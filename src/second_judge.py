"""Second judge on the stratified briefing subsample (playbook sec. 6: Groq-hosted or local model).

    python -m src.analyze export-second-judge-sample [--dry-run]   # writes the sample first
    python -m src.second_judge --dry-run                           # mock client
    python -m src.second_judge --approve                           # real, after the user approves

Same judge prompt as the primary judge (src/prompts.py). Every call attempt goes to the raw JSONL
log with role "second_judge"; one re-ask on a parse failure, then NaN. Config caps
(second_judge.rate_limits) are enforced: per-minute caps wait, per-day caps stop (resume later).
Groq's free tier costs no money, but it is an external API call, so a real run needs --approve.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src import prompts
from src.clients import (
    ChatClient,
    ChatRequest,
    ChatResponse,
    ClientError,
    DailyCapReached,
    FatalClientError,
    MockChatClient,
    OpenAIChatClient,
    RateLimiter,
    RateLimits,
    RetryPolicy,
    Role,
    TransientError,
    approx_tokens,
    call_with_retries,
)
from src.harness import ConfigError, JsonlLog, JudgeConfig, assistant, check_judge_family, load_config, user
from src.harness import read_judge_usage
from src.parsing import parse_probability

ROLE = "second_judge"
SCORE_COLUMNS: tuple[str, ...] = (
    "run_id", "dry_run", "briefing_id", "subject", "condition", "item_id", "sample", "tier",
    "judge2_provider", "judge2_model", "J2", "status",
)


def load_second_judge(config_path: Path) -> tuple[JudgeConfig, int]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        j = raw["second_judge"]
        rl = j.get("rate_limits") or {}
        cfg = JudgeConfig(
            provider=str(j["provider"]), family=j.get("family"), base_url=str(j["base_url"]),
            api_key_env=str(j.get("api_key_env") or ""), model=j.get("model"),
            temperature=float(j["temperature"]), max_tokens=int(j["max_tokens"]),
            price_in_per_1m=None, price_out_per_1m=None,
            rate_limits=RateLimits(
                requests_per_minute=rl.get("requests_per_minute"), tokens_per_minute=rl.get("tokens_per_minute"),
                requests_per_day=rl.get("requests_per_day"), tokens_per_day=rl.get("tokens_per_day"),
            ),
        )
        concurrency = int(raw["concurrency"]["second_judge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{config_path}: invalid second_judge settings: {exc!r}") from exc
    return cfg, concurrency


def _done_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise ConfigError(f"{path} has an unexpected header")
        return {row["briefing_id"] for row in reader}


async def score_all(sample: pd.DataFrame, *, client: ChatClient, judge: JudgeConfig, model: str,
                    log: JsonlLog, scores_path: Path, retry: RetryPolicy, limiter: RateLimiter | None,
                    concurrency: int, reasks: int) -> tuple[int, Counter[str], str | None]:
    new = not scores_path.exists() or scores_path.stat().st_size == 0
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    fh = scores_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(SCORE_COLUMNS))
    if new:
        writer.writeheader()
    sem = asyncio.Semaphore(max(1, concurrency))
    statuses: Counter[str] = Counter()
    written = 0

    async def call(row: Any, purpose: str, req: ChatRequest) -> ChatResponse | None:
        cell = {"briefing_id": row.briefing_id, "subject": row.subject, "condition": row.condition,
                "item_id": row.item_id, "sample": int(row.sample)}
        payload = req.to_payload()

        def log_attempt(n: int, resp: ChatResponse | None, exc: BaseException | None, latency: float) -> None:
            log.write("call", stream=ROLE, role=ROLE, arm="judge2", purpose=purpose, attempt=n, **cell,
                      base_url=client.base_url, request=payload,
                      response=resp.raw if resp else None, text=resp.text if resp else None,
                      usage=({"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}
                             if resp else None),
                      latency_s=round(latency, 3), error=None if exc is None else f"{type(exc).__name__}: {exc}")

        async def before() -> None:
            if limiter is not None:
                await limiter.acquire(approx_tokens(req.messages) + req.max_tokens)

        try:
            return await call_with_retries(client, req, retry, log_attempt, before)
        except (TransientError, ClientError):
            return None

    async def one(row: Any) -> None:
        nonlocal written
        async with sem:
            req = ChatRequest(model=model, messages=prompts.judge_messages(row.question, row.briefing),
                              temperature=judge.temperature, max_tokens=judge.max_tokens)
            resp = await call(row, "primary", req)
            value, status = math.nan, "nan_error"
            if resp is not None:
                parsed = parse_probability(resp.text)
                if parsed.value is not None:
                    value, status = parsed.value, "ok"
                else:
                    log.write("parse_error", stream=ROLE, role=ROLE, arm="judge2", purpose="primary",
                              briefing_id=row.briefing_id, reason=parsed.error, text=resp.text)
                    messages = list(req.messages)
                    last = resp.text
                    status = "nan_parse"
                    for _ in range(reasks):
                        messages = [*messages, assistant(last), user(prompts.JUDGE_REASK_MESSAGE)]
                        resp2 = await call(row, "reask", replace(req, messages=messages))
                        if resp2 is None:
                            status = "nan_error"
                            break
                        parsed = parse_probability(resp2.text)
                        if parsed.value is not None:
                            value, status = parsed.value, "reask_ok"
                            break
                        log.write("parse_error", stream=ROLE, role=ROLE, arm="judge2", purpose="reask",
                                  briefing_id=row.briefing_id, reason=parsed.error, text=resp2.text)
                        last = resp2.text
            writer.writerow({
                "run_id": log.run_id, "dry_run": log.dry_run, "briefing_id": row.briefing_id,
                "subject": row.subject, "condition": row.condition, "item_id": row.item_id,
                "sample": int(row.sample), "tier": int(row.tier), "judge2_provider": judge.provider,
                "judge2_model": model, "J2": "nan" if math.isnan(value) else repr(value), "status": status,
            })
            fh.flush()
            statuses[status] += 1
            written += 1

    stop: str | None = None
    try:
        async with asyncio.TaskGroup() as tg:
            for row in sample.itertuples(index=False):
                tg.create_task(one(row))
    except BaseExceptionGroup as eg:
        leaves = list(eg.exceptions)
        if all(isinstance(e, (DailyCapReached, FatalClientError)) for e in leaves):
            stop = "; ".join(sorted({f"{type(e).__name__}: {e}" for e in leaves}))
        else:
            raise
    finally:
        fh.close()
    return written, statuses, stop


def main(argv: list[str] | None = None, *, mock_factory: Callable[[Role], ChatClient] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.second_judge", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--approve", action="store_true", help="required for a real run (user approval)")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        judge, concurrency = load_second_judge(args.config)
        if args.dry_run:
            base = cfg.paths["dry_run_dir"]
            sample_path, scores_path, log_path = (base / "tables" / "second_judge_sample.csv",
                                                  base / "second_judge_scores.csv", base / "trajectories")
        else:
            sample_path = cfg.paths["tables"] / "second_judge_sample.csv"
            scores_path, log_path = cfg.paths["second_judge_scores"], cfg.paths["raw_log_dir"]

        if judge.family:
            check_judge_family(cfg.subjects, judge)
        elif args.dry_run:
            print("WARNING: second_judge.family is not set; a real run will refuse until it is.")
        else:
            raise ConfigError("second_judge.family is not set; cannot check it differs from the subjects")
        if not sample_path.exists():
            raise ConfigError(f"no sample at {sample_path}; run `python -m src.analyze "
                              f"export-second-judge-sample{' --dry-run' if args.dry_run else ''}` first")
        sample = pd.read_csv(sample_path, dtype={"briefing_id": str, "item_id": str, "subject": str,
                                                 "condition": str, "question": str, "briefing": str})
        done = _done_ids(scores_path)
        todo = sample[~sample["briefing_id"].isin(done)]
        print(f"== second judge: {len(sample)} briefings in sample, {len(done)} already scored, {len(todo)} to do")

        if args.dry_run:
            factory = mock_factory or (lambda role: MockChatClient(role))
            client, model = factory("judge"), judge.model or "mock-second-judge"
            retry, limiter = RetryPolicy(cfg.retry.max_attempts, 0.0, 0.0), None
        else:
            problems = []
            if not judge.model:
                problems.append("second_judge.model is not set in config.yaml")
            if not args.approve:
                problems.append(f"{len(todo)} external API calls to {judge.base_url}: ask the user, then "
                                "re-run with --approve")
            if not problems and judge.api_key_env:
                from dotenv import load_dotenv

                load_dotenv()
                if not os.environ.get(judge.api_key_env):
                    problems.append(f"environment variable {judge.api_key_env} is not set")
            if problems:
                print("REFUSING TO START:", *(f"  - {x}" for x in problems), sep="\n", file=sys.stderr)
                return 2
            key = os.environ[judge.api_key_env] if judge.api_key_env else "EMPTY"
            client, model = OpenAIChatClient(judge.base_url, key), str(judge.model)
            usage = read_judge_usage(log_path, role=ROLE)
            retry = cfg.retry
            limiter = RateLimiter(judge.rate_limits, used_today_requests=usage.requests_today,
                                  used_today_tokens=usage.tokens_today)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        log = JsonlLog(log_path, run_id, args.dry_run)
        log.write("run_start", stream=ROLE, role=ROLE, judge2_provider=judge.provider, judge2_model=model,
                  n_todo=len(todo), sample=str(sample_path))
        try:
            written, statuses, stop = asyncio.run(score_all(
                todo, client=client, judge=judge, model=model, log=log, scores_path=scores_path, retry=retry,
                limiter=limiter, concurrency=concurrency, reasks=cfg.reasks))
        finally:
            log.write("run_end", stream=ROLE, role=ROLE)
            log.close()
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"== wrote {written} rows -> {scores_path}; statuses: {dict(statuses)}")
    if stop:
        print(f"   STOPPED EARLY: {stop} (re-run later to resume)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
