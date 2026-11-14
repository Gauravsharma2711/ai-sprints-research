"""Multi-turn epistemic harness: every arm per subject x condition x item x sample.

Usage (from the repo root):
    python -m src.harness --dry-run --limit 3                # mock client, results/dry_run/
    python -m src.harness --show-histories --dry-run         # print one cell's history per arm
    python -m src.harness --estimate --models qwen2.5-7b     # call count + judge cost, then stop
    python -m src.harness --models qwen2.5-7b --approve-spend --allow-unverified   # real run

Rules (CLAUDE.md): every call attempt is appended to the JSONL log before parsing; a failed
parse gets one re-ask, then NaN plus a logged parse_error; paid judge calls are refused past
budget.max_spend_usd; keys are never printed or logged. Dry-run numbers are placeholders.

Arms (P = P(answer is YES)):
    T1     [sys, claim]                                       -> c1
    probe  [sys, claim + "answer Yes or No"], 1 token, logprobs -> L
    judge  [judge sys, question + stripped T1 prose]          -> J
    T2A    T1 + [r1, bluff]                                   -> c2A
    T2B    T1 + [r1, evidence_turn]                           -> c2B
    T2N    T1 + [r1, neutral]                                 -> c2N
    T3     T1 + [r1, bluff, r2A, release]                     -> c3
    T3a    T1 + [r1, bluff, NEUTRAL_PLACEHOLDER, release]     -> c3a
    T3b    T1 + [r1, bluff, stand-firm reply restating c1, release] -> c3b
    T3c    T1 + [r1, release_no_bluff]                        -> c3c
    T3N    T1 + [r1, neutral, r2N, neutral_again]             -> c3N
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Coroutine, Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, TypeVar

import yaml

from src import prompts
from src.benchmark import BenchmarkItem, load_items, load_turns, validate_items, validate_turns
from src.clients import (
    BudgetExceeded,
    ChatClient,
    ChatRequest,
    ChatResponse,
    ClientError,
    DailyCapReached,
    FatalClientError,
    Message,
    MockChatClient,
    OpenAIChatClient,
    RateLimiter,
    RateLimits,
    RetryPolicy,
    Role,
    SpendTracker,
    TransientError,
    approx_tokens,
    call_with_retries,
)
from src.parsing import parse_probability, probe_p_yes, strip_prose

NAN = float("nan")
T = TypeVar("T")

ALL_ARMS: tuple[str, ...] = ("T1", "probe", "judge", "T2A", "T2B", "T2N", "T3", "T3a", "T3b", "T3c", "T3N")
SUBJECT_PROB_ARMS: tuple[str, ...] = ("T1", "T2A", "T2B", "T2N", "T3", "T3a", "T3b", "T3c", "T3N")
VALUE_COLUMN: dict[str, str] = {
    "T1": "c1", "probe": "L", "judge": "J", "T2A": "c2A", "T2B": "c2B", "T2N": "c2N",
    "T3": "c3", "T3a": "c3a", "T3b": "c3b", "T3c": "c3c", "T3N": "c3N",
}
RESULT_COLUMNS: tuple[str, ...] = (
    "run_id", "protocol_hash", "dry_run", "timestamp", "subject", "hf_id", "family", "condition",
    "item_id", "domain", "tier", "truth", "evidence_direction", "sample", "paraphrase_idx",
    *VALUE_COLUMN.values(), "probe_mass", *(f"status_{a}" for a in ALL_ARMS),
)
SUBJECT_CALLS_PER_CELL = len(ALL_ARMS) - 1  # all arms except judge
STOP_ERRORS = (BudgetExceeded, DailyCapReached, FatalClientError)
PENDING = "pending"   # judge arm not yet scored (--skip-judge); filled in by --judge-only
JUDGED_OK = ("ok", "reask_ok")


class ConfigError(Exception):
    """Configuration or input problem. The CLI prints it and exits with status 2."""


class MissingDependency(Exception):
    """An arm cannot be built because an earlier arm produced no usable output."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectConfig:
    name: str
    hf_id: str
    family: str | None
    base_url: str
    api_key_env: str | None = None
    quantisation: str = "none"
    tensor_parallel: int = 1
    confirmed: bool = False


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    family: str | None
    base_url: str
    api_key_env: str
    model: str | None
    temperature: float
    max_tokens: int
    price_in_per_1m: float | None
    price_out_per_1m: float | None
    rate_limits: RateLimits = field(default_factory=RateLimits)
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class HarnessConfig:
    subjects: tuple[SubjectConfig, ...]
    judge: JudgeConfig
    temperature: float
    max_tokens: int
    samples_per_cell: int
    seed: int
    probe_top_logprobs: int
    subject_concurrency: int
    judge_concurrency: int
    retry: RetryPolicy
    reasks: int
    max_spend_usd: float
    paths: dict[str, Path]


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_config(path: Path) -> HarnessConfig:
    """Relative paths in the config resolve against the config file's directory."""
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    base = path.resolve().parent
    try:
        defaults = raw.get("subject_defaults", {}) or {}
        subjects = tuple(
            SubjectConfig(
                name=str(s["name"]),
                hf_id=str(s["hf_id"]),
                family=s.get("family"),
                base_url=str(s.get("base_url", defaults.get("base_url"))),
                api_key_env=s.get("api_key_env"),
                quantisation=str(s.get("quantisation", "none")),
                tensor_parallel=int(s.get("tensor_parallel", 1)),
                confirmed=bool(s.get("confirmed", False)),
            )
            for s in raw["subjects"]
        )
        j = raw["judge"]
        rl = j.get("rate_limits") or {}
        judge = JudgeConfig(
            provider=str(j["provider"]),
            family=j.get("family"),
            base_url=str(j["base_url"]),
            api_key_env=str(j["api_key_env"]),
            model=j.get("model"),
            temperature=float(j["temperature"]),
            max_tokens=int(j["max_tokens"]),
            price_in_per_1m=_opt_float(j.get("price_per_1m_input_usd")),
            price_out_per_1m=_opt_float(j.get("price_per_1m_output_usd")),
            rate_limits=RateLimits(
                requests_per_minute=_opt_int(rl.get("requests_per_minute")),
                tokens_per_minute=_opt_int(rl.get("tokens_per_minute")),
                requests_per_day=_opt_int(rl.get("requests_per_day")),
                tokens_per_day=_opt_int(rl.get("tokens_per_day")),
            ),
            reasoning_effort=j.get("reasoning_effort"),
        )
        sampling, conc, retries = raw["sampling"], raw["concurrency"], raw["retries"]
        cfg = HarnessConfig(
            subjects=subjects,
            judge=judge,
            temperature=float(sampling["temperature"]),
            max_tokens=int(sampling["max_tokens"]),
            samples_per_cell=int(sampling["samples_per_cell"]),
            seed=int(sampling["seed"]),
            probe_top_logprobs=int(sampling.get("probe_top_logprobs", 20)),
            subject_concurrency=int(conc["subject"]),
            judge_concurrency=int(conc["judge"]),
            retry=RetryPolicy(
                max_attempts=int(retries["max_attempts"]),
                initial_wait_s=float(retries.get("initial_wait_s", 2.0)),
                max_wait_s=float(retries.get("max_wait_s", 60.0)),
            ),
            reasks=int(retries["reask_on_parse_failure"]),
            max_spend_usd=float(raw["budget"]["max_spend_usd"]),
            paths={k: (Path(v) if Path(v).is_absolute() else base / v) for k, v in raw["paths"].items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: missing or invalid setting: {exc!r}") from exc
    names = [s.name for s in subjects]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate subject names: {names}")
    return cfg


def check_judge_family(subjects: Iterable[SubjectConfig], judge: JudgeConfig) -> None:
    """The judge must come from a different model family than every subject. Fails loudly."""
    subjects = list(subjects)
    if not judge.family or not str(judge.family).strip():
        raise ConfigError("judge.family is not set; cannot check it differs from the subjects")
    jf = str(judge.family).strip().lower()
    missing = [s.name for s in subjects if not s.family or not str(s.family).strip()]
    if missing:
        raise ConfigError(f"subjects without a family: {missing}")
    clash = [s.name for s in subjects if str(s.family).strip().lower() == jf]
    if clash:
        raise ConfigError(f"judge family {judge.family!r} is the same as subject(s) {clash}")
    if judge.model:
        named = [s.name for s in subjects if str(s.family).lower() in judge.model.lower()]
        if named:
            raise ConfigError(f"judge.model {judge.model!r} names the family of subject(s) {named}")
    named = [s.name for s in subjects if jf in s.hf_id.lower()]
    if named:
        raise ConfigError(f"subject(s) {named} have the judge family {judge.family!r} in their hf_id")


# ---------------------------------------------------------------------------
# Histories (pure functions; tested for every arm)
# ---------------------------------------------------------------------------


def user(text: str) -> Message:
    return {"role": "user", "content": text}


def assistant(text: str) -> Message:
    return {"role": "assistant", "content": text}


@dataclass(frozen=True)
class CellTurns:
    """Fixed user turns for one sample. Paraphrase index = sample % 3; the second neutral turn
    in T3N uses the next paraphrase so it is not a verbatim repeat."""

    paraphrase_idx: int
    bluff: str
    release: str
    release_no_bluff: str
    neutral: str
    neutral_again: str

    @classmethod
    def for_sample(cls, turns: dict[str, list[str]], sample: int) -> CellTurns:
        n = len(turns["bluff"])
        k = sample % n
        return cls(
            paraphrase_idx=k,
            bluff=turns["bluff"][k],
            release=turns["release"][k],
            release_no_bluff=turns["release_no_bluff"][k],
            neutral=turns["neutral"][k],
            neutral_again=turns["neutral"][(k + 1) % n],
        )


@dataclass(frozen=True)
class HistoryInputs:
    condition: str
    claim: str
    evidence_turn: str
    turns: CellTurns
    r1: str | None = None
    c1: float = NAN
    r2a: str | None = None
    r2n: str | None = None


def _need(value: str | None, what: str) -> str:
    if value is None:
        raise MissingDependency(f"{what} is missing")
    return value


def build_history(arm: str, h: HistoryInputs) -> list[Message]:
    system: Message = {"role": "system", "content": prompts.system_prompt(h.condition)}
    if arm == "T1":
        return [system, user(h.claim)]
    if arm == "probe":
        return [system, user(prompts.probe_user_message(h.claim))]
    if arm not in VALUE_COLUMN or arm == "judge":
        raise ValueError(f"no subject history for arm {arm!r}")
    t1 = [system, user(h.claim), assistant(_need(h.r1, "T1 reply"))]
    t = h.turns
    if arm == "T2A":
        return [*t1, user(t.bluff)]
    if arm == "T2B":
        return [*t1, user(h.evidence_turn)]
    if arm == "T2N":
        return [*t1, user(t.neutral)]
    if arm == "T3":
        return [*t1, user(t.bluff), assistant(_need(h.r2a, "T2A reply")), user(t.release)]
    if arm == "T3a":
        return [*t1, user(t.bluff), assistant(prompts.NEUTRAL_PLACEHOLDER), user(t.release)]
    if arm == "T3b":
        if math.isnan(h.c1):
            raise MissingDependency("c1 is NaN, so the stand-firm reply cannot restate it")
        return [*t1, user(t.bluff), assistant(prompts.stand_firm_reply(h.c1)), user(t.release)]
    if arm == "T3c":
        return [*t1, user(t.release_no_bluff)]
    if arm == "T3N":
        return [*t1, user(t.neutral), assistant(_need(h.r2n, "T2N reply")), user(t.neutral_again)]
    raise ValueError(f"unknown arm {arm!r}")  # pragma: no cover


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(json.dumps([base, *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def scrub_terms_for(subjects: Iterable[SubjectConfig]) -> list[str]:
    """Names the judge must never see (all configured subjects, not just the current one)."""
    terms: set[str] = set()
    for s in subjects:
        terms.update({s.name, s.hf_id, s.hf_id.split("/")[-1], s.hf_id.split("/")[0]})
        if s.family:
            terms.add(str(s.family))
    return sorted(terms)


# ---------------------------------------------------------------------------
# Logging and results
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


RUNS_STREAM = "_runs"
GITHUB_FILE_LIMIT_MB = 100  # GitHub rejects files above this
LOG_WARN_MB = 80            # warn early so there is time to react before a push fails


def _stream_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or RUNS_STREAM


class JsonlLog:
    """Append-only raw logs in a directory, one JSONL file per stream, flushed per record.

    The stream is the subject model (``subject`` field) unless given explicitly, so each model's
    calls (including the judge calls for its cells) live in ``<dir>/<model>.jsonl``. Records with no
    subject (run start/end) go to ``_runs.jsonl``. Splitting keeps each file under GitHub's limit.
    """

    def __init__(self, directory: Path, run_id: str, dry_run: bool) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.run_id = run_id
        self.dry_run = dry_run
        self._files: dict[str, TextIO] = {}

    def write(self, event: str, *, stream: str | None = None, **fields: Any) -> None:
        name = _stream_name(stream or str(fields.get("subject") or RUNS_STREAM))
        fh = self._files.get(name)
        if fh is None:
            fh = self._files[name] = (self.directory / f"{name}.jsonl").open("a", encoding="utf-8")
        record = {"event": event, "ts": _now(), "run_id": self.run_id, "dry_run": self.dry_run, **fields}
        fh.write(json.dumps(_json_safe(record), ensure_ascii=False, default=str) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def iter_log_records(directory: Path) -> Iterator[dict[str, Any]]:
    """Every record in every JSONL file of a log directory (files in name order)."""
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.jsonl")):
        bad = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1  # e.g. a line cut off by a crash mid-write
        if bad:
            print(f"WARNING: skipped {bad} unreadable line(s) in {path}", file=sys.stderr)


def log_size_warnings(directory: Path, warn_mb: float = LOG_WARN_MB) -> list[str]:
    warnings_: list[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.jsonl")):
            size_mb = path.stat().st_size / 1_000_000
            if size_mb >= warn_mb:
                warnings_.append(f"{path.name} is {size_mb:.0f} MB (GitHub limit {GITHUB_FILE_LIMIT_MB} MB per file)")
    return warnings_


CellKey = tuple[str, str, str, int]  # (subject, condition, item_id, sample)


def read_done_keys(path: Path, protocol_hash: str) -> set[CellKey]:
    """Completed cells in results.csv. Refuses to mix rows from a different protocol."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[CellKey] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
            raise ConfigError(f"{path} has a different header; write to a new output directory")
        for row in reader:
            if row["protocol_hash"] != protocol_hash:
                raise ConfigError(
                    f"{path} holds rows from protocol {row['protocol_hash']}, but prompts/items/turns "
                    f"now hash to {protocol_hash}. Use a new output directory instead of mixing them."
                )
            done.add((row["subject"], row["condition"], row["item_id"], int(row["sample"])))
    return done


class ResultsWriter:
    def __init__(self, path: Path, protocol_hash: str) -> None:
        self.done = read_done_keys(path, protocol_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(RESULT_COLUMNS))
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        self.done.add((row["subject"], row["condition"], row["item_id"], int(row["sample"])))

    def close(self) -> None:
        self._fh.close()


def protocol_hash(items_path: Path, turns: dict[str, list[str]]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(prompts.all_prompt_text()).encode())
    h.update(json.dumps(turns, sort_keys=True).encode())
    h.update(items_path.read_bytes())
    return h.hexdigest()[:12]


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else repr(float(value))


# ---------------------------------------------------------------------------
# Running cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRef:
    subject: str
    condition: str
    item_id: str
    sample: int

    def fields(self) -> dict[str, Any]:
        return {"subject": self.subject, "condition": self.condition,
                "item_id": self.item_id, "sample": self.sample}


@dataclass(frozen=True)
class ArmResult:
    text: str | None
    value: float
    status: str  # ok | reask_ok | nan_parse | nan_error | nan_dependency | nan_no_logprobs | ...
    mass: float = NAN


@dataclass
class Endpoint:
    client: ChatClient
    model: str
    semaphore: asyncio.Semaphore
    limiter: RateLimiter | None = None
    spend: SpendTracker | None = None


async def _all(*coros: Coroutine[Any, Any, T]) -> list[T]:
    """Run concurrently; if one fails, cancel the rest (TaskGroup semantics)."""
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(c) for c in coros]
    return [t.result() for t in tasks]


class Runner:
    def __init__(self, cfg: HarnessConfig, *, log: JsonlLog, writer: ResultsWriter | None,
                 judge: Endpoint | None, turns: dict[str, list[str]], retry: RetryPolicy, protocol: str,
                 scrub_terms: list[str], skip_judge: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.writer = writer
        self.judge = judge
        self.turns = turns
        self.retry = retry
        self.protocol = protocol
        self.scrub_terms = scrub_terms
        self.skip_judge = skip_judge
        self.rows: list[dict[str, Any]] = []

    # -- calls ---------------------------------------------------------------

    async def _call(self, ep: Endpoint, role: Role, cell: CellRef, arm: str, purpose: str,
                    req: ChatRequest) -> ChatResponse | None:
        est_in, est_out = approx_tokens(req.messages), req.max_tokens
        payload = req.to_payload()

        def log_attempt(n: int, resp: ChatResponse | None, exc: BaseException | None, latency: float) -> None:
            self.log.write(
                "call", role=role, arm=arm, purpose=purpose, attempt=n, **cell.fields(),
                base_url=ep.client.base_url, request=payload,
                response=resp.raw if resp else None, text=resp.text if resp else None,
                usage=({"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}
                       if resp else None),
                finish_reason=resp.finish_reason if resp else None,
                latency_s=round(latency, 3),
                error=None if exc is None else f"{type(exc).__name__}: {exc}",
            )

        async def before_attempt() -> None:
            if ep.limiter is not None:
                await ep.limiter.acquire(est_in + est_out)

        async with ep.semaphore:
            reserved = ep.spend.reserve(est_in, est_out) if ep.spend is not None else 0.0
            try:
                resp = await call_with_retries(ep.client, req, self.retry, log_attempt, before_attempt)
            except (TransientError, ClientError):
                if ep.spend is not None:
                    ep.spend.settle(reserved, None, None, est_in, est_out, succeeded=False)
                return None
            except BaseException:
                if ep.spend is not None:
                    ep.spend.settle(reserved, None, None, est_in, est_out, succeeded=False)
                raise
            if ep.spend is not None:
                ep.spend.settle(reserved, resp.prompt_tokens, resp.completion_tokens, est_in, est_out,
                                succeeded=True)
            return resp

    async def _ask_probability(self, ep: Endpoint, role: Role, cell: CellRef, arm: str,
                               req: ChatRequest) -> ArmResult:
        resp = await self._call(ep, role, cell, arm, "primary", req)
        if resp is None:
            return ArmResult(None, NAN, "nan_error")
        parsed = parse_probability(resp.text)
        if parsed.value is not None:
            return ArmResult(resp.text, parsed.value, "ok")
        self.log.write("parse_error", role=role, arm=arm, purpose="primary", **cell.fields(),
                       reason=parsed.error, text=resp.text)
        reask = prompts.JUDGE_REASK_MESSAGE if role == "judge" else prompts.REASK_MESSAGE
        messages = list(req.messages)
        last_text = resp.text
        for n in range(1, self.cfg.reasks + 1):
            messages = [*messages, assistant(last_text), user(reask)]
            seed = None if req.seed is None else (req.seed + n) & 0x7FFFFFFF
            resp_n = await self._call(ep, role, cell, arm, "reask", replace(req, messages=messages, seed=seed))
            if resp_n is None:
                return ArmResult(resp.text, NAN, "nan_error")
            parsed = parse_probability(resp_n.text)
            if parsed.value is not None:
                return ArmResult(resp.text, parsed.value, "reask_ok")
            self.log.write("parse_error", role=role, arm=arm, purpose="reask", **cell.fields(),
                           reason=parsed.error, text=resp_n.text)
            last_text = resp_n.text
        # The arm's reply text stays the original reply; only its probability is NaN.
        return ArmResult(resp.text, NAN, "nan_parse")

    # -- arms ----------------------------------------------------------------

    async def prob_arm(self, ep: Endpoint, cell: CellRef, arm: str, h: HistoryInputs) -> ArmResult:
        try:
            messages = build_history(arm, h)
        except MissingDependency as exc:
            self.log.write("skipped", role="subject", arm=arm, **cell.fields(), reason=str(exc))
            return ArmResult(None, NAN, "nan_dependency")
        req = ChatRequest(
            model=ep.model, messages=messages, temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            seed=stable_seed(self.cfg.seed, cell.subject, cell.condition, cell.item_id, cell.sample, arm),
        )
        return await self._ask_probability(ep, "subject", cell, arm, req)

    async def probe_arm(self, ep: Endpoint, cell: CellRef, h: HistoryInputs) -> ArmResult:
        req = ChatRequest(
            model=ep.model, messages=build_history("probe", h), temperature=0.0, max_tokens=1,
            seed=stable_seed(self.cfg.seed, cell.subject, cell.condition, cell.item_id, cell.sample, "probe"),
            top_logprobs=self.cfg.probe_top_logprobs,
        )
        resp = await self._call(ep, "subject", cell, "probe", "primary", req)
        if resp is None:
            return ArmResult(None, NAN, "nan_error")
        probe = probe_p_yes(resp.top_logprobs)
        mass = NAN if probe.mass is None else probe.mass
        if probe.p_yes is None:
            self.log.write("probe_nan", role="subject", arm="probe", **cell.fields(), reason=probe.error)
            return ArmResult(resp.text, NAN, f"nan_{probe.error}", mass)
        return ArmResult(resp.text, probe.p_yes, "ok", mass)

    async def judge_arm(self, cell: CellRef, item: BenchmarkItem, r1: str | None) -> ArmResult:
        if r1 is None:
            self.log.write("skipped", role="judge", arm="judge", **cell.fields(), reason="T1 reply is missing")
            return ArmResult(None, NAN, "nan_dependency")
        prose = strip_prose(r1, self.scrub_terms)
        if not prose.strip():
            self.log.write("skipped", role="judge", arm="judge", **cell.fields(),
                           reason="nothing left after stripping", text=r1)
            return ArmResult(None, NAN, "nan_empty_prose")
        if self.judge is None:
            raise ConfigError("judge_arm called without a judge endpoint")
        req = ChatRequest(model=self.judge.model, messages=prompts.judge_messages(item.claim, prose),
                          temperature=self.cfg.judge.temperature, max_tokens=self.cfg.judge.max_tokens,
                          reasoning_effort=self.cfg.judge.reasoning_effort)
        return await self._ask_probability(self.judge, "judge", cell, "judge", req)

    async def pending_judge(self) -> ArmResult:
        """--skip-judge: the judge arm is filled in later by --judge-only."""
        return ArmResult(None, NAN, PENDING)

    async def run_cell(self, subject: SubjectConfig, ep: Endpoint, condition: str,
                       item: BenchmarkItem, sample: int) -> dict[str, Any]:
        cell = CellRef(subject.name, condition, item.id, sample)
        turns = CellTurns.for_sample(self.turns, sample)
        h = HistoryInputs(condition, item.claim, item.evidence_turn, turns)
        res: dict[str, ArmResult] = {}

        res["T1"], res["probe"] = await _all(self.prob_arm(ep, cell, "T1", h), self.probe_arm(ep, cell, h))
        h = replace(h, r1=res["T1"].text, c1=res["T1"].value)

        phase2 = ("T2A", "T2B", "T2N", "T3a", "T3b", "T3c")
        judge = self.pending_judge() if self.skip_judge else self.judge_arm(cell, item, res["T1"].text)
        out = await _all(judge, *(self.prob_arm(ep, cell, a, h) for a in phase2))
        res["judge"] = out[0]
        res.update(zip(phase2, out[1:]))
        h = replace(h, r2a=res["T2A"].text, r2n=res["T2N"].text)

        res["T3"], res["T3N"] = await _all(self.prob_arm(ep, cell, "T3", h), self.prob_arm(ep, cell, "T3N", h))

        row: dict[str, Any] = {
            "run_id": self.log.run_id, "protocol_hash": self.protocol, "dry_run": self.log.dry_run,
            "timestamp": _now(), "subject": subject.name, "hf_id": subject.hf_id,
            "family": subject.family, "condition": condition, "item_id": item.id,
            "domain": item.domain, "tier": item.tier, "truth": item.truth or "",
            "evidence_direction": item.evidence_direction, "sample": sample,
            "paraphrase_idx": turns.paraphrase_idx,
        }
        for arm, column in VALUE_COLUMN.items():
            row[column] = _fmt(res[arm].value)
        row["probe_mass"] = _fmt(res["probe"].mass)
        for arm in ALL_ARMS:
            row[f"status_{arm}"] = res[arm].status
        return row

    async def run_subject(self, subject: SubjectConfig, ep: Endpoint,
                          cells: list[tuple[str, BenchmarkItem, int]]) -> None:
        gate = asyncio.Semaphore(max(1, self.cfg.subject_concurrency))

        async def one(condition: str, item: BenchmarkItem, sample: int) -> None:
            async with gate:
                row = await self.run_cell(subject, ep, condition, item, sample)
                assert self.writer is not None
                self.writer.write(row)  # checkpoint: one row per completed cell
                self.rows.append(row)

        async with asyncio.TaskGroup() as tg:
            for condition, item, sample in cells:
                tg.create_task(one(condition, item, sample))


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for e in exc.exceptions for leaf in _leaves(e)]
    return [exc]


@dataclass
class RunPlan:
    cfg: HarnessConfig
    subjects: list[SubjectConfig]
    conditions: list[str]
    items: list[BenchmarkItem]
    turns: dict[str, list[str]]
    samples: int
    log_dir: Path
    results_path: Path
    dry_run: bool
    protocol: str
    skip_judge: bool = False
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        + "-" + uuid.uuid4().hex[:6])


@dataclass(frozen=True)
class RunOutcome:
    rows: list[dict[str, Any]]
    stop_reason: str | None


async def execute(plan: RunPlan, *, subject_client_for: Callable[[SubjectConfig], ChatClient],
                  judge_client: ChatClient | None, judge_model: str, retry: RetryPolicy,
                  judge_limiter: RateLimiter | None, judge_spend: SpendTracker | None) -> RunOutcome:
    cfg = plan.cfg
    log = JsonlLog(plan.log_dir, plan.run_id, plan.dry_run)
    writer = ResultsWriter(plan.results_path, plan.protocol)
    log.write("run_start", protocol_hash=plan.protocol, conditions=plan.conditions,
              n_items=len(plan.items), samples=plan.samples,
              subjects=[{"name": s.name, "hf_id": s.hf_id, "family": s.family, "base_url": s.base_url,
                         "quantisation": s.quantisation, "tensor_parallel": s.tensor_parallel}
                        for s in plan.subjects],
              judge=(None if plan.skip_judge else
                     {"provider": cfg.judge.provider, "family": cfg.judge.family, "model": judge_model,
                      "base_url": judge_client.base_url if judge_client else None}),
              skip_judge=plan.skip_judge,
              sampling={"temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "seed": cfg.seed},
              max_spend_usd=cfg.max_spend_usd)
    judge_ep = (None if judge_client is None else
                Endpoint(judge_client, judge_model, asyncio.Semaphore(cfg.judge_concurrency),
                         judge_limiter, judge_spend))
    runner = Runner(cfg, log=log, writer=writer, judge=judge_ep, turns=plan.turns, retry=retry,
                    protocol=plan.protocol, scrub_terms=scrub_terms_for(cfg.subjects),
                    skip_judge=plan.skip_judge)
    stop_reason: str | None = None
    try:
        for subject in plan.subjects:
            cells = [(c, it, s) for c in plan.conditions for it in plan.items for s in range(plan.samples)
                     if (subject.name, c, it.id, s) not in writer.done]
            if not cells:
                continue
            ep = Endpoint(subject_client_for(subject), subject.hf_id,
                          asyncio.Semaphore(max(1, cfg.subject_concurrency)))
            await runner.run_subject(subject, ep, cells)
    except BaseException as exc:
        leaves = _leaves(exc)
        if leaves and all(isinstance(e, STOP_ERRORS) for e in leaves):
            stop_reason = "; ".join(sorted({f"{type(e).__name__}: {e}" for e in leaves}))
        else:
            raise
    finally:
        log.write("run_end", rows_written=len(runner.rows), stop_reason=stop_reason,
                  spent_usd=None if judge_spend is None else round(judge_spend.spent_usd, 6))
        log.close()
        writer.close()
    return RunOutcome(runner.rows, stop_reason)


# ---------------------------------------------------------------------------
# Estimate and spend accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    requests_today: int = 0
    tokens_today: int = 0


def read_judge_usage(log_dir: Path, role: str = "judge") -> JudgeUsage:
    """Judge usage already logged for ``role`` (non-dry-run calls) across every file in the log
    directory, all-time and today (UTC)."""
    today = datetime.now(timezone.utc).date().isoformat()
    t_in = t_out = req_today = tok_today = 0
    for rec in iter_log_records(log_dir):
        if rec.get("event") != "call" or rec.get("role") != role or rec.get("dry_run"):
            continue
        usage = rec.get("usage") or {}
        p_in, p_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        t_in += p_in
        t_out += p_out
        if str(rec.get("ts", "")).startswith(today):
            req_today += 1
            tok_today += p_in + p_out
    return JudgeUsage(t_in, t_out, req_today, tok_today)


@dataclass(frozen=True)
class Estimate:
    cells_total: int
    cells_remaining: int
    subject_calls: int
    subject_calls_max: int
    judge_calls: int
    judge_calls_max: int
    judge_tokens_in_max: int
    judge_tokens_out_max: int
    cost_expected_usd: float | None
    cost_max_usd: float | None
    prior_spend_usd: float | None
    cap_usd: float
    within_cap: bool
    reason: str


def estimate_run(cfg: HarnessConfig, items: list[BenchmarkItem], cells_total: int,
                 cells_remaining: int, usage: JudgeUsage, judge_cells: int | None = None) -> Estimate:
    """``judge_cells``: judge calls to price (default: one per remaining cell; 0 with --skip-judge)."""
    r = cells_remaining
    jc = r if judge_cells is None else judge_cells
    reasks = cfg.reasks
    prose = "x" * (cfg.max_tokens * 4)  # a reply at the token cap
    per_in = max((approx_tokens(prompts.judge_messages(i.claim, prose)) for i in items), default=0)
    per_out = cfg.judge.max_tokens
    reask_in = per_in + per_out + approx_tokens([user(prompts.JUDGE_REASK_MESSAGE)])
    tokens_in_exp, tokens_out_exp = jc * per_in, jc * per_out
    tokens_in_max = jc * (per_in + reasks * reask_in)
    tokens_out_max = jc * (1 + reasks) * per_out
    pin, pout = cfg.judge.price_in_per_1m, cfg.judge.price_out_per_1m

    if jc == 0:
        cost_exp = cost_max = 0.0
        prior = None if pin is None or pout is None else (usage.tokens_in * pin + usage.tokens_out * pout) / 1e6
        within, reason = True, "no judge calls in this run"
    elif pin is None or pout is None:
        cost_exp = cost_max = prior = None
        within, reason = False, "judge prices are not set in config.yaml (price_per_1m_*_usd)"
    else:
        cost_exp = (tokens_in_exp * pin + tokens_out_exp * pout) / 1e6
        cost_max = (tokens_in_max * pin + tokens_out_max * pout) / 1e6
        prior = (usage.tokens_in * pin + usage.tokens_out * pout) / 1e6
        within = prior + cost_max <= cfg.max_spend_usd
        reason = ("within cap" if within else
                  f"worst case ${prior + cost_max:.4f} (spent ${prior:.4f} + this run ${cost_max:.4f}) "
                  f"exceeds max_spend_usd=${cfg.max_spend_usd:.2f}")
    return Estimate(
        cells_total=cells_total, cells_remaining=r,
        subject_calls=r * SUBJECT_CALLS_PER_CELL,
        subject_calls_max=r * (SUBJECT_CALLS_PER_CELL + reasks * len(SUBJECT_PROB_ARMS)),
        judge_calls=jc, judge_calls_max=jc * (1 + reasks),
        judge_tokens_in_max=tokens_in_max, judge_tokens_out_max=tokens_out_max,
        cost_expected_usd=cost_exp, cost_max_usd=cost_max, prior_spend_usd=prior,
        cap_usd=cfg.max_spend_usd, within_cap=within, reason=reason,
    )


def print_estimate(est: Estimate, dry_run: bool) -> None:
    money = lambda v: "unknown" if v is None else f"${v:.4f}"  # noqa: E731
    print("== Estimate (approximate; token counts use a generous ~3 chars/token rule)")
    print(f"  cells: {est.cells_remaining} remaining of {est.cells_total}")
    print(f"  subject calls: {est.subject_calls} (up to {est.subject_calls_max} with re-asks)")
    print(f"  judge calls:   {est.judge_calls} (up to {est.judge_calls_max} with re-asks)")
    print(f"  judge tokens (worst case): {est.judge_tokens_in_max} in, {est.judge_tokens_out_max} out")
    print(f"  judge cost for a real run: expected {money(est.cost_expected_usd)}, "
          f"worst case {money(est.cost_max_usd)}; already spent {money(est.prior_spend_usd)}; "
          f"cap ${est.cap_usd:.2f}")
    print(f"  budget check: {'OK' if est.within_cap else 'REFUSED'} ({est.reason})")
    if dry_run:
        print("  (this dry run uses a mock client and spends nothing)")


# ---------------------------------------------------------------------------
# History viewer
# ---------------------------------------------------------------------------


def show_histories(log_dir: Path, *, subject: str | None, condition: str | None,
                   item_id: str | None, sample: int | None) -> int:
    if not log_dir.is_dir():
        print(f"no log directory at {log_dir}", file=sys.stderr)
        return 1
    groups: dict[CellKey, dict[str, dict[str, Any]]] = {}
    for rec in iter_log_records(log_dir):
        if (rec.get("event") != "call" or rec.get("purpose") != "primary"
                or rec.get("role") not in ("subject", "judge")):
            continue
        key = (rec["subject"], rec["condition"], rec["item_id"], int(rec["sample"]))
        if ((subject and key[0] != subject) or (condition and key[1] != condition)
                or (item_id and key[2] != item_id) or (sample is not None and key[3] != sample)):
            continue
        groups.setdefault(key, {})[rec["arm"]] = rec  # keep the last attempt (same request)
    complete = [(k, g) for k, g in groups.items() if set(ALL_ARMS) <= set(g)]
    if not complete:
        print("no cell with every arm found in the log for that selection", file=sys.stderr)
        return 1
    key, group = complete[0]
    print(f"Cell: subject={key[0]} condition={key[1]} item={key[2]} sample={key[3]}  (log: {log_dir})")
    for arm in ALL_ARMS:
        rec = group[arm]
        req = rec["request"]
        params = {k: v for k, v in req.items() if k != "messages"}
        print(f"\n{'=' * 22} {arm} {'=' * 22}\nparams: {json.dumps(params)}  (attempt {rec['attempt']})")
        for m in req["messages"]:
            print(f"--- [{m['role']}]\n{m['content']}")
        reply = rec.get("text")
        print(f"--> reply: {reply!r}" if reply is not None else f"--> error: {rec.get('error')}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Judge-only pass: score T1 briefings from the logs, separately from the GPU run
# ---------------------------------------------------------------------------

JUDGE_CHECKPOINT_EVERY = 25


def _cell_key(row: dict[str, Any]) -> CellKey:
    return (str(row["subject"]), str(row["condition"]), str(row["item_id"]), int(row["sample"]))


def judge_targets(rows: list[dict[str, str]], log_dir: Path, judge_model: str,
                  subjects: set[str] | None) -> tuple[list[dict[str, str]], dict[tuple[CellKey, str], str]]:
    """Rows needing a (re-)judgement, and the T1 reply text per (cell, run_id).

    A row needs judging unless its judge status is ok/reask_ok AND the latest successful judge call
    logged for that cell used ``judge_model``. So cells pending, failed, or scored by a different
    judge model are (re-)judged, and every J ends up from the same judge.
    """
    t1_text: dict[tuple[CellKey, str], str] = {}
    last_judge_model: dict[CellKey, str] = {}
    for rec in iter_log_records(log_dir):
        if rec.get("event") != "call" or rec.get("error") is not None or "subject" not in rec:
            continue
        key = (rec["subject"], rec["condition"], rec["item_id"], int(rec["sample"]))
        if rec.get("role") == "subject" and rec.get("arm") == "T1" and rec.get("purpose") == "primary":
            t1_text[(key, rec["run_id"])] = rec.get("text") or ""
        elif rec.get("role") == "judge":
            last_judge_model[key] = rec["request"]["model"]
    todo = [r for r in rows
            if (subjects is None or r["subject"] in subjects)
            and not (r["status_judge"] in JUDGED_OK and last_judge_model.get(_cell_key(r)) == judge_model)]
    return todo, t1_text


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


async def run_judge_pass(cfg: HarnessConfig, *, rows: list[dict[str, str]], todo: list[dict[str, str]],
                         t1_text: dict[tuple[CellKey, str], str], items: dict[str, BenchmarkItem],
                         results_path: Path, log: JsonlLog, judge_ep: Endpoint, retry: RetryPolicy,
                         turns: dict[str, list[str]], protocol: str) -> tuple[int, Counter[str], str | None]:
    runner = Runner(cfg, log=log, writer=None, judge=judge_ep, turns=turns, retry=retry, protocol=protocol,
                    scrub_terms=scrub_terms_for(cfg.subjects))
    index = {_cell_key(r): r for r in rows}
    statuses: Counter[str] = Counter()
    done = 0

    async def one(row: dict[str, str]) -> None:
        nonlocal done
        key = _cell_key(row)
        cell = CellRef(*key)
        item = items.get(row["item_id"])
        r1 = t1_text.get((key, row["run_id"]))
        if item is None or r1 is None:
            log.write("skipped", role="judge", arm="judge", **cell.fields(),
                      reason="item or T1 reply for this row's run_id not found in the logs")
            res = ArmResult(None, NAN, "nan_dependency")
        else:
            res = await runner.judge_arm(cell, item, r1)
        index[key]["J"] = _fmt(res.value)
        index[key]["status_judge"] = res.status
        statuses[res.status] += 1
        done += 1
        if done % JUDGE_CHECKPOINT_EVERY == 0:
            _write_rows(results_path, rows)

    stop: str | None = None
    try:
        async with asyncio.TaskGroup() as tg:
            for row in todo:
                tg.create_task(one(row))
    except BaseException as exc:
        leaves = _leaves(exc)
        if leaves and all(isinstance(e, STOP_ERRORS) for e in leaves):
            stop = "; ".join(sorted({f"{type(e).__name__}: {e}" for e in leaves}))
        else:
            raise
    finally:
        _write_rows(results_path, rows)   # always save what was judged
    return done, statuses, stop


def judge_only(cfg: HarnessConfig, args: argparse.Namespace, *, log_dir: Path, results_path: Path,
               subjects: list[SubjectConfig], mock_factory: Callable[[Role], ChatClient] | None) -> int:
    if not results_path.exists():
        raise ConfigError(f"no results at {results_path}; run the harness first")
    turns = load_turns(cfg.paths["turns"])
    protocol = protocol_hash(cfg.paths["benchmark"], turns)
    read_done_keys(results_path, protocol)                      # header + protocol checks
    with results_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    items_list, _ = load_items(cfg.paths["benchmark"])
    items = {i.id: i for i in items_list}
    judge_model = cfg.judge.model or ("mock-judge" if args.dry_run else "")
    todo, t1_text = judge_targets(rows, log_dir, judge_model, {s.name for s in subjects})
    print(f"== Judge-only pass with {cfg.judge.provider} {judge_model!r}: {len(todo)} of {len(rows)} cells "
          "need judging (pending, failed, or scored by another judge model)")

    usage_dirs = {cfg.paths["raw_log_dir"].resolve(), log_dir.resolve()}
    parts = [read_judge_usage(d) for d in sorted(usage_dirs)]
    usage = JudgeUsage(*(sum(getattr(p, f) for p in parts)
                         for f in ("tokens_in", "tokens_out", "requests_today", "tokens_today")))
    est = estimate_run(cfg, items_list, len(rows), 0, usage, judge_cells=len(todo))
    print_estimate(est, args.dry_run)
    if args.estimate or not todo:
        return 0 if est.within_cap else 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    if args.dry_run:
        client: ChatClient = (mock_factory or (lambda role: MockChatClient(role)))("judge")
        retry = replace(cfg.retry, initial_wait_s=0.0, max_wait_s=0.0)
        limiter = RateLimiter(cfg.judge.rate_limits)
        spend = None
    else:
        problems = []
        if not cfg.judge.model:
            problems.append("judge.model is not set in config.yaml")
        if not est.within_cap:
            problems.append(f"budget: {est.reason}")
        elif (est.cost_max_usd or 0.0) > 0 and not args.approve_spend:
            problems.append(f"judge spend up to ${est.cost_max_usd:.4f}: ask the user, then re-run with "
                            "--approve-spend")
        if not problems:
            from dotenv import load_dotenv

            load_dotenv()
            if not os.environ.get(cfg.judge.api_key_env):
                problems.append(f"environment variable {cfg.judge.api_key_env} is not set")
        if problems:
            print("\nREFUSING TO START THE JUDGE PASS:", *(f"  - {p}" for p in problems), sep="\n", file=sys.stderr)
            return 2
        client = OpenAIChatClient(cfg.judge.base_url, os.environ[cfg.judge.api_key_env])
        retry = cfg.retry
        limiter = RateLimiter(cfg.judge.rate_limits, used_today_requests=usage.requests_today,
                              used_today_tokens=usage.tokens_today)
        pin, pout = cfg.judge.price_in_per_1m, cfg.judge.price_out_per_1m
        prior = 0.0 if pin is None or pout is None else (usage.tokens_in * pin + usage.tokens_out * pout) / 1e6
        spend = SpendTracker(cfg.max_spend_usd, pin, pout, spent_usd=prior)

    log = JsonlLog(log_dir, run_id, args.dry_run)
    log.write("run_start", mode="judge_only", protocol_hash=protocol, n_todo=len(todo),
              judge={"provider": cfg.judge.provider, "family": cfg.judge.family, "model": judge_model,
                     "base_url": client.base_url, "max_tokens": cfg.judge.max_tokens,
                     "reasoning_effort": cfg.judge.reasoning_effort})
    try:
        judge_ep = Endpoint(client, judge_model, asyncio.Semaphore(cfg.judge_concurrency), limiter, spend)
        done, statuses, stop = asyncio.run(run_judge_pass(
            cfg, rows=rows, todo=todo, t1_text=t1_text, items=items, results_path=results_path, log=log,
            judge_ep=judge_ep, retry=retry, turns=turns, protocol=protocol))
    finally:
        log.write("run_end", mode="judge_only")
        log.close()
    remaining = len(judge_targets(rows, log_dir, judge_model, {s.name for s in subjects})[0])
    print(f"\n== Judged {done} cells -> {results_path}; statuses: {dict(sorted(statuses.items()))}")
    print(f"   cells still needing the judge: {remaining}")
    if stop:
        print(f"   STOPPED EARLY: {stop}  (judged cells are saved; re-run --judge-only later to continue)")
        return 3
    return 0


def _split(value: str | None) -> list[str] | None:
    return None if value is None else [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.harness", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="mock client; writes to paths.dry_run_dir")
    p.add_argument("--limit", type=int, default=None, help="first N items by id")
    p.add_argument("--models", default=None, help="comma-separated subject names (default: all)")
    p.add_argument("--conditions", default=None, help=f"comma-separated, from {prompts.CONDITIONS}")
    p.add_argument("--samples", type=int, default=None, help="samples per cell (default: config)")
    p.add_argument("--estimate", action="store_true", help="print call count and judge cost, then stop")
    p.add_argument("--approve-spend", action="store_true",
                   help="required for a real run with nonzero estimated judge cost (after user approval)")
    p.add_argument("--allow-unverified", action="store_true",
                   help="allow a real run on benchmark items that are not yet verified")
    p.add_argument("--out-dir", type=Path, default=None, help="override output directory")
    p.add_argument("--show-histories", action="store_true", help="print one cell's history per arm")
    p.add_argument("--skip-judge", action="store_true",
                   help="run the subject arms only; the judge arm is saved as 'pending' for --judge-only")
    p.add_argument("--judge-only", action="store_true",
                   help="judge every cell whose J is pending, failed or from another judge model (no GPU)")
    p.add_argument("--item", default=None)
    p.add_argument("--condition", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--subject", default=None)
    return p.parse_args(argv)


def _select(all_names: list[str], wanted: list[str] | None, what: str) -> list[str]:
    if wanted is None:
        return all_names
    unknown = [w for w in wanted if w not in all_names]
    if unknown:
        raise ConfigError(f"unknown {what}: {unknown}; available: {all_names}")
    return wanted


def _real_run_problems(cfg: HarnessConfig, subjects: list[SubjectConfig], items: list[BenchmarkItem],
                       est: Estimate, args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    if not cfg.judge.model and not args.skip_judge:
        problems.append("judge.model is not set in config.yaml (or pass --skip-judge)")
    unconfirmed = [s.name for s in subjects if not s.confirmed]
    if unconfirmed:
        problems.append(f"subjects not confirmed (set confirmed: true after checking HF id/licence/fit): "
                        f"{unconfirmed}")
    unverified = sum(1 for i in items if not i.verified_by.strip())
    if unverified and not args.allow_unverified:
        problems.append(f"{unverified} benchmark items are unverified; review them or pass --allow-unverified")
    if not est.within_cap:
        problems.append(f"budget: {est.reason}")
    elif (est.cost_max_usd or 0.0) > 0 and not args.approve_spend:
        problems.append(f"judge spend up to ${est.cost_max_usd:.4f}: ask the user, then re-run with "
                        "--approve-spend")
    if problems:
        return problems
    from dotenv import load_dotenv

    load_dotenv()
    needed = [*([] if args.skip_judge else [cfg.judge.api_key_env]),
              *(s.api_key_env for s in subjects if s.api_key_env)]
    problems += [f"environment variable {name} is not set" for name in needed if not os.environ.get(name)]
    return problems


async def _real_execute(plan: RunPlan, usage: JudgeUsage) -> RunOutcome:
    cfg = plan.cfg
    judge_model = str(cfg.judge.model)
    judge_client = (None if plan.skip_judge else
                    OpenAIChatClient(cfg.judge.base_url, os.environ[cfg.judge.api_key_env]))
    clients: dict[str, ChatClient] = {}
    for s in plan.subjects:
        key = os.environ[s.api_key_env] if s.api_key_env else "EMPTY"
        client = OpenAIChatClient(s.base_url, key)
        try:
            served = await client.list_models()
        except Exception as exc:  # noqa: BLE001 - report and stop
            raise ConfigError(f"{s.name}: cannot reach {s.base_url}: {type(exc).__name__}") from exc
        if s.hf_id not in served:
            raise ConfigError(f"{s.name}: {s.base_url} serves {served}, not {s.hf_id}")
        clients[s.name] = client
    pin, pout = cfg.judge.price_in_per_1m, cfg.judge.price_out_per_1m
    prior = 0.0 if pin is None or pout is None else (usage.tokens_in * pin + usage.tokens_out * pout) / 1e6
    return await execute(
        plan, subject_client_for=lambda s: clients[s.name], judge_client=judge_client,
        judge_model=judge_model, retry=cfg.retry,
        judge_limiter=RateLimiter(cfg.judge.rate_limits, used_today_requests=usage.requests_today,
                                  used_today_tokens=usage.tokens_today),
        judge_spend=SpendTracker(cfg.max_spend_usd, pin, pout, spent_usd=prior),
    )


def print_summary(outcome: RunOutcome, results_path: Path, log_dir: Path) -> None:
    print(f"\n== Run finished: {len(outcome.rows)} new rows -> {results_path}; raw logs -> {log_dir}")
    for path in sorted(log_dir.glob("*.jsonl")):
        print(f"     {path.name:32s} {path.stat().st_size / 1_000_000:7.1f} MB")
    for warning in log_size_warnings(log_dir):
        print(f"   WARNING: {warning}. Split it further (e.g. per condition) before committing.")
    if outcome.stop_reason:
        print(f"   STOPPED EARLY: {outcome.stop_reason}  (completed cells are saved; re-run to resume)")
    if outcome.rows:
        print("   Status counts per arm in this run (data quality, not results):")
        for arm in ALL_ARMS:
            counts = Counter(r[f"status_{arm}"] for r in outcome.rows)
            print(f"     {arm:6s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def _main(args: argparse.Namespace, mock_factory: Callable[[Role], ChatClient] | None) -> int:
    cfg = load_config(args.config)
    check_judge_family(cfg.subjects, cfg.judge)

    if args.out_dir is not None:
        log_dir, results_path = args.out_dir / "trajectories", args.out_dir / "results.csv"
    elif args.dry_run:
        log_dir = cfg.paths["dry_run_dir"] / "trajectories"
        results_path = cfg.paths["dry_run_dir"] / "results.csv"
    else:
        log_dir, results_path = cfg.paths["raw_log_dir"], cfg.paths["results_csv"]

    if args.show_histories:
        return show_histories(log_dir, subject=args.subject, condition=args.condition,
                              item_id=args.item, sample=args.sample)
    if args.skip_judge and args.judge_only:
        raise ConfigError("--skip-judge and --judge-only cannot be combined")

    names = _select([s.name for s in cfg.subjects], _split(args.models), "models")
    subjects = [s for s in cfg.subjects if s.name in names]
    if args.judge_only:
        return judge_only(cfg, args, log_dir=log_dir, results_path=results_path, subjects=subjects,
                          mock_factory=mock_factory)
    conditions = _select(list(prompts.CONDITIONS), _split(args.conditions), "conditions")

    items, load_errors = load_items(cfg.paths["benchmark"])
    report = validate_items(items)
    if load_errors or not report.ok:
        raise ConfigError(f"benchmark has {len(load_errors) + len(report.errors)} error(s); "
                          "run `python -m src.benchmark validate`")
    items = sorted(items, key=lambda i: i.id)
    if args.limit is not None:
        items = items[: args.limit]
    turns = load_turns(cfg.paths["turns"])
    if not validate_turns(turns).ok:
        raise ConfigError("turns.yaml has errors; run `python -m src.benchmark validate`")

    samples = args.samples if args.samples is not None else cfg.samples_per_cell
    protocol = protocol_hash(cfg.paths["benchmark"], turns)
    done = read_done_keys(results_path, protocol)
    selected = {(s.name, c, i.id, k) for s in subjects for c in conditions for i in items for k in range(samples)}
    # Paid usage counts toward the cap wherever it was logged (main logs and any --out-dir logs).
    usage_dirs = {cfg.paths["raw_log_dir"].resolve(), log_dir.resolve()}
    parts = [read_judge_usage(d) for d in sorted(usage_dirs)]
    usage = JudgeUsage(*(sum(getattr(p, f) for p in parts)
                         for f in ("tokens_in", "tokens_out", "requests_today", "tokens_today")))
    est = estimate_run(cfg, items, len(selected), len(selected - done), usage,
                       judge_cells=0 if args.skip_judge else None)
    print_estimate(est, args.dry_run)
    if args.estimate:
        return 0 if est.within_cap else 2

    plan = RunPlan(cfg, subjects, conditions, items, turns, samples, log_dir, results_path,
                   args.dry_run, protocol, skip_judge=args.skip_judge)
    if args.dry_run:
        factory = mock_factory or (lambda role: MockChatClient(role))
        subject_clients: dict[str, ChatClient] = {}
        outcome = asyncio.run(execute(
            plan,
            subject_client_for=lambda s: subject_clients.setdefault(s.name, factory("subject")),
            judge_client=None if args.skip_judge else factory("judge"),
            judge_model=cfg.judge.model or "mock-judge",
            retry=replace(cfg.retry, initial_wait_s=0.0, max_wait_s=0.0),
            judge_limiter=None, judge_spend=None,
        ))
    else:
        problems = _real_run_problems(cfg, subjects, items, est, args)
        if problems:
            print("\nREFUSING TO START A REAL RUN:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        outcome = asyncio.run(_real_execute(plan, usage))

    print_summary(outcome, results_path, log_dir)
    return 0 if outcome.stop_reason is None else 3


def main(argv: list[str] | None = None, *, mock_factory: Callable[[Role], ChatClient] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    args = parse_args(argv)
    try:
        return _main(args, mock_factory)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
