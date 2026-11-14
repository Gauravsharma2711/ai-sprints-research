"""Offline tests for the harness: parsing, prose stripping, histories for every arm, retries,
caps, estimate, and resume. No network, no keys: all model calls go to MockChatClient."""

from __future__ import annotations

import asyncio
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import prompts
from src.clients import (
    BudgetExceeded,
    ChatRequest,
    ChatResponse,
    DailyCapReached,
    MockChatClient,
    RateLimiter,
    RateLimits,
    RetryPolicy,
    SpendTracker,
    TransientError,
    call_with_retries,
)
from src.harness import (
    ALL_ARMS,
    CellTurns,
    ConfigError,
    HistoryInputs,
    JsonlLog,
    MissingDependency,
    build_history,
    check_judge_family,
    iter_log_records,
    load_config,
    log_size_warnings,
    main,
)
from src.parsing import parse_probability, probe_p_yes, strip_prose

REPO = Path(__file__).resolve().parents[1]
NO_FAILURES = dict(parse_failure_rate=0.0, reask_failure_rate=0.0, missing_logprobs_rate=0.0, rate_limit_rate=0.0)


# ---------------------------------------------------------------------------
# Probability parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning.\nPROBABILITY_YES: 0.73", 0.73),
        ("PROBABILITY_YES: .4", 0.4),
        ("**PROBABILITY_YES:** 0.2", 0.2),
        ("**PROBABILITY_YES**: 0.25", 0.25),
        ("probability_yes = 0.9", 0.9),
        ("PROBABILITY_YES: 70%", 0.7),
        ("PROBABILITY_YES: <0.35>", 0.35),
        ("PROBABILITY_YES: 0.1\nOn reflection...\nPROBABILITY_YES: 0.6", 0.6),  # last one wins
        ("PROBABILITY_YES: 0.55.", 0.55),
    ],
)
def test_parse_probability_ok(text: str, expected: float) -> None:
    parsed = parse_probability(text)
    assert parsed.error is None and parsed.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "I think the answer is yes.", "PROBABILITY_YES: high", "PROBABILITY_YES: 1.5",
     "PROBABILITY_YES: 70", "PROBABILITY_RANGE: 0.4 to 0.6"],
)
def test_parse_probability_failures_return_none_not_a_default(text: str) -> None:
    parsed = parse_probability(text)
    assert parsed.value is None and parsed.error


# ---------------------------------------------------------------------------
# Prose stripping (judge input)
# ---------------------------------------------------------------------------


def test_strip_contract_header_and_probability_line() -> None:
    text = ("EVIDENCE_TIER: 2\nPROBABILITY_RANGE: 0.40 to 0.60\nWOULD_CHANGE_IF: a trial result\n\n"
            "I recommend option X because of Y.\n\nPROBABILITY_YES: 0.55")
    assert strip_prose(text) == "I recommend option X because of Y."


def test_strip_markdown_and_numbered_header_variants() -> None:
    text = ("**Evidence tier:** 2\n- **PROBABILITY_RANGE**: 0.4-0.6\n1. WOULD_CHANGE_IF: x\n"
            "Body text stays.\n**PROBABILITY_YES: 0.5**")
    assert strip_prose(text) == "Body text stays."


def test_strip_inline_probability_keeps_sentence() -> None:
    assert strip_prose("The case is weak. PROBABILITY_YES: 0.3") == "The case is weak."


def test_strip_leaves_plain_prose_untouched() -> None:
    text = "Line one.\n\nLine two with 70% confidence."
    assert strip_prose(text) == text


def test_strip_scrubs_model_names_on_word_boundaries() -> None:
    out = strip_prose("As Qwen2.5-7B, a Qwen model, I discuss philosophy.", ["Qwen2.5-7B", "qwen", "phi"])
    assert out == "As [model], a [model] model, I discuss philosophy."


# ---------------------------------------------------------------------------
# Logit probe
# ---------------------------------------------------------------------------


def test_probe_normalises_over_yes_and_no() -> None:
    top = [("Yes", math.log(0.6)), (" No", math.log(0.3)), ("The", math.log(0.1))]
    probe = probe_p_yes(top)
    assert probe.p_yes == pytest.approx(0.6 / 0.9) and probe.mass == pytest.approx(0.9)


def test_probe_merges_token_variants() -> None:
    top = [("Yes", math.log(0.4)), ("ĠYes", math.log(0.2)), ("no", math.log(0.2))]
    assert probe_p_yes(top).p_yes == pytest.approx(0.75)


@pytest.mark.parametrize(("top", "error"), [(None, "no_logprobs"), ([], "no_logprobs"),
                                            ([("EVIDENCE", -0.1)], "no_yes_no")])
def test_probe_nan_cases(top: Any, error: str) -> None:
    probe = probe_p_yes(top)
    assert probe.p_yes is None and probe.error == error


# ---------------------------------------------------------------------------
# History construction for every arm
# ---------------------------------------------------------------------------

TURNS = CellTurns(paraphrase_idx=0, bluff="BLUFF", release="RELEASE", release_no_bluff="FRESH",
                  neutral="NEUTRAL1", neutral_again="NEUTRAL2")
H = HistoryInputs(condition="default", claim="CLAIM?", evidence_turn="EVIDENCE", turns=TURNS,
                  r1="R1", c1=0.7, r2a="R2A", r2n="R2N")


def _u(t: str) -> dict[str, str]:
    return {"role": "user", "content": t}


def _a(t: str) -> dict[str, str]:
    return {"role": "assistant", "content": t}


SYS = {"role": "system", "content": prompts.system_prompt("default")}
T1_HIST = [SYS, _u("CLAIM?"), _a("R1")]
EXPECTED: dict[str, list[dict[str, str]]] = {
    "T1": [SYS, _u("CLAIM?")],
    "probe": [SYS, _u(prompts.probe_user_message("CLAIM?"))],
    "T2A": [*T1_HIST, _u("BLUFF")],
    "T2B": [*T1_HIST, _u("EVIDENCE")],
    "T2N": [*T1_HIST, _u("NEUTRAL1")],
    "T3": [*T1_HIST, _u("BLUFF"), _a("R2A"), _u("RELEASE")],
    "T3a": [*T1_HIST, _u("BLUFF"), _a(prompts.NEUTRAL_PLACEHOLDER), _u("RELEASE")],
    "T3b": [*T1_HIST, _u("BLUFF"), _a(prompts.stand_firm_reply(0.7)), _u("RELEASE")],
    "T3c": [*T1_HIST, _u("FRESH")],
    "T3N": [*T1_HIST, _u("NEUTRAL1"), _a("R2N"), _u("NEUTRAL2")],
}


@pytest.mark.parametrize("arm", list(EXPECTED))
def test_history_for_every_arm(arm: str) -> None:
    assert build_history(arm, H) == EXPECTED[arm]


def test_stand_firm_reply_restates_c1() -> None:
    assert "PROBABILITY_YES: 0.70" in build_history("T3b", H)[-2]["content"]


def test_t3c_has_no_bluff_and_no_bluff_referencing_release() -> None:
    contents = [m["content"] for m in build_history("T3c", H)]
    assert "BLUFF" not in contents and "RELEASE" not in contents


@pytest.mark.parametrize(
    ("arm", "missing"),
    [("T2A", {"r1": None}), ("T2B", {"r1": None}), ("T3", {"r2a": None}), ("T3N", {"r2n": None}),
     ("T3b", {"c1": float("nan")}), ("T3c", {"r1": None})],
)
def test_missing_dependencies_raise(arm: str, missing: dict[str, Any]) -> None:
    with pytest.raises(MissingDependency):
        build_history(arm, replace(H, **missing))


def test_contract_condition_uses_contract_system_prompt() -> None:
    hist = build_history("T1", replace(H, condition="contract_prompt"))
    assert "EVIDENCE_TIER" in hist[0]["content"] and "PROBABILITY_YES" in hist[0]["content"]


def test_paraphrase_selection_by_sample() -> None:
    turns = {k: [f"{k}0", f"{k}1", f"{k}2"] for k in ("bluff", "release", "release_no_bluff", "neutral")}
    t = CellTurns.for_sample(turns, 4)
    assert (t.paraphrase_idx, t.bluff, t.release, t.release_no_bluff, t.neutral, t.neutral_again) == (
        1, "bluff1", "release1", "release_no_bluff1", "neutral1", "neutral2")


# ---------------------------------------------------------------------------
# Judge family check
# ---------------------------------------------------------------------------


def test_repo_config_passes_family_check() -> None:
    cfg = load_config(REPO / "config.yaml")
    check_judge_family(cfg.subjects, cfg.judge)


@pytest.mark.parametrize(
    "judge_change",
    [{"family": "llama"}, {"family": "LLaMA "}, {"family": None}, {"model": "llama-3.3-70b-versatile"}],
)
def test_family_clash_fails_loudly(judge_change: dict[str, Any]) -> None:
    cfg = load_config(REPO / "config.yaml")
    with pytest.raises(ConfigError):
        check_judge_family(cfg.subjects, replace(cfg.judge, **judge_change))


def test_subject_without_family_fails() -> None:
    cfg = load_config(REPO / "config.yaml")
    subjects = [replace(cfg.subjects[0], family=None), *cfg.subjects[1:]]
    with pytest.raises(ConfigError):
        check_judge_family(subjects, cfg.judge)


# ---------------------------------------------------------------------------
# Retries, rate limits, spend
# ---------------------------------------------------------------------------

REQ = ChatRequest(model="m", messages=[_u("hi")], temperature=0.0, max_tokens=5)


def test_429_is_retried_and_every_attempt_logged() -> None:
    client = MockChatClient("judge", **{**NO_FAILURES, "rate_limit_rate": 1.0})
    attempts: list[tuple[int, bool, str | None]] = []

    def log(n: int, resp: ChatResponse | None, exc: BaseException | None, _lat: float) -> None:
        attempts.append((n, resp is not None, type(exc).__name__ if exc else None))

    resp = asyncio.run(call_with_retries(client, REQ, RetryPolicy(4, 0.0, 0.0), log))
    assert resp.text.startswith("PROBABILITY_YES")
    assert attempts == [(1, False, "RateLimited"), (2, True, None)]


def test_transient_errors_give_up_after_max_attempts() -> None:
    class Broken:
        base_url = "mock://broken"

        async def complete(self, req: ChatRequest) -> ChatResponse:
            raise TransientError("connection refused")

        async def list_models(self) -> list[str]:
            return []

    n_logged: list[int] = []
    with pytest.raises(TransientError):
        asyncio.run(call_with_retries(Broken(), REQ, RetryPolicy(3, 0.0, 0.0),
                                      lambda n, *_: n_logged.append(n)))
    assert n_logged == [1, 2, 3]


def test_rate_limiter_waits_when_per_minute_cap_is_hit() -> None:
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    async def go() -> None:
        lim = RateLimiter(RateLimits(requests_per_minute=2), clock=lambda: now[0], sleep=fake_sleep)
        for _ in range(3):
            await lim.acquire(10)

    asyncio.run(go())
    assert len(slept) == 1 and slept[0] == pytest.approx(60.01)


def test_rate_limiter_stops_at_daily_cap() -> None:
    lim = RateLimiter(RateLimits(requests_per_day=5), used_today_requests=5)
    with pytest.raises(DailyCapReached):
        asyncio.run(lim.acquire(1))


def test_spend_tracker_refuses_past_cap_and_without_prices() -> None:
    tracker = SpendTracker(cap_usd=0.001, price_in_per_1m=1.0, price_out_per_1m=1.0)
    with pytest.raises(BudgetExceeded):
        tracker.reserve(2000, 0)
    with pytest.raises(BudgetExceeded):
        SpendTracker(10.0, None, None).reserve(1, 1)


# ---------------------------------------------------------------------------
# End to end (dry run with mock clients)
# ---------------------------------------------------------------------------


class Factory:
    def __init__(self, **rates: float) -> None:
        self.rates = rates
        self.clients: list[MockChatClient] = []

    def __call__(self, role: str) -> MockChatClient:
        client = MockChatClient(role, **self.rates)  # type: ignore[arg-type]
        self.clients.append(client)
        return client

    @property
    def calls(self) -> int:
        return sum(c.calls for c in self.clients)


def write_config(tmp_path: Path, *, judge: dict[str, Any] | None = None, cap: float = 0.0) -> Path:
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"].update({
        "benchmark": str(REPO / "data/benchmark/items.jsonl"),
        "turns": str(REPO / "data/benchmark/turns.yaml"),
        "raw_log_dir": str(tmp_path / "real/trajectories"),
        "results_csv": str(tmp_path / "real/results.csv"),
        "dry_run_dir": str(tmp_path / "dry"),
    })
    raw["judge"].update(judge or {})
    raw["budget"]["max_spend_usd"] = cap
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def dry(tmp_path: Path, factory: Factory, *extra: str, cfg: Path | None = None) -> int:
    cfg = cfg or write_config(tmp_path)
    args = ["--config", str(cfg), "--dry-run", "--limit", "2", "--conditions", "default",
            "--samples", "1", "--models", "llama-3.1-8b", *extra]
    return main(args, mock_factory=factory)


def rows(tmp_path: Path) -> list[dict[str, str]]:
    with (tmp_path / "dry/results.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def log_records(tmp_path: Path) -> list[dict[str, Any]]:
    return list(iter_log_records(tmp_path / "dry/trajectories"))


def test_dry_run_writes_one_row_per_cell_and_logs_every_call(tmp_path: Path) -> None:
    factory = Factory(**NO_FAILURES)
    assert dry(tmp_path, factory) == 0
    out = rows(tmp_path)
    assert len(out) == 2 and all(r["dry_run"] == "True" for r in out)
    assert all(r[f"status_{a}"] == "ok" for r in out for a in ALL_ARMS)
    calls = [r for r in log_records(tmp_path) if r["event"] == "call"]
    assert len(calls) == factory.calls == 2 * len(ALL_ARMS)
    assert {c["arm"] for c in calls} == set(ALL_ARMS)


def test_raw_log_is_split_one_file_per_model(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    args = ["--config", str(cfg), "--dry-run", "--limit", "2", "--conditions", "default", "--samples", "1",
            "--models", "llama-3.1-8b,qwen2.5-7b"]
    assert main(args, mock_factory=Factory(**NO_FAILURES)) == 0
    log_dir = tmp_path / "dry/trajectories"
    assert sorted(p.name for p in log_dir.glob("*.jsonl")) == ["_runs.jsonl", "llama-3.1-8b.jsonl",
                                                                "qwen2.5-7b.jsonl"]
    for name in ("llama-3.1-8b", "qwen2.5-7b"):
        with (log_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        assert {r["subject"] for r in recs} == {name}
        assert {r["role"] for r in recs if r["event"] == "call"} == {"subject", "judge"}  # judge calls stay with their cell
    with (log_dir / "_runs.jsonl").open(encoding="utf-8") as fh:
        assert {json.loads(line)["event"] for line in fh} == {"run_start", "run_end"}


def test_reader_skips_a_line_cut_off_by_a_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_dir = tmp_path / "logs"
    log = JsonlLog(log_dir, "r1", dry_run=True)
    log.write("call", subject="m1", role="subject")
    log.close()
    with (log_dir / "m1.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"event": "call", "subj')  # truncated
    assert [r["event"] for r in iter_log_records(log_dir)] == ["call"]
    assert "skipped 1 unreadable line" in capsys.readouterr().err


def test_log_size_warning(tmp_path: Path) -> None:
    (tmp_path / "small.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "big.jsonl").write_bytes(b"x" * 2_000_000)
    warnings = log_size_warnings(tmp_path, warn_mb=1)
    assert len(warnings) == 1 and warnings[0].startswith("big.jsonl")


def test_resume_skips_completed_cells(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    assert dry(tmp_path, Factory(**NO_FAILURES), cfg=cfg) == 0
    first = rows(tmp_path)

    again = Factory(**NO_FAILURES)
    assert dry(tmp_path, again, cfg=cfg) == 0
    assert again.calls == 0 and rows(tmp_path) == first

    # Drop the second row (as if the run died mid-cell) and resume: only that cell reruns.
    path = tmp_path / "dry/results.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header, kept = reader.fieldnames, list(reader)[:1]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header or [])
        writer.writeheader()
        writer.writerows(kept)
    third = Factory(**NO_FAILURES)
    assert dry(tmp_path, third, cfg=cfg) == 0
    assert third.calls == len(ALL_ARMS)
    assert sorted(r["item_id"] for r in rows(tmp_path)) == sorted(r["item_id"] for r in first)


def test_resume_refuses_rows_from_a_different_protocol(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    assert dry(tmp_path, Factory(**NO_FAILURES), cfg=cfg) == 0
    path = tmp_path / "dry/results.csv"
    text = path.read_text(encoding="utf-8")
    old_hash = rows(tmp_path)[0]["protocol_hash"]
    path.write_text(text.replace(old_hash, "000000000000"), encoding="utf-8")
    assert dry(tmp_path, Factory(**NO_FAILURES), cfg=cfg) == 2


def test_parse_failures_get_one_reask_then_nan(tmp_path: Path) -> None:
    factory = Factory(parse_failure_rate=1.0, reask_failure_rate=1.0, missing_logprobs_rate=0.0,
                      rate_limit_rate=0.0)
    assert dry(tmp_path, factory) == 0
    for row in rows(tmp_path):
        for arm, col in [("T1", "c1"), ("T2A", "c2A"), ("T3", "c3"), ("judge", "J")]:
            assert row[f"status_{arm}"] == "nan_parse" and row[col] == "nan"
        assert row["status_T3b"] == "nan_dependency"  # stand-firm reply needs c1
    records = log_records(tmp_path)
    parse_errors = [r for r in records if r["event"] == "parse_error"]
    reasks = [r for r in records if r["event"] == "call" and r["purpose"] == "reask"]
    # 9 parsed arms per cell (8 subject + judge; T3b skipped), primary + one re-ask each.
    assert len(reasks) == 2 * 9 and len(parse_errors) == 2 * 9 * 2


def test_judge_429_is_retried_in_a_run(tmp_path: Path) -> None:
    factory = Factory(**{**NO_FAILURES, "rate_limit_rate": 1.0})
    assert dry(tmp_path, factory) == 0
    assert all(r["status_judge"] == "ok" for r in rows(tmp_path))
    judge_calls = [r for r in log_records(tmp_path) if r["event"] == "call" and r["role"] == "judge"]
    assert any(c["error"] and "RateLimited" in c["error"] for c in judge_calls)
    assert any(c["attempt"] == 2 and c["error"] is None for c in judge_calls)


def test_missing_logprobs_make_probe_nan(tmp_path: Path) -> None:
    assert dry(tmp_path, Factory(**{**NO_FAILURES, "missing_logprobs_rate": 1.0})) == 0
    assert all(r["L"] == "nan" and r["status_probe"] == "nan_no_logprobs" for r in rows(tmp_path))


def test_judge_sees_only_question_and_stripped_prose(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    args = ["--config", str(cfg), "--dry-run", "--limit", "2", "--conditions", "contract_prompt",
            "--samples", "1", "--models", "qwen2.5-7b"]
    assert main(args, mock_factory=Factory(**NO_FAILURES)) == 0
    judge_reqs = [r["request"] for r in log_records(tmp_path)
                  if r["event"] == "call" and r["role"] == "judge" and r["purpose"] == "primary"]
    assert judge_reqs
    for req in judge_reqs:
        assert set(req) == {"model", "messages", "temperature", "max_tokens"}
        assert req["temperature"] == 0.0
        briefing = req["messages"][1]["content"].split("<<<")[1].split(">>>")[0]
        assert "PROBABILITY_YES" not in briefing and "EVIDENCE_TIER" not in briefing
        assert "WOULD_CHANGE_IF" not in briefing and "PROBABILITY_RANGE" not in briefing
        assert "qwen" not in json.dumps(req).lower()


def test_show_histories_prints_every_arm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    assert dry(tmp_path, Factory(**NO_FAILURES), cfg=cfg) == 0
    capsys.readouterr()
    assert main(["--config", str(cfg), "--dry-run", "--show-histories"]) == 0
    out = capsys.readouterr().out
    assert all(f"= {arm} =" in out for arm in ALL_ARMS)


# ---------------------------------------------------------------------------
# Estimate and refusal to start real runs
# ---------------------------------------------------------------------------


def test_estimate_refuses_without_prices(tmp_path: Path) -> None:
    assert main(["--config", str(write_config(tmp_path)), "--estimate", "--limit", "2"]) == 2
    assert not (tmp_path / "real").exists()


def test_estimate_refuses_over_cap_and_passes_under(tmp_path: Path) -> None:
    prices = {"price_per_1m_input_usd": 1.0, "price_per_1m_output_usd": 2.0}
    assert main(["--config", str(write_config(tmp_path, judge=prices, cap=0.0)), "--estimate"]) == 2
    assert main(["--config", str(write_config(tmp_path, judge=prices, cap=1000.0)), "--estimate"]) == 0


def test_spend_logged_in_out_dir_counts_toward_cap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    smoke_logs = tmp_path / "smoke/trajectories"
    log = JsonlLog(smoke_logs, "r0", dry_run=False)
    log.write("call", subject="llama-3.1-8b", role="judge",
              usage={"prompt_tokens": 1_000_000, "completion_tokens": 0})
    log.close()
    prices = {"price_per_1m_input_usd": 1.0, "price_per_1m_output_usd": 1.0}
    cfg = str(write_config(tmp_path, judge=prices, cap=1000.0))
    assert main(["--config", cfg, "--estimate", "--limit", "1"]) == 0
    assert "already spent $0.0000" in capsys.readouterr().out
    assert main(["--config", cfg, "--estimate", "--limit", "1", "--out-dir", str(tmp_path / "smoke")]) == 0
    assert "already spent $1.0000" in capsys.readouterr().out


def test_real_run_refuses_before_any_call(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    factory = Factory(**NO_FAILURES)
    rc = main(["--config", str(write_config(tmp_path)), "--limit", "1"], mock_factory=factory)
    err = capsys.readouterr().err
    assert rc == 2 and factory.calls == 0 and not (tmp_path / "real").exists()
    assert "judge.model is not set" in err and "not confirmed" in err and "unverified" in err
