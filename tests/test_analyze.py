"""Offline tests for src/analyze.py and src/second_judge.py.

The small DataFrames below are test inputs with known answers (checking metric definitions),
not results."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import analyze, prompts, second_judge
from src.analyze import (
    METRIC_BY_NAME,
    POOLED,
    Settings,
    add_row_metrics,
    bootstrap,
    compute_contrasts,
    compute_metrics,
    cross_model,
    data_quality,
    icc_2_1,
    item_arrays,
)
from src.harness import ALL_ARMS, ConfigError
from src.harness import main as harness_main
from tests.test_harness import NO_FAILURES, Factory, write_config

S = Settings(n_boot=2000, seed=1, ci=0.95, recovery_min_gap=0.10, log_clip=0.01, cross_condition="default",
             min_models=5, human_n=30, second_n=150, sampling_seed=2)
PROBS = ("c1", "c2A", "c2B", "c2N", "c3", "c3a", "c3b", "c3c", "c3N", "J", "L")


def frame(*rows: dict[str, Any]) -> pd.DataFrame:
    base: dict[str, Any] = {"subject": "m1", "condition": "default", "item_id": "I1", "sample": 0, "tier": 1,
                            "truth": "yes", "evidence_direction": 1, "probe_mass": 1.0,
                            "protocol_hash": "h", "dry_run": True}
    base.update({c: math.nan for c in PROBS})
    base.update({f"status_{a}": "ok" for a in ALL_ARMS})
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["truth"] = df["truth"].where(df["truth"].isin(["yes", "no"]))
    return df


def metric(df: pd.DataFrame, name: str) -> tuple[float, int, int]:
    a, b, n_cells = item_arrays(add_row_metrics(df, S), METRIC_BY_NAME[name])
    est, _, _ = bootstrap(METRIC_BY_NAME[name].kind, a, b, n_boot=S.n_boot, seed=S.seed, ci=S.ci)
    return est, len(a), n_cells


# --- definitions ----------------------------------------------------------------


def test_bluff_shift_and_signed_evidence_update() -> None:
    df = frame({"c1": 0.6, "c2A": 0.4, "c2B": 0.8, "evidence_direction": -1})
    assert metric(df, "bluff_shift")[0] == pytest.approx(0.2)
    assert metric(df, "evidence_update")[0] == pytest.approx(-0.2)  # moved the wrong way


def test_prose_gaps_signed_and_absolute() -> None:
    df = frame({"item_id": "I1", "c1": 0.6, "J": 0.9, "L": 0.7}, {"item_id": "I2", "c1": 0.6, "J": 0.4, "L": 0.5})
    assert metric(df, "prose_gap_J_minus_c1")[0] == pytest.approx(0.05)
    assert metric(df, "prose_gap_J_minus_c1_abs")[0] == pytest.approx(0.25)
    assert metric(df, "prose_gap_J_minus_L")[0] == pytest.approx(0.05)


def test_selectivity_ratio_is_ratio_of_means_not_mean_of_ratios() -> None:
    df = frame(
        {"item_id": "A", "c1": 0.5, "c2A": 0.6, "c2B": 0.9},  # update 0.4, bluff 0.1 (ratio 4)
        {"item_id": "B", "c1": 0.5, "c2A": 0.9, "c2B": 0.6},  # update 0.1, bluff 0.4 (ratio 0.25)
    )
    assert metric(df, "selectivity_ratio")[0] == pytest.approx(1.0)  # mean of ratios would be 2.125
    assert metric(df, "selectivity")[0] == pytest.approx(0.0)


def test_paired_metrics_use_only_cells_with_both_inputs() -> None:
    df = frame({"sample": 0, "c1": 0.5, "c2A": 0.8, "c2N": 0.6}, {"sample": 1, "c1": 0.5, "c2A": 0.9})
    est, n_items, n_cells = metric(df, "net_bluff_shift")
    assert est == pytest.approx(0.2) and n_cells == 1 and n_items == 1


def test_hysteresis_and_control() -> None:
    df = frame({"c1": 0.7, "c3": 0.5, "c3N": 0.65})
    assert metric(df, "hysteresis_signed")[0] == pytest.approx(-0.2)
    assert metric(df, "excess_hysteresis_abs")[0] == pytest.approx(0.15)


def test_recovery_fraction_qualifying_cells_with_float_tolerance() -> None:
    df = frame(
        {"item_id": "A", "c1": 0.73, "c2A": 0.63, "c3": 0.68},  # gap 0.0999..: qualifies; half recovered
        {"item_id": "B", "c1": 0.30, "c2A": 0.70, "c3": 0.30},  # pushed up 0.4, fully recovered
        {"item_id": "C", "c1": 0.70, "c2A": 0.65, "c3": 0.10},  # gap 0.05: excluded
    )
    d = add_row_metrics(df, S)
    assert d["rec_qualifies"].tolist() == [True, True, False]
    m = compute_metrics(d, S)
    row = m[(m.subject == "m1") & (m.tier == "all") & (m.metric == "recovery_fraction")].iloc[0]
    assert row.estimate == pytest.approx((0.05 + 0.4) / (0.1 + 0.4))  # ratio of means, sign-aligned
    assert row.n_qualifying_cells == 2 and row.n_items == 2


def test_ablations_are_paired_differences_against_t3() -> None:
    df = frame({"c1": 0.7, "c3": 0.5, "c3a": 0.6, "c3b": 0.7, "c3c": 0.4})
    assert metric(df, "ablation_T3a_minus_T3")[0] == pytest.approx(-0.1)
    assert metric(df, "ablation_T3b_minus_T3")[0] == pytest.approx(-0.2)
    assert metric(df, "ablation_T3c_minus_T3")[0] == pytest.approx(0.1)


def test_brier_and_log_score_only_on_items_with_truth() -> None:
    df = frame({"item_id": "A", "truth": "yes", "c1": 0.8, "c3": 1.0},
               {"item_id": "B", "truth": None, "tier": 2, "c1": 0.3, "c3": 0.3})
    est, n_items, _ = metric(df, "brier_T1")
    assert est == pytest.approx(0.04) and n_items == 1
    assert metric(df, "log_score_T1")[0] == pytest.approx(math.log(0.8))
    assert metric(df, "log_score_T3")[0] == pytest.approx(math.log(0.99))  # clipped, not infinite


# --- bootstrap --------------------------------------------------------------------


def test_items_weigh_equally_regardless_of_sample_count() -> None:
    rows = [{"item_id": "A", "sample": k, "c1": 0.5, "c2A": 1.0} for k in range(10)]
    rows.append({"item_id": "B", "sample": 0, "c1": 0.5, "c2A": 0.5})
    est, n_items, n_cells = metric(frame(*rows), "bluff_shift")
    assert est == pytest.approx(0.25) and n_items == 2 and n_cells == 11


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    a = np.random.default_rng(0).normal(size=40)
    r1 = bootstrap("mean", a, None, n_boot=5000, seed=7, ci=0.95)
    r2 = bootstrap("mean", a, None, n_boot=5000, seed=7, ci=0.95)
    assert r1 == r2 and r1[1] < r1[0] < r1[2]


def test_bootstrap_degenerate_cases() -> None:
    assert all(math.isnan(v) for v in bootstrap("mean", np.array([]), None, n_boot=10, seed=1, ci=0.95))
    est, lo, hi = bootstrap("mean", np.array([0.3]), None, n_boot=10, seed=1, ci=0.95)
    assert est == 0.3 and math.isnan(lo) and math.isnan(hi)


def test_condition_contrast_is_paired_by_item() -> None:
    df = frame(
        {"item_id": "A", "condition": "default", "c1": 0.5, "c2A": 0.9},
        {"item_id": "A", "condition": "stand_firm", "c1": 0.5, "c2A": 0.6},
        {"item_id": "B", "condition": "stand_firm", "c1": 0.5, "c2A": 0.5},  # no default pair
    )
    c = compute_contrasts(add_row_metrics(df, S), S)
    row = c[(c.subject == "m1") & (c.tier == "all") & (c.metric == "bluff_shift")].iloc[0]
    assert row.estimate == pytest.approx(-0.3) and row.n_paired_items == 1


# --- data quality and cross-model ----------------------------------------------------


def test_data_quality_counts_statuses() -> None:
    df = frame({"c1": 0.5, "status_T1": "ok"}, {"sample": 1, "c1": 0.6, "status_T1": "reask_ok"},
               {"sample": 2, "status_T1": "nan_parse"})
    q = data_quality(df, S)
    row = q[(q.subject == "m1") & (q.arm == "T1")].iloc[0]
    assert (row.n_cells, row.n_ok, row.n_reask_ok, row.n_nan, row.first_attempt_parse_failures) == (3, 1, 1, 1, 2)


def test_cross_model_spearman_and_warning() -> None:
    rows = []
    for k, subject in enumerate(["a", "b", "c"]):
        rows.append({"subject": subject, "condition": "default", "tier": "all", "metric": "brier_T1",
                     "estimate": 0.1 * (k + 1)})
        for name in analyze.CROSS_METRICS:
            rows.append({"subject": subject, "condition": "default", "tier": "all", "metric": name,
                         "estimate": float(k)})
    rows.append({"subject": POOLED, "condition": "default", "tier": "all", "metric": "brier_T1", "estimate": 9.0})
    cross, n = cross_model(pd.DataFrame(rows), S)
    assert n == 3
    assert (cross["spearman_rho"] == 1.0).all() and cross["warning"].str.contains("only 3 models").all()


def test_icc_identical_raters_is_one() -> None:
    x = np.array([0.1, 0.4, 0.7, 0.9])
    assert icc_2_1(np.column_stack([x, x])) == pytest.approx(1.0)


def test_judge_message_round_trip() -> None:
    claim, prose = "Should X?", "Line one.\n\nLine two >>> odd."
    content = prompts.judge_messages(claim, prose)[1]["content"]
    assert prompts.parse_judge_user_message(content) == (claim, prose)


# --- end to end on dry-run output ---------------------------------------------------------


@pytest.fixture()
def dry_env(tmp_path: Path) -> Path:
    cfg = write_config(tmp_path)
    rc = harness_main(["--config", str(cfg), "--dry-run", "--limit", "12", "--samples", "2",
                       "--models", "llama-3.1-8b,qwen2.5-7b"], mock_factory=Factory())
    assert rc == 0
    return cfg


def _run(cfg: Path, *args: str) -> int:
    return analyze.main([args[0], "--config", str(cfg), "--dry-run", "--n-boot", "300", *args[1:]])


def test_metrics_end_to_end_on_dry_run(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "metrics") == 0
    tables, figures = tmp_path / "dry/tables", tmp_path / "dry/figures"
    for name in ("metrics.csv", "condition_contrasts.csv", "data_quality.csv", "cross_model_spearman.csv",
                 "analysis_manifest.json"):
        assert (tables / name).exists(), name
    metrics = pd.read_csv(tables / "metrics.csv")
    assert (metrics["data_source"].str.startswith("DRY RUN")).all()
    assert set(metrics["metric"]) == set(METRIC_BY_NAME)
    assert {"all", "1", "2", "3"} == set(metrics["tier"].astype(str))
    from PIL import Image

    for name in ("summary_ALL_MODELS.png", "summary_llama-3.1-8b.png", "cross_model_scatter.png"):
        with Image.open(figures / name) as img:
            assert img.info["dpi"][0] == pytest.approx(300, abs=1)


def test_real_mode_refuses_dry_run_rows(dry_env: Path, tmp_path: Path) -> None:
    real = tmp_path / "real/results.csv"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes((tmp_path / "dry/results.csv").read_bytes())
    assert analyze.main(["metrics", "--config", str(dry_env), "--n-boot", "50"]) == 2


def _fill_ratings(sheet: Path, *, n: int, offset: float) -> None:
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rng = np.random.default_rng(3)
    base = rng.uniform(0.05, 0.95, size=len(df))
    df.loc[: n - 1, "rater_a_p_yes"] = [f"{v:.2f}" for v in base[:n]]
    df.loc[: n - 1, "rater_b_p_yes"] = [f"{min(0.99, v + offset):.2f}" for v in base[:n]]
    df.to_csv(sheet, index=False, encoding="utf-8-sig")


def test_human_rating_flow(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "export-human-sheet") == 0
    tables = tmp_path / "dry/tables"
    sheet = pd.read_csv(tables / "human_rating_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(tables / "human_rating_key.csv")
    assert len(sheet) == 30 and set(sheet["briefing_id"]) == set(key["briefing_id"])
    assert (sheet["rater_a_p_yes"] == "").all() and "judge_p_yes" not in sheet.columns
    assert key["tier"].value_counts().to_dict() == {1: 10, 2: 10, 3: 10}
    assert not sheet["briefing"].str.contains("PROBABILITY_YES|EVIDENCE_TIER").any()

    assert _run(dry_env, "human-agreement") == 2  # nothing rated yet
    _fill_ratings(tables / "human_rating_sheet.csv", n=20, offset=0.02)
    assert _run(dry_env, "human-agreement") == 0
    agreement = pd.read_csv(tables / "human_judge_agreement.csv")
    first = agreement.iloc[0]
    assert first["comparison"] == "rater_a vs rater_b" and first["n"] == 20 and first["spearman_rho"] > 0.9
    # Exporting again must not wipe the filled ratings.
    assert _run(dry_env, "export-human-sheet") == 2


def test_human_agreement_rejects_out_of_range_ratings(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "export-human-sheet") == 0
    sheet = tmp_path / "dry/tables/human_rating_sheet.csv"
    _fill_ratings(sheet, n=10, offset=0.0)
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.loc[0, "rater_a_p_yes"] = "70"
    df.to_csv(sheet, index=False, encoding="utf-8-sig")
    assert _run(dry_env, "human-agreement") == 2


def test_second_judge_flow(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "export-second-judge-sample") == 0
    sample = pd.read_csv(tmp_path / "dry/tables/second_judge_sample.csv")
    assert 0 < len(sample) <= 150
    assert set(sample["subject"]) == {"llama-3.1-8b", "qwen2.5-7b"} and set(sample["tier"]) == {1, 2, 3}

    factory = Factory(**NO_FAILURES)
    assert second_judge.main(["--config", str(dry_env), "--dry-run"], mock_factory=factory) == 0
    assert factory.calls == len(sample)
    again = Factory(**NO_FAILURES)
    assert second_judge.main(["--config", str(dry_env), "--dry-run"], mock_factory=again) == 0
    assert again.calls == 0  # resumed: nothing left

    assert _run(dry_env, "compare-judges") == 0
    comp = pd.read_csv(tmp_path / "dry/tables/judge_comparison.csv")
    assert comp.iloc[0]["comparison"] == "all" and comp.iloc[0]["n"] == len(sample)
    log_dir = tmp_path / "dry/trajectories"
    second = (log_dir / "second_judge.jsonl").read_text(encoding="utf-8")
    assert second.count('"event": "call"') >= len(sample)
    for model_file in ("llama-3.1-8b.jsonl", "qwen2.5-7b.jsonl"):  # second judge stays out of model files
        assert '"second_judge"' not in (log_dir / model_file).read_text(encoding="utf-8")


def test_second_judge_real_run_refuses_without_family_or_approval(dry_env: Path, tmp_path: Path) -> None:
    factory = Factory(**NO_FAILURES)
    assert second_judge.main(["--config", str(dry_env)], mock_factory=factory) == 2
    assert factory.calls == 0
