"""Offline tests for src/analyze.py and src/second_judge.py.

The small DataFrames below are test inputs with known answers (checking metric definitions),
not results."""

from __future__ import annotations

import csv
import json
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
    for name in ("metrics.csv", "condition_contrasts.csv", "data_quality.csv", "coverage.csv",
                 "analysis_manifest.json", "metrics.md", "condition_contrasts.md", "data_quality.md",
                 "coverage.md"):
        assert (tables / name).exists(), name
    # Cross-model outputs only with --cross-model.
    assert not (tables / "cross_model_spearman.csv").exists()
    assert not (figures / "cross_model_scatter.png").exists()
    metrics = pd.read_csv(tables / "metrics.csv", dtype={"samples": str})
    assert (metrics["data_source"].str.startswith("DRY RUN")).all()
    assert set(metrics["metric"]) == set(METRIC_BY_NAME)
    assert {"all", "1", "2", "3"} == set(metrics["tier"].astype(str))
    assert (metrics["samples"] == "0").all() and metrics["exclusions"].str.startswith("applied").all()
    contrasts = pd.read_csv(tables / "condition_contrasts.csv", dtype={"samples": str})
    assert {"n_items", "n_cells", "samples", "exclusions"} <= set(contrasts.columns)
    cov = pd.read_csv(tables / "coverage.csv")
    assert list(cov.columns) == list(analyze.COVERAGE_COLUMNS)
    assert set(cov["sample"]) == {0, 1}  # coverage lists every sample, before the sample policy
    manifest = json.loads((tables / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"]["samples"] == "0" and manifest["cross_model"] is False
    assert "DRY RUN" in (tables / "metrics.md").read_text(encoding="utf-8")
    from PIL import Image

    for name in ("summary_ALL_MODELS.png", "summary_llama-3.1-8b.png"):
        with Image.open(figures / name) as img:
            assert img.info["dpi"][0] == pytest.approx(300, abs=1)

    out = tmp_path / "robust"
    assert _run(dry_env, "metrics", "--all-samples", "--cross-model", "--out-dir", str(out)) == 0
    robust = pd.read_csv(out / "tables/metrics.csv", dtype={"samples": str})
    assert (robust["samples"] == "all").all()
    assert (out / "tables/cross_model_spearman.csv").exists()
    with Image.open(out / "figures/cross_model_scatter.png") as img:
        assert img.info["dpi"][0] == pytest.approx(300, abs=1)
    n_all = robust[(robust.subject == POOLED) & (robust.tier == "all") & (robust.metric == "bluff_shift")
                   & (robust.condition == "default")].iloc[0]["n_cells"]
    n_primary = metrics[(metrics.subject == POOLED) & (metrics.tier == "all") & (metrics.metric == "bluff_shift")
                        & (metrics.condition == "default")].iloc[0]["n_cells"]
    assert n_all > n_primary


def test_exclusions_end_to_end(dry_env: Path, tmp_path: Path) -> None:
    excl = tmp_path / "data/benchmark/exclusions.csv"  # analysis.exclusions_csv, relative to the config
    excl.parent.mkdir(parents=True, exist_ok=True)
    excl.write_text("item_id,reason\nEH001,test exclusion\n", encoding="utf-8")
    assert _run(dry_env, "metrics") == 0
    tables = tmp_path / "dry/tables"
    assert (pd.read_csv(tables / "coverage.csv")["n_items"] == 11).all()  # 12 dry-run items, 1 excluded
    manifest = json.loads((tables / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"]["exclusions"]["items"] == {"EH001": "test exclusion"}
    metrics = pd.read_csv(tables / "metrics.csv")
    assert metrics["exclusions"].eq("applied (1 items)").all()
    excluded_n = metrics[(metrics.subject == POOLED) & (metrics.tier == "all") & (metrics.metric == "bluff_shift")
                         & (metrics.condition == "default")].iloc[0]["n_items"]
    out = tmp_path / "noexcl"
    assert _run(dry_env, "metrics", "--no-exclusions", "--out-dir", str(out)) == 0
    full = pd.read_csv(out / "tables/metrics.csv")
    assert full["exclusions"].str.startswith("not applied").all()
    full_n = full[(full.subject == POOLED) & (full.tier == "all") & (full.metric == "bluff_shift")
                  & (full.condition == "default")].iloc[0]["n_items"]
    assert full_n == excluded_n + 1


def test_hypotheses_end_to_end_on_dry_run(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "hypotheses") == 0
    tables = tmp_path / "dry/tables"
    table = pd.read_csv(tables / "hypotheses.csv", dtype={"samples": str, "tier": str})
    assert list(table.columns) == list(analyze.HYPOTHESIS_COLUMNS)
    assert len(table) == len(analyze.HYPOTHESES) * 3  # two models + ALL_MODELS
    assert set(table["verdict"]) <= {"supported", "opposite", "inconclusive", "no data"}
    assert (table["samples"] == "0").all() and (tables / "hypotheses.md").exists()
    assert (tables / "hypotheses_manifest.json").exists()


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


# --- unbalanced data: different samples / items per model, pending judge --------------------


def test_pooled_rows_weigh_models_equally_within_an_item() -> None:
    rows = [{"subject": "m1", "item_id": "A", "sample": k, "c1": 0.5, "c2A": 1.0} for k in range(3)]
    rows.append({"subject": "m2", "item_id": "A", "sample": 0, "c1": 0.5, "c2A": 0.5})
    m = compute_metrics(add_row_metrics(frame(*rows), S), S)
    row = m[(m.subject == POOLED) & (m.tier == "all") & (m.metric == "bluff_shift")].iloc[0]
    # model means 0.5 and 0.0 -> 0.25 (pooling cells first would give 0.375)
    assert row.estimate == pytest.approx(0.25) and row.n_items == 1 and row.n_cells == 4


def test_pooled_contrast_pairs_items_within_each_model() -> None:
    df = frame(
        {"subject": "m1", "item_id": "A", "condition": "default", "c1": 0.5, "c2A": 0.9},
        {"subject": "m1", "item_id": "A", "condition": "stand_firm", "c1": 0.5, "c2A": 0.6},
        {"subject": "m2", "item_id": "A", "condition": "default", "c1": 0.5, "c2A": 0.5},  # no stand_firm
        {"subject": "m2", "item_id": "B", "condition": "stand_firm", "c1": 0.5, "c2A": 0.7},  # no default
    )
    c = compute_contrasts(add_row_metrics(df, S), S)
    pooled = c[(c.subject == POOLED) & (c.tier == "all") & (c.metric == "bluff_shift")].iloc[0]
    assert pooled.estimate == pytest.approx(-0.3)  # only m1's pair; m2 has no pair
    assert pooled.n_items == 1 and pooled.n_paired_items == 1 and pooled.n_cells == 2
    m2 = c[(c.subject == "m2") & (c.tier == "all") & (c.metric == "bluff_shift")].iloc[0]
    assert m2.n_items == 0 and math.isnan(m2.estimate)


def _unbalanced() -> pd.DataFrame:
    """m1: samples 0-2 (judge pending on 1-2), three conditions, contract_prompt on 3 items only.
    m2: sample 0 only, no contract_prompt."""
    rows: list[dict[str, Any]] = []
    tiers = {f"I{k}": t for k, t in enumerate((1, 2, 3, 1, 2, 3), start=1)}
    for subject, samples in (("m1", (0, 1, 2)), ("m2", (0,))):
        for cond in ("default", "stand_firm", "contract_prompt"):
            if subject == "m2" and cond == "contract_prompt":
                continue
            for k, (item, tier) in enumerate(tiers.items()):
                if cond == "contract_prompt" and k >= 3:
                    continue
                for sample in samples:
                    judged = sample == 0
                    rows.append({
                        "subject": subject, "condition": cond, "item_id": item, "tier": tier, "sample": sample,
                        "c1": 0.6, "c2N": 0.61, "c3": 0.7, "c3N": 0.6, "c3a": 0.7, "c3b": 0.7, "c3c": 0.7,
                        "c2A": (0.8 + 0.01 * k) if cond == "default" else 0.65,
                        "c2B": 0.75 if cond == "stand_firm" else 0.7,
                        "J": 0.9 if judged else math.nan, "status_judge": "ok" if judged else "pending",
                    })
    return frame(*rows)


def _settings(**kw: Any) -> Settings:
    return Settings(**{**analyze.asdict(S), "n_boot": 500, **kw})


def test_sample_policy_and_scope_labels() -> None:
    df = _unbalanced()
    s = _settings()
    primary, label = analyze.apply_samples(df, s, all_samples=False)
    assert label == "0" and set(primary["sample"]) == {0} and set(primary["subject"]) == {"m1", "m2"}
    every, label_all = analyze.apply_samples(df, s, all_samples=True)
    assert label_all == "all" and len(every) == len(df)
    _, two = analyze.apply_samples(df, _settings(primary_samples=(0, 1)), all_samples=False)
    assert two == "0,1"
    _, analysed, scope = analyze.apply_scope(df, s, all_samples=False, use_exclusions=True)
    assert scope.samples == "0" and scope.n_cells_analysed == len(analysed) < scope.n_cells_loaded
    assert scope.manifest()["exclusions"]["applied"] is True


def test_exclusions_file_handling(tmp_path: Path) -> None:
    path = tmp_path / "exclusions.csv"
    assert analyze.load_exclusions("")[0] == {}
    assert analyze.load_exclusions(str(path))[0] == {}  # missing file: no exclusions
    path.write_text("item_id,reason\n", encoding="utf-8")
    assert analyze.load_exclusions(str(path))[0] == {}  # header only
    path.write_text("item_id,reason\nI2,broken evidence turn\n", encoding="utf-8")
    df = _unbalanced()
    s = _settings(exclusions_csv=str(path))
    _, analysed, scope = analyze.apply_scope(df, s, all_samples=True, use_exclusions=True)
    assert "I2" not in set(analysed["item_id"]) and scope.exclusions == "applied (1 items)"
    _, kept, scope2 = analyze.apply_scope(df, s, all_samples=True, use_exclusions=False)
    assert "I2" in set(kept["item_id"]) and scope2.exclusions.startswith("not applied")
    for bad in ("id,why\nI2,x\n", "item_id,reason\nI2,\n", "item_id,reason\nI2,a\nI2,b\n"):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ConfigError):
            analyze.load_exclusions(str(path))


def test_coverage_counts_unbalanced_and_pending() -> None:
    cov = analyze.coverage(_unbalanced())
    assert list(cov.columns) == list(analyze.COVERAGE_COLUMNS)
    # m1: 3 conditions x 3 samples; m2: 2 conditions x 1 sample
    assert len(cov) == 9 + 2
    row = cov[(cov.subject == "m1") & (cov.condition == "contract_prompt") & (cov["sample"] == 1)].iloc[0]
    assert (row.n_items, row.n_cells, row.n_judged) == (3, 3, 0)
    row0 = cov[(cov.subject == "m1") & (cov.condition == "default") & (cov["sample"] == 0)].iloc[0]
    assert (row0.n_items, row0.n_cells, row0.n_judged) == (6, 6, 6)


def test_data_quality_per_model_and_arm_with_pending_judge() -> None:
    df = _unbalanced()
    df.loc[0, "status_T1"] = "reask_ok"
    df.loc[1, "status_T1"] = "nan_parse"
    q = data_quality(df, S)
    judge = q[(q.subject == "m1") & (q.condition == "all") & (q.arm == "judge")].iloc[0]
    n_m1 = int((df.subject == "m1").sum())
    assert judge.n_cells == n_m1 and judge.n_pending == n_m1 * 2 // 3 and judge.n_attempted == n_m1 // 3
    assert judge.ok_rate == pytest.approx(1.0) and judge.nan_rate_attempted == pytest.approx(0.0)
    t1 = q[(q.subject == "m1") & (q.condition == "all") & (q.arm == "T1")].iloc[0]
    assert t1.n_pending == 0 and t1.reask_ok_rate == pytest.approx(1 / n_m1)
    assert t1.nan_rate_attempted == pytest.approx(1 / n_m1)
    assert {"default", "stand_firm", "contract_prompt", "all"} == set(q["condition"])


def test_verdict_rules() -> None:
    v = analyze.verdict
    assert v(0.2, 0.1, 0.3, ">0") == "supported" and v(-0.2, -0.3, -0.1, ">0") == "opposite"
    assert v(0.1, -0.1, 0.3, ">0") == "inconclusive" and v(-0.2, -0.3, -0.1, "<0") == "supported"
    assert v(0.2, 0.1, 0.3, "<0") == "opposite" and v(-0.2, -0.3, -0.1, "!=0") == "supported"
    assert v(0.0, -0.1, 0.1, "!=0") == "inconclusive" and v(math.nan, math.nan, math.nan, ">0") == "no data"
    assert v(0.3, math.nan, math.nan, ">0") == "inconclusive"


def test_hypotheses_on_unbalanced_frame() -> None:
    df = _unbalanced()
    s = _settings()
    _, analysed, scope = analyze.apply_scope(df, s, all_samples=True, use_exclusions=False)
    h = analyze.compute_hypotheses(add_row_metrics(analysed, s), s, scope)
    assert list(h.columns) == list(analyze.HYPOTHESIS_COLUMNS)
    assert len(h) == len(analyze.HYPOTHESES) * 3 and set(h["samples"]) == {"all"}

    def row(hid: str, subject: str, metric: str) -> pd.Series:
        return h[(h.hypothesis == hid) & (h.subject == subject) & (h.metric == metric)].iloc[0]

    h1 = row("H1", "m1", "extremity_gap")  # tier 3 = I3, I6; judged sample 0 only
    assert h1.estimate == pytest.approx(0.3) and h1.n_items == 2 and h1.n_cells == 2 and h1.verdict == "supported"
    assert row("H2", "m1", "net_bluff_shift").verdict == "supported"
    assert row("H3", "m2", "bluff_shift").verdict == "supported"
    assert row("H3", "m2", "evidence_update").verdict == "opposite"
    assert row("H5", POOLED, "ablation_T3a_minus_T3").verdict == "inconclusive"
    h6 = row("H6", "m2", "selectivity")
    assert h6.verdict == "no data" and h6.n_items == 0 and h6.n_cells == 0
    h6_m1 = row("H6", "m1", "selectivity")
    assert h6_m1.n_items == 3 and h6_m1.n_cells == 2 * 3 * 3  # 3 items x 3 samples x 2 conditions
    assert row("H6", POOLED, "selectivity").n_items == 3  # pooled pairs only within m1
    assert row("H3", "m1", "bluff_shift").comparison == "stand_firm - default"


def test_pick_balanced_spreads_across_models() -> None:
    pool = pd.DataFrame({"subject": ["m1"] * 10 + ["m2"] * 2 + ["m3"] * 5,
                         "item_id": [f"I{k}" for k in range(10)] + ["I0", "I1"] + [f"I{k}" for k in range(5)]})
    picked = analyze._pick_balanced(pool, 7, np.random.default_rng(0))
    counts = picked["subject"].value_counts().to_dict()
    assert len(picked) == 7 and counts["m2"] == 2 and abs(counts["m1"] - counts["m3"]) <= 1
    assert len(analyze._pick_balanced(pool, 50, np.random.default_rng(0))) == len(pool)


def test_cohen_kappa_known_values() -> None:
    k = analyze.cohen_kappa
    assert k([1, 1, 2, 2, 3, 3], [1, 1, 2, 3, 3, 3], (1, 2, 3)) == pytest.approx(0.75)
    assert k([1, 2, 3], [1, 2, 3], (1, 2, 3)) == pytest.approx(1.0)
    assert math.isnan(k([2, 2], [2, 2], (1, 2, 3))) and math.isnan(k([], [], (1, 2, 3)))


def _annotation_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    from src.benchmark import review_sheet_header

    cfg = write_config(tmp_path)
    raw = analyze.yaml.safe_load(cfg.read_text(encoding="utf-8"))
    items_path, sheet_path = tmp_path / "bench/items.jsonl", tmp_path / "bench/review_sheet.csv"
    raw["paths"].update({"benchmark": str(items_path), "review_sheet": str(sheet_path)})
    cfg.write_text(analyze.yaml.safe_dump(raw), encoding="utf-8")
    items_path.parent.mkdir(parents=True, exist_ok=True)
    drafted = [("T01", 1, "yes", 1), ("T02", 2, None, -1), ("T03", 3, "no", 1), ("T04", 3, "no", -1)]
    with items_path.open("w", encoding="utf-8") as fh:
        for item_id, tier, truth, direction in drafted:
            fh.write(json.dumps({"id": item_id, "domain": "medicine", "tier": tier, "claim": "Should X?",
                                 "truth": truth, "evidence_turn": "E.", "evidence_direction": direction,
                                 "rationale": "test input", "verified_by": ""}) + "\n")
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        header = review_sheet_header()
        writer.writerow(header)
        for item_id, *_ in drafted:
            writer.writerow([item_id, "medicine", "Should X?", "E.", ""] + [""] * (len(header) - 5))
    return cfg, items_path, sheet_path


def test_annotation_agreement_blank_sheet_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg, _, _ = _annotation_env(tmp_path)
    assert analyze.main(["annotation-agreement", "--config", str(cfg), "--n-boot", "50"]) == 2
    captured = capsys.readouterr()
    assert "0 items labelled" in captured.out and "0 items labelled" in captured.err


def test_annotation_agreement_labels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg, items_path, sheet_path = _annotation_env(tmp_path)
    before = items_path.read_bytes()
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    # T01-T03 labelled (tiers 1,2,2 vs draft 1,2,3); T04 left blank
    sheet.loc[0:2, "a_tier"] = ["1", "2", "2"]
    sheet.loc[0:2, "a_truth"] = ["yes", "null", "yes"]
    sheet.loc[0:2, "a_evidence_direction"] = ["+1", "-1", "1"]
    sheet.loc[0:2, "a_keep"] = ["y", "n", "edit"]
    sheet.loc[1, "a_notes"] = "evidence, off topic"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    assert analyze.main(["annotation-agreement", "--config", str(cfg), "--n-boot", "50"]) == 0
    table = pd.read_csv(tmp_path / "results/tables/annotation_agreement.csv").set_index("measure")
    assert table.loc["items_labelled", "value"] == 3
    assert table.loc["tier_percent_agreement", "value"] == pytest.approx(200 / 3)
    assert table.loc["truth_percent_agreement", "value"] == pytest.approx(200 / 3)
    assert table.loc["evidence_direction_percent_agreement", "value"] == pytest.approx(100.0)
    # draft (1,2,3) vs label (1,2,2): po = 2/3, pe = (1*1 + 1*2 + 1*0)/9 = 1/3 -> kappa 0.5
    assert table.loc["tier_cohen_kappa", "value"] == pytest.approx(0.5)
    assert table.loc["keep=n", "value"] == 1 and table.loc["keep=edit", "value"] == 1
    assert (tmp_path / "results/tables/annotation_agreement.md").exists()
    out = capsys.readouterr().out
    assert "Proposed exclusions (1 items" in out and "T02,annotator a keep=n: evidence; off topic" in out
    assert items_path.read_bytes() == before
    assert not (tmp_path / "data/benchmark/exclusions.csv").exists()

    sheet.loc[0, "a_tier"] = "4"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    assert analyze.main(["annotation-agreement", "--config", str(cfg), "--n-boot", "50"]) == 2


def test_human_sheet_uses_primary_samples_and_balances_models(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "export-human-sheet") == 0
    tables = tmp_path / "dry/tables"
    sheet = pd.read_csv(tables / "human_rating_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(tables / "human_rating_key.csv")
    assert list(sheet.columns) == ["briefing_id", "question", "briefing", "rater_a_p_yes", "rater_b_p_yes",
                                   "rater_a_notes", "rater_b_notes"]  # no model, condition or judge score
    assert set(key["sample"]) == {0} and {"subject", "condition", "judge_p_yes"} <= set(key.columns)
    for _, per_tier in key.groupby("tier"):
        counts = per_tier["subject"].value_counts()
        assert len(counts) == 2 and counts.max() - counts.min() <= 1
    assert _run(dry_env, "export-human-sheet", "--all-samples") == 2  # exports always use primary samples

    assert _run(dry_env, "export-second-judge-sample") == 0
    sample = pd.read_csv(tables / "second_judge_sample.csv")
    assert set(sample["sample"]) == {0}


def test_human_agreement_single_rater(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "export-human-sheet") == 0
    tables = tmp_path / "dry/tables"
    sheet_path = tables / "human_rating_sheet.csv"
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rng = np.random.default_rng(5)
    sheet.loc[:11, "rater_a_p_yes"] = [f"{v:.2f}" for v in rng.uniform(0.05, 0.95, size=12)]
    sheet.loc[:2, "rater_b_p_yes"] = ["0.5", "0.5", "0.5"]  # too few for two-rater agreement
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    assert _run(dry_env, "human-agreement") == 0
    agreement = pd.read_csv(tables / "human_judge_agreement.csv")
    first = agreement.iloc[0]
    assert first["comparison"] == "rater_a vs judge" and first["n"] == 12
    assert (agreement["rater_setup"] == "single rater").all()
    assert not math.isnan(first["mean_abs_diff"]) and "icc_2_1" not in agreement.columns
    assert (tables / "human_judge_agreement.md").exists()


# --- robustness, per-tier, forest figure, `all` ---------------------------------------------


def test_per_tier_columns_and_rows_on_synthetic_frame() -> None:
    df = _unbalanced()
    s = _settings()
    _, analysed, scope = analyze.apply_scope(df, s, all_samples=False, use_exclusions=True)
    d = add_row_metrics(analysed, s)
    table = analyze.compute_per_tier(d, s, scope)
    assert list(table.columns) == ["subject", "condition", "metric", "tier", "estimate", "ci_low", "ci_high",
                                   "n_items", "n_cells", "samples", "exclusions"]
    assert set(table["tier"]) == {"1", "2", "3"} and set(table["metric"]) == set(analyze.PER_TIER_METRICS)
    assert set(table["subject"]) == {"m1", "m2", POOLED} and set(table["samples"]) == {"0"}
    # same numbers as the per-tier rows of the full metrics table
    from_metrics = analyze.per_tier_table(compute_metrics(d, s), scope)
    pd.testing.assert_frame_equal(table, from_metrics)
    row = table[(table.subject == "m1") & (table.condition == "default") & (table.metric == "extremity_gap")
                & (table.tier == "3")].iloc[0]
    assert row.estimate == pytest.approx(0.3) and row.n_items == 2 and row.n_cells == 2


def _hyp_frame() -> pd.DataFrame:
    df = _unbalanced()
    s = _settings()
    _, analysed, scope = analyze.apply_scope(df, s, all_samples=True, use_exclusions=False)
    return analyze.compute_hypotheses(add_row_metrics(analysed, s), s, scope)


def test_forest_plot_file_created(tmp_path: Path) -> None:
    from PIL import Image

    h = _hyp_frame()
    assert len(analyze._forest_rows(h)) == len(analyze.HYPOTHESES)
    for banner in ("", analyze.DRY_BANNER):
        out = tmp_path / f"forest_{len(banner)}.png"
        analyze.plot_forest(h, out, banner, 0.95)
        with Image.open(out) as img:
            assert img.info["dpi"][0] == pytest.approx(300, abs=1)


def test_metrics_and_hypotheses_write_per_tier_and_forest(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "metrics") == 0
    tables = tmp_path / "dry/tables"
    per_tier = pd.read_csv(tables / "per_tier.csv", dtype={"samples": str, "tier": str})
    assert list(per_tier.columns) == list(analyze.PER_TIER_COLUMNS)
    assert set(per_tier["subject"]) == {"llama-3.1-8b", "qwen2.5-7b", POOLED}
    assert "DRY RUN" in (tables / "per_tier.md").read_text(encoding="utf-8")
    out = tmp_path / "pt"
    assert _run(dry_env, "per-tier", "--out-dir", str(out)) == 0
    standalone = pd.read_csv(out / "tables/per_tier.csv", dtype={"samples": str, "tier": str})
    pd.testing.assert_frame_equal(standalone, per_tier)
    assert _run(dry_env, "hypotheses") == 0
    assert (tmp_path / "dry/figures/hypotheses_forest.png").exists()


def test_robustness_outputs(dry_env: Path, tmp_path: Path) -> None:
    assert _run(dry_env, "robustness") == 0
    base = tmp_path / "dry/tables/robustness"
    for variant, samples, applied in (("all_samples", "all", True), ("no_exclusions", "0", False)):
        d = base / variant
        for name in ("metrics.csv", "hypotheses.csv", "condition_contrasts.csv", "coverage.csv",
                     "analysis_manifest.json", "metrics.md", "hypotheses.md", "condition_contrasts.md",
                     "coverage.md"):
            assert (d / name).exists(), (variant, name)
        manifest = json.loads((d / "analysis_manifest.json").read_text(encoding="utf-8"))
        assert manifest["robustness_variant"] == variant
        assert manifest["scope"]["samples"] == samples and manifest["scope"]["exclusions"]["applied"] is applied
        hyp = pd.read_csv(d / "hypotheses.csv", dtype={"samples": str})
        assert (hyp["samples"] == samples).all()
        assert hyp["exclusions"].str.startswith("applied" if applied else "not applied").all()
    # robustness writes no figures and does not touch the primary tables
    assert not (tmp_path / "dry/figures").exists()
    assert not (tmp_path / "dry/tables/metrics.csv").exists()
    assert _run(dry_env, "robustness", "--all-samples") == 2


def _point_review_sheet(cfg: Path, sheet: Path) -> None:
    raw = analyze.yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw["paths"]["review_sheet"] = str(sheet)
    cfg.write_text(analyze.yaml.safe_dump(raw), encoding="utf-8")


def test_all_skips_validations_without_inputs(dry_env: Path, tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    _point_review_sheet(dry_env, tmp_path / "bench/review_sheet.csv")  # does not exist
    assert _run(dry_env, "all") == 0
    out = capsys.readouterr().out
    for name in ("annotation-agreement", "human-agreement", "compare-judges"):
        assert f"{name}: skipped: waiting for" in out
    tables, figures = tmp_path / "dry/tables", tmp_path / "dry/figures"
    for name in ("metrics.csv", "per_tier.csv", "hypotheses.csv", "robustness/all_samples/hypotheses.csv",
                 "robustness/no_exclusions/metrics.csv"):
        assert (tables / name).exists(), name
    assert (figures / "hypotheses_forest.png").exists() and (figures / "summary_ALL_MODELS.png").exists()
    assert not (tables / "human_judge_agreement.csv").exists()


def test_all_runs_validations_whose_inputs_exist(dry_env: Path, tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    _point_review_sheet(dry_env, tmp_path / "bench/review_sheet.csv")
    assert _run(dry_env, "export-human-sheet") == 0
    _fill_ratings(tmp_path / "dry/tables/human_rating_sheet.csv", n=12, offset=0.02)
    capsys.readouterr()
    assert _run(dry_env, "all") == 0
    out = capsys.readouterr().out
    assert "human-agreement: skipped" not in out and "compare-judges: skipped: waiting for" in out
    assert (tmp_path / "dry/tables/human_judge_agreement.csv").exists()
    # an input that is present but unusable makes `all` fail (exit 2) after running the rest
    sheet = tmp_path / "dry/tables/human_rating_sheet.csv"
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.loc[0, "rater_a_p_yes"] = "70"
    df.to_csv(sheet, index=False, encoding="utf-8-sig")
    assert _run(dry_env, "all") == 2
    assert (tmp_path / "dry/tables/hypotheses.csv").exists()


def test_input_hash_is_line_ending_independent(tmp_path: Path) -> None:
    crlf, lf = tmp_path / "crlf.csv", tmp_path / "lf.csv"
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    lf.write_bytes(b"a,b\n1,2\n")
    assert analyze.sha256_lf(crlf) == analyze.sha256_lf(lf) == analyze._sha256(lf)
    assert analyze._sha256(crlf) != analyze._sha256(lf)


def test_manifests_record_lf_hash_and_outputs_use_lf(dry_env: Path, tmp_path: Path) -> None:
    results = tmp_path / "dry/results.csv"
    expected = analyze.hashlib.sha256(results.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert _run(dry_env, "metrics") == 0 and _run(dry_env, "hypotheses") == 0 and _run(dry_env, "robustness") == 0
    tables = tmp_path / "dry/tables"
    manifests = [tables / "analysis_manifest.json", tables / "hypotheses_manifest.json",
                 *(tables / "robustness" / v / n for v in ("all_samples", "no_exclusions")
                   for n in ("analysis_manifest.json", "hypotheses_manifest.json"))]
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["input_sha256_lf"] == expected and "input_sha256" in manifest, path
    written = [p for p in tables.rglob("*") if p.suffix in (".csv", ".md", ".json")]
    assert written and all(b"\r\n" not in p.read_bytes() for p in written)


def test_validation_inputs_detects_filled_columns(tmp_path: Path) -> None:
    cfg_path, _, sheet_path = _annotation_env(tmp_path)
    cfg, _ = analyze.load_settings(cfg_path)
    paths = analyze.resolve_paths(cfg, True)
    waiting = analyze.validation_inputs(cfg, paths)
    assert all(v is not None for v in waiting.values())
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet.loc[0, "a_notes"] = "notes alone do not count"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    assert analyze.validation_inputs(cfg, paths)["annotation-agreement"] is not None
    sheet.loc[0, "a_tier"] = "1"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    assert analyze.validation_inputs(cfg, paths)["annotation-agreement"] is None
    paths.second_scores.parent.mkdir(parents=True, exist_ok=True)
    paths.second_scores.write_text("x\n", encoding="utf-8")
    assert analyze.validation_inputs(cfg, paths)["compare-judges"] is None
