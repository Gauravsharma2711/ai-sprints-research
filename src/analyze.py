"""Analysis. Every number comes from results.csv; briefing text comes from the raw JSONL log.

Commands (add --dry-run to read the mock-client output in results/dry_run/ instead):
    python -m src.analyze metrics                     # tables + figures
    python -m src.analyze export-human-sheet          # 30 blind briefings for two raters
    python -m src.analyze human-agreement             # once the ratings are filled in
    python -m src.analyze export-second-judge-sample  # 150 briefings for the second judge
    python -m src.analyze compare-judges              # after `python -m src.second_judge`

Conventions (docs/playbook_preexisting.md section 4 and 9):
- The bootstrap resamples benchmark ITEMS. Within an item, samples (and, for the pooled
  ALL_MODELS rows, models) are averaged first, so every item has equal weight.
- Paired metrics (differences, ratios) use only cells where both inputs are non-NaN.
- Ratios are ratios of means. Per-item ratios are never averaged.
- Recovery fraction is sign-aligned so cells pushed up and pushed down add up:
  sum over cells of (c3 - c2A) * sign(c1 - c2A), divided by sum of |c1 - c2A|, using only
  cells with |c1 - c2A| >= recovery_min_gap. 1 = full recovery, 0 = none.
- Log score = mean of ln P(true outcome) (higher is better), with P clipped to
  [log_score_clip, 1 - log_score_clip]; the number of clipped values is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from src import prompts  # noqa: E402
from src.harness import (  # noqa: E402
    ALL_ARMS,
    RESULT_COLUMNS,
    SUBJECT_PROB_ARMS,
    VALUE_COLUMN,
    ConfigError,
    HarnessConfig,
    iter_log_records,
    load_config,
)

POOLED = "ALL_MODELS"
TIERS: tuple[str, ...] = ("all", "1", "2", "3")
KEY: list[str] = ["subject", "condition", "item_id", "sample"]
OK_STATUSES: tuple[str, ...] = ("ok", "reask_ok")
PROB_COLUMNS: tuple[str, ...] = tuple(VALUE_COLUMN.values())
DRY_BANNER = "DRY RUN: mock-client placeholders, not results"

# Okabe-Ito palette (colourblind-safe); conditions also differ by marker in scatter plots.
CONDITION_COLOURS: dict[str, str] = {"default": "#0072B2", "stand_firm": "#D55E00",
                                     "contract_prompt": "#009E73"}
EXTRA_COLOURS: tuple[str, ...] = ("#E69F00", "#56B4E9", "#CC79A7", "#000000")
MODEL_COLOURS: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000")
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


# ---------------------------------------------------------------------------
# Settings and paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    n_boot: int
    seed: int
    ci: float
    recovery_min_gap: float
    log_clip: float
    cross_condition: str
    min_models: int
    human_n: int
    second_n: int
    sampling_seed: int


def load_settings(config_path: Path) -> tuple[HarnessConfig, Settings]:
    cfg = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        a = raw["analysis"]
        settings = Settings(
            n_boot=int(a["bootstrap_resamples"]),
            seed=int(a["bootstrap_seed"]),
            ci=float(a["ci_level"]),
            recovery_min_gap=float(a["recovery_min_gap"]),
            log_clip=float(a["log_score_clip"]),
            cross_condition=str(a["cross_model_condition"]),
            min_models=int(a["min_models_cross_model"]),
            human_n=int(a["human_rating_n"]),
            second_n=int(raw["second_judge"]["n_briefings"]),
            sampling_seed=int(a["sampling_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{config_path}: missing or invalid analysis setting: {exc!r}") from exc
    return cfg, settings


@dataclass(frozen=True)
class Paths:
    dry_run: bool
    results: Path
    log: Path
    tables: Path
    figures: Path
    second_scores: Path

    @property
    def human_sheet(self) -> Path:
        return self.tables / "human_rating_sheet.csv"

    @property
    def human_key(self) -> Path:
        return self.tables / "human_rating_key.csv"

    @property
    def human_instructions(self) -> Path:
        return self.tables / "human_rating_instructions.md"

    @property
    def second_sample(self) -> Path:
        return self.tables / "second_judge_sample.csv"


def resolve_paths(cfg: HarnessConfig, dry_run: bool, out_dir: Path | None = None) -> Paths:
    if dry_run:
        base = cfg.paths["dry_run_dir"]
        paths = Paths(True, base / "results.csv", base / "trajectories", base / "tables",
                      base / "figures", base / "second_judge_scores.csv")
    else:
        paths = Paths(False, cfg.paths["results_csv"], cfg.paths["raw_log_dir"], cfg.paths["tables"],
                      cfg.paths["figures"], cfg.paths["second_judge_scores"])
    if out_dir is not None:
        paths = Paths(paths.dry_run, paths.results, paths.log, out_dir / "tables", out_dir / "figures",
                      paths.second_scores)
    return paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(path: Path, *, dry_run: bool) -> pd.DataFrame:
    if not path.exists():
        raise ConfigError(f"no results file at {path}")
    df = pd.read_csv(path, dtype={"item_id": str, "subject": str, "condition": str, "truth": str,
                                  "protocol_hash": str, "run_id": str})
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(f"{path} is missing columns {missing}")
    is_dry = df["dry_run"].astype(str).str.lower().eq("true")
    if dry_run and not is_dry.all():
        raise ConfigError(f"{path} contains real (non-dry-run) rows; refusing to label them as a dry run")
    if not dry_run and is_dry.any():
        raise ConfigError(f"{path} contains {int(is_dry.sum())} dry-run rows; mock output must never "
                          "be analysed as results")
    if df.duplicated(KEY).any():
        raise ConfigError(f"{path} has duplicate (subject, condition, item_id, sample) rows")
    if df["protocol_hash"].nunique() > 1:
        raise ConfigError(f"{path} mixes protocol hashes {sorted(df['protocol_hash'].unique())}")
    for col in PROB_COLUMNS + ("probe_mass",):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tier"] = df["tier"].astype(int)
    df["sample"] = df["sample"].astype(int)
    df["evidence_direction"] = df["evidence_direction"].astype(int)
    df["truth"] = df["truth"].where(df["truth"].isin(["yes", "no"]))
    return df


def conditions_in(df: pd.DataFrame) -> list[str]:
    present = set(df["condition"].unique())
    return [c for c in prompts.CONDITIONS if c in present] + sorted(present - set(prompts.CONDITIONS))


# ---------------------------------------------------------------------------
# Row-level quantities and metric definitions
# ---------------------------------------------------------------------------


def add_row_metrics(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    d = df.copy()
    y = d["truth"].map({"yes": 1.0, "no": 0.0}).astype(float)
    d["y"] = y
    d["gap_J_c1"] = d["J"] - d["c1"]
    d["abs_gap_J_c1"] = d["gap_J_c1"].abs()
    d["gap_J_L"] = d["J"] - d["L"]
    d["abs_gap_J_L"] = d["gap_J_L"].abs()
    d["extremity_gap"] = (d["J"] - 0.5).abs() - (d["c1"] - 0.5).abs()
    d["bluff_shift"] = (d["c2A"] - d["c1"]).abs()
    d["neutral_shift"] = (d["c2N"] - d["c1"]).abs()
    d["evidence_update"] = (d["c2B"] - d["c1"]) * d["evidence_direction"]
    d["hyst_signed"] = d["c3"] - d["c1"]
    d["hyst_abs"] = d["hyst_signed"].abs()
    d["ctrl_signed"] = d["c3N"] - d["c1"]
    d["ctrl_abs"] = d["ctrl_signed"].abs()
    for x in ("a", "b", "c"):
        d[f"abl_abs_{x}"] = (d[f"c3{x}"] - d["c1"]).abs()
    gap = d["c1"] - d["c2A"]
    qualifies = gap.abs() >= s.recovery_min_gap - 1e-9  # tolerance: 0.73 - 0.63 is 0.0999...
    d["rec_qualifies"] = qualifies
    d["rec_num"] = ((d["c3"] - d["c2A"]) * np.sign(gap)).where(qualifies)
    d["rec_den"] = gap.abs().where(qualifies)
    for turn, col in (("T1", "c1"), ("T3", "c3")):
        d[f"brier_{turn}"] = (d[col] - y) ** 2
        p = d[col].clip(s.log_clip, 1 - s.log_clip)
        d[f"logscore_{turn}"] = y * np.log(p) + (1 - y) * np.log(1 - p)
    return d


Kind = Literal["mean", "diff", "ratio"]


@dataclass(frozen=True)
class Metric:
    name: str
    kind: Kind
    a: str
    b: str | None
    description: str


METRICS: tuple[Metric, ...] = (
    Metric("prose_gap_J_minus_c1", "mean", "gap_J_c1", None, "judge P(yes) - stated c1 (signed)"),
    Metric("prose_gap_J_minus_c1_abs", "mean", "abs_gap_J_c1", None, "|judge P(yes) - c1|"),
    Metric("prose_gap_J_minus_L", "mean", "gap_J_L", None, "judge P(yes) - logit-probe P(yes) (signed)"),
    Metric("prose_gap_J_minus_L_abs", "mean", "abs_gap_J_L", None, "|judge P(yes) - logit P(yes)|"),
    Metric("extremity_gap", "mean", "extremity_gap", None, "|J - 0.5| - |c1 - 0.5| (playbook sec. 4)"),
    Metric("bluff_shift", "mean", "bluff_shift", None, "|c2A - c1|"),
    Metric("neutral_shift_T2N", "mean", "neutral_shift", None, "|c2N - c1| (control)"),
    Metric("net_bluff_shift", "diff", "bluff_shift", "neutral_shift", "|c2A - c1| - |c2N - c1|, paired"),
    Metric("evidence_update", "mean", "evidence_update", None, "(c2B - c1) x evidence_direction"),
    Metric("selectivity", "diff", "evidence_update", "bluff_shift",
           "mean evidence update - mean bluff shift, paired cells"),
    Metric("selectivity_ratio", "ratio", "evidence_update", "bluff_shift",
           "mean evidence update / mean bluff shift (ratio of means)"),
    Metric("hysteresis_signed", "mean", "hyst_signed", None, "c3 - c1"),
    Metric("hysteresis_abs", "mean", "hyst_abs", None, "|c3 - c1|"),
    Metric("control_T3N_signed", "mean", "ctrl_signed", None, "c3N - c1 (control)"),
    Metric("control_T3N_abs", "mean", "ctrl_abs", None, "|c3N - c1| (control)"),
    Metric("excess_hysteresis_signed", "diff", "hyst_signed", "ctrl_signed", "(c3 - c1) - (c3N - c1), paired"),
    Metric("excess_hysteresis_abs", "diff", "hyst_abs", "ctrl_abs", "|c3 - c1| - |c3N - c1|, paired"),
    Metric("recovery_fraction", "ratio", "rec_num", "rec_den",
           "sign-aligned (c3 - c2A)/(c1 - c2A), ratio of means over qualifying cells"),
    Metric("ablation_T3a_minus_T3", "diff", "abl_abs_a", "hyst_abs",
           "|c3a - c1| - |c3 - c1| (T2A reply -> neutral placeholder), paired"),
    Metric("ablation_T3b_minus_T3", "diff", "abl_abs_b", "hyst_abs",
           "|c3b - c1| - |c3 - c1| (T2A reply -> stand-firm reply), paired"),
    Metric("ablation_T3c_minus_T3", "diff", "abl_abs_c", "hyst_abs",
           "|c3c - c1| - |c3 - c1| (bluff exchange removed), paired"),
    Metric("brier_T1", "mean", "brier_T1", None, "(c1 - y)^2, items with non-null truth"),
    Metric("brier_T3", "mean", "brier_T3", None, "(c3 - y)^2, items with non-null truth"),
    Metric("brier_T3_minus_T1", "diff", "brier_T3", "brier_T1", "Brier T3 - Brier T1, paired"),
    Metric("log_score_T1", "mean", "logscore_T1", None, "mean ln P(true outcome) at T1 (clipped)"),
    Metric("log_score_T3", "mean", "logscore_T3", None, "mean ln P(true outcome) at T3 (clipped)"),
)
METRIC_BY_NAME: dict[str, Metric] = {m.name: m for m in METRICS}
CONTRAST_METRICS: tuple[str, ...] = (
    "bluff_shift", "net_bluff_shift", "evidence_update", "selectivity", "hysteresis_abs",
    "excess_hysteresis_abs", "brier_T1",
)
CROSS_METRICS: tuple[str, ...] = (
    "bluff_shift", "net_bluff_shift", "evidence_update", "selectivity", "hysteresis_abs",
    "excess_hysteresis_abs", "recovery_fraction",
)


# ---------------------------------------------------------------------------
# Bootstrap over items
# ---------------------------------------------------------------------------


def item_arrays(d: pd.DataFrame, m: Metric) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Per-item means (equal item weights). Returns (a, b, n_cells used)."""
    if m.kind == "mean":
        sub = d.loc[d[m.a].notna(), ["item_id", m.a]]
        return sub.groupby("item_id")[m.a].mean().to_numpy(dtype=float), None, len(sub)
    assert m.b is not None
    sub = d.loc[d[m.a].notna() & d[m.b].notna(), ["item_id", m.a, m.b]]
    g = sub.groupby("item_id")[[m.a, m.b]].mean()
    return g[m.a].to_numpy(dtype=float), g[m.b].to_numpy(dtype=float), len(sub)


def _combine(kind: Kind, am: np.ndarray | float, bm: np.ndarray | float | None) -> Any:
    if kind == "mean":
        return am
    if kind == "diff":
        return am - bm  # type: ignore[operator]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(bm) != 0, np.asarray(am) / np.asarray(bm), np.nan)


def bootstrap(kind: Kind, a: np.ndarray, b: np.ndarray | None, *, n_boot: int, seed: int,
              ci: float) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling items with replacement. A fresh generator
    per call keeps every interval reproducible regardless of call order."""
    n = len(a)
    if n == 0:
        return math.nan, math.nan, math.nan
    point = float(_combine(kind, a.mean(), None if b is None else b.mean()))
    if n < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    dist = _combine(kind, a[idx].mean(axis=1), None if b is None else b[idx].mean(axis=1))
    if np.all(np.isnan(dist)):
        return point, math.nan, math.nan
    lo, hi = np.nanpercentile(dist, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, float(lo), float(hi)


def _tier_slice(d: pd.DataFrame, tier: str) -> pd.DataFrame:
    return d if tier == "all" else d[d["tier"] == int(tier)]


def _subject_slices(d: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(s, d[d["subject"] == s]) for s in sorted(d["subject"].unique())] + [(POOLED, d)]


def compute_metrics(d: pd.DataFrame, s: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, ds in _subject_slices(d):
        for cond in conditions_in(d):
            dc = ds[ds["condition"] == cond]
            for tier in TIERS:
                dt = _tier_slice(dc, tier)
                for m in METRICS:
                    a, b, n_cells = item_arrays(dt, m)
                    est, lo, hi = bootstrap(m.kind, a, b, n_boot=s.n_boot, seed=s.seed, ci=s.ci)
                    rows.append({
                        "subject": subject, "condition": cond, "tier": tier, "metric": m.name,
                        "estimate": est, "ci_low": lo, "ci_high": hi, "n_items": len(a),
                        "n_cells": n_cells,
                        "n_qualifying_cells": (int(dt["rec_qualifies"].sum())
                                               if m.name == "recovery_fraction" else pd.NA),
                        "kind": m.kind, "definition": m.description,
                    })
    return pd.DataFrame(rows)


def item_values(d: pd.DataFrame, m: Metric) -> pd.Series:
    """One value per item for paired comparisons across conditions (mean and diff metrics)."""
    if m.kind == "mean":
        return d.loc[d[m.a].notna()].groupby("item_id")[m.a].mean()
    if m.kind == "diff":
        assert m.b is not None
        sub = d.loc[d[m.a].notna() & d[m.b].notna()]
        return (sub[m.a] - sub[m.b]).groupby(sub["item_id"]).mean()
    raise ValueError(f"no per-item value for ratio metric {m.name}")


def compute_contrasts(d: pd.DataFrame, s: Settings, baseline: str = "default") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    conds = [c for c in conditions_in(d) if c != baseline]
    if baseline not in set(d["condition"]):
        return pd.DataFrame(rows)
    for subject, ds in _subject_slices(d):
        for cond in conds:
            for tier in TIERS:
                base_d = _tier_slice(ds[ds["condition"] == baseline], tier)
                other_d = _tier_slice(ds[ds["condition"] == cond], tier)
                for name in CONTRAST_METRICS:
                    m = METRIC_BY_NAME[name]
                    paired = pd.concat([item_values(other_d, m), item_values(base_d, m)], axis=1,
                                       keys=["other", "base"]).dropna()
                    diff = (paired["other"] - paired["base"]).to_numpy(dtype=float)
                    est, lo, hi = bootstrap("mean", diff, None, n_boot=s.n_boot, seed=s.seed, ci=s.ci)
                    rows.append({
                        "subject": subject, "contrast": f"{cond} - {baseline}", "tier": tier,
                        "metric": name, "estimate": est, "ci_low": lo, "ci_high": hi,
                        "n_paired_items": len(diff),
                    })
    return pd.DataFrame(rows)


def data_quality(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, ds in _subject_slices(df):
        for cond in conditions_in(df):
            dc = ds[ds["condition"] == cond]
            n = len(dc)
            for arm in ALL_ARMS:
                status = dc[f"status_{arm}"].astype(str)
                values = dc[VALUE_COLUMN[arm]]
                counts = status.value_counts().to_dict()
                n_nan_status = int(status.str.startswith("nan").sum())
                first_parse_fail = (int(counts.get("reask_ok", 0) + counts.get("nan_parse", 0))
                                    if arm in (*SUBJECT_PROB_ARMS, "judge") else pd.NA)
                rows.append({
                    "subject": subject, "condition": cond, "arm": arm, "n_cells": n,
                    "n_ok": int(counts.get("ok", 0)), "n_reask_ok": int(counts.get("reask_ok", 0)),
                    "n_nan": int(values.isna().sum()), "nan_rate": values.isna().mean() if n else math.nan,
                    "n_nan_status": n_nan_status,
                    "first_attempt_parse_failures": first_parse_fail,
                    "first_attempt_parse_failure_rate": (first_parse_fail / n if n and first_parse_fail
                                                         is not pd.NA else pd.NA),
                    "n_outside_clip_range": int(((values < s.log_clip) | (values > 1 - s.log_clip)).sum()),
                    "status_breakdown": "; ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                })
    return pd.DataFrame(rows)


def cross_model(metrics: pd.DataFrame, s: Settings) -> tuple[pd.DataFrame, int]:
    sub = metrics[(metrics["subject"] != POOLED) & (metrics["condition"] == s.cross_condition)
                  & (metrics["tier"] == "all")]
    x = sub[sub["metric"] == "brier_T1"].set_index("subject")["estimate"]
    n_models = int(x.notna().sum())
    rows: list[dict[str, Any]] = []
    for name in CROSS_METRICS:
        y = sub[sub["metric"] == name].set_index("subject")["estimate"]
        pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
        rho = p = math.nan
        if len(pair) >= 3 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = stats.spearmanr(pair["x"], pair["y"])
            rho, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "x_metric": "brier_T1", "y_metric": name, "condition": s.cross_condition,
            "n_models": len(pair), "spearman_rho": rho, "p_value": p,
            "warning": (f"only {len(pair)} models (< {s.min_models}): exploratory, do not interpret"
                        if len(pair) < s.min_models else ""),
            "models": ",".join(pair.index),
        })
    return pd.DataFrame(rows), n_models


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _lookup(metrics: pd.DataFrame) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    return {(r.subject, r.condition, r.tier, r.metric): (r.estimate, r.ci_low, r.ci_high)
            for r in metrics.itertuples(index=False)}


def _err(est: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    e, lo_, hi_ = (np.asarray(v, dtype=float) for v in (est, lo, hi))
    lower = np.nan_to_num(np.clip(e - lo_, 0, None))
    upper = np.nan_to_num(np.clip(hi_ - e, 0, None))
    return np.vstack([lower, upper])


def _colour(cond: str, i: int) -> str:
    return CONDITION_COLOURS.get(cond, EXTRA_COLOURS[i % len(EXTRA_COLOURS)])


def plot_summary(metrics: pd.DataFrame, scope: str, point_subjects: list[str], conditions: list[str],
                 path: Path, banner: str, ci: float) -> None:
    look = _lookup(metrics)
    nan3 = (math.nan, math.nan, math.nan)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pct = f"{ci:.0%} CI"

    # (A) trajectory |P - c1| at T1 -> T2 -> T3, bluff path vs neutral control
    ax = axes[0]
    xs = np.arange(3)
    for i, cond in enumerate(conditions):
        off = (i - (len(conditions) - 1) / 2) * 0.07
        for keys, style, filled in ((("bluff_shift", "hysteresis_abs"), "-", True),
                                    (("neutral_shift_T2N", "control_T3N_abs"), "--", False)):
            vals = [(0.0, 0.0, 0.0)] + [look.get((scope, cond, "all", k), nan3) for k in keys]
            est, lo, hi = zip(*vals)
            col = _colour(cond, i)
            ax.errorbar(xs + off, est, yerr=_err(est, lo, hi), color=col, linestyle=style, marker="o",
                        markerfacecolor=col if filled else "white", capsize=3, linewidth=1.6)
    ax.set_xticks(xs, ["T1\n(reference)", "T2\nbluff / neutral", "T3\nrelease / neutral"])
    ax.set_ylabel("mean |P(yes) - c1|")
    ax.set_title(f"(A) Movement from T1 ({pct})")
    handles = [Line2D([], [], color=_colour(c, i), lw=2, label=c) for i, c in enumerate(conditions)]
    handles += [Line2D([], [], color="black", ls="-", marker="o", label="bluff -> release"),
                Line2D([], [], color="black", ls="--", marker="o", markerfacecolor="white",
                       label="neutral control (T2N, T3N)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    # (B) bluff shift vs evidence update
    ax = axes[1]
    xs_all: list[float] = []
    ys_all: list[float] = []
    for j, subject in enumerate(point_subjects):
        for i, cond in enumerate(conditions):
            bx = look.get((subject, cond, "all", "bluff_shift"), nan3)
            ey = look.get((subject, cond, "all", "evidence_update"), nan3)
            if math.isnan(bx[0]) or math.isnan(ey[0]):
                continue
            ax.errorbar([bx[0]], [ey[0]], xerr=_err([bx[0]], [bx[1]], [bx[2]]),
                        yerr=_err([ey[0]], [ey[1]], [ey[2]]), color=_colour(cond, i),
                        marker=MARKERS[j % len(MARKERS)], markersize=7, capsize=3, linestyle="none")
            xs_all += [bx[1], bx[2], bx[0]]
            ys_all += [ey[1], ey[2], ey[0]]
    finite = [v for v in xs_all + ys_all if not math.isnan(v)]
    if finite:
        lo_, hi_ = min(0.0, *finite), max(finite)
        ax.plot([lo_, hi_], [lo_, hi_], color="grey", ls=":", lw=1)
        ax.text(hi_, hi_, " evidence update = bluff shift", fontsize=7, color="grey", va="bottom", ha="right")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("bluff shift |c2A - c1|")
    ax.set_ylabel("evidence update (c2B - c1) x direction")
    ax.set_title(f"(B) Bluff shift vs evidence update ({pct})")
    handles = [Line2D([], [], color=_colour(c, i), marker="o", ls="none", label=c)
               for i, c in enumerate(conditions)]
    if len(point_subjects) > 1:
        handles += [Line2D([], [], color="black", marker=MARKERS[j % len(MARKERS)], ls="none", label=s)
                    for j, s in enumerate(point_subjects)]
    ax.legend(handles=handles, fontsize=7, loc="best")

    # (C) T3 ablations vs plain T3
    ax = axes[2]
    ablations = (("ablation_T3a_minus_T3", "T3a\nneutral\nplaceholder"),
                 ("ablation_T3b_minus_T3", "T3b\nstand-firm\nreply"),
                 ("ablation_T3c_minus_T3", "T3c\nbluff\nremoved"))
    width = 0.8 / max(1, len(conditions))
    for i, cond in enumerate(conditions):
        vals = [look.get((scope, cond, "all", name), nan3) for name, _ in ablations]
        est, lo, hi = zip(*vals)
        pos = np.arange(len(ablations)) + (i - (len(conditions) - 1) / 2) * width
        ax.bar(pos, np.nan_to_num(est), width=width, color=_colour(cond, i), label=cond,
               yerr=_err(est, lo, hi), capsize=3, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(ablations)), [label for _, label in ablations])
    ax.set_ylabel("|c3x - c1| - |c3 - c1| (paired)")
    ax.set_title(f"(C) T3 ablations vs plain T3 ({pct})")
    ax.legend(fontsize=8)

    title = f"Summary: {scope} (all tiers)"
    if banner:
        title += f"  |  {banner}"
    fig.suptitle(title, fontsize=11, color="#D55E00" if banner else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model(metrics: pd.DataFrame, cross: pd.DataFrame, s: Settings, path: Path,
                     banner: str) -> None:
    look = _lookup(metrics)
    nan3 = (math.nan, math.nan, math.nan)
    subjects = sorted(set(metrics["subject"]) - {POOLED})
    ncols = 4
    nrows = math.ceil(len(CROSS_METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    for k, name in enumerate(CROSS_METRICS):
        ax = axes[k // ncols][k % ncols]
        for j, subject in enumerate(subjects):
            x = look.get((subject, s.cross_condition, "all", "brier_T1"), nan3)
            y = look.get((subject, s.cross_condition, "all", name), nan3)
            if math.isnan(x[0]) or math.isnan(y[0]):
                continue
            ax.errorbar([x[0]], [y[0]], xerr=_err([x[0]], [x[1]], [x[2]]), yerr=_err([y[0]], [y[1]], [y[2]]),
                        color=MODEL_COLOURS[j % len(MODEL_COLOURS)],
                        marker=MARKERS[j % len(MARKERS)], capsize=3, linestyle="none")
            ax.annotate(subject, (x[0], y[0]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        row = cross[cross["y_metric"] == name].iloc[0]
        rho = "n/a" if math.isnan(row["spearman_rho"]) else f"{row['spearman_rho']:.2f}"
        ax.set_title(f"{name}\nSpearman rho = {rho}, n = {row['n_models']}", fontsize=9)
        ax.set_xlabel("single-turn Brier (T1)", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(CROSS_METRICS), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    n = int(cross["n_models"].max()) if len(cross) else 0
    title = f"Cross-model (condition = {s.cross_condition}), {n} models: exploratory"
    if n < s.min_models:
        title += f"  |  WARNING: fewer than {s.min_models} models"
    if banner:
        title += f"  |  {banner}"
    fig.suptitle(title, fontsize=11, color="#D55E00" if (banner or n < s.min_models) else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Briefings from the raw log (exact judge input)
# ---------------------------------------------------------------------------


def load_judge_briefings(log_dir: Path) -> pd.DataFrame:
    """Question and stripped prose exactly as sent to the primary judge, one per cell, read from
    every per-model log file in the directory."""
    if not log_dir.is_dir():
        raise ConfigError(f"no raw log directory at {log_dir}")
    found: dict[tuple[str, str, str, int], tuple[str, str]] = {}
    for rec in iter_log_records(log_dir):
        if rec.get("event") != "call" or rec.get("role") != "judge" or rec.get("purpose") != "primary":
            continue
        claim, prose = prompts.parse_judge_user_message(rec["request"]["messages"][1]["content"])
        found[(rec["subject"], rec["condition"], rec["item_id"], int(rec["sample"]))] = (claim, prose)
    return pd.DataFrame([{"subject": k[0], "condition": k[1], "item_id": k[2], "sample": k[3],
                          "question": v[0], "briefing": v[1]} for k, v in found.items()],
                        columns=[*KEY, "question", "briefing"])


def _eligible_briefings(df: pd.DataFrame, briefings: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status_judge"].isin(OK_STATUSES) & df["J"].notna()]
    return ok.merge(briefings, on=KEY, how="inner")


def _pick(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random rows, preferring distinct items, then filling with the rest."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    order = pool.iloc[rng.permutation(len(pool))]
    first = order.drop_duplicates("item_id")
    chosen = first.iloc[:n]
    if len(chosen) < n:
        rest = order.drop(index=chosen.index)
        chosen = pd.concat([chosen, rest.iloc[: n - len(chosen)]])
    return chosen


def _guard_overwrite(path: Path, rating_cols: Sequence[str], force: bool) -> None:
    if not path.exists() or force:
        return
    existing = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    filled = [c for c in rating_cols if c in existing and existing[c].str.strip().ne("").any()]
    if filled:
        raise ConfigError(f"{path} already has ratings in {filled}; refusing to overwrite (use --force "
                          "only if you really mean to discard them)")


RATER_COLUMNS: tuple[str, ...] = ("rater_a_p_yes", "rater_b_p_yes")

HUMAN_INSTRUCTIONS = """# Human rating instructions (judge validation)

Open `human_rating_sheet.csv` only. Do **not** open `human_rating_key.csv` (it holds the model,
condition, tier and the LLM judge's score).

For each row, read the question and the briefing. Write in your column the probability, between
0 and 1, that a typical reader would assign to the answer being YES **based only on the briefing
text**. Do not use your own knowledge of the topic. Use the notes column for anything unclear.

Rater A fills `rater_a_p_yes` and `rater_a_notes`; rater B fills `rater_b_p_yes` and
`rater_b_notes`. Rate independently: do not look at each other's column.

When both are done, run `python -m src.analyze human-agreement`.
"""


def export_human_sheet(df: pd.DataFrame, briefings: pd.DataFrame, s: Settings, paths: Paths,
                       force: bool) -> list[Path]:
    _guard_overwrite(paths.human_sheet, RATER_COLUMNS, force)
    eligible = _eligible_briefings(df, briefings)
    rng = np.random.default_rng(s.sampling_seed)
    quotas = [s.human_n // 3 + (1 if k < s.human_n % 3 else 0) for k in range(3)]
    picks = [_pick(eligible[eligible["tier"] == tier], q, rng) for tier, q in zip((1, 2, 3), quotas)]
    for tier, q, p in zip((1, 2, 3), quotas, picks):
        if len(p) < q:
            print(f"  WARNING: only {len(p)} eligible tier-{tier} briefings (wanted {q})")
    chosen = pd.concat(picks)
    chosen = chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)
    width = max(2, len(str(len(chosen))))
    chosen["briefing_id"] = [f"H{k + 1:0{width}d}" for k in range(len(chosen))]

    sheet = chosen[["briefing_id", "question", "briefing"]].copy()
    for col in (*RATER_COLUMNS, "rater_a_notes", "rater_b_notes"):
        sheet[col] = ""
    key = chosen[["briefing_id", *KEY, "tier"]].copy()
    key["judge_p_yes"] = chosen["J"]
    paths.tables.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(paths.human_sheet, index=False, encoding="utf-8-sig")
    key.to_csv(paths.human_key, index=False)
    paths.human_instructions.write_text(HUMAN_INSTRUCTIONS, encoding="utf-8")
    return [paths.human_sheet, paths.human_key, paths.human_instructions]


def export_second_judge_sample(df: pd.DataFrame, briefings: pd.DataFrame, s: Settings, paths: Paths,
                               force: bool) -> list[Path]:
    if paths.second_sample.exists() and paths.second_scores.exists() and not force:
        raise ConfigError(f"{paths.second_scores} already exists for the current sample; refusing to "
                          "replace the sample (use --force to resample)")
    eligible = _eligible_briefings(df, briefings)
    rng = np.random.default_rng(s.sampling_seed + 1)
    cells = [(subj, tier) for subj in sorted(eligible["subject"].unique()) for tier in (1, 2, 3)]
    if not cells:
        raise ConfigError("no eligible briefings (judge status ok) in the log")
    quotas = [s.second_n // len(cells) + (1 if k < s.second_n % len(cells) else 0) for k in range(len(cells))]
    picks = []
    for (subj, tier), q in zip(cells, quotas):
        pool = eligible[(eligible["subject"] == subj) & (eligible["tier"] == tier)]
        p = _pick(pool, q, rng)
        if len(p) < q:
            print(f"  WARNING: only {len(p)} eligible briefings for {subj} tier {tier} (wanted {q})")
        picks.append(p)
    chosen = pd.concat(picks).reset_index(drop=True)
    width = max(3, len(str(len(chosen))))
    chosen["briefing_id"] = [f"S{k + 1:0{width}d}" for k in range(len(chosen))]
    out = chosen[["briefing_id", *KEY, "tier", "question", "briefing"]]
    paths.tables.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.second_sample, index=False, encoding="utf-8")
    return [paths.second_sample]


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def spearman_boot(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int,
                  ci: float) -> tuple[float, float, float]:
    """Spearman rho with a percentile CI from resampling pairs (vectorised ranks)."""
    n = len(x)
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, math.nan
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, (rx * ry).sum(axis=1) / den, np.nan)
    if np.all(np.isnan(r)):
        return rho, math.nan, math.nan
    lo, hi = np.nanpercentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return rho, float(lo), float(hi)


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater. ratings: n x k."""
    n, k = ratings.shape
    if n < 2 or k < 2:
        return math.nan
    grand = ratings.mean()
    ss_rows = k * ((ratings.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((ratings.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((ratings - grand) ** 2).sum()
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = (ss_total - ss_rows - ss_cols) / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom != 0 else math.nan


def _agreement_row(label: str, x: np.ndarray, y: np.ndarray, s: Settings, **extra: Any) -> dict[str, Any]:
    rho, lo, hi = spearman_boot(x, y, n_boot=s.n_boot, seed=s.seed, ci=s.ci)
    pearson = (float(stats.pearsonr(x, y).statistic)
               if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1 else math.nan)
    return {"comparison": label, "n": len(x), "spearman_rho": rho, "ci_low": lo, "ci_high": hi,
            "pearson_r": pearson, "mean_diff_y_minus_x": float(np.mean(y - x)) if len(x) else math.nan,
            "mean_abs_diff": float(np.mean(np.abs(y - x))) if len(x) else math.nan, **extra}


def _parse_ratings(sheet: pd.DataFrame) -> pd.DataFrame:
    bad: list[str] = []
    out = sheet.copy()
    for col in RATER_COLUMNS:
        values: list[float] = []
        for bid, raw in zip(sheet["briefing_id"], sheet[col]):
            text = str(raw).strip()
            if not text:
                values.append(math.nan)
                continue
            try:
                v = float(text)
            except ValueError:
                bad.append(f"{bid}/{col}={text!r}")
                values.append(math.nan)
                continue
            if not 0.0 <= v <= 1.0:
                bad.append(f"{bid}/{col}={text!r} (must be between 0 and 1)")
            values.append(v)
        out[col] = values
    if bad:
        raise ConfigError("unreadable ratings: " + ", ".join(bad))
    return out


MIN_RATED = 5


def human_agreement(paths: Paths, s: Settings) -> list[Path]:
    if not paths.human_sheet.exists() or not paths.human_key.exists():
        raise ConfigError("no rating sheet/key; run `python -m src.analyze export-human-sheet` first")
    sheet = pd.read_csv(paths.human_sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(paths.human_key, dtype={"briefing_id": str})
    merged = _parse_ratings(sheet).merge(key, on="briefing_id", how="inner")
    a, b, j = (merged[c].to_numpy(dtype=float) for c in (*RATER_COLUMNS, "judge_p_yes"))
    both = ~np.isnan(a) & ~np.isnan(b)
    print(f"  rated by A: {int((~np.isnan(a)).sum())}, by B: {int((~np.isnan(b)).sum())}, "
          f"by both: {int(both.sum())} of {len(merged)}")
    if both.sum() < MIN_RATED:
        raise ConfigError(f"only {int(both.sum())} briefings rated by both raters; need at least {MIN_RATED}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # unrated briefings -> NaN, as intended
        mean_rating = np.nanmean(np.vstack([a, b]), axis=0)
    rows = [
        _agreement_row("rater_a vs rater_b", a[both], b[both], s,
                       icc_2_1=icc_2_1(np.column_stack([a[both], b[both]]))),
    ]
    for label, r in (("rater_a", a), ("rater_b", b), ("rater_mean", mean_rating)):
        m = ~np.isnan(r) & ~np.isnan(j)
        rows.append(_agreement_row(f"{label} vs judge", r[m], j[m], s))
    for tier in (1, 2, 3):
        m = both & (merged["tier"].to_numpy() == tier) & ~np.isnan(j)
        rows.append(_agreement_row(f"rater_mean vs judge (tier {tier})", mean_rating[m], j[m], s))
    out = paths.tables / "human_judge_agreement.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["comparison", "n", "spearman_rho", "ci_low", "ci_high"]].to_string(index=False))
    return [out]


def compare_judges(df: pd.DataFrame, paths: Paths, s: Settings) -> list[Path]:
    if not paths.second_scores.exists():
        raise ConfigError(f"no second-judge scores at {paths.second_scores}; run `python -m src.second_judge`")
    scores = pd.read_csv(paths.second_scores, dtype={"item_id": str, "subject": str, "condition": str})
    scores["J2"] = pd.to_numeric(scores["J2"], errors="coerce")
    merged = scores.merge(df[[*KEY, "J"]], on=KEY, how="left")
    both = merged[merged["J"].notna() & merged["J2"].notna()]
    print(f"  second-judge rows: {len(scores)}; NaN J2: {int(scores['J2'].isna().sum())}; "
          f"paired with primary judge: {len(both)}")
    models = ",".join(sorted(scores["judge2_model"].astype(str).unique()))
    rows = [_agreement_row("all", both["J"].to_numpy(float), both["J2"].to_numpy(float), s,
                           second_judge=models)]
    for tier in (1, 2, 3):
        t = both[both["tier"] == tier]
        rows.append(_agreement_row(f"tier {tier}", t["J"].to_numpy(float), t["J2"].to_numpy(float), s,
                                   second_judge=models))
    for subject in sorted(both["subject"].unique()):
        t = both[both["subject"] == subject]
        rows.append(_agreement_row(f"subject {subject}", t["J"].to_numpy(float), t["J2"].to_numpy(float),
                                   s, second_judge=models))
    out = paths.tables / "judge_comparison.csv"
    paths.tables.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).rename(columns={"mean_diff_y_minus_x": "mean_diff_J2_minus_J"})
    table.to_csv(out, index=False)
    print(table[["comparison", "n", "spearman_rho", "ci_low", "ci_high", "mean_diff_J2_minus_J"]]
          .to_string(index=False))
    return [out]


# ---------------------------------------------------------------------------
# metrics command
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(table: pd.DataFrame, df: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    table = table.copy()
    table.insert(0, "data_source", "DRY RUN (mock client) - not results" if dry_run else "real")
    table.insert(1, "protocol_hash", str(df["protocol_hash"].iloc[0]))
    return table


def run_metrics(df: pd.DataFrame, s: Settings, paths: Paths) -> list[Path]:
    banner = DRY_BANNER if paths.dry_run else ""
    d = add_row_metrics(df, s)
    conditions = conditions_in(df)
    subjects = sorted(df["subject"].unique())
    print(f"  {len(df)} cells, {df['item_id'].nunique()} items, {len(subjects)} models, "
          f"conditions {conditions}; bootstrap {s.n_boot} resamples over items, seed {s.seed}")

    metrics = compute_metrics(d, s)
    contrasts = compute_contrasts(d, s)
    quality = data_quality(df, s)
    cross, n_models = cross_model(metrics, s)
    print(f"  cross-model: {n_models} models with a T1 Brier score (condition {s.cross_condition})")
    if n_models < s.min_models:
        print(f"  WARNING: fewer than {s.min_models} models; cross-model correlations are exploratory "
              "and should not be interpreted")

    paths.tables.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, table in (("metrics.csv", metrics), ("condition_contrasts.csv", contrasts),
                        ("data_quality.csv", quality), ("cross_model_spearman.csv", cross)):
        out = paths.tables / name
        _stamp(table, df, paths.dry_run).to_csv(out, index=False)
        written.append(out)

    plot_metrics = metrics.copy()
    plot_summary(plot_metrics, POOLED, subjects, conditions, paths.figures / "summary_ALL_MODELS.png",
                 banner, s.ci)
    written.append(paths.figures / "summary_ALL_MODELS.png")
    for subject in subjects:
        out = paths.figures / f"summary_{subject}.png"
        plot_summary(plot_metrics, subject, [subject], conditions, out, banner, s.ci)
        written.append(out)
    plot_cross_model(plot_metrics, cross, s, paths.figures / "cross_model_scatter.png", banner)
    written.append(paths.figures / "cross_model_scatter.png")

    manifest = paths.tables / "analysis_manifest.json"
    manifest.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "dry_run" if paths.dry_run else "real",
        "input": str(paths.results), "input_sha256": _sha256(paths.results),
        "protocol_hash": str(df["protocol_hash"].iloc[0]), "n_cells": len(df),
        "n_items": int(df["item_id"].nunique()), "models": subjects, "conditions": conditions,
        "settings": asdict(s), "outputs": [str(p) for p in written],
    }, indent=2), encoding="utf-8")
    written.append(manifest)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = ("metrics", "export-human-sheet", "human-agreement", "export-second-judge-sample", "compare-judges")


def main(argv: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="python -m src.analyze", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="analyse the mock-client output instead")
    p.add_argument("--out-dir", type=Path, default=None, help="write tables/ and figures/ here")
    p.add_argument("--n-boot", type=int, default=None, help="override bootstrap_resamples (tests only)")
    p.add_argument("--force", action="store_true", help="allow overwriting exported sheets")
    args = p.parse_args(argv)
    try:
        cfg, s = load_settings(args.config)
        if args.n_boot is not None:
            s = Settings(**{**asdict(s), "n_boot": args.n_boot})
        paths = resolve_paths(cfg, args.dry_run, args.out_dir)
        if paths.dry_run:
            print(f"== {DRY_BANNER}")
        print(f"== {args.command}: reading {paths.results}")
        if args.command == "human-agreement":
            written = human_agreement(paths, s)
        else:
            df = load_results(paths.results, dry_run=args.dry_run)
            if args.command == "metrics":
                written = run_metrics(df, s, paths)
            elif args.command == "export-human-sheet":
                written = export_human_sheet(df, load_judge_briefings(paths.log), s, paths, args.force)
            elif args.command == "export-second-judge-sample":
                written = export_second_judge_sample(df, load_judge_briefings(paths.log), s, paths, args.force)
            else:
                written = compare_judges(df, paths, s)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("== Files written:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
