"""Offline tests for the paper tooling (tools/build_paper.py, numbers_audit.py, export_prompts.py).

All tables here are synthetic and live in tmp_path; no real results are read or produced.
"""

from __future__ import annotations

import csv
import random
import struct
import zlib
from pathlib import Path

import pytest
import yaml

from src import prompts
from tools import build_paper as bp
from tools import export_prompts as ep
from tools import numbers_audit as na

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config.yaml"
DRAFT = REPO / "paper" / "draft.md"

SUBJECTS = ("qwen2.5-7b", "phi-3.5-mini", "mistral-7b")
CONDITIONS = ("default", "stand_firm", "contract_prompt")
TIERS = ("all", "1", "2", "3")
METRICS = (
    "prose_gap_J_minus_c1", "prose_gap_J_minus_c1_abs", "prose_gap_J_minus_L", "extremity_gap",
    "bluff_shift", "neutral_shift_T2N", "net_bluff_shift", "evidence_update", "selectivity",
    "selectivity_ratio", "hysteresis_signed", "hysteresis_abs", "control_T3N_abs",
    "excess_hysteresis_abs", "recovery_fraction", "ablation_T3a_minus_T3", "ablation_T3b_minus_T3",
    "ablation_T3c_minus_T3", "brier_T1", "brier_T3", "brier_T3_minus_T1", "log_score_T1", "log_score_T3",
)
# (hypothesis, comparison, metric, tier) rows, as in src/analyze.py's hypotheses table.
HYPOTHESES = (
    ("H1", "default", "extremity_gap", "3"),
    ("H2", "default", "net_bluff_shift", "all"),
    ("H3", "stand_firm - default", "bluff_shift", "all"),
    ("H3", "stand_firm - default", "evidence_update", "all"),
    ("H4", "default", "excess_hysteresis_abs", "all"),
    ("H5", "default", "ablation_T3a_minus_T3", "all"),
    ("H5", "default", "ablation_T3b_minus_T3", "all"),
    ("H5", "default", "ablation_T3c_minus_T3", "all"),
    ("H6", "contract_prompt - default", "selectivity", "all"),
    ("H6", "contract_prompt - default", "excess_hysteresis_abs", "all"),
)
ARMS = ("T1", "probe", "judge", "T2A", "T2B", "T2N", "T3", "T3a", "T3b", "T3c", "T3N")


# ---------------------------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def tiny_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def synthetic_tables(root: Path) -> tuple[Path, Path]:
    """Complete synthetic versions of every table the draft uses (random values, fixed seed)."""
    rng = random.Random(7)
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)

    def est() -> list[object]:
        e = round(rng.uniform(-0.3, 0.3), 6)
        return [e, round(e - rng.uniform(0.01, 0.1), 6), round(e + rng.uniform(0.01, 0.1), 6)]

    rows: list[list[object]] = []
    for subject in (*SUBJECTS, "ALL_MODELS"):
        for cond in CONDITIONS:
            for tier in TIERS:
                for metric in METRICS:
                    n_items = 60 if tier == "all" else 20
                    rows.append(["real", "abc123", subject, cond, tier, metric, *est(), n_items,
                                 n_items * 3, n_items if metric == "recovery_fraction" else ""])
    write_csv(tables / "metrics.csv", ["data_source", "protocol_hash", "subject", "condition", "tier", "metric",
                                       "estimate", "ci_low", "ci_high", "n_items", "n_cells", "n_qualifying_cells"],
              rows)
    rows = []
    for subject in (*SUBJECTS, "ALL_MODELS"):
        for cond in CONDITIONS[1:]:
            for tier in TIERS:
                for metric in ("bluff_shift", "net_bluff_shift", "evidence_update", "selectivity",
                               "hysteresis_abs", "excess_hysteresis_abs", "brier_T1"):
                    rows.append([subject, f"{cond} - default", tier, metric, *est(), 60, 60, 120])
    write_csv(tables / "condition_contrasts.csv", ["subject", "contrast", "tier", "metric", "estimate", "ci_low",
                                                   "ci_high", "n_items", "n_paired_items", "n_cells"], rows)
    rows = []
    for subject in (*SUBJECTS, "ALL_MODELS"):
        for cond in (*CONDITIONS, "all"):
            for arm in ARMS:
                n_nan, n_reask = rng.randint(0, 9), rng.randint(0, 9)
                rows.append([subject, cond, arm, 180, 180 - n_nan - n_reask, n_reask, n_nan, n_nan / 180,
                             n_reask / 180])
    write_csv(tables / "data_quality.csv", ["subject", "condition", "arm", "n_cells", "n_ok", "n_reask_ok",
                                            "n_nan", "nan_rate", "reask_ok_rate"], rows)
    rows = []
    for subject, samples in (("qwen2.5-7b", 3), ("phi-3.5-mini", 3), ("mistral-7b", 1)):
        for cond in CONDITIONS:
            for sample in range(samples):
                n = 33 if (subject == "phi-3.5-mini" and cond == "contract_prompt") else 60
                rows.append([subject, cond, sample, n, n, n if sample == 0 else 0])
    write_csv(tables / "coverage.csv", ["subject", "condition", "sample", "n_items", "n_cells", "n_judged"], rows)
    hyp_header = ["hypothesis", "description", "subject", "comparison", "metric", "tier", "predicted_sign",
                  "estimate", "ci_low", "ci_high", "n_items", "n_cells", "verdict", "samples", "exclusions",
                  "p_boot", "holm_p", "holm_verdict"]

    def hypotheses() -> list[list[object]]:
        return [[h, f"synthetic {h}", subject, comp, metric, tier, ">0", *est(), 60, 180, "supported", "0",
                 "applied (0 items)", *((0.01, 0.04, "supported") if subject == "ALL_MODELS" else ("", "", ""))]
                for h, comp, metric, tier in HYPOTHESES for subject in (*SUBJECTS, "ALL_MODELS")]
    write_csv(tables / "hypotheses.csv", hyp_header, hypotheses())
    write_csv(tables / "hypotheses_first_attempt.csv", [*hyp_header, "restriction"],
              [[*row, "first_attempt_only"] for row in hypotheses()])
    write_csv(tables / "per_paraphrase.csv",
              ["subject", "condition", "paraphrase_idx", "metric", "estimate", "ci_low", "ci_high", "n_items",
               "n_cells", "exclusions"],
              [[subject, cond, k, metric, *est(), 60, 60, "applied (0 items)"]
               for subject in (*SUBJECTS, "ALL_MODELS") for cond in CONDITIONS
               for k in ((0,) if subject == "mistral-7b" else (0, 1, 2))
               for metric in ("bluff_shift", "net_bluff_shift", "excess_hysteresis_abs",
                              "ablation_T3a_minus_T3")])
    write_csv(tables / "per_tier.csv",
              ["subject", "condition", "metric", "tier", "estimate", "ci_low", "ci_high", "n_items", "n_cells",
               "samples", "exclusions"],
              [[subject, cond, metric, tier,
                *(["", "", ""] if (metric.startswith("brier") and tier == "2") else est()), 20, 60, "0",
                "applied (0 items)"]
               for subject in (*SUBJECTS, "ALL_MODELS") for cond in CONDITIONS for metric in METRICS
               for tier in TIERS[1:]])
    for variant in ("all_samples", "no_exclusions"):
        sub = tables / "robustness" / variant
        sub.mkdir(parents=True)
        write_csv(sub / "hypotheses.csv", hyp_header, hypotheses())
        (sub / "coverage.csv").write_bytes((tables / "coverage.csv").read_bytes())
    tiny_png(figures / "summary_ALL_MODELS.png")
    tiny_png(figures / "hypotheses_forest.png")
    return tables, figures


def small_tables(root: Path) -> tuple[Path, Path]:
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    write_csv(tables / "metrics.csv",
              ["data_source", "subject", "condition", "tier", "metric", "estimate", "ci_low", "ci_high", "n_items"],
              [["real", "ALL_MODELS", "default", "all", "bluff_shift", 0.123456, 0.1, 0.15, 60],
               ["real", "ALL_MODELS", "stand_firm", "all", "bluff_shift", 0.0654, 0.04, 0.09, 60],
               ["real", "ALL_MODELS", "default", "all", "evidence_update", 0.31, 0.25, 0.37, 60],
               ["real", "ALL_MODELS", "stand_firm", "all", "evidence_update", 0.21, "", "", 60],
               ["real", "qwen2.5-7b", "default", "all", "bluff_shift", 0.2, 0.18, 0.22, 60],
               ["real", "qwen2.5-7b", "default", "all", "bluff_shift", 0.3, 0.28, 0.32, 60]])
    write_csv(tables / "coverage.csv", ["subject", "condition", "sample", "n_items", "n_cells", "n_judged"],
              [["qwen2.5-7b", "default", 0, 60, 60, 60], ["qwen2.5-7b", "default", 1, 60, 60, 0],
               ["mistral-7b", "default", 0, 60, 60, 60]])
    tiny_png(figures / "fig.png")
    return tables, figures


def make_ctx(tables: Path, figures: Path, out: Path, draft_dir: Path, **kw: bool) -> bp.BuildContext:
    return bp.BuildContext(tables_dir=tables, figures_dir=figures, config_path=CONFIG, draft_dir=draft_dir,
                           out_dir=out, **kw)


REFS_MD = """# References

- [smith2024] Smith, J. (2024). A Single-Author Paper. arXiv:2401.00001. UNCONFIRMED - user must check.
- [ab2025] Ab, C., & Cd, E. (2025). Two Authors, pp. 11-22. CONFIRMED by user 2026-11-16.
- [many2023] Zed, A., Young, B., & Xu, C. (2023). Three Authors. CONFIRMED.
- [etal2026] Abel, M., Baker, N., et al. (2026). Truncated Author List. UNCONFIRMED - user must check.
- [unused2020] Never, C. (2020). Not Cited. CONFIRMED.
"""


def build_text(tmp_path: Path, draft: str, **kw: bool) -> tuple[bp.BuildContext, list[str], Path]:
    tables, figures = small_tables(tmp_path / "in")
    (tmp_path / "references.md").write_text(REFS_MD, encoding="utf-8")
    out = tmp_path / "out"
    d = tmp_path / "draft.md"
    d.write_text(draft, encoding="utf-8")
    ctx = make_ctx(tables, figures, out, tmp_path, **kw)
    return ctx, bp.build(ctx, d), out


# ---------------------------------------------------------------------------------------------
# build_paper
# ---------------------------------------------------------------------------------------------

def test_num_ci_agg_cfg_data_fill_and_numbers_used(tmp_path: Path) -> None:
    draft = ("Bluff {{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:estimate:.3f}} "
             "{{ci:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:.2f}}; "
             "models {{agg:coverage::subject:nunique:d}}, cells {{agg:coverage:subject=qwen2.5-7b:n_cells:sum:d}}; "
             "T={{cfg:sampling.temperature:.1f}}, B={{cfg:analysis.bootstrap_resamples:,d}}, "
             "CI {{cfg:analysis.ci_level:.0%}}, hf {{cfg:subjects[name=qwen2.5-7b].hf_id:s}}, "
             "items {{data:n_items:d}}, arms {{data:n_arms:d}}.\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    text = (out / "paper.md").read_text(encoding="utf-8")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    qwen = next(s for s in cfg["subjects"] if s["name"] == "qwen2.5-7b")
    n_items = sum(1 for line in (REPO / cfg["paths"]["benchmark"]).read_text(encoding="utf-8").splitlines()
                  if line.strip())
    assert "Bluff 0.123 [0.10, 0.15]" in text
    assert "models 2, cells 120" in text
    assert f"T={cfg['sampling']['temperature']:.1f}" in text
    assert f"B={cfg['analysis']['bootstrap_resamples']:,d}" in text
    assert f"CI {cfg['analysis']['ci_level']:.0%}" in text
    assert f"hf {qwen['hf_id']}" in text
    assert f"items {n_items}," in text
    assert "{{" not in text
    with (out / "numbers_used.csv").open(encoding="utf-8") as fh:
        used = list(csv.DictReader(fh))
    assert list(used[0].keys()) == list(bp.NUMBERS_USED_COLUMNS)
    first = used[0]
    assert first["value"] == "0.123" and first["column"] == "estimate" and first["source_row"] == "0"
    assert first["source_file"].endswith("metrics.csv")
    assert [u["value"] for u in used if u["kind"] == "ci"] == ["0.10", "0.15"]
    assert any(u["kind"] == "agg_sum" and u["source_row"] == "0;1" and u["value"] == "120" for u in used)
    assert (out / "paper.docx").is_file()


@pytest.mark.parametrize("placeholder, message", [
    ("{{num:metrics:subject=qwen2.5-7b;metric=bluff_shift:estimate:.3f}}", "matches 2 rows"),
    ("{{num:metrics:subject=nobody:estimate:.3f}}", "matches 0 rows"),
    ("{{num:metrics:colour=red:estimate:.3f}}", "filter column 'colour'"),
    ("{{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:nope:.3f}}", "column 'nope'"),
    ("{{num:missing_table::estimate:.3f}}", "not found"),
    ("{{wat:metrics}}", "unknown placeholder kind"),
    ("{{fig:absent.png:Figure 1. x}}", "not found"),
    ("{{cfg:sampling.nope:d}}", "not found"),
    ("{{data:n_unicorns:d}}", "unknown data fact"),
    ("{{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:subject:.3f}}", "not numeric"),
])
def test_unresolvable_placeholders_fail_without_outputs(tmp_path: Path, placeholder: str, message: str) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "paper.md").write_text("stale", encoding="utf-8")
    ctx, errors, out = build_text(tmp_path, f"Text {placeholder}.\n")
    assert errors and any(message in e for e in errors), errors
    assert not (out / "paper.md").exists() and not (out / "numbers_used.csv").exists()


def test_cli_exit_1_on_unresolved(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tables, figures = small_tables(tmp_path / "in")
    draft = tmp_path / "draft.md"
    draft.write_text("{{num:metrics:subject=qwen2.5-7b:estimate:.3f}} and {{ broken\n", encoding="utf-8")
    code = bp.main(["--draft", str(draft), "--tables", str(tables), "--figures", str(figures),
                    "--out-dir", str(tmp_path / "out"), "--config", str(CONFIG)])
    assert code == 1
    assert "BUILD FAILED" in capsys.readouterr().err


def test_missing_value_renders_na_with_warning(tmp_path: Path) -> None:
    ctx, errors, out = build_text(
        tmp_path, "{{ci:metrics:subject=ALL_MODELS;condition=stand_firm;metric=evidence_update:.2f}}\n")
    assert errors == []
    assert "[n/a, n/a]" in (out / "paper.md").read_text(encoding="utf-8")
    assert ctx.warnings


def test_dry_run_rows_rejected(tmp_path: Path) -> None:
    tables, figures = small_tables(tmp_path / "in")
    path = tables / "metrics.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("real,", "DRY RUN (mock client) - not results,"),
                    encoding="utf-8")
    draft = tmp_path / "d.md"
    draft.write_text("{{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:estimate:.3f}}\n",
                     encoding="utf-8")
    errors = bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path), draft)
    assert any("dry-run" in e for e in errors)
    ok = bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path, allow_dry_run=True), draft)
    assert ok == []


def test_defaults_directive_applies_and_is_overridable(tmp_path: Path) -> None:
    draft = ("{{defaults:metrics:condition=stand_firm}}\n"
             "{{num:metrics:subject=ALL_MODELS;metric=bluff_shift:estimate:.4f}} "
             "{{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:estimate:.4f}}\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    text = (out / "paper.md").read_text(encoding="utf-8")
    assert "0.0654 0.1235" in text and "defaults" not in text


def test_table_and_pivot_render_in_filter_order(tmp_path: Path) -> None:
    draft = ("{{table:metrics:subject=ALL_MODELS;condition=stand_firm|default:condition,metric,estimate=.2f,ci=.2f}}"
             "\n\n{{pivot:metrics:subject=ALL_MODELS;metric=evidence_update|bluff_shift:metric:condition:"
             "estimate+ci:.2f}}\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    lines = (out / "paper.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "| condition | metric | estimate | ci |"
    assert lines[2].startswith("| stand_firm |")  # filter order, not file order
    assert "| evidence_update | 0.31 [0.25, 0.37] | 0.21 [n/a, n/a] |" in lines
    assert "| bluff_shift | 0.12 [0.10, 0.15] | 0.07 [0.04, 0.09] |" in lines


def test_table_names_with_subdirectories(tmp_path: Path) -> None:
    tables, figures = small_tables(tmp_path / "in")
    sub = tables / "robustness" / "all_samples"
    sub.mkdir(parents=True)
    write_csv(sub / "metrics.csv", ["data_source", "subject", "condition", "tier", "metric", "estimate", "ci_low",
                                    "ci_high", "n_items"],
              [["real", "ALL_MODELS", "default", "all", "bluff_shift", 0.4321, 0.4, 0.46, 60]])
    draft = tmp_path / "d.md"
    draft.write_text("{{defaults:metrics:condition=stand_firm}}\n"
                     "{{num:robustness/all_samples/metrics:subject=ALL_MODELS;metric=bluff_shift:estimate:.3f}} "
                     "{{ci:robustness/all_samples/metrics:subject=ALL_MODELS;metric=bluff_shift:.2f}}\n",
                     encoding="utf-8")
    ctx = make_ctx(tables, figures, tmp_path / "o", tmp_path)
    assert bp.build(ctx, draft, docx=False) == []
    # the main table's defaults do not leak into the robustness table of the same base name
    assert "0.432 [0.40, 0.46]" in (tmp_path / "o" / "paper.md").read_text(encoding="utf-8")
    assert all(u.source_file.endswith("robustness/all_samples/metrics.csv") for u in ctx.used)


def test_pivot_ambiguous_cell_errors(tmp_path: Path) -> None:
    ctx, errors, _ = build_text(tmp_path, "{{pivot:metrics:subject=qwen2.5-7b:metric:condition:estimate:.2f}}\n")
    assert any("add a filter" in e for e in errors)


def test_html_comments_dropped_and_include(tmp_path: Path) -> None:
    (tmp_path / "inc.md").write_text("<!-- hidden -->Included text.\n", encoding="utf-8")
    ctx, errors, out = build_text(tmp_path, "<!-- {{num:nonsense}} -->\n{{include:inc.md}}\n")
    assert errors == []
    text = (out / "paper.md").read_text(encoding="utf-8")
    assert "Included text." in text and "hidden" not in text and "nonsense" not in text


def test_final_flag_rejects_editorial_markers(tmp_path: Path) -> None:
    tables, figures = small_tables(tmp_path / "in")
    (tmp_path / "references.md").write_text(REFS_MD, encoding="utf-8")
    draft = tmp_path / "d.md"
    draft.write_text("As shown [CITE:smith2024]. [RESULT PROSE - after freeze]\n", encoding="utf-8")
    assert bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path), draft) == []
    errors = bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path), draft, final=True)
    assert len(errors) == 2
    assert any("unfinished marker [CITE:smith2024] (--final)" in e for e in errors)


def test_citations_render_author_year(tmp_path: Path) -> None:
    draft = ("One [CITE:smith2024]. Two [CITE:ab2025]. Three [CITE:many2023]. "
             "Many [CITE:etal2026, smith2024].\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    text = (out / "paper.md").read_text(encoding="utf-8")
    assert "One (Smith, 2024)." in text
    assert "Two (Ab & Cd, 2025)." in text
    assert "Three (Zed et al., 2023)." in text
    assert "Many (Abel et al., 2026; Smith, 2024)." in text
    assert "[CITE:" not in text


def test_citation_with_missing_entry_fails(tmp_path: Path) -> None:
    ctx, errors, out = build_text(tmp_path, "See [CITE:smith2024,nobody2030].\n")
    assert any("nobody2030" in e for e in errors)
    assert not (out / "paper.md").exists()


def test_refs_placeholder_lists_cited_entries_sorted(tmp_path: Path) -> None:
    draft = "A [CITE:smith2024] and [CITE:ab2025,many2023].\n\n## References\n\n{{refs}}\n"
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    text = (out / "paper.md").read_text(encoding="utf-8")
    refs = text.split("## References", 1)[1].strip().splitlines()
    assert refs == ["- Ab, C., & Cd, E. (2025). Two Authors, pp. 11-22.",
                    "- Smith, J. (2024). A Single-Author Paper. arXiv:2401.00001.",
                    "- Zed, A., Young, B., & Xu, C. (2023). Three Authors."]
    assert "Never" not in text  # uncited entries are not listed
    # numbers inside entries are traced to references.md, so the audit passes
    code = na.main([str(out / "paper.md"), "--numbers-used", str(out / "numbers_used.csv"), "--config", str(CONFIG)])
    assert code == 0


def test_final_renders_confirmed_citations_only(tmp_path: Path) -> None:
    tables, figures = small_tables(tmp_path / "in")
    (tmp_path / "references.md").write_text(REFS_MD, encoding="utf-8")
    draft = tmp_path / "d.md"
    draft.write_text("Ok [CITE:ab2025,many2023]. Bad [CITE:ab2025,smith2024].\n\n{{refs}}\n", encoding="utf-8")
    errors = bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path), draft, final=True)
    assert len(errors) == 1 and "[CITE:ab2025,smith2024]" in errors[0]
    draft.write_text("Ok [CITE:ab2025,many2023].\n\n{{refs}}\n", encoding="utf-8")
    assert bp.build(make_ctx(tables, figures, tmp_path / "o", tmp_path), draft, final=True) == []
    assert "Ok (Ab & Cd, 2025; Zed et al., 2023)." in (tmp_path / "o" / "paper.md").read_text(encoding="utf-8")


def test_missing_text_format_suffix(tmp_path: Path) -> None:
    draft = ("{{num:metrics:subject=ALL_MODELS;condition=stand_firm;metric=evidence_update:ci_low:.2f?\u2013}}\n\n"
             "{{pivot:metrics:subject=ALL_MODELS;metric=evidence_update:metric:condition:estimate+ci:.2f?\u2013}}\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == [] and not ctx.warnings
    text = (out / "paper.md").read_text(encoding="utf-8")
    assert text.startswith("\u2013\n")
    assert "| evidence_update | 0.31 [0.25, 0.37] | 0.21 [\u2013, \u2013] |" in text


def test_docx_created_and_opens(tmp_path: Path) -> None:
    from docx import Document

    draft = ("# Title\n\n## Section\n\nA **bold** and *italic* paragraph with `code`.\n\n- bullet one\n\n"
             "{{table:metrics:subject=ALL_MODELS:condition,metric,estimate=.3f}}\n\n"
             "{{fig:fig.png:Figure 1. A caption: with colon.}}\n\n```text\nline one\nline two\n```\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    doc = Document(str(out / "paper.docx"))
    texts = [p.text for p in doc.paragraphs]
    assert "Title" in texts and "Section" in texts
    assert any("bold" in t and "italic" in t for t in texts)
    assert "Figure 1. A caption: with colon." in texts
    assert any("line one" in t and "line two" in t for t in texts)
    assert len(doc.tables) == 1
    assert doc.tables[0].cell(0, 0).text == "Condition"
    assert doc.tables[0].cell(1, 2).text == "0.123"
    assert len(doc.inline_shapes) == 1


def test_config_lookup_list_selection() -> None:
    cfg = {"subjects": [{"name": "a.b-1", "tp": 2}, {"name": "c", "tp": 1}], "x": {"y": 3}}
    assert bp.config_lookup(cfg, "subjects[name=a.b-1].tp") == 2
    assert bp.config_lookup(cfg, "x.y") == 3
    with pytest.raises(bp.BuildError):
        bp.config_lookup(cfg, "subjects[name=zz].tp")


def test_format_value() -> None:
    assert bp.format_value("-0.0001", ".3f") == "0.000"
    assert bp.format_value("0.5", "+.2f") == "+0.50"
    assert bp.format_value("12.0", "d") == "12"
    assert bp.format_value("nan", ".2f") is None
    with pytest.raises(bp.BuildError):
        bp.format_value("12.5", "d")


# ---------------------------------------------------------------------------------------------
# numbers_audit
# ---------------------------------------------------------------------------------------------


def _build_small(tmp_path: Path) -> Path:
    draft = ("## Results\n\nBluff {{num:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:estimate:.3f}} "
             "{{ci:metrics:subject=ALL_MODELS;condition=default;metric=bluff_shift:.2f}} over "
             "{{data:n_items:d}} items at temperature {{cfg:sampling.temperature:.1f}} with "
             "{{cfg:analysis.bootstrap_resamples:,d}} resamples ({{cfg:analysis.ci_level:.0%}} CI).\n\n"
             "See Table 2 and Section 4.1 (H1-H6; qwen2.5-7b, Phi-3.5-mini, gpt-oss-120b; item EH001; T2A/T3a). "
             "Track 1, 13-15 November 2026, https://example.org/v2/x1, [CITE:smith2024]. "
             "Extremity gap `|J - 0.5| - |c1 - 0.5|`.\n")
    ctx, errors, out = build_text(tmp_path, draft)
    assert errors == []
    return out


def test_audit_passes_on_correct_build(tmp_path: Path) -> None:
    out = _build_small(tmp_path)
    code = na.main([str(out / "paper.md"), "--numbers-used", str(out / "numbers_used.csv"), "--config", str(CONFIG)])
    assert code == 0


def test_audit_fails_on_planted_number(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _build_small(tmp_path)
    paper = out / "paper.md"
    paper.write_text(paper.read_text(encoding="utf-8") + "\nThe effect was 0.4271 overall.\n", encoding="utf-8")
    code = na.main([str(paper), "--numbers-used", str(out / "numbers_used.csv"), "--config", str(CONFIG)])
    assert code == 1
    err = capsys.readouterr().err
    assert "'0.4271'" in err and "The effect was" in err


def test_audit_catches_hand_rounded_and_altered_values(tmp_path: Path) -> None:
    out = _build_small(tmp_path)
    allowed = na.build_whitelist(CONFIG, out / "numbers_used.csv")
    bad, _ = na.audit("Bluff 0.12 and 0.124 and -0.123.", allowed)  # build wrote 0.123
    assert [h.token for h in bad] == ["0.12", "0.124", "-0.123"]
    bad, _ = na.audit("Bluff 0.123 [0.10, 0.15].", allowed)
    assert bad == []


def test_audit_ignores_identifiers_dates_urls_and_section_numbers() -> None:
    text = ("H1 to H6, T2A, T3a, c3N, qwen2.5-7b, Phi-3.5-mini, gpt-oss-120b, Llama-3.1-8B, EH001, "
            "Qwen/Qwen2.5-7B-Instruct-AWQ, 7B. Table 3, Figure 1, Section 4.2, Appendix B, tiers 1-3, Turn 2, "
            "(1) and (2). 14 Nov 2026; 2026-11-15; https://arxiv.org/abs/2609.33672 [CITE:shnaidman2026reset]")
    assert na.extract_numbers(text) == []
    hits = na.extract_numbers("values 0.25, −0.5 and 10,000 and 95% and 4-bit")
    assert [str(h.value) for h in hits] == ["0.25", "-0.5", "10000", "95", "4"]


def test_audit_reads_pdf(tmp_path: Path) -> None:
    out = _build_small(tmp_path)
    pdf = tmp_path / "paper.pdf"
    _minimal_pdf(pdf, "Bluff shift was 0.123 and the planted value 0.9876 appears here.")
    assert "0.9876" in na.read_document(pdf)
    allowed = na.build_whitelist(CONFIG, out / "numbers_used.csv")
    bad, n = na.audit(na.read_document(pdf), allowed)
    assert n == 2 and [h.token for h in bad] == ["0.9876"]


def _minimal_pdf(path: Path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    body += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    body += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(body)


# ---------------------------------------------------------------------------------------------
# The real draft builds and audits cleanly on complete synthetic tables
# ---------------------------------------------------------------------------------------------


def test_real_draft_builds_on_synthetic_tables_and_passes_audit(tmp_path: Path) -> None:
    from docx import Document

    tables, figures = synthetic_tables(tmp_path / "in")
    out = tmp_path / "out"
    ctx = make_ctx(tables, figures, out, DRAFT.parent)
    errors = bp.build(ctx, DRAFT)
    assert errors == [], errors
    text = (out / "paper.md").read_text(encoding="utf-8")
    assert "{{" not in text and "}}" not in text
    assert "Limitations and Dual-Use / Ethical Considerations" in text
    for cond in prompts.CONDITIONS:
        assert prompts.system_prompt(cond) in text
    Document(str(out / "paper.docx"))
    code = na.main([str(out / "paper.md"), "--numbers-used", str(out / "numbers_used.csv"), "--config", str(CONFIG)])
    assert code == 0
    assert "## References" in text and "(Shnaidman, 2026)" in text and "[CITE:" not in text
    assert "| brier_T1 | " in text and "| \u2013 |" in text  # tier-2 Brier rendered as a dash
    assert bp.build(make_ctx(tables, figures, out, DRAFT.parent), DRAFT, final=True)  # markers remain


# ---------------------------------------------------------------------------------------------
# export_prompts
# ---------------------------------------------------------------------------------------------


def test_export_prompts_verbatim() -> None:
    turns_path = ep.default_turns_path()
    text = ep.render(turns_path)
    for cond in prompts.CONDITIONS:
        assert prompts.system_prompt(cond) in text
    for fixed in (prompts.PROBE_INSTRUCTION, prompts.REASK_MESSAGE, prompts.JUDGE_REASK_MESSAGE,
                  prompts.NEUTRAL_PLACEHOLDER, prompts._STAND_FIRM_TEMPLATE, prompts.JUDGE_SYSTEM,
                  prompts.judge_messages("{claim}", "{prose}")[1]["content"]):
        assert fixed in text
    turns = yaml.safe_load(turns_path.read_text(encoding="utf-8"))
    for paraphrases in turns.values():
        for p in paraphrases:
            assert p in text


def test_committed_appendix_is_up_to_date() -> None:
    assert ep.main(["--check"]) == 0


def test_publication_text_drops_working_notes() -> None:
    entry = ("Hong, J. (2025). Measuring Sycophancy. Findings of EMNLP 2025, pp. 2239-2259 (per arXiv journal-ref). "
             "arXiv:2505.23840. https://arxiv.org/abs/2505.23840. Author list needs checking. "
             "UNCONFIRMED - user must check.")
    out = bp.publication_text(entry)
    assert out.endswith("https://arxiv.org/abs/2505.23840.")
    assert "UNCONFIRMED" not in out and "per arXiv" not in out and "needs checking" not in out
    assert bp.publication_text("Doe, A. (2020). No link. CONFIRMED.") == "Doe, A. (2020). No link."


def test_typeset_numbers_minus_and_unbreakable_ci() -> None:
    out = bp.typeset_numbers("fell (-0.067 [-0.095, -0.040]) in qwen2.5-7b, stand-firm, tiers 1-3")
    assert "−0.067 [−0.095, −0.040]" in out
    assert "qwen2.5-7b" in out and "stand-firm" in out and "1-3" in out


def test_docx_layout_tables_and_references(tmp_path: Path) -> None:
    from docx import Document

    md = ("# Title\n\nJane Doe\n\n**Abstract.** Short.\n\n## Results\n\n**Table 1. Demo.**\n\n"
          "| metric | estimate | ci |\n|---|---|---|\n| net_bluff_shift | 0.184 | [0.157, 0.211] |\n\n"
          "## References\n\n- Doe, A. (2020). A paper. https://example.org.\n")
    out = tmp_path / "p.docx"
    bp.markdown_to_docx(md, out, tmp_path)
    doc = Document(str(out))
    assert doc.styles["Normal"].font.name == bp.BODY_FONT
    header = [c.text for c in doc.tables[0].rows[0].cells]
    assert header == ["Metric", "Estimate", "95% CI"]
    ref = [p for p in doc.paragraphs if p.text.startswith("Doe, A.")][0]
    assert ref.style.name == "Normal" and ref.paragraph_format.first_line_indent is not None
