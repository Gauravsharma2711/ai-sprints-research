"""Offline tests for tools/build_readme.py.

Tables are synthetic (random values, fixed seed) and live in tmp_path; no real results are read
except by the link/stale-string checks on the committed README.md.
"""

from __future__ import annotations

import csv
import random
import re
import struct
import zlib
from pathlib import Path

import pytest

from tools import build_readme as br

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config.yaml"
TEMPLATE = REPO / "README.template.md"
DRAFT = REPO / "paper" / "draft.md"

SUBJECTS = ("qwen2.5-7b", "phi-3.5-mini", "mistral-7b")
CONDITIONS = ("default", "stand_firm", "contract_prompt")
HYPOTHESES = (  # (id, description, comparison, metric, tier, predicted sign)
    ("H1", "prose more extreme", "default", "extremity_gap", "3", ">0"),
    ("H2", "bluff beats neutral", "default", "net_bluff_shift", "all", ">0"),
    ("H3", "stand-firm reduces bluff shift", "stand_firm - default", "bluff_shift", "all", "<0"),
    ("H3", "stand-firm reduces evidence update", "stand_firm - default", "evidence_update", "all", "<0"),
    ("H4", "excess hysteresis", "default", "excess_hysteresis_abs", "all", ">0"),
    ("H5", "ablation a", "default", "ablation_T3a_minus_T3", "all", "!=0"),
    ("H5", "ablation b", "default", "ablation_T3b_minus_T3", "all", "!=0"),
    ("H5", "ablation c", "default", "ablation_T3c_minus_T3", "all", "!=0"),
    ("H6", "contract selectivity", "contract_prompt - default", "selectivity", "all", ">0"),
    ("H6", "contract hysteresis", "contract_prompt - default", "excess_hysteresis_abs", "all", "<0"),
)
STALE = ("Not started", "dry-run only", "to be written", "_None yet", "Not yet run on Kaggle")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def tiny_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def synthetic(root: Path) -> tuple[Path, Path]:
    rng = random.Random(11)
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    rows: list[list[object]] = []
    for hid, desc, comp, metric, tier, sign in HYPOTHESES:
        for subject in (*SUBJECTS, "ALL_MODELS"):
            e = round(rng.uniform(-0.3, 0.3), 6)
            rows.append([hid, desc, subject, comp, metric, tier, sign, e, e - 0.05, e + 0.05, 20, 60,
                         rng.choice(["supported", "inconclusive", "opposite"]), "0", "applied (0 items)"])
    write_csv(tables / "hypotheses.csv",
              ["hypothesis", "description", "subject", "comparison", "metric", "tier", "predicted_sign",
               "estimate", "ci_low", "ci_high", "n_items", "n_cells", "verdict", "samples", "exclusions"], rows)
    cov: list[list[object]] = []
    for subject in SUBJECTS:
        for cond in CONDITIONS:
            for sample in (0, 1):
                cov.append([subject, cond, sample, 7, 7, 7 if sample == 0 else 0])
    write_csv(tables / "coverage.csv", ["subject", "condition", "sample", "n_items", "n_cells", "n_judged"], cov)
    met: list[list[object]] = []
    for subject in (*SUBJECTS, "ALL_MODELS"):
        for cond in CONDITIONS:
            e = round(rng.uniform(-0.3, 0.3), 6)
            met.append(["real", subject, cond, "all", "selectivity", e, e - 0.05, e + 0.05, 20])
    write_csv(tables / "metrics.csv",
              ["data_source", "subject", "condition", "tier", "metric", "estimate", "ci_low", "ci_high", "n_items"], met)
    tiny_png(figures / "summary_ALL_MODELS.png")
    return tables, figures


def run(tmp: Path, tables: Path, figures: Path, *extra: str, template: Path = TEMPLATE) -> int:
    return br.main(["--template", str(template), "--out", str(tmp / "README.md"), "--config", str(CONFIG),
                    "--draft", str(DRAFT), "--tables", str(tables), "--figures", str(figures), *extra])


def test_render_real_template(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    assert run(tmp_path, tables, figures) == 0
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "{{" not in text and "}}" not in text
    template = TEMPLATE.read_text(encoding="utf-8")
    names = re.findall(r"<!-- BEGIN GENERATED: ([\w.-]+) -->", template)
    assert names, "the template has no generated blocks"
    for name in names:
        assert f"<!-- BEGIN GENERATED: {name} -->" in text
        assert f"<!-- END GENERATED: {name} -->" in text
    assert "<!-- TEMPLATE:" not in text
    assert "summary_ALL_MODELS.png" in text
    # synthetic coverage: 2 samples x 3 conditions x 7 cells per model
    assert "| qwen2.5-7b | `" in text and "| 2 | 42 | 21 |" in text


def test_check_pass_and_fail(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    out = tmp_path / "README.md"
    assert run(tmp_path, tables, figures, "--check") == 1  # README missing
    assert run(tmp_path, tables, figures) == 0
    assert run(tmp_path, tables, figures, "--check") == 0
    out.write_text(out.read_text(encoding="utf-8") + "hand edit\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, "--check") == 1
    # a changed table makes the committed README stale
    assert run(tmp_path, tables, figures) == 0
    lines = (tables / "coverage.csv").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].rsplit(",", 3)[0] + ",8,8,8"
    (tables / "coverage.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, "--check") == 1


def test_missing_table_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tables, figures = synthetic(tmp_path / "src")
    (tables / "coverage.csv").unlink()
    assert run(tmp_path, tables, figures) == 1
    assert not (tmp_path / "README.md").exists()
    assert "coverage" in capsys.readouterr().err


def test_missing_figure_is_an_error(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    (figures / "summary_ALL_MODELS.png").unlink()
    assert run(tmp_path, tables, figures) == 1


def test_placeholder_outside_block_and_bad_markers(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    outside = tmp_path / "outside.md"
    outside.write_text("# T\n\n{{data:n_items:d}} items\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, template=outside) == 1
    unmatched = tmp_path / "unmatched.md"
    unmatched.write_text("<!-- BEGIN GENERATED: a -->\n{{data:n_items:d}}\n<!-- END GENERATED: b -->\n",
                         encoding="utf-8")
    assert run(tmp_path, tables, figures, template=unmatched) == 1
    unknown = tmp_path / "unknown.md"
    unknown.write_text("<!-- BEGIN GENERATED: a -->\n{{readme:nope}} {{bogus:x}}\n<!-- END GENERATED: a -->\n",
                       encoding="utf-8")
    assert run(tmp_path, tables, figures, template=unknown) == 1
    assert not (tmp_path / "README.md").exists()


def test_small_template_values(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    tpl = tmp_path / "t.md"
    tpl.write_text("<!-- TEMPLATE: dropped -->\n# Static\n\n<!-- BEGIN GENERATED: n -->\n"
                   "{{agg:coverage::n_cells:sum:d}} cells; title {{readme:paper_title}}\n"
                   "<!-- END GENERATED: n -->\n\nTail.\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, template=tpl) == 0
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED: n -->\n126 cells; title " in text
    assert "dropped" not in text and text.endswith("Tail.\n")


def test_committed_readme_links_and_stale_strings() -> None:
    readme = REPO / "README.md"
    text = readme.read_text(encoding="utf-8")
    for s in STALE:
        assert s not in text, f"stale string {s!r} in README.md"
    assert "{{" not in text
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    missing = []
    for t in targets:
        if re.match(r"^[a-z]+://", t) or t.startswith("#"):
            continue
        path = t.split("#", 1)[0]
        if not (REPO / path).exists():
            missing.append(t)
    assert not missing, f"README links that do not resolve: {missing}"


def _set_selectivity_ci(tables: Path, lo: str, hi: str) -> None:
    path = tables / "metrics.csv"
    rows = list(csv.reader(path.open(encoding="utf-8", newline="")))
    for r in rows[1:]:
        if r[1] == "ALL_MODELS" and r[2] == "default":
            r[6], r[7] = lo, hi
    write_csv(path, rows[0], rows[1:])


@pytest.mark.parametrize(("lo", "hi", "phrase"), [
    ("-0.01", "0.2", "inconclusive (CI includes 0)"),
    ("0.01", "0.2", "CI excludes 0, positive"),
    ("-0.2", "-0.01", "CI excludes 0, negative"),
])
def test_ci_zero_phrase(tmp_path: Path, lo: str, hi: str, phrase: str) -> None:
    tables, figures = synthetic(tmp_path / "src")
    _set_selectivity_ci(tables, lo, hi)
    tpl = tmp_path / "t.md"
    tpl.write_text("<!-- BEGIN GENERATED: s -->\n"
                   "{{readme:ci_zero:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity}}\n"
                   "<!-- END GENERATED: s -->\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, template=tpl) == 0
    assert phrase in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_ci_zero_missing_row_or_nan_is_an_error(tmp_path: Path) -> None:
    tables, figures = synthetic(tmp_path / "src")
    tpl = tmp_path / "t.md"
    tpl.write_text("<!-- BEGIN GENERATED: s -->\n{{readme:ci_zero:metrics:subject=nobody}}\n"
                   "<!-- END GENERATED: s -->\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, template=tpl) == 1
    _set_selectivity_ci(tables, "nan", "0.1")
    tpl.write_text("<!-- BEGIN GENERATED: s -->\n"
                   "{{readme:ci_zero:metrics:subject=ALL_MODELS;condition=default;tier=all;metric=selectivity}}\n"
                   "<!-- END GENERATED: s -->\n", encoding="utf-8")
    assert run(tmp_path, tables, figures, template=tpl) == 1
    assert not (tmp_path / "README.md").exists()


BANNED_JUDGE_WORDS = re.compile(r"groq|gpt-oss|\bxai\b|x\.ai|grok|second judge|second_judge|adaption",
                                re.IGNORECASE)


@pytest.mark.parametrize("rel", ["README.template.md", "README.md", "docs/kaggle_runbook.md",
                                 "tools/kaggle_sim/run_sim.py", "tools/kaggle_sim/fake_openai.py"])
def test_gemini_is_the_only_judge(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    hits = sorted({m.group(0) for m in BANNED_JUDGE_WORDS.finditer(text)})
    assert not hits, f"{rel} mentions {hits}"


def test_license_citation_and_ai_use() -> None:
    assert "Copyright (c) 2026 Gaurav Sharma" in (REPO / "LICENSE").read_text(encoding="utf-8")
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in cff and "family-names: Sharma" in cff and "license: MIT" in cff
    assert br.paper_title(DRAFT) in cff
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "author       = {Gaurav Sharma}" in text and "](LICENSE)" in text
    assert "The benchmark labels have not been independently verified by a human." in text
    for claim in ("reviewed the items", "rated briefings", "confirmed the citations", "being added in parallel"):
        assert claim not in text, claim