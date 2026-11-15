"""Offline tests for the pure checks in tools/gates.py. Synthetic text only; no subprocesses, no network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import gates as g

JUDGE = "judge-model-x"


def _words(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


def make_draft(abstract: str = "", *, authors: bool = False, limitations: str = g.LIMITATIONS_HEADING,
               related: bool = True, ai_use: str | None = None, main_extra: str = "") -> str:
    ai = ai_use if ai_use is not None else (
        f"Claude (Claude Code, Anthropic) wrote the code. The judge was {JUDGE}. "
        "The author reviewed the items and confirmed the citations."
    )
    parts = [
        "<!-- a comment with {{num:x}} and causal scrubbing -->",
        "# A Title",
        "",
        "[AUTHOR NAMES]\n\n[AFFILIATIONS]" if authors else "Jane Doe",
        "",
        f"**Abstract.** {abstract}",
        "",
        "## Introduction",
        "",
        "Intro prose here.",
        main_extra,
    ]
    if related:
        parts += ["## Related Work", "", "Related prose."]
    parts += [
        "## Methodology", "", "Method prose.",
        "## Results", "", "Results prose.",
        "## Discussion", "", "Discussion prose.",
        "## References", "", "- refs",
        "## Appendix A. Extra", "", "Appendix prose.",
        f"## {limitations}", "", "Limits.",
        "## Preexisting Work", "", "Plan.",
        "## AI-Use Statement", "", ai,
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------
# word counting and G7
# ---------------------------------------------------------------------------------------------


def test_count_words_counts_placeholders_and_markers_as_one() -> None:
    text = "We found {{num:metrics:subject=ALL MODELS;metric=x:estimate:.3f}} and [CITE:a, b] here ."
    # We, found, <placeholder>, and, <marker>, here  -> the lone "." is not a word
    assert g.count_words(text) == 6


def test_abstract_marker_fails_not_written() -> None:
    draft = make_draft("[ABSTRACT - write last]")
    status, detail = g.check_abstract(draft)
    assert status == "FAIL" and "not written" in detail


def test_abstract_word_count_with_placeholders() -> None:
    body = _words(198) + " {{num:a:b c d e}} {{ci:x y z}}"  # 198 + 2 placeholders = 200
    assert g.count_words(g.extract_abstract(make_draft(body)) or "") == 200
    assert g.check_abstract(make_draft(body))[0] == "PASS"
    status, detail = g.check_abstract(make_draft(body, authors=True))
    assert status == "BLOCKED" and "[AUTHOR NAMES]" in detail


def test_abstract_too_short_or_long_fails() -> None:
    assert g.check_abstract(make_draft(_words(149)))[0] == "FAIL"
    assert g.check_abstract(make_draft(_words(251)))[0] == "FAIL"
    assert g.check_abstract(make_draft(_words(150)))[0] == "PASS"
    assert g.check_abstract(make_draft(_words(250)))[0] == "PASS"


def test_abstract_under_heading_and_title_required() -> None:
    draft = "# T\n\n## Abstract\n\n" + _words(160) + "\n\n## Introduction\n\nx\n"
    assert g.check_abstract(draft)[0] == "PASS"
    no_title = draft.replace("# T\n", "")
    status, detail = g.check_abstract(no_title)
    assert status == "FAIL" and "title" in detail


# ---------------------------------------------------------------------------------------------
# G8 structure
# ---------------------------------------------------------------------------------------------


def test_structure_complete_passes() -> None:
    assert g.check_structure(make_draft(_words(160)), JUDGE)[0] == "PASS"


def test_structure_accepts_judge_placeholder() -> None:
    ai = "Claude drafted it. Judge {{cfg:judge.model:s}}. The author rated briefings."
    assert g.check_structure(make_draft(ai_use=ai), JUDGE)[0] == "PASS"


def test_structure_reports_missing_parts() -> None:
    draft = make_draft(related=False, limitations="Limitations", ai_use="Some model wrote this.")
    status, detail = g.check_structure(draft, JUDGE)
    assert status == "FAIL"
    assert "Related Work" in detail
    assert g.LIMITATIONS_HEADING in detail
    assert "Claude" in detail and JUDGE in detail and "reviewed" in detail


def test_structure_limitations_with_appendix_prefix() -> None:
    draft = make_draft(limitations=f"Appendix E. {g.LIMITATIONS_HEADING}")
    assert g.check_structure(draft, JUDGE)[0] == "PASS"


# ---------------------------------------------------------------------------------------------
# G9 length
# ---------------------------------------------------------------------------------------------


def test_length_counts_main_text_tables_and_figures() -> None:
    extra = "\n".join([
        "| a | b |", "|---|---|", "| 1 | 2 |",          # one markdown table, no words
        "",
        "{{table:metrics::a,b}}",                       # table placeholder line, no words
        "{{pivot:metrics:x:y:z:estimate:.3f}}",
        "{{fig:f.png:Figure 1. caption words}}",
        "```", "code words here", "```",
        "<!-- hidden words -->",
        "Prose with {{num:x}} inline.",                  # 4 words
    ])
    draft = make_draft(_words(160), main_extra=extra)
    words, tf = g.measure_length(draft)
    assert tf == 4
    # heading words ('##' is not a word): Introduction(1) Related Work(2) Methodology(1) Results(1)
    # Discussion(1) = 6; prose: "Intro prose here."(3) "Related prose."(2) "Method prose."(2)
    # "Results prose."(2) "Discussion prose."(2) = 11; extra prose line = 4. References onward excluded.
    assert words == 6 + 11 + 4
    status, detail = g.check_length(draft)
    assert status == "PASS" and "4 tables+figures" in detail


def test_length_fails_over_limits() -> None:
    many = "\n".join(f"{{{{table:t{i}::a}}}}" for i in range(6))
    assert g.check_length(make_draft(main_extra=many))[0] == "FAIL"
    assert g.check_length(make_draft(main_extra=_words(3801)))[0] == "FAIL"


# ---------------------------------------------------------------------------------------------
# G10 citations
# ---------------------------------------------------------------------------------------------

REFS = "\n".join([
    "# References",
    "- [a2024] A. (2024). Title. CONFIRMED by user.",
    "- [b2025] B. (2025). Title. UNCONFIRMED - user must check.",
])


def test_citations_pass_blocked_fail() -> None:
    assert g.check_citations("x [CITE:a2024] y", REFS)[0] == "PASS"
    status, detail = g.check_citations("x [CITE:a2024, b2025]", REFS)
    assert status == "BLOCKED" and "b2025" in detail and "a2024" not in detail.split(":")[-1]
    status, detail = g.check_citations("x [CITE:a2024] [CITE:zz]", REFS)
    assert status == "FAIL" and "zz" in detail


def test_reference_entries_confirmed_flag() -> None:
    assert g.reference_entries(REFS) == {"a2024": True, "b2025": False}


def test_classify_build_errors_human_vs_other() -> None:
    entries = g.reference_entries(REFS)
    stderr = "\n".join([
        "WARNING: something",
        "BUILD FAILED: 4 unresolved placeholder(s); no outputs written.",
        "  line 3: unfinished marker [AUTHOR NAMES] (--final)",
        "  line 9: unfinished marker [CITE:b2025] (--final)",
        "  line 9: unfinished marker [CITE:a2024] (--final)",
        "  line 12: unfinished marker [RESULT PROSE - after freeze] (--final)",
    ])
    human, other = g.classify_build_errors(stderr, entries)
    assert human == ["[AUTHOR NAMES]", "[CITE:b2025]"]
    assert other == ["[CITE:a2024]", "[RESULT PROSE - after freeze]"]


# ---------------------------------------------------------------------------------------------
# G11 wording lint
# ---------------------------------------------------------------------------------------------


def test_lint_hits_with_line_numbers() -> None:
    text = "Line one.\nThis proves it.\nA novel method.\nTo our knowledge, this novel design is new.\n" \
           "We test frontier LLMs.\nNo causal scrubbing here.\n<!-- proven in a comment -->\n"
    hits = {(h.line, h.term) for h in g.lint_text(text, "f.md")}
    assert hits == {(2, "proves"), (3, "novel"), (5, "frontier llm"), (6, "causal scrubbing")}


def test_lint_qualifier_only_in_same_sentence() -> None:
    text = "To our knowledge, this is fine. But this is novel.\n"
    hits = g.lint_text(text, "f.md")
    assert [(h.line, h.term) for h in hits] == [(1, "novel")]


def test_hypothesis_claims_flag_contradictions() -> None:
    verdicts = g.pooled_verdicts([
        {"hypothesis": "H1", "subject": "ALL_MODELS", "verdict": "inconclusive"},
        {"hypothesis": "H2", "subject": "ALL_MODELS", "verdict": "supported"},
        {"hypothesis": "H2", "subject": "qwen", "verdict": "opposite"},
    ])
    draft = "H1 was supported overall. H2 was supported.\nH2 was not supported in the pooled data.\n" \
            "{{table:hypotheses::H1,supported}}\n"
    problems = g.check_hypothesis_claims(draft, verdicts, "d.md")
    assert len(problems) == 2
    assert problems[0].startswith("d.md:1 H1")
    assert problems[1].startswith("d.md:2 H2") and "not supported" in problems[1]


# ---------------------------------------------------------------------------------------------
# G12 README
# ---------------------------------------------------------------------------------------------

GOOD_README = "\n".join([
    "# Project", "## TL;DR", "## Research question", "## Method", "## Results",
    "<!-- BEGIN GENERATED results -->", "x", "<!-- END GENERATED results -->",
    "## Models", "## Data", "## Reproducing", "## Repository structure", "## Limitations",
    "## Preexisting work vs. sprint", "## AI-use statement", "## Citation",
    "See [the draft](paper/draft.md), [site](https://example.org) and [top](#tldr).",
    "![fig](results/fig%20one.png)",
])


def test_readme_good_passes(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "draft.md").write_text("x", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fig one.png").write_bytes(b"png")
    assert g.readme_problems(GOOD_README, tmp_path) == []


def test_readme_stale_strings_broken_links_and_headings(tmp_path: Path) -> None:
    text = GOOD_README.replace("## Citation", "## Cite") + "\nStatus: Not started. Also _None yet.\n" \
        "[gone](docs/missing.md)\n```\n[in code](nope.md)\n```\n"
    problems = g.readme_problems(text, tmp_path)
    assert "no heading: citation" in problems
    assert "stale string 'Not started'" in problems
    assert "stale string '_None yet'" in problems
    assert "broken link docs/missing.md" in problems
    assert "broken link paper/draft.md" in problems
    assert not any("nope.md" in p for p in problems)


def test_readme_requires_generated_block(tmp_path: Path) -> None:
    text = GOOD_README.replace("<!-- BEGIN GENERATED results -->", "")
    assert any("GENERATED" in p for p in g.readme_problems(text, tmp_path))


# ---------------------------------------------------------------------------------------------
# G13 key scan
# ---------------------------------------------------------------------------------------------


def test_scan_secrets_reports_file_and_pattern_not_secret(capsys: pytest.CaptureFixture[str]) -> None:
    fake = "gsk" + "_" + "Q" * 24  # built at runtime so this file holds no key-shaped literal
    fake_google = "AI" + "za" + "z" * 33
    hits = g.scan_secrets([("a.py", f"KEY = '{fake}'"), ("b.md", "nothing here"), ("c.txt", fake_google)])
    assert hits == [("a.py", "groq_key"), ("c.txt", "google_api_key")]
    out = capsys.readouterr()
    assert fake not in repr(hits) and fake not in out.out + out.err
    assert fake_google not in repr(hits)


def test_short_prefixes_are_not_keys() -> None:
    assert g.scan_secrets([("x", "sk-short hf_abc gsk_ ghp_1234")]) == []


# ---------------------------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------------------------


def test_hash_match_tolerates_line_endings_only() -> None:
    lf = b"a,b\n1,2\n"
    crlf = b"a,b\r\n1,2\r\n"
    assert g.hash_match(hashlib.sha256(lf).hexdigest(), lf) == "exact"
    assert g.hash_match(hashlib.sha256(crlf).hexdigest(), lf) == "eol"
    assert g.hash_match(hashlib.sha256(lf).hexdigest(), crlf) == "eol"
    assert g.hash_match(hashlib.sha256(b"other").hexdigest(), lf) == "none"


def test_robustness_detection(tmp_path: Path) -> None:
    rob = tmp_path / "robustness"
    assert g.robustness_missing(rob) == ["all-samples", "no-exclusions"]
    (rob / "all_samples").mkdir(parents=True)
    (rob / "all_samples" / "metrics.csv").write_text("x", encoding="utf-8")
    assert g.robustness_missing(rob) == ["no-exclusions"]
    (rob / "no-exclusions").mkdir()
    (rob / "no-exclusions" / "hypotheses.csv").write_text("x", encoding="utf-8")
    assert g.robustness_missing(rob) == []


def test_appendix_mentions_only_after_first_appendix() -> None:
    draft = "# T\n## Intro\nrobustness checks\n## Appendix A. X\nnothing\n"
    assert not g.appendix_mentions(draft, "robustness")
    assert g.appendix_mentions(draft + "## Appendix B. Robustness\n", "robustness")


def test_run_gate_catches_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(ctx: g.Ctx) -> g.GateResult:
        raise ValueError("kaput")

    monkeypatch.setitem(g.GATES, "GX", ("boom", boom))
    out = g.run_gate("GX", g.Ctx(root=tmp_path, python="python"))
    assert out.status == "FAIL" and "ValueError: kaput" in out.detail


def test_skip_tests_gate_does_not_run_pytest(tmp_path: Path) -> None:
    assert g.g1_tests(g.Ctx(root=tmp_path, python="python", skip_tests=True))[0] == "SKIP"
