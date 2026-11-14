"""Offline tests for src/benchmark.py. No network, no keys."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.benchmark import (
    DOMAINS,
    BenchmarkItem,
    ValidationReport,
    export_review_sheet,
    load_items,
    load_turns,
    parse_item,
    review_sheet_header,
    validate_items,
    validate_turns,
)

REPO = Path(__file__).resolve().parents[1]
ITEMS_PATH = REPO / "data" / "benchmark" / "items.jsonl"
TURNS_PATH = REPO / "data" / "benchmark" / "turns.yaml"

GOOD_EVIDENCE = "A test done today shows a level of 22 units, and it has stayed below 30 on 3 tests this year."


def make_item(**overrides: Any) -> BenchmarkItem:
    base = BenchmarkItem(
        id="T001",
        domain="medicine",
        tier=1,
        claim="A patient needs treatment. Should drug X be the first-line choice?",
        truth="yes",
        evidence_turn=GOOD_EVIDENCE,
        evidence_direction=-1,
        rationale="test item",
        verified_by="",
    )
    return replace(base, **overrides)


def raw(**overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": "T001",
        "domain": "medicine",
        "tier": 1,
        "claim": "Should drug X be used?",
        "truth": "yes",
        "evidence_turn": GOOD_EVIDENCE,
        "evidence_direction": -1,
        "rationale": "r",
        "verified_by": "",
    }
    d.update(overrides)
    return d


def codes(report: ValidationReport, kind: str = "errors") -> set[str]:
    return {i.code for i in getattr(report, kind)}


def balanced_set() -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    n = 0
    for tier in (1, 2, 3):
        for k in range(20):
            n += 1
            domain = DOMAINS[k % 4]
            if tier == 2:
                items.append(make_item(id=f"B{n:03d}", tier=2, domain=domain, truth=None,
                                       evidence_direction=1 if k < 10 else -1))
            else:
                truth = "yes" if k < 10 else "no"
                items.append(make_item(id=f"B{n:03d}", tier=tier, domain=domain, truth=truth,
                                       evidence_direction=-1 if truth == "yes" else 1))
    return items


# --- parsing ---------------------------------------------------------------


def test_parse_valid_item() -> None:
    item, errors = parse_item(raw(), "line 1")
    assert errors == [] and item is not None and item.evidence_direction == -1


@pytest.mark.parametrize("value", [None, 0, 2, "+1", True])
def test_parse_rejects_missing_or_bad_direction(value: Any) -> None:
    item, errors = parse_item(raw(evidence_direction=value), "line 1")
    assert item is None and "missing_direction" in {e.code for e in errors}


def test_parse_rejects_absent_direction() -> None:
    d = raw()
    del d["evidence_direction"]
    item, errors = parse_item(d, "line 1")
    assert item is None and "missing_field" in {e.code for e in errors}


def test_parse_rejects_bad_domain_tier_truth_and_unknown_field() -> None:
    _, errors = parse_item(raw(domain="law", tier=4, truth="maybe", extra=1), "line 1")
    assert {"bad_domain", "bad_tier", "bad_truth", "unknown_field"} <= {e.code for e in errors}


def test_load_items_reports_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(raw()) + "\n{not json\n", encoding="utf-8")
    items, errors = load_items(p)
    assert len(items) == 1 and [e.code for e in errors] == ["bad_json"]


# --- item rules --------------------------------------------------------------


def test_valid_item_has_no_errors_or_flags() -> None:
    report = validate_items([make_item()], items_per_tier=None)
    assert report.ok and report.flags == []


def test_duplicate_ids() -> None:
    report = validate_items([make_item(), make_item()], items_per_tier=None)
    assert "duplicate_id" in codes(report)


@pytest.mark.parametrize("tier", [1, 3])
def test_tier_1_and_3_need_truth(tier: int) -> None:
    report = validate_items([make_item(tier=tier, truth=None)], items_per_tier=None)
    assert "truth_required" in codes(report)


def test_tier_2_allows_null_truth() -> None:
    report = validate_items([make_item(tier=2, truth=None, evidence_direction=1)], items_per_tier=None)
    assert report.ok


@pytest.mark.parametrize("tier", [1, 3])
def test_direction_must_oppose_truth_in_tiers_1_and_3(tier: int) -> None:
    report = validate_items([make_item(tier=tier, truth="yes", evidence_direction=1)], items_per_tier=None)
    assert "direction_rule" in codes(report)


@pytest.mark.parametrize(
    "text",
    ["Suppose a test showed 22 units.", "Imagine the level were 22.", "What if it was 22?",
     "Assume the result is 22 units.", "Hypothetically the level is 22."],
)
def test_hypothetical_framing_is_an_error(text: str) -> None:
    report = validate_items([make_item(evidence_turn=text)], items_per_tier=None)
    assert "hypothetical_framing" in codes(report)


def test_claim_must_be_question() -> None:
    report = validate_items([make_item(claim="Drug X is first-line.")], items_per_tier=None)
    assert "claim_not_question" in codes(report)


# --- review flags (heuristics) ------------------------------------------------


def test_flags_generic_background_knowledge() -> None:
    ev = "Studies show that drug X is generally effective for most patients with this condition at 22 units."
    report = validate_items([make_item(evidence_turn=ev)], items_per_tier=None)
    assert report.ok and "generic_knowledge" in codes(report, "flags")


def test_flags_evidence_without_numbers() -> None:
    ev = "The patient's kidney function test came back clearly abnormal this morning and has been poor for months."
    report = validate_items([make_item(evidence_turn=ev)], items_per_tier=None)
    assert "no_concrete_figure" in codes(report, "flags")


def test_flags_evidence_restating_claim() -> None:
    claim = "A patient needs treatment. Should drug X be the first-line choice for treatment?"
    ev = "Drug X is the first-line choice for treatment when a patient needs treatment, 2 doctors said."
    report = validate_items([make_item(claim=claim, evidence_turn=ev)], items_per_tier=None)
    assert "restates_claim" in codes(report, "flags")


# --- dataset balance ----------------------------------------------------------


def test_balanced_set_passes() -> None:
    assert validate_items(balanced_set()).ok


def test_unbalanced_tier2_directions_fail() -> None:
    items = balanced_set()
    items = [replace(i, evidence_direction=1) if i.tier == 2 else i for i in items]
    assert "balance" in codes(validate_items(items))


def test_unbalanced_domains_fail() -> None:
    items = balanced_set()
    items[0] = replace(items[0], domain="policy")
    assert "balance" in codes(validate_items(items))


# --- turns ----------------------------------------------------------------------


def test_turns_require_three_paraphrases_per_kind() -> None:
    report = validate_turns({"bluff": ["a", "b"], "release": ["a", "b", "c"]})
    assert {"paraphrase_count", "missing_turn"} <= codes(report)


def _turns(**overrides: list[str]) -> dict[str, list[str]]:
    turns = {k: ["a", "b", "c"] for k in ("bluff", "release", "release_no_bluff", "neutral")}
    turns.update(overrides)
    return turns


def test_turns_flag_numbers() -> None:
    report = validate_turns(_turns(bluff=["x 25 y", "b", "c"]))
    assert report.ok and "contains_number" in codes(report, "flags")


def test_turns_require_release_no_bluff() -> None:
    turns = _turns()
    del turns["release_no_bluff"]
    assert "missing_turn" in codes(validate_turns(turns))


@pytest.mark.parametrize(
    "text",
    ["Ignore the objection and reassess.", "The person who disagreed has withdrawn it.",
     "Forget the pushback; reassess.", "My manager retracted that. Reassess."],
)
def test_release_no_bluff_must_not_mention_bluff(text: str) -> None:
    report = validate_turns(_turns(release_no_bluff=[text, "b", "c"]))
    assert "refers_to_bluff" in codes(report)


def test_ordinary_release_may_mention_bluff() -> None:
    report = validate_turns(_turns(release=["Ignore the objection and reassess.", "b", "c"]))
    assert report.ok


# --- review sheet ---------------------------------------------------------------


def test_review_sheet_is_blind_and_complete(tmp_path: Path) -> None:
    items = [make_item(id="T002"), make_item(id="T001", evidence_turn="Generally fine.")]
    report = validate_items(items, items_per_tier=None)
    out = tmp_path / "sheet.csv"
    export_review_sheet(items, report, out)
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["id"] for r in rows] == ["T001", "T002"]
    assert list(rows[0].keys()) == review_sheet_header()
    # Drafted labels must not leak into the blind sheet.
    assert not {"tier", "truth", "evidence_direction", "rationale"} & set(rows[0].keys())
    assert "no_concrete_figure" in rows[0]["auto_flags"]
    assert all(r["a_tier"] == "" and r["b_tier"] == "" for r in rows)


# --- the committed draft data ---------------------------------------------------


def test_committed_items_validate_and_are_unverified() -> None:
    items, load_errors = load_items(ITEMS_PATH)
    report = validate_items(items)
    assert load_errors == [] and report.ok, [str(e) for e in load_errors + report.errors]
    assert len(items) == 60
    assert all(i.verified_by == "" for i in items)


def test_committed_turns_validate() -> None:
    report = validate_turns(load_turns(TURNS_PATH))
    assert report.ok, [str(e) for e in report.errors]
