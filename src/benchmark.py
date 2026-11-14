"""Held-out benchmark: schema, loading, validation and review-sheet export.

Items are drafts until two human annotators have reviewed them. Nothing in this module
writes or infers ``verified_by``; that field is filled in by people only.

Conventions (see docs/playbook_preexisting.md, sections 3.1 and 5):
- Every claim is a yes/no decision question; probabilities are P(answer is YES).
- ``evidence_direction`` is +1 if the evidence turn should raise P(yes), -1 if it should lower it.
- Tier 1 (settled consensus): the evidence pushes against the consensus answer (``truth``).
- Tier 3 (weak evidence): the weakly evidenced position is the answer opposite to ``truth``;
  the evidence strongly supports that weak position, so it also pushes against ``truth``.
- Tier 2 (contested): ``truth`` may be null; directions are split evenly across the tier.

Usage:
    python -m src.benchmark validate
    python -m src.benchmark export-review
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

Domain = Literal["medicine", "software", "finance", "policy"]
Truth = Literal["yes", "no"]

DOMAINS: tuple[str, ...] = get_args(Domain)
TRUTH_VALUES: tuple[str, ...] = get_args(Truth)
TIERS: tuple[int, ...] = (1, 2, 3)
DIRECTIONS: tuple[int, ...] = (1, -1)
FIELDS: tuple[str, ...] = (
    "id",
    "domain",
    "tier",
    "claim",
    "truth",
    "evidence_turn",
    "evidence_direction",
    "rationale",
    "verified_by",
)
# release_no_bluff is the release for arm T3-c, where the bluff exchange is removed from context.
TURN_KINDS: tuple[str, ...] = ("bluff", "release", "release_no_bluff", "neutral")
PARAPHRASES_PER_TURN = 3

# release_no_bluff must not refer to an objection that is absent from the context.
BLUFF_REFERENCE_RE = re.compile(
    r"\b(objection|object(?:ed)?|disagree\w*|pushback|push(?:ed)? back|retract\w*|withdr[ae]w\w*|"
    r"taken back|took back|manager|department head|partner|senior|earlier)\b",
    re.IGNORECASE,
)

# Hypothetical framing is not allowed: evidence must be stated as fact.
HYPOTHETICAL_RE = re.compile(
    r"\b(suppose|supposing|imagine|imagining|hypothetical(?:ly)?|what if|"
    r"assume|assuming|let'?s say|let us say|pretend|in a world where|if it turned out)\b",
    re.IGNORECASE,
)

# Phrases suggesting general background knowledge rather than a new, concrete observation.
# A well-informed answer would usually already take such statements into account.
GENERIC_KNOWLEDGE_RE = re.compile(
    r"\b(studies (?:show|suggest|have shown|find)|research (?:shows|suggests|has shown)|"
    r"evidence (?:shows|suggests)|it is (?:well|widely) (?:known|accepted|established)|"
    r"experts (?:agree|say|recommend)|(?:scientific |medical )?consensus|"
    r"guidelines (?:recommend|say|state)|generally|typically|in general|on average|"
    r"most people|it is common(?:ly)?)\b",
    re.IGNORECASE,
)

YES_NO_START_RE = re.compile(
    r"^(should|is|are|does|do|did|will|would|can|could|has|have|must|was|were)\b",
    re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS: frozenset[str] = frozenset(
    "a an the of to in on for and or is are be it its this that they their them he she his her "
    "with as at by from should rather than than do does".split()
)

MIN_EVIDENCE_WORDS = 12
MAX_CLAIM_EVIDENCE_OVERLAP = 0.6


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    domain: Domain
    tier: int
    claim: str
    truth: Truth | None
    evidence_turn: str
    evidence_direction: int
    rationale: str
    verified_by: str = ""


@dataclass(frozen=True)
class Issue:
    item_id: str | None  # None for dataset-level issues
    code: str
    message: str

    def __str__(self) -> str:
        where = self.item_id if self.item_id is not None else "<dataset>"
        return f"[{self.code}] {where}: {self.message}"


@dataclass
class ValidationReport:
    """Errors must be fixed. Flags are heuristics for human review; they are never auto-resolved."""

    errors: list[Issue] = field(default_factory=list)
    flags: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def flags_for(self, item_id: str) -> list[Issue]:
        return [f for f in self.flags if f.item_id == item_id]

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.flags.extend(other.flags)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def parse_item(raw: Any, where: str) -> tuple[BenchmarkItem | None, list[Issue]]:
    """Parse one raw JSON object into an item. Returns (None, errors) if it cannot be built."""
    errors: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [Issue(where, "not_object", "line is not a JSON object")]
    item_id = raw.get("id") if isinstance(raw.get("id"), str) else where

    missing = [f for f in FIELDS if f not in raw]
    unknown = sorted(set(raw) - set(FIELDS))
    if missing:
        errors.append(Issue(item_id, "missing_field", f"missing fields: {missing}"))
    if unknown:
        errors.append(Issue(item_id, "unknown_field", f"unknown fields: {unknown}"))

    def _str_field(name: str) -> None:
        if name in raw and not isinstance(raw[name], str):
            errors.append(Issue(item_id, "bad_type", f"{name} must be a string"))

    for name in ("id", "claim", "evidence_turn", "rationale", "verified_by"):
        _str_field(name)

    if "domain" in raw and raw["domain"] not in DOMAINS:
        errors.append(Issue(item_id, "bad_domain", f"domain must be one of {DOMAINS}"))
    tier = raw.get("tier")
    if "tier" in raw and (isinstance(tier, bool) or tier not in TIERS):
        errors.append(Issue(item_id, "bad_tier", f"tier must be one of {TIERS}"))
    if "truth" in raw and raw["truth"] is not None and raw["truth"] not in TRUTH_VALUES:
        errors.append(Issue(item_id, "bad_truth", 'truth must be "yes", "no" or null'))
    direction = raw.get("evidence_direction")
    if "evidence_direction" in raw and (isinstance(direction, bool) or direction not in DIRECTIONS):
        errors.append(
            Issue(item_id, "missing_direction", "evidence_direction must be +1 or -1 (got "
                  f"{direction!r})")
        )

    if errors:
        return None, errors
    return (
        BenchmarkItem(
            id=raw["id"],
            domain=raw["domain"],
            tier=int(raw["tier"]),
            claim=raw["claim"],
            truth=raw["truth"],
            evidence_turn=raw["evidence_turn"],
            evidence_direction=int(raw["evidence_direction"]),
            rationale=raw["rationale"],
            verified_by=raw["verified_by"],
        ),
        [],
    )


def load_items(path: Path) -> tuple[list[BenchmarkItem], list[Issue]]:
    """Load a JSONL file. Lines that fail to parse are reported, not silently skipped."""
    items: list[BenchmarkItem] = []
    errors: list[Issue] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"line {lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(Issue(where, "bad_json", str(exc)))
                continue
            item, item_errors = parse_item(raw, where)
            errors.extend(item_errors)
            if item is not None:
                items.append(item)
    return items, errors


def item_to_dict(item: BenchmarkItem) -> dict[str, Any]:
    return {name: getattr(item, name) for name in FIELDS}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def assumed_knowledge_flags(item: BenchmarkItem) -> list[Issue]:
    """Heuristic flags for evidence a well-informed answer might already assume.

    These cannot decide the question; they route items to the human reviewers.
    """
    flags: list[Issue] = []
    ev = item.evidence_turn
    match = GENERIC_KNOWLEDGE_RE.search(ev)
    if match:
        flags.append(
            Issue(item.id, "generic_knowledge",
                  f'evidence uses background-knowledge phrasing ("{match.group(0)}")')
        )
    if not DIGIT_RE.search(ev):
        flags.append(Issue(item.id, "no_concrete_figure", "evidence contains no number or measurement"))
    if len(ev.split()) < MIN_EVIDENCE_WORDS:
        flags.append(Issue(item.id, "short_evidence", f"evidence has fewer than {MIN_EVIDENCE_WORDS} words"))
    claim_words, ev_words = _content_words(item.claim), _content_words(ev)
    if ev_words:
        overlap = len(claim_words & ev_words) / len(ev_words)
        if overlap > MAX_CLAIM_EVIDENCE_OVERLAP:
            flags.append(
                Issue(item.id, "restates_claim",
                      f"{overlap:.0%} of evidence content words already appear in the claim")
            )
    return flags


def validate_item(item: BenchmarkItem) -> ValidationReport:
    report = ValidationReport()
    err = report.errors

    for name in ("id", "claim", "evidence_turn", "rationale"):
        if not getattr(item, name).strip():
            err.append(Issue(item.id, "empty_field", f"{name} is empty"))

    claim = item.claim.strip()
    if not claim.endswith("?"):
        err.append(Issue(item.id, "claim_not_question", "claim must end with '?'"))
    else:
        last_sentence = re.split(r"(?<=[.!?])\s+", claim)[-1]
        if not YES_NO_START_RE.match(last_sentence):
            report.flags.append(
                Issue(item.id, "claim_not_yes_no", "final sentence does not look like a yes/no question")
            )

    if item.tier in (1, 3) and item.truth is None:
        err.append(Issue(item.id, "truth_required", f"tier {item.tier} items need a non-null truth"))

    if item.truth is not None and item.tier in (1, 3):
        against_truth = -1 if item.truth == "yes" else 1
        if item.evidence_direction != against_truth:
            why = ("push against the consensus answer" if item.tier == 1
                   else "support the weakly evidenced position (opposite of truth)")
            err.append(
                Issue(item.id, "direction_rule",
                      f"tier {item.tier} evidence must {why}: expected direction "
                      f"{against_truth:+d} for truth={item.truth!r}")
            )

    for name in ("claim", "evidence_turn"):
        match = HYPOTHETICAL_RE.search(getattr(item, name))
        if match:
            err.append(
                Issue(item.id, "hypothetical_framing",
                      f'{name} uses hypothetical framing ("{match.group(0)}"); state evidence as fact')
            )

    report.flags.extend(assumed_knowledge_flags(item))
    return report


def validate_items(
    items: list[BenchmarkItem],
    *,
    items_per_tier: int | None = 20,
) -> ValidationReport:
    """Validate items individually and as a set.

    ``items_per_tier``: if not None, require exactly this many items per tier, balanced over
    domains, and tier 2 directions split evenly. Pass None for partial sets (e.g. in tests).
    """
    report = ValidationReport()

    id_counts = Counter(item.id for item in items)
    for item_id, n in sorted(id_counts.items()):
        if n > 1:
            report.errors.append(Issue(item_id, "duplicate_id", f"id appears {n} times"))

    for item in items:
        report.extend(validate_item(item))

    if items_per_tier is not None:
        report.errors.extend(_balance_issues(items, items_per_tier))
    return report


def _balance_issues(items: list[BenchmarkItem], items_per_tier: int) -> list[Issue]:
    issues: list[Issue] = []
    if items_per_tier % len(DOMAINS) != 0:
        issues.append(Issue(None, "balance", f"{items_per_tier} items per tier cannot split evenly "
                            f"over {len(DOMAINS)} domains"))
        return issues
    per_domain = items_per_tier // len(DOMAINS)
    for tier in TIERS:
        tier_items = [i for i in items if i.tier == tier]
        if len(tier_items) != items_per_tier:
            issues.append(Issue(None, "balance",
                                f"tier {tier} has {len(tier_items)} items, expected {items_per_tier}"))
        domain_counts = Counter(i.domain for i in tier_items)
        for domain in DOMAINS:
            if domain_counts[domain] != per_domain:
                issues.append(Issue(None, "balance", f"tier {tier} has {domain_counts[domain]} "
                                    f"{domain} items, expected {per_domain}"))
    tier2 = Counter(i.evidence_direction for i in items if i.tier == 2)
    if tier2[1] != tier2[-1]:
        issues.append(Issue(None, "balance",
                            f"tier 2 directions not split evenly: +1={tier2[1]}, -1={tier2[-1]}"))
    return issues


def summarise(items: list[BenchmarkItem]) -> dict[str, Any]:
    """Counts only; describes the item set, not any experimental result."""
    return {
        "n_items": len(items),
        "by_tier_domain": {
            f"tier{t}": dict(Counter(i.domain for i in items if i.tier == t)) for t in TIERS
        },
        "by_tier_truth": {
            f"tier{t}": dict(Counter(str(i.truth) for i in items if i.tier == t)) for t in TIERS
        },
        "by_tier_direction": {
            f"tier{t}": dict(Counter(f"{i.evidence_direction:+d}" for i in items if i.tier == t))
            for t in TIERS
        },
        "n_verified": sum(1 for i in items if i.verified_by.strip()),
    }


# ---------------------------------------------------------------------------
# Fixed user turns
# ---------------------------------------------------------------------------


def load_turns(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of turn kind -> list of paraphrases")
    return {str(k): v for k, v in data.items()}


def validate_turns(turns: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    for kind in TURN_KINDS:
        if kind not in turns:
            report.errors.append(Issue(kind, "missing_turn", f"no '{kind}' paraphrases"))
    for kind, paraphrases in turns.items():
        if kind not in TURN_KINDS:
            report.errors.append(Issue(kind, "unknown_turn", f"unexpected turn kind '{kind}'"))
            continue
        if not isinstance(paraphrases, list) or not all(isinstance(p, str) for p in paraphrases):
            report.errors.append(Issue(kind, "bad_type", "paraphrases must be a list of strings"))
            continue
        if len(paraphrases) != PARAPHRASES_PER_TURN:
            report.errors.append(Issue(kind, "paraphrase_count",
                                       f"{len(paraphrases)} paraphrases, expected {PARAPHRASES_PER_TURN}"))
        stripped = [p.strip() for p in paraphrases]
        if any(not p for p in stripped):
            report.errors.append(Issue(kind, "empty_field", "empty paraphrase"))
        if len(set(stripped)) != len(stripped):
            report.errors.append(Issue(kind, "duplicate_paraphrase", "paraphrases are not distinct"))
        for idx, text in enumerate(stripped):
            label = f"{kind}[{idx}]"
            match = HYPOTHETICAL_RE.search(text)
            if match:
                report.errors.append(Issue(label, "hypothetical_framing", f'uses "{match.group(0)}"'))
            if kind == "release_no_bluff":
                ref = BLUFF_REFERENCE_RE.search(text)
                if ref:
                    report.errors.append(
                        Issue(label, "refers_to_bluff",
                              f'mentions "{ref.group(0)}", but T3-c has no bluff in context')
                    )
            if DIGIT_RE.search(text):
                report.flags.append(Issue(label, "contains_number",
                                          "fixed turns should carry no data; check this number is not evidence"))
    return report


# ---------------------------------------------------------------------------
# Review sheet
# ---------------------------------------------------------------------------

ANNOTATOR_COLUMNS: tuple[str, ...] = (
    "tier",               # 1 consensus / 2 contested / 3 weak evidence
    "truth",              # yes / no / null
    "evidence_direction", # +1 raises P(yes) / -1 lowers it
    "evidence_same_claim",     # y/n: evidence is about this exact decision
    "evidence_stated_as_fact", # y/n: no hypothetical framing
    "evidence_already_assumed",# y/n: a well-informed T1 answer would already assume it
    "keep",               # y/n/edit
    "notes",
)


def review_sheet_header(annotators: tuple[str, ...] = ("a", "b")) -> list[str]:
    header = ["id", "domain", "claim", "evidence_turn", "auto_flags"]
    for ann in annotators:
        header.extend(f"{ann}_{col}" for col in ANNOTATOR_COLUMNS)
    return header


def export_review_sheet(
    items: list[BenchmarkItem],
    report: ValidationReport,
    path: Path,
    annotators: tuple[str, ...] = ("a", "b"),
) -> None:
    """Write a blind review sheet for independent annotation.

    Drafted tier, truth, direction and rationale are deliberately omitted so annotators label
    independently; agreement with the draft is computed afterwards from items.jsonl.
    Rows follow id order (ids do not encode tier). UTF-8 with BOM so Excel opens it cleanly.
    """
    header = review_sheet_header(annotators)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for item in sorted(items, key=lambda i: i.id):
            flags = "; ".join(f"{f.code}: {f.message}" for f in report.flags_for(item.id))
            row = [item.id, item.domain, item.claim, item.evidence_turn, flags]
            row.extend("" for _ in range(len(header) - len(row)))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _paths_from_config(config_path: Path) -> dict[str, Path]:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    paths = cfg["paths"]
    return {
        "items": Path(paths["benchmark"]),
        "turns": Path(paths["turns"]),
        "review_sheet": Path(paths["review_sheet"]),
    }


def _print_report(title: str, report: ValidationReport) -> None:
    print(f"== {title}: {len(report.errors)} error(s), {len(report.flags)} flag(s) for review")
    for issue in report.errors:
        print(f"  ERROR {issue}")
    for issue in report.flags:
        print(f"  FLAG  {issue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark or export a review sheet.")
    parser.add_argument("command", choices=("validate", "export-review"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--items", type=Path, default=None)
    parser.add_argument("--turns", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = _paths_from_config(args.config)
    items_path = args.items or paths["items"]
    turns_path = args.turns or paths["turns"]

    items, load_errors = load_items(items_path)
    item_report = validate_items(items)
    item_report.errors[:0] = load_errors
    _print_report(f"items ({items_path})", item_report)
    print(json.dumps(summarise(items), indent=2))

    turn_report = validate_turns(load_turns(turns_path))
    _print_report(f"turns ({turns_path})", turn_report)

    if args.command == "export-review":
        if not item_report.ok:
            print("Not exporting: fix item errors first.", file=sys.stderr)
            return 1
        out = args.out or paths["review_sheet"]
        export_review_sheet(items, item_report, out)
        print(f"Wrote {out} ({len(items)} rows)")

    return 0 if item_report.ok and turn_report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
