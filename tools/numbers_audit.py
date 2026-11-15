"""Audit every number in the built paper (paper.md, or a PDF exported from the template).

Each number must match a value in paper/numbers_used.csv (written by tools/build_paper.py) or a
whitelist built from config.yaml, the benchmark files and the prompt text. Numbers inside
identifiers (H1, T2A, c3N, qwen2.5-7b, Phi-3.5-mini, gpt-oss-120b, EH001), URLs, link targets,
[CITE:...] keys and dates are ignored, as are section/table/figure/tier/turn numbers, list and
enumeration markers, and years (1900-2099). Exits 1 and lists every offending number with context.

Usage (from the repo root):
    python tools/numbers_audit.py paper/paper.md
    python tools/numbers_audit.py submission.pdf --numbers-used paper/numbers_used.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_paper import config_lookup, data_facts, load_config  # noqa: E402

# Config values that are design constants (not results). Each may appear in the paper.
WHITELIST_CONFIG_PATHS: tuple[str, ...] = (
    "sampling.temperature", "sampling.max_tokens", "sampling.samples_per_cell", "sampling.seed",
    "sampling.probe_top_logprobs",
    "judge.temperature", "judge.max_tokens",
    
    "analysis.bootstrap_resamples", "analysis.bootstrap_seed", "analysis.ci_level",
    "analysis.recovery_min_gap", "analysis.log_score_clip", "analysis.min_models_cross_model",
    "analysis.human_rating_n", "analysis.sampling_seed",
    "retries.reask_on_parse_failure", "subject_defaults.max_model_len",
)
# Constants of the metric definitions (src/analyze.py), not results.
FORMULA_CONSTANTS: dict[str, str] = {
    "0.5": "extremity gap reference point |p - 0.5| (src/analyze.py)",
    "1": "evidence direction +1/-1 and truth label y in {0, 1}",
    "-1": "evidence direction -1",
    "0": "truth label y = 0",
    "2": "squared error exponent in the Brier score (c - y)^2",
}

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
CONTEXT_WORDS = (r"(?:Sections?|Sec\.|§|Tables?|Figures?|Figs?\.|Appendix|Appendices|Track|Turns?|Tiers?|"
                 r"Hypothes[ie]s|Steps?|Paraphrases?|Eq\.|Equations?|Pages?|Rows?|Columns?)")
NUM_LIST = r"\d+(?:\.\d+)*(?:\s*(?:,|-|–|to|and|or)\s*\d{1,2}(?![\d.]))*"
MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+|www\.\S+"),
    re.compile(r"\]\([^)]*\)"),                                         # Markdown link/image targets
    re.compile(r"\[CITE:[^\]]*\]"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                # ISO dates
    re.compile(rf"\b\d{{1,2}}(?:\s*(?:-|–|to)\s*\d{{1,2}})?\s+{MONTHS}(?:\s+\d{{4}})?", re.IGNORECASE),
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?", re.IGNORECASE),
    re.compile(rf"\b{CONTEXT_WORDS}\s+(?:[A-Z]\.?)?{NUM_LIST}", re.IGNORECASE),
    re.compile(r"\(\d{1,2}\)"),                                          # enumerators (1), (2)
    re.compile(r"^\s*\d{1,2}[.)](?=\s)", re.MULTILINE),                  # numbered-list markers
    re.compile(r"^#{1,6}\s*[A-Z]?[\d.]*", re.MULTILINE),                 # heading numbers
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.\-/]*[A-Za-z0-9_])?")
NUMBER_RE = re.compile(r"(?<![\w.])(?:[-+−](?=\d))?\d+(?:,\d{3})*(?:\.\d+)?%?")


@dataclass(frozen=True)
class Hit:
    token: str
    value: Decimal
    line: int
    context: str


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def _is_identifier(token: str) -> bool:
    """Letter before the first digit (H1, c2A, qwen2.5-7b, EH001), or digits+letters (7B, 1st)."""
    if not re.search(r"\d", token) or not re.search(r"[A-Za-z]", token):
        return False
    first_letter = re.search(r"[A-Za-z]", token)
    first_digit = re.search(r"\d", token)
    assert first_letter and first_digit
    return first_letter.start() < first_digit.start() or bool(re.fullmatch(r"\d+[A-Za-z]+", token))


def mask(text: str) -> str:
    """Replace everything that is not a candidate number with spaces (positions preserved)."""
    for pattern in MASKS:
        for m in list(pattern.finditer(text)):
            text = _blank(text, m.start(), m.end())
    for m in list(TOKEN_RE.finditer(text)):
        if _is_identifier(m.group(0)):
            text = _blank(text, m.start(), m.end())
    return text


def normalise(token: str) -> Decimal:
    s = token.replace("−", "-").replace(",", "").rstrip("%").lstrip("+")
    return Decimal(s)


def extract_numbers(text: str) -> list[Hit]:
    masked = mask(text)
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    hits: list[Hit] = []
    for m in NUMBER_RE.finditer(masked):
        try:
            value = normalise(m.group(0))
        except InvalidOperation:
            continue
        line = sum(1 for s in starts if s <= m.start())
        ctx = text[max(0, m.start() - 50): m.end() + 50].replace("\n", " ")
        hits.append(Hit(m.group(0), value, line, ctx))
    return hits


def _numbers_in(value: Any) -> list[Decimal]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [Decimal(repr(value)) if isinstance(value, float) else Decimal(value)]
    return [h.value for h in extract_numbers(str(value))]


def build_whitelist(config_path: Path, numbers_used: Path | None) -> dict[Decimal, str]:
    """Allowed values -> where they come from."""
    from src import prompts

    allowed: dict[Decimal, str] = {}

    def add(values: Iterable[Decimal], source: str) -> None:
        for v in values:
            allowed.setdefault(v, source)

    cfg = load_config(config_path)
    for path in WHITELIST_CONFIG_PATHS:
        try:
            value = config_lookup(cfg, path)
        except Exception:  # noqa: BLE001 - optional keys may be absent
            continue
        add(_numbers_in(value), f"config:{path}")
    for subject in cfg.get("subjects", []):
        add(_numbers_in(subject.get("tensor_parallel")), "config:subjects[].tensor_parallel")
    ci = cfg.get("analysis", {}).get("ci_level")
    if isinstance(ci, (int, float)):
        add([Decimal(repr(float(ci))) * 100], "config:analysis.ci_level (percent)")
    clip = cfg.get("analysis", {}).get("log_score_clip")
    if isinstance(clip, (int, float)):
        add([Decimal(1) - Decimal(repr(float(clip)))], "config:analysis.log_score_clip (upper)")
    for name, value in data_facts(config_path).items():
        add([Decimal(value)], f"data:{name}")
    turns_path = Path(cfg["paths"]["turns"])
    turns_path = turns_path if turns_path.is_absolute() else config_path.resolve().parent / turns_path
    prompt_text = [*prompts.all_prompt_text(), turns_path.read_text(encoding="utf-8")]
    for text in prompt_text:
        add((h.value for h in extract_numbers(text)), "prompt text (src/prompts.py, turns.yaml)")
    for token, why in FORMULA_CONSTANTS.items():
        add([Decimal(token)], why)
    if numbers_used is not None:
        with numbers_used.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                add(_numbers_in(row["value"]), f"numbers_used:{row['placeholder']}")
    return allowed


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def is_year(value: Decimal) -> bool:
    return value == value.to_integral_value() and 1900 <= value <= 2099


def audit(text: str, allowed: dict[Decimal, str]) -> tuple[list[Hit], int]:
    """Returns (offending numbers, numbers checked)."""
    hits = extract_numbers(text)
    bad = [h for h in hits if h.value not in allowed and not (is_year(h.value) and "." not in h.token)]
    return bad, len(hits)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("document", type=Path, help="paper.md or a PDF")
    ap.add_argument("--numbers-used", type=Path, default=REPO_ROOT / "paper" / "numbers_used.csv")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    args = ap.parse_args(argv)
    if not args.numbers_used.is_file():
        print(f"AUDIT FAILED: {args.numbers_used} not found (run tools/build_paper.py first)", file=sys.stderr)
        return 1
    allowed = build_whitelist(args.config, args.numbers_used)
    bad, n = audit(read_document(args.document), allowed)
    if bad:
        print(f"AUDIT FAILED: {len(bad)} of {n} numbers in {args.document} are not traceable to "
              f"numbers_used.csv or the config/data whitelist:", file=sys.stderr)
        for h in bad:
            print(f"  line {h.line}: {h.token!r} in ...{h.context}...", file=sys.stderr)
        return 1
    print(f"audit passed: {n} numbers in {args.document}, all traceable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
