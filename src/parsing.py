"""Parsing: stated probabilities, prose stripping for the judge, and the logit probe.

A failed parse returns ``None`` plus a reason; the caller turns it into NaN and logs it.
Nothing here substitutes a default value.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from src.prompts import CONTRACT_HEADER_KEYS, PROBABILITY_TAG

PROBABILITY_RE = re.compile(
    rf"{PROBABILITY_TAG}[*_`\s]*[:=][*_`\s<]*(?P<value>[0-9]*\.?[0-9]+)\s*(?P<pct>%)?",
    re.IGNORECASE,
)
_TAG_TO_EOL_RE = re.compile(rf"[*_`]*{PROBABILITY_TAG}.*$", re.IGNORECASE)
_HEADER_PREFIX_CHARS = " \t*_#>-•`0123456789.)"
_TOKEN_STRIP_CHARS = " \t\n.,!:;\"'*`Ġ▁"


@dataclass(frozen=True)
class ProbabilityParse:
    value: float | None
    error: str | None


def parse_probability(text: str | None) -> ProbabilityParse:
    """Read the LAST ``PROBABILITY_YES: x`` in the text. Accepts 0-1 or an explicit percent."""
    if not text or not text.strip():
        return ProbabilityParse(None, "empty reply")
    matches = list(PROBABILITY_RE.finditer(text))
    if not matches:
        if PROBABILITY_TAG in text.upper():
            return ProbabilityParse(None, "PROBABILITY_YES present but no readable number")
        return ProbabilityParse(None, "no PROBABILITY_YES line")
    match = matches[-1]
    value = float(match.group("value"))
    if match.group("pct"):
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        return ProbabilityParse(None, f"value {match.group(0).strip()!r} outside [0, 1]")
    return ProbabilityParse(value, None)


def _is_contract_header(line: str) -> bool:
    norm = line.strip().lstrip(_HEADER_PREFIX_CHARS).upper().replace(" ", "_")
    return any(norm.startswith(key) for key in CONTRACT_HEADER_KEYS)


def strip_prose(text: str, scrub_terms: Iterable[str] = ()) -> str:
    """Prose for the judge: drop contract-header lines and the PROBABILITY_YES line(s),
    and replace any model/family names with "[model]".

    Limitation: a contract header value that wraps onto a second line is not removed.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if _is_contract_header(line):
            continue
        cleaned = _TAG_TO_EOL_RE.sub("", line).rstrip()
        if cleaned != line.rstrip() and not cleaned.strip(" \t*_`-:."):
            continue  # the line held only the probability tag
        kept.append(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    for term in sorted({t for t in scrub_terms if t and len(t) >= 3}, key=len, reverse=True):
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", "[model]", out,
                     flags=re.IGNORECASE)
    return out


@dataclass(frozen=True)
class ProbeParse:
    p_yes: float | None
    mass: float | None  # total probability on Yes/No tokens among the returned top logprobs
    error: str | None


def probe_p_yes(top_logprobs: list[tuple[str, float]] | None) -> ProbeParse:
    """P(yes) = P(Yes-like tokens) / (P(Yes-like) + P(No-like)) at the first generated token."""
    if not top_logprobs:
        return ProbeParse(None, None, "no_logprobs")
    p_yes = p_no = 0.0
    for token, logprob in top_logprobs:
        norm = token.strip(_TOKEN_STRIP_CHARS).lower()
        if norm == "yes":
            p_yes += math.exp(logprob)
        elif norm == "no":
            p_no += math.exp(logprob)
    mass = p_yes + p_no
    if mass <= 0.0:
        return ProbeParse(None, 0.0, "no_yes_no")
    return ProbeParse(p_yes / mass, mass, None)
