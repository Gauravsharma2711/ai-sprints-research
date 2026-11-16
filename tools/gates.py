"""Offline submission gates: one PASS/FAIL/BLOCKED line per check of the repo's readiness.

Usage:  python tools/gates.py [--skip-tests] [--json] [--only G1,G5]

Statuses: PASS; FAIL (something the agent/code must fix); BLOCKED (waiting on a human input the agent
cannot produce: ratings, labels, confirmed citations, author names, PDF export); SKIP (not run).
Exit code is 1 if any gate FAILs, else 0 (BLOCKED and SKIP do not fail).

Every gate is a separate function returning ``(status, detail)``; the pure checks underneath take text
or paths so they can be unit-tested (tests/test_gates.py). No gate touches the network. The secret scan
reports file and pattern name only, never the matched text.

Interpretations worth knowing:
- G9 "main text" = from the Introduction heading to just before the References heading, HTML comments
  stripped; markdown table rows, code fences and lines that are only a {{table:..}}/{{pivot:..}}/
  {{fig:..}} placeholder are excluded from the word count. Every {{...}} placeholder and [...] editorial
  marker counts as one word.
- G11 hypothesis check (heuristic): a sentence that names H1..H6 and a verdict word ("not supported",
  "supported", "inconclusive", "opposite") is flagged when that verdict is not among the pooled
  ALL_MODELS verdicts for that hypothesis in results/tables/hypotheses.csv (a hypothesis may have several
  pooled rows, e.g. per tier; any match is accepted). Sentences inside placeholders are ignored.
- G5 human-owned markers: [AUTHOR NAMES], [AFFILIATIONS] and [CITE:k1,k2] where at least one key is not
  marked CONFIRMED in paper/references.md (a missing entry is also "not confirmed"; G10 reports it).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

Status = Literal["PASS", "FAIL", "BLOCKED", "SKIP"]
GateResult = tuple[Status, str]

EXPECTED_PROTOCOL_HASH = "677f941a762b"
ABSTRACT_MIN_WORDS = 150
ABSTRACT_MAX_WORDS = 250
MAX_MAIN_WORDS = 3800
MAX_MAIN_TABLES_FIGS = 5
SUBPROCESS_TIMEOUT_S = 600

PLACEHOLDER_RE = re.compile(r"\{\{[^{}\n]*\}\}")
MARKER_RE = re.compile(r"\[(?:CITE:[^\]]*|RESULT PROSE[^\]]*|ABSTRACT[^\]]*|AUTHOR NAMES|AFFILIATIONS|"
                       r"REFERENCES[^\]]*|DISCUSSION[^\]]*|CHECK[^\]]*)\]")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
CITE_RE = re.compile(r"\[CITE:([^\]]*)\]")
REF_ENTRY_RE = re.compile(r"^\s*[-*]\s*\[([^\]]+)\]\s*(.*)$")
CONFIRMED_RE = re.compile(r"(?<![A-Za-z])CONFIRMED\b")
TABLE_FIG_PLACEHOLDER_RE = re.compile(r"\{\{(?:table|pivot|fig):[^{}\n]*\}\}")

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    "groq_key": re.compile(r"gsk_[0-9A-Za-z]{20,}"),
    "huggingface_token": re.compile(r"hf_[0-9A-Za-z]{20,}"),
    "xai_key": re.compile(r"xai-[0-9A-Za-z]{20,}"),
    "openai_style_key": re.compile(r"sk-[0-9A-Za-z]{20,}"),
    "github_fine_grained_pat": re.compile(r"github_pat_[0-9A-Za-z_]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}"),
}

LINT_PATTERNS: dict[str, re.Pattern[str]] = {
    "causal scrubbing": re.compile(r"causal\s+scrubbing", re.IGNORECASE),
    "proves": re.compile(r"\bproves\b", re.IGNORECASE),
    "proven": re.compile(r"\bproven\b", re.IGNORECASE),
    "frontier llm": re.compile(r"\bfrontier\s+llm", re.IGNORECASE),
    "first to": re.compile(r"\bfirst\s+to\b", re.IGNORECASE),
    "we are the first": re.compile(r"\bwe\s+are\s+the\s+first\b", re.IGNORECASE),
    "novel": re.compile(r"\bnovel\b", re.IGNORECASE),
}
LINT_QUALIFIERS: dict[str, tuple[str, ...]] = {"novel": ("to our knowledge",)}

README_REQUIRED: tuple[tuple[str, ...], ...] = (
    ("tl;dr",), ("research question", "contributions"), ("method",), ("results",), ("models",),
    ("data",), ("reproduc",), ("repository structure",), ("limitations",), ("preexisting work",),
    ("ai-use",), ("citation",),
)
README_STALE = ("Not started", "dry-run only", "to be written", "_None yet", "Not yet run on Kaggle")
GITIGNORE_REQUIRED = (".env", ".venv/", "results/dry_run/", ".claude/", "__pycache__/", ".pytest_cache/")
UNTRACKED_REQUIRED = ("docs/agent_prompt_final.md", "docs/completion_plan_and_agent_prompt.md", "MASTER_PROMPT.md")
NOTEBOOK_PATH = "notebooks/kaggle_run.ipynb"

DRAFT_HEADINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Introduction", ("introduction",)),
    ("Related Work", ("related work",)),
    ("Methodology", ("methodology",)),
    ("Results", ("results",)),
    ("Discussion", ("discussion",)),
    ("References", ("references",)),
    ("Preexisting Work", ("preexisting work",)),
    ("AI-Use Statement", ("ai-use statement",)),
)
LIMITATIONS_HEADING = "Limitations and Dual-Use / Ethical Considerations"
VALIDATION_TABLES: tuple[tuple[str, str], ...] = (
    ("human_judge_agreement.csv", "human ratings of the judge-validation sheet"),
    ("annotation_agreement.csv", "author item/tier labels"),
)
ANALYSIS_TABLES = ("hypotheses.csv", "coverage.csv", "data_quality.csv", "metrics.csv", "condition_contrasts.csv")


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    """Remove HTML comments, keeping their newlines so line numbers stay valid."""
    return COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def count_words(text: str) -> int:
    """Words in prose; each {{placeholder}} and [EDITORIAL MARKER] counts as one word."""
    text = PLACEHOLDER_RE.sub(" PLACEHOLDER ", text)
    text = MARKER_RE.sub(" MARKER ", text)
    return sum(1 for tok in text.split() if re.search(r"[0-9A-Za-z]", tok))


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    line: int  # 0-based line index


def parse_headings(text: str) -> list[Heading]:
    """ATX headings outside code fences and HTML comments."""
    out: list[Heading] = []
    in_fence = False
    for i, line in enumerate(strip_comments(text).split("\n")):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            out.append(Heading(len(m.group(1)), m.group(2).strip(), i))
    return out


def _norm_title(title: str) -> str:
    """Lower-case; drop a leading 'Appendix X.' / numbering so appendix headings match by name."""
    t = title.strip().lower()
    t = re.sub(r"^appendix\s+[a-z0-9]+[.:]?\s*", "", t)
    t = re.sub(r"^\d+(\.\d+)*\.?\s+", "", t)
    return t


def _find_heading(heads: Sequence[Heading], *needles: str) -> Heading | None:
    for h in heads:
        t = _norm_title(h.title)
        if any(n in t for n in needles):
            return h
    return None


def section_text(text: str, heading: Heading) -> str:
    """Lines after ``heading`` up to the next heading of the same or higher level."""
    lines = strip_comments(text).split("\n")
    end = len(lines)
    for h in parse_headings(text):
        if h.line > heading.line and h.level <= heading.level:
            end = h.line
            break
    return "\n".join(lines[heading.line + 1:end])


# ---------------------------------------------------------------------------------------------
# G7 abstract and title
# ---------------------------------------------------------------------------------------------


def extract_abstract(text: str) -> str | None:
    """Abstract text: under an 'Abstract' heading, or a paragraph starting with **Abstract**."""
    clean = strip_comments(text)
    heads = parse_headings(clean)
    lines = clean.split("\n")
    for h in heads:
        if _norm_title(h.title).rstrip(".:") == "abstract":
            return section_text(clean, h).strip()
    intro = _find_heading(heads, "introduction")
    limit = intro.line if intro else len(lines)
    for i in range(limit):
        m = re.match(r"^\s*(\*\*|__)?abstract[.:]?(\*\*|__)?[.:]?\s*(.*)$", lines[i], re.IGNORECASE)
        if m and (m.group(1) or lines[i].strip().lower().startswith("abstract")):
            body = [m.group(3)]
            for j in range(i + 1, limit):
                if HEADING_RE.match(lines[j]):
                    break
                body.append(lines[j])
            return "\n".join(body).strip()
    return None


def check_abstract(text: str) -> GateResult:
    abstract = extract_abstract(text)
    if abstract is None or not abstract.strip():
        return "FAIL", "no abstract found (Abstract heading or **Abstract.** paragraph before Introduction)"
    if re.search(r"\[ABSTRACT[^\]]*\]", abstract):
        return "FAIL", "abstract not written (still the [ABSTRACT ...] marker)"
    n = count_words(abstract)
    problems: list[str] = []
    if not ABSTRACT_MIN_WORDS <= n <= ABSTRACT_MAX_WORDS:
        problems.append(f"abstract has {n} words (need {ABSTRACT_MIN_WORDS}-{ABSTRACT_MAX_WORDS})")
    heads = parse_headings(text)
    if not any(h.level == 1 and h.title for h in heads):
        problems.append("no title line ('# Title')")
    if problems:
        return "FAIL", "; ".join(problems)
    human = [m for m in ("[AUTHOR NAMES]", "[AFFILIATIONS]") if m in text]
    if human:
        return "BLOCKED", f"abstract {n} words, title ok; user fills {', '.join(human)}"
    return "PASS", f"abstract {n} words, title present"


# ---------------------------------------------------------------------------------------------
# G8 structure
# ---------------------------------------------------------------------------------------------


def check_structure(text: str, judge_model: str | None) -> GateResult:
    heads = parse_headings(text)
    missing: list[str] = []
    for label, needles in DRAFT_HEADINGS:
        if _find_heading(heads, *needles) is None:
            missing.append(f"heading '{label}'")
    if not any(_norm_title(h.title) == LIMITATIONS_HEADING.lower() for h in heads):
        missing.append(f"heading exactly '{LIMITATIONS_HEADING}'")
    ai = _find_heading(heads, "ai-use statement")
    if ai is not None:
        body = section_text(text, ai)
        if not re.search(r"\b(claude|anthropic)\b", body, re.IGNORECASE):
            missing.append("AI-use statement: mention of Claude / Claude Code / Anthropic")
        if not judge_model:
            missing.append("judge.model not set in config.yaml")
        elif judge_model not in body and "{{cfg:judge.model" not in body:
            missing.append(f"AI-use statement: judge model id '{judge_model}'")
        # The statement must say what the author did (their role), without forcing a claim of a
        # review that may not have happened: "directed", "responsible", "ran", "reviewed", ...
        role = r"\bauthors?\b[^.]*\b(direct\w*|responsib\w*|ran|run|design\w*|review\w*|rated|confirm\w*)"
        if not re.search(role, body, re.IGNORECASE):
            missing.append("AI-use statement: what the author did (their role)")
    if missing:
        return "FAIL", "missing: " + "; ".join(missing)
    return "PASS", f"{len(DRAFT_HEADINGS) + 1} required headings and AI-use statement present"


# ---------------------------------------------------------------------------------------------
# G9 length
# ---------------------------------------------------------------------------------------------


def main_text_lines(text: str) -> list[str]:
    """Lines from the Introduction heading up to (not including) the References heading."""
    clean = strip_comments(text)
    heads = parse_headings(clean)
    intro = _find_heading(heads, "introduction")
    refs = next((h for h in heads if _norm_title(h.title).startswith("references")), None)
    lines = clean.split("\n")
    start = intro.line if intro else 0
    end = refs.line if refs and refs.line > start else len(lines)
    return lines[start:end]


def measure_length(text: str) -> tuple[int, int]:
    """(words in main text, tables + figures in main text)."""
    words = 0
    tables = 0
    in_fence = False
    in_table = False
    for line in main_text_lines(text):
        stripped = line.strip()
        if FENCE_RE.match(line):
            in_fence = not in_fence
            in_table = False
            continue
        if in_fence:
            continue
        if stripped.startswith("|"):
            if not in_table:
                tables += 1
                in_table = True
            continue
        in_table = False
        n_tf = len(TABLE_FIG_PLACEHOLDER_RE.findall(stripped))
        tables += n_tf
        if n_tf and not TABLE_FIG_PLACEHOLDER_RE.sub("", stripped).strip():
            continue  # a line that is only a table/figure placeholder is not prose
        if re.match(r"^!\[[^\]]*\]\([^)]*\)$", stripped):
            tables += 1
            continue
        words += count_words(stripped)
    return words, tables


def check_length(text: str) -> GateResult:
    words, tf = measure_length(text)
    detail = f"main text {words} words (max {MAX_MAIN_WORDS}); {tf} tables+figures (max {MAX_MAIN_TABLES_FIGS})"
    ok = words <= MAX_MAIN_WORDS and tf <= MAX_MAIN_TABLES_FIGS
    return ("PASS" if ok else "FAIL"), detail


# ---------------------------------------------------------------------------------------------
# G10 citations
# ---------------------------------------------------------------------------------------------


def cited_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in CITE_RE.finditer(strip_comments(text)):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def reference_entries(refs_text: str) -> dict[str, bool]:
    """key -> confirmed? An entry is confirmed when it carries 'CONFIRMED' (not 'UNCONFIRMED')."""
    out: dict[str, bool] = {}
    for line in strip_comments(refs_text).split("\n"):
        m = REF_ENTRY_RE.match(line)
        if m:
            out[m.group(1).strip()] = bool(CONFIRMED_RE.search(m.group(2)))
    return out


def check_citations(draft: str, refs_text: str) -> GateResult:
    keys = cited_keys(draft)
    entries = reference_entries(refs_text)
    missing = [k for k in keys if k not in entries]
    if missing:
        return "FAIL", f"{len(missing)} cited key(s) with no references.md entry: {', '.join(missing)}"
    unconfirmed = [k for k in keys if not entries[k]]
    if unconfirmed:
        return "BLOCKED", f"{len(unconfirmed)} of {len(keys)} cited keys not CONFIRMED by user: {', '.join(unconfirmed)}"
    return "PASS", f"{len(keys)} cited keys, all CONFIRMED"


# ---------------------------------------------------------------------------------------------
# G11 wording lint
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LintHit:
    file: str
    line: int  # 1-based
    term: str


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for m in re.finditer(r"[.!?](?=\s)|\n\s*\n", text[:pos]):
        start = m.end()
    m = re.compile(r"[.!?](?=\s|$)|\n\s*\n").search(text, pos)
    end = m.end() if m else len(text)
    return start, end


def lint_text(text: str, filename: str) -> list[LintHit]:
    clean = strip_comments(text)
    hits: list[LintHit] = []
    for term, pat in LINT_PATTERNS.items():
        for m in pat.finditer(clean):
            quals = LINT_QUALIFIERS.get(term)
            if quals:
                s, e = _sentence_bounds(clean, m.start())
                if any(q in clean[s:e].lower() for q in quals):
                    continue
            hits.append(LintHit(filename, clean.count("\n", 0, m.start()) + 1, term))
    return sorted(hits, key=lambda h: (h.file, h.line, h.term))


VERDICT_PHRASES: tuple[tuple[str, str], ...] = (
    ("not supported", "not supported"), ("unsupported", "not supported"), ("opposite", "opposite"),
    ("inconclusive", "inconclusive"), ("supported", "supported"),
)


def _claimed_verdicts(sentence: str) -> set[str]:
    s = sentence.lower()
    found: set[str] = set()
    for phrase, verdict in VERDICT_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", s):
            found.add(verdict)
            s = s.replace(phrase, " ")
    return found


def pooled_verdicts(rows: Iterable[dict[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("subject") == "ALL_MODELS" and r.get("hypothesis"):
            out[r["hypothesis"].strip().upper()].add((r.get("verdict") or "").strip().lower())
    return dict(out)


def check_hypothesis_claims(draft: str, verdicts: dict[str, set[str]], filename: str) -> list[str]:
    """Best-effort: sentences naming Hn with a verdict word that contradicts the pooled verdicts."""
    clean = PLACEHOLDER_RE.sub(lambda m: " " * len(m.group(0)), strip_comments(draft))
    problems: list[str] = []
    for m in re.finditer(r"\bH([1-6])\b", clean):
        s, e = _sentence_bounds(clean, m.start())
        claimed = _claimed_verdicts(clean[s:e])
        if not claimed:
            continue
        hyp = f"H{m.group(1)}"
        actual = verdicts.get(hyp)
        line = clean.count("\n", 0, m.start()) + 1
        if actual is None:
            problems.append(f"{filename}:{line} {hyp} claimed {sorted(claimed)} but no pooled verdict in hypotheses.csv")
        elif not claimed & actual:
            problems.append(f"{filename}:{line} {hyp} claimed {sorted(claimed)}, hypotheses.csv says {sorted(actual)}")
    return problems


# ---------------------------------------------------------------------------------------------
# G12 README
# ---------------------------------------------------------------------------------------------

LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def broken_relative_links(text: str, base_dir: Path) -> list[str]:
    broken: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK_RE.finditer(re.sub(r"`[^`]*`", "", line)):
            target = m.group(1)
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
            if path and not (base_dir / path).exists():
                broken.append(target)
    return broken


def readme_problems(text: str, base_dir: Path) -> list[str]:
    problems: list[str] = []
    titles = [h.title.lower() for h in parse_headings(text)]
    for alts in README_REQUIRED:
        if not any(a in t for t in titles for a in alts):
            problems.append("no heading: " + " / ".join(alts))
    if "<!-- BEGIN GENERATED" not in text or "<!-- END GENERATED" not in text:
        problems.append("no <!-- BEGIN GENERATED ... END GENERATED --> block")
    for s in README_STALE:
        if s in text:
            problems.append(f"stale string {s!r}")
    for link in broken_relative_links(text, base_dir):
        problems.append(f"broken link {link}")
    return problems


# ---------------------------------------------------------------------------------------------
# G13 secrets
# ---------------------------------------------------------------------------------------------


def scan_secrets(files: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """(file, pattern name) for every key-like string. The match itself is never returned."""
    hits: list[tuple[str, str]] = []
    for name, text in files:
        for pname, pat in SECRET_PATTERNS.items():
            if pat.search(text):
                hits.append((name, pname))
    return hits


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------
# Repo context and subprocess helpers
# ---------------------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()[:200]}")
    return out.stdout


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=here, capture_output=True,
                             text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return here.parent


@dataclass
class Ctx:
    root: Path
    python: str
    skip_tests: bool = False

    @property
    def draft(self) -> Path:
        return self.root / "paper" / "draft.md"

    @property
    def refs(self) -> Path:
        return self.root / "paper" / "references.md"

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def results_csv(self) -> Path:
        return self.root / "results" / "raw" / "results.csv"

    @property
    def log_dir(self) -> Path:
        return self.root / "results" / "raw" / "trajectories"

    def run(self, args: Sequence[str], timeout: int = SUBPROCESS_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([self.python, *args], cwd=self.root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, env=env)

    def judge_model(self) -> str | None:
        import yaml  # local: only needed here
        with (self.root / "config.yaml").open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        model = (raw.get("judge") or {}).get("model")
        return str(model) if model else None


def _sys_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ---------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------


def g1_tests(ctx: Ctx) -> GateResult:
    if ctx.skip_tests:
        return "SKIP", "skipped (--skip-tests)"
    out = ctx.run(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
    lines = [ln for ln in (out.stdout + out.stderr).strip().split("\n") if ln.strip()]
    summary = lines[-1].strip() if lines else "(no output)"
    return ("PASS" if out.returncode == 0 else "FAIL"), summary


def g2_data_integrity(ctx: Ctx) -> GateResult:
    _sys_path(ctx.root)
    from src import benchmark, harness

    cfg = harness.load_config(ctx.root / "config.yaml")
    problems: list[str] = []
    with ctx.results_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    if header != harness.RESULT_COLUMNS:
        return "FAIL", "results.csv header != RESULT_COLUMNS"
    phash = harness.protocol_hash(cfg.paths["benchmark"], benchmark.load_turns(cfg.paths["turns"]))
    if phash != EXPECTED_PROTOCOL_HASH:
        problems.append(f"items/turns hash to {phash}, expected {EXPECTED_PROTOCOL_HASH}")
    try:
        harness.read_done_keys(ctx.results_csv, EXPECTED_PROTOCOL_HASH)
    except Exception as exc:  # ConfigError from mixed protocols
        problems.append(f"protocol_hash rows: {exc}")
    keys = Counter(harness._cell_key(r) for r in rows)
    dups = [k for k, n in keys.items() if n > 1]
    if dups:
        problems.append(f"{len(dups)} duplicate cell keys, e.g. {dups[0]}")
    t1: set[tuple[tuple[str, str, str, int], str]] = set()
    judge_models: Counter[str] = Counter()
    for rec in harness.iter_log_records(ctx.log_dir):
        if rec.get("event") != "call" or rec.get("error") is not None or "subject" not in rec:
            continue
        if rec.get("role") == "subject" and rec.get("arm") == "T1" and rec.get("purpose") == "primary":
            t1.add(((str(rec["subject"]), str(rec["condition"]), str(rec["item_id"]), int(rec["sample"])),
                    str(rec["run_id"])))
        elif rec.get("role") == "judge":
            judge_models[str((rec.get("request") or {}).get("model"))] += 1
    no_t1 = [r for r in rows if (harness._cell_key(r), r["run_id"]) not in t1]
    if no_t1:
        problems.append(f"{len(no_t1)} rows without a logged successful T1 call (same cell + run_id)")
    if len(judge_models) != 1:
        problems.append(f"successful judge calls use {len(judge_models)} models: {dict(judge_models)}")
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        counts[r["subject"]][r["condition"]][r["sample"]] += 1
    summary = "; ".join(
        f"{m}: " + " ".join(f"{c}[" + ",".join(f"s{s}={n}" for s, n in sorted(cs.items())) + "]"
                            for c, cs in conds.items())
        for m, conds in counts.items()
    )
    jm = ",".join(judge_models) or "none"
    if problems:
        return "FAIL", "; ".join(problems) + f" | {summary}"
    return "PASS", f"{len(rows)} rows, hash {phash}, judge {jm} | {summary}"


def hash_match(recorded: str | None, data: bytes) -> Literal["exact", "eol", "none"]:
    """Does ``recorded`` equal sha256 of ``data``, or of it with only line endings changed?

    The csv module writes CRLF; git (text=auto eol=lf) checks the file out with LF, so the same data can
    hash differently in a fresh clone. An eol-only match counts as fresh but is reported.
    """
    if recorded == hashlib.sha256(data).hexdigest():
        return "exact"
    lf = data.replace(b"\r\n", b"\n")
    for variant in (lf, lf.replace(b"\n", b"\r\n")):
        if recorded == hashlib.sha256(variant).hexdigest():
            return "eol"
    return "none"


def robustness_missing(rob_dir: Path) -> list[str]:
    """Which of the two robustness outputs (all samples, no exclusions) are missing."""
    found = {"all-samples": False, "no-exclusions": False}
    if rob_dir.is_dir():
        for p in rob_dir.rglob("*"):
            if not p.is_file() or p.name not in ("metrics.csv", "hypotheses.csv"):
                continue
            rel = p.relative_to(rob_dir).as_posix().lower()
            if re.search(r"no[_-]exclusions", rel):
                found["no-exclusions"] = True
            elif "all" in rel:
                found["all-samples"] = True
    return [k for k, v in found.items() if not v]


def appendix_mentions(text: str, word: str) -> bool:
    clean = strip_comments(text)
    heads = parse_headings(clean)
    app = next((h for h in heads if h.title.lower().startswith("appendix")), None)
    if app is None:
        return False
    after = "\n".join(clean.split("\n")[app.line:])
    return word.lower() in after.lower()


def g3_analysis_fresh(ctx: Ctx) -> GateResult:
    problems: list[str] = []
    notes: list[str] = []
    manifest = ctx.tables / "analysis_manifest.json"
    if not manifest.exists():
        problems.append("analysis_manifest.json missing")
    else:
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("input_sha256")
        match = hash_match(recorded, ctx.results_csv.read_bytes())
        if match == "none":
            problems.append("analysis_manifest input_sha256 != sha256(results.csv) (rerun analysis)")
        elif match == "eol":
            notes.append("manifest hash matches results.csv only after CRLF/LF normalisation")
    csv_mtime = ctx.results_csv.stat().st_mtime
    for name in ANALYSIS_TABLES:
        p = ctx.tables / name
        if not p.exists():
            problems.append(f"{name} missing")
        elif p.stat().st_mtime < csv_mtime:
            problems.append(f"{name} older than results.csv")
    missing = robustness_missing(ctx.tables / "robustness")
    if missing:
        problems.append("robustness/ lacks " + " and ".join(missing) + " outputs (metrics.csv or hypotheses.csv)")
    if not appendix_mentions(ctx.draft.read_text(encoding="utf-8"), "robustness"):
        problems.append("draft.md appendix does not mention the robustness tables")
    note = f" (note: {'; '.join(notes)})" if notes else ""
    if problems:
        return "FAIL", "; ".join(problems) + note
    return "PASS", "manifest hash matches results.csv; tables fresh; robustness outputs present and in appendix" + note


def g4_validation_tables(ctx: Ctx) -> GateResult:
    missing = [(n, why) for n, why in VALIDATION_TABLES if not (ctx.tables / n).exists()]
    if not missing:
        return "PASS", "human_judge_agreement, annotation_agreement present"
    return "BLOCKED", "missing " + "; ".join(f"{n} (waits for {why})" for n, why in missing)


def human_owned_marker(marker: str, entries: dict[str, bool]) -> bool:
    if marker in ("[AUTHOR NAMES]", "[AFFILIATIONS]"):
        return True
    m = CITE_RE.fullmatch(marker)
    if m:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        return any(not entries.get(k, False) for k in keys)
    return False


def classify_build_errors(stderr: str, entries: dict[str, bool]) -> tuple[list[str], list[str]]:
    """(human-owned markers, other problems) from build_paper --final stderr."""
    human: list[str] = []
    other: list[str] = []
    for line in stderr.split("\n"):
        s = line.strip()
        if not s or s.startswith("WARNING:") or s.startswith("BUILD FAILED"):
            continue
        m = re.search(r"unfinished marker (\[.*\]) \(--final\)", s)
        if m:
            (human if human_owned_marker(m.group(1), entries) else other).append(m.group(1))
        else:
            other.append(s)
    return human, other


def _summarise(items: Sequence[str], limit: int = 6) -> str:
    c = Counter(m if len(m) <= 40 else m[:37] + "...]" for m in items)
    parts = [f"{k} x{n}" if n > 1 else k for k, n in c.most_common(limit)]
    more = len(c) - limit
    return ", ".join(parts) + (f", +{more} more kinds" if more > 0 else "")


def g5_final_build(ctx: Ctx) -> GateResult:
    with tempfile.TemporaryDirectory() as tmp:
        out = ctx.run(["tools/build_paper.py", "--final", "--out-dir", tmp, "--no-docx"])
    if out.returncode == 0:
        return "PASS", "final build succeeded"
    entries = reference_entries(ctx.refs.read_text(encoding="utf-8")) if ctx.refs.exists() else {}
    human, other = classify_build_errors(out.stderr, entries)
    if not other and human:
        return "BLOCKED", f"only human-owned markers remain ({len(human)}): {_summarise(human)}"
    if not other and not human:
        return "FAIL", f"build failed: {out.stderr.strip()[-300:]}"
    return "FAIL", f"{len(other)} unresolved placeholder/marker(s): {_summarise(other)}" + (
        f"; plus {len(human)} human-owned" if human else "")


def g6_numbers_audit(ctx: Ctx) -> GateResult:
    with tempfile.TemporaryDirectory() as tmp:
        b = ctx.run(["tools/build_paper.py", "--out-dir", tmp, "--no-docx"])
        if b.returncode != 0:
            return "FAIL", "non-final build failed: " + (b.stderr.strip().split("\n") or [""])[-1][:200]
        paper = Path(tmp) / "paper.md"
        nums = Path(tmp) / "numbers_used.csv"
        a = ctx.run(["tools/numbers_audit.py", str(paper), "--numbers-used", str(nums)])
        md_line = ((a.stdout if a.returncode == 0 else a.stderr).strip().split("\n") or [""])[0]
        md_line = md_line.replace(str(paper), "paper.md")
        if a.returncode != 0:
            return "FAIL", f"md: {md_line[:240]}"
        pdfs = sorted((ctx.root / "paper").glob("*.pdf"))
        if not pdfs:
            return "BLOCKED", f"md: {md_line}; PDF: BLOCKED (export pending)"
        bad: list[str] = []
        for pdf in pdfs:
            r = ctx.run(["tools/numbers_audit.py", str(pdf), "--numbers-used", str(nums)])
            if r.returncode != 0:
                bad.append(f"{pdf.name}: {(r.stderr.strip().split(chr(10)) or [''])[0][:160]}")
        if bad:
            return "FAIL", f"md: {md_line}; PDF: " + "; ".join(bad)
        return "PASS", f"md: {md_line}; PDF: {len(pdfs)} file(s) audited"


def g7_abstract(ctx: Ctx) -> GateResult:
    return check_abstract(ctx.draft.read_text(encoding="utf-8"))


def g8_structure(ctx: Ctx) -> GateResult:
    return check_structure(ctx.draft.read_text(encoding="utf-8"), ctx.judge_model())


def g9_length(ctx: Ctx) -> GateResult:
    return check_length(ctx.draft.read_text(encoding="utf-8"))


def g10_citations(ctx: Ctx) -> GateResult:
    refs = ctx.refs.read_text(encoding="utf-8") if ctx.refs.exists() else ""
    return check_citations(ctx.draft.read_text(encoding="utf-8"), refs)


def g11_wording(ctx: Ctx) -> GateResult:
    hits: list[LintHit] = []
    for rel in ("paper/draft.md", "README.md"):
        p = ctx.root / rel
        if p.exists():
            hits.extend(lint_text(p.read_text(encoding="utf-8"), rel))
    problems = [f"{h.file}:{h.line} '{h.term}'" for h in hits]
    hyp = ctx.tables / "hypotheses.csv"
    if hyp.exists() and ctx.draft.exists():
        with hyp.open(encoding="utf-8", newline="") as fh:
            verdicts = pooled_verdicts(csv.DictReader(fh))
        problems += check_hypothesis_claims(ctx.draft.read_text(encoding="utf-8"), verdicts, "paper/draft.md")
    if problems:
        return "FAIL", f"{len(problems)} hit(s): " + "; ".join(problems[:12]) + (" ..." if len(problems) > 12 else "")
    return "PASS", "no banned wording; no hypothesis-verdict contradictions found (heuristic)"


def g12_readme(ctx: Ctx) -> GateResult:
    readme = ctx.root / "README.md"
    problems = readme_problems(readme.read_text(encoding="utf-8"), ctx.root)
    notes: list[str] = []
    br = ctx.root / "tools" / "build_readme.py"
    if br.exists():
        h = ctx.run([str(br.relative_to(ctx.root)), "--help"], timeout=120)
        if "--check" in h.stdout:
            c = ctx.run([str(br.relative_to(ctx.root)), "--check"], timeout=300)
            if c.returncode != 0:
                problems.append("build_readme.py --check failed (generated block stale)")
        else:
            notes.append("build_readme.py has no --check; sub-check skipped")
    else:
        notes.append("tools/build_readme.py absent; --check skipped")
    note = f" ({'; '.join(notes)})" if notes else ""
    if problems:
        return "FAIL", f"{len(problems)} problem(s): " + "; ".join(problems) + note
    return "PASS", "headings, generated block, links ok" + note


def _in_worktree(root: Path) -> bool:
    git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir").strip()).resolve()
    common = _git(root, "rev-parse", "--git-common-dir").strip()
    common_p = Path(common) if Path(common).is_absolute() else (root / common)
    return git_dir != common_p.resolve()


def g13_hygiene(ctx: Ctx) -> GateResult:
    root = ctx.root
    problems: list[str] = []
    notes: list[str] = []
    tracked = [p for p in _git(root, "ls-files", "-z").split("\0") if p]
    tracked_set = set(tracked)

    def texts() -> Iterable[tuple[str, str]]:
        for rel in tracked:
            t = _read_text(root / rel)
            if t is not None:
                yield rel, t

    for f, pname in scan_secrets(texts()):
        problems.append(f"key-like string ({pname}) in {f}")
    if ".env" in tracked_set:
        problems.append(".env is tracked")
    gi = root / ".gitignore"
    gi_lines = {ln.strip() for ln in gi.read_text(encoding="utf-8").split("\n")} if gi.exists() else set()
    lacking = [e for e in GITIGNORE_REQUIRED if e not in gi_lines]
    if lacking:
        problems.append(".gitignore lacks " + ", ".join(lacking))
    for rel in UNTRACKED_REQUIRED:
        if rel in tracked_set:
            problems.append(f"{rel} is tracked")
    if _in_worktree(root):
        notes.append("worktree/branch checks skipped: running inside a worktree")
    else:
        wt = [ln for ln in _git(root, "worktree", "list", "--porcelain").split("\n")
              if ln.startswith("worktree ") and ".claude/worktrees" in ln.replace("\\", "/")]
        if wt:
            problems.append(f"{len(wt)} agent worktree(s) under .claude/worktrees")
        branches = [b.strip() for b in _git(root, "branch", "--format=%(refname:short)").split("\n")
                    if b.strip().startswith("worktree-agent-")]
        if branches:
            problems.append(f"{len(branches)} local worktree-agent-* branch(es)")
    if NOTEBOOK_PATH not in tracked_set:
        problems.append(f"{NOTEBOOK_PATH} not tracked")
    else:
        nb = json.loads((root / NOTEBOOK_PATH).read_text(encoding="utf-8"))
        with_out = sum(1 for c in nb.get("cells", []) if c.get("outputs"))
        if with_out:
            problems.append(f"{NOTEBOOK_PATH} has outputs in {with_out} cell(s)")
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    for mention in ("tools/judge_subset.py", "tools/kaggle_sim"):
        if mention not in readme:
            problems.append(f"README does not mention {mention}")
    note = f" ({'; '.join(notes)})" if notes else ""
    if problems:
        return "FAIL", "; ".join(problems) + note
    return "PASS", f"{len(tracked)} tracked files scanned, no key-like strings; ignore rules ok" + note


def g14_eol_status(ctx: Ctx) -> GateResult:
    crlf = [ln.split("\t", 1)[-1] for ln in _git(ctx.root, "ls-files", "--eol").split("\n")
            if ln.startswith("i/crlf")]
    changes = [ln for ln in _git(ctx.root, "status", "--porcelain").split("\n") if ln.strip()]
    problems: list[str] = []
    if crlf:
        problems.append(f"{len(crlf)} file(s) committed with CRLF: {', '.join(crlf[:5])}")
    if changes:
        problems.append(f"working tree not clean: {len(changes)} change(s)")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "no CRLF in index; working tree clean"


GATES: dict[str, tuple[str, Callable[[Ctx], GateResult]]] = {
    "G1": ("tests", g1_tests),
    "G2": ("data integrity", g2_data_integrity),
    "G3": ("analysis fresh", g3_analysis_fresh),
    "G4": ("validation tables", g4_validation_tables),
    "G5": ("final build", g5_final_build),
    "G6": ("numbers audit", g6_numbers_audit),
    "G7": ("abstract & title", g7_abstract),
    "G8": ("structure", g8_structure),
    "G9": ("length", g9_length),
    "G10": ("citations", g10_citations),
    "G11": ("wording lint", g11_wording),
    "G12": ("README", g12_readme),
    "G13": ("hygiene", g13_hygiene),
    "G14": ("line endings & status", g14_eol_status),
}


@dataclass(frozen=True)
class GateOutcome:
    gate: str
    name: str
    status: Status
    detail: str
    seconds: float


def run_gate(gate: str, ctx: Ctx) -> GateOutcome:
    name, fn = GATES[gate]
    t0 = time.monotonic()
    try:
        status, detail = fn(ctx)
    except Exception as exc:  # each gate reports its own failure; the runner never crashes
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        status, detail = "FAIL", f"error: {type(exc).__name__}: {exc}{where}"
    return GateOutcome(gate, name, status, " ".join(detail.split()), time.monotonic() - t0)


def format_table(outcomes: Sequence[GateOutcome]) -> str:
    w = max((len(f"{o.gate} {o.name}") for o in outcomes), default=4)
    lines = [f"{'gate'.ljust(w)}  {'status':7}  detail", f"{'-' * w}  {'-' * 7}  {'-' * 40}"]
    for o in outcomes:
        lines.append(f"{f'{o.gate} {o.name}'.ljust(w)}  {o.status:7}  {o.detail}")
    c = Counter(o.status for o in outcomes)
    lines.append("")
    lines.append(" ".join(f"{k}={c.get(k, 0)}" for k in ("PASS", "FAIL", "BLOCKED", "SKIP")))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-tests", action="store_true", help="do not run pytest (G1 reports SKIP)")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--only", default=None, help="comma-separated gate ids, e.g. G1,G5")
    args = ap.parse_args(argv)
    wanted = list(GATES)
    if args.only:
        wanted = [g.strip().upper() for g in args.only.split(",") if g.strip()]
        unknown = [g for g in wanted if g not in GATES]
        if unknown:
            ap.error(f"unknown gate(s): {', '.join(unknown)}")
    ctx = Ctx(root=repo_root(), python=sys.executable, skip_tests=args.skip_tests)
    outcomes = [run_gate(g, ctx) for g in wanted]
    if args.json:
        print(json.dumps([o.__dict__ for o in outcomes], indent=2))
    else:
        print(format_table(outcomes))
    return 1 if any(o.status == "FAIL" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
