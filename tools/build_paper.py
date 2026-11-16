"""Fill paper/draft.md from generated tables and write paper.md, paper.docx and numbers_used.csv.

Every result number in the paper comes from a placeholder resolved against a CSV in
results/tables/ (or the --tables directory). Methods constants come from config.yaml or from the
benchmark files. Nothing is typed by hand. Any placeholder that cannot be resolved is an error and
the build exits with status 1 without writing outputs.

Placeholders (one line each, ``{{kind:...}}``; fields separated by ``:``):

    {{num:<table>:<filters>:<column>:<format>}}       one cell, formatted (``s`` = as text)
    {{ci:<table>:<filters>:<format>}}                 "[ci_low, ci_high]" of one row
    {{agg:<table>:<filters>:<column>:<func>:<format>}} func in count|nunique|sum|min|max
    {{table:<table>:<filters>:<columns>}}             Markdown table; columns = ``col`` or
                                                      ``col=format`` (comma-separated); the
                                                      pseudo-column ``ci=format`` renders [lo, hi]
    {{pivot:<table>:<filters>:<row col>:<column col>:<value>:<format>}}
                                                      value = a column, or ``estimate+ci``
    {{fig:<file in figures dir>:<caption>}}           figure (caption may contain ':')
    {{cfg:<dotted.path>:<format>}}                    config.yaml value, e.g.
                                                      ``subjects[name=qwen2.5-7b].hf_id``
    {{data:<fact>:<format>}}                          benchmark/design fact (see data_facts)
    {{include:<file in the draft's directory>}}       verbatim inclusion (e.g. appendix_prompts.md)
    {{defaults:<table>:<filters>}}                    extra filters for every lookup in <table>
                                                      (a placeholder's own key wins); line removed
    {{refs}}                                          reference list: the references.md entry of
                                                      every key cited anywhere in the draft, sorted

A format may end in ``?<text>`` (e.g. ``.3f?–``): a missing value then renders as <text>, with no
warning, instead of ``n/a``; in a pivot with ``estimate+ci`` a missing estimate renders <text> alone.

Citations: ``[CITE:k1,k2]`` renders as author-year text, e.g. "(Shnaidman, 2026; Ma et al., 2026)",
from the entries in references.md (next to the draft unless --refs is given). A cited key with no
entry is a build error. With --final, a citation whose keys are not all marked CONFIRMED is left as
the marker and fails the build; in normal builds every entry is rendered.

Filters: ``key=value;key=value``; ``key=a|b`` matches either value (and orders table rows/pivot
axes in that order). An empty filter matches every row. ``num``/``ci`` must match exactly one row.
HTML comments (``<!-- ... -->``) in the draft are dropped before placeholders are resolved.

Usage (from the repo root):
    python tools/build_paper.py                    # results/tables -> paper/paper.md, .docx, numbers_used.csv
    python tools/build_paper.py --final            # also fail on unconfirmed citations, [CHECK ...] etc.
    python tools/build_paper.py --tables T --figures F --out-dir D --draft paper/draft.md --refs R
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MISSING = "n/a"
PLACEHOLDER_RE = re.compile(r"\{\{(?P<body>[^{}\n]*)\}\}")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
DRY_RUN_RE = re.compile(r"dry", re.IGNORECASE)
# Markers that must be gone before submission (checked with --final).
FINAL_MARKERS = re.compile(
    r"\[(?:CITE:[^\]]*|RESULT PROSE[^\]]*|ABSTRACT[^\]]*|AUTHOR NAMES|AFFILIATIONS|REFERENCES[^\]]*|"
    r"DISCUSSION[^\]]*|CHECK[^\]]*)\]"
)
OUTPUT_NAMES = ("paper.md", "paper.docx", "numbers_used.csv")
NUMBERS_USED_COLUMNS = ("placeholder", "kind", "value", "source_file", "source_row", "column")


class BuildError(Exception):
    """A placeholder could not be resolved."""


@dataclass(frozen=True)
class UsedNumber:
    placeholder: str
    kind: str
    value: str
    source_file: str
    source_row: str   # 0-based data-row index (header excluded); ';'-joined for aggregates
    column: str


@dataclass
class Table:
    name: str
    path: Path
    header: list[str]
    rows: list[dict[str, str]]


@dataclass
class BuildContext:
    tables_dir: Path
    figures_dir: Path
    config_path: Path
    draft_dir: Path
    out_dir: Path
    allow_dry_run: bool = False
    refs_path: Path | None = None   # default: <draft_dir>/references.md
    final: bool = False             # set by build(); unconfirmed citations stay as markers
    used: list[UsedNumber] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    defaults: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    _tables: dict[str, Table] = field(default_factory=dict)
    _config: dict[str, Any] | None = None
    _facts: dict[str, int] | None = None
    _refs: dict[str, RefEntry] | None = None

    # -- sources ---------------------------------------------------------------------------
    def table(self, name: str) -> Table:
        key = name[:-4] if name.endswith(".csv") else name
        if key not in self._tables:
            path = self.tables_dir / f"{key}.csv"
            if not path.is_file():
                raise BuildError(f"table {key!r} not found at {path}")
            with path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                header = list(reader.fieldnames or [])
                rows = [{k: (v if v is not None else "") for k, v in r.items()} for r in reader]
            if not header:
                raise BuildError(f"table {key!r} at {path} has no header")
            self._tables[key] = Table(key, path, header, rows)
        return self._tables[key]

    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config

    def facts(self) -> dict[str, int]:
        if self._facts is None:
            self._facts = data_facts(self.config_path)
        return self._facts

    def references_path(self) -> Path:
        return self.refs_path if self.refs_path is not None else self.draft_dir / "references.md"

    def references(self) -> dict[str, RefEntry]:
        if self._refs is None:
            path = self.references_path()
            if not path.is_file():
                raise BuildError(f"references file not found at {path}")
            self._refs = parse_references(path.read_text(encoding="utf-8"))
        return self._refs

    def rel(self, path: Path) -> str:
        try:
            return Path(os.path.relpath(path.resolve(), REPO_ROOT)).as_posix()
        except ValueError:
            return path.resolve().as_posix()


# ---------------------------------------------------------------------------------------------
# Config and benchmark facts (shared with tools/numbers_audit.py)
# ---------------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise BuildError(f"{path}: not a YAML mapping")
    return data


_PATH_PART = re.compile(r"(?P<key>[A-Za-z_][\w]*)(?:\[(?P<k>[A-Za-z_]\w*)=(?P<v>[^\]]+)\])?")


def config_lookup(config: dict[str, Any], dotted: str) -> Any:
    """Resolve ``a.b[name=x].c`` (list elements selected by a key=value match)."""
    node: Any = config
    pos = 0
    while pos < len(dotted):
        m = _PATH_PART.match(dotted, pos)
        if m is None:
            raise BuildError(f"bad config path {dotted!r} at position {pos}")
        key = m.group("key")
        if not isinstance(node, dict) or key not in node:
            raise BuildError(f"config path {dotted!r}: key {key!r} not found")
        node = node[key]
        if m.group("k"):
            if not isinstance(node, list):
                raise BuildError(f"config path {dotted!r}: {key!r} is not a list")
            hits = [e for e in node if isinstance(e, dict) and str(e.get(m.group("k"))) == m.group("v")]
            if len(hits) != 1:
                raise BuildError(f"config path {dotted!r}: {len(hits)} elements of {key!r} match "
                                 f"{m.group('k')}={m.group('v')}")
            node = hits[0]
        pos = m.end()
        if pos < len(dotted):
            if dotted[pos] != ".":
                raise BuildError(f"bad config path {dotted!r} at position {pos}")
            pos += 1
    return node


def _resolve(config_path: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else config_path.resolve().parent / p


def data_facts(config_path: Path) -> dict[str, int]:
    """Design facts counted from the benchmark files and the code's arm/condition lists."""
    from src.harness import ALL_ARMS, SUBJECT_PROB_ARMS  # local import: heavy module
    from src.prompts import CONDITIONS

    cfg = load_config(config_path)
    items_path = _resolve(config_path, cfg["paths"]["benchmark"])
    turns_path = _resolve(config_path, cfg["paths"]["turns"])
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with turns_path.open(encoding="utf-8") as fh:
        turns = yaml.safe_load(fh)
    facts: dict[str, int] = {
        "n_items": len(items),
        "n_domains": len({it["domain"] for it in items}),
        "n_tiers": len({it["tier"] for it in items}),
        "n_items_with_truth": sum(1 for it in items if it.get("truth") not in (None, "")),
        "n_conditions": len(CONDITIONS),
        "n_arms": len(ALL_ARMS),
        "n_subject_prob_arms": len(SUBJECT_PROB_ARMS),
        "n_turn_kinds": len(turns),
    }
    for tier in sorted({it["tier"] for it in items}):
        facts[f"items_tier_{tier}"] = sum(1 for it in items if it["tier"] == tier)
    for dom in sorted({it["domain"] for it in items}):
        facts[f"items_domain_{dom}"] = sum(1 for it in items if it["domain"] == dom)
    per_tier = {facts[k] for k in facts if k.startswith("items_tier_")}
    if len(per_tier) == 1:
        facts["items_per_tier"] = per_tier.pop()
    per_domain = {facts[k] for k in facts if k.startswith("items_domain_")}
    if len(per_domain) == 1:
        facts["items_per_domain"] = per_domain.pop()
    n_para = {len(v) for v in turns.values()}
    if len(n_para) == 1:
        facts["n_paraphrases"] = n_para.pop()
    return facts


# ---------------------------------------------------------------------------------------------
# References and citations
# ---------------------------------------------------------------------------------------------

CITE_RE = re.compile(r"\[CITE:(?P<keys>[^\]]*)\]")
REF_ENTRY_RE = re.compile(r"^\s*[-*]\s*\[(?P<key>[^\]]+)\]\s*(?P<text>.*\S)\s*$")
CONFIRMED_RE = re.compile(r"(?<![A-Za-z])CONFIRMED\b")  # 'UNCONFIRMED' does not count
YEAR_RE = re.compile(r"\((?P<year>\d{4}[a-z]?)\)")


@dataclass(frozen=True)
class RefEntry:
    key: str
    text: str        # entry text without the leading [key]
    confirmed: bool
    line: int        # 0-based line in references.md


def parse_references(text: str) -> dict[str, RefEntry]:
    """``- [key] Entry text`` lines of references.md (HTML comments ignored)."""
    out: dict[str, RefEntry] = {}
    clean = COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    for n, line in enumerate(clean.split("\n")):
        m = REF_ENTRY_RE.match(line)
        if m:
            key = m.group("key").strip()
            out[key] = RefEntry(key, m.group("text"), bool(CONFIRMED_RE.search(m.group("text"))), n)
    return out


def cite_keys(marker_body: str) -> list[str]:
    return [k.strip() for k in marker_body.split(",") if k.strip()]


def cited_keys(text: str) -> list[str]:
    """Every key cited in ``text`` (HTML comments ignored), in order of first citation."""
    keys: list[str] = []
    for m in CITE_RE.finditer(COMMENT_RE.sub("", text)):
        for k in cite_keys(m.group("keys")):
            if k not in keys:
                keys.append(k)
    return keys


def author_year(entry: RefEntry) -> str:
    """'Shnaidman, 2026', 'Kim & Khashabi, 2025' or 'Ma et al., 2026' from an author-year entry."""
    m = YEAR_RE.search(entry.text)
    if m is None:
        raise BuildError(f"reference {entry.key!r}: no '(YYYY)' year in the entry")
    authors = entry.text[:m.start()].strip().rstrip(",")
    et_al = bool(re.search(r"\bet al\.?", authors))
    authors = re.sub(r",?\s*\bet al\.?", "", authors)
    authors = re.sub(r",?\s*&\s*", ", ", authors)
    parts = [p.strip() for p in authors.split(", ") if p.strip()]
    surnames = parts[0::2]  # "Surname, Initials, Surname, Initials, ..."
    if not surnames:
        raise BuildError(f"reference {entry.key!r}: cannot read the author list")
    if et_al or len(surnames) >= 3:
        names = f"{surnames[0]} et al."
    elif len(surnames) == 2:
        names = f"{surnames[0]} & {surnames[1]}"
    else:
        names = surnames[0]
    return f"{names}, {m.group('year')}"


def render_citations(ctx: BuildContext, text: str, errors: list[str]) -> str:
    """Replace every [CITE:k1,k2] by "(A, 2026; B et al., 2025)"."""
    def repl(m: re.Match[str]) -> str:
        keys = cite_keys(m.group("keys"))
        try:
            refs = ctx.references()
        except BuildError as exc:
            errors.append(f"{m.group(0)}: {exc}")
            return m.group(0)
        missing = [k for k in keys if k not in refs]
        if missing or not keys:
            errors.append(f"{m.group(0)}: no entry in {ctx.references_path()} for "
                          f"{', '.join(missing) or '(empty citation)'}")
            return m.group(0)
        if ctx.final and not all(refs[k].confirmed for k in keys):
            return m.group(0)  # left as the marker; build(final=True) reports it as unfinished
        try:
            return "(" + "; ".join(author_year(refs[k]) for k in keys) + ")"
        except BuildError as exc:
            errors.append(f"{m.group(0)}: {exc}")
            return m.group(0)
    return CITE_RE.sub(repl, text)


def h_refs(ctx: BuildContext, ph: str, keys: Sequence[str]) -> str:
    if not keys:
        raise BuildError("refs: the draft cites no keys")
    refs = ctx.references()
    missing = [k for k in keys if k not in refs]
    if missing:
        raise BuildError(f"no entry in {ctx.references_path()} for {', '.join(missing)}")
    entries = sorted((refs[k] for k in keys), key=lambda e: (e.text.lower(), e.key))
    for e in entries:  # numbers inside entries (pages, arXiv ids) are traced to references.md
        ctx.used.append(UsedNumber(ph, "ref", e.text, ctx.rel(ctx.references_path()), str(e.line), e.key))
    return "\n".join(f"- {publication_text(e.text)}" for e in entries)


_STATUS_RE = re.compile(r"\s*\b(?:UN)?CONFIRMED\b[^.]*\.?\s*$")
_PROVENANCE_RE = re.compile(r"\s*\(per arXiv [^)]*\)")


def publication_text(entry_text: str) -> str:
    """Reference as printed in the paper: everything up to and including its URL, without the working
    notes kept in references.md (confirmation status, provenance remarks, author-list caveats)."""
    m = re.search(r"https?://\S+?(?=\.?(?:\s|$))", entry_text)
    text = entry_text[:m.end()] + "." if m else _STATUS_RE.sub("", entry_text)
    return _PROVENANCE_RE.sub("", text).strip()


# ---------------------------------------------------------------------------------------------
# Formatting and filtering
# ---------------------------------------------------------------------------------------------


def parse_float(raw: str) -> float | None:
    s = raw.strip()
    if s.lower() in ("", "nan", "na", "n/a", "none", "null", "<na>"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def format_value(raw: Any, spec: str) -> str | None:
    """Format a cell. Returns None when the value is missing (NaN/empty). ``s`` = verbatim text."""
    if spec in ("", "s"):
        text = str(raw)
        return None if text.strip() == "" else text
    if isinstance(raw, bool):
        raise BuildError(f"cannot format boolean {raw!r} with {spec!r}")
    v = float(raw) if isinstance(raw, (int, float)) else parse_float(str(raw))
    if v is None or math.isnan(v):
        if not isinstance(raw, (int, float)) and str(raw).strip() not in ("", "nan", "NaN", "NA", "n/a",
                                                                          "None", "null", "<NA>"):
            raise BuildError(f"value {raw!r} is not numeric (format {spec!r}); use format 's'")
        return None
    try:
        if spec.endswith("d"):
            if abs(v - round(v)) > 1e-9:
                raise BuildError(f"value {raw!r} is not an integer (format {spec!r})")
            out = format(int(round(v)), spec)
        else:
            out = format(v, spec)
    except ValueError as exc:
        raise BuildError(f"bad format {spec!r}: {exc}") from exc
    if out.startswith("-") and not re.search(r"[1-9]", out):
        out = out[1:]  # no "-0.000"
    return out


def parse_filters(spec: str) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    for part in (p.strip() for p in spec.split(";")):
        if not part:
            continue
        if "=" not in part:
            raise BuildError(f"bad filter {part!r} (expected key=value)")
        key, value = part.split("=", 1)
        filters[key.strip()] = [v.strip() for v in value.split("|")]
    return filters


def select(ctx: BuildContext, table: Table, filter_spec: str) -> tuple[list[int], dict[str, list[str]]]:
    filters = {**ctx.defaults.get(table.name, {}), **parse_filters(filter_spec)}
    for key in filters:
        if key not in table.header:
            raise BuildError(f"filter column {key!r} not in {table.name}.csv (columns: {', '.join(table.header)})")
    idx = [i for i, row in enumerate(table.rows)
           if all(row.get(k, "").strip() in vals for k, vals in filters.items())]
    order_keys = [k for k, v in parse_filters(filter_spec).items() if len(v) > 1]
    if order_keys:
        k = order_keys[0]
        rank = {v: n for n, v in enumerate(filters[k])}
        idx.sort(key=lambda i: rank[table.rows[i][k].strip()])
    if not ctx.allow_dry_run and "data_source" in table.header:
        dry = [i for i in idx if DRY_RUN_RE.search(table.rows[i]["data_source"])]
        if dry:
            raise BuildError(f"{table.name}.csv rows {dry[:5]} are dry-run (mock) output, not results")
    return idx, filters


def one_row(ctx: BuildContext, table: Table, filter_spec: str) -> int:
    idx, _ = select(ctx, table, filter_spec)
    if len(idx) != 1:
        raise BuildError(f"filter {filter_spec!r} matches {len(idx)} rows in {table.name}.csv (need exactly 1)"
                         + (f"; rows {idx[:10]}" if idx else ""))
    return idx[0]


def need_column(table: Table, col: str) -> None:
    if col not in table.header:
        raise BuildError(f"column {col!r} not in {table.name}.csv (columns: {', '.join(table.header)})")


# ---------------------------------------------------------------------------------------------
# Placeholder handlers
# ---------------------------------------------------------------------------------------------


def split_spec(spec: str) -> tuple[str, str | None]:
    """``.3f?–`` -> (``.3f``, ``–``): the text after '?' replaces a missing value."""
    fmt, sep, missing = spec.partition("?")
    return fmt, (missing if sep else None)


def _cell(ctx: BuildContext, ph: str, kind: str, table: Table, row: int, col: str, spec: str) -> str:
    need_column(table, col)
    fmt, missing_text = split_spec(spec)
    out = format_value(table.rows[row][col], fmt)
    if out is None and missing_text is not None:
        return missing_text
    if out is None:
        ctx.warnings.append(f"{ph}: value missing in {table.name}.csv row {row} column {col}; rendered {MISSING!r}")
        out = MISSING
    ctx.used.append(UsedNumber(ph, kind, out, ctx.rel(table.path), str(row), col))
    return out


def _ci(ctx: BuildContext, ph: str, table: Table, row: int, spec: str) -> str:
    lo = _cell(ctx, ph, "ci", table, row, "ci_low", spec)
    hi = _cell(ctx, ph, "ci", table, row, "ci_high", spec)
    return f"[{lo}, {hi}]"


def h_num(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 4:
        raise BuildError("num needs <table>:<filters>:<column>:<format>")
    t = ctx.table(args[0])
    return _cell(ctx, ph, "num", t, one_row(ctx, t, args[1]), args[2], args[3])


def h_ci(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 3:
        raise BuildError("ci needs <table>:<filters>:<format>")
    t = ctx.table(args[0])
    return _ci(ctx, ph, t, one_row(ctx, t, args[1]), args[2])


def h_agg(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 5:
        raise BuildError("agg needs <table>:<filters>:<column>:<func>:<format>")
    t = ctx.table(args[0])
    col, func, spec = args[2], args[3], args[4]
    need_column(t, col)
    idx, _ = select(ctx, t, args[1])
    if not idx:
        raise BuildError(f"filter {args[1]!r} matches 0 rows in {t.name}.csv")
    cells = [t.rows[i][col].strip() for i in idx]
    value: float | int
    if func == "count":
        value = len(cells)
    elif func == "nunique":
        value = len(set(cells))
    elif func in ("sum", "min", "max"):
        nums = [parse_float(c) for c in cells]
        if any(n is None for n in nums):
            raise BuildError(f"agg {func} over {t.name}.{col}: missing or non-numeric values in rows {idx[:10]}")
        clean = [n for n in nums if n is not None]
        value = sum(clean) if func == "sum" else (min(clean) if func == "min" else max(clean))
    else:
        raise BuildError(f"unknown agg function {func!r} (count|nunique|sum|min|max)")
    out = format_value(value, spec)
    assert out is not None
    ctx.used.append(UsedNumber(ph, f"agg_{func}", out, ctx.rel(t.path), ";".join(map(str, idx)), col))
    return out


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def h_table(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 3:
        raise BuildError("table needs <table>:<filters>:<columns>")
    t = ctx.table(args[0])
    idx, _ = select(ctx, t, args[1])
    if not idx:
        raise BuildError(f"filter {args[1]!r} matches 0 rows in {t.name}.csv")
    cols: list[tuple[str, str]] = []
    for part in (p.strip() for p in args[2].split(",")):
        name, _, spec = part.partition("=")
        if name != "ci":
            need_column(t, name)
        cols.append((name, spec))
    lines = [_md_row([c for c, _ in cols]), _md_row(["---"] * len(cols))]
    for i in idx:
        cells = [(_ci(ctx, ph, t, i, spec) if name == "ci" else _cell(ctx, ph, "table", t, i, name, spec))
                 for name, spec in cols]
        lines.append(_md_row(cells))
    return "\n".join(lines)


def h_pivot(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 6:
        raise BuildError("pivot needs <table>:<filters>:<row col>:<column col>:<value>:<format>")
    t = ctx.table(args[0])
    row_col, col_col, value, spec = args[2], args[3], args[4], args[5]
    need_column(t, row_col)
    need_column(t, col_col)
    with_ci = value == "estimate+ci"
    value_col = "estimate" if with_ci else value
    need_column(t, value_col)
    idx, filters = select(ctx, t, args[1])
    if not idx:
        raise BuildError(f"filter {args[1]!r} matches 0 rows in {t.name}.csv")

    def axis(col: str) -> list[str]:
        if col in filters and len(filters[col]) > 1:
            return [v for v in filters[col] if any(t.rows[i][col].strip() == v for i in idx)]
        seen: list[str] = []
        for i in idx:
            v = t.rows[i][col].strip()
            if v not in seen:
                seen.append(v)
        return seen

    cell_rows: dict[tuple[str, str], int] = {}
    for i in idx:
        k = (t.rows[i][row_col].strip(), t.rows[i][col_col].strip())
        if k in cell_rows:
            raise BuildError(f"pivot: {row_col}={k[0]}, {col_col}={k[1]} matches rows {cell_rows[k]} and {i} "
                             f"in {t.name}.csv; add a filter")
        cell_rows[k] = i
    rows_ax, cols_ax = axis(row_col), axis(col_col)
    lines = [_md_row([row_col, *cols_ax]), _md_row(["---"] * (len(cols_ax) + 1))]
    for r in rows_ax:
        cells = [r]
        for c in cols_ax:
            i = cell_rows.get((r, c))
            if i is None:
                cells.append("missing")
                ctx.warnings.append(f"{ph}: no row for {row_col}={r}, {col_col}={c}")
                continue
            fmt, missing_text = split_spec(spec)
            if with_ci and missing_text is not None and format_value(t.rows[i][value_col], fmt) is None:
                cells.append(missing_text)
                continue
            text = _cell(ctx, ph, "pivot", t, i, value_col, spec)
            if with_ci:
                text += " " + _ci(ctx, ph, t, i, spec)
            cells.append(text)
        lines.append(_md_row(cells))
    return "\n".join(lines)


def h_fig(ctx: BuildContext, ph: str, body: str) -> str:
    rest = body[len("fig:"):]
    name, sep, caption = rest.partition(":")
    if not sep or not name.strip() or not caption.strip():
        raise BuildError("fig needs <file>:<caption>")
    path = ctx.figures_dir / name.strip()
    if not path.is_file():
        raise BuildError(f"figure {name.strip()!r} not found at {path}")
    try:
        rel = Path(os.path.relpath(path.resolve(), ctx.out_dir.resolve())).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    return f"![{caption.strip()}]({rel})\n\n*{caption.strip()}*"


def h_cfg(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 2:
        raise BuildError("cfg needs <dotted.path>:<format>")
    value = config_lookup(ctx.config(), args[0])
    if isinstance(value, (dict, list)) or value is None:
        raise BuildError(f"config value {args[0]!r} is {value!r}; not a scalar")
    out = format_value(value, args[1])
    if out is None:
        raise BuildError(f"config value {args[0]!r} is empty")
    ctx.used.append(UsedNumber(ph, "cfg", out, ctx.rel(ctx.config_path), args[0], ""))
    return out


def h_data(ctx: BuildContext, ph: str, args: list[str]) -> str:
    if len(args) != 2:
        raise BuildError("data needs <fact>:<format>")
    facts = ctx.facts()
    if args[0] not in facts:
        raise BuildError(f"unknown data fact {args[0]!r} (known: {', '.join(sorted(facts))})")
    out = format_value(facts[args[0]], args[1])
    assert out is not None
    ctx.used.append(UsedNumber(ph, "data", out, "data/benchmark + src (data_facts)", args[0], ""))
    return out


HANDLERS: dict[str, Callable[[BuildContext, str, list[str]], str]] = {
    "num": h_num, "ci": h_ci, "agg": h_agg, "table": h_table, "pivot": h_pivot, "cfg": h_cfg, "data": h_data,
}


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


def _expand_includes(ctx: BuildContext, text: str, errors: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        body = m.group("body").strip()
        if not body.startswith("include:"):
            return m.group(0)
        path = ctx.draft_dir / body[len("include:"):].strip()
        if not path.is_file():
            errors.append(f"{m.group(0)}: file not found at {path}")
            return m.group(0)
        return path.read_text(encoding="utf-8").rstrip("\n")
    return PLACEHOLDER_RE.sub(repl, text)


def _collect_defaults(ctx: BuildContext, text: str, errors: list[str]) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        m = re.fullmatch(r"\s*\{\{\s*defaults:(?P<t>[^:{}]+):(?P<f>[^{}]*)\}\}\s*", line)
        if m:
            try:
                table = ctx.table(m.group("t").strip())
                filters = parse_filters(m.group("f"))
                for key in filters:
                    need_column(table, key)
                ctx.defaults.setdefault(table.name, {}).update(filters)
            except BuildError as exc:
                errors.append(f"{line.strip()}: {exc}")
            continue
        out.append(line)
    return "\n".join(out)


def fill(ctx: BuildContext, draft: str) -> tuple[str, list[str]]:
    """Resolve every placeholder. Returns (text, errors)."""
    errors: list[str] = []
    text = COMMENT_RE.sub("", draft)
    text = COMMENT_RE.sub("", _expand_includes(ctx, text, errors))
    text = _collect_defaults(ctx, text, errors)
    keys = cited_keys(text)

    def repl(m: re.Match[str]) -> str:
        ph = m.group(0)
        body = m.group("body").strip()
        kind = body.split(":", 1)[0]
        try:
            if body == "refs":
                return h_refs(ctx, ph, keys)
            if kind == "fig":
                return h_fig(ctx, ph, body)
            if kind not in HANDLERS:
                raise BuildError(f"unknown placeholder kind {kind!r}")
            return HANDLERS[kind](ctx, ph, body.split(":")[1:])
        except BuildError as exc:
            errors.append(f"{ph}: {exc}")
            return ph

    text = PLACEHOLDER_RE.sub(repl, text)
    text = render_citations(ctx, text, errors)
    for line_no, line in enumerate(text.split("\n"), start=1):
        if ("{{" in line or "}}" in line) and not any(e.split(": ", 1)[0] in line for e in errors):
            errors.append(f"line {line_no}: unresolved or malformed placeholder: {line.strip()[:120]}")
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, errors


def write_numbers_used(used: Sequence[UsedNumber], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(NUMBERS_USED_COLUMNS)
        for u in used:
            w.writerow([u.placeholder, u.kind, u.value, u.source_file, u.source_row, u.column])


# ---------------------------------------------------------------------------------------------
# Markdown -> docx (headings, paragraphs, lists, tables, figures, code blocks)
# ---------------------------------------------------------------------------------------------

_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*$")


_NEG_NUMBER_RE = re.compile(r"(?<![\w.\-])-(?=\d)")
_CI_SPACE_RE = re.compile(r"(\[[−+]?\d[\d.]*%?),\s+(?=[−+]?\d)")


def typeset_numbers(text: str) -> str:
    """Word output only: a true minus sign for negative numbers (no line break after it) and a
    non-breaking space inside "[lo, hi]" so a CI never splits across lines."""
    text = _NEG_NUMBER_RE.sub("−", text)
    return _CI_SPACE_RE.sub("\\1,\u00a0", text)


def _add_inline(paragraph: Any, text: str) -> None:
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(typeset_numbers(part[2:-2])).bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            _set_font(run.font, MONO_FONT, None)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(typeset_numbers(part[1:-1])).italic = True
        else:
            paragraph.add_run(typeset_numbers(part.replace("\\|", "|")))


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


# Word layout: an academic single-column style (serif body, black headings, booktabs tables).
BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_PT = 11.0
TABLE_PT = 8.0
CAPTION_PT = 9.5
TEXT_WIDTH_IN = 6.5
MAX_FIG_HEIGHT_IN = 8.4
# Column headers shown in Word tables (paper.md keeps the raw table column names).
HEADER_LABELS: dict[str, str] = {
    "hypothesis": "Hyp.", "subject": "Model", "comparison": "Comparison", "metric": "Metric",
    "predicted_sign": "Pred.", "estimate": "Estimate", "ci": "95% CI", "n_items": "Items",
    "n_cells": "Cells", "n_judged": "Judged", "verdict": "Verdict", "holm_p": "Holm p",
    "holm_verdict": "Holm verdict", "paraphrase_idx": "Paraphrase", "condition": "Condition",
    "sample": "Sample", "tier": "Tier", "arm": "Arm", "contrast": "Contrast",
    "n_qualifying_cells": "Qualifying cells", "restriction": "Restriction",
}
_CAPTION_RE = re.compile(r"^\*{0,2}(Table|Tables|Figure)\s+[A-Z]?\d")
_PAGE_BREAK_BEFORE = re.compile(r"^(Limitations and Dual-Use|Appendix [ABC]\b)")


def _set_font(style_or_run_font: Any, name: str, size: float | None = None) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    style_or_run_font.name = name
    if size is not None:
        style_or_run_font.size = Pt(size)
    rpr = style_or_run_font.element.rPr if hasattr(style_or_run_font, "element") else None
    if rpr is not None:
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is not None:
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                fonts.set(qn(attr), name)


def _style_document(doc: Any) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.left_margin = section.right_margin = Inches(1)
        section.top_margin = section.bottom_margin = Inches(1)
        # centred page number in the footer
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("Page ").font.size = Pt(9)
        run = p.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(9)

    styles = doc.styles
    normal = styles["Normal"]
    _set_font(normal.font, BODY_FONT, BODY_PT)
    normal.element.get_or_add_rPr()
    rfonts = normal.element.rPr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        normal.element.rPr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), BODY_FONT)
    pf = normal.paragraph_format
    pf.space_before, pf.space_after = Pt(0), Pt(5)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size, before, after, italic in (("Title", 17, 0, 6, False), ("Heading 1", 13, 14, 5, False),
                                              ("Heading 2", 11.5, 10, 3, False), ("Heading 3", 11, 8, 2, True)):
        st = styles[name]
        _set_font(st.font, BODY_FONT, size)
        st.font.bold = True
        st.font.italic = italic
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(after)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if name == "Title" else WD_ALIGN_PARAGRAPH.LEFT
        ppr = st.element.get_or_add_pPr()
        for border in ppr.findall(qn("w:pBdr")):  # the default Title style has a rule under it
            ppr.remove(border)
        rpr = st.element.get_or_add_rPr()
        for el in rpr.findall(qn("w:rFonts")):
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                el.set(qn(attr), BODY_FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if el.get(qn(attr)) is not None:
                    del el.attrib[qn(attr)]
    for name in ("List Bullet", "List Number"):
        styles[name].paragraph_format.space_after = Pt(2)
        styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _breakable(text: str) -> str:
    """Let long identifiers (ablation_T3a_minus_T3) wrap at underscores instead of mid-number."""
    return text.replace("_", "_\u200b")


def _format_table(table: Any, header: Sequence[str], body: Sequence[Sequence[str]]) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    tbl = table._tbl
    tblpr = tbl.tblPr
    width = OxmlElement("w:tblW")
    width.set(qn("w:w"), "5000")
    width.set(qn("w:type"), "pct")
    for old in tblpr.findall(qn("w:tblW")):
        tblpr.remove(old)
    tblpr.append(width)
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        if val == "single":
            el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
        borders.append(el)
    tblpr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge, w in (("top", "15"), ("bottom", "15"), ("left", "60"), ("right", "60")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), w)
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    tblpr.append(margins)
    # Column widths proportional to content (capped), so wide text columns get the room.
    ncols = len(header)
    lengths = []
    for c in range(ncols):
        texts = [HEADER_LABELS.get(header[c], header[c])] + [row[c] if c < len(row) else "" for row in body]
        lengths.append(min(max(max(len(t) for t in texts), 4), 55))
    total = sum(lengths)
    table.autofit = False
    for c in range(ncols):
        w = Inches(TEXT_WIDTH_IN * lengths[c] / total)
        table.columns[c].width = w
        for cell in table.columns[c].cells:
            cell.width = w

    for r, cells in enumerate([header, *body]):
        row = table.rows[r]
        if r == 0:
            trpr = row._tr.get_or_add_trPr()
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            trpr.append(repeat)
            cant_split = OxmlElement("w:cantSplit")
            trpr.append(cant_split)
        for c in range(len(header)):
            cell = row.cells[c]
            text = cells[c] if c < len(cells) else ""
            if r == 0:
                text = HEADER_LABELS.get(text, text.replace("_", " "))
            par = cell.paragraphs[0]
            # non-breaking hyphen inside identifiers (Qwen2.5-7B-Instruct) so they never wrap at "-"
            _add_inline(par, _breakable(re.sub(r"(?<=\w)-(?=\w)", "\u2011", text)))
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in par.runs:
                run.font.size = Pt(TABLE_PT)
                if r == 0:
                    run.bold = True
            tcpr = cell._tc.get_or_add_tcPr()
            if r == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:val"), "clear")
                shade.set(qn("w:color"), "auto")
                shade.set(qn("w:fill"), "EDEDED")
                tcpr.append(shade)
                tcb = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "6")
                bottom.set(qn("w:color"), "000000")
                tcb.append(bottom)
                tcpr.append(tcb)


def markdown_to_docx(md: str, out: Path, base_dir: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.image.image import Image as DocxImage
    from docx.shared import Inches, Pt

    doc = Document()
    _style_document(doc)
    lines = md.split("\n")
    i = 0
    para: list[str] = []
    state: dict[str, Any] = {"front": False, "seen_h2": False, "section": ""}

    def flush() -> None:
        if not para:
            return
        text = " ".join(s.strip() for s in para)
        para.clear()
        p = doc.add_paragraph()
        if state["front"] and not state["seen_h2"]:
            if text.startswith("**Abstract"):
                p.paragraph_format.left_indent = p.paragraph_format.right_indent = Inches(0.4)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(8)
                _add_inline(p, text)
                for run in p.runs:
                    run.font.size = Pt(10)
                return
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER  # author and affiliation lines
            p.paragraph_format.space_after = Pt(1)
            _add_inline(p, text)
            return
        if _CAPTION_RE.match(text):
            p.paragraph_format.keep_with_next = text.lstrip("*").startswith("Table")
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _add_inline(p, text)
            for run in p.runs:
                run.font.size = Pt(CAPTION_PT)
            return
        _add_inline(p, text)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        fence = re.match(r"^(`{3,})", stripped)
        if fence:
            flush()
            ticks = fence.group(1)
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(ticks):
                block.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Inches(0.2)
            for n, code_line in enumerate(block):
                run = p.add_run(code_line)
                _set_font(run.font, MONO_FONT, 8)
                if n < len(block) - 1:
                    run.add_break()
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush()
            level = min(len(heading.group(1)) - 1, 9)
            title = heading.group(2).strip()
            if level == 0:
                state["front"] = True
            else:
                state["seen_h2"] = True
            if level == 1:
                state["section"] = title
            if level == 1 and _PAGE_BREAK_BEFORE.match(title):
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            doc.add_heading(title, level=level)
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            flush()
            header = _split_row(stripped)
            body: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(_split_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1 + len(body), cols=len(header))
            _format_table(table, header, body)
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_after = Pt(2)
            continue
        image = _IMAGE_RE.match(stripped)
        if image:
            flush()
            src = Path(image.group("src"))
            path = src if src.is_absolute() else base_dir / src
            img = DocxImage.from_file(str(path))
            width_in = min(TEXT_WIDTH_IN, MAX_FIG_HEIGHT_IN * img.px_width / img.px_height)
            doc.add_picture(str(path), width=Inches(width_in))
            pic_par = doc.paragraphs[-1]
            pic_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pic_par.paragraph_format.keep_with_next = True
            pic_par.paragraph_format.space_before = Pt(6)
            i += 1
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        number = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if bullet or number:
            flush()
            m = bullet or number
            assert m is not None
            if bullet and str(state["section"]).startswith("References"):
                ref = doc.add_paragraph()  # hanging indent, no bullet
                ref.alignment = WD_ALIGN_PARAGRAPH.LEFT
                ref.paragraph_format.left_indent = Inches(0.25)
                ref.paragraph_format.first_line_indent = Inches(-0.25)
                ref.paragraph_format.space_after = Pt(3)
                _add_inline(ref, m.group(1))
                for run in ref.runs:
                    run.font.size = Pt(9.5)
            else:
                _add_inline(doc.add_paragraph(style="List Bullet" if bullet else "List Number"), m.group(1))
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            _add_inline(doc.add_paragraph(style="Quote"), stripped.lstrip(">").strip())
            i += 1
            continue
        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            flush()
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    doc.save(str(out))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def build(ctx: BuildContext, draft_path: Path, *, final: bool = False, docx: bool = True) -> list[str]:
    """Fill the draft and write outputs. Returns errors (outputs written only if there are none)."""
    ctx.final = final
    text, errors = fill(ctx, draft_path.read_text(encoding="utf-8"))
    if final:
        for line_no, line in enumerate(text.split("\n"), start=1):
            for m in FINAL_MARKERS.finditer(line):
                errors.append(f"line {line_no}: unfinished marker {m.group(0)} (--final)")
    else:
        n_markers = len(FINAL_MARKERS.findall(text))
        if n_markers:
            ctx.warnings.append(f"{n_markers} editorial markers remain ([CITE:...], [RESULT PROSE ...], ...)")
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:  # never leave stale outputs from an earlier build
        (ctx.out_dir / name).unlink(missing_ok=True)
    if errors:
        return errors
    (ctx.out_dir / "paper.md").write_text(text, encoding="utf-8")
    write_numbers_used(ctx.used, ctx.out_dir / "numbers_used.csv")
    if docx:
        markdown_to_docx(text, ctx.out_dir / "paper.docx", ctx.out_dir)
    return []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    ap.add_argument("--draft", type=Path, default=REPO_ROOT / "paper" / "draft.md")
    ap.add_argument("--tables", type=Path, default=None, help="default: paths.tables in config.yaml")
    ap.add_argument("--figures", type=Path, default=None, help="default: paths.figures in config.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper")
    ap.add_argument("--refs", type=Path, default=None, help="default: references.md next to the draft")
    ap.add_argument("--final", action="store_true",
                    help="fail on citations not CONFIRMED in references.md and on remaining editorial markers")
    ap.add_argument("--no-docx", action="store_true")
    ap.add_argument("--allow-dry-run", action="store_true", help="accept rows whose data_source is dry-run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ctx = BuildContext(
        tables_dir=args.tables or _resolve(args.config, cfg["paths"]["tables"]),
        figures_dir=args.figures or _resolve(args.config, cfg["paths"]["figures"]),
        config_path=args.config, draft_dir=args.draft.resolve().parent, out_dir=args.out_dir,
        allow_dry_run=args.allow_dry_run, refs_path=args.refs,
    )
    errors = build(ctx, args.draft, final=args.final, docx=not args.no_docx)
    for w in ctx.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if errors:
        print(f"BUILD FAILED: {len(errors)} unresolved placeholder(s); no outputs written.", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"wrote {ctx.out_dir / 'paper.md'}, {len(ctx.used)} numbers -> {ctx.out_dir / 'numbers_used.csv'}"
          + ("" if args.no_docx else f", {ctx.out_dir / 'paper.docx'}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
