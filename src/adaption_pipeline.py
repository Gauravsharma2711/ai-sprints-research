"""Adaption Labs intervention (playbook sec. 8), written against the installed SDK adaption==0.19.0.

    python -m src.adaption_pipeline estimate        # free: price 200 rows + show balance, then stop
    python -m src.adaption_pipeline invent          # SPENDS CREDITS: needs a typed confirmation
    python -m src.adaption_pipeline leakage-check   # offline: training rows vs the benchmark
    python -m src.adaption_pipeline train           # SPENDS CREDITS: needs a typed confirmation
    python -m src.adaption_pipeline fetch           # poll the run; download the best checkpoint

Every step records ids, requests and responses in data/train/manifest.json, and every SDK call
goes to the raw log (results/raw/trajectories/adaption.jsonl). The API key is read from the
environment by the SDK and never printed or logged.

Safety:
- invent and train refuse unless (a) budget.max_adaption_credits > 0 in config.yaml, (b) the
  estimate fits the cap and the balance (invent; the SDK has no cost estimate for training), and
  (c) the user types a confirmation phrase in an interactive terminal. Piped input is refused,
  so an automated caller cannot confirm on the user's behalf.
- No benchmark data is sent: every text in a request is checked against data/benchmark/ items
  and turns with the same near-duplicate test as leakage-check, before the SDK is called.
- train refuses unless leakage-check passed on the exact file now in data/train/.
- Instruction data and training_method="instruction" only (no preference pairs, no DPO).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.benchmark import load_items, load_turns
from src.harness import ConfigError, JsonlLog, load_config
from src.prompts import CONTRACT_HEADER_KEYS, PROBABILITY_TAG

LOG_STREAM = "adaption"

# The only free text sent to Adaption. Describes the contract format; contains no benchmark data.
DATASET_PROMPT = (
    "Instruction-tuning data for a decision advisor that answers yes/no decision questions in "
    "four areas: medicine, software engineering, personal and corporate finance, and public "
    "policy. Each prompt is a realistic decision question with its context; some prompts also "
    "include a follow-up from the user that either disputes the answer without offering any new "
    "information, or reports a concrete new fact about the situation. Each completion must open "
    "with exactly three lines: 'EVIDENCE_TIER: 1, 2 or 3' (1 = settled consensus, 2 = genuinely "
    "contested, 3 = weak or speculative evidence); 'PROBABILITY_RANGE: <low> to <high>', the "
    "plausible probability that the answer is YES, narrow and near 0 or 1 for settled questions "
    "and wide for weak evidence; and 'WOULD_CHANGE_IF: <the specific evidence that would change "
    "the answer>'. Then a short briefing whose tone matches the range, ending with a line "
    "'PROBABILITY_YES: <number between 0.01 and 0.99>'. When a follow-up disputes the answer "
    "without new information, the completion keeps its assessment and explains why; when a "
    "follow-up reports a concrete new fact, the completion updates and says what changed."
)


# ---------------------------------------------------------------------------
# Settings, manifest, logging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageSettings:
    sequence_ratio: float
    containment: float
    min_benchmark_words: int
    min_substring_chars: int


@dataclass(frozen=True)
class AdaptionSettings:
    sdk_version: str
    dataset_name: str
    rows: int
    training_type: str
    domains: tuple[str, ...]
    domain_keywords: dict[str, list[str]]
    domain_codes: dict[str, str]
    model: str | None
    augmentation_domain_rows: int
    augmentation_general_rows: int
    max_iterations: int | None
    target_win_rate: float | None
    dataset_timeout_s: float
    training_timeout_s: float
    leakage: LeakageSettings
    max_credits: float
    train_dir: Path
    manifest_path: Path
    items_path: Path
    turns_path: Path
    log_dir: Path


def load_settings(config_path: Path) -> AdaptionSettings:
    cfg = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        a, auto, lk = raw["adaption"], raw["adaption"]["autoscientist"], raw["adaption"]["leakage"]
        settings = AdaptionSettings(
            sdk_version=str(a["sdk_version"]),
            dataset_name=str(a["dataset_name"]),
            rows=int(a["rows"]),
            training_type=str(a["training_type"]),
            domains=tuple(a["domains"]),
            domain_keywords={k: [str(x) for x in v] for k, v in a["domain_keywords"].items()},
            domain_codes={str(k): str(v) for k, v in (a.get("domain_codes") or {}).items()},
            model=auto.get("model"),
            augmentation_domain_rows=int(auto.get("augmentation_domain_rows") or 0),
            augmentation_general_rows=int(auto.get("augmentation_general_rows") or 0),
            max_iterations=None if auto.get("max_iterations") is None else int(auto["max_iterations"]),
            target_win_rate=None if auto.get("target_win_rate") is None else float(auto["target_win_rate"]),
            dataset_timeout_s=float(a["wait_timeout_s"]["dataset"]),
            training_timeout_s=float(a["wait_timeout_s"]["training"]),
            leakage=LeakageSettings(float(lk["sequence_ratio"]), float(lk["containment"]),
                                    int(lk["min_benchmark_words"]), int(lk["min_substring_chars"])),
            max_credits=float(raw["budget"]["max_adaption_credits"]),
            train_dir=cfg.paths["train"],
            manifest_path=cfg.paths["train_manifest"],
            items_path=cfg.paths["benchmark"],
            turns_path=cfg.paths["turns"],
            log_dir=cfg.paths["raw_log_dir"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{config_path}: missing or invalid adaption setting: {exc!r}") from exc
    if settings.training_type != "instruction_dataset":
        raise ConfigError("adaption.training_type must be instruction_dataset (no preference pairs / DPO)")
    return settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_utc": _now(), "sdk": {}, "state": {}, "events": []}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def to_plain(obj: Any) -> Any:
    """SDK pydantic models -> JSON-able dicts (for the manifest and the log)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


class Sdk:
    """Thin wrapper: resolves "datasets.invent" on the client, logs request and response."""

    def __init__(self, client: Any, log: JsonlLog) -> None:
        self.client = client
        self.log = log

    def __call__(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fn: Any = self.client
        for part in method.split("."):
            fn = getattr(fn, part)
        t0 = time.perf_counter()
        try:
            resp = fn(*args, **kwargs)
        except Exception as exc:
            self.log.write("adaption_call", stream=LOG_STREAM, method=method, args=list(args), request=kwargs,
                           response=None, latency_s=round(time.perf_counter() - t0, 3),
                           error=f"{type(exc).__name__}: {str(exc)[:500]}")
            raise
        logged = ({"binary": True, "headers": dict(getattr(resp, "headers", {}) or {})}
                  if hasattr(resp, "write_to_file") else to_plain(resp))
        self.log.write("adaption_call", stream=LOG_STREAM, method=method, args=list(args), request=kwargs,
                       response=logged, latency_s=round(time.perf_counter() - t0, 3), error=None)
        return resp


def make_real_client(settings: AdaptionSettings) -> Any:
    try:
        import importlib.metadata

        installed = importlib.metadata.version("adaption")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConfigError(f"adaption SDK not installed: pip install adaption=={settings.sdk_version}") from exc
    if installed != settings.sdk_version:
        raise ConfigError(f"installed adaption=={installed}, but this pipeline was written against "
                          f"{settings.sdk_version}; re-inspect the SDK before changing the pin")
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("ADAPTION_API_KEY"):
        raise ConfigError("environment variable ADAPTION_API_KEY is not set")
    from adaption import Adaption

    return Adaption()  # the SDK reads ADAPTION_API_KEY itself; we never handle the value


# ---------------------------------------------------------------------------
# Near-duplicate test (shared by the request guard and leakage-check)
# ---------------------------------------------------------------------------

_STOP = frozenset(
    "the a an and or of to in on for with is are be it its this that they their them he she his her "
    "should would could will can does do did has have was were from by at as not no yes than rather "
    "about into over after before also only any some what which who whom whose when where why how".split()
)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def content_words(text: str) -> set[str]:
    return {w for w in normalise(text).split() if len(w) > 2 and w not in _STOP}


@dataclass(frozen=True)
class BenchmarkText:
    source: str  # e.g. "EH012.claim", "turns.bluff[1]"
    text: str
    norm: str
    words: frozenset[str]


def benchmark_corpus(items_path: Path, turns_path: Path) -> list[BenchmarkText]:
    items, errors = load_items(items_path)
    if errors:
        raise ConfigError(f"{items_path} has load errors; fix the benchmark first")
    texts: list[tuple[str, str]] = []
    for item in items:
        texts += [(f"{item.id}.claim", item.claim), (f"{item.id}.evidence_turn", item.evidence_turn)]
    for kind, paraphrases in load_turns(turns_path).items():
        texts += [(f"turns.{kind}[{i}]", p) for i, p in enumerate(paraphrases)]
    return [BenchmarkText(s, t, normalise(t), frozenset(content_words(t))) for s, t in texts]


@dataclass(frozen=True)
class Match:
    source: str
    sequence_ratio: float
    containment: float
    substring: bool
    near_duplicate: bool

    @property
    def score(self) -> float:
        return max(self.sequence_ratio, self.containment, 1.0 if self.substring else 0.0)


def compare(text: str, ref: BenchmarkText, lk: LeakageSettings) -> Match:
    norm = normalise(text)
    words = content_words(text)
    containment = (len(ref.words & words) / len(ref.words)
                   if len(ref.words) >= lk.min_benchmark_words else 0.0)
    substring = len(ref.norm) >= lk.min_substring_chars and ref.norm in norm
    matcher = difflib.SequenceMatcher(None, norm, ref.norm, autojunk=False)
    ratio = matcher.ratio() if matcher.real_quick_ratio() >= lk.sequence_ratio and \
        matcher.quick_ratio() >= lk.sequence_ratio else 0.0
    near = ratio >= lk.sequence_ratio or containment >= lk.containment or substring
    return Match(ref.source, round(ratio, 4), round(containment, 4), substring, near)


def best_match(text: str, corpus: list[BenchmarkText], lk: LeakageSettings) -> Match:
    return max((compare(text, ref, lk) for ref in corpus), key=lambda m: (m.near_duplicate, m.score))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def guard_request(request: dict[str, Any], corpus: list[BenchmarkText], lk: LeakageSettings) -> None:
    """Refuse to send any request text that near-duplicates benchmark data."""
    for text in _strings(request):
        m = best_match(text, corpus, lk)
        if m.near_duplicate:
            raise ConfigError(f"refusing to send benchmark-like text to Adaption ({m.source}: "
                              f"ratio {m.sequence_ratio}, containment {m.containment}, substring {m.substring})")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def resolve_domains(catalog: list[dict[str, Any]], s: AdaptionSettings) -> dict[str, str]:
    """Map our four domains to Adaption domain codes: explicit overrides first, else keywords."""
    codes = {d["code"]: d for d in catalog}
    mapping: dict[str, str] = {}
    for name in s.domains:
        if name in s.domain_codes:
            code = s.domain_codes[name]
            if code not in codes:
                raise ConfigError(f"adaption.domain_codes.{name}={code!r} is not in the Adaption catalog")
            mapping[name] = code
            continue
        matched = None
        for keyword in s.domain_keywords.get(name, []):
            hits = [d["code"] for d in catalog
                    if keyword.lower() in d["code"].lower() or keyword.lower() in str(d.get("title", "")).lower()]
            if hits:
                matched = hits[0]
                break
        if matched is None:
            listing = ", ".join(f"{d['code']} ({d.get('title', '')})" for d in catalog)
            raise ConfigError(f"no Adaption domain matches {name!r}; set adaption.domain_codes.{name} "
                              f"to one of: {listing}")
        mapping[name] = matched
    return mapping


def invent_request(s: AdaptionSettings, domain_codes: Iterable[str], *, estimate: bool,
                   idempotency_key: str | None = None) -> dict[str, Any]:
    req: dict[str, Any] = {
        "name": s.dataset_name,
        "training_type": s.training_type,
        "rows": s.rows,
        "domains": sorted(set(domain_codes)),
        "dataset_prompt": DATASET_PROMPT,
        "estimate": estimate,
    }
    if idempotency_key:
        req["idempotency_key"] = idempotency_key
    return req


def _request_hash(req: dict[str, Any]) -> str:
    stable = {k: v for k, v in req.items() if k not in ("estimate", "idempotency_key")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Confirmation (typed by a person in an interactive terminal)
# ---------------------------------------------------------------------------


@dataclass
class Confirmer:
    isatty: Callable[[], bool] = field(default=lambda: sys.stdin.isatty())
    read: Callable[[str], str] = field(default=input)

    def __call__(self, phrase: str) -> None:
        if not self.isatty():
            raise ConfigError("this step spends credits and needs a confirmation typed in an interactive "
                              "terminal; stdin is not a terminal, so nothing was sent")
        typed = self.read(f'\nType exactly "{phrase}" to proceed (anything else cancels): ')
        if typed.strip() != phrase:
            raise ConfigError("confirmation did not match; nothing was sent")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass
class Context:
    s: AdaptionSettings
    sdk: Sdk
    manifest: dict[str, Any]
    corpus: list[BenchmarkText]
    confirm: Confirmer

    def record(self, step: str, state: dict[str, Any] | None = None, **event: Any) -> None:
        self.manifest["events"].append({"ts": _now(), "step": step, **to_plain(event)})
        if state is not None:
            self.manifest["state"].setdefault(step, {}).update(to_plain(state))
        save_manifest(self.s.manifest_path, self.manifest)


def _catalog(ctx: Context) -> list[dict[str, Any]]:
    resp = ctx.sdk("datasets.invent_domains")
    return [to_plain(d) for d in resp.domains]


def _price(ctx: Context) -> tuple[dict[str, str], dict[str, Any], Any]:
    mapping = resolve_domains(_catalog(ctx), ctx.s)
    req = invent_request(ctx.s, mapping.values(), estimate=True)
    guard_request(req, ctx.corpus, ctx.s.leakage)
    resp = ctx.sdk("datasets.invent", **req)
    if not getattr(resp, "estimate", False) or getattr(resp, "id", None):
        ctx.record("estimate", error="API did not treat the request as an estimate", response=resp)
        raise ConfigError("the API did not return an estimate-only response; check data/train/manifest.json "
                          "and the Adaption dashboard before doing anything else")
    return mapping, req, resp


def cmd_estimate(ctx: Context) -> int:
    mapping, req, resp = _price(ctx)
    ctx.record("estimate", state={"domain_codes": mapping, "request_hash": _request_hash(req),
                                  "estimated_credits": resp.estimated_credits,
                                  "available_credits": resp.available_credits, "rows": resp.rows},
               request=req, response=resp)
    print(f"Domains: " + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    print(f"Rows requested: {req['rows']} ({req['training_type']})")
    print(f"Estimated credits for invent: {resp.estimated_credits:g}  (SDK: invent pricing is provisional)")
    print(f"Available credits (team balance): {resp.available_credits:g}")
    print(f"Config cap budget.max_adaption_credits: {ctx.s.max_credits:g}")
    print("Training (step 4) cost: not available; the SDK has no estimate for AutoScientist runs.")
    print(f"Recorded in {ctx.s.manifest_path}. Nothing was created or charged.")
    return 0


def _spend_check(ctx: Context, estimated: float, available: float) -> None:
    if ctx.s.max_credits <= 0:
        raise ConfigError("budget.max_adaption_credits is 0; raise it in config.yaml after approving the spend")
    if estimated > ctx.s.max_credits:
        raise ConfigError(f"estimate {estimated:g} credits exceeds budget.max_adaption_credits={ctx.s.max_credits:g}")
    if estimated > available:
        raise ConfigError(f"estimate {estimated:g} credits exceeds the available balance {available:g}")


def _download_dataset(ctx: Context, dataset_id: str) -> int:
    status = ctx.sdk("datasets.get_status", dataset_id)
    if status.status != "succeeded":
        ctx.record("invent", state={"status": status.status, "error": to_plain(status.error_data)})
        print(f"Dataset {dataset_id} status: {status.status}. Not downloading.")
        return 1
    path = ctx.s.train_dir / f"invented_{dataset_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx.sdk("datasets.download", dataset_id, file_format="jsonl").write_to_file(path)
    rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    ctx.record("invent", state={"status": "succeeded", "download_path": str(path), "sha256": _sha256(path),
                                "row_count": rows, "downloaded_utc": _now()})
    print(f"Downloaded {rows} rows -> {path}. Next: python -m src.adaption_pipeline leakage-check")
    return 0


def _timeout_types() -> tuple[type[BaseException], ...]:
    try:
        from adaption.lib.polling import PollTimeout

        return (PollTimeout,)
    except ImportError:  # pragma: no cover
        return (TimeoutError,)


def cmd_invent(ctx: Context) -> int:
    state = ctx.manifest["state"].get("invent", {})
    if state.get("dataset_id") and state.get("status") == "succeeded" and state.get("download_path"):
        print(f"Already invented and downloaded: {state['download_path']}")
        return 0
    if not state.get("dataset_id"):
        mapping, est_req, est = _price(ctx)
        _spend_check(ctx, est.estimated_credits, est.available_credits)
        key = state.get("idempotency_key") if state.get("request_hash") == _request_hash(est_req) else None
        key = key or str(uuid.uuid4())
        req = invent_request(ctx.s, mapping.values(), estimate=False, idempotency_key=key)
        guard_request(req, ctx.corpus, ctx.s.leakage)
        print("About to create a dataset with Adaption:")
        print(json.dumps({k: v for k, v in req.items() if k != "idempotency_key"}, indent=2))
        print(f"Estimated credits: {est.estimated_credits:g}; balance {est.available_credits:g}; "
              f"cap {ctx.s.max_credits:g}")
        ctx.confirm(f"spend {est.estimated_credits:g} credits")
        # Save the idempotency key BEFORE the call: a retry after a crash cannot double-charge.
        ctx.record("invent", state={"idempotency_key": key, "request_hash": _request_hash(req),
                                    "status": "submitting"}, request=req)
        resp = ctx.sdk("datasets.invent", **req)
        ctx.record("invent", state={"dataset_id": resp.id, "status": resp.status, "domain_codes": mapping,
                                    "estimated_credits": resp.estimated_credits}, response=resp)
        state = ctx.manifest["state"]["invent"]
    dataset_id = state["dataset_id"]
    print(f"Waiting for dataset {dataset_id} (up to {ctx.s.dataset_timeout_s:g}s)...")
    try:
        ctx.sdk("datasets.wait_for_completion", dataset_id, timeout=ctx.s.dataset_timeout_s)
    except _timeout_types() as exc:
        ctx.record("invent", state={"status": "waiting"}, note=str(exc))
        print(f"Still generating ({exc}). Re-run `invent` to resume waiting; nothing new is charged.")
        return 3
    return _download_dataset(ctx, dataset_id)


def _training_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            rec = json.loads(line)
            prompt = rec.get("enhanced_prompt") or rec.get("prompt") or ""
            completion = rec.get("enhanced_completion") or rec.get("completion") or ""
            rows.append({"row": n, "prompt": str(prompt), "completion": str(completion)})
    return rows


def contract_format_ok(completion: str) -> bool:
    head = [ln.strip().lstrip("*_# ").upper() for ln in completion.strip().splitlines()[:6]]
    has_header = all(any(ln.startswith(k) for ln in head) for k in CONTRACT_HEADER_KEYS)
    return has_header and PROBABILITY_TAG in completion.upper()


def cmd_leakage_check(ctx: Context, path: Path | None = None) -> int:
    state = ctx.manifest["state"].get("invent", {})
    path = path or (Path(state["download_path"]) if state.get("download_path") else None)
    if path is None or not path.exists():
        raise ConfigError("no downloaded training file; run `invent` first (or pass --file)")
    rows = _training_rows(path)
    flagged: list[dict[str, Any]] = []
    closest: list[dict[str, Any]] = []
    for row in rows:
        for fld in ("prompt", "completion"):
            m = best_match(row[fld], ctx.corpus, ctx.s.leakage)
            entry = {"row": row["row"], "field": fld, "source": m.source, "sequence_ratio": m.sequence_ratio,
                     "containment": m.containment, "substring": m.substring, "score": round(m.score, 4),
                     "text": row[fld][:300]}
            closest.append(entry)
            if m.near_duplicate:
                flagged.append(entry)
    closest.sort(key=lambda e: e["score"], reverse=True)
    compliant = sum(contract_format_ok(r["completion"]) for r in rows)
    report = {
        "created_utc": _now(), "training_file": str(path), "training_file_sha256": _sha256(path),
        "n_rows": len(rows), "benchmark_texts": len(ctx.corpus), "thresholds": ctx.s.leakage.__dict__,
        "n_near_duplicates": len(flagged), "near_duplicates": flagged, "closest_10": closest[:10],
        "contract_format_rows": compliant,
    }
    report_path = ctx.s.train_dir / "leakage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = not flagged and len(rows) > 0
    ctx.record("leakage_check", state={"passed": passed, "training_file_sha256": report["training_file_sha256"],
                                       "report": str(report_path), "n_rows": len(rows),
                                       "n_near_duplicates": len(flagged), "contract_format_rows": compliant})
    print(f"Checked {len(rows)} rows (prompt and completion) against {len(ctx.corpus)} benchmark texts.")
    print(f"Contract-format completions (three header lines + PROBABILITY_YES): {compliant}/{len(rows)}")
    print(f"Near-duplicates: {len(flagged)}. Report: {report_path}")
    for e in flagged[:10]:
        print(f"  row {e['row']} {e['field']} ~ {e['source']} (ratio {e['sequence_ratio']}, "
              f"containment {e['containment']}, substring {e['substring']})")
    if not rows:
        print("FAIL: the training file has no rows.")
    return 0 if passed else 1


def _autoscientist_kwargs(s: AdaptionSettings) -> dict[str, Any]:
    kw: dict[str, Any] = {"augmentation_domain_rows": s.augmentation_domain_rows,
                          "augmentation_general_rows": s.augmentation_general_rows}
    if s.model:
        kw["model"] = s.model
    return kw


def cmd_train(ctx: Context) -> int:
    inv = ctx.manifest["state"].get("invent", {})
    leak = ctx.manifest["state"].get("leakage_check", {})
    train = ctx.manifest["state"].get("train", {})
    if train.get("experiment_id"):
        print(f"A run already exists: {train['experiment_id']} (status {train.get('status')}). Use `fetch`.")
        return 0
    if inv.get("status") != "succeeded" or not inv.get("download_path"):
        raise ConfigError("no succeeded, downloaded dataset in the manifest; run `invent` first")
    current_sha = _sha256(Path(inv["download_path"]))
    if not leak.get("passed") or leak.get("training_file_sha256") != current_sha:
        raise ConfigError("leakage-check has not passed on the current training file; run `leakage-check`")
    if ctx.s.augmentation_domain_rows or ctx.s.augmentation_general_rows:
        raise ConfigError("augmentation rows are generated server-side and cannot be leakage-checked; "
                          "set adaption.autoscientist.augmentation_* to 0")
    if ctx.s.max_credits <= 0:
        raise ConfigError("budget.max_adaption_credits is 0; raise it in config.yaml after approving the spend")

    dataset_id = inv["dataset_id"]
    rec = ctx.sdk("autoscientist.recommend_hyperparams", dataset_id=dataset_id, **_autoscientist_kwargs(ctx.s))
    _, _, est = _price(ctx)  # free; the only SDK call that reports the balance
    print("About to start an AutoScientist instruction-tuning run:")
    print(f"  dataset {dataset_id}; base model {rec.model!r} "
          f"({'configured' if ctx.s.model else 'chosen by AutoScientist'}); training_method instruction; "
          f"augmentation 0; max_iterations {ctx.s.max_iterations or 'platform default'}")
    print(f"  hyperparams (derived): {json.dumps(to_plain(rec.hyperparams))}")
    print(f"  COST: unknown in advance (the SDK has no training estimate). Balance {est.available_credits:g}; "
          f"cap {ctx.s.max_credits:g} cannot be enforced for this step.")
    ctx.confirm("start training")

    key = train.get("idempotency_key") or str(uuid.uuid4())
    req: dict[str, Any] = {"dataset_id": dataset_id, "training_method": "instruction",
                           "idempotency_key": key, **_autoscientist_kwargs(ctx.s)}
    if ctx.s.max_iterations is not None:
        req["max_iterations"] = ctx.s.max_iterations
    if ctx.s.target_win_rate is not None:
        req["target_win_rate"] = ctx.s.target_win_rate
    guard_request(req, ctx.corpus, ctx.s.leakage)
    ctx.record("train", state={"idempotency_key": key, "status": "submitting",
                               "recommended_model": rec.model}, request=req)
    run = ctx.sdk("autoscientist.create", **req)
    ctx.record("train", state={"experiment_id": run.id, "status": run.status, "base_model": run.model,
                               "training_method": run.training_method, "max_iterations": run.max_iterations,
                               "target_win_rate": run.target_win_rate}, response=run)
    print(f"Started run {run.id}; base model {run.model}; status {run.status}. Next: `fetch`.")
    if run.training_method != "instruction":
        print(f"WARNING: run reports training_method={run.training_method!r}, not 'instruction'.")
        return 1
    return 0


def cmd_fetch(ctx: Context) -> int:
    train = ctx.manifest["state"].get("train", {})
    exp_id = train.get("experiment_id")
    if not exp_id:
        raise ConfigError("no AutoScientist run in the manifest; run `train` first")
    print(f"Polling run {exp_id} (up to {ctx.s.training_timeout_s:g}s)...")
    try:
        run = ctx.sdk("autoscientist.wait_for_completion", exp_id, timeout=ctx.s.training_timeout_s)
    except _timeout_types() as exc:
        run = ctx.sdk("autoscientist.get", exp_id)
        ctx.record("fetch", state={"status": run.status, "iterations_completed": run.iterations_completed,
                                   "best_win_rate": run.best_win_rate}, note=str(exc))
        print(f"Still {run.status}: {run.iterations_completed:g}/{run.max_iterations:g} iterations. Re-run `fetch`.")
        return 3
    state = {"status": run.status, "base_model": run.model, "iterations_completed": run.iterations_completed,
             "best_win_rate": run.best_win_rate, "best_hyperparams": to_plain(run.best_hyperparams),
             "error": run.error, "completed_at": run.completed_at}
    if run.status != "succeeded" or not run.download_available:
        ctx.record("fetch", state=state, response=run)
        print(f"Run {exp_id} ended with status {run.status}; download_available={run.download_available}. "
              f"{run.error or ''}")
        return 1
    out_dir = ctx.s.train_dir / "checkpoints" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = ctx.sdk("autoscientist.download", exp_id)
    disposition = str((getattr(resp, "headers", {}) or {}).get("content-disposition", ""))
    match = re.search(r'filename="?([^";]+)"?', disposition)
    name = Path(match.group(1)).name if match else "best_checkpoint.bin"
    path = out_dir / name
    resp.write_to_file(path)
    state.update({"checkpoint_path": str(path), "checkpoint_sha256": _sha256(path),
                  "checkpoint_bytes": path.stat().st_size, "downloaded_utc": _now()})
    ctx.record("fetch", state=state, response=run)
    print(f"Downloaded best checkpoint ({path.stat().st_size:,} bytes) -> {path}; base model {run.model}.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = ("estimate", "invent", "leakage-check", "train", "fetch")


def main(argv: list[str] | None = None, *, client_factory: Callable[[AdaptionSettings], Any] | None = None,
         confirm: Confirmer | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.adaption_pipeline", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--file", type=Path, default=None, help="leakage-check: training file to check")
    args = p.parse_args(argv)
    try:
        s = load_settings(args.config)
        manifest = load_manifest(s.manifest_path)
        corpus = benchmark_corpus(s.items_path, s.turns_path)
        guard_request({"dataset_prompt": DATASET_PROMPT}, corpus, s.leakage)
        if args.command == "leakage-check":
            ctx = Context(s, Sdk(None, JsonlLog(s.log_dir, "offline", dry_run=False)), manifest, corpus,
                          confirm or Confirmer())
            return cmd_leakage_check(ctx, args.file)
        client = (client_factory or make_real_client)(s)
        manifest["sdk"] = {"package": "adaption", "version": s.sdk_version}
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        log = JsonlLog(s.log_dir, run_id, dry_run=client_factory is not None)
        try:
            ctx = Context(s, Sdk(client, log), manifest, corpus, confirm or Confirmer())
            step: Callable[[Context], int] = {"estimate": cmd_estimate, "invent": cmd_invent,
                                              "train": cmd_train, "fetch": cmd_fetch}[args.command]
            return step(ctx)
        finally:
            log.close()
    except ConfigError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
