"""Offline tests for src/adaption_pipeline.py with a fake SDK client (no network, no credits)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src import adaption_pipeline as ap
from src.benchmark import load_items
from src.harness import ConfigError

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("adaption")
from adaption.lib.polling import DatasetTimeout, TrainingTimeout  # noqa: E402


class M(SimpleNamespace):
    """Stands in for the SDK's pydantic response models."""

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {k: (v.model_dump() if isinstance(v, M) else [x.model_dump() if isinstance(x, M) else x for x in v]
                    if isinstance(v, list) else v) for k, v in vars(self).items()}


class FakeBinary:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        self.content, self.headers = content, headers or {}

    def write_to_file(self, path: Path) -> None:
        Path(path).write_bytes(self.content)


CLEAN_ROWS = [
    {"enhanced_prompt": "Should a regional hospital adopt barcode scanning for every medication given on its wards?",
     "enhanced_completion": "EVIDENCE_TIER: 1\nPROBABILITY_RANGE: 0.85 to 0.95\nWOULD_CHANGE_IF: an audit "
                            "showing more errors after rollout\n\nYes, broadly.\nPROBABILITY_YES: 0.9"},
    {"enhanced_prompt": "A startup asks whether to adopt property-based testing for its billing code.",
     "enhanced_completion": "Probably worth it for billing logic.\nPROBABILITY_YES: 0.7"},  # no header
]


class FakeClient:
    def __init__(self, *, training_rows: list[dict[str, str]] | None = None, estimate_ok: bool = True,
                 dataset_timeout: bool = False, training_timeout: bool = False,
                 estimated: float = 40.0, available: float = 1250.0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.training_rows = training_rows if training_rows is not None else CLEAN_ROWS
        self.estimate_ok, self.dataset_timeout, self.training_timeout = estimate_ok, dataset_timeout, training_timeout
        self.estimated, self.available = estimated, available
        self.datasets = SimpleNamespace(invent_domains=self.invent_domains, invent=self.invent,
                                        wait_for_completion=self.ds_wait, get_status=self.ds_status,
                                        download=self.ds_download)
        self.autoscientist = SimpleNamespace(recommend_hyperparams=self.recommend, create=self.create,
                                             wait_for_completion=self.run_wait, get=self.run_get,
                                             download=self.run_download)

    def _log(self, _method: str, **kw: Any) -> None:
        self.calls.append((_method, kw))

    def invent_domains(self) -> M:
        self._log("invent_domains")
        # Mirrors codes seen in the live catalog (14 Nov 2026) that config.yaml maps to.
        return M(domains=[M(code=c, title=c.replace("_", " ").capitalize(), subdomains=[])
                          for c in ("code", "governance", "legal", "medical", "personal_finance", "technology")])

    def invent(self, **kw: Any) -> M:
        self._log("invent", **kw)
        common = dict(name=kw["name"], training_type=kw["training_type"], domains=kw["domains"], subdomains=[],
                      rows=kw["rows"], estimated_credits=self.estimated, available_credits=self.available)
        if kw.get("estimate") and self.estimate_ok:
            return M(estimate=True, id=None, status=None, created_at=None, **common)
        return M(estimate=False, id="ds_1", status="running", created_at="2026-11-14T00:00:00Z", **common)

    def ds_wait(self, dataset_id: str, *, timeout: float) -> M:
        self._log("datasets.wait_for_completion", dataset_id=dataset_id)
        if self.dataset_timeout:
            raise DatasetTimeout(dataset_id, timeout, "running")
        return M(dataset_id=dataset_id, status="succeeded")

    def ds_status(self, dataset_id: str) -> M:
        return M(dataset_id=dataset_id, status="succeeded", error_data=None, row_count=len(self.training_rows))

    def ds_download(self, dataset_id: str, *, file_format: str) -> FakeBinary:
        self._log("datasets.download", file_format=file_format)
        return FakeBinary("\n".join(json.dumps(r) for r in self.training_rows).encode())

    def recommend(self, **kw: Any) -> M:
        self._log("recommend_hyperparams", **kw)
        return M(model="base-8b-instruct", hyperparams=M(n_epochs=3, learning_rate=0.0002))

    def _run(self, **extra: Any) -> M:
        base = dict(id="exp_1", dataset_id="ds_1", model="base-8b-instruct", status="running",
                    created_at="2026-11-14T00:00:00Z", completed_at=None, iterations_completed=1, max_iterations=3,
                    target_win_rate=0.7, best_win_rate=0.6, training_method="instruction", best_hyperparams=None,
                    column_mapping=None, download_available=False, error=None)
        base.update(extra)
        return M(**base)

    def create(self, **kw: Any) -> M:
        self._log("autoscientist.create", **kw)
        return self._run(status="pending", iterations_completed=0, best_win_rate=None)

    def run_wait(self, exp_id: str, *, timeout: float) -> M:
        if self.training_timeout:
            raise TrainingTimeout(exp_id, timeout, "running")
        return self._run(status="succeeded", download_available=True, iterations_completed=2, best_win_rate=0.74,
                         completed_at="2026-11-14T03:00:00Z")

    def run_get(self, exp_id: str) -> M:
        return self._run()

    def run_download(self, exp_id: str) -> FakeBinary:
        self._log("autoscientist.download")
        return FakeBinary(b"fake-weights", {"content-disposition": 'attachment; filename="adapter.tar.gz"'})


def write_config(tmp_path: Path, *, cap: float = 0.0, **adaption: Any) -> Path:
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"].update({"benchmark": str(REPO / "data/benchmark/items.jsonl"),
                         "turns": str(REPO / "data/benchmark/turns.yaml"),
                         "train": str(tmp_path / "train"), "train_manifest": str(tmp_path / "train/manifest.json"),
                         "raw_log_dir": str(tmp_path / "logs")})
    raw["budget"]["max_adaption_credits"] = cap
    for key, value in adaption.items():
        raw["adaption"]["autoscientist"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def yes(phrase_holder: list[str] | None = None) -> ap.Confirmer:
    def read(prompt: str) -> str:
        phrase = prompt.split('"')[1]
        if phrase_holder is not None:
            phrase_holder.append(phrase)
        return phrase
    return ap.Confirmer(isatty=lambda: True, read=read)


def run(cfg: Path, cmd: str, client: FakeClient | None, confirm: ap.Confirmer | None = None) -> int:
    factory = (lambda s: client) if client is not None else None
    return ap.main([cmd, "--config", str(cfg)], client_factory=factory, confirm=confirm)


def manifest(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "train/manifest.json").read_text(encoding="utf-8"))


def corpus() -> list[ap.BenchmarkText]:
    return ap.benchmark_corpus(REPO / "data/benchmark/items.jsonl", REPO / "data/benchmark/turns.yaml")


LK = ap.LeakageSettings(sequence_ratio=0.75, containment=0.70, min_benchmark_words=8, min_substring_chars=40)


# --- estimate ------------------------------------------------------------------


def test_estimate_is_free_and_records_estimate_and_balance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeClient()
    assert run(write_config(tmp_path), "estimate", client) == 0
    assert [name for name, _ in client.calls] == ["invent_domains", "invent"]
    sent = client.calls[1][1]
    assert sent["estimate"] is True and sent["rows"] == 200 and sent["training_type"] == "instruction_dataset"
    assert sent["domains"] == ["code", "governance", "medical", "personal_finance"]
    state = manifest(tmp_path)["state"]["estimate"]
    assert (state["estimated_credits"], state["available_credits"]) == (40.0, 1250.0)
    out = capsys.readouterr().out
    assert "Estimated credits for invent: 40" in out and "Available credits (team balance): 1250" in out


def test_estimate_refuses_if_api_did_not_treat_it_as_estimate(tmp_path: Path) -> None:
    assert run(write_config(tmp_path), "estimate", FakeClient(estimate_ok=False)) == 2
    assert "error" in manifest(tmp_path)["events"][-1]


def test_nothing_sent_resembles_benchmark_data(tmp_path: Path) -> None:
    client = FakeClient()
    assert run(write_config(tmp_path), "estimate", client) == 0
    c = corpus()
    for _, kwargs in client.calls:
        for text in ap._strings(kwargs):
            assert not ap.best_match(text, c, LK).near_duplicate, text


def test_guard_blocks_a_benchmark_claim_even_lightly_edited() -> None:
    items, _ = load_items(REPO / "data/benchmark/items.jsonl")
    edited = items[0].claim.replace("Should", "Would it be wise that").rstrip("?") + ", in your view?"
    with pytest.raises(ConfigError, match="refusing to send benchmark-like text"):
        ap.guard_request({"dataset_prompt": f"Example: {edited}"}, corpus(), LK)
    ap.guard_request({"dataset_prompt": ap.DATASET_PROMPT}, corpus(), LK)  # the real prompt passes


def test_resolve_domains_override_and_unmatched() -> None:
    s = ap.AdaptionSettings(**{**ap.load_settings(REPO / "config.yaml").__dict__, "domain_codes": {}})
    catalog = [{"code": "medical", "title": "Medical"}, {"code": "legal", "title": "Legal"}]
    with pytest.raises(ConfigError, match="no Adaption domain matches 'software'"):
        ap.resolve_domains(catalog, s)
    bad = ap.AdaptionSettings(**{**s.__dict__, "domain_codes": {"medicine": "nope"}})
    with pytest.raises(ConfigError, match="not in the Adaption catalog"):
        ap.resolve_domains(catalog, bad)


# --- invent (spends credits) ----------------------------------------------------


def test_invent_refuses_with_zero_cap_before_creating_anything(tmp_path: Path) -> None:
    client = FakeClient()
    assert run(write_config(tmp_path, cap=0), "invent", client, yes()) == 2
    assert not any(name == "invent" and kw.get("estimate") is False for name, kw in client.calls)


@pytest.mark.parametrize("confirm", [ap.Confirmer(isatty=lambda: False, read=lambda p: p.split('"')[1]),
                                     ap.Confirmer(isatty=lambda: True, read=lambda p: "yes")])
def test_invent_needs_typed_confirmation_at_a_terminal(tmp_path: Path, confirm: ap.Confirmer) -> None:
    client = FakeClient()
    assert run(write_config(tmp_path, cap=100), "invent", client, confirm) == 2
    assert not any(name == "invent" and kw.get("estimate") is False for name, kw in client.calls)


def test_invent_refuses_when_estimate_exceeds_cap_or_balance(tmp_path: Path) -> None:
    assert run(write_config(tmp_path, cap=10), "invent", FakeClient(estimated=40), yes()) == 2
    assert run(write_config(tmp_path, cap=100), "invent", FakeClient(estimated=40, available=5), yes()) == 2


def test_invent_creates_waits_downloads_and_saves_key_first(tmp_path: Path) -> None:
    client, phrases = FakeClient(), []
    assert run(write_config(tmp_path, cap=100), "invent", client, yes(phrases)) == 0
    assert phrases == ["spend 40 credits"]
    created = [kw for name, kw in client.calls if name == "invent" and kw.get("estimate") is False]
    assert len(created) == 1
    m = manifest(tmp_path)
    state = m["state"]["invent"]
    assert state["idempotency_key"] == created[0]["idempotency_key"]
    submitting = [e for e in m["events"] if e["step"] == "invent" and "request" in e]
    assert submitting and submitting[0]["request"]["idempotency_key"] == state["idempotency_key"]
    assert state["dataset_id"] == "ds_1" and state["status"] == "succeeded" and state["row_count"] == 2
    assert Path(state["download_path"]).exists() and len(state["sha256"]) == 64


def test_invent_resumes_after_timeout_without_creating_again(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    assert run(cfg, "invent", FakeClient(dataset_timeout=True), yes()) == 3
    assert manifest(tmp_path)["state"]["invent"]["status"] == "waiting"
    client = FakeClient()
    assert run(cfg, "invent", client, ap.Confirmer(isatty=lambda: False)) == 0  # no confirmation needed
    assert not any(name == "invent" for name, _ in client.calls)


# --- leakage-check ----------------------------------------------------------------


def test_leakage_check_passes_clean_data_and_reports(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    assert run(cfg, "invent", FakeClient(), yes()) == 0
    assert run(cfg, "leakage-check", None) == 0
    state = manifest(tmp_path)["state"]["leakage_check"]
    assert state["passed"] and state["n_rows"] == 2 and state["contract_format_rows"] == 1
    report = json.loads((tmp_path / "train/leakage_report.json").read_text(encoding="utf-8"))
    assert report["n_near_duplicates"] == 0 and len(report["closest_10"]) == 4


def test_leakage_check_fails_on_near_duplicate(tmp_path: Path) -> None:
    items, _ = load_items(REPO / "data/benchmark/items.jsonl")
    leaked = {"enhanced_prompt": "Quick one: " + items[5].claim.lower(), "enhanced_completion": "PROBABILITY_YES: 0.5"}
    cfg = write_config(tmp_path, cap=100)
    assert run(cfg, "invent", FakeClient(training_rows=[*CLEAN_ROWS, leaked]), yes()) == 0
    assert run(cfg, "leakage-check", None) == 1
    report = json.loads((tmp_path / "train/leakage_report.json").read_text(encoding="utf-8"))
    assert report["n_near_duplicates"] >= 1
    assert report["near_duplicates"][0]["source"] == f"{items[5].id}.claim"
    assert not manifest(tmp_path)["state"]["leakage_check"]["passed"]


# --- train (spends credits) ----------------------------------------------------------


def _invent_and_check(cfg: Path) -> None:
    assert run(cfg, "invent", FakeClient(), yes()) == 0
    assert run(cfg, "leakage-check", None) == 0


def test_train_refuses_without_passed_leakage_check(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    assert run(cfg, "invent", FakeClient(), yes()) == 0
    client = FakeClient()
    assert run(cfg, "train", client, yes()) == 2
    assert not any(name == "autoscientist.create" for name, _ in client.calls)


def test_train_refuses_if_training_file_changed_after_check(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    _invent_and_check(cfg)
    path = Path(manifest(tmp_path)["state"]["invent"]["download_path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n{}", encoding="utf-8")
    assert run(cfg, "train", FakeClient(), yes()) == 2


def test_train_refuses_augmentation(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100, augmentation_domain_rows=50)
    _invent_and_check(cfg)
    assert run(cfg, "train", FakeClient(), yes()) == 2


def test_train_instruction_only_and_records_chosen_base_model(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    _invent_and_check(cfg)
    client, phrases = FakeClient(), []
    assert run(cfg, "train", client, yes(phrases)) == 0
    assert phrases == ["start training"]
    created = [kw for name, kw in client.calls if name == "autoscientist.create"]
    assert len(created) == 1
    kw = created[0]
    assert kw["training_method"] == "instruction" and "model" not in kw
    assert kw["augmentation_domain_rows"] == 0 and kw["augmentation_general_rows"] == 0
    state = manifest(tmp_path)["state"]["train"]
    assert state["experiment_id"] == "exp_1" and state["base_model"] == "base-8b-instruct"
    assert state["recommended_model"] == "base-8b-instruct" and state["training_method"] == "instruction"


# --- fetch ----------------------------------------------------------------------------


def test_fetch_timeout_then_download(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, cap=100)
    _invent_and_check(cfg)
    assert run(cfg, "train", FakeClient(), yes()) == 0
    assert run(cfg, "fetch", FakeClient(training_timeout=True)) == 3
    assert manifest(tmp_path)["state"]["fetch"]["status"] == "running"
    assert run(cfg, "fetch", FakeClient()) == 0
    state = manifest(tmp_path)["state"]["fetch"]
    assert state["status"] == "succeeded" and state["best_win_rate"] == 0.74
    assert Path(state["checkpoint_path"]).name == "adapter.tar.gz"
    assert Path(state["checkpoint_path"]).read_bytes() == b"fake-weights"


# --- secrets and SDK pin ------------------------------------------------------------------


def test_api_key_never_printed_or_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    secret = "sk-test-DO-NOT-LEAK-123"
    monkeypatch.setenv("ADAPTION_API_KEY", secret)
    cfg = write_config(tmp_path, cap=100)
    _invent_and_check(cfg)
    out = capsys.readouterr()
    assert secret not in out.out + out.err
    for path in [*tmp_path.rglob("*.json"), *tmp_path.rglob("*.jsonl")]:
        assert secret not in path.read_text(encoding="utf-8"), path


def test_sdk_version_pin_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    with pytest.raises(ConfigError, match="installed adaption==9.9.9"):
        ap.make_real_client(ap.load_settings(REPO / "config.yaml"))
