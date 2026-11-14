"""Run docs/kaggle_runbook.md cells in a real IPython kernel, locally, against fake servers.

Test-only substitutions (everything else runs exactly as written in the runbook):
  /kaggle/working, /kaggle/input   -> a temp folder
  kaggle_secrets                   -> stub returning dummy values
  torch                            -> stub (no GPU here)
  Cell 5 (vLLM install)            -> skipped
  vllm serve                       -> fake_openai.py serving the same model id on :8000
  xAI judge                        -> fake_openai.py on :8001 (config judge.base_url patched)
  JUDGE_MODEL / prices             -> fake-judge, 1.0, 2.0
usage: python tools/kaggle_sim/run_sim.py docs/kaggle_runbook.md <empty temp dir> first|resume|second_model|smoke
(resume/second_model expect <dir>/input/prev-results/ to hold an unzipped results_*.zip; needs ipython)
"""
import re
import subprocess
import sys
import time
import types
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

runbook, root, session = Path(sys.argv[1]), Path(sys.argv[2]).as_posix(), sys.argv[3]
HERE = Path(__file__).parent.as_posix()
FAKE = f"{HERE}/fake_openai.py"

# --- split the runbook into cells: "## Cell N" heading -> its python code blocks ---
cells: dict[int, list[str]] = {}
current = None
for part in re.split(r"(^## Cell \d+.*$)", runbook.read_text(encoding="utf-8"), flags=re.M):
    m = re.match(r"## Cell (\d+)", part)
    if m:
        current = int(m.group(1))
        continue
    if current is not None:
        cells[current] = re.findall(r"```python\n(.*?)```", part, flags=re.S)
print("cells found:", {k: len(v) for k, v in cells.items()})

# --- stubs ---
ks = types.ModuleType("kaggle_secrets")
class UserSecretsClient:  # noqa: N801
    def get_secret(self, name):
        return f"dummy-{name.lower()}"
ks.UserSecretsClient = UserSecretsClient
sys.modules["kaggle_secrets"] = ks
torch = types.ModuleType("torch")
torch.__version__ = "stub"
torch.version = types.SimpleNamespace(cuda="stub")
torch.cuda = types.SimpleNamespace(device_count=lambda: 0, is_available=lambda: False)
sys.modules["torch"] = torch


def subst(code: str) -> str:
    code = code.replace("/kaggle/working", f"{root}/working").replace("/kaggle/input", f"{root}/input")
    code = re.sub(r'^JUDGE_MODEL  = .*$', 'JUDGE_MODEL  = "fake-judge"', code, flags=re.M)
    code = code.replace("PRICE_IN_1M  = None", "PRICE_IN_1M  = 1.0").replace("PRICE_OUT_1M = None", "PRICE_OUT_1M = 2.0")
    code = code.replace('"/tmp/vllm-env/bin/vllm", "serve", HF_MODEL,', f'sys.executable, "{FAKE}", HF_MODEL,')
    for url in ("https://api.x.ai/v1", "https://api.groq.com/openai/v1"):
        code = code.replace(url, "http://127.0.0.1:8001/v1")
    if session == "second_model":
        code = code.replace('MODEL = "llama-3.1-8b"', 'MODEL = "qwen2.5-7b"')
    return code


INJECT_AFTER_3 = f"""
_edit(lambda c: c["judge"].__setitem__("base_url", "http://127.0.0.1:8001/v1"))   # TEST ONLY
JUDGE_SERVER = subprocess.Popen([sys.executable, "{FAKE}", "fake-judge", "--port", "8001"])
time.sleep(1.5)
"""

plan = {"first": [1, 2, 3, "inject", 4, 6, 7, 8, 9, 10, 11],
        "resume": [1, 2, 3, "inject", 6, 9, 10, 11],
        "second_model": [1, 2, 3, "inject", 6, 9],
        "smoke": [1, 2, 3, "inject", 4, 6, 7, 8]}[session]
shell = InteractiveShell.instance()
failures = []
for step in plan:
    blocks = [INJECT_AFTER_3] if step == "inject" else cells[step]
    if step == 5:
        continue
    for i, code in enumerate(blocks):
        label = f"Cell {step}" + (f".{i + 1}" if len(blocks) > 1 else "")
        print(f"\n{'#' * 20} {label} {'#' * 20}", flush=True)
        shell.user_ns["_exit_code"] = 0
        t0 = time.time()
        res = shell.run_cell(subst(code), store_history=False)
        rc = shell.user_ns.get("_exit_code", 0)
        expected_refusal = "--estimate" in code and rc == 2   # cap still $0: refusal is by design
        ok = res.success and (rc == 0 or expected_refusal)
        print(f"--> {label}: {'OK' if ok else 'FAILED'} (shell exit {rc}, {time.time() - t0:.1f}s)", flush=True)
        if not ok:
            failures.append(label)

for name in ("server", "JUDGE_SERVER"):
    proc = shell.user_ns.get(name)
    if proc is not None and proc.poll() is None:
        proc.terminate()
print("\nFAILED CELLS:", failures or "none")
sys.exit(1 if failures else 0)
