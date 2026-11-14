# Kaggle runbook: copy-paste cells (one subject model per session)

**Notebook settings:** Accelerator **GPU T4 x2**, Internet **On**.

**Secrets** (Add-ons → Secrets, attach them to the notebook). The repo is public, so no GitHub
token is needed.

| Name | What |
|---|---|
| `HF_TOKEN` | Hugging Face token (accept the Llama 3.1 licence on HF first) |
| `GROQ_API_KEY` | Groq key: the judge (and later the second judge) |
| `XAI_API_KEY` | only if `JUDGE_PROVIDER = "xai"` |

**Before the first session, check the judge block in Cell 1.** It is set for Groq with
`openai/gpt-oss-120b` (the setup that passed Cell 7). Fill in `PRICE_IN_1M` / `PRICE_OUT_1M` from
the Groq console (0.0 only if your Groq account is on the free tier with no billing), and
`JUDGE_REQUESTS_PER_DAY` from Groq's rate-limit page. To use xAI instead, set
`JUDGE_PROVIDER = "xai"` and the xAI model, family and prices. Everything else is already set for the first model,
`llama-3.1-8b`. For the other models, change only the `MODEL` line in Cell 1 (table at the end).

Paste the cells in order.

**What has been verified (14 Nov 2026, locally):** every cell ran in a real IPython kernel via
`tools/kaggle_sim/run_sim.py`, with stand-ins for what only Kaggle has (stub Kaggle secrets and torch,
a fake OpenAI-compatible server in place of vLLM and xAI). Passed: anonymous clone of the public
repo, config edits, smoke test, history viewer, full 540-cell run through the real OpenAI client,
packaging, restoring the zip in a new session (0 cells redone), and switching to a second model.
**Not verifiable off Kaggle:** the vLLM install (Cell 5), the real GPU server and model, CUDA
versions, real xAI responses. The first Kaggle session is the test of those. If a cell
errors, paste the error and the last 30 lines of `/kaggle/working/vllm_server.log` to Claude.

---

## Cell 1: settings

```python
import time, os, sys
START = time.time()

# ---- the only lines you normally edit ----
MODEL = "llama-3.1-8b"        # llama-3.1-8b | qwen2.5-7b | mistral-7b | phi-3.5-mini
JUDGE_PROVIDER = "groq"       # "groq" or "xai"
JUDGE_MODEL  = "openai/gpt-oss-120b"   # exactly as the provider lists it
JUDGE_FAMILY = "openai"       # must differ from llama/qwen/mistral/phi (the harness checks)
JUDGE_MAX_TOKENS = 300        # reasoning models need room to think (gpt-oss used ~80); 20 for non-reasoning
PRICE_IN_1M  = None           # FILL IN: USD per 1M input tokens (provider console; 0.0 only if free)
PRICE_OUT_1M = None           # FILL IN: USD per 1M output tokens
JUDGE_REQUESTS_PER_DAY = None # FILL IN for Groq free tier (its daily request limit); None = no cap
LIMIT = 3                     # smoke-test items
ALLOW_UNVERIFIED = True       # set False once both annotators have reviewed the benchmark
VLLM_SPEC = "vllm"            # if the vLLM install fails once, change to "vllm==0.9.2"
PREV_DIR = "/kaggle/input/prev-results"   # earlier results_*.zip, uploaded as a dataset named prev-results
# ------------------------------------------

REPO_URL = "https://github.com/Gauravsharma2711/ai-sprints-research.git"
REPO = "/kaggle/working/repo"
PER_MODEL = {   # mirrors config.yaml: (tensor_parallel, vLLM --quantization)
    "llama-3.1-8b": (2, None),
    "qwen2.5-7b":   (1, "awq"),
    "mistral-7b":   (2, None),
    "phi-3.5-mini": (1, None),
}
TP, QUANT = PER_MODEL[MODEL]
JUDGE_BASE_URL, JUDGE_KEY = {"groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
                             "xai":  ("https://api.x.ai/v1", "XAI_API_KEY")}[JUDGE_PROVIDER]
UNV = "--allow-unverified" if ALLOW_UNVERIFIED else ""
LORA_PATH, LORA_NAME = None, "contract_sft"   # tuned-model session only (see the end)
print("model:", MODEL, "| TP:", TP, "| quant:", QUANT)
```

## Cell 2: secrets, clone the repo, install requirements

```python
import subprocess, shutil, glob
from kaggle_secrets import UserSecretsClient

def sh(cmd, timeout=None, show=True):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    if show:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
    return p.returncode

sec = UserSecretsClient()
for k in ["HF_TOKEN", "XAI_API_KEY", "GROQ_API_KEY"]:
    try:
        os.environ[k] = sec.get_secret(k)
        print(k, "loaded")                       # never prints the value
    except Exception:
        print(k, "MISSING - attach it under Add-ons > Secrets")
os.environ["HF_HOME"] = "/tmp/hf"

if not os.path.isdir(REPO):
    assert sh(f"git clone {REPO_URL} {REPO}") == 0, "clone failed: is Internet on?"
else:
    sh(f"cd {REPO} && git checkout -- config.yaml && git pull")   # Cell 3 edits config.yaml
os.chdir(REPO)
sh("git log --oneline -1")
sh(f'"{sys.executable}" -m pip install -q -r requirements.txt')

# Restore earlier results, keeping the layout the harness expects.
raw, traj = f"{REPO}/results/raw", f"{REPO}/results/raw/trajectories"
os.makedirs(traj, exist_ok=True)
if os.path.isdir(PREV_DIR):
    csvs = sorted(glob.glob(f"{PREV_DIR}/**/results.csv", recursive=True), key=os.path.getsize)
    if csvs:
        shutil.copy(csvs[-1], f"{raw}/results.csv")      # the largest = most complete
        print("results.csv from", csvs[-1])
    for f in glob.glob(f"{PREV_DIR}/**/trajectories/*.jsonl", recursive=True):
        dst = f"{traj}/{os.path.basename(f)}"
        if not os.path.exists(dst) or os.path.getsize(f) > os.path.getsize(dst):
            shutil.copy(f, dst)
    print("raw:", sorted(os.listdir(raw)), "| trajectories:", sorted(os.listdir(traj)))
else:
    print("no earlier results attached: starting fresh")
```

## Cell 3: apply settings to config.yaml (this session only)

```python
import yaml
assert JUDGE_MODEL and PRICE_IN_1M is not None and PRICE_OUT_1M is not None, \
    "Fill in the judge block (model and prices) in Cell 1, then rerun Cells 1 and 3."
CFG = f"{REPO}/config.yaml"

def _edit(fn):
    c = yaml.safe_load(open(CFG)); fn(c); yaml.safe_dump(c, open(CFG, "w"), sort_keys=False); return c

def apply_settings(c):
    s = next(x for x in c["subjects"] if x["name"] == MODEL)
    assert s["tensor_parallel"] == TP, (s["tensor_parallel"], TP)
    s["confirmed"] = True                      # you have checked the HF id, licence and T4 fit
    c["judge"].update(provider=JUDGE_PROVIDER, model=JUDGE_MODEL, family=JUDGE_FAMILY,
                      base_url=JUDGE_BASE_URL, api_key_env=JUDGE_KEY, max_tokens=JUDGE_MAX_TOKENS,
                      price_per_1m_input_usd=PRICE_IN_1M, price_per_1m_output_usd=PRICE_OUT_1M)
    c["judge"]["rate_limits"]["requests_per_day"] = JUDGE_REQUESTS_PER_DAY

def set_cap(usd):
    """Call only with an amount YOU approved after reading an --estimate printout."""
    _edit(lambda c: c["budget"].__setitem__("max_spend_usd", float(usd)))
    print("judge spend cap set to $", usd)

cfg = _edit(apply_settings)
HF_MODEL = next(x for x in cfg["subjects"] if x["name"] == MODEL)["hf_id"]
print("will serve:", HF_MODEL, "| judge:", JUDGE_PROVIDER, JUDGE_MODEL, "| cap: $", cfg["budget"]["max_spend_usd"])
```

The spend cap starts at $0, so every paid run refuses until you call `set_cap(...)` (Cells 7 and 10).

## Cell 4: hardware check

```python
import sys, torch
sh("nvidia-smi")
print(sys.version)
assert sys.version_info >= (3, 11), "the harness needs Python 3.11+"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs:", torch.cuda.device_count())
```

Expect 2 GPUs (Tesla T4). Note the `CUDA Version` in the `nvidia-smi` header; Cell 5 assumes 12.8.

## Cell 5: install vLLM in its own environment (up to 20 minutes)

```python
t0 = time.time()
rc = sh("pip install -q uv && uv venv /tmp/vllm-env --seed && "
        f"uv pip install --python /tmp/vllm-env/bin/python '{VLLM_SPEC}' --torch-backend=cu128",
        timeout=1200)
print("rc:", rc, "| minutes:", round((time.time() - t0) / 60, 1))
sh("/tmp/vllm-env/bin/python -c \"import vllm, torch; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())\"")
```

- If `nvidia-smi` shows a CUDA version other than 12.8, change `cu128` to match
  (a `libcudart.so.13` error later means a wrong build).
- If it fails: set `VLLM_SPEC = "vllm==0.9.2"` in Cell 1, rerun Cells 1 and 5 once. If that also
  fails, stop and ask Claude for the Hugging Face fallback backend.

## Cell 6: start the vLLM server (up to 15 minutes)

```python
import json, urllib.request
env = dict(os.environ, VLLM_ATTENTION_BACKEND="XFORMERS", HF_HOME="/tmp/hf")
cmd = ["/tmp/vllm-env/bin/vllm", "serve", HF_MODEL, "--dtype", "half", "--max-model-len", "4096",
       "--gpu-memory-utilization", "0.88", "--tensor-parallel-size", str(TP), "--port", "8000"]
if QUANT:
    cmd += ["--quantization", QUANT]
if LORA_PATH:
    cmd += ["--enable-lora", "--lora-modules", f"{LORA_NAME}={LORA_PATH}", "--max-lora-rank", "64"]
if TP == 1:
    env["CUDA_VISIBLE_DEVICES"] = "0"

server = subprocess.Popen(cmd, env=env, stdout=open("/kaggle/working/vllm_server.log", "w"),
                          stderr=subprocess.STDOUT)
ready = False
for i in range(180):
    if server.poll() is not None:
        break
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3) as r:
            print("READY:", [m["id"] for m in json.load(r)["data"]]); ready = True; break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    print("NOT READY. Log tail:\n", open("/kaggle/working/vllm_server.log").read()[-3000:])
```

## Cell 7: quick checks + smoke-test estimate

```python
from openai import OpenAI
# a) local logprobs (free)
local = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = local.chat.completions.create(model=LORA_NAME if LORA_PATH else HF_MODEL,
    messages=[{"role": "user", "content": "Answer with one word, Yes or No: Is the sky blue on a clear day?"}],
    max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
print("logprobs:", [(t.token, round(t.logprob, 3)) for t in r.choices[0].logprobs.content[0].top_logprobs])

# b) judge (one tiny call): must print exactly PROBABILITY_YES: 0.50
judge = OpenAI(base_url=JUDGE_BASE_URL, api_key=os.environ[JUDGE_KEY])
j = judge.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=JUDGE_MAX_TOKENS,
    messages=[{"role": "user", "content": "Reply exactly: PROBABILITY_YES: 0.50"}])
print("judge:", repr(j.choices[0].message.content), "| finish:", j.choices[0].finish_reason)
assert "PROBABILITY_YES" in (j.choices[0].message.content or ""), "empty judge reply: raise JUDGE_MAX_TOKENS"

```

```python
# c) smoke-test estimate (free): call counts and worst-case judge cost
!"{sys.executable}" -m src.harness --models {MODEL} --limit {LIMIT} --out-dir results/smoke --estimate
```

Read the worst-case cost. If you approve it, run `set_cap(<that amount, rounded up>)`:

```python
set_cap(1.00)   # EDIT to the amount you approved
```

## Cell 8: smoke test (writes to results/smoke, never mixed with real results)

```python
t1 = time.time()
!"{sys.executable}" -u -m src.harness --models {MODEL} --limit {LIMIT} --out-dir results/smoke --approve-spend {UNV}
smoke_seconds = time.time() - t1
print("smoke seconds:", round(smoke_seconds))
```

```python
# full transcript of every arm for one cell
!"{sys.executable}" -m src.harness --show-histories --out-dir results/smoke --condition contract_prompt
```

Pass checks: the status table at the end is mostly `ok` (few `nan_*`); the transcripts read
sensibly. Exit code 2 = refused (the message says why); 3 = stopped early (cap reached).
Under `contract_prompt` the logit probe may be NaN by design.

## Cell 9: time and cost for the full run

```python
if "smoke_seconds" in globals():           # set by Cell 8; not present in a resumed session
    per_item = smoke_seconds / LIMIT      # one item = 3 conditions x 3 samples x 11 calls
    print(f"~{per_item:.0f}s per item -> ~{per_item * 60 / 3600:.1f} h for all 60 items of {MODEL}")
else:
    print("no smoke test in this session: skipping the time estimate")
!"{sys.executable}" -m src.harness --models {MODEL} --estimate
```

If one model needs more than about 4 hours, tell Claude (the model list needs trimming).
Otherwise, if you approve the printed worst-case judge cost (it includes what was already spent):

```python
set_cap(5.00)   # EDIT to the total you approved
```

## Cell 10: full run

```python
!"{sys.executable}" -u -m src.harness --models {MODEL} --approve-spend {UNV}
```

Saves after every finished cell to `results/raw/results.csv` and
`results/raw/trajectories/<model>.jsonl`.

**If the session dies:** start a new session, attach the last `results_*.zip` as a dataset named
`prev-results`, then run Cells 1, 2, 3, 4, 5, 6, **9** (the spend cap resets to $0 in every new
session, so approve the remaining cost again with `set_cap`), then 10 and 11. Finished cells are
skipped: the estimate shows how many remain. Cells 7 and 8 can be skipped when resuming.

## Cell 11: package and stop

```python
import pandas as pd
server.terminate()
tag = MODEL + ("_lora" if LORA_PATH else "")
shutil.copy("/kaggle/working/vllm_server.log", f"{REPO}/results/raw/vllm_server_{tag}.log")
shutil.make_archive(f"/kaggle/working/results_{tag}", "zip", f"{REPO}/results/raw")

df = pd.read_csv(f"{REPO}/results/raw/results.csv")
print("rows:", len(df), "| by model:", df["subject"].value_counts().to_dict())
st = [c for c in df.columns if c.startswith("status_")]
print("share not ok, per arm:\n", (df[st] != "ok").mean().round(3).sort_values(ascending=False))
print("session hours:", round((time.time() - START) / 3600, 2))
```

Download `results_<model>.zip` from the Output tab.

## After each session (on your computer)

Unzip into the repo's `results/raw/` (keep the `trajectories/` folder), then:

```bash
git add results/raw && git commit -m "Results: <model>" && git push
```

For the next model: in Kaggle, upload that zip as a dataset, attach it so it appears under
`/kaggle/input/prev-results`, change `MODEL` in Cell 1, and run all cells again.

## Models

| MODEL | HF id (config.yaml) | GPUs | Note |
|---|---|---|---|
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 2 | accept the licence on HF first |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct-AWQ | 1 | AWQ 4-bit |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | 2 | |
| `phi-3.5-mini` | microsoft/Phi-3.5-mini-instruct | 1 | |

## Tuned model (later, after `python -m src.adaption_pipeline fetch`)

Not ready to paste yet. It needs: the base model AutoScientist chose (recorded in
`data/train/manifest.json`) run as an ordinary subject; a check that the downloaded checkpoint
is a LoRA adapter vLLM can load (the SDK only calls it "best trained model artifact"); and a
`contract-sft` subject in `config.yaml` whose `hf_id` is `contract_sft`. Ask Claude to prepare
that session when the checkpoint exists.
