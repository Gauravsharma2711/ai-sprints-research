# Kaggle runbook: one subject model per session

Paste the cells in order into a Kaggle notebook (GPU **T4 x2**, Internet **on**). Not yet run on
Kaggle: the first session is the real test. Revised from the original draft to match this repo;
changes are marked **[changed]**.

## What the code expects (checked against the repo)

- **[changed]** Run the harness as a module from the repo root: `python -m src.harness ...`.
  `python src/harness.py` fails (`No module named 'src'`).
- **[changed]** `--models` takes the subject **name** from `config.yaml` (`llama-3.1-8b`,
  `qwen2.5-7b`, `mistral-7b`, `phi-3.5-mini`), not the Hugging Face id.
- The harness reads `base_url` per subject from `config.yaml` (default `http://localhost:8000/v1`)
  and sends the subject's `hf_id` as the model name. Before the first call it checks that the
  server's `/v1/models` lists that id, so `vllm serve <hf_id>` (no renaming) is right.
- **[changed]** A real run refuses to start until all of these hold (cell 0b sets them for the
  session):
  - `judge.model` and the judge prices are set
  - the subject is `confirmed: true`
  - the worst-case judge cost fits `budget.max_spend_usd`
  - `--approve-spend` is passed (only after you have approved the printed estimate)
  - `--allow-unverified` is passed while the benchmark items are still unreviewed
- **[changed]** Raw logs are a directory, `results/raw/trajectories/<model>.jsonl`, not a single
  `trajectories.jsonl`. Restoring must keep that folder structure (cell 1).
- Results resume per cell from `results/raw/results.csv`. If prompts, items or turns change, the
  harness refuses to mix rows (protocol hash) rather than silently continuing.

## Cell 0: parameters

```python
import time, os
START = time.time()

REPO_URL    = "https://github.com/<your-user>/<your-repo>.git"
SUBJECT     = "llama-3.1-8b"   # [changed] subject NAME in config.yaml; HF id is read from there
TP          = 2                # 2 for 7-8B fp16; 1 for AWQ or models up to ~4B (must match config)
QUANT       = None             # "awq" for Qwen/Qwen2.5-7B-Instruct-AWQ
LIMIT       = 3                # smoke-test items
LORA_PATH   = None             # adapter dir for the tuned-model session (see "Tuned model" below)
LORA_NAME   = "contract_sft"
PREV_DIR    = "/kaggle/input/prev-results"   # attached dataset of earlier result zips, if any
VLLM_SPEC   = "vllm"           # second attempt if the first fails: "vllm==0.9.2"

# [changed] Judge settings and spend cap, applied to config.yaml for this session only (cell 0b).
JUDGE_MODEL     = "<model name from the xAI console>"
PRICE_IN_1M     = None   # USD per 1M input tokens, from the xAI console
PRICE_OUT_1M    = None   # USD per 1M output tokens
MAX_SPEND_USD   = 0.0    # raise only to the amount you approved after seeing --estimate
ALLOW_UNVERIFIED = True  # set False once both annotators have reviewed data/benchmark/items.jsonl
```

## Cell 1: secrets, repo, dependencies

```python
import subprocess, os, shutil, glob, base64
from kaggle_secrets import UserSecretsClient

def sh(cmd, timeout=None, show=True):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    if show:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
    return p.returncode

sec = UserSecretsClient()
for k in ["HF_TOKEN", "XAI_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN"]:
    try:
        os.environ[k] = sec.get_secret(k)
        print(k, "loaded")                       # never print the value
    except Exception:
        print(k, "MISSING - attach it under Add-ons > Secrets")
os.environ["HF_HOME"] = "/tmp/hf"

# [changed] Private repo: authenticate git through an environment-only header, so the token never
# appears in a command line, a URL or printed output. GITHUB_TOKEN = fine-grained, read-only, this repo.
if os.environ.get("GITHUB_TOKEN"):
    basic = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
    os.environ.update({"GIT_CONFIG_COUNT": "1",
                       "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                       "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})

REPO = "/kaggle/working/repo"
if not os.path.isdir(REPO):
    sh(f"git clone {REPO_URL} {REPO}")
else:
    sh(f"cd {REPO} && git pull")
os.chdir(REPO)
sh("pip install -q -r requirements.txt")

# [changed] Restore earlier results keeping the folder layout: results.csv -> results/raw/,
# per-model logs -> results/raw/trajectories/. Kaggle unzips attached zips; files may be nested.
raw, traj = f"{REPO}/results/raw", f"{REPO}/results/raw/trajectories"
os.makedirs(traj, exist_ok=True)
if os.path.isdir(PREV_DIR):
    csvs = sorted(glob.glob(f"{PREV_DIR}/**/results.csv", recursive=True), key=os.path.getsize)
    if csvs:
        shutil.copy(csvs[-1], f"{raw}/results.csv")       # the largest = most complete
        print("results.csv from", csvs[-1], "| other copies ignored:", csvs[:-1])
    for f in glob.glob(f"{PREV_DIR}/**/trajectories/*.jsonl", recursive=True):
        dst = f"{traj}/{os.path.basename(f)}"
        if not os.path.exists(dst) or os.path.getsize(f) > os.path.getsize(dst):
            shutil.copy(f, dst)
    print("raw now:", os.listdir(raw), "| trajectories:", os.listdir(traj))
```

## Cell 0b: apply session settings to config.yaml  **[new]**

```python
import yaml
cfg_path = f"{REPO}/config.yaml"
cfg = yaml.safe_load(open(cfg_path))
subj = next(s for s in cfg["subjects"] if s["name"] == SUBJECT)
HF_MODEL = subj["hf_id"]
assert subj["tensor_parallel"] == TP, f"config says tensor_parallel={subj['tensor_parallel']}, cell 0 says {TP}"
subj["confirmed"] = True             # you have checked the HF id, licence and T4 fit
cfg["judge"]["model"] = JUDGE_MODEL
cfg["judge"]["price_per_1m_input_usd"] = PRICE_IN_1M
cfg["judge"]["price_per_1m_output_usd"] = PRICE_OUT_1M
cfg["budget"]["max_spend_usd"] = MAX_SPEND_USD
yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
print("serving", HF_MODEL, "| judge", JUDGE_MODEL, "| cap $", MAX_SPEND_USD)
```

This edits only the Kaggle working copy. To make a setting permanent, change `config.yaml` in the
repo and push.

## Cell 2: hardware check

```python
sh("nvidia-smi")
sh("python --version")      # [changed] must be 3.11+ (the harness uses asyncio.TaskGroup)
import torch, sys
assert sys.version_info >= (3, 11), sys.version
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
```

Expect two T4s. Note the `CUDA Version` in the `nvidia-smi` header.

## Cell 3: vLLM in an isolated environment (time-boxed to 20 minutes)

```python
t0 = time.time()
rc = sh("pip install -q uv && uv venv /tmp/vllm-env --seed && "
        f"uv pip install --python /tmp/vllm-env/bin/python '{VLLM_SPEC}' --torch-backend=cu128",
        timeout=1200)
print("rc:", rc, "| minutes:", round((time.time() - t0) / 60, 1))
sh("/tmp/vllm-env/bin/python -c \"import vllm, torch; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())\"")
```

- `cu128` assumes CUDA 12.8; change it if `nvidia-smi` shows otherwise (`libcudart.so.13` = wrong build).
- If it fails, set `VLLM_SPEC = "vllm==0.9.2"` and rerun once. If both fail within 20 minutes,
  stop: see "If vLLM fails" below.

## Cell 4: start the server

```python
import requests
env = dict(os.environ, VLLM_ATTENTION_BACKEND="XFORMERS", HF_HOME="/tmp/hf")
cmd = ["/tmp/vllm-env/bin/vllm", "serve", HF_MODEL,
       "--dtype", "half", "--max-model-len", "4096",          # matches config.yaml
       "--gpu-memory-utilization", "0.88",
       "--tensor-parallel-size", str(TP), "--port", "8000"]
if QUANT:
    cmd += ["--quantization", QUANT]
if LORA_PATH:
    cmd += ["--enable-lora", "--lora-modules", f"{LORA_NAME}={LORA_PATH}", "--max-lora-rank", "64"]
if TP == 1:
    env["CUDA_VISIBLE_DEVICES"] = "0"

log = open("/kaggle/working/vllm_server.log", "w")
server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

ready = False
for i in range(180):                       # up to 15 minutes
    if server.poll() is not None:
        print("server exited; log tail:\n", open("/kaggle/working/vllm_server.log").read()[-3000:]); break
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=3)
        if r.ok:
            print("READY:", [m["id"] for m in r.json()["data"]]); ready = True; break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    print("Not ready. Log tail:\n", open("/kaggle/working/vllm_server.log").read()[-3000:])
```

## Cell 5: smoke tests, in order

```python
from openai import OpenAI
# 5a: local logprobs work? (free)
local = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = local.chat.completions.create(
    model=HF_MODEL if not LORA_PATH else LORA_NAME,
    messages=[{"role": "user", "content": "Answer with one word, Yes or No: Is the sky blue on a clear day?"}],
    max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
print([(t.token, round(t.logprob, 3)) for t in r.choices[0].logprobs.content[0].top_logprobs])

# 5b: xAI judge works? (one tiny paid call)
judge = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
j = judge.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=20,
    messages=[{"role": "user", "content": "Reply exactly: PROBABILITY_YES: 0.50"}])
print(j.choices[0].message.content)
```

```python
# 5c-1 [new]: estimate only (free). Prints call counts and worst-case judge cost vs your cap.
UNV = "--allow-unverified" if ALLOW_UNVERIFIED else ""
!python -m src.harness --models {SUBJECT} --limit {LIMIT} --out-dir results/smoke --estimate
```

Approve the printed smoke cost (and set `MAX_SPEND_USD` to cover it) before 5c-2.

```python
# 5c-2 [changed]: smoke run into a SEPARATE folder, so smoke rows never mix with the real results.
t1 = time.time()
!python -u -m src.harness --models {SUBJECT} --limit {LIMIT} --out-dir results/smoke --approve-spend {UNV}
smoke_seconds = time.time() - t1
print("smoke seconds:", round(smoke_seconds))
```

```python
# 5d [changed]: read a full transcript for every arm of one cell
!python -m src.harness --show-histories --out-dir results/smoke --condition contract_prompt
```

Pass checks:

- 5a prints real log-probabilities.
- 5b prints a probability line.
- 5c-2 ends with a status table per arm: mostly `ok`, few `nan_*`. Exit codes: 2 = refused (it
  says why), 3 = stopped early (budget or daily cap).
- 5d histories read sensibly. `probe_mass` near 1 means the probe sees Yes/No tokens. Under
  `contract_prompt` the probe may be NaN by design.

## Cell 6: time estimate

```python
N_ITEMS = 60
per_item = smoke_seconds / LIMIT        # one item = 3 conditions x 3 samples x 11 calls
print(f"~{per_item:.0f}s per item -> ~{per_item * N_ITEMS / 3600:.1f} h for this model")
!python -m src.harness --models {SUBJECT} --estimate      # [new] full-run judge cost vs cap
```

If one model needs more than about 4 hours, trim the model count. Approve the printed full-run
judge cost and raise `MAX_SPEND_USD` (rerun cell 0b) before cell 7.

## Cell 7: full run

```python
!python -u -m src.harness --models {SUBJECT} --approve-spend {UNV}
```

Writes `results/raw/results.csv` and `results/raw/trajectories/<model>.jsonl` after every cell.
If the session dies, start a new one from cell 0 with the zip from cell 8 attached as `PREV_DIR`;
the harness skips finished cells. It prints log file sizes at the end and warns if any file
approaches GitHub's 100 MB limit.

## Cell 8: package, stop server, report

```python
import pandas as pd
server.terminate()
tag = SUBJECT + ("_lora" if LORA_PATH else "")
shutil.copy("/kaggle/working/vllm_server.log", f"{REPO}/results/raw/vllm_server_{tag}.log")
shutil.make_archive(f"/kaggle/working/results_{tag}", "zip", f"{REPO}/results/raw")   # keeps trajectories/

df = pd.read_csv(f"{REPO}/results/raw/results.csv")
print("rows:", len(df), "| by subject:", df["subject"].value_counts().to_dict())
status_cols = [c for c in df.columns if c.startswith("status_")]
print((df[status_cols] != "ok").mean().round(3).sort_values(ascending=False))   # [changed] NaN/failure rate per arm
print("session hours:", round((time.time() - START) / 3600, 2))
```

Download `results_<tag>.zip` from the Output tab. Then, on your machine, unzip it into
`results/raw/` (keep `trajectories/`), commit and push. For the next model, attach the zip as
`PREV_DIR`, change `SUBJECT`/`TP`/`QUANT` in cell 0, and start again.

## Per-model settings

| SUBJECT | HF id (from config) | TP | QUANT |
|---|---|---|---|
| llama-3.1-8b | meta-llama/Llama-3.1-8B-Instruct (accept the licence on HF first) | 2 | None |
| qwen2.5-7b | Qwen/Qwen2.5-7B-Instruct-AWQ | 1 | "awq" |
| mistral-7b | mistralai/Mistral-7B-Instruct-v0.3 | 2 | None |
| phi-3.5-mini | microsoft/Phi-3.5-mini-instruct | 1 | None |

## Tuned model (after `python -m src.adaption_pipeline fetch`)

1. Run the AutoScientist base model as an ordinary subject first (the manifest records which
   model it chose). It needs its own `subjects:` entry if it is not already one of the four.
2. Check what the downloaded checkpoint is. vLLM needs a LoRA adapter directory
   (`adapter_config.json` + weights). The SDK only says "best trained model artifact", so this is
   unverified.
3. Add a subject whose `hf_id` is the served LoRA name, so the harness sends `contract_sft`:
   ```yaml
   - name: contract-sft
     <<: *subject_defaults
     hf_id: contract_sft        # = LORA_NAME passed to --lora-modules
     family: <base model family>
     quantisation: none
     tensor_parallel: <as base>
     confirmed: true
   ```
4. Set `HF_MODEL` (cell 0b sets it from the base subject; override it to the base id), `LORA_PATH`,
   and `SUBJECT = "contract-sft"`, then run cells 4 to 8. Compare against the base under the
   `default` condition.

## If vLLM fails

Not built yet: a Hugging Face backend (`attn_implementation="sdpa"`, batched generation, a Yes/No
probe read from output scores) as a second client type in `src/clients.py`. Ask for it if needed.
If a cell errors, paste the error and the last 30 lines of `vllm_server.log`.
