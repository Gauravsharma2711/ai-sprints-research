# CLAUDE.md: standing rules for this repo

This repo is for a 3-day research sprint (AI x Epistemics, Apart Research, Fri 13 to Sun 15 Nov 2026)
measuring multi-turn epistemic behaviour in LLMs: bluff shift, evidence update, and belief recovery
after a retracted challenge ("hysteresis").

`docs/playbook_preexisting.md` is the plan, written **before** the sprint. Treat it as preexisting
work. It is a source for the design and metric definitions (Section 4 is authoritative), **not** for
numbers or code: copy no numbers from it, and reuse none of its code or snippets unless the user has
specified the fixes.

## Non-negotiable rules

1. **Never invent, hardcode, estimate or "fill in" results.** Every number in a table, figure, README
   or paper must be computed by code from logged runs in `results/raw/`. No placeholder numbers in
   output files. If data is missing, the output says it is missing.
2. **Every model call is logged to raw JSONL** (`results/raw/trajectories/`, one append-only file per
   subject model plus `second_judge.jsonl` and `_runs.jsonl`, so no file exceeds GitHub's 100 MB
   limit; this replaces the playbook's single `trajectories.jsonl`): subject,
   judge and probe calls alike, including failures and retries. Each record holds at least a timestamp,
   run id, model id, base_url, arm, item id, condition, sample index, full request messages, sampling
   params, raw response text, usage, latency and error (if any). Log before parsing.
3. **Failed parses become NaN and are logged.** Never substitute a default value (no 0.5, no previous
   value). One re-ask is allowed where the design specifies it; after that, NaN plus a logged
   `parse_error` record. Report parse-failure/NaN rates per arm.
4. **Never print, log or commit keys.** Keys come from environment variables (`.env`, loaded with
   python-dotenv, or Kaggle Secrets). Do not read or display `.env`. Do not echo env vars. Strip
   `Authorization` headers and keys from anything written to logs. `.env` is gitignored; only
   `.env.example` (names only, no values) is committed.
5. **Ask the user before any step that spends money or sponsor credits** (judge API calls, any paid API, anything that consumes paid GPU time). Show an estimate of call count,
   tokens and cost first and wait for an explicit yes. Respect `budget` in `config.yaml`; code must
   refuse to exceed it rather than warn. Approval for one run does not carry over to the next.
6. **Python 3.11, fully typed** (type hints on all functions; dataclasses or TypedDicts for records).
7. **Tests are offline.** `pytest` must pass with no network and no keys, using a mock
   OpenAI-compatible client. Tests must never hit a real endpoint.

## Further rules from the plan

- All model endpoints go through the `openai` client with a configurable `base_url` (vLLM, xAI, Groq).
  Read endpoints, model ids and limits from `config.yaml`; do not hardcode them in code.
- Confidence convention: every probability is P(answer is YES), parsed from `PROBABILITY_YES: <x>`.
- The judge must be from a different model family than every subject; enforce this in code.
  The judge sees only stripped prose, never the subject identity, confidence line or contract header.
- Benchmark items: Claude may draft items but must never mark them verified (`verified_by`) or assign
  final tiers on the annotators' behalf.
- Bootstrap CIs resample **items**, not rows. Never average per-item ratios.
- Figures and tables are generated only from `results/raw/` files by scripts, never edited by hand.
- Wording: "context ablation" (not "causal scrubbing"); "consistent with self-conditioning" (not
  "proves"); "default-prompted open-weights models" (not "frontier LLMs").
- Keep the README's "Preexisting work vs. work done during the sprint" section accurate as work lands.
- Do not cite a paper unless the user has confirmed its details.

## Layout

```
src/              experiment code (harness, analysis)
data/benchmark/   benchmark items and fixed user turns
results/raw/      raw JSONL logs and results.csv (committed so judges can inspect)
results/tables/   generated tables
results/figures/  generated figures
notebooks/        Kaggle / exploration notebooks
tests/            offline pytest tests with a mock client
docs/             playbook and notes
```
