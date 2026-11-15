# Epistemic Hysteresis in Multi-Turn LLM Advice

> Status: scaffold only. No experiments have been run and there are no results yet.

Submission to the AI x Epistemics Research Sprint (Apart Research),
13 to 15 November 2026, Track 1: Model Epistemics & Decision-Making Evals.

## Question

When a user challenges an LLM advisor with no evidence and then explicitly retracts the challenge,
does the model's stated belief return to its original value? And does it still update appropriately
when given real evidence?

## Results

_None yet. Every number here will be generated from `results/raw/` by scripts in `src/`._

## Preexisting work vs. work done during the sprint

| Item | Status | Notes |
|---|---|---|
| `docs/playbook_preexisting.md` | **Preexisting** (written before the sprint) | Research plan, design and metric definitions. Contains no results. Its code has known flaws and was not reused without fixes. |
| Repo scaffold: `CLAUDE.md`, `config.yaml`, `requirements.txt`, `.gitignore`, `.env.example`, this README | Sprint (Sat 14 Nov 2026) | |
| `src/benchmark.py` (schema, validator, review-sheet export), `tests/test_benchmark.py` | Sprint (Sat 14 Nov 2026) | |
| `src/harness.py`, `src/prompts.py`, `src/parsing.py`, `src/clients.py`, `tests/test_harness.py` | Sprint (Sat 14 Nov 2026) | Harness written by Claude; only dry-run (mock client) tested so far. Prompt wording in `prompts.py` is a draft adapted from the playbook's section 12 templates. |
| `src/analyze.py`, `src/second_judge.py`, `tests/test_analyze.py` | Sprint (Sat 14 Nov 2026) | Analysis, judge validation and second-judge runner written by Claude; tested on dry-run (mock) output only. |
| Judge choice | Sprint (Sat 14 Nov 2026) | The playbook planned xAI Grok as judge. During the sprint the judge was changed to Google Gemini (OpenAI-compatible endpoint, free tier) after Groq's free-tier daily token limit proved too small. `--judge-only` re-judges any cell scored by another model, so all reported J values come from one Gemini model. |
| `docs/kaggle_runbook.md` | Sprint (Sat 14 Nov 2026) | Kaggle notebook cells drafted by the user, revised by Claude to match the code. Not yet run on Kaggle. |
| `data/benchmark/items.jsonl` (60 items), `turns.yaml`, `review_sheet.csv` | Sprint (Sat 14 Nov 2026) | **Drafted by Claude; not yet reviewed or verified by humans.** |
| `results/` | Sprint | Not started |
| `tests/` | Sprint | Not started |

AI-use statement: _to be written._

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env   # then fill in keys locally; never commit .env
```

Edit `config.yaml` (subject models, judge model name, caps) before running anything.

## Reproducing

_To be written once the harness exists._

## Repository layout

```
src/              experiment code
data/benchmark/   benchmark items and fixed user turns
results/raw/      trajectories/<model>.jsonl (every model call, one file per model), results.csv
results/tables/   generated tables
results/figures/  generated figures
notebooks/        Kaggle notebooks
tests/            offline tests (mock client)
docs/             playbook and notes
```

## Limitations and dual-use considerations

_To be written from actual findings._
