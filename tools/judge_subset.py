"""Run `src.harness --judge-only` on a subset: every remaining Qwen cell, and sample 0 for other models.

Free-tier judge capacity is limited, so new models are judged on sample 0 only (the primary prose-gap
analysis uses sample 0 for every model). Usage, from the repo root:

    python tools/judge_subset.py --judge-only --estimate
    python tools/judge_subset.py --judge-only --approve-spend
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.getcwd())

import src.harness as H  # noqa: E402

FINISH_ALL = {"qwen2.5-7b"}
_orig = H.judge_targets


def subset(rows: list[dict[str, Any]], log_dir: Any, judge_model: str, subjects: Any) -> tuple[list[dict[str, Any]], Any]:
    todo, t1 = _orig(rows, log_dir, judge_model, subjects)
    return [r for r in todo if r["subject"] in FINISH_ALL or str(r["sample"]) == "0"], t1


H.judge_targets = subset  # type: ignore[assignment]

if __name__ == "__main__":
    sys.exit(H.main(sys.argv[1:]))
