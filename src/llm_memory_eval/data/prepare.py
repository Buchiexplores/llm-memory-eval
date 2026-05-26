"""Convert raw benchmark dumps into a unified per-instance JSON array.

The output schema is shared by all three benchmarks:

.. code-block:: json

   {
     "instance_id": "LB_narrativeqa_000",
     "benchmark":   "LongBench",
     "task_type":   "narrativeqa",
     "context":     "<long input>",
     "question":    "<query>",
     "answer":      "<reference answer>",
     "all_answers": ["<ref 1>", "<ref 2>"],
     "context_tokens_approx": 23306,
     "session_count": null,
     "turn_count":    null,
     "length_category": "long"
   }

This module is intentionally thin: it normalises field names and computes
the length category. Benchmark-specific parsing is delegated to small
helpers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_memory_eval.data.length_buckets import assign_length_bucket
from llm_memory_eval.utils.logging import get_logger
from llm_memory_eval.utils.tokens import count_tokens

log = get_logger(__name__)


def prepare_all(raw_dir: Path, output_dir: Path) -> Path:
    """Walk *raw_dir* and write the unified instances to *output_dir*.

    Returns the path to ``all_instances.json``.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances: list[dict[str, Any]] = []
    instances.extend(_load_longbench(raw_dir / "LongBench"))
    instances.extend(_load_locomo(raw_dir / "LoCoMo"))
    instances.extend(_load_longmemeval(raw_dir / "LongMemEval"))

    target = output_dir / "all_instances.json"
    target.write_text(json.dumps(instances, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d unified instances to %s", len(instances), target)
    return target


def _load_longbench(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        log.warning("LongBench directory missing: %s", root)
        return
    for jsonl in sorted(root.glob("**/*.jsonl")):
        task = jsonl.parent.name if jsonl.parent != root else jsonl.stem
        with jsonl.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                context = row.get("context") or row.get("input") or ""
                question = row.get("input") if "context" in row else row.get("question", "")
                answers = row.get("answers") or [row.get("answer", "")]
                tokens = count_tokens(context)
                yield {
                    "instance_id": f"LB_{task}_{i:03d}",
                    "benchmark": "LongBench",
                    "task_type": task,
                    "context": context,
                    "question": question,
                    "answer": answers[0] if answers else "",
                    "all_answers": list(answers),
                    "context_tokens_approx": tokens,
                    "session_count": None,
                    "turn_count": None,
                    "length_category": assign_length_bucket(
                        benchmark="LongBench", token_count=tokens
                    ),
                }


def _load_locomo(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        log.warning("LoCoMo directory missing: %s", root)
        return
    for jsonl in sorted(root.glob("**/*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                context = row.get("conversation") or row.get("context", "")
                question = row.get("question", "")
                answer = row.get("answer", "")
                sessions = row.get("session_count") or row.get("num_sessions")
                turns = row.get("turn_count") or row.get("num_turns")
                tokens = count_tokens(context)
                yield {
                    "instance_id": f"LC_{i:05d}",
                    "benchmark": "LoCoMo",
                    "task_type": row.get("task_type", "qa"),
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "all_answers": [answer] if answer else [],
                    "context_tokens_approx": tokens,
                    "session_count": sessions,
                    "turn_count": turns,
                    "length_category": assign_length_bucket(
                        benchmark="LoCoMo",
                        token_count=tokens,
                        session_count=sessions,
                        turn_count=turns,
                    ),
                }


def _load_longmemeval(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        log.warning("LongMemEval directory missing: %s", root)
        return
    for jsonl in sorted(root.glob("**/*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                context = row.get("history") or row.get("context", "")
                question = row.get("question", "")
                answer = row.get("answer", "")
                tokens = count_tokens(context)
                yield {
                    "instance_id": f"LME_{i:03d}",
                    "benchmark": "LongMemEval",
                    "task_type": row.get("task_type", "qa"),
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "all_answers": [answer] if answer else [],
                    "context_tokens_approx": tokens,
                    "session_count": None,
                    "turn_count": None,
                    "length_category": assign_length_bucket(
                        benchmark="LongMemEval", token_count=tokens
                    ),
                }
