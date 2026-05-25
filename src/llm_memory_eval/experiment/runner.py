"""End-to-end experiment runner with checkpointing."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional

from llm_memory_eval import __version__
from llm_memory_eval.config import ExperimentConfig
from llm_memory_eval.experiment.schema import InstanceResult, RunMetadata
from llm_memory_eval.llm.base import LLMClient
from llm_memory_eval.llm.factory import build_client
from llm_memory_eval.memory.rag import RagMemory
from llm_memory_eval.memory.summarization import SummarizationMemory
from llm_memory_eval.metrics import (
    best_em,
    best_f1,
    consistency_indicator,
    contradiction_indicator,
)
from llm_memory_eval.utils.logging import get_logger


log = get_logger(__name__)


ANSWER_SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer the "
    "question. Be concise and precise: give the shortest answer that fully "
    "answers the question. Do not add explanations or restate the question. "
    "If the context truly contains no relevant information, give your best "
    "single-phrase guess based on what is in the context."
)


def _answer_prompt(memory_context: str, question: str) -> str:
    return f"Context:\n{memory_context}\n\nQuestion: {question}\n\nAnswer:"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:  # noqa: BLE001
        return None


class ExperimentRunner:
    """Orchestrates per-instance evaluation under both memory strategies."""

    def __init__(self, cfg: ExperimentConfig, client: Optional[LLMClient] = None) -> None:
        self.cfg = cfg
        self.client = client or build_client(cfg.backend)
        self.summ = SummarizationMemory(self.client, cfg.memory.summarization)
        self.rag = RagMemory(cfg.memory.rag)

    def run(
        self,
        instances: Iterable[dict],
        *,
        results_csv: Path,
        partial_csv: Optional[Path] = None,
        checkpoint_every: int = 3,
    ) -> List[InstanceResult]:
        """Run the experiment over *instances*, resuming from a partial CSV if present."""
        results_csv = Path(results_csv)
        partial_csv = (
            Path(partial_csv)
            if partial_csv
            else results_csv.with_name(results_csv.stem + "_partial.csv")
        )
        results_csv.parent.mkdir(parents=True, exist_ok=True)

        completed_ids: set[str] = set()
        rows: List[InstanceResult] = []

        if partial_csv.exists():
            with partial_csv.open("r", newline="", encoding="utf-8") as fh:
                for csv_row in csv.DictReader(fh):
                    rows.append(InstanceResult.model_validate(csv_row))
                    completed_ids.add(csv_row["instance_id"])
            log.info("Resuming from checkpoint: %d already done", len(completed_ids))

        instances_list = list(instances)
        total = len(instances_list)
        t_start = time.perf_counter()

        for idx, inst in enumerate(instances_list, start=1):
            iid = inst["instance_id"]
            if iid in completed_ids:
                continue

            elapsed = time.perf_counter() - t_start
            log.info(
                "[%d/%d] %s (%s, %s, ctx~%d tok) elapsed=%.0fs",
                idx,
                total,
                iid,
                inst["benchmark"],
                inst["length_category"],
                inst["context_tokens_approx"],
                elapsed,
            )

            row = self._run_one(inst)
            rows.append(row)

            if idx % checkpoint_every == 0 or idx == total:
                self._write_rows(partial_csv, rows)
                log.info("  checkpoint saved (%d/%d)", len(rows), total)

        self._write_rows(results_csv, rows)
        self._write_metadata(results_csv, len(rows))
        log.info("Experiment complete. %d instances saved to %s", len(rows), results_csv)
        return rows

    def _run_one(self, inst: dict) -> InstanceResult:
        context = inst["context"]
        question = inst["question"]
        answers = inst.get("all_answers", [inst["answer"]])
        budget = self.cfg.memory.context_budget_tokens
        cfg_dec = self.cfg.decoding

        summ_data = self._safe_strategy(self.summ.process, context, question, budget_tokens=budget)
        summ_ans = self._answer(summ_data.text, question)
        summ_metrics = self._score(summ_ans.text, answers)

        rag_data = self._safe_strategy(self.rag.process, context, question, budget_tokens=budget)
        rag_ans = self._answer(rag_data.text, question)
        rag_metrics = self._score(rag_ans.text, answers)

        return InstanceResult(
            instance_id=inst["instance_id"],
            benchmark=inst["benchmark"],
            task_type=inst["task_type"],
            context_tokens=int(inst["context_tokens_approx"]),
            length_category=inst["length_category"],
            ground_truth=inst["answer"],
            summ_answer=summ_ans.text[:500],
            summ_f1=summ_metrics["f1"],
            summ_em=summ_metrics["em"],
            summ_consistency=summ_metrics["consistency"],
            summ_contradiction=summ_metrics["contradiction"],
            summ_latency=summ_ans.latency_seconds,
            summ_prompt_tokens=summ_ans.prompt_tokens,
            summ_output_tokens=summ_ans.completion_tokens,
            summ_storage=summ_data.storage_bytes,
            summ_prep_time=summ_data.prep_seconds,
            rag_answer=rag_ans.text[:500],
            rag_f1=rag_metrics["f1"],
            rag_em=rag_metrics["em"],
            rag_consistency=rag_metrics["consistency"],
            rag_contradiction=rag_metrics["contradiction"],
            rag_latency=rag_ans.latency_seconds,
            rag_prompt_tokens=rag_ans.prompt_tokens,
            rag_output_tokens=rag_ans.completion_tokens,
            rag_storage=rag_data.storage_bytes,
            rag_prep_time=rag_data.prep_seconds,
        )

    def _answer(self, memory_context: str, question: str):
        cfg_dec = self.cfg.decoding
        return self.client.generate(
            _answer_prompt(memory_context, question),
            system=ANSWER_SYSTEM,
            max_tokens=cfg_dec.max_tokens,
            temperature=cfg_dec.temperature,
            top_p=cfg_dec.top_p,
            seed=cfg_dec.seed,
        )

    def _score(self, answer: str, references):
        thresh_cons = self.cfg.metrics.consistency_f1_threshold
        thresh_contra = self.cfg.metrics.contradiction_f1_threshold
        return {
            "f1": best_f1(answer, references),
            "em": best_em(answer, references),
            "consistency": consistency_indicator(answer, references, f1_threshold=thresh_cons),
            "contradiction": contradiction_indicator(
                answer, references, f1_threshold=thresh_contra
            ),
        }

    def _safe_strategy(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("Memory strategy failed: %s", e)
            from llm_memory_eval.memory.base import MemoryArtifact

            return MemoryArtifact(text="", storage_bytes=0, prep_seconds=0.0)

    def _write_rows(self, path: Path, rows: List[InstanceResult]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].model_dump().keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump())

    def _write_metadata(self, results_csv: Path, total: int) -> None:
        meta_dir = results_csv.parent / "logs"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta = RunMetadata(
            project_name=self.cfg.project_name,
            backend=self.cfg.backend.name,
            llm_model=self.cfg.backend.model,
            embedding_model=self.cfg.memory.rag.embedding_model,
            temperature=self.cfg.decoding.temperature,
            top_p=self.cfg.decoding.top_p,
            max_tokens=self.cfg.decoding.max_tokens,
            seed=self.cfg.decoding.seed,
            context_budget=self.cfg.memory.context_budget_tokens,
            chunk_size_words=self.cfg.memory.rag.chunk_size_words,
            chunk_overlap_words=self.cfg.memory.rag.chunk_overlap_words,
            rag_top_k=self.cfg.memory.rag.top_k,
            consistency_f1_threshold=self.cfg.metrics.consistency_f1_threshold,
            contradiction_f1_threshold=self.cfg.metrics.contradiction_f1_threshold,
            total_instances=total,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            commit=_git_commit(),
            package_version=__version__,
        )
        (meta_dir / "run_metadata.json").write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )
