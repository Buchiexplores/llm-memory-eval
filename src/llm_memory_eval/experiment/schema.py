"""Pydantic models for experiment outputs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InstanceResult(BaseModel):
    """One row of the experiment results CSV.

    Each row pairs a single benchmark instance with both memory strategy
    conditions, supporting the within-instance paired analyses.
    """

    instance_id: str
    benchmark: str
    task_type: str
    context_tokens: int
    length_category: str
    ground_truth: str

    summ_answer: str = ""
    summ_f1: float = 0.0
    summ_em: float = 0.0
    summ_consistency: float = 0.0
    summ_contradiction: float = 0.0
    summ_latency: float = 0.0
    summ_prompt_tokens: int = 0
    summ_output_tokens: int = 0
    summ_storage: int = 0
    summ_prep_time: float = 0.0

    rag_answer: str = ""
    rag_f1: float = 0.0
    rag_em: float = 0.0
    rag_consistency: float = 0.0
    rag_contradiction: float = 0.0
    rag_latency: float = 0.0
    rag_prompt_tokens: int = 0
    rag_output_tokens: int = 0
    rag_storage: int = 0
    rag_prep_time: float = 0.0


class RunMetadata(BaseModel):
    """Reproducibility metadata for a single experiment run."""

    project_name: str
    backend: str
    llm_model: str
    embedding_model: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    context_budget: int
    chunk_size_words: int
    chunk_overlap_words: int
    rag_top_k: int
    consistency_f1_threshold: float
    contradiction_f1_threshold: float
    total_instances: int
    timestamp: str
    commit: Optional[str] = Field(default=None, description="git commit SHA if available")
    package_version: str = ""
