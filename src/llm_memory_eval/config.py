"""Pydantic configuration model loaded from YAML.

Configurations select the LLM backend, the embedding model, decoding
parameters, memory-strategy hyperparameters, and statistical-analysis
options. Environment variables with the prefix ``LLM_MEMORY_EVAL_`` override
the YAML values, which makes it easy to switch between local pilot and
cloud production runs without editing files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

BackendName = Literal["ollama", "together", "openai_compat", "bedrock", "hf_endpoint"]


class BackendConfig(BaseModel):
    """LLM backend selection and per-backend connection details."""

    name: BackendName = "ollama"
    model: str = "llama3.1:8b"
    base_url: str | None = None
    api_key_env: str | None = None
    region: str | None = None
    keep_alive: str = "30m"
    request_timeout_seconds: int = 600
    max_retries: int = 3
    num_ctx: int = 4096  # only used by Ollama


class DecodingConfig(BaseModel):
    """Generation hyperparameters shared by both memory-strategy conditions."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 42


class SummarizationConfig(BaseModel):
    """Hyperparameters for the recursive summarization memory strategy."""

    max_chunks: int = 2
    chunk_tokens: int = 1800
    per_chunk_output_tokens: int = 300
    summary_budget_tokens: int = 1000


class RagConfig(BaseModel):
    """Hyperparameters for the dense-retrieval memory strategy."""

    embedding_model: str = "intfloat/e5-large-v2"
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    top_k: int = 8
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class MemoryConfig(BaseModel):
    context_budget_tokens: int = 3000
    summarization: SummarizationConfig = SummarizationConfig()
    rag: RagConfig = RagConfig()


class MetricsConfig(BaseModel):
    """Thresholds for the conversational-consistency and contradiction indicators."""

    consistency_f1_threshold: float = 0.30
    contradiction_f1_threshold: float = 0.30


class AnalysisConfig(BaseModel):
    alpha: float = 0.05
    multiple_comparison_correction: Literal["holm", "bonferroni"] = "holm"


class PathsConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    results: Path = Path("results")
    figures: Path = Path("results/figures")
    tables: Path = Path("results/tables")
    dissertation: Path = Path("dissertation")


class ExperimentConfig(BaseModel):
    """Top-level pipeline configuration."""

    project_name: str = "llm-memory-eval"
    backend: BackendConfig = BackendConfig()
    decoding: DecodingConfig = DecodingConfig()
    memory: MemoryConfig = MemoryConfig()
    metrics: MetricsConfig = MetricsConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    paths: PathsConfig = PathsConfig()
    log_level: str = "INFO"

    @field_validator("project_name")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        if any(c.isspace() for c in v):
            raise ValueError("project_name must not contain whitespace")
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load configuration from a YAML file and apply environment overrides."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = cls(**raw)
        cfg = _apply_env_overrides(cfg)
        return cfg


def _apply_env_overrides(cfg: ExperimentConfig) -> ExperimentConfig:
    """Apply ``LLM_MEMORY_EVAL_*`` overrides on top of *cfg*.

    Supported overrides:
    ``LLM_MEMORY_EVAL_BACKEND``, ``LLM_MEMORY_EVAL_MODEL``,
    ``LLM_MEMORY_EVAL_SEED``, ``LLM_MEMORY_EVAL_LOG_LEVEL``.
    """
    backend = os.environ.get("LLM_MEMORY_EVAL_BACKEND")
    if backend:
        cfg.backend.name = backend  # type: ignore[assignment]

    model = os.environ.get("LLM_MEMORY_EVAL_MODEL")
    if model:
        cfg.backend.model = model

    seed = os.environ.get("LLM_MEMORY_EVAL_SEED")
    if seed:
        cfg.decoding.seed = int(seed)

    log_level = os.environ.get("LLM_MEMORY_EVAL_LOG_LEVEL")
    if log_level:
        cfg.log_level = log_level
    return cfg
