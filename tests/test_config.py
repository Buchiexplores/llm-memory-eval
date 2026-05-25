"""Unit tests for configuration loading and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_memory_eval.config import ExperimentConfig


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "test.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.backend.name == "ollama"
        assert cfg.decoding.seed == 42
        assert cfg.memory.rag.embedding_model == "intfloat/e5-large-v2"

    def test_from_yaml(self, tmp_path: Path) -> None:
        body = """
project_name: "demo"
backend:
  name: "together"
  model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
decoding:
  max_tokens: 512
"""
        path = _write_yaml(tmp_path, body)
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.project_name == "demo"
        assert cfg.backend.name == "together"
        assert cfg.decoding.max_tokens == 512

    def test_env_override_backend(self, tmp_path: Path, monkeypatch) -> None:
        path = _write_yaml(tmp_path, "{}")
        monkeypatch.setenv("LLM_MEMORY_EVAL_BACKEND", "openai_compat")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.backend.name == "openai_compat"

    def test_env_override_seed(self, tmp_path: Path, monkeypatch) -> None:
        path = _write_yaml(tmp_path, "{}")
        monkeypatch.setenv("LLM_MEMORY_EVAL_SEED", "1234")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.decoding.seed == 1234

    def test_project_name_rejects_whitespace(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(project_name="bad name")
