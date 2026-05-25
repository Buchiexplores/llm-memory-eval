"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_memory_eval.llm.base import GenerateResponse, LLMClient


class StubLLMClient(LLMClient):
    """LLMClient that returns a deterministic, scripted response."""

    name = "stub"

    def __init__(self, text: str = "stub answer") -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> GenerateResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }
        )
        return GenerateResponse(
            text=self.text,
            latency_seconds=0.001,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.text.split()),
        )


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
