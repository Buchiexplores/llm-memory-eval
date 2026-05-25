"""Abstract LLM client interface implemented by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    """A single completion returned by an :class:`LLMClient`."""

    text: str = Field(description="Generated text, stripped of surrounding whitespace.")
    latency_seconds: float = Field(ge=0.0, description="Wall-clock latency in seconds.")
    prompt_tokens: int = Field(ge=0, description="Tokens consumed by the prompt.")
    completion_tokens: int = Field(ge=0, description="Tokens produced by the model.")


class LLMClient(ABC):
    """Backend-agnostic generation interface.

    Concrete subclasses must implement :meth:`generate`. Every call passes
    the full decoding configuration explicitly so callers can vary
    parameters across calls (for example, a shorter ``max_tokens`` for
    summarization passes and a longer one for the final answer).
    """

    name: str

    @abstractmethod
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
        """Generate a completion."""
