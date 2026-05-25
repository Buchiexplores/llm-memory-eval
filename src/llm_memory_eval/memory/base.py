"""Abstract memory strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryArtifact:
    """Result of running a memory strategy on a single instance.

    Attributes
    ----------
    text
        The context text that will be supplied to the LLM at answer time.
    storage_bytes
        Persistent storage required to maintain the memory artifact
        (serialized summary text for summarization; FAISS index + chunks +
        metadata for RAG).
    prep_seconds
        Wall-clock seconds spent preparing the artifact.
    """

    text: str
    storage_bytes: int
    prep_seconds: float


class MemoryStrategy(ABC):
    """Strategy that converts raw context into a model-ready memory artifact."""

    name: str

    @abstractmethod
    def process(
        self,
        context: str,
        question: str,
        *,
        budget_tokens: int,
    ) -> MemoryArtifact:
        """Produce a memory artifact for *context* given *question*."""
