"""Recursive abstractive summarization implemented via the base LLM.

The same LLM used for response
generation produces an evolving rolling summary that is reintroduced into
the prompt at answer time. To bound LLM call counts on long contexts we
limit the number of chunks; when the context exceeds ``max_chunks *
chunk_tokens``, a stratified sample (first chunk, evenly spaced middle
chunks, last chunk) is used so the rolling summary still represents the
entire document.
"""

from __future__ import annotations

import time

import numpy as np

from llm_memory_eval.config import SummarizationConfig
from llm_memory_eval.llm.base import LLMClient
from llm_memory_eval.memory.base import MemoryArtifact, MemoryStrategy
from llm_memory_eval.utils.tokens import (
    count_tokens,
    decode_tokens,
    encode_tokens,
    truncate_to_tokens,
)

SUMMARY_SYSTEM = (
    "You are a careful conversational memory module. Given prior context "
    "and a user question, produce a faithful, compact summary that retains "
    "all facts, entities, dates, numbers, named items, user preferences, "
    "and statements necessary to answer the question. Do not answer the "
    "question. Output only the summary as plain prose. Keep it under the "
    "requested length."
)


class SummarizationMemory(MemoryStrategy):
    """Recursive summarization memory implemented via the base LLM."""

    name = "summarization"

    def __init__(self, client: LLMClient, cfg: SummarizationConfig | None = None) -> None:
        self.client = client
        self.cfg = cfg or SummarizationConfig()

    def _chunk(self, text: str) -> list[str]:
        ids = encode_tokens(text)
        total = len(ids)
        chunk_tokens = self.cfg.chunk_tokens

        if total <= chunk_tokens:
            return [text]

        starts = list(range(0, total, chunk_tokens))
        if len(starts) <= self.cfg.max_chunks:
            return [decode_tokens(ids[s : s + chunk_tokens]) for s in starts]

        chosen = [0, starts[-1]]
        n_middle = self.cfg.max_chunks - 2
        if n_middle > 0:
            mid = np.linspace(starts[1], starts[-2], num=n_middle, dtype=int).tolist()
            chosen = sorted(set(chosen + mid))
        return [decode_tokens(ids[s : s + chunk_tokens]) for s in chosen]

    def process(
        self,
        context: str,
        question: str,
        *,
        budget_tokens: int,
    ) -> MemoryArtifact:
        t0 = time.perf_counter()
        chunks = self._chunk(context)
        current_summary = ""

        for ch in chunks:
            prompt = (
                f"Question that will eventually be asked:\n{question}\n\n"
                f"Existing rolling summary so far (may be empty):\n"
                f"{current_summary if current_summary else '[none]'}\n\n"
                f"New context chunk to integrate:\n{ch}\n\n"
                f"Update the rolling summary so it stays under "
                f"{self.cfg.per_chunk_output_tokens} words and retains all "
                f"facts, names, numbers, dates, and user statements relevant "
                f"to the eventual question. Output only the updated summary."
            )
            response = self.client.generate(
                prompt,
                system=SUMMARY_SYSTEM,
                max_tokens=self.cfg.per_chunk_output_tokens * 2,
                temperature=0.0,
                top_p=1.0,
                seed=42,
            )
            if response.text.strip():
                current_summary = response.text.strip()

        current_summary = truncate_to_tokens(current_summary, budget_tokens)
        prep = time.perf_counter() - t0
        storage = len(current_summary.encode("utf-8"))

        # Token count is computed for diagnostics even though it is not returned.
        _ = count_tokens(current_summary)

        return MemoryArtifact(
            text=current_summary,
            storage_bytes=storage,
            prep_seconds=prep,
        )
