"""Retrieval-Augmented Generation memory strategy.

Chunked context is encoded with the
``intfloat/e5-large-v2`` model, indexed with FAISS using inner-product
similarity on L2-normalised embeddings (equivalent to cosine similarity),
and the top-k passages are concatenated into the answer prompt up to the
configured token budget.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from llm_memory_eval.config import RagConfig
from llm_memory_eval.memory.base import MemoryArtifact, MemoryStrategy
from llm_memory_eval.utils.logging import get_logger
from llm_memory_eval.utils.tokens import count_tokens, truncate_to_tokens


log = get_logger(__name__)


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class RagMemory(MemoryStrategy):
    """Dense-retrieval memory backed by ``intfloat/e5-large-v2`` and FAISS."""

    name = "rag"

    def __init__(self, cfg: RagConfig | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.cfg = cfg or RagConfig()
        device = _select_device(self.cfg.embedding_device)
        log.info("Loading embedding model %s on device=%s", self.cfg.embedding_model, device)
        self.model = SentenceTransformer(self.cfg.embedding_model, device=device)

    def _chunk(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        step = self.cfg.chunk_size_words - self.cfg.chunk_overlap_words
        if step <= 0:
            raise ValueError("chunk_size_words must exceed chunk_overlap_words")
        start = 0
        while start < len(words):
            end = min(start + self.cfg.chunk_size_words, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 30:
                chunks.append(chunk)
            if end == len(words):
                break
            start += step
        return chunks

    def process(
        self,
        context: str,
        question: str,
        *,
        budget_tokens: int,
    ) -> MemoryArtifact:
        import faiss

        t0 = time.perf_counter()
        chunks = self._chunk(context)
        if not chunks:
            text = truncate_to_tokens(context, budget_tokens)
            return MemoryArtifact(
                text=text,
                storage_bytes=len(text.encode("utf-8")),
                prep_seconds=time.perf_counter() - t0,
            )

        passages = [f"passage: {c}" for c in chunks]
        query = f"query: {question}"

        chunk_embeddings = self.model.encode(
            passages,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
        k = min(self.cfg.top_k * 2, len(chunks))
        _, idxs = index.search(query_embedding, k)

        retrieved: List[str] = []
        tok_count = 0
        for i in idxs[0]:
            if i < 0 or i >= len(chunks):
                continue
            chunk_tokens = count_tokens(chunks[i])
            if tok_count + chunk_tokens <= budget_tokens:
                retrieved.append(chunks[i])
                tok_count += chunk_tokens
            if len(retrieved) >= self.cfg.top_k:
                break

        text = "\n\n".join(retrieved)
        emb_bytes = chunk_embeddings.nbytes
        chunk_bytes = sum(len(c.encode("utf-8")) for c in chunks)
        meta_bytes = sum(len(str(i).encode("utf-8")) for i in range(len(chunks)))
        storage = emb_bytes + chunk_bytes + meta_bytes
        prep = time.perf_counter() - t0

        return MemoryArtifact(text=text, storage_bytes=storage, prep_seconds=prep)
