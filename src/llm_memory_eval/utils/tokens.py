"""Token-counting and truncation helpers.

The package uses :mod:`tiktoken` with the ``cl100k_base`` encoding for all
token-budget accounting. Different model families (Llama, OpenAI, Anthropic)
use different tokenisers, but ``cl100k_base`` gives a stable, well-tested
estimate that is sufficient for *internal* budget control. The two memory
strategies share this estimator, so the comparison stays internally
consistent across backends.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* under the ``cl100k_base`` encoding."""
    if not text:
        return 0
    return len(_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens.

    Returns the original string if it is already short enough.
    """
    if max_tokens <= 0:
        return ""
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def decode_tokens(token_ids: list[int]) -> str:
    """Inverse of :func:`count_tokens` for testing utilities."""
    return _encoding().decode(token_ids)


def encode_tokens(text: str, *, max_tokens: Optional[int] = None) -> list[int]:
    """Return the token ids for *text*, optionally truncated to *max_tokens*."""
    ids = _encoding().encode(text)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids
