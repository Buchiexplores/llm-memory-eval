"""Deterministic exact-match and token-level F1 metrics.

The normalisation follows the standard SQuAD/LongBench convention used by
all three benchmarks: lowercase, strip surrounding punctuation, collapse
whitespace, and remove the English articles ``a``, ``an``, and ``the``.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Apply SQuAD-style answer normalisation."""
    if not text:
        return ""
    lowered = text.lower()
    no_articles = _ARTICLES.sub(" ", lowered)
    no_punct = "".join(ch for ch in no_articles if ch not in string.punctuation)
    return " ".join(no_punct.split())


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and a single reference."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_em(prediction: str, reference: str) -> float:
    """Exact match indicator (1.0 if equal after normalisation, else 0.0)."""
    norm = normalize(prediction)
    return 1.0 if norm and norm == normalize(reference) else 0.0


def best_f1(prediction: str, references: Sequence[str]) -> float:
    """Maximum token-level F1 over multiple reference answers."""
    return max((compute_f1(prediction, r) for r in references), default=0.0)


def best_em(prediction: str, references: Sequence[str]) -> float:
    """Maximum exact-match indicator over multiple reference answers."""
    return max((compute_em(prediction, r) for r in references), default=0.0)
