"""Conversational consistency and contradiction indicators.

Both indicators are derived from the per-instance F1 score against the
benchmark references combined with an abstention regex. The thresholds
are configurable via :class:`~llm_memory_eval.config.MetricsConfig`.
"""

from __future__ import annotations

import re
from typing import Sequence

from llm_memory_eval.metrics.text import best_em, best_f1


_ABSTENTION_PATTERNS = [
    r"\bunknown\b",
    r"\bcannot\s+(be\s+)?determined?\b",
    r"\bcannot\s+answer\b",
    r"\bnot\s+(mentioned|provided|specified|stated|given|available|in\s+the\s+context)\b",
    r"\bno\s+information\b",
    r"\binsufficient\s+(information|context)\b",
    r"\bthere\s+is\s+no\b.*\b(information|mention|reference)\b",
    r"\bi\s+(don'?t|do\s+not)\s+know\b",
]
_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), flags=re.IGNORECASE)


def is_abstention(prediction: str) -> bool:
    """Return True if *prediction* matches any abstention phrase."""
    if not prediction:
        return False
    return _ABSTENTION_RE.search(prediction) is not None


def consistency_indicator(
    prediction: str,
    references: Sequence[str],
    *,
    f1_threshold: float = 0.30,
) -> float:
    """Return 1.0 when the model's response aligns with at least one reference.

    Alignment is satisfied by an exact match OR by token-level F1 at or
    above ``f1_threshold``. The same indicator is computed for both memory
    strategy conditions on the same item, so it is a valid paired measure
    even though it depends on a threshold.
    """
    if best_em(prediction, references) > 0:
        return 1.0
    return 1.0 if best_f1(prediction, references) >= f1_threshold else 0.0


def contradiction_indicator(
    prediction: str,
    references: Sequence[str],
    *,
    f1_threshold: float = 0.30,
) -> float:
    """Return 1.0 when the model returns a substantive but disagreeing answer.

    A contradiction requires (a) a non-empty prediction, (b) the prediction
    is not an abstention, (c) no exact match, and (d) the best F1 is below
    ``f1_threshold``.
    """
    if not prediction or not prediction.strip():
        return 0.0
    if is_abstention(prediction):
        return 0.0
    if best_em(prediction, references) > 0:
        return 0.0
    return 1.0 if best_f1(prediction, references) < f1_threshold else 0.0
