"""Benchmark-specific length-category thresholds.

LongBench and LongMemEval use total input token count; LoCoMo uses
session and turn counts because the benchmark is structured around
multi-session dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


LengthBucket = Literal["short", "medium", "long"]


@dataclass(frozen=True)
class TokenThresholds:
    short_max: int = 4_000
    medium_max: int = 16_000  # short_max < tokens <= medium_max => "medium"


@dataclass(frozen=True)
class LoCoMoThresholds:
    short_max_sessions: int = 3
    short_max_turns: int = 20
    medium_max_sessions: int = 8
    medium_max_turns: int = 60


TOKEN_DEFAULTS = TokenThresholds()
LOCOMO_DEFAULTS = LoCoMoThresholds()


def assign_length_bucket(
    *,
    benchmark: str,
    token_count: Optional[int] = None,
    session_count: Optional[int] = None,
    turn_count: Optional[int] = None,
    token_thresholds: TokenThresholds = TOKEN_DEFAULTS,
    locomo_thresholds: LoCoMoThresholds = LOCOMO_DEFAULTS,
) -> LengthBucket:
    """Return the length category for a benchmark instance.

    Parameters
    ----------
    benchmark
        One of ``"LongBench"``, ``"LongMemEval"``, or ``"LoCoMo"``.
    token_count
        Total input tokens for the instance (required for the token-based
        benchmarks).
    session_count, turn_count
        Required for LoCoMo. The bucket is the *broader* (more demanding)
        of the two assignments.
    """
    bench = benchmark.lower()

    if bench == "locomo":
        if session_count is None and turn_count is None:
            raise ValueError("LoCoMo requires session_count or turn_count")
        bucket_session = _locomo_session_bucket(session_count, locomo_thresholds)
        bucket_turn = _locomo_turn_bucket(turn_count, locomo_thresholds)
        return _max_bucket(bucket_session, bucket_turn)

    if token_count is None:
        raise ValueError(f"{benchmark} requires token_count")
    if token_count <= token_thresholds.short_max:
        return "short"
    if token_count <= token_thresholds.medium_max:
        return "medium"
    return "long"


def _locomo_session_bucket(
    session_count: Optional[int],
    thresholds: LoCoMoThresholds,
) -> Optional[LengthBucket]:
    if session_count is None:
        return None
    if session_count <= thresholds.short_max_sessions:
        return "short"
    if session_count <= thresholds.medium_max_sessions:
        return "medium"
    return "long"


def _locomo_turn_bucket(
    turn_count: Optional[int],
    thresholds: LoCoMoThresholds,
) -> Optional[LengthBucket]:
    if turn_count is None:
        return None
    if turn_count <= thresholds.short_max_turns:
        return "short"
    if turn_count <= thresholds.medium_max_turns:
        return "medium"
    return "long"


_ORDER = {"short": 0, "medium": 1, "long": 2}


def _max_bucket(
    a: Optional[LengthBucket],
    b: Optional[LengthBucket],
) -> LengthBucket:
    if a is None and b is None:
        raise ValueError("at least one bucket must be supplied")
    if a is None:
        return b  # type: ignore[return-value]
    if b is None:
        return a
    return a if _ORDER[a] >= _ORDER[b] else b
