"""Unit tests for the length-bucket assignment rules from Chapter 3."""

from __future__ import annotations

import pytest

from llm_memory_eval.data.length_buckets import assign_length_bucket


class TestTokenBenchmarks:
    @pytest.mark.parametrize("benchmark", ["LongBench", "LongMemEval"])
    def test_short_threshold(self, benchmark: str) -> None:
        assert assign_length_bucket(benchmark=benchmark, token_count=2000) == "short"
        assert assign_length_bucket(benchmark=benchmark, token_count=4000) == "short"

    @pytest.mark.parametrize("benchmark", ["LongBench", "LongMemEval"])
    def test_medium_threshold(self, benchmark: str) -> None:
        assert assign_length_bucket(benchmark=benchmark, token_count=4001) == "medium"
        assert assign_length_bucket(benchmark=benchmark, token_count=16000) == "medium"

    @pytest.mark.parametrize("benchmark", ["LongBench", "LongMemEval"])
    def test_long_threshold(self, benchmark: str) -> None:
        assert assign_length_bucket(benchmark=benchmark, token_count=16001) == "long"
        assert assign_length_bucket(benchmark=benchmark, token_count=120000) == "long"

    def test_missing_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            assign_length_bucket(benchmark="LongBench")


class TestLoCoMo:
    def test_short_by_sessions(self) -> None:
        assert assign_length_bucket(benchmark="LoCoMo", session_count=2) == "short"

    def test_medium_by_sessions(self) -> None:
        assert assign_length_bucket(benchmark="LoCoMo", session_count=6) == "medium"

    def test_long_by_sessions(self) -> None:
        assert assign_length_bucket(benchmark="LoCoMo", session_count=10) == "long"

    def test_short_by_turns(self) -> None:
        assert assign_length_bucket(benchmark="LoCoMo", turn_count=15) == "short"

    def test_long_by_turns(self) -> None:
        assert assign_length_bucket(benchmark="LoCoMo", turn_count=120) == "long"

    def test_takes_broader_bucket(self) -> None:
        # 3 sessions = short by session rule
        # 80 turns  = long by turn rule
        # Combined assignment should be long.
        bucket = assign_length_bucket(benchmark="LoCoMo", session_count=3, turn_count=80)
        assert bucket == "long"

    def test_requires_session_or_turn(self) -> None:
        with pytest.raises(ValueError):
            assign_length_bucket(benchmark="LoCoMo")
