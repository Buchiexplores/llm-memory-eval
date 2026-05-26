"""Unit tests for the text and conversational metrics."""

from __future__ import annotations

import pytest

from llm_memory_eval.metrics.conversational import (
    consistency_indicator,
    contradiction_indicator,
    is_abstention,
)
from llm_memory_eval.metrics.text import (
    best_em,
    best_f1,
    compute_em,
    compute_f1,
    normalize,
)


class TestNormalize:
    def test_lowercases(self) -> None:
        assert normalize("Hello WORLD") == "hello world"

    def test_strips_articles(self) -> None:
        assert normalize("The dog ate a bone") == "dog ate bone"

    def test_collapses_whitespace(self) -> None:
        assert normalize("  hello   world  ") == "hello world"

    def test_removes_punctuation(self) -> None:
        assert normalize("Hello, world!") == "hello world"

    def test_empty_input(self) -> None:
        assert normalize("") == ""

    def test_integer_input_is_coerced(self) -> None:
        # Benchmark reference answers are sometimes ints (e.g. a year).
        assert normalize(2022) == "2022"

    def test_float_input_is_coerced(self) -> None:
        assert normalize(3.5) == "35"

    def test_none_input(self) -> None:
        assert normalize(None) == ""


class TestNonStringReferences:
    def test_f1_with_integer_reference(self) -> None:
        assert compute_f1("the answer is 2022", 2022) > 0.0

    def test_em_with_integer_reference(self) -> None:
        assert compute_em("2022", 2022) == 1.0

    def test_best_f1_with_mixed_reference_types(self) -> None:
        assert best_f1("2022", [2022, "something else"]) == 1.0


class TestF1:
    def test_identical(self) -> None:
        assert compute_f1("paris", "paris") == 1.0

    def test_no_overlap(self) -> None:
        assert compute_f1("paris", "london") == 0.0

    def test_partial_overlap(self) -> None:
        # "the cat sat" vs "the cat ran" -> after normalize: "cat sat" / "cat ran"
        score = compute_f1("the cat sat", "the cat ran")
        assert 0.4 < score < 0.6

    def test_empty_prediction(self) -> None:
        assert compute_f1("", "anything") == 0.0


class TestEM:
    def test_identical(self) -> None:
        assert compute_em("Charles", "Charles") == 1.0

    def test_different(self) -> None:
        assert compute_em("Mary", "Charles") == 0.0

    def test_empty_strings(self) -> None:
        assert compute_em("", "") == 0.0


class TestBestOverReferences:
    def test_best_f1(self) -> None:
        assert best_f1("paris", ["london", "paris"]) == 1.0

    def test_best_em(self) -> None:
        assert best_em("Paris", ["london", "PARIS"]) == 1.0

    def test_best_f1_empty_list(self) -> None:
        assert best_f1("paris", []) == 0.0


class TestAbstention:
    @pytest.mark.parametrize(
        "phrase",
        [
            "unknown",
            "The answer is unknown.",
            "I don't know.",
            "Not mentioned in the context.",
            "No information available.",
            "Cannot be determined.",
        ],
    )
    def test_detects_abstention(self, phrase: str) -> None:
        assert is_abstention(phrase)

    def test_substantive_answer_is_not_abstention(self) -> None:
        assert not is_abstention("Paris is the capital of France.")


class TestConsistencyIndicator:
    def test_high_f1_marks_consistent(self) -> None:
        assert consistency_indicator("paris", ["paris"], f1_threshold=0.30) == 1.0

    def test_low_f1_marks_inconsistent(self) -> None:
        assert consistency_indicator("london", ["paris"], f1_threshold=0.30) == 0.0

    def test_exact_match_overrides_threshold(self) -> None:
        assert consistency_indicator("PARIS", ["paris"], f1_threshold=0.99) == 1.0


class TestContradictionIndicator:
    def test_disagreement_counts(self) -> None:
        assert contradiction_indicator("london", ["paris"], f1_threshold=0.30) == 1.0

    def test_abstention_does_not_count(self) -> None:
        assert contradiction_indicator("unknown", ["paris"], f1_threshold=0.30) == 0.0

    def test_empty_does_not_count(self) -> None:
        assert contradiction_indicator("", ["paris"], f1_threshold=0.30) == 0.0

    def test_correct_does_not_count(self) -> None:
        assert contradiction_indicator("paris", ["paris"], f1_threshold=0.30) == 0.0
