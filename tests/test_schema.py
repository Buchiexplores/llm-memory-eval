"""Unit tests for experiment result schema coercion."""

from __future__ import annotations

from llm_memory_eval.experiment.schema import InstanceResult


def _base(**overrides) -> dict:
    data = {
        "instance_id": "LC_00_001",
        "benchmark": "LoCoMo",
        "task_type": "multi-hop",
        "context_tokens": 10961,
        "length_category": "medium",
        "ground_truth": "Paris",
    }
    data.update(overrides)
    return data


class TestInstanceResultCoercion:
    def test_integer_ground_truth_is_coerced(self) -> None:
        # Benchmark answers are sometimes integers (e.g. a year).
        row = InstanceResult(**_base(ground_truth=2022))
        assert row.ground_truth == "2022"
        assert isinstance(row.ground_truth, str)

    def test_integer_answers_are_coerced(self) -> None:
        row = InstanceResult(**_base(summ_answer=2022, rag_answer=1999))
        assert row.summ_answer == "2022"
        assert row.rag_answer == "1999"

    def test_none_answer_becomes_empty_string(self) -> None:
        row = InstanceResult(**_base(summ_answer=None))
        assert row.summ_answer == ""

    def test_normal_string_unchanged(self) -> None:
        row = InstanceResult(**_base(ground_truth="Charles"))
        assert row.ground_truth == "Charles"
