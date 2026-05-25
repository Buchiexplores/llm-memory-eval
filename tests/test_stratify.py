"""Unit tests for the stratified subsampling utility."""

from __future__ import annotations

from collections import Counter

from llm_memory_eval.data.stratify import stratified_subsample


def _build_instances() -> list[dict]:
    items = []
    for cat in ("short", "medium", "long"):
        for bench in ("LongBench", "LoCoMo", "LongMemEval"):
            for i in range(40):
                items.append(
                    {
                        "instance_id": f"{bench}_{cat}_{i:03d}",
                        "benchmark": bench,
                        "length_category": cat,
                    }
                )
    return items


class TestStratifiedSubsample:
    def test_per_length_count(self) -> None:
        instances = _build_instances()
        out = stratified_subsample(instances, per_length=30, seed=42)
        counts = Counter(i["length_category"] for i in out)
        assert counts["short"] == 30
        assert counts["medium"] == 30
        assert counts["long"] == 30

    def test_deterministic(self) -> None:
        instances = _build_instances()
        a = stratified_subsample(instances, per_length=30, seed=42)
        b = stratified_subsample(instances, per_length=30, seed=42)
        assert [i["instance_id"] for i in a] == [i["instance_id"] for i in b]

    def test_spreads_across_benchmarks(self) -> None:
        instances = _build_instances()
        out = stratified_subsample(instances, per_length=30, seed=42)
        benches = Counter(i["benchmark"] for i in out)
        assert min(benches.values()) >= 5  # roughly balanced

    def test_rejects_non_positive(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            stratified_subsample(_build_instances(), per_length=0)
