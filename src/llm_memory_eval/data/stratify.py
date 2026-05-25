"""Stratified subsampling for length-balanced experiment runs."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, List, Mapping, MutableMapping, Sequence


def stratified_subsample(
    instances: Sequence[Mapping[str, Any]],
    *,
    per_length: int = 30,
    seed: int = 42,
    length_key: str = "length_category",
    benchmark_key: str = "benchmark",
) -> List[Mapping[str, Any]]:
    """Return a stratified subset balanced across length categories.

    Within each length category the function spreads picks across the
    available benchmarks before topping up from a remaining pool. The
    output preserves the cell counts (``per_length`` per category) and is
    deterministic given *seed*.
    """
    if per_length <= 0:
        raise ValueError("per_length must be positive")

    rng = random.Random(seed)
    by_length: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for inst in instances:
        by_length[str(inst[length_key])].append(inst)

    selected: List[Mapping[str, Any]] = []
    for length, pool in by_length.items():
        pool_copy = list(pool)
        rng.shuffle(pool_copy)
        by_bench: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for item in pool_copy:
            by_bench[str(item[benchmark_key])].append(item)

        per_bench = max(1, per_length // max(1, len(by_bench)))
        chosen: List[Mapping[str, Any]] = []
        for bench_items in by_bench.values():
            chosen.extend(bench_items[:per_bench])

        if len(chosen) < per_length:
            seen_ids = {id(x) for x in chosen}
            extras = [x for x in pool_copy if id(x) not in seen_ids]
            chosen.extend(extras[: per_length - len(chosen)])

        selected.extend(chosen[:per_length])
    return selected
