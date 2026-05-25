"""Benchmark dataset download.

Downloads the LongBench, LoCoMo, and LongMemEval datasets from their
canonical Hugging Face Hub locations. Datasets are not redistributed with
this package; consumers must accept the upstream licences themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from llm_memory_eval.utils.logging import get_logger


log = get_logger(__name__)


DATASETS: list[dict[str, str]] = [
    {
        "name": "LongBench",
        "hf_repo": "THUDM/LongBench",
        "license": "MIT (see upstream)",
    },
    {
        "name": "LoCoMo",
        "hf_repo": "snap-research/locomo",
        "license": "Apache-2.0 (see upstream)",
    },
    {
        "name": "LongMemEval",
        "hf_repo": "xiaowu0162/LongMemEval",
        "license": "Apache-2.0 (see upstream)",
    },
]


def download_datasets(output_dir: Path, datasets: Iterable[dict[str, str]] = DATASETS) -> None:
    """Download each benchmark to ``output_dir/<name>/``.

    Uses the ``datasets`` library when available; otherwise instructs the
    caller to clone the repository manually.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to download benchmarks. "
            "Install it via: pip install datasets"
        ) from exc

    for ds in datasets:
        name = ds["name"]
        repo = ds["hf_repo"]
        log.info("Downloading %s from %s ...", name, repo)
        try:
            data = load_dataset(repo)
            local = output_dir / name
            local.mkdir(parents=True, exist_ok=True)
            for split, rows in data.items():
                target = local / f"{split}.jsonl"
                with target.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.append({"name": name, "repo": repo, "path": str(local)})
            log.info("  saved -> %s", local)
        except Exception as e:  # noqa: BLE001
            log.warning("  failed to download %s: %s", name, e)
            manifest.append({"name": name, "repo": repo, "error": str(e)})

    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("Download manifest written to %s/download_manifest.json", output_dir)
