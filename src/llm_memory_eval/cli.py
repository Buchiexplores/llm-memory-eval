"""Typer-based command-line interface.

Examples
--------
.. code-block:: console

   $ llm-memory-eval download-data
   $ llm-memory-eval prepare-data
   $ llm-memory-eval run --config configs/cloud-production.yaml
   $ llm-memory-eval analyze
   $ llm-memory-eval figures
   $ llm-memory-eval all --config configs/cloud-production.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from llm_memory_eval import __version__
from llm_memory_eval.config import ExperimentConfig
from llm_memory_eval.utils.logging import configure_logging, get_logger
from llm_memory_eval.utils.seed import set_global_seed

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Reproducible evaluation harness for Summarization-Based Memory vs "
        "Retrieval-Augmented Generation in long-context LLMs."
    ),
)
log = get_logger(__name__)


def _load_cfg(config_path: Path | None) -> ExperimentConfig:
    if config_path is None:
        config_path = Path("configs/default.yaml")
    return ExperimentConfig.from_yaml(config_path)


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command("download-data")
def download_data(
    output: Path = typer.Option(
        Path("data/raw"), "--output", "-o", help="Directory to write the raw datasets to."
    ),
) -> None:
    """Download LongBench, LoCoMo, and LongMemEval from the Hugging Face Hub."""
    from llm_memory_eval.data.download import download_datasets

    download_datasets(output)


@app.command("prepare-data")
def prepare_data(
    raw: Path = typer.Option(Path("data/raw"), "--raw"),
    output: Path = typer.Option(Path("data/processed"), "--output", "-o"),
) -> None:
    """Convert raw benchmark dumps into a unified instances JSON."""
    from llm_memory_eval.data.prepare import prepare_all

    prepare_all(raw, output)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    instances: Path = typer.Option(Path("data/processed/all_instances.json"), "--instances", "-i"),
    output: Path = typer.Option(Path("results"), "--output", "-o"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Run the experiment over every prepared benchmark instance."""
    cfg = _load_cfg(config)
    if seed is not None:
        cfg.decoding.seed = seed
    set_global_seed(cfg.decoding.seed)

    from llm_memory_eval.experiment.runner import ExperimentRunner

    instances_data = json.loads(instances.read_text(encoding="utf-8"))
    runner = ExperimentRunner(cfg)
    runner.run(instances_data, results_csv=output / "experiment_results.csv")


@app.command()
def analyze(
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
) -> None:
    """Run the full statistical analysis on the experiment results CSV."""
    from llm_memory_eval.analysis.pipeline import run_full_analysis

    run_full_analysis(results_dir)


@app.command()
def figures(
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
) -> None:
    """Generate the publication figures from the saved analyses."""
    from llm_memory_eval.reporting.figures import generate_figures

    generate_figures(results_dir)


@app.command("all")
def run_all(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    raw: Path = typer.Option(Path("data/raw"), "--raw"),
    processed: Path = typer.Option(Path("data/processed"), "--processed"),
    results: Path = typer.Option(Path("results"), "--results"),
) -> None:
    """Run the full pipeline: download, prepare, run, analyze, figures."""
    download_data(output=raw)
    prepare_data(raw=raw, output=processed)
    run(config=config, instances=processed / "all_instances.json", output=results, seed=None)
    analyze(results_dir=results)
    figures(results_dir=results)


if __name__ == "__main__":
    app()
