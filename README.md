# llm-memory-eval

[![CI](https://github.com/Buchiexplores/llm-memory-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Buchiexplores/llm-memory-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-pending%20Zenodo%20release-blue.svg)](https://zenodo.org/)

**Author:** Abuchi Okeke · **Affiliation:** University of the Cumberlands, Williamsburg, Kentucky, USA · **Contact:** okekeag@gmail.com

> Reproducible comparative evaluation of **Summarization-Based Memory** and
> **Retrieval-Augmented Generation** for long-term conversational
> performance in large language models, on three long-context benchmarks
> (LongBench, LoCoMo, LongMemEval).

`llm-memory-eval` is an experiment-running harness for comparing
long-term memory strategies in large language models under controlled,
reproducible conditions. It downloads and prepares the benchmark
instances, runs both memory-strategy conditions on a single base model,
computes the outcome metrics, executes the statistical analyses, and
renders publication-quality figures.

## Abstract

Large language models often forget earlier facts, contradict prior
statements, and lose user-specific context as conversations extend across
many turns or multiple sessions. Two memory-augmentation strategies
dominate the literature on long-term conversational use:
*Summarization-Based Memory*, which compresses prior dialogue into evolving
summaries, and *Retrieval-Augmented Generation* (RAG), which stores prior
content in an external vector index and retrieves relevant passages at
inference time. Because the two strategies are typically studied in
isolation, their relative strengths are difficult to assess. This package
operationalizes a controlled, within-instance, repeated-measures comparison
of the two strategies on a single base model under matched decoding
parameters, across three standardized long-context benchmarks. It measures
recall accuracy, exact match, conversational consistency, contradiction
rate, response latency, token usage, and memory storage overhead, and tests
whether conversation length moderates the strategy-outcome relationship. The
analysis is interpreted through Distributed Cognition Theory (Hutchins,
1995) and Cognitive Load Theory (Sweller, 1988). All hypotheses, dependent
variables, statistical tests, and corrections were specified prior to data
collection; the harness records every deterministic control required for
independent replication.

## Highlights

- **Backend-agnostic LLM interface.** One configuration file selects
  Ollama (local pilot), Together AI (default cloud), AWS Bedrock,
  Hugging Face Inference Endpoints, or any OpenAI-compatible endpoint.
- **Controlled, deterministic pipeline.** Greedy decoding, `seed = 42`,
  `intfloat/e5-large-v2` embeddings, FAISS inner-product retrieval, and
  recursive summarization, all held constant across both conditions.
- **Complete statistical analysis.** Paired-samples *t*-tests with
  Holm-Bonferroni correction, Wilcoxon confirmatory tests, 2 × 3 ANOVA
  on Strategy × Length, Shapiro-Wilk and Levene assumption diagnostics,
  Bonferroni-corrected simple effects, Cohen's *d*, partial η², and 95
  percent confidence intervals.
- **Publication-quality figures.** Seven matplotlib figures rendered
  directly from the recorded results.
- **Research-grade engineering.** Type hints, Pydantic configuration,
  unit-tested metrics and statistics, GitHub Actions CI on Python 3.10
  / 3.11 / 3.12, pre-commit hooks, multi-stage Dockerfile.

## Quickstart

```bash
git clone https://github.com/Buchiexplores/llm-memory-eval.git
cd llm-memory-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cloud,local,viz]"

cp .env.example .env             # then add TOGETHER_API_KEY=...
export $(grep -v '^#' .env | xargs)

scripts/reproduce_results.sh
```

The script runs the full pipeline: download benchmarks, prepare
instances, evaluate both strategies, run the statistical analyses, and
render figures.

A local laptop pilot is supported with the same command and a different
config:

```bash
ollama pull llama3.1:8b
CONFIG=configs/local-pilot.yaml scripts/reproduce_results.sh
```

## Documentation

| Document                                | Purpose                                        |
|-----------------------------------------|------------------------------------------------|
| [Installation](docs/installation.md)    | Python / system / extras setup.                |
| [Quickstart](docs/quickstart.md)        | Ten-command reproduction recipe.               |
| **[Cloud setup](docs/cloud-setup.md)**  | Together AI account, key, cost, alternatives.  |
| [Methodology](docs/methodology.md)      | Design, variables, benchmarks, statistics.     |
| [Reproducibility](docs/reproducibility.md) | Deterministic controls, verification recipe. |
| [API reference](docs/api-reference.md)  | Module-by-module summary.                      |
| [GitHub setup](docs/github-setup.md)    | Publishing the repo + Zenodo DOI.              |

## Project layout

```
.
├── src/llm_memory_eval/        # Python package
│   ├── cli.py                  # Typer CLI: download / prepare / run / analyze / figures / all
│   ├── config.py               # Pydantic-validated YAML configuration
│   ├── llm/                    # Backend abstraction + Ollama, Together, OpenAI-compat, Bedrock, HF
│   ├── memory/                 # Summarization and RAG strategies
│   ├── metrics/                # F1, EM, consistency, contradiction
│   ├── data/                   # Benchmark download, prepare, stratify, length buckets
│   ├── experiment/             # Runner + result schema with checkpointing
│   ├── analysis/               # Descriptive, paired, ANOVA, assumptions, pipeline
│   ├── reporting/              # Publication-quality figures
│   └── utils/                  # Logging, seeding, token counting
├── tests/                      # Pytest suite (deterministic, no live API calls)
├── configs/                    # default / local-pilot / cloud-production YAMLs
├── scripts/                    # reproduce_results.sh, verify_environment.sh
├── docker/Dockerfile           # Multi-stage image for cloud runs
├── docs/                       # User and developer docs
├── paper/                      # JOSS submission (paper.md + paper.bib)
├── pyproject.toml              # PEP 621 metadata + tool config
├── CITATION.cff                # Software + dissertation citation (CFF v1.2.0)
├── codemeta.json               # CodeMeta 2.0 software metadata
├── .zenodo.json                # Zenodo archival metadata
├── CHANGELOG.md                # Keep-a-Changelog format
├── CONTRIBUTING.md             # Setup, tests, scientific-replication rule
├── CODE_OF_CONDUCT.md          # Contributor Covenant 2.1 (linked)
└── LICENSE                     # MIT
```

## Running the test suite

```bash
pytest --cov=llm_memory_eval --cov-report=term-missing
```

CI runs the same command across Python 3.10, 3.11, and 3.12 on every
push and pull request.

## Citing

If this package contributes to your work, please cite **both** the software
and the associated dissertation. The machine-readable
[CITATION.cff](CITATION.cff) file is rendered automatically by GitHub's
"Cite this repository" widget and by Zenodo's DOI registration. A
[CodeMeta](codemeta.json) descriptor and a [Zenodo metadata](.zenodo.json)
record are also provided for indexing services.

**Software (APA 7th edition):**

```text
Okeke, A. (2026). llm-memory-eval: A reproducible evaluation harness for
Summarization-Based Memory and Retrieval-Augmented Generation in large
language models (Version 0.1.0) [Computer software].
https://doi.org/10.5281/zenodo.XXXXXXX
```

**Software (BibTeX):**

```bibtex
@software{okeke2026llmmemoryeval,
  author    = {Okeke, Abuchi},
  title      = {llm-memory-eval: A Reproducible Evaluation Harness for
                Summarization-Based Memory and Retrieval-Augmented
                Generation in Large Language Models},
  year       = {2026},
  version    = {0.1.0},
  publisher  = {Zenodo},
  doi        = {10.5281/zenodo.XXXXXXX},
  url        = {https://github.com/Buchiexplores/llm-memory-eval}
}
```

**Dissertation (BibTeX):**

```bibtex
@phdthesis{okeke2026memorystrategies,
  author      = {Okeke, Abuchi},
  title       = {A Comparative Evaluation of Summarization and
                 Retrieval-Augmented Memory Strategies for Long-Term
                 Conversational Performance in Large Language Models},
  school      = {University of the Cumberlands},
  year        = {2026},
  type        = {Doctoral dissertation},
  address     = {Williamsburg, Kentucky, USA}
}
```

> Replace `10.5281/zenodo.XXXXXXX` with the DOI minted when the first
> release is archived on Zenodo (see [docs/github-setup.md](docs/github-setup.md)).

## Licence

MIT - see [LICENSE](LICENSE). The benchmark datasets retrieved by
`llm-memory-eval download-data` retain their upstream licences; this
package only redistributes evaluation code, not data.

## Acknowledgements

This work builds on the LongBench, LoCoMo, and LongMemEval benchmarks
and on the broader literature on memory-augmented language models cited
in [docs/methodology.md](docs/methodology.md).
