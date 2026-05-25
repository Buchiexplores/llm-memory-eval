# Contributing to llm-memory-eval

Thank you for your interest in contributing. This package supports a
peer-reviewed dissertation, so contributions must respect the
reproducibility-first norms that govern the research itself.

## Ground rules

- Be respectful. See the [Code of Conduct](./CODE_OF_CONDUCT.md).
- Open an issue before starting non-trivial work so we can agree on the
  approach before code is written.
- Keep pull requests focused on a single concern.
- Do not commit experimental output, data files, or model weights. The
  `.gitignore` excludes `data/`, `results/`, and similar paths.

## Development setup

```bash
git clone https://github.com/Buchiexplores/llm-memory-eval.git
cd llm-memory-eval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cloud,viz]"
pre-commit install
```

## Running the test suite

```bash
pytest --cov=llm_memory_eval --cov-report=term-missing
```

The CI workflow runs the same command across Python 3.10, 3.11, and 3.12.
Tests must not require live LLM API calls; use the stub `LLMClient` and stub
embedders provided in `tests/conftest.py`.

## Running the linters

```bash
ruff check src tests
black --check src tests
mypy src
```

Pre-commit will run these automatically when configured.

## Opening a pull request

1. Create a feature branch from `main`.
2. Add or update tests.
3. Ensure `pytest`, `ruff`, `black --check`, and `mypy` all pass locally.
4. Update `CHANGELOG.md` under an `## [Unreleased]` heading.
5. Open a pull request describing the motivation, the change, and the test
   plan.

## Scientific-replication rule

Any change that affects evaluation logic - text normalization, F1
computation, abstention regex, chunking, retrieval ranking, statistical
tests, or any code path that produces a reported result -
**must** be accompanied by a regression test backed by a versioned fixture
in `tests/fixtures/`. This guarantees that downstream replicators can detect
behavioural drift between releases. PRs that change evaluation behaviour
without a fixture-backed test will be asked to add one before merge.

## Releasing

Release responsibility rests with the maintainer. The typical flow is:

1. Update `pyproject.toml` version and `CHANGELOG.md`.
2. Tag the release (`git tag vX.Y.Z && git push --tags`).
3. Archive on Zenodo via the GitHub-Zenodo integration.
4. Update `CITATION.cff` with the new DOI.
