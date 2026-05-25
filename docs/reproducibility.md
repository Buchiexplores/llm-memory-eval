# Reproducibility

This document lists every deterministic control in the pipeline and how
to verify that a re-run on your machine matches the published numbers.

## Deterministic controls

| Control                              | Source of truth                              |
|--------------------------------------|----------------------------------------------|
| Random seed (all RNGs)               | `decoding.seed` in the YAML config (default 42) |
| NumPy / `random` / torch (if used)   | `set_global_seed` in `utils/seed.py`         |
| `PYTHONHASHSEED`                     | set by `set_global_seed`                     |
| LLM decoding temperature             | 0.0 (greedy)                                 |
| LLM decoding top-p                   | 1.0                                          |
| LLM decoding `max_tokens`            | 512 (cloud) / 256 (pilot)                    |
| Cloud LLM `seed` argument            | passed via Together AI Chat Completions API  |
| Embedding model                      | `intfloat/e5-large-v2`                       |
| Embedding L2 normalisation           | always enabled                               |
| FAISS index                          | `IndexFlatIP` (inner product, exact)         |
| Chunk size / overlap                 | 220 words / 40 words                         |
| Stratified subsample                 | `data.stratify.stratified_subsample(seed=42)`|

## What is recorded per run

`results/logs/run_metadata.json` captures the full configuration, the
package version, the git commit SHA (if the repository is a git
checkout), and the run timestamp.

## How to verify

After re-running the pipeline:

```bash
diff <(jq -S . results/statistical_analyses.json) \
     <(jq -S . path/to/published_statistical_analyses.json)
```

Small floating-point differences (< 1e-6) at the periphery of the
descriptive statistics are expected when switching between cloud
backends. Inferential decisions (significant / not significant at α = .05
after Holm-Bonferroni) should be identical.

## Known sources of non-determinism

- Cloud LLM providers may deploy different model revisions over time.
  Pin the model identifier in your YAML config and record the value of
  the `system_fingerprint` field (Together AI exposes this) in your
  experiment notes for true bit-level reproducibility.
- BLAS routines used by NumPy / SciPy can produce platform-specific
  rounding differences in the final decimal place of effect sizes and
  *p*-values.
- The `sentence-transformers` library uses non-deterministic operations
  on some accelerators; embeddings are stable on CPU and on Apple
  silicon with `embedding_device: "cpu"` or `"mps"`.

## Replication recipe

1. Clone the repository at a tagged release: `git checkout v0.1.0`.
2. Recreate the Python environment with the lockfile (see `pyproject.toml`).
3. Set `TOGETHER_API_KEY`.
4. Run `scripts/reproduce_results.sh`.
5. Compare `results/statistical_analyses.json` against the archived
   artefact on Zenodo.
