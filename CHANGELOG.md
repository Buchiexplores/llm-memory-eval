# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-21

### Added
- Initial open-source release accompanying the dissertation
  *A Comparative Evaluation of Summarization and Retrieval-Augmented Memory
  Strategies for Long-Term Conversational Performance in Large Language Models*.
- `LLMClient` abstraction with backends for Ollama, Together AI, generic
  OpenAI-compatible endpoints, AWS Bedrock, and Hugging Face Inference
  Endpoints.
- Memory strategies: recursive abstractive summarization and dense-embedding
  RAG (intfloat/e5-large-v2 + FAISS inner-product index).
- Deterministic experiment runner with per-instance checkpointing.
- Statistical pipeline: descriptive statistics, paired-samples t-tests with
  Holm-Bonferroni correction, Wilcoxon signed-rank fallback, 2 × 3 ANOVA on
  Strategy × Length, Shapiro-Wilk normality, Levene homogeneity-of-variance,
  Bonferroni-corrected simple-effects post-hoc tests, and Cohen's d / partial
  eta-squared effect sizes.
- Publication-quality matplotlib figures rendered from the recorded results.
- Typer-based CLI (`llm-memory-eval`) with `download-data`, `prepare-data`,
  `run`, `analyze`, `figures`, and `all` commands.
- YAML-driven configuration with environment-variable overrides.
- Continuous integration on Python 3.10, 3.11, and 3.12.
- Multi-stage Docker image for cloud-backend execution.
- Documentation set: installation, quickstart, methodology, reproducibility,
  and API reference.

### Notes
- The DOI field in `CITATION.cff` is currently the placeholder
  `10.5281/zenodo.XXXXXXX`. The real Zenodo DOI will be inserted when the
  first archive is published.
