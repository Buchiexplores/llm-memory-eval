# llm-memory-eval

`llm-memory-eval` is an open-source experiment-running harness for
comparing long-term memory strategies in large language models under
controlled, reproducible conditions.

It implements:

- A backend-agnostic LLM client (Ollama for local pilots; Together AI,
  AWS Bedrock, Hugging Face Inference Endpoints, or any
  OpenAI-compatible endpoint for production).
- Two memory strategies under comparison: recursive abstractive
  summarization (compression-oriented) and dense-embedding
  retrieval-augmented generation (fidelity-oriented).
- Deterministic text metrics (F1, exact match) and conversational
  indicators (consistency, contradiction).
- A complete statistical analysis pipeline (paired-samples *t*-tests
  with Holm–Bonferroni correction, Wilcoxon confirmatory tests, 2 × 3
  ANOVA on Strategy × Length, Shapiro–Wilk and Levene assumption
  diagnostics, and Bonferroni-corrected simple effects).
- Publication-quality figures rendered directly from the recorded
  results.

## Where to start

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Cloud setup](cloud-setup.md)
- [Methodology](methodology.md)
- [Reproducibility](reproducibility.md)
- [API reference](api-reference.md)

## Citing this work

If you use this package in academic work please cite the accompanying
`CITATION.cff` file, which Zenodo's GitHub integration converts into a
versioned DOI on every release.
