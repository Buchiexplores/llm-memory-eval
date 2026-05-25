# llm-memory-eval

`llm-memory-eval` is the open-source companion package to the
dissertation *A Comparative Evaluation of Summarization and
Retrieval-Augmented Memory Strategies for Long-Term Conversational
Performance in Large Language Models* (Okeke, 2026).

It implements:

- A backend-agnostic LLM client (Ollama for local pilots; Together AI,
  AWS Bedrock, Hugging Face Inference Endpoints, or any
  OpenAI-compatible endpoint for production).
- Two memory strategies under comparison: recursive abstractive
  summarization (compression-oriented) and dense-embedding
  retrieval-augmented generation (fidelity-oriented).
- Deterministic text metrics (F1, exact match) and conversational
  indicators (consistency, contradiction).
- The full statistical analysis pipeline pre-registered in Chapter 3
  (paired-samples *t*-tests with Holm–Bonferroni correction,
  Wilcoxon confirmatory tests, 2 × 3 ANOVA on Strategy × Length,
  Shapiro–Wilk and Levene assumption diagnostics, and
  Bonferroni-corrected simple effects).
- Publication-quality figures and a Chapter 4 / Chapter 5 `.docx`
  builder that follows the University of the Cumberlands APA 7
  quantitative dissertation template.

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
