# Methodology

This document summarises the comparative experimental design implemented
in the package.

## Research questions

1. How do Summarization-Based Memory and Retrieval-Augmented Generation
   compare in terms of long-term recall accuracy, conversational
   consistency, and contradiction rate in large language models?
2. Do Summarization-Based Memory and Retrieval-Augmented Generation
   differ significantly in response latency, token usage, and memory
   storage overhead when evaluated across benchmark conditions?
3. Under what interaction conditions does each memory strategy perform
   best in sustaining human-like conversational persistence across
   varying dialogue lengths?

## Design

Quantitative, causal-comparative design with a repeated-measures
benchmark evaluation structure. Each benchmark instance is evaluated
under both memory-strategy conditions on the same base language model,
producing a within-instance paired-comparison structure.

| Variable                 | Type                          |
|--------------------------|-------------------------------|
| Memory strategy          | Independent (within-instance) |
| Conversation length      | Moderating (short, medium, long) |
| Recall accuracy (F1)     | Dependent                     |
| Exact match              | Dependent                     |
| Consistency rate         | Dependent                     |
| Contradiction rate       | Dependent                     |
| Response latency (s)     | Dependent                     |
| Token usage              | Dependent                     |
| Memory storage overhead  | Dependent                     |

## Benchmarks

- **LongBench** (Bai et al., 2024) — long-context understanding across
  question answering, summarization, code, and few-shot tasks.
- **LoCoMo** (Maharana et al., 2024) — very long-term, multi-session
  conversational memory.
- **LongMemEval** (Wu et al., 2025) — interactive memory abilities in
  chat-assistant settings.

## Length categorisation

Token-based for LongBench and LongMemEval; session- and turn-based for
LoCoMo (the broader of the two assignments wins). See
[`length_buckets.py`](../src/llm_memory_eval/data/length_buckets.py).

## Inference configuration

- Base model: Meta Llama 3.1 70B Instruct (cloud production) or 8B
  (laptop pilot).
- Decoding: `temperature = 0`, `top_p = 1.0`, `max_tokens = 512`,
  `seed = 42`.
- Embeddings: `intfloat/e5-large-v2`.
- Retrieval: FAISS inner-product index over L2-normalised embeddings,
  `top_k = 8`, 220-word chunks with 40-word overlap.
- Summarization: recursive chunked summarization performed by the same
  base LLM, capped at two chunks per instance for bounded latency.

## Statistical procedure

- Paired-samples *t*-tests for RQ1 and RQ2 with Holm-Bonferroni
  correction across each family of outcomes.
- Wilcoxon signed-rank test as a planned nonparametric confirmatory
  test alongside every parametric paired test.
- 2 × 3 ANOVA on Memory Strategy × Conversation Length for RQ3 on the
  five primary outcomes.
- Bonferroni-corrected simple effects within each length category for
  any outcome whose interaction term is significant.
- Shapiro-Wilk normality and Levene homogeneity-of-variance diagnostics.
- Natural-log transformation of response latency followed by Welch's
  one-way analysis as a robustness check.
- Effect sizes: Cohen's *d* for paired designs, partial η² for ANOVA.

## Theoretical framework

- **Distributed Cognition Theory** (Hutchins, 1995) frames memory
  externalisation through retrieval indices and summary artefacts.
- **Cognitive Load Theory** (Sweller, 1988) interprets the trade-off
  between in-context compression and pre-generation retrieval cost as
  competing load-management strategies.

## References

- Bai, Y., et al. (2024). LongBench: A bilingual, multitask benchmark
  for long context understanding. *ACL 2024*.
- Gao, Y., et al. (2023). Retrieval-augmented generation for large
  language models: A survey. *arXiv:2312.10997*.
- Hutchins, E. (1995). *Cognition in the Wild*. MIT Press.
- Lewis, P., et al. (2020). Retrieval-augmented generation for
  knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Maharana, A., et al. (2024). Evaluating very long-term conversational
  memory of LLM agents. *ACL 2024*.
- Rae, J. W., et al. (2020). Compressive transformers for long-range
  sequence modelling. *ICLR 2020*.
- Sweller, J. (1988). Cognitive load during problem solving.
  *Cognitive Science*, 12(2).
- Wu, D., et al. (2025). LongMemEval: Benchmarking chat assistants on
  long-term interactive memory. *arXiv*.
