---
title: "llm-memory-eval: A reproducible harness for comparing summarization-based and retrieval-augmented memory in large language models"
tags:
  - Python
  - large language models
  - retrieval-augmented generation
  - summarization
  - long-context evaluation
  - reproducible research
authors:
  - name: Abuchi Okeke
    orcid: 0000-0001-8684-1115
    affiliation: 1
affiliations:
  - name: University of the Cumberlands, Williamsburg, Kentucky, USA
    index: 1
date: 21 May 2026
bibliography: paper.bib
---

# Summary

Large language models (LLMs) frequently lose track of earlier facts,
contradict prior statements, and forget user-specific context as
conversations extend across many turns or multiple sessions
[@liu2024lost; @maharana2024locomo; @wu2025longmemeval]. Two memory
augmentation strategies dominate the literature on long-term
conversational use: *summarization-based memory*, which compresses prior
dialogue into evolving textual summaries [@rae2020compressive], and
*retrieval-augmented generation* (RAG), which stores prior content in an
external vector index and retrieves the most relevant passages at
inference time [@lewis2020rag; @karpukhin2020dpr]. These strategies are
typically studied in isolation, with different base models, prompts,
decoding parameters, and metrics, which makes their relative strengths
difficult to assess.

`llm-memory-eval` is an open-source Python package that operationalizes a
controlled, within-instance, repeated-measures comparison of the two
strategies on the same base model under matched decoding parameters,
across three standardized long-context benchmarks: LongBench
[@bai2024longbench], LoCoMo [@maharana2024locomo], and LongMemEval
[@wu2025longmemeval]. The package provides backend-agnostic inference, a
deterministic measurement layer, and a complete statistical-analysis
pipeline, and it renders both publication-quality figures and APA-7
dissertation chapter documents directly from the recorded results.

# Statement of need

Researchers and practitioners who must choose between summarization and
retrieval for persistent conversational systems currently lack a
controlled, reproducible basis for that decision. Existing evaluations
are fragmented across incompatible experimental setups, and few release
the code required to reproduce their numbers. `llm-memory-eval` addresses
this gap with four design commitments:

1. **Controlled comparison.** Both memory strategies are evaluated on the
   same benchmark instances with the same base model, prompt structure,
   and decoding configuration (`temperature = 0`, `top_p = 1`, fixed
   seed), so observed differences are attributable to the memory strategy
   rather than to confounds.

2. **Backend portability.** A single `LLMClient` abstraction supports
   local inference via Ollama and cloud inference via Together AI, AWS
   Bedrock, Hugging Face Inference Endpoints, and any OpenAI-compatible
   endpoint. Switching backends requires changing one configuration
   field, which lets a laptop pilot and a cloud production run share
   identical analysis code.

3. **Pre-registered statistics.** The analysis module implements the
   tests specified before data collection: paired-samples *t*-tests with
   Holm-Bonferroni correction, Wilcoxon signed-rank confirmatory tests,
   two-way ANOVA on Strategy x Conversation Length with
   Bonferroni-corrected simple effects, Shapiro-Wilk and Levene
   assumption diagnostics, and effect sizes (Cohen's *d*, partial
   eta-squared).

4. **End-to-end reproducibility.** Random seeds, model identifiers,
   embedding-model revisions, and the package version are recorded in
   run metadata; a single command reproduces the full pipeline from
   benchmark download through figure and chapter rendering. A
   deterministic unit-test suite runs in continuous integration across
   Python 3.10, 3.11, and 3.12.

The package was developed to support a doctoral dissertation comparing
the two strategies, and it interprets results through Distributed
Cognition Theory [@hutchins1995cognition] and Cognitive Load Theory
[@sweller1988cognitive]. It is, however, written as a general-purpose
harness: new memory strategies, benchmarks, metrics, or inference
backends can be added by implementing the corresponding abstract base
class, which makes the package a reusable foundation for future
comparative studies of memory-augmented LLMs.

# Functionality

`llm-memory-eval` exposes a `typer` command-line interface with stages
for data download, preparation, experiment execution, statistical
analysis, figure generation, and dissertation-chapter construction, as
well as an `all` command that runs the complete pipeline. Configuration
is supplied through validated YAML files, with environment-variable
overrides for backend selection and seeding. The retrieval strategy uses
the `intfloat/e5-large-v2` embedding model with a FAISS inner-product
index [@johnson2019faiss], and the summarization strategy performs
recursive abstractive summarization with the same base model used for
answer generation. Outcome measures include token-level F1, exact match,
consistency and contradiction indicators, response latency, token usage,
and serialized memory-storage overhead.

# Acknowledgements

This software builds on the LongBench, LoCoMo, and LongMemEval benchmarks
and on the open-source `sentence-transformers`, FAISS, SciPy, pandas, and
NumPy ecosystems. The author thanks the dissertation committee at the
University of the Cumberlands for their guidance.

# References
