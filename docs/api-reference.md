# API reference

This page summarises the public API by module. Use `pydoc
llm_memory_eval.<module>` or the docstrings directly for full signatures.

## `llm_memory_eval.config`

- `ExperimentConfig` — top-level Pydantic model loaded from YAML.
- `ExperimentConfig.from_yaml(path)` — load and apply env overrides.

## `llm_memory_eval.llm`

- `LLMClient` — abstract base class.
- `GenerateResponse` — typed completion response.
- `build_client(cfg.backend)` — factory that returns the configured backend.
- Concrete backends: `OllamaClient`, `TogetherClient`,
  `OpenAICompatibleClient`, `BedrockClient`, `HFEndpointClient`.

## `llm_memory_eval.memory`

- `MemoryStrategy` — abstract base class.
- `MemoryArtifact` — frozen dataclass returned by `process()`.
- `SummarizationMemory(client, cfg)` — recursive abstractive summarization.
- `RagMemory(cfg)` — dense embedding + FAISS retrieval.

## `llm_memory_eval.metrics`

- `normalize(text)` — SQuAD-style answer normalisation.
- `compute_f1(prediction, reference)`, `best_f1(prediction, references)`
- `compute_em(prediction, reference)`, `best_em(prediction, references)`
- `consistency_indicator(prediction, references, f1_threshold=0.30)`
- `contradiction_indicator(prediction, references, f1_threshold=0.30)`
- `is_abstention(prediction)`

## `llm_memory_eval.data`

- `assign_length_bucket(...)` — benchmark-specific length thresholds.
- `stratified_subsample(instances, per_length, seed)` — balanced sampling.
- `download_datasets(output_dir)` — fetches LongBench / LoCoMo / LongMemEval.
- `prepare_all(raw_dir, output_dir)` — produces a unified instances JSON.

## `llm_memory_eval.experiment`

- `ExperimentRunner(cfg, client=None)` — run end-to-end with checkpointing.
- `InstanceResult`, `RunMetadata` — typed result rows and metadata.

## `llm_memory_eval.analysis`

- `descriptive_summary(df, variables)`
- `paired_test(summ, rag, name)`, `holm_correction(results)`
- `two_way_anova(df, sv, rv, name)`, `simple_effects_tests(df, sv, rv, name)`
- `shapiro_normality(df, variables)`, `levene_test(df, variables)`
- `log_transform_check(summ_latency, rag_latency)`
- `run_full_analysis(results_dir)` — orchestrates the whole pipeline.

## `llm_memory_eval.reporting`

- `generate_figures(results_dir)` — renders all publication figures.

## CLI

```text
llm-memory-eval --help
llm-memory-eval download-data
llm-memory-eval prepare-data
llm-memory-eval run --config configs/cloud-production.yaml
llm-memory-eval analyze
llm-memory-eval figures
llm-memory-eval all --config configs/cloud-production.yaml
```
