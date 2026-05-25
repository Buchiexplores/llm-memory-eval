# Data directory

This directory is intentionally empty in the repository. Benchmark datasets
are downloaded at runtime and are **not** redistributed with this package
because their licences vary by upstream.

## How to populate

```bash
llm-memory-eval download-data --output data/raw
llm-memory-eval prepare-data --raw data/raw --output data/processed
```

The download step retrieves:

| Benchmark    | Hugging Face repo               | Upstream licence            |
|--------------|----------------------------------|------------------------------|
| LongBench    | THUDM/LongBench                  | See upstream LICENSE         |
| LoCoMo       | snap-research/locomo             | See upstream LICENSE         |
| LongMemEval  | xiaowu0162/LongMemEval           | See upstream LICENSE         |

By downloading you agree to the upstream licences. If your organisation
mirrors these datasets internally, point the prepare command at the local
mirror instead.
