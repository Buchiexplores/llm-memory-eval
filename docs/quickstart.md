# Quickstart

This guide reproduces the full experimental pipeline in under ten
commands. The default flow assumes the reference cloud configuration
(Together AI hosting Meta Llama 3.1 70B Instruct).

## 1. Install

```bash
git clone https://github.com/Buchiexplores/llm-memory-eval.git
cd llm-memory-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cloud,local,viz]"
```

## 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and set TOGETHER_API_KEY=<your key>
export $(grep -v '^#' .env | xargs)
```

See the [cloud setup guide](cloud-setup.md) for full details on Together
AI account creation, free credit, and alternative backends.

## 3. Run the pipeline

```bash
llm-memory-eval download-data
llm-memory-eval prepare-data
llm-memory-eval run --config configs/cloud-production.yaml
llm-memory-eval analyze
llm-memory-eval figures
```

or in one command:

```bash
scripts/reproduce_results.sh
```

## 4. Outputs

After the pipeline finishes you will have:

- `results/experiment_results.csv` — per-instance outcomes.
- `results/statistical_analyses.json` — every statistical test.
- `results/tables/*.csv` — descriptive and inferential result tables.
- `results/figures/*.png` — publication-quality figures.

## 5. Pilot first (recommended)

To validate the pipeline locally before paying for the cloud run, swap
the config:

```bash
ollama pull llama3.1:8b
llm-memory-eval run --config configs/local-pilot.yaml
```

A laptop pilot at N = 90 typically completes in 2-3 hours; the cloud run
at N = 90 completes in well under one hour and costs a few US dollars on
Together AI.
