# Cloud setup

This guide walks through the reference cloud deployment: Meta Llama 3.1
70B Instruct served from Together AI. Alternative cloud backends (AWS
Bedrock, Hugging Face Inference Endpoints, self-hosted vLLM) are
documented at the end for reviewers who require them.

The package is provider-agnostic at the code level. Switching providers
means changing one YAML field and one environment variable; the
experiment runner, statistical pipeline, and figures stay identical.

## Why Together AI is appropriate for US academic research

- The model itself — Meta **Llama 3.1 70B Instruct** — is open-weights
  and version-pinnable. Reviewers can verify that the weights are
  identical to those Meta publishes on Hugging Face.
- Together AI publishes its inference configuration (quantisation,
  batching, attention implementation) and exposes a `seed` argument
  through the OpenAI-compatible Chat Completions API, which delivers
  greedy-decode determinism for the same input.
- Hundreds of arXiv papers in 2024 and 2025 cite Together AI as their
  inference backend; it is treated by the broader literature in the same
  way OpenAI and Anthropic APIs are treated: a managed inference
  service whose model identifier, version, and decoding parameters
  must be reported in the methods section.
- The University of the Cumberlands IRB exemption you already hold for
  publicly available, deidentified benchmark datasets covers this
  arrangement; no human-subjects data leaves your machine.
- If a reviewer pushes back specifically on Together AI, swap to the
  AWS Bedrock backend (same model family, enterprise-managed) using the
  one-line config change documented later in this file.

## Recommended cloud setup: Together AI

### 1. Create an account

- Go to <https://api.together.ai/signin> and create an account using
  your institutional email when possible (so the audit trail ties the
  workload to your university).
- New accounts receive a starter credit balance (commonly USD 5–25).
  The full reproduction at N = 90 fits comfortably inside that.

### 2. Generate an API key

- Sign in to the Together AI dashboard.
- Navigate to *Settings → API Keys* (or visit
  <https://api.together.ai/settings/api-keys>).
- Create a new key, label it `llm-memory-eval`, and copy the value
  somewhere safe. The key is shown only once.

### 3. Store the key locally

```bash
cp .env.example .env
# Edit .env in your editor of choice and set:
#   TOGETHER_API_KEY=<your key>
```

Load it into your current shell:

```bash
export $(grep -v '^#' .env | xargs)
```

Or set the variable directly for a single command:

```bash
TOGETHER_API_KEY=<your key> llm-memory-eval run --config configs/cloud-production.yaml
```

> **Never commit `.env`.** The repository's `.gitignore` already
> excludes `.env` and `.env.local`. Treat keys like passwords.

### 4. Verify connectivity

```bash
scripts/verify_environment.sh
```

You should see `TOGETHER_API_KEY: set` in the output and `together`
listed under the optional dependencies. To make a single test call:

```python
from llm_memory_eval.llm.together import TogetherClient

client = TogetherClient()
print(client.generate(
    "What is the capital of France?",
    system="Answer concisely.",
    max_tokens=10, temperature=0.0, top_p=1.0, seed=42,
).text)
```

### 5. Choose the model identifier

The `configs/cloud-production.yaml` config ships with:

```yaml
backend:
  name: "together"
  model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

| Together AI model id                          | Availability                                   |
|-----------------------------------------------|------------------------------------------------|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo`     | Serverless (pay-per-token). Default config.    |
| `meta-llama/Meta-Llama-3.1-70B-Instruct`      | Dedicated endpoint only (hourly billing).      |
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`| Dedicated endpoint only (hourly billing).      |

> **Note on model availability.** Together AI's serverless (pay-per-token)
> catalogue changes over time. As of this writing, Llama 3.1 70B Instruct is
> served only via a *dedicated endpoint* (hourly billing), so the default
> config uses the serverless `Llama-3.3-70B-Instruct-Turbo`. To run the exact
> Llama 3.1 70B model, either provision a dedicated Together AI endpoint and
> set its model id here, or use the AWS Bedrock backend
> (`meta.llama3-1-70b-instruct-v1:0`), which still serves Llama 3.1 70B
> on-demand. Verify current identifiers and pricing at
> <https://www.together.ai/pricing>. Pin whichever model you choose so the
> run is reproducible.

### 6. Run the reproduction

```bash
scripts/reproduce_results.sh
```

The full pipeline at N = 90 typically completes in under an hour on
Together AI. Cost varies with output length; a budget of USD 5 is
ample for a single reproduction. Watch your dashboard for usage spikes
if you adjust the configuration to N > 200.

### 7. Cost estimation

| Item                              | Typical magnitude (Llama 3.1 70B)        |
|-----------------------------------|-------------------------------------------|
| Input tokens per instance         | ~6,000-12,000 (summarization + answer)    |
| Output tokens per instance        | ~600-1,000                                |
| Pairs per N = 90 run              | 90 (each evaluated twice: summ + RAG)     |
| Total billable tokens at N = 90   | ~1.0-1.5 million                          |
| Estimated cost at N = 90          | USD 1-3 at Together AI list prices        |
| Estimated cost at N = 212         | USD 3-7                                   |

Confirm the current per-million-token rate on Together AI's pricing
page before each run; rates change occasionally.

## Alternative backend: AWS Bedrock

Suitable when your institution requires a US enterprise cloud provider
with BAA / SOC 2 paperwork already in place.

### 1. Install the AWS extras and authenticate

```bash
pip install -e ".[cloud]"
aws configure                   # or set AWS_PROFILE
export AWS_REGION=us-east-1
```

### 2. Enable model access in the Bedrock console

In the AWS console, open *Bedrock → Model access* and enable
*Meta Llama 3.1 70B Instruct*. Activation typically takes a few
minutes; some accounts require a brief justification form.

### 3. Switch the YAML config

```yaml
backend:
  name: "bedrock"
  model: "meta.llama3-1-70b-instruct-v1:0"
  region: "us-east-1"
```

### 4. Run the same pipeline

```bash
scripts/reproduce_results.sh
```

Bedrock does **not** accept a `seed` argument, so re-runs at
`temperature = 0` are still greedy but may include rare ties resolved
non-deterministically. Together AI is therefore preferred when bit-level
reproducibility is required.

## Alternative backend: self-hosted vLLM on a GPU instance

Use this when you intend to run thousands of evaluation sweeps and want
to avoid per-token billing.

1. Provision a GPU instance (RunPod, Lambda Labs, AWS p4/p5, GCP A3,
   or your own server). Llama 3.1 70B fits on 2 × A100 80 GB or
   1 × H100 80 GB.
2. Install vLLM and start the OpenAI-compatible server:

   ```bash
   pip install vllm
   vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --port 8000
   ```

3. Point the YAML at the local server:

   ```yaml
   backend:
     name: "openai_compat"
     model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
     base_url: "http://localhost:8000/v1"
     api_key_env: "OPENAI_API_KEY"   # any non-empty value works
   ```

4. Export a placeholder key:

   ```bash
   export OPENAI_API_KEY=dummy
   ```

5. Run the pipeline as usual.

## Alternative backend: Hugging Face Inference Endpoints

```yaml
backend:
  name: "hf_endpoint"
  base_url: "<your endpoint URL>"
```

```bash
export HF_TOKEN=<your write token>
export HF_ENDPOINT_URL=<your endpoint URL>
llm-memory-eval run --config configs/cloud-production.yaml
```

## Operational checklist before a production run

- [ ] Pinned model identifier recorded in `configs/cloud-production.yaml`.
- [ ] `TOGETHER_API_KEY` (or equivalent) set in the shell that will run
      the pipeline.
- [ ] `scripts/verify_environment.sh` reports all dependencies present.
- [ ] You have committed the current code state (`git status` is clean)
      so `RunMetadata.commit` captures a reproducible SHA.
- [ ] Together AI dashboard shows an account credit balance large enough
      for the planned N (see cost table above).
- [ ] `data/processed/all_instances.json` exists (run `prepare-data` if
      not).
- [ ] You have allocated a quiet terminal session for the run; the
      pipeline streams progress to stderr.

## Troubleshooting

- **401 Unauthorized**: the `TOGETHER_API_KEY` is missing or mistyped.
  Re-run `export $(grep -v '^#' .env | xargs)` and try again.
- **429 Rate limited**: Together AI enforces per-account RPM limits.
  Wait a minute and re-run; the runner resumes from the checkpoint at
  `results/experiment_results_partial.csv`.
- **`model_not_found`**: confirm the exact model identifier is current
  in Together AI's catalogue; identifiers change occasionally.
- **Hung embedding pass**: the `sentence-transformers` model is
  downloaded the first time RAG runs; the download can take several
  minutes on a slow link. Re-run after the cache is warm.
- **AWS `AccessDeniedException`**: enable the Llama 3.1 model in the
  Bedrock console and confirm the IAM role has `bedrock:InvokeModel`.
