# Installation

`llm-memory-eval` requires Python 3.10 or newer. Optional dependency
groups install only the backends and tools you actually need.

## From source (recommended for now)

```bash
git clone https://github.com/okekeag/llm-memory-eval.git
cd llm-memory-eval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[cloud,viz]"
```

## Dependency extras

| Extra     | What it adds                                                            |
|-----------|-------------------------------------------------------------------------|
| (none)    | Core pipeline, statistical analysis, Ollama backend, docx output.       |
| `cloud`   | Together AI, AWS Bedrock, Hugging Face Endpoints, generic OpenAI-compat.|
| `local`   | sentence-transformers, FAISS CPU, torch (needed for RAG locally).       |
| `viz`     | matplotlib + kaleido (figure rendering).                                |
| `dev`     | pytest, ruff, black, mypy, pre-commit, type stubs.                      |

A complete development install:

```bash
pip install -e ".[cloud,local,viz,dev]"
pre-commit install
```

## System prerequisites

- For the local backend (Ollama): install Ollama and pull the model.
  ```bash
  brew install ollama        # macOS; see ollama.ai for other platforms
  ollama pull llama3.1:8b
  ```
- For dataset downloads: the `datasets` package (installed automatically
  as a runtime dependency) requires network access to the Hugging Face
  Hub.
- For the cloud backend: a Together AI account and API key (see the
  [cloud setup guide](cloud-setup.md)).

## Verifying the install

```bash
llm-memory-eval version
scripts/verify_environment.sh
```
