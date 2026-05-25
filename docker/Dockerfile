# Multi-stage Dockerfile for cloud-backend execution.
# Local backends (Ollama, sentence-transformers + faiss) require additional
# system libraries and are best run outside the container.

FROM python:3.11-slim AS build

WORKDIR /build
COPY pyproject.toml README.md LICENSE CHANGELOG.md CITATION.cff /build/
COPY src /build/src
RUN pip install --upgrade pip build && python -m build --wheel

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="llm-memory-eval" \
      org.opencontainers.image.description="Reproducible evaluation of Summarization vs RAG memory strategies for LLMs." \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY --from=build /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[cloud,viz] && rm /tmp/*.whl

COPY configs /app/configs

ENTRYPOINT ["llm-memory-eval"]
CMD ["--help"]
