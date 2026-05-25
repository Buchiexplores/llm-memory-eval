"""Backend factory.

Constructs the appropriate :class:`LLMClient` for an
:class:`ExperimentConfig`. Importing concrete backends lazily keeps
optional dependencies (e.g. ``boto3``) out of the import path when the
user only needs the local backend.
"""

from __future__ import annotations

from llm_memory_eval.config import BackendConfig
from llm_memory_eval.llm.base import LLMClient


def build_client(cfg: BackendConfig) -> LLMClient:
    """Instantiate the configured backend."""
    if cfg.name == "ollama":
        from llm_memory_eval.llm.ollama import OllamaClient

        return OllamaClient(
            model=cfg.model,
            base_url=cfg.base_url,
            num_ctx=cfg.num_ctx,
            keep_alive=cfg.keep_alive,
            request_timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.name == "together":
        from llm_memory_eval.llm.together import TogetherClient

        return TogetherClient(
            model=cfg.model,
            request_timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.name == "openai_compat":
        from llm_memory_eval.llm.openai_compat import OpenAICompatibleClient

        if not cfg.base_url:
            raise ValueError("backend.base_url is required for openai_compat")
        return OpenAICompatibleClient(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env or "OPENAI_API_KEY",
            request_timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.name == "bedrock":
        from llm_memory_eval.llm.bedrock import BedrockClient

        return BedrockClient(
            model=cfg.model,
            region=cfg.region,
            request_timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.name == "hf_endpoint":
        from llm_memory_eval.llm.hf_endpoint import HFEndpointClient

        return HFEndpointClient(
            endpoint_url=cfg.base_url,
            request_timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
        )

    raise ValueError(f"Unknown backend: {cfg.name}")
