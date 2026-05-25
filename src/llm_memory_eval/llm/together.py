"""Together AI backend (default cloud backend for production runs).

Together AI hosts a range of open-weights Llama models behind an
OpenAI-compatible API. This is the default cloud backend for production
experiment runs. The default model is a serverless (pay-per-token) 70B
chat model; see ``configs/cloud-production.yaml`` for the configured value.
"""

from __future__ import annotations

from llm_memory_eval.llm.openai_compat import OpenAICompatibleClient


TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class TogetherClient(OpenAICompatibleClient):
    """OpenAI-compatible client preconfigured for Together AI.

    Parameters
    ----------
    model
        Together AI model identifier. Defaults to
        ``meta-llama/Llama-3.3-70B-Instruct-Turbo`` (serverless). Note that
        Llama 3.1 70B Instruct is only available on Together AI via a
        dedicated endpoint, not the serverless tier.
    """

    name = "together"

    def __init__(
        self,
        model: str = DEFAULT_TOGETHER_MODEL,
        *,
        request_timeout_seconds: int = 600,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            base_url=TOGETHER_BASE_URL,
            api_key_env="TOGETHER_API_KEY",
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
        )
