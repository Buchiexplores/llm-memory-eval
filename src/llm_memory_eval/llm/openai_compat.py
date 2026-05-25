"""OpenAI-compatible HTTP backend.

Works with any provider that exposes the OpenAI Chat Completions API,
including Together AI, Fireworks AI, Anyscale, and a self-hosted vLLM
server. The Together AI backend (:mod:`llm_memory_eval.llm.together`)
is a thin specialization of this client with provider-specific defaults.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from llm_memory_eval.llm.base import GenerateResponse, LLMClient
from llm_memory_eval.utils.logging import get_logger


log = get_logger(__name__)


class OpenAICompatibleClient(LLMClient):
    name = "openai_compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        request_timeout_seconds: int = 600,
        max_retries: int = 3,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set; cannot authenticate "
                f"against {base_url}."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = request_timeout_seconds
        self.max_retries = max_retries
        self._client = self._make_client()

    def _make_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI-compatible backends. "
                "Install it via: pip install 'llm-memory-eval[cloud]'"
            ) from exc

        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> GenerateResponse:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                elapsed = time.perf_counter() - t0
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                comp_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                return GenerateResponse(
                    text=text,
                    latency_seconds=elapsed,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "OpenAI-compatible call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"OpenAI-compatible generation failed after {self.max_retries} attempts: {last_err}"
        )
