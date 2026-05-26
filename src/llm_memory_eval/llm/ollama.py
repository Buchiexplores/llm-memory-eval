"""Local Ollama backend.

Suitable for pilot evaluation on a laptop or workstation with a smaller
Llama model. For production runs, use a cloud backend serving the larger
target model (for example, ``Llama 3.1 70B Instruct``).
"""

from __future__ import annotations

import os
import time

import requests

from llm_memory_eval.llm.base import GenerateResponse, LLMClient
from llm_memory_eval.utils.logging import get_logger

log = get_logger(__name__)


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        *,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        request_timeout_seconds: int = 600,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = request_timeout_seconds
        self.max_retries = max_retries

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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "seed": seed,
            },
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                elapsed = time.perf_counter() - t0
                body = resp.json()
                return GenerateResponse(
                    text=(body.get("response") or "").strip(),
                    latency_seconds=elapsed,
                    prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
                    completion_tokens=int(body.get("eval_count", 0) or 0),
                )
            except Exception as e:  # noqa: BLE001 - any HTTP/network issue retried
                last_err = e
                log.warning(
                    "Ollama call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Ollama generation failed after {self.max_retries} attempts: {last_err}"
        )
