"""Hugging Face Inference Endpoints backend.

Use this when a dedicated, version-pinned model endpoint provides the
strongest reproducibility story (the endpoint is tied to a specific model
revision under your account).
"""

from __future__ import annotations

import os
import time

import requests

from llm_memory_eval.llm.base import GenerateResponse, LLMClient
from llm_memory_eval.utils.logging import get_logger

log = get_logger(__name__)


class HFEndpointClient(LLMClient):
    name = "hf_endpoint"

    def __init__(
        self,
        endpoint_url: str | None = None,
        *,
        token_env: str = "HF_TOKEN",
        request_timeout_seconds: int = 600,
        max_retries: int = 3,
    ) -> None:
        resolved_url = endpoint_url or os.environ.get("HF_ENDPOINT_URL")
        if not resolved_url:
            raise RuntimeError(
                "HF_ENDPOINT_URL is not set; provide it explicitly or via the environment."
            )
        self.endpoint_url: str = resolved_url
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Environment variable {token_env} is not set.")
        self.token = token
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
        full_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 1e-3),  # HF TGI rejects 0.0
                "top_p": top_p,
                "seed": seed,
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    self.endpoint_url, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                elapsed = time.perf_counter() - t0
                body = resp.json()
                if isinstance(body, list) and body:
                    text = (body[0].get("generated_text") or "").strip()
                else:
                    text = (body.get("generated_text") or "").strip()
                # HF Inference Endpoints does not always return usage counts;
                # leave at 0 when absent so analyses can flag missing data.
                return GenerateResponse(
                    text=text,
                    latency_seconds=elapsed,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "HF Endpoint call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"HF Endpoint generation failed after {self.max_retries} attempts: {last_err}"
        )
