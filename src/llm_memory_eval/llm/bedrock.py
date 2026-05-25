"""AWS Bedrock backend for Meta Llama 3.1 70B Instruct."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from llm_memory_eval.llm.base import GenerateResponse, LLMClient
from llm_memory_eval.utils.logging import get_logger


log = get_logger(__name__)

DEFAULT_BEDROCK_MODEL = "meta.llama3-1-70b-instruct-v1:0"


class BedrockClient(LLMClient):
    """Bedrock client for the Llama 3.1 family via the ``invoke_model`` API.

    Credentials are resolved by the default boto3 chain (environment
    variables, shared config file, IAM role). The caller may also set the
    ``AWS_PROFILE`` and ``AWS_REGION`` environment variables.
    """

    name = "bedrock"

    def __init__(
        self,
        model: str = DEFAULT_BEDROCK_MODEL,
        *,
        region: str | None = None,
        request_timeout_seconds: int = 600,
        max_retries: int = 3,
    ) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'boto3' package is required for the Bedrock backend. "
                "Install it via: pip install 'llm-memory-eval[cloud]'"
            ) from exc

        import boto3

        self.model = model
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.timeout = request_timeout_seconds
        self.max_retries = max_retries
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def _build_prompt(self, system: str, user: str) -> str:
        # Llama 3.1 Instruct prompt format
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,  # noqa: ARG002 - Bedrock Llama does not accept a seed
    ) -> GenerateResponse:
        body = json.dumps(
            {
                "prompt": self._build_prompt(system, prompt),
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self._client.invoke_model(
                    modelId=self.model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                elapsed = time.perf_counter() - t0
                payload: dict[str, Any] = json.loads(resp["body"].read())
                text = (payload.get("generation") or "").strip()
                prompt_tok = int(payload.get("prompt_token_count", 0) or 0)
                comp_tok = int(payload.get("generation_token_count", 0) or 0)
                return GenerateResponse(
                    text=text,
                    latency_seconds=elapsed,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "Bedrock call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Bedrock generation failed after {self.max_retries} attempts: {last_err}"
        )
