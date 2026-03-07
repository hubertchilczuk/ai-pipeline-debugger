"""Ollama provider — local inference path."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx
from ollama import AsyncClient, ResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings
from llm.base import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderName

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


class OllamaProvider(LLMProvider):
    name = ProviderName.OLLAMA

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncClient(host=settings.ollama_base_url)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        try:
            result = await self._chat_with_retry(request, options)
        except (ResponseError, httpx.HTTPError) as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        content = result.get("message", {}).get("content", "")
        prompt_tokens = int(result.get("prompt_eval_count", 0))
        completion_tokens = int(result.get("eval_count", 0))
        confidence = self._estimate_confidence(
            content=content,
            json_mode=request.json_mode,
            done_reason=str(result.get("done_reason") or ""),
            completion_tokens=completion_tokens,
            max_tokens=request.max_tokens,
        )

        return LLMResponse(
            content=content,
            provider=self.name,
            model=self._settings.ollama_model,
            confidence_score=confidence,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=dict(result),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _chat_with_retry(self, request: LLMRequest, options: dict[str, Any]) -> dict[str, Any]:
        return await self._client.chat(
            model=self._settings.ollama_model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            format="json" if request.json_mode else "",
            options=options,
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._settings.ollama_base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def _estimate_confidence(
        content: str,
        json_mode: bool,
        done_reason: str,
        completion_tokens: int,
        max_tokens: int,
    ) -> float:
        """Hybrid heuristic combining several signals.

        Ollama doesn't expose token logprobs, so we blend: validity of output,
        self-reported confidence (when JSON), and whether generation was truncated.
        """
        if not content.strip():
            return 0.0
        # Truncation = low trust. Ollama signals via done_reason="length".
        truncated = done_reason == "length" or (
            max_tokens > 0 and completion_tokens >= max_tokens
        )
        truncation_penalty = 0.3 if truncated else 0.0

        if not json_mode:
            base = 0.7
            return max(0.0, base - truncation_penalty)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return 0.2
        reported = parsed.get("confidence")
        if isinstance(reported, (int, float)):
            base = max(0.0, min(1.0, float(reported)))
        else:
            base = 0.6
        return max(0.0, base - truncation_penalty)
