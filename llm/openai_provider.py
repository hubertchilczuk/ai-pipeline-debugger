"""OpenAI provider — accurate / fallback path."""
from __future__ import annotations

import math
import time

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings
from llm.base import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderName

_RETRYABLE = (APIConnectionError, RateLimitError)


class OpenAIProvider(LLMProvider):
    name = ProviderName.OPENAI

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_s,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            completion = await self._call_with_retry(request)
        except APIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = completion.choices[0]
        content = choice.message.content or ""

        confidence = self._estimate_confidence(choice)

        return LLMResponse(
            content=content,
            provider=self.name,
            model=self._settings.openai_model,
            confidence_score=confidence,
            latency_ms=latency_ms,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            raw=completion.model_dump(),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _call_with_retry(self, request: LLMRequest):
        return await self._client.chat.completions.create(
            model=self._settings.openai_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format={"type": "json_object"} if request.json_mode else {"type": "text"},
            logprobs=True,
            top_logprobs=1,
            messages=[
                {"role": "system", "content": request.system_prompt},
