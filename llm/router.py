"""Smart LLM router with confidence-based fallback.

Routing matrix:
    mode=cheap     -> Ollama only (no fallback)
    mode=accurate  -> OpenAI only
    mode=auto      -> Ollama first; fallback to OpenAI if:
                       * Ollama errored, OR
                       * confidence_score < threshold
"""
from __future__ import annotations

from core.config import LLMMode, Settings
from core.logger import get_logger
from llm.base import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderName

logger = get_logger(__name__)


class LLMRouter:
    def __init__(
        self,
        ollama: LLMProvider,
        openai: LLMProvider,
        settings: Settings,
    ) -> None:
        self._ollama = ollama
        self._openai = openai
        self._settings = settings

    async def generate(self, request: LLMRequest) -> LLMResponse:
        mode = self._settings.llm_mode

        if mode is LLMMode.ACCURATE:
            return await self._call(self._openai, request)

        if mode is LLMMode.CHEAP:
            return await self._call(self._ollama, request)

        # AUTO: try local first, fall back on error or low confidence.
        try:
            primary = await self._call(self._ollama, request)
        except LLMProviderError as exc:
            logger.warning("ollama_failed_falling_back", error=str(exc))
            return await self._call(self._openai, request)

        threshold = self._settings.llm_confidence_threshold
        if primary.confidence_score < threshold:
            logger.info(
                "low_confidence_fallback",
                confidence=primary.confidence_score,
                threshold=threshold,
            )
            try:
                return await self._call(self._openai, request)
            except LLMProviderError as exc:
                logger.warning("fallback_failed_returning_primary", error=str(exc))
                return primary

        return primary

    @staticmethod
    async def _call(provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        response = await provider.generate(request)
        logger.info(
            "llm_call",
            provider=provider.name,
            model=response.model,
            latency_ms=response.latency_ms,
            confidence=response.confidence_score,
        )
        return response

    async def health(self) -> dict[ProviderName, bool]:
        return {
            ProviderName.OLLAMA: await self._ollama.health_check(),
            ProviderName.OPENAI: await self._openai.health_check(),
        }
