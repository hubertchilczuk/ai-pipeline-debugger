"""Tests for LLM router fallback logic."""
from __future__ import annotations

import pytest

from core.config import LLMMode, Settings
from llm.base import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderName
from llm.router import LLMRouter


class FakeProvider(LLMProvider):
    def __init__(self, name: ProviderName, confidence: float = 0.9, fail: bool = False) -> None:
        self.name = name
        self._confidence = confidence
        self._fail = fail
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self._fail:
            raise LLMProviderError("boom")
        return LLMResponse(
            content='{"ok": true}',
            provider=self.name,
            model=f"fake-{self.name.value}",
            confidence_score=self._confidence,
            latency_ms=10,
        )

    async def health_check(self) -> bool:
        return not self._fail


def _settings(mode: LLMMode, threshold: float = 0.65) -> Settings:
    return Settings(llm_mode=mode, llm_confidence_threshold=threshold)


@pytest.mark.asyncio
async def test_auto_uses_ollama_when_high_confidence() -> None:
    ollama = FakeProvider(ProviderName.OLLAMA, confidence=0.9)
    openai = FakeProvider(ProviderName.OPENAI)
    router = LLMRouter(ollama, openai, _settings(LLMMode.AUTO))

    resp = await router.generate(LLMRequest("s", "u"))
    assert resp.provider is ProviderName.OLLAMA
    assert openai.calls == 0


@pytest.mark.asyncio
async def test_auto_falls_back_on_low_confidence() -> None:
    ollama = FakeProvider(ProviderName.OLLAMA, confidence=0.2)
    openai = FakeProvider(ProviderName.OPENAI, confidence=0.95)
    router = LLMRouter(ollama, openai, _settings(LLMMode.AUTO, threshold=0.5))

    resp = await router.generate(LLMRequest("s", "u"))
    assert resp.provider is ProviderName.OPENAI
    assert ollama.calls == 1
    assert openai.calls == 1


@pytest.mark.asyncio
async def test_auto_falls_back_on_error() -> None:
    ollama = FakeProvider(ProviderName.OLLAMA, fail=True)
    openai = FakeProvider(ProviderName.OPENAI)
    router = LLMRouter(ollama, openai, _settings(LLMMode.AUTO))

    resp = await router.generate(LLMRequest("s", "u"))
    assert resp.provider is ProviderName.OPENAI


@pytest.mark.asyncio
async def test_accurate_mode_skips_ollama() -> None:
    ollama = FakeProvider(ProviderName.OLLAMA)
    openai = FakeProvider(ProviderName.OPENAI)
    router = LLMRouter(ollama, openai, _settings(LLMMode.ACCURATE))

    await router.generate(LLMRequest("s", "u"))
    assert ollama.calls == 0
    assert openai.calls == 1


@pytest.mark.asyncio
async def test_cheap_mode_never_calls_openai_even_on_failure() -> None:
    ollama = FakeProvider(ProviderName.OLLAMA, fail=True)
    openai = FakeProvider(ProviderName.OPENAI)
    router = LLMRouter(ollama, openai, _settings(LLMMode.CHEAP))

    with pytest.raises(LLMProviderError):
        await router.generate(LLMRequest("s", "u"))
    assert openai.calls == 0
