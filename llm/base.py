"""Abstract LLM provider interface.

All concrete providers (Ollama, OpenAI, ...) implement this contract so that
the router can swap them transparently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class ProviderName(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 1024
    json_mode: bool = True


@dataclass(slots=True)
class LLMResponse:
    """Provider-agnostic response.

    `confidence_score` is provider-supplied (e.g. derived from logprobs / self-eval)
    and is used by the router to decide on fallback. Range: [0.0, 1.0].
    """

    content: str
    provider: ProviderName
    model: str
    confidence_score: float
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, object] = field(default_factory=dict)


class LLMProviderError(Exception):
    """Raised when a provider fails (network, auth, rate limit, malformed output)."""


class LLMProvider(ABC):
    """Base class — concrete providers MUST implement `generate`."""

    name: ProviderName

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run inference and return a normalized response."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True when the provider is reachable and ready."""
