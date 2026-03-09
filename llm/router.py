"""LLM provider router."""
from .base import LLMProvider


class Router:
    """Selects an LLM provider for a given request."""

    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers

    def get(self, name: str) -> LLMProvider:
        return self.providers[name]
