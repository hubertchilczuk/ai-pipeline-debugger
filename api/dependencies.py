"""FastAPI dependency providers (singletons via lru_cache)."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import get_settings
from ingestion.parser import LogParser
from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider
from llm.router import LLMRouter
from vector_db.client import get_chroma_client
from vector_db.indexer import IncidentIndexer
from vector_db.retriever import IncidentRetriever


@lru_cache
def _build_router() -> LLMRouter:
    s = get_settings()
    return LLMRouter(ollama=OllamaProvider(s), openai=OpenAIProvider(s), settings=s)


def get_router() -> LLMRouter:
    return _build_router()


@lru_cache
def _build_indexer() -> IncidentIndexer:
    s = get_settings()
    return IncidentIndexer(client=get_chroma_client(), collection=s.chroma_collection)


def get_indexer() -> IncidentIndexer:
