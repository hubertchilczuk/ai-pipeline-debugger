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
    return _build_indexer()


@lru_cache
def _build_retriever() -> IncidentRetriever:
    s = get_settings()
    return IncidentRetriever(client=get_chroma_client(), collection=s.chroma_collection)


def get_retriever() -> IncidentRetriever:
    return _build_retriever()


def get_parser(router: Annotated[LLMRouter, Depends(get_router)]) -> LogParser:
    s = get_settings()
    # When PARSER_USE_LLM_FALLBACK=true the router itself becomes the fallback path,
    # so unstructured logs get a second chance instead of returning UNKNOWN.
    fallback = router if s.parser_use_llm_fallback else None
    return LogParser(llm_fallback=fallback)


RouterDep = Annotated[LLMRouter, Depends(get_router)]
IndexerDep = Annotated[IncidentIndexer, Depends(get_indexer)]
RetrieverDep = Annotated[IncidentRetriever, Depends(get_retriever)]
ParserDep = Annotated[LogParser, Depends(get_parser)]
