"""Integration tests for the /analyze endpoint with mocked LLM + Chroma."""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient

from api import dependencies as deps
from api.main import app
from core.config import LLMMode, Settings
from core.logger import configure_logging
from ingestion.parser import LogParser
from llm.base import LLMProviderError, LLMRequest, LLMResponse, ProviderName
from llm.router import LLMRouter
from vector_db.indexer import IncidentIndexer
from vector_db.retriever import IncidentRetriever


class _FakeProvider:
    name = ProviderName.OLLAMA

    def __init__(self, payload: dict, confidence: float = 0.9, fail: bool = False) -> None:
        self._payload = payload
        self._confidence = confidence
        self._fail = fail
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self._fail:
            raise LLMProviderError("boom")
        return LLMResponse(
            content=json.dumps(self._payload),
            provider=self.name,
            model="fake",
            confidence_score=self._confidence,
            latency_ms=5,
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_mode=LLMMode.CHEAP,
        chroma_persist_dir=tmp_path / "chroma",
        api_key=None,
        rate_limit_per_minute=0,
    )


@pytest.fixture
def fake_payload() -> dict:
    return {
        "error_type": "KeyError",
        "root_cause": "Missing user_id column.",
        "suggested_fix": ["Validate schema", "Backfill column"],
        "severity": "high",
        "confidence": 0.88,
        "tags": ["pandas", "schema"],
    }


@pytest.fixture
def client(settings: Settings, fake_payload: dict, monkeypatch: pytest.MonkeyPatch):
    configure_logging()
    chroma = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    fake = _FakeProvider(fake_payload, confidence=0.9)
    router = LLMRouter(ollama=fake, openai=fake, settings=settings)
    indexer = IncidentIndexer(client=chroma, collection="test_incidents")
    retriever = IncidentRetriever(client=chroma, collection="test_incidents")
    parser = LogParser()

    app.dependency_overrides[deps.get_router] = lambda: router
    app.dependency_overrides[deps.get_indexer] = lambda: indexer
    app.dependency_overrides[deps.get_retriever] = lambda: retriever
    app.dependency_overrides[deps.get_parser] = lambda: parser

    with TestClient(app) as c:
        yield c, fake

    app.dependency_overrides.clear()


def test_analyze_happy_path(client) -> None:
    c, _ = client
    resp = c.post(
        "/analyze",
        json={
            "pipeline": "etl",
            "stage": "transform",
            "log_excerpt": 'KeyError: "user_id" not found',
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["error_type"] == "KeyError"
    assert data["severity"] == "high"
    assert "Validate schema" in data["suggested_fix"]
    assert data["llm"]["provider"] == "ollama"
    assert data["incident_id"]


