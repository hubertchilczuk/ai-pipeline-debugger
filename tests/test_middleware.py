"""Tests for API-key auth and in-memory rate limiting middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import APIKeyMiddleware, InMemoryRateLimitMiddleware


def _make_app(api_key: str | None = None, rpm: int = 0) -> FastAPI:
    app = FastAPI()

    @app.get("/echo")
    def echo() -> dict:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.add_middleware(InMemoryRateLimitMiddleware, per_minute=rpm)
    app.add_middleware(APIKeyMiddleware, expected=api_key)
    return app


def test_api_key_allows_when_unset() -> None:
    c = TestClient(_make_app(api_key=None))
    assert c.get("/echo").status_code == 200


def test_api_key_blocks_missing_header() -> None:
    c = TestClient(_make_app(api_key="secret"))
    assert c.get("/echo").status_code == 401


def test_api_key_health_is_open() -> None:
    c = TestClient(_make_app(api_key="secret"))
    assert c.get("/health").status_code == 200


def test_api_key_passes_with_correct_header() -> None:
    c = TestClient(_make_app(api_key="secret"))
    assert c.get("/echo", headers={"X-API-Key": "secret"}).status_code == 200


def test_rate_limit_blocks_after_threshold() -> None:
    c = TestClient(_make_app(rpm=2))
    assert c.get("/echo").status_code == 200
    assert c.get("/echo").status_code == 200
    resp = c.get("/echo")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_rate_limit_disabled_when_zero() -> None:
    c = TestClient(_make_app(rpm=0))
    for _ in range(5):
        assert c.get("/echo").status_code == 200
