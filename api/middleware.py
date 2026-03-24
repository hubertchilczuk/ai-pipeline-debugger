"""HTTP middleware: API-key auth, in-process rate limiting, CORS."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import Settings
from core.logger import get_logger

logger = get_logger(__name__)

_OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforces X-API-Key when settings.api_key is set. No-op otherwise."""

    def __init__(self, app, expected: str | None) -> None:
        super().__init__(app)
        self._expected = expected

    async def dispatch(self, request: Request, call_next):
        if self._expected and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if provided != self._expected:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "unauthorized", "detail": "invalid or missing X-API-Key"},
                )
        return await call_next(request)


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limit per client (API key, else client IP).

    Single-process only — for multi-replica deployments swap for Redis-backed limiter.
    """

    def __init__(self, app, per_minute: int) -> None:
        super().__init__(app)
        self._limit = per_minute
        self._window = 60.0
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self._limit <= 0 or request.url.path in _OPEN_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
        now = time.monotonic()
        bucket = self._buckets[key]
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self._limit:
            retry_after = int(self._window - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "rate_limited", "detail": f"retry after {retry_after}s"},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)


def install_middleware(app: FastAPI, settings: Settings) -> None:
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(InMemoryRateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
    app.add_middleware(APIKeyMiddleware, expected=settings.api_key)
    if settings.api_key:
        logger.info("api_key_auth_enabled")
    if settings.rate_limit_per_minute:
        logger.info("rate_limit_enabled", per_minute=settings.rate_limit_per_minute)
