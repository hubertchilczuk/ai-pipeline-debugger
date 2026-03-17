"""Pydantic v2 schemas for API I/O."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------- Request models ----------
class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: Annotated[str, Field(min_length=1, max_length=200, examples=["etl_users_daily"])]
    stage: Annotated[str, Field(min_length=1, max_length=200, examples=["transform_users"])]
    log_excerpt: Annotated[str, Field(min_length=10, max_length=64_000)]
    timestamp: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class AnalyzeBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: Annotated[list[AnalyzeRequest], Field(min_length=1, max_length=50)]
    concurrency: Annotated[int, Field(ge=1, le=16, description="Max parallel analyses")] = 4


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    helpful: bool
    actual_fix: str | None = Field(None, max_length=4000)
    notes: str | None = Field(None, max_length=2000)


# ---------- Response models ----------
class SimilarIncident(BaseModel):
    incident_id: str
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    error_type: str | None
    suggested_fix: str | None
    pipeline: str | None


class LLMTrace(BaseModel):
    provider: str
    model: str
    confidence: float
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    fallback_used: bool = False


class AnalyzeResponse(BaseModel):
    incident_id: str
    error_type: str
    root_cause: str
    suggested_fix: str
    severity: Severity
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    tags: list[str] = Field(default_factory=list)
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    llm: LLMTrace


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class AnalyzeBatchResponse(BaseModel):
    results: list[AnalyzeResponse] = Field(default_factory=list)
    errors: list[ErrorResponse] = Field(default_factory=list)


class FeedbackView(BaseModel):
    incident_id: str
    helpful: str | None = None
    suggested_fix: str | None = None
    notes: str | None = None
    error_type: str | None = None
    pipeline: str | None = None
