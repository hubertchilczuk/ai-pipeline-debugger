"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status

from api.dependencies import IndexerDep, ParserDep, RetrieverDep, RouterDep
from api.middleware import install_middleware
from api.schemas import (
    AnalyzeBatchRequest,
    AnalyzeBatchResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackView,
    HealthResponse,
    LLMTrace,
    Severity,
    SimilarIncident,
)
from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.utils import to_str
from ingestion.parser import ParsedLog
from llm.base import LLMRequest, LLMResponse, ProviderName
from llm.prompt_templates import SYSTEM_ANALYZE, USER_ANALYZE_TEMPLATE
from observability.tracing import init_tracing, instrument_app

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_tracing(service_name="ai-pipeline-debugger")
    logger.info("api_starting")
    yield
    logger.info("api_stopping")


app = FastAPI(
    title="AI Pipeline Debugger",
    version="0.1.0",
    description="Analyze data-pipeline failures with LLMs and semantic incident retrieval.",
    lifespan=lifespan,
)
install_middleware(app, get_settings())
instrument_app(app)


@app.get("/health", response_model=HealthResponse)
async def health(router: RouterDep) -> HealthResponse:
    providers = await router.health()
    overall = "ok" if any(providers.values()) else "degraded"
    return HealthResponse(status=overall, providers={p.value: ok for p, ok in providers.items()})


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={502: {"model": ErrorResponse}},
)
async def analyze(
    payload: AnalyzeRequest,
    router: RouterDep,
    parser: ParserDep,
    retriever: RetrieverDep,
    indexer: IndexerDep,
) -> AnalyzeResponse:
    return await _analyze_one(payload, router, parser, retriever, indexer)


@app.post(
    "/analyze/batch",
    response_model=AnalyzeBatchResponse,
)
async def analyze_batch(
    payload: AnalyzeBatchRequest,
    router: RouterDep,
    parser: ParserDep,
    retriever: RetrieverDep,
    indexer: IndexerDep,
) -> AnalyzeBatchResponse:
    """Analyze up to N logs concurrently. Per-item failures are reported, not raised."""
    sem = asyncio.Semaphore(payload.concurrency)

    async def _bounded(item: AnalyzeRequest) -> AnalyzeResponse | ErrorResponse:
        async with sem:
            try:
                return await _analyze_one(item, router, parser, retriever, indexer)
            except HTTPException as exc:
                return ErrorResponse(error="analyze_failed", detail=str(exc.detail))
            except Exception as exc:  # noqa: BLE001 — batch must not abort on one item
                logger.exception("batch_item_failed", error=str(exc))
                return ErrorResponse(error="internal_error", detail=str(exc))

    results = await asyncio.gather(*(_bounded(item) for item in payload.items))
    return AnalyzeBatchResponse(
        results=[r for r in results if isinstance(r, AnalyzeResponse)],
        errors=[r for r in results if isinstance(r, ErrorResponse)],
    )


@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(payload: FeedbackRequest, indexer: IndexerDep) -> None:
    await indexer.attach_feedback(
        incident_id=payload.incident_id,
        helpful=payload.helpful,
        actual_fix=payload.actual_fix,
        notes=payload.notes,
    )


@app.get(
    "/analyze/{incident_id}/feedback",
    response_model=FeedbackView,
    responses={404: {"model": ErrorResponse}},
)
async def get_feedback(incident_id: str, indexer: IndexerDep) -> FeedbackView:
    record = await indexer.get_record(incident_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")
    return FeedbackView(
        incident_id=incident_id,
        helpful=record.get("feedback_helpful") or None,
        suggested_fix=record.get("suggested_fix"),
        notes=record.get("feedback_notes"),
        error_type=record.get("error_type"),
        pipeline=record.get("pipeline"),
    )


# ---------- internal ----------
async def _analyze_one(
    payload: AnalyzeRequest,
    router,
    parser,
    retriever,
    indexer,
) -> AnalyzeResponse:
    parsed: ParsedLog = await parser.parse(payload.log_excerpt)

    similar = await retriever.find_similar(
        query=parsed.message or payload.log_excerpt[:500],
        filters={"pipeline": payload.pipeline} if payload.pipeline else None,
        top_k=5,
    )
    context = _format_context(similar)

    llm_request = LLMRequest(
        system_prompt=SYSTEM_ANALYZE,
        user_prompt=USER_ANALYZE_TEMPLATE.format(
            pipeline=payload.pipeline,
            stage=payload.stage,
            timestamp=payload.timestamp.isoformat() if payload.timestamp else "unknown",
            log_excerpt=payload.log_excerpt[:8000],
            retrieved_context=context,
        ),
    )
    response: LLMResponse = await router.generate(llm_request)

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM returned non-JSON output: {exc}",
        ) from exc

    incident_id = uuid.uuid4().hex
    await indexer.index_analysis(
        incident_id=incident_id,
        request=payload,
        parsed=parsed,
        analysis=data,
    )

    return AnalyzeResponse(
        incident_id=incident_id,
        error_type=to_str(data.get("error_type") or parsed.error_class or "Unknown"),
        root_cause=to_str(data.get("root_cause")),
        suggested_fix=to_str(data.get("suggested_fix")),
        severity=Severity(data.get("severity", "medium")),
        confidence=float(data.get("confidence", response.confidence_score)),
        tags=list(data.get("tags", [])),
        similar_incidents=[
            SimilarIncident(
                incident_id=hit.id,
                similarity=hit.similarity,
                error_type=hit.metadata.get("error_type"),
                suggested_fix=to_str(hit.metadata.get("suggested_fix")),
                pipeline=hit.metadata.get("pipeline"),
            )
            for hit in similar
        ],
        llm=LLMTrace(
            provider=response.provider.value,
            model=response.model,
            confidence=response.confidence_score,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            fallback_used=response.provider is ProviderName.OPENAI,
        ),
    )


def _format_context(hits: list) -> str:
    if not hits:
        return "(no similar incidents found)"
    return "\n".join(
        f"- [{h.similarity:.2f}] {h.metadata.get('error_type', '?')}: "
        f"{h.metadata.get('suggested_fix', '')[:200]}"
        for h in hits
    )
