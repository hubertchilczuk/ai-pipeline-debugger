"""Index parsed incidents into ChromaDB."""
from __future__ import annotations

from typing import Any

import chromadb

from api.schemas import AnalyzeRequest
from core.utils import to_str
from ingestion.collectors import RawLog
from ingestion.parser import ParsedLog


class IncidentIndexer:
    def __init__(self, client: chromadb.ClientAPI, collection: str) -> None:
        self._collection = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def index(self, raw: RawLog, parsed: ParsedLog) -> None:
        """Index a parsed log without an LLM analysis (ingestion path)."""
        doc_id = self._make_id(raw.pipeline, raw.stage, parsed)
        self._collection.upsert(
            ids=[doc_id],
            documents=[self._build_document(parsed)],
            metadatas=[self._build_metadata(raw, parsed)],
        )

    async def index_analysis(
        self,
        incident_id: str,
        request: AnalyzeRequest,
        parsed: ParsedLog,
        analysis: dict[str, Any],
    ) -> None:
        """Index after LLM analysis — enriches metadata with the suggested fix."""
        metadata: dict[str, Any] = {
            "pipeline": request.pipeline,
            "stage": request.stage,
            "error_type": analysis.get("error_type") or (parsed.error_class or "Unknown"),
            "severity": analysis.get("severity", "medium"),
            "suggested_fix": to_str(analysis.get("suggested_fix"))[:1000],
            "tags": ",".join(analysis.get("tags", []) or []),
            "extraction_method": parsed.extraction_method.value,
            "feedback_helpful": "",
        }
        if parsed.timestamp:
            metadata["timestamp"] = parsed.timestamp.isoformat()

        self._collection.upsert(
            ids=[incident_id],
            documents=[self._build_document(parsed)],
            metadatas=[metadata],
        )

    async def attach_feedback(
        self,
        incident_id: str,
        helpful: bool,
        actual_fix: str | None,
        notes: str | None,
    ) -> None:
        update: dict[str, Any] = {"feedback_helpful": "yes" if helpful else "no"}
        if actual_fix:
            update["suggested_fix"] = actual_fix[:1000]
        if notes:
            update["feedback_notes"] = notes[:1000]
        self._collection.update(ids=[incident_id], metadatas=[update])

    async def get_record(self, incident_id: str) -> dict[str, Any] | None:
        """Fetch raw metadata + document for an incident, or None if missing."""
        result = self._collection.get(ids=[incident_id], include=["metadatas", "documents"])
        ids = result.get("ids") or []
        if not ids:
            return None
        metas = result.get("metadatas") or [{}]
        docs = result.get("documents") or [""]
        record = dict(metas[0] or {})
        record["document"] = docs[0] if docs else ""
        return record

    # ---------- helpers ----------
    @staticmethod
    def _build_document(parsed: ParsedLog) -> str:
        """Document = the semantic surface we want to retrieve against.

        We concatenate error class + message + stack summary because users typically
        search by symptom (e.g. paste an error line). Long raw logs hurt recall.
        """
        parts = [
            parsed.error_class or "",
            parsed.message or "",
            parsed.stack_summary or "",
        ]
        return "\n".join(p for p in parts if p)[:4000]

    @staticmethod
    def _build_metadata(raw: RawLog, parsed: ParsedLog) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "pipeline": raw.pipeline,
            "stage": raw.stage,
            "source": raw.source,
            "error_type": parsed.error_class or "Unknown",
            "extraction_method": parsed.extraction_method.value,
        }
        if parsed.timestamp:
            meta["timestamp"] = parsed.timestamp.isoformat()
        if parsed.tags:
            meta["tags"] = ",".join(parsed.tags)
        return meta

    @staticmethod
    def _make_id(pipeline: str, stage: str, parsed: ParsedLog) -> str:
        ts = parsed.timestamp.isoformat() if parsed.timestamp else "no-ts"
        ec = parsed.error_class or "unknown"
        return f"{pipeline}::{stage}::{ec}::{ts}"
