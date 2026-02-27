"""Ingestion orchestration: collect -> parse -> index."""
from __future__ import annotations

from ingestion.collectors import RawLog
from ingestion.parser import LogParser, ParsedLog
from vector_db.indexer import IncidentIndexer


class IngestionPipeline:
    def __init__(self, parser: LogParser, indexer: IncidentIndexer) -> None:
        self._parser = parser
        self._indexer = indexer

    async def ingest(self, raw: RawLog) -> ParsedLog:
        parsed = await self._parser.parse(raw.content)
        await self._indexer.index(raw=raw, parsed=parsed)
        return parsed
