"""ChromaDB client factory (persistent local store)."""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import get_settings


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    s = get_settings()
    s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(s.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
    )
