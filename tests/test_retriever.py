"""Smoke tests for semantic retrieval against a temporary ChromaDB."""
from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from vector_db.retriever import IncidentRetriever


@pytest.fixture
def chroma_client(tmp_path: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.mark.asyncio
async def test_retrieves_semantically_similar_incident(chroma_client: chromadb.ClientAPI) -> None:
    coll = chroma_client.get_or_create_collection("test", metadata={"hnsw:space": "cosine"})
    coll.upsert(
        ids=["inc-1", "inc-2"],
        documents=[
            "OutOfMemoryError Java heap space in spark executor",
            "KeyError 'user_id' in pandas dataframe",
        ],
        metadatas=[
            {"pipeline": "etl", "error_type": "OutOfMemoryError"},
            {"pipeline": "ml", "error_type": "KeyError"},
        ],
    )

    retriever = IncidentRetriever(client=chroma_client, collection="test")
    hits = await retriever.find_similar("spark OOM heap", top_k=2)

    assert hits, "expected at least one hit"
    assert hits[0].metadata["error_type"] == "OutOfMemoryError"
    assert 0.0 <= hits[0].similarity <= 1.0


@pytest.mark.asyncio
async def test_filter_by_pipeline(chroma_client: chromadb.ClientAPI) -> None:
    coll = chroma_client.get_or_create_collection("test2", metadata={"hnsw:space": "cosine"})
    coll.upsert(
        ids=["a", "b"],
        documents=["timeout", "timeout"],
        metadatas=[{"pipeline": "etl"}, {"pipeline": "ml"}],
    )
    retriever = IncidentRetriever(client=chroma_client, collection="test2")
    hits = await retriever.find_similar("timeout", filters={"pipeline": "ml"})
    assert all(h.metadata["pipeline"] == "ml" for h in hits)


@pytest.mark.asyncio
async def test_empty_query_returns_no_hits(chroma_client: chromadb.ClientAPI) -> None:
    retriever = IncidentRetriever(client=chroma_client, collection="empty")
    assert await retriever.find_similar("   ") == []
