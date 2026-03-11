"""Semantic retrieval of similar past incidents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb


@dataclass(slots=True)
class RetrievalHit:
    id: str
    similarity: float           # 1 - cosine_distance, clipped to [0, 1]
    document: str
    metadata: dict[str, Any]


class IncidentRetriever:
    def __init__(self, client: chromadb.ClientAPI, collection: str) -> None:
        self._collection = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def find_similar(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        # Chroma should return parallel lists of equal length; if it doesn't, that's
        # a contract violation we want to surface (silent truncation hides bugs).
        if not (len(ids) == len(docs) == len(metas) == len(dists)):
            raise RuntimeError(
                "Chroma returned mismatched result lists: "
                f"ids={len(ids)} docs={len(docs)} metas={len(metas)} dists={len(dists)}"
            )

        hits: list[RetrievalHit] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(
                RetrievalHit(
                    id=_id,
                    similarity=similarity,
                    document=doc or "",
                    metadata=dict(meta or {}),
                )
            )
        return hits
