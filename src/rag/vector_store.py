"""
FILE: src/rag/vector_store.py
PURPOSE: Thin wrapper around the ChromaDB HTTP client.

Responsibilities:
  - Connect to the ChromaDB server (Docker service).
  - Create or load a persistent collection.
  - Expose upsert / query / count operations.
  - Use fastembed (ONNX, no PyTorch) for embedding generation.

Design decisions:
  - Wraps chromadb.HttpClient rather than PersistentClient so the vector
    data is stored in the dedicated ChromaDB container (not inside ecombot).
  - FastEmbedEmbeddingFunction uses BAAI/bge-small-en-v1.5 (35 MB ONNX).
  - The embedding function is instantiated once and shared across calls.
  - All ChromaDB errors are caught and re-raised as VectorStoreError so
    callers can provide graceful fallback to TF-IDF KnowledgeService.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB is unavailable or a vector operation fails."""


class _FastEmbedFunction:
    """
    Minimal chromadb.EmbeddingFunction wrapper around fastembed.
    fastembed uses ONNX Runtime — no PyTorch required.
    The model (BAAI/bge-small-en-v1.5, ~35 MB) is pre-downloaded in the
    Docker build stage and cached to FASTEMBED_CACHE_DIR.
    """

    _instance: "_FastEmbedFunction | None" = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import]
        except ImportError as exc:
            raise VectorStoreError(
                "fastembed is not installed. Add 'fastembed' to requirements.txt."
            ) from exc
        logger.info("Loading fastembed model: %s", model_name)
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [emb.tolist() for emb in self._model.embed(input)]

    @classmethod
    def get(cls) -> "_FastEmbedFunction":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class ChromaVectorStore:
    """
    Manages a single ChromaDB collection for the knowledge base.

    Usage:
        store = ChromaVectorStore.connect(host="chromadb", port=8000)
        store.upsert(ids=["doc-1"], documents=["text..."], metadatas=[{...}])
        results = store.query("return policy", n_results=3)
    """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        try:
            import chromadb  # type: ignore[import]

            self._client = chromadb.HttpClient(host=host, port=port)
            self._ef = _FastEmbedFunction.get()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._ef,  # type: ignore[arg-type]
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaVectorStore connected | host=%s:%d | collection=%s | docs=%d",
                host,
                port,
                collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to ChromaDB at {host}:{port}: {exc}") from exc

    @classmethod
    def connect(cls, host: str, port: int, collection_name: str) -> "ChromaVectorStore":
        return cls(host=host, port=port, collection_name=collection_name)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas or [{} for _ in ids],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return top-N results as list of dicts with keys:
            id, document, metadata, distance, score
        """
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": min(n_results, max(self._collection.count(), 1)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self._collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Convert cosine distance (0-2) to a 0-1 similarity score
            score = max(0.0, round(1.0 - (dist / 2.0), 4))
            hits.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "score": score,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def get_all_ids(self) -> list[str]:
        try:
            result = self._collection.get(include=[])
            return result.get("ids", [])
        except Exception:
            return []

