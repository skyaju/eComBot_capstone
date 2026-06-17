"""
FILE: src/rag/retrieval_service.py
PURPOSE: ChromaDB-backed knowledge retrieval service.

Replaces the TF-IDF KnowledgeService (src/tools/services.py) when ChromaDB is
configured, while preserving the same output contract so existing tool wrappers
and agent tests are unaffected.

Design decisions:
  - Returns the exact same dict shape as KnowledgeService.search() so the
    existing adk_tools.py search_knowledge implementation requires no changes.
  - Session context is appended to the query text for personalised retrieval
    (same strategy as TF-IDF KnowledgeService).
  - Minimum confidence threshold is 0.25 (vs. 0.12 for TF-IDF) because
    cosine similarity from embeddings is a better-calibrated score.
  - Falls back gracefully: if ChromaDB is unreachable, raises VectorStoreError
    so the caller can fall back to TF-IDF.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.vector_store import ChromaVectorStore, VectorStoreError
from src.services.session_service import SessionService

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 0.25


class ChromaKnowledgeService:
    """
    Vector-search knowledge retrieval using ChromaDB + fastembed embeddings.

    Usage:
        service = ChromaKnowledgeService(
            store=ChromaVectorStore.connect(host="chromadb", port=8000),
            session_service=get_session_service(),
        )
        result = service.search(question="What is your return policy?", top_k=3)
    """

    def __init__(
        self,
        store: ChromaVectorStore,
        session_service: SessionService | None = None,
    ) -> None:
        self._store = store
        self._session_service = session_service

    def search(
        self,
        question: str,
        top_k: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant knowledge passages.

        Returns the same dict shape as KnowledgeService.search():
          {
            "success": bool,
            "question": str,
            "confidence": float,
            "results": [{"source", "category", "score", "snippet"}],
            "sources": [str],
          }
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error_code": "empty_question",
                "message": "Please provide a question so I can search the knowledge base.",
            }

        # Augment query with session context if available
        augmented_query = question
        if self._session_service and session_id:
            hint = self._session_service.build_context_hint(session_id)
            if hint:
                augmented_query = f"{question} {hint}"

        try:
            hits = self._store.query(augmented_query, n_results=top_k)
        except VectorStoreError as exc:
            logger.error("ChromaDB retrieval failed: %s", exc)
            return {
                "success": False,
                "error_code": "knowledge_not_found",
                "message": "Knowledge retrieval service is temporarily unavailable.",
                "question": question,
            }

        if not hits:
            return {
                "success": False,
                "error_code": "knowledge_not_found",
                "message": "I could not find relevant policy or product documentation for that question.",
                "question": question,
            }

        top_score = hits[0]["score"]
        if top_score < _MIN_CONFIDENCE:
            return {
                "success": False,
                "error_code": "low_confidence_retrieval",
                "message": "I found only weak matches. Please rephrase your question with more specifics.",
                "question": question,
                "confidence": round(top_score, 4),
            }

        snippets: list[dict[str, Any]] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            snippets.append(
                {
                    "source": meta.get("source", hit["id"]),
                    "category": meta.get("category", "general"),
                    "score": hit["score"],
                    "snippet": self._build_snippet(hit["document"], question),
                }
            )

        return {
            "success": True,
            "question": question,
            "confidence": round(top_score, 4),
            "results": snippets,
            "sources": [s["source"] for s in snippets],
        }

    @staticmethod
    def _build_snippet(text: str, query: str, max_length: int = 300) -> str:
        """Extract a query-anchored snippet from the document chunk."""
        clean = " ".join(text.split())
        lowered = clean.lower()
        anchor = 0
        for token in query.lower().split():
            pos = lowered.find(token)
            if pos != -1:
                anchor = max(pos - 80, 0)
                break
        snippet = clean[anchor: anchor + max_length]
        if anchor > 0:
            snippet = "..." + snippet
        if anchor + max_length < len(clean):
            snippet += "..."
        return snippet

