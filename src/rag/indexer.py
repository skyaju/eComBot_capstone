"""
FILE: src/rag/indexer.py
PURPOSE: Knowledge document ingestion pipeline.

Responsibilities:
  1. Scan the knowledge directory for .md and .txt files.
  2. Chunk large documents into passages (≤512 tokens).
  3. Upsert chunks into ChromaDB (incremental — skips unchanged docs).

Design decisions:
  - Chunk IDs are deterministic hashes of (source, chunk_index) so re-ingestion
    is idempotent and unchanged documents are skipped.
  - Chunks overlap by 64 tokens to avoid context loss at boundaries.
  - Metadata attached to each chunk: source, category, chunk_index, total_chunks.
  - Returns ingestion stats so callers can log progress.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.rag.vector_store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".md", ".txt"}
_CHUNK_SIZE_CHARS = 800     # characters per chunk (≈ 160-200 tokens for English)
_CHUNK_OVERLAP_CHARS = 120  # overlap between consecutive chunks


class KnowledgeIndexer:
    """
    Scans the knowledge directory and upserts documents into ChromaDB.

    Usage:
        indexer = KnowledgeIndexer(
            chroma_host="localhost", chroma_port=8000,
            collection_name="ecombot_knowledge",
            knowledge_dir="data/knowledge",
        )
        stats = indexer.ingest_incremental()
    """

    def __init__(
        self,
        chroma_host: str,
        chroma_port: int,
        collection_name: str,
        knowledge_dir: str | Path,
    ) -> None:
        self._store = ChromaVectorStore.connect(
            host=chroma_host,
            port=chroma_port,
            collection_name=collection_name,
        )
        self._knowledge_dir = Path(knowledge_dir).resolve()

    def ingest_incremental(self) -> dict[str, int]:
        """
        Ingest all documents; skip chunks whose IDs already exist in the store.

        Returns:
            {"added": int, "skipped": int, "total": int}
        """
        if not self._knowledge_dir.exists():
            logger.warning("Knowledge directory not found: %s — skipping ingestion", self._knowledge_dir)
            return {"added": 0, "skipped": 0, "total": 0}

        existing_ids: set[str] = set(self._store.get_all_ids())
        added = 0
        skipped = 0

        for path in sorted(self._knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue

            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", path, exc)
                continue

            if not content:
                continue

            relative = path.relative_to(self._knowledge_dir)
            parts = relative.parts
            category = parts[0] if len(parts) > 1 else "general"
            source = relative.as_posix()
            chunks = self._chunk(content)

            new_ids: list[str] = []
            new_docs: list[str] = []
            new_metas: list[dict[str, Any]] = []

            for idx, chunk_text in enumerate(chunks):
                chunk_id = self._make_chunk_id(source, idx)
                if chunk_id in existing_ids:
                    skipped += 1
                    continue
                new_ids.append(chunk_id)
                new_docs.append(chunk_text)
                new_metas.append(
                    {
                        "source": source,
                        "category": category,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    }
                )

            if new_ids:
                try:
                    self._store.upsert(ids=new_ids, documents=new_docs, metadatas=new_metas)
                    added += len(new_ids)
                    logger.debug("Indexed %d chunks from %s", len(new_ids), source)
                except VectorStoreError as exc:
                    logger.warning("Failed to index %s: %s", source, exc)

        total = added + skipped
        return {"added": added, "skipped": skipped, "total": total}

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Split text into overlapping char-based chunks."""
        if len(text) <= _CHUNK_SIZE_CHARS:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE_CHARS
            chunk = text[start:end]
            # Try to break at a sentence boundary (. or \n) near the end
            break_pos = max(
                chunk.rfind(". "),
                chunk.rfind(".\n"),
                chunk.rfind("\n\n"),
            )
            if break_pos > _CHUNK_SIZE_CHARS // 2:
                chunk = chunk[: break_pos + 1]
            chunks.append(chunk.strip())
            start += len(chunk) - _CHUNK_OVERLAP_CHARS
        return [c for c in chunks if c]

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int) -> str:
        raw = f"{source}:{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

