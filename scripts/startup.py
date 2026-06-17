"""
FILE: scripts/startup.py
PURPOSE: Container initialisation sequence executed before ADK starts.

Responsibilities:
  1. Wait for Redis, PostgreSQL, and ChromaDB to become available.
  2. Run Alembic migrations (idempotent — safe to run on every restart).
  3. Ingest knowledge documents into ChromaDB (incremental — skips unchanged docs).

Design decisions:
  - Uses exponential back-off for service readiness checks.
  - Each phase is independent; if ChromaDB is not configured it is skipped.
  - Exits with non-zero code on critical failures so Docker restarts the container.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | startup | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("startup")


# ─────────────────────────────────────────────────────────────────────────────
# Service readiness helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for(name: str, check_fn, max_attempts: int = 20, delay_s: float = 3.0) -> None:
    """Retry check_fn with exponential backoff until it succeeds or we give up."""
    for attempt in range(1, max_attempts + 1):
        try:
            check_fn()
            logger.info("✓ %s is ready (attempt %d/%d)", name, attempt, max_attempts)
            return
        except Exception as exc:
            wait = min(delay_s * (1.5 ** (attempt - 1)), 30.0)
            logger.warning(
                "✗ %s not ready (attempt %d/%d): %s — retrying in %.1fs",
                name, attempt, max_attempts, exc, wait,
            )
            time.sleep(wait)
    logger.error("FATAL: %s did not become ready after %d attempts.", name, max_attempts)
    sys.exit(1)


def _check_redis(url: str) -> None:
    import redis as redis_lib
    client = redis_lib.from_url(url, socket_connect_timeout=3)
    client.ping()
    client.close()


def _check_postgres(url: str) -> None:
    import psycopg2
    conn = psycopg2.connect(url, connect_timeout=3)
    conn.close()


def _check_chromadb(host: str, port: int) -> None:
    import chromadb
    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()


# ─────────────────────────────────────────────────────────────────────────────
# Migrations
# ─────────────────────────────────────────────────────────────────────────────

def _run_migrations(database_url: str) -> None:
    logger.info("Running Alembic migrations...")
    try:
        from alembic import command as alembic_cmd
        from alembic.config import Config

        alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        alembic_cmd.upgrade(alembic_cfg, "head")
        logger.info("✓ Migrations complete.")
    except Exception as exc:
        # Keep startup non-blocking until migration scaffolding is present.
        logger.warning("Migrations skipped (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_knowledge(chroma_host: str, chroma_port: int, knowledge_dir: str, collection_name: str) -> None:
    logger.info("Ingesting knowledge documents into ChromaDB...")
    try:
        from src.rag.indexer import KnowledgeIndexer
        indexer = KnowledgeIndexer(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=collection_name,
            knowledge_dir=knowledge_dir,
        )
        stats = indexer.ingest_incremental()
        logger.info(
            "✓ Knowledge ingestion complete: added=%d skipped=%d total=%d",
            stats["added"],
            stats["skipped"],
            stats["total"],
        )
    except Exception as exc:
        logger.warning("Knowledge ingestion failed (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    redis_url = os.getenv("REDIS_URL", "")
    database_url = os.getenv("DATABASE_URL", "")
    chroma_host = os.getenv("CHROMA_HOST", "")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_collection = os.getenv("CHROMA_COLLECTION", "ecombot_knowledge")
    knowledge_dir = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge")
    skip_ingestion = os.getenv("SKIP_KNOWLEDGE_INGESTION", "false").lower() in {"1", "true", "yes"}

    # ── Wait for services ────────────────────────────────────────────────────
    if redis_url:
        _wait_for("Redis", lambda: _check_redis(redis_url))

    if database_url:
        _wait_for("PostgreSQL", lambda: _check_postgres(database_url))

    if chroma_host:
        _wait_for("ChromaDB", lambda: _check_chromadb(chroma_host, chroma_port))

    # ── Database migrations ──────────────────────────────────────────────────
    if database_url:
        _run_migrations(database_url)

    # ── Knowledge ingestion ──────────────────────────────────────────────────
    if chroma_host and not skip_ingestion:
        _ingest_knowledge(chroma_host, chroma_port, knowledge_dir, chroma_collection)
    elif chroma_host and skip_ingestion:
        logger.info("Knowledge ingestion skipped (SKIP_KNOWLEDGE_INGESTION=true).")

    logger.info("============================================================")
    logger.info("  Startup complete. Handing off to ADK web server.")
    logger.info("============================================================")


if __name__ == "__main__":
    main()

