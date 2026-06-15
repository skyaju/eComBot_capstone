"""
FILE: src/services/redis_session_service.py
PURPOSE: Redis-backed implementation of SessionRepository.

Replaces InMemorySessionRepository for persistent, cross-container session storage.

Design decisions:
  - Uses redis-py with connection pooling (thread-safe, production-ready).
  - SessionState is serialised as JSON (pydantic model_dump_json).
  - TTL is set on every save() so hot sessions never expire unexpectedly.
  - Falls back gracefully: if Redis is unreachable the factory in
    session_service.py returns InMemorySessionRepository instead.
  - Keys follow the pattern: ecombot:session:<session_id>
  - Designed for future Redis Cluster/Sentinel migration with no API changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import redis

from src.services.session_service import SessionRepository, SessionState

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "ecombot:session:"


class RedisSessionRepository(SessionRepository):
    """
    Redis-backed session repository with connection pooling and TTL support.

    Usage:
        repo = RedisSessionRepository(redis_url="redis://localhost:6379/0")
        service = SessionService(repository=repo)
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 86_400) -> None:
        self._ttl = ttl_seconds
        # Connection pool: max_connections=20 handles concurrent ADK requests.
        self._pool = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        logger.info(
            "RedisSessionRepository initialised | url=%s | ttl=%ds",
            self._redact_url(redis_url),
            ttl_seconds,
        )

    # ── SessionRepository Protocol ────────────────────────────────────────────

    def create(self, session_id: str, user_id: str) -> SessionState:
        existing = self.get(session_id)
        if existing is not None:
            return existing
        session = SessionState(session_id=session_id, user_id=user_id)
        self.save(session)
        logger.debug("Session created in Redis | session_id=%s", session_id)
        return session

    def get(self, session_id: str) -> SessionState | None:
        try:
            raw = self._client.get(self._key(session_id))
            if raw is None:
                return None
            return SessionState.model_validate_json(raw)
        except redis.RedisError as exc:
            logger.error("Redis get failed for session %s: %s", session_id, exc)
            return None

    def save(self, session: SessionState) -> None:
        try:
            session.updated_at = datetime.now(timezone.utc)
            self._client.setex(
                name=self._key(session.session_id),
                time=self._ttl,
                value=session.model_dump_json(),
            )
        except redis.RedisError as exc:
            logger.error("Redis save failed for session %s: %s", session.session_id, exc)

    def clear(self, session_id: str) -> bool:
        try:
            result = self._client.delete(self._key(session_id))
            return bool(result)
        except redis.RedisError as exc:
            logger.error("Redis clear failed for session %s: %s", session_id, exc)
            return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self._client.ping()  # type: ignore[return-value]
        except redis.RedisError:
            return False

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{session_id}"

    @staticmethod
    def _redact_url(url: str) -> str:
        """Mask password in Redis URL for safe logging."""
        if "@" in url:
            parts = url.split("@")
            credentials = parts[0].split("//")[-1]
            if ":" in credentials:
                user, _ = credentials.split(":", 1)
                return url.replace(credentials, f"{user}:***")
        return url

