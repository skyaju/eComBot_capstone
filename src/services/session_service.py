from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from threading import RLock
from typing import Iterator, Literal, Protocol
import contextvars

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ORDER_ID_PATTERN = re.compile(r"ORD-\d{5}", re.IGNORECASE)
_NAME_PATTERN = re.compile(
    r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\-']{1,30}(?:\s+[A-Za-z][A-Za-z\-']{1,30})?)",
    re.IGNORECASE,
)

SupportTopic = Literal[
    "product_search",
    "order_lookup",
    "shipping",
    "returns_refunds",
    "knowledge",
    "faq",
    "general",
]
ToolName = Literal["search_products", "lookup_order", "retrieve_faq", "search_knowledge"]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    message: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionContext(BaseModel):
    customer_name: str | None = None
    recent_products: list[str] = Field(default_factory=list)
    recent_order_ids: list[str] = Field(default_factory=list)
    active_support_topic: SupportTopic | None = None
    last_tool_used: ToolName | None = None
    conversation_summary: str = ""
    shipping_inquiry_active: bool = False
    return_refund_discussed: bool = False


class SessionState(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: SessionContext = Field(default_factory=SessionContext)
    turns: list[ConversationTurn] = Field(default_factory=list)


class SessionRepository(Protocol):
    def create(self, session_id: str, user_id: str) -> SessionState: ...

    def get(self, session_id: str) -> SessionState | None: ...

    def save(self, session: SessionState) -> None: ...

    def clear(self, session_id: str) -> bool: ...


class InMemorySessionRepository(SessionRepository):
    """Local repository implementation; can be replaced by Redis-backed storage later."""

    def __init__(self) -> None:
        self._store: dict[str, SessionState] = {}
        self._lock = RLock()

    def create(self, session_id: str, user_id: str) -> SessionState:
        with self._lock:
            existing = self._store.get(session_id)
            if existing is not None:
                return existing
            session = SessionState(session_id=session_id, user_id=user_id)
            self._store[session_id] = session
            return session

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._store.get(session_id)

    def save(self, session: SessionState) -> None:
        with self._lock:
            session.updated_at = datetime.now(timezone.utc)
            self._store[session.session_id] = session

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._store.pop(session_id, None) is not None


class SessionService:
    """Session manager with context extraction and memory-aware helpers."""

    def __init__(self, repository: SessionRepository | None = None) -> None:
        self._repository = repository or InMemorySessionRepository()
        self._active_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "active_session_id", default=None
        )

    def create_session(self, session_id: str, user_id: str) -> SessionState:
        session = self._repository.create(session_id=session_id, user_id=user_id)
        logger.debug("Session created/loaded | session_id=%s | user_id=%s", session_id, user_id)
        return session

    def load_session(self, session_id: str) -> SessionState | None:
        return self._repository.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        cleared = self._repository.clear(session_id)
        logger.info("Session cleared=%s | session_id=%s", cleared, session_id)
        return cleared

    def get_current_session_id(self) -> str | None:
        return self._active_session_id.get()

    @contextmanager
    def activate_session(self, session_id: str) -> Iterator[None]:
        token = self._active_session_id.set(session_id)
        try:
            yield
        finally:
            self._active_session_id.reset(token)

    def remember_user_message(self, session_id: str, message: str) -> SessionState:
        session = self._require_session(session_id)
        session.turns.append(ConversationTurn(role="user", message=message))

        self._extract_customer_name(session, message)
        self._extract_order_ids(session, message)
        self._update_topic_from_text(session, message)

        self._repository.save(session)
        return session

    def remember_assistant_message(self, session_id: str, message: str) -> SessionState:
        session = self._require_session(session_id)
        if message.strip():
            session.turns.append(ConversationTurn(role="assistant", message=message.strip()))
        self._repository.save(session)
        return session

    def remember_tool_result(self, session_id: str, tool_name: ToolName, result: dict[str, object]) -> SessionState:
        session = self._require_session(session_id)
        session.context.last_tool_used = tool_name

        if tool_name == "search_products":
            session.context.active_support_topic = "product_search"
            if bool(result.get("success")):
                products_payload = result.get("products", [])
                if not isinstance(products_payload, list):
                    products_payload = []
                for product in products_payload:
                    if isinstance(product, dict) and isinstance(product.get("name"), str):
                        self._push_unique(session.context.recent_products, product["name"], max_items=5)

        if tool_name == "lookup_order":
            session.context.active_support_topic = "order_lookup"
            if bool(result.get("success")) and isinstance(result.get("order_id"), str):
                self._push_unique(session.context.recent_order_ids, str(result["order_id"]), max_items=5)
                tracking = result.get("tracking")
                session.context.shipping_inquiry_active = isinstance(tracking, dict)

        if tool_name == "retrieve_faq":
            session.context.active_support_topic = "faq"

        if tool_name == "search_knowledge":
            session.context.active_support_topic = "knowledge"

        self._repository.save(session)
        return session

    def resolve_order_id(self, session_id: str, explicit_order_id: str | None) -> str | None:
        if explicit_order_id and explicit_order_id.strip():
            return explicit_order_id.strip().upper()

        session = self.load_session(session_id)
        if not session or not session.context.recent_order_ids:
            return None
        return session.context.recent_order_ids[-1]

    def summarize_conversation(self, session_id: str) -> str:
        session = self._require_session(session_id)
        context = session.context

        customer = context.customer_name or "customer"
        recent_product = context.recent_products[-1] if context.recent_products else "none"
        recent_order = context.recent_order_ids[-1] if context.recent_order_ids else "none"
        topic = context.active_support_topic or "general"

        summary = (
            f"Customer={customer}; topic={topic}; "
            f"recent_product={recent_product}; recent_order={recent_order}; "
            f"return_refund_discussed={context.return_refund_discussed}; "
            f"shipping_inquiry_active={context.shipping_inquiry_active}."
        )

        context.conversation_summary = summary
        self._repository.save(session)
        return summary

    def build_context_hint(self, session_id: str) -> str:
        session = self.load_session(session_id)
        if session is None:
            return ""

        context = session.context
        if not context.conversation_summary:
            self.summarize_conversation(session_id)
        return context.conversation_summary

    def _require_session(self, session_id: str) -> SessionState:
        session = self.load_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    @staticmethod
    def _push_unique(buffer: list[str], value: str, max_items: int) -> None:
        if value in buffer:
            buffer.remove(value)
        buffer.append(value)
        if len(buffer) > max_items:
            del buffer[0 : len(buffer) - max_items]

    @staticmethod
    def _extract_customer_name(session: SessionState, message: str) -> None:
        match = _NAME_PATTERN.search(message)
        if match:
            session.context.customer_name = match.group(1).strip()

    @staticmethod
    def _extract_order_ids(session: SessionState, message: str) -> None:
        for order_id in _ORDER_ID_PATTERN.findall(message):
            normalized = order_id.upper()
            SessionService._push_unique(session.context.recent_order_ids, normalized, max_items=5)

    @staticmethod
    def _update_topic_from_text(session: SessionState, message: str) -> None:
        text = message.lower()

        if any(word in text for word in {"return", "refund", "exchange"}):
            session.context.return_refund_discussed = True
            session.context.active_support_topic = "returns_refunds"
            return

        if any(word in text for word in {"shipping", "deliver", "tracking", "shipped"}):
            session.context.shipping_inquiry_active = True
            session.context.active_support_topic = "shipping"
            return

        if any(word in text for word in {"order", "ord-"}):
            session.context.active_support_topic = "order_lookup"
            return

        if any(word in text for word in {"product", "laptop", "shoe", "watch", "jacket", "alternatives"}):
            session.context.active_support_topic = "product_search"
            return

        if any(word in text for word in {"policy", "faq", "warranty", "payment"}):
            session.context.active_support_topic = "faq"
            return

        if any(word in text for word in {"loyalty", "spec", "feature", "guarantee", "coverage"}):
            session.context.active_support_topic = "knowledge"
            return

        session.context.active_support_topic = "general"


@lru_cache(maxsize=1)
def get_session_service() -> SessionService:
    """Singleton session service shared by runner and tools in local mode."""

    return SessionService()

