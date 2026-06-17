from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class UISessionState:
    session_id: str
    user_id: str
    explainability_enabled: bool = True
    last_agent: str | None = None
    last_tool: str | None = None
    last_sources: list[str] = field(default_factory=list)
    last_response_message: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def new_ui_session_id(prefix: str = "chainlit") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def display_agent_name(agent_name: str) -> str:
    mapping = {
        "support_agent": "Support Agent",
        "product_agent": "Product Agent",
        "order_agent": "Order Agent",
        "knowledge_agent": "Knowledge Agent",
        "router_agent": "Router Agent",
    }
    return mapping.get(agent_name, agent_name.replace("_", " ").title())

