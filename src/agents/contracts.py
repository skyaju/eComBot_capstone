from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.services.session_service import SessionContext

IntentName = Literal["product", "order", "knowledge", "support"]


class AgentContext(BaseModel):
    session_id: str
    user_id: str
    session_context: SessionContext = Field(default_factory=SessionContext)


class AgentRequest(BaseModel):
    message_id: str
    message: str
    context: AgentContext
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_name: str
    handled: bool
    message: str
    tool_name: str | None = None
    confidence: float | None = None
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    intent: IntentName
    selected_agent: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

