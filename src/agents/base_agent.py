from __future__ import annotations

from typing import Protocol

from src.agents.contracts import AgentRequest, AgentResponse


class DomainAgent(Protocol):
    name: str

    def handle(self, request: AgentRequest) -> AgentResponse:
        """Process one routed request and return a structured response."""

