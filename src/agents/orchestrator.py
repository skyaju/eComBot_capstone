"""
FILE: src/agents/orchestrator.py
PURPOSE: Multi-agent CLI orchestrator that:
  - Uses RouterAgent to classify customer intent
  - Delegates to the appropriate domain agent
  - Records execution traces for observability
  - Propagates shared SessionContext across all agents

Design decisions:
  - Domain agents are pure-Python classes; no LLM call needed for routing.
  - LLM synthesis is handled by the ADK LlmAgent layer; orchestrator only
    provides context-enriched dispatch.
  - Stateless per request: all state lives in SessionService.
  - New agents are added by registering them in the `_registry` dict.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.agents.contracts import AgentContext, AgentRequest, AgentResponse, RoutingDecision
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.order_agent import OrderAgent
from src.agents.product_agent import ProductAgent
from src.agents.router_agent import RouterAgent
from src.agents.support_runtime_agent import SupportAgent
from src.observability.tracing import ExecutionTrace, TraceLogger
from src.services.session_service import SessionService
from src.tools.data_loader import MockDataStore

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates requests through intent-based agent routing.

    Usage:
        orchestrator = MultiAgentOrchestrator.build(data_store, session_service)
        response = orchestrator.dispatch(session_id="s1", user_id="u1", message="...")
    """

    def __init__(
        self,
        router: RouterAgent,
        registry: dict[str, Any],
        session_service: SessionService,
        trace_logger: TraceLogger,
    ) -> None:
        self._router = router
        self._registry = registry  # agent_name -> domain agent instance
        self._session_service = session_service
        self._trace_logger = trace_logger

    @classmethod
    def build(
        cls,
        data_store: MockDataStore,
        session_service: SessionService,
        trace_logger: TraceLogger | None = None,
    ) -> "MultiAgentOrchestrator":
        """Factory: wire all domain agents and the router."""
        registry: dict[str, Any] = {
            "product_agent": ProductAgent(data_store=data_store, session_service=session_service),
            "order_agent": OrderAgent(data_store=data_store, session_service=session_service),
            "knowledge_agent": KnowledgeAgent(data_store=data_store, session_service=session_service),
            "support_agent": SupportAgent(),
        }
        return cls(
            router=RouterAgent.default(),
            registry=registry,
            session_service=session_service,
            trace_logger=trace_logger or TraceLogger(),
        )

    def dispatch(self, session_id: str, user_id: str, message: str) -> tuple[AgentResponse, RoutingDecision]:
        """
        Route one customer message to the best domain agent.

        Returns:
            (AgentResponse, RoutingDecision) so callers can log or display routing info.
        """
        start = self._trace_logger.time_block()
        message_id = uuid.uuid4().hex[:8]

        # Load session context for routing decisions
        session = self._session_service.load_session(session_id)
        session_ctx = session.context if session else None
        from src.services.session_service import SessionContext  # local import avoids circular
        agent_context = AgentContext(
            session_id=session_id,
            user_id=user_id,
            session_context=session_ctx or SessionContext(),
        )

        request = AgentRequest(
            message_id=message_id,
            message=message,
            context=agent_context,
        )

        # Route
        decision = self._router.route(request)
        logger.info(
            "routing | session_id=%s | intent=%s | agent=%s | confidence=%.2f | reason=%s",
            session_id,
            decision.intent,
            decision.selected_agent,
            decision.confidence,
            decision.reason,
        )

        # Dispatch to selected agent; fall back to support_agent if not in registry
        agent = self._registry.get(decision.selected_agent) or self._registry["support_agent"]
        response: AgentResponse = agent.handle(request)

        # Trace
        self._trace_logger.record(
            ExecutionTrace(
                session_id=session_id,
                message_id=message_id,
                selected_agent=response.agent_name,
                routing_reason=decision.reason,
                tool_name=response.tool_name,
                retrieval_sources=response.sources,
                duration_ms=self._trace_logger.duration_ms(start),
                success=response.handled,
                metadata={
                    "intent": decision.intent,
                    "routing_confidence": decision.confidence,
                },
            )
        )

        return response, decision

    def get_traces(self, session_id: str) -> list[ExecutionTrace]:
        return self._trace_logger.list_traces(session_id=session_id)

