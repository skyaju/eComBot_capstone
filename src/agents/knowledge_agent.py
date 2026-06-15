from __future__ import annotations

from src.agents.contracts import AgentRequest, AgentResponse
from src.services.session_service import SessionService
from src.tools.data_loader import MockDataStore
from src.tools.services import KnowledgeService


class KnowledgeAgent:
    name = "knowledge_agent"

    def __init__(self, data_store: MockDataStore, session_service: SessionService) -> None:
        self._service = KnowledgeService(data_store=data_store, session_service=session_service)
        self._session_service = session_service

    def handle(self, request: AgentRequest) -> AgentResponse:
        result = self._service.search(
            question=request.message,
            top_k=3,
            session_id=request.context.session_id,
        )
        self._session_service.remember_tool_result(request.context.session_id, "search_knowledge", result)

        if not result.get("success"):
            return AgentResponse(
                agent_name=self.name,
                handled=False,
                tool_name="search_knowledge",
                message="I could not find a reliable policy or FAQ match. Please rephrase with a bit more detail.",
                confidence=result.get("confidence"),
                metadata={"error_code": result.get("error_code")},
            )

        snippets = result.get("results", [])
        top_snippet = snippets[0]["snippet"] if snippets else ""
        sources = result.get("sources", [])
        source_lines = "\n".join(f"- {source}" for source in sources)
        return AgentResponse(
            agent_name=self.name,
            handled=True,
            tool_name="search_knowledge",
            message=f"{top_snippet}\n\nSource:\n{source_lines}",
            confidence=result.get("confidence"),
            sources=sources,
            metadata={"result_count": len(snippets)},
        )

