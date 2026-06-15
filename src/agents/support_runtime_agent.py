from __future__ import annotations

from src.agents.contracts import AgentRequest, AgentResponse


class SupportAgent:
    """Fallback conversational agent for greetings and uncategorized requests."""

    name = "support_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        message = request.message.strip().lower()
        customer_name = request.context.session_context.customer_name

        if message in {"hi", "hello", "hey", "good morning", "good evening"}:
            if customer_name:
                text = f"Hello {customer_name}! How can I help you with products, orders, or policies today?"
            else:
                text = "Hello! How can I help you with products, orders, or policies today?"
            return AgentResponse(agent_name=self.name, handled=True, message=text)

        if "help" in message:
            return AgentResponse(
                agent_name=self.name,
                handled=True,
                message=(
                    "I can help with product discovery, order tracking, returns/refunds, shipping questions, "
                    "and policy/FAQ answers. Tell me what you need and I will route it correctly."
                ),
            )

        return AgentResponse(
            agent_name=self.name,
            handled=True,
            message=(
                "I am here to help with e-commerce support topics such as products, orders, shipping, "
                "returns, and policies. Could you share what you would like to do?"
            ),
        )

