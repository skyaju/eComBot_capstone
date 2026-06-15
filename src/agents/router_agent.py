from __future__ import annotations

import re
from dataclasses import dataclass

from src.agents.contracts import AgentRequest, RoutingDecision

_ORDER_PATTERN = re.compile(r"ORD-\d{5}", re.IGNORECASE)


@dataclass(frozen=True)
class RoutingRule:
    intent: str
    selected_agent: str
    keywords: tuple[str, ...]
    confidence: float


class RouterAgent:
    """Intent router with configurable rules to keep delegation extensible."""

    name = "router_agent"

    def __init__(self, rules: list[RoutingRule], fallback_agent: str = "support_agent") -> None:
        self._rules = rules
        self._fallback_agent = fallback_agent

    @classmethod
    def default(cls) -> "RouterAgent":
        return cls(
            rules=[
                RoutingRule(
                    intent="order",
                    selected_agent="order_agent",
                    keywords=("order", "tracking", "shipment", "shipped", "delivery", "eta", "where is"),
                    confidence=0.92,
                ),
                RoutingRule(
                    intent="product",
                    selected_agent="product_agent",
                    keywords=(
                        "product",
                        "laptop",
                        "phone",
                        "watch",
                        "gaming",
                        "compare",
                        "alternative",
                        "cheaper",
                        "price",
                    ),
                    confidence=0.9,
                ),
                RoutingRule(
                    intent="knowledge",
                    selected_agent="knowledge_agent",
                    keywords=(
                        "policy",
                        "return",
                        "refund",
                        "warranty",
                        "shipping policy",
                        "loyalty",
                        "faq",
                        "payment",
                    ),
                    confidence=0.88,
                ),
            ],
            fallback_agent="support_agent",
        )

    def route(self, request: AgentRequest) -> RoutingDecision:
        message = request.message.lower().strip()
        session_context = request.context.session_context

        if _ORDER_PATTERN.search(message):
            return RoutingDecision(
                intent="order",
                selected_agent="order_agent",
                reason="Detected order ID pattern in customer message.",
                confidence=0.97,
            )

        if message in {"has it shipped yet?", "has it shipped", "where is it now?", "where is it"}:
            if session_context.recent_order_ids:
                return RoutingDecision(
                    intent="order",
                    selected_agent="order_agent",
                    reason="Follow-up order status question resolved from session memory.",
                    confidence=0.94,
                )

        if "what warranty" in message and session_context.recent_products:
            return RoutingDecision(
                intent="knowledge",
                selected_agent="knowledge_agent",
                reason="Warranty follow-up tied to previous product context.",
                confidence=0.9,
            )

        for rule in self._rules:
            if any(keyword in message for keyword in rule.keywords):
                return RoutingDecision(
                    intent=rule.intent,
                    selected_agent=rule.selected_agent,
                    reason=f"Matched rule keywords for {rule.intent} intent.",
                    confidence=rule.confidence,
                )

        if session_context.active_support_topic == "order_lookup":
            return RoutingDecision(
                intent="order",
                selected_agent="order_agent",
                reason="No explicit keyword match; reused active order topic from memory.",
                confidence=0.7,
            )

        return RoutingDecision(
            intent="support",
            selected_agent=self._fallback_agent,
            reason="No intent rule matched; using fallback support agent.",
            confidence=0.55,
        )

