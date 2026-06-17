from __future__ import annotations

from pydantic import BaseModel


class UIAction(BaseModel):
    name: str
    label: str
    value: str


def actions_for_agent(agent_name: str) -> list[UIAction]:
    if agent_name == "product_agent":
        return [
            UIAction(name="budget_under_500", label="Budget Under $500", value="budget_under_500"),
            UIAction(name="budget_500_1000", label="Budget $500-$1000", value="budget_500_1000"),
            UIAction(name="premium_products", label="Premium Products", value="premium_products"),
        ]
    if agent_name == "order_agent":
        return [
            UIAction(name="track_order", label="Track Order", value="track_order"),
            UIAction(name="start_return", label="Start Return", value="start_return"),
            UIAction(name="contact_support", label="Contact Support", value="contact_support"),
        ]
    if agent_name == "knowledge_agent":
        return [
            UIAction(name="show_source", label="Show Source", value="show_source"),
            UIAction(name="related_policies", label="Related Policies", value="related_policies"),
        ]
    return []


def prompt_for_action(action_name: str) -> str | None:
    mapping = {
        "budget_under_500": "Show me product options under $500.",
        "budget_500_1000": "Show me product options between $500 and $1000.",
        "premium_products": "Show me premium product options.",
        "track_order": "Track my current order status.",
        "start_return": "I want to start a return for my order.",
        "contact_support": "Connect me with support options.",
        "related_policies": "Show related policies for this topic.",
    }
    return mapping.get(action_name)

