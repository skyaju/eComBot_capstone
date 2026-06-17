from __future__ import annotations

from typing import Any


def render_order_card(metadata: dict[str, Any]) -> str:
    return (
        "### Order Card\n"
        f"- **Order ID:** {metadata.get('order_id', 'N/A')}\n"
        f"- **Status:** {metadata.get('status', 'N/A')}\n"
        f"- **ETA:** {metadata.get('estimated_delivery_date', 'N/A')}\n"
        f"- **Tracking Number:** {metadata.get('tracking_number', 'N/A')}"
    )


def render_product_card(metadata: dict[str, Any]) -> str:
    products = metadata.get("products") or []
    if not products:
        return ""

    first = products[0]
    features = first.get("description", "")
    return (
        "### Product Card\n"
        f"- **Product Name:** {first.get('name', 'N/A')}\n"
        f"- **Price:** {first.get('price', 'N/A')}\n"
        f"- **Availability:** {first.get('availability', 'N/A')}\n"
        f"- **Key Features:** {features[:180]}{'...' if len(features) > 180 else ''}"
    )


def render_knowledge_card(metadata: dict[str, Any]) -> str:
    rows = metadata.get("knowledge_results") or []
    if not rows:
        return ""

    top = rows[0]
    source = top.get("source", "N/A")
    section = top.get("category", "general")
    score = top.get("score", metadata.get("retrieval_confidence", 0.0))

    return (
        "### Knowledge Card\n"
        f"- **Source Document:** {source}\n"
        f"- **Section:** {section}\n"
        "- **Page:** N/A\n"
        f"- **Confidence Score:** {score:.2f}"
    )


def render_card_for_agent(agent_name: str, metadata: dict[str, Any]) -> str:
    if agent_name == "order_agent":
        return render_order_card(metadata)
    if agent_name == "product_agent":
        return render_product_card(metadata)
    if agent_name == "knowledge_agent":
        return render_knowledge_card(metadata)
    return ""

