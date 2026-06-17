from __future__ import annotations

from typing import Any

from src.ui.session_ui import display_agent_name


_TOOL_STEP_LABELS: dict[str, str] = {
    "search_products": "Searching Product Catalog",
    "lookup_order": "Checking Order Status",
    "search_knowledge": "Searching Knowledge Base",
    "retrieve_faq": "Retrieving FAQ Answer",
}


def tool_step_label(tool_name: str | None) -> str:
    if not tool_name:
        return "Generating Support Response"
    return _TOOL_STEP_LABELS.get(tool_name, f"Running {tool_name}")


def result_summary(tool_name: str | None, metadata: dict[str, Any]) -> str:
    if tool_name == "search_products":
        return f"Found {metadata.get('result_count', 0)} product option(s)."
    if tool_name == "lookup_order":
        order_id = metadata.get("order_id", "unknown")
        status = metadata.get("status", "unknown")
        return f"Order {order_id}: {status}."
    if tool_name == "search_knowledge":
        count = metadata.get("result_count", 0)
        conf = metadata.get("retrieval_confidence")
        if conf is None:
            return f"Retrieved {count} source snippet(s)."
        return f"Retrieved {count} source snippet(s), confidence {conf:.2f}."
    return "Response generated successfully."


def explainability_text(
    selected_agent: str,
    tool_name: str | None,
    sources: list[str],
    response_text: str,
) -> str:
    source_lines = "\n".join(f"- {src}" for src in sources) if sources else "- No external source used"
    tool_label = tool_name or "No tool"
    return (
        "### How this answer was generated\n"
        f"- **Selected agent:** {display_agent_name(selected_agent)}\n"
        f"- **Tool used:** {tool_label}\n"
        f"- **Documents retrieved:**\n{source_lines}\n"
        f"- **Final answer basis:** {response_text[:220]}{'...' if len(response_text) > 220 else ''}"
    )

