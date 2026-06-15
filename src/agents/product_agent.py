from __future__ import annotations

import re
from typing import Any

from src.agents.contracts import AgentRequest, AgentResponse
from src.services.session_service import SessionService
from src.tools.data_loader import MockDataStore
from src.tools.services import ProductService

_BUDGET_PATTERN = re.compile(r"under\s*\$?(\d+(?:\.\d{1,2})?)", re.IGNORECASE)


class ProductAgent:
    name = "product_agent"

    def __init__(self, data_store: MockDataStore, session_service: SessionService) -> None:
        self._service = ProductService(data_store=data_store, session_service=session_service)
        self._session_service = session_service

    def handle(self, request: AgentRequest) -> AgentResponse:
        filters = self._build_filters(request.message)
        result = self._service.search(
            product_name=filters["product_name"],
            category=filters["category"],
            keywords=filters["keywords"],
            session_id=request.context.session_id,
        )
        self._session_service.remember_tool_result(request.context.session_id, "search_products", result)

        if not result.get("success"):
            return AgentResponse(
                agent_name=self.name,
                handled=False,
                tool_name="search_products",
                message="I could not find products for that request. Share a product name, category, or key feature and I will refine the options.",
                metadata={"error_code": result.get("error_code")},
            )

        products = self._apply_budget(result["products"], request.message)
        if not products:
            return AgentResponse(
                agent_name=self.name,
                handled=True,
                tool_name="search_products",
                message="I found matching products, but none are within that budget. If you share a higher range, I can suggest the best value options.",
                metadata={"result_count": 0},
            )

        lines = ["Here are the best product options for your request:"]
        for item in products[:3]:
            lines.append(
                f"- {item['name']} ({item['category']}): {item['price']} | stock={item['availability']}"
            )
        lines.append("Would you like a side-by-side comparison of any two options?")

        return AgentResponse(
            agent_name=self.name,
            handled=True,
            tool_name="search_products",
            message="\n".join(lines),
            metadata={"result_count": len(products)},
        )

    @staticmethod
    def _build_filters(message: str) -> dict[str, Any]:
        lowered = message.lower()
        category = None
        if "laptop" in lowered or "gaming" in lowered:
            category = "Electronics"
        elif "shoe" in lowered:
            category = "Footwear"
        elif "watch" in lowered:
            category = "Wearables"

        keywords = [token for token in re.findall(r"[a-zA-Z0-9]+", lowered) if len(token) > 2]
        return {
            "product_name": None,
            "category": category,
            "keywords": keywords,
        }

    @staticmethod
    def _apply_budget(products: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
        match = _BUDGET_PATTERN.search(message)
        if not match:
            return products

        budget = float(match.group(1))

        def parse_price(raw: str) -> float:
            parts = raw.split(" ", 1)
            return float(parts[0])

        return [item for item in products if parse_price(item["price"]) <= budget]

