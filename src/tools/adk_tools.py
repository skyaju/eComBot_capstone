from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from src.tools.data_loader import DataLoadError, get_mock_data_store
from src.tools.services import FAQService, OrderService, ProductService

logger = logging.getLogger(__name__)


class EComSupportTools:
    """Tool facade that isolates ADK callables from lower-level service classes."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._data_store = get_mock_data_store(data_dir)
        self._product_service = ProductService(self._data_store)
        self._order_service = OrderService(self._data_store)
        self._faq_service = FAQService(self._data_store)

    def search_products(
        self,
        product_name: str | None = None,
        category: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search catalog products by name/category/keywords and return details, price, and stock status."""

        try:
            return self._product_service.search(
                product_name=product_name,
                category=category,
                keywords=keywords,
            )
        except ValidationError as exc:
            logger.warning("Product search validation failed: %s", exc)
            return {
                "success": False,
                "error_code": "empty_search_criteria",
                "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Product search tool execution failed")
            return {
                "success": False,
                "error_code": "tool_execution_failed",
                "message": f"Product search failed: {exc}",
            }

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        """Lookup an order by order ID and return status, ETA, and tracking details."""

        try:
            return self._order_service.lookup(order_id=order_id)
        except ValidationError as exc:
            logger.warning("Order lookup validation failed: %s", exc)
            return {
                "success": False,
                "error_code": "invalid_order_id",
                "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Order lookup tool execution failed")
            return {
                "success": False,
                "error_code": "tool_execution_failed",
                "message": f"Order lookup failed: {exc}",
            }

    def retrieve_faq(self, customer_question: str) -> dict[str, Any]:
        """Retrieve the best matching FAQ answer for a customer question."""

        try:
            return self._faq_service.answer(question=customer_question)
        except ValidationError as exc:
            logger.warning("FAQ retrieval validation failed: %s", exc)
            return {
                "success": False,
                "error_code": "empty_question",
                "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("FAQ tool execution failed")
            return {
                "success": False,
                "error_code": "tool_execution_failed",
                "message": f"FAQ retrieval failed: {exc}",
            }


def build_tools(data_dir: str | None = None) -> tuple[list[Any], EComSupportTools]:
    """Create ADK-compatible tool callables and the owning facade instance."""

    try:
        support_tools = EComSupportTools(data_dir=data_dir)
    except DataLoadError as exc:
        logger.exception("Failed to initialize tool data store")
        raise RuntimeError(f"Tool initialization failed: {exc}") from exc

    tools: list[Any] = [
        support_tools.search_products,
        support_tools.lookup_order,
        support_tools.retrieve_faq,
    ]
    return tools, support_tools

