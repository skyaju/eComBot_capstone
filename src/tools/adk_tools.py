from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import ValidationError

from src.services.session_service import SessionService, get_session_service
from src.tools.data_loader import DataLoadError, get_mock_data_store
from src.tools.services import FAQService, KnowledgeService, OrderService, ProductService

logger = logging.getLogger(__name__)


class EComSupportTools:
    """Tool facade that isolates ADK callables from lower-level service classes."""

    def __init__(
        self,
        data_dir: str | None = None,
        knowledge_dir: str | None = None,
        session_service: SessionService | None = None,
        session_id_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._session_service = session_service or get_session_service()
        self._session_id_getter = session_id_getter or self._session_service.get_current_session_id
        self._data_store = get_mock_data_store(data_dir=data_dir, knowledge_dir=knowledge_dir)
        self._product_service = ProductService(self._data_store, session_service=self._session_service)
        self._order_service = OrderService(self._data_store, session_service=self._session_service)
        self._faq_service = FAQService(self._data_store)
        self._knowledge_service = KnowledgeService(self._data_store, session_service=self._session_service)

    def _active_session_id(self) -> str | None:
        return self._session_id_getter()

    def search_products(
        self,
        product_name: str | None = None,
        category: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search catalog products by name/category/keywords and return details, price, and stock status."""

        try:
            result = self._product_service.search(
                product_name=product_name,
                category=category,
                keywords=keywords,
                session_id=self._active_session_id(),
            )
            active_session_id = self._active_session_id()
            if active_session_id is not None:
                self._session_service.remember_tool_result(active_session_id, "search_products", result)
            return result
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

    def lookup_order(self, order_id: str | None = None) -> dict[str, Any]:
        """Lookup an order by order ID and return status, ETA, and tracking details."""

        try:
            result = self._order_service.lookup(
                order_id=order_id,
                session_id=self._active_session_id(),
            )
            active_session_id = self._active_session_id()
            if active_session_id is not None:
                self._session_service.remember_tool_result(active_session_id, "lookup_order", result)
            return result
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
            result = self._faq_service.answer(question=customer_question)
            active_session_id = self._active_session_id()
            if active_session_id is not None:
                self._session_service.remember_tool_result(active_session_id, "retrieve_faq", result)
            return result
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

    def search_knowledge(self, customer_question: str, top_k: int = 3) -> dict[str, Any]:
        """KnowledgeSearchTool: retrieve policy/faq/spec snippets with confidence and source attribution."""

        try:
            result = self._knowledge_service.search(
                question=customer_question,
                top_k=top_k,
                session_id=self._active_session_id(),
            )
            active_session_id = self._active_session_id()
            if active_session_id is not None:
                self._session_service.remember_tool_result(active_session_id, "search_knowledge", result)
            return result
        except ValidationError as exc:
            logger.warning("Knowledge search validation failed: %s", exc)
            return {
                "success": False,
                "error_code": "empty_question",
                "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Knowledge search tool execution failed")
            return {
                "success": False,
                "error_code": "tool_execution_failed",
                "message": f"Knowledge search failed: {exc}",
            }


def build_tools(
    data_dir: str | None = None,
    knowledge_dir: str | None = None,
    session_service: SessionService | None = None,
    session_id_getter: Callable[[], str | None] | None = None,
) -> tuple[list[Any], EComSupportTools]:
    """Create ADK-compatible tool callables and the owning facade instance."""

    try:
        support_tools = EComSupportTools(
            data_dir=data_dir,
            knowledge_dir=knowledge_dir,
            session_service=session_service,
            session_id_getter=session_id_getter,
        )
    except DataLoadError as exc:
        logger.exception("Failed to initialize tool data store")
        raise RuntimeError(f"Tool initialization failed: {exc}") from exc

    tools: list[Any] = [
        support_tools.search_products,
        support_tools.lookup_order,
        support_tools.retrieve_faq,
        support_tools.search_knowledge,
    ]
    return tools, support_tools

