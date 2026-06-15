from __future__ import annotations

import re

from src.agents.contracts import AgentRequest, AgentResponse
from src.services.session_service import SessionService
from src.tools.data_loader import MockDataStore
from src.tools.services import OrderService

_ORDER_PATTERN = re.compile(r"ORD-\d{5}", re.IGNORECASE)


class OrderAgent:
    name = "order_agent"

    def __init__(self, data_store: MockDataStore, session_service: SessionService) -> None:
        self._service = OrderService(data_store=data_store, session_service=session_service)
        self._session_service = session_service

    def handle(self, request: AgentRequest) -> AgentResponse:
        explicit_order_id = self._extract_order_id(request.message)
        result = self._service.lookup(order_id=explicit_order_id, session_id=request.context.session_id)
        self._session_service.remember_tool_result(request.context.session_id, "lookup_order", result)

        if not result.get("success"):
            error_code = result.get("error_code")
            if error_code == "order_id_required":
                message = "Please share your order number in the format ORD-12345 so I can check the status."
            elif error_code == "order_not_found":
                message = "I could not find that order ID. Please verify the number and I will check again."
            else:
                message = "I could not complete the order lookup right now. Please try again in a moment."
            return AgentResponse(
                agent_name=self.name,
                handled=False,
                tool_name="lookup_order",
                message=message,
                metadata={"error_code": error_code},
            )

        tracking = result.get("tracking") or {}
        return AgentResponse(
            agent_name=self.name,
            handled=True,
            tool_name="lookup_order",
            message=(
                f"Order {result['order_id']} is currently {result['order_status']}. "
                f"Estimated delivery: {result.get('estimated_delivery_date') or 'not available'}. "
                f"Tracking: {tracking.get('tracking_number') or 'pending'} via {tracking.get('carrier') or 'carrier not assigned'}."
            ),
            metadata={"order_id": result["order_id"], "status": result["order_status"]},
        )

    @staticmethod
    def _extract_order_id(message: str) -> str | None:
        match = _ORDER_PATTERN.search(message)
        if not match:
            return None
        return match.group(0).upper()

