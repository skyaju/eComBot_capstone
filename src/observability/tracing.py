from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecutionTrace(BaseModel):
    session_id: str
    message_id: str
    selected_agent: str
    routing_reason: str
    tool_name: str | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    duration_ms: float
    success: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceLogger:
    """Simple in-memory trace collector that can be replaced by OTEL/Redis later."""

    def __init__(self) -> None:
        self._traces: list[ExecutionTrace] = []

    def time_block(self) -> float:
        return perf_counter()

    @staticmethod
    def duration_ms(start_time: float) -> float:
        return round((perf_counter() - start_time) * 1000, 3)

    def record(self, trace: ExecutionTrace) -> None:
        self._traces.append(trace)
        logger.info(
            "trace | session_id=%s | agent=%s | tool=%s | duration_ms=%.3f | success=%s",
            trace.session_id,
            trace.selected_agent,
            trace.tool_name,
            trace.duration_ms,
            trace.success,
        )

    def list_traces(self, session_id: str | None = None) -> list[ExecutionTrace]:
        if session_id is None:
            return list(self._traces)
        return [trace for trace in self._traces if trace.session_id == session_id]

