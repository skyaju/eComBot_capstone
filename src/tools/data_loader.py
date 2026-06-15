from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from src.tools.models import FAQRecord, OrderRecord, ProductRecord

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class DataLoadError(RuntimeError):
    """Raised when JSON mock data cannot be loaded or validated."""


class MockDataStore:
    """Validated, immutable-ish in-memory store for mock support data."""

    def __init__(
        self,
        products: list[ProductRecord],
        orders: list[OrderRecord],
        faqs: list[FAQRecord],
    ) -> None:
        self._products = products
        self._orders = orders
        self._faqs = faqs
        self._orders_by_id = {order.order_id: order for order in orders}

    @property
    def products(self) -> list[ProductRecord]:
        return self._products

    @property
    def orders(self) -> list[OrderRecord]:
        return self._orders

    @property
    def faqs(self) -> list[FAQRecord]:
        return self._faqs

    def find_order(self, order_id: str) -> OrderRecord | None:
        return self._orders_by_id.get(order_id)


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataLoadError(f"Mock data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in mock data file: {path} ({exc})") from exc


def _load_records(path: Path, model_type: type[T]) -> list[T]:
    payload = _read_json_file(path)
    adapter = TypeAdapter(list[model_type])
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        raise DataLoadError(f"Validation failed for {path.name}: {exc}") from exc


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir).resolve()


def load_mock_data_store(data_dir: str | Path | None = None) -> MockDataStore:
    """Load and validate all mock data JSON files from the data directory."""

    resolved_dir = _resolve_data_dir(data_dir)
    products_path = resolved_dir / "products.json"
    orders_path = resolved_dir / "orders.json"
    faq_path = resolved_dir / "faq.json"

    logger.info("Loading mock data from %s", resolved_dir)
    products = _load_records(products_path, ProductRecord)
    orders = _load_records(orders_path, OrderRecord)
    faqs = _load_records(faq_path, FAQRecord)

    logger.info(
        "Mock data loaded | products=%d | orders=%d | faqs=%d",
        len(products),
        len(orders),
        len(faqs),
    )
    return MockDataStore(products=products, orders=orders, faqs=faqs)


@lru_cache(maxsize=1)
def get_mock_data_store(data_dir: str | Path | None = None) -> MockDataStore:
    """Return a cached data store instance to avoid re-reading JSON each turn."""

    return load_mock_data_store(data_dir)

