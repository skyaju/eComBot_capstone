from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from src.tools.models import FAQRecord, KnowledgeDocument, OrderRecord, ProductRecord

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"
_SUPPORTED_KNOWLEDGE_SUFFIXES = {".md", ".txt"}


class DataLoadError(RuntimeError):
    """Raised when JSON mock data cannot be loaded or validated."""


class MockDataStore:
    """Validated, immutable-ish in-memory store for mock support data."""

    def __init__(
        self,
        products: list[ProductRecord],
        orders: list[OrderRecord],
        faqs: list[FAQRecord],
        knowledge_documents: list[KnowledgeDocument],
    ) -> None:
        self._products = products
        self._orders = orders
        self._faqs = faqs
        self._knowledge_documents = knowledge_documents
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

    @property
    def knowledge_documents(self) -> list[KnowledgeDocument]:
        return self._knowledge_documents

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
    if not isinstance(payload, list):
        raise DataLoadError(f"Validation failed for {path.name}: expected a JSON array")
    adapter = TypeAdapter(model_type)
    try:
        return [adapter.validate_python(item) for item in payload]
    except ValidationError as exc:
        raise DataLoadError(f"Validation failed for {path.name}: {exc}") from exc


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir).resolve()


def _resolve_knowledge_dir(knowledge_dir: str | Path | None) -> Path:
    if knowledge_dir is None:
        return DEFAULT_KNOWLEDGE_DIR
    return Path(knowledge_dir).resolve()


def _load_knowledge_documents(knowledge_dir: Path) -> list[KnowledgeDocument]:
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        raise DataLoadError(f"Knowledge directory not found: {knowledge_dir}")

    documents: list[KnowledgeDocument] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_KNOWLEDGE_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DataLoadError(f"Unable to read knowledge file: {path}") from exc
        if not content:
            continue

        relative = path.relative_to(knowledge_dir)
        parts = relative.parts
        category = parts[0] if len(parts) > 1 else "general"
        doc_id = relative.as_posix().replace("/", "-").replace(".", "-")
        documents.append(
            KnowledgeDocument(
                doc_id=doc_id,
                source=relative.as_posix(),
                category=category,
                content=content,
            )
        )

    if not documents:
        raise DataLoadError(f"No markdown/text knowledge documents found in {knowledge_dir}")
    return documents


def load_mock_data_store(
    data_dir: str | Path | None = None,
    knowledge_dir: str | Path | None = None,
) -> MockDataStore:
    """Load and validate all mock data JSON files from the data directory."""

    resolved_dir = _resolve_data_dir(data_dir)
    resolved_knowledge_dir = _resolve_knowledge_dir(knowledge_dir)
    products_path = resolved_dir / "products.json"
    orders_path = resolved_dir / "orders.json"
    faq_path = resolved_dir / "faq.json"

    logger.info("Loading mock data from %s", resolved_dir)
    products = _load_records(products_path, ProductRecord)
    orders = _load_records(orders_path, OrderRecord)
    faqs = _load_records(faq_path, FAQRecord)
    knowledge_documents = _load_knowledge_documents(resolved_knowledge_dir)

    logger.info(
        "Mock data loaded | products=%d | orders=%d | faqs=%d | knowledge_docs=%d",
        len(products),
        len(orders),
        len(faqs),
        len(knowledge_documents),
    )
    return MockDataStore(
        products=products,
        orders=orders,
        faqs=faqs,
        knowledge_documents=knowledge_documents,
    )


@lru_cache(maxsize=4)
def get_mock_data_store(
    data_dir: str | Path | None = None,
    knowledge_dir: str | Path | None = None,
) -> MockDataStore:
    """Return a cached data store instance to avoid re-reading JSON each turn."""

    return load_mock_data_store(data_dir=data_dir, knowledge_dir=knowledge_dir)

