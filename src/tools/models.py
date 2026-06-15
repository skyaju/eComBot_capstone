from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProductRecord(BaseModel):
    """Canonical product shape loaded from the mock product catalog."""

    product_id: str = Field(min_length=3)
    name: str = Field(min_length=2)
    category: str = Field(min_length=2)
    keywords: list[str] = Field(default_factory=list)
    price: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    availability: Literal["in_stock", "low_stock", "out_of_stock"]
    description: str = Field(min_length=10)


class OrderRecord(BaseModel):
    """Canonical order shape loaded from mock order data."""

    order_id: str = Field(min_length=6)
    status: Literal[
        "processing",
        "packed",
        "shipped",
        "out_for_delivery",
        "delivered",
        "cancelled",
    ]
    estimated_delivery_date: date | None = None
    tracking_number: str | None = None
    carrier: str | None = None
    shipping_address: str = Field(min_length=10)
    items: list[str] = Field(default_factory=list)


class FAQRecord(BaseModel):
    """Structured FAQ record with searchable tags and answer body."""

    faq_id: str = Field(min_length=3)
    question: str = Field(min_length=5)
    answer: str = Field(min_length=10)
    tags: list[str] = Field(default_factory=list)


class KnowledgeDocument(BaseModel):
    """Normalized documentation record loaded from markdown/text knowledge files."""

    doc_id: str = Field(min_length=3)
    source: str = Field(min_length=3)
    category: str = Field(min_length=2)
    content: str = Field(min_length=20)


class ProductSearchInput(BaseModel):
    """Input contract for product search tool."""

    product_name: str | None = None
    category: str | None = None
    keywords: list[str] = Field(default_factory=list)

    @field_validator("product_name", "category", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return str(value)
        stripped = value.strip()
        return stripped if stripped else None

    @field_validator("keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return [part for part in parts if part]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    @model_validator(mode="after")
    def require_at_least_one_filter(self) -> "ProductSearchInput":
        if not self.product_name and not self.category and not self.keywords:
            raise ValueError("Provide at least one filter: product_name, category, or keywords.")
        return self


class OrderLookupInput(BaseModel):
    """Input contract for order lookup tool."""

    order_id: str = Field(min_length=6)

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized.startswith("ORD-"):
            raise ValueError("Order ID must start with 'ORD-' (example: ORD-10001).")
        suffix = normalized.replace("ORD-", "", 1)
        if not suffix.isdigit() or len(suffix) != 5:
            raise ValueError("Order ID format is invalid. Expected pattern: ORD-12345.")
        return normalized


class FAQQueryInput(BaseModel):
    """Input contract for FAQ retrieval tool."""

    question: str = Field(min_length=3)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty.")
        return stripped


class KnowledgeSearchInput(BaseModel):
    """Input contract for retrieval-augmented documentation search."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty.")
        return stripped


