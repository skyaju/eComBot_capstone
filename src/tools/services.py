from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.tools.data_loader import MockDataStore
from src.tools.models import FAQQueryInput, OrderLookupInput, ProductSearchInput

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class ProductService:
    """Business logic for product retrieval and comparison-friendly search."""

    def __init__(self, data_store: MockDataStore) -> None:
        self._data_store = data_store

    def search(
        self,
        product_name: str | None,
        category: str | None,
        keywords: list[str] | None,
    ) -> dict[str, Any]:
        query = ProductSearchInput(
            product_name=product_name,
            category=category,
            keywords=keywords or [],
        )

        name_needle = query.product_name.lower() if query.product_name else None
        category_needle = query.category.lower() if query.category else None
        keyword_needles = [kw.lower() for kw in query.keywords]

        scored_results: list[tuple[int, Any]] = []
        for product in self._data_store.products:
            score = 0

            if name_needle and name_needle in product.name.lower():
                score += 3

            if category_needle and category_needle in product.category.lower():
                score += 2

            if keyword_needles:
                searchable = " ".join(
                    [product.name, product.description, product.category, " ".join(product.keywords)]
                ).lower()
                score += sum(1 for keyword in keyword_needles if keyword in searchable)

            if score > 0:
                scored_results.append((score, product))

        scored_results.sort(key=lambda pair: (pair[0], pair[1].name), reverse=True)

        if not scored_results:
            return {
                "success": False,
                "error_code": "product_not_found",
                "message": "I could not find products matching that criteria.",
                "query": query.model_dump(),
                "products": [],
            }

        products = [
            {
                "product_id": product.product_id,
                "name": product.name,
                "category": product.category,
                "price": f"{product.price:.2f} {product.currency}",
                "availability": product.availability,
                "description": product.description,
            }
            for _, product in scored_results[:5]
        ]

        return {
            "success": True,
            "query": query.model_dump(),
            "count": len(products),
            "products": products,
        }


class OrderService:
    """Business logic for strict order ID validation and order lookup."""

    def __init__(self, data_store: MockDataStore) -> None:
        self._data_store = data_store

    def lookup(self, order_id: str) -> dict[str, Any]:
        query = OrderLookupInput(order_id=order_id)
        order = self._data_store.find_order(query.order_id)

        if order is None:
            return {
                "success": False,
                "error_code": "order_not_found",
                "message": "No order was found for that ID.",
                "order_id": query.order_id,
            }

        return {
            "success": True,
            "order_id": order.order_id,
            "order_status": order.status,
            "estimated_delivery_date": (
                order.estimated_delivery_date.isoformat() if order.estimated_delivery_date else None
            ),
            "tracking": {
                "tracking_number": order.tracking_number,
                "carrier": order.carrier,
            },
            "items": order.items,
        }


class FAQService:
    """Business logic for FAQ retrieval using simple lexical ranking."""

    def __init__(self, data_store: MockDataStore) -> None:
        self._data_store = data_store

    def answer(self, question: str) -> dict[str, Any]:
        query = FAQQueryInput(question=question)
        query_tokens = self._tokenize(query.question)

        if not query_tokens:
            return {
                "success": False,
                "error_code": "empty_question",
                "message": "Please provide a question so I can search the FAQ database.",
            }

        best = None
        best_score = 0

        for faq in self._data_store.faqs:
            haystack_tokens = self._tokenize(" ".join([faq.question, faq.answer, " ".join(faq.tags)]))
            overlap = sum((Counter(query_tokens) & Counter(haystack_tokens)).values())

            if overlap > best_score:
                best = faq
                best_score = overlap

        if best is None or best_score == 0:
            return {
                "success": False,
                "error_code": "faq_not_found",
                "message": "I could not find a FAQ answer that matches that question.",
                "question": query.question,
            }

        return {
            "success": True,
            "faq_id": best.faq_id,
            "question": best.question,
            "answer": best.answer,
            "match_score": best_score,
        }

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return [token.lower() for token in _WORD_PATTERN.findall(value)]

