from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any

from src.services.session_service import SessionService
from src.tools.data_loader import MockDataStore
from src.tools.models import FAQQueryInput, KnowledgeSearchInput, OrderLookupInput, ProductSearchInput

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class ProductService:
    """Business logic for product retrieval and comparison-friendly search."""

    def __init__(self, data_store: MockDataStore, session_service: SessionService | None = None) -> None:
        self._data_store = data_store
        self._session_service = session_service

    def search(
        self,
        product_name: str | None,
        category: str | None,
        keywords: list[str] | None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        product_name, category, keywords = self._hydrate_missing_query_from_context(
            product_name=product_name,
            category=category,
            keywords=keywords,
            session_id=session_id,
        )

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

        scored_results = self._apply_alternative_pricing_filter(
            scored_results=scored_results,
            keywords=keyword_needles,
            session_id=session_id,
        )

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

    def _hydrate_missing_query_from_context(
        self,
        product_name: str | None,
        category: str | None,
        keywords: list[str] | None,
        session_id: str | None,
    ) -> tuple[str | None, str | None, list[str] | None]:
        if product_name or category:
            return product_name, category, keywords

        if self._session_service is None or session_id is None:
            return product_name, category, keywords

        session = self._session_service.load_session(session_id)
        if session is None or not session.context.recent_products:
            return product_name, category, keywords

        recent_product_name = session.context.recent_products[-1]
        reference_product = next(
            (product for product in self._data_store.products if product.name == recent_product_name),
            None,
        )
        if reference_product is not None:
            return recent_product_name, reference_product.category, keywords

        return recent_product_name, category, keywords

    def _apply_alternative_pricing_filter(
        self,
        scored_results: list[tuple[int, Any]],
        keywords: list[str],
        session_id: str | None,
    ) -> list[tuple[int, Any]]:
        if self._session_service is None or session_id is None:
            return scored_results

        intent_tokens = {"cheaper", "budget", "lower", "affordable", "alternative", "alternatives"}
        if not any(token in keywords for token in intent_tokens):
            return scored_results

        session = self._session_service.load_session(session_id)
        if session is None or not session.context.recent_products:
            return scored_results

        reference_name = session.context.recent_products[-1].lower()
        reference_product = next(
            (p for p in self._data_store.products if p.name.lower() == reference_name),
            None,
        )
        if reference_product is None:
            return scored_results

        filtered = [
            pair
            for pair in scored_results
            if pair[1].price < reference_product.price and pair[1].name.lower() != reference_product.name.lower()
        ]
        if filtered:
            filtered.sort(key=lambda pair: pair[1].price)
            return filtered

        broader_fallback = [
            (1, product)
            for product in self._data_store.products
            if product.price < reference_product.price and product.name.lower() != reference_product.name.lower()
        ]
        if broader_fallback:
            broader_fallback.sort(key=lambda pair: pair[1].price)
            return broader_fallback

        return scored_results


class OrderService:
    """Business logic for strict order ID validation and order lookup."""

    def __init__(self, data_store: MockDataStore, session_service: SessionService | None = None) -> None:
        self._data_store = data_store
        self._session_service = session_service

    def lookup(self, order_id: str | None, session_id: str | None = None) -> dict[str, Any]:
        resolved_order_id = order_id
        if self._session_service is not None and session_id is not None:
            resolved_order_id = self._session_service.resolve_order_id(session_id, order_id)

        if resolved_order_id is None:
            return {
                "success": False,
                "error_code": "order_id_required",
                "message": "Please share your order ID (example: ORD-12345).",
            }

        query = OrderLookupInput(order_id=resolved_order_id)
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


class KnowledgeService:
    """Retrieval-oriented knowledge search over markdown/text documentation."""

    _SYNONYMS: dict[str, set[str]] = {
        "warranty": {"guarantee", "coverage", "repair"},
        "shipping": {"delivery", "carrier", "ship", "shipped", "transit"},
        "returns": {"return", "refund", "exchange", "rma"},
        "policy": {"policies", "rules", "terms"},
        "loyalty": {"rewards", "points", "membership"},
        "spec": {"specs", "specification", "features"},
    }

    def __init__(self, data_store: MockDataStore, session_service: SessionService | None = None) -> None:
        self._data_store = data_store
        self._session_service = session_service
        self._doc_vectors = [Counter(self._expand_tokens(doc.content)) for doc in self._data_store.knowledge_documents]
        self._idf = self._build_idf(self._doc_vectors)

    def search(
        self,
        question: str,
        top_k: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        query = KnowledgeSearchInput(question=question, top_k=top_k)

        query_tokens = self._expand_tokens(query.question)
        if self._session_service is not None and session_id is not None:
            context_hint = self._session_service.build_context_hint(session_id)
            if context_hint:
                query_tokens.extend(self._expand_tokens(context_hint))

        if not query_tokens:
            return {
                "success": False,
                "error_code": "empty_question",
                "message": "Please provide a question so I can search the knowledge base.",
            }

        ranked: list[tuple[float, int]] = []
        query_counter = Counter(query_tokens)

        for idx, doc_vector in enumerate(self._doc_vectors):
            score = self._cosine_similarity(query_counter, doc_vector)
            if score <= 0:
                continue

            doc = self._data_store.knowledge_documents[idx]
            if doc.category in {"shipping", "returns", "policies", "faq"}:
                score *= 1.05
            ranked.append((score, idx))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_results = ranked[: query.top_k]

        if not best_results:
            return {
                "success": False,
                "error_code": "knowledge_not_found",
                "message": "I could not find relevant policy or product documentation for that question.",
                "question": query.question,
            }

        top_score = best_results[0][0]
        confidence = min(round(top_score, 4), 1.0)
        if confidence < 0.12:
            return {
                "success": False,
                "error_code": "low_confidence_retrieval",
                "message": "I found only weak matches. Please rephrase your question with more specifics.",
                "question": query.question,
                "confidence": confidence,
            }

        snippets: list[dict[str, Any]] = []
        for score, idx in best_results:
            doc = self._data_store.knowledge_documents[idx]
            snippets.append(
                {
                    "source": doc.source,
                    "category": doc.category,
                    "score": round(score, 4),
                    "snippet": self._build_snippet(doc.content, query_tokens),
                }
            )

        return {
            "success": True,
            "question": query.question,
            "confidence": confidence,
            "results": snippets,
            "sources": [item["source"] for item in snippets],
        }

    def _build_idf(self, vectors: list[Counter[str]]) -> dict[str, float]:
        if not vectors:
            return {}
        doc_count = len(vectors)
        token_doc_freq: Counter[str] = Counter()
        for vector in vectors:
            token_doc_freq.update(vector.keys())
        return {
            token: math.log((1 + doc_count) / (1 + frequency)) + 1.0
            for token, frequency in token_doc_freq.items()
        }

    def _cosine_similarity(self, lhs: Counter[str], rhs: Counter[str]) -> float:
        if not lhs or not rhs:
            return 0.0

        dot = 0.0
        for token, value in lhs.items():
            if token not in rhs:
                continue
            weight = self._idf.get(token, 1.0)
            dot += value * rhs[token] * weight * weight

        if dot == 0.0:
            return 0.0

        lhs_norm = math.sqrt(sum((count * self._idf.get(token, 1.0)) ** 2 for token, count in lhs.items()))
        rhs_norm = math.sqrt(sum((count * self._idf.get(token, 1.0)) ** 2 for token, count in rhs.items()))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return 0.0
        return dot / (lhs_norm * rhs_norm)

    def _expand_tokens(self, value: str) -> list[str]:
        tokens = [token.lower() for token in _WORD_PATTERN.findall(value)]
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            for base, synonyms in self._SYNONYMS.items():
                if token == base or token in synonyms:
                    expanded.append(base)
                    expanded.extend(sorted(synonyms))
        return expanded

    @staticmethod
    def _build_snippet(content: str, query_tokens: list[str], max_length: int = 220) -> str:
        text = " ".join(content.split())
        lowered = text.lower()
        anchor = 0
        for token in query_tokens:
            pos = lowered.find(token)
            if pos != -1:
                anchor = max(pos - 60, 0)
                break
        snippet = text[anchor : anchor + max_length]
        if anchor > 0:
            snippet = "..." + snippet
        if anchor + max_length < len(text):
            snippet += "..."
        return snippet


def build_knowledge_service(
    data_store: MockDataStore,
    session_service: SessionService | None = None,
) -> Any:
    """
    Create the active knowledge retrieval service.

    Preference order:
      1) ChromaKnowledgeService when CHROMA_HOST is configured and reachable.
      2) KnowledgeService lexical fallback (always available).
    """
    chroma_host = os.getenv("CHROMA_HOST")
    chroma_port_raw = os.getenv("CHROMA_PORT", "8000")
    chroma_collection = os.getenv("CHROMA_COLLECTION", "ecombot_knowledge")

    if chroma_host:
        try:
            from src.rag.retrieval_service import ChromaKnowledgeService
            from src.rag.vector_store import ChromaVectorStore

            chroma_port = int(chroma_port_raw)
            store = ChromaVectorStore.connect(
                host=chroma_host,
                port=chroma_port,
                collection_name=chroma_collection,
            )
            logger.info(
                "Knowledge service using ChromaDB | host=%s | port=%d | collection=%s",
                chroma_host,
                chroma_port,
                chroma_collection,
            )
            return ChromaKnowledgeService(store=store, session_service=session_service)
        except Exception as exc:
            logger.warning("ChromaDB unavailable; falling back to lexical KnowledgeService: %s", exc)

    return KnowledgeService(data_store=data_store, session_service=session_service)


