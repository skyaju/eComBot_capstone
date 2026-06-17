# eComBot (v3 — Multi-Agent + Chainlit UI)

eComBot is a Google ADK-powered e-commerce support assistant with a **production-style multi-agent backend** and a **Chainlit generative UI frontend**. The UI layer reuses the existing Router, Product, Order, Support, and Knowledge agents; Redis memory; and ChromaDB RAG retrieval.

---

## Multi-Agent Architecture Diagram

```text
User (CLI / ADK Web)
  │
  ▼
run_agent.py / src/agent.py
  │
  ├─ [single-agent mode] ─────────────────────────────────────────────────────┐
  │   create_support_agent()                                                   │
  │     └─ LlmAgent (eComBot)                                                  │
  │          ├─ search_products                                                 │
  │          ├─ lookup_order                                                    │
  │          ├─ retrieve_faq                                                    │
  │          └─ search_knowledge                                                │
  │                                                                             │
  └─ [multi-agent mode] ──────────────────────────────────────────────────────┘
      create_multi_agent_system()
        └─ LlmAgent "eComBot" (root orchestrator + sub_agents)
             ├─ product_agent  (LlmAgent + search_products)
             ├─ order_agent    (LlmAgent + lookup_order)
             └─ knowledge_agent (LlmAgent + retrieve_faq + search_knowledge)

MultiAgentOrchestrator  [CLI + tests layer]
  ├─ RouterAgent (intent classification)
  │    ├─ product intent  → ProductAgent
  │    ├─ order intent    → OrderAgent
  │    ├─ knowledge intent→ KnowledgeAgent
  │    └─ fallback        → SupportAgent
  └─ TraceLogger (observability)

Shared across all agents:
  SessionService ──► SessionContext (name, orders, products, topic, summary)
  MockDataStore  ──► products.json + orders.json + faq.json + knowledge/**
```

## Chainlit UI Architecture

```text
User
  │
  ▼
Chainlit (`src/ui/chainlit_app.py`)
  │
  ▼
MultiAgentOrchestrator.dispatch()
  │
  ▼
RouterAgent
  ├─ ProductAgent   → ProductService
  ├─ OrderAgent     → OrderService
  ├─ KnowledgeAgent → ChromaKnowledgeService (fallback: KnowledgeService)
  └─ SupportAgent
  │
  ▼
Chainlit response + structured card + action buttons + explainability

Shared services:
  Redis SessionService, ChromaDB, Postgres (analytics/audit)
```

---

## Chainlit Features

- Agent visibility: every message includes `[Support Agent]`, `[Product Agent]`, `[Order Agent]`, or `[Knowledge Agent]`.
- Tool visibility via Chainlit Steps: routing, tool execution summary, and knowledge retrieval context.
- Structured cards:
  - Order Card: order ID, status, ETA, tracking number
  - Product Card: name, price, availability, key features
  - Knowledge Card: source document, section, confidence score
- Quick actions for common follow-ups (`budget`, `track order`, `show source`, `related policies`).
- Explainability action: "How was this answer generated?" shows selected agent, tool used, and source set.

---

## Run Chainlit UI

```bash
source .venv/bin/activate
PYTHONPATH=. chainlit run src/ui/chainlit_app.py --host 0.0.0.0 --port 8001
```

With Docker Compose:

```bash
docker compose up -d --build
docker compose ps
```

Then open:

- ADK web: `http://localhost:8080`
- Chainlit UI: `http://localhost:8001`

---

---

## Routing Flow

```text
Customer Message
  │
  ▼
RouterAgent.route()
  │  keyword matching → ORD-\d{5} pattern → session context
  ▼
RoutingDecision { intent, selected_agent, confidence, reason }
  │
  ▼
Domain Agent.handle(AgentRequest)
  │  calls domain service (ProductService / OrderService / KnowledgeService)
  │  updates SessionService memory
  ▼
AgentResponse { message, tool_name, sources, handled }
  │
  ▼
TraceLogger.record(ExecutionTrace)
  │  session_id, agent, tool, duration_ms, success
  ▼
Final response to customer
```

---

## Agent Responsibilities

| Agent | Intent | Tools |
|---|---|---|
| `product_agent` | Product discovery, comparison, alternatives | `search_products` |
| `order_agent` | Order status, tracking, ETA | `lookup_order` |
| `knowledge_agent` | Policy, FAQ, warranty, returns, loyalty | `retrieve_faq` + `search_knowledge` |
| `support_agent` | Greetings, help, fallback | Direct LLM response |
| `router_agent` | Intent classification, agent delegation | None (pure Python) |

---

## RAG Flow

```text
User Question
  │
  ▼
search_knowledge tool
  │
  ▼
KnowledgeService ranking (TF-IDF cosine + synonym expansion)
  │
  ▼
Top-K snippets + confidence score + source file names
  │
  ▼
LLM response with source citations:
  Source: - returns_policy.md
```

---

## Session Lifecycle

1. Create/attach session in ADK + local `SessionService`.
2. Store each user message; extract customer name, order IDs, support topic.
3. Activate session context for each turn (ContextVar propagation).
4. Tool calls (and domain agents) read/write session context.
5. `ConversationSummary` rebuilt after each assistant turn.
6. Context used by `RouterAgent` for follow-up routing decisions.
7. Clear session when requested.

---

## Memory Model (`SessionContext`)

| Field | Type | Purpose |
|---|---|---|
| `customer_name` | `str \| None` | Extracted from "my name is …" |
| `recent_products` | `list[str]` | Last 5 product names viewed |
| `recent_order_ids` | `list[str]` | Last 5 order IDs mentioned |
| `active_support_topic` | `SupportTopic` | Current topic for follow-up routing |
| `last_tool_used` | `ToolName \| None` | Used for observability |
| `conversation_summary` | `str` | Human-readable per-turn summary |
| `shipping_inquiry_active` | `bool` | True when tracking question active |
| `return_refund_discussed` | `bool` | True once return topic raised |

---

## Knowledge Base Layout

```text
data/knowledge/
  products/
    asus_rog_strix_g16.md
    pulsex_smartwatch_2.md
  shipping/
    shipping_policy.md
  returns/
    returns_policy.md
  faq/
    payment_methods.md
  policies/
    warranty_policy.md
    loyalty_program.txt
```

---

## Observability

`TraceLogger` records one `ExecutionTrace` per customer turn:

```python
ExecutionTrace(
    session_id, message_id, selected_agent,
    routing_reason, tool_name, retrieval_sources,
    duration_ms, success, metadata
)
```

Traces are accessible via `orchestrator.get_traces(session_id)`.

Future: replace `TraceLogger` with OpenTelemetry exporter or Datadog agent.

---

## Cleanup Report (accumulated through Day 06)

| Artifact | Reason | Status |
|---|---|---|
| `src/agents/instruction.txt` | Prompt generated in code; no references | Removed |
| `_tool_owner` variable in `support_agent.py` | Second tuple element never used | Cleaned (renamed `_`) |
| `_agent` variable in `run_agent.py` | Agent not accessed directly; Runner wraps it | Suppressed with `_agent` |
| `src/rag/embeddings.py` | No inbound references; duplicate embedding path vs `vector_store.py` fastembed wrapper | Removed |
| `__pycache__` directories | Auto-regenerated bytecode; not source-managed | Excluded via `.gitignore` |

No other actively-referenced files were removed.

---

## Project Structure

```text
ecombot/
  run_agent.py                       # CLI runner (--multi-agent flag added)
  requirements.txt
  README.md
  data/
    knowledge/
      products/ shipping/ returns/ faq/ policies/
  src/
    agent.py                         # ADK Web entrypoint (multi-agent aware)
    agents/
      support_agent.py               # ADK LlmAgent factory (single + multi)
      contracts.py                   # AgentRequest/Response/RoutingDecision models
      base_agent.py                  # DomainAgent Protocol
      router_agent.py                # RouterAgent + RoutingRule
      product_agent.py               # ProductAgent (pure Python)
      order_agent.py                 # OrderAgent (pure Python)
      knowledge_agent.py             # KnowledgeAgent (pure Python)
      support_runtime_agent.py       # SupportAgent fallback (pure Python)
      orchestrator.py                # MultiAgentOrchestrator
    config/
      settings.py                    # EComBotSettings (multi_agent_mode added)
    data/
      products.json / orders.json / faq.json
    observability/
      __init__.py
      tracing.py                     # ExecutionTrace + TraceLogger
    services/
      __init__.py
      session_service.py             # SessionService + SessionContext
    tools/
      __init__.py
      adk_tools.py                   # EComSupportTools (ADK callables)
      data_loader.py                 # MockDataStore + JSON loader
      models.py                      # Pydantic data models
      services.py                    # ProductService/OrderService/…
  tests/
    test_day03_tool_workflows.py
    test_day04_session_memory.py
    test_day05_rag_workflows.py
    test_day06_multi_agent.py        # NEW – 21 multi-agent tests
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` example:

```env
OPENROUTER_API_KEY=your_real_openrouter_key
MODEL_NAME=openrouter/google/gemini-2.5-flash
AGENT_PERSONA=friendly
TOOLS_ENABLED=true
MOCK_DATA_DIR=src/data
KNOWLEDGE_BASE_DIR=data/knowledge
MULTI_AGENT_MODE=false
```

---

## Run

```bash
# Single-agent mode (default)
python run_agent.py

# Single-agent formal persona
python run_agent.py --persona formal

# Multi-agent mode (ProductAgent + OrderAgent + KnowledgeAgent)
python run_agent.py --multi-agent

# ADK Web (single-agent by default; set MULTI_AGENT_MODE=true in .env for multi-agent)
adk web

# Disable tools
python run_agent.py --no-tools
```

---

## Test

```bash
# Full suite (41 tests)
python -m pytest tests/ -v

# Day 06 only
python -m pytest tests/test_day06_multi_agent.py -v
```

---

## Future Extensibility

Adding a new agent requires three steps:

1. Create `src/agents/my_agent.py` with a class that implements `.handle(request) -> AgentResponse`.
2. Register it in `MultiAgentOrchestrator.build()` registry dict.
3. Add a `RoutingRule` in `RouterAgent.default()` with appropriate keywords.

No other files need changes. Candidates for future agents:

- `RecommendationAgent` — personalised upsell suggestions
- `InventoryAgent` — real-time stock availability from warehouse API
- `FraudDetectionAgent` — pattern-match suspicious order activity
- `HumanEscalationAgent` — handoff to live support queue
- `MarketingAgent` — promotions, discounts, loyalty rewards

---

## Future Redis Migration Plan

- Implement `RedisSessionRepository` satisfying the `SessionRepository` protocol in `session_service.py`.
- Swap `InMemorySessionRepository` for `RedisSessionRepository` in `get_session_service()`.
- Store `SessionState` as JSON with TTL; use `WATCH`/optimistic locking for concurrent updates.
- No changes required in domain agents, tools, or the orchestrator.
