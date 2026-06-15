# eComBot (Day 05 - Memory + RAG Knowledge Assistant)

eComBot is a Google ADK-powered e-commerce support assistant with OpenRouter model routing, tool-based workflows, in-session memory, and retrieval-augmented knowledge answering.

## Architecture Summary

```text
User (CLI / ADK Web)
  |
  v
run_agent.py / src/agent.py
  |
  v
create_support_agent(...) [src/agents/support_agent.py]
  |
  +--> LiteLlm (OpenRouter)
  |
  +--> ADK Runner + InMemorySessionService (conversation history)
  |
  +--> EComSupportTools [src/tools/adk_tools.py]
		  |
		  +--> ProductService      [search_products]
		  +--> OrderService        [lookup_order]
		  +--> FAQService          [retrieve_faq]
		  +--> KnowledgeService    [search_knowledge]
		  |
		  +--> MockDataStore + Knowledge Loader [src/tools/data_loader.py]
		  |      - src/data/products.json
		  |      - src/data/orders.json
		  |      - src/data/faq.json
		  |      - data/knowledge/**/*.md|txt
		  |
		  +--> SessionService [src/services/session_service.py]
				 - Context model
				 - Active session tracking
				 - Topic inference
				 - Conversation summary
```

## RAG Flow

```text
User Question
   |
   v
search_knowledge tool
   |
   v
KnowledgeService ranking (semantic-style token expansion + IDF cosine)
   |
   v
Top-K snippets + confidence + source files
   |
   v
LLM response with concise source citations
```

## Session Lifecycle

1. Create/attach session in ADK + local `SessionService`.
2. Store each user message and extract context (name, order IDs, support topic).
3. Activate session context for each turn.
4. Tool calls read/write session context (`last_tool_used`, recent products/orders).
5. Save assistant response and refresh summary for follow-up understanding.
6. Clear session when requested.

## Memory Model

`SessionContext` includes:
- `customer_name`
- `recent_products`
- `recent_order_ids`
- `active_support_topic`
- `last_tool_used`
- `conversation_summary`
- `shipping_inquiry_active`
- `return_refund_discussed`

## Memory + RAG Interaction

- User: `Tell me about the ASUS ROG Strix G16`
- User: `What warranty comes with it?`
- Behavior:
  1. Product context is remembered in session.
  2. `search_knowledge` uses session summary for query expansion.
  3. Response includes warranty details and source file citations.

## Tool Routing Policy

- Product discovery/comparison/availability -> `search_products`
- Order status/tracking/ETA -> `lookup_order`
- FAQ answer lookup -> `retrieve_faq`
- Policy/specification/warranty/shipping/returns/loyalty docs -> `search_knowledge`
- Greetings/small talk -> direct LLM response

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

## Cleanup Report

Audit results and removals:

1. `src/agents/instruction.txt`
   - Why unused: agent prompt is generated in `src/agents/support_agent.py`; file has no runtime imports.
   - Dependency analysis: only README historical mention, no code path references.
   - Safe removal: confirmed and removed.

2. `__pycache__` artifacts (under `src/` and `tests/`)
   - Why unused: bytecode cache files are regenerated automatically and should not be source-managed.
   - Dependency analysis: never imported directly by application code.
   - Safe removal: confirmed and removed.

## Project Structure

```text
ecombot/
  run_agent.py
  requirements.txt
  README.md
  data/
	knowledge/
	  products/
	  shipping/
	  returns/
	  faq/
	  policies/
  src/
	agent.py
	agents/
	  support_agent.py
	config/
	  settings.py
	data/
	  products.json
	  orders.json
	  faq.json
	services/
	  __init__.py
	  session_service.py
	tools/
	  __init__.py
	  adk_tools.py
	  data_loader.py
	  models.py
	  services.py
  tests/
	test_day03_tool_workflows.py
	test_day04_session_memory.py
	test_day05_rag_workflows.py
```

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
```

## Run

```bash
python run_agent.py
python run_agent.py --persona formal
python run_agent.py --no-tools
adk web
```

## Test

```bash
python -m pytest tests/test_day03_tool_workflows.py tests/test_day04_session_memory.py tests/test_day05_rag_workflows.py -q
```

## Future Redis Migration Plan

- Keep `SessionRepository` protocol and implement `RedisSessionRepository`.
- Preserve `SessionService` public API for tool/runner compatibility.
- Store session context + summary with TTL and optimistic locking.
- Keep retrieval layer stateless; only pass `session_id` and read context through `SessionService`.
