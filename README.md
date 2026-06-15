# eComBot (Day 03 - Tool-Based Support Workflows)

eComBot is a Google ADK-powered e-commerce support assistant with OpenRouter integration.

Day 03 upgrades the assistant from pure LLM reasoning to intelligent tool-driven workflows for:
- Product search
- Order lookup
- FAQ retrieval

## Architecture

```text
User (CLI / ADK Web)
		|
		v
run_agent.py / src/agent.py
		|
		v
create_support_agent(...)  [src/agents/support_agent.py]
		|
		+--> LiteLlm(OpenRouter)
		|
		+--> ADK Tools (search_products, lookup_order, retrieve_faq)
				 |
				 v
			EComSupportTools facade      [src/tools/adk_tools.py]
				 |
				 v
			Domain services              [src/tools/services.py]
				 |
				 v
			MockDataStore + validators   [src/tools/data_loader.py, src/tools/models.py]
				 |
				 v
			JSON knowledge/data files    [src/data/*.json]
```

## Tool Flow

1. Customer message arrives.
2. Agent instruction enforces intent-aware behavior:
   - Product intent -> `search_products`
   - Order intent -> `lookup_order`
   - Policy/FAQ intent -> `retrieve_faq`
   - Greeting/small-talk -> direct conversational response
3. Tool returns structured output.
4. Assistant responds naturally (without exposing internal tool calls).
5. If a tool fails or returns no results, the assistant recovers gracefully.

## Project Structure

```text
ecombot/
  run_agent.py
  requirements.txt
  README.md
  src/
	agent.py
	agents/
	  support_agent.py
	  instruction.txt
	config/
	  settings.py
	data/
	  products.json
	  orders.json
	  faq.json
	tools/
	  __init__.py
	  adk_tools.py
	  data_loader.py
	  models.py
	  services.py
  tests/
	test_day03_tool_workflows.py
	day03_scenarios.md
```

## Setup

1. Create and activate your Python 3.11+ environment.
2. Install dependencies.
3. Set your OpenRouter key in `.env`.

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
```

## Running

CLI mode:

```bash
python run_agent.py
python run_agent.py --persona formal
python run_agent.py --no-tools
```

ADK Web:

```bash
adk web
```

## Tests

Run the Day 03 workflow tests:

```bash
python -m pytest tests/test_day03_tool_workflows.py -q
```

## Example Conversations

### Product Search
**User:** "Show me lightweight running shoes under sports category."

**Assistant behavior:** Calls product search, returns matching item(s) with price and availability.

### Product Comparison
**User:** "Compare your Home products for comfort and kitchen use."

**Assistant behavior:** Calls product search with category + keywords, then compares returned products.

### Order Lookup
**User:** "Track ORD-10001"

**Assistant behavior:** Calls order lookup and replies with status, ETA, and tracking details.

### FAQ Retrieval
**User:** "What is your return policy?"

**Assistant behavior:** Retrieves structured FAQ answer.

### Invalid Order ID
**User:** "Track 10001"

**Assistant behavior:** Returns a graceful validation message with expected format `ORD-12345`.

## Day 03 Error Handling

The tool layer handles and normalizes:
- `product_not_found`
- `order_not_found`
- `invalid_order_id`
- `empty_search_criteria`
- `faq_not_found`
- `tool_execution_failed`

The agent instructions require graceful recovery for each case.
