from __future__ import annotations

import logging
import importlib
from functools import lru_cache
from time import perf_counter
from typing import Any

cl = importlib.import_module("chainlit")

from src.agents.orchestrator import MultiAgentOrchestrator
from src.config.settings import get_settings
from src.services.session_service import SessionService, get_session_service
from src.tools.data_loader import get_mock_data_store
from src.ui.actions.definitions import actions_for_agent, prompt_for_action
from src.ui.cards.renderers import render_card_for_agent
from src.ui.components.formatters import explainability_text, result_summary, tool_step_label
from src.ui.session_ui import UISessionState, display_agent_name, new_ui_session_id

logger = logging.getLogger(__name__)


def _settings_safe():
    try:
        return get_settings()
    except Exception:
        # Chainlit UI still works for local tests without full LLM env setup.
        class _Fallback:
            session_id_prefix = "ecombot-session"
            user_id = "default_user"
            mock_data_dir = "src/data"
            knowledge_base_dir = "data/knowledge"

        return _Fallback()


@lru_cache(maxsize=1)
def _session_service() -> SessionService:
    return get_session_service()


@lru_cache(maxsize=1)
def _orchestrator() -> MultiAgentOrchestrator:
    cfg = _settings_safe()
    data_store = get_mock_data_store(data_dir=cfg.mock_data_dir, knowledge_dir=cfg.knowledge_base_dir)
    return MultiAgentOrchestrator.build(data_store=data_store, session_service=_session_service())


async def _dispatch_and_render(user_text: str) -> None:
    state: UISessionState = cl.user_session.get("ui_state")
    session_service = _session_service()
    orchestrator = _orchestrator()

    session_service.remember_user_message(state.session_id, user_text)

    route_start = perf_counter()
    response, decision = orchestrator.dispatch(
        session_id=state.session_id,
        user_id=state.user_id,
        message=user_text,
    )
    elapsed_ms = (perf_counter() - route_start) * 1000

    state.last_agent = response.agent_name
    state.last_tool = response.tool_name
    state.last_sources = response.sources
    state.last_response_message = response.message
    cl.user_session.set("ui_state", state)

    with cl.Step(name="Routing Message") as step:
        step.output = (
            f"Selected **{display_agent_name(decision.selected_agent)}** "
            f"(confidence {decision.confidence:.2f}) in {elapsed_ms:.1f} ms."
        )

    with cl.Step(name=tool_step_label(response.tool_name)) as step:
        step.output = result_summary(response.tool_name, response.metadata)

    if response.agent_name == "knowledge_agent" and response.tool_name == "search_knowledge":
        with cl.Step(name="Retrieving ChromaDB Context") as step:
            sources = response.sources or ["No source"]
            step.output = f"Top source: {sources[0]}"

    session_service.remember_assistant_message(state.session_id, response.message)
    session_service.summarize_conversation(state.session_id)

    actions = [
        cl.Action(name=item.name, label=item.label, payload={"value": item.value})
        for item in actions_for_agent(response.agent_name)
    ]
    actions.append(cl.Action(name="explain_answer", label="How was this answer generated?", payload={}))

    header = f"[{display_agent_name(response.agent_name)}]"
    await cl.Message(content=f"{header}\n\n{response.message}", actions=actions).send()

    card_text = render_card_for_agent(response.agent_name, response.metadata)
    if card_text:
        await cl.Message(content=card_text).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cfg = _settings_safe()
    session_id = new_ui_session_id(prefix=cfg.session_id_prefix)
    user_id = cfg.user_id

    session_service = _session_service()
    session_service.create_session(session_id=session_id, user_id=user_id)

    cl.user_session.set("ui_state", UISessionState(session_id=session_id, user_id=user_id))

    await cl.Message(
        content=(
            "Welcome to eComBot UI. I route your request to the best agent "
            "(Support, Product, Order, or Knowledge) and show source-grounded answers."
        )
    ).send()


@cl.on_message
async def on_message(message: Any) -> None:
    await _dispatch_and_render(message.content)


async def _run_action(action_name: str) -> None:
    prompt = prompt_for_action(action_name)
    if prompt:
        await _dispatch_and_render(prompt)


@cl.action_callback("budget_under_500")
@cl.action_callback("budget_500_1000")
@cl.action_callback("premium_products")
@cl.action_callback("track_order")
@cl.action_callback("start_return")
@cl.action_callback("contact_support")
@cl.action_callback("related_policies")
async def on_quick_action(action: Any) -> None:
    await _run_action(action.name)


@cl.action_callback("show_source")
async def on_show_source(action: Any) -> None:
    _ = action
    state: UISessionState = cl.user_session.get("ui_state")
    sources = state.last_sources or ["No sources available"]
    lines = "\n".join(f"- {src}" for src in sources)
    await cl.Message(content=f"### Retrieved Sources\n{lines}").send()


@cl.action_callback("explain_answer")
async def on_explain_answer(action: Any) -> None:
    _ = action
    state: UISessionState = cl.user_session.get("ui_state")
    explanation = explainability_text(
        selected_agent=state.last_agent or "support_agent",
        tool_name=state.last_tool,
        sources=state.last_sources,
        response_text=state.last_response_message,
    )
    await cl.Message(content=explanation).send()

