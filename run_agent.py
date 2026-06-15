"""
FILE: run_agent.py
PURPOSE: Interactive CLI runner for local development and quick smoke-testing.

Features:
  - Full conversation loop with session memory
  - Streaming token output (feels responsive)
  - Graceful exit via 'quit', 'exit', or Ctrl-C
  - Coloured prompt for readability
  - Startup banner with active configuration
  - --multi-agent flag to enable multi-agent orchestration mode

Usage:
    python run_agent.py
    python run_agent.py --persona formal
    python run_agent.py --session-id my-test-session-001
    python run_agent.py --multi-agent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

# ── Ensure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Now safe to import project modules ─────────────────────────────────────
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from src.agents.support_agent import create_multi_agent_system, create_support_agent
from src.config.settings import EComBotSettings, get_settings
from src.services.session_service import get_session_service

# ── ANSI colour codes (disabled on Windows without colorama) ───────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_DIM = "\033[2m"
_MAGENTA = "\033[95m"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Async chat loop
# ─────────────────────────────────────────────────────────────────────────────

async def chat_loop(
    runner: Runner,
    session_service: InMemorySessionService,
    settings: EComBotSettings,
    session_id: str,
) -> None:
    """
    Run an interactive REPL against the eComBot agent.

    Each message is sent to the Runner which handles:
      1. Loading session history from SessionService
      2. Calling the LLM with full context
      3. Persisting the updated session
      4. Streaming response events back to us
    """
    # Create the session (first-time only; subsequent messages reuse it)
    await session_service.create_session(
        app_name=settings.app_name,
        user_id=settings.user_id,
        session_id=session_id,
    )
    memory_service = get_session_service()
    memory_service.create_session(session_id=session_id, user_id=settings.user_id)

    _print_banner(settings, session_id)

    while True:
        # ── Get user input ────────────────────────────────────────────────
        try:
            user_input = input(f"\n{_BOLD}{_CYAN}You:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{_DIM}Session ended. Goodbye! 👋{_RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print(f"\n{_DIM}Thanks for using eComBot. Have a great day! 🛍️{_RESET}")
            break

        # ── Stream response ───────────────────────────────────────────────
        memory_service.remember_user_message(session_id=session_id, message=user_input)
        print(f"\n{_BOLD}{_GREEN}eComBot:{_RESET} ", end="", flush=True)

        full_response = ""
        try:
            with memory_service.activate_session(session_id):
                async for event in runner.run_async(
                    user_id=settings.user_id,
                    session_id=session_id,
                    new_message=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=user_input)],
                    ),
                ):
                    # ADK streams multiple event types; we only print text chunks
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                print(part.text, end="", flush=True)
                                full_response += part.text

        except Exception as exc:
            print(f"\n{_YELLOW}⚠ Error communicating with the model: {exc}{_RESET}")
            logger.exception("Runner error during chat loop")
            continue

        print()  # newline after streaming response
        memory_service.remember_assistant_message(session_id=session_id, message=full_response)
        memory_service.summarize_conversation(session_id=session_id)

        if not full_response:
            print(f"{_DIM}(No response received — check your API key and model name){_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(settings: EComBotSettings, session_id: str) -> None:
    """Print a startup banner showing active configuration."""
    persona_icon = "😊" if settings.agent_persona == "friendly" else "🎩"
    mode = "multi-agent 🤖" if settings.multi_agent_mode else "single-agent"
    print(
        f"\n{'─' * 60}\n"
        f"  {_BOLD}eComBot — AI Customer Support Assistant{_RESET}\n"
        f"{'─' * 60}\n"
        f"  Model   : {settings.model_name}\n"
        f"  Persona : {settings.agent_persona} {persona_icon}\n"
        f"  Mode    : {mode}\n"
        f"  Session : {session_id}\n"
        f"  Temp    : {settings.model_temperature}\n"
        f"{'─' * 60}\n"
        f"  {_DIM}Type 'quit' or press Ctrl-C to exit{_RESET}\n"
        f"{'─' * 60}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="eComBot interactive CLI runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_agent.py\n"
            "  python run_agent.py --persona formal\n"
            "  python run_agent.py --session-id test-001 --persona friendly\n"
        ),
    )
    parser.add_argument(
        "--persona",
        choices=["friendly", "formal"],
        default=None,
        help="Override AGENT_PERSONA from .env",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Reuse a specific session ID (default: new UUID per run)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override LOG_LEVEL from .env",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable Day 03 tool workflows and run as pure conversational mode.",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Enable multi-agent mode (Day 06): ProductAgent, OrderAgent, KnowledgeAgent sub-agents.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Apply CLI overrides ───────────────────────────────────────────────
    settings = get_settings()

    # Build an override settings object when CLI flags are present
    overrides: dict = {}
    if args.persona:
        overrides["agent_persona"] = args.persona
    if args.log_level:
        overrides["log_level"] = args.log_level
    if args.no_tools:
        overrides["tools_enabled"] = False

    if overrides:
        # Pydantic allows model_copy(update=...) for immutable override
        settings = settings.model_copy(update=overrides)

    # ── Apply --multi-agent flag ──────────────────────────────────────────
    if args.multi_agent and not settings.multi_agent_mode:
        settings = settings.model_copy(update={"multi_agent_mode": True})

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,   # keep stderr clean; stdout is for chat
    )

    # ── Build agent ───────────────────────────────────────────────────────
    session_id = args.session_id or f"{settings.session_id_prefix}-{uuid.uuid4().hex[:8]}"

    if settings.multi_agent_mode:
        _agent, session_service, runner = create_multi_agent_system(settings)
    else:
        _agent, session_service, runner = create_support_agent(settings)

    # ── Run async loop ────────────────────────────────────────────────────
    asyncio.run(
        chat_loop(
            runner=runner,
            session_service=session_service,
            settings=settings,
            session_id=session_id,
        )
    )


if __name__ == "__main__":
    main()
