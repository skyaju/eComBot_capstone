"""
FILE: src/agent.py  (ADK convention: must be named `agent.py` at package root)
PURPOSE: ADK Web discovery entrypoint.

When you run `adk web` from the project root, ADK scans for a module named
`agent` that exposes a variable named `root_agent`.  This file satisfies that
contract by delegating to our factory.

Design decision: We expose only `root_agent` here.  The Runner and
SessionService are managed internally by ADK Web when using `adk web`.
For the CLI runner (run_agent.py) we build our own Runner so we can manage
session IDs and stream responses ourselves.
"""

from __future__ import annotations

import logging

from src.agents.support_agent import create_support_agent
from src.config.settings import get_settings

# Configure logging early so ADK Web picks up our log level
settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ADK Web looks for this specific variable name
root_agent, _session_service, _runner = create_support_agent(settings)

logger.info("root_agent exposed for ADK Web ✓  (persona=%s)", settings.agent_persona)
