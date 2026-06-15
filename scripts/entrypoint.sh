#!/usr/bin/env bash
# FILE: scripts/entrypoint.sh
# PURPOSE: Container startup sequence:
#   1. Wait for all dependent services (Redis, Postgres, ChromaDB)
#   2. Run database migrations
#   3. Ingest knowledge documents into ChromaDB
#   4. Start the ADK web server
#
# Design decisions:
#   - Pure bash for minimal runtime overhead.
#   - Python startup script handles complex initialisation.
#   - exec replaces shell with ADK process (clean PID 1 / signal handling).

set -euo pipefail

echo "============================================================"
echo "  eComBot — Container Startup"
echo "============================================================"

# ── 1. Run Python initialisation (migrations + knowledge ingestion) ─────────
echo "[startup] Running initialisation script..."
python scripts/startup.py

# ── 2. Start ADK Web Server ──────────────────────────────────────────────────
# 'exec' makes ADK PID 1 so Docker signals (SIGTERM) reach it directly.
# 'src' tells ADK to scan the src/ directory for the root_agent.
echo "[startup] Starting ADK web server on 0.0.0.0:8080..."
exec adk web src --host 0.0.0.0 --port 8080

