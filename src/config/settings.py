"""
FILE: src/config/settings.py
PURPOSE: Centralised, type-safe configuration using Pydantic BaseSettings.
         All environment variables are validated at startup; the rest of the
         codebase never touches os.getenv() directly.

Design decision: Pydantic V2 BaseSettings gives us:
  - Automatic .env loading (no manual dotenv calls needed)
  - Type coercion  (e.g. "true" → True)
  - Fail-fast validation with readable error messages
  - A single source of truth for every configurable value
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class EComBotSettings(BaseSettings):
    """
    Application-wide settings for eComBot.

    Priority order (highest → lowest):
      1. Real environment variables  (CI/CD secrets, Docker env)
      2. .env file                   (local development)
      3. Field defaults              (safe fallbacks)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,          # OPENROUTER_API_KEY == openrouter_api_key
        extra="ignore",                # silently drop unknown vars (e.g. SHELL, PATH)
        validate_default=True,
    )

    # ------------------------------------------------------------------ #
    # OpenRouter / LLM                                                     #
    # ------------------------------------------------------------------ #
    openrouter_api_key: str = Field(
        ...,                           # required – no default
        description="Secret API key from https://openrouter.ai/keys",
    )
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter-compatible endpoint.",
    )
    model_name: str = Field(
        default="openrouter/google/gemini-2.5-flash",
        description=(
            "Model identifier as listed on OpenRouter, e.g. "
            "'openai/gpt-4o-mini', 'anthropic/claude-3-haiku', "
            "'google/gemini-flash-1.5'."
        ),
    )
    model_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more deterministic.",
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Hard cap on tokens the model may generate per turn.",
    )

    # ------------------------------------------------------------------ #
    # Agent Identity                                                        #
    # ------------------------------------------------------------------ #
    agent_name: str = Field(
        default="eComBot",
        description="Internal agent identifier used by ADK runner.",
    )
    agent_persona: Literal["friendly", "formal"] = Field(
        default="friendly",
        description=(
            "'friendly' – warm, first-name basis, uses light emoji.  "
            "'formal'   – professional, no emoji, complete sentences."
        ),
    )

    # ------------------------------------------------------------------ #
    # Session / Memory                                                      #
    # ------------------------------------------------------------------ #
    session_id_prefix: str = Field(
        default="ecombot-session",
        description="Prefix for generated session IDs.",
    )
    app_name: str = Field(
        default="ecombot_app",
        description="ADK application name (used by SessionService).",
    )
    user_id: str = Field(
        default="default_user",
        description="Default user ID when running locally via run_agent.py.",
    )

    # ------------------------------------------------------------------ #
    # Logging                                                               #
    # ------------------------------------------------------------------ #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )

    # ------------------------------------------------------------------ #
    # Day 03 - Tooling / Data                                               #
    # ------------------------------------------------------------------ #
    tools_enabled: bool = Field(
        default=True,
        description="Enable ADK tool-based product/order/FAQ workflows.",
    )
    mock_data_dir: str = Field(
        default="src/data",
        description="Path to JSON mock data directory (products/orders/faq).",
    )
    knowledge_base_dir: str = Field(
        default="data/knowledge",
        description="Path to markdown/text knowledge documents for RAG retrieval.",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                            #
    # ------------------------------------------------------------------ #
    @field_validator("openrouter_api_key")
    @classmethod
    def api_key_must_not_be_placeholder(cls, v: str) -> str:
        """Catch the classic 'I forgot to set my key' mistake."""
        bad_values = {"your_openrouter_api_key_here", "sk-or-...", "", "changeme"}
        if v.lower() in bad_values:
            raise ValueError(
                "OPENROUTER_API_KEY looks like a placeholder. "
                "Set a real key in your .env file."
            )
        return v

    @model_validator(mode="after")
    def log_loaded_settings(self) -> "EComBotSettings":
        """Emit a startup log so operators can verify configuration."""
        masked_key = (
            self.openrouter_api_key[:8] + "..." + self.openrouter_api_key[-4:]
            if len(self.openrouter_api_key) > 12
            else "***"
        )
        logger.info(
            "EComBotSettings loaded | model=%s | persona=%s | temp=%.1f | key=%s",
            self.model_name,
            self.agent_persona,
            self.model_temperature,
            masked_key,
        )
        return self


@lru_cache(maxsize=1)
def get_settings() -> EComBotSettings:
    """
    Return the singleton Settings instance.

    Using @lru_cache means .env is parsed exactly once per process lifetime.
    Call get_settings.cache_clear() in tests to reload fresh config.
    """
    return EComBotSettings()
