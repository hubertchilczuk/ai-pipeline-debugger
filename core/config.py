"""Centralized configuration via Pydantic Settings."""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(StrEnum):
    CHEAP = "cheap"        # Ollama only
    ACCURATE = "accurate"  # OpenAI only
    AUTO = "auto"          # Ollama -> fallback OpenAI when low confidence


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # LLM Routing
    llm_mode: LLMMode = LLMMode.AUTO
    llm_confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)

    # OpenAI
    openai_api_key: str = "sk-replace-me"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_s: int = 30

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_s: int = 60

    # Vector DB
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection: str = "pipeline_incidents"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Ingestion
    airflow_base_url: str | None = None
    spark_history_url: str | None = None

    # Parser
    parser_use_llm_fallback: bool = Field(
        False,
        description="When true, unstructured logs are sent to the LLM router for extraction.",
    )

    # API security / observability
    api_key: str | None = Field(
        None,
        description="If set, all endpoints require X-API-Key header.",
    )
    rate_limit_per_minute: int = Field(
        60,
        ge=0,
        description="Per-client (API key or IP) requests per minute. 0 disables.",
    )
    cors_origins: list[str] = Field(default_factory=list)
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
