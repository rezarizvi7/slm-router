from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs are env vars prefixed with ROUTER_ (e.g. ROUTER_COMPLEXITY_THRESHOLD=0.6)."""

    model_config = SettingsConfigDict(env_prefix="ROUTER_", env_file=".env", extra="ignore")

    # --- Local tier (any OpenAI-compatible server: Ollama, vLLM, llama.cpp server) ---
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen2.5:1.5b-instruct"
    local_api_key: str = "ollama"
    local_timeout_s: float = 20.0
    # Time-to-first-token budget for streamed local calls; blow it and we escalate.
    local_ttft_budget_s: float = 3.0

    # --- Frontier tier ---
    frontier_provider: Literal["anthropic", "openai_compat"] = "anthropic"
    frontier_model: str = "claude-opus-5-5"
    frontier_base_url: str = "https://api.openai.com/v1"  # only used by openai_compat
    frontier_api_key: str = ""  # anthropic: falls back to ANTHROPIC_API_KEY / SDK credential chain
    frontier_timeout_s: float = 600.0
    frontier_max_tokens: int = 16000
    frontier_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    frontier_refusal_fallback: bool = True

    # --- Routing ---
    complexity_threshold: float = 0.5
    # Scores inside this band are ambiguous; if judge is enabled the SLM is asked to classify.
    judge_enabled: bool = True
    judge_band_low: float = 0.35
    judge_band_high: float = 0.65
    # After a local answer, escalate if it looks like the SLM gave up / got truncated.
    cascade_on_weak_answer: bool = True
    # Hard daily cap on estimated frontier spend (USD). 0 disables. When hit, everything stays local.
    frontier_daily_budget_usd: float = 0.0

    # --- Semantic cache ---
    cache_enabled: bool = True
    cache_backend: Literal["redis", "memory"] = "redis"
    redis_url: str = "redis://localhost:6379/0"
    cache_max_distance: float = 0.08  # cosine distance; 0 = identical
    cache_ttl_s: int = 24 * 3600
    embed_provider: Literal["ollama", "hashing"] = "ollama"
    embed_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
