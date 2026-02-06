"""Application settings using Pydantic BaseSettings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    ENV: Literal["dev", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:8501", "http://localhost:5173"])

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./quantdash.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Data Provider
    DATA_PROVIDER_DEFAULT: str = "yfinance"
    CACHE_TTL_SECONDS: int = 3600

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-pro"

    # Backtest defaults
    DEFAULT_FEE_BPS: int = 10
    DEFAULT_SLIPPAGE_BPS: int = 5
    DEFAULT_TIMEFRAME: str = "1d"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
