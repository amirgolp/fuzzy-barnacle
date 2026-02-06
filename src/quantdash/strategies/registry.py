"""Strategy registry for discovering and instantiating strategies."""

from typing import Any

from .base import Strategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(strategy_id: str, cls: type[Strategy]) -> None:
    """Register a strategy class by ID."""
    STRATEGY_REGISTRY[strategy_id] = cls


def get_strategy(strategy_id: str, params: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a registered strategy by ID."""
    if strategy_id not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {strategy_id}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_id](params)


def list_strategies() -> list[dict[str, Any]]:
    """List all registered strategies with metadata."""
    result = []
    for sid, cls in STRATEGY_REGISTRY.items():
        instance = cls()
        info = instance.get_info()
        info["id"] = sid
        result.append(info)
    return result
