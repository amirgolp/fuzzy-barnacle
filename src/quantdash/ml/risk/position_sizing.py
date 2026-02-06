"""Position sizing based on model confidence and risk profile.

Maps confidence score to position size with leverage limits.

Confidence < 0.55  → no trade (size = 0)
Confidence 0.55-1.0 → linear ramp from 1.0× to max_leverage×
"""

from __future__ import annotations

from quantdash.ml.config import AssetRiskProfile, RISK_PROFILES


def confidence_to_size(
    confidence: float,
    risk_profile: AssetRiskProfile,
    min_confidence: float = 0.55,
) -> float:
    """Map model confidence to position size multiplier.

    Args:
        confidence: Model confidence [0, 1].
        risk_profile: Per-asset risk limits.
        min_confidence: Minimum confidence to trade.

    Returns:
        Position size multiplier (0.0 to max_leverage).
    """
    if confidence < min_confidence:
        return 0.0

    # Linear interpolation: [min_confidence, 1.0] → [1.0, max_leverage]
    t = (confidence - min_confidence) / (1.0 - min_confidence)
    size = 1.0 + t * (risk_profile.max_leverage - 1.0)

    return min(size, risk_profile.max_leverage)


def compute_position(
    action: int,
    confidence: float,
    symbol: str,
    current_equity: float,
    min_confidence: float = 0.55,
) -> dict:
    """Compute full position specification.

    Args:
        action: Model action (-1=sell, 0=hold, 1=buy).
        confidence: Model confidence [0, 1].
        symbol: Asset symbol for risk profile lookup.
        current_equity: Current portfolio equity.
        min_confidence: Minimum confidence threshold.

    Returns:
        Dict with direction, size, dollar_amount, leverage.
    """
    risk_profile = RISK_PROFILES.get(symbol)
    if risk_profile is None:
        # Default conservative profile
        risk_profile = AssetRiskProfile(
            max_leverage=1.0,
            max_drawdown_pct=10.0,
            max_position_pct=20.0,
            fee_bps=10,
            session_type="session",
        )

    if action == 0:
        return {
            "direction": "hold",
            "size": 0.0,
            "dollar_amount": 0.0,
            "leverage": 0.0,
        }

    leverage = confidence_to_size(confidence, risk_profile, min_confidence)

    # Cap position as percentage of equity
    max_dollar = current_equity * (risk_profile.max_position_pct / 100)
    dollar_amount = min(current_equity * leverage, max_dollar)

    direction = "long" if action == 1 else "short"

    return {
        "direction": direction,
        "size": leverage,
        "dollar_amount": round(dollar_amount, 2),
        "leverage": round(leverage, 2),
    }
