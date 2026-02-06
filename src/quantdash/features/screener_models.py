"""
Screener Pydantic models for type-safe signal aggregation.

Defines data structures for technical screening, signal detection,
scoring, and recommendation generation.
"""

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class SignalType(str, Enum):
    """Technical signal types that can be detected."""

    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    STOCH_OVERSOLD = "stoch_oversold"
    STOCH_OVERBOUGHT = "stoch_overbought"
    BOLLINGER_OVERSOLD = "bollinger_oversold"
    BOLLINGER_OVERBOUGHT = "bollinger_overbought"
    MFI_OVERSOLD = "mfi_oversold"
    MFI_OVERBOUGHT = "mfi_overbought"
    OBV_DIVERGENCE_BULL = "obv_divergence_bull"
    OBV_DIVERGENCE_BEAR = "obv_divergence_bear"
    EMA_CROSSOVER_BULL = "ema_crossover_bull"
    EMA_CROSSOVER_BEAR = "ema_crossover_bear"
    PRICE_ABOVE_EMA = "price_above_ema"
    PRICE_BELOW_EMA = "price_below_ema"
    VOLUME_SPIKE = "volume_spike"


SignalStrength = Literal["strong", "moderate", "weak"]

Recommendation = Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]


_BULLISH_SIGNALS = {
    SignalType.GOLDEN_CROSS,
    SignalType.RSI_OVERSOLD,
    SignalType.MACD_BULLISH,
    SignalType.BREAKOUT,
    SignalType.STOCH_OVERSOLD,
    SignalType.BOLLINGER_OVERSOLD,
    SignalType.MFI_OVERSOLD,
    SignalType.OBV_DIVERGENCE_BULL,
    SignalType.EMA_CROSSOVER_BULL,
    SignalType.PRICE_ABOVE_EMA,
    SignalType.VOLUME_SPIKE,
}

_BEARISH_SIGNALS = {
    SignalType.DEATH_CROSS,
    SignalType.RSI_OVERBOUGHT,
    SignalType.MACD_BEARISH,
    SignalType.BREAKDOWN,
    SignalType.STOCH_OVERBOUGHT,
    SignalType.BOLLINGER_OVERBOUGHT,
    SignalType.MFI_OVERBOUGHT,
    SignalType.OBV_DIVERGENCE_BEAR,
    SignalType.EMA_CROSSOVER_BEAR,
    SignalType.PRICE_BELOW_EMA,
}


class TechnicalSignal(BaseModel):
    """A detected technical signal with metadata."""

    signal_type: SignalType = Field(
        ...,
        description="Type of technical signal detected"
    )
    strength: SignalStrength = Field(
        ...,
        description="Signal strength level"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the signal"
    )
    date_detected: date = Field(
        ...,
        description="Date when signal was detected"
    )
    value: Optional[float] = Field(
        default=None,
        description="Numeric value associated with signal (e.g., RSI value)"
    )

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish signal."""
        return self.signal_type in _BULLISH_SIGNALS

    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish signal."""
        return self.signal_type in _BEARISH_SIGNALS

    @property
    def score_value(self) -> int:
        """Get numeric score for this signal's strength."""
        return {"strong": 3, "moderate": 2, "weak": 1}[self.strength]


class ScreenerConfig(BaseModel):
    """Configuration for the technical screener."""

    # RSI thresholds
    rsi_oversold: float = Field(default=30.0, ge=0, le=50)
    rsi_overbought: float = Field(default=70.0, ge=50, le=100)

    # Stochastic thresholds
    stoch_oversold: float = Field(default=20.0, ge=0, le=50)
    stoch_overbought: float = Field(default=80.0, ge=50, le=100)

    # MFI thresholds
    mfi_oversold: float = Field(default=20.0, ge=0, le=50)
    mfi_overbought: float = Field(default=80.0, ge=50, le=100)

    # Moving average periods for crossovers (SMA)
    ma_fast: int = Field(default=50, ge=5, le=100)
    ma_slow: int = Field(default=200, ge=50, le=500)

    # EMA crossover periods
    ema_fast: int = Field(default=12, ge=3, le=50)
    ema_slow: int = Field(default=26, ge=10, le=100)

    # Breakout detection
    breakout_lookback: int = Field(default=20, ge=5, le=100)
    volume_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)

    # Crossover detection window (days to look back for crossovers)
    crossover_window: int = Field(default=10, ge=1, le=30)

    # Pattern weight multiplier
    pattern_weight: float = Field(default=1.0, ge=0.1, le=3.0)

    @field_validator("rsi_overbought")
    @classmethod
    def validate_rsi_overbought(cls, v: float, info) -> float:
        """Ensure overbought > oversold."""
        if "rsi_oversold" in info.data and v <= info.data["rsi_oversold"]:
            raise ValueError("rsi_overbought must be greater than rsi_oversold")
        return v

    @field_validator("ma_slow")
    @classmethod
    def validate_ma_slow(cls, v: int, info) -> int:
        """Ensure slow MA > fast MA."""
        if "ma_fast" in info.data and v <= info.data["ma_fast"]:
            raise ValueError("ma_slow must be greater than ma_fast")
        return v

    @field_validator("ema_slow")
    @classmethod
    def validate_ema_slow(cls, v: int, info) -> int:
        """Ensure slow EMA > fast EMA."""
        if "ema_fast" in info.data and v <= info.data["ema_fast"]:
            raise ValueError("ema_slow must be greater than ema_fast")
        return v


class ScreenerResult(BaseModel):
    """Complete screening result for a symbol."""

    symbol: str = Field(..., description="Screened symbol")
    screening_date: date = Field(..., description="Date of screening")
    signals: list[TechnicalSignal] = Field(default_factory=list)

    # Composite metrics
    score: float = Field(default=0.0, description="Composite signal score")
    bullish_score: float = Field(default=0.0, description="Sum of bullish signal scores")
    bearish_score: float = Field(default=0.0, description="Sum of bearish signal scores")

    # Recommendation
    recommendation: Recommendation = Field(default="hold")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Current values
    current_price: Optional[float] = Field(default=None)
    current_rsi: Optional[float] = Field(default=None)

    # Ranking (set during portfolio screening)
    rank: Optional[int] = Field(default=None, description="Rank among screened symbols")

    # Summary
    notes: list[str] = Field(default_factory=list)

    @property
    def signal_count(self) -> int:
        """Total number of signals detected."""
        return len(self.signals)

    @property
    def strong_signals(self) -> list[TechnicalSignal]:
        """Get only strong signals."""
        return [s for s in self.signals if s.strength == "strong"]

    @property
    def matches_criteria(self) -> bool:
        """Check if signals meet minimum screening criteria."""
        strong = sum(1 for s in self.signals if s.strength == "strong")
        moderate = sum(1 for s in self.signals if s.strength == "moderate")
        weak = sum(1 for s in self.signals if s.strength == "weak")
        return strong >= 1 or moderate >= 2 or weak >= 3

    @computed_field
    @property
    def gauge_value(self) -> float:
        """
        Calculate gauge value from -100 (strong sell) to +100 (strong buy).

        Uses net score normalized to gauge range.
        """
        net = self.bullish_score - self.bearish_score
        # Max possible score: 9 bullish signals * 3 points each = 27
        max_score = 27.0
        normalized = (net / max_score) * 100
        return max(-100.0, min(100.0, normalized))

    @computed_field
    @property
    def gauge_label(self) -> str:
        """Human-readable label for gauge position."""
        v = self.gauge_value
        if v >= 60:
            return "Strong Buy"
        elif v >= 20:
            return "Buy"
        elif v > -20:
            return "Neutral"
        elif v > -60:
            return "Sell"
        else:
            return "Strong Sell"


class PortfolioScreeningResult(BaseModel):
    """Result of screening multiple symbols."""

    screening_date: date = Field(..., description="Date of screening")
    total_screened: int = Field(..., description="Total symbols screened")
    tickers_matching: int = Field(default=0, description="Symbols meeting criteria")
    results: list[ScreenerResult] = Field(default_factory=list, description="Ranked results")
    top_picks: list[str] = Field(default_factory=list, description="Top matching symbols")
    summary: str = Field(default="", description="Human-readable summary")


# Type exports
__all__ = [
    "SignalType",
    "SignalStrength",
    "Recommendation",
    "TechnicalSignal",
    "ScreenerConfig",
    "ScreenerResult",
    "PortfolioScreeningResult",
]
