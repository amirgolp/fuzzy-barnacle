"""Core domain models using Pydantic."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Timeframe(str, Enum):
    """Supported timeframes for data."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1wk"
    MO1 = "1mo"


class ChartStyle(str, Enum):
    """Chart rendering styles."""
    CANDLE = "Candle"
    LINE = "Line"
    HEIKIN_ASHI = "Heikin Ashi"


class IndicatorType(str, Enum):
    """Indicator display type."""
    OVERLAY = "overlay"  # Rendered on price chart
    SUBCHART = "subchart"  # Rendered in separate panel


class IndicatorCategory(str, Enum):
    """Indicator categories."""
    TREND = "Trend"
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    VOLUME = "Volume"


class PatternDirection(str, Enum):
    """Pattern signal direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# =============================================================================
# Data Models
# =============================================================================


class Instrument(BaseModel):
    """Tradable instrument."""
    symbol: str
    name: Optional[str] = None
    asset_class: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"


class Bar(BaseModel):
    """Single OHLCV bar."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DatasetRef(BaseModel):
    """Reference to a dataset with version tracking."""
    provider: str
    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    adjusted: bool = True
    version_hash: Optional[str] = None


# =============================================================================
# Indicator Models
# =============================================================================


class IndicatorConfig(BaseModel):
    """Configuration for a technical indicator."""
    name: str
    func_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    indicator_type: IndicatorType = IndicatorType.SUBCHART
    category: IndicatorCategory = IndicatorCategory.TREND
    description: str = ""
    color: str = "#2962FF"


class ActiveIndicator(BaseModel):
    """An indicator active on a chart."""
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Pattern Models
# =============================================================================


class PatternEvent(BaseModel):
    """Detected chart pattern."""
    pattern_type: str
    start_index: int
    end_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    direction: PatternDirection
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Fundamental Models
# =============================================================================


class FundamentalData(BaseModel):
    """Fundamental metrics for a symbol."""
    symbol: str
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    peg_ratio: Optional[float] = None
    enterprise_value: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    last_updated: Optional[datetime] = None


# =============================================================================
# Chart Configuration Models
# =============================================================================


class PinnedChart(BaseModel):
    """Saved chart configuration."""
    id: str
    symbol: str
    timeframe: str
    chart_style: ChartStyle = ChartStyle.CANDLE
    indicators: list[ActiveIndicator] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Strategy & Backtest Models
# =============================================================================


class ExecutionConfig(BaseModel):
    """Execution parameters for backtesting."""
    fee_bps: int = Field(default=10, ge=0, description="Trading fee in basis points")
    slippage_bps: int = Field(default=5, ge=0, description="Slippage in basis points")
    allow_short: bool = False


class PortfolioConfig(BaseModel):
    """Portfolio configuration for backtesting."""
    initial_cash: float = Field(default=100_000.0, gt=0)
    rebalance_frequency: Optional[str] = None  # "daily", "weekly", "monthly"


class RiskLimits(BaseModel):
    """Risk constraints for backtesting."""
    max_leverage: float = Field(default=1.0, ge=1.0)
    max_drawdown: float = Field(default=0.25, ge=0.0, le=1.0)
    var_limit: Optional[float] = None
    cvar_limit: Optional[float] = None


class StrategySpec(BaseModel):
    """Strategy specification."""
    strategy_id: str
    name: str
    description: Optional[str] = None
    type: str = Field(default="builtin", description="builtin or dsl")
    params: dict[str, Any] = Field(default_factory=dict)
    universe: list[str] = Field(default_factory=list)


class BacktestSpec(BaseModel):
    """Full backtest specification."""
    dataset: DatasetRef
    strategy: StrategySpec
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)
    seed: int = 42


class TradeRecord(BaseModel):
    """A single trade from a backtest."""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    pnl: float = 0.0
    return_pct: float = 0.0


class BacktestResult(BaseModel):
    """Complete backtest result."""
    # Summary metrics
    cagr: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    exposure: float = 0.0

    # Series data
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[TradeRecord] = Field(default_factory=list)

    # Metadata
    strategy_id: str = ""
    symbol: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class PortfolioOptimizationResult(BaseModel):
    """Result from portfolio optimization."""
    method: str  # "mean_variance" or "risk_parity"
    weights: dict[str, float] = Field(default_factory=dict)
    risk_contributions: dict[str, float] = Field(default_factory=dict)
    expected_return: float = 0.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
