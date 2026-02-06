"""ML configuration: model, training, labeling, and per-asset configs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class SessionType(str, Enum):
    """Asset trading session type."""
    TWENTY_FOUR_SEVEN = "24/7"
    SESSION = "session"


class SignalAction(int, Enum):
    """Model output actions mapped to position direction."""
    SELL = -1
    HOLD = 0
    BUY = 1


# =============================================================================
# Model Architecture Config
# =============================================================================


class ModelConfig(BaseModel):
    """TemporalFusionSignalNet architecture hyperparameters."""
    # Branch input dimensions (defaults match typical feature counts)
    price_channels: int = 42  # OHLCV(5) + returns(1) + accel(1) + indicators(~35)
    num_pattern_features: int = 106  # 53 patterns x 2 (detected + confidence)
    num_macro_features: int = 13
    cross_asset_channels: int = 20  # varies per asset (num_correlated × 5)
    max_articles: int = 10
    finbert_dim: int = 768

    # Temporal dimensions
    lookback_1h: int = 120
    lookback_15m: int = 480

    # Branch output dimensions
    price_d_model: int = 128
    volume_d_model: int = 64
    pattern_d_model: int = 64
    news_d_model: int = 64
    macro_d_model: int = 32
    cross_d_model: int = 64

    # Transformer settings
    price_nhead: int = 4
    price_num_layers_1h: int = 2
    price_num_layers_15m: int = 1
    volume_nhead: int = 2
    news_nhead: int = 4
    fusion_nhead: int = 4
    fusion_num_layers: int = 2
    fusion_d: int = 64
    output_d: int = 256

    # Regularization
    dropout: float = 0.1
    fusion_dropout: float = 0.2


# =============================================================================
# Training Config
# =============================================================================


class LabelingConfig(BaseModel):
    """Triple-barrier labeling parameters."""
    tp_mult: float = Field(default=2.0, description="TP barrier = tp_mult * ATR")
    sl_mult: float = Field(default=1.5, description="SL barrier = sl_mult * ATR")
    max_bars: int = Field(default=5, description="Time barrier in bars")
    atr_period: int = Field(default=14, description="ATR period for barrier sizing")
    fee_bps: int = Field(default=10, description="Fee in basis points for barrier adjustment")


class TrainingConfig(BaseModel):
    """Supervised training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 10
    focal_alpha: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.25])
    focal_gamma: float = 2.0
    confidence_loss_weight: float = 0.1
    grad_clip_norm: float = 1.0
    use_fp16: bool = True
    num_workers: int = 4


class RLConfig(BaseModel):
    """PPO fine-tuning hyperparameters."""
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    lambda_drawdown: float = 0.5
    freeze_encoders: bool = True


class WalkForwardConfig(BaseModel):
    """Walk-forward expanding window validation."""
    initial_train_size: int = 2000
    val_size: int = 500
    step_size: int = 500
    min_train_size: int = 1000


# =============================================================================
# Normalization Config
# =============================================================================


class NormalizationConfig(BaseModel):
    """Feature normalization settings."""
    method: str = "rolling_zscore"
    lookback: int = 500
    min_periods: int = 50
    clip_value: float = 5.0  # Clip z-scores to [-5, 5]


# =============================================================================
# Risk Profile
# =============================================================================


@dataclass
class AssetRiskProfile:
    """Per-asset risk limits."""
    max_leverage: float
    max_drawdown_pct: float
    max_position_pct: float
    fee_bps: int
    session_type: SessionType


# =============================================================================
# Per-Asset Configuration
# =============================================================================


@dataclass
class AssetSession:
    """Market session timing (UTC hours)."""
    open_hour_utc: int
    close_hour_utc: int
    trading_days: str  # "Mon-Fri", "Sun-Fri", "24/7"


class AssetConfig(BaseModel):
    """Complete per-asset ML configuration."""
    symbol: str
    correlated_assets: list[str]
    session_type: SessionType
    session_open_utc: int = 0
    session_close_utc: int = 0
    trading_days: str = "Mon-Fri"
    arch_config: ModelConfig = Field(default_factory=ModelConfig)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    labeling_config: LabelingConfig = Field(default_factory=LabelingConfig)
    rl_config: RLConfig = Field(default_factory=RLConfig)
    walk_forward_config: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    normalization_config: NormalizationConfig = Field(default_factory=NormalizationConfig)

    @property
    def num_cross_asset_channels(self) -> int:
        return len(self.correlated_assets) * 5  # OHLCV per correlated asset


# =============================================================================
# Risk Profiles
# =============================================================================

RISK_PROFILES: dict[str, AssetRiskProfile] = {
    "GC=F": AssetRiskProfile(
        max_leverage=3.0, max_drawdown_pct=15.0, max_position_pct=25.0,
        fee_bps=10, session_type=SessionType.SESSION,
    ),
    "BTC-USD": AssetRiskProfile(
        max_leverage=2.0, max_drawdown_pct=20.0, max_position_pct=15.0,
        fee_bps=10, session_type=SessionType.TWENTY_FOUR_SEVEN,
    ),
    "CL=F": AssetRiskProfile(
        max_leverage=2.5, max_drawdown_pct=15.0, max_position_pct=20.0,
        fee_bps=10, session_type=SessionType.SESSION,
    ),
    "SPY": AssetRiskProfile(
        max_leverage=2.0, max_drawdown_pct=10.0, max_position_pct=30.0,
        fee_bps=5, session_type=SessionType.SESSION,
    ),
    "QQQ": AssetRiskProfile(
        max_leverage=2.0, max_drawdown_pct=10.0, max_position_pct=30.0,
        fee_bps=5, session_type=SessionType.SESSION,
    ),
    "AAPL": AssetRiskProfile(
        max_leverage=2.0, max_drawdown_pct=15.0, max_position_pct=15.0,
        fee_bps=5, session_type=SessionType.SESSION,
    ),
}


# =============================================================================
# Pre-built Asset Configs
# =============================================================================

ASSET_CONFIGS: dict[str, AssetConfig] = {
    "GC=F": AssetConfig(
        symbol="GC=F",
        correlated_assets=["BTC-USD", "CL=F", "SPY", "^IRX"],
        session_type=SessionType.SESSION,
        session_open_utc=22, session_close_utc=21,
        trading_days="Sun-Fri",
        arch_config=ModelConfig(cross_asset_channels=20),
    ),
    "BTC-USD": AssetConfig(
        symbol="BTC-USD",
        correlated_assets=["GC=F", "SPY", "^TNX"],
        session_type=SessionType.TWENTY_FOUR_SEVEN,
        trading_days="24/7",
        arch_config=ModelConfig(cross_asset_channels=15),
    ),
    "CL=F": AssetConfig(
        symbol="CL=F",
        correlated_assets=["GC=F", "SPY", "^TNX"],
        session_type=SessionType.SESSION,
        session_open_utc=23, session_close_utc=22,
        trading_days="Sun-Fri",
        arch_config=ModelConfig(cross_asset_channels=15),
    ),
    "SPY": AssetConfig(
        symbol="SPY",
        correlated_assets=["GC=F", "^TNX", "DX-Y.NYB"],
        session_type=SessionType.SESSION,
        session_open_utc=14, session_close_utc=21,
        trading_days="Mon-Fri",
        arch_config=ModelConfig(cross_asset_channels=15),
    ),
    "QQQ": AssetConfig(
        symbol="QQQ",
        correlated_assets=["GC=F", "^TNX", "DX-Y.NYB"],
        session_type=SessionType.SESSION,
        session_open_utc=14, session_close_utc=21,
        trading_days="Mon-Fri",
        arch_config=ModelConfig(cross_asset_channels=15),
    ),
    "AAPL": AssetConfig(
        symbol="AAPL",
        correlated_assets=["SPY", "QQQ", "^TNX"],
        session_type=SessionType.SESSION,
        session_open_utc=14, session_close_utc=21,
        trading_days="Mon-Fri",
        arch_config=ModelConfig(cross_asset_channels=15),
    ),
}


# =============================================================================
# Inference Result Models
# =============================================================================


class SignalPrediction(BaseModel):
    """Single prediction output from the model."""
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    position_size: float = Field(ge=0.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class CircuitBreakerState(BaseModel):
    """Current state of circuit breaker checks."""
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    triggered: list[str] = Field(default_factory=list)


# =============================================================================
# FRED Macro Series
# =============================================================================

FRED_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
}


# =============================================================================
# Paths
# =============================================================================

ML_DATA_DIR = Path("data/ml")
ML_MODELS_DIR = Path("models")
ML_LOGS_DIR = Path("logs/ml")
