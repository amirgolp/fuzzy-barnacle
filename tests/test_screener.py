"""Unit tests for technical screener module."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from quantdash.features.screener import TechnicalScreener, screen_symbol, screen_portfolio
from quantdash.features.screener_models import (
    PortfolioScreeningResult,
    ScreenerConfig,
    ScreenerResult,
    SignalType,
    TechnicalSignal,
)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 250  # Enough for MA200

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000000, 5000000, n),
    }, index=dates)

    return df


@pytest.fixture
def oversold_ohlcv():
    """Generate data with oversold RSI condition."""
    np.random.seed(123)
    n = 100

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Create downtrend for oversold RSI
    close = 150 - np.arange(n) * 0.5 - np.abs(np.random.randn(n)) * 0.3

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.3,
        "high": close + abs(np.random.randn(n)) * 0.5,
        "low": close - abs(np.random.randn(n)) * 0.8,
        "close": close,
        "volume": np.random.randint(1000000, 5000000, n),
    }, index=dates)

    return df


@pytest.fixture
def bollinger_oversold_ohlcv():
    """Generate data where price is below lower Bollinger Band."""
    np.random.seed(77)
    n = 50

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Stable price then sharp drop at end
    close = np.full(n, 100.0)
    close[-3:] = [90.0, 85.0, 80.0]  # Sharp drop below lower band

    df = pd.DataFrame({
        "open": close + 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(1000000, 5000000, n),
    }, index=dates)

    return df


class TestTechnicalSignal:
    """Test TechnicalSignal model."""

    def test_signal_is_bullish(self):
        """Bullish signals should be identified correctly."""
        bullish_types = [
            SignalType.GOLDEN_CROSS, SignalType.RSI_OVERSOLD,
            SignalType.MACD_BULLISH, SignalType.BREAKOUT,
            SignalType.STOCH_OVERSOLD, SignalType.BOLLINGER_OVERSOLD,
            SignalType.MFI_OVERSOLD, SignalType.OBV_DIVERGENCE_BULL,
            SignalType.EMA_CROSSOVER_BULL,
        ]
        for sig_type in bullish_types:
            signal = TechnicalSignal(
                signal_type=sig_type,
                strength="strong",
                description="Test signal",
                date_detected=date.today(),
            )
            assert signal.is_bullish, f"{sig_type} should be bullish"
            assert not signal.is_bearish, f"{sig_type} should not be bearish"

    def test_signal_is_bearish(self):
        """Bearish signals should be identified correctly."""
        bearish_types = [
            SignalType.DEATH_CROSS, SignalType.RSI_OVERBOUGHT,
            SignalType.MACD_BEARISH, SignalType.BREAKDOWN,
            SignalType.STOCH_OVERBOUGHT, SignalType.BOLLINGER_OVERBOUGHT,
            SignalType.MFI_OVERBOUGHT, SignalType.OBV_DIVERGENCE_BEAR,
            SignalType.EMA_CROSSOVER_BEAR,
        ]
        for sig_type in bearish_types:
            signal = TechnicalSignal(
                signal_type=sig_type,
                strength="strong",
                description="Test signal",
                date_detected=date.today(),
            )
            assert signal.is_bearish, f"{sig_type} should be bearish"
            assert not signal.is_bullish, f"{sig_type} should not be bullish"

    def test_signal_score_value(self):
        """Score values should be calculated correctly."""
        for strength, expected in [("strong", 3), ("moderate", 2), ("weak", 1)]:
            signal = TechnicalSignal(
                signal_type=SignalType.RSI_OVERSOLD,
                strength=strength,
                description="Test",
                date_detected=date.today(),
            )
            assert signal.score_value == expected


class TestScreenerConfig:
    """Test ScreenerConfig validation."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ScreenerConfig()
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.ma_fast == 50
        assert config.ma_slow == 200
        assert config.ema_fast == 12
        assert config.ema_slow == 26
        assert config.mfi_oversold == 20.0
        assert config.mfi_overbought == 80.0

    def test_config_validation(self):
        """Config should validate constraints."""
        with pytest.raises(ValueError):
            ScreenerConfig(ma_fast=100, ma_slow=50)

    def test_ema_validation(self):
        """EMA slow must be > EMA fast."""
        with pytest.raises(ValueError):
            ScreenerConfig(ema_fast=30, ema_slow=10)


class TestScreenerResult:
    """Test ScreenerResult model."""

    def test_gauge_value_range(self):
        """Gauge value should be in -100 to +100 range."""
        result = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            bullish_score=9.0,
            bearish_score=0.0,
        )
        assert -100 <= result.gauge_value <= 100

    def test_gauge_label(self):
        """Gauge label should match value ranges."""
        strong_buy = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            bullish_score=20.0,
            bearish_score=0.0,
        )
        assert strong_buy.gauge_label == "Strong Buy"

        neutral = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            bullish_score=2.0,
            bearish_score=2.0,
        )
        assert neutral.gauge_label == "Neutral"

    def test_matches_criteria_strong(self):
        """One strong signal should match criteria."""
        result = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            signals=[
                TechnicalSignal(
                    signal_type=SignalType.RSI_OVERSOLD,
                    strength="strong",
                    description="Test",
                    date_detected=date.today(),
                )
            ],
        )
        assert result.matches_criteria

    def test_matches_criteria_moderate(self):
        """Two moderate signals should match criteria."""
        result = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            signals=[
                TechnicalSignal(
                    signal_type=SignalType.RSI_OVERSOLD,
                    strength="moderate",
                    description="Test",
                    date_detected=date.today(),
                ),
                TechnicalSignal(
                    signal_type=SignalType.MACD_BULLISH,
                    strength="moderate",
                    description="Test",
                    date_detected=date.today(),
                ),
            ],
        )
        assert result.matches_criteria

    def test_does_not_match_criteria(self):
        """One weak signal should not match criteria."""
        result = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
            signals=[
                TechnicalSignal(
                    signal_type=SignalType.RSI_OVERSOLD,
                    strength="weak",
                    description="Test",
                    date_detected=date.today(),
                )
            ],
        )
        assert not result.matches_criteria

    def test_rank_field(self):
        """Rank field should be settable."""
        result = ScreenerResult(
            symbol="TEST",
            screening_date=date.today(),
        )
        assert result.rank is None
        result.rank = 1
        assert result.rank == 1


class TestTechnicalScreener:
    """Test TechnicalScreener class."""

    def test_screener_returns_result(self, sample_ohlcv):
        """Screener should return a ScreenerResult."""
        screener = TechnicalScreener()
        result = screener.screen(sample_ohlcv, symbol="TEST")

        assert isinstance(result, ScreenerResult)
        assert result.symbol == "TEST"
        assert result.screening_date is not None

    def test_screener_with_custom_config(self, sample_ohlcv):
        """Screener should respect custom config."""
        config = ScreenerConfig(
            rsi_oversold=25.0,
            rsi_overbought=75.0,
            ma_fast=20,
            ma_slow=100,
        )
        screener = TechnicalScreener(config)
        result = screener.screen(sample_ohlcv)

        assert isinstance(result, ScreenerResult)

    def test_screener_detects_signals(self, sample_ohlcv):
        """Screener should be able to detect signals."""
        screener = TechnicalScreener()
        result = screener.screen(sample_ohlcv)

        assert isinstance(result.signals, list)
        assert isinstance(result.score, float)
        assert result.recommendation in ["strong_buy", "buy", "hold", "sell", "strong_sell"]

    def test_screener_recommendations(self, sample_ohlcv):
        """Recommendations should have valid confidence."""
        screener = TechnicalScreener()
        result = screener.screen(sample_ohlcv)

        assert 0.0 <= result.confidence <= 1.0

    def test_screener_notes_generation(self, sample_ohlcv):
        """Screener should generate notes."""
        screener = TechnicalScreener()
        result = screener.screen(sample_ohlcv)

        assert isinstance(result.notes, list)

    def test_screener_bollinger_detection(self, bollinger_oversold_ohlcv):
        """Screener should detect Bollinger oversold when price drops sharply."""
        screener = TechnicalScreener()
        result = screener.screen(bollinger_oversold_ohlcv, symbol="BB_TEST")

        bollinger_signals = [
            s for s in result.signals
            if s.signal_type == SignalType.BOLLINGER_OVERSOLD
        ]
        assert len(bollinger_signals) > 0, "Should detect Bollinger oversold on sharp drop"

    def test_screener_has_18_detectors(self):
        """Screener should have 18 detector methods."""
        screener = TechnicalScreener()
        detector_methods = [
            m for m in dir(screener) if m.startswith("_detect_")
        ]
        assert len(detector_methods) == 18


class TestPortfolioScreening:
    """Test portfolio-level screening."""

    def test_portfolio_screening(self, sample_ohlcv):
        """Portfolio screening should return ranked results."""
        # Create two datasets with different seeds
        np.random.seed(99)
        n = 250
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close2 = 50 + np.cumsum(np.random.randn(n) * 2)
        df2 = pd.DataFrame({
            "open": close2 + np.random.randn(n) * 0.5,
            "high": close2 + abs(np.random.randn(n)),
            "low": close2 - abs(np.random.randn(n)),
            "close": close2,
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=dates)

        portfolio = {"AAPL": sample_ohlcv, "TSLA": df2}
        screener = TechnicalScreener()
        result = screener.screen_portfolio(portfolio)

        assert isinstance(result, PortfolioScreeningResult)
        assert result.total_screened == 2
        assert len(result.results) == 2
        # Results should be ranked
        assert result.results[0].rank == 1
        assert result.results[1].rank == 2
        # Higher score should be rank 1
        assert result.results[0].score >= result.results[1].score

    def test_portfolio_top_picks(self, sample_ohlcv):
        """Top picks should only include matching symbols."""
        portfolio = {"AAPL": sample_ohlcv}
        result = screen_portfolio(portfolio)

        assert isinstance(result, PortfolioScreeningResult)
        assert isinstance(result.top_picks, list)
        assert isinstance(result.summary, str)

    def test_portfolio_empty(self):
        """Empty portfolio should return empty results."""
        screener = TechnicalScreener()
        result = screener.screen_portfolio({})

        assert result.total_screened == 0
        assert len(result.results) == 0
        assert result.tickers_matching == 0

    def test_portfolio_summary(self, sample_ohlcv):
        """Portfolio summary should be generated."""
        portfolio = {"AAPL": sample_ohlcv}
        result = screen_portfolio(portfolio)

        assert len(result.summary) > 0


class TestConvenienceFunction:
    """Test screen_symbol convenience function."""

    def test_screen_symbol_works(self, sample_ohlcv):
        """screen_symbol should work without explicit screener."""
        result = screen_symbol(sample_ohlcv, symbol="AAPL")

        assert isinstance(result, ScreenerResult)
        assert result.symbol == "AAPL"

    def test_screen_symbol_with_config(self, sample_ohlcv):
        """screen_symbol should accept custom config."""
        config = ScreenerConfig(rsi_oversold=25.0)
        result = screen_symbol(sample_ohlcv, symbol="AAPL", config=config)

        assert isinstance(result, ScreenerResult)
