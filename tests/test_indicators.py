"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from quantdash.features import (
    INDICATORS_CONFIG,
    compute_indicator,
    get_indicators_by_category,
    get_overlay_indicators,
    get_subchart_indicators,
)
from quantdash.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    stochastic_oscillator,
)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
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


class TestSimpleIndicators:
    """Test simple moving average indicators."""

    def test_sma_computes_correctly(self, sample_ohlcv):
        """SMA should compute rolling mean of close prices."""
        result = sma(sample_ohlcv, period=10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        # First 9 values should be NaN (not enough data)
        assert result.isna().sum() == 9
        # Manual check for 10th value
        expected = sample_ohlcv["close"].iloc[:10].mean()
        np.testing.assert_almost_equal(result.iloc[9], expected, decimal=5)

    def test_ema_computes_correctly(self, sample_ohlcv):
        """EMA should compute exponential moving average."""
        result = ema(sample_ohlcv, period=10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        # EMA starts from the first value
        assert not np.isnan(result.iloc[-1])


class TestMomentumIndicators:
    """Test momentum indicators."""

    def test_rsi_range(self, sample_ohlcv):
        """RSI should be bounded between 0 and 100."""
        result = rsi(sample_ohlcv, period=14)
        
        valid_values = result.dropna()
        assert all(valid_values >= 0)
        assert all(valid_values <= 100)

    def test_macd_returns_dict(self, sample_ohlcv):
        """MACD should return dict with macd_line, signal_line, histogram."""
        result = macd(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result
        assert isinstance(result["macd_line"], pd.Series)

    def test_stochastic_range(self, sample_ohlcv):
        """Stochastic should be bounded between 0 and 100."""
        result = stochastic_oscillator(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert "percent_k" in result
        assert "percent_d" in result
        
        k_valid = result["percent_k"].dropna()
        assert all(k_valid >= 0)
        assert all(k_valid <= 100)


class TestVolatilityIndicators:
    """Test volatility indicators."""

    def test_atr_positive(self, sample_ohlcv):
        """ATR should always be positive."""
        result = atr(sample_ohlcv, period=14)
        
        valid_values = result.dropna()
        assert all(valid_values >= 0)

    def test_bollinger_bands_returns_dict(self, sample_ohlcv):
        """Bollinger Bands should return upper, middle, lower."""
        result = bollinger_bands(sample_ohlcv)
        
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        
        # Upper should be above middle, middle above lower
        valid_idx = ~(result["upper"].isna() | result["lower"].isna())
        assert all(result["upper"][valid_idx] >= result["middle"][valid_idx])
        assert all(result["middle"][valid_idx] >= result["lower"][valid_idx])


class TestIndicatorsConfig:
    """Test indicator configuration registry."""

    def test_config_has_required_fields(self):
        """Each indicator config should have required fields."""
        required_fields = ["func", "params", "type", "category", "desc"]
        
        for name, config in INDICATORS_CONFIG.items():
            for field in required_fields:
                assert field in config, f"{name} missing {field}"

    def test_compute_indicator_factory(self, sample_ohlcv):
        """compute_indicator should work for all registered indicators."""
        for name in INDICATORS_CONFIG:
            try:
                result = compute_indicator(sample_ohlcv, name)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Failed to compute {name}: {e}")

    def test_get_indicators_by_category(self):
        """Should filter indicator names by category."""
        trend = get_indicators_by_category("Trend")
        assert len(trend) > 0
        assert isinstance(trend, list)
        # Verify all returned names have Trend category
        for name in trend:
            assert INDICATORS_CONFIG[name]["category"] == "Trend"

    def test_get_overlay_indicators(self):
        """Should return only overlay indicator names."""
        overlays = get_overlay_indicators()
        assert isinstance(overlays, list)
        for name in overlays:
            assert INDICATORS_CONFIG[name]["type"] == "overlay"

    def test_get_subchart_indicators(self):
        """Should return only subchart indicator names."""
        subcharts = get_subchart_indicators()
        assert isinstance(subcharts, list)
        for name in subcharts:
            assert INDICATORS_CONFIG[name]["type"] == "subchart"
