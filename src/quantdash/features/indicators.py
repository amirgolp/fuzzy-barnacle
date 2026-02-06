"""
Technical indicators library with comprehensive INDICATORS_CONFIG registry.

Includes value-based and volume-based indicators per specification.
"""

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# VALUE-BASED INDICATORS - Trend
# =============================================================================


def sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """
    Simple Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for the moving average
        column: Column to calculate SMA on

    Returns:
        Series with SMA values
    """
    return df[column].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for the EMA
        column: Column to calculate EMA on

    Returns:
        Series with EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()


def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """
    Parabolic Stop and Reverse (SAR).

    Args:
        df: DataFrame with high, low, close columns
        af_start: Starting acceleration factor
        af_step: Acceleration factor increment
        af_max: Maximum acceleration factor

    Returns:
        Series with SAR values
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    sar = np.zeros(n)
    trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
    ep = np.zeros(n)  # Extreme point
    af = np.full(n, af_start)

    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    trend[0] = 1

    for i in range(1, n):
        if trend[i - 1] == 1:  # Uptrend
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])

            if low[i] < sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
        else:  # Downtrend
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])

            if high[i] > sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

    return pd.Series(sar, index=df.index, name="parabolic_sar")


def ichimoku_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict[str, pd.Series]:
    """
    Ichimoku Cloud indicator.

    Returns dict with: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A) - shifted forward by kijun_period
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B) - shifted forward by kijun_period
    senkou_b_high = high.rolling(window=senkou_b_period).max()
    senkou_b_low = low.rolling(window=senkou_b_period).min()
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span) - shifted back by kijun_period
    chikou_span = close.shift(-kijun_period)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


# =============================================================================
# VALUE-BASED INDICATORS - Momentum
# =============================================================================


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """
    Relative Strength Index.

    Args:
        df: DataFrame with price data
        period: RSI period
        column: Column to calculate RSI on

    Returns:
        Series with RSI values (0-100)
    """
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values.rename("rsi")


def macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "close",
) -> dict[str, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Returns dict with: macd_line, signal_line, histogram
    """
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd_line": macd_line.rename("macd"),
        "signal_line": signal_line.rename("macd_signal"),
        "histogram": histogram.rename("macd_histogram"),
    }


def stochastic_oscillator(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> dict[str, pd.Series]:
    """
    Stochastic Oscillator.

    Returns dict with: percent_k, percent_d
    """
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()

    # Fast %K
    fast_k = 100 * (df["close"] - low_min) / (high_max - low_min)

    # Slow %K (smoothed)
    percent_k = fast_k.rolling(window=smooth_k).mean()

    # %D (signal line)
    percent_d = percent_k.rolling(window=d_period).mean()

    return {
        "percent_k": percent_k.rename("stoch_k"),
        "percent_d": percent_d.rename("stoch_d"),
    }


# =============================================================================
# VALUE-BASED INDICATORS - Volatility
# =============================================================================


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Args:
        df: DataFrame with high, low, close columns
        period: ATR period

    Returns:
        Series with ATR values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = true_range.ewm(span=period, adjust=False).mean()

    return atr_values.rename("atr")


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
) -> dict[str, pd.Series]:
    """
    Bollinger Bands.

    Returns dict with: upper, middle, lower
    """
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return {
        "upper": upper.rename("bb_upper"),
        "middle": middle.rename("bb_middle"),
        "lower": lower.rename("bb_lower"),
    }


# =============================================================================
# VALUE-BASED INDICATORS - Support/Resistance
# =============================================================================


def fibonacci_retracement(
    df: pd.DataFrame,
    lookback: int = 100,
) -> dict[str, float]:
    """
    Fibonacci Retracement levels based on recent high/low.

    Returns dict with: level_0, level_236, level_382, level_500, level_618, level_100
    """
    recent = df.tail(lookback)
    high = recent["high"].max()
    low = recent["low"].min()
    diff = high - low

    return {
        "level_0": high,
        "level_236": high - (0.236 * diff),
        "level_382": high - (0.382 * diff),
        "level_500": high - (0.500 * diff),
        "level_618": high - (0.618 * diff),
        "level_100": low,
    }


def pivot_points(
    df: pd.DataFrame,
    method: str = "standard",
) -> dict[str, pd.Series]:
    """
    Pivot Points (Standard, Fibonacci, Camarilla).

    Uses previous bar's high, low, close to calculate current pivot levels.

    Returns dict with: pivot, r1, r2, r3, s1, s2, s3
    """
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    close = df["close"].shift(1)

    pivot = (high + low + close) / 3

    if method == "standard":
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

    elif method == "fibonacci":
        diff = high - low
        r1 = pivot + 0.382 * diff
        s1 = pivot - 0.382 * diff
        r2 = pivot + 0.618 * diff
        s2 = pivot - 0.618 * diff
        r3 = pivot + 1.000 * diff
        s3 = pivot - 1.000 * diff

    elif method == "camarilla":
        diff = high - low
        r1 = close + diff * 1.1 / 12
        s1 = close - diff * 1.1 / 12
        r2 = close + diff * 1.1 / 6
        s2 = close - diff * 1.1 / 6
        r3 = close + diff * 1.1 / 4
        s3 = close - diff * 1.1 / 4

    else:
        raise ValueError(f"Unknown pivot method: {method}")

    return {
        "pivot": pivot.rename("pivot"),
        "r1": r1.rename("r1"),
        "r2": r2.rename("r2"),
        "r3": r3.rename("r3"),
        "s1": s1.rename("s1"),
        "s2": s2.rename("s2"),
        "s3": s3.rename("s3"),
    }


# =============================================================================
# VOLUME-BASED INDICATORS
# =============================================================================


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    Cumulative volume that adds on up days and subtracts on down days.
    """
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    obv_values = (direction * df["volume"]).cumsum()

    return obv_values.rename("obv")


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.

    Typically calculated intraday, resets each session.
    For daily data, provides cumulative VWAP.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_values = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    return vwap_values.rename("vwap")


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Accumulation/Distribution Line.

    Measures cumulative money flow based on close location within the range.
    """
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    clv = clv.fillna(0)  # Handle zero-range bars
    ad_line = (clv * df["volume"]).cumsum()

    return ad_line.rename("ad_line")


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow.

    Measures money flow volume over a specified period.
    """
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    clv = clv.fillna(0)
    mf_volume = clv * df["volume"]

    cmf = mf_volume.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()

    return cmf.rename("cmf")


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index.

    Volume-weighted RSI that incorporates volume into momentum analysis.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0)
    negative_flow = raw_money_flow.where(direction < 0, 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi.rename("mfi")


# =============================================================================
# ADDITIONAL UTILITY INDICATORS
# =============================================================================


def returns(df: pd.DataFrame, periods: int = 1, log: bool = False) -> pd.Series:
    """Calculate simple or log returns."""
    if log:
        return np.log(df["close"] / df["close"].shift(periods)).rename("log_returns")
    return df["close"].pct_change(periods).rename("returns")


def rolling_volatility(df: pd.DataFrame, period: int = 20, annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns)."""
    ret = df["close"].pct_change()
    vol = ret.rolling(window=period).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename("volatility")


def rolling_high(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling highest high."""
    return df["high"].rolling(window=period).max().rename("rolling_high")


def rolling_low(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling lowest low."""
    return df["low"].rolling(window=period).min().rename("rolling_low")


def breakout_signal(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Breakout signal: 1 if close > rolling high, -1 if close < rolling low, else 0.
    """
    r_high = df["high"].rolling(window=period).max().shift(1)
    r_low = df["low"].rolling(window=period).min().shift(1)

    signal = pd.Series(0, index=df.index)
    signal[df["close"] > r_high] = 1
    signal[df["close"] < r_low] = -1

    return signal.rename("breakout")


# =============================================================================
# INDICATORS CONFIG REGISTRY
# =============================================================================


IndicatorFunc = Callable[[pd.DataFrame], pd.Series | dict[str, pd.Series]]


INDICATORS_CONFIG: dict[str, dict[str, Any]] = {
    # =========================================================================
    # VALUE-BASED - TREND
    # =========================================================================
    "SMA": {
        "func": sma,
        "params": {"period": 20},
        "type": "overlay",
        "category": "Trend",
        "desc": "Simple Moving Average",
        "color": "#2962FF",
    },
    "EMA": {
        "func": ema,
        "params": {"period": 20},
        "type": "overlay",
        "category": "Trend",
        "desc": "Exponential Moving Average",
        "color": "#FF6D00",
    },
    "Parabolic SAR": {
        "func": parabolic_sar,
        "params": {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2},
        "type": "overlay",
        "category": "Trend",
        "desc": "Parabolic Stop and Reverse",
        "color": "#AB47BC",
    },
    "Ichimoku Cloud": {
        "func": ichimoku_cloud,
        "params": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52},
        "type": "overlay",
        "category": "Trend",
        "desc": "Ichimoku Kinko Hyo (Cloud)",
        "color": "#26C6DA",
        "multi_output": True,
    },
    # =========================================================================
    # VALUE-BASED - MOMENTUM
    # =========================================================================
    "RSI": {
        "func": rsi,
        "params": {"period": 14},
        "type": "subchart",
        "category": "Momentum",
        "desc": "Relative Strength Index",
        "color": "#7B1FA2",
        "overbought": 70,
        "oversold": 30,
    },
    "MACD": {
        "func": macd,
        "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "type": "subchart",
        "category": "Momentum",
        "desc": "Moving Average Convergence Divergence",
        "color": "#2196F3",
        "multi_output": True,
    },
    "Stochastic Oscillator": {
        "func": stochastic_oscillator,
        "params": {"k_period": 14, "d_period": 3, "smooth_k": 3},
        "type": "subchart",
        "category": "Momentum",
        "desc": "Stochastic %K and %D",
        "color": "#009688",
        "multi_output": True,
        "overbought": 80,
        "oversold": 20,
    },
    # =========================================================================
    # VALUE-BASED - VOLATILITY
    # =========================================================================
    "ATR": {
        "func": atr,
        "params": {"period": 14},
        "type": "subchart",
        "category": "Volatility",
        "desc": "Average True Range",
        "color": "#F44336",
    },
    "Bollinger Bands": {
        "func": bollinger_bands,
        "params": {"period": 20, "std_dev": 2.0},
        "type": "overlay",
        "category": "Volatility",
        "desc": "Bollinger Bands (Upper, Middle, Lower)",
        "color": "#9C27B0",
        "multi_output": True,
    },
    # =========================================================================
    # VALUE-BASED - SUPPORT/RESISTANCE
    # =========================================================================
    "Fibonacci Retracement": {
        "func": fibonacci_retracement,
        "params": {"lookback": 100},
        "type": "overlay",
        "category": "Trend",
        "desc": "Fibonacci Retracement Levels",
        "color": "#FFD700",
        "levels_output": True,
    },
    "Pivot Points": {
        "func": pivot_points,
        "params": {"method": "standard"},
        "type": "overlay",
        "category": "Trend",
        "desc": "Pivot Points (Standard/Fibonacci/Camarilla)",
        "color": "#00BCD4",
        "multi_output": True,
    },
    # =========================================================================
    # VOLUME-BASED
    # =========================================================================
    "OBV": {
        "func": obv,
        "params": {},
        "type": "subchart",
        "category": "Volume",
        "desc": "On-Balance Volume",
        "color": "#4CAF50",
    },
    "VWAP": {
        "func": vwap,
        "params": {},
        "type": "overlay",
        "category": "Volume",
        "desc": "Volume Weighted Average Price",
        "color": "#E91E63",
    },
    "Accumulation/Distribution": {
        "func": accumulation_distribution,
        "params": {},
        "type": "subchart",
        "category": "Volume",
        "desc": "Accumulation/Distribution Line",
        "color": "#3F51B5",
    },
    "CMF": {
        "func": chaikin_money_flow,
        "params": {"period": 20},
        "type": "subchart",
        "category": "Volume",
        "desc": "Chaikin Money Flow",
        "color": "#00796B",
    },
    "MFI": {
        "func": money_flow_index,
        "params": {"period": 14},
        "type": "subchart",
        "category": "Volume",
        "desc": "Money Flow Index",
        "color": "#795548",
        "overbought": 80,
        "oversold": 20,
    },
    # =========================================================================
    # UTILITY
    # =========================================================================
    "Volatility": {
        "func": rolling_volatility,
        "params": {"period": 20, "annualize": True},
        "type": "subchart",
        "category": "Volatility",
        "desc": "Rolling Volatility (Annualized)",
        "color": "#FF5722",
    },
}


def compute_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    params: Optional[dict[str, Any]] = None,
) -> pd.Series | dict[str, pd.Series]:
    """
    Compute an indicator by name with optional parameter overrides.

    Args:
        df: OHLCV DataFrame
        indicator_name: Name from INDICATORS_CONFIG
        params: Optional parameter overrides

    Returns:
        Series or dict of Series depending on indicator
    """
    if indicator_name not in INDICATORS_CONFIG:
        raise ValueError(f"Unknown indicator: {indicator_name}")

    config = INDICATORS_CONFIG[indicator_name]
    func = config["func"]

    # Merge default params with overrides
    final_params = {**config["params"]}
    if params:
        final_params.update(params)

    return func(df, **final_params)


def get_indicators_by_category(category: str) -> list[str]:
    """Get indicator names filtered by category."""
    return [
        name
        for name, config in INDICATORS_CONFIG.items()
        if config["category"] == category
    ]


def get_overlay_indicators() -> list[str]:
    """Get indicators that render on price chart."""
    return [
        name
        for name, config in INDICATORS_CONFIG.items()
        if config["type"] == "overlay"
    ]


def get_subchart_indicators() -> list[str]:
    """Get indicators that render in separate panel."""
    return [
        name
        for name, config in INDICATORS_CONFIG.items()
        if config["type"] == "subchart"
    ]
