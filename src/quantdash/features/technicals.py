"""
TradingView-style technical analysis summary.

Computes individual indicator values grouped into Oscillators and Moving Averages,
each with a Buy/Neutral/Sell action, and produces 3 gauges.
"""

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field

from quantdash.features.indicators import (
    bollinger_bands, ema, macd, money_flow_index, rsi, sma, stochastic_oscillator,
)

Action = Literal["Buy", "Sell", "Neutral"]


class IndicatorReading(BaseModel):
    name: str
    value: float
    action: Action


class CategorySummary(BaseModel):
    title: str
    indicators: list[IndicatorReading] = Field(default_factory=list)

    @computed_field
    @property
    def buy_count(self) -> int:
        return sum(1 for i in self.indicators if i.action == "Buy")

    @computed_field
    @property
    def sell_count(self) -> int:
        return sum(1 for i in self.indicators if i.action == "Sell")

    @computed_field
    @property
    def neutral_count(self) -> int:
        return sum(1 for i in self.indicators if i.action == "Neutral")

    @computed_field
    @property
    def recommendation(self) -> str:
        b, s, n = self.buy_count, self.sell_count, self.neutral_count
        diff = b - s
        total = b + s + n
        if total == 0:
            return "Neutral"
        ratio = diff / total
        if ratio > 0.5:
            return "Strong Buy"
        elif ratio > 0.1:
            return "Buy"
        elif ratio < -0.5:
            return "Strong Sell"
        elif ratio < -0.1:
            return "Sell"
        return "Neutral"

    @computed_field
    @property
    def gauge_value(self) -> float:
        """Maps to -100..+100."""
        b, s = self.buy_count, self.sell_count
        total = b + s + self.neutral_count
        if total == 0:
            return 0.0
        return ((b - s) / total) * 100


class TechnicalsResult(BaseModel):
    symbol: str
    date: str
    oscillators: CategorySummary
    moving_averages: CategorySummary

    @computed_field
    @property
    def summary(self) -> CategorySummary:
        all_indicators = self.oscillators.indicators + self.moving_averages.indicators
        return CategorySummary(title="Summary", indicators=all_indicators)


def compute_technicals(df: pd.DataFrame, symbol: str = "UNKNOWN") -> TechnicalsResult:
    """Compute TradingView-style technical analysis for OHLCV data."""
    close = df["close"]
    last = close.iloc[-1]

    oscillator_readings: list[IndicatorReading] = []
    ma_readings: list[IndicatorReading] = []

    # --- Oscillators ---

    # RSI (14)
    rsi_val = rsi(df, 14)
    v = _last(rsi_val)
    if not np.isnan(v):
        action: Action = "Neutral"
        if v < 30:
            action = "Buy"
        elif v > 70:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Relative Strength Index (14)", value=round(v, 2), action=action))

    # Stochastic %K (14, 3, 3)
    stoch = stochastic_oscillator(df, k_period=14, d_period=3)
    v = _last(stoch["percent_k"])
    if not np.isnan(v):
        action = "Neutral"
        if v < 20:
            action = "Buy"
        elif v > 80:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Stochastic %K (14, 3, 3)", value=round(v, 2), action=action))

    # CCI (20) - Commodity Channel Index
    cci_val = _cci(df, 20)
    v = _last(cci_val)
    if not np.isnan(v):
        action = "Neutral"
        if v < -100:
            action = "Buy"
        elif v > 100:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Commodity Channel Index (20)", value=round(v, 2), action=action))

    # ADX (14) - Average Directional Index
    adx_val = _adx(df, 14)
    v = _last(adx_val)
    if not np.isnan(v):
        action = "Neutral"
        if v > 25:
            action = "Buy"  # Strong trend
        oscillator_readings.append(IndicatorReading(name="Average Directional Index (14)", value=round(v, 2), action=action))

    # Awesome Oscillator
    ao_val = sma(df, 5, "close") - sma(df, 34, "close")
    v = _last(ao_val)
    if not np.isnan(v):
        prev = ao_val.iloc[-2] if len(ao_val) > 1 else 0
        action = "Buy" if v > 0 and v > prev else ("Sell" if v < 0 and v < prev else "Neutral")
        oscillator_readings.append(IndicatorReading(name="Awesome Oscillator", value=round(v, 2), action=action))

    # Momentum (10)
    mom = close - close.shift(10)
    v = _last(mom)
    if not np.isnan(v):
        action = "Buy" if v > 0 else ("Sell" if v < 0 else "Neutral")
        oscillator_readings.append(IndicatorReading(name="Momentum (10)", value=round(v, 2), action=action))

    # MACD Level (12, 26)
    macd_result = macd(df, fast_period=12, slow_period=26, signal_period=9)
    v = _last(macd_result["macd_line"])
    sig = _last(macd_result["signal_line"])
    if not np.isnan(v) and not np.isnan(sig):
        action = "Buy" if v > sig else ("Sell" if v < sig else "Neutral")
        oscillator_readings.append(IndicatorReading(name="MACD Level (12, 26)", value=round(v, 2), action=action))

    # Stochastic RSI Fast (3, 3, 14, 14)
    rsi_series = rsi(df, 14)
    stoch_rsi_k = ((rsi_series - rsi_series.rolling(14).min()) /
                   (rsi_series.rolling(14).max() - rsi_series.rolling(14).min()) * 100)
    stoch_rsi_k = stoch_rsi_k.rolling(3).mean()
    v = _last(stoch_rsi_k)
    if not np.isnan(v):
        action = "Neutral"
        if v < 20:
            action = "Buy"
        elif v > 80:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Stochastic RSI Fast (3, 3, 14, 14)", value=round(v, 2), action=action))

    # Williams %R (14)
    highest = df["high"].rolling(14).max()
    lowest = df["low"].rolling(14).min()
    wpr = ((highest - close) / (highest - lowest)) * -100
    v = _last(wpr)
    if not np.isnan(v):
        action = "Neutral"
        if v < -80:
            action = "Buy"
        elif v > -20:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Williams Percent Range (14)", value=round(v, 2), action=action))

    # Bull Bear Power
    ema13 = ema(df, 13)
    bull_power = df["high"] - ema13
    bear_power = df["low"] - ema13
    bbp = bull_power + bear_power
    v = _last(bbp)
    if not np.isnan(v):
        action = "Buy" if v > 0 else ("Sell" if v < 0 else "Neutral")
        oscillator_readings.append(IndicatorReading(name="Bull Bear Power", value=round(v, 2), action=action))

    # Ultimate Oscillator (7, 14, 28)
    uo = _ultimate_oscillator(df)
    v = _last(uo)
    if not np.isnan(v):
        action = "Neutral"
        if v < 30:
            action = "Buy"
        elif v > 70:
            action = "Sell"
        oscillator_readings.append(IndicatorReading(name="Ultimate Oscillator (7, 14, 28)", value=round(v, 2), action=action))

    # --- Moving Averages ---
    ma_periods = [10, 20, 30, 50, 100, 200]

    for p in ma_periods:
        # EMA
        ema_val = ema(df, p)
        v = _last(ema_val)
        if not np.isnan(v):
            action = "Buy" if last > v else "Sell"
            ma_readings.append(IndicatorReading(name=f"Exponential Moving Average ({p})", value=round(v, 2), action=action))

        # SMA
        sma_val = sma(df, p)
        v = _last(sma_val)
        if not np.isnan(v):
            action = "Buy" if last > v else "Sell"
            ma_readings.append(IndicatorReading(name=f"Simple Moving Average ({p})", value=round(v, 2), action=action))

    # Ichimoku Base Line (9, 26, 52, 26)
    high9 = df["high"].rolling(9).max()
    low9 = df["low"].rolling(9).min()
    high26 = df["high"].rolling(26).max()
    low26 = df["low"].rolling(26).min()
    base_line = (high26 + low26) / 2
    v = _last(base_line)
    if not np.isnan(v):
        action = "Buy" if last > v else ("Sell" if last < v else "Neutral")
        ma_readings.append(IndicatorReading(name="Ichimoku Base Line (9, 26, 52, 26)", value=round(v, 2), action=action))

    # VWAP (20)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap_val = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    v = _last(vwap_val)
    if not np.isnan(v):
        action = "Buy" if last > v else "Sell"
        ma_readings.append(IndicatorReading(name="Volume Weighted Moving Average (20)", value=round(v, 2), action=action))

    # Hull MA (9)
    hma = _hull_ma(df, 9)
    v = _last(hma)
    if not np.isnan(v):
        action = "Buy" if last > v else "Sell"
        ma_readings.append(IndicatorReading(name="Hull Moving Average (9)", value=round(v, 2), action=action))

    return TechnicalsResult(
        symbol=symbol,
        date=str(date.today()),
        oscillators=CategorySummary(title="Oscillators", indicators=oscillator_readings),
        moving_averages=CategorySummary(title="Moving Averages", indicators=ma_readings),
    )


# --- Helper functions ---

def _last(series: pd.Series) -> float:
    v = series.iloc[-1] if len(series) > 0 else np.nan
    return float(v) if not pd.isna(v) else np.nan


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # When plus_dm <= minus_dm, set plus_dm to 0 and vice versa
    mask = plus_dm <= minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def _ultimate_oscillator(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7


def _hull_ma(df: pd.DataFrame, period: int = 9) -> pd.Series:
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = df["close"].rolling(half).mean()  # Simplified; true HMA uses WMA
    wma_full = df["close"].rolling(period).mean()
    hull_raw = 2 * wma_half - wma_full
    return hull_raw.rolling(sqrt_p).mean()
