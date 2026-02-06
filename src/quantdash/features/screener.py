"""
Technical Screener Engine for QuantDash.

Implements multi-indicator technical screening with signal aggregation,
composite scoring, and recommendation generation.

Inspired by Finance-Guru's TechnicalScreener.
"""

from datetime import date
from typing import Optional

import pandas as pd

from quantdash.features.indicators import (
    bollinger_bands,
    ema,
    macd,
    money_flow_index,
    obv,
    rsi,
    sma,
    stochastic_oscillator,
)
from quantdash.features.screener_models import (
    PortfolioScreeningResult,
    Recommendation,
    ScreenerConfig,
    ScreenerResult,
    SignalType,
    TechnicalSignal,
)


class TechnicalScreener:
    """
    Multi-indicator technical screening engine.

    Detects technical signals, calculates composite scores,
    and generates buy/sell recommendations with confidence levels.

    Usage:
        screener = TechnicalScreener()
        result = screener.screen(df, symbol="TSLA")
        print(f"Recommendation: {result.recommendation}")
        print(f"Gauge: {result.gauge_value}")
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or ScreenerConfig()

    def screen(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> ScreenerResult:
        """
        Screen a symbol's data for technical signals.

        Args:
            df: OHLCV DataFrame with lowercase columns (open, high, low, close, volume)
            symbol: Symbol name for result labeling

        Returns:
            ScreenerResult with signals, score, and recommendation
        """
        signals: list[TechnicalSignal] = []

        detectors = [
            self._detect_golden_cross,
            self._detect_death_cross,
            self._detect_rsi_oversold,
            self._detect_rsi_overbought,
            self._detect_macd_bullish,
            self._detect_macd_bearish,
            self._detect_stoch_oversold,
            self._detect_stoch_overbought,
            self._detect_breakout,
            self._detect_breakdown,
            self._detect_bollinger_oversold,
            self._detect_bollinger_overbought,
            self._detect_mfi_oversold,
            self._detect_mfi_overbought,
            self._detect_obv_divergence_bull,
            self._detect_obv_divergence_bear,
            self._detect_ema_crossover_bull,
            self._detect_ema_crossover_bear,
            self._detect_price_above_ema,
            self._detect_price_below_ema,
            self._detect_volume_spike,
        ]

        for detector in detectors:
            try:
                signal = detector(df)
                if signal:
                    signals.append(signal)
            except Exception:
                continue

        # Calculate scores
        bullish_score = sum(s.score_value for s in signals if s.is_bullish)
        bearish_score = sum(s.score_value for s in signals if s.is_bearish)
        total_score = (bullish_score + bearish_score) * self.config.pattern_weight

        recommendation, confidence = self._generate_recommendation(
            signals, bullish_score, bearish_score
        )

        current_price = float(df["close"].iloc[-1]) if len(df) > 0 else None
        current_rsi = self._get_current_rsi(df)
        notes = self._generate_notes(signals)

        screening_date = (
            df.index[-1].date()
            if hasattr(df.index[-1], "date")
            else date.today()
        )

        return ScreenerResult(
            symbol=symbol,
            screening_date=screening_date,
            signals=signals,
            score=total_score,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            recommendation=recommendation,
            confidence=confidence,
            current_price=current_price,
            current_rsi=current_rsi,
            notes=notes,
        )

    def screen_portfolio(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> PortfolioScreeningResult:
        """
        Screen multiple symbols and rank results.

        Args:
            portfolio_data: Dict mapping symbol -> OHLCV DataFrame

        Returns:
            PortfolioScreeningResult with ranked results and top picks
        """
        results: list[ScreenerResult] = []

        for symbol, df in portfolio_data.items():
            try:
                result = self.screen(df, symbol=symbol)
                results.append(result)
            except Exception:
                continue

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Assign ranks
        for rank, result in enumerate(results, start=1):
            result.rank = rank

        matching_count = sum(1 for r in results if r.matches_criteria)
        top_picks = [r.symbol for r in results[:5] if r.matches_criteria]

        # Generate summary
        total = len(results)
        if matching_count == 0:
            summary = f"Screened {total} symbols. None met criteria."
        elif results:
            top = results[0]
            summary = (
                f"Found {matching_count} of {total} symbols meeting criteria. "
                f"Top pick: {top.symbol} (score {top.score:.1f}, "
                f"{top.signal_count} signals)"
            )
        else:
            summary = f"Screened {total} symbols."

        return PortfolioScreeningResult(
            screening_date=date.today(),
            total_screened=total,
            tickers_matching=matching_count,
            results=results,
            top_picks=top_picks,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # Crossover detectors
    # -------------------------------------------------------------------------

    def _detect_golden_cross(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect golden cross (fast MA crosses above slow MA)."""
        if len(df) < self.config.ma_slow:
            return None

        ma_fast = sma(df, period=self.config.ma_fast)
        ma_slow = sma(df, period=self.config.ma_slow)

        window = self.config.crossover_window
        recent_fast = ma_fast.iloc[-window:]
        recent_slow = ma_slow.iloc[-window:]

        for i in range(1, len(recent_fast)):
            prev_below = recent_fast.iloc[i - 1] < recent_slow.iloc[i - 1]
            now_above = recent_fast.iloc[i] > recent_slow.iloc[i]

            if prev_below and now_above:
                separation = abs(recent_fast.iloc[i] - recent_slow.iloc[i])
                separation_pct = separation / df["close"].mean()

                strength = (
                    "strong" if separation_pct > 0.05
                    else "moderate" if separation_pct > 0.02
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.GOLDEN_CROSS,
                    strength=strength,
                    description=f"Golden Cross: {self.config.ma_fast}MA crossed above {self.config.ma_slow}MA",
                    date_detected=df.index[-window + i].date() if hasattr(df.index[-window + i], "date") else date.today(),
                    value=separation_pct,
                )

        return None

    def _detect_death_cross(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect death cross (fast MA crosses below slow MA)."""
        if len(df) < self.config.ma_slow:
            return None

        ma_fast = sma(df, period=self.config.ma_fast)
        ma_slow = sma(df, period=self.config.ma_slow)

        window = self.config.crossover_window
        recent_fast = ma_fast.iloc[-window:]
        recent_slow = ma_slow.iloc[-window:]

        for i in range(1, len(recent_fast)):
            prev_above = recent_fast.iloc[i - 1] > recent_slow.iloc[i - 1]
            now_below = recent_fast.iloc[i] < recent_slow.iloc[i]

            if prev_above and now_below:
                separation = abs(recent_fast.iloc[i] - recent_slow.iloc[i])
                separation_pct = separation / df["close"].mean()

                strength = (
                    "strong" if separation_pct > 0.05
                    else "moderate" if separation_pct > 0.02
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.DEATH_CROSS,
                    strength=strength,
                    description=f"Death Cross: {self.config.ma_fast}MA crossed below {self.config.ma_slow}MA",
                    date_detected=df.index[-window + i].date() if hasattr(df.index[-window + i], "date") else date.today(),
                    value=separation_pct,
                )

        return None

    def _detect_ema_crossover_bull(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect bullish EMA crossover (fast EMA crosses above slow EMA)."""
        if len(df) < self.config.ema_slow:
            return None

        ema_fast = ema(df, period=self.config.ema_fast)
        ema_slow = ema(df, period=self.config.ema_slow)

        window = self.config.crossover_window
        recent_fast = ema_fast.iloc[-window:]
        recent_slow = ema_slow.iloc[-window:]

        for i in range(1, len(recent_fast)):
            prev_below = recent_fast.iloc[i - 1] < recent_slow.iloc[i - 1]
            now_above = recent_fast.iloc[i] > recent_slow.iloc[i]

            if prev_below and now_above:
                separation = abs(recent_fast.iloc[i] - recent_slow.iloc[i])
                separation_pct = separation / df["close"].mean()

                strength = (
                    "strong" if separation_pct > 0.03
                    else "moderate" if separation_pct > 0.01
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.EMA_CROSSOVER_BULL,
                    strength=strength,
                    description=f"EMA Bullish: {self.config.ema_fast}EMA crossed above {self.config.ema_slow}EMA",
                    date_detected=df.index[-window + i].date() if hasattr(df.index[-window + i], "date") else date.today(),
                    value=separation_pct,
                )

        return None

    def _detect_ema_crossover_bear(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect bearish EMA crossover (fast EMA crosses below slow EMA)."""
        if len(df) < self.config.ema_slow:
            return None

        ema_fast = ema(df, period=self.config.ema_fast)
        ema_slow = ema(df, period=self.config.ema_slow)

        window = self.config.crossover_window
        recent_fast = ema_fast.iloc[-window:]
        recent_slow = ema_slow.iloc[-window:]

        for i in range(1, len(recent_fast)):
            prev_above = recent_fast.iloc[i - 1] > recent_slow.iloc[i - 1]
            now_below = recent_fast.iloc[i] < recent_slow.iloc[i]

            if prev_above and now_below:
                separation = abs(recent_fast.iloc[i] - recent_slow.iloc[i])
                separation_pct = separation / df["close"].mean()

                strength = (
                    "strong" if separation_pct > 0.03
                    else "moderate" if separation_pct > 0.01
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.EMA_CROSSOVER_BEAR,
                    strength=strength,
                    description=f"EMA Bearish: {self.config.ema_fast}EMA crossed below {self.config.ema_slow}EMA",
                    date_detected=df.index[-window + i].date() if hasattr(df.index[-window + i], "date") else date.today(),
                    value=separation_pct,
                )

        return None

    def _detect_price_above_ema(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price trading above slow EMA (trend confirmation)."""
        if len(df) < self.config.ema_slow:
            return None

        ema_slow = ema(df, period=self.config.ema_slow)
        current_price = df["close"].iloc[-1]
        current_ema = ema_slow.iloc[-1]

        if pd.isna(current_ema):
            return None

        if current_price > current_ema:
            diff_pct = (current_price - current_ema) / current_ema

            strength = (
                "strong" if diff_pct > 0.05
                else "moderate" if diff_pct > 0.02
                else "weak"
            )

            return TechnicalSignal(
                signal_type=SignalType.PRICE_ABOVE_EMA,
                strength=strength,
                description=f"Price above {self.config.ema_slow}EMA (Uptrend)",
                date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                value=diff_pct,
            )

        return None

    def _detect_price_below_ema(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price trading below slow EMA (downtrend confirmation)."""
        if len(df) < self.config.ema_slow:
            return None

        ema_slow = ema(df, period=self.config.ema_slow)
        current_price = df["close"].iloc[-1]
        current_ema = ema_slow.iloc[-1]

        if pd.isna(current_ema):
            return None

        if current_price < current_ema:
            diff_pct = (current_ema - current_price) / current_ema

            strength = (
                "strong" if diff_pct > 0.05
                else "moderate" if diff_pct > 0.02
                else "weak"
            )

            return TechnicalSignal(
                signal_type=SignalType.PRICE_BELOW_EMA,
                strength=strength,
                description=f"Price below {self.config.ema_slow}EMA (Downtrend)",
                date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                value=diff_pct,
            )

        return None

    # -------------------------------------------------------------------------
    # Momentum detectors
    # -------------------------------------------------------------------------

    def _detect_rsi_oversold(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect RSI oversold condition."""
        if len(df) < 14:
            return None

        rsi_series = rsi(df, period=14)
        current_rsi = rsi_series.iloc[-1]

        if pd.isna(current_rsi) or current_rsi >= self.config.rsi_oversold:
            return None

        strength = (
            "strong" if current_rsi < 20
            else "moderate" if current_rsi < 25
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.RSI_OVERSOLD,
            strength=strength,
            description=f"RSI Oversold: RSI at {current_rsi:.1f} (below {self.config.rsi_oversold})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=current_rsi,
        )

    def _detect_rsi_overbought(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect RSI overbought condition."""
        if len(df) < 14:
            return None

        rsi_series = rsi(df, period=14)
        current_rsi = rsi_series.iloc[-1]

        if pd.isna(current_rsi) or current_rsi <= self.config.rsi_overbought:
            return None

        strength = (
            "strong" if current_rsi > 80
            else "moderate" if current_rsi > 75
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.RSI_OVERBOUGHT,
            strength=strength,
            description=f"RSI Overbought: RSI at {current_rsi:.1f} (above {self.config.rsi_overbought})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=current_rsi,
        )

    def _detect_macd_bullish(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect MACD bullish crossover."""
        if len(df) < 26:
            return None

        macd_result = macd(df)
        macd_line = macd_result["macd_line"]
        signal_line = macd_result["signal_line"]
        histogram = macd_result["histogram"]

        window = self.config.crossover_window
        for i in range(-window, -1):
            if i - 1 < -len(macd_line):
                continue
            prev_below = macd_line.iloc[i - 1] < signal_line.iloc[i - 1]
            now_above = macd_line.iloc[i] > signal_line.iloc[i]

            if prev_below and now_above:
                hist_value = abs(histogram.iloc[-1])
                strength = (
                    "strong" if hist_value > 0.5
                    else "moderate" if hist_value > 0.2
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.MACD_BULLISH,
                    strength=strength,
                    description="MACD Bullish: MACD line crossed above signal line",
                    date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                    value=float(macd_line.iloc[-1]),
                )

        return None

    def _detect_macd_bearish(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect MACD bearish crossover."""
        if len(df) < 26:
            return None

        macd_result = macd(df)
        macd_line = macd_result["macd_line"]
        signal_line = macd_result["signal_line"]
        histogram = macd_result["histogram"]

        window = self.config.crossover_window
        for i in range(-window, -1):
            if i - 1 < -len(macd_line):
                continue
            prev_above = macd_line.iloc[i - 1] > signal_line.iloc[i - 1]
            now_below = macd_line.iloc[i] < signal_line.iloc[i]

            if prev_above and now_below:
                hist_value = abs(histogram.iloc[-1])
                strength = (
                    "strong" if hist_value > 0.5
                    else "moderate" if hist_value > 0.2
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.MACD_BEARISH,
                    strength=strength,
                    description="MACD Bearish: MACD line crossed below signal line",
                    date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                    value=float(macd_line.iloc[-1]),
                )

        return None

    def _detect_stoch_oversold(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect Stochastic oversold condition."""
        if len(df) < 14:
            return None

        stoch = stochastic_oscillator(df)
        k_value = stoch["percent_k"].iloc[-1]

        if pd.isna(k_value) or k_value >= self.config.stoch_oversold:
            return None

        strength = (
            "strong" if k_value < 10
            else "moderate" if k_value < 15
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.STOCH_OVERSOLD,
            strength=strength,
            description=f"Stochastic Oversold: %K at {k_value:.1f} (below {self.config.stoch_oversold})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=k_value,
        )

    def _detect_stoch_overbought(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect Stochastic overbought condition."""
        if len(df) < 14:
            return None

        stoch = stochastic_oscillator(df)
        k_value = stoch["percent_k"].iloc[-1]

        if pd.isna(k_value) or k_value <= self.config.stoch_overbought:
            return None

        strength = (
            "strong" if k_value > 90
            else "moderate" if k_value > 85
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.STOCH_OVERBOUGHT,
            strength=strength,
            description=f"Stochastic Overbought: %K at {k_value:.1f} (above {self.config.stoch_overbought})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=k_value,
        )

    def _detect_mfi_oversold(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect Money Flow Index oversold condition."""
        if len(df) < 14 or "volume" not in df.columns:
            return None

        mfi_series = money_flow_index(df, period=14)
        current_mfi = mfi_series.iloc[-1]

        if pd.isna(current_mfi) or current_mfi >= self.config.mfi_oversold:
            return None

        strength = (
            "strong" if current_mfi < 10
            else "moderate" if current_mfi < 15
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.MFI_OVERSOLD,
            strength=strength,
            description=f"MFI Oversold: MFI at {current_mfi:.1f} (below {self.config.mfi_oversold})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=current_mfi,
        )

    def _detect_mfi_overbought(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect Money Flow Index overbought condition."""
        if len(df) < 14 or "volume" not in df.columns:
            return None

        mfi_series = money_flow_index(df, period=14)
        current_mfi = mfi_series.iloc[-1]

        if pd.isna(current_mfi) or current_mfi <= self.config.mfi_overbought:
            return None

        strength = (
            "strong" if current_mfi > 90
            else "moderate" if current_mfi > 85
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.MFI_OVERBOUGHT,
            strength=strength,
            description=f"MFI Overbought: MFI at {current_mfi:.1f} (above {self.config.mfi_overbought})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=current_mfi,
        )

    # -------------------------------------------------------------------------
    # Volatility detectors
    # -------------------------------------------------------------------------

    def _detect_bollinger_oversold(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price at or below lower Bollinger Band."""
        if len(df) < 20:
            return None

        bands = bollinger_bands(df)
        current_price = df["close"].iloc[-1]
        lower = bands["lower"].iloc[-1]
        upper = bands["upper"].iloc[-1]

        if pd.isna(lower) or pd.isna(upper):
            return None

        band_width = upper - lower
        if band_width <= 0:
            return None

        # %B: position of price relative to bands (0 = lower, 1 = upper)
        pct_b = (current_price - lower) / band_width

        if pct_b >= 0.2:
            return None

        strength = (
            "strong" if pct_b < 0.0
            else "moderate" if pct_b < 0.1
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.BOLLINGER_OVERSOLD,
            strength=strength,
            description=f"Bollinger Oversold: Price near lower band (%B: {pct_b:.2f})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=pct_b,
        )

    def _detect_bollinger_overbought(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price at or above upper Bollinger Band."""
        if len(df) < 20:
            return None

        bands = bollinger_bands(df)
        current_price = df["close"].iloc[-1]
        lower = bands["lower"].iloc[-1]
        upper = bands["upper"].iloc[-1]

        if pd.isna(lower) or pd.isna(upper):
            return None

        band_width = upper - lower
        if band_width <= 0:
            return None

        pct_b = (current_price - lower) / band_width

        if pct_b <= 0.8:
            return None

        strength = (
            "strong" if pct_b > 1.0
            else "moderate" if pct_b > 0.9
            else "weak"
        )

        return TechnicalSignal(
            signal_type=SignalType.BOLLINGER_OVERBOUGHT,
            strength=strength,
            description=f"Bollinger Overbought: Price near upper band (%B: {pct_b:.2f})",
            date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            value=pct_b,
        )

    # -------------------------------------------------------------------------
    # Volume detectors
    # -------------------------------------------------------------------------

    def _detect_volume_spike(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect volume spike without necessarily a breakout."""
        if "volume" not in df.columns or len(df) < 20:
            return None

        avg_volume = df["volume"].rolling(window=20).mean()
        current_volume = df["volume"].iloc[-1]
        recent_avg_volume = avg_volume.iloc[-1]

        if pd.isna(recent_avg_volume) or recent_avg_volume == 0:
            return None

        volume_mult = current_volume / recent_avg_volume

        if volume_mult >= 1.2:  # 120%
            strength = (
                "strong" if volume_mult > 2.0
                else "moderate" if volume_mult > 1.5
                else "weak"
            )

            return TechnicalSignal(
                signal_type=SignalType.VOLUME_SPIKE,
                strength=strength,
                description=f"Volume Spike: {volume_mult:.1f}x average volume",
                date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                value=volume_mult,
            )

        return None

    def _detect_breakout(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price breakout with volume confirmation."""
        if "volume" not in df.columns or len(df) < self.config.breakout_lookback:
            return None

        lookback = self.config.breakout_lookback
        high_n = df["close"].rolling(window=lookback).max()
        avg_volume = df["volume"].rolling(window=lookback).mean()

        current_price = df["close"].iloc[-1]
        recent_high = high_n.iloc[-2]
        current_volume = df["volume"].iloc[-1]
        recent_avg_volume = avg_volume.iloc[-1]

        if pd.isna(recent_high) or pd.isna(recent_avg_volume):
            return None

        if current_price > recent_high * 1.01:
            volume_mult = current_volume / recent_avg_volume

            if volume_mult >= self.config.volume_multiplier:
                strength = (
                    "strong" if volume_mult > 2.5
                    else "moderate" if volume_mult > 2.0
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.BREAKOUT,
                    strength=strength,
                    description=f"Breakout: Price broke above {lookback}-day high with {volume_mult:.1f}x volume",
                    date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                    value=volume_mult,
                )

        return None

    def _detect_breakdown(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect price breakdown with volume confirmation."""
        if "volume" not in df.columns or len(df) < self.config.breakout_lookback:
            return None

        lookback = self.config.breakout_lookback
        low_n = df["close"].rolling(window=lookback).min()
        avg_volume = df["volume"].rolling(window=lookback).mean()

        current_price = df["close"].iloc[-1]
        recent_low = low_n.iloc[-2]
        current_volume = df["volume"].iloc[-1]
        recent_avg_volume = avg_volume.iloc[-1]

        if pd.isna(recent_low) or pd.isna(recent_avg_volume):
            return None

        if current_price < recent_low * 0.99:
            volume_mult = current_volume / recent_avg_volume

            if volume_mult >= self.config.volume_multiplier:
                strength = (
                    "strong" if volume_mult > 2.5
                    else "moderate" if volume_mult > 2.0
                    else "weak"
                )

                return TechnicalSignal(
                    signal_type=SignalType.BREAKDOWN,
                    strength=strength,
                    description=f"Breakdown: Price broke below {lookback}-day low with {volume_mult:.1f}x volume",
                    date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                    value=volume_mult,
                )

        return None

    def _detect_obv_divergence_bull(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect bullish OBV divergence (price falling but OBV rising)."""
        if "volume" not in df.columns or len(df) < 20:
            return None

        obv_series = obv(df)
        lookback = 20

        price_change = df["close"].iloc[-1] - df["close"].iloc[-lookback]
        obv_change = obv_series.iloc[-1] - obv_series.iloc[-lookback]

        # Bullish divergence: price down, OBV up
        if price_change < 0 and obv_change > 0:
            price_pct = abs(price_change) / df["close"].iloc[-lookback]

            strength = (
                "strong" if price_pct > 0.10
                else "moderate" if price_pct > 0.05
                else "weak"
            )

            return TechnicalSignal(
                signal_type=SignalType.OBV_DIVERGENCE_BULL,
                strength=strength,
                description=f"OBV Bullish Divergence: Price down {price_pct:.1%} but OBV rising",
                date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                value=price_pct,
            )

        return None

    def _detect_obv_divergence_bear(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Detect bearish OBV divergence (price rising but OBV falling)."""
        if "volume" not in df.columns or len(df) < 20:
            return None

        obv_series = obv(df)
        lookback = 20

        price_change = df["close"].iloc[-1] - df["close"].iloc[-lookback]
        obv_change = obv_series.iloc[-1] - obv_series.iloc[-lookback]

        # Bearish divergence: price up, OBV down
        if price_change > 0 and obv_change < 0:
            price_pct = price_change / df["close"].iloc[-lookback]

            strength = (
                "strong" if price_pct > 0.10
                else "moderate" if price_pct > 0.05
                else "weak"
            )

            return TechnicalSignal(
                signal_type=SignalType.OBV_DIVERGENCE_BEAR,
                strength=strength,
                description=f"OBV Bearish Divergence: Price up {price_pct:.1%} but OBV falling",
                date_detected=df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                value=price_pct,
            )

        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_current_rsi(self, df: pd.DataFrame) -> Optional[float]:
        """Get current RSI value."""
        if len(df) < 14:
            return None
        try:
            rsi_series = rsi(df, period=14)
            val = rsi_series.iloc[-1]
            return float(val) if not pd.isna(val) else None
        except Exception:
            return None

    def _generate_recommendation(
        self,
        signals: list[TechnicalSignal],
        bullish_score: float,
        bearish_score: float,
    ) -> tuple[Recommendation, float]:
        """Generate recommendation based on signal scores."""
        if not signals:
            return ("hold", 0.5)

        net_score = bullish_score - bearish_score

        if net_score >= 6:
            return ("strong_buy", 0.85)
        elif net_score >= 3:
            return ("buy", 0.70)
        elif net_score > -3:
            return ("hold", 0.55)
        elif net_score > -6:
            return ("sell", 0.65)
        else:
            return ("strong_sell", 0.80)

    def _generate_notes(self, signals: list[TechnicalSignal]) -> list[str]:
        """Generate helpful notes about detected signals."""
        notes: list[str] = []

        if not signals:
            notes.append("No technical signals detected")
            return notes

        signal_types = {s.signal_type for s in signals}
        strong_count = sum(1 for s in signals if s.strength == "strong")

        # Confluence detection
        if SignalType.GOLDEN_CROSS in signal_types and SignalType.RSI_OVERSOLD in signal_types:
            notes.append("Strong setup: Golden cross + oversold RSI indicates bullish reversal")

        if SignalType.DEATH_CROSS in signal_types and SignalType.RSI_OVERBOUGHT in signal_types:
            notes.append("Bearish confluence: Death cross + overbought RSI")

        if SignalType.BOLLINGER_OVERSOLD in signal_types and SignalType.MFI_OVERSOLD in signal_types:
            notes.append("Strong reversal setup: Price at lower Bollinger band with oversold MFI")

        if SignalType.BOLLINGER_OVERSOLD in signal_types and SignalType.RSI_OVERSOLD in signal_types:
            notes.append("Multiple oversold signals: Bollinger + RSI suggest potential bounce")

        if SignalType.OBV_DIVERGENCE_BULL in signal_types:
            notes.append("Bullish OBV divergence: Smart money may be accumulating")

        if SignalType.OBV_DIVERGENCE_BEAR in signal_types:
            notes.append("Bearish OBV divergence: Smart money may be distributing")

        if SignalType.BREAKOUT in signal_types:
            notes.append("Volume-confirmed breakout increases continuation probability")

        if strong_count >= 2:
            notes.append(f"Multiple strong signals ({strong_count}) increase conviction")

        bullish = [s for s in signals if s.is_bullish]
        bearish = [s for s in signals if s.is_bearish]

        if bullish and bearish:
            notes.append("Mixed signals detected - consider waiting for clarity")

        return notes


# Convenience functions
def screen_symbol(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    config: Optional[ScreenerConfig] = None,
) -> ScreenerResult:
    """
    Convenience function to screen a symbol.

    Args:
        df: OHLCV DataFrame with lowercase columns
        symbol: Symbol name
        config: Optional ScreenerConfig

    Returns:
        ScreenerResult with signals and recommendation
    """
    screener = TechnicalScreener(config)
    return screener.screen(df, symbol)


def screen_portfolio(
    portfolio_data: dict[str, pd.DataFrame],
    config: Optional[ScreenerConfig] = None,
) -> PortfolioScreeningResult:
    """
    Convenience function to screen multiple symbols.

    Args:
        portfolio_data: Dict mapping symbol -> OHLCV DataFrame
        config: Optional ScreenerConfig

    Returns:
        PortfolioScreeningResult with ranked results
    """
    screener = TechnicalScreener(config)
    return screener.screen_portfolio(portfolio_data)


__all__ = [
    "TechnicalScreener",
    "screen_symbol",
    "screen_portfolio",
]
