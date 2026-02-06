"""
Candlestick pattern detection algorithms.

Implements single-bar, dual-bar, and triple-bar candlestick pattern recognition.
"""

from enum import Enum

import numpy as np
import pandas as pd

from quantdash.core.models import PatternDirection, PatternEvent


class CandlestickType(str, Enum):
    """Supported candlestick patterns."""
    # Single bar
    DOJI = "doji"
    DOJI_LONG_LEGGED = "doji_long_legged"
    DOJI_DRAGONFLY = "doji_dragonfly"
    DOJI_GRAVESTONE = "doji_gravestone"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"
    SPINNING_TOP = "spinning_top"
    PIN_BAR_BULLISH = "pin_bar_bullish"
    PIN_BAR_BEARISH = "pin_bar_bearish"

    # Dual bar
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    BULLISH_HARAMI = "bullish_harami"
    BEARISH_HARAMI = "bearish_harami"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"

    # Triple bar
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"


# =============================================================================
# HELPERS
# =============================================================================


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range(h: float, l: float) -> float:
    return h - l


def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _is_bullish(o: float, c: float) -> bool:
    return c > o


def _is_bearish(o: float, c: float) -> bool:
    return c < o


def _trend_direction(close: pd.Series, i: int, lookback: int = 20) -> float:
    """Return positive for uptrend, negative for downtrend via SMA slope."""
    if i < lookback:
        return 0.0
    window = close.iloc[i - lookback:i]
    sma = window.mean()
    return (close.iloc[i] - sma) / sma


# =============================================================================
# SINGLE-BAR PATTERNS
# =============================================================================


def detect_doji(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Doji variants (standard, long-legged, dragonfly, gravestone)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(1, len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        body = _body(o[i], c[i])
        body_ratio = body / rng
        upper = _upper_shadow(o[i], h[i], c[i])
        lower = _lower_shadow(o[i], l[i], c[i])

        if body_ratio > 0.1:
            continue  # Not a doji

        # Dragonfly: long lower shadow, negligible upper
        if lower > 3 * body and upper < body * 1.5:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.DOJI_DRAGONFLY.value,
                start_index=i, end_index=i, confidence=0.70,
                direction=PatternDirection.BULLISH,
            ))
        # Gravestone: long upper shadow, negligible lower
        elif upper > 3 * body and lower < body * 1.5:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.DOJI_GRAVESTONE.value,
                start_index=i, end_index=i, confidence=0.70,
                direction=PatternDirection.BEARISH,
            ))
        # Long-legged: both shadows long
        elif upper > 2 * body and lower > 2 * body:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.DOJI_LONG_LEGGED.value,
                start_index=i, end_index=i, confidence=0.55,
                direction=PatternDirection.NEUTRAL,
            ))
        # Standard doji
        else:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.DOJI.value,
                start_index=i, end_index=i, confidence=0.50,
                direction=PatternDirection.NEUTRAL,
            ))

    return patterns


def detect_hammer_hanging_man(df: pd.DataFrame) -> list[PatternEvent]:
    """
    Detect Hammer (bullish, in downtrend) and Hanging Man (bearish, in uptrend).

    Shape: small body at top, long lower shadow >= 2x body, negligible upper shadow.
    """
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    close_s = df["close"]

    for i in range(20, len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        body = _body(o[i], c[i])
        lower = _lower_shadow(o[i], l[i], c[i])
        upper = _upper_shadow(o[i], h[i], c[i])

        # Body in upper third, long lower shadow
        body_pos = (min(o[i], c[i]) - l[i]) / rng
        if body_pos < 0.6 or lower < 2 * body or upper > body * 0.5:
            continue

        trend = _trend_direction(close_s, i)
        confidence = min(0.80, 0.55 + lower / rng * 0.3)

        if trend < -0.01:  # Downtrend → Hammer (bullish reversal)
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.HAMMER.value,
                start_index=i, end_index=i, confidence=round(confidence, 2),
                direction=PatternDirection.BULLISH,
            ))
        elif trend > 0.01:  # Uptrend → Hanging Man (bearish reversal)
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.HANGING_MAN.value,
                start_index=i, end_index=i, confidence=round(confidence, 2),
                direction=PatternDirection.BEARISH,
            ))

    return patterns


def detect_inverted_hammer_shooting_star(df: pd.DataFrame) -> list[PatternEvent]:
    """
    Detect Inverted Hammer (bullish, in downtrend) and Shooting Star (bearish, in uptrend).

    Shape: small body at bottom, long upper shadow >= 2x body, negligible lower shadow.
    """
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    close_s = df["close"]

    for i in range(20, len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        body = _body(o[i], c[i])
        upper = _upper_shadow(o[i], h[i], c[i])
        lower = _lower_shadow(o[i], l[i], c[i])

        body_pos = (min(o[i], c[i]) - l[i]) / rng
        if body_pos > 0.4 or upper < 2 * body or lower > body * 0.5:
            continue

        trend = _trend_direction(close_s, i)
        confidence = min(0.80, 0.55 + upper / rng * 0.3)

        if trend < -0.01:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.INVERTED_HAMMER.value,
                start_index=i, end_index=i, confidence=round(confidence, 2),
                direction=PatternDirection.BULLISH,
            ))
        elif trend > 0.01:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.SHOOTING_STAR.value,
                start_index=i, end_index=i, confidence=round(confidence, 2),
                direction=PatternDirection.BEARISH,
            ))

    return patterns


def detect_marubozu(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Marubozu (full-body candle with no/minimal shadows)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        body = _body(o[i], c[i])
        if body / rng < 0.95:
            continue

        if _is_bullish(o[i], c[i]):
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.MARUBOZU_BULLISH.value,
                start_index=i, end_index=i, confidence=0.75,
                direction=PatternDirection.BULLISH,
            ))
        else:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.MARUBOZU_BEARISH.value,
                start_index=i, end_index=i, confidence=0.75,
                direction=PatternDirection.BEARISH,
            ))

    return patterns


def detect_spinning_top(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Spinning Top (small body, shadows on both sides)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        body = _body(o[i], c[i])
        upper = _upper_shadow(o[i], h[i], c[i])
        lower = _lower_shadow(o[i], l[i], c[i])

        # Small body (< 1/3 of range), shadows on both sides
        if body / rng < 0.33 and upper > body * 0.5 and lower > body * 0.5:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.SPINNING_TOP.value,
                start_index=i, end_index=i, confidence=0.50,
                direction=PatternDirection.NEUTRAL,
            ))

    return patterns


def detect_pin_bar(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Pin Bar (long shadow >= 2/3 of range on one side)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(len(df)):
        rng = _range(h[i], l[i])
        if rng < 1e-10:
            continue

        lower = _lower_shadow(o[i], l[i], c[i])
        upper = _upper_shadow(o[i], h[i], c[i])

        if lower / rng >= 0.67 and upper / rng < 0.2:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.PIN_BAR_BULLISH.value,
                start_index=i, end_index=i, confidence=0.70,
                direction=PatternDirection.BULLISH,
            ))
        elif upper / rng >= 0.67 and lower / rng < 0.2:
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.PIN_BAR_BEARISH.value,
                start_index=i, end_index=i, confidence=0.70,
                direction=PatternDirection.BEARISH,
            ))

    return patterns


# =============================================================================
# DUAL-BAR PATTERNS
# =============================================================================


def detect_engulfing(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Bullish and Bearish Engulfing patterns."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(1, len(df)):
        prev_body_top = max(o[i - 1], c[i - 1])
        prev_body_bot = min(o[i - 1], c[i - 1])
        curr_body_top = max(o[i], c[i])
        curr_body_bot = min(o[i], c[i])

        # Current body must fully engulf previous body
        if curr_body_top <= prev_body_top or curr_body_bot >= prev_body_bot:
            continue

        if _is_bearish(o[i - 1], c[i - 1]) and _is_bullish(o[i], c[i]):
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.BULLISH_ENGULFING.value,
                start_index=i - 1, end_index=i, confidence=0.75,
                direction=PatternDirection.BULLISH,
            ))
        elif _is_bullish(o[i - 1], c[i - 1]) and _is_bearish(o[i], c[i]):
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.BEARISH_ENGULFING.value,
                start_index=i - 1, end_index=i, confidence=0.75,
                direction=PatternDirection.BEARISH,
            ))

    return patterns


def detect_harami(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Bullish and Bearish Harami patterns."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(1, len(df)):
        prev_body_top = max(o[i - 1], c[i - 1])
        prev_body_bot = min(o[i - 1], c[i - 1])
        curr_body_top = max(o[i], c[i])
        curr_body_bot = min(o[i], c[i])
        prev_body = _body(o[i - 1], c[i - 1])
        curr_body = _body(o[i], c[i])

        # Previous body must be large, current small and inside
        if prev_body < 1e-10 or curr_body >= prev_body * 0.6:
            continue
        if curr_body_top > prev_body_top or curr_body_bot < prev_body_bot:
            continue

        if _is_bearish(o[i - 1], c[i - 1]) and _is_bullish(o[i], c[i]):
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.BULLISH_HARAMI.value,
                start_index=i - 1, end_index=i, confidence=0.65,
                direction=PatternDirection.BULLISH,
            ))
        elif _is_bullish(o[i - 1], c[i - 1]) and _is_bearish(o[i], c[i]):
            patterns.append(PatternEvent(
                pattern_type=CandlestickType.BEARISH_HARAMI.value,
                start_index=i - 1, end_index=i, confidence=0.65,
                direction=PatternDirection.BEARISH,
            ))

    return patterns


def detect_tweezers(df: pd.DataFrame, tolerance: float = 0.001) -> list[PatternEvent]:
    """Detect Tweezer Top and Bottom patterns."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    close_s = df["close"]

    for i in range(20, len(df)):
        trend = _trend_direction(close_s, i)

        # Tweezer Top: matching highs in uptrend
        if trend > 0.01 and abs(h[i] - h[i - 1]) / h[i] < tolerance:
            if _is_bullish(o[i - 1], c[i - 1]) and _is_bearish(o[i], c[i]):
                patterns.append(PatternEvent(
                    pattern_type=CandlestickType.TWEEZER_TOP.value,
                    start_index=i - 1, end_index=i, confidence=0.65,
                    direction=PatternDirection.BEARISH,
                ))

        # Tweezer Bottom: matching lows in downtrend
        if trend < -0.01 and abs(l[i] - l[i - 1]) / l[i] < tolerance:
            if _is_bearish(o[i - 1], c[i - 1]) and _is_bullish(o[i], c[i]):
                patterns.append(PatternEvent(
                    pattern_type=CandlestickType.TWEEZER_BOTTOM.value,
                    start_index=i - 1, end_index=i, confidence=0.65,
                    direction=PatternDirection.BULLISH,
                ))

    return patterns


# =============================================================================
# TRIPLE-BAR PATTERNS
# =============================================================================


def detect_morning_evening_star(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Morning Star (bullish) and Evening Star (bearish)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(2, len(df)):
        body0 = _body(o[i - 2], c[i - 2])
        body1 = _body(o[i - 1], c[i - 1])
        body2 = _body(o[i], c[i])
        rng0 = _range(h[i - 2], l[i - 2])

        if rng0 < 1e-10 or body0 < 1e-10:
            continue

        # Middle bar must be small (star)
        if body1 > body0 * 0.4:
            continue

        # Morning Star: bearish → small → bullish
        if (_is_bearish(o[i - 2], c[i - 2]) and _is_bullish(o[i], c[i])
                and body2 > body0 * 0.5):
            # Third bar should close above midpoint of first bar
            mid_first = (o[i - 2] + c[i - 2]) / 2
            if c[i] > mid_first:
                patterns.append(PatternEvent(
                    pattern_type=CandlestickType.MORNING_STAR.value,
                    start_index=i - 2, end_index=i, confidence=0.75,
                    direction=PatternDirection.BULLISH,
                ))

        # Evening Star: bullish → small → bearish
        if (_is_bullish(o[i - 2], c[i - 2]) and _is_bearish(o[i], c[i])
                and body2 > body0 * 0.5):
            mid_first = (o[i - 2] + c[i - 2]) / 2
            if c[i] < mid_first:
                patterns.append(PatternEvent(
                    pattern_type=CandlestickType.EVENING_STAR.value,
                    start_index=i - 2, end_index=i, confidence=0.75,
                    direction=PatternDirection.BEARISH,
                ))

    return patterns


def detect_three_soldiers_crows(df: pd.DataFrame) -> list[PatternEvent]:
    """Detect Three White Soldiers (bullish) and Three Black Crows (bearish)."""
    patterns = []
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    for i in range(2, len(df)):
        # Three White Soldiers: 3 consecutive bullish bars with higher closes
        if (all(_is_bullish(o[i - j], c[i - j]) for j in range(3))
                and c[i] > c[i - 1] > c[i - 2]):
            # Each opens within previous body
            if (o[i] >= o[i - 1] and o[i] <= c[i - 1]
                    and o[i - 1] >= o[i - 2] and o[i - 1] <= c[i - 2]):
                # Small upper shadows (strong conviction)
                bodies = [_body(o[i - j], c[i - j]) for j in range(3)]
                ranges = [_range(h[i - j], l[i - j]) for j in range(3)]
                uppers = [_upper_shadow(o[i - j], h[i - j], c[i - j]) for j in range(3)]
                if all(u < b * 0.5 for u, b in zip(uppers, bodies)):
                    patterns.append(PatternEvent(
                        pattern_type=CandlestickType.THREE_WHITE_SOLDIERS.value,
                        start_index=i - 2, end_index=i, confidence=0.80,
                        direction=PatternDirection.BULLISH,
                    ))

        # Three Black Crows: 3 consecutive bearish bars with lower closes
        if (all(_is_bearish(o[i - j], c[i - j]) for j in range(3))
                and c[i] < c[i - 1] < c[i - 2]):
            if (o[i] <= o[i - 1] and o[i] >= c[i - 1]
                    and o[i - 1] <= o[i - 2] and o[i - 1] >= c[i - 2]):
                bodies = [_body(o[i - j], c[i - j]) for j in range(3)]
                lowers = [_lower_shadow(o[i - j], l[i - j], c[i - j]) for j in range(3)]
                if all(lw < b * 0.5 for lw, b in zip(lowers, bodies)):
                    patterns.append(PatternEvent(
                        pattern_type=CandlestickType.THREE_BLACK_CROWS.value,
                        start_index=i - 2, end_index=i, confidence=0.80,
                        direction=PatternDirection.BEARISH,
                    ))

    return patterns


# =============================================================================
# MASTER DETECTOR
# =============================================================================


def detect_all_candlestick_patterns(df: pd.DataFrame) -> list[PatternEvent]:
    """Run all candlestick pattern detectors."""
    all_patterns = []
    all_patterns.extend(detect_doji(df))
    all_patterns.extend(detect_hammer_hanging_man(df))
    all_patterns.extend(detect_inverted_hammer_shooting_star(df))
    all_patterns.extend(detect_marubozu(df))
    all_patterns.extend(detect_spinning_top(df))
    all_patterns.extend(detect_pin_bar(df))
    all_patterns.extend(detect_engulfing(df))
    all_patterns.extend(detect_harami(df))
    all_patterns.extend(detect_tweezers(df))
    all_patterns.extend(detect_morning_evening_star(df))
    all_patterns.extend(detect_three_soldiers_crows(df))
    all_patterns.sort(key=lambda p: p.end_index, reverse=True)
    return all_patterns


# =============================================================================
# PATTERNS CONFIG REGISTRY
# =============================================================================

CANDLESTICK_PATTERNS_CONFIG: dict[str, dict] = {
    "Doji": {"type": CandlestickType.DOJI, "direction": "neutral", "category": "single", "desc": "Open and close nearly equal, indecision"},
    "Dragonfly Doji": {"type": CandlestickType.DOJI_DRAGONFLY, "direction": "bullish", "category": "single", "desc": "Long lower shadow, bullish reversal signal"},
    "Gravestone Doji": {"type": CandlestickType.DOJI_GRAVESTONE, "direction": "bearish", "category": "single", "desc": "Long upper shadow, bearish reversal signal"},
    "Hammer": {"type": CandlestickType.HAMMER, "direction": "bullish", "category": "single", "desc": "Long lower shadow in downtrend, bullish reversal"},
    "Hanging Man": {"type": CandlestickType.HANGING_MAN, "direction": "bearish", "category": "single", "desc": "Hammer shape in uptrend, bearish reversal"},
    "Shooting Star": {"type": CandlestickType.SHOOTING_STAR, "direction": "bearish", "category": "single", "desc": "Long upper shadow in uptrend, bearish reversal"},
    "Marubozu": {"type": CandlestickType.MARUBOZU_BULLISH, "direction": "continuation", "category": "single", "desc": "Full body candle with no shadows, strong conviction"},
    "Pin Bar": {"type": CandlestickType.PIN_BAR_BULLISH, "direction": "reversal", "category": "single", "desc": "Long shadow >= 2/3 of range on one side"},
    "Bullish Engulfing": {"type": CandlestickType.BULLISH_ENGULFING, "direction": "bullish", "category": "dual", "desc": "Current bullish body engulfs previous bearish body"},
    "Bearish Engulfing": {"type": CandlestickType.BEARISH_ENGULFING, "direction": "bearish", "category": "dual", "desc": "Current bearish body engulfs previous bullish body"},
    "Bullish Harami": {"type": CandlestickType.BULLISH_HARAMI, "direction": "bullish", "category": "dual", "desc": "Small bullish body inside large bearish body"},
    "Bearish Harami": {"type": CandlestickType.BEARISH_HARAMI, "direction": "bearish", "category": "dual", "desc": "Small bearish body inside large bullish body"},
    "Tweezer Top": {"type": CandlestickType.TWEEZER_TOP, "direction": "bearish", "category": "dual", "desc": "Matching highs in uptrend, bearish reversal"},
    "Tweezer Bottom": {"type": CandlestickType.TWEEZER_BOTTOM, "direction": "bullish", "category": "dual", "desc": "Matching lows in downtrend, bullish reversal"},
    "Morning Star": {"type": CandlestickType.MORNING_STAR, "direction": "bullish", "category": "triple", "desc": "Bearish, star, bullish — strong bullish reversal"},
    "Evening Star": {"type": CandlestickType.EVENING_STAR, "direction": "bearish", "category": "triple", "desc": "Bullish, star, bearish — strong bearish reversal"},
    "Three White Soldiers": {"type": CandlestickType.THREE_WHITE_SOLDIERS, "direction": "bullish", "category": "triple", "desc": "Three consecutive bullish bars with higher closes"},
    "Three Black Crows": {"type": CandlestickType.THREE_BLACK_CROWS, "direction": "bearish", "category": "triple", "desc": "Three consecutive bearish bars with lower closes"},
}
