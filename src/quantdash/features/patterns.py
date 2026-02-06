"""
Chart pattern detection algorithms.

Implements rule-based detection for classic technical analysis patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quantdash.core.models import PatternDirection, PatternEvent


class PatternType(str, Enum):
    """Supported chart patterns."""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"
    ISLAND_REVERSAL = "island_reversal"

    # Continuation patterns
    CUP_AND_HANDLE = "cup_and_handle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"

    # Gap patterns
    GAP_COMMON = "gap_common"
    GAP_BREAKAWAY = "gap_breakaway"
    GAP_RUNAWAY = "gap_runaway"
    GAP_EXHAUSTION = "gap_exhaustion"


@dataclass
class PatternConfig:
    """Configuration for pattern detection."""
    min_pattern_bars: int = 10
    max_pattern_bars: int = 100
    price_tolerance: float = 0.02  # 2% tolerance for equality
    volume_confirmation: bool = True
    min_confidence: float = 0.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_local_extrema(
    series: pd.Series,
    order: int = 5,
) -> tuple[list[int], list[int]]:
    """
    Find local maxima and minima indices.

    Args:
        series: Price series
        order: Number of bars on each side to compare

    Returns:
        Tuple of (maxima_indices, minima_indices)
    """
    maxima = []
    minima = []
    values = series.values

    for i in range(order, len(values) - order):
        # Check for local maximum
        if all(values[i] >= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] >= values[i + j] for j in range(1, order + 1)):
            maxima.append(i)

        # Check for local minimum
        if all(values[i] <= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] <= values[i + j] for j in range(1, order + 1)):
            minima.append(i)

    return maxima, minima


def prices_equal(price1: float, price2: float, tolerance: float = 0.02) -> bool:
    """Check if two prices are approximately equal within tolerance."""
    return abs(price1 - price2) / max(price1, price2) <= tolerance


def calculate_slope(prices: pd.Series) -> float:
    """Calculate linear regression slope of a price series."""
    x = np.arange(len(prices))
    slope, _ = np.polyfit(x, prices.values, 1)
    return slope


# =============================================================================
# REVERSAL PATTERNS
# =============================================================================


def detect_head_and_shoulders(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Head and Shoulders patterns (regular and inverse).

    Regular H&S: Bearish reversal at top
    Inverse H&S: Bullish reversal at bottom
    """
    patterns = []
    high = df["high"]
    low = df["low"]
    close = df["close"]

    maxima, minima = find_local_extrema(high, order=5)

    # Regular Head and Shoulders
    for i in range(len(maxima) - 2):
        left_shoulder = maxima[i]
        head = maxima[i + 1]
        right_shoulder = maxima[i + 2]

        # Head must be higher than shoulders
        if high.iloc[head] > high.iloc[left_shoulder] and \
           high.iloc[head] > high.iloc[right_shoulder]:

            # Shoulders approximately equal
            if prices_equal(
                high.iloc[left_shoulder],
                high.iloc[right_shoulder],
                config.price_tolerance * 2,
            ):
                # Find neckline (troughs between peaks)
                troughs_between = [
                    m for m in minima
                    if left_shoulder < m < right_shoulder
                ]

                if len(troughs_between) >= 2:
                    neckline_left = troughs_between[0]
                    neckline_right = troughs_between[-1]
                    neckline_price = (
                        low.iloc[neckline_left] + low.iloc[neckline_right]
                    ) / 2

                    # Calculate target (head to neckline distance)
                    pattern_height = high.iloc[head] - neckline_price
                    target = neckline_price - pattern_height

                    # Confidence based on symmetry and volume
                    symmetry = 1 - abs(
                        high.iloc[left_shoulder] - high.iloc[right_shoulder]
                    ) / high.iloc[head]
                    confidence = min(0.9, 0.5 + symmetry * 0.4)

                    if confidence >= config.min_confidence:
                        patterns.append(PatternEvent(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS.value,
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            confidence=round(confidence, 2),
                            direction=PatternDirection.BEARISH,
                            target_price=target,
                            stop_price=high.iloc[head],
                            metadata={
                                "neckline": neckline_price,
                                "head_price": high.iloc[head],
                            },
                        ))

    # Inverse Head and Shoulders
    for i in range(len(minima) - 2):
        left_shoulder = minima[i]
        head = minima[i + 1]
        right_shoulder = minima[i + 2]

        # Head must be lower than shoulders
        if low.iloc[head] < low.iloc[left_shoulder] and \
           low.iloc[head] < low.iloc[right_shoulder]:

            # Shoulders approximately equal
            if prices_equal(
                low.iloc[left_shoulder],
                low.iloc[right_shoulder],
                config.price_tolerance * 2,
            ):
                # Find neckline (peaks between troughs)
                peaks_between = [
                    m for m in maxima
                    if left_shoulder < m < right_shoulder
                ]

                if len(peaks_between) >= 2:
                    neckline_price = (
                        high.iloc[peaks_between[0]] + high.iloc[peaks_between[-1]]
                    ) / 2

                    pattern_height = neckline_price - low.iloc[head]
                    target = neckline_price + pattern_height

                    symmetry = 1 - abs(
                        low.iloc[left_shoulder] - low.iloc[right_shoulder]
                    ) / abs(low.iloc[head])
                    confidence = min(0.9, 0.5 + symmetry * 0.4)

                    if confidence >= config.min_confidence:
                        patterns.append(PatternEvent(
                            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS.value,
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            confidence=round(confidence, 2),
                            direction=PatternDirection.BULLISH,
                            target_price=target,
                            stop_price=low.iloc[head],
                            metadata={"neckline": neckline_price},
                        ))

    return patterns


def detect_double_top_bottom(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Double Top and Double Bottom patterns.
    """
    patterns = []
    high = df["high"]
    low = df["low"]

    maxima, minima = find_local_extrema(high, order=5)

    # Double Top
    for i in range(len(maxima) - 1):
        first = maxima[i]
        second = maxima[i + 1]

        # Peaks approximately equal
        if prices_equal(high.iloc[first], high.iloc[second], config.price_tolerance):
            # Find trough between peaks
            troughs_between = [m for m in minima if first < m < second]

            if troughs_between:
                trough = troughs_between[0]
                neckline = low.iloc[trough]
                pattern_height = high.iloc[first] - neckline
                target = neckline - pattern_height

                confidence = 1 - abs(
                    high.iloc[first] - high.iloc[second]
                ) / high.iloc[first]

                if confidence >= config.min_confidence:
                    patterns.append(PatternEvent(
                        pattern_type=PatternType.DOUBLE_TOP.value,
                        start_index=first,
                        end_index=second,
                        confidence=round(confidence, 2),
                        direction=PatternDirection.BEARISH,
                        target_price=target,
                        stop_price=max(high.iloc[first], high.iloc[second]),
                        metadata={"neckline": neckline},
                    ))

    # Double Bottom
    for i in range(len(minima) - 1):
        first = minima[i]
        second = minima[i + 1]

        # Troughs approximately equal
        if prices_equal(low.iloc[first], low.iloc[second], config.price_tolerance):
            # Find peak between troughs
            peaks_between = [m for m in maxima if first < m < second]

            if peaks_between:
                peak = peaks_between[0]
                neckline = high.iloc[peak]
                pattern_height = neckline - low.iloc[first]
                target = neckline + pattern_height

                confidence = 1 - abs(
                    low.iloc[first] - low.iloc[second]
                ) / low.iloc[first]

                if confidence >= config.min_confidence:
                    patterns.append(PatternEvent(
                        pattern_type=PatternType.DOUBLE_BOTTOM.value,
                        start_index=first,
                        end_index=second,
                        confidence=round(confidence, 2),
                        direction=PatternDirection.BULLISH,
                        target_price=target,
                        stop_price=min(low.iloc[first], low.iloc[second]),
                        metadata={"neckline": neckline},
                    ))

    return patterns


def detect_triple_top_bottom(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Triple Top and Triple Bottom patterns.
    """
    patterns = []
    high = df["high"]
    low = df["low"]

    maxima, minima = find_local_extrema(high, order=5)

    # Triple Top
    for i in range(len(maxima) - 2):
        first, second, third = maxima[i], maxima[i + 1], maxima[i + 2]

        if prices_equal(high.iloc[first], high.iloc[second], config.price_tolerance) and \
           prices_equal(high.iloc[second], high.iloc[third], config.price_tolerance):

            avg_peak = (high.iloc[first] + high.iloc[second] + high.iloc[third]) / 3

            # Find support level
            troughs_between = [
                m for m in minima if first < m < third
            ]
            if troughs_between:
                support = min(low.iloc[t] for t in troughs_between)
                target = support - (avg_peak - support)

                confidence = min(0.85, 0.6 + 0.1 * len(troughs_between))

                if confidence >= config.min_confidence:
                    patterns.append(PatternEvent(
                        pattern_type=PatternType.TRIPLE_TOP.value,
                        start_index=first,
                        end_index=third,
                        confidence=round(confidence, 2),
                        direction=PatternDirection.BEARISH,
                        target_price=target,
                        stop_price=avg_peak * 1.01,
                    ))

    # Triple Bottom
    for i in range(len(minima) - 2):
        first, second, third = minima[i], minima[i + 1], minima[i + 2]

        if prices_equal(low.iloc[first], low.iloc[second], config.price_tolerance) and \
           prices_equal(low.iloc[second], low.iloc[third], config.price_tolerance):

            avg_trough = (low.iloc[first] + low.iloc[second] + low.iloc[third]) / 3

            peaks_between = [m for m in maxima if first < m < third]
            if peaks_between:
                resistance = max(high.iloc[p] for p in peaks_between)
                target = resistance + (resistance - avg_trough)

                confidence = min(0.85, 0.6 + 0.1 * len(peaks_between))

                if confidence >= config.min_confidence:
                    patterns.append(PatternEvent(
                        pattern_type=PatternType.TRIPLE_BOTTOM.value,
                        start_index=first,
                        end_index=third,
                        confidence=round(confidence, 2),
                        direction=PatternDirection.BULLISH,
                        target_price=target,
                        stop_price=avg_trough * 0.99,
                    ))

    return patterns


# =============================================================================
# CONTINUATION PATTERNS
# =============================================================================


def detect_cup_and_handle(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Cup and Handle pattern (bullish continuation).
    """
    patterns = []
    high = df["high"]
    low = df["low"]
    close = df["close"]

    maxima, minima = find_local_extrema(high, order=10)

    for i in range(len(maxima) - 1):
        left_lip = maxima[i]
        right_lip = maxima[i + 1]

        # Lips should be at similar levels
        if not prices_equal(
            high.iloc[left_lip], high.iloc[right_lip], config.price_tolerance * 1.5
        ):
            continue

        # Find bottom of cup
        cup_bottom_candidates = [
            m for m in minima if left_lip < m < right_lip
        ]
        if not cup_bottom_candidates:
            continue

        cup_bottom = min(cup_bottom_candidates, key=lambda x: low.iloc[x])

        # Cup should be U-shaped (gradual, not V-shaped)
        cup_length = right_lip - left_lip
        cup_depth = high.iloc[left_lip] - low.iloc[cup_bottom]

        # Check cup proportions
        if cup_length < config.min_pattern_bars:
            continue

        # Look for handle (small consolidation after right lip)
        handle_start = right_lip
        handle_end = min(right_lip + cup_length // 4, len(df) - 1)

        if handle_end > handle_start + 3:
            handle_data = df.iloc[handle_start:handle_end]
            handle_range = handle_data["high"].max() - handle_data["low"].min()
            handle_retracement = handle_range / cup_depth

            # Handle should retrace 10-50% of cup
            if 0.1 <= handle_retracement <= 0.5:
                lip_price = (high.iloc[left_lip] + high.iloc[right_lip]) / 2
                target = lip_price + cup_depth

                confidence = min(0.85, 0.5 + 0.15 * (1 - handle_retracement))

                if confidence >= config.min_confidence:
                    patterns.append(PatternEvent(
                        pattern_type=PatternType.CUP_AND_HANDLE.value,
                        start_index=left_lip,
                        end_index=handle_end,
                        confidence=round(confidence, 2),
                        direction=PatternDirection.BULLISH,
                        target_price=target,
                        stop_price=low.iloc[cup_bottom],
                        metadata={
                            "cup_depth": cup_depth,
                            "handle_retracement": handle_retracement,
                        },
                    ))

    return patterns


def detect_flag_pennant(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Flag and Pennant patterns.

    Flags: Parallel channel consolidation
    Pennants: Converging trendlines consolidation
    """
    patterns = []
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Look for strong moves (flagpole)
    returns = close.pct_change(5)

    for i in range(20, len(df) - 10):
        # Check for strong prior move (flagpole)
        flagpole_return = returns.iloc[i]

        if abs(flagpole_return) < 0.05:  # At least 5% move
            continue

        direction = PatternDirection.BULLISH if flagpole_return > 0 else PatternDirection.BEARISH

        # Look at consolidation zone
        consolidation = df.iloc[i:min(i + 15, len(df))]
        if len(consolidation) < 5:
            continue

        high_slope = calculate_slope(consolidation["high"])
        low_slope = calculate_slope(consolidation["low"])

        # Flag: parallel lines (similar slopes)
        if abs(high_slope - low_slope) < 0.1:
            # Counter-trend flag
            if (direction == PatternDirection.BULLISH and high_slope < 0) or \
               (direction == PatternDirection.BEARISH and high_slope > 0):

                flagpole_size = abs(close.iloc[i] - close.iloc[i - 5])
                if direction == PatternDirection.BULLISH:
                    target = consolidation["high"].iloc[-1] + flagpole_size
                else:
                    target = consolidation["low"].iloc[-1] - flagpole_size

                patterns.append(PatternEvent(
                    pattern_type=PatternType.FLAG.value,
                    start_index=i - 5,
                    end_index=i + len(consolidation) - 1,
                    confidence=0.65,
                    direction=direction,
                    target_price=target,
                ))

        # Pennant: converging lines
        elif (high_slope < 0 and low_slope > 0) or \
             abs(high_slope) > abs(low_slope) and high_slope * low_slope < 0:

            flagpole_size = abs(close.iloc[i] - close.iloc[i - 5])
            apex_price = (consolidation["high"].iloc[-1] + consolidation["low"].iloc[-1]) / 2

            if direction == PatternDirection.BULLISH:
                target = apex_price + flagpole_size
            else:
                target = apex_price - flagpole_size

            patterns.append(PatternEvent(
                pattern_type=PatternType.PENNANT.value,
                start_index=i - 5,
                end_index=i + len(consolidation) - 1,
                confidence=0.60,
                direction=direction,
                target_price=target,
            ))

    return patterns


def detect_wedge(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Rising and Falling Wedge patterns.

    Rising Wedge: Bearish (converging upward)
    Falling Wedge: Bullish (converging downward)
    """
    patterns = []

    for i in range(config.min_pattern_bars, len(df) - 5):
        window = df.iloc[i - config.min_pattern_bars:i]

        high_slope = calculate_slope(window["high"])
        low_slope = calculate_slope(window["low"])

        # Both slopes same direction and converging
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            # Rising Wedge - bearish
            patterns.append(PatternEvent(
                pattern_type=PatternType.WEDGE_RISING.value,
                start_index=i - config.min_pattern_bars,
                end_index=i,
                confidence=0.60,
                direction=PatternDirection.BEARISH,
            ))

        elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            # Falling Wedge - bullish
            patterns.append(PatternEvent(
                pattern_type=PatternType.WEDGE_FALLING.value,
                start_index=i - config.min_pattern_bars,
                end_index=i,
                confidence=0.60,
                direction=PatternDirection.BULLISH,
            ))

    return patterns


def detect_triangle(
    df: pd.DataFrame,
    config: PatternConfig = PatternConfig(),
) -> list[PatternEvent]:
    """
    Detect Triangle patterns (ascending, descending, symmetrical).
    """
    patterns = []

    for i in range(config.min_pattern_bars, len(df) - 5):
        window = df.iloc[i - config.min_pattern_bars:i]

        high_slope = calculate_slope(window["high"])
        low_slope = calculate_slope(window["low"])

        # Ascending: flat top, rising bottom
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            patterns.append(PatternEvent(
                pattern_type=PatternType.TRIANGLE_ASCENDING.value,
                start_index=i - config.min_pattern_bars,
                end_index=i,
                confidence=0.65,
                direction=PatternDirection.BULLISH,
            ))

        # Descending: falling top, flat bottom
        elif high_slope < -0.01 and abs(low_slope) < 0.01:
            patterns.append(PatternEvent(
                pattern_type=PatternType.TRIANGLE_DESCENDING.value,
                start_index=i - config.min_pattern_bars,
                end_index=i,
                confidence=0.65,
                direction=PatternDirection.BEARISH,
            ))

        # Symmetrical: converging from both sides
        elif high_slope < -0.01 and low_slope > 0.01:
            patterns.append(PatternEvent(
                pattern_type=PatternType.TRIANGLE_SYMMETRICAL.value,
                start_index=i - config.min_pattern_bars,
                end_index=i,
                confidence=0.55,
                direction=PatternDirection.NEUTRAL,
            ))

    return patterns


# =============================================================================
# GAP PATTERNS
# =============================================================================


def detect_gaps(
    df: pd.DataFrame,
    min_gap_percent: float = 0.02,
) -> list[PatternEvent]:
    """
    Detect gap patterns (common, breakaway, runaway, exhaustion).
    """
    patterns = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else None

    for i in range(1, len(df)):
        prev_high = high.iloc[i - 1]
        prev_low = low.iloc[i - 1]
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]
        prev_close = close.iloc[i - 1]

        # Gap up
        if curr_low > prev_high:
            gap_size = (curr_low - prev_high) / prev_close

            if gap_size >= min_gap_percent:
                # Determine gap type based on context
                # Simple heuristic: check prior trend and volume
                prior_trend = close.iloc[max(0, i - 10):i].pct_change().sum()

                if abs(prior_trend) < 0.03:
                    gap_type = PatternType.GAP_COMMON
                    confidence = 0.50
                elif prior_trend < 0 and gap_size > 0.03:
                    gap_type = PatternType.GAP_BREAKAWAY
                    confidence = 0.70
                else:
                    gap_type = PatternType.GAP_RUNAWAY
                    confidence = 0.60

                patterns.append(PatternEvent(
                    pattern_type=gap_type.value,
                    start_index=i - 1,
                    end_index=i,
                    confidence=confidence,
                    direction=PatternDirection.BULLISH,
                    metadata={"gap_percent": round(gap_size * 100, 2)},
                ))

        # Gap down
        elif curr_high < prev_low:
            gap_size = (prev_low - curr_high) / prev_close

            if gap_size >= min_gap_percent:
                prior_trend = close.iloc[max(0, i - 10):i].pct_change().sum()

                if abs(prior_trend) < 0.03:
                    gap_type = PatternType.GAP_COMMON
                    confidence = 0.50
                elif prior_trend > 0 and gap_size > 0.03:
                    gap_type = PatternType.GAP_BREAKAWAY
                    confidence = 0.70
                else:
                    gap_type = PatternType.GAP_RUNAWAY
                    confidence = 0.60

                patterns.append(PatternEvent(
                    pattern_type=gap_type.value,
                    start_index=i - 1,
                    end_index=i,
                    confidence=confidence,
                    direction=PatternDirection.BEARISH,
                    metadata={"gap_percent": round(gap_size * 100, 2)},
                ))

    return patterns


# =============================================================================
# MASTER PATTERN DETECTOR
# =============================================================================


def detect_all_patterns(
    df: pd.DataFrame,
    config: Optional[PatternConfig] = None,
) -> list[PatternEvent]:
    """
    Run all pattern detection algorithms on the DataFrame.

    Args:
        df: OHLCV DataFrame with lowercase column names
        config: Optional PatternConfig for detection parameters

    Returns:
        List of all detected patterns sorted by end_index
    """
    if config is None:
        config = PatternConfig()

    all_patterns = []

    # Run all detectors
    all_patterns.extend(detect_head_and_shoulders(df, config))
    all_patterns.extend(detect_double_top_bottom(df, config))
    all_patterns.extend(detect_triple_top_bottom(df, config))
    all_patterns.extend(detect_cup_and_handle(df, config))
    all_patterns.extend(detect_flag_pennant(df, config))
    all_patterns.extend(detect_wedge(df, config))
    all_patterns.extend(detect_triangle(df, config))
    all_patterns.extend(detect_gaps(df))

    # Sort by end index (most recent first)
    all_patterns.sort(key=lambda p: p.end_index, reverse=True)

    return all_patterns


# =============================================================================
# PATTERNS CONFIG REGISTRY
# =============================================================================


PATTERNS_CONFIG: dict[str, dict] = {
    "Head and Shoulders": {
        "type": PatternType.HEAD_AND_SHOULDERS,
        "direction": "bearish",
        "category": "reversal",
        "desc": "Classic bearish reversal pattern with three peaks",
    },
    "Inverse Head and Shoulders": {
        "type": PatternType.INVERSE_HEAD_AND_SHOULDERS,
        "direction": "bullish",
        "category": "reversal",
        "desc": "Classic bullish reversal pattern with three troughs",
    },
    "Double Top": {
        "type": PatternType.DOUBLE_TOP,
        "direction": "bearish",
        "category": "reversal",
        "desc": "Two peaks at similar levels indicating resistance",
    },
    "Double Bottom": {
        "type": PatternType.DOUBLE_BOTTOM,
        "direction": "bullish",
        "category": "reversal",
        "desc": "Two troughs at similar levels indicating support",
    },
    "Triple Top": {
        "type": PatternType.TRIPLE_TOP,
        "direction": "bearish",
        "category": "reversal",
        "desc": "Three peaks at similar levels",
    },
    "Triple Bottom": {
        "type": PatternType.TRIPLE_BOTTOM,
        "direction": "bullish",
        "category": "reversal",
        "desc": "Three troughs at similar levels",
    },
    "Cup and Handle": {
        "type": PatternType.CUP_AND_HANDLE,
        "direction": "bullish",
        "category": "continuation",
        "desc": "U-shaped recovery with small consolidation",
    },
    "Flag": {
        "type": PatternType.FLAG,
        "direction": "continuation",
        "category": "continuation",
        "desc": "Parallel channel consolidation after strong move",
    },
    "Pennant": {
        "type": PatternType.PENNANT,
        "direction": "continuation",
        "category": "continuation",
        "desc": "Converging triangle after strong move",
    },
    "Rising Wedge": {
        "type": PatternType.WEDGE_RISING,
        "direction": "bearish",
        "category": "reversal",
        "desc": "Converging upward pattern, typically bearish",
    },
    "Falling Wedge": {
        "type": PatternType.WEDGE_FALLING,
        "direction": "bullish",
        "category": "reversal",
        "desc": "Converging downward pattern, typically bullish",
    },
    "Ascending Triangle": {
        "type": PatternType.TRIANGLE_ASCENDING,
        "direction": "bullish",
        "category": "continuation",
        "desc": "Flat top with rising bottom, bullish breakout",
    },
    "Descending Triangle": {
        "type": PatternType.TRIANGLE_DESCENDING,
        "direction": "bearish",
        "category": "continuation",
        "desc": "Falling top with flat bottom, bearish breakout",
    },
    "Symmetrical Triangle": {
        "type": PatternType.TRIANGLE_SYMMETRICAL,
        "direction": "neutral",
        "category": "continuation",
        "desc": "Converging from both sides, breakout direction unclear",
    },
    "Gap Patterns": {
        "type": PatternType.GAP_BREAKAWAY,
        "direction": "continuation",
        "category": "gap",
        "desc": "Price gaps indicating momentum",
    },
}
