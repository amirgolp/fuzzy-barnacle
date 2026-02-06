"""
Advanced chart pattern detection algorithms.

Implements detection for Quasimodo, Dead Cat Bounce, Island Reversal,
Tower Top/Bottom, and other advanced patterns.
"""

from enum import Enum

import numpy as np
import pandas as pd

from quantdash.core.models import PatternDirection, PatternEvent
from quantdash.features.patterns import find_local_extrema, prices_equal


class AdvancedPatternType(str, Enum):
    """Supported advanced patterns."""
    QUASIMODO_BULLISH = "quasimodo_bullish"
    QUASIMODO_BEARISH = "quasimodo_bearish"
    DEAD_CAT_BOUNCE = "dead_cat_bounce"
    ISLAND_REVERSAL_BULLISH = "island_reversal_bullish"
    ISLAND_REVERSAL_BEARISH = "island_reversal_bearish"
    TOWER_TOP = "tower_top"
    TOWER_BOTTOM = "tower_bottom"


def detect_quasimodo(
    df: pd.DataFrame, order: int = 5
) -> list[PatternEvent]:
    """
    Detect Quasimodo (QM) / Over-Under pattern.

    Bearish QM: Higher High followed by Lower Low (structure break).
    Bullish QM: Lower Low followed by Higher High (structure break).
    """
    patterns = []
    high = df["high"]
    low = df["low"]
    maxima, minima = find_local_extrema(high, order=order)

    # Also find minima on the low series
    minima_low, _ = find_local_extrema(-low, order=order)

    # Bearish Quasimodo: H → HH → LL (failed higher high)
    for i in range(len(maxima) - 1):
        h1_idx = maxima[i]
        h2_idx = maxima[i + 1]

        # Higher high
        if high.iloc[h2_idx] <= high.iloc[h1_idx]:
            continue

        # Find lows between and after
        lows_between = [m for m in minima_low if h1_idx < m < h2_idx]
        lows_after = [m for m in minima_low if m > h2_idx]

        if not lows_between or not lows_after:
            continue

        l1_idx = lows_between[0]
        l2_idx = lows_after[0]

        # Lower low after the higher high → structure break
        if low.iloc[l2_idx] < low.iloc[l1_idx]:
            pattern_height = high.iloc[h2_idx] - low.iloc[l2_idx]
            target = low.iloc[l2_idx] - pattern_height * 0.5
            conf = min(0.80, 0.55 + (high.iloc[h2_idx] - high.iloc[h1_idx]) / high.iloc[h1_idx] * 5)

            patterns.append(PatternEvent(
                pattern_type=AdvancedPatternType.QUASIMODO_BEARISH.value,
                start_index=h1_idx,
                end_index=l2_idx,
                confidence=round(conf, 2),
                direction=PatternDirection.BEARISH,
                target_price=target,
                stop_price=high.iloc[h2_idx],
            ))

    # Bullish Quasimodo: L → LL → HH (failed lower low)
    for i in range(len(minima_low) - 1):
        l1_idx = minima_low[i]
        l2_idx = minima_low[i + 1]

        if low.iloc[l2_idx] >= low.iloc[l1_idx]:
            continue

        highs_between = [m for m in maxima if l1_idx < m < l2_idx]
        highs_after = [m for m in maxima if m > l2_idx]

        if not highs_between or not highs_after:
            continue

        h1_idx = highs_between[0]
        h2_idx = highs_after[0]

        if high.iloc[h2_idx] > high.iloc[h1_idx]:
            pattern_height = high.iloc[h2_idx] - low.iloc[l2_idx]
            target = high.iloc[h2_idx] + pattern_height * 0.5
            conf = min(0.80, 0.55 + (low.iloc[l1_idx] - low.iloc[l2_idx]) / low.iloc[l1_idx] * 5)

            patterns.append(PatternEvent(
                pattern_type=AdvancedPatternType.QUASIMODO_BULLISH.value,
                start_index=l1_idx,
                end_index=h2_idx,
                confidence=round(conf, 2),
                direction=PatternDirection.BULLISH,
                target_price=target,
                stop_price=low.iloc[l2_idx],
            ))

    return patterns


def detect_dead_cat_bounce(
    df: pd.DataFrame,
    decline_threshold: float = 0.05,
    fib_max: float = 0.382,
) -> list[PatternEvent]:
    """
    Detect Dead Cat Bounce pattern.

    Sharp decline (> threshold), followed by a small recovery (< fib_max retracement),
    followed by continued decline.
    """
    patterns = []
    close = df["close"]
    high = df["high"]
    low = df["low"]

    for i in range(10, len(df) - 10):
        # Look for sharp decline ending at bar i
        lookback_prices = close.iloc[max(0, i - 10):i + 1]
        decline = (lookback_prices.iloc[0] - lookback_prices.iloc[-1]) / lookback_prices.iloc[0]

        if decline < decline_threshold:
            continue

        # Look for bounce
        bounce_window = df.iloc[i:min(i + 10, len(df))]
        if len(bounce_window) < 3:
            continue

        bounce_high = bounce_window["high"].max()
        bounce_recovery = (bounce_high - close.iloc[i]) / (lookback_prices.iloc[0] - close.iloc[i])

        if bounce_recovery > fib_max or bounce_recovery < 0.05:
            continue

        # Check for continued decline after bounce
        bounce_high_idx = bounce_window["high"].idxmax()
        if not isinstance(bounce_high_idx, int):
            bounce_high_bar = bounce_window.index.get_loc(bounce_high_idx)
        else:
            bounce_high_bar = bounce_high_idx - i

        post_bounce_start = i + bounce_high_bar + 1
        if post_bounce_start >= len(df) - 2:
            continue

        post_bounce = close.iloc[post_bounce_start:min(post_bounce_start + 5, len(df))]
        if len(post_bounce) < 2:
            continue

        if post_bounce.iloc[-1] < close.iloc[i]:
            conf = min(0.75, 0.50 + decline * 3)
            patterns.append(PatternEvent(
                pattern_type=AdvancedPatternType.DEAD_CAT_BOUNCE.value,
                start_index=max(0, i - 10),
                end_index=min(post_bounce_start + len(post_bounce) - 1, len(df) - 1),
                confidence=round(conf, 2),
                direction=PatternDirection.BEARISH,
                metadata={
                    "decline_pct": round(decline * 100, 1),
                    "recovery_pct": round(bounce_recovery * 100, 1),
                },
            ))

    return patterns


def detect_island_reversal(
    df: pd.DataFrame, min_gap_pct: float = 0.005
) -> list[PatternEvent]:
    """
    Detect Island Reversal pattern.

    A group of bars isolated by gaps on both sides (gap up then gap down, or vice versa).
    """
    patterns = []
    high = df["high"].values
    low = df["low"].values

    # Find all gaps
    gap_ups = []   # (index, gap_size)
    gap_downs = []

    for i in range(1, len(df)):
        prev_high = high[i - 1]
        curr_low = low[i]
        prev_low = low[i - 1]
        curr_high = high[i]

        if curr_low > prev_high:
            gap_ups.append((i, (curr_low - prev_high) / prev_high))
        elif curr_high < prev_low:
            gap_downs.append((i, (prev_low - curr_high) / prev_low))

    # Bearish island: gap up followed by gap down
    for gu_idx, gu_size in gap_ups:
        if gu_size < min_gap_pct:
            continue
        for gd_idx, gd_size in gap_downs:
            if gd_size < min_gap_pct:
                continue
            if gd_idx <= gu_idx:
                continue
            # Island should be relatively short (1-10 bars)
            island_len = gd_idx - gu_idx
            if 1 <= island_len <= 10:
                conf = min(0.80, 0.55 + (gu_size + gd_size) * 10)
                patterns.append(PatternEvent(
                    pattern_type=AdvancedPatternType.ISLAND_REVERSAL_BEARISH.value,
                    start_index=gu_idx,
                    end_index=gd_idx,
                    confidence=round(conf, 2),
                    direction=PatternDirection.BEARISH,
                    metadata={"island_bars": island_len},
                ))
                break  # Match first gap down after gap up

    # Bullish island: gap down followed by gap up
    for gd_idx, gd_size in gap_downs:
        if gd_size < min_gap_pct:
            continue
        for gu_idx, gu_size in gap_ups:
            if gu_size < min_gap_pct:
                continue
            if gu_idx <= gd_idx:
                continue
            island_len = gu_idx - gd_idx
            if 1 <= island_len <= 10:
                conf = min(0.80, 0.55 + (gu_size + gd_size) * 10)
                patterns.append(PatternEvent(
                    pattern_type=AdvancedPatternType.ISLAND_REVERSAL_BULLISH.value,
                    start_index=gd_idx,
                    end_index=gu_idx,
                    confidence=round(conf, 2),
                    direction=PatternDirection.BULLISH,
                    metadata={"island_bars": island_len},
                ))
                break

    return patterns


def detect_tower(
    df: pd.DataFrame,
    strong_move_pct: float = 0.03,
    consolidation_bars: int = 3,
) -> list[PatternEvent]:
    """
    Detect Tower Top and Tower Bottom patterns.

    Tower Top: Strong rally → 2-3 bar consolidation at top → strong decline.
    Tower Bottom: Strong decline → 2-3 bar consolidation at bottom → strong rally.
    """
    patterns = []
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for i in range(10, len(df) - consolidation_bars - 5):
        # Check for strong rally before consolidation
        pre_move = (close[i] - close[max(0, i - 5)]) / close[max(0, i - 5)]

        if abs(pre_move) < strong_move_pct:
            continue

        # Check for consolidation (small range)
        consol = df.iloc[i:i + consolidation_bars]
        consol_range = consol["high"].max() - consol["low"].min()
        avg_price = consol["close"].mean()
        consol_pct = consol_range / avg_price

        if consol_pct > strong_move_pct * 0.5:
            continue  # Consolidation too wide

        # Check for strong move after consolidation
        post_start = i + consolidation_bars
        post_end = min(post_start + 5, len(df))
        if post_end <= post_start + 1:
            continue

        post_move = (close[post_end - 1] - close[post_start]) / close[post_start]

        # Tower Top: rally → consolidation → decline
        if pre_move > strong_move_pct and post_move < -strong_move_pct:
            conf = min(0.75, 0.50 + abs(post_move) * 5)
            patterns.append(PatternEvent(
                pattern_type=AdvancedPatternType.TOWER_TOP.value,
                start_index=max(0, i - 5),
                end_index=post_end - 1,
                confidence=round(conf, 2),
                direction=PatternDirection.BEARISH,
                metadata={
                    "rally_pct": round(pre_move * 100, 1),
                    "decline_pct": round(post_move * 100, 1),
                },
            ))

        # Tower Bottom: decline → consolidation → rally
        if pre_move < -strong_move_pct and post_move > strong_move_pct:
            conf = min(0.75, 0.50 + abs(post_move) * 5)
            patterns.append(PatternEvent(
                pattern_type=AdvancedPatternType.TOWER_BOTTOM.value,
                start_index=max(0, i - 5),
                end_index=post_end - 1,
                confidence=round(conf, 2),
                direction=PatternDirection.BULLISH,
                metadata={
                    "decline_pct": round(pre_move * 100, 1),
                    "rally_pct": round(post_move * 100, 1),
                },
            ))

    return patterns


# =============================================================================
# MASTER DETECTOR
# =============================================================================


def detect_all_advanced_patterns(df: pd.DataFrame) -> list[PatternEvent]:
    """Run all advanced pattern detectors."""
    all_patterns = []
    all_patterns.extend(detect_quasimodo(df))
    all_patterns.extend(detect_dead_cat_bounce(df))
    all_patterns.extend(detect_island_reversal(df))
    all_patterns.extend(detect_tower(df))
    all_patterns.sort(key=lambda p: p.end_index, reverse=True)
    return all_patterns


# =============================================================================
# PATTERNS CONFIG REGISTRY
# =============================================================================

ADVANCED_PATTERNS_CONFIG: dict[str, dict] = {
    "Quasimodo (Bullish)": {"type": AdvancedPatternType.QUASIMODO_BULLISH, "direction": "bullish", "category": "advanced", "desc": "Failed lower low followed by higher high breakout"},
    "Quasimodo (Bearish)": {"type": AdvancedPatternType.QUASIMODO_BEARISH, "direction": "bearish", "category": "advanced", "desc": "Failed higher high followed by lower low breakdown"},
    "Dead Cat Bounce": {"type": AdvancedPatternType.DEAD_CAT_BOUNCE, "direction": "bearish", "category": "advanced", "desc": "Sharp decline, weak recovery, continued decline"},
    "Island Reversal (Bullish)": {"type": AdvancedPatternType.ISLAND_REVERSAL_BULLISH, "direction": "bullish", "category": "advanced", "desc": "Bars isolated by gap down then gap up"},
    "Island Reversal (Bearish)": {"type": AdvancedPatternType.ISLAND_REVERSAL_BEARISH, "direction": "bearish", "category": "advanced", "desc": "Bars isolated by gap up then gap down"},
    "Tower Top": {"type": AdvancedPatternType.TOWER_TOP, "direction": "bearish", "category": "advanced", "desc": "Strong rally, consolidation at top, strong decline"},
    "Tower Bottom": {"type": AdvancedPatternType.TOWER_BOTTOM, "direction": "bullish", "category": "advanced", "desc": "Strong decline, consolidation at bottom, strong rally"},
}
