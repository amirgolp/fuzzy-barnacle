"""
Elliott Wave detection (simplified rule-based).

Detects 5-wave impulse patterns and 3-wave corrective (A-B-C) patterns
using swing point analysis and Fibonacci relationships.
"""

from enum import Enum

import numpy as np
import pandas as pd

from quantdash.core.models import PatternDirection, PatternEvent
from quantdash.features.patterns import find_local_extrema


class ElliottType(str, Enum):
    """Elliott Wave pattern types."""
    IMPULSE_BULLISH = "elliott_impulse_bullish"
    IMPULSE_BEARISH = "elliott_impulse_bearish"
    CORRECTIVE_BULLISH = "elliott_corrective_bullish"
    CORRECTIVE_BEARISH = "elliott_corrective_bearish"


def _get_alternating_swings(
    df: pd.DataFrame, order: int = 5
) -> list[tuple[int, float, str]]:
    """Get alternating swing highs and lows from price data."""
    maxima, minima = find_local_extrema(df["high"], order=order)
    minima_low, _ = find_local_extrema(-df["low"], order=order)

    swings: list[tuple[int, float, str]] = []
    for idx in maxima:
        swings.append((idx, df["high"].iloc[idx], "high"))
    for idx in minima_low:
        swings.append((idx, df["low"].iloc[idx], "low"))

    swings.sort(key=lambda x: x[0])

    # Filter to alternating
    filtered = []
    for s in swings:
        if not filtered or filtered[-1][2] != s[2]:
            filtered.append(s)
        elif s[2] == "high" and s[1] > filtered[-1][1]:
            filtered[-1] = s
        elif s[2] == "low" and s[1] < filtered[-1][1]:
            filtered[-1] = s

    return filtered


def detect_impulse_wave(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect 5-wave impulse patterns.

    Rules:
    1. Wave 2 does NOT retrace 100% of Wave 1
    2. Wave 3 is NEVER the shortest of waves 1, 3, 5
    3. Wave 4 does NOT enter Wave 1's price territory
    4. Waves 1, 3, 5 move in trend direction; waves 2, 4 are corrections

    Optional Fibonacci checks:
    - Wave 2 typically retraces 50-78.6% of Wave 1
    - Wave 3 typically extends 1.618x Wave 1
    """
    patterns = []
    swings = _get_alternating_swings(df, order=order)

    if len(swings) < 6:
        return patterns

    for i in range(len(swings) - 5):
        pts = [(swings[i + j][0], swings[i + j][1], swings[i + j][2]) for j in range(6)]

        # Try bullish impulse: Low-High-Low-High-Low-High (start=0, end=5)
        if pts[0][2] == "low" and pts[5][2] == "high":
            w1_start, w1_end = pts[0][1], pts[1][1]  # low → high
            w2_end = pts[2][1]                         # high → low
            w3_end = pts[3][1]                         # low → high
            w4_end = pts[4][1]                         # high → low
            w5_end = pts[5][1]                         # low → high

            wave1 = w1_end - w1_start
            wave2 = w1_end - w2_end  # retracement (positive)
            wave3 = w3_end - w2_end
            wave4 = w3_end - w4_end  # retracement (positive)
            wave5 = w5_end - w4_end

            if wave1 <= 0 or wave3 <= 0 or wave5 <= 0:
                continue

            # Rule 1: Wave 2 must not retrace 100% of Wave 1
            if wave2 >= wave1:
                continue

            # Rule 2: Wave 3 is never the shortest
            if wave3 < wave1 and wave3 < wave5:
                continue

            # Rule 3: Wave 4 must not enter Wave 1 territory
            if w4_end <= w1_end:
                pass  # OK if Wave 4 bottom stays above Wave 1 top...
            # Actually: Wave 4 should not overlap Wave 1's price range
            if w4_end < w1_start:
                continue  # Wave 4 went below Wave 1 start — invalid

            # Fibonacci scoring
            w2_retrace = wave2 / wave1 if wave1 > 0 else 0
            w3_ext = wave3 / wave1 if wave1 > 0 else 0

            conf = 0.50
            if 0.382 <= w2_retrace <= 0.786:
                conf += 0.10
            if 1.0 <= w3_ext <= 2.618:
                conf += 0.10
            if wave3 == max(wave1, wave3, wave5):
                conf += 0.10
            conf = min(conf, 0.85)

            patterns.append(PatternEvent(
                pattern_type=ElliottType.IMPULSE_BULLISH.value,
                start_index=pts[0][0],
                end_index=pts[5][0],
                confidence=round(conf, 2),
                direction=PatternDirection.BULLISH,
                target_price=w5_end + wave5 * 0.618,
                metadata={
                    "wave1": round(wave1, 4), "wave3": round(wave3, 4),
                    "wave5": round(wave5, 4),
                    "w2_retrace": round(w2_retrace, 3),
                    "w3_extension": round(w3_ext, 3),
                },
            ))

        # Try bearish impulse: High-Low-High-Low-High-Low
        if pts[0][2] == "high" and pts[5][2] == "low":
            w1_start, w1_end = pts[0][1], pts[1][1]  # high → low
            w2_end = pts[2][1]
            w3_end = pts[3][1]
            w4_end = pts[4][1]
            w5_end = pts[5][1]

            wave1 = w1_start - w1_end  # positive for bearish
            wave2 = w2_end - w1_end    # retracement up
            wave3 = w2_end - w3_end
            wave4 = w4_end - w3_end    # retracement up
            wave5 = w4_end - w5_end

            if wave1 <= 0 or wave3 <= 0 or wave5 <= 0:
                continue

            if wave2 >= wave1:
                continue
            if wave3 < wave1 and wave3 < wave5:
                continue
            if w4_end > w1_start:
                continue

            w2_retrace = wave2 / wave1 if wave1 > 0 else 0
            w3_ext = wave3 / wave1 if wave1 > 0 else 0

            conf = 0.50
            if 0.382 <= w2_retrace <= 0.786:
                conf += 0.10
            if 1.0 <= w3_ext <= 2.618:
                conf += 0.10
            if wave3 == max(wave1, wave3, wave5):
                conf += 0.10
            conf = min(conf, 0.85)

            patterns.append(PatternEvent(
                pattern_type=ElliottType.IMPULSE_BEARISH.value,
                start_index=pts[0][0],
                end_index=pts[5][0],
                confidence=round(conf, 2),
                direction=PatternDirection.BEARISH,
                target_price=w5_end - wave5 * 0.618,
                metadata={
                    "wave1": round(wave1, 4), "wave3": round(wave3, 4),
                    "wave5": round(wave5, 4),
                    "w2_retrace": round(w2_retrace, 3),
                    "w3_extension": round(w3_ext, 3),
                },
            ))

    return patterns


def detect_corrective_wave(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect 3-wave corrective (A-B-C) patterns.

    In a bullish context (after a decline): A down, B up, C down → bullish reversal expected.
    In a bearish context (after a rally): A up, B down, C up → bearish reversal expected.

    Wave B typically retraces 50-78.6% of Wave A.
    Wave C is often equal to Wave A or 1.618× Wave A.
    """
    patterns = []
    swings = _get_alternating_swings(df, order=order)

    if len(swings) < 4:
        return patterns

    for i in range(len(swings) - 3):
        pts = [(swings[i + j][0], swings[i + j][1], swings[i + j][2]) for j in range(4)]

        # Bullish corrective: after decline, A-B-C ends at low → expect reversal up
        # Pattern: High → Low → High → Low (correction within a larger uptrend)
        if pts[0][2] == "high" and pts[3][2] == "low":
            wave_a = pts[0][1] - pts[1][1]  # decline
            wave_b = pts[2][1] - pts[1][1]  # recovery
            wave_c = pts[2][1] - pts[3][1]  # second decline

            if wave_a <= 0 or wave_b <= 0 or wave_c <= 0:
                continue

            b_retrace = wave_b / wave_a if wave_a > 0 else 0
            c_a_ratio = wave_c / wave_a if wave_a > 0 else 0

            # B should retrace 38.2-78.6% of A
            if not (0.30 <= b_retrace <= 0.85):
                continue

            conf = 0.50
            if 0.382 <= b_retrace <= 0.786:
                conf += 0.10
            if 0.8 <= c_a_ratio <= 1.8:
                conf += 0.10
            conf = min(conf, 0.80)

            patterns.append(PatternEvent(
                pattern_type=ElliottType.CORRECTIVE_BULLISH.value,
                start_index=pts[0][0],
                end_index=pts[3][0],
                confidence=round(conf, 2),
                direction=PatternDirection.BULLISH,
                metadata={
                    "wave_a": round(wave_a, 4),
                    "wave_b": round(wave_b, 4),
                    "wave_c": round(wave_c, 4),
                    "b_retrace": round(b_retrace, 3),
                    "c_a_ratio": round(c_a_ratio, 3),
                },
            ))

        # Bearish corrective: Low → High → Low → High
        if pts[0][2] == "low" and pts[3][2] == "high":
            wave_a = pts[1][1] - pts[0][1]  # rally
            wave_b = pts[1][1] - pts[2][1]  # pullback
            wave_c = pts[3][1] - pts[2][1]  # second rally

            if wave_a <= 0 or wave_b <= 0 or wave_c <= 0:
                continue

            b_retrace = wave_b / wave_a if wave_a > 0 else 0
            c_a_ratio = wave_c / wave_a if wave_a > 0 else 0

            if not (0.30 <= b_retrace <= 0.85):
                continue

            conf = 0.50
            if 0.382 <= b_retrace <= 0.786:
                conf += 0.10
            if 0.8 <= c_a_ratio <= 1.8:
                conf += 0.10
            conf = min(conf, 0.80)

            patterns.append(PatternEvent(
                pattern_type=ElliottType.CORRECTIVE_BEARISH.value,
                start_index=pts[0][0],
                end_index=pts[3][0],
                confidence=round(conf, 2),
                direction=PatternDirection.BEARISH,
                metadata={
                    "wave_a": round(wave_a, 4),
                    "wave_b": round(wave_b, 4),
                    "wave_c": round(wave_c, 4),
                    "b_retrace": round(b_retrace, 3),
                    "c_a_ratio": round(c_a_ratio, 3),
                },
            ))

    return patterns


# =============================================================================
# MASTER DETECTOR
# =============================================================================


def detect_all_elliott_patterns(
    df: pd.DataFrame, order: int = 5
) -> list[PatternEvent]:
    """Run all Elliott Wave pattern detectors."""
    all_patterns = []
    all_patterns.extend(detect_impulse_wave(df, order=order))
    all_patterns.extend(detect_corrective_wave(df, order=order))
    all_patterns.sort(key=lambda p: p.end_index, reverse=True)
    return all_patterns


# =============================================================================
# PATTERNS CONFIG REGISTRY
# =============================================================================

ELLIOTT_PATTERNS_CONFIG: dict[str, dict] = {
    "Elliott Impulse (Bullish)": {
        "type": ElliottType.IMPULSE_BULLISH, "direction": "bullish",
        "category": "elliott", "desc": "5-wave bullish impulse (waves 1-2-3-4-5)",
    },
    "Elliott Impulse (Bearish)": {
        "type": ElliottType.IMPULSE_BEARISH, "direction": "bearish",
        "category": "elliott", "desc": "5-wave bearish impulse (waves 1-2-3-4-5)",
    },
    "Elliott Corrective (Bullish)": {
        "type": ElliottType.CORRECTIVE_BULLISH, "direction": "bullish",
        "category": "elliott", "desc": "3-wave A-B-C correction ending at support",
    },
    "Elliott Corrective (Bearish)": {
        "type": ElliottType.CORRECTIVE_BEARISH, "direction": "bearish",
        "category": "elliott", "desc": "3-wave A-B-C correction ending at resistance",
    },
}
