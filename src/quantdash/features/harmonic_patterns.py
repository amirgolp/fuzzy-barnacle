"""
Harmonic pattern detection algorithms.

Implements detection for Fibonacci ratio-based harmonic patterns (Gartley, Butterfly,
Bat, Crab, Shark, Cypher, AB=CD, Three Drives, Wolfe Wave).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quantdash.core.models import PatternDirection, PatternEvent
from quantdash.features.patterns import find_local_extrema


class HarmonicType(str, Enum):
    """Supported harmonic patterns."""
    GARTLEY_BULLISH = "gartley_bullish"
    GARTLEY_BEARISH = "gartley_bearish"
    BUTTERFLY_BULLISH = "butterfly_bullish"
    BUTTERFLY_BEARISH = "butterfly_bearish"
    BAT_BULLISH = "bat_bullish"
    BAT_BEARISH = "bat_bearish"
    CRAB_BULLISH = "crab_bullish"
    CRAB_BEARISH = "crab_bearish"
    SHARK_BULLISH = "shark_bullish"
    SHARK_BEARISH = "shark_bearish"
    CYPHER_BULLISH = "cypher_bullish"
    CYPHER_BEARISH = "cypher_bearish"
    ABCD_BULLISH = "abcd_bullish"
    ABCD_BEARISH = "abcd_bearish"
    THREE_DRIVES_BULLISH = "three_drives_bullish"
    THREE_DRIVES_BEARISH = "three_drives_bearish"
    WOLFE_WAVE_BULLISH = "wolfe_wave_bullish"
    WOLFE_WAVE_BEARISH = "wolfe_wave_bearish"


@dataclass
class FibRatios:
    """Fibonacci ratio requirements for a harmonic pattern."""
    name: str
    ab_xa: tuple[float, float]  # (min, max) retracement of XA
    bc_ab: tuple[float, float]  # (min, max) retracement of AB
    cd_bc: tuple[float, float]  # (min, max) extension of BC
    xd_xa: tuple[float, float]  # (min, max) retracement/extension of XA


# Ratio definitions per pattern
HARMONIC_RATIOS: dict[str, FibRatios] = {
    "gartley": FibRatios("gartley", (0.58, 0.66), (0.38, 0.89), (1.27, 1.62), (0.75, 0.82)),
    "butterfly": FibRatios("butterfly", (0.74, 0.82), (0.38, 0.89), (1.62, 2.62), (1.27, 1.62)),
    "bat": FibRatios("bat", (0.38, 0.52), (0.38, 0.89), (1.62, 2.62), (0.85, 0.92)),
    "crab": FibRatios("crab", (0.38, 0.62), (0.38, 0.89), (2.24, 3.62), (1.58, 1.68)),
    "shark": FibRatios("shark", (0.38, 0.62), (1.13, 1.62), (1.62, 2.24), (0.88, 1.13)),
    "cypher": FibRatios("cypher", (0.38, 0.62), (1.13, 1.42), (0.38, 0.62), (0.75, 0.82)),
}

TOLERANCE = 0.05  # 5% tolerance on Fibonacci ratios


def _ratio_in_range(ratio: float, expected: tuple[float, float]) -> bool:
    """Check if a ratio falls within expected range (with tolerance)."""
    low = expected[0] - TOLERANCE
    high = expected[1] + TOLERANCE
    return low <= ratio <= high


def _ratio_confidence(ratio: float, expected: tuple[float, float]) -> float:
    """Calculate confidence based on how close ratio is to ideal midpoint."""
    mid = (expected[0] + expected[1]) / 2
    max_dev = (expected[1] - expected[0]) / 2 + TOLERANCE
    dev = abs(ratio - mid)
    return max(0.0, 1.0 - dev / max_dev) if max_dev > 0 else 0.0


def _get_swing_points(df: pd.DataFrame, order: int = 5) -> list[tuple[int, float, str]]:
    """
    Get alternating swing highs and lows.

    Returns list of (index, price, 'high'|'low') sorted by index.
    """
    maxima, minima = find_local_extrema(df["high"], order=order)
    minima_low, _ = find_local_extrema(-df["low"], order=order)  # invert for minima on low

    swings: list[tuple[int, float, str]] = []
    for idx in maxima:
        swings.append((idx, df["high"].iloc[idx], "high"))
    for idx in minima_low:
        swings.append((idx, df["low"].iloc[idx], "low"))

    swings.sort(key=lambda x: x[0])

    # Filter to alternating high/low
    filtered = []
    for s in swings:
        if not filtered or filtered[-1][2] != s[2]:
            filtered.append(s)
        elif s[2] == "high" and s[1] > filtered[-1][1]:
            filtered[-1] = s  # Keep higher high
        elif s[2] == "low" and s[1] < filtered[-1][1]:
            filtered[-1] = s  # Keep lower low

    return filtered


def detect_xabcd_patterns(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect XABCD harmonic patterns (Gartley, Butterfly, Bat, Crab, Shark, Cypher).

    Scans swing points for 5-point sequences matching Fibonacci ratio requirements.
    """
    patterns = []
    swings = _get_swing_points(df, order=order)

    if len(swings) < 5:
        return patterns

    for i in range(len(swings) - 4):
        x_idx, x_price, x_type = swings[i]
        a_idx, a_price, a_type = swings[i + 1]
        b_idx, b_price, b_type = swings[i + 2]
        c_idx, c_price, c_type = swings[i + 3]
        d_idx, d_price, d_type = swings[i + 4]

        xa = abs(a_price - x_price)
        ab = abs(b_price - a_price)
        bc = abs(c_price - b_price)
        cd = abs(d_price - c_price)

        if xa < 1e-10 or ab < 1e-10 or bc < 1e-10:
            continue

        ab_xa_ratio = ab / xa
        bc_ab_ratio = bc / ab
        cd_bc_ratio = cd / bc if bc > 1e-10 else 0
        xd_xa_ratio = abs(d_price - x_price) / xa

        # Determine direction
        is_bullish = x_type == "high"  # X is high, D is low → bullish reversal

        for name, ratios in HARMONIC_RATIOS.items():
            if (_ratio_in_range(ab_xa_ratio, ratios.ab_xa)
                    and _ratio_in_range(bc_ab_ratio, ratios.bc_ab)
                    and _ratio_in_range(cd_bc_ratio, ratios.cd_bc)
                    and _ratio_in_range(xd_xa_ratio, ratios.xd_xa)):

                # Calculate confidence as average of how well each ratio matches
                conf = (
                    _ratio_confidence(ab_xa_ratio, ratios.ab_xa)
                    + _ratio_confidence(bc_ab_ratio, ratios.bc_ab)
                    + _ratio_confidence(cd_bc_ratio, ratios.cd_bc)
                    + _ratio_confidence(xd_xa_ratio, ratios.xd_xa)
                ) / 4
                conf = max(0.5, min(0.90, 0.5 + conf * 0.4))

                direction = PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH
                ptype = f"{name}_{'bullish' if is_bullish else 'bearish'}"

                # Target: CD leg reversal
                if is_bullish:
                    target = d_price + xa * 0.618
                    stop = d_price - xa * 0.13
                else:
                    target = d_price - xa * 0.618
                    stop = d_price + xa * 0.13

                patterns.append(PatternEvent(
                    pattern_type=ptype,
                    start_index=x_idx,
                    end_index=d_idx,
                    confidence=round(conf, 2),
                    direction=direction,
                    target_price=target,
                    stop_price=stop,
                    metadata={
                        "X": x_price, "A": a_price, "B": b_price,
                        "C": c_price, "D": d_price,
                        "AB/XA": round(ab_xa_ratio, 3),
                        "BC/AB": round(bc_ab_ratio, 3),
                        "CD/BC": round(cd_bc_ratio, 3),
                        "XD/XA": round(xd_xa_ratio, 3),
                    },
                ))

    return patterns


def detect_abcd(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect AB=CD patterns.

    AB and CD legs should be approximately equal in length and/or time.
    """
    patterns = []
    swings = _get_swing_points(df, order=order)

    if len(swings) < 4:
        return patterns

    for i in range(len(swings) - 3):
        a_idx, a_price, a_type = swings[i]
        b_idx, b_price, _ = swings[i + 1]
        c_idx, c_price, _ = swings[i + 2]
        d_idx, d_price, d_type = swings[i + 3]

        ab = abs(b_price - a_price)
        cd = abs(d_price - c_price)

        if ab < 1e-10:
            continue

        # AB ≈ CD (within 15% tolerance)
        ratio = cd / ab
        if 0.85 <= ratio <= 1.15:
            is_bullish = d_type == "low"
            direction = PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH
            conf = max(0.55, 0.75 - abs(ratio - 1.0) * 2)

            patterns.append(PatternEvent(
                pattern_type=HarmonicType.ABCD_BULLISH.value if is_bullish else HarmonicType.ABCD_BEARISH.value,
                start_index=a_idx,
                end_index=d_idx,
                confidence=round(conf, 2),
                direction=direction,
                metadata={"AB": ab, "CD": cd, "ratio": round(ratio, 3)},
            ))

    return patterns


def detect_three_drives(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect Three Drives pattern.

    Three successive pushes in the same direction, each approximately 1.272 or 1.618
    extension of the previous correction.
    """
    patterns = []
    swings = _get_swing_points(df, order=order)

    if len(swings) < 6:
        return patterns

    for i in range(len(swings) - 5):
        points = [swings[i + j] for j in range(6)]

        # Check alternating: all odd same type, all even same type
        types = [p[2] for p in points]
        if not (types[0] == types[2] == types[4] and types[1] == types[3] == types[5]):
            continue

        # Three drives should be progressively reaching
        drives = [points[1][1], points[3][1], points[5][1]]
        corrections = [points[0][1], points[2][1], points[4][1]]

        is_bullish_drives = types[1] == "high"

        if is_bullish_drives:
            # Three drives up
            if not (drives[0] < drives[1] < drives[2]):
                continue
        else:
            # Three drives down
            if not (drives[0] > drives[1] > drives[2]):
                continue

        # Check extension ratios (~1.272 or ~1.618)
        drive1 = abs(drives[0] - corrections[0])
        correction1 = abs(corrections[1] - drives[0])
        drive2 = abs(drives[1] - corrections[1])

        if correction1 < 1e-10 or drive1 < 1e-10:
            continue

        ext1 = drive2 / correction1
        if not (1.1 <= ext1 <= 1.8):
            continue

        direction = PatternDirection.BEARISH if is_bullish_drives else PatternDirection.BULLISH

        patterns.append(PatternEvent(
            pattern_type=(HarmonicType.THREE_DRIVES_BEARISH.value if is_bullish_drives
                          else HarmonicType.THREE_DRIVES_BULLISH.value),
            start_index=points[0][0],
            end_index=points[5][0],
            confidence=0.65,
            direction=direction,
            metadata={"extension_ratio": round(ext1, 3)},
        ))

    return patterns


def detect_wolfe_wave(df: pd.DataFrame, order: int = 5) -> list[PatternEvent]:
    """
    Detect Wolfe Wave pattern (5-point channel pattern).

    Points 1-3-5 on one trendline, points 2-4 on another.
    Point 5 exceeds the 1-3 trendline slightly (sweet spot entry).
    """
    patterns = []
    swings = _get_swing_points(df, order=order)

    if len(swings) < 5:
        return patterns

    for i in range(len(swings) - 4):
        pts = [(swings[i + j][0], swings[i + j][1]) for j in range(5)]
        types = [swings[i + j][2] for j in range(5)]

        # Must alternate
        if not (types[0] == types[2] == types[4] and types[1] == types[3]):
            continue

        # Check 1-3-5 roughly collinear (on a trendline)
        idx1, p1 = pts[0]
        idx3, p3 = pts[2]
        idx5, p5 = pts[4]

        if idx5 == idx1:
            continue

        slope_135 = (p3 - p1) / (idx3 - idx1) if idx3 != idx1 else 0
        expected_p5 = p1 + slope_135 * (idx5 - idx1)
        deviation = abs(p5 - expected_p5) / abs(p3 - p1) if abs(p3 - p1) > 1e-10 else float("inf")

        # Point 5 should be near or slightly beyond the 1-3 line (< 20% deviation)
        if deviation > 0.20:
            continue

        # Check 2-4 form a channel line
        idx2, p2 = pts[1]
        idx4, p4 = pts[3]
        slope_24 = (p4 - p2) / (idx4 - idx2) if idx4 != idx2 else 0

        # Slopes should be roughly parallel (or converging)
        is_bullish = types[0] == "low"
        direction = PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH

        # Target: project 1-4 line to the right
        target = p4 + slope_24 * (idx5 - idx4)

        patterns.append(PatternEvent(
            pattern_type=HarmonicType.WOLFE_WAVE_BULLISH.value if is_bullish else HarmonicType.WOLFE_WAVE_BEARISH.value,
            start_index=idx1,
            end_index=idx5,
            confidence=0.60,
            direction=direction,
            target_price=target,
            metadata={"deviation_pct": round(deviation * 100, 1)},
        ))

    return patterns


# =============================================================================
# MASTER DETECTOR
# =============================================================================


def detect_all_harmonic_patterns(
    df: pd.DataFrame, order: int = 5
) -> list[PatternEvent]:
    """Run all harmonic pattern detectors."""
    all_patterns = []
    all_patterns.extend(detect_xabcd_patterns(df, order=order))
    all_patterns.extend(detect_abcd(df, order=order))
    all_patterns.extend(detect_three_drives(df, order=order))
    all_patterns.extend(detect_wolfe_wave(df, order=order))
    all_patterns.sort(key=lambda p: p.end_index, reverse=True)
    return all_patterns


# =============================================================================
# PATTERNS CONFIG REGISTRY
# =============================================================================

HARMONIC_PATTERNS_CONFIG: dict[str, dict] = {
    "Gartley": {"type": "gartley", "direction": "reversal", "category": "harmonic", "desc": "0.786 XD retracement, classic harmonic pattern"},
    "Butterfly": {"type": "butterfly", "direction": "reversal", "category": "harmonic", "desc": "1.27 XD extension, extreme reversal pattern"},
    "Bat": {"type": "bat", "direction": "reversal", "category": "harmonic", "desc": "0.886 XD retracement, precise reversal pattern"},
    "Crab": {"type": "crab", "direction": "reversal", "category": "harmonic", "desc": "1.618 XD extension, deep reversal pattern"},
    "Shark": {"type": "shark", "direction": "reversal", "category": "harmonic", "desc": "0/5 pattern, newer harmonic variant"},
    "Cypher": {"type": "cypher", "direction": "reversal", "category": "harmonic", "desc": "0.786 XD with 1.13-1.414 BC extension"},
    "AB=CD": {"type": "abcd", "direction": "reversal", "category": "harmonic", "desc": "Equal AB and CD legs, simplest harmonic"},
    "Three Drives": {"type": "three_drives", "direction": "reversal", "category": "harmonic", "desc": "Three successive drives with Fibonacci extensions"},
    "Wolfe Wave": {"type": "wolfe_wave", "direction": "reversal", "category": "harmonic", "desc": "Five-point channel pattern for reversal trading"},
}
