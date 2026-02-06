"""Feature engineering: flatten indicators, compute derivatives, normalize.

Builds the feature matrices consumed by each model branch:
- Price/Indicator branch: OHLCV + returns + accel + flattened indicator cols
- Volume branch: volume + volume_return + volume_accel
- Pattern branch: detected flag + confidence per pattern
- Session/macro features built separately (macro.py)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantdash.features.indicators import INDICATORS_CONFIG, compute_indicator
from quantdash.ml.config import NormalizationConfig


# ── Indicator keys that return dict[str, pd.Series] ──────────────────────────
_MULTI_OUTPUT_KEYS: dict[str, list[str]] = {
    "Ichimoku Cloud": [
        "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span",
    ],
    "MACD": ["macd_line", "signal_line", "histogram"],
    "Bollinger Bands": ["upper", "middle", "lower"],
    "Stochastic Oscillator": ["percent_k", "percent_d"],
    "Pivot Points": ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"],
}

# Indicators to skip for the ML feature matrix
_SKIP_INDICATORS = {"Fibonacci Retracement"}  # returns floats, not per-bar


def flatten_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators from INDICATORS_CONFIG and flatten to columns.

    Multi-output indicators are expanded into separate columns.
    Fibonacci (levels_output) is skipped as it returns constants, not series.

    Returns:
        DataFrame with one column per indicator output, same index as df.
    """
    cols: dict[str, pd.Series] = {}

    for name, config in INDICATORS_CONFIG.items():
        if name in _SKIP_INDICATORS:
            continue

        result = compute_indicator(df, name)

        if isinstance(result, dict):
            # Multi-output indicator
            for key, series in result.items():
                col_name = f"ind_{name.lower().replace(' ', '_')}_{key}"
                if isinstance(series, pd.Series):
                    cols[col_name] = series
        elif isinstance(result, pd.Series):
            col_name = f"ind_{name.lower().replace(' ', '_')}"
            cols[col_name] = result

    return pd.DataFrame(cols, index=df.index)


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the price/indicator feature matrix for the price branch.

    Columns: OHLCV (5) + returns (1) + acceleration (1) + indicator cols (~40)
    """
    features = pd.DataFrame(index=df.index)

    # Base OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        features[col] = df[col]

    # Price derivatives
    features["returns"] = df["close"].pct_change()
    features["acceleration"] = features["returns"].diff()

    # Flattened indicators
    indicator_df = flatten_indicators(df)
    features = pd.concat([features, indicator_df], axis=1)

    # Derived features for Bollinger
    if "ind_bollinger_bands_upper" in features.columns:
        bb_upper = features["ind_bollinger_bands_upper"]
        bb_lower = features["ind_bollinger_bands_lower"]
        bb_middle = features["ind_bollinger_bands_middle"]
        close = features["close"]

        features["bb_width"] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
        features["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Distance from moving averages (normalized)
    if "ind_sma" in features.columns:
        features["close_sma_ratio"] = (
            features["close"] / features["ind_sma"].replace(0, np.nan)
        ) - 1.0
    if "ind_ema" in features.columns:
        features["close_ema_ratio"] = (
            features["close"] / features["ind_ema"].replace(0, np.nan)
        ) - 1.0

    return features


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build volume branch features: volume + derivatives."""
    features = pd.DataFrame(index=df.index)
    features["volume"] = df["volume"]
    features["volume_return"] = df["volume"].pct_change()
    features["volume_accel"] = features["volume_return"].diff()
    return features


def build_pattern_features(
    df: pd.DataFrame,
    pattern_configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build pattern branch features: detected flag + confidence per pattern.

    Detects all patterns on the data and creates binary+confidence columns.
    Each pattern gets 2 columns: {name}_detected (0/1) and {name}_confidence (0-1).
    """
    from quantdash.features.advanced_patterns import (
        ADVANCED_PATTERNS_CONFIG,
        detect_all_advanced_patterns,
    )
    from quantdash.features.candlestick_patterns import (
        CANDLESTICK_PATTERNS_CONFIG,
        detect_all_candlestick_patterns,
    )
    from quantdash.features.elliott_wave import (
        ELLIOTT_PATTERNS_CONFIG,
        detect_all_elliott_patterns,
    )
    from quantdash.features.harmonic_patterns import (
        HARMONIC_PATTERNS_CONFIG,
        detect_all_harmonic_patterns,
    )
    from quantdash.features.patterns import PATTERNS_CONFIG, detect_all_patterns

    # Collect all pattern configs for column names
    all_configs: dict[str, dict] = {}
    all_configs.update(PATTERNS_CONFIG)
    all_configs.update(CANDLESTICK_PATTERNS_CONFIG)
    all_configs.update(HARMONIC_PATTERNS_CONFIG)
    all_configs.update(ELLIOTT_PATTERNS_CONFIG)
    all_configs.update(ADVANCED_PATTERNS_CONFIG)

    if pattern_configs is not None:
        all_configs = pattern_configs

    # Initialize columns
    n = len(df)
    feature_cols: dict[str, np.ndarray] = {}
    for name in all_configs:
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        feature_cols[f"pat_{safe_name}_det"] = np.zeros(n)
        feature_cols[f"pat_{safe_name}_conf"] = np.zeros(n)

    # Run all detectors
    all_events = []
    all_events.extend(detect_all_patterns(df))
    all_events.extend(detect_all_candlestick_patterns(df))
    all_events.extend(detect_all_harmonic_patterns(df))
    all_events.extend(detect_all_elliott_patterns(df))
    all_events.extend(detect_all_advanced_patterns(df))

    # Map pattern_type back to config name
    type_to_name: dict[str, str] = {}
    for name, cfg in all_configs.items():
        ptype = cfg.get("type")
        if ptype is not None:
            if hasattr(ptype, "value"):
                type_to_name[ptype.value] = name
            else:
                type_to_name[str(ptype)] = name

    # Fill in detected patterns at their end_index
    for event in all_events:
        config_name = type_to_name.get(event.pattern_type)
        if config_name is None:
            continue

        safe_name = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        det_key = f"pat_{safe_name}_det"
        conf_key = f"pat_{safe_name}_conf"

        idx = event.end_index
        if 0 <= idx < n and det_key in feature_cols:
            feature_cols[det_key][idx] = 1.0
            # Keep highest confidence if multiple detections at same bar
            feature_cols[conf_key][idx] = max(
                feature_cols[conf_key][idx], event.confidence
            )

    return pd.DataFrame(feature_cols, index=df.index)


def build_session_features(
    df: pd.DataFrame,
    session_type: str = "session",
    open_hour_utc: int = 14,
    close_hour_utc: int = 21,
) -> pd.DataFrame:
    """Build session/timing features with cyclical encoding.

    Features: sin/cos hour, sin/cos day-of-week, is_market_open,
    minutes_since_open, minutes_to_close (normalized).
    """
    features = pd.DataFrame(index=df.index)
    idx = df.index

    if hasattr(idx, "hour"):
        hours = idx.hour
        dow = idx.dayofweek
    else:
        # Try to get datetime from index
        try:
            dt_index = pd.DatetimeIndex(idx)
            hours = dt_index.hour
            dow = dt_index.dayofweek
        except (TypeError, ValueError):
            # Fallback: zeros
            features["sin_hour"] = 0.0
            features["cos_hour"] = 0.0
            features["sin_dow"] = 0.0
            features["cos_dow"] = 0.0
            features["is_market_open"] = 1.0
            features["minutes_since_open"] = 0.0
            features["minutes_to_close"] = 0.0
            return features

    # Cyclical encoding
    features["sin_hour"] = np.sin(2 * np.pi * hours / 24)
    features["cos_hour"] = np.cos(2 * np.pi * hours / 24)
    features["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    features["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    if session_type == "24/7":
        features["is_market_open"] = 1.0
        # Encode NYC session overlap (14-21 UTC) as feature
        features["nyc_session"] = ((hours >= 14) & (hours < 21)).astype(float)
        features["minutes_since_open"] = 0.0
        features["minutes_to_close"] = 0.0
    else:
        # Session-based market hours
        if open_hour_utc < close_hour_utc:
            is_open = (hours >= open_hour_utc) & (hours < close_hour_utc)
        else:
            # Wraps midnight (e.g., open=22, close=21)
            is_open = (hours >= open_hour_utc) | (hours < close_hour_utc)
        features["is_market_open"] = is_open.astype(float)

        # Minutes since/to session (simplified, normalized 0-1)
        session_hours = (
            (close_hour_utc - open_hour_utc) % 24
        ) or 24
        hours_since = (hours - open_hour_utc) % 24
        features["minutes_since_open"] = np.clip(hours_since / session_hours, 0, 1)
        features["minutes_to_close"] = 1.0 - features["minutes_since_open"]

    return features


# ── Normalization ────────────────────────────────────────────────────────────


def rolling_zscore(
    df: pd.DataFrame,
    config: NormalizationConfig | None = None,
) -> pd.DataFrame:
    """Apply rolling z-score normalization (no look-ahead bias).

    Each feature is normalized using a rolling window of past values only.
    Z-scores are clipped to [-clip_value, clip_value].
    """
    if config is None:
        config = NormalizationConfig()

    mean = df.rolling(
        window=config.lookback, min_periods=config.min_periods
    ).mean()
    std = df.rolling(
        window=config.lookback, min_periods=config.min_periods
    ).std()

    # Avoid division by zero
    std = std.replace(0, np.nan)

    normalized = (df - mean) / std
    normalized = normalized.clip(-config.clip_value, config.clip_value)

    return normalized


def normalize_features(
    df: pd.DataFrame,
    config: NormalizationConfig | None = None,
    skip_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize feature DataFrame, skipping binary/categorical columns."""
    if config is None:
        config = NormalizationConfig()

    if skip_cols is None:
        # Don't normalize binary pattern detection flags or cyclical features
        skip_cols = [
            c for c in df.columns
            if c.endswith("_det")
            or c.startswith("is_")
            or c.startswith("sin_")
            or c.startswith("cos_")
            or c == "nyc_session"
        ]

    cols_to_norm = [c for c in df.columns if c not in skip_cols]
    cols_to_skip = [c for c in df.columns if c in skip_cols]

    normalized = rolling_zscore(df[cols_to_norm], config)

    result = pd.concat([normalized, df[cols_to_skip]], axis=1)
    # Maintain original column order
    return result[df.columns]
