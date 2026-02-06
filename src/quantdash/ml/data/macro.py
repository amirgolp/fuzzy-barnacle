"""FRED macro data fetcher.

Fetches low-frequency macro series (fed funds rate, treasury yields, CPI,
unemployment) from FRED API and forward-fills to align with high-frequency
OHLCV data timestamps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantdash.ml.config import FRED_SERIES

logger = logging.getLogger(__name__)


def fetch_fred_series(
    series_id: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    api_key: str | None = None,
) -> pd.Series:
    """Fetch a single FRED series.

    Args:
        series_id: FRED series ID (e.g., 'FEDFUNDS').
        start: Start date.
        end: End date.
        api_key: FRED API key. If None, reads from FRED_API_KEY env var.

    Returns:
        Series indexed by date with the macro values.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi is required for macro data. "
            "Install with: pip install 'quantdash[ml]'"
        )

    import os
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY env var or pass api_key."
        )

    fred = Fred(api_key=key)

    kwargs = {}
    if start is not None:
        kwargs["observation_start"] = start
    if end is not None:
        kwargs["observation_end"] = end

    data = fred.get_series(series_id, **kwargs)
    data.name = series_id
    return data


def fetch_all_macro(
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    api_key: str | None = None,
    series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series into a single DataFrame.

    Args:
        start: Start date.
        end: End date.
        api_key: FRED API key.
        series_map: Override default FRED_SERIES mapping.

    Returns:
        DataFrame with columns named by feature name (not FRED ID).
    """
    if series_map is None:
        series_map = FRED_SERIES

    frames = {}
    for feature_name, series_id in series_map.items():
        try:
            data = fetch_fred_series(series_id, start=start, end=end, api_key=api_key)
            frames[feature_name] = data
            logger.info(f"Fetched {series_id} ({feature_name}): {len(data)} observations")
        except Exception as e:
            logger.warning(f"Failed to fetch {series_id}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    return df


def align_macro_to_ohlcv(
    macro_df: pd.DataFrame,
    ohlcv_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill macro data to align with high-frequency OHLCV timestamps.

    Macro data is low-frequency (monthly/daily). We reindex to the OHLCV
    timestamps and forward-fill, so each bar gets the most recent macro value.

    Args:
        macro_df: DataFrame from fetch_all_macro().
        ohlcv_index: DatetimeIndex of the OHLCV data.

    Returns:
        DataFrame with macro features aligned to OHLCV timestamps.
    """
    if macro_df.empty:
        return pd.DataFrame(index=ohlcv_index)

    # Normalize both indices to tz-naive for joining
    macro_idx = macro_df.index
    if macro_idx.tz is not None:
        macro_idx = macro_idx.tz_localize(None)
        macro_df = macro_df.copy()
        macro_df.index = macro_idx

    target_idx = ohlcv_index
    if hasattr(target_idx, "tz") and target_idx.tz is not None:
        target_idx = target_idx.tz_localize(None)

    # Combine indices and forward-fill
    combined_idx = macro_idx.union(target_idx).sort_values()
    aligned = macro_df.reindex(combined_idx).ffill()

    # Select only the OHLCV timestamps
    result = aligned.reindex(target_idx)

    # Back-fill any leading NaNs (before first macro observation)
    result = result.bfill()

    result.index = ohlcv_index  # restore original index
    return result


def create_macro_stub(
    ohlcv_index: pd.DatetimeIndex,
    series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a zero-filled macro DataFrame when FRED data is unavailable.

    Useful for testing or when no API key is configured.
    """
    if series_map is None:
        series_map = FRED_SERIES

    return pd.DataFrame(
        0.0, index=ohlcv_index, columns=list(series_map.keys())
    )
