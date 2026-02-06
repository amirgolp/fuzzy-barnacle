"""Data pipeline orchestrator.

Coordinates: fetch OHLCV → compute features → label → normalize → save.
Produces ready-to-train datasets for a given asset.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdash.ml.config import (
    ASSET_CONFIGS,
    ML_DATA_DIR,
    AssetConfig,
    LabelingConfig,
    ModelConfig,
    NormalizationConfig,
)
from quantdash.ml.data.dataset import (
    TradingSignalDataset,
    create_dataset_from_dataframes,
    save_dataset_hdf5,
)
from quantdash.ml.data.features import (
    build_pattern_features,
    build_price_features,
    build_session_features,
    build_volume_features,
    normalize_features,
)
from quantdash.ml.data.labeling import label_distribution, triple_barrier_label
from quantdash.ml.data.macro import align_macro_to_ohlcv, create_macro_stub
from quantdash.ml.data.splits import SplitFold, walk_forward_splits

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    period: str = "730d",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data using yfinance.

    Args:
        symbol: Ticker symbol.
        timeframe: Bar interval (e.g., '1h', '15m').
        period: Lookback period (e.g., '730d').
        start: Explicit start date (overrides period).
        end: Explicit end date.

    Returns:
        DataFrame with lowercase OHLCV columns and DatetimeIndex.
    """
    import yfinance as yf

    kwargs: dict = {
        "tickers": symbol,
        "interval": timeframe,
        "auto_adjust": True,
        "threads": False,
        "progress": False,
    }

    if start is not None:
        kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = yf.download(**kwargs)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({timeframe})")

    # Normalize columns to lowercase
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]

    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def fetch_cross_asset_data(
    correlated_assets: list[str],
    target_index: pd.DatetimeIndex,
    timeframe: str = "1h",
    period: str = "730d",
) -> pd.DataFrame:
    """Fetch cross-asset OHLCV and align to target timestamps.

    Returns DataFrame with columns: {symbol}_open, {symbol}_high, etc.
    """
    frames = []

    for symbol in correlated_assets:
        try:
            df = fetch_ohlcv(symbol, timeframe=timeframe, period=period)
            # Prefix columns with symbol
            prefix = symbol.replace("=", "").replace("-", "").replace(".", "").replace("^", "")
            renamed = df[["open", "high", "low", "close", "volume"]].rename(
                columns={c: f"{prefix}_{c}" for c in ["open", "high", "low", "close", "volume"]}
            )
            frames.append(renamed)
            logger.info(f"Fetched cross-asset {symbol}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"Failed to fetch cross-asset {symbol}: {e}")
            continue

    if not frames:
        return pd.DataFrame(index=target_index)

    # Merge all on index, forward-fill missing timestamps
    merged = pd.concat(frames, axis=1)

    # Align to target index
    combined_idx = merged.index.union(target_index).sort_values()
    aligned = merged.reindex(combined_idx).ffill()
    result = aligned.reindex(target_index)

    return result


def build_dataset(
    symbol: str,
    asset_config: AssetConfig | None = None,
    macro_df: pd.DataFrame | None = None,
    news_embeddings: np.ndarray | None = None,
    save_path: Path | None = None,
    fetch_cross: bool = True,
) -> TradingSignalDataset:
    """Build a complete training dataset for a single asset.

    Pipeline:
    1. Fetch 1H OHLCV data
    2. Compute price/indicator features
    3. Compute volume features
    4. Detect patterns → pattern features
    5. Compute session/timing features
    6. Align macro data
    7. Fetch cross-asset data
    8. Compute labels (triple-barrier)
    9. Normalize features
    10. Package into TradingSignalDataset

    Args:
        symbol: Ticker symbol.
        asset_config: Per-asset config (defaults to ASSET_CONFIGS lookup).
        macro_df: Pre-fetched macro data (skips FRED API call if provided).
        news_embeddings: Pre-computed FinBERT embeddings [N, max_articles, 768].
        save_path: If set, save processed arrays to HDF5.
        fetch_cross: Whether to fetch cross-asset data.

    Returns:
        TradingSignalDataset ready for DataLoader.
    """
    if asset_config is None:
        asset_config = ASSET_CONFIGS.get(symbol)
        if asset_config is None:
            raise ValueError(
                f"No config for {symbol}. Available: {list(ASSET_CONFIGS.keys())}"
            )

    config = asset_config.arch_config
    lookback = config.lookback_1h

    # ── 1. Fetch OHLCV ──────────────────────────────────────────────────────
    logger.info(f"Fetching 1H data for {symbol}...")
    df = fetch_ohlcv(symbol, timeframe="1h")
    logger.info(f"Got {len(df)} bars for {symbol}")

    # ── 2. Price/indicator features ──────────────────────────────────────────
    logger.info("Computing price/indicator features...")
    price_df = build_price_features(df)
    logger.info(f"Price features: {price_df.shape[1]} columns")

    # ── 3. Volume features ───────────────────────────────────────────────────
    volume_df = build_volume_features(df)

    # ── 4. Pattern features ──────────────────────────────────────────────────
    logger.info("Detecting patterns...")
    pattern_df = build_pattern_features(df)
    logger.info(f"Pattern features: {pattern_df.shape[1]} columns")

    # ── 5. Session features ──────────────────────────────────────────────────
    session_df = build_session_features(
        df,
        session_type=asset_config.session_type.value,
        open_hour_utc=asset_config.session_open_utc,
        close_hour_utc=asset_config.session_close_utc,
    )

    # ── 6. Macro data ────────────────────────────────────────────────────────
    if macro_df is not None:
        macro_aligned = align_macro_to_ohlcv(macro_df, df.index)
    else:
        macro_aligned = create_macro_stub(df.index)

    # Combine macro + session into one feature set
    macro_session_df = pd.concat([macro_aligned, session_df], axis=1)

    # ── 7. Cross-asset data ──────────────────────────────────────────────────
    cross_asset_df = None
    if fetch_cross and asset_config.correlated_assets:
        logger.info(f"Fetching cross-asset data: {asset_config.correlated_assets}")
        cross_asset_df = fetch_cross_asset_data(
            asset_config.correlated_assets, df.index
        )
        if cross_asset_df.empty:
            cross_asset_df = None
        else:
            logger.info(f"Cross-asset features: {cross_asset_df.shape[1]} columns")

    # ── 8. Labels ────────────────────────────────────────────────────────────
    logger.info("Computing triple-barrier labels...")
    labels = triple_barrier_label(df, config=asset_config.labeling_config)
    dist = label_distribution(labels)
    logger.info(
        f"Label distribution — BUY: {dist['buy_pct']}%, "
        f"HOLD: {dist['hold_pct']}%, SELL: {dist['sell_pct']}%"
    )

    # ── 9. Normalize ─────────────────────────────────────────────────────────
    logger.info("Normalizing features...")
    norm_config = asset_config.normalization_config

    price_norm = normalize_features(price_df, norm_config)
    volume_norm = normalize_features(volume_df, norm_config)
    # Pattern features: binary flags + confidence — no normalization needed
    macro_session_norm = normalize_features(macro_session_df, norm_config)

    cross_asset_norm = None
    if cross_asset_df is not None:
        cross_asset_norm = normalize_features(cross_asset_df, norm_config)

    # ── 10. Build dataset ────────────────────────────────────────────────────
    dataset = create_dataset_from_dataframes(
        price_df=price_norm,
        volume_df=volume_norm,
        pattern_df=pattern_df,  # not normalized (binary)
        macro_session_df=macro_session_norm,
        labels=labels,
        cross_asset_df=cross_asset_norm,
        news_embeddings=news_embeddings,
        lookback=lookback,
        config=config,
    )

    logger.info(
        f"Dataset built: {len(dataset)} samples, "
        f"price_channels={price_norm.shape[1]}, "
        f"pattern_features={pattern_df.shape[1]}, "
        f"macro_features={macro_session_df.shape[1]}"
    )

    # ── Save to HDF5 ────────────────────────────────────────────────────────
    if save_path is not None:
        logger.info(f"Saving dataset to {save_path}")
        arrays = {
            "price": price_norm.fillna(0).values,
            "volume": volume_norm.fillna(0).values,
            "pattern": pattern_df.fillna(0).values,
            "macro_session": macro_session_norm.fillna(0).values,
            "labels": labels.values,
        }
        if cross_asset_norm is not None:
            arrays["cross_asset"] = cross_asset_norm.fillna(0).values
        if news_embeddings is not None:
            arrays["news"] = news_embeddings
        save_dataset_hdf5(save_path, **arrays)

    return dataset


def get_dataset_path(symbol: str) -> Path:
    """Get the default HDF5 path for a symbol's dataset."""
    safe_symbol = symbol.replace("=", "").replace("-", "_").replace("^", "")
    return ML_DATA_DIR / f"{safe_symbol}_dataset.h5"
