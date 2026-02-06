"""ML Signal Strategy — integrates TemporalFusionSignalNet with Strategy framework.

Registered in STRATEGY_REGISTRY, backtestable via existing VectorBT engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy

logger = logging.getLogger(__name__)


class MLSignalStrategy(Strategy):
    """Trading strategy powered by TemporalFusionSignalNet predictions.

    Loads a trained model checkpoint, runs inference on OHLCV data,
    and produces signal column compatible with the backtest engine.

    Params:
        checkpoint_path: Path to trained model .pt file.
        symbol: Asset symbol for config lookup.
        min_confidence: Minimum confidence to generate trade signal (default 0.55).
    """

    name = "ml_signal"
    description = "ML-based trading signals from TemporalFusionSignalNet"
    default_params = {
        "checkpoint_path": "models/model_best.pt",
        "symbol": "GC=F",
        "min_confidence": 0.55,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using the trained ML model.

        Args:
            df: OHLCV DataFrame with lowercase columns.

        Returns:
            DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
            and 'ml_confidence' column.
        """
        checkpoint_path = Path(self.params.get("checkpoint_path", "models/model_best.pt"))
        symbol = self.params.get("symbol", "GC=F")
        min_confidence = self.params.get("min_confidence", 0.55)

        result_df = df.copy()
        result_df["signal"] = 0
        result_df["ml_confidence"] = 0.0

        if not checkpoint_path.exists():
            logger.warning(
                "Model checkpoint not found at %s. Returning hold signals.",
                checkpoint_path,
            )
            return result_df

        try:
            from quantdash.ml.config import ASSET_CONFIGS, ModelConfig
            from quantdash.ml.data.features import (
                build_pattern_features,
                build_price_features,
                build_session_features,
                build_volume_features,
            )
            from quantdash.ml.data.macro import create_macro_stub
            from quantdash.ml.data.news_embeddings import create_empty_embeddings
            from quantdash.ml.inference.predictor import SignalPredictor
        except ImportError as e:
            logger.error("ML dependencies not available: %s", e)
            return result_df

        # Load model
        predictor = SignalPredictor.from_checkpoint(checkpoint_path)
        config = ASSET_CONFIGS.get(symbol)
        lookback = predictor.config.lookback_1h

        if len(df) < lookback + 50:
            logger.warning("Insufficient data for ML signals: %d bars", len(df))
            return result_df

        # Compute features once
        price_feats = build_price_features(df).fillna(0)
        volume_feats = build_volume_features(df).fillna(0)
        pattern_feats = build_pattern_features(df).fillna(0)

        session_type = "session"
        open_h, close_h = 14, 21
        if config is not None:
            session_type = config.session_type.value
            open_h = config.session_open_utc
            close_h = config.session_close_utc

        session_feats = build_session_features(df, session_type, open_h, close_h)
        macro_feats = create_macro_stub(df.index)
        macro_session = pd.concat([macro_feats, session_feats], axis=1).fillna(0)

        n_price_ch = price_feats.shape[1]
        n_cross_ch = predictor.config.cross_asset_channels

        # Run predictions for each bar with enough lookback
        for i in range(lookback, len(df)):
            price_window = price_feats.values[i - lookback : i][np.newaxis]
            volume_window = volume_feats.values[i - lookback : i][np.newaxis]
            pattern_vec = pattern_feats.values[i : i + 1]
            macro_vec = macro_session.values[i : i + 1]
            news_arr = create_empty_embeddings(
                1, max_articles=predictor.config.max_articles
            )
            cross_arr = np.zeros((1, lookback, n_cross_ch), dtype=np.float32)

            try:
                prediction = predictor.predict_from_tensors(
                    price_window, volume_window, pattern_vec,
                    news_arr, macro_vec, cross_arr,
                )

                result_df.iloc[i, result_df.columns.get_loc("ml_confidence")] = (
                    prediction.confidence
                )

                if prediction.confidence >= min_confidence:
                    result_df.iloc[i, result_df.columns.get_loc("signal")] = (
                        prediction.action.value
                    )
            except Exception as e:
                logger.debug("Prediction failed at bar %d: %s", i, e)
                continue

        return result_df
