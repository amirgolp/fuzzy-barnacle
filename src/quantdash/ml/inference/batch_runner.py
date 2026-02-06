"""Batch inference runner for cron-based hourly predictions.

Loads model, fetches latest data, computes features, runs prediction,
applies position sizing and circuit breakers, outputs signal.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantdash.ml.config import (
    ASSET_CONFIGS,
    RISK_PROFILES,
    CircuitBreakerState,
    SignalAction,
)
from quantdash.ml.data.features import (
    build_pattern_features,
    build_price_features,
    build_session_features,
    build_volume_features,
)
from quantdash.ml.data.macro import create_macro_stub
from quantdash.ml.data.news_embeddings import create_empty_embeddings
from quantdash.ml.inference.predictor import SignalPredictor
from quantdash.ml.risk.circuit_breakers import get_effective_leverage
from quantdash.ml.risk.position_sizing import compute_position

logger = logging.getLogger(__name__)


def run_inference(
    symbol: str,
    checkpoint_path: str | Path,
    equity: float = 100_000.0,
    circuit_state: CircuitBreakerState | None = None,
    device: str | None = None,
) -> dict:
    """Run inference for a single asset.

    Pipeline:
    1. Load model from checkpoint
    2. Fetch latest OHLCV data (lookback window)
    3. Compute features
    4. Run prediction
    5. Apply position sizing
    6. Check circuit breakers
    7. Return signal

    Args:
        symbol: Ticker symbol.
        checkpoint_path: Path to trained model checkpoint.
        equity: Current portfolio equity.
        circuit_state: Current circuit breaker state.
        device: Inference device.

    Returns:
        Dict with signal details.
    """
    from quantdash.ml.data.builder import fetch_ohlcv

    config = ASSET_CONFIGS.get(symbol)
    if config is None:
        raise ValueError(f"No config for {symbol}")

    lookback = config.arch_config.lookback_1h

    # 1. Load model
    predictor = SignalPredictor.from_checkpoint(checkpoint_path, device=device)

    # 2. Fetch data (need lookback + some extra for indicator warmup)
    warmup = 200  # extra bars for indicator computation
    total_bars = lookback + warmup
    df = fetch_ohlcv(symbol, timeframe="1h", period=f"{total_bars * 2}d")

    if len(df) < lookback + 50:
        return {"error": f"Insufficient data: {len(df)} bars (need {lookback + 50})"}

    # 3. Compute features
    price_feats = build_price_features(df)
    volume_feats = build_volume_features(df)
    pattern_feats = build_pattern_features(df)
    session_feats = build_session_features(
        df,
        session_type=config.session_type.value,
        open_hour_utc=config.session_open_utc,
        close_hour_utc=config.session_close_utc,
    )
    macro_feats = create_macro_stub(df.index)
    macro_session = pd.concat([macro_feats, session_feats], axis=1)

    # Fill NaNs
    price_arr = price_feats.fillna(0).values[-lookback:][np.newaxis]  # [1, lookback, C]
    volume_arr = volume_feats.fillna(0).values[-lookback:][np.newaxis]
    pattern_arr = pattern_feats.fillna(0).values[-1:][np.newaxis]  # [1, 1, P] → squeeze later
    pattern_arr = pattern_feats.fillna(0).values[-1:].reshape(1, -1)  # [1, P]
    macro_arr = macro_session.fillna(0).values[-1:].reshape(1, -1)  # [1, M]

    # News: empty
    news_arr = create_empty_embeddings(1, max_articles=config.arch_config.max_articles)

    # Cross-asset: zeros (simplified — full version would fetch cross data)
    cross_arr = np.zeros(
        (1, lookback, config.arch_config.cross_asset_channels), dtype=np.float32
    )

    # 4. Predict
    prediction = predictor.predict_from_tensors(
        price_arr, volume_arr, pattern_arr, news_arr, macro_arr, cross_arr
    )

    # 5. Position sizing
    position = compute_position(
        action=prediction.action.value,
        confidence=prediction.confidence,
        symbol=symbol,
        current_equity=equity,
    )
    prediction.position_size = position["size"]

    # 6. Circuit breakers
    risk_profile = RISK_PROFILES.get(symbol)
    if circuit_state is not None and risk_profile is not None:
        effective_leverage = get_effective_leverage(circuit_state, risk_profile)
        if effective_leverage == 0:
            position["direction"] = "flat"
            position["size"] = 0.0
            prediction.position_size = 0.0
            prediction.action = SignalAction.HOLD
        elif effective_leverage < position["leverage"]:
            position["leverage"] = effective_leverage

    # 7. Build result
    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "action": prediction.action.name,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "position": position,
        "latest_close": float(df["close"].iloc[-1]),
    }

    logger.info(
        "Signal for %s: %s (conf=%.3f, size=%.2f×)",
        symbol, prediction.action.name,
        prediction.confidence, position["size"],
    )

    return result


def run_batch_inference(
    symbols: list[str],
    model_dir: Path,
    equity: float = 100_000.0,
    state_file: Path | None = None,
    device: str | None = None,
) -> list[dict]:
    """Run inference for multiple assets.

    Args:
        symbols: List of ticker symbols.
        model_dir: Directory containing model checkpoints ({symbol}_best.pt).
        equity: Current portfolio equity.
        state_file: JSON file with circuit breaker states.
        device: Inference device.

    Returns:
        List of signal dicts.
    """
    # Load circuit breaker states
    states = {}
    if state_file is not None and state_file.exists():
        with open(state_file) as f:
            raw_states = json.load(f)
        for sym, s in raw_states.items():
            states[sym] = CircuitBreakerState(**s)

    results = []
    for symbol in symbols:
        safe = symbol.replace("=", "").replace("-", "_").replace("^", "")
        checkpoint = model_dir / f"{safe}_best.pt"

        if not checkpoint.exists():
            logger.warning("No checkpoint for %s at %s", symbol, checkpoint)
            results.append({"symbol": symbol, "error": "no checkpoint"})
            continue

        try:
            result = run_inference(
                symbol=symbol,
                checkpoint_path=checkpoint,
                equity=equity,
                circuit_state=states.get(symbol),
                device=device,
            )
            results.append(result)
        except Exception as e:
            logger.error("Inference failed for %s: %s", symbol, e)
            results.append({"symbol": symbol, "error": str(e)})

    return results
