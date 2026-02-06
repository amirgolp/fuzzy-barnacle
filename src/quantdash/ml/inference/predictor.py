"""Signal predictor: loads trained model and runs inference.

Handles model loading, feature preparation, and prediction output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from quantdash.ml.config import (
    ASSET_CONFIGS,
    AssetConfig,
    ModelConfig,
    SignalAction,
    SignalPrediction,
)

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Load a trained model and produce trading signal predictions.

    Usage:
        predictor = SignalPredictor.from_checkpoint("models/model_best.pt")
        signal = predictor.predict(df)
    """

    def __init__(
        self,
        model,
        config: ModelConfig,
        device: str = "cpu",
    ):
        if torch is None:
            raise ImportError("PyTorch required for inference.")

        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> "SignalPredictor":
        """Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Device ('cuda', 'cpu', or None for auto).
        """
        if torch is None:
            raise ImportError("PyTorch required for inference.")

        from quantdash.ml.models.signal_net import TemporalFusionSignalNet

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Reconstruct config
        config_dict = state.get("config", {})
        config = ModelConfig(**config_dict) if config_dict else ModelConfig()

        model = TemporalFusionSignalNet(config)
        model.load_state_dict(state["model_state_dict"])

        logger.info(
            "Loaded model from %s (epoch=%d)", checkpoint_path, state.get("epoch", -1)
        )

        return cls(model, config, device=device)

    def predict_from_tensors(
        self,
        price: np.ndarray,
        volume: np.ndarray,
        pattern: np.ndarray,
        news: np.ndarray,
        macro: np.ndarray,
        cross_asset: np.ndarray,
    ) -> SignalPrediction:
        """Run prediction from prepared numpy arrays.

        All inputs should have batch dimension (even if batch=1).
        """
        with torch.no_grad():
            result = self.model.predict(
                torch.from_numpy(price).float().to(self.device),
                torch.from_numpy(volume).float().to(self.device),
                torch.from_numpy(pattern).float().to(self.device),
                torch.from_numpy(news).float().to(self.device),
                torch.from_numpy(macro).float().to(self.device),
                torch.from_numpy(cross_asset).float().to(self.device),
            )

        # Extract first sample from batch
        action_idx = result["action"][0].item()
        confidence = result["confidence"][0].item()
        probs = result["probabilities"][0].cpu().numpy()

        # Map from {0,1,2} to SignalAction {-1,0,+1}
        action = SignalAction(action_idx - 1)

        return SignalPrediction(
            action=action,
            confidence=round(confidence, 4),
            position_size=0.0,  # filled by position sizing module
            probabilities={
                "sell": round(float(probs[0]), 4),
                "hold": round(float(probs[1]), 4),
                "buy": round(float(probs[2]), 4),
            },
        )

    def predict_batch(
        self,
        price: np.ndarray,
        volume: np.ndarray,
        pattern: np.ndarray,
        news: np.ndarray,
        macro: np.ndarray,
        cross_asset: np.ndarray,
    ) -> list[SignalPrediction]:
        """Run batch prediction."""
        with torch.no_grad():
            result = self.model.predict(
                torch.from_numpy(price).float().to(self.device),
                torch.from_numpy(volume).float().to(self.device),
                torch.from_numpy(pattern).float().to(self.device),
                torch.from_numpy(news).float().to(self.device),
                torch.from_numpy(macro).float().to(self.device),
                torch.from_numpy(cross_asset).float().to(self.device),
            )

        predictions = []
        batch_size = result["action"].shape[0]

        for i in range(batch_size):
            action = SignalAction(result["action"][i].item() - 1)
            confidence = result["confidence"][i].item()
            probs = result["probabilities"][i].cpu().numpy()

            predictions.append(SignalPrediction(
                action=action,
                confidence=round(confidence, 4),
                position_size=0.0,
                probabilities={
                    "sell": round(float(probs[0]), 4),
                    "hold": round(float(probs[1]), 4),
                    "buy": round(float(probs[2]), 4),
                },
            ))

        return predictions
