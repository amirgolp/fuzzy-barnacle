"""TemporalFusionSignalNet — the full multi-modal trading signal model.

Assembles 6 branches + fusion transformer + classification/confidence heads.
~1.15M parameters, fits in 8GB VRAM with FP16 batch=128.

Output:
    - action_logits: [batch, 3] → Softmax → P(sell), P(hold), P(buy)
    - confidence: [batch, 1] → Sigmoid → calibrated prediction confidence
    - fused_embedding: [batch, 256] → for RL fine-tuning observation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantdash.ml.config import ModelConfig
from quantdash.ml.models.branches import (
    CrossAssetBranch,
    MacroBranch,
    NewsBranch,
    PatternBranch,
    PriceBranch,
    VolumeBranch,
)
from quantdash.ml.models.fusion import FusionTransformer


class TemporalFusionSignalNet(nn.Module):
    """Multi-modal trading signal prediction network.

    6 specialized branches process different data modalities,
    a fusion transformer merges them, and output heads produce
    trading signals with calibrated confidence.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Branch encoders
        self.price_branch = PriceBranch(self.config)
        self.volume_branch = VolumeBranch(self.config)
        self.pattern_branch = PatternBranch(self.config)
        self.news_branch = NewsBranch(self.config)
        self.macro_branch = MacroBranch(self.config)
        self.cross_branch = CrossAssetBranch(self.config)

        # Fusion
        self.fusion = FusionTransformer(self.config)

        # Output heads
        self.action_head = nn.Linear(self.config.output_d, 3)
        self.confidence_head = nn.Linear(self.config.output_d, 1)

        # Initialize output heads for stable start
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.confidence_head.weight, gain=0.01)
        nn.init.zeros_(self.confidence_head.bias)  # sigmoid(0) = 0.5

    def forward(
        self,
        price: torch.Tensor,
        volume: torch.Tensor,
        pattern: torch.Tensor,
        news: torch.Tensor,
        macro: torch.Tensor,
        cross_asset: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through all branches, fusion, and output heads.

        Args:
            price: [batch, lookback, price_channels]
            volume: [batch, lookback, 3]
            pattern: [batch, num_pattern_features]
            news: [batch, max_articles, 768]
            macro: [batch, num_macro_features]
            cross_asset: [batch, lookback, cross_asset_channels]

        Returns:
            Dict with:
                - action_logits: [batch, 3] (raw logits, not softmaxed)
                - confidence: [batch, 1]
                - fused_embedding: [batch, output_d]
        """
        # Encode each modality
        price_embed = self.price_branch(price)
        volume_embed = self.volume_branch(volume)
        pattern_embed = self.pattern_branch(pattern)
        news_embed = self.news_branch(news)
        macro_embed = self.macro_branch(macro)
        cross_embed = self.cross_branch(cross_asset)

        # Fuse
        fused = self.fusion(
            price_embed, volume_embed, pattern_embed,
            news_embed, macro_embed, cross_embed,
        )

        # Output heads
        action_logits = self.action_head(fused)
        confidence = self.confidence_head(fused)

        return {
            "action_logits": action_logits,
            "confidence": confidence,
            "fused_embedding": fused,
        }

    def predict(
        self,
        price: torch.Tensor,
        volume: torch.Tensor,
        pattern: torch.Tensor,
        news: torch.Tensor,
        macro: torch.Tensor,
        cross_asset: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference mode: returns probabilities and predicted action.

        Returns:
            Dict with:
                - action: [batch] int tensor (0=sell, 1=hold, 2=buy)
                - probabilities: [batch, 3]
                - confidence: [batch]
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(price, volume, pattern, news, macro, cross_asset)

        probs = F.softmax(out["action_logits"], dim=-1)
        action = probs.argmax(dim=-1)

        return {
            "action": action,
            "probabilities": probs,
            "confidence": torch.sigmoid(out["confidence"]).squeeze(-1),
            "fused_embedding": out["fused_embedding"],
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoders(self) -> None:
        """Freeze all branch encoders (for RL fine-tuning)."""
        for branch in [
            self.price_branch, self.volume_branch, self.pattern_branch,
            self.news_branch, self.macro_branch, self.cross_branch,
        ]:
            for param in branch.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze all branch encoders."""
        for branch in [
            self.price_branch, self.volume_branch, self.pattern_branch,
            self.news_branch, self.macro_branch, self.cross_branch,
        ]:
            for param in branch.parameters():
                param.requires_grad = True


def create_model(config: ModelConfig | None = None) -> TemporalFusionSignalNet:
    """Create a new model with default or custom config."""
    return TemporalFusionSignalNet(config)
