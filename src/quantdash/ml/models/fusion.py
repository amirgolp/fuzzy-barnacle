"""Fusion Transformer — merges all 6 branch embeddings via cross-attention.

Projects each branch output to a common dimension, stacks them as a sequence,
then applies multi-head self-attention to learn inter-branch relationships.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quantdash.ml.config import ModelConfig


class FusionTransformer(nn.Module):
    """Multi-head cross-attention fusion over 6 branch embeddings.

    Architecture:
        1. Project each branch to fusion_d dimension
        2. Stack as [batch, 6, fusion_d]
        3. Apply N-layer Transformer encoder
        4. Mean pool → Linear → output embedding
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.fusion_d

        # Branch projection layers (project each branch to fusion_d)
        self.proj_price = nn.Linear(config.price_d_model, d)
        self.proj_volume = nn.Linear(config.volume_d_model, d)
        self.proj_pattern = nn.Linear(config.pattern_d_model, d)
        self.proj_news = nn.Linear(config.news_d_model, d)
        self.proj_macro = nn.Linear(config.macro_d_model, d)
        self.proj_cross = nn.Linear(config.cross_d_model, d)

        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.fusion_nhead,
            dim_feedforward=d * 4,
            dropout=config.fusion_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.fusion_num_layers
        )

        # Output projection: fusion_d → output_d (256)
        self.output_proj = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(128, config.output_d),
        )

        self.layer_norm = nn.LayerNorm(d)

    def forward(
        self,
        price_embed: torch.Tensor,
        volume_embed: torch.Tensor,
        pattern_embed: torch.Tensor,
        news_embed: torch.Tensor,
        macro_embed: torch.Tensor,
        cross_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse 6 branch embeddings into a single output embedding.

        Args:
            price_embed: [batch, price_d_model]
            volume_embed: [batch, volume_d_model]
            pattern_embed: [batch, pattern_d_model]
            news_embed: [batch, news_d_model]
            macro_embed: [batch, macro_d_model]
            cross_embed: [batch, cross_d_model]

        Returns:
            [batch, output_d] fused embedding
        """
        # Project each to fusion_d
        branches = torch.stack([
            self.proj_price(price_embed),
            self.proj_volume(volume_embed),
            self.proj_pattern(pattern_embed),
            self.proj_news(news_embed),
            self.proj_macro(macro_embed),
            self.proj_cross(cross_embed),
        ], dim=1)  # [batch, 6, fusion_d]

        branches = self.layer_norm(branches)

        # Self-attention across branches
        fused = self.transformer(branches)  # [batch, 6, fusion_d]

        # Mean pool over branches
        pooled = fused.mean(dim=1)  # [batch, fusion_d]

        # Project to output dimension
        return self.output_proj(pooled)  # [batch, output_d]
