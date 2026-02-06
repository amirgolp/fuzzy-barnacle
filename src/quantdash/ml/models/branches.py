"""TemporalFusionSignalNet branch modules.

Six specialized branches that each process a different data modality:
1. PriceBranch — Conv1D + Transformer on 1H price/indicator features
2. VolumeBranch — Conv1D + Transformer on volume features
3. PatternBranch — MLP on binary pattern detection + confidence
4. NewsBranch — Multi-head attention on FinBERT embeddings
5. MacroBranch — MLP on macro/session features
6. CrossAssetBranch — Conv1D + cross-attention on correlated asset features
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from quantdash.ml.config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PriceBranch(nn.Module):
    """Process 1H price/indicator temporal features.

    Architecture: Conv1D → BatchNorm → GELU → Conv1D → PositionalEncoding
    → TransformerEncoder → last hidden state → Linear → output [batch, d_model]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.price_d_model

        # Conv feature extraction
        self.conv1 = nn.Conv1d(config.price_channels, d, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d)
        self.act = nn.GELU()

        # Positional encoding
        self.pos_enc = PositionalEncoding(d, max_len=config.lookback_1h, dropout=config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.price_nhead,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.price_num_layers_1h
        )

        self.output_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, price_channels]
        Returns:
            [batch, price_d_model]
        """
        # Conv: transpose to [batch, channels, seq]
        h = x.transpose(1, 2)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = h.transpose(1, 2)  # back to [batch, seq, d]

        # Transformer
        h = self.pos_enc(h)
        h = self.transformer(h)

        # Take last timestep
        out = h[:, -1, :]
        return self.output_proj(out)


class VolumeBranch(nn.Module):
    """Process volume temporal features.

    Architecture: Conv1D(3→32→64) → 1-layer Transformer → output [batch, 64]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.volume_d_model
        in_channels = 3  # volume, volume_return, volume_accel

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, d, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d)
        self.act = nn.GELU()

        self.pos_enc = PositionalEncoding(d, max_len=config.lookback_1h, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.volume_nhead,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, 3]
        Returns:
            [batch, volume_d_model]
        """
        h = x.transpose(1, 2)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = h.transpose(1, 2)

        h = self.pos_enc(h)
        h = self.transformer(h)
        return h[:, -1, :]


class PatternBranch(nn.Module):
    """Process pattern detection features.

    Architecture: Linear(N→64) → GELU → Dropout → Linear(64→64)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.pattern_d_model

        self.net = nn.Sequential(
            nn.Linear(config.num_pattern_features, d),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_pattern_features]
        Returns:
            [batch, pattern_d_model]
        """
        return self.net(x)


class NewsBranch(nn.Module):
    """Process FinBERT news embeddings.

    Architecture: MultiheadAttention(768, 4) → masked mean pool → Linear(768→128→64)
    Zero-filled articles are masked out.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.news_d_model

        self.attention = nn.MultiheadAttention(
            embed_dim=config.finbert_dim,
            num_heads=config.news_nhead,
            dropout=config.dropout,
            batch_first=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(config.finbert_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, max_articles, 768]
        Returns:
            [batch, news_d_model]
        """
        # Create mask: True where all embeddings are zero (no article)
        key_padding_mask = (x.abs().sum(dim=-1) == 0)  # [batch, max_articles]

        # If ALL articles in the entire batch are masked, skip attention
        # to avoid NaN from softmax(-inf) — NaN * 0 = NaN in IEEE 754
        all_masked = key_padding_mask.all(dim=-1)  # [batch]
        if all_masked.all():
            return torch.zeros(
                x.size(0), self.proj[-1].out_features,
                device=x.device, dtype=x.dtype,
            )

        # For samples where all articles are masked, inject a dummy token
        # to prevent per-sample NaN in attention
        needs_dummy = all_masked.any()
        if needs_dummy:
            # Replace fully-masked rows with a small epsilon so attention
            # doesn't produce NaN, then zero the output afterward
            dummy = torch.full_like(x[:1, :1, :], 1e-8)
            dummy_expanded = dummy.expand(x.size(0), x.size(1), -1)
            x = torch.where(all_masked[:, None, None], dummy_expanded, x)
            # Recompute mask after injection
            key_padding_mask = (x.abs().sum(dim=-1) == 0)

        # Self-attention over articles
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)

        # Masked mean pool
        mask_expanded = (~key_padding_mask).unsqueeze(-1).float()  # [batch, articles, 1]
        pooled = (attn_out * mask_expanded).sum(dim=1)  # [batch, 768]
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
        pooled = pooled / counts

        # Zero out fully-masked samples (safe now — no NaN to propagate)
        if needs_dummy:
            pooled = pooled * (~all_masked).unsqueeze(-1).float()

        return self.proj(pooled)


class MacroBranch(nn.Module):
    """Process macro + session features.

    Architecture: Linear(N→32) → GELU → Dropout → Linear(32→32)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.macro_d_model

        self.net = nn.Sequential(
            nn.Linear(config.num_macro_features, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_macro_features]
        Returns:
            [batch, macro_d_model]
        """
        return self.net(x)


class CrossAssetBranch(nn.Module):
    """Process cross-asset temporal features.

    Architecture: Conv1D(A→64→64) → CrossAttention(query=price_embed, kv=cross)
    The cross-attention lets the price branch query attend to cross-asset features.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.cross_d_model

        # Conv to process cross-asset temporal features
        self.conv1 = nn.Conv1d(config.cross_asset_channels, d, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d)
        self.act = nn.GELU()

        # Pool temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Linear(d, d)

    def forward(
        self,
        x: torch.Tensor,
        price_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, cross_asset_channels]
            price_embed: [batch, price_d_model] (unused in simplified version)
        Returns:
            [batch, cross_d_model]
        """
        h = x.transpose(1, 2)  # [batch, channels, seq]
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))

        # Pool to single vector
        h = self.pool(h).squeeze(-1)  # [batch, d]

        return self.output_proj(h)
