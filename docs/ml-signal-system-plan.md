# TemporalFusionSignalNet — Full Implementation Plan

## Overview

A **6-branch multi-modal PyTorch neural network** that predicts buy/sell/hold signals on 1H bars. Separate models per asset (gold, BTC, crude, ETFs, AAPL). Two-phase training: supervised classification with triple-barrier labels, then RL fine-tuning with PPO.

**Hardware**: RTX 3080 Laptop (8GB VRAM) for dev/inference + Google Colab (T4/A100) for heavy training.
**Inference**: Batch mode via hourly cron job per bar close.

---

## 1. Architecture: TemporalFusionSignalNet

### 1.1 High-Level Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         OUTPUT HEADS                             │
│                                                                  │
│   Classification Head              Confidence Head               │
│   Linear(256 → 3) → Softmax       Linear(256 → 1) → Sigmoid    │
│   → P(sell), P(hold), P(buy)      → [0, 1] calibrated conf     │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────────────┐
│                    FUSION TRANSFORMER                             │
│                                                                  │
│   Project all branches → d=64, stack: [batch, 6, 64]            │
│   2-layer Multi-Head Self-Attention (4 heads, dropout=0.1)      │
│   Mean pool across 6 branch tokens → [batch, 64]               │
│   Linear(64 → 128) → GELU → Dropout(0.2) → Linear(128 → 256)  │
│   → fused embedding [batch, 256]                                 │
│                                                                  │
└──┬─────────┬─────────┬─────────┬──────────┬──────────┬──────────┘
   │         │         │         │          │          │
┌──┴──┐  ┌───┴──┐  ┌──┴───┐  ┌──┴──┐  ┌───┴───┐  ┌──┴─────┐
│PRICE│  │VOLUME│  │PATT- │  │NEWS │  │MACRO/ │  │CROSS-  │
│/IND │  │      │  │ERNS  │  │/NLP │  │SESSION│  │ASSET   │
│     │  │      │  │      │  │     │  │       │  │        │
│[128]│  │ [64] │  │ [64] │  │[64] │  │ [32]  │  │ [64]   │
└─────┘  └──────┘  └──────┘  └─────┘  └───────┘  └────────┘
```

### 1.2 Branch 1 — Price/Indicator Branch (Temporal, Multi-Resolution)

**Purpose**: Encode price action, derivatives, and all technical indicators at two resolutions.

**Input Shapes**:
- 1H path: `[batch, 120, C_1h]` — 120 hourly bars, C_1h ~ 47 channels
- 15m path: `[batch, 480, C_15m]` — 480 fifteen-minute bars, C_15m ~ 47 channels

**Feature Channels (~47 per bar)**:
```
OHLCV                           5 channels (open, high, low, close, volume)
Price return (1st derivative)   1 channel  (close.pct_change())
Price accel (2nd derivative)    1 channel  (close.pct_change().diff())
SMA(20)                         1 channel
SMA(50)                         1 channel
EMA(12)                         1 channel
EMA(26)                         1 channel
RSI(14)                         1 channel
MACD                            3 channels (macd_line, signal_line, histogram)
Bollinger Bands                 3 channels (upper, middle, lower)
ATR(14)                         1 channel
Stochastic                      2 channels (%K, %D)
Parabolic SAR                   1 channel
Ichimoku Cloud                  5 channels (tenkan, kijun, senkou_a, senkou_b, chikou)
OBV                             1 channel
VWAP                            1 channel
Accumulation/Distribution       1 channel
CMF(20)                         1 channel
MFI(14)                         1 channel
Fibonacci Retracement           5 channels (0%, 23.6%, 38.2%, 50%, 61.8%)
Pivot Points                    3 channels (pivot, support1, resistance1)
Rolling Volatility              1 channel
Rolling High/Low                2 channels
Breakout Signal                 1 channel
Returns (log)                   1 channel
─────────────────────────────────────
Total                          ~47 channels
```

**Architecture (PyTorch pseudocode)**:
```python
class PriceBranch(nn.Module):
    def __init__(self, in_channels=47, d_model=128, nhead=4):
        # --- 1H path ---
        self.conv1h_1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1h_1 = nn.BatchNorm1d(64)
        self.conv1h_2 = nn.Conv1d(64, d_model, kernel_size=3, padding=1)
        self.bn1h_2 = nn.BatchNorm1d(d_model)
        self.pos_enc_1h = PositionalEncoding(d_model, max_len=120)
        self.transformer_1h = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=256, dropout=0.1,
                                       activation='gelu', batch_first=True),
            num_layers=2
        )

        # --- 15m path ---
        self.conv15m_1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=2)  # 480→238
        self.bn15m_1 = nn.BatchNorm1d(64)
        self.conv15m_2 = nn.Conv1d(64, d_model, kernel_size=3, stride=2)      # 238→118
        self.bn15m_2 = nn.BatchNorm1d(d_model)
        self.pos_enc_15m = PositionalEncoding(d_model, max_len=120)
        self.transformer_15m = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=256, dropout=0.1,
                                       activation='gelu', batch_first=True),
            num_layers=1
        )

        # --- Merge ---
        self.merge = nn.Linear(d_model * 2, d_model)  # 256 → 128

    def forward(self, x_1h, x_15m):
        # x_1h: [B, 120, 47] → transpose for conv → [B, 47, 120]
        h1 = F.gelu(self.bn1h_1(self.conv1h_1(x_1h.transpose(1,2))))
        h1 = F.gelu(self.bn1h_2(self.conv1h_2(h1)))
        h1 = h1.transpose(1,2)              # back to [B, 120, 128]
        h1 = self.pos_enc_1h(h1)
        h1 = self.transformer_1h(h1)
        h1 = h1[:, -1, :]                   # last hidden: [B, 128]

        # x_15m: [B, 480, 47]
        h2 = F.gelu(self.bn15m_1(self.conv15m_1(x_15m.transpose(1,2))))
        h2 = F.gelu(self.bn15m_2(self.conv15m_2(h2)))
        h2 = h2.transpose(1,2)              # [B, ~118, 128]
        h2 = self.pos_enc_15m(h2)
        h2 = self.transformer_15m(h2)
        h2 = h2[:, -1, :]                   # [B, 128]

        return self.merge(torch.cat([h1, h2], dim=-1))  # [B, 128]
```

### 1.3 Branch 2 — Volume Branch (Temporal)

**Purpose**: Encode volume dynamics separately (volume often leads price).

**Input**: `[batch, 120, 3]`
- Channel 0: raw volume (normalized)
- Channel 1: volume return (1st derivative)
- Channel 2: volume acceleration (2nd derivative)

```python
class VolumeBranch(nn.Module):
    def __init__(self, in_channels=3, d_model=64, nhead=2):
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=128, dropout=0.1,
                                       activation='gelu', batch_first=True),
            num_layers=1
        )

    def forward(self, x):
        # x: [B, 120, 3]
        h = F.gelu(self.bn1(self.conv1(x.transpose(1,2))))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.transformer(h.transpose(1,2))
        return h[:, -1, :]  # [B, 64]
```

### 1.4 Branch 3 — Pattern Branch (Categorical/Binary)

**Purpose**: Encode detected chart patterns, candlestick patterns, harmonic patterns, and Elliott Wave signals.

**Input**: `[batch, P]` where P ~ 100
- ~50 patterns × 2 values each (detected: 0/1, confidence: 0.0-1.0)
- Breakdown:
  - 19 existing chart patterns (H&S, double top/bottom, flags, wedges, triangles, gaps, etc.)
  - 15 candlestick patterns (doji, hammer, engulfing, harami, morning/evening star, etc.)
  - 9 harmonic patterns (Gartley, butterfly, bat, crab, shark, cypher, AB=CD, three drives, Wolfe wave)
  - 5 advanced patterns (Quasimodo, dead cat bounce, island reversal, tower top/bottom)
  - ~2 Elliott Wave indicators (impulse wave detected, corrective wave detected)

```python
class PatternBranch(nn.Module):
    def __init__(self, num_features=100, d_model=64):
        self.net = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, 100]
        return self.net(x)  # [B, 64]
```

### 1.5 Branch 4 — News/NLP Branch

**Purpose**: Encode recent news sentiment from pre-computed FinBERT article embeddings.

**Input**:
- `news`: `[batch, N_max, 768]` — up to 10 articles per prediction window, pre-computed FinBERT embeddings
- `news_mask`: `[batch, N_max]` — boolean mask for valid articles (padded positions = False)

**Design**: Temporal attention over articles (newer articles weighted more), then project to branch embedding.

```python
class NewsBranch(nn.Module):
    def __init__(self, embed_dim=768, d_model=64, nhead=4):
        self.attention = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=0.1)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model),
        )

    def forward(self, news, news_mask):
        # news: [B, N, 768], news_mask: [B, N]
        # Self-attention across articles
        key_padding_mask = ~news_mask  # True = ignore position
        h, _ = self.attention(news, news, news, key_padding_mask=key_padding_mask)
        # Masked mean pooling
        mask_expanded = news_mask.unsqueeze(-1).float()  # [B, N, 1]
        h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # [B, 768]
        return self.project(h)  # [B, 64]
```

**FinBERT Pre-computation Pipeline**:
```
Articles (RSS/API) → ProsusAI/finbert → 768-dim embedding per article
                   → stored in HDF5: {timestamp, symbol, embedding, source, title}
                   → loaded at training/inference time
```

### 1.6 Branch 5 — Macro/Session Branch (Tabular)

**Purpose**: Encode macroeconomic conditions and market session timing.

**Input**: `[batch, M]` where M ~ 13

| Feature | Source | Freq | Encoding |
|---------|--------|------|----------|
| Fed Funds Rate | FRED FEDFUNDS | Daily | Raw value |
| 2Y Treasury Yield | FRED DGS2 | Daily | Raw value |
| 10Y Treasury Yield | FRED DGS10 | Daily | Raw value |
| 30Y Treasury Yield | FRED DGS30 | Daily | Raw value |
| CPI (YoY change) | FRED CPIAUCSL | Monthly | Percentage change |
| Unemployment Rate | FRED UNRATE | Monthly | Raw value |
| Hour of Day | Timestamp | Per bar | sin(2pi*h/24), cos(2pi*h/24) |
| Day of Week | Timestamp | Per bar | sin(2pi*d/7), cos(2pi*d/7) |
| Is Market Open | Session config | Per bar | Binary (0/1) |
| Minutes Since Open | Session config | Per bar | Normalized [0, 1] |
| Minutes To Close | Session config | Per bar | Normalized [0, 1] |

**Total**: 13 features (6 macro + 4 cyclical + 3 session)

```python
class MacroBranch(nn.Module):
    def __init__(self, num_features=13, d_model=32):
        self.net = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, 13]
        return self.net(x)  # [B, 32]
```

### 1.7 Branch 6 — Cross-Asset Branch (Temporal with Cross-Attention)

**Purpose**: Capture inter-market dynamics and correlations.

**Input**: `[batch, 120, A]` where A = num_correlated_assets × 5 (OHLCV per asset)

**Cross-Asset Pairs per Model**:
| Primary Asset | Correlated Assets | A |
|--------------|-------------------|---|
| GC=F (Gold) | BTC-USD, CL=F, SPY, ^IRX (2Y Treasury) | 20 |
| BTC-USD | GC=F, SPY, ^TNX (10Y Treasury) | 15 |
| CL=F (Crude) | GC=F, SPY, ^TNX | 15 |
| SPY/QQQ | GC=F, ^TNX, DX-Y.NYB (Dollar Index) | 15 |
| AAPL | SPY, QQQ, ^TNX | 15 |

**Architecture**: Conv encoder for cross-asset data, then cross-attention where the price branch attends to cross-asset features.

```python
class CrossAssetBranch(nn.Module):
    def __init__(self, in_channels, d_model=64, nhead=2):
        self.conv1 = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        # Cross-attention: primary price branch queries, cross-asset is key/value
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.query_proj = nn.Linear(128, d_model)  # project price_embed to d_model

    def forward(self, cross_asset_data, price_embed):
        # cross_asset_data: [B, 120, A]
        h = F.gelu(self.bn1(self.conv1(cross_asset_data.transpose(1,2))))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = h.transpose(1,2)  # [B, 120, 64]

        # Cross-attention: price_embed queries cross-asset features
        query = self.query_proj(price_embed).unsqueeze(1)  # [B, 1, 64]
        out, _ = self.cross_attn(query, h, h)              # [B, 1, 64]
        return out.squeeze(1)                               # [B, 64]
```

### 1.8 Fusion Transformer

```python
class FusionTransformer(nn.Module):
    def __init__(self, branch_dims=[128, 64, 64, 64, 32, 64], d_fusion=64, nhead=4, num_layers=2):
        # Project each branch to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, d_fusion) for dim in branch_dims
        ])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_fusion, nhead=nhead,
                                       dim_feedforward=128, dropout=0.1,
                                       activation='gelu', batch_first=True),
            num_layers=num_layers
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_fusion, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
        )

    def forward(self, branch_embeds: list[torch.Tensor]):
        # Project each branch: [B, dim_i] → [B, 64]
        projected = [proj(embed) for proj, embed in zip(self.projections, branch_embeds)]
        stacked = torch.stack(projected, dim=1)  # [B, 6, 64]

        # Self-attention across modalities
        fused = self.transformer(stacked)        # [B, 6, 64]
        fused = fused.mean(dim=1)                # [B, 64] — mean pool

        return self.output_proj(fused)           # [B, 256]
```

### 1.9 Full Model Assembly

```python
class TemporalFusionSignalNet(nn.Module):
    def __init__(self, config: ModelConfig):
        self.price_branch = PriceBranch(in_channels=config.price_channels)
        self.volume_branch = VolumeBranch()
        self.pattern_branch = PatternBranch(num_features=config.num_patterns * 2)
        self.news_branch = NewsBranch()
        self.macro_branch = MacroBranch(num_features=config.num_macro_features)
        self.cross_asset_branch = CrossAssetBranch(in_channels=config.cross_asset_channels)
        self.fusion = FusionTransformer()

        # Output heads
        self.classifier = nn.Linear(256, 3)
        self.confidence_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, batch: dict[str, torch.Tensor]):
        # Encode each modality
        price_embed = self.price_branch(batch["price_1h"], batch["price_15m"])
        volume_embed = self.volume_branch(batch["volume"])
        pattern_embed = self.pattern_branch(batch["patterns"])
        news_embed = self.news_branch(batch["news"], batch["news_mask"])
        macro_embed = self.macro_branch(batch["macro"])
        cross_embed = self.cross_asset_branch(batch["cross_asset"], price_embed)

        # Fuse
        fused = self.fusion([price_embed, volume_embed, pattern_embed,
                             news_embed, macro_embed, cross_embed])

        # Predict
        logits = self.classifier(fused)          # [B, 3]
        confidence = self.confidence_head(fused)  # [B, 1]

        return logits, confidence.squeeze(-1)

    def get_fused_embedding(self, batch):
        """For RL: return fused embedding without heads."""
        # Same as forward but return fused before heads
        ...
```

### 1.10 Model Size Estimate

| Component | Parameters |
|-----------|-----------|
| Price/Indicator Branch (1H + 15m) | ~800K |
| Volume Branch | ~50K |
| Pattern Branch | ~12K |
| News Branch | ~120K |
| Macro Branch | ~2K |
| Cross-Asset Branch | ~60K |
| Fusion Transformer | ~80K |
| Output Heads | ~1K |
| **Total** | **~1.15M** |

**VRAM Usage** (FP16, batch=128): ~200MB model + ~500MB activations = well under 8GB.

### 1.11 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return self.dropout(x + self.pe[:, :x.size(1)])
```

---

## 2. Data Pipeline — Detailed Design

### 2.1 Triple-Barrier Labeling

```python
def triple_barrier_label(
    df: pd.DataFrame,
    tp_mult: float = 2.0,     # TP barrier = tp_mult × ATR
    sl_mult: float = 1.5,     # SL barrier = sl_mult × ATR
    max_bars: int = 5,        # time barrier (prediction horizon)
    atr_period: int = 14,
    fee_bps: int = 10,        # 10 bps = 0.1%
) -> pd.Series:
    """
    For each bar i:
      1. Compute ATR at bar i
      2. Set TP = close[i] + tp_mult * ATR[i] - 2 * fee (round-trip)
      3. Set SL = close[i] - sl_mult * ATR[i] + 2 * fee
      4. Look forward up to max_bars:
         - If high[j] >= TP before low[j] <= SL → label = +1 (buy signal)
         - If low[j] <= SL before high[j] >= TP → label = -1 (sell signal)
         - If time barrier reached → label = 0 (hold)

    Fee adjustment ensures only trades that overcome fees get labeled.
    """
```

**Label Distribution (expected)**:
- Hold (0): ~50-60% of bars (most common)
- Buy (+1): ~20-25%
- Sell (-1): ~20-25%
- This imbalance is why we use Focal Loss

### 2.2 Feature Engineering Pipeline

```python
def build_feature_matrix(
    df_1h: pd.DataFrame,           # Primary OHLCV at 1H
    df_15m: pd.DataFrame,          # Sub-bar OHLCV at 15m
    cross_asset_dfs: dict[str, pd.DataFrame],  # Correlated asset OHLCV
    news_embeddings: np.ndarray,   # Pre-computed FinBERT [N_articles, 768]
    news_timestamps: np.ndarray,   # Article timestamps
    macro_data: pd.DataFrame,      # FRED macro data (forward-filled to 1H)
    asset_config: AssetConfig,     # Session hours, asset type, etc.
) -> dict[str, np.ndarray]:
    """
    Build all feature tensors for the full dataset.

    Steps:
    1. Compute all technical indicators on df_1h using existing INDICATORS_CONFIG
    2. Flatten multi-output indicators into separate columns
    3. Compute price/volume derivatives
    4. Apply rolling z-score normalization (500-bar lookback)
    5. Detect all patterns (chart + candlestick + harmonic + Elliott)
    6. Encode session timing features (cyclical + binary)
    7. Align cross-asset data to same timestamps
    8. Create sliding window samples of length 120 (1H) / 480 (15m)

    Returns dict of numpy arrays ready for TradingSignalDataset.
    """
```

**Normalization Strategy**: Rolling z-score with 500-bar lookback.
```python
def rolling_zscore(series: pd.Series, lookback: int = 500) -> pd.Series:
    mean = series.rolling(lookback, min_periods=50).mean()
    std = series.rolling(lookback, min_periods=50).std()
    return (series - mean) / std.clip(lower=1e-8)
```

- Applied per feature column independently
- Lookback uses only past data (no look-ahead bias)
- First 50 bars are NaN (discarded from training)
- OHLCV prices normalized relative to each other (divide by close)

### 2.3 Macro Data Fetcher

```python
class MacroDataFetcher:
    """Fetch macroeconomic data from FRED API."""

    SERIES_MAP = {
        "fed_funds_rate": "FEDFUNDS",
        "treasury_2y": "DGS2",
        "treasury_10y": "DGS10",
        "treasury_30y": "DGS30",
        "cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
    }

    def fetch(self, start_date, end_date) -> pd.DataFrame:
        """
        Fetch all FRED series, resample/forward-fill to 1H frequency.

        CPI: convert to YoY percentage change before forward-fill.
        All others: forward-fill directly.
        """
```

### 2.4 Session Timing Features

```python
ASSET_SESSIONS = {
    # (open_hour_utc, close_hour_utc, trading_days)
    "GC=F":    (22, 21, "Sun-Fri"),    # Nearly 24h Sun-Fri
    "CL=F":    (23, 22, "Sun-Fri"),    # Nearly 24h Sun-Fri
    "SPY":     (14, 21, "Mon-Fri"),    # 9:30am-4pm ET = 14:30-21:00 UTC
    "QQQ":     (14, 21, "Mon-Fri"),
    "AAPL":    (14, 21, "Mon-Fri"),
    "BTC-USD": (0, 0, "24/7"),         # 24/7, encode NYC session overlap
}

def encode_session_features(timestamps: pd.DatetimeIndex, asset: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - hour_sin, hour_cos: cyclical hour of day
    - dow_sin, dow_cos: cyclical day of week
    - is_market_open: 1 if within session hours, 0 otherwise
    - minutes_since_open: normalized [0, 1] within session
    - minutes_to_close: normalized [0, 1] within session
    """
```

### 2.5 Walk-Forward Validation Splits

```
Gold (~3,500 bars) example:

Split 0:  TRAIN [0 ────── 2000]  VAL [2000 ── 2500]
Split 1:  TRAIN [0 ──────────── 2500]  VAL [2500 ── 3000]
Split 2:  TRAIN [0 ────────────────── 3000]  VAL [3000 ── 3500]

BTC (~17,500 bars) example:

Split 0:  TRAIN [0 ────── 5000]  VAL [5000 ── 6000]
Split 1:  TRAIN [0 ──────────── 6000]  VAL [6000 ── 7000]
...
Split 11: TRAIN [0 ────────────────── 16500]  VAL [16500 ── 17500]
```

### 2.6 PyTorch Dataset

```python
class TradingSignalDataset(torch.utils.data.Dataset):
    """
    Loads pre-built feature arrays from HDF5 files.

    Directory structure:
    data/ml/{asset}/{split}/
        price_1h.h5       # [N_samples, 120, C_1h]
        price_15m.h5      # [N_samples, 480, C_15m]
        volume.h5         # [N_samples, 120, 3]
        patterns.h5       # [N_samples, P]
        news.h5           # [N_samples, 10, 768]
        news_mask.h5      # [N_samples, 10]
        macro.h5          # [N_samples, M]
        cross_asset.h5    # [N_samples, 120, A]
        labels.h5         # [N_samples] — values {0, 1, 2}
        weights.h5        # [N_samples] — class balancing weights
    """

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {
            "price_1h": torch.tensor(self.price_1h[idx], dtype=torch.float32),
            "price_15m": torch.tensor(self.price_15m[idx], dtype=torch.float32),
            "volume": torch.tensor(self.volume[idx], dtype=torch.float32),
            "patterns": torch.tensor(self.patterns[idx], dtype=torch.float32),
            "news": torch.tensor(self.news[idx], dtype=torch.float32),
            "news_mask": torch.tensor(self.news_mask[idx], dtype=torch.bool),
            "macro": torch.tensor(self.macro[idx], dtype=torch.float32),
            "cross_asset": torch.tensor(self.cross_asset[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }
```

### 2.7 Data Builder Orchestrator

```python
class DatasetBuilder:
    """
    Orchestrates end-to-end dataset construction:

    1. Download 1H + 15m bars via YFinanceProvider (reuse existing)
    2. Download cross-asset data for correlated symbols
    3. Fetch macro data from FRED
    4. Load pre-computed FinBERT embeddings from HDF5
    5. Compute all technical indicators (reuse INDICATORS_CONFIG)
    6. Detect all patterns (chart + candlestick + harmonic + Elliott)
    7. Build feature matrices with rolling z-score normalization
    8. Generate triple-barrier labels
    9. Create walk-forward splits
    10. Save as HDF5 files in data/ml/{asset}/{split}/
    """

    def build(self, asset_config: AssetConfig):
        ...
```

---

## 3. Training Pipeline — Detailed Design

### 3.1 Phase 1: Supervised Pre-training

**Loss Function: Focal Loss**
```python
def focal_loss(logits, targets, alpha=[0.25, 0.5, 0.25], gamma=2.0):
    """
    Focal Loss for multi-class classification.

    Handles class imbalance (hold >> buy/sell) by down-weighting easy examples.
    alpha: per-class weights — [sell, hold, buy]. Hold gets 0.5 (down-weighted).
    gamma: focusing parameter. Higher = more focus on hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    ce = F.cross_entropy(logits, targets, weight=torch.tensor(alpha), reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
```

**Confidence Loss (auxiliary)**:
```python
# BCE: predicted confidence vs actual correctness
conf_target = (logits.argmax(-1) == labels).float()
conf_loss = F.binary_cross_entropy(confidence, conf_target)
total_loss = focal_loss + 0.1 * conf_loss
```

**Training Loop**:
```python
class SupervisedTrainer:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=1e-3, total_steps=config.total_steps
        )
        self.scaler = torch.amp.GradScaler('cuda')  # FP16

    def train_epoch(self, loader):
        self.model.train()
        for batch in loader:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, confidence = self.model(batch)
                loss = self.compute_loss(logits, confidence, batch["label"])

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def validate(self, loader) -> dict:
        """Returns accuracy, F1 (macro), per-class metrics, Sharpe ratio."""
        ...
```

**Hyperparameters**:
| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 128 (laptop) / 256 (Colab) | FP16 |
| Learning rate | 1e-3 | OneCycleLR with cosine annealing |
| Weight decay | 1e-4 | AdamW regularization |
| Epochs | 50 max | Early stopping patience=10 |
| Gradient clipping | max_norm=1.0 | Stabilize training |
| Dropout | 0.1-0.2 | Per-branch (see branch specs) |
| Label smoothing | 0.0 | Not used (Focal Loss handles it) |

### 3.2 Phase 2: RL Fine-tuning (PPO)

**Gymnasium Environment**:
```python
class TradingEnv(gymnasium.Env):
    """
    Custom trading environment for RL fine-tuning.

    State: fused embedding [256] from frozen pre-trained encoder
    Action: discrete {0=sell, 1=hold, 2=buy}
    Reward: risk-adjusted per-bar P&L
    """

    def __init__(self, data, fee_bps=10, lambda_drawdown=0.5):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(256,))
        self.action_space = spaces.Discrete(3)
        self.data = data
        self.fee = fee_bps / 10000
        self.lambda_dd = lambda_drawdown

    def step(self, action):
        # Map action to position: {0: -1, 1: 0, 2: +1}
        position = action - 1

        # Compute bar return
        bar_return = self.data[self.idx]["close_return"]

        # P&L
        pnl = position * bar_return

        # Fee for position change
        if position != self.prev_position:
            pnl -= abs(position - self.prev_position) * self.fee

        # Drawdown penalty
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-8)
        drawdown_penalty = self.lambda_dd * drawdown

        reward = pnl - drawdown_penalty

        self.prev_position = position
        self.idx += 1
        done = self.idx >= len(self.data)

        return self.get_obs(), reward, done, False, {"equity": self.equity}
```

**RL Training**:
```python
from stable_baselines3 import PPO

# Freeze encoder, fine-tune fusion + head
for param in model.price_branch.parameters():
    param.requires_grad = False
# ... freeze all branch encoders

env = TradingEnv(validation_data, fee_bps=10)
ppo_model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, verbose=1)
ppo_model.learn(total_timesteps=500_000)
```

### 3.3 Walk-Forward Training Orchestration

```
For each walk-forward fold:
  1. Create DataLoaders for train/val splits
  2. Train supervised model (Phase 1)
  3. Evaluate on validation fold → classification + financial metrics
  4. Save checkpoint

After all folds:
  5. Ensemble: load all fold models, average logits at inference

Optional Phase 2 (per-fold):
  6. Fine-tune with PPO on training data
  7. Compare supervised vs. supervised+RL on validation fold
```

---

## 4. Position Sizing & Risk Management

### 4.1 Confidence-Based Position Sizing

```python
def confidence_to_size(
    confidence: float,
    base_size: float = 1.0,       # 1.0 = full single unit
    max_leverage: float = 3.0,
    min_confidence: float = 0.55,  # below this = no trade
) -> float:
    """
    Maps model confidence to position size.

    confidence < 0.55           → 0 (no trade)
    confidence = 0.55           → 1.0× (base size)
    confidence = 0.775          → 2.0×
    confidence = 1.0            → 3.0× (max leverage)

    Linear ramp between min_confidence and 1.0.
    """
    if confidence < min_confidence:
        return 0.0
    scale = (confidence - min_confidence) / (1.0 - min_confidence)
    return base_size + scale * (max_leverage - base_size)
```

### 4.2 Per-Asset Risk Profiles

```python
@dataclass
class AssetRiskProfile:
    max_leverage: float
    max_drawdown_pct: float     # max allowed drawdown before reducing
    max_position_pct: float     # max % of portfolio in single position
    fee_bps: int                # trading fees in basis points
    session_type: str           # "24/7" or "session"

RISK_PROFILES = {
    "GC=F":    AssetRiskProfile(3.0, 15.0, 25.0, 10, "session"),
    "BTC-USD": AssetRiskProfile(2.0, 20.0, 15.0, 10, "24/7"),
    "CL=F":    AssetRiskProfile(2.5, 15.0, 20.0, 10, "session"),
    "SPY":     AssetRiskProfile(2.0, 10.0, 30.0, 5,  "session"),
    "QQQ":     AssetRiskProfile(2.0, 10.0, 30.0, 5,  "session"),
    "AAPL":    AssetRiskProfile(2.0, 15.0, 15.0, 5,  "session"),
}
```

### 4.3 Circuit Breakers

```python
class CircuitBreaker:
    """
    Safety mechanisms that override model predictions.

    Triggers:
    1. Daily loss > 3% of portfolio → flatten ALL positions immediately
    2. Drawdown > 15% from equity peak → reduce max leverage to 1×
    3. Cross-asset correlation > 0.9 → reduce total exposure by 50%
       (prevents concentrated correlated bets)
    """

    def check(self, portfolio_state) -> list[str]:
        """Returns list of triggered circuit breaker names."""
        ...

    def adjust_position(self, signal, confidence, portfolio_state) -> tuple[int, float]:
        """
        Apply circuit breaker adjustments.
        Returns adjusted (signal, position_size).
        """
        ...
```

---

## 5. New Pattern Modules — Detailed Specs

### 5.1 Candlestick Patterns

File: `src/quantdash/features/candlestick_patterns.py`

Convention: Follow existing `patterns.py` — each detector returns `list[PatternEvent]` from `core/models.py`.

| Pattern | Detection Logic | Direction |
|---------|----------------|-----------|
| Standard Doji | `abs(open - close) / (high - low) < 0.1` | Neutral |
| Long-legged Doji | Doji + upper shadow > 2× body AND lower shadow > 2× body | Neutral |
| Dragonfly Doji | Doji + lower shadow > 3× body, negligible upper shadow | Bullish |
| Gravestone Doji | Doji + upper shadow > 3× body, negligible lower shadow | Bearish |
| Hammer | Small body at top 1/3 of range, lower shadow >= 2× body, in downtrend | Bullish |
| Inverted Hammer | Small body at bottom 1/3, upper shadow >= 2× body, in downtrend | Bullish |
| Hanging Man | Same shape as Hammer but in uptrend | Bearish |
| Shooting Star | Same shape as Inverted Hammer but in uptrend | Bearish |
| Bullish Engulfing | Previous bearish, current bullish body fully engulfs previous body | Bullish |
| Bearish Engulfing | Previous bullish, current bearish body fully engulfs previous body | Bearish |
| Bullish Harami | Previous large bearish, current small bullish inside previous | Bullish |
| Bearish Harami | Previous large bullish, current small bearish inside previous | Bearish |
| Morning Star | 3-bar: large bearish → small body/doji → large bullish (gap optional) | Bullish |
| Evening Star | 3-bar: large bullish → small body/doji → large bearish | Bearish |
| Three White Soldiers | 3 consecutive bullish bars, each closing higher, small upper shadows | Bullish |
| Three Black Crows | 3 consecutive bearish bars, each closing lower, small lower shadows | Bearish |
| Marubozu (Bullish) | body/range > 0.95, no shadows, close > open | Bullish |
| Marubozu (Bearish) | body/range > 0.95, no shadows, close < open | Bearish |
| Pin Bar | Long shadow >= 2/3 of total range, body in opposite 1/3 | Reversal |
| Spinning Top | Small body (< 1/3 range), shadows on both sides | Neutral |
| Tweezer Top | Two bars with matching highs (within tolerance), in uptrend | Bearish |
| Tweezer Bottom | Two bars with matching lows (within tolerance), in downtrend | Bullish |

**Trend Detection for Context**: Use 20-bar SMA slope to determine if in uptrend/downtrend.

### 5.2 Harmonic Patterns

File: `src/quantdash/features/harmonic_patterns.py`

Harmonic patterns use Fibonacci ratio relationships between 5 swing points (X-A-B-C-D):

```
X ─── swing start
│
A ─── first impulse
│
B ─── first retracement (AB = fib ratio of XA)
│
C ─── second impulse (BC = fib ratio of AB)
│
D ─── completion point / PRZ (CD = fib ratio of BC)
```

| Pattern | AB/XA Ratio | BC/AB Ratio | CD/BC Ratio | XD/XA Ratio |
|---------|------------|------------|------------|-------------|
| Gartley | 0.618 | 0.382-0.886 | 1.272-1.618 | 0.786 |
| Butterfly | 0.786 | 0.382-0.886 | 1.618-2.618 | 1.272-1.618 |
| Bat | 0.382-0.5 | 0.382-0.886 | 1.618-2.618 | 0.886 |
| Crab | 0.382-0.618 | 0.382-0.886 | 2.24-3.618 | 1.618 |
| Shark | - | 1.13-1.618 | 1.618-2.24 | - |
| Cypher | 0.382-0.618 | 1.13-1.414 | - | 0.786 |
| AB=CD | - | - | 1.0 (CD = AB) | - |
| Three Drives | Each drive 1.272-1.618 extension of previous | - | - | - |
| Wolfe Wave | 5-point channel pattern (1-2, 2-3, 3-4, 4-5) | - | - | - |

**Implementation Approach**:
1. Reuse `find_local_extrema()` from existing `patterns.py`
2. Identify candidate X-A-B-C-D swing sequences
3. Check Fibonacci ratios within tolerance (±5%)
4. Calculate PRZ (Potential Reversal Zone) at D
5. Confidence = how closely ratios match ideal values

### 5.3 Elliott Wave (Simplified)

File: `src/quantdash/features/elliott_wave.py`

```
Impulse Wave (5 waves in trend direction):
    Wave 1: Initial move
    Wave 2: Correction (does NOT retrace 100% of Wave 1)
    Wave 3: Strongest move (NEVER shortest of 1,3,5)
    Wave 4: Correction (does NOT overlap Wave 1 price territory)
    Wave 5: Final move in trend direction

Corrective Wave (3 waves against trend):
    Wave A: Counter-trend move
    Wave B: Partial retracement of A
    Wave C: Final counter-trend move
```

**Rules checked**:
1. Wave 2 retraces 50-78.6% of Wave 1
2. Wave 3 is never the shortest impulse wave
3. Wave 4 does not enter Wave 1's price range
4. Wave 3 typically extends 1.618× Wave 1

**Output**: Binary flag per bar (impulse_detected, corrective_detected) + wave position.

### 5.4 Advanced Chart Patterns

File: `src/quantdash/features/advanced_patterns.py`

| Pattern | Detection | Direction |
|---------|-----------|-----------|
| Quasimodo | Higher high, then lower low breaking structure | Reversal |
| Dead Cat Bounce | Sharp decline (>5%), small recovery (<38.2% fib), continued decline | Bearish |
| Island Reversal | Gap up + gap down (or vice versa) isolating bars | Reversal |
| Tower Top | Strong rally, 2-3 bar consolidation at top, strong decline | Bearish |
| Tower Bottom | Strong decline, 2-3 bar consolidation at bottom, strong rally | Bullish |

---

## 6. News/NLP Pipeline — Detailed Design

### 6.1 FinBERT Embedding Pipeline

```python
class FinBERTEmbedder:
    """
    Pre-compute FinBERT embeddings for financial news articles.
    Uses ProsusAI/finbert (110M params, 1.7GB).

    Run offline on GPU, store embeddings in HDF5.
    NOT loaded during training — only the pre-computed 768-dim vectors.
    """

    def __init__(self, model_name="ProsusAI/finbert", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def embed_articles(self, articles: list[dict]) -> np.ndarray:
        """
        articles: list of {"text": str, "timestamp": datetime, "source": str, "title": str}
        Returns: [N, 768] float32 array
        """
        embeddings = []
        with torch.no_grad():
            for article in articles:
                tokens = self.tokenizer(article["text"], return_tensors="pt",
                                        max_length=512, truncation=True, padding=True)
                output = self.model(**tokens.to(self.device))
                cls_embed = output.last_hidden_state[:, 0, :]  # CLS token
                embeddings.append(cls_embed.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def save_to_hdf5(self, embeddings, metadata, path):
        """Save embeddings + metadata to HDF5 indexed by timestamp."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings, compression='gzip')
            f.create_dataset('timestamps', data=metadata['timestamps'])
            f.create_dataset('sources', data=metadata['sources'])
```

### 6.2 News Data Sources

Existing infrastructure in `src/quantdash/services/api/routes/news.py`:
- RSS feeds from Google News and Yahoo Finance
- Sentiment scoring via keyword matching

**For ML pipeline, extend to**:
- More comprehensive RSS feeds per asset (e.g., "gold price news", "bitcoin news")
- Optional: NewsAPI, Alpha Vantage news, or Benzinga API for higher quality
- Store raw articles in SQLite (existing DB infrastructure) with timestamps
- Pre-compute FinBERT embeddings in batch

### 6.3 Article-to-Bar Alignment

For each bar at time `t`, find articles within a window:
- **Window**: [t - 24h, t] for recent news impact
- **Max articles**: 10 per bar window (sorted by recency)
- **Time-decay**: More recent articles have higher attention weights (implicit via attention mechanism)
- **No articles**: Mask all positions → NewsBranch returns zero vector

---

## 7. Inference & Deployment

### 7.1 Batch Inference Pipeline

```python
class SignalPredictor:
    """
    Loads trained model and runs batch inference.

    Workflow:
    1. Load latest data (1H + 15m bars, cross-asset, macro, news)
    2. Build feature matrix for current window
    3. Run model forward pass
    4. Apply confidence-based position sizing
    5. Apply circuit breaker checks
    6. Output: {action, confidence, position_size, reasoning}
    """

    def __init__(self, model_path: Path, asset_config: AssetConfig):
        self.model = TemporalFusionSignalNet.load(model_path)
        self.model.eval()
        self.asset_config = asset_config

    def predict(self) -> dict:
        batch = self.build_current_features()
        with torch.no_grad():
            logits, confidence = self.model(batch)
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax().item() - 1  # map {0,1,2} → {-1,0,+1}
        conf = confidence.item()

        # Position sizing
        size = confidence_to_size(conf, **self.asset_config.risk_profile)

        # Circuit breaker check
        breaker = CircuitBreaker()
        action, size = breaker.adjust_position(action, size, self.portfolio_state)

        return {
            "action": action,          # -1, 0, +1
            "confidence": conf,        # [0, 1]
            "position_size": size,     # 0 to max_leverage
            "probabilities": {
                "sell": probs[0, 0].item(),
                "hold": probs[0, 1].item(),
                "buy": probs[0, 2].item(),
            }
        }
```

### 7.2 Cron Job Setup

```bash
# Hourly cron (runs at minute :05 after bar close)
5 * * * * cd /home/amir/workspace/quant && uv run python scripts/run_inference.py --asset GC=F
5 * * * * cd /home/amir/workspace/quant && uv run python scripts/run_inference.py --asset BTC-USD
5 * * * * cd /home/amir/workspace/quant && uv run python scripts/run_inference.py --asset CL=F
# ... etc for each asset
```

**Output**: JSON signal logged to file + optional Telegram alert (reuses existing TelegramBot from `alerts/telegram.py`).

### 7.3 Strategy Integration

```python
# src/quantdash/ml/strategy/ml_signal_strategy.py

class MLSignalStrategy(Strategy):
    """
    Integrates TemporalFusionSignalNet with existing strategy framework.

    Registered in STRATEGY_REGISTRY so it can be:
    - Backtested via VectorBT engine (poe test / API endpoint)
    - Compared against other strategies
    - Used in portfolio optimization
    """

    name = "ml_signal"
    description = "Multi-modal neural network trading signals"
    default_params = {
        "asset": "GC=F",
        "model_path": "models/gc_signal_net.pt",
        "min_confidence": 0.55,
        "max_leverage": 3.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        predictor = SignalPredictor(Path(self.params["model_path"]),
                                    ASSET_CONFIGS[self.params["asset"]])
        result = df.copy()
        actions, confidences = predictor.predict_batch(df)
        result["signal"] = actions
        result["confidence"] = confidences
        result["position_size"] = [
            confidence_to_size(c, max_leverage=self.params["max_leverage"])
            for c in confidences
        ]
        return result

# Register with existing strategy system
from quantdash.strategies.registry import register_strategy
register_strategy("ml_signal", MLSignalStrategy)
```

---

## 8. Configuration

### 8.1 ML Config (Pydantic Settings)

```python
class ModelConfig(BaseModel):
    """Per-model architecture configuration."""
    price_channels: int = 47
    num_patterns: int = 50
    num_macro_features: int = 13
    cross_asset_channels: int = 20  # varies per asset
    lookback_1h: int = 120
    lookback_15m: int = 480
    max_articles: int = 10
    d_fusion: int = 64
    d_output: int = 256

class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 10
    focal_alpha: list[float] = [0.25, 0.5, 0.25]
    focal_gamma: float = 2.0
    confidence_loss_weight: float = 0.1
    grad_clip_norm: float = 1.0
    use_fp16: bool = True

class LabelingConfig(BaseModel):
    """Triple-barrier labeling parameters."""
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    max_bars: int = 5
    atr_period: int = 14
    fee_bps: int = 10

class AssetConfig(BaseModel):
    """Per-asset configuration."""
    symbol: str
    correlated_assets: list[str]
    session_type: str  # "24/7" or "session"
    session_open_utc: int = 0
    session_close_utc: int = 0
    risk_profile: AssetRiskProfile
    model_config: ModelConfig
    training_config: TrainingConfig
    labeling_config: LabelingConfig

ASSET_CONFIGS = {
    "GC=F": AssetConfig(
        symbol="GC=F",
        correlated_assets=["BTC-USD", "CL=F", "SPY", "^IRX"],
        session_type="session",
        session_open_utc=22, session_close_utc=21,
        risk_profile=AssetRiskProfile(3.0, 15.0, 25.0, 10, "session"),
        model_config=ModelConfig(cross_asset_channels=20),
        training_config=TrainingConfig(),
        labeling_config=LabelingConfig(),
    ),
    "BTC-USD": AssetConfig(
        symbol="BTC-USD",
        correlated_assets=["GC=F", "SPY", "^TNX"],
        session_type="24/7",
        risk_profile=AssetRiskProfile(2.0, 20.0, 15.0, 10, "24/7"),
        model_config=ModelConfig(cross_asset_channels=15),
        training_config=TrainingConfig(),
        labeling_config=LabelingConfig(),
    ),
    # ... similar for CL=F, SPY, QQQ, AAPL
}
```

---

## 9. Implementation Phases — Detailed

### Phase 1: Foundation

**Files to create**:
- `src/quantdash/ml/__init__.py`
- `src/quantdash/ml/config.py` — MLConfig, AssetConfig, ASSET_CONFIGS, all Pydantic models
- `src/quantdash/ml/data/__init__.py`
- `src/quantdash/ml/models/__init__.py`
- `src/quantdash/ml/training/__init__.py`
- `src/quantdash/ml/risk/__init__.py`
- `src/quantdash/ml/inference/__init__.py`
- `src/quantdash/ml/strategy/__init__.py`

**Files to modify**:
- `pyproject.toml` — add `ml` optional dependency group

**Verification**: `uv sync --extra ml` succeeds, `import quantdash.ml` works.

---

### Phase 2: New Patterns

**Files to create**:
- `src/quantdash/features/candlestick_patterns.py` — 15+ candlestick patterns
- `src/quantdash/features/harmonic_patterns.py` — 9 harmonic patterns
- `src/quantdash/features/elliott_wave.py` — Elliott Wave detection
- `src/quantdash/features/advanced_patterns.py` — 5 advanced patterns
- `tests/unit/test_candlestick_patterns.py`
- `tests/unit/test_harmonic_patterns.py`
- `tests/unit/test_elliott_wave.py`
- `tests/unit/test_advanced_patterns.py`

**Convention**: Follow `patterns.py` — each function returns `list[PatternEvent]`, register in `*_PATTERNS_CONFIG` dicts.

**Verification**: `poe test-fast tests/unit/test_candlestick_patterns.py` etc. all pass.

---

### Phase 3: Data Pipeline

**Files to create**:
- `src/quantdash/ml/data/features.py` — `build_feature_matrix()`, indicator flattening, normalization
- `src/quantdash/ml/data/labeling.py` — `triple_barrier_label()` with fee adjustment
- `src/quantdash/ml/data/dataset.py` — `TradingSignalDataset` (PyTorch Dataset)
- `src/quantdash/ml/data/splits.py` — `walk_forward_splits()`
- `src/quantdash/ml/data/macro.py` — FRED API fetcher
- `src/quantdash/ml/data/builder.py` — orchestrator
- `scripts/build_dataset.py` — CLI script
- `tests/unit/test_labeling.py`
- `tests/unit/test_features.py`
- `tests/unit/test_dataset.py`

**Key reuse**:
- `src/quantdash/features/indicators.py` → INDICATORS_CONFIG and all indicator functions
- `src/quantdash/features/patterns.py` → detect_* functions
- `src/quantdash/data/providers/yfinance.py` → YFinanceProvider for data download

**Verification**: Build gold dataset end-to-end, inspect shapes and label distribution.

---

### Phase 4: News/NLP

**Files to create**:
- `src/quantdash/ml/data/news_embeddings.py` — FinBERT pipeline
- `scripts/precompute_finbert.py` — offline embedding script

**Verification**: Pre-compute embeddings for gold news articles, verify HDF5 output shapes.

---

### Phase 5: Model

**Files to create**:
- `src/quantdash/ml/models/branches.py` — all 6 branch modules
- `src/quantdash/ml/models/fusion.py` — FusionTransformer
- `src/quantdash/ml/models/signal_net.py` — TemporalFusionSignalNet
- `tests/unit/test_model.py` — shape tests, forward pass, FP16 verification

**Verification**:
- Forward pass with random data succeeds
- All output shapes correct: logits [B, 3], confidence [B]
- FP16 mode works without NaN
- Total params matches estimate (~1.15M)
- VRAM usage < 2GB with batch=128

---

### Phase 6: Supervised Training

**Files to create**:
- `src/quantdash/ml/training/losses.py` — focal_loss, confidence_loss
- `src/quantdash/ml/training/supervised.py` — SupervisedTrainer
- `src/quantdash/ml/training/callbacks.py` — EarlyStopping, CheckpointSaver
- `src/quantdash/ml/training/train_runner.py` — CLI entrypoint
- `scripts/train_model.py` — training script

**Verification**:
- Smoke test: overfit on 100 bars (loss → 0, accuracy → 100%)
- Train gold model on full data with walk-forward
- Validation F1 > random baseline (33%)
- TensorBoard shows loss curves converging

---

### Phase 7: RL Fine-tuning

**Files to create**:
- `src/quantdash/ml/training/rl_finetune.py` — TradingEnv + PPO wrapper

**Verification**:
- Environment passes `gymnasium.utils.env_checker.check_env()`
- PPO trains without crashes for 10K timesteps
- Compare validation Sharpe: supervised-only vs. supervised+RL

---

### Phase 8: Risk & Inference

**Files to create**:
- `src/quantdash/ml/risk/position_sizing.py` — confidence_to_size()
- `src/quantdash/ml/risk/circuit_breakers.py` — CircuitBreaker
- `src/quantdash/ml/inference/predictor.py` — SignalPredictor
- `src/quantdash/ml/inference/batch_runner.py` — cron-compatible script
- `src/quantdash/ml/strategy/ml_signal_strategy.py` — Strategy subclass
- `scripts/run_inference.py` — inference script

**Verification**:
- End-to-end: raw data → features → model → signal → backtest via VectorBT
- `MLSignalStrategy` appears in `list_strategies()` output
- Backtest produces valid BacktestResult with Sharpe, CAGR, win rate

---

### Phase 9: Multi-Asset

**Deliverables**:
- Build datasets for BTC-USD, CL=F, SPY, QQQ, AAPL
- Train per-asset models
- Colab notebook for GPU training on T4/A100
- Set up hourly cron jobs per asset
- Portfolio-level risk: correlation-aware total exposure limits

---

## 10. Existing Code to Reuse

| File | What to Reuse |
|------|---------------|
| `src/quantdash/features/indicators.py` | All 25+ indicator functions + INDICATORS_CONFIG dict |
| `src/quantdash/features/patterns.py` | `find_local_extrema()`, PatternConfig, detection convention |
| `src/quantdash/core/models.py` | PatternEvent, PatternDirection, Instrument, Timeframe |
| `src/quantdash/strategies/base.py` | Strategy ABC, generate_signals() interface, apply_tp_sl() |
| `src/quantdash/strategies/registry.py` | STRATEGY_REGISTRY, register_strategy() |
| `src/quantdash/data/providers/yfinance.py` | YFinanceProvider.download_bars() |
| `src/quantdash/data/providers/base.py` | DataProvider interface |
| `src/quantdash/data/cache.py` | OHLCVCache for caching downloaded data |
| `src/quantdash/config/settings.py` | Settings pattern (Pydantic BaseSettings) |
| `src/quantdash/config/symbols.py` | SYMBOLS dict, AssetType enum |
| `src/quantdash/alerts/telegram.py` | TelegramBot for sending inference alerts |
| `src/quantdash/engine/backtest/vectorbt_engine.py` | VectorbtEngine for backtesting MLSignalStrategy |
| `src/quantdash/engine/backtest/metrics.py` | extract_metrics() for BacktestResult |

---

## 11. Dependencies

```toml
# Add to pyproject.toml under [project.optional-dependencies]
ml = [
    "torch>=2.1.0",                    # Core deep learning framework
    "transformers>=4.35.0",            # FinBERT model (offline embedding only)
    "stable-baselines3>=2.2.0",        # PPO reinforcement learning
    "gymnasium>=0.29.0",               # RL environment interface
    "fredapi>=0.5.0",                  # Federal Reserve macro data
    "h5py>=3.10.0",                    # HDF5 dataset storage
    "tensorboard>=2.15.0",             # Training visualization
    "scikit-learn>=1.3.0",             # Metrics, preprocessing, confusion matrix
]
```

Update `all` group:
```toml
all = [
    "quantdash[api,ui,worker,backtest,optimize,sentiment,dev,ml]",
]
```

---

## 12. Key Design Rationale

1. **Separate models per asset** — Assets have fundamentally different dynamics (gold vs BTC vs equities). Shared architecture with independent weights.

2. **Pre-computed FinBERT** — Too large (110M params) for joint training on 8GB VRAM. 768-dim embeddings stored as features = same information, no GPU cost during training.

3. **1D Conv → Transformer** — Conv extracts local patterns, Transformer captures long-range dependencies. Better than pure LSTM (parallelizable, more stable gradients).

4. **Focal Loss** — Triple-barrier labels are imbalanced (hold >> buy/sell). Focal loss auto-focuses on hard (signal) examples.

5. **Walk-forward validation** — Financial time series are non-stationary. Standard k-fold would leak future information.

6. **Rolling z-score normalization** — No global stats → no look-ahead bias. Each bar normalized using only past 500 bars.

7. **Strategy registry integration** — ML model slots into existing infrastructure (backtest, API, screener) with zero changes to the rest of the codebase.

8. **Circuit breakers** — Model confidence can be miscalibrated. Hard risk limits prevent catastrophic losses regardless of model output.

9. **Two-phase training** — Supervised pre-training gives a strong foundation. RL fine-tuning can improve risk-adjusted returns by learning position timing that classification alone cannot capture.

10. **HDF5 storage** — Efficient for large numerical arrays. Supports partial loading and compression. Better than raw numpy files for multi-modal datasets.
