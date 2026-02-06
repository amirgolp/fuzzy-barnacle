import {
  Delete,
  ExpandMore,
  Pause,
  PlayArrow,
  Refresh,
  Remove,
  Source,
  Star,
  StarBorder,
  TrendingDown,
  TrendingUp,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material'
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Card,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  LinearProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import React, { useCallback, useEffect, useState } from 'react'

import { api } from '../../api/client'
import { screenerApi } from '../../api/screener'
import { technicalsApi } from '../../api/technicals'
import { useFavoritesStore } from '../../stores/favoritesStore'
import type { ScreenerResult, TechnicalSignal } from '../../types/screener'
import type { IndicatorReading, TechnicalsResult } from '../../types/technicals'
import { NewsSentimentPanel } from './NewsSentimentPanel'

// ── Callback types for applying strategies/indicators to chart ─────────
export interface ApplyOverlay {
  name: string
  params: Record<string, unknown>
  color: string
  type: 'indicator' | 'strategy'
}

// Strategy colors for chart overlays
const STRATEGY_COLORS: Record<string, string> = {
  sma_crossover: '#2962FF',
  chandelier_exit: '#FF6D00',
  breakout_trading: '#00BFA5',
  mean_reversion: '#AA00FF',
  bollinger_reversion: '#9C27B0',
  rsi_extremes: '#E91E63',
  squeeze_momentum: '#00E676',
  atr_breakout: '#FF3D00',
  macd_divergence: '#2196F3',
  adx_trend: '#FFC400',
  fibonacci_pullback: '#FFD700',
  gap_trading: '#26C6DA',
  ml_momentum: '#7C4DFF',
  ml_momentum_v2: '#651FFF',
  multi_agent_broker: '#F50057',
  multi_agent_broker_v2: '#D500F9',
  vwap_reversion: '#E91E63',
  ichimoku_cloud: '#26C6DA',
  pairs_trading: '#69F0AE',
  heikin_ashi_momentum: '#FFAB40',
  order_flow_imbalance: '#40C4FF',
  regime_detection: '#B388FF',
}

// Map indicator display names from technicals API to indicator registry names
const INDICATOR_NAME_MAP: Record<
  string,
  { name: string; color: string; params?: Record<string, any> }
> = {
  RSI: { name: 'RSI', color: '#7B1FA2', params: { period: 14 } },
  'Stochastic %K': {
    name: 'Stochastic Oscillator',
    color: '#009688',
    params: { k_period: 14 },
  },
  MACD: { name: 'MACD', color: '#2196F3' },
  ADX: { name: 'ADX', color: '#FFC400' },
  MFI: { name: 'MFI', color: '#795548', params: { period: 14 } },
  ATR: { name: 'ATR', color: '#F44336', params: { period: 14 } },
  'Bollinger Bands': { name: 'Bollinger Bands', color: '#9C27B0' },
  VWAP: { name: 'VWAP', color: '#E91E63' },
  OBV: { name: 'OBV', color: '#4CAF50' },
  CMF: { name: 'CMF', color: '#00796B', params: { period: 20 } },
  'Ichimoku Cloud': { name: 'Ichimoku Cloud', color: '#26C6DA' },
  'Parabolic SAR': { name: 'Parabolic SAR', color: '#AB47BC' },
}

// ── Asset Class Detection & Recommendations ─────────────────────────────
// Determines which indicators/strategies are most relevant for each asset class
const ASSET_CLASS_PATTERNS: { class: string; patterns: RegExp[] }[] = [
  {
    class: 'crypto',
    patterns: [
      /-USD$/,
      /^BTC/,
      /^ETH/,
      /^SOL/,
      /^AVAX/,
      /^ADA/,
      /^DOT/,
      /^XRP/,
    ],
  },
  {
    class: 'precious_metals',
    patterns: [
      /^GLD$/,
      /^SLV$/,
      /^IAU$/,
      /^PPLT$/,
      /^PALL$/,
      /^GDX/,
      /^SIL$/,
      /^GOLD$/,
      /^SILVER$/,
    ],
  },
  {
    class: 'energy',
    patterns: [
      /^USO$/,
      /^UNG$/,
      /^XLE$/,
      /^OIH$/,
      /^XOP$/,
      /^UCO$/,
      /^BOIL$/,
      /^CL/,
      /^NG/,
    ],
  },
  {
    class: 'commodities',
    patterns: [
      /^DBA$/,
      /^DBC$/,
      /^GSG$/,
      /^PDBC$/,
      /^COPX$/,
      /^CPER$/,
      /^WEAT$/,
      /^CORN$/,
    ],
  },
  {
    class: 'index_etf',
    patterns: [
      /^SPY$/,
      /^QQQ$/,
      /^DIA$/,
      /^IWM$/,
      /^VTI$/,
      /^TLT$/,
      /^HYG$/,
      /^LQD$/,
    ],
  },
  {
    class: 'tech',
    patterns: [
      /^AAPL$/,
      /^MSFT$/,
      /^GOOGL$/,
      /^AMZN$/,
      /^TSLA$/,
      /^NVDA$/,
      /^META$/,
      /^NFLX$/,
      /^AMD$/,
      /^INTC$/,
    ],
  },
  {
    class: 'finance',
    patterns: [/^JPM$/, /^BAC$/, /^WFC$/, /^C$/, /^GS$/, /^MS$/, /^V$/, /^MA$/],
  },
]

const getAssetClass = (symbol: string): string => {
  for (const { class: cls, patterns } of ASSET_CLASS_PATTERNS) {
    if (patterns.some((p) => p.test(symbol))) return cls
  }
  return 'equity' // default for individual stocks
}

interface AssetRecommendation {
  indicators: { name: string; reason: string; weight: number }[]
  strategies: { id: string; reason: string; weight: number }[]
  notes: string[]
}

const ASSET_CLASS_RECOMMENDATIONS: Record<string, AssetRecommendation> = {
  crypto: {
    indicators: [
      {
        name: 'RSI',
        reason: 'Crypto momentum swings are extreme - RSI helps spot reversals',
        weight: 9,
      },
      {
        name: 'Bollinger Bands',
        reason: 'High volatility makes BB squeeze/expansion signals reliable',
        weight: 8,
      },
      {
        name: 'OBV',
        reason: 'Exchange volume confirms trend direction',
        weight: 7,
      },
      {
        name: 'ATR',
        reason: 'Position sizing in high-vol environment',
        weight: 8,
      },
    ],
    strategies: [
      {
        id: 'squeeze_momentum',
        reason: 'Crypto consolidations lead to explosive moves',
        weight: 9,
      },
      {
        id: 'bollinger_reversion',
        reason: 'Mean reversion works well in ranging crypto markets',
        weight: 8,
      },
      {
        id: 'atr_breakout',
        reason: 'Volatility breakouts capture trending moves',
        weight: 8,
      },
      {
        id: 'regime_detection',
        reason: 'Crypto cycles between trending and ranging',
        weight: 7,
      },
    ],
    notes: [
      '24/7 market - signals can trigger anytime',
      'High correlation to BTC during risk-off',
      'Consider on-chain metrics',
    ],
  },
  precious_metals: {
    indicators: [
      {
        name: 'ADX',
        reason: 'Gold trends strongly - ADX confirms trend strength',
        weight: 9,
      },
      {
        name: 'ATR',
        reason: 'Volatility clusters in metals during macro events',
        weight: 8,
      },
      {
        name: 'MACD',
        reason: 'Momentum divergences are reliable in gold',
        weight: 7,
      },
      {
        name: 'Ichimoku Cloud',
        reason: 'Long-term support/resistance levels well-defined',
        weight: 8,
      },
    ],
    strategies: [
      {
        id: 'chandelier_exit',
        reason: 'Trailing stops work well in gold trends',
        weight: 9,
      },
      {
        id: 'adx_trend',
        reason: 'Gold trends last - ADX identifies entry points',
        weight: 8,
      },
      {
        id: 'ichimoku_cloud',
        reason: 'Multiple TF confluence in precious metals',
        weight: 8,
      },
      {
        id: 'fibonacci_pullback',
        reason: 'Gold respects fib levels precisely',
        weight: 7,
      },
    ],
    notes: [
      'Safe haven - inversely correlated to USD/rates',
      'Watch real yields (TIPS)',
      'Strong seasonal patterns',
    ],
  },
  energy: {
    indicators: [
      {
        name: 'ATR',
        reason: 'Energy is volatile - ATR sizes positions correctly',
        weight: 9,
      },
      {
        name: 'RSI',
        reason: 'Mean reversion in oil works at extremes',
        weight: 7,
      },
      {
        name: 'VWAP',
        reason: 'Institutional flow visible via VWAP',
        weight: 8,
      },
      {
        name: 'OBV',
        reason: 'Volume confirms supply/demand imbalance',
        weight: 8,
      },
    ],
    strategies: [
      {
        id: 'mean_reversion',
        reason: 'Oil mean-reverts around production costs',
        weight: 8,
      },
      {
        id: 'atr_breakout',
        reason: 'Geopolitical events cause breakouts',
        weight: 9,
      },
      {
        id: 'order_flow_imbalance',
        reason: 'Physical/commercial flow impacts price',
        weight: 8,
      },
      {
        id: 'regime_detection',
        reason: 'Contango/backwardation regimes matter',
        weight: 7,
      },
    ],
    notes: [
      'Contango/backwardation affects ETF returns',
      'OPEC decisions cause gaps',
      'Seasonal demand patterns (summer/winter)',
    ],
  },
  commodities: {
    indicators: [
      {
        name: 'MACD',
        reason: 'Commodity super-cycles show clear momentum',
        weight: 8,
      },
      {
        name: 'ADX',
        reason: 'Agricultural trends last through seasons',
        weight: 8,
      },
      {
        name: 'Bollinger Bands',
        reason: 'Weather events cause extreme moves',
        weight: 7,
      },
      { name: 'ATR', reason: 'Commodities can be very volatile', weight: 9 },
    ],
    strategies: [
      {
        id: 'breakout_trading',
        reason: 'Supply shocks cause breakouts',
        weight: 8,
      },
      { id: 'adx_trend', reason: 'Commodity trends persist', weight: 8 },
      {
        id: 'mean_reversion',
        reason: 'Production costs anchor long-term mean',
        weight: 7,
      },
      {
        id: 'regime_detection',
        reason: 'Inflation/deflation regimes matter',
        weight: 7,
      },
    ],
    notes: [
      'Weather-dependent assets',
      'Seasonal planting/harvest cycles',
      'USD strength impacts dollar-denominated commodities',
    ],
  },
  index_etf: {
    indicators: [
      {
        name: 'RSI',
        reason: 'Index oversold/overbought signals reliable',
        weight: 8,
      },
      {
        name: 'VWAP',
        reason: 'Institutional rebalancing visible at VWAP',
        weight: 9,
      },
      { name: 'ADX', reason: 'Index trends are persistent', weight: 7 },
      {
        name: 'Bollinger Bands',
        reason: 'Volatility regimes clear in indices',
        weight: 7,
      },
    ],
    strategies: [
      {
        id: 'vwap_reversion',
        reason: 'Indices revert to VWAP intraday',
        weight: 9,
      },
      {
        id: 'regime_detection',
        reason: 'Risk-on/risk-off regimes dominate',
        weight: 8,
      },
      {
        id: 'sma_crossover',
        reason: 'Classic MA signals work in indices',
        weight: 7,
      },
      {
        id: 'bollinger_reversion',
        reason: 'SPY mean-reverts at 2σ bands',
        weight: 7,
      },
    ],
    notes: [
      'Highly liquid - tight spreads',
      'Options flow impacts price',
      'Fed/macro events dominate',
    ],
  },
  tech: {
    indicators: [
      {
        name: 'RSI',
        reason: 'Tech stocks swing between fear/greed',
        weight: 8,
      },
      {
        name: 'MACD',
        reason: 'Momentum shows earnings-driven trends',
        weight: 8,
      },
      {
        name: 'OBV',
        reason: 'Institutional accumulation/distribution visible',
        weight: 8,
      },
      {
        name: 'Bollinger Bands',
        reason: 'Earnings volatility makes BB useful',
        weight: 7,
      },
    ],
    strategies: [
      {
        id: 'squeeze_momentum',
        reason: 'Pre-earnings squeezes are tradeable',
        weight: 8,
      },
      {
        id: 'breakout_trading',
        reason: 'New highs in tech lead to momentum',
        weight: 8,
      },
      {
        id: 'macd_divergence',
        reason: 'Divergences at tops/bottoms reliable',
        weight: 7,
      },
      { id: 'chandelier_exit', reason: 'Trail strong tech trends', weight: 7 },
    ],
    notes: [
      'Earnings drive price action',
      'High beta to QQQ/rates',
      'Growth vs value rotation matters',
    ],
  },
  finance: {
    indicators: [
      { name: 'ADX', reason: 'Banks trend with yield curve', weight: 8 },
      {
        name: 'RSI',
        reason: 'Financials mean-revert in stable rates',
        weight: 7,
      },
      { name: 'MACD', reason: 'Sector rotation visible in MACD', weight: 7 },
      {
        name: 'ATR',
        reason: 'Volatility spikes during credit events',
        weight: 8,
      },
    ],
    strategies: [
      {
        id: 'adx_trend',
        reason: 'Rate trends drive bank performance',
        weight: 8,
      },
      {
        id: 'mean_reversion',
        reason: 'Book value anchors financials',
        weight: 7,
      },
      {
        id: 'regime_detection',
        reason: 'Credit cycle regimes matter',
        weight: 8,
      },
      {
        id: 'pairs_trading',
        reason: 'Bank pairs have stable relationships',
        weight: 7,
      },
    ],
    notes: [
      'Yield curve slope drives earnings',
      'Credit spreads signal risk',
      'Fed policy impacts heavily',
    ],
  },
  equity: {
    indicators: [
      { name: 'RSI', reason: 'Standard momentum indicator', weight: 7 },
      { name: 'MACD', reason: 'Trend/momentum confirmation', weight: 7 },
      {
        name: 'Bollinger Bands',
        reason: 'Volatility-based entries',
        weight: 7,
      },
      { name: 'OBV', reason: 'Volume confirms moves', weight: 7 },
    ],
    strategies: [
      { id: 'breakout_trading', reason: 'Works for most stocks', weight: 7 },
      { id: 'mean_reversion', reason: 'Oversold bounces common', weight: 7 },
      { id: 'sma_crossover', reason: 'Classic trend signals', weight: 7 },
      {
        id: 'macd_divergence',
        reason: 'Divergences at turning points',
        weight: 7,
      },
    ],
    notes: [
      'Check sector correlation',
      'Earnings dates important',
      'Consider market cap for liquidity',
    ],
  },
}

// ── Strategy-Signal Relevance Map ──────────────────────────────────────
// Maps detected signal types to the strategies most relevant for that setup.
const SIGNAL_STRATEGY_MAP: Record<
  string,
  { id: string; label: string; why: string }[]
> = {
  GOLDEN_CROSS: [
    {
      id: 'sma_crossover',
      label: 'SMA Crossover',
      why: 'Directly trades MA crossovers',
    },
    {
      id: 'adx_trend',
      label: 'ADX Trend',
      why: 'Confirms trend strength after cross',
    },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trails the new uptrend',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'TK cross + cloud confirms trend',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed trend confirmation',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Auto-selects trend-following mode',
    },
  ],
  DEATH_CROSS: [
    {
      id: 'sma_crossover',
      label: 'SMA Crossover',
      why: 'Directly trades MA crossovers',
    },
    { id: 'adx_trend', label: 'ADX Trend', why: 'Confirms downtrend strength' },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trails the new downtrend',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'TK cross + below cloud confirms',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed bear trend entry',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Auto-selects trend-following mode',
    },
  ],
  RSI_OVERSOLD: [
    {
      id: 'rsi_extremes',
      label: 'RSI Extremes',
      why: 'Directly trades RSI extremes',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Oversold = reversion setup',
    },
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'Combines BB + RSI oversold',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Oversold below VWAP = snap back',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Spread at extreme = reversion',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Activates mean-reversion mode',
    },
  ],
  RSI_OVERBOUGHT: [
    {
      id: 'rsi_extremes',
      label: 'RSI Extremes',
      why: 'Directly trades RSI extremes',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Overbought = reversion setup',
    },
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'Combines BB + RSI overbought',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Overbought above VWAP = fade',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Spread at extreme = reversion',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Activates mean-reversion mode',
    },
  ],
  MACD_BULLISH: [
    {
      id: 'macd_divergence',
      label: 'MACD Divergence',
      why: 'Momentum confirmation',
    },
    {
      id: 'squeeze_momentum',
      label: 'Squeeze Momentum',
      why: 'Momentum aligns with squeeze release',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed momentum confirms',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'Momentum + trend alignment',
    },
  ],
  MACD_BEARISH: [
    {
      id: 'macd_divergence',
      label: 'MACD Divergence',
      why: 'Bearish momentum signal',
    },
    {
      id: 'squeeze_momentum',
      label: 'Squeeze Momentum',
      why: 'Momentum aligns with squeeze release',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed bear momentum',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'Momentum + trend alignment',
    },
  ],
  BREAKOUT: [
    {
      id: 'breakout_trading',
      label: 'Breakout Trading',
      why: 'Directly trades breakouts',
    },
    {
      id: 'atr_breakout',
      label: 'ATR Breakout',
      why: 'Volatility expansion confirmation',
    },
    {
      id: 'squeeze_momentum',
      label: 'Squeeze Momentum',
      why: 'Post-squeeze breakout',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Detects regime shift to trending',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'Volume confirms institutional buying',
    },
  ],
  BREAKDOWN: [
    {
      id: 'breakout_trading',
      label: 'Breakout Trading',
      why: 'Directly trades breakdowns',
    },
    {
      id: 'atr_breakout',
      label: 'ATR Breakout',
      why: 'Volatility expansion confirmation',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Detects regime shift to bear trend',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'Volume confirms distribution',
    },
  ],
  BOLLINGER_OVERSOLD: [
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'At lower band = entry zone',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'BB oversold = mean reversion',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'BB + VWAP deviation = strong setup',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Spread stretched = reversion',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Activates mean-reversion mode',
    },
  ],
  BOLLINGER_OVERBOUGHT: [
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'At upper band = short zone',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'BB overbought = mean reversion',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'BB + VWAP deviation = fade setup',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Spread stretched = reversion',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Activates mean-reversion mode',
    },
  ],
  STOCH_OVERSOLD: [
    {
      id: 'rsi_extremes',
      label: 'RSI Extremes',
      why: 'Stochastic confirms RSI oversold',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Oversold = reversion candidate',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Multiple oversold = VWAP snap',
    },
  ],
  STOCH_OVERBOUGHT: [
    {
      id: 'rsi_extremes',
      label: 'RSI Extremes',
      why: 'Stochastic confirms overbought',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Overbought = reversion candidate',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Multiple overbought = VWAP fade',
    },
  ],
  VOLUME_SPIKE: [
    {
      id: 'breakout_trading',
      label: 'Breakout Trading',
      why: 'Volume confirms breakout',
    },
    {
      id: 'gap_trading',
      label: 'Gap Trading',
      why: 'Volume spike on gap open',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'Spike = institutional activity',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'High-volume deviation from VWAP',
    },
  ],
  EMA_CROSSOVER_BULL: [
    {
      id: 'sma_crossover',
      label: 'SMA Crossover',
      why: 'EMA cross confirms trend',
    },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trail the bullish trend',
    },
    {
      id: 'fibonacci_pullback',
      label: 'Fibonacci Pullback',
      why: 'Wait for pullback to enter',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'EMA + TK cross alignment',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed EMA trend',
    },
  ],
  EMA_CROSSOVER_BEAR: [
    {
      id: 'sma_crossover',
      label: 'SMA Crossover',
      why: 'EMA cross confirms downtrend',
    },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trail the bearish trend',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'EMA + TK cross alignment',
    },
    {
      id: 'heikin_ashi_momentum',
      label: 'Heikin-Ashi Momentum',
      why: 'Smoothed bear EMA trend',
    },
  ],
  PRICE_ABOVE_EMA: [
    { id: 'adx_trend', label: 'ADX Trend', why: 'Uptrend confirmed by EMA' },
    {
      id: 'fibonacci_pullback',
      label: 'Fibonacci Pullback',
      why: 'Buy pullbacks in uptrend',
    },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trail the uptrend',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'Price above cloud + EMA',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Bull regime confirmed',
    },
  ],
  PRICE_BELOW_EMA: [
    { id: 'adx_trend', label: 'ADX Trend', why: 'Downtrend confirmed by EMA' },
    {
      id: 'chandelier_exit',
      label: 'Chandelier Exit',
      why: 'Trail the downtrend',
    },
    {
      id: 'ichimoku_cloud',
      label: 'Ichimoku Cloud',
      why: 'Price below cloud + EMA',
    },
    {
      id: 'regime_detection',
      label: 'Regime Detection',
      why: 'Bear regime confirmed',
    },
  ],
  MFI_OVERSOLD: [
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Money flow extreme = reversal',
    },
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'Volume-weighted oversold',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'MFI confirms accumulation',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Flow extreme + VWAP deviation',
    },
  ],
  MFI_OVERBOUGHT: [
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Money flow extreme = reversal',
    },
    {
      id: 'bollinger_reversion',
      label: 'Bollinger Reversion',
      why: 'Volume-weighted overbought',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'MFI confirms distribution',
    },
    {
      id: 'vwap_reversion',
      label: 'VWAP Reversion',
      why: 'Flow extreme + VWAP deviation',
    },
  ],
  OBV_DIVERGENCE_BULL: [
    {
      id: 'macd_divergence',
      label: 'MACD Divergence',
      why: 'Divergence confluence',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Accumulation signal',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'OBV confirms institutional buying',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Divergence = spread stretched',
    },
  ],
  OBV_DIVERGENCE_BEAR: [
    {
      id: 'macd_divergence',
      label: 'MACD Divergence',
      why: 'Divergence confluence',
    },
    {
      id: 'mean_reversion',
      label: 'Mean Reversion',
      why: 'Distribution signal',
    },
    {
      id: 'order_flow_imbalance',
      label: 'Order Flow Imbalance',
      why: 'OBV confirms institutional selling',
    },
    {
      id: 'pairs_trading',
      label: 'Pairs Trading',
      why: 'Divergence = spread stretched',
    },
  ],
}

function getRecommendedStrategies(signals: TechnicalSignal[]) {
  const stratMap = new Map<
    string,
    { id: string; label: string; reasons: string[]; count: number }
  >()
  for (const sig of signals) {
    const matches = SIGNAL_STRATEGY_MAP[sig.signal_type] || []
    for (const m of matches) {
      const existing = stratMap.get(m.id)
      if (existing) {
        existing.reasons.push(m.why)
        existing.count++
      } else {
        stratMap.set(m.id, {
          id: m.id,
          label: m.label,
          reasons: [m.why],
          count: 1,
        })
      }
    }
  }
  return Array.from(stratMap.values()).sort((a, b) => b.count - a.count)
}

// ── Merged data per symbol ─────────────────────────────────────────────
interface SymbolIntel {
  symbol: string
  technicals?: TechnicalsResult
  screener?: ScreenerResult
}

// ── Colors ─────────────────────────────────────────────────────────────
const recColor = (rec: string) => {
  const r = rec.toLowerCase()
  if (r.includes('strong_buy') || r.includes('strong buy')) return '#26a69a'
  if (r.includes('buy')) return '#66bb6a'
  if (r.includes('strong_sell') || r.includes('strong sell')) return '#f44336'
  if (r.includes('sell')) return '#ff5252'
  return '#9e9e9e'
}

const recChipColor = (rec: string): 'success' | 'error' | 'default' => {
  const r = rec.toLowerCase()
  if (r.includes('buy')) return 'success'
  if (r.includes('sell')) return 'error'
  return 'default'
}

const actionColor = (action: string) => {
  if (action === 'Buy') return '#66bb6a'
  if (action === 'Sell') return '#ff5252'
  return '#9e9e9e'
}

// ── Default symbols (includes tech, finance, commodities, crypto) ──────
const DEFAULT_SYMBOLS = [
  'AAPL',
  'MSFT',
  'GOOGL',
  'NVDA',
  'TSLA', // Tech
  'SPY',
  'QQQ', // Indices
  'GLD',
  'SLV',
  'USO', // Commodities
  'BTC-USD',
  'ETH-USD', // Crypto
]
const POLL_INTERVAL = 60_000

// ── Indicator detail rows ──────────────────────────────────────────────
function IndicatorTable({
  title,
  indicators,
  activeOverlays,
  onToggleIndicator,
}: {
  title: string
  indicators: IndicatorReading[]
  activeOverlays: Set<string>
  onToggleIndicator?: (indicatorDisplayName: string) => void
}) {
  return (
    <Box>
      <Typography
        variant="caption"
        sx={{
          fontWeight: 700,
          color: 'text.secondary',
          textTransform: 'uppercase',
          fontSize: '0.6rem',
          letterSpacing: 1,
        }}
      >
        {title}
        {onToggleIndicator && (
          <Typography
            component="span"
            variant="caption"
            sx={{ fontSize: '0.5rem', color: 'info.main', ml: 0.5 }}
          >
            (click to chart)
          </Typography>
        )}
      </Typography>
      <Table size="small" sx={{ mt: 0.5 }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontSize: '0.65rem', py: 0.2, fontWeight: 600 }}>
              Indicator
            </TableCell>
            <TableCell
              align="right"
              sx={{ fontSize: '0.65rem', py: 0.2, fontWeight: 600 }}
            >
              Value
            </TableCell>
            <TableCell
              align="center"
              sx={{ fontSize: '0.65rem', py: 0.2, fontWeight: 600 }}
            >
              Action
            </TableCell>
            {onToggleIndicator && (
              <TableCell
                align="center"
                sx={{
                  fontSize: '0.65rem',
                  py: 0.2,
                  fontWeight: 600,
                  width: 28,
                }}
              ></TableCell>
            )}
          </TableRow>
        </TableHead>
        <TableBody>
          {indicators.map((ind) => {
            const mapped = INDICATOR_NAME_MAP[ind.name]
            const canChart = !!mapped && !!onToggleIndicator
            const isActive = mapped ? activeOverlays.has(mapped.name) : false
            return (
              <TableRow
                key={ind.name}
                hover={canChart}
                sx={{ cursor: canChart ? 'pointer' : 'default' }}
                onClick={() => canChart && onToggleIndicator?.(ind.name)}
              >
                <TableCell sx={{ fontSize: '0.65rem', py: 0.2, border: 0 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {isActive && (
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          bgcolor: mapped?.color || '#2962FF',
                          flexShrink: 0,
                        }}
                      />
                    )}
                    {ind.name}
                  </Box>
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontSize: '0.65rem',
                    py: 0.2,
                    border: 0,
                    fontFamily: 'monospace',
                  }}
                >
                  {typeof ind.value === 'number'
                    ? ind.value.toFixed(2)
                    : ind.value}
                </TableCell>
                <TableCell
                  align="center"
                  sx={{
                    fontSize: '0.65rem',
                    py: 0.2,
                    border: 0,
                    color: actionColor(ind.action),
                    fontWeight: 600,
                  }}
                >
                  {ind.action}
                </TableCell>
                {onToggleIndicator && (
                  <TableCell
                    align="center"
                    sx={{ py: 0.2, border: 0, width: 28 }}
                  >
                    {canChart &&
                      (isActive ? (
                        <Visibility
                          sx={{
                            fontSize: 12,
                            color: mapped?.color || 'primary.main',
                          }}
                        />
                      ) : (
                        <VisibilityOff
                          sx={{ fontSize: 12, color: 'text.disabled' }}
                        />
                      ))}
                  </TableCell>
                )}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </Box>
  )
}

function SignalsList({ signals }: { signals: TechnicalSignal[] }) {
  if (!signals.length)
    return (
      <Typography variant="caption" color="text.secondary">
        No signals detected
      </Typography>
    )
  return (
    <Stack direction="row" flexWrap="wrap" gap={0.5}>
      {signals.map((s, i) => (
        <Chip
          key={i}
          label={s.description.split(':')[0]}
          size="small"
          color={s.is_bullish ? 'success' : 'error'}
          variant={s.strength === 'strong' ? 'filled' : 'outlined'}
          sx={{ fontSize: '0.6rem', height: 20 }}
        />
      ))}
    </Stack>
  )
}

function StrategyRecommendations({
  signals,
  activeOverlays,
  onToggleStrategy,
}: {
  signals: TechnicalSignal[]
  activeOverlays: Set<string>
  onToggleStrategy?: (id: string, label: string) => void
}) {
  const strats = getRecommendedStrategies(signals)
  if (!strats.length) return null
  return (
    <Box>
      <Typography
        variant="caption"
        sx={{
          fontWeight: 700,
          color: 'text.secondary',
          textTransform: 'uppercase',
          fontSize: '0.6rem',
          letterSpacing: 1,
        }}
      >
        Recommended Strategies
        {onToggleStrategy && (
          <Typography
            component="span"
            variant="caption"
            sx={{ fontSize: '0.5rem', color: 'info.main', ml: 0.5 }}
          >
            (click to overlay on chart)
          </Typography>
        )}
      </Typography>
      <Stack spacing={0.5} sx={{ mt: 0.5 }}>
        {strats.slice(0, 5).map((s) => {
          const isActive = activeOverlays.has(s.id)
          return (
            <Box
              key={s.id}
              sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
            >
              <Chip
                label={s.label}
                size="small"
                color={isActive ? 'primary' : 'default'}
                variant={isActive ? 'filled' : 'outlined'}
                onClick={() => onToggleStrategy?.(s.id, s.label)}
                icon={
                  onToggleStrategy ? (
                    isActive ? (
                      <Visibility sx={{ fontSize: 12 }} />
                    ) : (
                      <VisibilityOff sx={{ fontSize: 12 }} />
                    )
                  ) : undefined
                }
                sx={{
                  fontSize: '0.6rem',
                  height: 22,
                  minWidth: 120,
                  cursor: onToggleStrategy ? 'pointer' : 'default',
                  borderColor: isActive
                    ? STRATEGY_COLORS[s.id] || '#2962FF'
                    : undefined,
                  bgcolor: isActive
                    ? `${STRATEGY_COLORS[s.id] || '#2962FF'}22`
                    : undefined,
                }}
              />
              <Chip
                label={`${s.count} signal${s.count > 1 ? 's' : ''}`}
                size="small"
                sx={{
                  fontSize: '0.55rem',
                  height: 18,
                  bgcolor: 'action.hover',
                }}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.6rem' }}
              >
                {s.reasons[0]}
              </Typography>
            </Box>
          )
        })}
      </Stack>
    </Box>
  )
}

// ── Asset Class Recommendations Panel ───────────────────────────────────
function AssetClassRecommendations({
  symbol,
  activeOverlays,
  onToggleStrategy,
  onToggleIndicator,
}: {
  symbol: string
  activeOverlays: Set<string>
  onToggleStrategy?: (id: string, label: string) => void
  onToggleIndicator?: (displayName: string) => void
}) {
  const assetClass = getAssetClass(symbol)
  const rec = ASSET_CLASS_RECOMMENDATIONS[assetClass]
  if (!rec) return null

  const classLabel = assetClass
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())

  return (
    <Box>
      <Typography
        variant="caption"
        sx={{
          fontWeight: 700,
          color: 'text.secondary',
          textTransform: 'uppercase',
          fontSize: '0.6rem',
          letterSpacing: 1,
        }}
      >
        {classLabel} Recommendations
      </Typography>

      {/* Recommended Indicators */}
      <Box sx={{ mt: 0.5 }}>
        <Typography
          variant="caption"
          sx={{ fontSize: '0.65rem', color: 'info.main', fontWeight: 700 }}
        >
          Top Indicators
        </Typography>
        <Stack direction="row" flexWrap="wrap" gap={0.5} sx={{ mt: 0.25 }}>
          {rec.indicators.slice(0, 4).map((ind) => {
            const mapped = Object.entries(INDICATOR_NAME_MAP).find(
              ([k]) => k === ind.name,
            )?.[1]
            const isActive = mapped ? activeOverlays.has(mapped.name) : false
            return (
              <Tooltip key={ind.name} title={ind.reason}>
                <Chip
                  label={
                    <Box
                      sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                    >
                      {ind.name}
                      <Star sx={{ fontSize: 10, color: '#FFD700' }} />
                    </Box>
                  }
                  size="small"
                  variant={isActive ? 'filled' : 'outlined'}
                  color={isActive ? 'primary' : 'default'}
                  onClick={() => onToggleIndicator?.(ind.name)}
                  sx={{
                    fontSize: '0.65rem',
                    height: 20,
                    cursor: onToggleIndicator ? 'pointer' : 'default',
                    borderColor: isActive ? mapped?.color : undefined,
                    bgcolor: isActive
                      ? `${mapped?.color || '#2962FF'}22`
                      : undefined,
                  }}
                />
              </Tooltip>
            )
          })}
        </Stack>
      </Box>

      {/* Recommended Strategies */}
      <Box sx={{ mt: 0.5 }}>
        <Typography
          variant="caption"
          sx={{ fontSize: '0.55rem', color: 'info.main', fontWeight: 600 }}
        >
          Top Strategies
        </Typography>
        <Stack direction="row" flexWrap="wrap" gap={0.5} sx={{ mt: 0.25 }}>
          {rec.strategies.slice(0, 4).map((strat) => {
            const isActive = activeOverlays.has(strat.id)
            const stratLabel = strat.id
              .replace(/_/g, ' ')
              .replace(/\b\w/g, (c) => c.toUpperCase())
            return (
              <Tooltip key={strat.id} title={strat.reason}>
                <Chip
                  label={
                    <Box
                      sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                    >
                      {stratLabel}
                      <Star sx={{ fontSize: 10, color: '#FFD700' }} />
                    </Box>
                  }
                  size="small"
                  variant={isActive ? 'filled' : 'outlined'}
                  color={isActive ? 'primary' : 'default'}
                  onClick={() => onToggleStrategy?.(strat.id, stratLabel)}
                  sx={{
                    fontSize: '0.65rem',
                    height: 20,
                    cursor: onToggleStrategy ? 'pointer' : 'default',
                    borderColor: isActive
                      ? STRATEGY_COLORS[strat.id]
                      : undefined,
                    bgcolor: isActive
                      ? `${STRATEGY_COLORS[strat.id] || '#2962FF'}22`
                      : undefined,
                  }}
                />
              </Tooltip>
            )
          })}
        </Stack>
      </Box>
    </Box>
  )
}

// ── Signal Detail (TP/SL/freshness) when strategy is active ────────────
interface StrategyMarker {
  time: number
  text: 'Buy' | 'Sell'
  entry_price?: number
  sl?: number
  tp?: number
  bars_ago: number
}

const freshnessLabel = (barsAgo: number): { label: string; color: string } => {
  if (barsAgo <= 2) return { label: 'Fresh', color: '#26a69a' }
  if (barsAgo <= 5) return { label: 'Recent', color: '#FFC400' }
  return { label: 'Stale', color: '#ef5350' }
}

function SignalDetail({
  symbol,
  strategyId,
  currentPrice,
}: {
  symbol: string
  strategyId: string
  currentPrice?: number
}) {
  const [marker, setMarker] = useState<StrategyMarker | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    const fetchOverlay = async () => {
      setLoading(true)
      setError(false)
      try {
        const res = await api.post('/strategies/overlay', {
          symbol,
          strategy_id: strategyId,
          params: {},
          timeframe: '1d',
        })
        if (cancelled) return
        const markers: StrategyMarker[] = res.data.markers || []
        // Get the most recent buy or sell signal
        const latest = markers
          .filter((m) => m.text === 'Buy' || m.text === 'Sell')
          .pop()
        setMarker(latest || null)
      } catch (e) {
        if (!cancelled) setError(true)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchOverlay()
    return () => {
      cancelled = true
    }
  }, [symbol, strategyId])

  if (loading) return <CircularProgress size={12} />
  if (error || !marker) return null

  const { label: freshLabel, color: freshColor } = freshnessLabel(
    marker.bars_ago,
  )
  const isBuy = marker.text === 'Buy'
  const entry = marker.entry_price
  const sl = marker.sl
  const tp = marker.tp
  const price = currentPrice

  // Progress toward TP (or SL for loss)
  let progress = 0
  let progressColor = '#9e9e9e'
  if (entry && price && tp && sl) {
    const tpDist = Math.abs(tp - entry)
    const slDist = Math.abs(sl - entry)
    const priceDist = isBuy ? price - entry : entry - price
    if (priceDist >= 0 && tpDist > 0) {
      progress = Math.min(100, (priceDist / tpDist) * 100)
      progressColor = '#26a69a'
    } else if (priceDist < 0 && slDist > 0) {
      progress = Math.min(100, (Math.abs(priceDist) / slDist) * 100)
      progressColor = '#ef5350'
    }
  }

  return (
    <Box sx={{ mt: 0.5, p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
      <Box
        sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}
      >
        <Chip
          label={marker.text}
          size="small"
          sx={{
            fontSize: '0.6rem',
            height: 18,
            bgcolor: isBuy ? '#26a69a' : '#ef5350',
            color: '#fff',
            fontWeight: 700,
          }}
        />
        <Chip
          label={freshLabel}
          size="small"
          sx={{
            fontSize: '0.55rem',
            height: 16,
            bgcolor: `${freshColor}22`,
            color: freshColor,
            fontWeight: 600,
          }}
        />
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontSize: '0.55rem' }}
        >
          {marker.bars_ago} bar{marker.bars_ago !== 1 ? 's' : ''} ago
        </Typography>
      </Box>
      {entry && (
        <Box sx={{ display: 'flex', gap: 2, mt: 0.5, flexWrap: 'wrap' }}>
          <Box>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: '0.5rem' }}
            >
              Entry
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontFamily: 'monospace',
                fontWeight: 600,
                fontSize: '0.65rem',
                display: 'block',
              }}
            >
              ${entry.toFixed(2)}
            </Typography>
          </Box>
          {sl && (
            <Box>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.5rem' }}
              >
                SL
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'monospace',
                  fontWeight: 600,
                  fontSize: '0.65rem',
                  display: 'block',
                  color: '#ef5350',
                }}
              >
                ${sl.toFixed(2)}
                {price && (
                  <span style={{ fontSize: '0.5rem', color: '#9e9e9e' }}>
                    {' '}
                    ({(((sl - price) / price) * 100).toFixed(1)}%)
                  </span>
                )}
              </Typography>
            </Box>
          )}
          {tp && (
            <Box>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.5rem' }}
              >
                TP
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'monospace',
                  fontWeight: 600,
                  fontSize: '0.65rem',
                  display: 'block',
                  color: '#26a69a',
                }}
              >
                ${tp.toFixed(2)}
                {price && (
                  <span style={{ fontSize: '0.5rem', color: '#9e9e9e' }}>
                    {' '}
                    ({(((tp - price) / price) * 100).toFixed(1)}%)
                  </span>
                )}
              </Typography>
            </Box>
          )}
        </Box>
      )}
      {progress > 0 && (
        <Box sx={{ mt: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: '0.5rem' }}
            >
              Progress
            </Typography>
            <Box
              sx={{
                flex: 1,
                height: 4,
                bgcolor: 'action.disabledBackground',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <Box
                sx={{
                  width: `${progress}%`,
                  height: '100%',
                  bgcolor: progressColor,
                  borderRadius: 2,
                }}
              />
            </Box>
            <Typography
              variant="caption"
              sx={{ fontSize: '0.5rem', color: progressColor, fontWeight: 600 }}
            >
              {progress.toFixed(0)}%
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  )
}

// SentimentInline removed - replaced by NewsSentimentPanel

// ── Accordion Row ──────────────────────────────────────────────────────
function SymbolRow({
  data,
  onSymbolClick,
  onRemove,
  activeOverlays,
  onToggleStrategy,
  onToggleIndicator,
}: {
  data: SymbolIntel
  onSymbolClick?: (symbol: string) => void
  onRemove: (symbol: string) => void
  activeOverlays: Set<string>
  onToggleStrategy?: (id: string, label: string) => void
  onToggleIndicator?: (displayName: string) => void
}) {
  const { symbol, technicals, screener } = data
  const summaryRec = technicals?.summary.recommendation || 'N/A'
  const score = screener?.score ?? 0
  const price = screener?.current_price
  const signals = screener?.signals || []

  const { isFavorite, toggleFavorite } = useFavoritesStore()
  const isFav = isFavorite(symbol)

  return (
    <Accordion
      disableGutters
      elevation={0}
      sx={{
        '&:before': { display: 'none' },
        borderBottom: '1px solid',
        borderColor: 'divider',
        bgcolor: 'transparent',
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMore sx={{ fontSize: '1rem' }} />}
        sx={{
          minHeight: 36,
          px: 1,
          '& .MuiAccordionSummary-content': {
            my: 0.5,
            alignItems: 'center',
            width: '100%',
            ml: 0,
          },
        }}
      >
        {/* Grid Layout: [Favorite 30px] [Symbol 110px] [Price 70px] [Rating 80px] [Score 40px] [Signals 1fr] [Delete 30px] */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns:
              'minmax(30px, auto) 110px 70px 80px 40px 1fr 30px',
            gap: 1,
            width: '100%',
            alignItems: 'center',
          }}
        >
          {/* Favorite (30px) */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <Tooltip title="Favorite">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation()
                  toggleFavorite(symbol)
                }}
                sx={{ p: 0.5 }}
              >
                {isFav ? (
                  <Star sx={{ color: '#FFD700', fontSize: 18 }} />
                ) : (
                  <StarBorder sx={{ fontSize: 18, color: 'action.disabled' }} />
                )}
              </IconButton>
            </Tooltip>
          </Box>

          {/* Symbol (100px) */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              overflow: 'hidden',
            }}
          >
            <Typography
              variant="body1"
              sx={{
                fontWeight: 700,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                cursor: onSymbolClick ? 'pointer' : 'default',
                fontSize: '1rem',
              }}
              onClick={(e) => {
                e.stopPropagation()
                onSymbolClick?.(symbol)
              }}
              title={symbol}
            >
              {symbol}
            </Typography>
          </Box>

          {/* Price (100px) */}
          <Box>
            {price != null && (
              <Typography
                variant="body2"
                sx={{
                  color: 'text.secondary',
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                }}
              >
                ${price.toFixed(2)}
              </Typography>
            )}
          </Box>

          {/* Rating (100px) */}
          <Box>
            <Chip
              label={summaryRec}
              size="small"
              color={recChipColor(summaryRec)}
              variant="filled"
              sx={{ fontSize: '0.6rem', height: 20 }}
            />
          </Box>

          {/* Score (80px) */}
          <Box>
            <Typography
              variant="body2"
              sx={{
                fontWeight: 700,
                color: score > 70 ? 'success.main' : 'text.primary',
                fontSize: '0.9rem',
              }}
            >
              {score.toFixed(1)}
            </Typography>
          </Box>

          {/* Signals (1fr) */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              overflow: 'hidden',
            }}
          >
            {signals.length > 0 && (
              <Chip
                label={`${signals.length} sig`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.55rem', height: 18, flexShrink: 0 }}
              />
            )}
            {signals.slice(0, 2).map((s, i) => (
              <Chip
                key={i}
                label={s.description.split(':')[0]}
                size="small"
                color={s.is_bullish ? 'success' : 'error'}
                variant={s.strength === 'strong' ? 'filled' : 'outlined'}
                sx={{
                  fontSize: '0.55rem',
                  height: 18,
                  flexShrink: 1,
                  maxWidth: 80,
                  '& .MuiChip-label': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            ))}
            {/* Trend icon */}
            <Box sx={{ flexShrink: 0, display: 'flex', ml: 1 }}>
              {score > 2 && (
                <TrendingUp sx={{ fontSize: 14, color: '#66bb6a' }} />
              )}
              {score < -2 && (
                <TrendingDown sx={{ fontSize: 14, color: '#ff5252' }} />
              )}
              {score >= -2 && score <= 2 && (
                <Remove sx={{ fontSize: 14, color: '#9e9e9e' }} />
              )}
            </Box>
          </Box>

          {/* Delete (40px) */}
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Tooltip title="Remove symbol">
              <IconButton
                size="small"
                sx={{
                  opacity: 0.6,
                  '&:hover': { opacity: 1, color: 'error.main' },
                  p: 0.25,
                }}
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove(symbol)
                }}
              >
                <Delete sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </AccordionSummary>

      <AccordionDetails sx={{ pt: 0, pb: 1.5, px: 2 }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' },
            gap: 2,
          }}
        >
          {/* Column 1: Oscillators */}
          {technicals && (
            <IndicatorTable
              title="Oscillators"
              indicators={technicals.oscillators.indicators}
              activeOverlays={activeOverlays}
              onToggleIndicator={onToggleIndicator}
            />
          )}

          {/* Column 2: Moving Averages */}
          {technicals && (
            <IndicatorTable
              title="Moving Averages"
              indicators={technicals.moving_averages.indicators}
              activeOverlays={activeOverlays}
              onToggleIndicator={onToggleIndicator}
            />
          )}

          {/* Column 3: Screener Signals + Strategies */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <Box>
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 700,
                  color: 'text.secondary',
                  textTransform: 'uppercase',
                  fontSize: '0.6rem',
                  letterSpacing: 1,
                }}
              >
                Detected Signals
              </Typography>
              <Box sx={{ mt: 0.5 }}>
                <SignalsList signals={signals} />
              </Box>
            </Box>

            <Divider />

            <StrategyRecommendations
              signals={signals}
              activeOverlays={activeOverlays}
              onToggleStrategy={onToggleStrategy}
            />

            <Divider />

            {/* Asset-class-specific recommendations */}
            <AssetClassRecommendations
              symbol={symbol}
              activeOverlays={activeOverlays}
              onToggleStrategy={onToggleStrategy}
              onToggleIndicator={onToggleIndicator}
            />

            {/* Signal details for active strategies */}
            {Array.from(activeOverlays)
              .filter((id) => STRATEGY_COLORS[id])
              .map((stratId) => (
                <SignalDetail
                  key={stratId}
                  symbol={symbol}
                  strategyId={stratId}
                  currentPrice={price}
                />
              ))}

            {/* Merged Notes (Screener + Asset Class) */}
            {(() => {
              const assetClass = getAssetClass(symbol)
              const acNotes =
                ASSET_CLASS_RECOMMENDATIONS[assetClass]?.notes || []
              const screenerNotes = screener?.notes || []
              const allNotes = [...acNotes, ...screenerNotes]

              if (allNotes.length === 0) return null

              return (
                <Box>
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 700,
                      color: 'text.secondary',
                      textTransform: 'uppercase',
                      fontSize: '0.65rem',
                      letterSpacing: 1,
                    }}
                  >
                    Notes
                  </Typography>
                  <Typography
                    variant="body2"
                    display="block"
                    sx={{
                      fontSize: '0.8rem',
                      color: 'text.secondary',
                      mt: 0.5,
                      lineHeight: 1.5,
                    }}
                  >
                    {allNotes.map((n, i) => (
                      <span key={i}>
                        {i > 0 && (
                          <span style={{ margin: '0 8px', color: '#666' }}>
                            •
                          </span>
                        )}
                        {n}
                      </span>
                    ))}
                  </Typography>
                </Box>
              )
            })()}

            <Divider />

            <NewsSentimentPanel symbol={symbol} compact />
          </Box>
        </Box>

        {/* Summary bar */}
        {technicals && (
          <Box
            sx={{
              display: 'flex',
              gap: 2,
              mt: 1.5,
              pt: 1,
              borderTop: 1,
              borderColor: 'divider',
            }}
          >
            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.6rem' }}
              >
                Osc:
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: recColor(technicals.oscillators.recommendation),
                  fontWeight: 600,
                  fontSize: '0.65rem',
                }}
              >
                {technicals.oscillators.recommendation}
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.55rem' }}
              >
                ({technicals.oscillators.buy_count}B/
                {technicals.oscillators.sell_count}S/
                {technicals.oscillators.neutral_count}N)
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.6rem' }}
              >
                MAs:
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: recColor(technicals.moving_averages.recommendation),
                  fontWeight: 600,
                  fontSize: '0.65rem',
                }}
              >
                {technicals.moving_averages.recommendation}
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.55rem' }}
              >
                ({technicals.moving_averages.buy_count}B/
                {technicals.moving_averages.sell_count}S/
                {technicals.moving_averages.neutral_count}N)
              </Typography>
            </Box>
            {screener && (
              <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontSize: '0.6rem' }}
                >
                  Screener:
                </Typography>
                <Chip
                  label={screener.recommendation
                    .toUpperCase()
                    .replace('_', ' ')}
                  size="small"
                  color={recChipColor(screener.recommendation)}
                  sx={{ fontSize: '0.55rem', height: 18 }}
                />
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontSize: '0.55rem' }}
                >
                  (conf: {(screener.confidence * 100).toFixed(0)}%)
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  )
}

// ── Main Component ─────────────────────────────────────────────────────
interface MarketIntelProps {
  symbols?: string[]
  onSymbolClick?: (symbol: string) => void
  onApplyOverlay?: (overlay: ApplyOverlay) => void
  onRemoveOverlay?: (name: string) => void
}

export const MarketIntel: React.FC<MarketIntelProps> = ({
  symbols = DEFAULT_SYMBOLS,
  onSymbolClick,
  onApplyOverlay,
  onRemoveOverlay,
}) => {
  const [localSymbols, setLocalSymbols] = useState(symbols)
  const [activeOverlays, setActiveOverlays] = useState<Set<string>>(new Set())
  const { isFavorite } = useFavoritesStore()

  const handleToggleStrategy = useCallback(
    (id: string) => {
      setActiveOverlays((prev) => {
        const next = new Set(prev)
        if (next.has(id)) {
          next.delete(id)
          onRemoveOverlay?.(id)
        } else {
          next.add(id)
          onApplyOverlay?.({
            name: id,
            params: {},
            color: STRATEGY_COLORS[id] || '#2962FF',
            type: 'strategy',
          })
        }
        return next
      })
    },
    [onApplyOverlay, onRemoveOverlay],
  )

  const handleToggleIndicator = useCallback(
    (displayName: string) => {
      const mapped = INDICATOR_NAME_MAP[displayName]
      if (!mapped) return
      setActiveOverlays((prev) => {
        const next = new Set(prev)
        if (next.has(mapped.name)) {
          next.delete(mapped.name)
          onRemoveOverlay?.(mapped.name)
        } else {
          next.add(mapped.name)
          onApplyOverlay?.({
            name: mapped.name,
            params: mapped.params || {},
            color: mapped.color,
            type: 'indicator',
          })
        }
        return next
      })
    },
    [onApplyOverlay, onRemoveOverlay],
  )
  const [liveEnabled, setLiveEnabled] = useState(false)
  const [addInput, setAddInput] = useState('')
  const [sortField, setSortField] = useState<
    'symbol' | 'price' | 'rating' | 'score' | 'favorite'
  >('score')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

  const handleSort = (
    field: 'symbol' | 'price' | 'rating' | 'score' | 'favorite',
  ) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDirection('desc') // Default to desc for new field
    }
  }

  // Stable key for caching: sort symbols to avoid order-dependent cache misses
  // If symbols change (e.g. switching universe), we want a fresh cache entry or hit existing one
  const stableKey = localSymbols.slice().sort().join(',')

  const {
    data: queryData,
    isLoading,
    isRefetching,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['marketIntel', stableKey],
    queryFn: async () => {
      if (localSymbols.length === 0) return { tech: [], screen: [] }
      const [techResults, screenResults] = await Promise.all([
        technicalsApi.watchlist(localSymbols),
        screenerApi
          .analyzePortfolio(localSymbols)
          .then((r) => r.results)
          .catch(() => [] as ScreenerResult[]),
      ])
      return { tech: techResults, screen: screenResults }
    },
    enabled: localSymbols.length > 0,
    refetchInterval: liveEnabled ? POLL_INTERVAL : false,
    staleTime: 5 * 60 * 1000, // 5 minutes frontend cache to persist across tab switches
    gcTime: 10 * 60 * 1000, // Keep unused data for 10 minutes (renamed from cacheTime in v5)
    placeholderData: (previousData) => previousData, // Keep previous data while refetching to avoid flicker
  })

  const techData = queryData?.tech || []
  const screenData = queryData?.screen || []
  const loading = isLoading // Only true on initial load
  const lastUpdate = dataUpdatedAt ? new Date(dataUpdatedAt) : null

  // Check if data is from cache (checking the first item is enough as they come together)
  const isCached = screenData.length > 0 && !!screenData[0].from_cache

  useEffect(() => {
    setLocalSymbols(symbols)
  }, [symbols])

  const handleAdd = () => {
    const sym = addInput.trim().toUpperCase()
    if (sym && !localSymbols.includes(sym)) {
      setLocalSymbols((prev) => [...prev, sym])
      setAddInput('')
    }
  }

  const handleRemove = (sym: string) => {
    setLocalSymbols((prev) => prev.filter((s) => s !== sym))
  }

  // Merge technicals + screener data per symbol
  const merged: SymbolIntel[] = localSymbols.map((sym) => ({
    symbol: sym,
    technicals: techData.find((t) => t.symbol === sym),
    screener: screenData.find((s) => s.symbol === sym),
  }))

  // Sort logic
  const sorted = [...merged].sort((a, b) => {
    let valA: string | number = ''
    let valB: string | number = ''

    switch (sortField) {
      case 'symbol':
        valA = a.symbol
        valB = b.symbol
        break
      case 'price':
        valA = a.screener?.current_price ?? 0
        valB = b.screener?.current_price ?? 0
        break
      case 'rating':
        // Map recommendation string to numeric value for sorting
        const getRatingVal = (r: string) => {
          if (r === 'Strong Buy') return 5
          if (r === 'Buy') return 4
          if (r === 'Neutral') return 3
          if (r === 'Sell') return 2
          if (r === 'Strong Sell') return 1
          return 0
        }
        valA = getRatingVal(a.technicals?.summary.recommendation || '')
        valB = getRatingVal(b.technicals?.summary.recommendation || '')
        break
      case 'score':
        valA = a.screener?.score ?? 0
        valB = b.screener?.score ?? 0
        break
      case 'favorite':
        // Sort by boolean (true=1, false=0)
        valA = isFavorite(a.symbol) ? 1 : 0
        valB = isFavorite(b.symbol) ? 1 : 0
        break
    }

    if (valA < valB) return sortDirection === 'asc' ? -1 : 1
    if (valA > valB) return sortDirection === 'asc' ? 1 : -1
    return 0
  })

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1.5,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            Market Intel
          </Typography>
          {lastUpdate && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: '0.7rem' }}
              >
                Updated {lastUpdate.toLocaleTimeString()}
              </Typography>
              {isCached && (
                <Chip
                  label="Cached"
                  size="small"
                  icon={<Source sx={{ fontSize: 10 }} />}
                  sx={{
                    height: 16,
                    fontSize: '0.6rem',
                    '.MuiChip-icon': { fontSize: 10 },
                  }}
                />
              )}
            </Box>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
          <input
            value={addInput}
            onChange={(e) => setAddInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            placeholder="Add symbol.."
            style={{
              backgroundColor: '#2a2e39',
              color: '#e5e7eb',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '0.75rem',
              width: '80px',
              border: '1px solid #374151',
              outline: 'none',
            }}
          />
          <IconButton
            size="small"
            onClick={handleAdd}
            disabled={!addInput.trim()}
          >
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              +
            </Typography>
          </IconButton>
          <div
            style={{
              height: '16px',
              width: '1px',
              backgroundColor: '#374151',
              margin: '0 4px',
            }}
          />
          <Tooltip
            title={liveEnabled ? 'Pause live updates' : 'Resume live updates'}
          >
            <IconButton size="small" onClick={() => setLiveEnabled((v) => !v)}>
              {liveEnabled ? (
                <Pause fontSize="small" color="success" />
              ) : (
                <PlayArrow fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh now">
            <IconButton
              size="small"
              onClick={() => refetch()}
              disabled={isLoading || isRefetching}
            >
              {isLoading || isRefetching ? (
                <CircularProgress size={16} />
              ) : (
                <Refresh fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && techData.length === 0 && <LinearProgress />}

      {/* Headers with Sort Clicks */}
      {/* Grid Layout: [ExpandIcon 24px] [Symbol 100px] [Price 120px] [Rating 100px] [Score 80px] [Signals 1fr] [Delete 40px] */}
      {/* Headers with Sort Clicks */}
      {/* Grid Layout: [Spacer 24px] [Symbol 60px] [Price 70px] [Rating 80px] [Score 50px] [Signals 1fr] [Delete 30px] */}
      {/* Headers with Sort Clicks */}
      {/* Grid Layout: [Spacer 18px] [Symbol 85px] [Price 70px] [Rating 80px] [Score 40px] [Signals 1fr] [Delete 30px] */}
      {/* Headers with Sort Clicks */}
      {/* Grid Layout: [Symbol 110px] [Price 70px] [Rating 80px] [Score 40px] [Signals 1fr] [Delete 30px] */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns:
            'minmax(30px, auto) 110px 70px 80px 40px 1fr 30px',
          alignItems: 'center',
          gap: 1,
          px: 1,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'action.hover',
        }}
      >
        {/* Favorite Header */}
        <Box
          onClick={() => handleSort('favorite')}
          sx={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {sortField === 'favorite' ? (
            <Star sx={{ fontSize: 16, color: 'primary.main' }} />
          ) : (
            <StarBorder sx={{ fontSize: 16, color: 'text.secondary' }} />
          )}
        </Box>
        {/* Symbol Header */}
        <Box
          onClick={() => handleSort('symbol')}
          sx={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontWeight: 700,
              color: sortField === 'symbol' ? 'primary.main' : 'text.secondary',
            }}
          >
            SYMBOL
          </Typography>
        </Box>
        <Box
          onClick={() => handleSort('price')}
          sx={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontWeight: 700,
              color: sortField === 'price' ? 'primary.main' : 'text.secondary',
            }}
          >
            PRICE
          </Typography>
        </Box>
        <Box
          onClick={() => handleSort('rating')}
          sx={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontWeight: 700,
              color: sortField === 'rating' ? 'primary.main' : 'text.secondary',
            }}
          >
            RATING
          </Typography>
        </Box>
        <Box
          onClick={() => handleSort('score')}
          sx={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontWeight: 700,
              color: sortField === 'score' ? 'primary.main' : 'text.secondary',
            }}
          >
            SCR
          </Typography>
        </Box>
        <Box>
          <Typography
            variant="caption"
            sx={{ fontWeight: 700, color: 'text.secondary' }}
          >
            SIGNALS
          </Typography>
        </Box>
        <Box /> {/* Spacer for Delete Button */}
      </Box>

      {/* Rows */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {sorted.map((item) => (
          <SymbolRow
            key={item.symbol}
            data={item}
            onSymbolClick={onSymbolClick}
            onRemove={handleRemove}
            activeOverlays={activeOverlays}
            onToggleStrategy={onApplyOverlay ? handleToggleStrategy : undefined}
            onToggleIndicator={
              onApplyOverlay ? handleToggleIndicator : undefined
            }
          />
        ))}
        {sorted.length === 0 && !loading && (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              No symbols added
            </Typography>
          </Box>
        )}
      </Box>
    </Card>
  )
}
