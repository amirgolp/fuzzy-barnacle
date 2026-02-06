export type SignalType =
  | 'GOLDEN_CROSS'
  | 'DEATH_CROSS'
  | 'RSI_OVERSOLD'
  | 'RSI_OVERBOUGHT'
  | 'MACD_BULLISH'
  | 'MACD_BEARISH'
  | 'STOCH_OVERSOLD'
  | 'STOCH_OVERBOUGHT'
  | 'BREAKOUT'
  | 'BREAKDOWN'
  | 'BOLLINGER_OVERSOLD'
  | 'BOLLINGER_OVERBOUGHT'
  | 'MFI_OVERSOLD'
  | 'MFI_OVERBOUGHT'
  | 'OBV_DIVERGENCE_BULL'
  | 'OBV_DIVERGENCE_BEAR'
  | 'EMA_CROSSOVER_BULL'
  | 'EMA_CROSSOVER_BEAR'
  | 'PRICE_ABOVE_EMA'
  | 'PRICE_BELOW_EMA'
  | 'VOLUME_SPIKE'

export type SignalStrength = 'strong' | 'moderate' | 'weak'

export type Recommendation =
  | 'strong_buy'
  | 'buy'
  | 'hold'
  | 'sell'
  | 'strong_sell'

export interface TechnicalSignal {
  signal_type: SignalType
  strength: SignalStrength
  description: string
  date_detected: string // ISO date string
  value?: number
  is_bullish: boolean
  is_bearish: boolean
  score_value: number
}

export interface ScreenerResult {
  symbol: string
  screening_date: string // ISO date string
  signals: TechnicalSignal[]
  score: number
  bullish_score: number
  bearish_score: number
  recommendation: Recommendation
  confidence: number
  current_price?: number
  current_rsi?: number
  rank?: number
  notes: string[]
  gauge_value: number
  gauge_label: string
  from_cache?: boolean
}

export interface PortfolioScreeningResult {
  screening_date: string
  total_screened: number
  tickers_matching: number
  results: ScreenerResult[]
  top_picks: string[]
  summary: string
}
