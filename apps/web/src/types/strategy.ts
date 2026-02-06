export interface StrategyInfo {
  id: string
  name: string
  description: string
  params: Record<string, any>
  default_params: Record<string, any>
}

export interface BacktestRequest {
  symbol: string
  strategy_id: string
  params?: Record<string, any>
  timeframe?: string
  start?: string
  end?: string
  initial_cash?: number
  fee_bps?: number
  slippage_bps?: number
}

export interface TradeRecord {
  entry_date: string
  exit_date: string | null
  symbol: string
  side: string
  entry_price: number
  exit_price: number | null
  quantity: number
  pnl: number
  return_pct: number
}

export interface BacktestResult {
  cagr: number
  total_return: number
  volatility: number
  sharpe_ratio: number
  sortino_ratio: number
  calmar_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  exposure: number
  equity_curve: { time: string; value: number }[]
  drawdown_curve: { time: string; value: number }[]
  trades: TradeRecord[]
  strategy_id: string
  symbol: string
  start_date: string | null
  end_date: string | null
}

export interface OptimizeRequest {
  symbols: string[]
  method: string
  timeframe?: string
  start?: string
  end?: string
  risk_aversion?: number
}

export interface PortfolioOptimizationResult {
  method: string
  weights: Record<string, number>
  risk_contributions: Record<string, number>
  expected_return: number
  portfolio_volatility: number
  sharpe_ratio: number
}
