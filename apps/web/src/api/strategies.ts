import type {
  BacktestRequest,
  BacktestResult,
  OptimizeRequest,
  PortfolioOptimizationResult,
  StrategyInfo,
} from '../types/strategy'
import { api } from './client'

export const strategiesApi = {
  list: async (): Promise<StrategyInfo[]> => {
    const response = await api.get<StrategyInfo[]>('/strategies/')
    return response.data
  },

  get: async (strategyId: string): Promise<StrategyInfo> => {
    const response = await api.get<StrategyInfo>(`/strategies/${strategyId}`)
    return response.data
  },

  backtest: async (req: BacktestRequest): Promise<BacktestResult> => {
    const response = await api.post<BacktestResult>('/strategies/backtest', req)
    return response.data
  },

  optimize: async (
    req: OptimizeRequest,
  ): Promise<PortfolioOptimizationResult> => {
    const response = await api.post<PortfolioOptimizationResult>(
      '/strategies/optimize',
      req,
    )
    return response.data
  },
}
