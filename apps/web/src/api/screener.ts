import type {
  PortfolioScreeningResult,
  ScreenerResult,
} from '../types/screener'
import { api } from './client'

export const screenerApi = {
  analyze: async (
    symbol: string,
    timeframe: string = '1d',
  ): Promise<ScreenerResult> => {
    const response = await api.get<ScreenerResult>(
      '/features/screener/analyze',
      {
        params: {
          symbol,
          timeframe,
        },
      },
    )
    return response.data
  },

  analyzePortfolio: async (
    symbols: string[],
    timeframe: string = '1d',
  ): Promise<PortfolioScreeningResult> => {
    const response = await api.post<PortfolioScreeningResult>(
      '/features/screener/portfolio',
      {
        symbols,
        timeframe,
      },
    )
    return response.data
  },
}
