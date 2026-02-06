import type { TechnicalsResult } from '../types/technicals'
import { api } from './client'

export const technicalsApi = {
  analyze: async (
    symbol: string,
    timeframe: string = '1d',
  ): Promise<TechnicalsResult> => {
    const response = await api.get<TechnicalsResult>('/features/technicals', {
      params: { symbol, timeframe },
    })
    return response.data
  },
  watchlist: async (
    symbols: string[],
    timeframe: string = '1d',
  ): Promise<TechnicalsResult[]> => {
    const response = await api.get<TechnicalsResult[]>('/features/watchlist', {
      params: { symbols: symbols.join(','), timeframe },
    })
    return response.data
  },
}
