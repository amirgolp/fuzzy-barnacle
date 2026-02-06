import { api } from './client'

export interface Bar {
  time: string // ISO string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface BarsResponse {
  symbol: string
  timeframe: string
  count: number
  data: Bar[]
}

function defaultDaysForTimeframe(tf: string): number {
  switch (tf) {
    case '1m':
      return 5
    case '5m':
      return 10
    case '15m':
      return 20
    case '1h':
      return 60
    case '4h':
      return 120
    case '1d':
      return 365
    case '1w':
      return 365 * 3
    default:
      return 365
  }
}

function toBars(data: Bar[]): Bar[] {
  return data.map((bar) => ({
    ...bar,
    time: Math.floor(new Date(bar.time).getTime() / 1000) as any,
  }))
}

export const dataApi = {
  getBars: async (symbol: string, timeframe: string = '1d'): Promise<Bar[]> => {
    const end = new Date()
    const start = new Date()
    const days = Math.ceil(defaultDaysForTimeframe(timeframe) * 1.5)
    start.setDate(start.getDate() - days)

    const response = await api.get<BarsResponse>('/data/bars', {
      params: {
        symbol,
        timeframe,
        start: start.toISOString(),
        end: end.toISOString(),
        adjusted: true,
      },
    })
    return toBars(response.data.data)
  },

  getBarsByRange: async (
    symbol: string,
    timeframe: string,
    start: Date,
    end: Date,
  ): Promise<Bar[]> => {
    const response = await api.get<BarsResponse>('/data/bars', {
      params: {
        symbol,
        timeframe,
        start: start.toISOString(),
        end: end.toISOString(),
        adjusted: true,
      },
    })
    return toBars(response.data.data)
  },
}
