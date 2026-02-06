export type Action = 'Buy' | 'Sell' | 'Neutral'

export interface IndicatorReading {
  name: string
  value: number
  action: Action
}

export interface CategorySummary {
  title: string
  indicators: IndicatorReading[]
  buy_count: number
  sell_count: number
  neutral_count: number
  recommendation: string
  gauge_value: number
}

export interface TechnicalsResult {
  symbol: string
  date: string
  oscillators: CategorySummary
  moving_averages: CategorySummary
  summary: CategorySummary
}
