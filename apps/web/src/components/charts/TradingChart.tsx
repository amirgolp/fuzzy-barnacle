import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type HistogramData,
  type Time,
  type SeriesMarker,
  type UTCTimestamp,
  type ChartOptions,
  type DeepPartial,
} from 'lightweight-charts'
import { useCallback, useEffect, useRef } from 'react'

interface CustomSeriesData {
  data: (CandlestickData<Time> | HistogramData<Time>)[]
}

interface TradingChartProps {
  data: CandlestickData<Time>[]
  volumeData?: HistogramData<Time>[]
  indicators?: {
    name: string
    color: string
    data: (CandlestickData<Time> | HistogramData<Time>)[]
  }[]
  strategyMarkers?: SeriesMarker<Time>[]
  symbol: string
  currency: 'USD' | 'EUR'
  timeframe?: string
  onVisibleRangeChange?: (from: number) => void
  customSeries?: CustomSeriesData[]
}

// Only show session markers for very short intraday timeframes (not 1h - too noisy)
const MARKER_TIMEFRAMES = ['1m', '5m', '15m']

const MARKET_SESSIONS = [
  {
    name: 'NYC',
    openH: 9,
    openM: 30,
    closeH: 16,
    closeM: 0,
    dstRange: [2, 10],
    stdOff: -5,
    dstOff: -4,
    color: '#2962FF',
  },
  {
    name: 'LDN',
    openH: 8,
    openM: 0,
    closeH: 16,
    closeM: 30,
    dstRange: [2, 9],
    stdOff: 0,
    dstOff: 1,
    color: '#FF6D00',
  },
  {
    name: 'FRA',
    openH: 9,
    openM: 0,
    closeH: 17,
    closeM: 30,
    dstRange: [2, 9],
    stdOff: 1,
    dstOff: 2,
    color: '#AA00FF',
  },
]

function getUtcOffset(session: (typeof MARKET_SESSIONS)[0], d: Date): number {
  const m = d.getUTCMonth()
  return m >= session.dstRange[0] && m <= session.dstRange[1]
    ? session.dstOff
    : session.stdOff
}


function buildSessionMarkers(
  data: CandlestickData<Time>[],
  timeframe: string | undefined,
) {
  if (!MARKER_TIMEFRAMES.includes(timeframe || '') || data.length === 0)
    return []

  const dateSet = new Set<string>()
  for (const bar of data) {
    const d = new Date((bar.time as number) * 1000)
    dateSet.add(d.toISOString().slice(0, 10))
  }

  const timestamps = data.map((b) => b.time as number)
  const markers: SeriesMarker<Time>[] = []

  for (const dateStr of dateSet) {
    const dayBase = Math.floor(
      new Date(dateStr + 'T00:00:00Z').getTime() / 1000,
    )
    const d = new Date(dayBase * 1000)
    const dow = d.getUTCDay()
    if (dow === 0 || dow === 6) continue

    for (const session of MARKET_SESSIONS) {
      const off = getUtcOffset(session, d)
      const openTs = (dayBase +
        (session.openH - off) * 3600 +
        session.openM * 60) as UTCTimestamp
      const closeTs = (dayBase +
        (session.closeH - off) * 3600 +
        session.closeM * 60) as UTCTimestamp

      const openBar = findNearest(timestamps, openTs, 7200)
      const closeBar = findNearest(timestamps, closeTs, 7200)

      if (openBar !== null) {
        markers.push({
          time: openBar as UTCTimestamp,
          position: 'belowBar',
          color: session.color,
          shape: 'arrowUp',
          text: `${session.name} Open`,
        })
      }
      if (closeBar !== null && closeBar !== openBar) {
        markers.push({
          time: closeBar as UTCTimestamp,
          position: 'aboveBar',
          color: session.color,
          shape: 'arrowDown',
          text: `${session.name} Close`,
        })
      }
    }
  }

  markers.sort((a, b) => (a.time as number) - (b.time as number))
  return markers
}

function findNearest(
  sorted: number[],
  target: number,
  maxDist: number,
): number | null {
  if (sorted.length === 0) return null
  let lo = 0,
    hi = sorted.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < target) lo = mid + 1
    else hi = mid
  }
  const candidates = [lo, lo - 1].filter((i) => i >= 0 && i < sorted.length)
  let best = candidates[0]
  for (const c of candidates) {
    if (Math.abs(sorted[c] - target) < Math.abs(sorted[best] - target)) best = c
  }
  return Math.abs(sorted[best] - target) <= maxDist ? sorted[best] : null
}

export const TradingChart = ({
  data,
  volumeData,
  indicators,
  strategyMarkers,
  customSeries,
  symbol,
  currency,
  timeframe,
  onVisibleRangeChange,
}: TradingChartProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const indicatorSeriesRefs = useRef<ISeriesApi<'Line'>[]>([])
  const customSeriesRefs = useRef<ISeriesApi<'Custom'>[]>([])
  const markersPluginRef = useRef<unknown>(null)
  const dataRef = useRef(data)
  dataRef.current = data

  const colors = {
    background: '#131722',
    text: '#d1d4dc',
    grid: 'rgba(42, 46, 57, 0.6)',
    up: '#26a69a',
    down: '#ef5350',
  }

  // Debounced scroll-left handler
  const handleVisibleRangeChange = useCallback(() => {
    if (
      !chartRef.current ||
      !onVisibleRangeChange ||
      dataRef.current.length === 0
    )
      return
    const range = chartRef.current.timeScale().getVisibleLogicalRange()
    if (!range) return
    // If user scrolled so that logical range starts < 5 bars from beginning, request more
    if (range.from < 5) {
      const firstTs = dataRef.current[0]?.time as number
      if (firstTs) onVisibleRangeChange(firstTs)
    }
  }, [onVisibleRangeChange])

  // Initialize Chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    type ExtendedChartOptions = DeepPartial<ChartOptions> & {
      attributionLogo?: boolean
    }

    const chartOptions: ExtendedChartOptions = {
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.text,
        fontFamily: "'Inter', sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0)' },
        horzLines: { color: colors.grid },
      },
      width: chartContainerRef.current.clientWidth,
      height: 600,
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      attributionLogo: false,
      timeScale: {
        borderColor: colors.grid,
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: colors.grid,
      },
    }

    const chart = createChart(chartContainerRef.current, chartOptions)

    chartRef.current = chart

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    })
    candlestickSeriesRef.current = candlestickSeries

    if (volumeData) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })

      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })

      volumeSeries.setData(volumeData)
      volumeSeriesRef.current = volumeSeries
    }

    // Scroll-left detection
    chart
      .timeScale()
      .subscribeVisibleLogicalRangeChange(handleVisibleRangeChange)

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update Data + session markers + strategy markers
  useEffect(() => {
    if (!candlestickSeriesRef.current) return
    candlestickSeriesRef.current.setData(data)

    const sessionMarkers = buildSessionMarkers(data, timeframe)
    const allMarkers = [...sessionMarkers, ...(strategyMarkers || [])].sort(
      (a, b) => (a.time as number) - (b.time as number),
    )

    if (markersPluginRef.current) {
      // Use unknown and type assertion to avoid any
      ; (
        markersPluginRef.current as {
          setMarkers: (
            markers: SeriesMarker<Time>[],
          ) => void
        }
      ).setMarkers(allMarkers)
    } else if (allMarkers.length > 0) {
      markersPluginRef.current = createSeriesMarkers(
        candlestickSeriesRef.current,
        allMarkers,
      )
    }
  }, [data, timeframe, strategyMarkers])

  useEffect(() => {
    if (!volumeSeriesRef.current || !volumeData) return
    volumeSeriesRef.current.setData(volumeData)
  }, [volumeData])

  // Update Indicators
  useEffect(() => {
    if (!chartRef.current) return

    indicatorSeriesRefs.current.forEach((series) => {
      chartRef.current?.removeSeries(series)
    })
    indicatorSeriesRefs.current = []

    indicators?.forEach((ind) => {
      const series = chartRef.current!.addSeries(LineSeries, {
        color: ind.color,
        lineWidth: 2,
        title: ind.name,
        crosshairMarkerVisible: true,
      })
      series.setData(ind.data)
      indicatorSeriesRefs.current.push(series)
    })
  }, [indicators])

  // Update Custom Series (Fills)
  useEffect(() => {
    if (!chartRef.current) return

    // Cleanup previous custom series
    customSeriesRefs.current.forEach((series) => {
      chartRef.current?.removeSeries(series)
    })
    customSeriesRefs.current = []

    if (!customSeries) return

    import('./series/AreaFillSeries').then(({ AreaFillSeries }) => {
      if (!chartRef.current) return
      customSeries.forEach((cs) => {
        const series = chartRef.current!.addCustomSeries(new AreaFillSeries(), {
          priceLineVisible: false,
        })
        series.setData(cs.data)
        customSeriesRefs.current.push(series)
      })
    })
  }, [customSeries])

  // Update Options
  useEffect(() => {
    if (!chartRef.current) return
    chartRef.current.applyOptions({
      rightPriceScale: {
        scaleMargins: {
          top: 0.1,
          bottom: volumeData ? 0.2 : 0.1,
        },
      },
    })
  }, [symbol, currency, volumeData])

  return (
    <div
      ref={chartContainerRef}
      style={{ width: '100%', height: 600, position: 'relative' }}
    />
  )
}
