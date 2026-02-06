import {
  BaselineSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  LineSeries,
  createChart,
} from 'lightweight-charts'
import { useEffect, useRef } from 'react'

interface IndicatorPanelProps {
  type: 'rsi' | 'macd' | 'stoch' | 'line'
  data: any | any[]
  height?: number
  config?: any
}

export const IndicatorPanel = ({
  type,
  data,
  height = 200,
  config = {},
}: IndicatorPanelProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  // We'll store a list of series for cleanup, though this panel usually has static 1-3 series types
  const seriesRefs = useRef<ISeriesApi<any>[]>([])

  const colors = {
    background: '#131722',
    text: '#d1d4dc',
    grid: 'rgba(42, 46, 57, 0.6)',
  }

  // Initialize Chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      height,
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
      crosshair: { mode: CrosshairMode.Magnet },
      timeScale: { visible: false },
      rightPriceScale: {
        borderColor: colors.grid,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
    })

    chartRef.current = chart

    // Resize handler
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
  }, []) // Only run once on mount

  // Update Dimensions
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.applyOptions({ height })
    }
  }, [height])

  // Render Series Logic
  useEffect(() => {
    if (!chartRef.current) return

    // Cleanup previous series
    seriesRefs.current.forEach((series) =>
      chartRef.current?.removeSeries(series),
    )
    seriesRefs.current = []

    const chart = chartRef.current

    switch (type) {
      case 'rsi': {
        const series = chart.addSeries(BaselineSeries, {
          baseValue: { type: 'price', price: 50 },
          topLineColor: '#ef5350',
          bottomLineColor: '#26a69a',
          topFillColor1: 'rgba(239, 83, 80, 0.05)',
          topFillColor2: 'rgba(239, 83, 80, 0.28)',
          bottomFillColor1: 'rgba(38, 166, 154, 0.28)',
          bottomFillColor2: 'rgba(38, 166, 154, 0.05)',
          lineWidth: 2,
        })
        series.setData(data)
        seriesRefs.current.push(series)
        break
      }
      case 'macd': {
        const histSeries = chart.addSeries(HistogramSeries, {
          color: '#26a69a',
          priceScaleId: '',
        })
        histSeries.setData(data.histogram)
        seriesRefs.current.push(histSeries)

        const macdSeries = chart.addSeries(LineSeries, {
          color: '#2962FF',
          lineWidth: 2,
          priceScaleId: '',
        })
        macdSeries.setData(data.macd)
        seriesRefs.current.push(macdSeries)

        const signalSeries = chart.addSeries(LineSeries, {
          color: '#FF6D00',
          lineWidth: 2,
          priceScaleId: '',
        })
        signalSeries.setData(data.signal)
        seriesRefs.current.push(signalSeries)
        break
      }
      default: {
        const series = chart.addSeries(LineSeries, {
          color: config.color || '#2962FF',
          lineWidth: 2,
        })
        series.setData(data)
        seriesRefs.current.push(series)
      }
    }
  }, [type, data, config.color]) // Re-run when type/data changes

  return (
    <div
      style={{
        width: '100%',
        position: 'relative',
        borderTop: '1px solid #2a2e39',
      }}
    >
      <div ref={chartContainerRef} style={{ width: '100%' }} />
      <div
        style={{
          position: 'absolute',
          top: 8,
          left: 8,
          fontSize: '0.75rem',
          color: '#9ca3af',
          fontWeight: 500,
          zIndex: 10,
          pointerEvents: 'none',
        }}
      >
        {config.name || type.toUpperCase()}
      </div>
    </div>
  )
}
