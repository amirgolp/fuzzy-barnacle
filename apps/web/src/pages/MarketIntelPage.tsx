import { Alert, Box, Paper, Snackbar } from '@mui/material'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../api/client'
import { type Bar, dataApi } from '../api/data'
import { ChartToolbar } from '../components/charts/ChartToolbar'
import type { IndicatorSelection } from '../components/charts/IndicatorDialog'
import { IndicatorLegend } from '../components/charts/IndicatorLegend'
import { IndicatorSettingsDialog } from '../components/charts/IndicatorSettingsDialog'
import { TradingChart } from '../components/charts/TradingChart'
import { MarketIntel } from '../components/signals/MarketIntel'
import { useBars, useUpdateBarsCache } from '../hooks/useMarketData'

interface ChartIndicator {
  name: string
  color: string
  data: { time: any; value: number }[]
}

export const MarketIntelPage = () => {
  const [symbol, setSymbol] = useState('AAPL')
  const [timeframe, setTimeframe] = useState('1d')

  // Additional bars loaded via scroll (merged with query data)
  const [additionalBars, setAdditionalBars] = useState<Bar[]>([])

  const [activeIndicators, setActiveIndicators] = useState<
    IndicatorSelection[]
  >([])
  const [chartIndicators, setChartIndicators] = useState<ChartIndicator[]>([])
  const [chartCustomSeries, setChartCustomSeries] = useState<any[]>([])
  const [strategyMarkers, setStrategyMarkers] = useState<any[]>([])

  const fetchingMoreRef = useRef(false)
  const updateBarsCache = useUpdateBarsCache()

  const [editDialog, setEditDialog] = useState<{
    open: boolean
    name: string
    params: any
  }>({
    open: false,
    name: '',
    params: {},
  })

  // Use React Query for data fetching with caching
  const {
    data: queryBars,
    isLoading: loading,
    error: queryError,
  } = useBars(symbol, timeframe)

  // Reset additional bars when symbol/timeframe changes
  useEffect(() => {
    setAdditionalBars([])
  }, [symbol, timeframe])

  // Merge query bars with additional bars loaded via scroll
  const bars = useMemo(() => {
    if (!queryBars) return []
    if (additionalBars.length === 0) return queryBars

    const existingTimes = new Set(queryBars.map((b) => b.time))
    const uniqueAdditional = additionalBars.filter(
      (b) => !existingTimes.has(b.time),
    )
    return [...uniqueAdditional, ...queryBars].sort(
      (a, b) => (a.time as any) - (b.time as any),
    )
  }, [queryBars, additionalBars])

  const [snackbarError, setSnackbarError] = useState<string | null>(null)

  useEffect(() => {
    if (queryError) setSnackbarError('Failed to load chart data')
  }, [queryError])

  const toVolume = (barData: Bar[]) =>
    barData.map((d) => ({
      time: d.time,
      value: d.volume,
      color:
        d.close >= d.open
          ? 'rgba(38, 166, 154, 0.5)'
          : 'rgba(239, 83, 80, 0.5)',
    }))

  const volumeData = useMemo(() => toVolume(bars), [bars])

  useEffect(() => {
    if (bars.length === 0 || activeIndicators.length === 0) {
      setChartIndicators([])
      setChartCustomSeries([])
      setStrategyMarkers([])
      return
    }

    const computeAll = async () => {
      const results: ChartIndicator[] = []
      const customSeries: any[] = []
      let allMarkers: any[] = []
      const startDate = new Date((bars[0].time as any) * 1000)
      const endDate = new Date((bars[bars.length - 1].time as any) * 1000)

      for (const ind of activeIndicators) {
        try {
          if (ind.type === 'strategy') {
            const res = await api.post('/strategies/overlay', {
              symbol,
              strategy_id: ind.name,
              params: ind.params,
              timeframe,
              start: startDate.toISOString(),
              end: endDate.toISOString(),
            })

            const overlayData = res.data
            for (const line of overlayData.lines || []) {
              results.push({
                name: line.name,
                color: line.color,
                data: line.data,
              })
            }
            if (overlayData.markers) {
              allMarkers = [...allMarkers, ...overlayData.markers]
            }
          } else {
            const res = await api.post('/features/compute', {
              symbol,
              timeframe,
              start: startDate.toISOString(),
              end: endDate.toISOString(),
              indicator: ind.name,
              params: ind.params,
            })

            const values = res.data.values || {}
            const data = Object.entries(values)
              .map(([dateStr, val]) => ({
                time: Math.floor(new Date(dateStr).getTime() / 1000),
                value: val as number,
              }))
              .sort((a, b) => a.time - b.time)

            results.push({ name: ind.name, color: ind.color, data })
          }
        } catch (e) {
          console.error(`Failed to compute ${ind.name}:`, e)
        }
      }

      setChartIndicators(results)
      setChartCustomSeries(customSeries)
      setStrategyMarkers(allMarkers.sort((a, b) => a.time - b.time))
    }

    computeAll()
  }, [bars, activeIndicators, symbol, timeframe])

  const handleScrollLeft = useCallback(
    async (firstTimestamp: number) => {
      if (fetchingMoreRef.current) return
      fetchingMoreRef.current = true
      try {
        const endDate = new Date(firstTimestamp * 1000)
        const startDate = new Date(endDate)
        startDate.setDate(startDate.getDate() - 90)
        const olderBars = await dataApi.getBarsByRange(
          symbol,
          timeframe,
          startDate,
          endDate,
        )
        if (olderBars.length > 0) {
          // Add to additional bars (will be merged with query data)
          setAdditionalBars((prev) => {
            const existingTimes = new Set(prev.map((b) => b.time))
            const newBars = olderBars.filter((b) => !existingTimes.has(b.time))
            return [...newBars, ...prev].sort(
              (a, b) => (a.time as any) - (b.time as any),
            )
          })
          // Also update the cache so data persists across tab switches
          updateBarsCache(symbol, timeframe, olderBars, true)
        }
      } catch (e) {
        console.error('Failed to fetch older data:', e)
      } finally {
        fetchingMoreRef.current = false
      }
    },
    [symbol, timeframe, updateBarsCache],
  )

  const handleAddIndicator = useCallback((ind: IndicatorSelection) => {
    setActiveIndicators((prev) => {
      if (prev.some((i) => i.name === ind.name)) return prev
      return [...prev, ind]
    })
  }, [])

  const handleRemoveIndicator = useCallback((name: string) => {
    setActiveIndicators((prev) => prev.filter((i) => i.name !== name))
  }, [])

  const handleEditIndicator = useCallback(
    (name: string) => {
      const ind = activeIndicators.find((i) => i.name === name)
      if (ind) {
        setEditDialog({ open: true, name: ind.name, params: ind.params })
      }
    },
    [activeIndicators],
  )

  const handleSaveSettings = useCallback((name: string, newParams: any) => {
    setActiveIndicators((prev) =>
      prev.map((i) => (i.name === name ? { ...i, params: newParams } : i)),
    )
  }, [])

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
        color: 'text.primary',
      }}
    >
      <ChartToolbar
        symbol={symbol}
        onSymbolChange={setSymbol}
        timeframe={timeframe}
        onTimeframeChange={setTimeframe}
        onAddIndicator={handleAddIndicator}
        onRemoveIndicator={handleRemoveIndicator}
        activeIndicators={activeIndicators.map((i) => i.name)}
      />

      <Box sx={{ flex: 1, display: 'flex', gap: 2, p: 2, overflow: 'hidden' }}>
        {/* Left: MarketIntel panel */}
        <Box sx={{ width: 520, flexShrink: 0, overflow: 'auto' }}>
          <MarketIntel
            onSymbolClick={setSymbol}
            onApplyOverlay={handleAddIndicator}
            onRemoveOverlay={handleRemoveIndicator}
          />
        </Box>

        {/* Right: Chart */}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            minWidth: 0,
          }}
        >
          <Paper
            sx={{
              flex: 1,
              minHeight: 600,
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 2,
            }}
          >
            <IndicatorLegend
              indicators={activeIndicators}
              onEdit={handleEditIndicator}
              onRemove={handleRemoveIndicator}
            />
            {loading && (
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  zIndex: 10,
                }}
              >
                Loading Chart...
              </Box>
            )}
            <TradingChart
              data={bars}
              volumeData={volumeData}
              symbol={symbol}
              currency="USD"
              timeframe={timeframe}
              indicators={chartIndicators}
              strategyMarkers={strategyMarkers}
              customSeries={chartCustomSeries}
              onVisibleRangeChange={handleScrollLeft}
            />
          </Paper>
        </Box>
      </Box>

      <IndicatorSettingsDialog
        open={editDialog.open}
        indicatorName={editDialog.name}
        params={editDialog.params}
        onClose={() => setEditDialog((prev) => ({ ...prev, open: false }))}
        onSave={handleSaveSettings}
      />

      <Snackbar
        open={!!snackbarError}
        autoHideDuration={6000}
        onClose={() => setSnackbarError(null)}
      >
        <Alert severity="error" onClose={() => setSnackbarError(null)}>
          {snackbarError}
        </Alert>
      </Snackbar>
    </Box>
  )
}
