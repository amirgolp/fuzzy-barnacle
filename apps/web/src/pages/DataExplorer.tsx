import { ChevronRight } from '@mui/icons-material'
import {
  Alert,
  Box,
  Divider,
  Drawer,
  IconButton,
  Paper,
  Snackbar,
  Typography,
} from '@mui/material'
import { LineChart } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { ChartToolbar } from '../components/charts/ChartToolbar'
import type { IndicatorSelection } from '../components/charts/IndicatorDialog'
import { IndicatorSettingsDialog } from '../components/charts/IndicatorSettingsDialog'
import { NewsSentimentPanel } from '../components/signals/NewsSentimentPanel'
import { TechnicalsPanel } from '../components/signals/TechnicalsPanel'
import { TVDashboard } from '../components/charts/TVDashboard'



export const DataExplorer = () => {
  const [symbol, setSymbol] = useState('AAPL')
  const [timeframe, setTimeframe] = useState('1d')

  const [activeIndicators, setActiveIndicators] = useState<
    IndicatorSelection[]
  >([])

  // Dialog State
  const [editDialog, setEditDialog] = useState<{
    open: boolean
    name: string
    params: any
  }>({
    open: false,
    name: '',
    params: {},
  })

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Drawer State
  const [drawerOpen, setDrawerOpen] = useState(false)

  // Use React Query for data fetching with caching
  // Reset error when symbol/timeframe changes
  useEffect(() => {
    setError(null)
  }, [symbol, timeframe])

  const handleAddIndicator = useCallback((ind: IndicatorSelection) => {
    setActiveIndicators((prev) => {
      if (prev.some((i) => i.name === ind.name)) return prev
      return [...prev, ind]
    })
  }, [])

  const handleRemoveIndicator = useCallback((name: string) => {
    setActiveIndicators((prev) => prev.filter((i) => i.name !== name))
  }, [])



  const handleSaveSettings = useCallback((name: string, newParams: any) => {
    setActiveIndicators((prev) =>
      prev.map((i) => (i.name === name ? { ...i, params: newParams } : i)),
    )
  }, [])

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
        color: 'text.primary',
        overflow: 'hidden',
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
        extraActions={
          <button
            onClick={() => setDrawerOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              fontSize: '0.875rem',
              fontWeight: 500,
              color: '#d1d4dc',
              backgroundColor: 'transparent',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'color 0.2s, background-color 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#fff'
              e.currentTarget.style.backgroundColor = '#2a2e39'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#d1d4dc'
              e.currentTarget.style.backgroundColor = 'transparent'
            }}
            title="Open Analysis Panel"
          >
            <LineChart size={16} />
            <span>Analysis</span>
          </button>
        }
      />

      <Box
        sx={{
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          flexGrow: 1,
          overflow: 'hidden',
        }}
      >
        {/* Chart Section */}
        <Paper
          sx={{
            flexGrow: 1,
            position: 'relative',
            overflow: 'hidden',
            borderRadius: 2,
            bgcolor: 'background.paper',
            display: 'flex',
            flexDirection: 'column',
          }}
        >

          <TVDashboard
            symbol={symbol}
            interval={
              timeframe === '1d' ? 'D' :
                timeframe === '1w' ? 'W' :
                  timeframe === '1M' ? 'M' :
                    timeframe === '1m' ? '1' :
                      timeframe === '5m' ? '5' :
                        timeframe === '15m' ? '15' :
                          timeframe === '30m' ? '30' :
                            timeframe === '1h' ? '60' :
                              timeframe === '4h' ? '240' :
                                timeframe
            }
          />
        </Paper>
      </Box>

      {/* Analysis Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
          sx: {
            width: 600,
            bgcolor: 'background.default',
            p: 2,
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <IconButton onClick={() => setDrawerOpen(false)} sx={{ mr: 1 }}>
            <ChevronRight />
          </IconButton>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Analysis Panel
          </Typography>
        </Box>
        <Divider sx={{ mb: 3 }} />

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Box>
            <TechnicalsPanel symbol={symbol} timeframe={timeframe} />
          </Box>
          <Box>
            <NewsSentimentPanel symbol={symbol} />
          </Box>
        </Box>
      </Drawer>

      {/* Settings Dialog */}
      <IndicatorSettingsDialog
        open={editDialog.open}
        indicatorName={editDialog.name}
        params={editDialog.params}
        onClose={() => setEditDialog((prev) => ({ ...prev, open: false }))}
        onSave={handleSaveSettings}
      />

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  )
}
