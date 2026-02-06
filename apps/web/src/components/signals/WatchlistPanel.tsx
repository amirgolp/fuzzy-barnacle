import { Pause, PlayArrow, Refresh } from '@mui/icons-material'
import {
  Box,
  Card,
  Chip,
  CircularProgress,
  IconButton,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material'
import React, { useCallback, useEffect, useRef, useState } from 'react'

import { technicalsApi } from '../../api/technicals'
import type { TechnicalsResult } from '../../types/technicals'

const DEFAULT_SYMBOLS = [
  'AAPL',
  'MSFT',
  'GOOGL',
  'AMZN',
  'TSLA',
  'NVDA',
  'META',
  'JPM',
  'V',
  'SPY',
]
const POLL_INTERVAL = 60_000 // 60 seconds

interface WatchlistPanelProps {
  symbols?: string[]
  onSymbolClick?: (symbol: string) => void
}

const recColor = (rec: string) => {
  const r = rec.toLowerCase()
  if (r.includes('strong buy')) return '#26a69a'
  if (r.includes('buy')) return '#66bb6a'
  if (r.includes('strong sell')) return '#f44336'
  if (r.includes('sell')) return '#ff5252'
  return '#9e9e9e'
}

const recChipColor = (
  rec: string,
): 'success' | 'error' | 'default' | 'warning' => {
  const r = rec.toLowerCase()
  if (r.includes('buy')) return 'success'
  if (r.includes('sell')) return 'error'
  return 'default'
}

export const WatchlistPanel: React.FC<WatchlistPanelProps> = ({
  symbols = DEFAULT_SYMBOLS,
  onSymbolClick,
}) => {
  const [localSymbols, setLocalSymbols] = useState(symbols)
  const [data, setData] = useState<TechnicalsResult[]>([])
  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [liveEnabled, setLiveEnabled] = useState(false)
  const [addInput, setAddInput] = useState('')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const results = await technicalsApi.watchlist(localSymbols)
      setData(results)
      setLastUpdate(new Date())
    } catch (e) {
      console.error('Watchlist fetch failed:', e)
    } finally {
      setLoading(false)
    }
  }, [localSymbols])

  useEffect(() => {
    setLocalSymbols(symbols)
  }, [symbols])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (liveEnabled) {
      intervalRef.current = setInterval(fetchData, POLL_INTERVAL)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [liveEnabled, fetchData])

  const handleAdd = () => {
    if (addInput && !localSymbols.includes(addInput.toUpperCase())) {
      setLocalSymbols((prev) => [...prev, addInput.toUpperCase()])
      setAddInput('')
    }
  }

  const handleRemove = (sym: string) => {
    setLocalSymbols((prev) => prev.filter((s) => s !== sym))
  }

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1.5,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            Watchlist
          </Typography>
          {lastUpdate && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: '0.6rem' }}
            >
              Updated {lastUpdate.toLocaleTimeString()}
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
          <input
            value={addInput}
            onChange={(e) => setAddInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            placeholder="Add.."
            style={{
              backgroundColor: '#2a2e39',
              color: '#e5e7eb',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '0.75rem',
              width: '64px',
              border: '1px solid #374151',
              outline: 'none',
            }}
          />
          <IconButton size="small" onClick={handleAdd} disabled={!addInput}>
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              +
            </Typography>
          </IconButton>
          <div className="h-4 w-px bg-gray-700 mx-1" />
          <Tooltip
            title={liveEnabled ? 'Pause live updates' : 'Resume live updates'}
          >
            <IconButton size="small" onClick={() => setLiveEnabled((v) => !v)}>
              {liveEnabled ? (
                <Pause fontSize="small" color="success" />
              ) : (
                <PlayArrow fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh now">
            <IconButton size="small" onClick={fetchData} disabled={loading}>
              {loading ? (
                <CircularProgress size={16} />
              ) : (
                <Refresh fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && data.length === 0 && <LinearProgress />}

      <TableContainer sx={{ flex: 1 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}>
                Symbol
              </TableCell>
              <TableCell
                align="center"
                sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
              >
                Summary
              </TableCell>
              <TableCell
                align="center"
                sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
              >
                Osc
              </TableCell>
              <TableCell
                align="center"
                sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
              >
                MAs
              </TableCell>
              <TableCell
                align="center"
                sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
              ></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((item) => (
              <TableRow
                key={item.symbol}
                hover
                sx={{
                  cursor: onSymbolClick ? 'pointer' : 'default',
                  '&:hover button': { opacity: 1 },
                }}
                onClick={() => onSymbolClick?.(item.symbol)}
              >
                <TableCell
                  sx={{ fontSize: '0.75rem', fontWeight: 600, py: 0.4 }}
                >
                  {item.symbol}
                </TableCell>
                <TableCell align="center" sx={{ py: 0.4 }}>
                  <Chip
                    label={item.summary.recommendation}
                    size="small"
                    color={recChipColor(item.summary.recommendation)}
                    variant="filled"
                    sx={{ fontSize: '0.6rem', height: 20 }}
                  />
                </TableCell>
                <TableCell align="center" sx={{ py: 0.4 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: recColor(item.oscillators.recommendation),
                      fontWeight: 600,
                      fontSize: '0.65rem',
                    }}
                  >
                    {item.oscillators.recommendation}
                  </Typography>
                </TableCell>
                <TableCell align="center" sx={{ py: 0.4 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: recColor(item.moving_averages.recommendation),
                      fontWeight: 600,
                      fontSize: '0.65rem',
                    }}
                  >
                    {item.moving_averages.recommendation}
                  </Typography>
                </TableCell>
                <TableCell align="center" sx={{ py: 0.4, width: 20 }}>
                  <IconButton
                    size="small"
                    sx={{
                      opacity: 0,
                      transition: 'opacity 0.2s',
                      padding: 0.5,
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRemove(item.symbol)
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ fontWeight: 600, color: 'text.secondary' }}
                    >
                      ×
                    </Typography>
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {data.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  <Typography variant="caption" color="text.secondary">
                    No data
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Card>
  )
}
