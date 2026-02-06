import {
  Box,
  Card,
  CircularProgress,
  Divider,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Typography,
} from '@mui/material'
import React, { useEffect, useState } from 'react'

import { technicalsApi } from '../../api/technicals'
import type {
  Action,
  CategorySummary,
  TechnicalsResult,
} from '../../types/technicals'
import { TechnicalsGauge } from './TechnicalsGauge'

interface TechnicalsPanelProps {
  symbol: string
  timeframe: string
}

const actionColor = (action: Action) => {
  if (action === 'Buy') return '#26a69a'
  if (action === 'Sell') return '#ff5252'
  return '#9e9e9e'
}

const IndicatorTable: React.FC<{ category: CategorySummary }> = ({
  category,
}) => (
  <TableContainer sx={{ maxHeight: 300 }}>
    <Table size="small" stickyHeader>
      <TableHead>
        <TableRow>
          <TableCell sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}>
            Name
          </TableCell>
          <TableCell
            align="right"
            sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
          >
            Value
          </TableCell>
          <TableCell
            align="right"
            sx={{ fontSize: '0.7rem', fontWeight: 600, py: 0.5 }}
          >
            Action
          </TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {category.indicators.map((ind) => (
          <TableRow
            key={ind.name}
            sx={{ '&:last-child td': { borderBottom: 0 } }}
          >
            <TableCell sx={{ fontSize: '0.7rem', py: 0.3 }}>
              {ind.name}
            </TableCell>
            <TableCell align="right" sx={{ fontSize: '0.7rem', py: 0.3 }}>
              {typeof ind.value === 'number' ? ind.value.toFixed(2) : '—'}
            </TableCell>
            <TableCell
              align="right"
              sx={{
                fontSize: '0.7rem',
                py: 0.3,
                color: actionColor(ind.action),
                fontWeight: 600,
              }}
            >
              {ind.action}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
)

export const TechnicalsPanel: React.FC<TechnicalsPanelProps> = ({
  symbol,
  timeframe,
}) => {
  const [data, setData] = useState<TechnicalsResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState(0)

  useEffect(() => {
    let cancelled = false
    const fetch = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await technicalsApi.analyze(symbol, timeframe)
        if (!cancelled) setData(result)
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : 'Failed to load'
          setError(msg)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetch()
    return () => {
      cancelled = true
    }
  }, [symbol, timeframe])

  if (loading) {
    return (
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <CircularProgress size={32} />
      </Card>
    )
  }

  if (error || !data) {
    return (
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {error || 'No data'}
        </Typography>
      </Card>
    )
  }

  const categories = [data.oscillators, data.moving_averages]
  const tabLabels = ['Oscillators', 'Moving Averages']

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* 3 Gauges row - evenly distributed */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-evenly',
          alignItems: 'flex-start',
          pt: 1,
          pb: 0.5,
          px: 0.5,
        }}
      >
        <TechnicalsGauge
          title="Oscillators"
          value={data.oscillators.gauge_value}
          recommendation={data.oscillators.recommendation}
          buyCount={data.oscillators.buy_count}
          sellCount={data.oscillators.sell_count}
          neutralCount={data.oscillators.neutral_count}
        />
        <TechnicalsGauge
          title="Summary"
          value={data.summary.gauge_value}
          recommendation={data.summary.recommendation}
          buyCount={data.summary.buy_count}
          sellCount={data.summary.sell_count}
          neutralCount={data.summary.neutral_count}
        />
        <TechnicalsGauge
          title="Moving Avgs"
          value={data.moving_averages.gauge_value}
          recommendation={data.moving_averages.recommendation}
          buyCount={data.moving_averages.buy_count}
          sellCount={data.moving_averages.sell_count}
          neutralCount={data.moving_averages.neutral_count}
        />
      </Box>

      <Divider />

      {/* Tabs for indicator detail tables */}
      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        variant="fullWidth"
        sx={{
          minHeight: 32,
          '& .MuiTab-root': { minHeight: 32, py: 0.5, fontSize: '0.75rem' },
        }}
      >
        {tabLabels.map((l) => (
          <Tab key={l} label={l} />
        ))}
      </Tabs>

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <IndicatorTable category={categories[tab]} />
      </Box>
    </Card>
  )
}
