import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'

import { strategiesApi } from '../api/strategies'
import type { BacktestResult, StrategyInfo } from '../types/strategy'

const MetricCard = ({
  label,
  value,
  format,
}: {
  label: string
  value: number
  format?: string
}) => {
  let display: string
  if (format === 'pct') display = `${(value * 100).toFixed(2)}%`
  else if (format === 'ratio') display = value.toFixed(3)
  else if (format === 'int') display = value.toString()
  else display = value.toFixed(4)

  const color =
    value > 0 ? 'success.main' : value < 0 ? 'error.main' : 'text.primary'

  return (
    <Paper sx={{ p: 2, textAlign: 'center' }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" sx={{ color }}>
        {display}
      </Typography>
    </Paper>
  )
}

export const BacktestLab = () => {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([])
  const [strategyId, setStrategyId] = useState('')
  const [symbol, setSymbol] = useState('AAPL')
  const [startDate, setStartDate] = useState('2022-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')
  const [initialCash, setInitialCash] = useState(100000)
  const [feeBps, setFeeBps] = useState(10)
  const [takeProfitPct, setTakeProfitPct] = useState<string>('')
  const [stopLossPct, setStopLossPct] = useState<string>('')
  const [trailingStopPct, setTrailingStopPct] = useState<string>('')
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    strategiesApi.list().then((data) => {
      setStrategies(data)
      if (data.length > 0) setStrategyId(data[0].id)
    })
  }, [])

  const runBacktest = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const params: Record<string, any> = {}
      if (takeProfitPct) params.take_profit_pct = parseFloat(takeProfitPct)
      if (stopLossPct) params.stop_loss_pct = parseFloat(stopLossPct)
      if (trailingStopPct)
        params.trailing_stop_pct = parseFloat(trailingStopPct)

      const res = await strategiesApi.backtest({
        symbol,
        strategy_id: strategyId,
        params: Object.keys(params).length > 0 ? params : undefined,
        start: startDate,
        end: endDate,
        initial_cash: initialCash,
        fee_bps: feeBps,
      })
      setResult(res)
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Backtest Lab
      </Typography>

      {/* Config */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12, sm: 3 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Strategy</InputLabel>
              <Select
                value={strategyId}
                label="Strategy"
                onChange={(e) => setStrategyId(e.target.value)}
              >
                {strategies.map((s) => (
                  <MenuItem key={s.id} value={s.id}>
                    {s.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 6, sm: 2 }}>
            <TextField
              label="Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              size="small"
              fullWidth
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 2 }}>
            <TextField
              label="Start"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              size="small"
              fullWidth
              slotProps={{ inputLabel: { shrink: true } }}
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 2 }}>
            <TextField
              label="End"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              size="small"
              fullWidth
              slotProps={{ inputLabel: { shrink: true } }}
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 1 }}>
            <TextField
              label="Cash"
              type="number"
              value={initialCash}
              onChange={(e) => setInitialCash(+e.target.value)}
              size="small"
              fullWidth
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 1 }}>
            <TextField
              label="Fee (bps)"
              type="number"
              value={feeBps}
              onChange={(e) => setFeeBps(+e.target.value)}
              size="small"
              fullWidth
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 1 }}>
            <Button
              variant="contained"
              onClick={runBacktest}
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Run'}
            </Button>
          </Grid>
        </Grid>

        {/* TP/SL Row */}
        <Grid container spacing={2} alignItems="center" sx={{ mt: 1 }}>
          <Grid size={{ xs: 12, sm: 2 }}>
            <TextField
              label="Take Profit %"
              type="number"
              value={takeProfitPct}
              onChange={(e) => setTakeProfitPct(e.target.value)}
              size="small"
              fullWidth
              placeholder="e.g. 5"
              inputProps={{ step: 0.5, min: 0 }}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 2 }}>
            <TextField
              label="Stop Loss %"
              type="number"
              value={stopLossPct}
              onChange={(e) => setStopLossPct(e.target.value)}
              size="small"
              fullWidth
              placeholder="e.g. 3"
              inputProps={{ step: 0.5, min: 0 }}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 2 }}>
            <TextField
              label="Trailing Stop %"
              type="number"
              value={trailingStopPct}
              onChange={(e) => setTrailingStopPct(e.target.value)}
              size="small"
              fullWidth
              placeholder="e.g. 2"
              inputProps={{ step: 0.5, min: 0 }}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <Typography variant="caption" color="text.secondary">
              Leave blank to disable. TP/SL override strategy exit signals.
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Metrics Grid */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard label="CAGR" value={result.cagr} format="pct" />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Total Return"
                value={result.total_return}
                format="pct"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Sharpe"
                value={result.sharpe_ratio}
                format="ratio"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Sortino"
                value={result.sortino_ratio}
                format="ratio"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Max DD"
                value={-result.max_drawdown}
                format="pct"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Win Rate"
                value={result.win_rate}
                format="pct"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Volatility"
                value={result.volatility}
                format="pct"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Calmar"
                value={result.calmar_ratio}
                format="ratio"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Trades"
                value={result.total_trades}
                format="int"
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 3, md: 2 }}>
              <MetricCard
                label="Exposure"
                value={result.exposure}
                format="pct"
              />
            </Grid>
          </Grid>

          {/* Equity Curve (simple table-based for now) */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Equity Curve
            </Typography>
            <Box sx={{ height: 200, overflow: 'hidden', position: 'relative' }}>
              <svg
                width="100%"
                height="100%"
                viewBox={`0 0 ${result.equity_curve.length} 100`}
                preserveAspectRatio="none"
              >
                {(() => {
                  const vals = result.equity_curve.map((p) => p.value)
                  const min = Math.min(...vals)
                  const max = Math.max(...vals)
                  const range = max - min || 1
                  const points = vals
                    .map((v, i) => `${i},${100 - ((v - min) / range) * 100}`)
                    .join(' ')
                  return (
                    <polyline
                      points={points}
                      fill="none"
                      stroke="#2962FF"
                      strokeWidth="0.5"
                    />
                  )
                })()}
              </svg>
              <Box sx={{ position: 'absolute', top: 4, left: 8 }}>
                <Typography variant="caption" color="text.secondary">
                  ${result.equity_curve[0]?.value.toLocaleString()} → $
                  {result.equity_curve[
                    result.equity_curve.length - 1
                  ]?.value.toLocaleString()}
                </Typography>
              </Box>
            </Box>
          </Paper>

          {/* Trades Table */}
          {result.trades.length > 0 && (
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Trades ({result.trades.length})
              </Typography>
              <TableContainer sx={{ maxHeight: 300 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Entry</TableCell>
                      <TableCell>Exit</TableCell>
                      <TableCell align="right">Entry $</TableCell>
                      <TableCell align="right">Exit $</TableCell>
                      <TableCell align="right">Qty</TableCell>
                      <TableCell align="right">PnL</TableCell>
                      <TableCell align="right">Return</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.trades.map((t, i) => (
                      <TableRow key={i}>
                        <TableCell>{t.entry_date?.slice(0, 10)}</TableCell>
                        <TableCell>
                          {t.exit_date?.slice(0, 10) || '—'}
                        </TableCell>
                        <TableCell align="right">
                          ${t.entry_price.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          {t.exit_price ? `$${t.exit_price.toFixed(2)}` : '—'}
                        </TableCell>
                        <TableCell align="right">
                          {t.quantity.toFixed(2)}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            color: t.pnl >= 0 ? 'success.main' : 'error.main',
                          }}
                        >
                          ${t.pnl.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${(t.return_pct * 100).toFixed(2)}%`}
                            size="small"
                            color={t.return_pct >= 0 ? 'success' : 'error'}
                            variant="outlined"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}
        </>
      )}
    </Box>
  )
}
