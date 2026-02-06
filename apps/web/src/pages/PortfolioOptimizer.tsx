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
import { useState } from 'react'

import { strategiesApi } from '../api/strategies'
import type { PortfolioOptimizationResult } from '../types/strategy'

export const PortfolioOptimizer = () => {
  const [symbolsInput, setSymbolsInput] = useState(
    'AAPL, MSFT, GOOGL, AMZN, TSLA',
  )
  const [method, setMethod] = useState('mean_variance')
  const [riskAversion, setRiskAversion] = useState(1.0)
  const [startDate, setStartDate] = useState('2022-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')
  const [result, setResult] = useState<PortfolioOptimizationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runOptimize = async () => {
    const symbols = symbolsInput
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean)
    if (symbols.length < 2) {
      setError('Need at least 2 symbols')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await strategiesApi.optimize({
        symbols,
        method,
        start: startDate,
        end: endDate,
        risk_aversion: riskAversion,
      })
      setResult(res)
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Optimization failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ p: 3, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Portfolio Optimizer
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12 }}>
            <TextField
              label="Symbols (comma-separated)"
              value={symbolsInput}
              onChange={(e) => setSymbolsInput(e.target.value)}
              fullWidth
              size="small"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 3 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Method</InputLabel>
              <Select
                value={method}
                label="Method"
                onChange={(e) => setMethod(e.target.value)}
              >
                <MenuItem value="mean_variance">Mean-Variance</MenuItem>
                <MenuItem value="risk_parity">Risk Parity</MenuItem>
              </Select>
            </FormControl>
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
          {method === 'mean_variance' && (
            <Grid size={{ xs: 6, sm: 2 }}>
              <TextField
                label="Risk Aversion"
                type="number"
                value={riskAversion}
                onChange={(e) => setRiskAversion(+e.target.value)}
                size="small"
                fullWidth
                inputProps={{ step: 0.5, min: 0.1 }}
              />
            </Grid>
          )}
          <Grid size={{ xs: 6, sm: 2 }}>
            <Button
              variant="contained"
              onClick={runOptimize}
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Optimize'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <>
          {/* Summary metrics */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Expected Return
                </Typography>
                <Typography variant="h6">
                  {(result.expected_return * 100).toFixed(2)}%
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Volatility
                </Typography>
                <Typography variant="h6">
                  {(result.portfolio_volatility * 100).toFixed(2)}%
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Sharpe Ratio
                </Typography>
                <Typography variant="h6">
                  {result.sharpe_ratio.toFixed(3)}
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Method
                </Typography>
                <Typography variant="h6">
                  <Chip
                    label={
                      result.method === 'mean_variance'
                        ? 'Mean-Variance'
                        : 'Risk Parity'
                    }
                    color="primary"
                  />
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Weights & Risk Contributions */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Allocation
            </Typography>

            {/* Weight bars */}
            <Box sx={{ mb: 3 }}>
              {Object.entries(result.weights)
                .sort(([, a], [, b]) => b - a)
                .map(([sym, w]) => (
                  <Box
                    key={sym}
                    sx={{ display: 'flex', alignItems: 'center', mb: 1 }}
                  >
                    <Typography
                      sx={{
                        width: 60,
                        fontWeight: 'bold',
                        fontSize: '0.85rem',
                      }}
                    >
                      {sym}
                    </Typography>
                    <Box sx={{ flex: 1, mx: 1 }}>
                      <Box
                        sx={{
                          height: 20,
                          width: `${(w * 100).toFixed(1)}%`,
                          bgcolor: 'primary.main',
                          borderRadius: 1,
                          minWidth: 4,
                        }}
                      />
                    </Box>
                    <Typography
                      sx={{
                        width: 60,
                        textAlign: 'right',
                        fontSize: '0.85rem',
                      }}
                    >
                      {(w * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                ))}
            </Box>

            {/* Table */}
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Weight</TableCell>
                    <TableCell align="right">Risk Contribution</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(result.weights)
                    .sort(([, a], [, b]) => b - a)
                    .map(([sym, w]) => (
                      <TableRow key={sym}>
                        <TableCell>{sym}</TableCell>
                        <TableCell align="right">
                          {(w * 100).toFixed(2)}%
                        </TableCell>
                        <TableCell align="right">
                          {(
                            (result.risk_contributions[sym] || 0) * 100
                          ).toFixed(2)}
                          %
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </>
      )}
    </Box>
  )
}
