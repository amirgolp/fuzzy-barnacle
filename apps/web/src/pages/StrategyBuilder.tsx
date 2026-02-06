import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  FormControl,
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
import type { StrategyInfo } from '../types/strategy'

export const StrategyBuilder = () => {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([])
  const [selected, setSelected] = useState<StrategyInfo | null>(null)
  const [params, setParams] = useState<Record<string, any>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    strategiesApi
      .list()
      .then((data) => {
        setStrategies(data)
        if (data.length > 0) {
          setSelected(data[0])
          setParams({ ...data[0].default_params })
        }
      })
      .catch(() => setError('Failed to load strategies'))
      .finally(() => setLoading(false))
  }, [])

  const handleSelect = (id: string) => {
    const s = strategies.find((s) => s.id === id)
    if (s) {
      setSelected(s)
      setParams({ ...s.default_params })
    }
  }

  if (loading)
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    )
  if (error)
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    )

  return (
    <Box sx={{ p: 3, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Strategy Builder
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Strategy</InputLabel>
          <Select
            value={selected?.id || ''}
            label="Strategy"
            onChange={(e) => handleSelect(e.target.value)}
          >
            {strategies.map((s) => (
              <MenuItem key={s.id} value={s.id}>
                {s.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {selected && (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {selected.description}
            </Typography>

            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Parameters
            </Typography>
            {Object.entries(params).map(([key, value]) => (
              <TextField
                key={key}
                label={key}
                type="number"
                value={value}
                onChange={(e) =>
                  setParams({
                    ...params,
                    [key]: parseFloat(e.target.value) || 0,
                  })
                }
                size="small"
                sx={{ mr: 2, mb: 1, width: 150 }}
              />
            ))}
          </>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Available Strategies
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Parameters</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {strategies.map((s) => (
                <TableRow
                  key={s.id}
                  hover
                  onClick={() => handleSelect(s.id)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell>{s.name}</TableCell>
                  <TableCell>
                    <Chip label={s.id} size="small" />
                  </TableCell>
                  <TableCell>
                    {Object.entries(s.default_params).map(([k, v]) => (
                      <Chip
                        key={k}
                        label={`${k}=${v}`}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 0.5 }}
                      />
                    ))}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  )
}
