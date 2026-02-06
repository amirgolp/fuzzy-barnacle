import {
  Box,
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { useState } from 'react'

import { MarketIntel } from '../components/signals/MarketIntel'

// Predefined symbol universes
const UNIVERSES: Record<string, string[]> = {
  'Big Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
  Finance: ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA'],
  Semiconductors: [
    'NVDA',
    'AMD',
    'INTC',
    'TSM',
    'QCOM',
    'TXN',
    'MU',
    'AVGO',
    'ASML',
  ],
  'Indices (ETFs)': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'TLT', 'HYG', 'LQD'],
  'Precious Metals': [
    'GLD',
    'SLV',
    'IAU',
    'PPLT',
    'PALL',
    'GDX',
    'GDXJ',
    'SIL',
  ],
  Energy: ['USO', 'UNG', 'XLE', 'OIH', 'XOP', 'UCO', 'BOIL'],
  Commodities: ['DBA', 'DBC', 'GSG', 'PDBC', 'COPX', 'CPER', 'WEAT', 'CORN'],
  Crypto: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ADA-USD', 'DOT-USD'],
}

export function Screener() {
  const [selectedUniverse, setSelectedUniverse] = useState<string>('Big Tech')
  const symbols = UNIVERSES[selectedUniverse] || []

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography
          variant="h4"
          gutterBottom
          component="div"
          sx={{ fontWeight: 'bold' }}
        >
          Market Screener
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Scan for intraweek trading opportunities. Expand any row for full
          indicator detail and strategy recommendations.
        </Typography>
      </Box>

      <Paper sx={{ p: 3, mb: 4 }}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={3}
          alignItems="center"
        >
          <Box sx={{ flexGrow: 1, width: '100%' }}>
            <FormControl fullWidth>
              <InputLabel id="universe-select-label">
                Select Universe
              </InputLabel>
              <Select
                labelId="universe-select-label"
                value={selectedUniverse}
                label="Select Universe"
                onChange={(e) => setSelectedUniverse(e.target.value)}
              >
                {Object.keys(UNIVERSES).map((name) => (
                  <MenuItem key={name} value={name}>
                    {name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Stack>
      </Paper>

      <Paper sx={{ minHeight: 400 }}>
        <MarketIntel key={selectedUniverse} symbols={symbols} />
      </Paper>
    </Container>
  )
}
