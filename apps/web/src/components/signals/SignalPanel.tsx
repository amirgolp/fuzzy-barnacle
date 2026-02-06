import { Remove, Speed, TrendingDown, TrendingUp } from '@mui/icons-material'
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material'
import React from 'react'

import type { ScreenerResult, TechnicalSignal } from '../../types/screener'
import { SignalGauge } from './SignalGauge'

interface SignalPanelProps {
  result: ScreenerResult | null
  loading?: boolean
}

const getSignalIcon = (signal: TechnicalSignal) => {
  if (signal.is_bullish) return <TrendingUp color="success" />
  if (signal.is_bearish) return <TrendingDown color="error" />
  return <Remove color="warning" />
}

const getStrengthColor = (strength: string) => {
  switch (strength) {
    case 'strong':
      return 'success'
    case 'moderate':
      return 'info'
    case 'weak':
      return 'default'
    default:
      return 'default'
  }
}

export const SignalPanel: React.FC<SignalPanelProps> = ({
  result,
  loading,
}) => {
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
        <Typography>Loading analysis...</Typography>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              opacity: 0.5,
              py: 4,
            }}
          >
            <Speed sx={{ fontSize: 48, mb: 2 }} />
            <Typography>No analysis data available</Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'auto',
      }}
    >
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Technical Analysis</Typography>
        <Typography variant="caption" color="text.secondary">
          Based on {result.signals.length} detected signal
          {result.signals.length !== 1 ? 's' : ''}
        </Typography>
      </Box>

      {/* Gauge section */}
      <Box
        sx={{
          px: 2,
          pt: 2,
          pb: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <SignalGauge
          value={result.gauge_value}
          recommendation={result.recommendation}
          label={`${result.gauge_label} (${result.score.toFixed(1)})`}
        />

        <Stack
          direction="row"
          spacing={3}
          sx={{ mt: 1, justifyContent: 'center' }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h5" color="success.main">
              {result.bullish_score}
            </Typography>
            <Typography variant="caption">Bullish</Typography>
          </Box>
          <Divider orientation="vertical" flexItem />
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h5" color="error.main">
              {result.bearish_score}
            </Typography>
            <Typography variant="caption">Bearish</Typography>
          </Box>
        </Stack>
      </Box>

      <Divider sx={{ mx: 2 }} />

      {/* Signals list */}
      <List disablePadding sx={{ flex: 1, overflowY: 'auto' }}>
        {result.signals.length === 0 ? (
          <ListItem>
            <ListItemText
              primary="No signals detected"
              secondary="Try a different timeframe"
            />
          </ListItem>
        ) : (
          result.signals.map((signal, idx) => (
            <React.Fragment key={idx}>
              <ListItem sx={{ py: 1 }}>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  {getSignalIcon(signal)}
                </ListItemIcon>
                <ListItemText
                  primary={signal.signal_type.replace(/_/g, ' ')}
                  secondary={signal.description}
                  primaryTypographyProps={{
                    variant: 'body2',
                    sx: { textTransform: 'capitalize', fontWeight: 500 },
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    noWrap: true,
                  }}
                />
                <Chip
                  label={signal.strength}
                  size="small"
                  color={getStrengthColor(signal.strength) as any}
                  variant="outlined"
                  sx={{ textTransform: 'capitalize', ml: 0.5 }}
                />
              </ListItem>
              {idx < result.signals.length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))
        )}
      </List>

      {/* Notes */}
      {result.notes.length > 0 && (
        <Box sx={{ p: 1 }}>
          {result.notes.map((note, i) => (
            <Alert
              severity="info"
              key={i}
              sx={{
                mb: 0.5,
                py: 0,
                '& .MuiAlert-message': { fontSize: '0.75rem' },
              }}
            >
              {note}
            </Alert>
          ))}
        </Box>
      )}
    </Card>
  )
}
