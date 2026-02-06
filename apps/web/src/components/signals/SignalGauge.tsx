import { Box, Typography, useTheme } from '@mui/material'
import React from 'react'

interface SignalGaugeProps {
  value: number // -100 to +100 (gauge_value from backend)
  recommendation: string
  label?: string
}

export const SignalGauge: React.FC<SignalGaugeProps> = ({
  value,
  recommendation,
  label,
}) => {
  const theme = useTheme()

  // Clamp value to -100..+100
  // Map to angle on semicircle: -100 → π (left), 0 → π/2 (up), +100 → 0 (right)
  const clamped = Math.max(-100, Math.min(100, value))
  const angleRad = (Math.PI * (100 - clamped)) / 200

  // Needle endpoint (pivot at 110,110, length 80)
  // In SVG, x+ is right, y+ is down, so we negate sin for upward arc
  const nx = 110 + 80 * Math.cos(angleRad)
  const ny = 110 - 80 * Math.sin(angleRad)

  let statusColor = theme.palette.text.secondary
  if (recommendation.includes('buy')) statusColor = theme.palette.success.main
  if (recommendation.includes('sell')) statusColor = theme.palette.error.main

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: '100%',
      }}
    >
      <svg width="100%" viewBox="0 0 220 130" style={{ maxWidth: 220 }}>
        {/* Background arc */}
        <path
          d="M 10 110 A 100 100 0 0 1 210 110"
          fill="none"
          stroke={theme.palette.divider}
          strokeWidth="14"
          strokeLinecap="round"
        />

        {/* Colored segments */}
        <path
          d="M 10 110 A 100 100 0 0 1 50 50"
          fill="none"
          stroke={theme.palette.error.main}
          strokeWidth="14"
          strokeLinecap="round"
          opacity="0.35"
        />
        <path
          d="M 50 50 A 100 100 0 0 1 80 25"
          fill="none"
          stroke={theme.palette.error.light}
          strokeWidth="14"
          opacity="0.25"
        />
        <path
          d="M 80 25 A 100 100 0 0 1 140 25"
          fill="none"
          stroke={theme.palette.warning.main}
          strokeWidth="14"
          opacity="0.25"
        />
        <path
          d="M 140 25 A 100 100 0 0 1 170 50"
          fill="none"
          stroke={theme.palette.success.light}
          strokeWidth="14"
          opacity="0.25"
        />
        <path
          d="M 170 50 A 100 100 0 0 1 210 110"
          fill="none"
          stroke={theme.palette.success.main}
          strokeWidth="14"
          strokeLinecap="round"
          opacity="0.35"
        />

        {/* Needle */}
        <line
          x1="110"
          y1="110"
          x2={nx}
          y2={ny}
          stroke={statusColor}
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx="110" cy="110" r="5" fill={statusColor} />

        {/* Labels */}
        <text
          x="10"
          y="128"
          fontSize="9"
          fill={theme.palette.error.main}
          textAnchor="start"
        >
          Strong Sell
        </text>
        <text
          x="110"
          y="128"
          fontSize="9"
          fill={theme.palette.text.secondary}
          textAnchor="middle"
        >
          Neutral
        </text>
        <text
          x="210"
          y="128"
          fontSize="9"
          fill={theme.palette.success.main}
          textAnchor="end"
        >
          Strong Buy
        </text>
      </svg>

      <Typography
        variant="subtitle1"
        sx={{
          color: statusColor,
          textTransform: 'uppercase',
          letterSpacing: 1,
          mt: 0.5,
          fontWeight: 700,
        }}
      >
        {recommendation.replace('_', ' ')}
      </Typography>
      {label && (
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
      )}
    </Box>
  )
}
