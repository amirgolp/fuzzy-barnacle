import { Box, Typography, useTheme } from '@mui/material'
import React from 'react'

interface TechnicalsGaugeProps {
  value: number // -100 to +100
  recommendation: string
  buyCount: number
  sellCount: number
  neutralCount: number
  title: string
  /** Compact mode for smaller displays */
  compact?: boolean
}

export const TechnicalsGauge: React.FC<TechnicalsGaugeProps> = ({
  value,
  recommendation,
  buyCount,
  sellCount,
  neutralCount,
  title,
  compact = false,
}) => {
  const theme = useTheme()
  const isDark = theme.palette.mode === 'dark'

  const clamped = Math.max(-100, Math.min(100, value))
  // Map -100..+100 to π..0 (left to right semicircle)
  const angleRad = (Math.PI * (100 - clamped)) / 200
  const cx = 50,
    cy = 45,
    r = 35
  const nx = cx + r * Math.cos(angleRad)
  const ny = cy - r * Math.sin(angleRad)

  // 5 segments: strong sell, sell, neutral, buy, strong buy
  const colors = ['#f44336', '#ff5252', '#9e9e9e', '#66bb6a', '#26a69a']
  const segAngles = [
    [Math.PI, Math.PI * 0.8],
    [Math.PI * 0.8, Math.PI * 0.6],
    [Math.PI * 0.6, Math.PI * 0.4],
    [Math.PI * 0.4, Math.PI * 0.2],
    [Math.PI * 0.2, 0],
  ]

  const arcPath = (startA: number, endA: number) => {
    const x1 = cx + r * Math.cos(startA)
    const y1 = cy - r * Math.sin(startA)
    const x2 = cx + r * Math.cos(endA)
    const y2 = cy - r * Math.sin(endA)
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`
  }

  let recColor = theme.palette.text.secondary
  const recLower = recommendation.toLowerCase()
  if (recLower.includes('strong buy')) recColor = '#26a69a'
  else if (recLower.includes('buy')) recColor = '#66bb6a'
  else if (recLower.includes('strong sell')) recColor = '#f44336'
  else if (recLower.includes('sell')) recColor = '#ff5252'

  const svgWidth = compact ? 120 : 180
  const svgHeight = compact ? 70 : 100

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        minWidth: compact ? 120 : 180,
      }}
    >
      <Typography
        variant="caption"
        sx={{
          fontWeight: 600,
          mb: 0.25,
          color: 'text.secondary',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          fontSize: compact ? '0.65rem' : '0.8rem',
          lineHeight: 1,
        }}
      >
        {title}
      </Typography>
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox="0 0 100 60"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Segments */}
        {segAngles.map(([s, e], i) => (
          <path
            key={i}
            d={arcPath(s, e)}
            fill="none"
            stroke={colors[i]}
            strokeWidth="6"
            strokeLinecap="butt"
            opacity={0.8}
          />
        ))}
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={nx}
          y2={ny}
          stroke={recColor}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r="2.5" fill={recColor} />
        {/* Labels at edges */}
        <text
          x="10"
          y="56"
          fontSize="6"
          fill={isDark ? '#888' : '#666'}
          textAnchor="middle"
        >
          Sell
        </text>
        <text
          x="50"
          y="8"
          fontSize="6"
          fill={isDark ? '#888' : '#666'}
          textAnchor="middle"
        >
          Neutral
        </text>
        <text
          x="90"
          y="56"
          fontSize="6"
          fill={isDark ? '#888' : '#666'}
          textAnchor="middle"
        >
          Buy
        </text>
      </svg>
      <Typography
        sx={{
          color: recColor,
          fontWeight: 700,
          mt: -0.25,
          fontSize: compact ? '0.75rem' : '0.9rem',
          lineHeight: 1.2,
        }}
      >
        {recommendation}
      </Typography>
      {/* Counts row */}
      <Box sx={{ display: 'flex', gap: compact ? 1 : 1.5, mt: 0.25 }}>
        <Box sx={{ textAlign: 'center' }}>
          <Typography
            sx={{
              fontWeight: 700,
              color: '#ff5252',
              fontSize: compact ? '0.7rem' : '0.8rem',
              lineHeight: 1,
            }}
          >
            {sellCount}
          </Typography>
          <Typography
            sx={{
              fontSize: compact ? '0.55rem' : '0.6rem',
              color: 'text.secondary',
              lineHeight: 1,
            }}
          >
            Sell
          </Typography>
        </Box>
        <Box sx={{ textAlign: 'center' }}>
          <Typography
            sx={{
              fontWeight: 700,
              color: 'text.secondary',
              fontSize: compact ? '0.7rem' : '0.8rem',
              lineHeight: 1,
            }}
          >
            {neutralCount}
          </Typography>
          <Typography
            sx={{
              fontSize: compact ? '0.55rem' : '0.6rem',
              color: 'text.secondary',
              lineHeight: 1,
            }}
          >
            Neutral
          </Typography>
        </Box>
        <Box sx={{ textAlign: 'center' }}>
          <Typography
            sx={{
              fontWeight: 700,
              color: '#26a69a',
              fontSize: compact ? '0.7rem' : '0.8rem',
              lineHeight: 1,
            }}
          >
            {buyCount}
          </Typography>
          <Typography
            sx={{
              fontSize: compact ? '0.55rem' : '0.6rem',
              color: 'text.secondary',
              lineHeight: 1,
            }}
          >
            Buy
          </Typography>
        </Box>
      </Box>
    </Box>
  )
}
