import { Box, Chip, Typography } from '@mui/material'
import React, { useEffect, useState } from 'react'

interface Market {
  name: string
  timezone: string
  openHour: number
  openMin: number
  closeHour: number
  closeMin: number
  flag: string
}

const MARKETS: Market[] = [
  {
    name: 'NYC',
    timezone: 'America/New_York',
    openHour: 9,
    openMin: 30,
    closeHour: 16,
    closeMin: 0,
    flag: '🇺🇸',
  },
  {
    name: 'London',
    timezone: 'Europe/London',
    openHour: 8,
    openMin: 0,
    closeHour: 16,
    closeMin: 30,
    flag: '🇬🇧',
  },
  {
    name: 'Frankfurt',
    timezone: 'Europe/Berlin',
    openHour: 9,
    openMin: 0,
    closeHour: 17,
    closeMin: 30,
    flag: '🇩🇪',
  },
]

function getMarketTime(timezone: string): Date {
  return new Date(new Date().toLocaleString('en-US', { timeZone: timezone }))
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function isWeekday(d: Date): boolean {
  const day = d.getDay()
  return day !== 0 && day !== 6
}

function isOpen(d: Date, market: Market): boolean {
  if (!isWeekday(d)) return false
  const mins = d.getHours() * 60 + d.getMinutes()
  const openMins = market.openHour * 60 + market.openMin
  const closeMins = market.closeHour * 60 + market.closeMin
  return mins >= openMins && mins < closeMins
}

function getCountdown(d: Date, market: Market): string {
  const mins = d.getHours() * 60 + d.getMinutes()
  const secs = d.getSeconds()
  const openMins = market.openHour * 60 + market.openMin
  const closeMins = market.closeHour * 60 + market.closeMin

  let targetMins: number
  let prefix: string

  if (!isWeekday(d)) {
    // Weekend — show time until Monday open
    const daysUntilMon = d.getDay() === 0 ? 1 : 8 - d.getDay()
    const totalSecs = daysUntilMon * 86400 - (mins * 60 + secs) + openMins * 60
    return `Opens ${formatCountdownSecs(totalSecs)}`
  }

  if (mins < openMins) {
    // Before open
    targetMins = openMins
    prefix = 'Opens'
  } else if (mins < closeMins) {
    // Currently open — countdown to close
    targetMins = closeMins
    prefix = 'Closes'
  } else {
    // After close — show next open (tomorrow or Monday)
    const day = d.getDay()
    const daysUntil = day === 5 ? 3 : 1 // Friday → Monday, else tomorrow
    const totalSecs = daysUntil * 86400 - (mins * 60 + secs) + openMins * 60
    return `Opens ${formatCountdownSecs(totalSecs)}`
  }

  const diffSecs = (targetMins - mins) * 60 - secs
  return `${prefix} ${formatCountdownSecs(diffSecs)}`
}

function formatCountdownSecs(totalSecs: number): string {
  if (totalSecs < 0) totalSecs = 0
  const h = Math.floor(totalSecs / 3600)
  const m = Math.floor((totalSecs % 3600) / 60)
  const s = totalSecs % 60
  if (h > 24) {
    const days = Math.floor(h / 24)
    const remH = h % 24
    return `${days}d ${remH}h`
  }
  return `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`
}

export const MarketClocks: React.FC = () => {
  const [, setTick] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
      {MARKETS.map((market) => {
        const time = getMarketTime(market.timezone)
        const open = isOpen(time, market)
        const countdown = getCountdown(time, market)
        return (
          <Box
            key={market.name}
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
          >
            <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
              {market.flag}
            </Typography>
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                >
                  {market.name}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}
                >
                  {formatTime(time)}
                </Typography>
                <Chip
                  label={open ? 'OPEN' : 'CLOSED'}
                  size="small"
                  color={open ? 'success' : 'default'}
                  variant={open ? 'filled' : 'outlined'}
                  sx={{
                    height: 16,
                    fontSize: '0.55rem',
                    '& .MuiChip-label': { px: 0.5 },
                  }}
                />
              </Box>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.6rem',
                  color: open ? 'warning.main' : 'text.secondary',
                }}
              >
                {countdown}
              </Typography>
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}
