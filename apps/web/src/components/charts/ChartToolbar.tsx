import { BarChart2, Camera, Maximize2, Settings } from 'lucide-react'
import { type ReactNode, useState } from 'react'

import { IndicatorDialog, type IndicatorSelection } from './IndicatorDialog'
import { MarketClocks } from './MarketClocks'
import { SymbolSearchDialog } from './SymbolSearchDialog'

interface ChartToolbarProps {
  symbol: string
  onSymbolChange: (s: string) => void
  timeframe: string
  onTimeframeChange: (t: string) => void
  onAddIndicator: (ind: IndicatorSelection) => void
  onRemoveIndicator: (name: string) => void
  activeIndicators?: string[]
  /** Extra content to render after indicators (with divider) */
  extraActions?: ReactNode
}

const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

export const ChartToolbar = ({
  symbol,
  onSymbolChange,
  timeframe,
  onTimeframeChange,
  onAddIndicator,
  onRemoveIndicator,
  activeIndicators = [],
  extraActions,
}: ChartToolbarProps) => {
  const [symbolDialogOpen, setSymbolDialogOpen] = useState(false)
  const [indicatorDialogOpen, setIndicatorDialogOpen] = useState(false)

  return (
    <>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 16px',
          borderBottom: '1px solid #1f2937',
          backgroundColor: '#131722',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Symbol Search */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setSymbolDialogOpen(true)}
              style={{
                backgroundColor: '#2a2e39',
                color: '#e5e7eb',
                padding: '6px 12px',
                borderRadius: '4px',
                fontSize: '0.875rem',
                fontWeight: 500,
                width: '128px',
                textAlign: 'left',
                border: 'none',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#363a45')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#2a2e39')}
            >
              {symbol}
            </button>
          </div>

          <div style={{ height: '24px', width: '1px', backgroundColor: '#1f2937' }} />

          {/* Timeframes */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => onTimeframeChange(tf)}
                style={{
                  padding: '4px 8px',
                  fontSize: '0.75rem',
                  fontWeight: 500,
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'color 0.2s, background-color 0.2s',
                  color: timeframe === tf ? '#60a5fa' : '#9ca3af',
                  backgroundColor: timeframe === tf ? 'rgba(96, 165, 250, 0.1)' : 'transparent',
                }}
                onMouseEnter={(e) => {
                  if (timeframe !== tf) {
                    e.currentTarget.style.color = '#e5e7eb'
                    e.currentTarget.style.backgroundColor = '#2a2e39'
                  }
                }}
                onMouseLeave={(e) => {
                  if (timeframe !== tf) {
                    e.currentTarget.style.color = '#9ca3af'
                    e.currentTarget.style.backgroundColor = 'transparent'
                  }
                }}
              >
                {tf.toUpperCase()}
              </button>
            ))}
          </div>

          <div style={{ height: '24px', width: '1px', backgroundColor: '#1f2937' }} />

          {/* Indicators */}
          <button
            onClick={() => setIndicatorDialogOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              fontSize: '0.875rem',
              fontWeight: 500,
              color: '#d1d4dc',
              backgroundColor: 'transparent',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'color 0.2s, background-color 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#fff'
              e.currentTarget.style.backgroundColor = '#2a2e39'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#d1d4dc'
              e.currentTarget.style.backgroundColor = 'transparent'
            }}
          >
            <BarChart2 size={16} />
            <span>Indicators</span>
            {activeIndicators.length > 0 && (
              <span
                style={{
                  backgroundColor: 'rgba(59, 130, 246, 0.2)',
                  color: '#60a5fa',
                  fontSize: '10px',
                  padding: '0 6px',
                  borderRadius: '9999px',
                }}
              >
                {activeIndicators.length}
              </span>
            )}
          </button>

          {/* Extra Actions (e.g., Analysis Panel) */}
          {extraActions && (
            <>
              <div style={{ height: '24px', width: '1px', backgroundColor: '#1f2937' }} />
              {extraActions}
            </>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MarketClocks />
          {[
            { Icon: Settings, color: '#9ca3af' },
            { Icon: Camera, color: '#9ca3af' },
            { Icon: Maximize2, color: '#60a5fa' },
          ].map(({ Icon, color }, i) => (
            <button
              key={i}
              style={{
                padding: '8px',
                color,
                backgroundColor: 'transparent',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'color 0.2s, background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = i === 2 ? color : '#e5e7eb'
                e.currentTarget.style.backgroundColor = i === 2 ? 'rgba(96, 165, 250, 0.1)' : '#2a2e39'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = color
                e.currentTarget.style.backgroundColor = 'transparent'
              }}
            >
              <Icon size={18} />
            </button>
          ))}
        </div>
      </div>

      <SymbolSearchDialog
        open={symbolDialogOpen}
        onClose={() => setSymbolDialogOpen(false)}
        onSelect={onSymbolChange}
      />
      <IndicatorDialog
        open={indicatorDialogOpen}
        onClose={() => setIndicatorDialogOpen(false)}
        onAdd={onAddIndicator}
        onRemove={onRemoveIndicator}
        activeIndicators={activeIndicators}
      />
    </>
  )
}
