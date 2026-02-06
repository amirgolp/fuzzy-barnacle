import CloseIcon from '@mui/icons-material/Close'
import SettingsIcon from '@mui/icons-material/Settings'
import { IconButton } from '@mui/material'
import React from 'react'

interface LegendItem {
  name: string
  params?: Record<string, any>
  color: string
}

interface IndicatorLegendProps {
  indicators: LegendItem[]
  onEdit: (name: string) => void
  onRemove: (name: string) => void
}

export const IndicatorLegend: React.FC<IndicatorLegendProps> = ({
  indicators,
  onEdit,
  onRemove,
}) => {
  if (indicators.length === 0) return null

  return (
    <div
      style={{
        position: 'absolute',
        top: '8px',
        left: '8px',
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        pointerEvents: 'none',
      }}
    >
      {indicators.map((ind) => (
        <div
          key={ind.name}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            pointerEvents: 'auto',
            backgroundColor: 'rgba(19, 23, 34, 0.8)',
            backdropFilter: 'blur(4px)',
            padding: '4px 8px',
            borderRadius: '4px',
            border: '1px solid rgba(120, 120, 120, 0.5)',
            transition: 'background-color 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#1e222d'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(19, 23, 34, 0.8)'
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.75rem',
              fontWeight: 500,
              color: ind.color,
            }}
          >
            <span style={{ textTransform: 'capitalize' }}>
              {ind.name.replace(/_/g, ' ')}
            </span>

            {/* Show simple params summary if available */}
            {ind.params && (
              <span
                style={{
                  fontSize: '10px',
                  color: '#6b7280',
                  display: 'inline-block',
                }}
              >
                {Object.values(ind.params).slice(0, 3).join(', ')}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center' }}>
            <IconButton
              onClick={() => onEdit(ind.name)}
              size="small"
              sx={{
                color: '#888',
                padding: '2px',
                '&:hover': { color: '#d1d4dc' },
              }}
            >
              <SettingsIcon sx={{ fontSize: 14 }} />
            </IconButton>
            <IconButton
              onClick={() => onRemove(ind.name)}
              size="small"
              sx={{
                color: '#888',
                padding: '2px',
                '&:hover': { color: '#ef5350' },
              }}
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </div>
        </div>
      ))}
    </div>
  )
}
