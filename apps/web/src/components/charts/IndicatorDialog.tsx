import PushPinIcon from '@mui/icons-material/PushPin'
import PushPinOutlinedIcon from '@mui/icons-material/PushPinOutlined'
import { IconButton } from '@mui/material'
import React, { useEffect, useRef, useState } from 'react'

import { api } from '../../api/client'

interface IndicatorInfo {
  name: string
  category: string
  type: string
  description: string
  params: Record<string, any>
  color: string
}

export interface IndicatorSelection {
  name: string
  params: Record<string, any>
  color: string
  type?: 'indicator' | 'strategy'
}

interface IndicatorDialogProps {
  open: boolean
  onClose: () => void
  onAdd: (indicator: IndicatorSelection) => void
  onRemove: (name: string) => void
  activeIndicators: string[]
}

const CATEGORIES = [
  'All',
  'Trend',
  'Momentum',
  'Volatility',
  'Volume',
  'Strategy',
  'Favorites',
]

export const IndicatorDialog: React.FC<IndicatorDialogProps> = ({
  open,
  onClose,
  onAdd,
  onRemove,
  activeIndicators,
}) => {
  const [indicators, setIndicators] = useState<IndicatorInfo[]>([])
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All')
  const [loading, setLoading] = useState(false)
  const [pinned, setPinned] = useState<string[]>(() => {
    try {
      return JSON.parse(
        localStorage.getItem('quantdash_pinned_indicators') || '[]',
      )
    } catch {
      return []
    }
  })
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setCategory('All')
      setTimeout(() => inputRef.current?.focus(), 50)
      fetchIndicators()
    }
  }, [open])

  const handlePin = (e: React.MouseEvent, name: string) => {
    e.stopPropagation()
    setPinned((prev) => {
      const newPinned = prev.includes(name)
        ? prev.filter((p) => p !== name)
        : [...prev, name]
      localStorage.setItem(
        'quantdash_pinned_indicators',
        JSON.stringify(newPinned),
      )
      return newPinned
    })
  }

  const fetchIndicators = async () => {
    setLoading(true)
    try {
      const [indRes, stratRes] = await Promise.all([
        api.get<IndicatorInfo[]>('/features/indicators'),
        api.get<any[]>('/strategies/'),
      ])
      // Filter out strategies that might duplicate built-in indicators if needed, or just list them all
      const stratItems: IndicatorInfo[] = stratRes.data.map((s: any) => ({
        name: s.id || s.name,
        category: 'Strategy',
        type: 'strategy',
        description: s.description || '',
        params: s.default_params || {},
        color: '#FF6D00',
      }))

      // Merge lists
      setIndicators([...indRes.data, ...stratItems])
    } catch {
      setIndicators([])
    } finally {
      setLoading(false)
    }
  }

  const filtered = indicators
    .filter((ind) => {
      const matchesQuery =
        !query ||
        ind.name.toLowerCase().includes(query.toLowerCase()) ||
        ind.description.toLowerCase().includes(query.toLowerCase())

      if (category === 'Favorites') {
        return matchesQuery && pinned.includes(ind.name)
      }

      const matchesCat = category === 'All' || ind.category === category
      return matchesQuery && matchesCat
    })
    .sort((a, b) => {
      // Sort pinned first if not in Favorites category (where they are all pinned)
      if (category !== 'Favorites') {
        const aPinned = pinned.includes(a.name)
        const bPinned = pinned.includes(b.name)
        if (aPinned && !bPinned) return -1
        if (!aPinned && bPinned) return 1
      }
      return 0 // Keep original order or alphabetical could be better
    })

  if (!open) return null

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '10vh',
      }}
      onClick={onClose}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.6)',
        }}
      />
      <div
        style={{
          position: 'relative',
          backgroundColor: '#1e222d',
          borderRadius: '8px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid #374151',
          width: '520px',
          maxHeight: '70vh',
          display: 'flex',
          flexDirection: 'column',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            borderBottom: '1px solid #374151',
            padding: '12px 16px',
          }}
        >
          <svg
            style={{
              width: '20px',
              height: '20px',
              color: '#9ca3af',
              marginRight: '12px',
              flexShrink: 0,
            }}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
            }}
            placeholder="Search indicators..."
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              color: '#f3f4f6',
              fontSize: '0.875rem',
              border: 'none',
              outline: 'none',
            }}
          />
        </div>

        <div
          style={{
            display: 'flex',
            gap: '6px',
            padding: '8px 16px',
            borderBottom: '1px solid #374151',
            overflowX: 'auto',
          }}
        >
          {CATEGORIES.map((c) => (
            <button
              key={c}
              onClick={() => setCategory(c)}
              style={{
                padding: '4px 10px',
                fontSize: '0.75rem',
                borderRadius: '9999px',
                whiteSpace: 'nowrap',
                cursor: 'pointer',
                transition: 'color 0.2s, background-color 0.2s',
                backgroundColor: category === c ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                color: category === c ? '#60a5fa' : '#9ca3af',
                border: category === c ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid transparent',
              }}
              onMouseEnter={(e) => {
                if (category !== c) {
                  e.currentTarget.style.color = '#e5e7eb'
                  e.currentTarget.style.backgroundColor = '#2a2e39'
                }
              }}
              onMouseLeave={(e) => {
                if (category !== c) {
                  e.currentTarget.style.color = '#9ca3af'
                  e.currentTarget.style.backgroundColor = 'transparent'
                }
              }}
            >
              {c}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading && (
            <div
              style={{
                padding: '24px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '0.875rem',
              }}
            >
              Loading...
            </div>
          )}
          {!loading && filtered.length === 0 && (
            <div
              style={{
                padding: '24px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '0.875rem',
              }}
            >
              No indicators found
            </div>
          )}
          {!loading &&
            filtered.map((ind) => {
              const isActive = activeIndicators.includes(ind.name)
              const isPinned = pinned.includes(ind.name)
              return (
                <div
                  key={ind.name}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 16px',
                    cursor: 'pointer',
                    transition: 'background-color 0.2s',
                    backgroundColor: isActive ? 'rgba(59, 130, 246, 0.05)' : 'transparent',
                  }}
                  onClick={() => {
                    if (isActive) {
                      onRemove(ind.name)
                    } else {
                      onAdd({
                        name: ind.name,
                        params: ind.params,
                        color: ind.color,
                        type:
                          ind.type === 'strategy' ? 'strategy' : 'indicator',
                      })
                    }
                  }}
                  onMouseEnter={(e) => {
                    // Using inline style mutation for hover
                    if (!isActive) e.currentTarget.style.backgroundColor = '#2a2e39'
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = 'transparent'
                  }}
                >
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div
                          style={{
                            width: '10px',
                            height: '10px',
                            borderRadius: '50%',
                            backgroundColor: ind.color,
                          }}
                        />
                        <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#f3f4f6' }}>
                          {ind.name}
                        </span>
                        {isActive && (
                          <span
                            style={{
                              fontSize: '10px',
                              color: '#60a5fa',
                              backgroundColor: 'rgba(59, 130, 246, 0.1)',
                              padding: '0 6px',
                              borderRadius: '4px',
                            }}
                          >
                            Active
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '2px', marginLeft: '18px', marginRight: '8px' }}>
                        {ind.description}
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <IconButton
                        size="small"
                        onClick={(e) => handlePin(e, ind.name)}
                        sx={{
                          color: isPinned ? '#2962FF' : '#444',
                          '&:hover': { color: '#2962FF' },
                          p: 0.5,
                        }}
                      >
                        {isPinned ? (
                          <PushPinIcon sx={{ fontSize: 16 }} />
                        ) : (
                          <PushPinOutlinedIcon sx={{ fontSize: 16 }} />
                        )}
                      </IconButton>
                      <span
                        style={{
                          fontSize: '10px',
                          color: '#4b5563',
                          textTransform: 'uppercase',
                          width: '64px',
                          textAlign: 'right',
                          flexShrink: 0,
                        }}
                      >
                        {ind.category}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
        </div>
      </div>
    </div>
  )
}
