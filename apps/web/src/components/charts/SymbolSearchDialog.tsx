import React, { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '../../api/client'

interface SymbolResult {
  symbol: string
  name: string
  asset_type: string
  exchange: string
}

interface SymbolSearchDialogProps {
  open: boolean
  onClose: () => void
  onSelect: (symbol: string) => void
}

const ASSET_TYPES = ['All', 'Stocks', 'ETF', 'Crypto', 'Forex', 'Futures']

export const SymbolSearchDialog: React.FC<SymbolSearchDialogProps> = ({
  open,
  onClose,
  onSelect,
}) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SymbolResult[]>([])
  const [loading, setLoading] = useState(false)
  const [activeFilter, setActiveFilter] = useState('All')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setResults([])
      setSelectedIdx(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const doSearch = useCallback(
    async (q: string) => {
      if (q.length < 1) {
        setResults([])
        return
      }
      setLoading(true)
      try {
        const params: any = { q }
        if (activeFilter !== 'All') {
          params.asset_type = activeFilter.toLowerCase()
        }
        const res = await api.get<SymbolResult[]>('/symbols/search', { params })
        setResults(res.data.slice(0, 20))
        setSelectedIdx(0)
      } catch {
        setResults([])
      } finally {
        setLoading(false)
      }
    },
    [activeFilter],
  )

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(query), 200)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [query, doSearch])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIdx((i) => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && results.length > 0) {
      onSelect(results[selectedIdx].symbol)
      onClose()
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

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
        {/* Search input */}
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
            onChange={(e) => setQuery(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            placeholder="Search symbol or company..."
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              color: '#f3f4f6',
              fontSize: '0.875rem',
              border: 'none',
              outline: 'none',
            }}
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              style={{
                color: '#6b7280',
                marginLeft: '8px',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              <svg
                style={{ width: '16px', height: '16px' }}
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 011.414-1.414L10 8.586z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          )}
        </div>

        {/* Filter chips */}
        <div
          style={{
            display: 'flex',
            gap: '6px',
            padding: '8px 16px',
            borderBottom: '1px solid #374151',
          }}
        >
          {ASSET_TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setActiveFilter(t)}
              style={{
                padding: '4px 10px',
                fontSize: '0.75rem',
                borderRadius: '9999px',
                cursor: 'pointer',
                transition: 'color 0.2s, background-color 0.2s',
                backgroundColor: activeFilter === t ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                color: activeFilter === t ? '#60a5fa' : '#9ca3af',
                border: activeFilter === t ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid transparent',
              }}
              onMouseEnter={(e) => {
                if (activeFilter !== t) {
                  e.currentTarget.style.color = '#e5e7eb'
                  e.currentTarget.style.backgroundColor = '#2a2e39'
                }
              }}
              onMouseLeave={(e) => {
                if (activeFilter !== t) {
                  e.currentTarget.style.color = '#9ca3af'
                  e.currentTarget.style.backgroundColor = 'transparent'
                }
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Results */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '24px',
                color: '#6b7280',
                fontSize: '0.875rem',
              }}
            >
              Searching...
            </div>
          )}
          {!loading && query && results.length === 0 && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '24px',
                color: '#6b7280',
                fontSize: '0.875rem',
              }}
            >
              No results found
            </div>
          )}
          {!loading &&
            results.map((r, i) => (
              <div
                key={r.symbol + r.exchange}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 16px',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                  backgroundColor: i === selectedIdx ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                }}
                onClick={() => {
                  onSelect(r.symbol)
                  onClose()
                }}
                onMouseEnter={(e) => {
                  if (i !== selectedIdx) e.currentTarget.style.backgroundColor = '#2a2e39'
                  setSelectedIdx(i)
                }}
                onMouseLeave={(e) => {
                  if (i !== selectedIdx) e.currentTarget.style.backgroundColor = 'transparent'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span
                    style={{
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      color: '#f3f4f6',
                      width: '80px',
                    }}
                  >
                    {r.symbol}
                  </span>
                  <span
                    style={{
                      fontSize: '0.75rem',
                      color: '#9ca3af',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      maxWidth: '250px',
                    }}
                  >
                    {r.name}
                  </span>
                </div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '0.75rem',
                    color: '#6b7280',
                  }}
                >
                  <span>{r.asset_type}</span>
                  <span style={{ color: '#4b5563' }}>{r.exchange}</span>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}
