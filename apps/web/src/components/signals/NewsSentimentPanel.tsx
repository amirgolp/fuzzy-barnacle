import { Refresh } from '@mui/icons-material'
import {
  Box,
  Button,
  Card,
  CircularProgress,
  IconButton,
  LinearProgress,
  Link,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material'
import React, { useEffect, useMemo, useState } from 'react'

import { api } from '../../api/client'
import { TechnicalsGauge } from './TechnicalsGauge'

interface NewsItem {
  title: string
  link: string
  published: string | null
  source: string
  sentiment_score: number
  sentiment_label: string
}

interface SentimentData {
  symbol: string
  score: number
  label: string
  news_count: number
  gauge_news_count: number
  bullish_count: number
  bearish_count: number
  neutral_count: number
  items: NewsItem[]
}

interface NewsSentimentPanelProps {
  symbol: string
  /** Compact mode with smaller fonts */
  compact?: boolean
}

type SortOption = 'recent' | 'importance' | 'buy' | 'sell' | 'neutral'

const scoreColor = (score: number) => {
  if (score > 0.1) return '#26a69a'
  if (score < -0.1) return '#ef5350'
  return '#9e9e9e'
}

export const NewsSentimentPanel: React.FC<NewsSentimentPanelProps> = ({
  symbol,
  compact = false,
}) => {
  const [data, setData] = useState<SentimentData | null>(null)
  const [allItems, setAllItems] = useState<NewsItem[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [displayCount, setDisplayCount] = useState(5)
  const [sortBy, setSortBy] = useState<SortOption>('importance')

  const fetchData = React.useCallback(async () => {
    setLoading(true)
    setDisplayCount(5)
    try {
      // Fetch 20 articles for gauge, display 5 initially
      const res = await api.get<SentimentData>('/news/sentiment', {
        params: { symbol, gauge_count: 20, display_count: 20 },
      })
      setData(res.data)
      setAllItems(res.data.items || [])
    } catch {
      setData(null)
      setAllItems([])
    } finally {
      setLoading(false)
    }
  }, [symbol])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Sort items based on selected option
  const sortedItems = useMemo(() => {
    const items = [...allItems]
    switch (sortBy) {
      case 'importance':
        // Absolute score magnitude (strongest sentiment first)
        return items.sort(
          (a, b) => Math.abs(b.sentiment_score) - Math.abs(a.sentiment_score),
        )

      case 'buy':
        return items.filter((i) => i.sentiment_label === 'Bullish')
      case 'sell':
        return items.filter((i) => i.sentiment_label === 'Bearish')
      case 'neutral':
        return items.filter((i) => i.sentiment_label === 'Neutral')
      case 'recent':
      default:
        // Sort by date if available, otherwise API order (which is usually recent)
        return items.sort((a, b) => {
          if (a.published && b.published) {
            return (
              new Date(b.published).getTime() - new Date(a.published).getTime()
            )
          }
          return 0 // Keep original order
        })
    }
  }, [allItems, sortBy])

  const displayedItems = sortedItems.slice(0, displayCount)
  const hasMore = displayCount < allItems.length

  const handleLoadMore = async () => {
    if (displayCount < allItems.length) {
      setDisplayCount((prev) => Math.min(prev + 5, allItems.length))
    } else {
      // Fetch more from API
      setLoadingMore(true)
      try {
        const res = await api.get('/news/feed', {
          params: { symbol, limit: 50 },
        })
        const newItems = res.data as NewsItem[]
        setAllItems((prev) => {
          const existingTitles = new Set(
            prev.map((i) => i.title.toLowerCase().slice(0, 50)),
          )
          const unique = newItems.filter(
            (i) => !existingTitles.has(i.title.toLowerCase().slice(0, 50)),
          )
          return [...prev, ...unique]
        })
        setDisplayCount((prev) => prev + 5)
      } catch (e) {
        console.error('Failed to load more news:', e)
      } finally {
        setLoadingMore(false)
      }
    }
  }

  const fontSize = compact ? '0.6rem' : '0.65rem'
  const titleFontSize = compact ? '0.65rem' : '0.7rem'

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1,
          py: 0.5,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Typography sx={{ fontWeight: 700, fontSize: titleFontSize }}>
          News Sentiment
        </Typography>
        <IconButton
          size="small"
          onClick={fetchData}
          disabled={loading}
          sx={{ p: 0.5 }}
        >
          {loading ? (
            <CircularProgress size={12} />
          ) : (
            <Refresh sx={{ fontSize: 14 }} />
          )}
        </IconButton>
      </Box>

      {loading && <LinearProgress />}

      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          px: 1,
          py: 0.5,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {data && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 0.5 }}>
            <TechnicalsGauge
              title={`Sentiment (${data.gauge_news_count || data.news_count})`}
              value={data.score * 100}
              recommendation={data.label}
              buyCount={data.bullish_count ?? 0}
              sellCount={data.bearish_count ?? 0}
              neutralCount={data.neutral_count ?? 0}
              compact={compact}
            />
          </Box>
        )}

        {/* Sort options */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 0.5,
            px: 0.5,
          }}
        >
          <Typography
            sx={{
              color: '#888',
              fontWeight: 600,
              fontSize: '0.55rem',
              textTransform: 'uppercase',
            }}
          >
            News ({displayedItems.length}/{allItems.length})
          </Typography>
          <ToggleButtonGroup
            value={sortBy}
            exclusive
            onChange={(_, v) => v && setSortBy(v)}
            size="small"
            sx={{
              flexWrap: 'wrap',
              '& .MuiToggleButton-root': {
                px: 0.8,
                py: 0.2,
                fontSize: '0.55rem',
                textTransform: 'none',
                minWidth: 0,
                lineHeight: 1.4,
                border: '1px solid rgba(255,255,255,0.05)',
                color: 'text.secondary',
                m: 0.2,
                borderRadius: '4px !important',
                '&.Mui-selected': {
                  bgcolor: 'primary.main',
                  color: '#fff',
                  '&:hover': { bgcolor: 'primary.dark' },
                },
              },
            }}
          >
            <ToggleButton value="recent">Relevance</ToggleButton>
            <ToggleButton value="importance">Importance</ToggleButton>

            <ToggleButton
              value="buy"
              sx={{
                color: '#26a69a !important',
                '&.Mui-selected': {
                  bgcolor: '#26a69a !important',
                  color: '#fff !important',
                },
              }}
            >
              Buy
            </ToggleButton>
            <ToggleButton value="neutral">Neutral</ToggleButton>
            <ToggleButton
              value="sell"
              sx={{
                color: '#ef5350 !important',
                '&.Mui-selected': {
                  bgcolor: '#ef5350 !important',
                  color: '#fff !important',
                },
              }}
            >
              Sell
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {displayedItems.map((item, i) => (
          <Box
            key={i}
            sx={{
              py: 0.3,
              borderBottom:
                i < displayedItems.length - 1 ? '1px solid' : 'none',
              borderColor: 'divider',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.5 }}>
              <Box
                sx={{
                  width: 4,
                  height: 4,
                  borderRadius: '50%',
                  bgcolor: scoreColor(item.sentiment_score),
                  mt: 0.6,
                  flexShrink: 0,
                }}
              />
              <Tooltip title={item.title} placement="left">
                <Link
                  href={item.link}
                  target="_blank"
                  rel="noopener"
                  underline="hover"
                  sx={{
                    fontSize,
                    color: 'text.primary',
                    lineHeight: 1.25,
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                  }}
                >
                  {item.title}
                </Link>
              </Tooltip>
            </Box>
          </Box>
        ))}

        {/* Load more button */}
        {(hasMore || allItems.length > 0) && (
          <Button
            variant="outlined"
            fullWidth
            size="small"
            onClick={handleLoadMore}
            disabled={loadingMore}
            sx={{
              mt: 1,
              fontSize: '0.65rem',
              py: 0.5,
              textTransform: 'none',
              borderColor: 'divider',
              color: 'text.secondary',
              '&:hover': {
                borderColor: 'text.secondary',
                color: 'text.primary',
              },
            }}
          >
            {loadingMore
              ? 'Loading...'
              : hasMore
                ? `Load more (${allItems.length - displayCount} more)`
                : 'Fetch more news'}
          </Button>
        )}

        {!loading && allItems.length === 0 && (
          <Typography
            color="text.secondary"
            sx={{ display: 'block', textAlign: 'center', py: 1, fontSize }}
          >
            No news available
          </Typography>
        )}
      </Box>
    </Card>
  )
}
