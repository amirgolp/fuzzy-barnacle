/**
 * React Query hooks for market data fetching with intelligent caching.
 *
 * Caching strategy:
 * - Data is cached in memory and persists across tab switches
 * - staleTime: Data is considered fresh for 5 minutes (won't refetch)
 * - gcTime: Data stays in cache for 30 minutes after last use
 * - Refetches only happen when data is stale AND component remounts
 */
import { useQuery, useQueryClient } from '@tanstack/react-query'

import { type Bar, dataApi } from '../api/data'

// Query key factory for consistent cache keys
export const marketDataKeys = {
  all: ['marketData'] as const,
  bars: (symbol: string, timeframe: string) =>
    [...marketDataKeys.all, 'bars', symbol.toUpperCase(), timeframe] as const,
  barsByRange: (
    symbol: string,
    timeframe: string,
    start: string,
    end: string,
  ) =>
    [
      ...marketDataKeys.all,
      'barsByRange',
      symbol.toUpperCase(),
      timeframe,
      start,
      end,
    ] as const,
}

interface UseBarsOptions {
  enabled?: boolean
  staleTime?: number
  refetchOnWindowFocus?: boolean
}

/**
 * Hook to fetch OHLCV bars for a symbol.
 * Data is cached and won't refetch unless stale.
 */
export function useBars(
  symbol: string,
  timeframe: string = '1d',
  options: UseBarsOptions = {},
) {
  const {
    enabled = true,
    staleTime = 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus = false,
  } = options

  return useQuery({
    queryKey: marketDataKeys.bars(symbol, timeframe),
    queryFn: () => dataApi.getBars(symbol, timeframe),
    enabled: enabled && !!symbol,
    staleTime,
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
    refetchOnWindowFocus,
    refetchOnMount: false, // Don't refetch if data exists in cache
    refetchOnReconnect: false,
  })
}

/**
 * Hook to fetch OHLCV bars for a specific date range.
 * Used for loading historical data when scrolling.
 */
export function useBarsByRange(
  symbol: string,
  timeframe: string,
  start: Date | null,
  end: Date | null,
  options: UseBarsOptions = {},
) {
  const {
    enabled = true,
    staleTime = 30 * 60 * 1000, // 30 minutes for historical data
    refetchOnWindowFocus = false,
  } = options

  const startStr = start?.toISOString() ?? ''
  const endStr = end?.toISOString() ?? ''

  return useQuery({
    queryKey: marketDataKeys.barsByRange(symbol, timeframe, startStr, endStr),
    queryFn: () => dataApi.getBarsByRange(symbol, timeframe, start!, end!),
    enabled: enabled && !!symbol && !!start && !!end,
    staleTime,
    gcTime: 60 * 60 * 1000, // Keep historical data for 1 hour
    refetchOnWindowFocus,
    refetchOnMount: false,
    refetchOnReconnect: false,
  })
}

/**
 * Hook to prefetch bars data for a symbol.
 * Useful for preloading data before user navigates.
 */
export function usePrefetchBars() {
  const queryClient = useQueryClient()

  return (symbol: string, timeframe: string = '1d') => {
    return queryClient.prefetchQuery({
      queryKey: marketDataKeys.bars(symbol, timeframe),
      queryFn: () => dataApi.getBars(symbol, timeframe),
      staleTime: 5 * 60 * 1000,
    })
  }
}

/**
 * Hook to manually update/merge bars data in the cache.
 * Used when fetching additional historical data.
 */
export function useUpdateBarsCache() {
  const queryClient = useQueryClient()

  return (
    symbol: string,
    timeframe: string,
    newBars: Bar[],
    prepend: boolean = true,
  ) => {
    queryClient.setQueryData<Bar[]>(
      marketDataKeys.bars(symbol, timeframe),
      (oldBars) => {
        if (!oldBars) return newBars

        // Merge bars, avoiding duplicates by timestamp
        const existingTimes = new Set(oldBars.map((b) => b.time))
        const uniqueNewBars = newBars.filter((b) => !existingTimes.has(b.time))

        const merged = prepend
          ? [...uniqueNewBars, ...oldBars]
          : [...oldBars, ...uniqueNewBars]

        // Sort by time
        return merged.sort((a, b) => (a.time as any) - (b.time as any))
      },
    )
  }
}

/**
 * Hook to invalidate cached data for a symbol.
 * Forces refetch on next access.
 */
export function useInvalidateBars() {
  const queryClient = useQueryClient()

  return (symbol?: string, timeframe?: string) => {
    if (symbol && timeframe) {
      queryClient.invalidateQueries({
        queryKey: marketDataKeys.bars(symbol, timeframe),
      })
    } else if (symbol) {
      queryClient.invalidateQueries({
        queryKey: [...marketDataKeys.all, 'bars', symbol.toUpperCase()],
      })
    } else {
      queryClient.invalidateQueries({
        queryKey: marketDataKeys.all,
      })
    }
  }
}

/**
 * Get cached bars data without triggering a fetch.
 */
export function useGetCachedBars() {
  const queryClient = useQueryClient()

  return (symbol: string, timeframe: string): Bar[] | undefined => {
    return queryClient.getQueryData<Bar[]>(
      marketDataKeys.bars(symbol, timeframe),
    )
  }
}
