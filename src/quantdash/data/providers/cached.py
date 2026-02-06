"""
Caching data provider wrapper.

Wraps any DataProvider with intelligent caching:
- Checks local SQLite cache first
- Only fetches missing data from upstream provider
- Automatically updates cache with new data
- Supports incremental updates for real-time data
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from quantdash.config import get_logger
from quantdash.data.cache import OHLCVCache, get_cache

from .base import DataProvider, normalize_dataframe

logger = get_logger("data.cached")


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime to naive UTC for consistent comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        from datetime import timezone
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class CachedDataProvider(DataProvider):
    """
    Caching wrapper for any DataProvider.

    Uses a local SQLite database to cache OHLCV data and minimize
    redundant API calls to the upstream provider.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[OHLCVCache] = None,
        max_cache_age_minutes: int = 5,
    ):
        """
        Initialize cached provider.

        Args:
            provider: Upstream data provider to wrap
            cache: Cache instance (uses global cache if not specified)
            max_cache_age_minutes: Maximum age of cached data before refresh
                                   (applies to the most recent bar only)
        """
        self._provider = provider
        self._cache = cache or get_cache()
        self._max_cache_age = timedelta(minutes=max_cache_age_minutes)

    @property
    def name(self) -> str:
        return f"cached_{self._provider.name}"

    def download_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Download OHLCV bars with caching.

        First checks the local cache, then fetches only missing data
        from the upstream provider.
        """
        if not symbols:
            return pd.DataFrame()

        # For now, handle single symbol (most common case)
        # Multi-symbol support can be added later
        if len(symbols) > 1:
            # Fall back to upstream for multi-symbol requests
            logger.debug(f"Multi-symbol request, bypassing cache")
            return self._provider.download_bars(symbols, timeframe, start, end, adjusted)

        symbol = symbols[0].upper()

        # Normalize datetimes to naive UTC for consistent comparison
        start_naive = _to_naive_utc(start)
        end_naive = _to_naive_utc(end)

        # Get cached data
        cached_df = self._cache.get_bars(symbol, timeframe, start_naive, end_naive)
        logger.debug(f"Cache returned {len(cached_df)} bars for {symbol}/{timeframe}")

        # Determine what's missing
        missing_ranges = self._cache.get_missing_ranges(symbol, timeframe, start_naive, end_naive)

        # Check if we need to refresh the most recent data
        need_refresh = self._should_refresh(symbol, timeframe, end_naive)
        if need_refresh and not any(r[1] >= end_naive - timedelta(hours=1) for r in missing_ranges):
            # Add a range to fetch recent data
            cached_start, cached_end = self._cache.get_cached_range(symbol, timeframe)
            if cached_end:
                missing_ranges.append((cached_end, end_naive))
                logger.debug(f"Adding refresh range: {cached_end} to {end_naive}")

        # Fetch missing data
        if missing_ranges:
            logger.info(f"Fetching {len(missing_ranges)} missing range(s) for {symbol}/{timeframe}")

            all_new_data = []
            for range_start, range_end in missing_ranges:
                try:
                    new_df = self._provider.download_bars(
                        [symbol], timeframe, range_start, range_end, adjusted
                    )
                    if not new_df.empty:
                        all_new_data.append(new_df)
                        self._cache.store_bars(symbol, timeframe, new_df)
                        logger.info(f"Fetched and cached {len(new_df)} bars for {symbol}/{timeframe}")
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}/{timeframe}: {e}")

            # Combine cached and new data
            if all_new_data:
                combined = pd.concat([cached_df] + all_new_data)
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()

                # Filter to requested range (using naive UTC timestamps)
                if start_naive:
                    combined = combined[combined.index >= pd.Timestamp(start_naive, tz="UTC")]
                if end_naive:
                    combined = combined[combined.index <= pd.Timestamp(end_naive, tz="UTC")]

                return combined

        # Return cached data if no fetch needed
        return cached_df

    def _should_refresh(self, symbol: str, timeframe: str, end: datetime) -> bool:
        """
        Determine if cached data should be refreshed.

        Returns True if:
        - No cached data exists
        - The cached data is older than max_cache_age
        - The requested end time is recent (within 24 hours)
        """
        cached_start, cached_end = self._cache.get_cached_range(symbol, timeframe)

        if cached_end is None:
            return True

        now = datetime.utcnow()
        # Normalize end to naive UTC for comparison
        end_naive = _to_naive_utc(end)

        # Only refresh if requesting recent data
        if end_naive < now - timedelta(days=1):
            return False

        # Check if cached data is stale
        cache_age = now - cached_end
        return cache_age > self._max_cache_age

    def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """Get fundamentals (not cached, always from upstream)."""
        return self._provider.get_fundamentals(symbol)

    def get_bars_smart(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 500,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Smart bar fetching that automatically determines the date range.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe
            bars: Approximate number of bars to fetch
            end: End datetime (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        if end is None:
            end = datetime.utcnow()

        # Estimate start date based on timeframe and number of bars
        tf_minutes = self._timeframe_to_minutes(timeframe)
        estimated_duration = timedelta(minutes=tf_minutes * bars * 1.5)  # Add buffer

        # For daily+ timeframes, account for weekends
        if timeframe in ("1d", "1wk", "1mo"):
            estimated_duration = estimated_duration * 1.5

        start = end - estimated_duration

        df = self.download_bars([symbol], timeframe, start, end)

        # Trim to requested number of bars
        if len(df) > bars:
            df = df.iloc[-bars:]

        return df

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1wk": 10080,
            "1mo": 43200,
        }
        return mapping.get(timeframe, 1440)

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear (clears all if None)
            timeframe: Specific timeframe to clear

        Returns:
            Number of rows deleted
        """
        if symbol:
            return self._cache.clear_symbol(symbol, timeframe)
        return self._cache.clear_all()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.get_cache_stats()
