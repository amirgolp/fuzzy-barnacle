"""
SQLite-based cache for OHLCV time series data.

Features:
- Stores OHLCV bars by symbol and timeframe
- Smart fetching: only requests missing data from provider
- Automatic gap detection and filling
- Thread-safe for concurrent access
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from quantdash.config import get_logger

logger = get_logger("data.cache")

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".quantdash" / "cache"
DEFAULT_DB_NAME = "ohlcv_cache.db"


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime to naive UTC for consistent comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC then remove tzinfo
        from datetime import timezone
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class OHLCVCache:
    """
    SQLite-based cache for OHLCV time series data.

    Stores data by (symbol, timeframe) and supports:
    - Incremental updates (only fetch missing bars)
    - Range queries with gap detection
    - Automatic schema creation
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the cache.

        Args:
            db_path: Full path to SQLite database file
            cache_dir: Directory for cache (uses default if not specified)
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            cache_dir = cache_dir or DEFAULT_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / DEFAULT_DB_NAME

        self._local = threading.local()
        self._init_schema()
        logger.info(f"OHLCV cache initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
            ON ohlcv(symbol, timeframe, timestamp);

            CREATE TABLE IF NOT EXISTS cache_meta (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                first_ts INTEGER,
                last_ts INTEGER,
                last_updated INTEGER NOT NULL,
                PRIMARY KEY (symbol, timeframe)
            );
        """)
        conn.commit()

    def get_cached_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the cached data range for a symbol/timeframe.

        Returns:
            Tuple of (first_datetime, last_datetime) or (None, None) if no data
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT first_ts, last_ts FROM cache_meta WHERE symbol = ? AND timeframe = ?",
            (symbol.upper(), timeframe),
        ).fetchone()

        if row and row["first_ts"] and row["last_ts"]:
            return (
                datetime.utcfromtimestamp(row["first_ts"]),
                datetime.utcfromtimestamp(row["last_ts"]),
            )
        return None, None

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get cached bars for a symbol/timeframe within a date range.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe (1m, 5m, 1h, 1d, etc.)
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            DataFrame with OHLCV data, empty if no cached data
        """
        symbol = symbol.upper()
        conn = self._get_connection()

        # Normalize to naive UTC for consistent timestamp conversion
        start = _to_naive_utc(start)
        end = _to_naive_utc(end)

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params: list = [symbol, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(int(start.timestamp()))
        if end:
            query += " AND timestamp <= ?"
            params.append(int(end.timestamp()))

        query += " ORDER BY timestamp ASC"

        rows = conn.execute(query, params).fetchall()

        if not rows:
            return pd.DataFrame()

        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        timestamps = []

        for row in rows:
            timestamps.append(datetime.utcfromtimestamp(row["timestamp"]))
            data["open"].append(row["open"])
            data["high"].append(row["high"])
            data["low"].append(row["low"])
            data["close"].append(row["close"])
            data["volume"].append(row["volume"])

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, tz="UTC"))
        df.index.name = None
        return df

    def store_bars(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> int:
        """
        Store OHLCV bars in the cache.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe
            df: DataFrame with OHLCV data (index must be datetime)

        Returns:
            Number of bars stored/updated
        """
        if df.empty:
            return 0

        symbol = symbol.upper()
        conn = self._get_connection()
        now = int(datetime.utcnow().timestamp())

        # Prepare data for insertion
        rows = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else int(idx)
            rows.append((
                symbol,
                timeframe,
                ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                now,
            ))

        # Insert or replace
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv
            (symbol, timeframe, timestamp, open, high, low, close, volume, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Update metadata
        timestamps = [r[2] for r in rows]
        first_ts, last_ts = min(timestamps), max(timestamps)

        # Get existing range and expand if necessary
        existing = conn.execute(
            "SELECT first_ts, last_ts FROM cache_meta WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()

        if existing:
            first_ts = min(first_ts, existing["first_ts"] or first_ts)
            last_ts = max(last_ts, existing["last_ts"] or last_ts)

        conn.execute(
            """
            INSERT OR REPLACE INTO cache_meta (symbol, timeframe, first_ts, last_ts, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, timeframe, first_ts, last_ts, now),
        )

        conn.commit()
        logger.debug(f"Cached {len(rows)} bars for {symbol}/{timeframe}")
        return len(rows)

    def get_missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Identify date ranges that are not cached.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe
            start: Desired start datetime
            end: Desired end datetime

        Returns:
            List of (start, end) tuples representing missing ranges
        """
        cached_start, cached_end = self.get_cached_range(symbol, timeframe)

        # Normalize to naive UTC for consistent comparison
        start = _to_naive_utc(start)
        end = _to_naive_utc(end)

        if cached_start is None or cached_end is None:
            # No cache at all, need everything
            return [(start, end)]

        missing = []

        # Need data before cache?
        if start < cached_start:
            missing.append((start, cached_start - timedelta(seconds=1)))

        # Need data after cache?
        if end > cached_end:
            missing.append((cached_end + timedelta(seconds=1), end))

        return missing

    def clear_symbol(self, symbol: str, timeframe: Optional[str] = None) -> int:
        """
        Clear cached data for a symbol.

        Args:
            symbol: Ticker symbol
            timeframe: Optional timeframe (clears all if not specified)

        Returns:
            Number of rows deleted
        """
        symbol = symbol.upper()
        conn = self._get_connection()

        if timeframe:
            result = conn.execute(
                "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            conn.execute(
                "DELETE FROM cache_meta WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
        else:
            result = conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
            conn.execute("DELETE FROM cache_meta WHERE symbol = ?", (symbol,))

        conn.commit()
        return result.rowcount

    def clear_all(self) -> int:
        """Clear all cached data. Returns number of rows deleted."""
        conn = self._get_connection()
        result = conn.execute("DELETE FROM ohlcv")
        conn.execute("DELETE FROM cache_meta")
        conn.commit()
        logger.info(f"Cleared {result.rowcount} cached bars")
        return result.rowcount

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_connection()

        total_bars = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        total_symbols = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM ohlcv"
        ).fetchone()[0]

        # Size in MB
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)

        symbols = conn.execute(
            """
            SELECT symbol, timeframe, COUNT(*) as bars,
                   MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
        ).fetchall()

        return {
            "total_bars": total_bars,
            "total_symbols": total_symbols,
            "size_mb": round(size_mb, 2),
            "db_path": str(self.db_path),
            "symbols": [
                {
                    "symbol": r["symbol"],
                    "timeframe": r["timeframe"],
                    "bars": r["bars"],
                    "first": datetime.utcfromtimestamp(r["first_ts"]).isoformat(),
                    "last": datetime.utcfromtimestamp(r["last_ts"]).isoformat(),
                }
                for r in symbols
            ],
        }


# Global cache instance
_cache: Optional[OHLCVCache] = None


def get_cache() -> OHLCVCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = OHLCVCache()
    return _cache
