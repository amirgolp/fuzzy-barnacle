"""
SQLite-based cache for screener results.

Stores screening results by symbol/portfolio with:
- Automatic expiration (configurable TTL)
- JSON serialization for complex results
"""

from __future__ import annotations

import json
import sqlite3
import threading
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from quantdash.config import get_logger

logger = get_logger("data.screener_cache")

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".quantdash" / "cache"
DEFAULT_DB_NAME = "screener_cache.db"


class ScreenerCache:
    """
    SQLite-based cache for screener and technical analysis results.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ttl_minutes: int = 10,  # Screener results expire after 10 minutes
    ):
        """
        Initialize the screener cache.

        Args:
            db_path: Full path to SQLite database file
            cache_dir: Directory for cache (uses default if not specified)
            ttl_minutes: Time-to-live for cached results in minutes
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            cache_dir = cache_dir or DEFAULT_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / DEFAULT_DB_NAME

        self._ttl = timedelta(minutes=ttl_minutes)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"Screener cache initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS screener_results (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                fetched_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_screener_fetched
            ON screener_results(fetched_at);
        """)
        conn.commit()

    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a unique key for the cache based on parameters."""
        # Sort kwargs to ensure consistent key generation
        sorted_items = sorted(kwargs.items())
        key_str = f"{prefix}:" + "&".join(f"{k}={v}" for k, v in sorted_items)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_result(self, prefix: str, **kwargs) -> Optional[dict]:
        """
        Get cached result.

        Args:
            prefix: Key prefix (e.g. 'symbol', 'portfolio')
            **kwargs: Parameters identifying the request (e.g. symbol='AAPL')

        Returns:
            Cached result dict or None if not found/expired
        """
        key = self._generate_key(prefix, **kwargs)
        conn = self._get_connection()

        row = conn.execute(
            "SELECT result_json, fetched_at FROM screener_results WHERE key = ?",
            (key,)
        ).fetchone()

        if not row:
            return None

        fetched_at = datetime.utcfromtimestamp(row["fetched_at"])
        if datetime.utcnow() - fetched_at > self._ttl:
            # Expired
            return None

        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            return None

    def store_result(self, result: dict, prefix: str, **kwargs) -> None:
        """
        Store result in cache.

        Args:
            result: Dictionary to store
            prefix: Key prefix
            **kwargs: Parameters identifying the request
        """
        key = self._generate_key(prefix, **kwargs)
        conn = self._get_connection()
        now = int(datetime.utcnow().timestamp())

        try:
            result_json = json.dumps(result)
            conn.execute(
                """
                INSERT OR REPLACE INTO screener_results (key, result_json, fetched_at)
                VALUES (?, ?, ?)
                """,
                (key, result_json, now)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache screener result: {e}")

    def clear_expired(self) -> int:
        """Clear expired entries."""
        conn = self._get_connection()
        cutoff = int((datetime.utcnow() - self._ttl).timestamp())
        result = conn.execute(
            "DELETE FROM screener_results WHERE fetched_at < ?",
            (cutoff,)
        )
        conn.commit()
        return result.rowcount


# Global cache instance
_screener_cache: Optional[ScreenerCache] = None


def get_screener_cache() -> ScreenerCache:
    """Get or create the global screener cache instance."""
    global _screener_cache
    if _screener_cache is None:
        _screener_cache = ScreenerCache()
    return _screener_cache
