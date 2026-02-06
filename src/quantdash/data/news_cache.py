"""
SQLite-based cache for news articles and sentiment data.

Stores news articles by symbol with:
- Automatic expiration (configurable TTL)
- Deduplication by title
- Sentiment scores
"""

from __future__ import annotations

import sqlite3
import threading
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from quantdash.config import get_logger

logger = get_logger("data.news_cache")

# Default cache location (same directory as OHLCV cache)
DEFAULT_CACHE_DIR = Path.home() / ".quantdash" / "cache"
DEFAULT_DB_NAME = "news_cache.db"


class NewsCache:
    """
    SQLite-based cache for news articles.

    Stores news by symbol with automatic expiration.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ttl_minutes: int = 30,  # News expires after 30 minutes by default
    ):
        """
        Initialize the news cache.

        Args:
            db_path: Full path to SQLite database file
            cache_dir: Directory for cache (uses default if not specified)
            ttl_minutes: Time-to-live for cached news in minutes
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
        logger.info(f"News cache initialized at {self.db_path}")

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
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                published TEXT,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                fetched_at INTEGER NOT NULL,
                UNIQUE(symbol, title_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_news_symbol_fetched
            ON news_articles(symbol, fetched_at DESC);

            CREATE TABLE IF NOT EXISTS news_cache_meta (
                symbol TEXT PRIMARY KEY,
                last_fetched INTEGER NOT NULL,
                article_count INTEGER NOT NULL
            );
        """)
        conn.commit()

    def _title_hash(self, title: str) -> str:
        """Generate hash for title deduplication."""
        normalized = title.lower().strip()[:100]
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_cached_news(
        self,
        symbol: str,
        limit: int = 20,
    ) -> tuple[list[dict], bool]:
        """
        Get cached news for a symbol.

        Args:
            symbol: Ticker symbol
            limit: Maximum number of articles to return

        Returns:
            Tuple of (articles, is_valid) where is_valid indicates if cache is fresh
        """
        symbol = symbol.upper()
        conn = self._get_connection()
        now = int(datetime.utcnow().timestamp())

        # Check if cache is still valid
        meta = conn.execute(
            "SELECT last_fetched FROM news_cache_meta WHERE symbol = ?",
            (symbol,)
        ).fetchone()

        is_valid = False
        if meta:
            last_fetched = datetime.utcfromtimestamp(meta["last_fetched"])
            if datetime.utcnow() - last_fetched < self._ttl:
                is_valid = True

        # Fetch articles
        rows = conn.execute(
            """
            SELECT title, link, published, source, sentiment_score, sentiment_label
            FROM news_articles
            WHERE symbol = ?
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            (symbol, limit)
        ).fetchall()

        articles = [
            {
                "title": r["title"],
                "link": r["link"],
                "published": r["published"],
                "source": r["source"],
                "sentiment_score": r["sentiment_score"],
                "sentiment_label": r["sentiment_label"],
            }
            for r in rows
        ]

        return articles, is_valid

    def store_news(
        self,
        symbol: str,
        articles: list[dict],
    ) -> int:
        """
        Store news articles in the cache.

        Args:
            symbol: Ticker symbol
            articles: List of article dicts with title, link, published, source,
                     sentiment_score, sentiment_label

        Returns:
            Number of new articles stored
        """
        if not articles:
            return 0

        symbol = symbol.upper()
        conn = self._get_connection()
        now = int(datetime.utcnow().timestamp())

        stored = 0
        for article in articles:
            title_hash = self._title_hash(article["title"])
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO news_articles
                    (symbol, title_hash, title, link, published, source,
                     sentiment_score, sentiment_label, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        title_hash,
                        article["title"],
                        article["link"],
                        article.get("published"),
                        article["source"],
                        article["sentiment_score"],
                        article["sentiment_label"],
                        now,
                    )
                )
                stored += 1
            except Exception as e:
                logger.debug(f"Failed to store article: {e}")

        # Update metadata
        conn.execute(
            """
            INSERT OR REPLACE INTO news_cache_meta (symbol, last_fetched, article_count)
            VALUES (?, ?, ?)
            """,
            (symbol, now, len(articles))
        )

        conn.commit()
        logger.debug(f"Cached {stored} news articles for {symbol}")
        return stored

    def clear_symbol(self, symbol: str) -> int:
        """Clear cached news for a symbol."""
        symbol = symbol.upper()
        conn = self._get_connection()
        result = conn.execute("DELETE FROM news_articles WHERE symbol = ?", (symbol,))
        conn.execute("DELETE FROM news_cache_meta WHERE symbol = ?", (symbol,))
        conn.commit()
        return result.rowcount

    def clear_expired(self) -> int:
        """Clear all expired news from cache."""
        conn = self._get_connection()
        cutoff = int((datetime.utcnow() - self._ttl * 2).timestamp())
        result = conn.execute(
            "DELETE FROM news_articles WHERE fetched_at < ?",
            (cutoff,)
        )
        conn.commit()
        if result.rowcount > 0:
            logger.info(f"Cleared {result.rowcount} expired news articles")
        return result.rowcount

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_connection()

        total_articles = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        total_symbols = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM news_articles"
        ).fetchone()[0]

        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        size_kb = size_bytes / 1024

        return {
            "total_articles": total_articles,
            "total_symbols": total_symbols,
            "size_kb": round(size_kb, 2),
            "db_path": str(self.db_path),
            "ttl_minutes": self._ttl.total_seconds() / 60,
        }


# Global cache instance
_news_cache: Optional[NewsCache] = None


def get_news_cache() -> NewsCache:
    """Get or create the global news cache instance."""
    global _news_cache
    if _news_cache is None:
        _news_cache = NewsCache()
    return _news_cache
