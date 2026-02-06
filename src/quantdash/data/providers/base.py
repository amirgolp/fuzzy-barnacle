"""Base data provider interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    def download_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Download OHLCV bars for given symbols.

        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            adjusted: Whether to use adjusted prices

        Returns:
            DataFrame with canonical columns:
                - index: datetime (tz-aware normalized)
                - columns: open, high, low, close, volume
        """
        ...

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary of fundamental metrics or None if unavailable
        """
        ...


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names to lowercase, stripped.

    This handles yfinance returning MultiIndex columns or mixed-case columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names
    """
    if df.empty:
        return df

    # Handle MultiIndex columns (e.g., from multi-symbol yfinance download)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten to just the first level (e.g., 'Close' from ('Close', 'AAPL'))
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = df.columns.astype(str).str.lower().str.strip()

    return df


def get_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extract single-symbol data from a potentially multi-symbol DataFrame.

    Handles yfinance MultiIndex columns and returns normalized single-symbol data.

    Args:
        df: DataFrame with potentially MultiIndex columns
        symbol: Symbol to extract

    Returns:
        DataFrame with normalized lowercase column names
    """
    if df.empty:
        return df

    # If MultiIndex, select the symbol
    if isinstance(df.columns, pd.MultiIndex):
        # Get unique symbols from second level
        if symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, level=1, axis=1)
        else:
            # Single symbol case - just take first level
            df.columns = df.columns.get_level_values(0)

    return normalize_dataframe(df)
