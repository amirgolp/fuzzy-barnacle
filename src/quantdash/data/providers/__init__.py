"""Data providers module."""

from .base import DataProvider, get_symbol_data, normalize_dataframe
from .yfinance import YFinanceProvider, default_provider
from .cached import CachedDataProvider

# Create a cached version of the default provider
cached_provider = CachedDataProvider(default_provider)

__all__ = [
    "DataProvider",
    "normalize_dataframe",
    "get_symbol_data",
    "YFinanceProvider",
    "default_provider",
    "CachedDataProvider",
    "cached_provider",
]
