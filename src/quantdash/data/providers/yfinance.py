"""Yahoo Finance data provider implementation."""

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from quantdash.config import get_logger

from .base import DataProvider, normalize_dataframe

logger = get_logger("data.yfinance")


class YFinanceProvider(DataProvider):
    """Data provider using yfinance library."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the yfinance provider.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def name(self) -> str:
        return "yfinance"

    def download_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Download OHLCV bars from Yahoo Finance.

        CRITICAL: Uses threads=False to avoid Streamlit concurrency issues.
        """
        if not symbols:
            return pd.DataFrame()

        # Convert timeframe to yfinance interval format
        interval = self._to_yf_interval(timeframe)

        # Build ticker string
        tickers = symbols if len(symbols) > 1 else symbols[0]

        for attempt in range(self._max_retries):
            try:
                logger.info(
                    f"Downloading {tickers} from {start} to {end} ({interval})"
                )

                df = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=adjusted,
                    threads=False,  # CRITICAL for Streamlit
                    progress=False,
                )

                if df.empty:
                    logger.warning(f"No data returned for {tickers}")
                    return pd.DataFrame()

                # Normalize column names
                df = normalize_dataframe(df)

                # Ensure timezone-aware index
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                logger.info(f"Downloaded {len(df)} bars for {tickers}")
                return df

            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise

        return pd.DataFrame()

    def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """
        Get fundamental data from Yahoo Finance.

        Returns dict with P/E, EPS, P/B, dividend yield, ROE, debt-to-equity,
        free cash flow, PEG, enterprise value, beta, and market cap.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cash_flow": info.get("freeCashflow"),
                "peg_ratio": info.get("pegRatio"),
                "enterprise_value": info.get("enterpriseValue"),
                "beta": info.get("beta"),
                "market_cap": info.get("marketCap"),
            }

        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            return None

    @staticmethod
    def _to_yf_interval(timeframe: str) -> str:
        """Convert our timeframe format to yfinance interval."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # yfinance doesn't support 4h, use 1h
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
        }
        return mapping.get(timeframe, "1d")


# Default provider instance
default_provider = YFinanceProvider()
