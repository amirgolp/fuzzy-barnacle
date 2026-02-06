#!/usr/bin/env python3
"""
Headless Screener Runner for Raspberry Pi / Docker.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quantdash.alerts.telegram import TelegramBot
from quantdash.data import YFinanceProvider, get_symbol_data
from quantdash.features import screen_symbol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pi_screener")

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "3600"))  # Default 1 hour
SYMBOLS = os.getenv(
    "WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,BTC-USD,ETH-USD"
).split(",")


async def main():
    """Main screening loop."""
    logger.info("Starting Pi Screener...")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not found. Alerts will be disabled.")

    bot = TelegramBot(TELEGRAM_TOKEN or "", TELEGRAM_CHAT_ID or "")
    provider = YFinanceProvider()

    # Sending startup message
    await bot.send_message(f"🚀 Pi Screener Started\nFollowing: {', '.join(SYMBOLS)}")

    while True:
        logger.info("Running scan...")
        
        try:
            # Download data for all symbols at once (more efficient)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            clean_symbols = [s.strip().upper() for s in SYMBOLS if s.strip()]
            
            # Note: YFinanceProvider might need updates to handle bulk properly or just loop
            # For simplicity in this script, we loop
            
            for symbol in clean_symbols:
                try:
                    logger.info(f"Screening {symbol}...")
                    df = provider.download_bars([symbol], "1d", start_date, end_date)
                    df = get_symbol_data(df, symbol)
                    
                    if df.empty:
                        logger.warning(f"No data for {symbol}")
                        continue
                        
                    result = screen_symbol(df, symbol=symbol)
                    
                    # Alert logic
                    # 1. Strong signals (Strong Buy/Sell)
                    # 2. High confidence (> 70%)
                    if (
                        "strong" in result.recommendation.lower() 
                        or result.confidence > 0.7
                        or result.score > 20
                        or result.score < -20
                    ):
                        logger.info(f"🔥 Alert triggered for {symbol}: {result.recommendation}")
                        await bot.send_screener_result(result)
                        
                except Exception as e:
                    logger.error(f"Error screening {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Scan cycle failed: {e}")

        logger.info(f"Scan complete. Sleeping for {CHECK_INTERVAL_SECONDS}s...")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
