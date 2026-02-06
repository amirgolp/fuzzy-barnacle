"""
Telegram Alert Logic for QuantDash.
"""

import logging
from typing import Optional

import httpx

from quantdash.features.screener_models import ScreenerResult

logger = logging.getLogger(__name__)


class TelegramBot:
    """Simple Telegram Bot wrapper for sending alerts."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    async def send_message(self, text: str) -> bool:
        """Send a simple text message."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not set. Skipping message.")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_screener_result(self, result: ScreenerResult) -> bool:
        """Format and send a screener result."""
        icon = "🟢" if result.recommendation in ["buy", "strong_buy"] else "🔴"
        if result.recommendation == "hold":
            icon = "⚪"

        lines = [
            f"{icon} *{result.symbol}* - {result.recommendation.upper().replace('_', ' ')}",
            f"Score: `{result.score:.1f}` | Conf: `{int(result.confidence * 100)}%`",
            f"Price: `${result.current_price:.2f}`" if result.current_price else "",
            "",
            "*Signals:*",
        ]

        for s in result.signals:
            sig_icon = "📈" if s.is_bullish else "📉"
            lines.append(f"{sig_icon} {s.description}")

        if result.notes:
            lines.append("")
            lines.append("*Notes:*")
            for n in result.notes:
                lines.append(f"• {n}")

        message = "\n".join(filter(None, lines))
        return await self.send_message(message)
