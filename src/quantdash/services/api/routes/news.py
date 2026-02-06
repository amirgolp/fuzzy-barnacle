"""News feed and sentiment analysis routes."""

import re
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quantdash.data.news_cache import get_news_cache
from quantdash.config import get_logger

router = APIRouter()
logger = get_logger("api.news")

# Simple keyword-based sentiment scorer
POSITIVE_WORDS = {
    "surge", "surges", "rally", "rallies", "gain", "gains", "jump", "jumps", "soar", "soars",
    "rise", "rises", "climbs", "up", "higher", "high", "bullish", "bull", "positive", "strong",
    "strength", "growth", "profit", "profits", "beat", "beats", "record", "upgrade", "upgrades",
    "outperform", "buy", "boost", "boosts", "recovery", "recover", "optimism", "optimistic",
    "breakout", "momentum", "upside", "opportunity",
}
NEGATIVE_WORDS = {
    "crash", "crashes", "plunge", "plunges", "drop", "drops", "fall", "falls", "decline", "declines",
    "sink", "sinks", "tumble", "tumbles", "down", "lower", "low", "bearish", "bear", "negative",
    "weak", "weakness", "loss", "losses", "miss", "misses", "downgrade", "downgrades", "sell",
    "selloff", "sell-off", "risk", "risks", "fear", "fears", "recession", "crisis", "warning",
    "underperform", "cut", "cuts", "downside", "concern", "concerns",
}


def _sentiment_score(text: str) -> float:
    """Return sentiment score from -1.0 (very negative) to +1.0 (very positive)."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def _sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "Bullish"
    elif score <= -0.3:
        return "Bearish"
    return "Neutral"


class NewsItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    source: str
    sentiment_score: float
    sentiment_label: str


RSS_FEEDS = {
    "google": "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
    "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
}


async def _fetch_rss(url: str) -> list[dict]:
    """Fetch and parse RSS feed."""
    items = []
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "QuantDash/1.0"})
            resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)
        # Standard RSS 2.0
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                items.append({
                    "title": title_el.text.strip(),
                    "link": link_el.text.strip() if link_el is not None and link_el.text else "",
                    "published": pub_el.text.strip() if pub_el is not None and pub_el.text else None,
                })
    except Exception:
        pass
    return items


@router.get("/feed")
async def get_news_feed(
    symbol: str = Query(..., description="Symbol to search news for"),
    limit: int = Query(5, description="Number of news items", ge=1, le=50),
    use_cache: bool = Query(True, description="Use cached news if available"),
) -> list[NewsItem]:
    """Fetch latest news for a symbol with sentiment analysis."""
    cache = get_news_cache()
    cached_articles: list[dict] = []

    # Check cache first
    if use_cache:
        cached_articles, is_valid = cache.get_cached_news(symbol, limit=limit)
        if is_valid and cached_articles:
            logger.debug(f"Returning {len(cached_articles)} cached articles for {symbol}")
            return [NewsItem(**a) for a in cached_articles[:limit]]

    # Fetch fresh news
    all_items: list[dict] = []
    for source_name, url_template in RSS_FEEDS.items():
        # Strip yfinance suffixes for search
        clean_sym = symbol.replace("=F", "").replace("=X", "").replace("-USD", "")
        url = url_template.format(symbol=clean_sym)
        items = await _fetch_rss(url)
        for item in items:
            item["source"] = source_name
        all_items.extend(items)

    if not all_items:
        # Return stale cache if available
        if cached_articles:
            logger.debug(f"Returning stale cache for {symbol} (fetch failed)")
            return [NewsItem(**a) for a in cached_articles[:limit]]
        return []

    # Deduplicate by title
    seen = set()
    unique = []
    for item in all_items:
        key = item["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Score articles
    results = []
    articles_to_cache = []
    for item in unique:
        score = _sentiment_score(item["title"])
        label = _sentiment_label(score)
        results.append(NewsItem(
            title=item["title"],
            link=item["link"],
            published=item.get("published"),
            source=item["source"],
            sentiment_score=score,
            sentiment_label=label,
        ))
        articles_to_cache.append({
            "title": item["title"],
            "link": item["link"],
            "published": item.get("published"),
            "source": item["source"],
            "sentiment_score": score,
            "sentiment_label": label,
        })

    # Store in cache
    if articles_to_cache:
        cache.store_news(symbol, articles_to_cache)
        logger.info(f"Cached {len(articles_to_cache)} articles for {symbol}")

    return results[:limit]


@router.get("/sentiment")
async def get_sentiment_summary(
    symbol: str = Query(..., description="Symbol to analyze"),
    gauge_count: int = Query(20, description="Number of articles for sentiment gauge", ge=5, le=50),
    display_count: int = Query(5, description="Number of articles to display", ge=1, le=50),
) -> dict:
    """
    Get aggregated sentiment from latest news.

    - Fetches `gauge_count` articles for sentiment calculation (default 20)
    - Returns `display_count` most recent articles for display (default 5)
    - Gauge score is weighted: recent articles and strong sentiment have more impact
    """
    all_items = await get_news_feed(symbol=symbol, limit=gauge_count)
    if not all_items:
        return {"symbol": symbol, "score": 0.0, "label": "Neutral", "news_count": 0, "items": [], "gauge_news_count": 0}

    # Calculate weighted sentiment: recent + strong sentiment articles matter more
    total_weight = 0.0
    weighted_score = 0.0
    for i, item in enumerate(all_items):
        # Recency weight: first article gets weight 1.0, last gets 0.5
        recency_weight = 1.0 - (i / len(all_items)) * 0.5
        # Strength weight: strong sentiments (|score| > 0.3) get 1.5x weight
        strength_weight = 1.5 if abs(item.sentiment_score) > 0.3 else 1.0
        weight = recency_weight * strength_weight
        weighted_score += item.sentiment_score * weight
        total_weight += weight

    avg_score = round(weighted_score / total_weight, 2) if total_weight > 0 else 0.0

    # Count sentiments for gauge
    bullish_count = sum(1 for i in all_items if i.sentiment_label == "Bullish")
    bearish_count = sum(1 for i in all_items if i.sentiment_label == "Bearish")
    neutral_count = len(all_items) - bullish_count - bearish_count

    return {
        "symbol": symbol,
        "score": avg_score,
        "label": _sentiment_label(avg_score),
        "news_count": len(all_items),
        "gauge_news_count": len(all_items),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "items": [i.model_dump() for i in all_items[:display_count]],
    }
