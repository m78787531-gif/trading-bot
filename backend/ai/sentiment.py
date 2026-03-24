"""
ai/sentiment.py
─────────────────────────────────────────────────────────────────────────────
Multi-source sentiment analysis layer.

Sources:
  1. CryptoPanic news API  (free tier, no auth required for basic use)
  2. Reddit (pushshift / public JSON endpoints)
  3. Twitter v2 API       (requires Bearer token in env)
  4. Fallback: keyword-scored headline scraping

Returns a unified SentimentResult per symbol with:
  • composite_score  –1.0 (very bearish) → +1.0 (very bullish)
  • confidence       0–1
  • breakdown per source
  • top bullish / bearish headlines

Results are cached in-process for 15 minutes to avoid rate limits.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config — all from environment
# ─────────────────────────────────────────────────────────────────────────────

TWITTER_BEARER  = os.getenv("TWITTER_BEARER_TOKEN", "")
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")   # free at cryptopanic.com
REDDIT_CLIENT   = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET   = os.getenv("REDDIT_SECRET", "")

CACHE_TTL_SECONDS = 900   # 15 minutes

# In-memory cache: key → (timestamp, SentimentResult)
_cache: dict[str, tuple[float, "SentimentResult"]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword lexicons
# ─────────────────────────────────────────────────────────────────────────────

BULLISH_WORDS = {
    "bullish", "moon", "breakout", "surge", "rally", "pump", "accumulate",
    "buy", "long", "bull", "uptrend", "recover", "rebound", "support",
    "institutional", "adoption", "partnership", "upgrade", "approval",
    "all-time high", "ath", "breakout", "soar", "skyrocket", "outperform",
    "undervalued", "gem", "nfa", "potential", "golden", "opportunity",
    "demand", "inflow", "halving", "etf approved",
}

BEARISH_WORDS = {
    "bearish", "crash", "dump", "sell", "short", "downtrend", "breakdown",
    "resistance", "overvalued", "ban", "regulate", "hack", "exploit",
    "rug", "scam", "fraud", "liquidation", "outflow", "fear", "panic",
    "fud", "correction", "collapse", "plunge", "plummet", "warning",
    "investigation", "lawsuit", "sec", "crackdown", "sanction",
}

STRONG_MULTIPLIER_WORDS = {
    "massive", "huge", "enormous", "critical", "major", "historic",
    "all-time", "record", "breaking", "extreme", "unprecedented",
}


def _score_text(text: str) -> float:
    """
    Simple lexicon-based scorer.
    Returns –1.0 to +1.0.
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w[\w-]*\b', text_lower)
    word_set = set(words)

    # Count phrase matches too
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    strong = sum(1 for w in STRONG_MULTIPLIER_WORDS if w in text_lower)

    multiplier = 1.0 + (strong * 0.2)
    net = (bull - bear) * multiplier

    # Normalise: assume ±5 is very strong
    score = max(-1.0, min(1.0, net / 5.0))
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceSentiment:
    source: str
    score: float             # –1 to +1
    num_items: int
    top_bullish: list[str]   # top 3 bullish headlines
    top_bearish: list[str]   # top 3 bearish headlines
    confidence: float        # 0–1 (based on volume and freshness)
    error: Optional[str] = None


@dataclass
class SentimentResult:
    symbol: str
    composite_score: float       # –1.0 to +1.0
    label: str                   # very_bearish | bearish | neutral | bullish | very_bullish
    confidence: float            # 0–1
    sources: list[SourceSentiment]
    top_headlines: list[str]
    fetched_at: float            # unix timestamp

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _label(score: float) -> str:
    if score <= -0.6:   return "very_bearish"
    if score <= -0.2:   return "bearish"
    if score <= 0.2:    return "neutral"
    if score <= 0.6:    return "bullish"
    return "very_bullish"


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: CryptoPanic
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_cryptopanic(symbol: str) -> SourceSentiment:
    """
    CryptoPanic free API — top crypto news with vote counts.
    https://cryptopanic.com/api/v1/posts/
    """
    coin = symbol.replace("USDT", "").replace("BUSD", "").upper()
    url = "https://cryptopanic.com/api/v1/posts/"
    params: dict = {
        "currencies": coin,
        "kind": "news",
        "public": "true",
        "filter": "hot",
    }
    if CRYPTOPANIC_KEY:
        params["auth_token"] = CRYPTOPANIC_KEY

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return SourceSentiment("cryptopanic", 0.0, 0, [], [], 0.0, f"HTTP {r.status_code}")
            data = r.json()
    except Exception as e:
        return SourceSentiment("cryptopanic", 0.0, 0, [], [], 0.0, str(e))

    items = data.get("results", [])[:30]
    if not items:
        return SourceSentiment("cryptopanic", 0.0, 0, [], [], 0.2)

    scored: list[tuple[float, str]] = []
    for item in items:
        title = item.get("title", "")
        # Use CryptoPanic's own votes as weighting
        votes = item.get("votes", {})
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)
        vote_signal = (positive - negative) / max(positive + negative, 1)

        lex_score = _score_text(title)
        # Blend lexicon (70%) + vote signal (30%)
        combined = lex_score * 0.7 + vote_signal * 0.3
        scored.append((combined, title))

    avg_score = sum(s for s, _ in scored) / len(scored)
    bullish = sorted([t for s, t in scored if s > 0.1], key=lambda t: _score_text(t), reverse=True)[:3]
    bearish = sorted([t for s, t in scored if s < -0.1], key=lambda t: _score_text(t))[:3]

    confidence = min(0.9, 0.3 + len(items) / 30 * 0.6)

    return SourceSentiment(
        source="cryptopanic",
        score=round(avg_score, 4),
        num_items=len(items),
        top_bullish=bullish,
        top_bearish=bearish,
        confidence=round(confidence, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: Reddit
# ─────────────────────────────────────────────────────────────────────────────

CRYPTO_SUBREDDITS = ["CryptoCurrency", "Bitcoin", "ethtrader", "CryptoMarkets", "SatoshiStreetBets"]

async def _fetch_reddit(symbol: str) -> SourceSentiment:
    """
    Reddit public JSON API — no authentication required for read-only.
    Searches top posts across crypto subreddits.
    """
    coin = symbol.replace("USDT", "").replace("BUSD", "")

    headers = {"User-Agent": "TradingBot/1.0"}
    all_posts: list[tuple[float, str]] = []

    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as client:
            for subreddit in CRYPTO_SUBREDDITS[:3]:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": coin,
                    "sort": "top",
                    "t": "day",
                    "limit": 15,
                    "restrict_sr": "on",
                }
                try:
                    r = await client.get(url, params=params)
                    if r.status_code != 200:
                        continue
                    posts = r.json().get("data", {}).get("children", [])
                    for p in posts:
                        d = p.get("data", {})
                        title = d.get("title", "")
                        body = d.get("selftext", "")[:200]
                        upvote_ratio = d.get("upvote_ratio", 0.5)
                        score_r = d.get("score", 0)

                        # Combine lexicon + upvote ratio
                        lex = _score_text(title + " " + body)
                        vote_signal = (upvote_ratio - 0.5) * 2  # –1 to +1
                        combined = lex * 0.8 + vote_signal * 0.2

                        # Weight by post score (more upvotes = more weight)
                        weight = min(1.0, (score_r + 100) / 1000)
                        all_posts.append((combined * weight, title))
                except Exception:
                    continue
    except Exception as e:
        return SourceSentiment("reddit", 0.0, 0, [], [], 0.0, str(e))

    if not all_posts:
        return SourceSentiment("reddit", 0.0, 0, [], [], 0.2)

    avg = sum(s for s, _ in all_posts) / len(all_posts)
    bullish = [t for s, t in sorted(all_posts, reverse=True) if s > 0][:3]
    bearish = [t for s, t in sorted(all_posts) if s < 0][:3]

    confidence = min(0.85, 0.2 + len(all_posts) / 30 * 0.65)

    return SourceSentiment(
        source="reddit",
        score=round(avg, 4),
        num_items=len(all_posts),
        top_bullish=bullish,
        top_bearish=bearish,
        confidence=round(confidence, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: Twitter / X v2 API
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_twitter(symbol: str) -> SourceSentiment:
    """
    Twitter v2 recent search. Requires TWITTER_BEARER_TOKEN env var.
    Falls back gracefully if not configured.
    """
    if not TWITTER_BEARER:
        return SourceSentiment("twitter", 0.0, 0, [], [], 0.0, "No bearer token configured")

    coin = symbol.replace("USDT", "")
    query = f"#{coin} OR ${coin} lang:en -is:retweet"
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    params = {
        "query": query,
        "max_results": 50,
        "tweet.fields": "public_metrics,created_at",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers, params=params)
            if r.status_code == 401:
                return SourceSentiment("twitter", 0.0, 0, [], [], 0.0, "Invalid bearer token")
            if r.status_code != 200:
                return SourceSentiment("twitter", 0.0, 0, [], [], 0.0, f"HTTP {r.status_code}")
            data = r.json()
    except Exception as e:
        return SourceSentiment("twitter", 0.0, 0, [], [], 0.0, str(e))

    tweets = data.get("data", [])
    if not tweets:
        return SourceSentiment("twitter", 0.0, 0, [], [], 0.2)

    scored: list[tuple[float, str]] = []
    for tweet in tweets:
        text = tweet.get("text", "")
        metrics = tweet.get("public_metrics", {})
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)

        lex = _score_text(text)
        # Weight by engagement
        engagement_weight = min(1.5, 1.0 + (likes + retweets * 2) / 500)
        scored.append((lex * engagement_weight, text[:140]))

    avg = sum(s for s, _ in scored) / len(scored)
    bullish = [t for s, t in sorted(scored, reverse=True) if s > 0][:3]
    bearish = [t for s, t in sorted(scored) if s < 0][:3]
    confidence = min(0.9, 0.3 + len(tweets) / 50 * 0.6)

    return SourceSentiment(
        source="twitter",
        score=round(avg, 4),
        num_items=len(tweets),
        top_bullish=bullish,
        top_bearish=bearish,
        confidence=round(confidence, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_WEIGHTS = {
    "cryptopanic": 0.40,   # most reliable crypto-specific news
    "reddit":      0.35,   # community sentiment
    "twitter":     0.25,   # real-time but noisy
}


async def get_sentiment(symbol: str, force_refresh: bool = False) -> SentimentResult:
    """
    Fetch and aggregate sentiment from all sources.
    Results are cached for CACHE_TTL_SECONDS.
    """
    cache_key = symbol.upper()
    now = time.time()

    if not force_refresh and cache_key in _cache:
        ts, cached = _cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            logger.debug(f"Sentiment cache hit for {symbol}")
            return cached

    logger.info(f"Fetching sentiment for {symbol}...")

    # Fetch all sources concurrently
    results = await asyncio.gather(
        _fetch_cryptopanic(symbol),
        _fetch_reddit(symbol),
        _fetch_twitter(symbol),
        return_exceptions=True,
    )

    sources: list[SourceSentiment] = []
    for r in results:
        if isinstance(r, SourceSentiment):
            sources.append(r)
        else:
            logger.warning(f"Sentiment source error: {r}")

    if not sources:
        result = SentimentResult(
            symbol=symbol, composite_score=0.0, label="neutral",
            confidence=0.0, sources=[], top_headlines=[], fetched_at=now,
        )
        _cache[cache_key] = (now, result)
        return result

    # Weighted average, weighted further by per-source confidence
    total_weight = 0.0
    weighted_sum = 0.0
    all_headlines: list[tuple[float, str]] = []

    for src in sources:
        if src.error and src.num_items == 0:
            continue
        w = SOURCE_WEIGHTS.get(src.source, 0.2) * src.confidence
        weighted_sum += src.score * w
        total_weight += w
        for h in src.top_bullish:
            all_headlines.append((1.0, h))
        for h in src.top_bearish:
            all_headlines.append((-1.0, h))

    composite = weighted_sum / total_weight if total_weight > 0 else 0.0
    composite = round(max(-1.0, min(1.0, composite)), 4)

    avg_confidence = sum(s.confidence for s in sources) / len(sources) if sources else 0.0

    # Top headlines: mix bull/bear
    top = [h for _, h in sorted(all_headlines, key=lambda x: abs(x[0]), reverse=True)[:6]]

    result = SentimentResult(
        symbol=symbol,
        composite_score=composite,
        label=_label(composite),
        confidence=round(avg_confidence, 3),
        sources=sources,
        top_headlines=top,
        fetched_at=now,
    )

    _cache[cache_key] = (now, result)
    logger.info(f"Sentiment [{symbol}]: {composite:+.3f} ({result.label}) conf={avg_confidence:.2f}")
    return result


def get_sentiment_sync(symbol: str) -> SentimentResult:
    """Synchronous wrapper for use outside async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In FastAPI — create a new coroutine safely
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, get_sentiment(symbol))
                return future.result(timeout=20)
        return loop.run_until_complete(get_sentiment(symbol))
    except Exception as e:
        logger.error(f"Sentiment sync error: {e}")
        return SentimentResult(
            symbol=symbol, composite_score=0.0, label="neutral",
            confidence=0.0, sources=[], top_headlines=[], fetched_at=time.time(),
        )


def clear_cache(symbol: str = None):
    """Clear sentiment cache for a symbol or all symbols."""
    global _cache
    if symbol:
        _cache.pop(symbol.upper(), None)
    else:
        _cache.clear()
