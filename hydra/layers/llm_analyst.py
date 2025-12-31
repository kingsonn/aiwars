"""
HYDRA LLM News Analyst

Qualitative news analysis using Groq Qwen3-32B for:
- Per-pair sentiment analysis (bullish/bearish/exit)
- Runs independently every 30 minutes
- Helps Layer 4 make trading decisions

Rate Limits (Groq Qwen3-32B):
- 60 requests/minute
- 1,000 requests/day
- 6,000 tokens/minute

Strategy:
- Scan all 8 pairs every 30 min = 8 calls/scan
- 48 scans/day × 8 pairs = 384 calls/day (safe under 1K limit)
- Call once on initialization, then every 30 minutes
"""

from __future__ import annotations

import asyncio
import os
import time
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import aiohttp
from loguru import logger


class Sentiment(Enum):
    """Market sentiment classification."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class NewsItem:
    """A single news item."""
    title: str
    summary: str
    source: str
    timestamp: datetime
    symbols: list[str] = field(default_factory=list)
    sentiment: Optional[Sentiment] = None
    relevance_score: float = 0.0


@dataclass
class PairAnalysis:
    """LLM analysis result for a single trading pair."""
    symbol: str
    sentiment: Sentiment
    action: Literal["long", "short", "hold", "exit"]  # What to do
    confidence: float  # 0-1 how confident the LLM is
    reasoning: str  # Brief explanation
    news_summary: str  # Key news affecting this pair
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_bullish(self) -> bool:
        return self.sentiment in [Sentiment.BULLISH, Sentiment.VERY_BULLISH]
    
    @property
    def is_bearish(self) -> bool:
        return self.sentiment in [Sentiment.BEARISH, Sentiment.VERY_BEARISH]
    
    @property
    def should_long(self) -> bool:
        return self.action == "long"
    
    @property
    def should_short(self) -> bool:
        return self.action == "short"
    
    @property
    def should_exit(self) -> bool:
        return self.action == "exit"
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sentiment": self.sentiment.value,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "news_summary": self.news_summary,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def get_log_string(self) -> str:
        """Get a formatted string for verbose logging."""
        action_emoji = {"long": "📈", "short": "📉", "hold": "⏸️", "exit": "🚪"}.get(self.action, "❓")
        return (
            f"{action_emoji} {self.action.upper()} | {self.sentiment.value} ({self.confidence:.0%}) | "
            f"{self.reasoning[:60]}..."
        )


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(
        self,
        requests_per_minute: int = 50,  # Leave buffer below 60
        requests_per_day: int = 900,     # Leave buffer below 1000
        tokens_per_minute: int = 5000,   # Leave buffer below 6000
    ):
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day
        self.tpm_limit = tokens_per_minute
        
        self._minute_requests: list[float] = []
        self._day_requests: list[float] = []
        self._minute_tokens: list[tuple[float, int]] = []
        
    def can_request(self, estimated_tokens: int = 1000) -> tuple[bool, str]:
        """Check if we can make a request."""
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400
        
        # Clean old entries
        self._minute_requests = [t for t in self._minute_requests if t > minute_ago]
        self._day_requests = [t for t in self._day_requests if t > day_ago]
        self._minute_tokens = [(t, tok) for t, tok in self._minute_tokens if t > minute_ago]
        
        # Check limits
        if len(self._minute_requests) >= self.rpm_limit:
            return False, f"Rate limit: {len(self._minute_requests)}/{self.rpm_limit} requests/min"
        
        if len(self._day_requests) >= self.rpd_limit:
            return False, f"Daily limit: {len(self._day_requests)}/{self.rpd_limit} requests/day"
        
        tokens_used = sum(tok for _, tok in self._minute_tokens)
        if tokens_used + estimated_tokens > self.tpm_limit:
            return False, f"Token limit: {tokens_used}/{self.tpm_limit} tokens/min"
        
        return True, "OK"
    
    def record_request(self, tokens_used: int = 1000):
        """Record a request."""
        now = time.time()
        self._minute_requests.append(now)
        self._day_requests.append(now)
        self._minute_tokens.append((now, tokens_used))
    
    def get_stats(self) -> dict:
        """Get current rate limit stats."""
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400
        
        minute_reqs = len([t for t in self._minute_requests if t > minute_ago])
        day_reqs = len([t for t in self._day_requests if t > day_ago])
        minute_tokens = sum(tok for t, tok in self._minute_tokens if t > minute_ago)
        
        return {
            "requests_per_minute": f"{minute_reqs}/{self.rpm_limit}",
            "requests_per_day": f"{day_reqs}/{self.rpd_limit}",
            "tokens_per_minute": f"{minute_tokens}/{self.tpm_limit}",
        }


class CryptoNewsScraper:
    """Scraper for crypto news from free sources."""
    
    SYMBOL_KEYWORDS = {
        "cmt_btcusdt": ["bitcoin", "btc", "₿"],
        "cmt_ethusdt": ["ethereum", "eth", "ether"],
        "cmt_solusdt": ["solana", "sol"],
        "cmt_dogeusdt": ["dogecoin", "doge"],
        "cmt_xrpusdt": ["xrp", "ripple"],
        "cmt_adausdt": ["cardano", "ada"],
        "cmt_bnbusdt": ["bnb", "binance coin"],
        "cmt_ltcusdt": ["litecoin", "ltc"],
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[datetime, list[NewsItem]]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_news(self, hours_back: int = 1) -> list[NewsItem]:
        """Fetch recent crypto news from multiple sources."""
        # Check cache
        cache_key = f"news_{hours_back}"
        if cache_key in self._cache:
            cache_time, cached_news = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self._cache_ttl:
                return cached_news
        
        news_items = []
        
        # Try CryptoPanic API (free tier)
        try:
            cryptopanic_news = await self._fetch_cryptopanic()
            news_items.extend(cryptopanic_news)
        except Exception as e:
            logger.debug(f"CryptoPanic fetch failed: {e}")
        
        # Try CoinGecko news
        try:
            coingecko_news = await self._fetch_coingecko_news()
            news_items.extend(coingecko_news)
        except Exception as e:
            logger.debug(f"CoinGecko news fetch failed: {e}")
        
        # Filter to recent news
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        news_items = [n for n in news_items if n.timestamp > cutoff]
        
        # Deduplicate by title similarity
        seen_titles = set()
        unique_news = []
        for item in news_items:
            title_key = item.title[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        # Sort by timestamp
        unique_news.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Cache result
        self._cache[cache_key] = (datetime.now(timezone.utc), unique_news[:20])
        
        return unique_news[:20]
    
    async def _fetch_cryptopanic(self) -> list[NewsItem]:
        """Fetch from CryptoPanic public API."""
        session = await self._get_session()
        news = []
        
        try:
            # Public endpoint (no auth needed, limited results)
            url = "https://cryptopanic.com/api/free/v1/posts/?public=true"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("results", [])[:15]:
                        try:
                            timestamp = datetime.fromisoformat(
                                item.get("published_at", "").replace("Z", "+00:00")
                            )
                        except:
                            timestamp = datetime.now(timezone.utc)
                        
                        news_item = NewsItem(
                            title=item.get("title", ""),
                            summary=item.get("title", ""),  # Free tier has no summary
                            source="CryptoPanic",
                            timestamp=timestamp,
                            symbols=self._extract_symbols(item.get("title", "")),
                        )
                        news.append(news_item)
        except Exception as e:
            logger.debug(f"CryptoPanic error: {e}")
        
        return news
    
    async def _fetch_coingecko_news(self) -> list[NewsItem]:
        """Fetch trending/news from CoinGecko."""
        session = await self._get_session()
        news = []
        
        try:
            # Get trending coins as a proxy for news
            url = "https://api.coingecko.com/api/v3/search/trending"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for coin in data.get("coins", [])[:5]:
                        item = coin.get("item", {})
                        news_item = NewsItem(
                            title=f"{item.get('name', '')} trending - Rank #{item.get('market_cap_rank', 'N/A')}",
                            summary=f"Market cap rank: {item.get('market_cap_rank', 'N/A')}, Score: {item.get('score', 0)}",
                            source="CoinGecko Trending",
                            timestamp=datetime.now(timezone.utc),
                            symbols=self._extract_symbols(item.get("name", "") + " " + item.get("symbol", "")),
                        )
                        news.append(news_item)
        except Exception as e:
            logger.debug(f"CoinGecko error: {e}")
        
        return news
    
    def _extract_symbols(self, text: str) -> list[str]:
        """Extract relevant trading symbols from text."""
        text_lower = text.lower()
        symbols = []
        
        for symbol, keywords in self.SYMBOL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    symbols.append(symbol)
                    break
        
        return symbols


class LLMNewsAnalyst:
    """
    LLM-powered news analyst for trading decisions.
    
    Runs independently every 30 minutes to analyze news for each pair.
    Provides bullish/bearish/hold/exit recommendations to Layer 4.
    """
    
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # All trading pairs to analyze
    PAIRS = [
        "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_bnbusdt",
        "cmt_adausdt", "cmt_xrpusdt", "cmt_ltcusdt", "cmt_dogeusdt"
    ]
    
    PAIR_NAMES = {
        "cmt_btcusdt": "Bitcoin (BTC)",
        "cmt_ethusdt": "Ethereum (ETH)",
        "cmt_solusdt": "Solana (SOL)",
        "cmt_bnbusdt": "BNB",
        "cmt_adausdt": "Cardano (ADA)",
        "cmt_xrpusdt": "XRP",
        "cmt_ltcusdt": "Litecoin (LTC)",
        "cmt_dogeusdt": "Dogecoin (DOGE)",
    }
    
    def __init__(self, api_key: str = "", scan_interval_minutes: int = 30):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "qwen/qwen3-32b")
        self.scan_interval = scan_interval_minutes
        
        self.rate_limiter = RateLimiter()
        self.news_scraper = CryptoNewsScraper()
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_scan_time: Optional[datetime] = None
        self._pair_analysis: dict[str, PairAnalysis] = {}  # symbol -> latest analysis
        self._initialized = False
        
        # Stats
        self.total_calls = 0
        self.total_tokens = 0
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set - LLM analysis disabled")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        await self.news_scraper.close()
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _call_llm(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        """Make a rate-limited call to Groq API."""
        if not self.api_key:
            return None
        
        # Check rate limits
        can_call, reason = self.rate_limiter.can_request(max_tokens)
        if not can_call:
            logger.warning(f"LLM rate limited: {reason}")
            return None
        
        session = await self._get_session()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a crypto trading analyst. Analyze news and provide trading recommendations. Always respond with valid JSON only, no markdown."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            
            async with session.post(self.GROQ_API_URL, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Record usage
                    usage = data.get("usage", {})
                    tokens_used = usage.get("total_tokens", max_tokens)
                    self.rate_limiter.record_request(tokens_used)
                    self.total_calls += 1
                    self.total_tokens += tokens_used
                    
                    return data["choices"][0]["message"]["content"]
                elif resp.status == 429:
                    logger.warning("Groq rate limit hit, backing off")
                    await asyncio.sleep(5)
                    return None
                else:
                    text = await resp.text()
                    logger.error(f"Groq API error {resp.status}: {text[:200]}")
                    return None
                    
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def should_scan(self) -> bool:
        """Check if it's time to run a new scan."""
        if not self._initialized:
            return True
        if self._last_scan_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_scan_time).total_seconds() / 60
        return elapsed >= self.scan_interval
    
    def get_pair_analysis(self, symbol: str) -> Optional[PairAnalysis]:
        """Get the latest LLM analysis for a trading pair."""
        return self._pair_analysis.get(symbol.lower())
    
    def get_all_analyses(self) -> dict[str, PairAnalysis]:
        """Get all pair analyses."""
        return self._pair_analysis.copy()
    
    async def initialize(self) -> dict[str, PairAnalysis]:
        """
        Run initial scan on bot startup.
        Analyzes all pairs and caches results.
        """
        if not self.api_key:
            logger.warning("LLM disabled - no API key")
            return {}
        
        logger.info("🧠 [LLM] Running initial news scan for all pairs...")
        return await self.scan_all_pairs(force=True)
    
    async def scan_all_pairs(self, force: bool = False) -> dict[str, PairAnalysis]:
        """
        Scan news and analyze all trading pairs.
        Runs every 30 minutes or on force=True.
        
        Returns dict of symbol -> PairAnalysis
        """
        if not self.api_key:
            return {}
        
        now = datetime.now(timezone.utc)
        
        # Check if we should scan
        if not force and not self.should_scan():
            return self._pair_analysis
        
        logger.info(f"🧠 [LLM] Scanning news for {len(self.PAIRS)} pairs...")
        
        # Fetch all news (last hour)
        all_news = await self.news_scraper.fetch_news(hours_back=1)
        news_text = "\n".join([f"- {n.title}" for n in all_news[:15]]) if all_news else "No recent news"
        
        # Analyze each pair
        for symbol in self.PAIRS:
            try:
                analysis = await self._analyze_pair(symbol, news_text)
                if analysis:
                    self._pair_analysis[symbol] = analysis
                    logger.info(f"  [{self.PAIR_NAMES.get(symbol, symbol)[:3]}] {analysis.get_log_string()}")
                
                # Small delay between calls to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
        
        self._last_scan_time = now
        self._initialized = True
        
        logger.info(f"🧠 [LLM] Scan complete. {len(self._pair_analysis)} pairs analyzed. Next scan in {self.scan_interval} min.")
        
        return self._pair_analysis
    
    async def _analyze_pair(self, symbol: str, news_text: str) -> Optional[PairAnalysis]:
        """Analyze a single pair based on news."""
        coin_name = self.PAIR_NAMES.get(symbol, symbol)
        short_name = coin_name.split()[0] if " " in coin_name else coin_name[:3]
        
        prompt = f"""Based on recent crypto news, provide trading recommendation for {coin_name}.

Recent News (last hour):
{news_text}

What should a trader do with {short_name}? Respond with JSON only:
{{
    "sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "action": "long|short|hold|exit",
    "confidence": 0.0 to 1.0,
    "reasoning": "1-2 sentence explanation",
    "news_summary": "key news affecting {short_name}"
}}"""
        
        response = await self._call_llm(prompt, max_tokens=300)
        
        if not response:
            return None
        
        try:
            # Parse JSON (handle markdown code blocks)
            json_str = response
            if "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    json_str = parts[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
            
            data = json.loads(json_str.strip())
            
            return PairAnalysis(
                symbol=symbol,
                sentiment=Sentiment(data.get("sentiment", "neutral")),
                action=data.get("action", "hold"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No analysis available"),
                news_summary=data.get("news_summary", ""),
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse LLM response for {symbol}: {e}")
            return None
    
    def should_trade(self, symbol: str, proposed_side: str) -> tuple[bool, str]:
        """
        Check if LLM recommends trading this pair in this direction.
        Used by Layer 4 to gate trades.
        
        Returns: (should_proceed, reason)
        """
        analysis = self.get_pair_analysis(symbol)
        
        if not analysis:
            return True, "No LLM analysis available"
        
        # Check if LLM says exit
        if analysis.should_exit:
            return False, f"LLM recommends EXIT: {analysis.reasoning}"
        
        # Check alignment with proposed trade
        if proposed_side == "long":
            if analysis.is_bearish:
                return False, f"LLM bearish on {symbol}: {analysis.reasoning}"
            if analysis.should_short:
                return False, f"LLM recommends SHORT not LONG: {analysis.reasoning}"
        elif proposed_side == "short":
            if analysis.is_bullish:
                return False, f"LLM bullish on {symbol}: {analysis.reasoning}"
            if analysis.should_long:
                return False, f"LLM recommends LONG not SHORT: {analysis.reasoning}"
        
        return True, f"LLM: {analysis.action.upper()} ({analysis.sentiment.value})"
    
    def get_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "pairs_analyzed": len(self._pair_analysis),
            "rate_limits": self.rate_limiter.get_stats(),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "initialized": self._initialized,
        }


# Singleton instance
_llm_analyst: Optional[LLMNewsAnalyst] = None


def get_llm_analyst() -> LLMNewsAnalyst:
    """Get or create the LLM analyst singleton."""
    global _llm_analyst
    if _llm_analyst is None:
        _llm_analyst = LLMNewsAnalyst()
    return _llm_analyst
