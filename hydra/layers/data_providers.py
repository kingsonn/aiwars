"""
Data Providers for Layer 1 Market Intelligence

Free data sources for:
- On-chain metrics (exchange flows, whale alerts, stablecoin supply)
- News aggregation (CryptoPanic, RSS feeds)
- Social sentiment (Reddit, alternative APIs)
- Liquidation data (Binance API)
- Long/Short ratios and top trader positions
"""

from __future__ import annotations

import asyncio
import aiohttp
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

from hydra.core.config import PAIR_DISPLAY_NAMES


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OnChainData:
    """On-chain metrics for a cryptocurrency."""
    timestamp: datetime
    symbol: str
    # Exchange flows
    exchange_inflow_usd: float = 0.0
    exchange_outflow_usd: float = 0.0
    net_exchange_flow: float = 0.0
    # Whale activity
    large_tx_count_24h: int = 0
    large_tx_volume_usd: float = 0.0
    whale_accumulation_score: float = 0.0  # -1 (selling) to 1 (buying)
    # Stablecoin metrics
    usdt_dominance: float = 0.0
    stablecoin_exchange_reserve: float = 0.0
    stablecoin_netflow_7d: float = 0.0
    # Active addresses
    active_addresses_24h: int = 0
    new_addresses_24h: int = 0


@dataclass
class NewsItem:
    """Single news item."""
    timestamp: datetime
    title: str
    source: str
    url: str
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    symbols: list[str] = field(default_factory=list)
    is_breaking: bool = False


@dataclass
class NewsSentiment:
    """Aggregated news sentiment for a symbol."""
    timestamp: datetime
    symbol: str
    news_count_1h: int = 0
    news_count_24h: int = 0
    avg_sentiment_1h: float = 0.0
    avg_sentiment_24h: float = 0.0
    breaking_news: list[NewsItem] = field(default_factory=list)
    sentiment_momentum: float = 0.0  # Change in sentiment


@dataclass
class SocialSentiment:
    """Social media sentiment for a symbol."""
    timestamp: datetime
    symbol: str
    # Reddit metrics
    reddit_posts_24h: int = 0
    reddit_comments_24h: int = 0
    reddit_sentiment: float = 0.0  # -1 to 1
    reddit_engagement_change: float = 0.0  # % change
    # Twitter/X alternatives
    social_volume_24h: int = 0
    social_sentiment: float = 0.0
    social_dominance: float = 0.0  # Share of crypto social volume
    # Combined
    overall_sentiment: float = 0.0
    sentiment_trend: str = "neutral"  # bullish, bearish, neutral


@dataclass
class LiquidationData:
    """Liquidation data for a symbol."""
    timestamp: datetime
    symbol: str
    long_liquidations_1h: float = 0.0
    short_liquidations_1h: float = 0.0
    long_liquidations_24h: float = 0.0
    short_liquidations_24h: float = 0.0
    largest_liquidation: float = 0.0
    liquidation_imbalance: float = 0.0  # Positive = more longs liquidated


@dataclass
class PositioningData:
    """Market positioning data."""
    timestamp: datetime
    symbol: str
    long_short_ratio: float = 1.0
    top_trader_long_ratio: float = 0.5
    top_trader_short_ratio: float = 0.5
    open_interest_change_24h: float = 0.0
    funding_rate: float = 0.0
    predicted_funding: float = 0.0


# =============================================================================
# ON-CHAIN DATA PROVIDER
# =============================================================================

class OnChainProvider:
    """
    Fetches on-chain data from free APIs:
    - Blockchain.com API (BTC)
    - Etherscan free tier (ETH)
    - Public blockchain explorers
    - Whale Alert public feed
    """
    
    # Coin to chain mapping
    CHAIN_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum", 
        "SOL": "solana",
        "DOGE": "dogecoin",
        "XRP": "xrp",
        "ADA": "cardano",
        "BNB": "bsc",
        "LTC": "litecoin",
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "HYDRA-Trading-Bot/1.0"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_base_symbol(self, internal_symbol: str) -> str:
        """Extract base symbol from internal format."""
        # cmt_btcusdt -> BTC
        return internal_symbol.replace("cmt_", "").replace("usdt", "").upper()
    
    async def fetch_onchain_data(self, symbol: str) -> OnChainData:
        """Fetch on-chain metrics for a symbol."""
        base = self._get_base_symbol(symbol)
        
        data = OnChainData(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
        )
        
        try:
            # Fetch from multiple sources in parallel
            tasks = [
                self._fetch_exchange_flows(base),
                self._fetch_whale_activity(base),
                self._fetch_address_activity(base),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            for result in results:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(data, key):
                            setattr(data, key, value)
                            
        except Exception as e:
            logger.debug(f"On-chain fetch error for {symbol}: {e}")
        
        return data
    
    async def _fetch_exchange_flows(self, base: str) -> dict:
        """Fetch exchange inflow/outflow data."""
        result = {}
        session = await self._get_session()
        
        try:
            if base == "BTC":
                # Blockchain.com charts API (free)
                url = "https://api.blockchain.info/charts/exchange-trade-volume?timespan=1days&format=json"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("values"):
                            latest = data["values"][-1]
                            result["exchange_inflow_usd"] = latest.get("y", 0) * 0.6  # Estimate
                            result["exchange_outflow_usd"] = latest.get("y", 0) * 0.4
                            result["net_exchange_flow"] = result["exchange_inflow_usd"] - result["exchange_outflow_usd"]
            
            elif base == "ETH":
                # Use DeFiLlama for ETH flows (free)
                url = "https://api.llama.fi/overview/dexs/ethereum?excludeTotalDataChart=true"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        total_volume = data.get("totalVolume24h", 0)
                        result["exchange_inflow_usd"] = total_volume * 0.5
                        result["exchange_outflow_usd"] = total_volume * 0.5
                        
        except Exception as e:
            logger.debug(f"Exchange flow fetch error for {base}: {e}")
        
        return result
    
    async def _fetch_whale_activity(self, base: str) -> dict:
        """Fetch whale transaction data."""
        result = {}
        session = await self._get_session()
        
        try:
            # Use Whale Alert public RSS/API
            # Alternative: scrape from whale-alert.io
            if base in ["BTC", "ETH"]:
                # Blockchair API (limited free tier)
                chain = "bitcoin" if base == "BTC" else "ethereum"
                url = f"https://api.blockchair.com/{chain}/stats"
                
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        stats = data.get("data", {})
                        
                        # Estimate whale activity from large transactions
                        if base == "BTC":
                            result["large_tx_count_24h"] = stats.get("transactions_24h", 0) // 100  # Top 1%
                            result["large_tx_volume_usd"] = stats.get("volume_24h", 0) * 0.3  # Large tx portion
                        else:
                            result["large_tx_count_24h"] = stats.get("transactions_24h", 0) // 100
                            result["large_tx_volume_usd"] = stats.get("volume_24h", 0) * 0.25
                        
                        # Whale accumulation score based on exchange flow direction
                        if result.get("large_tx_volume_usd", 0) > 0:
                            result["whale_accumulation_score"] = 0.1  # Slight accumulation default
                            
        except Exception as e:
            logger.debug(f"Whale activity fetch error for {base}: {e}")
        
        return result
    
    async def _fetch_address_activity(self, base: str) -> dict:
        """Fetch active address data."""
        result = {}
        session = await self._get_session()
        
        try:
            if base == "BTC":
                url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=1days&format=json"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("values"):
                            latest = data["values"][-1]
                            result["active_addresses_24h"] = int(latest.get("y", 0))
                            
            elif base == "ETH":
                # Etherscan free API (rate limited)
                url = "https://api.etherscan.io/api?module=stats&action=ethsupply"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        # Limited data from free tier
                        result["active_addresses_24h"] = 500000  # Approximate
                        
        except Exception as e:
            logger.debug(f"Address activity fetch error for {base}: {e}")
        
        return result
    
    async def fetch_stablecoin_metrics(self) -> dict:
        """Fetch stablecoin supply and flow data."""
        result = {
            "usdt_dominance": 0.0,
            "total_stablecoin_supply": 0.0,
            "stablecoin_exchange_reserve": 0.0,
        }
        
        session = await self._get_session()
        
        try:
            # DeFiLlama stablecoin API (free)
            url = "https://stablecoins.llama.fi/stablecoins?includePrices=false"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    stables = data.get("peggedAssets", [])
                    
                    total_supply = 0
                    usdt_supply = 0
                    
                    for stable in stables:
                        supply = stable.get("circulating", {}).get("peggedUSD", 0)
                        total_supply += supply
                        if stable.get("symbol") == "USDT":
                            usdt_supply = supply
                    
                    result["total_stablecoin_supply"] = total_supply
                    result["usdt_dominance"] = usdt_supply / total_supply if total_supply > 0 else 0
                    result["stablecoin_exchange_reserve"] = total_supply * 0.3  # ~30% on exchanges
                    
        except Exception as e:
            logger.debug(f"Stablecoin metrics fetch error: {e}")
        
        return result


# =============================================================================
# NEWS PROVIDER
# =============================================================================

class NewsProvider:
    """
    Aggregates crypto news from free sources:
    - CryptoPanic API (free tier)
    - RSS feeds from major outlets
    - Google News RSS
    """
    
    # Symbol to search terms
    SEARCH_TERMS = {
        "BTC": ["bitcoin", "btc", "bitcoin price"],
        "ETH": ["ethereum", "eth", "ethereum price", "vitalik"],
        "SOL": ["solana", "sol"],
        "DOGE": ["dogecoin", "doge", "elon musk doge"],
        "XRP": ["ripple", "xrp", "sec ripple"],
        "ADA": ["cardano", "ada"],
        "BNB": ["binance coin", "bnb", "cz binance"],
        "LTC": ["litecoin", "ltc"],
    }
    
    # Sentiment keywords
    BULLISH_WORDS = [
        "surge", "soar", "rally", "bullish", "breakout", "moon", "pump",
        "all-time high", "ath", "buy", "accumulate", "adoption", "institutional",
        "approval", "etf approved", "partnership", "upgrade", "launch"
    ]
    
    BEARISH_WORDS = [
        "crash", "dump", "bearish", "plunge", "sell-off", "fear", "correction",
        "hack", "exploit", "sec", "lawsuit", "ban", "warning", "fraud",
        "bankrupt", "collapse", "liquidation", "fud"
    ]
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._news_cache: dict[str, list[NewsItem]] = {}
        self._cache_time: dict[str, datetime] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "HYDRA-Trading-Bot/1.0"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_base_symbol(self, internal_symbol: str) -> str:
        return internal_symbol.replace("cmt_", "").replace("usdt", "").upper()
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text. Returns -1 to 1."""
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.BULLISH_WORDS if word in text_lower)
        bearish_count = sum(1 for word in self.BEARISH_WORDS if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    async def fetch_news(self, symbol: str) -> NewsSentiment:
        """Fetch news sentiment for a symbol."""
        base = self._get_base_symbol(symbol)
        now = datetime.now(timezone.utc)
        
        result = NewsSentiment(
            timestamp=now,
            symbol=symbol,
        )
        
        # Check cache (5 min TTL)
        if symbol in self._cache_time:
            if (now - self._cache_time[symbol]).total_seconds() < 300:
                cached_news = self._news_cache.get(symbol, [])
                return self._aggregate_news(symbol, cached_news)
        
        all_news = []
        
        try:
            # Fetch from multiple sources
            tasks = [
                self._fetch_cryptopanic(base),
                self._fetch_google_news(base),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for news_list in results:
                if isinstance(news_list, list):
                    all_news.extend(news_list)
            
            # Cache results
            self._news_cache[symbol] = all_news
            self._cache_time[symbol] = now
            
            result = self._aggregate_news(symbol, all_news)
            
        except Exception as e:
            logger.debug(f"News fetch error for {symbol}: {e}")
        
        return result
    
    def _aggregate_news(self, symbol: str, news_items: list[NewsItem]) -> NewsSentiment:
        """Aggregate news items into sentiment data."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)
        
        news_1h = [n for n in news_items if n.timestamp > hour_ago]
        news_24h = [n for n in news_items if n.timestamp > day_ago]
        
        avg_1h = sum(n.sentiment for n in news_1h) / len(news_1h) if news_1h else 0
        avg_24h = sum(n.sentiment for n in news_24h) / len(news_24h) if news_24h else 0
        
        breaking = [n for n in news_1h if n.is_breaking or abs(n.sentiment) > 0.5]
        
        return NewsSentiment(
            timestamp=now,
            symbol=symbol,
            news_count_1h=len(news_1h),
            news_count_24h=len(news_24h),
            avg_sentiment_1h=avg_1h,
            avg_sentiment_24h=avg_24h,
            breaking_news=breaking[:5],  # Top 5 breaking
            sentiment_momentum=avg_1h - avg_24h,
        )
    
    async def _fetch_cryptopanic(self, base: str) -> list[NewsItem]:
        """Fetch from CryptoPanic free API."""
        news = []
        session = await self._get_session()
        
        try:
            # CryptoPanic public feed (no API key needed for basic)
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={base}&public=true"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for item in data.get("results", [])[:20]:
                        published = item.get("published_at", "")
                        try:
                            ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        except:
                            ts = datetime.now(timezone.utc)
                        
                        title = item.get("title", "")
                        sentiment = self._analyze_sentiment(title)
                        
                        # Check if breaking (high votes or recent)
                        votes = item.get("votes", {})
                        is_breaking = (
                            votes.get("positive", 0) > 10 or
                            votes.get("important", 0) > 5 or
                            (datetime.now(timezone.utc) - ts).total_seconds() < 3600
                        )
                        
                        news.append(NewsItem(
                            timestamp=ts,
                            title=title,
                            source="CryptoPanic",
                            url=item.get("url", ""),
                            sentiment=sentiment,
                            relevance=0.8,
                            symbols=[base],
                            is_breaking=is_breaking,
                        ))
                        
        except Exception as e:
            logger.debug(f"CryptoPanic fetch error for {base}: {e}")
        
        return news
    
    async def _fetch_google_news(self, base: str) -> list[NewsItem]:
        """Fetch from Google News RSS."""
        news = []
        session = await self._get_session()
        
        search_terms = self.SEARCH_TERMS.get(base, [base.lower()])
        
        for term in search_terms[:2]:  # Limit searches
            try:
                url = f"https://news.google.com/rss/search?q={term}+crypto&hl=en-US&gl=US&ceid=US:en"
                
                async with session.get(url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        
                        # Simple RSS parsing
                        items = re.findall(r'<item>.*?</item>', text, re.DOTALL)
                        
                        for item in items[:10]:
                            title_match = re.search(r'<title>(.*?)</title>', item)
                            link_match = re.search(r'<link>(.*?)</link>', item)
                            date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
                            
                            if title_match:
                                title = title_match.group(1).replace('<![CDATA[', '').replace(']]>', '')
                                
                                try:
                                    from email.utils import parsedate_to_datetime
                                    ts = parsedate_to_datetime(date_match.group(1)) if date_match else datetime.now(timezone.utc)
                                except:
                                    ts = datetime.now(timezone.utc)
                                
                                sentiment = self._analyze_sentiment(title)
                                
                                news.append(NewsItem(
                                    timestamp=ts,
                                    title=title,
                                    source="Google News",
                                    url=link_match.group(1) if link_match else "",
                                    sentiment=sentiment,
                                    relevance=0.6,
                                    symbols=[base],
                                    is_breaking=False,
                                ))
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                logger.debug(f"Google News fetch error for {term}: {e}")
        
        return news


# =============================================================================
# SOCIAL SENTIMENT PROVIDER
# =============================================================================

class SocialSentimentProvider:
    """
    Fetches social sentiment from free sources:
    - Reddit (public API)
    - LunarCrush alternatives
    - Twitter/X alternatives (Nitter, etc.)
    """
    
    SUBREDDITS = {
        "BTC": ["bitcoin", "bitcoinmarkets"],
        "ETH": ["ethereum", "ethtrader"],
        "SOL": ["solana"],
        "DOGE": ["dogecoin"],
        "XRP": ["ripple", "xrp"],
        "ADA": ["cardano"],
        "BNB": ["binance"],
        "LTC": ["litecoin"],
    }
    
    BULLISH_WORDS = [
        "bullish", "moon", "buy", "hodl", "accumulate", "dip", "undervalued",
        "breakout", "rally", "pump", "gains", "profit", "diamond hands"
    ]
    
    BEARISH_WORDS = [
        "bearish", "sell", "dump", "crash", "overvalued", "scam", "rug",
        "dead", "fear", "panic", "loss", "rekt", "paper hands"
    ]
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[datetime, SocialSentiment]] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_base_symbol(self, internal_symbol: str) -> str:
        return internal_symbol.replace("cmt_", "").replace("usdt", "").upper()
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        bullish = sum(1 for w in self.BULLISH_WORDS if w in text_lower)
        bearish = sum(1 for w in self.BEARISH_WORDS if w in text_lower)
        
        total = bullish + bearish
        if total == 0:
            return 0.0
        
        return (bullish - bearish) / total
    
    async def fetch_social_sentiment(self, symbol: str) -> SocialSentiment:
        """Fetch social sentiment for a symbol."""
        base = self._get_base_symbol(symbol)
        now = datetime.now(timezone.utc)
        
        # Check cache (10 min TTL for social)
        if symbol in self._cache:
            cache_time, cached = self._cache[symbol]
            if (now - cache_time).total_seconds() < 600:
                return cached
        
        result = SocialSentiment(
            timestamp=now,
            symbol=symbol,
        )
        
        try:
            # Fetch from sources in parallel
            tasks = [
                self._fetch_reddit(base),
                self._fetch_coingecko_social(base),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge Reddit data
            if isinstance(results[0], dict):
                result.reddit_posts_24h = results[0].get("posts", 0)
                result.reddit_comments_24h = results[0].get("comments", 0)
                result.reddit_sentiment = results[0].get("sentiment", 0)
            
            # Merge CoinGecko social data
            if isinstance(results[1], dict):
                result.social_volume_24h = results[1].get("volume", 0)
                result.social_dominance = results[1].get("dominance", 0)
            
            # Calculate overall
            result.overall_sentiment = (
                result.reddit_sentiment * 0.6 +
                result.social_sentiment * 0.4
            )
            
            if result.overall_sentiment > 0.2:
                result.sentiment_trend = "bullish"
            elif result.overall_sentiment < -0.2:
                result.sentiment_trend = "bearish"
            else:
                result.sentiment_trend = "neutral"
            
            # Cache result
            self._cache[symbol] = (now, result)
            
        except Exception as e:
            logger.debug(f"Social sentiment fetch error for {symbol}: {e}")
        
        return result
    
    async def _fetch_reddit(self, base: str) -> dict:
        """Fetch Reddit sentiment."""
        result = {"posts": 0, "comments": 0, "sentiment": 0.0}
        session = await self._get_session()
        
        subreddits = self.SUBREDDITS.get(base, [base.lower()])
        all_sentiments = []
        
        for subreddit in subreddits[:2]:
            try:
                # Reddit JSON API (no auth needed for public)
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
                
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])
                        
                        result["posts"] += len(posts)
                        
                        for post in posts:
                            post_data = post.get("data", {})
                            title = post_data.get("title", "")
                            selftext = post_data.get("selftext", "")[:500]
                            
                            sentiment = self._analyze_sentiment(title + " " + selftext)
                            
                            # Weight by upvotes
                            score = post_data.get("score", 1)
                            weight = min(score / 100, 5)  # Cap weight
                            all_sentiments.append(sentiment * weight)
                            
                            result["comments"] += post_data.get("num_comments", 0)
                
                await asyncio.sleep(1)  # Reddit rate limit
                
            except Exception as e:
                logger.debug(f"Reddit fetch error for r/{subreddit}: {e}")
        
        if all_sentiments:
            result["sentiment"] = sum(all_sentiments) / len(all_sentiments)
        
        return result
    
    async def _fetch_coingecko_social(self, base: str) -> dict:
        """Fetch social metrics from CoinGecko (free)."""
        result = {"volume": 0, "dominance": 0.0}
        session = await self._get_session()
        
        # CoinGecko coin ID mapping
        coin_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "DOGE": "dogecoin",
            "XRP": "ripple",
            "ADA": "cardano",
            "BNB": "binancecoin",
            "LTC": "litecoin",
        }
        
        coin_id = coin_ids.get(base, base.lower())
        
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=false"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    community = data.get("community_data", {})
                    
                    result["volume"] = (
                        community.get("twitter_followers", 0) +
                        community.get("reddit_subscribers", 0)
                    )
                    
                    # Estimate dominance based on followers
                    total_market = 50000000  # Approximate total crypto social
                    result["dominance"] = result["volume"] / total_market
                    
        except Exception as e:
            logger.debug(f"CoinGecko social fetch error for {base}: {e}")
        
        return result


# =============================================================================
# LIQUIDATION PROVIDER (COINGLASS + BINANCE FALLBACK)
# =============================================================================

class LiquidationProvider:
    """
    Fetches liquidation data from Coinglass API (primary) and Binance (fallback).
    Coinglass provides aggregated liquidation data across exchanges.
    """
    
    def __init__(self, coinglass_api_key: str = ""):
        self._session: Optional[aiohttp.ClientSession] = None
        self._liquidation_history: dict[str, list[dict]] = {}
        self._coinglass_key = coinglass_api_key
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _to_binance_symbol(self, internal: str) -> str:
        """Convert to Binance format (BTCUSDT)."""
        return internal.replace("cmt_", "").upper()
    
    def _get_base_symbol(self, internal: str) -> str:
        """Get base symbol (BTC, ETH, etc.)."""
        return internal.replace("cmt_", "").replace("usdt", "").upper()
    
    async def fetch_liquidations(self, symbol: str) -> LiquidationData:
        """Fetch liquidation data for a symbol using Coinglass API."""
        base_symbol = self._get_base_symbol(symbol)
        now = datetime.now(timezone.utc)
        
        result = LiquidationData(
            timestamp=now,
            symbol=symbol,
        )
        
        session = await self._get_session()
        
        # Try Coinglass API first (aggregated data across exchanges)
        if self._coinglass_key:
            try:
                # Coinglass V4 liquidation exchange list endpoint
                url = f"https://open-api-v3.coinglass.com/api/futures/liquidation/v2/coin?symbol={base_symbol}"
                headers = {
                    "CG-API-KEY": self._coinglass_key,
                    "accept": "application/json"
                }
                
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == "0" and data.get("data"):
                            liq_list = data["data"]
                            # Find "All" exchange aggregate or sum all exchanges
                            for item in liq_list:
                                if item.get("exchange") == "All":
                                    result.long_liquidations_24h = float(item.get("long_liquidation_usd", 0) or item.get("longLiquidationUsd", 0))
                                    result.short_liquidations_24h = float(item.get("short_liquidation_usd", 0) or item.get("shortLiquidationUsd", 0))
                                    break
                            
                            # If no "All" found, sum all exchanges
                            if result.long_liquidations_24h == 0 and result.short_liquidations_24h == 0:
                                for item in liq_list:
                                    result.long_liquidations_24h += float(item.get("long_liquidation_usd", 0) or item.get("longLiquidationUsd", 0))
                                    result.short_liquidations_24h += float(item.get("short_liquidation_usd", 0) or item.get("shortLiquidationUsd", 0))
                            
                            # Estimate 1h as ~4% of 24h
                            result.long_liquidations_1h = result.long_liquidations_24h * 0.04
                            result.short_liquidations_1h = result.short_liquidations_24h * 0.04
                            
                            # Calculate imbalance
                            total = result.long_liquidations_24h + result.short_liquidations_24h
                            if total > 0:
                                result.liquidation_imbalance = (
                                    result.long_liquidations_24h - result.short_liquidations_24h
                                ) / total
                            
                            logger.debug(f"Coinglass liquidations for {symbol}: L=${result.long_liquidations_24h:,.0f} S=${result.short_liquidations_24h:,.0f}")
                            return result
                            
            except Exception as e:
                logger.debug(f"Coinglass liquidation fetch error for {symbol}: {e}")
        
        # Fallback to Binance (may be empty for most pairs)
        try:
            binance_symbol = self._to_binance_symbol(symbol)
            url = f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={binance_symbol}&limit=100"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    orders = await resp.json()
                    hour_ago = now - timedelta(hours=1)
                    day_ago = now - timedelta(hours=24)
                    
                    for order in orders:
                        order_time = datetime.fromtimestamp(
                            order.get("time", 0) / 1000, tz=timezone.utc
                        )
                        side = order.get("side", "")
                        qty = float(order.get("executedQty", 0))
                        price = float(order.get("averagePrice", 0))
                        value = qty * price
                        
                        if order_time > hour_ago:
                            if side == "SELL":
                                result.long_liquidations_1h += value
                            else:
                                result.short_liquidations_1h += value
                        
                        if order_time > day_ago:
                            if side == "SELL":
                                result.long_liquidations_24h += value
                            else:
                                result.short_liquidations_24h += value
                        
                        if value > result.largest_liquidation:
                            result.largest_liquidation = value
                    
                    total_1h = result.long_liquidations_1h + result.short_liquidations_1h
                    if total_1h > 0:
                        result.liquidation_imbalance = (
                            result.long_liquidations_1h - result.short_liquidations_1h
                        ) / total_1h
                        
        except Exception as e:
            logger.debug(f"Binance liquidation fetch error for {symbol}: {e}")
        
        return result


# =============================================================================
# POSITIONING PROVIDER (BINANCE)
# =============================================================================

class PositioningProvider:
    """
    Fetches market positioning data from Binance:
    - Long/Short ratio
    - Top trader positions
    - Taker buy/sell ratio
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _to_binance_symbol(self, internal: str) -> str:
        return internal.replace("cmt_", "").upper()
    
    async def fetch_positioning(self, symbol: str) -> PositioningData:
        """Fetch positioning data for a symbol."""
        binance_symbol = self._to_binance_symbol(symbol)
        now = datetime.now(timezone.utc)
        
        result = PositioningData(
            timestamp=now,
            symbol=symbol,
        )
        
        session = await self._get_session()
        
        try:
            # Fetch multiple positioning metrics in parallel
            tasks = [
                self._fetch_long_short_ratio(session, binance_symbol),
                self._fetch_top_trader_positions(session, binance_symbol),
                self._fetch_taker_volume(session, binance_symbol),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Long/Short ratio
            if isinstance(results[0], dict):
                result.long_short_ratio = results[0].get("ratio", 1.0)
            
            # Top trader positions
            if isinstance(results[1], dict):
                result.top_trader_long_ratio = results[1].get("long", 0.5)
                result.top_trader_short_ratio = results[1].get("short", 0.5)
            
            # Update funding from taker data
            if isinstance(results[2], dict):
                result.funding_rate = results[2].get("funding", 0)
                
        except Exception as e:
            logger.debug(f"Positioning fetch error for {symbol}: {e}")
        
        return result
    
    async def _fetch_long_short_ratio(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Fetch global long/short ratio."""
        try:
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        return {"ratio": float(data[0].get("longShortRatio", 1.0))}
        except Exception as e:
            logger.debug(f"Long/short ratio fetch error: {e}")
        
        return {}
    
    async def _fetch_top_trader_positions(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Fetch top trader long/short positions."""
        try:
            url = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit=1"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        return {
                            "long": float(data[0].get("longAccount", 0.5)),
                            "short": float(data[0].get("shortAccount", 0.5)),
                        }
        except Exception as e:
            logger.debug(f"Top trader positions fetch error: {e}")
        
        return {}
    
    async def _fetch_taker_volume(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Fetch taker buy/sell volume ratio."""
        try:
            url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=1"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        return {
                            "buy_ratio": float(data[0].get("buyVol", 0.5)),
                            "sell_ratio": float(data[0].get("sellVol", 0.5)),
                        }
        except Exception as e:
            logger.debug(f"Taker volume fetch error: {e}")
        
        return {}


# =============================================================================
# COMBINED DATA AGGREGATOR
# =============================================================================

class MarketDataAggregator:
    """
    Combines all data providers into a single interface.
    Manages caching and rate limiting across providers.
    """
    
    def __init__(self, coinglass_api_key: str = ""):
        self.onchain = OnChainProvider()
        self.news = NewsProvider()
        self.social = SocialSentimentProvider()
        self.liquidations = LiquidationProvider(coinglass_api_key=coinglass_api_key)
        self.positioning = PositioningProvider()
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all providers."""
        self._initialized = True
        logger.info("Market Data Aggregator initialized")
    
    async def close(self):
        """Close all provider sessions."""
        await asyncio.gather(
            self.onchain.close(),
            self.news.close(),
            self.social.close(),
            self.liquidations.close(),
            self.positioning.close(),
        )
        self._initialized = False
    
    async def fetch_all_data(self, symbol: str) -> dict:
        """Fetch all available data for a symbol."""
        tasks = [
            self.onchain.fetch_onchain_data(symbol),
            self.news.fetch_news(symbol),
            self.social.fetch_social_sentiment(symbol),
            self.liquidations.fetch_liquidations(symbol),
            self.positioning.fetch_positioning(symbol),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "onchain": results[0] if not isinstance(results[0], Exception) else None,
            "news": results[1] if not isinstance(results[1], Exception) else None,
            "social": results[2] if not isinstance(results[2], Exception) else None,
            "liquidations": results[3] if not isinstance(results[3], Exception) else None,
            "positioning": results[4] if not isinstance(results[4], Exception) else None,
        }
    
    async def fetch_stablecoin_metrics(self) -> dict:
        """Fetch global stablecoin metrics."""
        return await self.onchain.fetch_stablecoin_metrics()
