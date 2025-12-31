"""
Layer 1: Market Intelligence

Collects ALL market data needed for decisions. No opinions — just facts.

Data Sources:
- Binance Futures API: OHLCV, Funding Rate, Open Interest, Order Book, Trades
- Coinalyse API: Liquidation history (aggregated across exchanges)
- Binance API: Long/Short Ratio, Taker Buy/Sell

Output: MarketState for each symbol
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import deque

import aiohttp
from loguru import logger

from hydra.core.types import (
    OHLCV, FundingRate, OpenInterest, Liquidation, 
    OrderBookSnapshot, MarketState, Side
)


# =============================================================================
# CONSTANTS
# =============================================================================

PERMITTED_PAIRS = [
    "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_bnbusdt",
    "cmt_adausdt", "cmt_xrpusdt", "cmt_ltcusdt", "cmt_dogeusdt"
]

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

BINANCE_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"
}

COINALYSE_SYMBOL_MAP = {
    "cmt_btcusdt": "BTCUSDT_PERP.A",
    "cmt_ethusdt": "ETHUSDT_PERP.A",
    "cmt_solusdt": "SOLUSDT_PERP.A",
    "cmt_bnbusdt": "BNBUSDT_PERP.A",
    "cmt_adausdt": "ADAUSDT_PERP.A",
    "cmt_xrpusdt": "XRPUSDT_PERP.A",
    "cmt_ltcusdt": "LTCUSDT_PERP.A",
    "cmt_dogeusdt": "DOGEUSDT_PERP.A",
}

# Data retention
CANDLES_TO_KEEP = 500
LIQUIDATIONS_TO_KEEP = 100
TRADES_TO_KEEP = 100

# Update intervals (seconds)
UPDATE_INTERVAL_OHLCV = 60
UPDATE_INTERVAL_FUNDING = 300
UPDATE_INTERVAL_OI = 15
UPDATE_INTERVAL_ORDERBOOK = 5
UPDATE_INTERVAL_LIQUIDATIONS = 30


# =============================================================================
# POSITIONING DATA
# =============================================================================

@dataclass
class PositioningData:
    """Market positioning data from Binance."""
    timestamp: datetime
    symbol: str
    long_short_ratio: float = 1.0
    top_trader_long_ratio: float = 0.5
    top_trader_short_ratio: float = 0.5
    taker_buy_ratio: float = 0.5
    taker_sell_ratio: float = 0.5


# =============================================================================
# BINANCE FUTURES CLIENT
# =============================================================================

class BinanceFuturesClient:
    """
    Async client for Binance Futures API.
    
    Endpoints used:
    - GET /fapi/v1/klines - OHLCV candles
    - GET /fapi/v1/fundingRate - Funding rate
    - GET /fapi/v1/openInterest - Open interest
    - GET /fapi/v1/depth - Order book
    - GET /fapi/v1/trades - Recent trades
    - GET /fapi/v1/premiumIndex - Mark price and funding info
    - GET /futures/data/globalLongShortAccountRatio - L/S ratio
    - GET /futures/data/topLongShortPositionRatio - Top traders
    - GET /futures/data/takerlongshortRatio - Taker volume
    """
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 1200
        self._rate_limit_reset = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "HYDRA-Trading-Bot/1.0"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _to_binance_symbol(self, internal: str) -> str:
        """Convert cmt_btcusdt -> BTCUSDT"""
        return internal.replace("cmt_", "").upper()
    
    async def _request(self, endpoint: str, params: dict = None) -> Any:
        """Make a request to Binance API with error handling."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning(f"Binance rate limit hit, waiting...")
                    await asyncio.sleep(60)
                    return None
                else:
                    text = await resp.text()
                    logger.debug(f"Binance API error {resp.status}: {text[:200]}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Binance API timeout: {endpoint}")
            return None
        except Exception as e:
            logger.debug(f"Binance API error: {e}")
            return None
    
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> list[OHLCV]:
        """Fetch OHLCV candles."""
        binance_symbol = self._to_binance_symbol(symbol)
        binance_tf = BINANCE_TIMEFRAME_MAP.get(timeframe, timeframe)
        
        data = await self._request("/fapi/v1/klines", {
            "symbol": binance_symbol,
            "interval": binance_tf,
            "limit": limit
        })
        
        if not data:
            return []
        
        candles = []
        for item in data:
            candles.append(OHLCV(
                timestamp=datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                symbol=symbol,
                timeframe=timeframe
            ))
        
        return candles
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Fetch current and predicted funding rate."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        # Get premium index which includes funding info
        data = await self._request("/fapi/v1/premiumIndex", {
            "symbol": binance_symbol
        })
        
        if not data:
            return None
        
        try:
            next_funding_ts = data.get("nextFundingTime", 0)
            next_funding_time = datetime.fromtimestamp(
                next_funding_ts / 1000, tz=timezone.utc
            ) if next_funding_ts else None
            
            return FundingRate(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                rate=float(data.get("lastFundingRate", 0)),
                predicted_rate=float(data.get("interestRate", 0)),
                next_funding_time=next_funding_time
            )
        except Exception as e:
            logger.debug(f"Funding rate parse error for {symbol}: {e}")
            return None
    
    async def fetch_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        """Fetch current open interest."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/fapi/v1/openInterest", {
            "symbol": binance_symbol
        })
        
        if not data:
            return None
        
        try:
            oi_contracts = float(data.get("openInterest", 0))
            
            # Get current price for USD value
            ticker = await self._request("/fapi/v1/ticker/price", {
                "symbol": binance_symbol
            })
            price = float(ticker.get("price", 0)) if ticker else 0
            oi_usd = oi_contracts * price
            
            return OpenInterest(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                open_interest=oi_contracts,
                open_interest_usd=oi_usd,
                delta=0.0,
                delta_pct=0.0
            )
        except Exception as e:
            logger.debug(f"Open interest parse error for {symbol}: {e}")
            return None
    
    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Optional[OrderBookSnapshot]:
        """Fetch order book depth."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/fapi/v1/depth", {
            "symbol": binance_symbol,
            "limit": limit
        })
        
        if not data:
            return None
        
        try:
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
            
            return OrderBookSnapshot(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                bids=bids,
                asks=asks
            )
        except Exception as e:
            logger.debug(f"Order book parse error for {symbol}: {e}")
            return None
    
    async def fetch_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        """Fetch recent trades for CVD calculation."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/fapi/v1/trades", {
            "symbol": binance_symbol,
            "limit": limit
        })
        
        if not data:
            return []
        
        trades = []
        for t in data:
            trades.append({
                "timestamp": datetime.fromtimestamp(t["time"] / 1000, tz=timezone.utc),
                "price": float(t["price"]),
                "quantity": float(t["qty"]),
                "is_buyer_maker": t.get("isBuyerMaker", False)
            })
        
        return trades
    
    async def fetch_long_short_ratio(self, symbol: str) -> Optional[float]:
        """Fetch global long/short account ratio."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/futures/data/globalLongShortAccountRatio", {
            "symbol": binance_symbol,
            "period": "1h",
            "limit": 1
        })
        
        if data and len(data) > 0:
            return float(data[0].get("longShortRatio", 1.0))
        return None
    
    async def fetch_top_trader_positions(self, symbol: str) -> Optional[dict]:
        """Fetch top trader long/short positions."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/futures/data/topLongShortPositionRatio", {
            "symbol": binance_symbol,
            "period": "1h",
            "limit": 1
        })
        
        if data and len(data) > 0:
            return {
                "long": float(data[0].get("longAccount", 0.5)),
                "short": float(data[0].get("shortAccount", 0.5))
            }
        return None
    
    async def fetch_taker_buy_sell(self, symbol: str) -> Optional[dict]:
        """Fetch taker buy/sell volume ratio."""
        binance_symbol = self._to_binance_symbol(symbol)
        
        data = await self._request("/futures/data/takerlongshortRatio", {
            "symbol": binance_symbol,
            "period": "1h",
            "limit": 1
        })
        
        if data and len(data) > 0:
            return {
                "buy": float(data[0].get("buyVol", 0.5)),
                "sell": float(data[0].get("sellVol", 0.5))
            }
        return None
    
    async def fetch_positioning(self, symbol: str) -> PositioningData:
        """Fetch all positioning data for a symbol."""
        result = PositioningData(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol
        )
        
        try:
            ls_ratio, top_traders, taker = await asyncio.gather(
                self.fetch_long_short_ratio(symbol),
                self.fetch_top_trader_positions(symbol),
                self.fetch_taker_buy_sell(symbol),
                return_exceptions=True
            )
            
            if isinstance(ls_ratio, float):
                result.long_short_ratio = ls_ratio
            
            if isinstance(top_traders, dict):
                result.top_trader_long_ratio = top_traders.get("long", 0.5)
                result.top_trader_short_ratio = top_traders.get("short", 0.5)
            
            if isinstance(taker, dict):
                result.taker_buy_ratio = taker.get("buy", 0.5)
                result.taker_sell_ratio = taker.get("sell", 0.5)
                
        except Exception as e:
            logger.debug(f"Positioning fetch error for {symbol}: {e}")
        
        return result


# =============================================================================
# COINALYSE LIQUIDATION CLIENT
# =============================================================================

class CoinalyseLiquidationClient:
    """
    Client for Coinalyse liquidation history API.
    
    API: https://api.coinalyze.net/v1/liquidation-history
    
    Response format:
    [
        {
            "symbol": "BTCUSDT_PERP.A",
            "history": [
                {"t": 1234567890, "l": 1000000, "s": 500000}
            ]
        }
    ]
    
    t = timestamp (seconds)
    l = long liquidations (USD)
    s = short liquidations (USD)
    """
    
    BASE_URL = "https://api.coinalyze.net/v1"
    
    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.getenv("COINALYSE_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[datetime, list[Liquidation]]] = {}
        self._cache_ttl = 30  # seconds
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "HYDRA-Trading-Bot/1.0",
                    "api_key": self._api_key
                }
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def fetch_liquidations(
        self, 
        symbol: str, 
        interval: str = "1min",
        hours_back: int = 1
    ) -> list[Liquidation]:
        """
        Fetch liquidation history for a symbol.
        
        Args:
            symbol: Internal symbol (cmt_btcusdt)
            interval: Granularity (1min, 5min, 15min, 30min, 1hour, etc.)
            hours_back: How many hours of history to fetch
        
        Returns:
            List of Liquidation objects
        """
        # Check cache first
        cache_key = f"{symbol}_{interval}"
        if cache_key in self._cache:
            cache_time, cached_data = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self._cache_ttl:
                return cached_data
        
        if not self._api_key:
            logger.warning("COINALYSE_API_KEY not set, liquidation data unavailable")
            return []
        
        coinalyse_symbol = COINALYSE_SYMBOL_MAP.get(symbol)
        if not coinalyse_symbol:
            logger.debug(f"No Coinalyse mapping for {symbol}")
            return []
        
        now = int(time.time())
        from_ts = now - (hours_back * 3600)
        
        session = await self._get_session()
        
        try:
            url = f"{self.BASE_URL}/liquidation-history"
            params = {
                "symbols": coinalyse_symbol,
                "interval": interval,
                "from": from_ts,
                "to": now,
                "convert_to_usd": "true"
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    liquidations = self._parse_liquidations(symbol, data)
                    
                    # Cache result
                    self._cache[cache_key] = (datetime.now(timezone.utc), liquidations)
                    
                    return liquidations
                elif resp.status == 401:
                    logger.error("Coinalyse API key invalid or expired")
                    return []
                else:
                    text = await resp.text()
                    logger.debug(f"Coinalyse API error {resp.status}: {text[:200]}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.debug(f"Coinalyse API timeout for {symbol}")
            return []
        except Exception as e:
            logger.debug(f"Coinalyse API error: {e}")
            return []
    
    def _parse_liquidations(self, symbol: str, data: list) -> list[Liquidation]:
        """Parse Coinalyse API response into Liquidation objects."""
        liquidations = []
        
        if not data or not isinstance(data, list):
            return []
        
        for item in data:
            history = item.get("history", [])
            
            for entry in history:
                ts = entry.get("t", 0)
                long_liq = float(entry.get("l", 0))
                short_liq = float(entry.get("s", 0))
                
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                # Create separate liquidation entries for longs and shorts
                if long_liq > 0:
                    liquidations.append(Liquidation(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.LONG,
                        quantity=0,  # Aggregated data doesn't have quantity
                        price=0,      # Aggregated data doesn't have price
                        usd_value=long_liq
                    ))
                
                if short_liq > 0:
                    liquidations.append(Liquidation(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.SHORT,
                        quantity=0,
                        price=0,
                        usd_value=short_liq
                    ))
        
        # Sort by timestamp descending (most recent first)
        liquidations.sort(key=lambda x: x.timestamp, reverse=True)
        
        return liquidations[:LIQUIDATIONS_TO_KEEP]
    
    async def fetch_liquidations_batch(
        self,
        symbols: list[str],
        interval: str = "5min",
        hours_back: int = 1
    ) -> dict[str, list[Liquidation]]:
        """
        Fetch liquidations for multiple symbols in one API call.
        
        Coinalyse allows up to 20 symbols per request.
        """
        if not self._api_key:
            return {s: [] for s in symbols}
        
        # Map to Coinalyse symbols
        coinalyse_symbols = []
        symbol_map = {}
        
        for s in symbols:
            cs = COINALYSE_SYMBOL_MAP.get(s)
            if cs:
                coinalyse_symbols.append(cs)
                symbol_map[cs] = s
        
        if not coinalyse_symbols:
            return {s: [] for s in symbols}
        
        now = int(time.time())
        from_ts = now - (hours_back * 3600)
        
        session = await self._get_session()
        
        try:
            url = f"{self.BASE_URL}/liquidation-history"
            params = {
                "symbols": ",".join(coinalyse_symbols),
                "interval": interval,
                "from": from_ts,
                "to": now,
                "convert_to_usd": "true"
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    result = {s: [] for s in symbols}
                    
                    for item in data:
                        cs = item.get("symbol")
                        internal_symbol = symbol_map.get(cs)
                        
                        if internal_symbol:
                            liqs = self._parse_liquidations(internal_symbol, [item])
                            result[internal_symbol] = liqs
                    
                    return result
                else:
                    logger.debug(f"Coinalyse batch API error {resp.status}")
                    return {s: [] for s in symbols}
                    
        except Exception as e:
            logger.debug(f"Coinalyse batch API error: {e}")
            return {s: [] for s in symbols}


# =============================================================================
# MARKET INTELLIGENCE LAYER
# =============================================================================

class MarketIntelligenceLayer:
    """
    Layer 1: Market Intelligence
    
    Collects ALL market data needed for decisions:
    - OHLCV candles (5 timeframes, 500 candles each)
    - Funding rates
    - Open interest with delta tracking
    - Order book depth
    - Recent trades for CVD
    - Liquidation data from Coinalyse
    - Positioning data (L/S ratio, top traders, taker flow)
    
    Outputs MarketState for each symbol.
    """
    
    def __init__(self, coinalyse_api_key: str = ""):
        self.binance = BinanceFuturesClient()
        self.coinalyse = CoinalyseLiquidationClient(api_key=coinalyse_api_key)
        
        # Data storage
        self._ohlcv: dict[str, dict[str, list[OHLCV]]] = {}  # symbol -> tf -> candles
        self._funding: dict[str, FundingRate] = {}
        self._oi: dict[str, OpenInterest] = {}
        self._oi_history: dict[str, deque] = {}  # For delta calculation
        self._order_books: dict[str, OrderBookSnapshot] = {}
        self._liquidations: dict[str, list[Liquidation]] = {}
        self._trades: dict[str, list[dict]] = {}
        self._positioning: dict[str, PositioningData] = {}
        
        # Last update times
        self._last_update: dict[str, dict[str, datetime]] = {}
        
        # Initialize storage for all pairs
        for symbol in PERMITTED_PAIRS:
            self._ohlcv[symbol] = {tf: [] for tf in TIMEFRAMES}
            self._oi_history[symbol] = deque(maxlen=100)
            self._last_update[symbol] = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize layer and fetch initial data."""
        logger.info("Initializing Market Intelligence Layer...")
        
        # Fetch initial data for all symbols
        await self.refresh_all()
        
        self._initialized = True
        logger.info("Market Intelligence Layer initialized")
    
    async def close(self):
        """Close all connections."""
        await self.binance.close()
        await self.coinalyse.close()
        self._initialized = False
    
    def _should_update(self, symbol: str, data_type: str, interval: int) -> bool:
        """Check if data should be updated based on interval."""
        last = self._last_update.get(symbol, {}).get(data_type)
        if last is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= interval
    
    def _mark_updated(self, symbol: str, data_type: str):
        """Mark data as just updated."""
        if symbol not in self._last_update:
            self._last_update[symbol] = {}
        self._last_update[symbol][data_type] = datetime.now(timezone.utc)
    
    async def refresh_ohlcv(self, symbol: str, timeframes: list[str] = None):
        """Refresh OHLCV data for a symbol."""
        timeframes = timeframes or TIMEFRAMES
        
        tasks = []
        for tf in timeframes:
            if self._should_update(symbol, f"ohlcv_{tf}", UPDATE_INTERVAL_OHLCV):
                tasks.append(self._fetch_and_store_ohlcv(symbol, tf))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_and_store_ohlcv(self, symbol: str, timeframe: str):
        """Fetch and store OHLCV data."""
        candles = await self.binance.fetch_ohlcv(symbol, timeframe, CANDLES_TO_KEEP)
        
        if candles:
            self._ohlcv[symbol][timeframe] = candles
            self._mark_updated(symbol, f"ohlcv_{timeframe}")
            logger.debug(f"Updated {symbol} {timeframe}: {len(candles)} candles")
    
    async def refresh_funding(self, symbol: str):
        """Refresh funding rate for a symbol."""
        if not self._should_update(symbol, "funding", UPDATE_INTERVAL_FUNDING):
            return
        
        funding = await self.binance.fetch_funding_rate(symbol)
        
        if funding:
            self._funding[symbol] = funding
            self._mark_updated(symbol, "funding")
            logger.debug(f"Updated {symbol} funding: {funding.rate:.6f}")
    
    async def refresh_open_interest(self, symbol: str):
        """Refresh open interest with delta calculation."""
        if not self._should_update(symbol, "oi", UPDATE_INTERVAL_OI):
            return
        
        oi = await self.binance.fetch_open_interest(symbol)
        
        if oi:
            # Calculate delta from history
            if self._oi_history[symbol]:
                prev_oi = self._oi_history[symbol][-1]
                oi.delta = oi.open_interest_usd - prev_oi.open_interest_usd
                if prev_oi.open_interest_usd > 0:
                    oi.delta_pct = oi.delta / prev_oi.open_interest_usd
            
            # Store in history
            self._oi_history[symbol].append(oi)
            self._oi[symbol] = oi
            self._mark_updated(symbol, "oi")
            logger.debug(f"Updated {symbol} OI: ${oi.open_interest_usd:,.0f} (Δ{oi.delta_pct:+.2%})")
    
    async def refresh_order_book(self, symbol: str):
        """Refresh order book for a symbol."""
        if not self._should_update(symbol, "orderbook", UPDATE_INTERVAL_ORDERBOOK):
            return
        
        ob = await self.binance.fetch_order_book(symbol, limit=20)
        
        if ob:
            self._order_books[symbol] = ob
            self._mark_updated(symbol, "orderbook")
            logger.debug(f"Updated {symbol} orderbook: spread={ob.spread:.4%}, imbalance={ob.imbalance:+.2f}")
    
    async def refresh_liquidations(self, symbol: str):
        """Refresh liquidation data from Coinalyse."""
        if not self._should_update(symbol, "liquidations", UPDATE_INTERVAL_LIQUIDATIONS):
            return
        
        liqs = await self.coinalyse.fetch_liquidations(symbol, interval="5min", hours_back=1)
        
        if liqs:
            self._liquidations[symbol] = liqs
            self._mark_updated(symbol, "liquidations")
            
            total_long = sum(l.usd_value for l in liqs if l.side == Side.LONG)
            total_short = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
            logger.debug(f"Updated {symbol} liquidations: L=${total_long:,.0f} S=${total_short:,.0f}")
    
    async def refresh_trades(self, symbol: str):
        """Refresh recent trades."""
        trades = await self.binance.fetch_recent_trades(symbol, TRADES_TO_KEEP)
        
        if trades:
            self._trades[symbol] = trades
    
    async def refresh_positioning(self, symbol: str):
        """Refresh positioning data."""
        positioning = await self.binance.fetch_positioning(symbol)
        self._positioning[symbol] = positioning
    
    async def refresh_symbol(self, symbol: str):
        """Refresh all data for a single symbol."""
        tasks = [
            self.refresh_ohlcv(symbol),
            self.refresh_funding(symbol),
            self.refresh_open_interest(symbol),
            self.refresh_order_book(symbol),
            self.refresh_liquidations(symbol),
            self.refresh_positioning(symbol),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def refresh_all(self):
        """Refresh data for all permitted pairs."""
        logger.debug("Refreshing all market data...")
        
        # Refresh in batches to avoid rate limits
        for symbol in PERMITTED_PAIRS:
            await self.refresh_symbol(symbol)
            await asyncio.sleep(0.1)  # Small delay between symbols
        
        # Batch fetch liquidations (more efficient)
        liqs = await self.coinalyse.fetch_liquidations_batch(PERMITTED_PAIRS)
        for symbol, liq_list in liqs.items():
            if liq_list:
                self._liquidations[symbol] = liq_list
                self._mark_updated(symbol, "liquidations")
    
    def get_market_state(self, symbol: str) -> MarketState:
        """
        Get complete market state for a symbol.
        
        This is the primary output of Layer 1.
        """
        now = datetime.now(timezone.utc)
        
        # Get latest price from most recent candle
        candles_5m = self._ohlcv.get(symbol, {}).get("5m", [])
        price = candles_5m[-1].close if candles_5m else 0.0
        
        # Calculate derived metrics
        volatility = self._calculate_volatility(symbol)
        volume_24h = self._calculate_volume_24h(symbol)
        price_change_24h = self._calculate_price_change_24h(symbol)
        
        # Get mark price and index from funding data
        funding = self._funding.get(symbol)
        mark_price = price  # Use last price if no mark price available
        index_price = price
        basis = 0.0
        
        return MarketState(
            timestamp=now,
            symbol=symbol,
            price=price,
            ohlcv=self._ohlcv.get(symbol, {}),
            funding_rate=funding,
            open_interest=self._oi.get(symbol),
            recent_liquidations=self._liquidations.get(symbol, []),
            order_book=self._order_books.get(symbol),
            volatility=volatility,
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            index_price=index_price,
            mark_price=mark_price,
            basis=basis
        )
    
    def get_all_market_states(self) -> dict[str, MarketState]:
        """Get market state for all symbols."""
        return {symbol: self.get_market_state(symbol) for symbol in PERMITTED_PAIRS}
    
    def get_positioning(self, symbol: str) -> Optional[PositioningData]:
        """Get positioning data for a symbol."""
        return self._positioning.get(symbol)
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate annualized volatility from 5m returns."""
        import numpy as np
        
        candles = self._ohlcv.get(symbol, {}).get("5m", [])
        if len(candles) < 20:
            return 0.0
        
        closes = [c.close for c in candles[-100:]]
        returns = np.diff(np.log(closes))
        
        # Annualize: 5min candles, 288 per day, 365 days
        vol = np.std(returns) * np.sqrt(288 * 365)
        
        return float(vol)
    
    def _calculate_volume_24h(self, symbol: str) -> float:
        """Calculate 24h volume from 1h candles."""
        candles = self._ohlcv.get(symbol, {}).get("1h", [])
        if len(candles) < 24:
            return 0.0
        
        return sum(c.volume for c in candles[-24:])
    
    def _calculate_price_change_24h(self, symbol: str) -> float:
        """Calculate 24h price change percentage."""
        candles = self._ohlcv.get(symbol, {}).get("1h", [])
        if len(candles) < 24:
            return 0.0
        
        old_price = candles[-24].close
        new_price = candles[-1].close
        
        if old_price <= 0:
            return 0.0
        
        return (new_price - old_price) / old_price
    
    def get_liquidation_imbalance(self, symbol: str) -> float:
        """
        Calculate liquidation imbalance.
        
        Returns: -1 to 1
        - Positive = more longs liquidated
        - Negative = more shorts liquidated
        """
        liqs = self._liquidations.get(symbol, [])
        if not liqs:
            return 0.0
        
        long_liq = sum(l.usd_value for l in liqs if l.side == Side.LONG)
        short_liq = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
        
        total = long_liq + short_liq
        if total == 0:
            return 0.0
        
        return (long_liq - short_liq) / total
    
    def get_liquidation_velocity(self, symbol: str) -> float:
        """
        Calculate liquidation velocity (liquidations / OI).
        
        High values indicate market stress.
        """
        liqs = self._liquidations.get(symbol, [])
        oi = self._oi.get(symbol)
        
        if not liqs or not oi or oi.open_interest_usd <= 0:
            return 0.0
        
        total_liq = sum(l.usd_value for l in liqs)
        
        return total_liq / oi.open_interest_usd


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_market_intel_layer() -> MarketIntelligenceLayer:
    """Factory function to create and initialize Market Intelligence Layer."""
    coinalyse_key = os.getenv("COINALYSE_API_KEY", "")
    
    layer = MarketIntelligenceLayer(coinalyse_api_key=coinalyse_key)
    await layer.initialize()
    
    return layer
