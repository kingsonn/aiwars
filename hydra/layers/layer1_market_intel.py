"""
Layer 1: Market Intelligence Layer

Primary data ingestion and preprocessing for HYDRA.
Collects: Price, Funding, OI, Liquidations, Order Book, On-Chain, Sentiment.
"""

from __future__ import annotations

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from hydra.core.config import HydraConfig, PAIR_DISPLAY_NAMES
from hydra.core.types import (
    MarketState,
    OHLCV,
    FundingRate,
    OpenInterest,
    Liquidation,
    OrderBookSnapshot,
    Side,
)
from hydra.layers.data_providers import (
    MarketDataAggregator,
    OnChainData,
    NewsSentiment,
    SocialSentiment,
    LiquidationData,
    PositioningData,
)
from hydra.layers.enhanced_sentiment import (
    EnhancedSentimentAggregator,
    FearGreedData,
    TechnicalSentiment,
    EnhancedSentiment,
)


def _to_exchange_symbol(internal: str) -> str:
    """Convert internal symbol (cmt_btcusdt) to exchange format (BTC/USDT:USDT)."""
    display = PAIR_DISPLAY_NAMES.get(internal, internal)
    if "/" not in display:
        # Already in internal format, convert it
        base = internal.replace("cmt_", "").replace("usdt", "").upper()
        return f"{base}/USDT:USDT"
    return f"{display}:USDT"


def _to_internal_symbol(exchange: str) -> str:
    """Convert exchange symbol (BTC/USDT:USDT) to internal format (cmt_btcusdt)."""
    # Remove :USDT suffix and convert
    base = exchange.replace(":USDT", "").replace("/USDT", "").lower()
    return f"cmt_{base}usdt"


@dataclass
class AggregatedFlow:
    """Aggregated trade flow data."""
    timestamp: datetime
    symbol: str
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    large_buy_volume: float  # Trades > threshold
    large_sell_volume: float
    
    @property
    def net_flow(self) -> float:
        return self.buy_volume - self.sell_volume
    
    @property
    def flow_imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        return self.net_flow / total if total > 0 else 0


@dataclass
class SentimentData:
    """Aggregated sentiment data."""
    timestamp: datetime
    symbol: str
    social_score: float  # -1 to 1
    funding_sentiment: float  # -1 to 1
    crowd_positioning: float  # -1 (everyone short) to 1 (everyone long)
    narrative_velocity: float  # Rate of narrative change
    fear_greed_index: float  # 0 to 100


@dataclass 
class OnChainMetrics:
    """On-chain data for context."""
    timestamp: datetime
    symbol: str
    exchange_inflow_usd: float
    exchange_outflow_usd: float
    net_flow_usd: float
    whale_transactions: int
    stablecoin_flow: float


class MarketIntelligenceLayer:
    """
    Layer 1: Market Intelligence
    
    Responsibilities:
    - Connect to exchange WebSocket feeds
    - Aggregate OHLCV across timeframes
    - Track funding rates and changes
    - Monitor open interest deltas
    - Capture liquidation streams
    - Maintain order book snapshots
    - Integrate on-chain data (contextual)
    - Process sentiment signals
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self._exchange = None
        self._ws_connections: dict[str, Any] = {}
        self._running = False
        
        # Data stores
        self._ohlcv: dict[str, dict[str, list[OHLCV]]] = {}  # symbol -> timeframe -> candles
        self._funding: dict[str, FundingRate] = {}
        self._open_interest: dict[str, OpenInterest] = {}
        self._liquidations: dict[str, list[Liquidation]] = {}  # Last N liquidations
        self._order_books: dict[str, OrderBookSnapshot] = {}
        self._flows: dict[str, AggregatedFlow] = {}
        self._sentiment: dict[str, SentimentData] = {}
        self._onchain: dict[str, OnChainMetrics] = {}
        
        # Enhanced data stores (from new providers)
        self._onchain_data: dict[str, OnChainData] = {}
        self._news_sentiment: dict[str, NewsSentiment] = {}
        self._social_sentiment: dict[str, SocialSentiment] = {}
        self._liquidation_data: dict[str, LiquidationData] = {}
        self._positioning_data: dict[str, PositioningData] = {}
        self._stablecoin_metrics: dict = {}
        
        # Binance public API data (Long/Short ratios, Taker volume)
        self._long_short_ratio: dict[str, dict] = {}  # symbol -> {longAccount, shortAccount, ratio}
        self._taker_volume: dict[str, dict] = {}  # symbol -> {buyVol, sellVol, ratio}
        self._http_session = None  # For direct Binance API calls
        
        # CVD (Cumulative Volume Delta) tracking
        self._cvd: dict[str, float] = {}  # symbol -> cumulative delta
        self._cvd_history: dict[str, list[tuple[datetime, float]]] = {}
        
        # Data aggregator for external sources (with Coinglass API key for liquidations)
        self._data_aggregator = MarketDataAggregator(
            coinglass_api_key=config.data.coinglass_api_key
        )
        
        # Enhanced sentiment aggregator
        self._sentiment_aggregator = EnhancedSentimentAggregator()
        self._fear_greed: Optional[FearGreedData] = None
        self._technical_sentiment: dict[str, TechnicalSentiment] = {}
        self._enhanced_sentiment: dict[str, EnhancedSentiment] = {}
        
        # Callbacks for real-time updates
        self._callbacks: dict[str, list[Callable]] = {
            "ohlcv": [],
            "funding": [],
            "oi": [],
            "liquidation": [],
            "orderbook": [],
            "sentiment": [],
            "onchain": [],
        }
        
        # Cache settings
        self._max_candles = 500
        self._max_liquidations = 100
        self._orderbook_depth = 20
        
        logger.info("Market Intelligence Layer initialized with enhanced data providers")
    
    async def setup(self) -> None:
        """Initialize exchange connections."""
        import ccxt.async_support as ccxt
        
        exchange_id = self.config.exchange.primary_exchange
        
        if exchange_id == "binance":
            self._exchange = ccxt.binanceusdm({
                'apiKey': self.config.exchange.binance_api_key,
                'secret': self.config.exchange.binance_api_secret,
                'sandbox': self.config.exchange.binance_testnet,
                'options': {
                    'defaultType': 'future',
                }
            })
        elif exchange_id == "bybit":
            self._exchange = ccxt.bybit({
                'apiKey': self.config.exchange.bybit_api_key,
                'secret': self.config.exchange.bybit_api_secret,
                'sandbox': self.config.exchange.bybit_testnet,
                'options': {
                    'defaultType': 'linear',
                }
            })
        
        await self._exchange.load_markets()
        
        # Initialize data stores for each symbol
        for symbol in self.config.trading.symbols:
            self._ohlcv[symbol] = {tf: [] for tf in self.config.system.timeframes}
            self._liquidations[symbol] = []
            self._cvd[symbol] = 0.0
            self._cvd_history[symbol] = []
        
        # Initialize data aggregator
        await self._data_aggregator.initialize()
        
        # Load initial data (OHLCV + immediate funding/OI/orderbook)
        await self._load_historical_data()
        await self._load_immediate_data()
        
        # Load Binance positioning data (L/S ratio, taker volume)
        await self._load_binance_positioning_data()
        
        # Load enhanced data (on-chain, sentiment, etc.)
        await self._load_enhanced_data()
        
        logger.info(f"Exchange {exchange_id} connected, {len(self._exchange.markets)} markets loaded")
    
    async def _load_historical_data(self) -> None:
        """Load historical OHLCV data for all symbols and timeframes."""
        for symbol in self.config.trading.symbols:
            exchange_symbol = _to_exchange_symbol(symbol)
            for timeframe in self.config.system.timeframes:
                try:
                    ohlcv = await self._exchange.fetch_ohlcv(
                        exchange_symbol, timeframe, limit=self._max_candles
                    )
                    
                    candles = [
                        OHLCV(
                            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=row[5],
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                        for row in ohlcv
                    ]
                    
                    self._ohlcv[symbol][timeframe] = candles
                    logger.debug(f"Loaded {len(candles)} {timeframe} candles for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {timeframe} data for {symbol}: {e}")
                
                await asyncio.sleep(0.1)  # Rate limit
    
    async def _load_immediate_data(self) -> None:
        """Load funding, OI, and orderbook data immediately on setup."""
        logger.info("Loading immediate market data (funding, OI, orderbook)...")
        
        for symbol in self.config.trading.symbols:
            exchange_symbol = _to_exchange_symbol(symbol)
            
            # Fetch funding rate
            try:
                funding = await self._exchange.fetch_funding_rate(exchange_symbol)
                self._funding[symbol] = FundingRate(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    rate=funding.get('fundingRate', 0),
                    predicted_rate=funding.get('fundingRatePredicted'),
                    next_funding_time=datetime.fromtimestamp(
                        funding.get('fundingTimestamp', 0) / 1000, tz=timezone.utc
                    ) if funding.get('fundingTimestamp') else None,
                )
                logger.debug(f"Loaded funding rate for {symbol}: {funding.get('fundingRate', 0):.6f}")
            except Exception as e:
                logger.debug(f"Failed to load funding for {symbol}: {e}")
            
            await asyncio.sleep(0.05)
            
            # Fetch open interest
            try:
                oi_value = 0.0
                oi_amount = 0.0
                
                # Get current price first for USD calculation
                ticker = await self._exchange.fetch_ticker(exchange_symbol)
                last_price = float(ticker.get('last') or 0)
                
                if hasattr(self._exchange, 'fetch_open_interest'):
                    oi_data = await self._exchange.fetch_open_interest(exchange_symbol)
                    oi_amount = float(oi_data.get('openInterestAmount') or 0)
                    # openInterestValue is often None, calculate from price
                    oi_value = float(oi_data.get('openInterestValue') or 0)
                    if oi_value == 0 and oi_amount > 0 and last_price > 0:
                        oi_value = oi_amount * last_price
                else:
                    oi_raw = ticker.get('info', {}).get('openInterest')
                    oi_amount = float(oi_raw) if oi_raw else 0
                    oi_value = oi_amount * last_price
                
                # Calculate delta from previous OI
                prev_oi = self._open_interest.get(symbol)
                delta = 0.0
                delta_pct = 0.0
                if prev_oi and prev_oi.open_interest_usd > 0:
                    delta = oi_value - prev_oi.open_interest_usd
                    delta_pct = (delta / prev_oi.open_interest_usd) * 100
                
                self._open_interest[symbol] = OpenInterest(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    open_interest=oi_amount,
                    open_interest_usd=oi_value,
                    delta=delta,
                    delta_pct=delta_pct,
                )
                logger.info(f"OI {symbol}: {oi_amount:,.0f} contracts = ${oi_value/1e6:.1f}M (Δ{delta_pct:+.2f}%)")
            except Exception as e:
                logger.warning(f"Failed to load OI for {symbol}: {e}")
            
            await asyncio.sleep(0.05)
            
            # Fetch orderbook
            try:
                book = await self._exchange.fetch_order_book(exchange_symbol, limit=self._orderbook_depth)
                self._order_books[symbol] = OrderBookSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    bids=[(b[0], b[1]) for b in book.get('bids', [])],
                    asks=[(a[0], a[1]) for a in book.get('asks', [])],
                )
                logger.debug(f"Loaded orderbook for {symbol}")
            except Exception as e:
                logger.debug(f"Failed to load orderbook for {symbol}: {e}")
            
            await asyncio.sleep(0.05)
        
        logger.info("Immediate market data loaded")
    
    async def _load_enhanced_data(self) -> None:
        """Load enhanced data from external providers (on-chain, news, social, liquidations)."""
        logger.info("Loading enhanced market intelligence data...")
        
        # Fetch Fear & Greed Index (global, very important)
        try:
            self._fear_greed = await self._sentiment_aggregator.get_fear_greed()
            logger.info(f"Fear & Greed Index: {self._fear_greed.value} ({self._fear_greed.classification})")
        except Exception as e:
            logger.warning(f"Failed to load Fear & Greed Index: {e}")
        
        # Fetch stablecoin metrics (global)
        try:
            self._stablecoin_metrics = await self._data_aggregator.fetch_stablecoin_metrics()
            logger.debug(f"Loaded stablecoin metrics: USDT dominance {self._stablecoin_metrics.get('usdt_dominance', 0):.2%}")
        except Exception as e:
            logger.debug(f"Failed to load stablecoin metrics: {e}")
        
        # Fetch per-symbol enhanced data in parallel batches
        for symbol in self.config.trading.symbols:
            try:
                all_data = await self._data_aggregator.fetch_all_data(symbol)
                
                if all_data.get("onchain"):
                    self._onchain_data[symbol] = all_data["onchain"]
                    logger.debug(f"Loaded on-chain data for {symbol}")
                
                if all_data.get("news"):
                    self._news_sentiment[symbol] = all_data["news"]
                    logger.debug(f"Loaded news sentiment for {symbol}: {all_data['news'].news_count_24h} articles")
                
                if all_data.get("social"):
                    self._social_sentiment[symbol] = all_data["social"]
                    logger.debug(f"Loaded social sentiment for {symbol}: {all_data['social'].sentiment_trend}")
                
                if all_data.get("liquidations"):
                    self._liquidation_data[symbol] = all_data["liquidations"]
                    liq = all_data["liquidations"]
                    logger.debug(f"Loaded liquidations for {symbol}: ${liq.long_liquidations_24h + liq.short_liquidations_24h:,.0f} 24h")
                
                if all_data.get("positioning"):
                    self._positioning_data[symbol] = all_data["positioning"]
                    logger.debug(f"Loaded positioning for {symbol}: L/S ratio {all_data['positioning'].long_short_ratio:.2f}")
                
            except Exception as e:
                logger.debug(f"Failed to load enhanced data for {symbol}: {e}")
            
            await asyncio.sleep(0.2)  # Rate limit for external APIs
        
        # Calculate technical sentiment and enhanced sentiment for each symbol
        for symbol in self.config.trading.symbols:
            self._update_enhanced_sentiment(symbol)
        
        logger.info("Enhanced market intelligence data loaded")
    
    async def refresh_all_data(self) -> None:
        """Refresh all market data - call this each cycle for fresh data."""
        logger.info("Refreshing all market data...")
        await self._load_historical_data()
        await self._load_immediate_data()
        await self._load_binance_positioning_data()
        await self._load_enhanced_data()
        logger.info("All market data refreshed")
    
    async def _load_binance_positioning_data(self) -> None:
        """Fetch Long/Short ratio and Taker volume from Binance public API."""
        logger.info("Loading Binance positioning data (Long/Short ratio, Taker volume)...")
        
        async with aiohttp.ClientSession() as session:
            for symbol in self.config.trading.symbols:
                # Convert to Binance symbol format (e.g., BTCUSDT)
                binance_symbol = symbol.replace("cmt_", "").upper()
                
                try:
                    # Fetch Global Long/Short Account Ratio
                    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                    params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data and len(data) > 0:
                                latest = data[0]
                                self._long_short_ratio[symbol] = {
                                    'longAccount': float(latest.get('longAccount', 0)),
                                    'shortAccount': float(latest.get('shortAccount', 0)),
                                    'ratio': float(latest.get('longShortRatio', 1)),
                                    'timestamp': latest.get('timestamp', 0),
                                }
                                logger.info(f"L/S Ratio {symbol}: {self._long_short_ratio[symbol]['ratio']:.2f} (L:{self._long_short_ratio[symbol]['longAccount']:.1%} S:{self._long_short_ratio[symbol]['shortAccount']:.1%})")
                    
                    await asyncio.sleep(0.1)  # Rate limit
                    
                    # Fetch Taker Buy/Sell Volume
                    url2 = "https://fapi.binance.com/futures/data/takerlongshortRatio"
                    params2 = {"symbol": binance_symbol, "period": "5m", "limit": 1}
                    async with session.get(url2, params=params2) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data and len(data) > 0:
                                latest = data[0]
                                self._taker_volume[symbol] = {
                                    'buyVol': float(latest.get('buyVol', 0)),
                                    'sellVol': float(latest.get('sellVol', 0)),
                                    'ratio': float(latest.get('buySellRatio', 1)),
                                    'timestamp': latest.get('timestamp', 0),
                                }
                                logger.debug(f"Taker Volume {symbol}: Buy/Sell ratio {self._taker_volume[symbol]['ratio']:.2f}")
                    
                    await asyncio.sleep(0.1)  # Rate limit
                    
                except Exception as e:
                    logger.debug(f"Failed to load Binance positioning data for {symbol}: {e}")
        
        logger.info("Binance positioning data loaded")
    
    def get_long_short_ratio(self, symbol: str) -> Optional[dict]:
        """Get Long/Short account ratio for a symbol."""
        return self._long_short_ratio.get(symbol)
    
    def get_taker_volume(self, symbol: str) -> Optional[dict]:
        """Get Taker buy/sell volume ratio for a symbol."""
        return self._taker_volume.get(symbol)
    
    async def start_feeds(self) -> None:
        """Start real-time data feeds."""
        self._running = True
        
        # Start update tasks
        asyncio.create_task(self._ohlcv_update_loop())
        asyncio.create_task(self._funding_update_loop())
        asyncio.create_task(self._oi_update_loop())
        asyncio.create_task(self._orderbook_update_loop())
        asyncio.create_task(self._liquidation_watch_loop())
        asyncio.create_task(self._enhanced_data_update_loop())
        asyncio.create_task(self._cvd_update_loop())
        
        logger.info("Real-time data feeds started (including enhanced intelligence)")
    
    async def stop_feeds(self) -> None:
        """Stop all data feeds."""
        self._running = False
        
        if self._exchange:
            await self._exchange.close()
        
        # Close data aggregator sessions
        await self._data_aggregator.close()
        
        logger.info("Data feeds stopped")
    
    async def _ohlcv_update_loop(self) -> None:
        """Periodically update OHLCV data."""
        interval = self.config.system.ohlcv_update_interval
        
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    exchange_symbol = _to_exchange_symbol(symbol)
                    for timeframe in self.config.system.timeframes:
                        # Fetch latest candle
                        ohlcv = await self._exchange.fetch_ohlcv(
                            exchange_symbol, timeframe, limit=2
                        )
                        
                        if ohlcv:
                            latest = OHLCV(
                                timestamp=datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=timezone.utc),
                                open=ohlcv[-1][1],
                                high=ohlcv[-1][2],
                                low=ohlcv[-1][3],
                                close=ohlcv[-1][4],
                                volume=ohlcv[-1][5],
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                            
                            # Update or append
                            candles = self._ohlcv[symbol][timeframe]
                            if candles and candles[-1].timestamp == latest.timestamp:
                                candles[-1] = latest
                            else:
                                candles.append(latest)
                                if len(candles) > self._max_candles:
                                    candles.pop(0)
                        
                        await asyncio.sleep(0.05)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"OHLCV update error: {e}")
                await asyncio.sleep(1)
    
    async def _funding_update_loop(self) -> None:
        """Update funding rate data."""
        interval = self.config.system.funding_update_interval
        
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    exchange_symbol = _to_exchange_symbol(symbol)
                    funding = await self._exchange.fetch_funding_rate(exchange_symbol)
                    
                    self._funding[symbol] = FundingRate(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        rate=funding.get('fundingRate', 0),
                        predicted_rate=funding.get('fundingRatePredicted'),
                        next_funding_time=datetime.fromtimestamp(
                            funding.get('fundingTimestamp', 0) / 1000, tz=timezone.utc
                        ) if funding.get('fundingTimestamp') else None,
                    )
                    
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Funding update error: {e}")
                await asyncio.sleep(5)
    
    async def _oi_update_loop(self) -> None:
        """Update open interest data."""
        interval = self.config.system.oi_update_interval
        previous_oi: dict[str, float] = {}
        
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    exchange_symbol = _to_exchange_symbol(symbol)
                    try:
                        # Get price first for USD calculation
                        ticker = await self._exchange.fetch_ticker(exchange_symbol)
                        last_price = float(ticker.get('last') or 0)
                        
                        # Fetch OI
                        if hasattr(self._exchange, 'fetch_open_interest'):
                            oi_data = await self._exchange.fetch_open_interest(exchange_symbol)
                            oi_amount = float(oi_data.get('openInterestAmount') or 0)
                            # Calculate USD value (often None from API)
                            oi_value = float(oi_data.get('openInterestValue') or 0)
                            if oi_value == 0 and oi_amount > 0 and last_price > 0:
                                oi_value = oi_amount * last_price
                        else:
                            oi_raw = ticker.get('info', {}).get('openInterest', 0)
                            oi_amount = float(oi_raw) if oi_raw else 0
                            oi_value = oi_amount * last_price
                        
                        prev = previous_oi.get(symbol, oi_value)
                        delta = oi_value - prev
                        delta_pct = (delta / prev * 100) if prev > 0 else 0
                        
                        self._open_interest[symbol] = OpenInterest(
                            timestamp=datetime.now(timezone.utc),
                            symbol=symbol,
                            open_interest=oi_amount,
                            open_interest_usd=oi_value,
                            delta=delta,
                            delta_pct=delta_pct,
                        )
                        
                        previous_oi[symbol] = oi_value
                        
                    except Exception as e:
                        logger.debug(f"OI fetch failed for {symbol}: {e}")
                    
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"OI update error: {e}")
                await asyncio.sleep(5)
    
    async def _orderbook_update_loop(self) -> None:
        """Update order book snapshots."""
        interval = self.config.system.orderbook_update_interval
        
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    exchange_symbol = _to_exchange_symbol(symbol)
                    try:
                        book = await self._exchange.fetch_order_book(
                            exchange_symbol, limit=self._orderbook_depth
                        )
                        
                        self._order_books[symbol] = OrderBookSnapshot(
                            timestamp=datetime.now(timezone.utc),
                            symbol=symbol,
                            bids=[(b[0], b[1]) for b in book.get('bids', [])],
                            asks=[(a[0], a[1]) for a in book.get('asks', [])],
                        )
                        
                    except Exception as e:
                        logger.debug(f"Orderbook fetch failed for {symbol}: {e}")
                    
                    await asyncio.sleep(0.05)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Orderbook update error: {e}")
                await asyncio.sleep(1)
    
    async def _liquidation_watch_loop(self) -> None:
        """Monitor for liquidation events using Binance forceOrders API."""
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    try:
                        # Use our liquidation provider for real data
                        liq_data = await self._data_aggregator.liquidations.fetch_liquidations(symbol)
                        if liq_data:
                            self._liquidation_data[symbol] = liq_data
                    except Exception as e:
                        logger.debug(f"Liquidation fetch failed for {symbol}: {e}")
                    
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Liquidation watch error: {e}")
                await asyncio.sleep(5)
    
    async def _enhanced_data_update_loop(self) -> None:
        """Periodically update enhanced data (news, social, on-chain, positioning)."""
        interval = 300  # 5 minutes for external APIs
        
        while self._running:
            try:
                # Update stablecoin metrics
                try:
                    self._stablecoin_metrics = await self._data_aggregator.fetch_stablecoin_metrics()
                except Exception as e:
                    logger.debug(f"Stablecoin metrics update failed: {e}")
                
                # Update per-symbol enhanced data
                for symbol in self.config.trading.symbols:
                    try:
                        all_data = await self._data_aggregator.fetch_all_data(symbol)
                        
                        if all_data.get("onchain"):
                            self._onchain_data[symbol] = all_data["onchain"]
                        if all_data.get("news"):
                            self._news_sentiment[symbol] = all_data["news"]
                        if all_data.get("social"):
                            self._social_sentiment[symbol] = all_data["social"]
                        if all_data.get("positioning"):
                            self._positioning_data[symbol] = all_data["positioning"]
                            
                    except Exception as e:
                        logger.debug(f"Enhanced data update failed for {symbol}: {e}")
                    
                    await asyncio.sleep(0.5)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Enhanced data update error: {e}")
                await asyncio.sleep(30)
    
    async def _cvd_update_loop(self) -> None:
        """Track Cumulative Volume Delta from recent trades."""
        while self._running:
            try:
                for symbol in self.config.trading.symbols:
                    exchange_symbol = _to_exchange_symbol(symbol)
                    try:
                        # Fetch recent trades
                        trades = await self._exchange.fetch_trades(exchange_symbol, limit=100)
                        
                        if trades:
                            # Calculate delta: buy volume - sell volume
                            buy_vol = sum(t['amount'] for t in trades if t.get('side') == 'buy')
                            sell_vol = sum(t['amount'] for t in trades if t.get('side') == 'sell')
                            delta = buy_vol - sell_vol
                            
                            # Update CVD
                            self._cvd[symbol] += delta
                            
                            # Store history (keep last 100 points)
                            now = datetime.now(timezone.utc)
                            self._cvd_history[symbol].append((now, self._cvd[symbol]))
                            if len(self._cvd_history[symbol]) > 100:
                                self._cvd_history[symbol].pop(0)
                                
                    except Exception as e:
                        logger.debug(f"CVD update failed for {symbol}: {e}")
                    
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"CVD update error: {e}")
                await asyncio.sleep(5)
    
    async def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Get complete market state for a symbol."""
        if symbol not in self._ohlcv:
            return None
        
        # Get latest price from 1m candles
        candles_1m = self._ohlcv[symbol].get("1m", [])
        if not candles_1m:
            return None
        
        latest_candle = candles_1m[-1]
        current_price = latest_candle.close
        
        # Calculate 24h metrics
        price_24h_ago = candles_1m[-min(1440, len(candles_1m))].close if len(candles_1m) > 1 else current_price
        price_change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
        volume_24h = sum(c.volume for c in candles_1m[-min(1440, len(candles_1m)):])
        
        # Calculate volatility (24h)
        if len(candles_1m) > 20:
            prices = np.array([c.close for c in candles_1m[-100:]])
            returns = np.diff(prices) / prices[:-1]  # n-1 diffs / n-1 prices
            volatility = np.std(returns) * np.sqrt(1440)  # Annualized from 1m
        else:
            volatility = 0.0
        
        # Get funding and OI
        funding = self._funding.get(symbol)
        oi = self._open_interest.get(symbol)
        orderbook = self._order_books.get(symbol)
        liquidations = self._liquidations.get(symbol, [])
        
        # Basis calculation
        exchange_symbol = _to_exchange_symbol(symbol)
        try:
            ticker = await self._exchange.fetch_ticker(exchange_symbol)
            index_price = ticker.get('info', {}).get('indexPrice', current_price)
            mark_price = ticker.get('info', {}).get('markPrice', current_price)
            index_price = float(index_price) if index_price else current_price
            mark_price = float(mark_price) if mark_price else current_price
            basis = (mark_price - index_price) / index_price if index_price > 0 else 0
        except Exception:
            index_price = current_price
            mark_price = current_price
            basis = 0
        
        return MarketState(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            price=current_price,
            ohlcv=self._ohlcv[symbol],
            funding_rate=funding,
            open_interest=oi,
            recent_liquidations=liquidations[-20:],
            order_book=orderbook,
            volatility=volatility,
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            index_price=index_price,
            mark_price=mark_price,
            basis=basis,
        )
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> list[OHLCV]:
        """Get recent candles for a symbol and timeframe."""
        candles = self._ohlcv.get(symbol, {}).get(timeframe, [])
        return candles[-limit:]
    
    def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Get current funding rate for a symbol."""
        return self._funding.get(symbol)
    
    def get_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        """Get current open interest for a symbol."""
        return self._open_interest.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot."""
        return self._order_books.get(symbol)
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for real-time updates."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    # =========================================================================
    # ENHANCED DATA GETTERS
    # =========================================================================
    
    def get_news_sentiment(self, symbol: str) -> Optional[NewsSentiment]:
        """Get news sentiment for a symbol."""
        return self._news_sentiment.get(symbol)
    
    def get_social_sentiment(self, symbol: str) -> Optional[SocialSentiment]:
        """Get social sentiment for a symbol."""
        return self._social_sentiment.get(symbol)
    
    def get_onchain_data(self, symbol: str) -> Optional[OnChainData]:
        """Get on-chain data for a symbol."""
        return self._onchain_data.get(symbol)
    
    def get_liquidation_data(self, symbol: str) -> Optional[LiquidationData]:
        """Get liquidation data for a symbol."""
        return self._liquidation_data.get(symbol)
    
    def get_positioning_data(self, symbol: str) -> Optional[PositioningData]:
        """Get market positioning data for a symbol."""
        return self._positioning_data.get(symbol)
    
    def get_stablecoin_metrics(self) -> dict:
        """Get global stablecoin metrics."""
        return self._stablecoin_metrics
    
    def get_cvd(self, symbol: str) -> float:
        """Get current CVD (Cumulative Volume Delta) for a symbol."""
        return self._cvd.get(symbol, 0.0)
    
    def get_cvd_history(self, symbol: str, limit: int = 50) -> list[tuple[datetime, float]]:
        """Get CVD history for a symbol."""
        history = self._cvd_history.get(symbol, [])
        return history[-limit:]
    
    def _update_enhanced_sentiment(self, symbol: str) -> None:
        """Calculate and update enhanced sentiment for a symbol."""
        # Calculate technical sentiment from candles
        candles = self._ohlcv.get(symbol, {}).get("5m", [])
        if candles:
            self._technical_sentiment[symbol] = self._sentiment_aggregator.calculate_technical(
                candles, symbol
            )
        
        # Get individual sentiment components
        news = self._news_sentiment.get(symbol)
        social = self._social_sentiment.get(symbol)
        funding = self._funding.get(symbol)
        positioning = self._positioning_data.get(symbol)
        onchain = self._onchain_data.get(symbol)
        technical = self._technical_sentiment.get(symbol)
        
        # Calculate news score with confidence
        news_score = 0.0
        news_confidence = 0.0
        if news and news.news_count_24h > 0:
            news_score = news.avg_sentiment_24h
            news_confidence = min(1.0, news.news_count_24h / 10)
        
        # Calculate social score with confidence
        social_score = 0.0
        social_confidence = 0.0
        if social:
            social_score = social.overall_sentiment
            social_confidence = 0.6 if social.reddit_posts_24h > 5 else 0.3
        
        # Aggregate using enhanced sentiment aggregator
        self._enhanced_sentiment[symbol] = self._sentiment_aggregator.aggregate_sentiment(
            symbol=symbol,
            fear_greed=self._fear_greed,
            news_score=news_score,
            news_confidence=news_confidence,
            social_score=social_score,
            social_confidence=social_confidence,
            technical=technical,
            funding_rate=funding.rate if funding else 0.0,
            long_short_ratio=positioning.long_short_ratio if positioning else 1.0,
            exchange_netflow=onchain.net_exchange_flow if onchain else 0.0,
        )
    
    def get_fear_greed(self) -> Optional[FearGreedData]:
        """Get current Fear & Greed Index."""
        return self._fear_greed
    
    def get_technical_sentiment(self, symbol: str) -> Optional[TechnicalSentiment]:
        """Get technical sentiment for a symbol."""
        return self._technical_sentiment.get(symbol)
    
    def get_enhanced_sentiment(self, symbol: str) -> Optional[EnhancedSentiment]:
        """Get enhanced sentiment (most comprehensive signal) for a symbol."""
        # Update if stale
        if symbol not in self._enhanced_sentiment:
            self._update_enhanced_sentiment(symbol)
        return self._enhanced_sentiment.get(symbol)
    
    def get_combined_sentiment(self, symbol: str) -> dict:
        """Get combined sentiment score from all sources (legacy - use get_enhanced_sentiment)."""
        # Use enhanced sentiment if available
        enhanced = self._enhanced_sentiment.get(symbol)
        if enhanced:
            return {
                "symbol": symbol,
                "overall_score": enhanced.overall_score,
                "news_score": enhanced.news_score,
                "social_score": enhanced.social_score,
                "funding_score": enhanced.funding_score,
                "positioning_score": enhanced.positioning_score,
                "technical_score": enhanced.technical_score,
                "fear_greed_score": enhanced.fear_greed_score,
                "signal": enhanced.signal,
                "confidence": enhanced.confidence,
                "regime": enhanced.regime,
                "key_factors": enhanced.key_factors,
                "warnings": enhanced.warnings,
            }
        
        # Fallback to basic calculation
        result = {
            "symbol": symbol,
            "overall_score": 0.0,
            "news_score": 0.0,
            "social_score": 0.0,
            "funding_score": 0.0,
            "positioning_score": 0.0,
            "signal": "neutral",
        }
        
        weights = {"news": 0.25, "social": 0.25, "funding": 0.3, "positioning": 0.2}
        scores = []
        
        # News sentiment
        news = self._news_sentiment.get(symbol)
        if news:
            result["news_score"] = news.avg_sentiment_24h
            scores.append(("news", news.avg_sentiment_24h))
        
        # Social sentiment
        social = self._social_sentiment.get(symbol)
        if social:
            result["social_score"] = social.overall_sentiment
            scores.append(("social", social.overall_sentiment))
        
        # Funding sentiment (inverted - high funding = bearish for longs)
        funding = self._funding.get(symbol)
        if funding:
            funding_sentiment = -np.tanh(funding.rate * 1000)
            result["funding_score"] = funding_sentiment
            scores.append(("funding", funding_sentiment))
        
        # Positioning sentiment
        positioning = self._positioning_data.get(symbol)
        if positioning:
            # High L/S ratio = crowded long = bearish contrarian signal
            pos_sentiment = -np.tanh((positioning.long_short_ratio - 1) * 2)
            result["positioning_score"] = pos_sentiment
            scores.append(("positioning", pos_sentiment))
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights.get(s[0], 0.25) for s in scores)
            weighted_sum = sum(weights.get(s[0], 0.25) * s[1] for s in scores)
            result["overall_score"] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine signal
        if result["overall_score"] > 0.3:
            result["signal"] = "bullish"
        elif result["overall_score"] < -0.3:
            result["signal"] = "bearish"
        else:
            result["signal"] = "neutral"
        
        return result
    
    def get_market_intelligence_summary(self, symbol: str) -> dict:
        """Get comprehensive market intelligence summary for a symbol."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price_data": {
                "funding": self._funding.get(symbol),
                "open_interest": self._open_interest.get(symbol),
                "orderbook": self._order_books.get(symbol),
            },
            "sentiment": self.get_combined_sentiment(symbol),
            "news": self._news_sentiment.get(symbol),
            "social": self._social_sentiment.get(symbol),
            "onchain": self._onchain_data.get(symbol),
            "liquidations": self._liquidation_data.get(symbol),
            "positioning": self._positioning_data.get(symbol),
            "cvd": self._cvd.get(symbol, 0.0),
            "stablecoin_metrics": self._stablecoin_metrics,
        }
    
    # Legacy methods for backward compatibility
    async def fetch_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Fetch sentiment data (legacy - use get_combined_sentiment instead)."""
        funding = self._funding.get(symbol)
        social = self._social_sentiment.get(symbol)
        news = self._news_sentiment.get(symbol)
        
        funding_sentiment = -np.tanh(funding.rate * 1000) if funding else 0
        crowd_positioning = np.tanh(funding.rate * 500) if funding else 0
        social_score = social.overall_sentiment if social else 0
        
        return SentimentData(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            social_score=social_score,
            funding_sentiment=funding_sentiment,
            crowd_positioning=crowd_positioning,
            narrative_velocity=news.sentiment_momentum if news else 0,
            fear_greed_index=50.0 + (funding_sentiment * 25),
        )
    
    async def fetch_onchain_metrics(self, symbol: str) -> Optional[OnChainMetrics]:
        """Fetch on-chain metrics (legacy - use get_onchain_data instead)."""
        onchain = self._onchain_data.get(symbol)
        if onchain:
            return OnChainMetrics(
                timestamp=onchain.timestamp,
                symbol=symbol,
                exchange_inflow_usd=onchain.exchange_inflow_usd,
                exchange_outflow_usd=onchain.exchange_outflow_usd,
                net_flow_usd=onchain.net_exchange_flow,
                whale_transactions=onchain.large_tx_count_24h,
                stablecoin_flow=self._stablecoin_metrics.get("stablecoin_netflow_7d", 0),
            )
        return None
