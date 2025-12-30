"""
Enhanced Sentiment Analysis for Layer 1

Provides much more accurate and actionable sentiment signals:
- Fear & Greed Index (Alternative.me)
- Advanced NLP-like sentiment scoring
- Technical sentiment from price action
- Market regime detection
- Volume-weighted signals
- Multi-source aggregation with confidence scoring
"""

from __future__ import annotations

import asyncio
import aiohttp
import re
import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger


# =============================================================================
# ENHANCED DATA CLASSES
# =============================================================================

@dataclass
class FearGreedData:
    """Fear & Greed Index data."""
    timestamp: datetime
    value: int  # 0-100
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    value_yesterday: int
    value_week_ago: int
    value_month_ago: int
    trend: str  # improving, worsening, stable


@dataclass
class TechnicalSentiment:
    """Technical analysis based sentiment."""
    timestamp: datetime
    symbol: str
    # Trend indicators
    trend_1h: str  # bullish, bearish, neutral
    trend_4h: str
    trend_1d: str
    trend_strength: float  # 0-1
    # Momentum
    rsi_14: float
    rsi_signal: str  # overbought, oversold, neutral
    # Volatility regime
    volatility_percentile: float  # 0-100 vs historical
    volatility_regime: str  # low, normal, high, extreme
    # Price action
    price_vs_ema20: float  # % above/below
    price_vs_ema50: float
    higher_highs: bool
    lower_lows: bool
    # Combined score
    technical_score: float  # -1 to 1


@dataclass 
class EnhancedSentiment:
    """Combined enhanced sentiment from all sources."""
    timestamp: datetime
    symbol: str
    # Individual scores (-1 to 1)
    fear_greed_score: float
    news_score: float
    social_score: float
    technical_score: float
    funding_score: float
    positioning_score: float
    onchain_score: float
    # Combined
    overall_score: float
    confidence: float  # 0-1 based on data availability
    signal: str  # strong_buy, buy, neutral, sell, strong_sell
    regime: str  # risk_on, risk_off, uncertain
    # Actionable insights
    key_factors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# FEAR & GREED INDEX PROVIDER
# =============================================================================

class FearGreedProvider:
    """
    Fetches Fear & Greed Index from Alternative.me (free, reliable).
    This is one of the best sentiment indicators available.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Optional[tuple[datetime, FearGreedData]] = None
        self._cache_ttl = 300  # 5 minutes
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_fear_greed(self) -> FearGreedData:
        """Fetch current Fear & Greed Index."""
        now = datetime.now(timezone.utc)
        
        # Check cache
        if self._cache:
            cache_time, cached_data = self._cache
            if (now - cache_time).total_seconds() < self._cache_ttl:
                return cached_data
        
        session = await self._get_session()
        
        try:
            # Alternative.me Fear & Greed API (free, no key needed)
            url = "https://api.alternative.me/fng/?limit=31"
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fng_data = data.get("data", [])
                    
                    if fng_data:
                        current = fng_data[0]
                        yesterday = fng_data[1] if len(fng_data) > 1 else current
                        week_ago = fng_data[7] if len(fng_data) > 7 else current
                        month_ago = fng_data[30] if len(fng_data) > 30 else current
                        
                        value = int(current.get("value", 50))
                        value_yesterday = int(yesterday.get("value", 50))
                        
                        # Determine trend
                        if value > value_yesterday + 5:
                            trend = "improving"
                        elif value < value_yesterday - 5:
                            trend = "worsening"
                        else:
                            trend = "stable"
                        
                        result = FearGreedData(
                            timestamp=now,
                            value=value,
                            classification=current.get("value_classification", "Neutral"),
                            value_yesterday=value_yesterday,
                            value_week_ago=int(week_ago.get("value", 50)),
                            value_month_ago=int(month_ago.get("value", 50)),
                            trend=trend,
                        )
                        
                        self._cache = (now, result)
                        return result
                        
        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")
        
        # Return default if fetch fails
        return FearGreedData(
            timestamp=now,
            value=50,
            classification="Neutral",
            value_yesterday=50,
            value_week_ago=50,
            value_month_ago=50,
            trend="stable",
        )


# =============================================================================
# ADVANCED SENTIMENT ANALYZER
# =============================================================================

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis with:
    - Weighted keyword matching
    - Context awareness
    - Intensity detection
    - Negation handling
    """
    
    # Weighted bullish keywords (word -> weight)
    BULLISH_KEYWORDS = {
        # Strong bullish (weight 2.0)
        "moon": 2.0, "mooning": 2.0, "skyrocket": 2.0, "parabolic": 2.0,
        "all-time high": 2.0, "ath": 2.0, "new high": 2.0,
        "etf approved": 2.0, "etf approval": 2.0, "sec approval": 2.0,
        "massive adoption": 2.0, "institutional buying": 2.0,
        "huge pump": 2.0, "mega rally": 2.0,
        
        # Medium bullish (weight 1.5)
        "bullish": 1.5, "surge": 1.5, "soar": 1.5, "rally": 1.5,
        "breakout": 1.5, "breaking out": 1.5, "pump": 1.5, "pumping": 1.5,
        "accumulate": 1.5, "accumulation": 1.5, "buy the dip": 1.5,
        "institutional": 1.5, "adoption": 1.5, "partnership": 1.5,
        "upgrade": 1.5, "launch": 1.5, "launched": 1.5,
        "milestone": 1.5, "record": 1.5,
        
        # Light bullish (weight 1.0)
        "buy": 1.0, "long": 1.0, "hold": 1.0, "hodl": 1.0,
        "positive": 1.0, "optimistic": 1.0, "confident": 1.0,
        "growth": 1.0, "growing": 1.0, "gains": 1.0, "profit": 1.0,
        "green": 1.0, "up": 1.0, "higher": 1.0, "rising": 1.0,
        "support": 1.0, "bounce": 1.0, "recovery": 1.0,
        "diamond hands": 1.0, "whale": 1.0, "whales buying": 1.5,
        
        # Weak bullish (weight 0.5)
        "stable": 0.5, "steady": 0.5, "holding": 0.5,
    }
    
    # Weighted bearish keywords
    BEARISH_KEYWORDS = {
        # Strong bearish (weight 2.0)
        "crash": 2.0, "crashing": 2.0, "collapse": 2.0, "collapsing": 2.0,
        "plummet": 2.0, "plunge": 2.0, "tank": 2.0, "tanking": 2.0,
        "bankrupt": 2.0, "bankruptcy": 2.0, "insolvent": 2.0,
        "rug pull": 2.0, "rugpull": 2.0, "scam": 2.0, "fraud": 2.0,
        "hack": 2.0, "hacked": 2.0, "exploit": 2.0, "exploited": 2.0,
        "death cross": 2.0, "capitulation": 2.0,
        
        # Medium bearish (weight 1.5)
        "bearish": 1.5, "dump": 1.5, "dumping": 1.5, "sell-off": 1.5,
        "correction": 1.5, "correcting": 1.5, "pullback": 1.5,
        "fear": 1.5, "panic": 1.5, "fud": 1.5,
        "sec": 1.5, "lawsuit": 1.5, "sued": 1.5, "investigation": 1.5,
        "ban": 1.5, "banned": 1.5, "regulation": 1.5, "crackdown": 1.5,
        "warning": 1.5, "risk": 1.5, "danger": 1.5,
        "liquidation": 1.5, "liquidated": 1.5, "rekt": 1.5,
        
        # Light bearish (weight 1.0)
        "sell": 1.0, "short": 1.0, "exit": 1.0,
        "negative": 1.0, "pessimistic": 1.0, "worried": 1.0,
        "decline": 1.0, "declining": 1.0, "drop": 1.0, "dropping": 1.0,
        "down": 1.0, "lower": 1.0, "falling": 1.0, "fell": 1.0,
        "red": 1.0, "loss": 1.0, "losses": 1.0,
        "resistance": 1.0, "rejected": 1.0, "rejection": 1.0,
        "paper hands": 1.0,
        
        # Weak bearish (weight 0.5)
        "uncertainty": 0.5, "uncertain": 0.5, "volatile": 0.5,
    }
    
    # Negation words that flip sentiment
    NEGATION_WORDS = [
        "not", "no", "never", "neither", "nobody", "nothing",
        "won't", "wouldn't", "couldn't", "shouldn't", "don't", "doesn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "failed", "fails", "fail", "unlikely", "doubt", "doubtful"
    ]
    
    # Intensity modifiers
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "super": 1.5, "really": 1.3,
        "incredibly": 1.8, "absolutely": 1.8, "totally": 1.5,
        "massive": 1.8, "huge": 1.8, "major": 1.5, "significant": 1.3,
    }
    
    DIMINISHERS = {
        "slightly": 0.5, "somewhat": 0.7, "a bit": 0.6, "little": 0.5,
        "minor": 0.5, "small": 0.5, "modest": 0.7,
    }
    
    def analyze(self, text: str) -> tuple[float, float]:
        """
        Analyze text sentiment with advanced NLP-like scoring.
        Returns (sentiment_score, confidence).
        """
        if not text:
            return 0.0, 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        bullish_score = 0.0
        bearish_score = 0.0
        matches = 0
        
        # Check for keyword matches with context
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                # Check for negation
                negated = self._is_negated(text_lower, keyword)
                # Check for intensity
                intensity = self._get_intensity(text_lower, keyword)
                
                if negated:
                    bearish_score += weight * intensity * 0.8
                else:
                    bullish_score += weight * intensity
                matches += 1
        
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                negated = self._is_negated(text_lower, keyword)
                intensity = self._get_intensity(text_lower, keyword)
                
                if negated:
                    bullish_score += weight * intensity * 0.8
                else:
                    bearish_score += weight * intensity
                matches += 1
        
        # Calculate final score
        total = bullish_score + bearish_score
        if total == 0:
            return 0.0, 0.0
        
        # Normalize to -1 to 1
        sentiment = (bullish_score - bearish_score) / total
        
        # Confidence based on number of matches and total signal strength
        confidence = min(1.0, (matches / 5) * (total / 10))
        
        return sentiment, confidence
    
    def _is_negated(self, text: str, keyword: str) -> bool:
        """Check if keyword is negated."""
        # Find position of keyword
        pos = text.find(keyword)
        if pos == -1:
            return False
        
        # Check preceding words (within 5 words)
        before = text[max(0, pos-50):pos]
        for neg in self.NEGATION_WORDS:
            if neg in before.split()[-5:]:
                return True
        return False
    
    def _get_intensity(self, text: str, keyword: str) -> float:
        """Get intensity modifier for keyword."""
        pos = text.find(keyword)
        if pos == -1:
            return 1.0
        
        before = text[max(0, pos-30):pos].lower()
        
        for intensifier, mult in self.INTENSIFIERS.items():
            if intensifier in before:
                return mult
        
        for diminisher, mult in self.DIMINISHERS.items():
            if diminisher in before:
                return mult
        
        return 1.0


# =============================================================================
# TECHNICAL SENTIMENT CALCULATOR
# =============================================================================

class TechnicalSentimentCalculator:
    """
    Calculate sentiment from technical analysis of price data.
    """
    
    def calculate(self, candles: list, symbol: str) -> TechnicalSentiment:
        """Calculate technical sentiment from OHLCV candles."""
        now = datetime.now(timezone.utc)
        
        if not candles or len(candles) < 50:
            return self._default_result(symbol)
        
        # Extract prices
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        
        # Calculate EMAs
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        
        current_price = closes[-1]
        
        # Price vs EMAs
        price_vs_ema20 = (current_price - ema20) / ema20 * 100
        price_vs_ema50 = (current_price - ema50) / ema50 * 100
        
        # RSI
        rsi = self._calculate_rsi(closes, 14)
        
        # Determine RSI signal
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
        
        # Trend detection
        trend_1h = self._detect_trend(closes[-12:])  # Last 12 candles for 1h if 5m data
        trend_4h = self._detect_trend(closes[-48:])
        trend_1d = self._detect_trend(closes[-288:] if len(closes) >= 288 else closes)
        
        # Trend strength
        trend_strength = abs(price_vs_ema20) / 5  # Normalize
        trend_strength = min(1.0, trend_strength)
        
        # Volatility
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        current_vol = self._std(returns[-20:]) if len(returns) >= 20 else 0
        historical_vol = self._std(returns) if returns else 0
        
        if historical_vol > 0:
            vol_percentile = min(100, (current_vol / historical_vol) * 50)
        else:
            vol_percentile = 50
        
        # Volatility regime
        if vol_percentile > 80:
            vol_regime = "extreme"
        elif vol_percentile > 60:
            vol_regime = "high"
        elif vol_percentile < 30:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        # Higher highs / Lower lows
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        higher_highs = recent_highs[-1] > max(recent_highs[:-1]) if len(recent_highs) > 1 else False
        lower_lows = recent_lows[-1] < min(recent_lows[:-1]) if len(recent_lows) > 1 else False
        
        # Calculate technical score
        score = 0.0
        
        # EMA positioning
        if price_vs_ema20 > 2:
            score += 0.3
        elif price_vs_ema20 < -2:
            score -= 0.3
        
        if price_vs_ema50 > 3:
            score += 0.2
        elif price_vs_ema50 < -3:
            score -= 0.2
        
        # RSI
        if rsi < 30:
            score += 0.25  # Oversold = bullish
        elif rsi > 70:
            score -= 0.25  # Overbought = bearish
        elif rsi > 50:
            score += 0.1
        else:
            score -= 0.1
        
        # Trend alignment
        trend_scores = {"bullish": 0.15, "bearish": -0.15, "neutral": 0}
        score += trend_scores.get(trend_1h, 0)
        score += trend_scores.get(trend_4h, 0)
        score += trend_scores.get(trend_1d, 0)
        
        # Clamp to -1 to 1
        score = max(-1.0, min(1.0, score))
        
        return TechnicalSentiment(
            timestamp=now,
            symbol=symbol,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_1d=trend_1d,
            trend_strength=trend_strength,
            rsi_14=rsi,
            rsi_signal=rsi_signal,
            volatility_percentile=vol_percentile,
            volatility_regime=vol_regime,
            price_vs_ema20=price_vs_ema20,
            price_vs_ema50=price_vs_ema50,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
            technical_score=score,
        )
    
    def _default_result(self, symbol: str) -> TechnicalSentiment:
        return TechnicalSentiment(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            trend_1h="neutral",
            trend_4h="neutral",
            trend_1d="neutral",
            trend_strength=0.0,
            rsi_14=50.0,
            rsi_signal="neutral",
            volatility_percentile=50.0,
            volatility_regime="normal",
            price_vs_ema20=0.0,
            price_vs_ema50=0.0,
            higher_highs=False,
            lower_lows=False,
            technical_score=0.0,
        )
    
    def _ema(self, data: list, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _detect_trend(self, prices: list) -> str:
        """Detect trend from price series."""
        if len(prices) < 5:
            return "neutral"
        
        # Simple linear regression slope
        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n
        
        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "neutral"
        
        slope = numerator / denominator
        
        # Normalize slope by price
        normalized_slope = (slope / y_mean) * 100
        
        if normalized_slope > 0.1:
            return "bullish"
        elif normalized_slope < -0.1:
            return "bearish"
        else:
            return "neutral"
    
    def _std(self, data: list) -> float:
        """Calculate standard deviation."""
        if len(data) < 2:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)


# =============================================================================
# ENHANCED SENTIMENT AGGREGATOR
# =============================================================================

class EnhancedSentimentAggregator:
    """
    Combines all sentiment sources into actionable signals.
    Uses adaptive weighting based on data quality and market conditions.
    """
    
    def __init__(self):
        self.fear_greed = FearGreedProvider()
        self.text_analyzer = AdvancedSentimentAnalyzer()
        self.technical_calc = TechnicalSentimentCalculator()
        
        self._fear_greed_cache: Optional[FearGreedData] = None
    
    async def close(self):
        await self.fear_greed.close()
    
    async def get_fear_greed(self) -> FearGreedData:
        """Get Fear & Greed Index."""
        self._fear_greed_cache = await self.fear_greed.fetch_fear_greed()
        return self._fear_greed_cache
    
    def analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze text sentiment."""
        return self.text_analyzer.analyze(text)
    
    def calculate_technical(self, candles: list, symbol: str) -> TechnicalSentiment:
        """Calculate technical sentiment."""
        return self.technical_calc.calculate(candles, symbol)
    
    def aggregate_sentiment(
        self,
        symbol: str,
        fear_greed: Optional[FearGreedData] = None,
        news_score: float = 0.0,
        news_confidence: float = 0.0,
        social_score: float = 0.0,
        social_confidence: float = 0.0,
        technical: Optional[TechnicalSentiment] = None,
        funding_rate: float = 0.0,
        long_short_ratio: float = 1.0,
        exchange_netflow: float = 0.0,
    ) -> EnhancedSentiment:
        """
        Aggregate all sentiment sources with adaptive weighting.
        """
        now = datetime.now(timezone.utc)
        key_factors = []
        warnings = []
        
        # =====================================================================
        # 1. Fear & Greed Score (-1 to 1)
        # =====================================================================
        if fear_greed:
            # Convert 0-100 to -1 to 1
            fg_score = (fear_greed.value - 50) / 50
            fg_confidence = 0.9  # High confidence - reliable source
            
            # Add key factors
            if fear_greed.value <= 25:
                key_factors.append(f"Extreme Fear ({fear_greed.value}) - Contrarian BUY signal")
            elif fear_greed.value <= 40:
                key_factors.append(f"Fear ({fear_greed.value}) - Caution, potential opportunity")
            elif fear_greed.value >= 75:
                key_factors.append(f"Extreme Greed ({fear_greed.value}) - Contrarian SELL signal")
                warnings.append("Market euphoria - high risk of correction")
            elif fear_greed.value >= 60:
                key_factors.append(f"Greed ({fear_greed.value}) - Momentum but elevated risk")
            
            if fear_greed.trend == "worsening" and fear_greed.value < 40:
                warnings.append("Fear increasing - potential capitulation")
        else:
            fg_score = 0.0
            fg_confidence = 0.0
        
        # =====================================================================
        # 2. Technical Score
        # =====================================================================
        if technical:
            tech_score = technical.technical_score
            tech_confidence = 0.8  # Good confidence
            
            if technical.rsi_signal == "oversold":
                key_factors.append(f"RSI Oversold ({technical.rsi_14:.0f}) - Bounce potential")
            elif technical.rsi_signal == "overbought":
                key_factors.append(f"RSI Overbought ({technical.rsi_14:.0f}) - Pullback risk")
            
            if technical.trend_1d == "bullish" and technical.trend_4h == "bullish":
                key_factors.append("Multi-timeframe bullish trend alignment")
            elif technical.trend_1d == "bearish" and technical.trend_4h == "bearish":
                key_factors.append("Multi-timeframe bearish trend alignment")
            
            if technical.volatility_regime == "extreme":
                warnings.append("Extreme volatility - reduce position size")
        else:
            tech_score = 0.0
            tech_confidence = 0.0
        
        # =====================================================================
        # 3. Funding Rate Score (contrarian)
        # =====================================================================
        # High funding = crowded long = bearish contrarian signal
        funding_score = -math.tanh(funding_rate * 500)  # More sensitive
        funding_confidence = 0.7
        
        if abs(funding_rate) > 0.0005:  # 0.05%
            if funding_rate > 0:
                key_factors.append(f"High funding ({funding_rate*100:.3f}%) - Longs overleveraged")
            else:
                key_factors.append(f"Negative funding ({funding_rate*100:.3f}%) - Shorts overleveraged")
        
        # =====================================================================
        # 4. Positioning Score (contrarian)
        # =====================================================================
        # High L/S ratio = crowded long = bearish
        pos_score = -math.tanh((long_short_ratio - 1) * 1.5)  # More sensitive
        pos_confidence = 0.7
        
        if long_short_ratio > 2.0:
            key_factors.append(f"L/S Ratio {long_short_ratio:.2f} - Extremely crowded long")
            warnings.append("Potential long squeeze setup")
        elif long_short_ratio < 0.5:
            key_factors.append(f"L/S Ratio {long_short_ratio:.2f} - Extremely crowded short")
            warnings.append("Potential short squeeze setup")
        
        # =====================================================================
        # 5. On-chain Score
        # =====================================================================
        # Positive netflow to exchanges = selling pressure
        if exchange_netflow != 0:
            onchain_score = -math.tanh(exchange_netflow / 100000000)  # Normalize
            onchain_confidence = 0.5
        else:
            onchain_score = 0.0
            onchain_confidence = 0.0
        
        # =====================================================================
        # 6. Calculate Weighted Average
        # =====================================================================
        scores_and_weights = [
            (fg_score, 0.20, fg_confidence),         # Fear & Greed
            (news_score, 0.15, news_confidence),      # News
            (social_score, 0.10, social_confidence),  # Social
            (tech_score, 0.25, tech_confidence),      # Technical
            (funding_score, 0.15, funding_confidence), # Funding
            (pos_score, 0.10, pos_confidence),        # Positioning
            (onchain_score, 0.05, onchain_confidence), # On-chain
        ]
        
        weighted_sum = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for score, weight, confidence in scores_and_weights:
            if confidence > 0:
                effective_weight = weight * confidence
                weighted_sum += score * effective_weight
                total_weight += effective_weight
                total_confidence += confidence
        
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
            confidence = total_confidence / 7  # Average confidence
        else:
            overall_score = 0.0
            confidence = 0.0
        
        # =====================================================================
        # 7. Determine Signal (more granular thresholds)
        # =====================================================================
        if overall_score >= 0.4:
            signal = "strong_buy"
        elif overall_score >= 0.15:
            signal = "buy"
        elif overall_score <= -0.4:
            signal = "strong_sell"
        elif overall_score <= -0.15:
            signal = "sell"
        else:
            signal = "neutral"
        
        # =====================================================================
        # 8. Determine Regime
        # =====================================================================
        if fg_score > 0.2 and tech_score > 0:
            regime = "risk_on"
        elif fg_score < -0.2 and tech_score < 0:
            regime = "risk_off"
        else:
            regime = "uncertain"
        
        return EnhancedSentiment(
            timestamp=now,
            symbol=symbol,
            fear_greed_score=fg_score,
            news_score=news_score,
            social_score=social_score,
            technical_score=tech_score if technical else 0.0,
            funding_score=funding_score,
            positioning_score=pos_score,
            onchain_score=onchain_score,
            overall_score=overall_score,
            confidence=confidence,
            signal=signal,
            regime=regime,
            key_factors=key_factors,
            warnings=warnings,
        )
