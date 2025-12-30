"""
Opponent & Crowd Modeling

HYDRA models other traders, not just price.

Techniques:
- Behavioral clustering from OI + funding
- Inverse RL on liquidation patterns
- Identification of trader archetypes

Trader Types:
- Momentum CTAs
- Funding farmers
- Retail leverage chasers
- Passive liquidity providers

Enables:
- Pre-squeeze positioning
- Fade-the-crowd setups
- Early exit before cascades reverse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import MarketState, Side, OHLCV


class TraderArchetype(Enum):
    """Identified trader archetypes in the market."""
    MOMENTUM_CTA = "momentum_cta"  # Trend followers
    FUNDING_FARMER = "funding_farmer"  # Collects funding
    RETAIL_LEVERAGE = "retail_leverage"  # High leverage retail
    PASSIVE_LP = "passive_lp"  # Market makers
    SMART_MONEY = "smart_money"  # Institutional
    UNKNOWN = "unknown"


@dataclass
class CrowdState:
    """Current state of the crowd."""
    timestamp: datetime
    symbol: str
    
    # Dominant archetypes
    dominant_archetype: TraderArchetype
    archetype_confidence: float
    archetype_distribution: dict[TraderArchetype, float]
    
    # Crowd behavior
    crowd_direction: Side  # Net positioning
    crowd_conviction: float  # 0 to 1
    crowd_leverage: float  # Estimated average leverage
    
    # Crowding metrics
    crowding_percentile: float  # 0-100, how crowded vs history
    position_concentration: float  # Herfindahl-like index
    
    # Behavioral signals
    fomo_score: float  # 0 to 1
    capitulation_score: float  # 0 to 1
    greed_fear_ratio: float  # >1 = greed, <1 = fear
    
    # Predicted behavior
    squeeze_vulnerability: float  # 0 to 1
    likely_squeeze_direction: Side
    estimated_liquidation_volume: float  # USD if squeezed
    
    # Counter-trading opportunity
    fade_opportunity: bool
    fade_direction: Side
    fade_confidence: float


@dataclass
class TraderCluster:
    """A cluster of similar trader behaviors."""
    archetype: TraderArchetype
    size_estimate: float  # Fraction of OI
    avg_entry_price: float
    avg_leverage: float
    avg_holding_period: float  # hours
    pnl_estimate: float  # Estimated unrealized PnL
    stress_level: float  # 0 to 1, higher = more likely to exit
    
    # Behavior patterns
    entry_pattern: str  # "momentum", "mean_reversion", "random"
    exit_pattern: str  # "stop_loss", "take_profit", "forced"
    funding_sensitivity: float  # How much funding affects them


class BehaviorFeatureExtractor:
    """Extract behavioral features from market data."""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
    
    def extract(self, market_state: MarketState) -> np.ndarray:
        """Extract behavioral features for clustering."""
        features = []
        
        candles = market_state.ohlcv.get("5m", [])[-self.lookback:]
        if len(candles) < 20:
            return np.zeros(12)
        
        prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        returns = np.diff(prices) / prices[:-1]
        
        # 1. Price momentum (normalized)
        mom_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        mom_20 = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        features.extend([mom_5 * 100, mom_20 * 100])
        
        # 2. Volume profile
        vol_ratio = volumes[-5:].mean() / volumes.mean() if volumes.mean() > 0 else 1
        vol_trend = np.polyfit(range(min(20, len(volumes))), volumes[-20:], 1)[0] if len(volumes) >= 20 else 0
        features.extend([vol_ratio, vol_trend / (volumes.mean() + 1e-10)])
        
        # 3. Funding rate behavior
        if market_state.funding_rate:
            funding = market_state.funding_rate.rate * 10000  # bps
        else:
            funding = 0
        features.append(funding)
        
        # 4. OI behavior
        if market_state.open_interest:
            oi_delta = market_state.open_interest.delta_pct
        else:
            oi_delta = 0
        features.append(oi_delta)
        
        # 5. Order book imbalance
        if market_state.order_book:
            imbalance = market_state.order_book.imbalance
        else:
            imbalance = 0
        features.append(imbalance)
        
        # 6. Volatility regime
        volatility = np.std(returns) if len(returns) > 0 else 0
        features.append(volatility * 100)
        
        # 7. Return distribution
        skew = stats.skew(returns) if len(returns) > 2 else 0
        kurt = stats.kurtosis(returns) if len(returns) > 3 else 0
        features.extend([skew, kurt])
        
        # 8. Trend strength (ADX-like)
        if len(candles) >= 14:
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            
            up_moves = np.maximum(highs[1:] - highs[:-1], 0)
            down_moves = np.maximum(lows[:-1] - lows[1:], 0)
            
            avg_up = np.mean(up_moves[-14:])
            avg_down = np.mean(down_moves[-14:])
            
            if avg_up + avg_down > 0:
                trend_strength = abs(avg_up - avg_down) / (avg_up + avg_down)
            else:
                trend_strength = 0
        else:
            trend_strength = 0
        features.append(trend_strength)
        
        return np.array(features)


class OpponentCrowdModel:
    """
    Model opponent traders and crowd behavior.
    
    This component answers:
    - Who is in the market right now?
    - What are they likely to do?
    - When will they be forced to act?
    - How can we position before them?
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        self.feature_extractor = BehaviorFeatureExtractor()
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
        
        # Historical data for clustering
        self._feature_history: list[np.ndarray] = []
        self._fitted = False
        
        # Archetype patterns (learned from data)
        self._archetype_centroids: dict[TraderArchetype, np.ndarray] = {}
        
        logger.info("Opponent Crowd Model initialized")
    
    async def setup(self) -> None:
        """Initialize the model."""
        # Pre-define archetype characteristics
        # These would be learned from historical data in practice
        self._archetype_centroids = {
            TraderArchetype.MOMENTUM_CTA: np.array([
                2.0, 5.0,  # Positive momentum
                1.5, 0.1,  # Increasing volume
                0.0, 1.0,  # OI increasing
                0.0, 0.5,  # Low volatility
                0.0, 0.0,  # Normal distribution
                0.7,  # Strong trend
            ]),
            TraderArchetype.FUNDING_FARMER: np.array([
                0.0, 0.0,  # No momentum
                0.8, 0.0,  # Stable volume
                5.0, -0.5,  # High funding, decreasing OI
                -0.2, 0.3,  # Slight book imbalance against
                0.0, 1.0,  # Fat tails
                0.2,  # Weak trend
            ]),
            TraderArchetype.RETAIL_LEVERAGE: np.array([
                3.0, 2.0,  # Chasing momentum
                2.0, 0.5,  # Spiking volume
                2.0, 3.0,  # Funding and OI spiking
                0.5, 1.0,  # High volatility
                -0.5, 2.0,  # Left skew, fat tails
                0.5,  # Moderate trend
            ]),
            TraderArchetype.PASSIVE_LP: np.array([
                0.0, 0.0,  # No directional bias
                1.0, 0.0,  # Stable volume
                0.0, 0.0,  # Neutral
                0.0, 0.2,  # Low volatility
                0.0, 0.0,  # Normal distribution
                0.1,  # No trend
            ]),
            TraderArchetype.SMART_MONEY: np.array([
                -1.0, 1.0,  # Counter-trend short-term
                0.7, -0.1,  # Decreasing volume (quiet)
                -2.0, 0.5,  # Opposite funding, OI increasing
                -0.3, 0.4,  # Moderate volatility
                0.2, 0.5,  # Slight right skew
                0.4,  # Some trend
            ]),
        }
    
    async def analyze(self, market_state: MarketState) -> CrowdState:
        """
        Analyze current crowd state and opponent positioning.
        """
        features = self.feature_extractor.extract(market_state)
        
        # Update history
        self._feature_history.append(features)
        if len(self._feature_history) > 1000:
            self._feature_history = self._feature_history[-1000:]
        
        # Identify dominant archetype
        archetype, confidence, distribution = self._identify_archetype(features)
        
        # Determine crowd direction and conviction
        crowd_direction, crowd_conviction = self._determine_crowd_positioning(
            market_state, features
        )
        
        # Estimate crowd leverage
        crowd_leverage = self._estimate_leverage(market_state)
        
        # Calculate crowding percentile
        crowding_percentile = self._calculate_crowding_percentile(market_state)
        
        # Behavioral scores
        fomo_score = self._calculate_fomo_score(market_state, features)
        capitulation_score = self._calculate_capitulation_score(market_state, features)
        
        # Squeeze analysis
        squeeze_vuln, squeeze_dir, liq_volume = self._analyze_squeeze_vulnerability(
            market_state, crowd_direction, crowd_leverage
        )
        
        # Fade opportunity
        fade_opp, fade_dir, fade_conf = self._identify_fade_opportunity(
            crowding_percentile, crowd_direction, squeeze_vuln
        )
        
        greed_fear = fomo_score / max(capitulation_score, 0.1)
        
        return CrowdState(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            dominant_archetype=archetype,
            archetype_confidence=confidence,
            archetype_distribution=distribution,
            crowd_direction=crowd_direction,
            crowd_conviction=crowd_conviction,
            crowd_leverage=crowd_leverage,
            crowding_percentile=crowding_percentile,
            position_concentration=0.5,  # Would need more data
            fomo_score=fomo_score,
            capitulation_score=capitulation_score,
            greed_fear_ratio=greed_fear,
            squeeze_vulnerability=squeeze_vuln,
            likely_squeeze_direction=squeeze_dir,
            estimated_liquidation_volume=liq_volume,
            fade_opportunity=fade_opp,
            fade_direction=fade_dir,
            fade_confidence=fade_conf,
        )
    
    def _identify_archetype(
        self, features: np.ndarray
    ) -> tuple[TraderArchetype, float, dict[TraderArchetype, float]]:
        """Identify dominant trader archetype from features."""
        distances = {}
        
        for archetype, centroid in self._archetype_centroids.items():
            # Euclidean distance in feature space
            if len(features) >= len(centroid):
                dist = np.linalg.norm(features[:len(centroid)] - centroid)
            else:
                dist = float('inf')
            distances[archetype] = dist
        
        # Convert distances to similarities (inverse)
        max_dist = max(distances.values()) + 1e-10
        similarities = {a: 1 - d / max_dist for a, d in distances.items()}
        
        # Normalize to probabilities
        total = sum(similarities.values())
        distribution = {a: s / total for a, s in similarities.items()}
        
        # Find dominant
        dominant = max(distribution, key=distribution.get)
        confidence = distribution[dominant]
        
        return dominant, confidence, distribution
    
    def _determine_crowd_positioning(
        self, market_state: MarketState, features: np.ndarray
    ) -> tuple[Side, float]:
        """Determine net crowd direction and conviction."""
        signals = []
        
        # From funding rate
        if market_state.funding_rate:
            rate = market_state.funding_rate.rate
            if rate > 0.0001:
                signals.append((Side.LONG, min(1.0, rate * 5000)))
            elif rate < -0.0001:
                signals.append((Side.SHORT, min(1.0, abs(rate) * 5000)))
        
        # From order book imbalance
        if market_state.order_book:
            imb = market_state.order_book.imbalance
            if imb > 0.2:
                signals.append((Side.LONG, imb))
            elif imb < -0.2:
                signals.append((Side.SHORT, abs(imb)))
        
        # From momentum
        if len(features) > 1:
            mom = features[1]  # 20-period momentum
            if mom > 1:
                signals.append((Side.LONG, min(1.0, mom / 5)))
            elif mom < -1:
                signals.append((Side.SHORT, min(1.0, abs(mom) / 5)))
        
        if not signals:
            return Side.FLAT, 0.0
        
        # Aggregate
        long_score = sum(w for s, w in signals if s == Side.LONG)
        short_score = sum(w for s, w in signals if s == Side.SHORT)
        
        if long_score > short_score:
            return Side.LONG, min(1.0, long_score / (len(signals) + 1))
        elif short_score > long_score:
            return Side.SHORT, min(1.0, short_score / (len(signals) + 1))
        
        return Side.FLAT, 0.0
    
    def _estimate_leverage(self, market_state: MarketState) -> float:
        """Estimate average crowd leverage."""
        base_leverage = 5.0  # Assumed baseline
        
        if market_state.funding_rate:
            # High funding = high leverage
            rate_abs = abs(market_state.funding_rate.rate)
            leverage_mult = 1 + rate_abs * 1000
            base_leverage *= leverage_mult
        
        if market_state.open_interest:
            # High OI delta = leverage building
            if market_state.open_interest.delta_pct > 5:
                base_leverage *= 1.2
        
        return min(base_leverage, 25.0)  # Cap at 25x
    
    def _calculate_crowding_percentile(self, market_state: MarketState) -> float:
        """Calculate how crowded vs historical."""
        if not market_state.funding_rate:
            return 50.0
        
        rate_abs = abs(market_state.funding_rate.rate)
        
        # Simple mapping based on typical funding ranges
        if rate_abs > 0.001:
            return 95.0
        elif rate_abs > 0.0005:
            return 80.0
        elif rate_abs > 0.0002:
            return 60.0
        elif rate_abs > 0.0001:
            return 40.0
        
        return 20.0
    
    def _calculate_fomo_score(
        self, market_state: MarketState, features: np.ndarray
    ) -> float:
        """Calculate FOMO (fear of missing out) score."""
        score = 0.0
        
        # Momentum chasing
        if len(features) > 1:
            mom = features[1]
            if mom > 2:
                score += 0.3
        
        # Volume spike
        if len(features) > 2:
            vol_ratio = features[2]
            if vol_ratio > 1.5:
                score += 0.3
        
        # OI building with price
        if market_state.open_interest and market_state.price_change_24h > 0.02:
            if market_state.open_interest.delta_pct > 3:
                score += 0.4
        
        return min(1.0, score)
    
    def _calculate_capitulation_score(
        self, market_state: MarketState, features: np.ndarray
    ) -> float:
        """Calculate capitulation score."""
        score = 0.0
        
        # Sharp negative momentum
        if len(features) > 0:
            mom = features[0]
            if mom < -3:
                score += 0.4
        
        # Volume spike on down move
        if len(features) > 2 and market_state.price_change_24h < -0.03:
            vol_ratio = features[2]
            if vol_ratio > 2:
                score += 0.3
        
        # Liquidation evidence
        if market_state.recent_liquidations:
            liq_count = len(market_state.recent_liquidations)
            if liq_count > 10:
                score += 0.3
        
        return min(1.0, score)
    
    def _analyze_squeeze_vulnerability(
        self,
        market_state: MarketState,
        crowd_direction: Side,
        crowd_leverage: float,
    ) -> tuple[float, Side, float]:
        """Analyze vulnerability to squeeze."""
        vulnerability = 0.0
        squeeze_dir = Side.FLAT
        liq_volume = 0.0
        
        if crowd_direction == Side.FLAT:
            return 0.0, Side.FLAT, 0.0
        
        # High leverage = high vulnerability
        leverage_score = min(1.0, crowd_leverage / 20)
        vulnerability += leverage_score * 0.4
        
        # High funding = stressed positions
        if market_state.funding_rate:
            rate_abs = abs(market_state.funding_rate.rate)
            funding_score = min(1.0, rate_abs * 2000)
            vulnerability += funding_score * 0.3
        
        # Orderbook thinness
        if market_state.order_book:
            spread = market_state.order_book.spread
            if spread > 0.001:
                vulnerability += 0.3
        
        # Squeeze direction is opposite of crowd
        if crowd_direction == Side.LONG:
            squeeze_dir = Side.SHORT  # Squeeze will push shorts out first, then longs
        else:
            squeeze_dir = Side.LONG
        
        # Estimate liquidation volume
        if market_state.open_interest and market_state.open_interest.open_interest_usd:
            # Assume 20% of OI is vulnerable at high leverage
            liq_volume = market_state.open_interest.open_interest_usd * 0.2 * vulnerability
        
        return vulnerability, squeeze_dir, liq_volume
    
    def _identify_fade_opportunity(
        self,
        crowding_percentile: float,
        crowd_direction: Side,
        squeeze_vuln: float,
    ) -> tuple[bool, Side, float]:
        """Identify opportunity to fade the crowd."""
        if crowding_percentile < 70:
            return False, Side.FLAT, 0.0
        
        if squeeze_vuln < 0.4:
            return False, Side.FLAT, 0.0
        
        # Fade direction is opposite of crowd
        if crowd_direction == Side.LONG:
            fade_dir = Side.SHORT
        elif crowd_direction == Side.SHORT:
            fade_dir = Side.LONG
        else:
            return False, Side.FLAT, 0.0
        
        # Confidence based on crowding and vulnerability
        confidence = min(1.0, (crowding_percentile / 100) * squeeze_vuln)
        
        return True, fade_dir, confidence
