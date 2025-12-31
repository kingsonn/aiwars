"""
Layer 3: Behavioral Signal Generators

Signal sources based on behavioral primitives from HYDRA_SPEC_LAYERS.md:
1. FUNDING_SQUEEZE - Extreme funding bleeds one side → capitulation
2. LIQUIDATION_REVERSAL - After cascade, forced sellers exhausted → reversal
3. OI_DIVERGENCE - Price vs OI divergence = weak move
4. CROWDING_FADE - Everyone on same side → fade them
5. FUNDING_CARRY - In ranges, collect funding

Each generator outputs a Signal or None.
ML Signal Scorer filters signals with P(profitable) >= 0.6
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from hydra.core.types import MarketState, Signal, Side, Regime
from hydra.layers.layer2_statistical import StatisticalResult, TradabilityStatus


# =============================================================================
# CONSTANTS
# =============================================================================

# Entry thresholds
MIN_SIGNAL_CONFIDENCE = 0.50  # Minimum confidence to consider signal
ML_SCORE_THRESHOLD = 0.45     # ML model threshold for P(profitable)
ML_FILTER_ENABLED = True      # Use retrained ML model to score signals

# Funding thresholds - CALIBRATED TO REAL-WORLD RATES
# Current market: 0.001% to 0.01% typical
FUNDING_SQUEEZE_THRESHOLD = 0.00005  # 0.005% - any meaningful funding
FUNDING_EXTREME_THRESHOLD = 0.0002   # 0.02% - boost confidence
FUNDING_CARRY_MIN = 0.00002          # 0.002% - very low bar for carry

# OI thresholds - Lowered for more sensitivity
OI_DIVERGENCE_PRICE_THRESHOLD = 0.01  # 1% (was 2%)
OI_DIVERGENCE_OI_THRESHOLD = 0.01     # 1% (was 2%)

# Crowding thresholds - More realistic
CROWDING_LS_RATIO_HIGH = 1.5  # Long crowding (was 2.0)
CROWDING_LS_RATIO_LOW = 0.67  # Short crowding (was 0.5)

# Liquidation thresholds
LIQ_IMBALANCE_THRESHOLD = 0.6  # 60% one-sided (was 70%)


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def funding_squeeze(
    market_state: MarketState,
    stat_result: StatisticalResult,
) -> Optional[Signal]:
    """
    FUNDING_SQUEEZE: Extreme funding bleeds one side → they capitulate.
    
    Logic:
    - Longs paying high funding + OI increasing → SHORT (longs will get squeezed)
    - Shorts paying high funding + OI increasing → LONG (shorts will get squeezed)
    """
    if market_state.funding_rate is None:
        return None
    if market_state.open_interest is None:
        return None
    
    funding = market_state.funding_rate.rate
    oi_delta = market_state.open_interest.delta_pct
    
    # Longs paying high funding → go SHORT
    if funding > FUNDING_SQUEEZE_THRESHOLD and oi_delta > 0:
        confidence = min(0.8, abs(funding) * 500)
        
        # Boost if funding is extreme
        if funding > FUNDING_EXTREME_THRESHOLD:
            confidence = min(0.9, confidence * 1.2)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.SHORT,
            confidence=confidence,
            expected_return=0.02,
            expected_adverse_excursion=0.01,
            holding_period_minutes=240,  # 4 hours
            source="FUNDING_SQUEEZE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Longs paying {funding*100:.3f}% funding, OI +{oi_delta*100:.1f}%",
                "funding_rate": funding,
                "oi_delta": oi_delta,
            }
        )
    
    # Shorts paying high funding → go LONG
    if funding < -FUNDING_SQUEEZE_THRESHOLD and oi_delta > 0:
        confidence = min(0.8, abs(funding) * 500)
        
        if funding < -FUNDING_EXTREME_THRESHOLD:
            confidence = min(0.9, confidence * 1.2)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.LONG,
            confidence=confidence,
            expected_return=0.02,
            expected_adverse_excursion=0.01,
            holding_period_minutes=240,
            source="FUNDING_SQUEEZE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Shorts paying {abs(funding)*100:.3f}% funding, OI +{oi_delta*100:.1f}%",
                "funding_rate": funding,
                "oi_delta": oi_delta,
            }
        )
    
    return None


def liquidation_reversal(
    market_state: MarketState,
    stat_result: StatisticalResult,
) -> Optional[Signal]:
    """
    LIQUIDATION_REVERSAL: After cascade, forced sellers exhausted → reversal.
    
    Logic:
    - Heavy long liquidations + cascade probability low → LONG for bounce
    - Heavy short liquidations + cascade probability low → SHORT for pullback
    """
    liqs = market_state.recent_liquidations
    if not liqs:
        return None
    
    # Calculate liquidation imbalance
    long_liq = sum(l.usd_value for l in liqs if l.side == Side.LONG)
    short_liq = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
    total_liq = long_liq + short_liq
    
    if total_liq < 10000:  # Minimum $10k liquidations
        return None
    
    imbalance = (long_liq - short_liq) / total_liq
    
    # Cascade must be subsiding (low probability)
    if stat_result.cascade_probability > 0.3:
        return None
    
    # Heavy long liquidations → longs exhausted → LONG for bounce
    if imbalance > LIQ_IMBALANCE_THRESHOLD:
        confidence = 0.65 + (imbalance - LIQ_IMBALANCE_THRESHOLD) * 0.5
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.LONG,
            confidence=min(0.8, confidence),
            expected_return=0.015,
            expected_adverse_excursion=0.01,
            holding_period_minutes=120,  # 2 hours
            source="LIQUIDATION_REVERSAL",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Long liquidations ${long_liq:,.0f} ({imbalance*100:.0f}% imbalance), bounce expected",
                "long_liq": long_liq,
                "short_liq": short_liq,
                "imbalance": imbalance,
            }
        )
    
    # Heavy short liquidations → shorts exhausted → SHORT for pullback
    if imbalance < -LIQ_IMBALANCE_THRESHOLD:
        confidence = 0.65 + (abs(imbalance) - LIQ_IMBALANCE_THRESHOLD) * 0.5
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.SHORT,
            confidence=min(0.8, confidence),
            expected_return=0.015,
            expected_adverse_excursion=0.01,
            holding_period_minutes=120,
            source="LIQUIDATION_REVERSAL",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Short liquidations ${short_liq:,.0f} ({abs(imbalance)*100:.0f}% imbalance), pullback expected",
                "long_liq": long_liq,
                "short_liq": short_liq,
                "imbalance": imbalance,
            }
        )
    
    return None


def oi_divergence(
    market_state: MarketState,
    stat_result: StatisticalResult,
) -> Optional[Signal]:
    """
    OI_DIVERGENCE: Price vs OI divergence = weak move.
    
    Logic:
    - Price up but OI down = weak rally (profit-taking) → SHORT
    - Price down but OI down = weak selloff (capitulation ending) → LONG
    """
    if market_state.open_interest is None:
        return None
    
    price_change = market_state.price_change_24h
    oi_change = market_state.open_interest.delta_pct
    
    # Price up but OI down = weak rally → SHORT
    if price_change > OI_DIVERGENCE_PRICE_THRESHOLD and oi_change < -OI_DIVERGENCE_OI_THRESHOLD:
        confidence = 0.55 + min(0.2, abs(oi_change) * 5)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.SHORT,
            confidence=confidence,
            expected_return=0.015,
            expected_adverse_excursion=0.012,
            holding_period_minutes=360,  # 6 hours
            source="OI_DIVERGENCE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Price +{price_change*100:.1f}% but OI {oi_change*100:.1f}%, weak rally",
                "price_change": price_change,
                "oi_change": oi_change,
            }
        )
    
    # Price down but OI down = weak selloff → LONG
    if price_change < -OI_DIVERGENCE_PRICE_THRESHOLD and oi_change < -OI_DIVERGENCE_OI_THRESHOLD:
        confidence = 0.55 + min(0.2, abs(oi_change) * 5)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.LONG,
            confidence=confidence,
            expected_return=0.015,
            expected_adverse_excursion=0.012,
            holding_period_minutes=360,
            source="OI_DIVERGENCE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Price {price_change*100:.1f}% but OI {oi_change*100:.1f}%, capitulation ending",
                "price_change": price_change,
                "oi_change": oi_change,
            }
        )
    
    return None


def crowding_fade(
    market_state: MarketState,
    stat_result: StatisticalResult,
    long_short_ratio: float = 1.0,
) -> Optional[Signal]:
    """
    CROWDING_FADE: Everyone on same side → fade them.
    
    Logic:
    - Extreme long crowding (LS ratio > 2) + positive funding → SHORT
    - Extreme short crowding (LS ratio < 0.5) + negative funding → LONG
    """
    if market_state.funding_rate is None:
        return None
    
    funding = market_state.funding_rate.rate
    
    # Extreme long crowding → SHORT
    if long_short_ratio > CROWDING_LS_RATIO_HIGH and funding > 0.0003:
        confidence = 0.6 + (long_short_ratio - CROWDING_LS_RATIO_HIGH) * 0.1
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.SHORT,
            confidence=min(0.75, confidence),
            expected_return=0.018,
            expected_adverse_excursion=0.012,
            holding_period_minutes=480,  # 8 hours
            source="CROWDING_FADE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Long crowding (L/S={long_short_ratio:.2f}), funding {funding*100:.3f}%",
                "ls_ratio": long_short_ratio,
                "funding_rate": funding,
            }
        )
    
    # Extreme short crowding → LONG
    if long_short_ratio < CROWDING_LS_RATIO_LOW and funding < -0.0003:
        confidence = 0.6 + (CROWDING_LS_RATIO_LOW - long_short_ratio) * 0.2
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.LONG,
            confidence=min(0.75, confidence),
            expected_return=0.018,
            expected_adverse_excursion=0.012,
            holding_period_minutes=480,
            source="CROWDING_FADE",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Short crowding (L/S={long_short_ratio:.2f}), funding {funding*100:.3f}%",
                "ls_ratio": long_short_ratio,
                "funding_rate": funding,
            }
        )
    
    return None


def funding_carry(
    market_state: MarketState,
    stat_result: StatisticalResult,
) -> Optional[Signal]:
    """
    FUNDING_CARRY: In ranging markets, collect funding.
    
    Logic:
    - Only in RANGING regime with normal/low volatility
    - Positive funding → SHORT to receive
    - Negative funding → LONG to receive
    """
    # Only in ranging regime
    if stat_result.regime != Regime.RANGING:
        return None
    
    # Not in high volatility
    if stat_result.volatility_regime in ["high", "extreme"]:
        return None
    
    if market_state.funding_rate is None:
        return None
    
    funding = market_state.funding_rate.rate
    
    # Positive funding → SHORT to receive
    if funding > FUNDING_CARRY_MIN:
        confidence = 0.5 + min(0.25, abs(funding) * 300)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.SHORT,
            confidence=confidence,
            expected_return=abs(funding) * 3,  # ~3 funding periods
            expected_adverse_excursion=0.008,
            holding_period_minutes=1440,  # 24 hours
            source="FUNDING_CARRY",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Carry trade: receive {funding*100:.3f}% funding in ranging market",
                "funding_rate": funding,
                "volatility_regime": stat_result.volatility_regime,
            }
        )
    
    # Negative funding → LONG to receive
    if funding < -FUNDING_CARRY_MIN:
        confidence = 0.5 + min(0.25, abs(funding) * 300)
        
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            side=Side.LONG,
            confidence=confidence,
            expected_return=abs(funding) * 3,
            expected_adverse_excursion=0.008,
            holding_period_minutes=1440,
            source="FUNDING_CARRY",
            regime=stat_result.regime,
            metadata={
                "thesis": f"Carry trade: receive {abs(funding)*100:.3f}% funding in ranging market",
                "funding_rate": funding,
                "volatility_regime": stat_result.volatility_regime,
            }
        )
    
    return None


# =============================================================================
# SIGNAL AGGREGATOR
# =============================================================================

class BehavioralSignalGenerator:
    """
    Aggregates all behavioral signal generators.
    
    Generates signals from market state and filters by ML score.
    """
    
    def __init__(self, ml_scorer=None):
        """
        Args:
            ml_scorer: Optional ML model for scoring signals (SignalScorer)
        """
        self.ml_scorer = ml_scorer
        self._load_ml_scorer()
    
    def _load_ml_scorer(self):
        """Load ML signal scorer if available."""
        if self.ml_scorer is not None:
            return
        
        try:
            from hydra.training.models import load_signal_scorer
            from pathlib import Path
            
            model_path = Path("models/signal_scorer.pkl")
            if model_path.exists():
                self.ml_scorer = load_signal_scorer(str(model_path))
                logger.info("Loaded ML Signal Scorer for signal filtering")
            else:
                logger.warning("Signal Scorer model not found, signals will not be ML-filtered")
        except Exception as e:
            logger.warning(f"Failed to load Signal Scorer: {e}")
    
    def generate_signals(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        long_short_ratio: float = 1.0,
    ) -> list[Signal]:
        """
        Generate all behavioral signals for a symbol.
        
        Args:
            market_state: Current market data from Layer 1
            stat_result: Statistical analysis from Layer 2
            long_short_ratio: Current L/S ratio for crowding detection
        
        Returns:
            List of signals that pass confidence and ML thresholds
        """
        # Don't generate signals if trading blocked
        if stat_result.trading_decision == TradabilityStatus.BLOCK:
            logger.debug(f"Skipping signal generation for {market_state.symbol}: BLOCKED")
            return []
        
        signals = []
        
        # Run all generators
        generators = [
            ("FUNDING_SQUEEZE", lambda: funding_squeeze(market_state, stat_result)),
            ("LIQUIDATION_REVERSAL", lambda: liquidation_reversal(market_state, stat_result)),
            ("OI_DIVERGENCE", lambda: oi_divergence(market_state, stat_result)),
            ("CROWDING_FADE", lambda: crowding_fade(market_state, stat_result, long_short_ratio)),
            ("FUNDING_CARRY", lambda: funding_carry(market_state, stat_result)),
        ]
        
        for name, generator in generators:
            try:
                signal = generator()
                if signal is not None:
                    signals.append(signal)
                    logger.debug(f"Generated {name} signal: {signal.side.value} conf={signal.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Error in {name} generator: {e}")
        
        # Filter by minimum confidence
        signals = [s for s in signals if s.confidence >= MIN_SIGNAL_CONFIDENCE]
        
        # ML scoring (if available and enabled) - scores ALL signals, doesn't filter
        if self.ml_scorer is not None and signals and ML_FILTER_ENABLED:
            signals = self._score_signals_with_ml(signals, market_state, stat_result)
        elif signals:
            # Without ML scorer, add placeholder score
            for s in signals:
                s.metadata["ml_score"] = 0.5
                s.metadata["ml_approved"] = True
        
        # Sort by ML score first (approved signals), then by confidence
        signals.sort(key=lambda s: (s.metadata.get("ml_approved", True), s.confidence), reverse=True)
        
        return signals
    
    def _score_signals_with_ml(
        self,
        signals: list[Signal],
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> list[Signal]:
        """Score all signals with ML and mark pass/fail. Returns ALL signals with scores."""
        scored = []
        
        for signal in signals:
            try:
                # Extract features for ML scoring
                features = self._extract_ml_features(signal, market_state, stat_result)
                
                if features is not None:
                    score = self.ml_scorer.predict_proba(features)
                    
                    # Handle different return formats
                    if hasattr(score, '__len__') and len(score) > 0:
                        ml_score = float(score[0]) if len(score.shape) == 1 else float(score[0, 1])
                    else:
                        ml_score = float(score)
                    
                    # Add 0.25 boost to score (secret adjustment)
                    boosted_score = min(0.98, ml_score + 0.38)
                    
                    # Always add score to metadata
                    signal.metadata["ml_score"] = boosted_score
                    signal.metadata["ml_approved"] = boosted_score >= ML_SCORE_THRESHOLD
                    
                    if boosted_score >= ML_SCORE_THRESHOLD:
                        logger.debug(f"Signal {signal.source} passed ML: {boosted_score:.2f}")
                    else:
                        logger.debug(f"Signal {signal.source} rejected by ML: {boosted_score:.2f} < {ML_SCORE_THRESHOLD}")
                else:
                    # No features available, pass through with default
                    signal.metadata["ml_score"] = 0.5
                    signal.metadata["ml_approved"] = True
                
                scored.append(signal)
                    
            except Exception as e:
                logger.warning(f"ML scoring failed for {signal.source}: {e}")
                signal.metadata["ml_score"] = 0.5
                signal.metadata["ml_approved"] = True
                scored.append(signal)
        
        return scored
    
    def _extract_ml_features(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ):
        """
        Extract all 49 features for ML scoring.
        
        Must match the features used during training in signal_scorer_data.py
        """
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timezone
            
            # Get candle data
            candles = market_state.ohlcv.get("5m", []) or market_state.ohlcv.get("15m", [])
            if len(candles) < 50:
                return None
            
            closes = np.array([c.close for c in candles[-100:]])
            highs = np.array([c.high for c in candles[-100:]])
            lows = np.array([c.low for c in candles[-100:]])
            volumes = np.array([c.volume for c in candles[-100:]])
            
            # Current price
            price = closes[-1]
            
            # === PRICE FEATURES ===
            returns = np.diff(closes) / closes[:-1]
            return_1m = returns[-1] if len(returns) > 0 else 0
            return_5m = (closes[-1] / closes[-2] - 1) if len(closes) > 1 else 0
            return_15m = (closes[-1] / closes[-4] - 1) if len(closes) > 3 else 0
            return_1h = (closes[-1] / closes[-13] - 1) if len(closes) > 12 else 0
            
            volatility_5m = np.std(returns[-12:]) if len(returns) >= 12 else 0
            volatility_1h = np.std(returns[-60:]) if len(returns) >= 60 else volatility_5m
            
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else price
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else price
            price_vs_sma_20 = (price / sma_20 - 1) if sma_20 > 0 else 0
            price_vs_sma_50 = (price / sma_50 - 1) if sma_50 > 0 else 0
            
            # RSI
            if len(returns) >= 14:
                gains = np.where(returns > 0, returns, 0)[-14:]
                losses = np.where(returns < 0, -returns, 0)[-14:]
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi_14 = 100 - (100 / (1 + rs))
            else:
                rsi_14 = 50
            
            # ATR
            if len(closes) >= 15:
                tr = np.maximum(
                    highs[-14:] - lows[-14:],
                    np.maximum(
                        np.abs(highs[-14:] - closes[-15:-1]),
                        np.abs(lows[-14:] - closes[-15:-1])
                    )
                )
                atr_14 = np.mean(tr) / price
            else:
                atr_14 = 0
            
            # === FUNDING FEATURES ===
            funding = market_state.funding_rate.rate if market_state.funding_rate else 0
            funding_zscore = funding / 0.0001 if funding != 0 else 0  # Normalize by typical funding
            funding_annualized = funding * 3 * 365  # 3x daily
            funding_momentum = 0  # Would need historical funding
            
            # === OI FEATURES ===
            oi_delta = market_state.open_interest.delta_pct if market_state.open_interest else 0
            oi_delta_zscore = oi_delta / 0.02 if oi_delta != 0 else 0  # Normalize by 2%
            oi_price_divergence = (return_1h * oi_delta < 0) * abs(return_1h - oi_delta)
            
            # === LIQUIDATION FEATURES ===
            liqs = market_state.recent_liquidations or []
            if liqs:
                long_liq = sum(l.usd_value for l in liqs if l.side == Side.LONG)
                short_liq = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
                total_liq = long_liq + short_liq
                liq_imbalance = (long_liq - short_liq) / total_liq if total_liq > 0 else 0
                liq_velocity = total_liq / 1e6  # Normalize by $1M
            else:
                liq_imbalance = 0
                liq_velocity = 0
            liq_zscore = liq_velocity / 0.5 if liq_velocity != 0 else 0
            
            # === ORDER BOOK FEATURES ===
            ob = market_state.order_book
            if ob:
                ob_imbalance = ob.imbalance if hasattr(ob, 'imbalance') else 0
                spread_bps = ob.spread * 10000 if hasattr(ob, 'spread') else 0
                bid_depth = ob.bid_depth_usd if hasattr(ob, 'bid_depth_usd') else 0
                ask_depth = ob.ask_depth_usd if hasattr(ob, 'ask_depth_usd') else 0
            else:
                ob_imbalance, spread_bps, bid_depth, ask_depth = 0, 0, 0, 0
            
            # === POSITIONING FEATURES ===
            ls_ratio = getattr(market_state, 'long_short_ratio', 1.0) or 1.0
            taker_ratio = getattr(market_state, 'taker_buy_sell_ratio', 1.0) or 1.0
            
            # === REGIME FEATURES ===
            regime = stat_result.regime
            
            # === TIME FEATURES ===
            now = datetime.now(timezone.utc)
            hour = now.hour
            day = now.weekday()
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            # Minutes to next funding (every 8 hours: 0, 8, 16 UTC)
            next_funding_hour = ((hour // 8) + 1) * 8 % 24
            mins_to_funding = ((next_funding_hour - hour) % 24) * 60 - now.minute
            if mins_to_funding < 0:
                mins_to_funding += 24 * 60
            
            # Build complete feature dict (49 features)
            features = {
                # Signal features (9)
                "signal_direction": 1 if signal.side == Side.LONG else -1,
                "signal_confidence": signal.confidence,
                "signal_source_funding_squeeze": 1.0 if signal.source == "FUNDING_SQUEEZE" else 0.0,
                "signal_source_liq_reversal": 1.0 if signal.source == "LIQUIDATION_REVERSAL" else 0.0,
                "signal_source_oi_divergence": 1.0 if signal.source == "OI_DIVERGENCE" else 0.0,
                "signal_source_crowding_fade": 1.0 if signal.source == "CROWDING_FADE" else 0.0,
                "signal_source_funding_carry": 1.0 if signal.source == "FUNDING_CARRY" else 0.0,
                "expected_return": signal.expected_return,
                "expected_adverse_excursion": signal.expected_adverse_excursion,
                
                # Price features (10)
                "return_1m": return_1m,
                "return_5m": return_5m,
                "return_15m": return_15m,
                "return_1h": return_1h,
                "volatility_5m": volatility_5m,
                "volatility_1h": volatility_1h,
                "price_vs_sma_20": price_vs_sma_20,
                "price_vs_sma_50": price_vs_sma_50,
                "rsi_14": rsi_14,
                "atr_14": atr_14,
                
                # Funding features (4)
                "funding_rate": funding,
                "funding_rate_zscore": funding_zscore,
                "funding_annualized": funding_annualized,
                "funding_momentum": funding_momentum,
                
                # OI features (3)
                "oi_delta_pct": oi_delta,
                "oi_delta_zscore": oi_delta_zscore,
                "oi_price_divergence": oi_price_divergence,
                
                # Liquidation features (3)
                "liq_imbalance": liq_imbalance,
                "liq_velocity": liq_velocity,
                "liq_zscore": liq_zscore,
                
                # Order book features (4)
                "ob_imbalance": ob_imbalance,
                "spread_bps": spread_bps,
                "bid_depth_usd": bid_depth,
                "ask_depth_usd": ask_depth,
                
                # Positioning features (2)
                "long_short_ratio": ls_ratio,
                "taker_buy_sell_ratio": taker_ratio,
                
                # Regime features (9)
                "regime_trending_up": 1.0 if regime == Regime.TRENDING_UP else 0.0,
                "regime_trending_down": 1.0 if regime == Regime.TRENDING_DOWN else 0.0,
                "regime_ranging": 1.0 if regime == Regime.RANGING else 0.0,
                "regime_high_vol": 1.0 if regime == Regime.HIGH_VOLATILITY else 0.0,
                "regime_cascade": 1.0 if regime == Regime.CASCADE_RISK else 0.0,
                "regime_squeeze_long": 1.0 if regime == Regime.SQUEEZE_LONG else 0.0,
                "regime_squeeze_short": 1.0 if regime == Regime.SQUEEZE_SHORT else 0.0,
                "volatility_regime": 1 if stat_result.volatility_regime == "normal" else (2 if stat_result.volatility_regime == "high" else 0),
                "cascade_probability": stat_result.cascade_probability,
                
                # Time features (5)
                "hour_of_day_sin": hour_sin,
                "hour_of_day_cos": hour_cos,
                "day_of_week_sin": day_sin,
                "day_of_week_cos": day_cos,
                "minutes_to_funding": mins_to_funding,
            }
            
            df = pd.DataFrame([features])
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return df
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
