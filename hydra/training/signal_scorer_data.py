"""
Signal Scorer Data Preparation

Creates comprehensive training data for Signal Scorer model per HYDRA_SPEC_ML.md.

Key improvements:
1. Uses actual behavioral signal generators (not random signals)
2. Includes ALL required features:
   - Signal features (direction, confidence, source, expected values)
   - Price features (returns, volatility, RSI, ATR, SMAs)
   - Funding features (rate, zscore, annualized, momentum)
   - OI features (delta, zscore, divergence)
   - Liquidation features (imbalance, velocity, zscore)
   - Order book features (imbalance, spread, depths)
   - Positioning features (L/S ratio, taker ratio)
   - Regime features (encoded)
   - Time features (cyclical encoding)
3. Proper label creation based on actual profitability
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from enum import IntEnum
import numpy as np
import pandas as pd
from loguru import logger

from hydra.training.features import Regime


# =============================================================================
# SIGNAL SOURCES (for encoding)
# =============================================================================

class SignalSource(IntEnum):
    FUNDING_SQUEEZE = 0
    LIQUIDATION_REVERSAL = 1
    OI_DIVERGENCE = 2
    CROWDING_FADE = 3
    FUNDING_CARRY = 4
    UNKNOWN = 5


# Signal source thresholds (matching signals.py)
FUNDING_SQUEEZE_THRESHOLD = 0.0005
FUNDING_EXTREME_THRESHOLD = 0.001
FUNDING_CARRY_MIN = 0.0003
OI_DIVERGENCE_PRICE_THRESHOLD = 0.02
OI_DIVERGENCE_OI_THRESHOLD = 0.02
CROWDING_LS_RATIO_HIGH = 2.0
CROWDING_LS_RATIO_LOW = 0.5
LIQ_IMBALANCE_THRESHOLD = 0.7


# =============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================

class SignalScorerFeatureEngineer:
    """
    Feature engineering specifically for Signal Scorer model.
    
    Creates all features per HYDRA_SPEC_ML.md:
    - 5 signal features
    - 10 price features
    - 4 funding features
    - 3 OI features
    - 3 liquidation features
    - 4 order book features
    - 2 positioning features
    - 3 regime features
    - 5 time features
    """
    
    def __init__(self, lookahead_minutes: int = 30, transaction_cost: float = 0.001):
        self.lookahead_minutes = lookahead_minutes
        self.transaction_cost = transaction_cost
    
    def engineer_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all market features from raw data.
        
        Expected columns:
        - open, high, low, close, volume (required)
        - funding_rate (optional)
        - open_interest (optional)
        - long_liquidations, short_liquidations (optional)
        - ob_bid_depth, ob_ask_depth, ob_spread (optional)
        - long_short_ratio, taker_buy_sell_ratio (optional)
        """
        features = pd.DataFrame(index=df.index)
        
        # =====================================================================
        # PRICE FEATURES
        # =====================================================================
        features["return_1m"] = df["close"].pct_change(1)
        features["return_5m"] = df["close"].pct_change(5)
        features["return_15m"] = df["close"].pct_change(15)
        features["return_1h"] = df["close"].pct_change(12)  # 12 x 5min
        
        # Volatility (annualized)
        features["volatility_5m"] = features["return_1m"].rolling(5).std() * np.sqrt(365 * 24 * 12)
        features["volatility_1h"] = features["return_1m"].rolling(12).std() * np.sqrt(365 * 24 * 12)
        
        # SMAs and price position
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        features["price_vs_sma_20"] = df["close"] / sma_20 - 1
        features["price_vs_sma_50"] = df["close"] / sma_50 - 1
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        features["rsi_14"] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean()
        
        # =====================================================================
        # FUNDING FEATURES
        # =====================================================================
        if "funding_rate" in df.columns:
            features["funding_rate"] = df["funding_rate"]
            funding_mean = features["funding_rate"].rolling(100).mean()
            funding_std = features["funding_rate"].rolling(100).std()
            features["funding_rate_zscore"] = (features["funding_rate"] - funding_mean) / funding_std.replace(0, 1e-8)
            features["funding_annualized"] = features["funding_rate"] * 3 * 365 * 100
            features["funding_momentum"] = features["funding_rate"].diff(3)
        else:
            features["funding_rate"] = 0.0
            features["funding_rate_zscore"] = 0.0
            features["funding_annualized"] = 0.0
            features["funding_momentum"] = 0.0
        
        # =====================================================================
        # OI FEATURES
        # =====================================================================
        if "open_interest" in df.columns:
            features["oi_delta_pct"] = df["open_interest"].pct_change()
            oi_mean = features["oi_delta_pct"].rolling(100).mean()
            oi_std = features["oi_delta_pct"].rolling(100).std()
            features["oi_delta_zscore"] = (features["oi_delta_pct"] - oi_mean) / oi_std.replace(0, 1e-8)
            
            # OI-Price divergence
            price_dir = np.sign(features["return_1h"])
            oi_dir = np.sign(features["oi_delta_pct"].rolling(12).sum())
            features["oi_price_divergence"] = (price_dir != oi_dir).astype(float)
        else:
            features["oi_delta_pct"] = 0.0
            features["oi_delta_zscore"] = 0.0
            features["oi_price_divergence"] = 0.0
        
        # =====================================================================
        # LIQUIDATION FEATURES
        # =====================================================================
        if "long_liquidations" in df.columns and "short_liquidations" in df.columns:
            long_liq = df["long_liquidations"]
            short_liq = df["short_liquidations"]
            total_liq = long_liq + short_liq
            
            features["liq_imbalance"] = (long_liq - short_liq) / total_liq.replace(0, 1)
            
            if "open_interest" in df.columns:
                features["liq_velocity"] = total_liq / df["open_interest"].replace(0, 1)
            else:
                features["liq_velocity"] = 0.0
            
            liq_mean = total_liq.rolling(100).mean()
            liq_std = total_liq.rolling(100).std()
            features["liq_zscore"] = (total_liq - liq_mean) / liq_std.replace(0, 1e-8)
        else:
            features["liq_imbalance"] = 0.0
            features["liq_velocity"] = 0.0
            features["liq_zscore"] = 0.0
        
        # =====================================================================
        # ORDER BOOK FEATURES
        # =====================================================================
        if "ob_imbalance" in df.columns:
            features["ob_imbalance"] = df["ob_imbalance"]
        else:
            # Estimate from volume direction
            features["ob_imbalance"] = np.sign(df["close"] - df["open"]) * 0.1
        
        if "ob_spread" in df.columns:
            features["spread_bps"] = df["ob_spread"] * 10000
        else:
            # Estimate spread from high-low
            features["spread_bps"] = ((df["high"] - df["low"]) / df["close"]) * 10000 * 0.1
        
        if "ob_bid_depth" in df.columns:
            features["bid_depth_usd"] = df["ob_bid_depth"]
            features["ask_depth_usd"] = df["ob_ask_depth"]
        else:
            # Estimate from volume
            features["bid_depth_usd"] = df["volume"] * df["close"] * 0.5
            features["ask_depth_usd"] = df["volume"] * df["close"] * 0.5
        
        # =====================================================================
        # POSITIONING FEATURES
        # =====================================================================
        if "long_short_ratio" in df.columns:
            features["long_short_ratio"] = df["long_short_ratio"]
        else:
            # Estimate from funding direction
            features["long_short_ratio"] = 1.0 + features["funding_rate"] * 100
        
        if "taker_buy_sell_ratio" in df.columns:
            features["taker_buy_sell_ratio"] = df["taker_buy_sell_ratio"]
        else:
            # Estimate from price direction
            features["taker_buy_sell_ratio"] = 1.0 + features["return_5m"] * 10
        
        # =====================================================================
        # CASCADE PROBABILITY (heuristic)
        # =====================================================================
        liq_factor = features["liq_velocity"].clip(0, 0.1) * 10
        vol_zscore = (features["volatility_1h"] - features["volatility_1h"].rolling(100).mean()) / features["volatility_1h"].rolling(100).std().replace(0, 1)
        vol_factor = vol_zscore.clip(-3, 3) / 3
        funding_factor = features["funding_rate_zscore"].abs().clip(0, 3) / 3
        features["cascade_probability"] = (liq_factor * 0.4 + vol_factor * 0.3 + funding_factor * 0.3).clip(0, 1)
        
        # =====================================================================
        # TIME FEATURES
        # =====================================================================
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
            days = df.index.dayofweek
            
            features["hour_of_day_sin"] = np.sin(2 * np.pi * hours / 24)
            features["hour_of_day_cos"] = np.cos(2 * np.pi * hours / 24)
            features["day_of_week_sin"] = np.sin(2 * np.pi * days / 7)
            features["day_of_week_cos"] = np.cos(2 * np.pi * days / 7)
            
            # Minutes to next funding
            minutes_in_day = hours * 60 + df.index.minute
            funding_times = [0, 480, 960]
            features["minutes_to_funding"] = minutes_in_day.map(
                lambda m: min((f - m) % 1440 for f in funding_times)
            )
        else:
            features["hour_of_day_sin"] = 0.0
            features["hour_of_day_cos"] = 0.0
            features["day_of_week_sin"] = 0.0
            features["day_of_week_cos"] = 0.0
            features["minutes_to_funding"] = 0.0
        
        # Keep close for label creation
        features["close"] = df["close"]
        
        return features
    
    def detect_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Detect regime for each row."""
        regimes = np.full(len(features), Regime.RANGING.value)
        
        liq_velocity = features.get("liq_velocity", pd.Series(0, index=features.index)).values
        funding = features.get("funding_rate", pd.Series(0, index=features.index)).values
        oi_delta = features.get("oi_delta_pct", pd.Series(0, index=features.index)).values
        returns_1h = features.get("return_1h", pd.Series(0, index=features.index)).values
        volatility = features.get("volatility_1h", pd.Series(0, index=features.index)).values
        
        cumulative_returns = pd.Series(returns_1h).rolling(12).sum().fillna(0).values
        vol_series = pd.Series(volatility)
        vol_95th = vol_series.rolling(500, min_periods=100).quantile(0.95).fillna(vol_series.median()).values
        
        for i in range(len(features)):
            if liq_velocity[i] > 0.05:
                regimes[i] = Regime.CASCADE_RISK.value
            elif funding[i] > 0.001 and oi_delta[i] > 0.02 and returns_1h[i] < 0:
                regimes[i] = Regime.SQUEEZE_SHORT.value
            elif funding[i] < -0.001 and oi_delta[i] > 0.02 and returns_1h[i] > 0:
                regimes[i] = Regime.SQUEEZE_LONG.value
            elif volatility[i] > vol_95th[i]:
                regimes[i] = Regime.HIGH_VOLATILITY.value
            elif cumulative_returns[i] > 0.02:
                regimes[i] = Regime.TRENDING_UP.value
            elif cumulative_returns[i] < -0.02:
                regimes[i] = Regime.TRENDING_DOWN.value
        
        return regimes
    
    def detect_volatility_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Detect volatility regime: 0=low, 1=normal, 2=high, 3=extreme."""
        volatility = features.get("volatility_1h", pd.Series(0, index=features.index))
        vol_zscore = (volatility - volatility.rolling(100).mean()) / volatility.rolling(100).std().replace(0, 1)
        
        regimes = np.ones(len(features), dtype=int)  # Default normal
        regimes[vol_zscore < -1] = 0  # Low
        regimes[vol_zscore > 1] = 2   # High
        regimes[vol_zscore > 2] = 3   # Extreme
        
        return regimes


# =============================================================================
# SIGNAL GENERATION (from behavioral primitives)
# =============================================================================

@dataclass
class SyntheticSignal:
    """Synthetic signal for training data."""
    timestamp: datetime
    side: int  # 1=LONG, -1=SHORT
    confidence: float
    source: SignalSource
    expected_return: float
    expected_adverse_excursion: float


def generate_signals_from_data(
    features: pd.DataFrame,
    lookahead: int = 6,  # 30 min at 5min candles
    min_profitable_move: float = 0.002,  # 0.2% minimum move for profit
) -> List[SyntheticSignal]:
    """
    Generate signals using behavioral signal logic on historical data.
    
    Uses hindsight to only generate signals at high-quality setups.
    This is valid for training data since we want to teach the model
    what good signals look like.
    """
    signals = []
    
    funding = features.get("funding_rate", pd.Series(0, index=features.index)).values
    oi_delta = features.get("oi_delta_pct", pd.Series(0, index=features.index)).values
    liq_imbalance = features.get("liq_imbalance", pd.Series(0, index=features.index)).values
    cascade_prob = features.get("cascade_probability", pd.Series(0, index=features.index)).values
    return_1h = features.get("return_1h", pd.Series(0, index=features.index)).values
    volatility = features.get("volatility_1h", pd.Series(0, index=features.index)).values
    ls_ratio = features.get("long_short_ratio", pd.Series(1, index=features.index)).values
    close = features.get("close", pd.Series(0, index=features.index)).values
    
    timestamps = features.index
    
    # Pre-compute future returns for hindsight filtering
    future_returns = np.zeros(len(features))
    for i in range(len(features) - lookahead):
        future_returns[i] = (close[i + lookahead] - close[i]) / close[i]
    
    for i in range(100, len(features) - lookahead):
        # Skip if cascade risk
        if cascade_prob[i] > 0.3:
            continue
        
        # =====================================================================
        # FUNDING_SQUEEZE
        # =====================================================================
        if funding[i] > FUNDING_SQUEEZE_THRESHOLD and oi_delta[i] > 0:
            confidence = min(0.8, abs(funding[i]) * 500)
            if funding[i] > FUNDING_EXTREME_THRESHOLD:
                confidence = min(0.9, confidence * 1.2)
            
            signals.append(SyntheticSignal(
                timestamp=timestamps[i],
                side=-1,  # SHORT
                confidence=confidence,
                source=SignalSource.FUNDING_SQUEEZE,
                expected_return=0.02,
                expected_adverse_excursion=0.01,
            ))
        
        elif funding[i] < -FUNDING_SQUEEZE_THRESHOLD and oi_delta[i] > 0:
            confidence = min(0.8, abs(funding[i]) * 500)
            if funding[i] < -FUNDING_EXTREME_THRESHOLD:
                confidence = min(0.9, confidence * 1.2)
            
            signals.append(SyntheticSignal(
                timestamp=timestamps[i],
                side=1,  # LONG
                confidence=confidence,
                source=SignalSource.FUNDING_SQUEEZE,
                expected_return=0.02,
                expected_adverse_excursion=0.01,
            ))
        
        # =====================================================================
        # LIQUIDATION_REVERSAL
        # =====================================================================
        elif abs(liq_imbalance[i]) > LIQ_IMBALANCE_THRESHOLD and cascade_prob[i] < 0.3:
            confidence = 0.65 + (abs(liq_imbalance[i]) - LIQ_IMBALANCE_THRESHOLD) * 0.5
            
            if liq_imbalance[i] > LIQ_IMBALANCE_THRESHOLD:
                # Heavy long liquidations -> LONG for bounce
                signals.append(SyntheticSignal(
                    timestamp=timestamps[i],
                    side=1,
                    confidence=min(0.8, confidence),
                    source=SignalSource.LIQUIDATION_REVERSAL,
                    expected_return=0.015,
                    expected_adverse_excursion=0.01,
                ))
            else:
                # Heavy short liquidations -> SHORT for pullback
                signals.append(SyntheticSignal(
                    timestamp=timestamps[i],
                    side=-1,
                    confidence=min(0.8, confidence),
                    source=SignalSource.LIQUIDATION_REVERSAL,
                    expected_return=0.015,
                    expected_adverse_excursion=0.01,
                ))
        
        # =====================================================================
        # OI_DIVERGENCE
        # =====================================================================
        elif abs(return_1h[i]) > OI_DIVERGENCE_PRICE_THRESHOLD and oi_delta[i] < -OI_DIVERGENCE_OI_THRESHOLD:
            confidence = 0.55 + min(0.2, abs(oi_delta[i]) * 5)
            
            if return_1h[i] > OI_DIVERGENCE_PRICE_THRESHOLD:
                # Price up, OI down = weak rally -> SHORT
                signals.append(SyntheticSignal(
                    timestamp=timestamps[i],
                    side=-1,
                    confidence=confidence,
                    source=SignalSource.OI_DIVERGENCE,
                    expected_return=0.015,
                    expected_adverse_excursion=0.012,
                ))
            else:
                # Price down, OI down = capitulation ending -> LONG
                signals.append(SyntheticSignal(
                    timestamp=timestamps[i],
                    side=1,
                    confidence=confidence,
                    source=SignalSource.OI_DIVERGENCE,
                    expected_return=0.015,
                    expected_adverse_excursion=0.012,
                ))
        
        # =====================================================================
        # CROWDING_FADE
        # =====================================================================
        elif ls_ratio[i] > CROWDING_LS_RATIO_HIGH and funding[i] > 0.0003:
            confidence = 0.6 + (ls_ratio[i] - CROWDING_LS_RATIO_HIGH) * 0.1
            signals.append(SyntheticSignal(
                timestamp=timestamps[i],
                side=-1,  # Fade longs -> SHORT
                confidence=min(0.75, confidence),
                source=SignalSource.CROWDING_FADE,
                expected_return=0.018,
                expected_adverse_excursion=0.012,
            ))
        
        elif ls_ratio[i] < CROWDING_LS_RATIO_LOW and funding[i] < -0.0003:
            confidence = 0.6 + (CROWDING_LS_RATIO_LOW - ls_ratio[i]) * 0.2
            signals.append(SyntheticSignal(
                timestamp=timestamps[i],
                side=1,  # Fade shorts -> LONG
                confidence=min(0.75, confidence),
                source=SignalSource.CROWDING_FADE,
                expected_return=0.018,
                expected_adverse_excursion=0.012,
            ))
        
        # =====================================================================
        # FUNDING_CARRY (in ranging, low vol)
        # =====================================================================
        elif volatility[i] < np.nanpercentile(volatility[:i+1], 50):
            cum_ret = pd.Series(return_1h[:i+1]).rolling(12).sum().iloc[-1] if i > 12 else 0
            
            if abs(cum_ret) < 0.02:  # Ranging
                if funding[i] > FUNDING_CARRY_MIN:
                    confidence = 0.5 + min(0.25, abs(funding[i]) * 300)
                    signals.append(SyntheticSignal(
                        timestamp=timestamps[i],
                        side=-1,  # SHORT to receive
                        confidence=confidence,
                        source=SignalSource.FUNDING_CARRY,
                        expected_return=abs(funding[i]) * 3,
                        expected_adverse_excursion=0.008,
                    ))
                elif funding[i] < -FUNDING_CARRY_MIN:
                    confidence = 0.5 + min(0.25, abs(funding[i]) * 300)
                    signals.append(SyntheticSignal(
                        timestamp=timestamps[i],
                        side=1,  # LONG to receive
                        confidence=confidence,
                        source=SignalSource.FUNDING_CARRY,
                        expected_return=abs(funding[i]) * 3,
                        expected_adverse_excursion=0.008,
                    ))
    
    # Also generate "anti-signals" - times when conditions look similar but move goes wrong
    # This helps the model learn what NOT to trade
    anti_signals = []
    
    for i in range(100, len(features) - lookahead):
        if cascade_prob[i] > 0.3:
            continue
        
        # Generate anti-signals at random market points that look tradeable but aren't
        # These become negative examples when the move doesn't go in signal direction
        if np.random.random() < 0.05:  # 5% of candles
            # Random direction
            direction = 1 if np.random.random() > 0.5 else -1
            
            # Check if this would have been profitable
            if direction == 1:
                would_profit = future_returns[i] > min_profitable_move
            else:
                would_profit = future_returns[i] < -min_profitable_move
            
            # Only add as anti-signal if it would have LOST (helps balance classes)
            if not would_profit:
                anti_signals.append(SyntheticSignal(
                    timestamp=timestamps[i],
                    side=direction,
                    confidence=0.5 + np.random.random() * 0.3,  # 0.5-0.8
                    source=SignalSource.UNKNOWN,
                    expected_return=0.01,
                    expected_adverse_excursion=0.015,
                ))
    
    # Combine and limit anti-signals to not overwhelm real signals
    max_anti = len(signals) // 2
    if len(anti_signals) > max_anti:
        np.random.shuffle(anti_signals)
        anti_signals = anti_signals[:max_anti]
    
    all_signals = signals + anti_signals
    
    logger.info(f"Generated {len(signals)} behavioral signals + {len(anti_signals)} anti-signals from {len(features)} candles")
    
    # Log distribution
    source_counts = {}
    for sig in all_signals:
        source_counts[sig.source.name] = source_counts.get(sig.source.name, 0) + 1
    for source, count in source_counts.items():
        pct = count / len(all_signals) * 100 if all_signals else 0
        logger.info(f"  {source}: {count} ({pct:.1f}%)")
    
    return all_signals


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_signal_scorer_dataset(
    raw_df: pd.DataFrame,
    lookahead_minutes: int = 30,
    transaction_cost: float = 0.001,
    min_confidence: float = 0.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create complete training dataset for Signal Scorer.
    
    Args:
        raw_df: Raw DataFrame with OHLCV + optional funding/OI/liquidations
        lookahead_minutes: How far ahead to look for profitability
        transaction_cost: Round-trip transaction cost
        min_confidence: Minimum signal confidence to include
    
    Returns:
        (features DataFrame, labels Series)
    """
    logger.info(f"Creating Signal Scorer dataset from {len(raw_df)} candles")
    
    # Engineer market features
    engineer = SignalScorerFeatureEngineer(lookahead_minutes, transaction_cost)
    market_features = engineer.engineer_market_features(raw_df)
    
    # Detect regimes
    regimes = engineer.detect_regime(market_features)
    vol_regimes = engineer.detect_volatility_regime(market_features)
    
    # Generate signals
    signals = generate_signals_from_data(market_features)
    
    if len(signals) == 0:
        logger.warning("No signals generated! Check data quality.")
        return pd.DataFrame(), pd.Series()
    
    # Filter by minimum confidence
    signals = [s for s in signals if s.confidence >= min_confidence]
    logger.info(f"Signals after confidence filter: {len(signals)}")
    
    # Build feature rows and labels
    X_rows = []
    y_labels = []
    
    lookahead = lookahead_minutes // 5  # Convert to candles
    close_values = market_features["close"].values
    
    for signal in signals:
        if signal.timestamp not in market_features.index:
            continue
        
        idx = market_features.index.get_loc(signal.timestamp)
        
        # Skip if not enough lookahead
        if idx + lookahead >= len(close_values):
            continue
        
        # Get market features at signal time
        market_row = market_features.iloc[idx]
        
        # Build complete feature row
        row = {
            # === SIGNAL FEATURES ===
            "signal_direction": signal.side,
            "signal_confidence": signal.confidence,
            "signal_source_funding_squeeze": 1.0 if signal.source == SignalSource.FUNDING_SQUEEZE else 0.0,
            "signal_source_liq_reversal": 1.0 if signal.source == SignalSource.LIQUIDATION_REVERSAL else 0.0,
            "signal_source_oi_divergence": 1.0 if signal.source == SignalSource.OI_DIVERGENCE else 0.0,
            "signal_source_crowding_fade": 1.0 if signal.source == SignalSource.CROWDING_FADE else 0.0,
            "signal_source_funding_carry": 1.0 if signal.source == SignalSource.FUNDING_CARRY else 0.0,
            "expected_return": signal.expected_return,
            "expected_adverse_excursion": signal.expected_adverse_excursion,
            
            # === PRICE FEATURES ===
            "return_1m": market_row.get("return_1m", 0),
            "return_5m": market_row.get("return_5m", 0),
            "return_15m": market_row.get("return_15m", 0),
            "return_1h": market_row.get("return_1h", 0),
            "volatility_5m": market_row.get("volatility_5m", 0),
            "volatility_1h": market_row.get("volatility_1h", 0),
            "price_vs_sma_20": market_row.get("price_vs_sma_20", 0),
            "price_vs_sma_50": market_row.get("price_vs_sma_50", 0),
            "rsi_14": market_row.get("rsi_14", 50),
            "atr_14": market_row.get("atr_14", 0),
            
            # === FUNDING FEATURES ===
            "funding_rate": market_row.get("funding_rate", 0),
            "funding_rate_zscore": market_row.get("funding_rate_zscore", 0),
            "funding_annualized": market_row.get("funding_annualized", 0),
            "funding_momentum": market_row.get("funding_momentum", 0),
            
            # === OI FEATURES ===
            "oi_delta_pct": market_row.get("oi_delta_pct", 0),
            "oi_delta_zscore": market_row.get("oi_delta_zscore", 0),
            "oi_price_divergence": market_row.get("oi_price_divergence", 0),
            
            # === LIQUIDATION FEATURES ===
            "liq_imbalance": market_row.get("liq_imbalance", 0),
            "liq_velocity": market_row.get("liq_velocity", 0),
            "liq_zscore": market_row.get("liq_zscore", 0),
            
            # === ORDER BOOK FEATURES ===
            "ob_imbalance": market_row.get("ob_imbalance", 0),
            "spread_bps": market_row.get("spread_bps", 0),
            "bid_depth_usd": market_row.get("bid_depth_usd", 0),
            "ask_depth_usd": market_row.get("ask_depth_usd", 0),
            
            # === POSITIONING FEATURES ===
            "long_short_ratio": market_row.get("long_short_ratio", 1),
            "taker_buy_sell_ratio": market_row.get("taker_buy_sell_ratio", 1),
            
            # === REGIME FEATURES ===
            "regime_trending_up": 1.0 if regimes[idx] == Regime.TRENDING_UP.value else 0.0,
            "regime_trending_down": 1.0 if regimes[idx] == Regime.TRENDING_DOWN.value else 0.0,
            "regime_ranging": 1.0 if regimes[idx] == Regime.RANGING.value else 0.0,
            "regime_high_vol": 1.0 if regimes[idx] == Regime.HIGH_VOLATILITY.value else 0.0,
            "regime_cascade": 1.0 if regimes[idx] == Regime.CASCADE_RISK.value else 0.0,
            "regime_squeeze_long": 1.0 if regimes[idx] == Regime.SQUEEZE_LONG.value else 0.0,
            "regime_squeeze_short": 1.0 if regimes[idx] == Regime.SQUEEZE_SHORT.value else 0.0,
            "volatility_regime": vol_regimes[idx],
            "cascade_probability": market_row.get("cascade_probability", 0),
            
            # === TIME FEATURES ===
            "hour_of_day_sin": market_row.get("hour_of_day_sin", 0),
            "hour_of_day_cos": market_row.get("hour_of_day_cos", 0),
            "day_of_week_sin": market_row.get("day_of_week_sin", 0),
            "day_of_week_cos": market_row.get("day_of_week_cos", 0),
            "minutes_to_funding": market_row.get("minutes_to_funding", 0),
        }
        
        X_rows.append(row)
        
        # === CREATE LABEL ===
        entry_price = close_values[idx]
        exit_price = close_values[idx + lookahead]
        
        if signal.side == 1:  # LONG
            return_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            return_pct = (entry_price - exit_price) / entry_price
        
        # Profitable after fees?
        net_return = return_pct - transaction_cost
        y_labels.append(1 if net_return > 0 else 0)
    
    X = pd.DataFrame(X_rows)
    y = pd.Series(y_labels)
    
    # Replace NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Created dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    logger.info(f"Win rate: {y.mean()*100:.1f}%")
    
    return X, y


def get_signal_scorer_feature_names() -> List[str]:
    """Get ordered list of feature names for Signal Scorer."""
    return [
        # Signal features (9)
        "signal_direction", "signal_confidence",
        "signal_source_funding_squeeze", "signal_source_liq_reversal",
        "signal_source_oi_divergence", "signal_source_crowding_fade", "signal_source_funding_carry",
        "expected_return", "expected_adverse_excursion",
        
        # Price features (10)
        "return_1m", "return_5m", "return_15m", "return_1h",
        "volatility_5m", "volatility_1h",
        "price_vs_sma_20", "price_vs_sma_50", "rsi_14", "atr_14",
        
        # Funding features (4)
        "funding_rate", "funding_rate_zscore", "funding_annualized", "funding_momentum",
        
        # OI features (3)
        "oi_delta_pct", "oi_delta_zscore", "oi_price_divergence",
        
        # Liquidation features (3)
        "liq_imbalance", "liq_velocity", "liq_zscore",
        
        # Order book features (4)
        "ob_imbalance", "spread_bps", "bid_depth_usd", "ask_depth_usd",
        
        # Positioning features (2)
        "long_short_ratio", "taker_buy_sell_ratio",
        
        # Regime features (9)
        "regime_trending_up", "regime_trending_down", "regime_ranging",
        "regime_high_vol", "regime_cascade", "regime_squeeze_long", "regime_squeeze_short",
        "volatility_regime", "cascade_probability",
        
        # Time features (5)
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos", "minutes_to_funding",
    ]
