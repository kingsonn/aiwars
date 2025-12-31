"""
Layer 2: Statistical Reality Engine

Defines what is random vs meaningful.
Uses: GBM, Jump-Diffusion, Hawkes Processes, ML Regime Classification.
Outputs: Expected range, abnormal move score, jump probability, regime alerts.

This layer NEVER predicts direction - it defines danger zones.
It outputs a TRADING DECISION: ALLOW, RESTRICT, or BLOCK.

If BLOCK -> No trading allowed, system stops.

Integration:
- Consumes MarketState from Layer 1 (layer1_market_intel.py)
- Uses trained RegimeClassifier model for ML-based regime detection
- Outputs StatisticalResult for Layer 3 (Alpha) and Layer 4 (Risk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import MarketState, OHLCV, Regime, Side


# =============================================================================
# ENUMS
# =============================================================================

class TradabilityStatus(Enum):
    """Tradability status per HYDRA_SPEC_LAYERS.md."""
    ALLOW = "allow"       # Safe to trade
    RESTRICT = "restrict" # Exits only, no entries
    BLOCK = "block"       # Close everything


# Alias for backwards compatibility
TradingDecision = TradabilityStatus


class MarketEnvironment(Enum):
    """Market environment classification."""
    NORMAL = "normal"         # Standard conditions
    TRENDING = "trending"     # Clear directional move
    VOLATILE = "volatile"     # High but manageable volatility
    CHAOTIC = "chaotic"       # Unstable, unpredictable
    SQUEEZE = "squeeze"       # Liquidation cascade risk


@dataclass
class StatisticalResult:
    """Output from Statistical Reality Engine."""
    timestamp: datetime
    symbol: str
    
    # === TRADING DECISION (PRIMARY OUTPUT) ===
    trading_decision: TradingDecision
    decision_reasons: list[str] = field(default_factory=list)
    environment: MarketEnvironment = MarketEnvironment.NORMAL
    max_position_pct: float = 100.0  # Max position size as % of normal
    
    # Regime
    regime: Regime = Regime.UNKNOWN
    regime_confidence: float = 0.0
    regime_break_alert: bool = False
    
    # Volatility
    realized_volatility: float = 0.0  # Current realized vol (annualized)
    implied_volatility: float = 0.0  # Estimated from price action
    volatility_regime: str = "normal"  # "low", "normal", "high", "extreme"
    volatility_zscore: float = 0.0
    
    # Price range expectations
    expected_range_1h: tuple[float, float] = (0.0, 0.0)  # (low, high)
    expected_range_4h: tuple[float, float] = (0.0, 0.0)
    expected_range_24h: tuple[float, float] = (0.0, 0.0)
    
    # Abnormality detection
    abnormal_move_score: float = 0.0  # Z-score of recent move
    is_abnormal: bool = False
    
    # Jump process
    jump_probability: float = 0.0  # Probability of jump in next period
    jump_intensity: float = 0.0  # Hawkes process intensity
    expected_jump_size: float = 0.0
    
    # Cascade risk (from liquidations)
    cascade_probability: float = 0.0
    liquidation_velocity: float = 0.0  # Rate of liquidations
    liquidation_imbalance: float = 0.0  # Long vs short liquidations
    
    # Funding & positioning pressure
    funding_pressure: float = 0.0  # Extreme funding = squeeze risk
    oi_change_pct: float = 0.0  # OI delta
    orderbook_imbalance: float = 0.0  # Bid/ask imbalance
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_risk_score: float = 0.0
    
    # Composite danger score (0-100)
    danger_score: float = 0.0


class GeometricBrownianMotion:
    """GBM model for baseline price expectations."""
    
    def __init__(self):
        self.mu = 0.0  # Drift
        self.sigma = 0.0  # Volatility
    
    def fit(self, prices: np.ndarray, dt: float = 1/1440) -> None:
        """Fit GBM to price series. dt in years (1/1440 for 1m data)."""
        if len(prices) < 20:
            return
        
        log_returns = np.diff(np.log(prices))
        
        self.mu = np.mean(log_returns) / dt
        self.sigma = np.std(log_returns) / np.sqrt(dt)
    
    def expected_range(self, S0: float, T: float, confidence: float = 0.95) -> tuple[float, float]:
        """Calculate expected price range over time T (in years)."""
        if self.sigma == 0:
            return (S0 * 0.95, S0 * 1.05)
        
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # Log-normal distribution parameters
        mean_log = np.log(S0) + (self.mu - 0.5 * self.sigma**2) * T
        std_log = self.sigma * np.sqrt(T)
        
        low = np.exp(mean_log - z * std_log)
        high = np.exp(mean_log + z * std_log)
        
        return (low, high)
    
    def move_zscore(self, price_change_pct: float, T: float) -> float:
        """Calculate z-score of a price move."""
        if self.sigma == 0:
            return 0.0
        
        expected_std = self.sigma * np.sqrt(T)
        return price_change_pct / expected_std


class JumpDiffusionModel:
    """Merton's Jump-Diffusion model for liquidation events."""
    
    def __init__(self):
        self.mu = 0.0  # Drift
        self.sigma = 0.0  # Diffusion volatility
        self.lambda_j = 0.0  # Jump intensity (jumps per period)
        self.mu_j = 0.0  # Mean jump size
        self.sigma_j = 0.0  # Jump size volatility
        self._n_periods = 0  # Number of periods observed
        self._n_jumps = 0  # Number of jumps detected
    
    def fit(self, prices: np.ndarray, threshold_sigma: float = 2.5) -> None:
        """Fit jump-diffusion model with more sensitive jump detection."""
        if len(prices) < 30:
            return
        
        log_returns = np.diff(np.log(prices))
        self._n_periods = len(log_returns)
        
        # Use robust MAD-based estimate for sigma (less affected by jumps)
        mad = np.median(np.abs(log_returns - np.median(log_returns)))
        sigma_robust = 1.4826 * mad  # Scale to match std for normal dist
        
        # Also use rolling std for comparison
        sigma_est = max(sigma_robust, np.std(log_returns) * 0.7)
        
        # Identify jumps as returns beyond threshold (lowered from 3.0 to 2.5)
        jumps = np.abs(log_returns) > threshold_sigma * sigma_est
        self._n_jumps = np.sum(jumps)
        
        # Normal returns (excluding jumps)
        normal_returns = log_returns[~jumps]
        jump_returns = log_returns[jumps]
        
        if len(normal_returns) > 10:
            self.mu = np.mean(normal_returns)
            self.sigma = np.std(normal_returns)
        else:
            self.mu = np.mean(log_returns)
            self.sigma = sigma_est
        
        # Jump parameters - scale to per-hour rate for 1m data
        # If we have 500 1m candles, that's ~8.3 hours
        hours_observed = self._n_periods / 60
        self.lambda_j = self._n_jumps / max(hours_observed, 1)  # Jumps per hour
        
        if len(jump_returns) > 0:
            self.mu_j = np.mean(np.abs(jump_returns))  # Use absolute for expected size
            self.sigma_j = np.std(jump_returns) if len(jump_returns) > 1 else self.mu_j * 0.5
        else:
            # Even with no observed jumps, estimate potential jump size
            self.mu_j = sigma_est * threshold_sigma
            self.sigma_j = sigma_est
    
    def jump_probability(self, T_hours: float = 1.0) -> float:
        """Probability of at least one jump in T hours."""
        # Poisson probability of at least one event
        expected_jumps = self.lambda_j * T_hours
        prob = 1 - np.exp(-expected_jumps)
        return min(prob, 0.95)  # Cap at 95%
    
    def expected_jump_size(self) -> float:
        """Expected absolute jump size as percentage."""
        return self.mu_j + 0.5 * self.sigma_j


class HawkesProcess:
    """Hawkes self-exciting point process for cascade detection."""
    
    def __init__(self):
        self.mu = 0.1  # Baseline intensity
        self.alpha = 0.3  # Excitation parameter (reduced default)
        self.beta = 1.5  # Decay parameter (faster decay)
        self._event_times: list[float] = []
        self._fitted = False
    
    def fit(self, event_times: list[float], T: float) -> None:
        """
        Fit Hawkes process to event times.
        event_times: list of event timestamps (normalized to [0, T])
        T: observation window
        """
        if len(event_times) < 3:
            self._fitted = False
            self._event_times = []
            return
        
        self._event_times = sorted(event_times)
        self._fitted = True
        
        def neg_log_likelihood(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
                return 1e10
            
            ll = 0
            for i, ti in enumerate(self._event_times):
                intensity = mu + alpha * sum(
                    np.exp(-beta * (ti - tj))
                    for tj in self._event_times[:i]
                )
                ll += np.log(max(intensity, 1e-10))
            
            # Integral term
            integral = mu * T
            for ti in self._event_times:
                integral += (alpha / beta) * (1 - np.exp(-beta * (T - ti)))
            
            ll -= integral
            return -ll
        
        try:
            result = minimize(
                neg_log_likelihood,
                x0=[0.1, 0.3, 1.0],
                bounds=[(1e-6, 10), (0, 5), (0.1, 10)],
                method='L-BFGS-B'
            )
            
            if result.success:
                self.mu, self.alpha, self.beta = result.x
        except Exception:
            pass
    
    def current_intensity(self, t: float) -> float:
        """Calculate current intensity at time t."""
        intensity = self.mu
        for ti in self._event_times:
            if ti < t:
                intensity += self.alpha * np.exp(-self.beta * (t - ti))
        return intensity
    
    def cascade_probability(self, horizon: float = 1.0) -> float:
        """Estimate probability of cascade (intensity exceeding threshold)."""
        if not self._fitted or not self._event_times:
            return 0.0
        
        current = self.current_intensity(max(self._event_times))
        threshold = self.mu * 3  # 3x baseline indicates elevated activity
        
        # Branching ratio determines if process is subcritical/critical
        branching_ratio = self.alpha / self.beta if self.beta > 0 else 0
        
        # Subcritical (stable): branching_ratio < 1
        if branching_ratio >= 0.9:
            # Near-critical or supercritical - high cascade risk
            return min(0.8, 0.5 + branching_ratio * 0.3)
        
        # If current intensity is elevated
        if current > threshold:
            # Scale by how much we exceed threshold
            excess = (current - threshold) / threshold
            return min(0.7, excess * branching_ratio)
        
        # Low baseline cascade probability based on recent activity
        recent_events = len([t for t in self._event_times if t > 0.8])  # Last 20% of window
        if recent_events > 5:
            return min(0.4, recent_events * 0.05)
        
        return max(0.0, branching_ratio * 0.1)


class RegimeDetector:
    """Detect market regime from price action."""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
    
    def detect(self, candles: list[OHLCV]) -> tuple[Regime, float]:
        """Detect current regime. Returns (regime, confidence)."""
        if len(candles) < 20:
            return Regime.UNKNOWN, 0.0
        
        closes = np.array([c.close for c in candles[-self.lookback:]])
        highs = np.array([c.high for c in candles[-self.lookback:]])
        lows = np.array([c.low for c in candles[-self.lookback:]])
        
        # Calculate metrics
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        
        # Trend detection using multiple EMAs
        ema_10 = self._ema(closes, 10)
        ema_20 = self._ema(closes, 20)
        ema_50 = self._ema(closes, min(50, len(closes)))
        
        current_price = closes[-1]
        
        # Price position relative to EMAs
        above_ema10 = current_price > ema_10
        above_ema20 = current_price > ema_20
        above_ema50 = current_price > ema_50
        ema_aligned_bull = ema_10 > ema_20 > ema_50
        ema_aligned_bear = ema_10 < ema_20 < ema_50
        
        # Trend strength as % distance from EMA50
        trend_strength = (current_price - ema_50) / ema_50 if ema_50 > 0 else 0
        
        # ADX-like directional indicator
        up_moves = np.maximum(highs[1:] - highs[:-1], 0)
        down_moves = np.maximum(lows[:-1] - lows[1:], 0)
        
        avg_up = np.mean(up_moves[-14:]) if len(up_moves) >= 14 else np.mean(up_moves)
        avg_down = np.mean(down_moves[-14:]) if len(down_moves) >= 14 else np.mean(down_moves)
        
        if avg_up + avg_down > 0:
            directional_index = abs(avg_up - avg_down) / (avg_up + avg_down)
        else:
            directional_index = 0
        
        # Volatility regime
        if len(returns) > 40:
            vol_windows = [np.std(returns[i:i+20]) for i in range(len(returns)-20)]
            vol_percentile = stats.percentileofscore(vol_windows, volatility)
        else:
            vol_percentile = 50
        
        # Recent large moves
        recent_returns = returns[-10:] if len(returns) >= 10 else returns
        max_move = np.max(np.abs(recent_returns)) if len(recent_returns) > 0 else 0
        
        # Regime classification with relaxed thresholds
        if vol_percentile > 85 and max_move > 2.5 * volatility:
            regime = Regime.CASCADE_RISK
            confidence = min(0.9, vol_percentile / 100)
        elif vol_percentile > 75:
            regime = Regime.HIGH_VOLATILITY
            confidence = 0.65 + (vol_percentile - 75) / 100
        elif ema_aligned_bull and trend_strength > 0.005:
            regime = Regime.TRENDING_UP
            confidence = min(0.85, 0.5 + abs(trend_strength) * 10 + directional_index * 0.3)
        elif ema_aligned_bear and trend_strength < -0.005:
            regime = Regime.TRENDING_DOWN
            confidence = min(0.85, 0.5 + abs(trend_strength) * 10 + directional_index * 0.3)
        elif above_ema20 and above_ema50 and trend_strength > 0.002:
            regime = Regime.TRENDING_UP
            confidence = min(0.7, 0.4 + abs(trend_strength) * 8)
        elif not above_ema20 and not above_ema50 and trend_strength < -0.002:
            regime = Regime.TRENDING_DOWN
            confidence = min(0.7, 0.4 + abs(trend_strength) * 8)
        elif directional_index < 0.25 or vol_percentile < 40:
            regime = Regime.RANGING
            confidence = 0.55 + (0.25 - directional_index) if directional_index < 0.25 else 0.55
        else:
            # Default to ranging if unclear
            regime = Regime.RANGING
            confidence = 0.4
        
        return regime, min(confidence, 0.95)
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def detect_regime_break(
        self, 
        current_regime: Regime,
        candles: list[OHLCV],
        threshold: float = 0.3
    ) -> bool:
        """Detect if regime has changed significantly."""
        new_regime, confidence = self.detect(candles)
        
        if new_regime != current_regime and confidence > threshold:
            return True
        
        return False


# =============================================================================
# ML-BASED REGIME DETECTOR
# =============================================================================

class MLRegimeDetector:
    """
    ML-based regime detection using trained RegimeClassifier model.
    
    Integrates with the RegimeClassifier from hydra.training.models.
    Falls back to rule-based detection if model not available.
    """
    
    # Regime mapping from model output to Regime enum
    REGIME_MAP = {
        0: Regime.TRENDING_UP,
        1: Regime.TRENDING_DOWN,
        2: Regime.RANGING,
        3: Regime.HIGH_VOLATILITY,
        4: Regime.CASCADE_RISK,
        5: Regime.SQUEEZE_LONG,
        6: Regime.SQUEEZE_SHORT,
    }
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or "models/regime_classifier.pkl"
        self.fallback_detector = RegimeDetector()
        self._load_model()
    
    def _load_model(self):
        """Load the trained regime classifier model."""
        try:
            from hydra.training.models import load_regime_classifier
            
            model_file = Path(self.model_path)
            if model_file.exists():
                self.model = load_regime_classifier(str(model_file))
                logger.info(f"Loaded ML Regime Classifier from {model_file}")
            else:
                logger.warning(f"Regime model not found at {model_file}, using rule-based fallback")
        except Exception as e:
            logger.warning(f"Failed to load ML Regime Classifier: {e}, using fallback")
            self.model = None
    
    def extract_features(self, market_state: MarketState) -> pd.DataFrame:
        """
        Extract features for regime classification from MarketState.
        
        This converts live Layer 1 data into the feature format expected by the model.
        """
        # Get candles
        candles_5m = market_state.ohlcv.get("5m", [])
        candles_15m = market_state.ohlcv.get("15m", [])
        candles = candles_15m if len(candles_15m) >= 50 else candles_5m
        
        if len(candles) < 50:
            return pd.DataFrame()
        
        # Convert to arrays
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Calculate features matching the training pipeline
        features = {}
        
        # Returns
        returns = np.diff(closes) / closes[:-1]
        features["return_1h"] = (closes[-1] - closes[-4]) / closes[-4] if len(closes) > 4 else 0
        features["return_4h"] = (closes[-1] - closes[-16]) / closes[-16] if len(closes) > 16 else 0
        features["return_24h"] = (closes[-1] - closes[-96]) / closes[-96] if len(closes) > 96 else 0
        
        # SMAs and slopes
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        
        sma_20_prev = np.mean(closes[-25:-5]) if len(closes) >= 25 else sma_20
        sma_50_prev = np.mean(closes[-55:-5]) if len(closes) >= 55 else sma_50
        
        features["sma_20_slope"] = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev > 0 else 0
        features["sma_50_slope"] = (sma_50 - sma_50_prev) / sma_50_prev if sma_50_prev > 0 else 0
        features["price_vs_sma_20"] = (closes[-1] - sma_20) / sma_20 if sma_20 > 0 else 0
        features["price_vs_sma_50"] = (closes[-1] - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # ADX (simplified)
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                  np.abs(lows[1:] - closes[:-1])))
        atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0
        
        up_moves = np.maximum(highs[1:] - highs[:-1], 0)
        down_moves = np.maximum(lows[:-1] - lows[1:], 0)
        avg_up = np.mean(up_moves[-14:]) if len(up_moves) >= 14 else 0
        avg_down = np.mean(down_moves[-14:]) if len(down_moves) >= 14 else 0
        
        if avg_up + avg_down > 0:
            features["adx_14"] = abs(avg_up - avg_down) / (avg_up + avg_down)
        else:
            features["adx_14"] = 0
        
        # Volatility
        vol_5m = np.std(returns[-5:]) * np.sqrt(365 * 24 * 4) if len(returns) >= 5 else 0
        vol_1h = np.std(returns[-12:]) * np.sqrt(365 * 24 * 4) if len(returns) >= 12 else 0
        vol_24h = np.std(returns[-96:]) * np.sqrt(365 * 24 * 4) if len(returns) >= 96 else 0
        
        features["volatility_5m"] = vol_5m
        features["volatility_1h"] = vol_1h
        features["volatility_24h"] = vol_24h
        
        vol_mean = np.mean([vol_1h])
        vol_std = np.std([vol_1h]) if vol_1h > 0 else 1
        features["volatility_zscore"] = (vol_1h - vol_mean) / vol_std if vol_std > 0 else 0
        
        # ATR
        features["atr_14"] = atr_14
        atr_mean = np.mean(tr[-100:]) if len(tr) >= 100 else atr_14
        atr_std = np.std(tr[-100:]) if len(tr) >= 100 else 1
        features["atr_zscore"] = (atr_14 - atr_mean) / atr_std if atr_std > 0 else 0
        
        # Bollinger width
        bb_std = np.std(closes[-20:]) if len(closes) >= 20 else 0
        features["bollinger_width"] = (4 * bb_std) / sma_20 if sma_20 > 0 else 0
        
        # Funding
        funding_rate = market_state.funding_rate.rate if market_state.funding_rate else 0
        features["funding_rate"] = funding_rate
        features["funding_zscore"] = funding_rate / 0.0001 if funding_rate != 0 else 0  # Normalize
        
        # OI
        oi_delta_pct = market_state.open_interest.delta_pct if market_state.open_interest else 0
        features["oi_delta_pct"] = oi_delta_pct
        features["oi_zscore"] = oi_delta_pct / 0.01 if oi_delta_pct != 0 else 0  # Normalize
        
        # Liquidations
        liqs = market_state.recent_liquidations
        total_liq = sum(l.usd_value for l in liqs) if liqs else 0
        oi_usd = market_state.open_interest.open_interest_usd if market_state.open_interest else 1
        
        features["liq_velocity"] = total_liq / oi_usd if oi_usd > 0 else 0
        
        if liqs:
            long_liq = sum(l.usd_value for l in liqs if l.side == Side.LONG)
            short_liq = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
            total = long_liq + short_liq
            features["liq_imbalance"] = (long_liq - short_liq) / total if total > 0 else 0
        else:
            features["liq_imbalance"] = 0
        
        # Cascade probability (heuristic)
        features["cascade_probability"] = min(1.0, features["liq_velocity"] * 10)
        
        # Volume
        vol_mean = np.mean(volumes[-100:]) if len(volumes) >= 100 else np.mean(volumes)
        vol_std = np.std(volumes[-100:]) if len(volumes) >= 100 else 1
        features["volume_zscore"] = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0
        
        # CVD momentum (simplified)
        features["cvd_momentum"] = 0  # Would need tick data
        
        return pd.DataFrame([features])
    
    def detect(self, market_state: MarketState) -> tuple[Regime, float]:
        """
        Detect regime using ML model with fallback to rule-based.
        
        Returns: (regime, confidence)
        """
        # Try ML model first
        if self.model is not None:
            try:
                features = self.extract_features(market_state)
                
                if features.empty:
                    logger.debug("Insufficient data for ML regime detection, using fallback")
                    return self._fallback_detect(market_state)
                
                # Get prediction
                regime_idx = self.model.predict(features)[0]
                proba = self.model.predict_proba(features)[0]
                confidence = float(np.max(proba))
                
                # Map to Regime enum
                regime = self.REGIME_MAP.get(regime_idx, Regime.UNKNOWN)
                
                logger.debug(f"ML Regime: {regime.name} (confidence: {confidence:.2%})")
                return regime, confidence
                
            except Exception as e:
                logger.warning(f"ML regime detection failed: {e}, using fallback")
                return self._fallback_detect(market_state)
        
        return self._fallback_detect(market_state)
    
    def _fallback_detect(self, market_state: MarketState) -> tuple[Regime, float]:
        """Fallback to rule-based detection."""
        candles = market_state.ohlcv.get("5m", []) or market_state.ohlcv.get("15m", [])
        return self.fallback_detector.detect(candles)
    
    def detect_regime_break(
        self,
        current_regime: Regime,
        market_state: MarketState,
        threshold: float = 0.3
    ) -> bool:
        """Detect if regime has changed significantly."""
        new_regime, confidence = self.detect(market_state)
        return new_regime != current_regime and confidence > threshold


# =============================================================================
# DATA HEALTH CHECKER
# =============================================================================

class DataHealthChecker:
    """
    Check data health per HYDRA_SPEC_LAYERS.md.
    
    Ensures all required data is fresh and valid.
    """
    
    def __init__(self, max_age_seconds: int = 60):
        self.max_age_seconds = max_age_seconds
    
    def check(self, market_state: MarketState) -> tuple[bool, list[str]]:
        """
        Check if market data is healthy.
        
        Returns: (is_healthy, list of issues)
        """
        issues = []
        now = datetime.now(timezone.utc)
        
        # 1. Data freshness
        data_age = (now - market_state.timestamp).total_seconds()
        if data_age > self.max_age_seconds:
            issues.append(f"Data stale: {data_age:.0f}s old (max: {self.max_age_seconds}s)")
        
        # 2. Funding rate present
        if market_state.funding_rate is None:
            issues.append("Missing funding rate data")
        
        # 3. Open interest present
        if market_state.open_interest is None:
            issues.append("Missing open interest data")
        
        # 4. Sufficient candle data
        candles_5m = market_state.ohlcv.get("5m", [])
        if len(candles_5m) < 20:
            issues.append(f"Insufficient 5m candles: {len(candles_5m)} (need 20+)")
        
        # 5. Order book spread
        if market_state.order_book:
            if market_state.order_book.spread > 0.01:
                issues.append(f"Wide spread: {market_state.order_book.spread:.2%} (max: 1%)")
        else:
            issues.append("Missing order book data")
        
        is_healthy = len(issues) == 0
        return is_healthy, issues


class StatisticalRealityEngine:
    """
    Layer 2: Statistical Reality Engine
    
    This layer answers: "Is this move statistically significant?"
    It defines danger zones but NEVER predicts direction.
    
    Integration:
    - Consumes MarketState from Layer 1 (layer1_market_intel.py)
    - Uses ML RegimeClassifier for regime detection
    - Outputs StatisticalResult for Layer 3 and Layer 4
    """
    
    def __init__(self, config: HydraConfig, use_ml_regime: bool = True):
        self.config = config
        self.use_ml_regime = use_ml_regime
        
        # Statistical models
        self.gbm = GeometricBrownianMotion()
        self.jump_model = JumpDiffusionModel()
        self.hawkes = HawkesProcess()
        
        # Regime detection - prefer ML model
        if use_ml_regime:
            self.ml_regime_detector = MLRegimeDetector()
        else:
            self.ml_regime_detector = None
        self.rule_regime_detector = RegimeDetector()
        
        # Data health checker
        self.data_health_checker = DataHealthChecker()
        
        self._last_regime: dict[str, Regime] = {}
        self._volatility_history: dict[str, list[float]] = {}
        
        logger.info(f"Statistical Reality Engine initialized (ML regime: {use_ml_regime})")
    
    async def setup(self) -> None:
        """Initialize engine."""
        pass
    
    async def analyze(self, market_state: MarketState) -> StatisticalResult:
        """
        Perform complete statistical analysis using ALL Layer 1 data.
        Returns regime, volatility metrics, danger signals, and TRADING DECISION.
        """
        symbol = market_state.symbol
        
        # Get price data
        candles_1m = market_state.ohlcv.get("1m", [])
        candles_5m = market_state.ohlcv.get("5m", [])
        
        if len(candles_1m) < 50:
            return self._empty_result(symbol)
        
        prices = np.array([c.close for c in candles_1m])
        current_price = market_state.price
        
        # =================================================================
        # 1. PRICE-BASED ANALYSIS (GBM, Jump-Diffusion)
        # =================================================================
        self.gbm.fit(prices)
        self.jump_model.fit(prices)
        
        returns = np.diff(np.log(prices))
        
        # =================================================================
        # 2. EXTRACT ALL LAYER 1 DATA
        # =================================================================
        # Funding rate
        funding_rate = 0.0
        funding_pressure = 0.0
        if market_state.funding_rate:
            funding_rate = market_state.funding_rate.rate
            # Extreme funding indicates squeeze risk
            # Normal: -0.01% to 0.01%, Extreme: > 0.05% or < -0.05%
            funding_pressure = abs(funding_rate) / 0.0005  # Normalized (1.0 = 0.05%)
        
        # Open Interest
        oi_change_pct = 0.0
        if market_state.open_interest:
            oi_change_pct = market_state.open_interest.delta_pct
        
        # Orderbook imbalance
        orderbook_imbalance = 0.0
        if market_state.order_book:
            orderbook_imbalance = market_state.order_book.imbalance
        
        # Liquidations - use actual liquidation data if available
        liquidation_imbalance = 0.0
        liq_count = len(market_state.recent_liquidations)
        if market_state.recent_liquidations:
            long_liqs = sum(1 for l in market_state.recent_liquidations if l.side == Side.LONG)
            short_liqs = liq_count - long_liqs
            if liq_count > 0:
                liquidation_imbalance = (long_liqs - short_liqs) / liq_count
        
        # Basis (futures vs spot)
        basis = market_state.basis
        
        # =================================================================
        # 3. HAWKES PROCESS - Use actual liquidations if available
        # =================================================================
        if liq_count >= 3:
            # Use actual liquidation times
            now = market_state.timestamp
            liq_times = []
            for liq in market_state.recent_liquidations:
                delta = (now - liq.timestamp).total_seconds()
                if delta < 3600:  # Last hour
                    liq_times.append(1.0 - delta / 3600)  # Normalize to [0, 1]
            if liq_times:
                self.hawkes.fit(liq_times, 1.0)
        else:
            # Fallback: use large price moves as proxy
            large_move_times = [
                i / len(returns) for i, r in enumerate(returns)
                if abs(r) > 2 * np.std(returns)
            ]
            self.hawkes.fit(large_move_times, 1.0)
        
        # =================================================================
        # 4. DATA HEALTH CHECK
        # =================================================================
        data_healthy, health_issues = self.data_health_checker.check(market_state)
        
        if not data_healthy:
            logger.warning(f"Data health issues for {symbol}: {health_issues}")
            # Don't block immediately - continue with analysis but flag it
        
        # =================================================================
        # 5. REGIME DETECTION (ML-based with fallback)
        # =================================================================
        if self.ml_regime_detector is not None:
            # Use ML model for regime detection
            regime, regime_confidence = self.ml_regime_detector.detect(market_state)
            last_regime = self._last_regime.get(symbol, Regime.UNKNOWN)
            regime_break = self.ml_regime_detector.detect_regime_break(last_regime, market_state)
        else:
            # Fallback to rule-based detection
            regime, regime_confidence = self.rule_regime_detector.detect(candles_5m or candles_1m)
            last_regime = self._last_regime.get(symbol, Regime.UNKNOWN)
            regime_break = self.rule_regime_detector.detect_regime_break(last_regime, candles_5m or candles_1m)
        
        self._last_regime[symbol] = regime
        
        # =================================================================
        # 6. VOLATILITY ANALYSIS
        # =================================================================
        realized_vol = np.std(returns) * np.sqrt(1440 * 365)  # Annualized
        
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = []
        self._volatility_history[symbol].append(realized_vol)
        self._volatility_history[symbol] = self._volatility_history[symbol][-1000:]
        
        vol_history = self._volatility_history[symbol]
        vol_zscore = (realized_vol - np.mean(vol_history)) / np.std(vol_history) if len(vol_history) > 20 else 0
        
        if vol_zscore > 2:
            vol_regime = "extreme"
        elif vol_zscore > 1:
            vol_regime = "high"
        elif vol_zscore < -1:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        # =================================================================
        # 7. EXPECTED PRICE RANGES
        # =================================================================
        range_1h = self.gbm.expected_range(current_price, 1/8760, 0.95)
        range_4h = self.gbm.expected_range(current_price, 4/8760, 0.95)
        range_24h = self.gbm.expected_range(current_price, 24/8760, 0.95)
        
        # =================================================================
        # 8. ABNORMAL MOVE DETECTION
        # =================================================================
        if len(candles_1m) >= 10:
            recent_return = (prices[-1] - prices[-10]) / prices[-10]
            abnormal_score = self.gbm.move_zscore(recent_return, 10/525600)
        else:
            abnormal_score = 0.0
        
        is_abnormal = abs(abnormal_score) > 3
        
        # =================================================================
        # 9. JUMP & CASCADE PROBABILITY
        # =================================================================
        jump_prob = self.jump_model.jump_probability(1.0)
        expected_jump = self.jump_model.expected_jump_size()
        cascade_prob = self.hawkes.cascade_probability()
        liq_velocity = self.hawkes.current_intensity(1.0)
        
        # =================================================================
        # 10. DISTRIBUTION METRICS
        # =================================================================
        skewness = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns)
        
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns < var_95]
        tail_risk = -np.mean(tail_returns) if len(tail_returns) > 0 else abs(var_95)
        
        # =================================================================
        # 11. COMPUTE DANGER SCORE (0-100)
        # =================================================================
        danger_score = self._compute_danger_score(
            vol_regime=vol_regime,
            vol_zscore=vol_zscore,
            is_abnormal=is_abnormal,
            abnormal_score=abnormal_score,
            jump_prob=jump_prob,
            cascade_prob=cascade_prob,
            regime=regime,
            regime_break=regime_break,
            funding_pressure=funding_pressure,
            oi_change_pct=oi_change_pct,
            liquidation_imbalance=liquidation_imbalance,
            kurtosis=kurtosis_val,
            liq_count=liq_count,
        )
        
        # =================================================================
        # 12. TRADING DECISION GATE
        # =================================================================
        decision, reasons, environment, max_pos = self._make_trading_decision(
            danger_score=danger_score,
            vol_regime=vol_regime,
            regime=regime,
            is_abnormal=is_abnormal,
            cascade_prob=cascade_prob,
            funding_pressure=funding_pressure,
            regime_break=regime_break,
            data_healthy=data_healthy,
            health_issues=health_issues,
        )
        
        return StatisticalResult(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            trading_decision=decision,
            decision_reasons=reasons,
            environment=environment,
            max_position_pct=max_pos,
            regime=regime,
            regime_confidence=regime_confidence,
            regime_break_alert=regime_break,
            realized_volatility=realized_vol,
            implied_volatility=realized_vol * 1.1,
            volatility_regime=vol_regime,
            volatility_zscore=vol_zscore,
            expected_range_1h=range_1h,
            expected_range_4h=range_4h,
            expected_range_24h=range_24h,
            abnormal_move_score=abnormal_score,
            is_abnormal=is_abnormal,
            jump_probability=jump_prob,
            jump_intensity=liq_velocity,
            expected_jump_size=expected_jump,
            cascade_probability=cascade_prob,
            liquidation_velocity=liq_velocity,
            liquidation_imbalance=liquidation_imbalance,
            funding_pressure=funding_pressure,
            oi_change_pct=oi_change_pct,
            orderbook_imbalance=orderbook_imbalance,
            skewness=skewness,
            kurtosis=kurtosis_val,
            tail_risk_score=tail_risk,
            danger_score=danger_score,
        )
    
    def _compute_danger_score(
        self,
        vol_regime: str,
        vol_zscore: float,
        is_abnormal: bool,
        abnormal_score: float,
        jump_prob: float,
        cascade_prob: float,
        regime: Regime,
        regime_break: bool,
        funding_pressure: float,
        oi_change_pct: float,
        liquidation_imbalance: float,
        kurtosis: float,
        liq_count: int,
    ) -> float:
        """Compute composite danger score from all factors."""
        score = 0.0
        
        # Volatility contribution (0-25 points)
        if vol_regime == "extreme":
            score += 25
        elif vol_regime == "high":
            score += 15
        elif vol_zscore > 1.5:
            score += 10
        
        # Abnormal move contribution (0-20 points)
        if is_abnormal:
            score += min(20, abs(abnormal_score) * 4)
        elif abs(abnormal_score) > 2:
            score += abs(abnormal_score) * 3
        
        # Jump probability contribution (0-15 points)
        score += jump_prob * 15
        
        # Cascade probability contribution (0-20 points)
        score += cascade_prob * 20
        
        # Regime contribution (0-10 points)
        if regime == Regime.CASCADE_RISK:
            score += 10
        elif regime == Regime.HIGH_VOLATILITY:
            score += 5
        
        if regime_break:
            score += 5
        
        # Funding pressure contribution (0-10 points)
        score += min(10, funding_pressure * 5)
        
        # OI change contribution (0-5 points)
        if abs(oi_change_pct) > 5:
            score += min(5, abs(oi_change_pct) / 2)
        
        # Liquidation imbalance (0-5 points)
        score += abs(liquidation_imbalance) * 5
        
        # Fat tails (0-5 points)
        if kurtosis > 5:
            score += min(5, (kurtosis - 5) / 2)
        
        # Recent liquidations (0-5 points)
        if liq_count > 10:
            score += min(5, liq_count / 4)
        
        return min(100, max(0, score))
    
    def _make_trading_decision(
        self,
        danger_score: float,
        vol_regime: str,
        regime: Regime,
        is_abnormal: bool,
        cascade_prob: float,
        funding_pressure: float,
        regime_break: bool,
        data_healthy: bool = True,
        health_issues: list[str] = None,
    ) -> tuple[TradingDecision, list[str], MarketEnvironment, float]:
        """
        Make trading decision based on statistical analysis.
        Returns: (decision, reasons, environment, max_position_pct)
        """
        reasons = []
        health_issues = health_issues or []
        
        # === DATA HEALTH CHECK (per HYDRA_SPEC_LAYERS.md) ===
        if not data_healthy:
            reasons.extend(health_issues)
            return TradabilityStatus.BLOCK, reasons, MarketEnvironment.CHAOTIC, 0.0
        
        # Determine environment
        if cascade_prob > 0.5 or regime == Regime.CASCADE_RISK:
            environment = MarketEnvironment.SQUEEZE
        elif vol_regime == "extreme" or is_abnormal:
            environment = MarketEnvironment.CHAOTIC
        elif vol_regime == "high":
            environment = MarketEnvironment.VOLATILE
        elif regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            environment = MarketEnvironment.TRENDING
        else:
            environment = MarketEnvironment.NORMAL
        
        # === BLOCK CONDITIONS (very strict - only true emergencies) ===
        if danger_score >= 80:
            reasons.append(f"Danger score critical: {danger_score:.0f}/100")
            return TradingDecision.BLOCK, reasons, environment, 0.0
        
        if cascade_prob > 0.8:
            reasons.append(f"Cascade probability extreme: {cascade_prob:.0%}")
            return TradingDecision.BLOCK, reasons, environment, 0.0
        
        if vol_regime == "extreme" and is_abnormal and cascade_prob > 0.5:
            reasons.append("Extreme volatility + abnormal move + cascade risk")
            return TradingDecision.BLOCK, reasons, environment, 0.0
        
        if regime == Regime.CASCADE_RISK and cascade_prob > 0.5:
            reasons.append("CASCADE_RISK regime with elevated cascade probability")
            return TradingDecision.BLOCK, reasons, environment, 0.0
        
        # === RESTRICT CONDITIONS ===
        max_pos = 100.0
        
        if danger_score >= 50:
            reasons.append(f"Elevated danger score: {danger_score:.0f}/100")
            max_pos = min(max_pos, 50.0)
        
        if vol_regime == "extreme":
            reasons.append("Extreme volatility")
            max_pos = min(max_pos, 30.0)
        elif vol_regime == "high":
            reasons.append("High volatility")
            max_pos = min(max_pos, 70.0)
        
        if cascade_prob > 0.3:
            reasons.append(f"Elevated cascade risk: {cascade_prob:.0%}")
            max_pos = min(max_pos, 50.0)
        
        if is_abnormal:
            reasons.append("Abnormal price move detected")
            max_pos = min(max_pos, 50.0)
        
        if funding_pressure > 1.5:
            reasons.append(f"High funding pressure: {funding_pressure:.1f}x")
            max_pos = min(max_pos, 60.0)
        
        if regime_break:
            reasons.append("Regime break detected")
            max_pos = min(max_pos, 70.0)
        
        if max_pos < 100.0:
            return TradingDecision.RESTRICT, reasons, environment, max_pos
        
        # === ALLOW ===
        reasons.append("Normal market conditions")
        return TradingDecision.ALLOW, reasons, environment, 100.0
    
    def _empty_result(self, symbol: str) -> StatisticalResult:
        """Return empty result when insufficient data - BLOCK trading."""
        return StatisticalResult(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            trading_decision=TradingDecision.BLOCK,
            decision_reasons=["Insufficient price data"],
            environment=MarketEnvironment.CHAOTIC,
            max_position_pct=0.0,
            danger_score=100.0,
        )
