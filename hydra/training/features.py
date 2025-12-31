"""
Feature Engineering for ML Models

Implements features as specified in HYDRA_SPEC_ML.md:
- Signal Scorer features
- Regime Classifier features
- Label creation for both models
"""

from __future__ import annotations

from enum import IntEnum
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# REGIME ENUM (Matches spec)
# =============================================================================

class Regime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOLATILITY = 3
    CASCADE_RISK = 4
    SQUEEZE_LONG = 5
    SQUEEZE_SHORT = 6


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering following HYDRA_SPEC_ML.md specifications.
    
    Creates features for:
    1. Signal Scorer model
    2. Regime Classifier model
    """
    
    def __init__(self):
        self._volatility_history = []  # For percentile calculation
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from raw OHLCV + OI + Funding + Liquidation data.
        
        Input DataFrame must have columns:
        - open, high, low, close, volume
        - open_interest
        - funding_rate
        - long_liquidations, short_liquidations, total_liquidations
        
        Returns DataFrame with all engineered features.
        """
        features = pd.DataFrame(index=df.index)
        
        # === PRICE FEATURES ===
        features["return_1m"] = df["close"].pct_change(1)
        features["return_5m"] = df["close"].pct_change(5)
        features["return_15m"] = df["close"].pct_change(15)
        features["return_1h"] = df["close"].pct_change(12)  # 12 x 5min = 1h
        features["return_4h"] = df["close"].pct_change(48)  # 48 x 5min = 4h
        features["return_24h"] = df["close"].pct_change(288)  # 288 x 5min = 24h
        
        # === VOLATILITY FEATURES ===
        features["volatility_5m"] = features["return_1m"].rolling(5).std() * np.sqrt(365 * 24 * 12)
        features["volatility_1h"] = features["return_1m"].rolling(12).std() * np.sqrt(365 * 24 * 12)
        features["volatility_24h"] = features["return_1m"].rolling(288).std() * np.sqrt(365 * 24 * 12)
        
        # Volatility z-score
        vol_mean = features["volatility_1h"].rolling(100).mean()
        vol_std = features["volatility_1h"].rolling(100).std()
        features["volatility_zscore"] = (features["volatility_1h"] - vol_mean) / vol_std.replace(0, 1)
        
        # === TECHNICAL INDICATORS ===
        # SMAs
        features["sma_20"] = df["close"].rolling(20).mean()
        features["sma_50"] = df["close"].rolling(50).mean()
        features["price_vs_sma_20"] = df["close"] / features["sma_20"] - 1
        features["price_vs_sma_50"] = df["close"] / features["sma_50"] - 1
        
        # SMA slopes
        features["sma_20_slope"] = features["sma_20"].pct_change(5)
        features["sma_50_slope"] = features["sma_50"].pct_change(10)
        
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
        features["atr_zscore"] = (features["atr_14"] - features["atr_14"].rolling(100).mean()) / features["atr_14"].rolling(100).std().replace(0, 1)
        
        # ADX (simplified)
        plus_dm = df["high"].diff().where(lambda x: x > 0, 0)
        minus_dm = (-df["low"].diff()).where(lambda x: x > 0, 0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / features["atr_14"].replace(0, 1))
        minus_di = 100 * (minus_dm.rolling(14).mean() / features["atr_14"].replace(0, 1))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        features["adx_14"] = dx.rolling(14).mean()
        
        # Bollinger Bands
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        features["bollinger_width"] = (4 * bb_std) / bb_mid
        features["bb_position"] = (df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std)
        
        # === FUNDING FEATURES ===
        if "funding_rate" in df.columns:
            features["funding_rate"] = df["funding_rate"]
            funding_mean = features["funding_rate"].rolling(100).mean()
            funding_std = features["funding_rate"].rolling(100).std()
            features["funding_zscore"] = (features["funding_rate"] - funding_mean) / funding_std.replace(0, 1)
            features["funding_annualized"] = features["funding_rate"] * 3 * 365 * 100  # Annualized %
            features["funding_momentum"] = features["funding_rate"].diff(3)  # Change over 3 periods
        else:
            features["funding_rate"] = 0.0
            features["funding_zscore"] = 0.0
            features["funding_annualized"] = 0.0
            features["funding_momentum"] = 0.0
        
        # === OI FEATURES ===
        if "open_interest" in df.columns:
            features["oi_delta_pct"] = df["open_interest"].pct_change()
            oi_mean = features["oi_delta_pct"].rolling(100).mean()
            oi_std = features["oi_delta_pct"].rolling(100).std()
            features["oi_zscore"] = (features["oi_delta_pct"] - oi_mean) / oi_std.replace(0, 1)
            
            # OI-Price divergence: OI direction vs price direction
            price_dir = np.sign(features["return_1h"])
            oi_dir = np.sign(features["oi_delta_pct"].rolling(12).sum())
            features["oi_price_divergence"] = price_dir * oi_dir * -1  # -1 when diverging
        else:
            features["oi_delta_pct"] = 0.0
            features["oi_zscore"] = 0.0
            features["oi_price_divergence"] = 0.0
        
        # === LIQUIDATION FEATURES ===
        if "long_liquidations" in df.columns and "short_liquidations" in df.columns:
            long_liq = df["long_liquidations"]
            short_liq = df["short_liquidations"]
            total_liq = long_liq + short_liq
            
            # Imbalance: positive = more longs liquidated
            features["liq_imbalance"] = (long_liq - short_liq) / total_liq.replace(0, 1)
            
            # Velocity: liquidations / OI
            if "open_interest" in df.columns:
                features["liq_velocity"] = total_liq / df["open_interest"].replace(0, 1)
            else:
                features["liq_velocity"] = 0.0
            
            # Liquidation z-score
            liq_mean = total_liq.rolling(100).mean()
            liq_std = total_liq.rolling(100).std()
            features["liq_zscore"] = (total_liq - liq_mean) / liq_std.replace(0, 1)
        else:
            features["liq_imbalance"] = 0.0
            features["liq_velocity"] = 0.0
            features["liq_zscore"] = 0.0
        
        # === VOLUME FEATURES ===
        vol_mean = df["volume"].rolling(100).mean()
        vol_std = df["volume"].rolling(100).std()
        features["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, 1)
        
        # CVD momentum (simplified: using volume * price direction)
        cvd = (df["close"] - df["open"]) / df["open"] * df["volume"]
        features["cvd_momentum"] = cvd.rolling(12).sum()
        
        # === TIME FEATURES ===
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
            days = df.index.dayofweek
            
            features["hour_of_day_sin"] = np.sin(2 * np.pi * hours / 24)
            features["hour_of_day_cos"] = np.cos(2 * np.pi * hours / 24)
            features["day_of_week_sin"] = np.sin(2 * np.pi * days / 7)
            features["day_of_week_cos"] = np.cos(2 * np.pi * days / 7)
            
            # Minutes to next funding (funding at 00:00, 08:00, 16:00 UTC)
            minutes_in_day = hours * 60 + df.index.minute
            funding_times = [0, 480, 960]  # 00:00, 08:00, 16:00
            minutes_to_funding = []
            for m in minutes_in_day:
                next_fundings = [(f - m) % 1440 for f in funding_times]
                minutes_to_funding.append(min(next_fundings))
            features["minutes_to_funding"] = minutes_to_funding
        else:
            features["hour_of_day_sin"] = 0.0
            features["hour_of_day_cos"] = 0.0
            features["day_of_week_sin"] = 0.0
            features["day_of_week_cos"] = 0.0
            features["minutes_to_funding"] = 0.0
        
        # === CASCADE PROBABILITY (heuristic) ===
        # Based on: high liq velocity + high volatility + extreme funding
        liq_factor = features["liq_velocity"].clip(0, 0.1) * 10
        vol_factor = features["volatility_zscore"].clip(-3, 3) / 3
        funding_factor = features["funding_zscore"].abs().clip(0, 3) / 3
        features["cascade_probability"] = (liq_factor * 0.4 + vol_factor * 0.3 + funding_factor * 0.3).clip(0, 1)
        
        # Add close price for reference
        features["close"] = df["close"]
        
        return features
    
    def get_signal_scorer_features(self, features_df: pd.DataFrame) -> list[str]:
        """Get list of features for Signal Scorer model."""
        return [
            # Price features
            "return_1m", "return_5m", "return_15m", "return_1h",
            "volatility_5m", "volatility_1h",
            "price_vs_sma_20", "price_vs_sma_50",
            "rsi_14", "atr_14",
            
            # Funding features
            "funding_rate", "funding_zscore", "funding_annualized", "funding_momentum",
            
            # OI features
            "oi_delta_pct", "oi_zscore", "oi_price_divergence",
            
            # Liquidation features
            "liq_imbalance", "liq_velocity", "liq_zscore",
            
            # Volume
            "volume_zscore",
            
            # Time features
            "hour_of_day_sin", "hour_of_day_cos",
            "day_of_week_sin", "day_of_week_cos",
            "minutes_to_funding",
            
            # Risk
            "cascade_probability",
        ]
    
    def get_regime_classifier_features(self, features_df: pd.DataFrame) -> list[str]:
        """Get list of features for Regime Classifier model."""
        return [
            # Trend features
            "return_1h", "return_4h", "return_24h",
            "sma_20_slope", "sma_50_slope",
            "price_vs_sma_20", "price_vs_sma_50",
            "adx_14",
            
            # Volatility features
            "volatility_5m", "volatility_1h", "volatility_24h",
            "volatility_zscore",
            "atr_14", "atr_zscore",
            "bollinger_width",
            
            # Funding/Positioning
            "funding_rate", "funding_zscore",
            "oi_delta_pct", "oi_zscore",
            
            # Liquidation
            "liq_velocity", "liq_imbalance", "cascade_probability",
            
            # Volume
            "volume_zscore", "cvd_momentum",
        ]


# =============================================================================
# LABEL CREATION
# =============================================================================

class LabelCreator:
    """Creates labels for ML models as specified in HYDRA_SPEC_ML.md."""
    
    def __init__(self, lookahead_minutes: int = 30, transaction_cost: float = 0.001):
        """
        Args:
            lookahead_minutes: How far ahead to look for profit calculation
            transaction_cost: Round-trip transaction cost (default 0.1%)
        """
        self.lookahead_minutes = lookahead_minutes
        self.transaction_cost = transaction_cost
    
    def create_signal_scorer_labels(
        self,
        features_df: pd.DataFrame,
        signal_side: np.ndarray,  # 1=LONG, -1=SHORT
    ) -> pd.Series:
        """
        Create labels for Signal Scorer.
        
        Label = 1 if trade would have been profitable after fees.
        
        Args:
            features_df: DataFrame with 'close' column
            signal_side: Array of signal directions (1=LONG, -1=SHORT)
        
        Returns:
            Series of binary labels
        """
        close = features_df["close"].values
        labels = np.zeros(len(close))
        
        # Lookahead in candles (assuming 5min candles)
        lookahead = self.lookahead_minutes // 5
        
        for i in range(len(close) - lookahead):
            entry_price = close[i]
            exit_price = close[i + lookahead]
            
            if signal_side[i] == 1:  # LONG
                return_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                return_pct = (entry_price - exit_price) / entry_price
            
            # Profitable after fees?
            net_return = return_pct - self.transaction_cost
            labels[i] = 1 if net_return > 0 else 0
        
        return pd.Series(labels, index=features_df.index)
    
    def create_regime_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Create labels for Regime Classifier.
        
        Labels based on spec:
        - CASCADE_RISK if liq_velocity > 0.05
        - SQUEEZE_SHORT if funding > 0.001 and oi_delta > 0.02 and returns < 0
        - SQUEEZE_LONG if funding < -0.001 and oi_delta > 0.02 and returns > 0
        - HIGH_VOLATILITY if volatility > 95th percentile
        - TRENDING_UP if trend > 0.02
        - TRENDING_DOWN if trend < -0.02
        - RANGING otherwise
        """
        labels = np.full(len(features_df), Regime.RANGING.value)
        
        # Get required features
        liq_velocity = features_df.get("liq_velocity", pd.Series(0, index=features_df.index)).values
        funding = features_df.get("funding_rate", pd.Series(0, index=features_df.index)).values
        oi_delta = features_df.get("oi_delta_pct", pd.Series(0, index=features_df.index)).values
        returns_1h = features_df.get("return_1h", pd.Series(0, index=features_df.index)).values
        volatility = features_df.get("volatility_1h", pd.Series(0, index=features_df.index)).values
        
        # Calculate cumulative returns for trend (sum of last 12 5-min returns)
        cumulative_returns = pd.Series(returns_1h).rolling(12).sum().fillna(0).values
        
        # Calculate volatility percentile
        vol_series = pd.Series(volatility)
        vol_95th = vol_series.rolling(500, min_periods=100).quantile(0.95).fillna(vol_series.median()).values
        
        for i in range(len(features_df)):
            # Priority order as per spec
            
            # 1. Cascade risk
            if liq_velocity[i] > 0.05:
                labels[i] = Regime.CASCADE_RISK.value
                continue
            
            # 2. Squeeze detection
            if funding[i] > 0.001 and oi_delta[i] > 0.02 and returns_1h[i] < 0:
                labels[i] = Regime.SQUEEZE_SHORT.value
                continue
            
            if funding[i] < -0.001 and oi_delta[i] > 0.02 and returns_1h[i] > 0:
                labels[i] = Regime.SQUEEZE_LONG.value
                continue
            
            # 3. High volatility
            if volatility[i] > vol_95th[i]:
                labels[i] = Regime.HIGH_VOLATILITY.value
                continue
            
            # 4. Trending
            if cumulative_returns[i] > 0.02:
                labels[i] = Regime.TRENDING_UP.value
                continue
            
            if cumulative_returns[i] < -0.02:
                labels[i] = Regime.TRENDING_DOWN.value
                continue
            
            # 5. Default: Ranging
            labels[i] = Regime.RANGING.value
        
        return pd.Series(labels, index=features_df.index)
    
    def create_direction_labels(
        self,
        features_df: pd.DataFrame,
        horizon_minutes: int = 30,
    ) -> pd.Series:
        """
        Create direction prediction labels.
        
        Label = 1 if price higher in horizon_minutes, else 0.
        """
        close = features_df["close"].values
        horizon = horizon_minutes // 5  # Convert to candles
        
        labels = np.zeros(len(close))
        
        for i in range(len(close) - horizon):
            labels[i] = 1 if close[i + horizon] > close[i] else 0
        
        return pd.Series(labels, index=features_df.index)


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_training_data(
    raw_df: pd.DataFrame,
    model_type: str = "regime",  # "regime" or "signal_scorer"
    lookahead_minutes: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data for a specific model.
    
    Args:
        raw_df: Raw DataFrame with OHLCV + OI + funding + liquidations
        model_type: "regime" or "signal_scorer"
        lookahead_minutes: Lookahead for label creation
    
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    engineer = FeatureEngineer()
    label_creator = LabelCreator(lookahead_minutes=lookahead_minutes)
    
    # Engineer features
    features_df = engineer.engineer_all_features(raw_df)
    
    # Get feature columns for model
    if model_type == "regime":
        feature_cols = engineer.get_regime_classifier_features(features_df)
        labels = label_creator.create_regime_labels(features_df)
    elif model_type == "signal_scorer":
        feature_cols = engineer.get_signal_scorer_features(features_df)
        # Generate synthetic signals for training (random for now, will be improved)
        signal_side = np.random.choice([1, -1], size=len(features_df))
        labels = label_creator.create_signal_scorer_labels(features_df, signal_side)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Select only the required features
    X = features_df[feature_cols].copy()
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | labels.isna())
    X = X[valid_mask]
    labels = labels[valid_mask]
    
    # Also exclude last lookahead rows (no labels)
    lookahead = lookahead_minutes // 5
    X = X.iloc[:-lookahead]
    labels = labels.iloc[:-lookahead]
    
    logger.info(f"Prepared {len(X)} samples for {model_type} model")
    logger.info(f"Features: {len(feature_cols)}")
    
    if model_type == "regime":
        # Log class distribution
        class_counts = labels.value_counts().sort_index()
        for regime_val, count in class_counts.items():
            regime_name = Regime(regime_val).name
            logger.info(f"  {regime_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    return X, labels
