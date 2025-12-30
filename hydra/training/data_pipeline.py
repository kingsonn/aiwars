"""
Data Pipeline for HYDRA Training

Handles:
- Historical data fetching
- Feature engineering
- Dataset creation for different model types
- Train/validation/test splits with regime awareness
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Generator
import numpy as np
import pandas as pd
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import OHLCV, Regime


@dataclass
class TrainingExample:
    """Single training example for transformer."""
    # Features (sequence)
    price_features: np.ndarray  # (seq_len, 5)
    funding_features: np.ndarray  # (seq_len, 3)
    oi_features: np.ndarray  # (seq_len, 3)
    orderbook_features: np.ndarray  # (seq_len, 4)
    liq_features: np.ndarray  # (seq_len, 4)
    vol_features: np.ndarray  # (seq_len, 3)
    
    # Labels
    direction_label: int  # 0=long, 1=short, 2=flat
    adverse_excursion: float
    favorable_excursion: float
    regime_label: int
    volatility_1h: float
    
    # New: Cost-aware labels for profitability
    profit_potential: float = 0.0  # favorable - adverse - costs
    risk_adjusted_return: float = 0.0  # return / volatility
    raw_return: float = 0.0  # actual future return
    
    # Metadata
    timestamp: datetime = None
    symbol: str = ""


@dataclass
class RLTrajectory:
    """Trajectory for RL training."""
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    next_states: list[np.ndarray]
    dones: list[bool]
    
    # Metadata
    total_pnl: float
    max_drawdown: float
    regime: Regime


class FeatureEngineer:
    """Feature engineering for HYDRA models."""
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.sequence_length = config.model.transformer_sequence_length
        self._btc_data: Optional[pd.DataFrame] = None  # Cache for BTC lead signal
    
    def set_btc_reference(self, btc_df: pd.DataFrame) -> None:
        """Set BTC data for computing lead signals on alt pairs."""
        self._btc_data = btc_df.copy()
        # Pre-compute BTC momentum features
        self._btc_data['btc_return_5m'] = self._btc_data['close'].pct_change()
        self._btc_data['btc_return_15m'] = self._btc_data['close'].pct_change(3)
        self._btc_data['btc_return_1h'] = self._btc_data['close'].pct_change(12)
        self._btc_data['btc_vol'] = self._btc_data['btc_return_5m'].rolling(20).std()
    
    def compute_price_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute normalized price features."""
        # Normalize by first close in sequence
        base = df['close'].iloc[0]
        
        features = np.column_stack([
            df['open'].values / base,
            df['high'].values / base,
            df['low'].values / base,
            df['close'].values / base,
            np.log1p(df['volume'].values),
        ])
        
        return features
    
    def compute_btc_lead_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """
        Compute BTC lead signal features for alt pairs.
        
        Key insight: BTC often leads alts by 5-30 minutes.
        If BTC just moved +2%, alts will likely follow.
        """
        # If this IS BTC, return zeros (no lead signal for BTC itself)
        if 'btc' in symbol.lower():
            return np.zeros((len(df), 4))
        
        # If no BTC reference data, return zeros
        if self._btc_data is None:
            return np.zeros((len(df), 4))
        
        # Align BTC data with current dataframe by timestamp
        try:
            aligned = self._btc_data.reindex(df.index, method='ffill')
            
            features = np.column_stack([
                aligned['btc_return_5m'].fillna(0).values * 100,   # Recent BTC move
                aligned['btc_return_15m'].fillna(0).values * 100,  # 15m BTC trend
                aligned['btc_return_1h'].fillna(0).values * 100,   # 1h BTC trend
                aligned['btc_vol'].fillna(0.01).values * 100,      # BTC volatility
            ])
            return features
        except Exception:
            return np.zeros((len(df), 4))
    
    def compute_funding_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute funding rate features."""
        if 'funding_rate' not in df.columns:
            return np.zeros((len(df), 3))
        
        features = np.column_stack([
            df['funding_rate'].values * 10000,  # bps
            df['funding_rate'].rolling(8).mean().fillna(0).values * 10000,
            df['funding_rate'].diff().fillna(0).values * 10000,
        ])
        
        return features
    
    def compute_oi_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute open interest features."""
        if 'open_interest' not in df.columns:
            return np.zeros((len(df), 3))
        
        oi = df['open_interest'].values
        oi_norm = oi / oi[0] if oi[0] > 0 else oi
        
        features = np.column_stack([
            oi_norm,
            np.diff(oi, prepend=oi[0]) / np.maximum(oi, 1) * 100,
            pd.Series(oi).pct_change().fillna(0).values * 100,
        ])
        
        return features
    
    def compute_volatility_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute volatility features."""
        returns = df['close'].pct_change().fillna(0)
        
        # Rolling volatility
        vol_20 = returns.rolling(20).std().fillna(0) * np.sqrt(1440 * 365)
        vol_100 = returns.rolling(100).std().fillna(0) * np.sqrt(1440 * 365)
        
        # Vol of vol
        vol_of_vol = vol_20.rolling(20).std().fillna(0)
        
        features = np.column_stack([
            vol_20.values,
            (vol_20 / vol_100.replace(0, 1)).values,  # Vol ratio
            vol_of_vol.values,
        ])
        
        return features
    
    def compute_labels(
        self, 
        df: pd.DataFrame, 
        horizon: int = 12,  # 1 hour for 5m data
        threshold: float = 0.003,  # Lower threshold: 0.3% (was 0.5%)
        transaction_cost: float = 0.0015,  # 0.15% round-trip (fees + slippage)
    ) -> dict:
        """
        Compute training labels with cost-awareness.
        
        Key improvements:
        1. Cost-adjusted threshold (must exceed transaction costs)
        2. Risk-adjusted return labels
        3. Optimal entry detection (not just direction)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Future returns (raw)
        future_returns = np.zeros(len(df))
        for i in range(len(df) - horizon):
            future_returns[i] = (close[i + horizon] - close[i]) / close[i]
        
        # Cost-adjusted returns (what we actually keep after fees)
        cost_adjusted_returns = future_returns - transaction_cost
        
        # Realized volatility for risk adjustment
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        realized_vol = pd.Series(returns).rolling(20).std().fillna(0.01).values
        
        # Risk-adjusted return (Sharpe-like)
        risk_adjusted_returns = cost_adjusted_returns / np.maximum(realized_vol, 0.001)
        
        # Direction labels with cost-aware threshold
        # Only signal if expected profit > transaction cost
        min_profitable_move = threshold + transaction_cost  # ~0.45%
        
        direction = np.where(
            future_returns > min_profitable_move, 0,  # Long (profitable after costs)
            np.where(future_returns < -min_profitable_move, 1, 2)  # Short or Flat
        )
        
        # Adverse and favorable excursion (for stop-loss/take-profit)
        adverse = np.zeros(len(df))
        favorable = np.zeros(len(df))
        
        for i in range(len(df) - horizon):
            future_high = high[i:i+horizon].max()
            future_low = low[i:i+horizon].min()
            
            if future_returns[i] > 0:
                # Long position
                adverse[i] = (close[i] - future_low) / close[i]
                favorable[i] = (future_high - close[i]) / close[i]
            else:
                # Short position
                adverse[i] = (future_high - close[i]) / close[i]
                favorable[i] = (close[i] - future_low) / close[i]
        
        # Volatility label (annualized)
        vol_1h = pd.Series(returns).rolling(12).std().fillna(0).values * np.sqrt(12 * 24 * 365)
        
        # Profit potential: favorable - adverse - costs
        profit_potential = favorable - adverse - transaction_cost
        
        return {
            'direction': direction,
            'adverse_excursion': adverse,
            'favorable_excursion': favorable,
            'volatility_1h': vol_1h,
            'risk_adjusted_return': risk_adjusted_returns,
            'profit_potential': profit_potential,
            'raw_return': future_returns,
        }
    
    def detect_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Detect regime for each row."""
        regimes = np.zeros(len(df), dtype=int)
        
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std()
        vol_zscore = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()
        
        # Trend detection
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        trend = (sma_20 - sma_50) / sma_50
        
        for i in range(100, len(df)):
            if vol_zscore.iloc[i] > 2:
                if abs(returns.iloc[i-5:i].sum()) > 0.05:
                    regimes[i] = 4  # CASCADE_RISK
                else:
                    regimes[i] = 3  # HIGH_VOLATILITY
            elif trend.iloc[i] > 0.02:
                regimes[i] = 0  # TRENDING_UP
            elif trend.iloc[i] < -0.02:
                regimes[i] = 1  # TRENDING_DOWN
            else:
                regimes[i] = 2  # RANGING
        
        return regimes


class DataPipeline:
    """
    Data pipeline for HYDRA training.
    
    Handles data fetching, preprocessing, and dataset creation.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self._cache_dir = Path("./data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100000,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        import ccxt.async_support as ccxt
        
        exchange = ccxt.binanceusdm({'options': {'defaultType': 'future'}})
        
        try:
            await exchange.load_markets()
            
            all_data = []
            since = int(start_date.timestamp() * 1000) if start_date else None
            
            while len(all_data) < limit:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if end_date and ohlcv[-1][0] > end_date.timestamp() * 1000:
                    break
                
                await asyncio.sleep(0.1)
            
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        finally:
            await exchange.close()
    
    async def fetch_funding_history(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical funding rates."""
        import ccxt.async_support as ccxt
        
        exchange = ccxt.binanceusdm({'options': {'defaultType': 'future'}})
        
        try:
            await exchange.load_markets()
            
            # Binance specific
            params = {'limit': limit}
            if start_date:
                params['startTime'] = int(start_date.timestamp() * 1000)
            
            funding = await exchange.fetch_funding_rate_history(symbol, **params)
            
            df = pd.DataFrame([
                {
                    'timestamp': f['timestamp'],
                    'funding_rate': f['fundingRate'],
                }
                for f in funding
            ])
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            return df
            
        finally:
            await exchange.close()
    
    def prepare_transformer_dataset(
        self,
        df: pd.DataFrame,
        shuffle: bool = True,
    ) -> Generator[TrainingExample, None, None]:
        """Prepare dataset for transformer training."""
        seq_len = self.config.model.transformer_sequence_length
        
        # Compute features
        price_feat = self.feature_engineer.compute_price_features(df)
        funding_feat = self.feature_engineer.compute_funding_features(df)
        oi_feat = self.feature_engineer.compute_oi_features(df)
        vol_feat = self.feature_engineer.compute_volatility_features(df)
        
        # Placeholder features
        orderbook_feat = np.zeros((len(df), 4))
        liq_feat = np.zeros((len(df), 4))
        
        # Compute labels
        labels = self.feature_engineer.compute_labels(df)
        regimes = self.feature_engineer.detect_regime(df)
        
        # Generate examples
        indices = list(range(seq_len, len(df) - 12))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in indices:
            start = i - seq_len
            
            yield TrainingExample(
                price_features=price_feat[start:i],
                funding_features=funding_feat[start:i],
                oi_features=oi_feat[start:i],
                orderbook_features=orderbook_feat[start:i],
                liq_features=liq_feat[start:i],
                vol_features=vol_feat[start:i],
                direction_label=labels['direction'][i],
                adverse_excursion=labels['adverse_excursion'][i],
                favorable_excursion=labels['favorable_excursion'][i],
                regime_label=regimes[i],
                volatility_1h=labels['volatility_1h'][i],
                profit_potential=labels['profit_potential'][i],
                risk_adjusted_return=labels['risk_adjusted_return'][i],
                raw_return=labels['raw_return'][i],
                timestamp=df.index[i],
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
            )
    
    def create_regime_splits(
        self,
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Split data by regime for regime-aware training."""
        regimes = self.feature_engineer.detect_regime(df)
        
        splits = {}
        regime_names = ['trending_up', 'trending_down', 'ranging', 'high_vol', 'cascade']
        
        for i, name in enumerate(regime_names):
            mask = regimes == i
            if mask.sum() > 0:
                splits[name] = df[mask].copy()
        
        return splits
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        n_folds: int = 5,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """Generate walk-forward validation splits."""
        total_len = len(df)
        fold_size = total_len // n_folds
        
        for fold in range(n_folds):
            # Expanding window
            train_end = int((fold + 1) * fold_size * train_pct)
            val_end = train_end + int(fold_size * val_pct)
            test_end = min(val_end + int(fold_size * (1 - train_pct - val_pct)), total_len)
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:test_end]
            
            if len(train_df) > 100 and len(val_df) > 10 and len(test_df) > 10:
                yield train_df, val_df, test_df
    
    def save_dataset(self, df: pd.DataFrame, name: str) -> Path:
        """Save dataset to cache."""
        path = self._cache_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved dataset to {path}")
        return path
    
    def load_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Load dataset from cache."""
        path = self._cache_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
