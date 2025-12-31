"""
ML Model Training Script for HYDRA

This script:
1. Fetches historical data from Binance (OHLCV) and Coinalyze (OI, funding, liquidations)
2. Engineers features as per HYDRA_SPEC_ML.md
3. Trains Signal Scorer (Model 1) and Regime Classifier (Model 2)
4. Saves trained models for production use

Usage:
    python scripts/train_ml_models.py --days 90 --interval 5min --gpu
    python scripts/train_ml_models.py --test-only  # Test data pipeline only
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from loguru import logger

from hydra.training.historical_data import (
    HistoricalDataCollector,
    PERMITTED_PAIRS,
    collect_training_data,
)
from hydra.training.features import (
    FeatureEngineer,
    LabelCreator,
    Regime,
    prepare_training_data,
)
from hydra.training.models import (
    SignalScorer,
    RegimeClassifier,
    train_signal_scorer,
    train_regime_classifier,
)


# =============================================================================
# DATA COLLECTION
# =============================================================================

async def collect_data(
    symbols: list[str],
    interval: str,
    days_back: int,
    end_date: datetime = None,
) -> dict[str, pd.DataFrame]:
    """Collect historical data for all symbols."""
    end_date = end_date or datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Collecting data for {len(symbols)} symbols")
    logger.info(f"Interval: {interval}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    coinalyze_key = os.getenv("COINALYSE_API_KEY", "")
    if not coinalyze_key or coinalyze_key == "your_coinalyse_api_key":
        logger.warning("COINALYSE_API_KEY not set - OI, funding, liquidation data will be unavailable")
    
    collector = HistoricalDataCollector(coinalyze_api_key=coinalyze_key)
    
    try:
        data = await collector.collect_all_symbols(
            symbols, interval, start_date, end_date, use_cache=True
        )
        return data
    finally:
        await collector.close()


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(
    raw_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Preprocess and combine data from all symbols.
    
    Returns single DataFrame with all symbols' data concatenated.
    """
    all_dfs = []
    
    for symbol, df in raw_data.items():
        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue
        
        # Add symbol column
        df = df.copy()
        df["symbol"] = symbol
        
        all_dfs.append(df)
        logger.info(f"{symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    
    if not all_dfs:
        raise ValueError("No data available for any symbol")
    
    combined = pd.concat(all_dfs, axis=0)
    combined = combined.sort_index()
    
    logger.info(f"Combined dataset: {len(combined)} rows")
    
    return combined


def engineer_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for a single symbol's data."""
    engineer = FeatureEngineer()
    return engineer.engineer_all_features(df)


# =============================================================================
# TRAINING
# =============================================================================

def train_models(
    data: dict[str, pd.DataFrame],
    use_gpu: bool = True,
    train_signal_scorer_model: bool = True,
    train_regime_model: bool = True,
) -> dict:
    """
    Train ML models on the collected data.
    
    Returns dict of training metrics for each model.
    """
    results = {}
    
    # Process each symbol and combine features
    all_features = []
    all_regime_labels = []
    all_signal_labels = []
    
    engineer = FeatureEngineer()
    label_creator = LabelCreator(lookahead_minutes=30)
    
    for symbol, raw_df in data.items():
        if raw_df.empty:
            continue
        
        logger.info(f"Processing {symbol}...")
        
        # Engineer features
        features_df = engineer.engineer_all_features(raw_df)
        
        # Create regime labels
        regime_labels = label_creator.create_regime_labels(features_df)
        
        # Create signal scorer labels (using random signals for initial training)
        # In production, you'd use actual signals from your strategies
        signal_side = np.random.choice([1, -1], size=len(features_df))
        signal_labels = label_creator.create_signal_scorer_labels(features_df, signal_side)
        
        # Add symbol identifier
        features_df["symbol_id"] = hash(symbol) % 1000
        
        all_features.append(features_df)
        all_regime_labels.append(regime_labels)
        all_signal_labels.append(signal_labels)
    
    # Combine all data
    combined_features = pd.concat(all_features, axis=0)
    combined_regime_labels = pd.concat(all_regime_labels, axis=0)
    combined_signal_labels = pd.concat(all_signal_labels, axis=0)
    
    # Align indices
    valid_idx = combined_features.index.intersection(combined_regime_labels.index)
    combined_features = combined_features.loc[valid_idx]
    combined_regime_labels = combined_regime_labels.loc[valid_idx]
    combined_signal_labels = combined_signal_labels.loc[valid_idx]
    
    # Drop NaN rows
    valid_mask = ~combined_features.isna().any(axis=1)
    combined_features = combined_features[valid_mask]
    combined_regime_labels = combined_regime_labels[valid_mask]
    combined_signal_labels = combined_signal_labels[valid_mask]
    
    logger.info(f"Total samples after preprocessing: {len(combined_features)}")
    
    # === TRAIN REGIME CLASSIFIER ===
    if train_regime_model:
        logger.info("\n" + "="*60)
        logger.info("TRAINING REGIME CLASSIFIER")
        logger.info("="*60)
        
        regime_feature_cols = engineer.get_regime_classifier_features(combined_features)
        X_regime = combined_features[regime_feature_cols]
        y_regime = combined_regime_labels
        
        # Remove last 6 rows (30min lookahead / 5min = 6)
        X_regime = X_regime.iloc[:-6]
        y_regime = y_regime.iloc[:-6]
        
        model, metrics = train_regime_classifier(X_regime, y_regime, use_gpu=use_gpu)
        results["regime_classifier"] = metrics
        
        # Evaluate on last 20% as holdout
        split_idx = int(len(X_regime) * 0.8)
        X_test = X_regime.iloc[split_idx:]
        y_test = y_regime.iloc[split_idx:]
        
        eval_metrics = model.evaluate(X_test, y_test)
        logger.info(f"\nHoldout Test Accuracy: {eval_metrics['accuracy']:.4f}")
        logger.info(f"Holdout Test F1: {eval_metrics['f1_weighted']:.4f}")
    
    # === TRAIN SIGNAL SCORER ===
    if train_signal_scorer_model:
        logger.info("\n" + "="*60)
        logger.info("TRAINING SIGNAL SCORER")
        logger.info("="*60)
        
        signal_feature_cols = engineer.get_signal_scorer_features(combined_features)
        X_signal = combined_features[signal_feature_cols]
        y_signal = combined_signal_labels
        
        # Remove last 6 rows
        X_signal = X_signal.iloc[:-6]
        y_signal = y_signal.iloc[:-6]
        
        model, metrics = train_signal_scorer(X_signal, y_signal, use_gpu=use_gpu)
        results["signal_scorer"] = metrics
    
    return results


# =============================================================================
# TEST PIPELINE
# =============================================================================

async def test_data_pipeline(symbols: list[str], interval: str, days: int):
    """Test the data pipeline without training models."""
    logger.info("="*60)
    logger.info("TESTING DATA PIPELINE")
    logger.info("="*60)
    
    # Collect data
    data = await collect_data(symbols[:2], interval, days)  # Just 2 symbols for testing
    
    logger.info("\n--- RAW DATA ---")
    for symbol, df in data.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Check data quality
        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 0:
                logger.info(f"  {col}: {null_pct:.1f}% null")
        
        logger.info(f"\n  Sample data:\n{df.head(3)}")
    
    # Test feature engineering
    logger.info("\n--- FEATURE ENGINEERING ---")
    engineer = FeatureEngineer()
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        features = engineer.engineer_all_features(df)
        logger.info(f"\n{symbol} features:")
        logger.info(f"  Total features: {len(features.columns)}")
        
        # Check for NaN in features
        nan_counts = features.isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            logger.info(f"  Features with NaN:")
            for feat, count in nan_features.head(10).items():
                logger.info(f"    {feat}: {count} ({count/len(features)*100:.1f}%)")
        
        # Sample feature values
        logger.info(f"\n  Sample features:\n{features.head(3)}")
        break  # Just test first symbol
    
    # Test label creation
    logger.info("\n--- LABEL CREATION ---")
    label_creator = LabelCreator(lookahead_minutes=30)
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        features = engineer.engineer_all_features(df)
        
        # Regime labels
        regime_labels = label_creator.create_regime_labels(features)
        logger.info(f"\n{symbol} regime labels:")
        for regime_val in range(7):
            count = (regime_labels == regime_val).sum()
            if count > 0:
                regime_name = Regime(regime_val).name
                logger.info(f"  {regime_name}: {count} ({count/len(regime_labels)*100:.1f}%)")
        
        # Direction labels
        direction_labels = label_creator.create_direction_labels(features, horizon_minutes=30)
        up_pct = direction_labels.mean() * 100
        logger.info(f"\n{symbol} direction labels:")
        logger.info(f"  Up (1): {up_pct:.1f}%")
        logger.info(f"  Down (0): {100-up_pct:.1f}%")
        
        break  # Just test first symbol
    
    logger.info("\n" + "="*60)
    logger.info("DATA PIPELINE TEST COMPLETE")
    logger.info("="*60)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Train HYDRA ML Models")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--interval", type=str, default="5min", help="Data interval")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--test-only", action="store_true", help="Test data pipeline only")
    parser.add_argument("--regime-only", action="store_true", help="Train only regime classifier")
    parser.add_argument("--signal-only", action="store_true", help="Train only signal scorer")
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = PERMITTED_PAIRS
    
    logger.info("="*60)
    logger.info("HYDRA ML MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Days: {args.days}")
    logger.info(f"GPU: {args.gpu}")
    
    # Test mode
    if args.test_only:
        await test_data_pipeline(symbols, args.interval, args.days)
        return
    
    # Collect data
    logger.info("\n--- COLLECTING DATA ---")
    data = await collect_data(symbols, args.interval, args.days)
    
    if not data:
        logger.error("No data collected, aborting")
        return
    
    # Train models
    logger.info("\n--- TRAINING MODELS ---")
    
    train_regime = not args.signal_only
    train_signal = not args.regime_only
    
    results = train_models(
        data,
        use_gpu=args.gpu,
        train_signal_scorer_model=train_signal,
        train_regime_model=train_regime,
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"  {key}: <dict with {len(value)} items>")
            else:
                logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
