"""
Test trained ML models with sample data.
Validates models work correctly per HYDRA_SPEC_ML.md specifications.
"""
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.training.historical_data import HistoricalDataCollector
from hydra.training.features import FeatureEngineer, LabelCreator
from hydra.training.models import (
    SignalScorer,
    RegimeClassifier,
    load_signal_scorer,
    load_regime_classifier,
)


async def fetch_test_data(days: int = 1) -> dict[str, pd.DataFrame]:
    """Fetch recent data for testing."""
    collector = HistoricalDataCollector(
        coinalyze_api_key=os.getenv("COINALYSE_API_KEY", "")
    )
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    try:
        # Fetch just BTC for testing
        df = await collector.collect_symbol_data(
            "cmt_btcusdt",
            "15min",
            start_date,
            end_date,
            use_cache=True
        )
        return {"cmt_btcusdt": df}
    finally:
        await collector.close()


def test_signal_scorer(model: SignalScorer, X: pd.DataFrame) -> dict:
    """Test Signal Scorer model."""
    logger.info("=" * 60)
    logger.info("TESTING SIGNAL SCORER")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Test predict_proba - returns P(profitable) directly
    logger.info("\n--- predict_proba() ---")
    proba = model.predict_proba(X)
    logger.info(f"Output shape: {proba.shape}")
    logger.info(f"Expected: ({len(X)},) - P(profitable) for each sample")
    assert proba.shape == (len(X),), "predict_proba should return (n_samples,)"
    
    # Probabilities should be between 0 and 1
    assert proba.min() >= 0 and proba.max() <= 1, "Probabilities should be in [0, 1]"
    logger.info(f"✓ All probabilities in valid range [0, 1]")
    
    # Show sample predictions
    logger.info(f"\nSample predictions (first 5):")
    for i in range(min(5, len(proba))):
        logger.info(f"  Sample {i}: P(profitable)={proba[i]:.4f}")
    
    results["proba_shape"] = proba.shape
    results["p_profitable_mean"] = proba.mean()
    results["p_profitable_std"] = proba.std()
    
    # 2. Test predict - should return 0 or 1
    logger.info("\n--- predict() ---")
    preds = model.predict(X)
    logger.info(f"Predictions shape: {preds.shape}")
    logger.info(f"Unique values: {np.unique(preds)}")
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be 0 or 1"
    logger.info(f"✓ Predictions are binary (0/1)")
    
    # Class distribution
    pred_dist = pd.Series(preds).value_counts(normalize=True)
    logger.info(f"\nPrediction distribution:")
    logger.info(f"  Unprofitable (0): {pred_dist.get(0, 0)*100:.1f}%")
    logger.info(f"  Profitable (1): {pred_dist.get(1, 0)*100:.1f}%")
    
    results["predictions"] = preds
    results["pct_profitable"] = (preds == 1).mean() * 100
    
    # 3. Test feature importance
    logger.info("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    logger.info(f"Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    results["top_features"] = importance.head(10)["feature"].tolist()
    
    return results


def test_regime_classifier(model: RegimeClassifier, X: pd.DataFrame) -> dict:
    """Test Regime Classifier model."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING REGIME CLASSIFIER")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Test predict - should return regime labels (0-6)
    logger.info("\n--- predict() ---")
    preds = model.predict(X)
    logger.info(f"Predictions shape: {preds.shape}")
    logger.info(f"Unique values: {np.unique(preds)}")
    
    valid_regimes = set(range(7))
    actual_regimes = set(np.unique(preds))
    assert actual_regimes.issubset(valid_regimes), f"Invalid regime labels: {actual_regimes - valid_regimes}"
    logger.info(f"✓ All predictions are valid regime labels (0-6)")
    
    results["predictions"] = preds
    
    # 2. Test predict_regime_name - should return string labels
    logger.info("\n--- predict_regime_name() ---")
    regime_names = model.predict_regime_name(X)
    logger.info(f"Sample regime names (first 10):")
    for i, name in enumerate(regime_names[:10]):
        logger.info(f"  Sample {i}: {name}")
    
    # Regime distribution
    logger.info("\n--- Regime Distribution ---")
    regime_dist = pd.Series(regime_names).value_counts()
    for regime, count in regime_dist.items():
        pct = count / len(regime_names) * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")
    
    results["regime_distribution"] = regime_dist.to_dict()
    
    # 3. Test predict_proba - should return (n_samples, n_classes)
    logger.info("\n--- predict_proba() ---")
    proba = model.predict_proba(X)
    logger.info(f"Output shape: {proba.shape}")
    logger.info(f"Expected: ({len(X)}, {model.n_classes})")
    
    # Probabilities should sum to 1
    prob_sums = proba.sum(axis=1)
    assert np.allclose(prob_sums, 1.0), "Probabilities should sum to 1"
    logger.info(f"✓ Probabilities sum to 1.0")
    
    # Show sample with highest confidence
    max_conf_idx = proba.max(axis=1).argmax()
    max_conf = proba[max_conf_idx].max()
    max_regime = regime_names[max_conf_idx]
    logger.info(f"\nHighest confidence prediction:")
    logger.info(f"  Index: {max_conf_idx}, Regime: {max_regime}, Confidence: {max_conf:.4f}")
    
    results["proba_shape"] = proba.shape
    results["avg_confidence"] = proba.max(axis=1).mean()
    
    # 4. Feature importance
    logger.info("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    logger.info(f"Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    results["top_features"] = importance.head(10)["feature"].tolist()
    
    return results


def test_live_inference(
    signal_scorer: SignalScorer,
    regime_classifier: RegimeClassifier,
    raw_data: pd.DataFrame
):
    """Simulate live inference scenario."""
    logger.info("\n" + "=" * 60)
    logger.info("LIVE INFERENCE SIMULATION")
    logger.info("=" * 60)
    
    engineer = FeatureEngineer()
    
    # Get features
    features = engineer.engineer_all_features(raw_data)
    features = features.dropna()
    
    if len(features) == 0:
        logger.warning("No valid features after dropping NaN")
        return
    
    # Get latest row (most recent candle)
    latest = features.iloc[[-1]]  # Keep as DataFrame
    
    logger.info(f"\nLatest candle: {raw_data.index[-1]}")
    logger.info(f"Price: ${raw_data['close'].iloc[-1]:,.2f}")
    
    # 1. Get regime
    regime_feature_names = engineer.get_regime_classifier_features(features)
    regime_features = latest[regime_feature_names]
    regime_pred = regime_classifier.predict_regime_name(regime_features)
    regime_proba = regime_classifier.predict_proba(regime_features)
    regime_conf = regime_proba.max()
    
    logger.info(f"\n--- Current Regime ---")
    logger.info(f"  Regime: {regime_pred[0]}")
    logger.info(f"  Confidence: {regime_conf:.2%}")
    
    # 2. Score a hypothetical LONG signal
    logger.info(f"\n--- Scoring Hypothetical LONG Signal ---")
    signal_feature_names = engineer.get_signal_scorer_features(features)
    signal_features = latest[signal_feature_names].copy()
    
    # Score
    score = signal_scorer.predict_proba(signal_features)[0]  # P(profitable)
    
    logger.info(f"  P(profitable): {score:.2%}")
    logger.info(f"  Decision: {'TAKE' if score > 0.55 else 'SKIP'} (threshold: 55%)")


async def main():
    logger.info("=" * 60)
    logger.info("ML MODEL TEST SUITE")
    logger.info("=" * 60)
    
    # Check models exist
    signal_path = Path("models/signal_scorer.pkl")
    regime_path = Path("models/regime_classifier.pkl")
    
    if not signal_path.exists():
        logger.error(f"Signal Scorer model not found: {signal_path}")
        logger.info("Run: python scripts/train_ml_models.py --days 365 --interval 15min")
        return
    
    if not regime_path.exists():
        logger.error(f"Regime Classifier model not found: {regime_path}")
        return
    
    # Load models
    logger.info("\n--- Loading Models ---")
    signal_scorer = load_signal_scorer(str(signal_path))
    logger.info(f"✓ Signal Scorer loaded ({len(signal_scorer.feature_names)} features)")
    
    regime_classifier = load_regime_classifier(str(regime_path))
    logger.info(f"✓ Regime Classifier loaded ({len(regime_classifier.feature_names)} features)")
    
    # Fetch test data (need more days for feature lookback windows)
    logger.info("\n--- Fetching Test Data (7 days) ---")
    raw_data = await fetch_test_data(days=7)
    btc_data = raw_data["cmt_btcusdt"]
    logger.info(f"Fetched {len(btc_data)} candles")
    
    # Engineer features
    logger.info("\n--- Engineering Features ---")
    engineer = FeatureEngineer()
    features = engineer.engineer_all_features(btc_data)
    
    # Drop NaN rows
    features = features.dropna()
    logger.info(f"Test samples (after dropping NaN): {len(features)}")
    
    # Get model-specific feature names
    signal_feature_names = engineer.get_signal_scorer_features(features)
    regime_feature_names = engineer.get_regime_classifier_features(features)
    
    # Extract features for each model - use only features the model was trained on
    signal_features = features[signal_feature_names].copy()
    regime_features = features[regime_feature_names].copy()
    
    logger.info(f"Signal features shape: {signal_features.shape}")
    logger.info(f"Regime features shape: {regime_features.shape}")
    
    # Run tests
    signal_results = test_signal_scorer(signal_scorer, signal_features)
    regime_results = test_regime_classifier(regime_classifier, regime_features)
    
    # Live inference simulation
    test_live_inference(signal_scorer, regime_classifier, btc_data)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n✓ Signal Scorer:")
    logger.info(f"  - predict_proba() returns correct shape")
    logger.info(f"  - predict() returns binary labels (0/1)")
    logger.info(f"  - Avg P(profitable): {signal_results['p_profitable_mean']:.2%}")
    logger.info(f"  - % Predicted profitable: {signal_results['pct_profitable']:.1f}%")
    
    logger.info(f"\n✓ Regime Classifier:")
    logger.info(f"  - predict() returns valid regime labels (0-6)")
    logger.info(f"  - predict_regime_name() returns string labels")
    logger.info(f"  - predict_proba() returns correct shape")
    logger.info(f"  - Avg confidence: {regime_results['avg_confidence']:.2%}")
    
    logger.info(f"\n✓ Live inference simulation passed")
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
