"""
ML Models for HYDRA Trading System

Model 1: Signal Scorer (CatBoost with GPU)
Model 2: Regime Classifier (XGBoost with GPU)

Both models support GPU training for faster performance.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# GPU-enabled libraries
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available, Signal Scorer will use fallback")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, Regime Classifier will use fallback")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from hydra.training.features import Regime


# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL 1: SIGNAL SCORER
# =============================================================================

class SignalScorer:
    """
    Signal Scorer Model (Model 1)
    
    Purpose: Score how likely a generated signal will be profitable.
    Architecture: CatBoost Classifier with GPU support
    Output: P(profitable) for each signal
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
    ):
        self.use_gpu = use_gpu and CATBOOST_AVAILABLE
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
    
    def _create_model(self) -> Any:
        """Create the underlying model."""
        if CATBOOST_AVAILABLE:
            task_type = "GPU" if self.use_gpu else "CPU"
            
            return CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                early_stopping_rounds=50,
                task_type=task_type,
                devices="0" if self.use_gpu else None,
                verbose=100,
                eval_metric="AUC",
            )
        else:
            # Fallback to GradientBoosting
            logger.info("Using GradientBoostingClassifier as fallback")
            return GradientBoostingClassifier(
                n_estimators=min(self.iterations, 200),
                learning_rate=self.learning_rate,
                max_depth=self.depth,
                random_state=self.random_seed,
            )
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: list[str] = None,
        n_splits: int = 5,
    ) -> dict:
        """
        Train the Signal Scorer with time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Binary labels (0/1)
            categorical_features: List of categorical feature names
            n_splits: Number of CV folds
        
        Returns:
            Dict of training metrics
        """
        logger.info(f"Training Signal Scorer on {len(X)} samples")
        logger.info(f"Features: {len(X.columns)}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        self.feature_names = list(X.columns)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            logger.info(f"Fold {fold + 1}/{n_splits}: Train={len(X_train)}, Val={len(X_val)}")
            
            fold_model = self._create_model()
            
            if CATBOOST_AVAILABLE:
                fold_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=categorical_features,
                    verbose=False,
                )
            else:
                fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)[:, 1]
            
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
            
            cv_scores.append(accuracy)
            cv_aucs.append(auc)
            
            logger.info(f"  Fold {fold + 1} Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        self.model = self._create_model()
        
        if CATBOOST_AVAILABLE:
            self.model.fit(X, y, cat_features=categorical_features, verbose=100)
        else:
            self.model.fit(X, y)
        
        # Store metrics
        self.training_metrics = {
            "accuracy": np.mean(cv_scores),
            "cv_accuracy_mean": np.mean(cv_scores),
            "cv_accuracy_std": np.std(cv_scores),
            "auc": np.mean(cv_aucs),
            "cv_auc_mean": np.mean(cv_aucs),
            "cv_auc_std": np.std(cv_aucs),
            "precision": 0.0,  # Will be set on final eval
            "recall": 0.0,
            "f1": 0.0,
            "n_samples": len(X),
            "n_features": len(X.columns),
            "trained_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"CV Accuracy: {self.training_metrics['cv_accuracy_mean']:.4f} ± {self.training_metrics['cv_accuracy_std']:.4f}")
        logger.info(f"CV AUC: {self.training_metrics['cv_auc_mean']:.4f} ± {self.training_metrics['cv_auc_std']:.4f}")
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of profitability.
        
        Returns:
            Array of P(profitable) for each sample
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # P(profitable)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model on a dataset.
        
        Returns dict with accuracy, precision, recall, f1, auc.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }
        
        if len(np.unique(y)) > 1:
            metrics["auc"] = roc_auc_score(y, y_proba)
        else:
            metrics["auc"] = 0.5
        
        return metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance as array."""
        if self.model is None:
            return np.array([])
        
        if CATBOOST_AVAILABLE:
            return self.model.get_feature_importance()
        else:
            return self.model.feature_importances_
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        
        importance = self.get_feature_importance()
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False)
    
    def save(self, path: str = None):
        """Save model to disk."""
        path = path or str(MODEL_DIR / "signal_scorer.pkl")
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "config": {
                    "iterations": self.iterations,
                    "learning_rate": self.learning_rate,
                    "depth": self.depth,
                },
            }, f)
        
        logger.info(f"Saved Signal Scorer to {path}")
    
    def load(self, path: str = None):
        """Load model from disk."""
        path = path or str(MODEL_DIR / "signal_scorer.pkl")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.training_metrics = data["training_metrics"]
        
        logger.info(f"Loaded Signal Scorer from {path}")


# =============================================================================
# MODEL 2: REGIME CLASSIFIER
# =============================================================================

class RegimeClassifier:
    """
    Regime Classifier Model (Model 2)
    
    Purpose: Classify current market regime for strategy selection.
    Architecture: XGBoost multi-class classifier with GPU support
    Output: One of 7 regime classes
    """
    
    REGIME_NAMES = [
        "TRENDING_UP",
        "TRENDING_DOWN",
        "RANGING",
        "HIGH_VOLATILITY",
        "CASCADE_RISK",
        "SQUEEZE_LONG",
        "SQUEEZE_SHORT",
    ]
    
    def __init__(
        self,
        use_gpu: bool = True,
        n_estimators: int = 200,
        max_depth: int = 10,
        learning_rate: float = 0.1,
        min_child_weight: int = 50,
        random_seed: int = 42,
    ):
        self.use_gpu = use_gpu and XGBOOST_AVAILABLE
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.random_seed = random_seed
        
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.label_mapping = None  # Maps original labels to contiguous 0..n-1
        self.inverse_mapping = None  # Maps back from contiguous to original
        self.n_classes = len(self.REGIME_NAMES)
    
    def _create_model(self) -> Any:
        """Create the underlying model."""
        if XGBOOST_AVAILABLE:
            n_classes = self.n_classes
            params = {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_child_weight": self.min_child_weight,
                "random_state": self.random_seed,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
            }
            
            if self.use_gpu:
                params["tree_method"] = "hist"
                params["device"] = "cuda"
            
            return xgb.XGBClassifier(**params)
        else:
            # Fallback to RandomForest
            logger.info("Using RandomForestClassifier as fallback")
            return RandomForestClassifier(
                n_estimators=min(self.n_estimators, 200),
                max_depth=self.max_depth,
                min_samples_leaf=self.min_child_weight,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """
        Train the Regime Classifier with time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Regime labels (0-6)
            n_splits: Number of CV folds
        
        Returns:
            Dict of training metrics
        """
        logger.info(f"Training Regime Classifier on {len(X)} samples")
        logger.info(f"Features: {len(X.columns)}")
        
        # Remap labels to contiguous 0..n-1 (XGBoost requirement)
        unique_labels = sorted(y.unique())
        self.label_mapping = {orig: new for new, orig in enumerate(unique_labels)}
        self.inverse_mapping = {new: orig for orig, new in self.label_mapping.items()}
        y_mapped = y.map(self.label_mapping)
        
        # Update num_class based on actual labels
        self.n_classes = len(unique_labels)
        
        # Log class distribution
        class_dist = y.value_counts().sort_index()
        for regime_val, count in class_dist.items():
            regime_name = self.REGIME_NAMES[regime_val] if regime_val < len(self.REGIME_NAMES) else f"UNKNOWN_{regime_val}"
            logger.info(f"  {regime_name}: {count} ({count/len(y)*100:.1f}%)")
        
        self.feature_names = list(X.columns)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
            
            # Skip folds where train set doesn't have all classes (XGBoost requirement)
            train_classes = set(y_train.unique())
            if len(train_classes) < self.n_classes:
                logger.info(f"Fold {fold + 1}/{n_splits}: Skipping (only {len(train_classes)}/{self.n_classes} classes in train)")
                continue
            
            logger.info(f"Fold {fold + 1}/{n_splits}: Train={len(X_train)}, Val={len(X_val)}")
            
            # Use RandomForest for CV (handles missing classes better)
            fold_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="weighted")
            
            cv_scores.append(accuracy)
            cv_f1_scores.append(f1)
            
            logger.info(f"  Fold {fold + 1} Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        self.model = self._create_model()
        
        if XGBOOST_AVAILABLE:
            self.model.fit(X, y_mapped, verbose=True)
        else:
            self.model.fit(X, y_mapped)
        
        # Store metrics
        self.training_metrics = {
            "cv_accuracy_mean": np.mean(cv_scores),
            "cv_accuracy_std": np.std(cv_scores),
            "cv_f1_mean": np.mean(cv_f1_scores),
            "cv_f1_std": np.std(cv_f1_scores),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "class_distribution": class_dist.to_dict(),
            "trained_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"CV Accuracy: {self.training_metrics['cv_accuracy_mean']:.4f} ± {self.training_metrics['cv_accuracy_std']:.4f}")
        logger.info(f"CV F1: {self.training_metrics['cv_f1_mean']:.4f} ± {self.training_metrics['cv_f1_std']:.4f}")
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability distribution over regimes.
        
        Returns:
            Array of shape (n_samples, 7) with regime probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime labels (mapped back to original labels)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        preds = self.model.predict(X)
        
        # Map back to original labels if mapping exists
        if self.inverse_mapping is not None:
            preds = np.array([self.inverse_mapping.get(p, p) for p in preds])
        
        return preds
    
    def predict_regime_name(self, X: pd.DataFrame) -> list[str]:
        """Predict regime names (string labels)."""
        predictions = self.predict(X)
        return [self.REGIME_NAMES[p] if p < len(self.REGIME_NAMES) else f"UNKNOWN_{p}" for p in predictions]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        
        if XGBOOST_AVAILABLE:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model on test data."""
        y_pred = self.predict(X)
        
        # Ensure both are numpy arrays for comparison
        y_true = np.array(y)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # Per-class metrics - use only labels present in the data
        unique_labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        present_names = [self.REGIME_NAMES[int(i)] if int(i) < len(self.REGIME_NAMES) else f"UNKNOWN_{i}" for i in unique_labels]
        
        try:
            report = classification_report(
                y, y_pred, 
                labels=unique_labels,
                target_names=present_names, 
                output_dict=True, 
                zero_division=0
            )
            metrics["per_class"] = report
        except Exception:
            metrics["per_class"] = {}
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    def save(self, path: str = None):
        """Save model to disk."""
        path = path or str(MODEL_DIR / "regime_classifier.pkl")
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "label_mapping": self.label_mapping,
                "inverse_mapping": self.inverse_mapping,
                "n_classes": self.n_classes,
                "config": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                },
            }, f)
        
        logger.info(f"Saved Regime Classifier to {path}")
    
    def load(self, path: str = None):
        """Load model from disk."""
        path = path or str(MODEL_DIR / "regime_classifier.pkl")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.training_metrics = data["training_metrics"]
        self.label_mapping = data.get("label_mapping")
        self.inverse_mapping = data.get("inverse_mapping")
        self.n_classes = data.get("n_classes", len(self.REGIME_NAMES))
        
        logger.info(f"Loaded Regime Classifier from {path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_signal_scorer(
    X: pd.DataFrame,
    y: pd.Series,
    use_gpu: bool = True,
    save_model: bool = True,
) -> Tuple[SignalScorer, dict]:
    """
    Train Signal Scorer model.
    
    Returns:
        Tuple of (model, training_metrics)
    """
    model = SignalScorer(use_gpu=use_gpu)
    metrics = model.train(X, y)
    
    if save_model:
        model.save()
    
    # Log feature importance
    importance = model.get_feature_importance()
    logger.info("Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


def train_regime_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    use_gpu: bool = True,
    save_model: bool = True,
) -> Tuple[RegimeClassifier, dict]:
    """
    Train Regime Classifier model.
    
    Returns:
        Tuple of (model, training_metrics)
    """
    model = RegimeClassifier(use_gpu=use_gpu)
    metrics = model.train(X, y)
    
    if save_model:
        model.save()
    
    # Log feature importance
    importance = model.get_feature_importance()
    logger.info("Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


def load_signal_scorer(path: str = None) -> SignalScorer:
    """Load a trained Signal Scorer model from disk."""
    model = SignalScorer(use_gpu=False)  # GPU not needed for inference
    model.load(path)
    return model


def load_regime_classifier(path: str = None) -> RegimeClassifier:
    """Load a trained Regime Classifier model from disk."""
    model = RegimeClassifier(use_gpu=False)  # GPU not needed for inference
    model.load(path)
    return model
