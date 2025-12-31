#!/usr/bin/env python3
"""
Train Signal Scorer Model

Uses comprehensive feature engineering from HYDRA_SPEC_ML.md:
- Signal features (direction, confidence, source, expected values)
- Price features (returns, volatility, RSI, ATR, SMAs)
- Funding features (rate, zscore, annualized, momentum)
- OI features (delta, zscore, divergence)
- Liquidation features (imbalance, velocity, zscore)
- Order book features (imbalance, spread, depths)
- Positioning features (L/S ratio, taker ratio)
- Regime features (encoded)
- Time features (cyclical encoding)

Usage:
    python scripts/train_signal_scorer.py
    python scripts/train_signal_scorer.py --days 180
    python scripts/train_signal_scorer.py --symbols cmt_btcusdt,cmt_ethusdt
"""

import argparse
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.training.historical_data import (
    collect_training_data, HistoricalDataCollector, PERMITTED_PAIRS
)
from hydra.training.signal_scorer_data import (
    create_signal_scorer_dataset, get_signal_scorer_feature_names
)
from hydra.training.models import SignalScorer

console = Console()


async def fetch_data(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Fetch historical data for training."""
    console.print(f"\n[cyan]Fetching {days} days of data for {len(symbols)} symbols...[/cyan]")
    
    data = await collect_training_data(
        symbols=symbols,
        interval="5min",
        days_back=days,
    )
    
    console.print(f"[green]✓ Fetched data for {len(data)} symbols[/green]")
    
    for symbol, df in data.items():
        console.print(f"  {symbol}: {len(df):,} candles")
        
        # Check data quality
        missing_cols = []
        for col in ["open_interest", "funding_rate", "long_liquidations"]:
            if col not in df.columns or df[col].isna().all():
                missing_cols.append(col)
        
        if missing_cols:
            console.print(f"    [yellow]Missing: {', '.join(missing_cols)}[/yellow]")
    
    return data


def prepare_datasets(
    data: dict[str, pd.DataFrame],
    lookahead_minutes: int = 30,
    test_split: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare training and test datasets from all symbols."""
    console.print(f"\n[cyan]Preparing datasets (lookahead: {lookahead_minutes}min)...[/cyan]")
    
    all_X = []
    all_y = []
    
    for symbol, raw_df in data.items():
        console.print(f"  Processing {symbol}...")
        
        try:
            X, y = create_signal_scorer_dataset(
                raw_df,
                lookahead_minutes=lookahead_minutes,
                transaction_cost=0.001,
                min_confidence=0.5,
            )
            
            if len(X) > 0:
                X["symbol"] = symbol  # Add symbol for tracking
                all_X.append(X)
                all_y.append(y)
                console.print(f"    [green]{len(X):,} samples, win rate: {y.mean()*100:.1f}%[/green]")
            else:
                console.print(f"    [yellow]No samples generated[/yellow]")
                
        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")
            logger.exception(f"Error processing {symbol}")
    
    if not all_X:
        console.print("[red]No training data generated![/red]")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    
    # Combine all data
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    console.print(f"\n[green]Total samples: {len(X_combined):,}[/green]")
    console.print(f"Overall win rate: {y_combined.mean()*100:.1f}%")
    
    # Time-based split (last test_split % for testing)
    split_idx = int(len(X_combined) * (1 - test_split))
    
    X_train = X_combined.iloc[:split_idx].drop(columns=["symbol"])
    y_train = y_combined.iloc[:split_idx]
    X_test = X_combined.iloc[split_idx:].drop(columns=["symbol"])
    y_test = y_combined.iloc[split_idx:]
    
    console.print(f"Train: {len(X_train):,} samples")
    console.print(f"Test: {len(X_test):,} samples")
    
    return X_train, y_train, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> SignalScorer:
    """Train the Signal Scorer model."""
    console.print(f"\n[cyan]Training Signal Scorer...[/cyan]")
    console.print(f"Features: {len(X_train.columns)}")
    
    # Check for GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    console.print(f"GPU: {'Available' if gpu_available else 'Not available'}")
    
    # Create and train model
    model = SignalScorer(
        use_gpu=gpu_available,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
    )
    
    # Train with cross-validation
    metrics = model.train(X_train, y_train, n_splits=5)
    
    # Evaluate on test set
    console.print(f"\n[cyan]Evaluating on test set...[/cyan]")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Print metrics
    table = Table(title="Model Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Train (CV)", style="white")
    table.add_column("Test", style="green")
    
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        train_val = metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        table.add_row(
            metric.capitalize(),
            f"{train_val:.3f}" if train_val else "-",
            f"{test_val:.3f}" if test_val else "-",
        )
    
    console.print(table)
    
    return model


def analyze_features(model: SignalScorer, feature_names: list[str]):
    """Analyze feature importance."""
    console.print(f"\n[cyan]Feature Importance Analysis...[/cyan]")
    
    try:
        importance = model.get_feature_importance()
        
        if importance is not None:
            # Sort by importance
            sorted_features = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            table = Table(title="Top 15 Features")
            table.add_column("Rank", style="dim")
            table.add_column("Feature", style="cyan")
            table.add_column("Importance", style="green")
            
            for i, (name, imp) in enumerate(sorted_features[:15], 1):
                table.add_row(str(i), name, f"{imp:.4f}")
            
            console.print(table)
            
            # Show signal features specifically
            console.print("\n[cyan]Signal Feature Importance:[/cyan]")
            signal_features = [f for f in feature_names if f.startswith("signal_")]
            for name in signal_features:
                idx = feature_names.index(name)
                if idx < len(importance):
                    console.print(f"  {name}: {importance[idx]:.4f}")
    
    except Exception as e:
        console.print(f"[yellow]Could not analyze features: {e}[/yellow]")


def save_model(model: SignalScorer, path: str = "models/signal_scorer.pkl"):
    """Save the trained model."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(str(model_path))
    console.print(f"\n[green]✓ Model saved to {model_path}[/green]")


async def main():
    parser = argparse.ArgumentParser(description="Train Signal Scorer Model")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all)")
    parser.add_argument("--lookahead", type=int, default=30, help="Lookahead minutes for labels")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Signal Scorer Training[/bold cyan]\n"
        "Comprehensive feature engineering per HYDRA_SPEC_ML.md",
        border_style="cyan"
    ))
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = PERMITTED_PAIRS
    
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Days: {args.days}")
    console.print(f"Lookahead: {args.lookahead} minutes")
    
    # Check for API key
    if not os.getenv("COINALYSE_API_KEY"):
        console.print("[yellow]Warning: COINALYSE_API_KEY not set. OI/Funding/Liquidation data will be missing.[/yellow]")
    
    # Fetch data
    data = await fetch_data(symbols, args.days)
    
    if not data:
        console.print("[red]No data fetched. Exiting.[/red]")
        return
    
    # Prepare datasets
    X_train, y_train, X_test, y_test = prepare_datasets(
        data,
        lookahead_minutes=args.lookahead,
    )
    
    if len(X_train) == 0:
        console.print("[red]No training data. Check data quality.[/red]")
        return
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Analyze features
    feature_names = list(X_train.columns)
    analyze_features(model, feature_names)
    
    # Save model
    save_model(model)
    
    console.print("\n[green]✓ Training complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
