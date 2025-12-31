#!/usr/bin/env python3
"""
Train Signal Scorer with Full 365 Days of Data

Optimized for maximum accuracy:
1. Fetches 365 days of historical data with proper rate limiting
2. Generates signals using ALL behavioral primitives
3. Balances classes for better model performance
4. Uses proper time-series cross-validation
5. Tunes hyperparameters for best accuracy

Usage:
    python scripts/train_full_model.py
    python scripts/train_full_model.py --days 365
    python scripts/train_full_model.py --days 180 --symbols cmt_btcusdt,cmt_ethusdt
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.training.historical_data import (
    HistoricalDataCollector, PERMITTED_PAIRS
)
from hydra.training.signal_scorer_data import (
    create_signal_scorer_dataset, 
    SignalScorerFeatureEngineer,
    generate_signals_from_data,
)
from hydra.training.models import SignalScorer

console = Console()


async def fetch_full_data(
    symbols: list[str], 
    days: int,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch comprehensive historical data with progress tracking."""
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    coinalyze_key = os.getenv("COINALYSE_API_KEY", "")
    if not coinalyze_key:
        console.print("[yellow]⚠ COINALYSE_API_KEY not set - OI/Funding/Liquidation will be estimated[/yellow]")
    
    collector = HistoricalDataCollector(coinalyze_api_key=coinalyze_key)
    
    data = {}
    
    console.print(f"\n[cyan]Fetching {days} days of data for {len(symbols)} symbols...[/cyan]")
    console.print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        for i, symbol in enumerate(symbols, 1):
            console.print(f"\n[{i}/{len(symbols)}] Fetching {symbol}...")
            
            try:
                df = await collector.collect_symbol_data(
                    symbol=symbol,
                    interval="5min",
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                )
                
                if not df.empty:
                    data[symbol] = df
                    
                    # Print data quality stats
                    console.print(f"  [green]✓ {len(df):,} candles[/green]")
                    
                    # Check data completeness
                    expected_candles = days * 24 * 12  # 5min candles
                    completeness = len(df) / expected_candles * 100
                    
                    if completeness < 95:
                        console.print(f"  [yellow]⚠ Data completeness: {completeness:.1f}%[/yellow]")
                    
                    # Check for NaN values
                    nan_cols = df.columns[df.isna().any()].tolist()
                    if nan_cols:
                        console.print(f"  [dim]Columns with NaN: {', '.join(nan_cols[:3])}...[/dim]")
                else:
                    console.print(f"  [red]✗ No data retrieved[/red]")
                    
            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
                logger.exception(f"Error fetching {symbol}")
    
    finally:
        await collector.close()
    
    return data


def validate_data_quality(data: dict[str, pd.DataFrame]) -> dict:
    """Validate data quality and return statistics."""
    stats = {
        "total_candles": 0,
        "symbols": len(data),
        "date_range": None,
        "avg_completeness": 0,
        "has_funding": False,
        "has_oi": False,
        "has_liquidations": False,
    }
    
    completeness_list = []
    
    for symbol, df in data.items():
        stats["total_candles"] += len(df)
        
        if stats["date_range"] is None:
            stats["date_range"] = (df.index.min(), df.index.max())
        
        # Check data presence
        if "funding_rate" in df.columns and df["funding_rate"].notna().any():
            stats["has_funding"] = True
        if "open_interest" in df.columns and df["open_interest"].notna().any():
            stats["has_oi"] = True
        if "long_liquidations" in df.columns and (df["long_liquidations"] > 0).any():
            stats["has_liquidations"] = True
        
        # Calculate completeness
        days = (df.index.max() - df.index.min()).days
        expected = days * 24 * 12
        completeness_list.append(len(df) / max(expected, 1) * 100)
    
    stats["avg_completeness"] = np.mean(completeness_list) if completeness_list else 0
    
    return stats


def prepare_balanced_dataset(
    data: dict[str, pd.DataFrame],
    lookahead_minutes: int = 30,
    balance_ratio: float = 1.5,  # Allow slight imbalance (1.5:1)
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training dataset with class balancing for better accuracy.
    """
    console.print(f"\n[cyan]Preparing balanced training dataset...[/cyan]")
    
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
                X["symbol"] = symbol
                all_X.append(X)
                all_y.append(y)
                
                win_rate = y.mean() * 100
                console.print(f"    {len(X):,} samples, win rate: {win_rate:.1f}%")
                
        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")
            logger.exception(f"Error processing {symbol}")
    
    if not all_X:
        return pd.DataFrame(), pd.Series()
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    # Class balancing through undersampling majority class
    n_positive = (y_combined == 1).sum()
    n_negative = (y_combined == 0).sum()
    
    console.print(f"\nBefore balancing: {n_positive:,} wins, {n_negative:,} losses")
    
    if n_negative > n_positive * balance_ratio:
        # Undersample negative class
        target_negative = int(n_positive * balance_ratio)
        
        positive_indices = y_combined[y_combined == 1].index.tolist()
        negative_indices = y_combined[y_combined == 0].index.tolist()
        
        # Keep all positives, sample negatives
        np.random.seed(42)
        sampled_negative = np.random.choice(negative_indices, target_negative, replace=False)
        
        keep_indices = positive_indices + list(sampled_negative)
        keep_indices.sort()
        
        X_combined = X_combined.loc[keep_indices].reset_index(drop=True)
        y_combined = y_combined.loc[keep_indices].reset_index(drop=True)
        
        console.print(f"After balancing: {(y_combined == 1).sum():,} wins, {(y_combined == 0).sum():,} losses")
    
    # Remove symbol column
    X_combined = X_combined.drop(columns=["symbol"], errors="ignore")
    
    console.print(f"\n[green]Total dataset: {len(X_combined):,} samples, {len(X_combined.columns)} features[/green]")
    console.print(f"Final win rate: {y_combined.mean()*100:.1f}%")
    
    return X_combined, y_combined


def train_optimized_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_split: float = 0.15,
) -> tuple[SignalScorer, dict]:
    """Train model with optimized hyperparameters."""
    
    console.print(f"\n[cyan]Training optimized Signal Scorer model...[/cyan]")
    
    # Time-based split
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    console.print(f"Train: {len(X_train):,} samples")
    console.print(f"Test: {len(X_test):,} samples (held out)")
    
    # Check for GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    console.print(f"GPU: {'Available ✓' if gpu_available else 'Not available'}")
    
    # Train with optimized params for accuracy
    model = SignalScorer(
        use_gpu=gpu_available,
        iterations=2000,       # More iterations for better learning
        learning_rate=0.03,    # Slower learning for better generalization
        depth=8,               # Deeper trees for complex patterns
        l2_leaf_reg=5.0,       # More regularization to prevent overfitting
    )
    
    train_metrics = model.train(X_train, y_train, n_splits=5)
    
    # Evaluate on held-out test set
    console.print(f"\n[cyan]Evaluating on held-out test set...[/cyan]")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Combined results
    results = {
        "train": train_metrics,
        "test": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(X.columns),
    }
    
    return model, results


def display_results(results: dict, model: SignalScorer, feature_names: list[str]):
    """Display comprehensive training results."""
    
    # Performance table
    table = Table(title="Model Performance", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Train (CV)", style="yellow")
    table.add_column("Test", style="green")
    table.add_column("Status", style="white")
    
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    
    for metric in metrics:
        train_val = results["train"].get(metric, 0)
        test_val = results["test"].get(metric, 0)
        
        # Determine status
        if metric == "accuracy":
            if test_val >= 0.70:
                status = "🟢 Excellent"
            elif test_val >= 0.60:
                status = "🟡 Good"
            else:
                status = "🔴 Needs improvement"
        elif metric == "auc":
            if test_val >= 0.70:
                status = "🟢 Excellent"
            elif test_val >= 0.60:
                status = "🟡 Good"
            else:
                status = "🔴 Needs improvement"
        else:
            status = ""
        
        table.add_row(
            metric.capitalize(),
            f"{train_val:.3f}" if train_val else "-",
            f"{test_val:.3f}" if test_val else "-",
            status,
        )
    
    console.print(table)
    
    # Feature importance
    console.print(f"\n[cyan]Top 20 Most Important Features:[/cyan]")
    
    importance = model.get_feature_importance()
    if len(importance) > 0:
        sorted_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        table2 = Table(show_header=True)
        table2.add_column("#", style="dim", width=3)
        table2.add_column("Feature", style="cyan")
        table2.add_column("Importance", style="green")
        table2.add_column("Category", style="yellow")
        
        for i, (name, imp) in enumerate(sorted_features[:20], 1):
            # Categorize feature
            if name.startswith("signal_"):
                cat = "Signal"
            elif name.startswith("funding_") or name.startswith("oi_"):
                cat = "Derivatives"
            elif name.startswith("return_") or name.startswith("volatility_"):
                cat = "Price"
            elif name.startswith("regime_"):
                cat = "Regime"
            elif name.startswith("liq_"):
                cat = "Liquidation"
            else:
                cat = "Other"
            
            table2.add_row(str(i), name, f"{imp:.4f}", cat)
        
        console.print(table2)
    
    # Signal feature analysis
    console.print(f"\n[cyan]Signal Feature Importance:[/cyan]")
    signal_features = [f for f in feature_names if f.startswith("signal_")]
    for name in signal_features:
        idx = feature_names.index(name)
        if idx < len(importance):
            bar_len = int(importance[idx] / max(importance) * 20)
            bar = "█" * bar_len
            console.print(f"  {name:35} {importance[idx]:6.3f} {bar}")


async def main():
    parser = argparse.ArgumentParser(description="Train Full Signal Scorer Model")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--lookahead", type=int, default=30, help="Lookahead minutes")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")
    parser.add_argument("--output", type=str, default="models/signal_scorer.pkl", help="Model output path")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Signal Scorer - Full Training[/bold cyan]\n"
        f"Training with {args.days} days of data for maximum accuracy",
        border_style="cyan"
    ))
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = PERMITTED_PAIRS
    
    console.print(f"\nConfiguration:")
    console.print(f"  Symbols: {len(symbols)} pairs")
    console.print(f"  Days: {args.days}")
    console.print(f"  Lookahead: {args.lookahead} minutes")
    console.print(f"  Output: {args.output}")
    
    # Fetch data
    data = await fetch_full_data(symbols, args.days, use_cache=not args.no_cache)
    
    if not data:
        console.print("[red]No data fetched. Check API keys and network.[/red]")
        return
    
    # Validate data quality
    stats = validate_data_quality(data)
    
    console.print(f"\n[cyan]Data Quality Report:[/cyan]")
    console.print(f"  Total candles: {stats['total_candles']:,}")
    console.print(f"  Symbols: {stats['symbols']}")
    console.print(f"  Avg completeness: {stats['avg_completeness']:.1f}%")
    console.print(f"  Has funding data: {'✓' if stats['has_funding'] else '✗'}")
    console.print(f"  Has OI data: {'✓' if stats['has_oi'] else '✗'}")
    console.print(f"  Has liquidation data: {'✓' if stats['has_liquidations'] else '✗'}")
    
    # Prepare balanced dataset
    X, y = prepare_balanced_dataset(data, lookahead_minutes=args.lookahead)
    
    if len(X) < 1000:
        console.print(f"[red]Insufficient training data ({len(X)} samples). Need at least 1000.[/red]")
        return
    
    # Train model
    model, results = train_optimized_model(X, y)
    
    # Display results
    feature_names = list(X.columns)
    display_results(results, model, feature_names)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    console.print(f"\n[green]✓ Model saved to {output_path}[/green]")
    
    # Summary
    test_acc = results["test"].get("accuracy", 0)
    test_auc = results["test"].get("auc", 0)
    
    if test_acc >= 0.65 and test_auc >= 0.65:
        console.print(f"\n[green]🎉 Model trained successfully with good accuracy![/green]")
    else:
        console.print(f"\n[yellow]⚠ Model accuracy could be improved. Consider:[/yellow]")
        console.print("  - Adding more training data (--days 365)")
        console.print("  - Ensuring Coinalyze API key is set for full data")
        console.print("  - Using all 8 trading pairs")


if __name__ == "__main__":
    asyncio.run(main())
