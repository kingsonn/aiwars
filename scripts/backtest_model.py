#!/usr/bin/env python3
"""
Fast Backtest using trained Transformer model.

Usage:
    python scripts/backtest_model.py
    python scripts/backtest_model.py --symbol cmt_ethusdt --days 90
    python scripts/backtest_model.py --all-pairs --capital 50000
"""

import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.training.data_pipeline import DataPipeline
from hydra.layers.layer3_alpha.transformer_model import FuturesTransformer

console = Console()


def to_exchange_symbol(internal: str) -> str:
    """Convert cmt_btcusdt -> BTC/USDT:USDT"""
    display = PAIR_DISPLAY_NAMES.get(internal, internal)
    return f"{display}:USDT"


def load_model(config: HydraConfig) -> FuturesTransformer:
    """Load trained transformer model."""
    models_dir = Path("models")
    model_path = models_dir / "final_transformer.pt"
    
    if not model_path.exists():
        model_path = models_dir / "final_transformer.pt"
    
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run training first.")
    
    model = FuturesTransformer(
        d_model=config.model.transformer_hidden_size,
        nhead=config.model.transformer_num_heads,
        num_layers=config.model.transformer_num_layers,
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    console.print(f"[green]Loaded model from {model_path}[/green]")
    return model


async def run_backtest(
    model: FuturesTransformer,
    pipeline: DataPipeline,
    symbol: str,
    days: int,
    capital: float,
    device: str,
) -> dict:
    """Run backtest on a single pair using the trained model."""
    model = model.to(device)
    exchange_symbol = to_exchange_symbol(symbol)
    
    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    console.print(f"[cyan]Fetching {days} days of data for {symbol}...[/cyan]")
    df = await pipeline.fetch_historical_data(
        exchange_symbol, "5m",
        start_date=start_date,
        end_date=end_date,
        limit=days * 288  # 288 5-min candles per day
    )
    
    if df is None or len(df) < 200:
        return {"error": f"Insufficient data for {symbol}", "symbol": symbol}
    
    console.print(f"[green]Loaded {len(df)} candles[/green]")
    
    # Prepare features (vectorized for speed)
    examples = list(pipeline.prepare_transformer_dataset(df, symbol))
    
    if len(examples) < 100:
        return {"error": f"Not enough examples for {symbol}", "symbol": symbol}
    
    # Get symbol index
    symbol_idx = PERMITTED_PAIRS.index(symbol) if symbol in PERMITTED_PAIRS else 0
    
    # Batch inference for speed
    console.print("[cyan]Running model inference...[/cyan]")
    
    batch_size = 64
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i+batch_size]
            
            # Stack features
            price = torch.stack([torch.FloatTensor(ex.price_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            funding = torch.stack([torch.FloatTensor(ex.funding_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            oi = torch.stack([torch.FloatTensor(ex.oi_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            orderbook = torch.stack([torch.FloatTensor(ex.orderbook_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            liq = torch.stack([torch.FloatTensor(ex.liq_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            vol = torch.stack([torch.FloatTensor(ex.vol_features) for ex in batch_examples]).permute(1, 0, 2).to(device)
            
            symbol_tensor = torch.LongTensor([symbol_idx] * len(batch_examples)).to(device)
            
            output = model(price, funding, oi, orderbook, liq, vol, symbol_idx=symbol_tensor)
            
            # Get predictions
            dir_probs = output['direction_probs'].cpu().numpy()
            pred_dirs = np.argmax(dir_probs, axis=1)
            confidences = np.max(dir_probs, axis=1)
            
            for j, ex in enumerate(batch_examples):
                all_predictions.append({
                    'pred_dir': pred_dirs[j],
                    'confidence': confidences[j],
                    'actual_dir': ex.direction_label,
                    'raw_return': ex.raw_return,
                })
    
    # Simulate trading
    console.print("[cyan]Simulating trades...[/cyan]")
    
    equity = capital
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    trades = []
    equity_curve = [capital]
    
    # Trading parameters
    position_size = 0.1  # 10% of capital per trade
    fee_rate = 0.0004  # 0.04% per side
    min_confidence = 0.6  # Only trade with high confidence
    
    for i, pred in enumerate(all_predictions):
        current_price = df['close'].iloc[i + 100] if i + 100 < len(df) else df['close'].iloc[-1]  # Offset for sequence length
        
        # Skip low confidence predictions
        if pred['confidence'] < min_confidence:
            if position != 0:
                # Update equity for holding position
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                equity_curve.append(capital + capital * position_size * pnl_pct)
            else:
                equity_curve.append(equity)
            continue
        
        pred_dir = pred['pred_dir']
        
        # Close existing position if direction changes
        if position != 0:
            if (position == 1 and pred_dir != 0) or (position == -1 and pred_dir != 1):
                # Close position
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price - fee_rate
                else:
                    pnl_pct = (entry_price - current_price) / entry_price - fee_rate
                
                trade_pnl = capital * position_size * pnl_pct
                equity += trade_pnl
                
                trades.append({
                    'side': 'LONG' if position == 1 else 'SHORT',
                    'pnl': trade_pnl,
                    'pnl_pct': pnl_pct,
                })
                
                position = 0
        
        # Open new position
        if position == 0 and pred_dir != 2:  # Not flat prediction
            position = 1 if pred_dir == 0 else -1
            entry_price = current_price * (1 + fee_rate if position == 1 else 1 - fee_rate)
        
        equity_curve.append(equity)
    
    # Close any remaining position
    if position != 0:
        current_price = df['close'].iloc[-1]
        if position == 1:
            pnl_pct = (current_price - entry_price) / entry_price - fee_rate
        else:
            pnl_pct = (entry_price - current_price) / entry_price - fee_rate
        
        trade_pnl = capital * position_size * pnl_pct
        equity += trade_pnl
        
        trades.append({
            'side': 'LONG' if position == 1 else 'SHORT',
            'pnl': trade_pnl,
            'pnl_pct': pnl_pct,
        })
    
    # Calculate metrics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    total_return = equity - capital
    total_return_pct = total_return / capital * 100
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    # Sharpe ratio (simplified)
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288) if np.std(returns) > 0 else 0
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # Direction accuracy
    correct = sum(1 for p in all_predictions if p['pred_dir'] == p['actual_dir'])
    accuracy = correct / len(all_predictions) * 100
    
    return {
        "symbol": symbol,
        "days": days,
        "candles": len(df),
        "examples": len(all_predictions),
        "initial_capital": capital,
        "final_equity": equity,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "direction_accuracy": accuracy,
    }


def print_results(results: dict):
    """Print backtest results."""
    if "error" in results:
        console.print(f"[red]Error for {results['symbol']}: {results['error']}[/red]")
        return
    
    table = Table(title=f"Backtest Results - {results['symbol']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Period", f"{results['days']} days")
    table.add_row("Candles", f"{results['candles']:,}")
    table.add_row("Initial Capital", f"${results['initial_capital']:,.0f}")
    table.add_row("Final Equity", f"${results['final_equity']:,.2f}")
    table.add_row("Total Return", f"${results['total_return']:,.2f}")
    table.add_row("Return %", f"{results['total_return_pct']:.2f}%")
    table.add_row("Trades", str(results['num_trades']))
    table.add_row("Win Rate", f"{results['win_rate']:.1f}%")
    table.add_row("Avg Win", f"${results['avg_win']:.2f}")
    table.add_row("Avg Loss", f"${results['avg_loss']:.2f}")
    table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{results['max_drawdown']:.2f}%")
    table.add_row("Direction Accuracy", f"{results['direction_accuracy']:.1f}%")
    
    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Backtest trained HYDRA model")
    parser.add_argument("--symbol", type=str, default="cmt_btcusdt", help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--all-pairs", action="store_true", help="Backtest all pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Model Backtest[/bold cyan]\n"
        f"Testing trained Transformer on historical data",
        border_style="cyan"
    ))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[yellow]Using device: {device}[/yellow]\n")
    
    config = HydraConfig.load()
    model = load_model(config)
    pipeline = DataPipeline(config)
    
    if args.all_pairs:
        symbols = list(PERMITTED_PAIRS)
    else:
        symbols = [args.symbol]
    
    all_results = []
    
    for symbol in symbols:
        results = await run_backtest(
            model, pipeline, symbol,
            args.days, args.capital, device
        )
        print_results(results)
        all_results.append(results)
        console.print()
    
    # Summary for all pairs
    if len(all_results) > 1:
        valid_results = [r for r in all_results if "error" not in r]
        if valid_results:
            total_return = sum(r['total_return'] for r in valid_results)
            avg_win_rate = np.mean([r['win_rate'] for r in valid_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
            
            console.print(Panel.fit(
                f"[bold green]Portfolio Summary[/bold green]\n\n"
                f"Pairs tested: {len(valid_results)}\n"
                f"Total PnL: ${total_return:,.2f}\n"
                f"Avg Win Rate: {avg_win_rate:.1f}%\n"
                f"Avg Sharpe: {avg_sharpe:.2f}",
                border_style="green"
            ))
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
