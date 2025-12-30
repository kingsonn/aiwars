#!/usr/bin/env python3
"""
Test trained Transformer and RL models.

Usage:
    python scripts/test_models.py                    # Quick test on recent data
    python scripts/test_models.py --backtest         # Run full backtest
    python scripts/test_models.py --symbol cmt_ethusdt  # Test specific pair
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.training.data_pipeline import DataPipeline

# Convert internal symbol to exchange format
def to_exchange_symbol(internal: str) -> str:
    """Convert cmt_btcusdt -> BTC/USDT:USDT"""
    display = PAIR_DISPLAY_NAMES.get(internal, internal)
    return f"{display}:USDT"

from hydra.layers.layer3_alpha.transformer_model import FuturesTransformer
from hydra.layers.layer3_alpha.rl_agent import PolicyNetwork
from hydra.training.simulator import MarketSimulator

console = Console()


def load_transformer(model_path: Path, config: HydraConfig) -> FuturesTransformer:
    """Load trained transformer model."""
    model = FuturesTransformer(
        d_model=config.model.transformer_hidden_size,
        nhead=config.model.transformer_num_heads,
        num_layers=config.model.transformer_num_layers,
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_rl_policy(model_path: Path) -> PolicyNetwork:
    """Load trained RL policy."""
    policy = PolicyNetwork(state_dim=8, hidden_dim=128, num_actions=7)
    policy.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    policy.eval()
    
    return policy


async def test_transformer(
    model: FuturesTransformer,
    pipeline: DataPipeline,
    symbol: str,
    num_samples: int = 100,
) -> dict:
    """Test transformer on recent data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Convert to exchange symbol format
    exchange_symbol = to_exchange_symbol(symbol)
    
    # Fetch recent data
    console.print(f"[cyan]Fetching data for {symbol} ({exchange_symbol})...[/cyan]")
    df = await pipeline.fetch_historical_data(exchange_symbol, "5m", limit=5000)
    
    if df is None or len(df) < 200:
        return {"error": f"Insufficient data for {symbol}"}
    
    # Prepare test examples (convert generator to list)
    examples = list(pipeline.prepare_transformer_dataset(df, symbol))
    
    if len(examples) < num_samples:
        num_samples = len(examples)
    
    # Test on last N examples
    test_examples = examples[-num_samples:]
    
    results = {
        "symbol": symbol,
        "samples": num_samples,
        "predictions": {"long": 0, "short": 0, "flat": 0},
        "correct_direction": 0,
        "avg_confidence": 0.0,
        "avg_predicted_vol": 0.0,
    }
    
    confidences = []
    volatilities = []
    
    for ex in test_examples:
        # Build input tensors - model expects (seq_len, batch, features)
        with torch.no_grad():
            # Add batch dim and transpose to (seq_len, batch, features)
            price = torch.FloatTensor(ex.price_features).unsqueeze(1).to(device)  # (seq, 1, feat)
            funding = torch.FloatTensor(ex.funding_features).unsqueeze(1).to(device)
            oi = torch.FloatTensor(ex.oi_features).unsqueeze(1).to(device)
            orderbook = torch.FloatTensor(ex.orderbook_features).unsqueeze(1).to(device)
            liq = torch.FloatTensor(ex.liq_features).unsqueeze(1).to(device)
            vol = torch.FloatTensor(ex.vol_features).unsqueeze(1).to(device)
            
            # Get symbol index (needs to be (batch,) shape)
            symbol_idx = PERMITTED_PAIRS.index(symbol) if symbol in PERMITTED_PAIRS else 0
            symbol_tensor = torch.LongTensor([symbol_idx]).to(device)
            
            output = model(price, funding, oi, orderbook, liq, vol, symbol_idx=symbol_tensor)
        
        # Parse prediction
        dir_probs = torch.softmax(output['direction_logits'], dim=-1).cpu().numpy()[0]
        pred_dir = np.argmax(dir_probs)
        confidence = dir_probs[pred_dir]
        
        confidences.append(confidence)
        volatilities.append(output['predicted_vol'].item())
        
        # Count predictions
        if pred_dir == 0:
            results["predictions"]["long"] += 1
        elif pred_dir == 1:
            results["predictions"]["short"] += 1
        else:
            results["predictions"]["flat"] += 1
        
        # Check if correct (comparing to actual label)
        if pred_dir == ex.direction_label:
            results["correct_direction"] += 1
    
    results["avg_confidence"] = np.mean(confidences)
    results["avg_predicted_vol"] = np.mean(volatilities)
    results["direction_accuracy"] = results["correct_direction"] / num_samples
    
    return results


def test_rl_policy(policy: PolicyNetwork, num_episodes: int = 10) -> dict:
    """Test RL policy on simulated environment."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = policy.to(device)
    
    env = MarketSimulator(
        initial_cash=10000.0,
        initial_price=50000.0,
        volatility=0.02,
    )
    
    results = {
        "episodes": num_episodes,
        "total_rewards": [],
        "final_pnls": [],
        "action_distribution": {i: 0 for i in range(7)},
    }
    
    action_names = ["HOLD", "LONG", "SHORT", "EXIT_PARTIAL", "EXIT_FULL", "FLIP_LONG", "FLIP_SHORT"]
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value, size = policy(state_t)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            results["action_distribution"][action] += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        results["total_rewards"].append(episode_reward)
        results["final_pnls"].append(env.pnl)
    
    results["avg_reward"] = np.mean(results["total_rewards"])
    results["avg_pnl"] = np.mean(results["final_pnls"])
    results["action_names"] = action_names
    
    return results


def print_transformer_results(results: dict):
    """Print transformer test results."""
    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        return
    
    table = Table(title=f"Transformer Test - {results['symbol']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Samples Tested", str(results["samples"]))
    table.add_row("Direction Accuracy", f"{results['direction_accuracy']:.1%}")
    table.add_row("Avg Confidence", f"{results['avg_confidence']:.3f}")
    table.add_row("Avg Predicted Vol", f"{results['avg_predicted_vol']:.4f}")
    table.add_row("Long Predictions", str(results["predictions"]["long"]))
    table.add_row("Short Predictions", str(results["predictions"]["short"]))
    table.add_row("Flat Predictions", str(results["predictions"]["flat"]))
    
    console.print(table)


def print_rl_results(results: dict):
    """Print RL policy test results."""
    table = Table(title="RL Policy Test (Simulated)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Episodes", str(results["episodes"]))
    table.add_row("Avg Reward", f"{results['avg_reward']:.2f}")
    table.add_row("Avg Final PnL", f"${results['avg_pnl']:.2f}")
    table.add_row("Best Episode", f"${max(results['final_pnls']):.2f}")
    table.add_row("Worst Episode", f"${min(results['final_pnls']):.2f}")
    
    console.print(table)
    
    # Action distribution
    action_table = Table(title="Action Distribution")
    action_table.add_column("Action", style="cyan")
    action_table.add_column("Count", style="yellow")
    action_table.add_column("Percentage", style="green")
    
    total_actions = sum(results["action_distribution"].values())
    for action_id, count in results["action_distribution"].items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        action_table.add_row(
            results["action_names"][action_id],
            str(count),
            f"{pct:.1f}%"
        )
    
    console.print(action_table)


async def main():
    parser = argparse.ArgumentParser(description="Test HYDRA models")
    parser.add_argument("--symbol", type=str, default="cmt_btcusdt", help="Symbol to test")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--episodes", type=int, default=10, help="RL test episodes")
    parser.add_argument("--transformer-only", action="store_true", help="Test only transformer")
    parser.add_argument("--rl-only", action="store_true", help="Test only RL policy")
    parser.add_argument("--all-pairs", action="store_true", help="Test all permitted pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Model Testing[/bold cyan]\n"
        "Testing trained Transformer and RL models",
        border_style="cyan"
    ))
    
    # Check model files exist
    models_dir = Path("models")
    transformer_path = models_dir / "best_transformer.pt"
    rl_path = models_dir / "rl_policy.pt"
    
    if not transformer_path.exists():
        transformer_path = models_dir / "final_transformer.pt"
    
    if not transformer_path.exists() and not args.rl_only:
        console.print("[red]Error: No transformer model found in models/[/red]")
        console.print("Run training first: python scripts/train_all_pairs.py")
        return 1
    
    if not rl_path.exists() and not args.transformer_only:
        console.print("[yellow]Warning: No RL policy found, skipping RL tests[/yellow]")
        args.transformer_only = True
    
    config = HydraConfig.load()
    
    # Test Transformer
    if not args.rl_only:
        console.print("\n[bold]Testing Transformer Model[/bold]")
        model = load_transformer(transformer_path, config)
        pipeline = DataPipeline(config)
        
        if args.all_pairs:
            symbols = PERMITTED_PAIRS
        else:
            symbols = [args.symbol]
        
        for symbol in symbols:
            results = await test_transformer(model, pipeline, symbol, args.samples)
            print_transformer_results(results)
            console.print()
    
    # Test RL Policy
    if not args.transformer_only and rl_path.exists():
        console.print("\n[bold]Testing RL Policy[/bold]")
        policy = load_rl_policy(rl_path)
        results = test_rl_policy(policy, args.episodes)
        print_rl_results(results)
    
    console.print("\n[green]✓ Testing complete![/green]")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
