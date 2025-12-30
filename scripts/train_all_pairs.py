#!/usr/bin/env python3
"""
HYDRA Full Training Script

Trains all models on ALL permitted pairs with production-grade data volume.
This script should be run to prepare HYDRA for paper/live trading.

Usage:
    python scripts/train_all_pairs.py
    python scripts/train_all_pairs.py --candles 100000 --epochs 200
    python scripts/train_all_pairs.py --pairs cmt_btcusdt,cmt_ethusdt
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.training.data_pipeline import DataPipeline
from hydra.training.trainer import HydraTrainer

console = Console()

# Production training parameters
DEFAULT_CANDLES = 100000  # ~347 days of 5m data - much more robust
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Train HYDRA on all permitted pairs")
    parser.add_argument(
        "--candles", 
        type=int, 
        default=DEFAULT_CANDLES,
        help=f"Number of candles per pair (default: {DEFAULT_CANDLES})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs to train (default: all permitted)"
    )
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["transformer", "rl", "all"],
        help="Component to train (default: all)"
    )
    parser.add_argument(
        "--skip-data-fetch",
        action="store_true",
        help="Skip data fetching, use cached data"
    )
    return parser.parse_args()


def symbol_to_exchange_format(symbol: str) -> str:
    """Convert cmt_btcusdt to BTC/USDT:USDT for exchange API."""
    base = symbol.replace("cmt_", "").replace("usdt", "").upper()
    return f"{base}/USDT:USDT"


async def fetch_training_data(
    pipeline: DataPipeline,
    pairs: list[str],
    candles: int,
    progress: Progress,
) -> dict[str, any]:
    """Fetch historical data for all pairs."""
    from datetime import timedelta
    
    data = {}
    
    # Calculate date range based on candle count
    # 5m candles: candles * 5 minutes
    minutes_needed = candles * 5
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=minutes_needed + 1440)  # Add 1 day buffer
    
    logger.info(f"Fetching {candles} candles from {start_date.date()} to {end_date.date()}")
    
    task = progress.add_task("Fetching data...", total=len(pairs))
    
    for pair in pairs:
        exchange_symbol = symbol_to_exchange_format(pair)
        progress.update(task, description=f"Fetching {PAIR_DISPLAY_NAMES.get(pair, pair)}...")
        
        try:
            df = await pipeline.fetch_historical_data(
                exchange_symbol, 
                "5m",
                start_date=start_date,
                end_date=end_date,
                limit=candles
            )
            df.attrs['symbol'] = pair
            data[pair] = df
            logger.info(f"Fetched {len(df)} candles for {pair}")
        except Exception as e:
            logger.error(f"Failed to fetch {pair}: {e}")
            continue
        
        progress.advance(task)
    
    return data


def prepare_datasets(
    pipeline: DataPipeline,
    data: dict[str, any],
    progress: Progress,
) -> tuple[list, list]:
    """Prepare training datasets from all pairs with BTC lead signal."""
    all_train = []
    all_val = []
    
    # Set BTC reference data for alt pair lead signals
    # BTC often leads alts by 5-30 minutes - this is a key trading edge
    btc_pair = 'cmt_btcusdt'
    if btc_pair in data:
        pipeline.feature_engineer.set_btc_reference(data[btc_pair])
        logger.info("BTC reference set for alt pair lead signals")
    
    task = progress.add_task("Preparing datasets...", total=len(data))
    
    for pair, df in data.items():
        progress.update(task, description=f"Processing {PAIR_DISPLAY_NAMES.get(pair, pair)}...")
        
        try:
            examples = list(pipeline.prepare_transformer_dataset(df))
            
            if not examples:
                logger.warning(f"No examples generated for {pair}")
                continue
            
            # 80/20 split
            split_idx = int(len(examples) * 0.8)
            all_train.extend(examples[:split_idx])
            all_val.extend(examples[split_idx:])
            
            logger.info(f"{pair}: {len(examples)} examples (train: {split_idx}, val: {len(examples) - split_idx})")
            
        except Exception as e:
            logger.error(f"Failed to process {pair}: {e}")
            continue
        
        progress.advance(task)
    
    return all_train, all_val


def train_transformer(
    trainer: HydraTrainer,
    train_data: list,
    val_data: list,
    epochs: int,
    progress: Progress,
) -> None:
    """Train the transformer model."""
    task = progress.add_task(f"Training Transformer ({epochs} epochs)...", total=epochs)
    
    # Custom callback to update progress
    original_fit = trainer.transformer_trainer.fit
    
    def fit_with_progress(*args, **kwargs):
        for epoch in range(epochs):
            # Train one epoch
            trainer.transformer_trainer._train_epoch(epoch)
            trainer.transformer_trainer._validate_epoch(epoch)
            progress.advance(task)
    
    # Just call the trainer directly
    model = trainer.train_transformer(train_data, val_data)
    progress.update(task, completed=epochs)
    
    return model


async def main():
    args = parse_args()
    
    # Determine pairs to train
    if args.pairs:
        pairs = [p.strip().lower() for p in args.pairs.split(",")]
        # Validate
        invalid = [p for p in pairs if p not in PERMITTED_PAIRS]
        if invalid:
            console.print(f"[red]Invalid pairs: {invalid}[/red]")
            console.print(f"Permitted: {list(PERMITTED_PAIRS)}")
            return 1
    else:
        pairs = list(PERMITTED_PAIRS)
    
    # Print banner
    console.print(Panel.fit(
        "[bold cyan]HYDRA Full Training Pipeline[/bold cyan]\n"
        f"Training on {len(pairs)} pairs with {args.candles:,} candles each\n"
        f"Epochs: {args.epochs} | Component: {args.component}",
        border_style="cyan"
    ))
    
    # Show pairs
    pairs_table = Table(title="Training Pairs", show_header=False)
    pairs_table.add_column("Pair")
    pairs_table.add_column("Exchange Symbol")
    for p in pairs:
        pairs_table.add_row(
            PAIR_DISPLAY_NAMES.get(p, p),
            symbol_to_exchange_format(p)
        )
    console.print(pairs_table)
    console.print()
    
    # Initialize
    config = HydraConfig.load()
    config.model.epochs = args.epochs
    
    pipeline = DataPipeline(config)
    trainer = HydraTrainer(config)
    
    start_time = datetime.now()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        # Step 1: Fetch data
        console.print("\n[bold]Step 1: Fetching Historical Data[/bold]")
        data = await fetch_training_data(pipeline, pairs, args.candles, progress)
        
        if not data:
            console.print("[red]No data fetched! Check your exchange connection.[/red]")
            return 1
        
        console.print(f"[green]✓ Fetched data for {len(data)} pairs[/green]")
        
        # Step 2: Prepare datasets
        console.print("\n[bold]Step 2: Preparing Training Datasets[/bold]")
        train_data, val_data = prepare_datasets(pipeline, data, progress)
        
        if not train_data:
            console.print("[red]No training examples generated![/red]")
            return 1
        
        console.print(f"[green]✓ Prepared {len(train_data):,} training, {len(val_data):,} validation examples[/green]")
        
        # Step 3: Train Transformer
        if args.component in ["transformer", "all"]:
            console.print("\n[bold]Step 3: Training Transformer Model[/bold]")
            
            task = progress.add_task(f"Training Transformer...", total=None)
            model = trainer.train_transformer(train_data, val_data)
            progress.update(task, description="Transformer complete!")
            
            console.print("[green]✓ Transformer training complete[/green]")
        
        # Step 4: Train RL Agent
        if args.component in ["rl", "all"]:
            console.print("\n[bold]Step 4: Training RL Agent[/bold]")
            
            from hydra.training.simulator import MarketSimulator
            
            task = progress.add_task("Training RL Agent...", total=None)
            
            # Create simulator with realistic parameters
            env = MarketSimulator(
                initial_cash=10000.0,
                initial_price=50000.0,
                volatility=0.02,
            )
            
            # Train with more timesteps for robustness
            policy = trainer.train_rl_agent(env, total_timesteps=100000)
            progress.update(task, description="RL Agent complete!")
            
            # Save RL policy
            import torch
            rl_path = Path("models") / "rl_policy.pt"
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), rl_path)
            logger.info(f"Saved RL policy to {rl_path}")
            
            console.print("[green]✓ RL Agent training complete[/green]")
    
    # Summary
    elapsed = datetime.now() - start_time
    
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n\n"
        f"Duration: {elapsed}\n"
        f"Pairs trained: {len(data)}\n"
        f"Total examples: {len(train_data) + len(val_data):,}\n"
        f"Models saved to: models/",
        title="Summary",
        border_style="green"
    ))
    
    # Show model files
    models_dir = Path("models")
    if models_dir.exists():
        console.print("\n[bold]Saved Models:[/bold]")
        for f in models_dir.glob("*.pt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  [green]✓[/green] {f.name} ({size_mb:.2f} MB)")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        logger.exception("Training error")
        sys.exit(1)
