"""
HYDRA Command Line Interface

Commands:
- run: Start the trading engine
- backtest: Run backtests
- train: Train models
- status: Check system status
- config: Manage configuration
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.engine import HydraEngine

app = typer.Typer(
    name="hydra",
    help="HYDRA - AI-Native Crypto Perpetual Futures Trading System",
    add_completion=False,
)
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        "logs/hydra_{time}.log",
        rotation="1 day",
        retention="30 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )
    logger.add(
        lambda msg: console.print(f"[dim]{msg}[/dim]") if level == "DEBUG" else None,
        level=level,
    )


def print_banner() -> None:
    """Print HYDRA banner."""
    banner = """
    ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ 
    ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗
    ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║
    ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║
    ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║
    ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    
    AI-Native Perpetual Futures Trading System
    """
    console.print(Panel(banner, style="bold cyan"))


@app.command()
def run(
    mode: str = typer.Option("paper", help="Trading mode: live, paper, or backtest"),
    symbols: Optional[str] = typer.Option(None, help="Comma-separated symbols to trade"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Start the HYDRA trading engine."""
    print_banner()
    
    # Load config
    if config_file:
        config = HydraConfig.load(str(config_file))
    else:
        config = HydraConfig.load()
    
    # Override mode
    config.trading.trading_mode = mode
    
    # Override symbols
    if symbols:
        config.trading.symbols = [s.strip() for s in symbols.split(",")]
    
    # Validate for live trading
    if mode == "live":
        errors = config.validate_for_live_trading()
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for err in errors:
                console.print(f"  • {err}")
            raise typer.Exit(1)
    
    setup_logging(config.system.log_level)
    
    console.print(f"\n[green]Starting HYDRA in {mode.upper()} mode[/green]")
    console.print(f"Symbols: {', '.join(config.trading.symbols)}")
    console.print(f"Max Leverage: {config.trading.max_leverage}x")
    console.print(f"Max Position: ${config.trading.max_position_size_usd:,.0f}")
    console.print()
    
    # Create and run engine
    engine = HydraEngine(config)
    
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        asyncio.run(engine.stop())
        console.print("[green]HYDRA stopped gracefully[/green]")


@app.command()
def backtest(
    symbol: str = typer.Option("BTC/USDT:USDT", help="Symbol to backtest"),
    start: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-12-01", help="End date (YYYY-MM-DD)"),
    capital: float = typer.Option(100000, help="Initial capital"),
) -> None:
    """Run a backtest."""
    print_banner()
    console.print(f"[cyan]Running backtest for {symbol}[/cyan]")
    console.print(f"Period: {start} to {end}")
    console.print(f"Capital: ${capital:,.0f}\n")
    
    from hydra.training.backtester import Backtester
    from hydra.training.data_pipeline import DataPipeline
    
    config = HydraConfig.load()
    pipeline = DataPipeline(config)
    backtester = Backtester(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Fetch data
        task = progress.add_task("Fetching historical data...", total=None)
        
        from datetime import datetime
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        data = asyncio.run(pipeline.fetch_historical_data(
            symbol, "5m", start_dt, end_dt, limit=100000
        ))
        data.attrs['symbol'] = symbol
        
        progress.update(task, description=f"Loaded {len(data)} candles")
        
        # Run backtest
        progress.update(task, description="Running backtest...")
        
        async def simple_strategy(market_state):
            from hydra.core.types import Signal, Side
            from datetime import datetime, timezone
            
            # Simple momentum strategy for demo
            candles = market_state.ohlcv.get("5m", [])
            if len(candles) < 20:
                return None
            
            prices = [c.close for c in candles[-20:]]
            sma = sum(prices) / len(prices)
            
            if market_state.price > sma * 1.01:
                return Signal(
                    timestamp=datetime.now(timezone.utc),
                    symbol=market_state.symbol,
                    side=Side.LONG,
                    confidence=0.7,
                    expected_return=0.01,
                    expected_adverse_excursion=0.02,
                    holding_period_minutes=60,
                )
            elif market_state.price < sma * 0.99:
                return Signal(
                    timestamp=datetime.now(timezone.utc),
                    symbol=market_state.symbol,
                    side=Side.SHORT,
                    confidence=0.7,
                    expected_return=0.01,
                    expected_adverse_excursion=0.02,
                    holding_period_minutes=60,
                )
            return None
        
        result = asyncio.run(backtester.run_backtest(data, simple_strategy, capital))
        
        progress.update(task, description="Complete!")
    
    # Print results
    backtester.print_results(result)


@app.command()
def paper(
    balance: float = typer.Option(1000.0, help="Initial balance in USDC"),
    symbols: Optional[str] = typer.Option(None, help="Comma-separated symbols (default: all permitted)"),
) -> None:
    """Start paper trading with live market data."""
    print_banner()
    
    from hydra.paper_trading import PaperTradingEngine
    from hydra.core.config import PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
    
    config = HydraConfig.load()
    
    # Set symbols
    if symbols:
        config.trading.symbols = [s.strip().lower() for s in symbols.split(",")]
    
    console.print(f"[green]Starting Paper Trading[/green]")
    console.print(f"Initial Balance: [bold]${balance:,.2f} USDC[/bold]")
    console.print(f"Trading Pairs: {', '.join(PAIR_DISPLAY_NAMES.get(s, s) for s in config.trading.symbols)}")
    console.print()
    
    engine = PaperTradingEngine(
        config=config,
        initial_balance=balance,
    )
    
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping paper trading...[/yellow]")
        asyncio.run(engine.stop())


@app.command()
def portfolio(
    action: str = typer.Argument("show", help="Action: show, positions, trades, reset"),
) -> None:
    """View paper trading portfolio."""
    from hydra.paper_trading import Portfolio, TradingDashboard
    
    portfolio = Portfolio()
    portfolio.load()
    
    dashboard = TradingDashboard(portfolio)
    
    if action == "show":
        dashboard.print_dashboard()
    elif action == "positions":
        dashboard.print_positions()
    elif action == "trades":
        dashboard.print_trades(limit=30)
    elif action == "reset":
        import os
        state_file = portfolio.data_dir / "portfolio_state.json"
        if state_file.exists():
            os.remove(state_file)
            console.print("[green]Portfolio reset![/green]")
        else:
            console.print("[yellow]No saved portfolio to reset[/yellow]")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def train(
    component: str = typer.Option("transformer", help="Component to train: transformer, rl, all"),
    symbol: str = typer.Option("cmt_btcusdt", help="Symbol for training data (use permitted pairs)"),
    epochs: int = typer.Option(100, help="Training epochs"),
) -> None:
    """Train HYDRA models on permitted pairs."""
    print_banner()
    
    from hydra.core.config import PERMITTED_PAIRS
    
    # Validate symbol
    symbol_lower = symbol.lower()
    if symbol_lower not in PERMITTED_PAIRS:
        console.print(f"[red]Invalid symbol: {symbol}[/red]")
        console.print(f"Permitted pairs: {', '.join(PERMITTED_PAIRS)}")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Training {component} model on {symbol}[/cyan]\n")
    
    from hydra.training.trainer import HydraTrainer
    from hydra.training.data_pipeline import DataPipeline
    
    config = HydraConfig.load()
    config.model.epochs = epochs
    
    pipeline = DataPipeline(config)
    trainer = HydraTrainer(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching training data...", total=None)
        
        # Map cmt_* to exchange format for data fetching
        exchange_symbol = symbol_lower.replace("cmt_", "").upper()
        exchange_symbol = f"{exchange_symbol[:len(exchange_symbol)-4]}/USDT:USDT"
        
        data = asyncio.run(pipeline.fetch_historical_data(exchange_symbol, "5m", limit=100000))
        data.attrs['symbol'] = symbol_lower
        
        progress.update(task, description="Preparing dataset...")
        
        examples = list(pipeline.prepare_transformer_dataset(data))
        
        if not examples:
            console.print("[red]No training examples generated![/red]")
            raise typer.Exit(1)
        
        # Split
        split_idx = int(len(examples) * 0.8)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        console.print(f"Training examples: {len(train_examples)}, Validation: {len(val_examples)}")
        progress.update(task, description=f"Training on {len(train_examples)} examples...")
        
        if component in ["transformer", "all"]:
            model = trainer.train_transformer(train_examples, val_examples)
            console.print("[green]Transformer training complete![/green]")
        
        if component in ["rl", "all"]:
            from hydra.training.simulator import MarketSimulator
            env = MarketSimulator()
            policy = trainer.train_rl_agent(env, total_timesteps=50000)
            console.print("[green]RL training complete![/green]")


@app.command()
def status() -> None:
    """Check HYDRA system status."""
    print_banner()
    
    config = HydraConfig.load()
    
    # Config status
    table = Table(title="Configuration Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Trading Mode", config.trading.trading_mode)
    table.add_row("Primary Exchange", config.exchange.primary_exchange)
    table.add_row("Symbols", ", ".join(config.trading.symbols))
    table.add_row("Max Leverage", f"{config.trading.max_leverage}x")
    table.add_row("Max Position", f"${config.trading.max_position_size_usd:,.0f}")
    table.add_row("LLM Provider", config.llm.llm_provider)
    table.add_row("LLM Model", config.llm.llm_model)
    
    console.print(table)
    
    # Validation
    console.print("\n[bold]Validation:[/bold]")
    errors = config.validate_for_live_trading()
    
    if not errors:
        console.print("[green]✓ Configuration valid for live trading[/green]")
    else:
        for err in errors:
            console.print(f"[yellow]⚠ {err}[/yellow]")
    
    # Check models
    console.print("\n[bold]Models:[/bold]")
    models_dir = Path(config.model.models_dir)
    
    if models_dir.exists():
        models = list(models_dir.glob("*.pt"))
        if models:
            for m in models:
                console.print(f"[green]✓ {m.name}[/green]")
        else:
            console.print("[yellow]No trained models found[/yellow]")
    else:
        console.print("[yellow]Models directory not found[/yellow]")


@app.command()
def kill() -> None:
    """Emergency kill switch - flatten all positions."""
    console.print("[red bold]⚠️  KILL SWITCH ACTIVATED ⚠️[/red bold]")
    
    if not typer.confirm("This will close ALL positions. Continue?"):
        raise typer.Abort()
    
    config = HydraConfig.load()
    engine = HydraEngine(config)
    
    asyncio.run(engine.setup())
    asyncio.run(engine.manual_kill("Manual CLI kill switch"))
    
    console.print("[green]All positions flattened[/green]")


@app.command()
def version() -> None:
    """Show HYDRA version."""
    from hydra import __version__, __codename__
    console.print(f"HYDRA {__codename__} v{__version__}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
