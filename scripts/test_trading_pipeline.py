#!/usr/bin/env python3
"""
Test Trading Pipeline

Tests the complete Layer 3, 4, 5 integration:
- Behavioral signal generation
- Risk evaluation
- Entry/exit decision logic
- Execution flow (paper trading)

Usage:
    python scripts/test_trading_pipeline.py
    python scripts/test_trading_pipeline.py --symbol cmt_btcusdt
    python scripts/test_trading_pipeline.py --all
"""

import argparse
import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import Side, Regime
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer, PERMITTED_PAIRS
from hydra.layers.layer2_statistical import StatisticalRealityEngine, TradabilityStatus
from hydra.layers.layer3_alpha.signals import BehavioralSignalGenerator
from hydra.layers.layer4_risk import RiskCapitalBrain, PortfolioState
from hydra.layers.layer5_executor import BinanceFuturesExecutor

console = Console()

PAIR_DISPLAY_NAMES = {
    "cmt_btcusdt": "BTC/USDT",
    "cmt_ethusdt": "ETH/USDT",
    "cmt_solusdt": "SOL/USDT",
    "cmt_bnbusdt": "BNB/USDT",
    "cmt_adausdt": "ADA/USDT",
    "cmt_xrpusdt": "XRP/USDT",
    "cmt_ltcusdt": "LTC/USDT",
    "cmt_dogeusdt": "DOGE/USDT",
}


async def test_signal_generation(symbol: str, layer1, layer2, layer3):
    """Test signal generation for a symbol."""
    console.print(f"\n[cyan]Testing signal generation for {PAIR_DISPLAY_NAMES.get(symbol, symbol)}...[/cyan]")
    
    # Refresh data
    await layer1.refresh_symbol(symbol)
    market_state = layer1.get_market_state(symbol)
    
    if not market_state or market_state.price == 0:
        console.print(f"[red]No market data for {symbol}[/red]")
        return None, None, []
    
    console.print(f"  Price: ${market_state.price:,.2f}")
    
    if market_state.funding_rate:
        console.print(f"  Funding: {market_state.funding_rate.rate*100:.4f}%")
    if market_state.open_interest:
        console.print(f"  OI Delta: {market_state.open_interest.delta_pct*100:.2f}%")
    
    # Layer 2 analysis
    stat_result = await layer2.analyze(market_state)
    console.print(f"  Regime: {stat_result.regime.name} ({stat_result.regime_confidence:.0%})")
    console.print(f"  Tradability: {stat_result.trading_decision.value}")
    console.print(f"  Volatility: {stat_result.volatility_regime}")
    
    # Layer 3 signals
    signals = layer3.generate_signals(
        market_state=market_state,
        stat_result=stat_result,
        long_short_ratio=1.0,
    )
    
    if signals:
        console.print(f"\n[green]Generated {len(signals)} signal(s):[/green]")
        for sig in signals:
            color = "green" if sig.side == Side.LONG else "red"
            console.print(
                f"  [{color}]{sig.side.value.upper()}[/{color}] "
                f"conf={sig.confidence:.2f} "
                f"source={sig.source}"
            )
            console.print(f"    Thesis: {sig.metadata.get('thesis', 'N/A')}")
    else:
        console.print("[yellow]No signals generated[/yellow]")
    
    return market_state, stat_result, signals


async def test_risk_evaluation(signal, market_state, stat_result, layer4):
    """Test risk evaluation for a signal."""
    if signal is None:
        console.print("[yellow]No signal to evaluate[/yellow]")
        return None
    
    console.print(f"\n[cyan]Evaluating through Risk Brain...[/cyan]")
    
    # Build mock portfolio state
    portfolio = PortfolioState(
        total_equity=10000.0,
        available_margin=10000.0,
        used_margin=0.0,
        total_exposure_usd=0.0,
        net_exposure_usd=0.0,
        gross_leverage=0.0,
        unrealized_pnl=0.0,
        realized_pnl_today=0.0,
        peak_equity=10000.0,
        current_drawdown=0.0,
        max_drawdown_today=0.0,
        num_positions=0,
        position_correlation=0.0,
    )
    
    risk_decision = await layer4.evaluate(
        signal=signal,
        market_state=market_state,
        stat_result=stat_result,
        current_position=None,
        all_positions={},
    )
    
    color = "green" if risk_decision.approved else "red"
    console.print(f"  Approved: [{color}]{risk_decision.approved}[/{color}]")
    
    if risk_decision.veto:
        console.print(f"  [red]VETO: {risk_decision.veto_reason}[/red]")
    
    console.print(f"  Size: ${risk_decision.recommended_size_usd:.0f}")
    console.print(f"  Leverage: {risk_decision.recommended_leverage:.1f}x")
    console.print(f"  Stop-loss: ${risk_decision.stop_loss_price:.2f}")
    console.print(f"  Take-profit: ${risk_decision.take_profit_price:.2f}")
    console.print(f"  Max holding: {risk_decision.max_holding_time_hours:.1f}h")
    
    return risk_decision


async def test_execution(signal, risk_decision, market_state, executor):
    """Test execution (paper trading)."""
    if signal is None or risk_decision is None:
        console.print("[yellow]No signal/decision to execute[/yellow]")
        return
    
    if not risk_decision.approved:
        console.print("[yellow]Risk not approved, skipping execution[/yellow]")
        return
    
    console.print(f"\n[cyan]Testing execution (PAPER TRADING)...[/cyan]")
    
    result = await executor.execute_entry(
        signal=signal,
        risk_decision=risk_decision,
        current_price=market_state.price,
    )
    
    color = "green" if result.status.value == "success" else "red"
    console.print(f"  Status: [{color}]{result.status.value}[/{color}]")
    console.print(f"  Order ID: {result.order_id}")
    console.print(f"  Filled: {result.filled_qty:.6f} @ ${result.filled_price:.2f}")
    console.print(f"  Value: ${result.filled_usd:.2f}")
    
    # Show paper positions
    positions = executor.get_paper_positions()
    if positions:
        console.print(f"\n[green]Paper Positions:[/green]")
        for sym, pos in positions.items():
            console.print(f"  {sym}: {pos['side'].value} {pos['size']:.6f} @ ${pos['price']:.2f}")


async def main():
    parser = argparse.ArgumentParser(description="Test Trading Pipeline")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    parser.add_argument("--all", action="store_true", help="Test all pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Trading Pipeline Test[/bold cyan]\n"
        "Testing Layer 3 (Signals) → Layer 4 (Risk) → Layer 5 (Execution)",
        border_style="cyan"
    ))
    
    # Load config
    config = HydraConfig.load()
    
    # Initialize layers
    console.print("\n[cyan]Initializing layers...[/cyan]")
    
    # Layer 1
    layer1 = MarketIntelligenceLayer()
    await layer1.initialize()
    console.print("[green]✓ Layer 1 (Market Intelligence)[/green]")
    
    # Layer 2
    layer2 = StatisticalRealityEngine(config, use_ml_regime=True)
    await layer2.setup()
    console.print("[green]✓ Layer 2 (Statistical Reality)[/green]")
    
    # Layer 3
    layer3 = BehavioralSignalGenerator()
    console.print("[green]✓ Layer 3 (Signal Generation)[/green]")
    
    # Layer 4
    layer4 = RiskCapitalBrain(config)
    await layer4.setup()
    console.print("[green]✓ Layer 4 (Risk Brain)[/green]")
    
    # Layer 5
    executor = BinanceFuturesExecutor(paper_trading=True)
    await executor.initialize()
    console.print("[green]✓ Layer 5 (Executor - Paper Trading)[/green]")
    
    # Determine symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = list(PERMITTED_PAIRS)
    else:
        symbols = ["cmt_btcusdt"]
    
    # Test each symbol
    all_signals = []
    
    for symbol in symbols:
        console.print(f"\n{'='*60}")
        
        # Test signal generation
        market_state, stat_result, signals = await test_signal_generation(
            symbol, layer1, layer2, layer3
        )
        
        if signals:
            signal = signals[0]
            all_signals.append((symbol, signal))
            
            # Test risk evaluation
            risk_decision = await test_risk_evaluation(
                signal, market_state, stat_result, layer4
            )
            
            # Test execution (paper)
            await test_execution(signal, risk_decision, market_state, executor)
    
    # Summary
    if len(symbols) > 1:
        console.print(f"\n{'='*60}")
        console.print("[bold]Summary[/bold]")
        
        table = Table(show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Signal", style="white")
        table.add_column("Confidence", style="white")
        table.add_column("Source", style="white")
        
        for symbol, signal in all_signals:
            color = "green" if signal.side == Side.LONG else "red"
            table.add_row(
                PAIR_DISPLAY_NAMES.get(symbol, symbol),
                f"[{color}]{signal.side.value.upper()}[/{color}]",
                f"{signal.confidence:.2f}",
                signal.source,
            )
        
        if all_signals:
            console.print(table)
        else:
            console.print("[yellow]No signals generated across all pairs[/yellow]")
    
    # Cleanup
    await layer1.close()
    await executor.close()
    
    console.print("\n[green]✓ Trading pipeline test complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
