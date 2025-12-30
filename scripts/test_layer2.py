#!/usr/bin/env python3
"""
Test Layer 2: Statistical Reality Engine

Tests statistical analysis for all 8 pairs:
- Regime detection (trending, ranging, cascade risk)
- Volatility analysis (realized vol, z-score, regime)
- Expected price ranges (1h, 4h, 24h)
- Abnormal move detection
- Jump probability (liquidation events)
- Cascade probability (Hawkes process)
- Distribution metrics (skewness, kurtosis, tail risk)

Usage:
    python scripts/test_layer2.py
    python scripts/test_layer2.py --symbol cmt_btcusdt
    python scripts/test_layer2.py --all
"""

import argparse
import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
from hydra.layers.layer2_statistical import (
    StatisticalRealityEngine, StatisticalResult, 
    TradingDecision, MarketEnvironment
)
from hydra.core.types import Regime

console = Console()

# Trading decision colors and icons
DECISION_COLORS = {
    TradingDecision.ALLOW: "bold green",
    TradingDecision.RESTRICT: "bold yellow",
    TradingDecision.BLOCK: "bold red",
}

DECISION_ICONS = {
    TradingDecision.ALLOW: "✅",
    TradingDecision.RESTRICT: "⚠️",
    TradingDecision.BLOCK: "⛔",
}

# Environment colors
ENV_COLORS = {
    MarketEnvironment.NORMAL: "green",
    MarketEnvironment.TRENDING: "cyan",
    MarketEnvironment.VOLATILE: "yellow",
    MarketEnvironment.CHAOTIC: "red",
    MarketEnvironment.SQUEEZE: "bold red",
}

# Regime colors
REGIME_COLORS = {
    Regime.TRENDING_UP: "green",
    Regime.TRENDING_DOWN: "red",
    Regime.RANGING: "yellow",
    Regime.HIGH_VOLATILITY: "magenta",
    Regime.CASCADE_RISK: "bold red",
    Regime.UNKNOWN: "white",
}

REGIME_ICONS = {
    Regime.TRENDING_UP: "📈",
    Regime.TRENDING_DOWN: "📉",
    Regime.RANGING: "↔️",
    Regime.HIGH_VOLATILITY: "⚡",
    Regime.CASCADE_RISK: "🚨",
    Regime.UNKNOWN: "❓",
}


def format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_price(value: float) -> str:
    """Format price."""
    if value >= 1000:
        return f"${value:,.2f}"
    elif value >= 1:
        return f"${value:.4f}"
    else:
        return f"${value:.6f}"


def print_statistical_result(result: StatisticalResult, current_price: float):
    """Print comprehensive statistical analysis result."""
    display = PAIR_DISPLAY_NAMES.get(result.symbol, result.symbol)
    
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold white]{display}[/bold white] - Statistical Reality Analysis")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    # =========================================================================
    # TRADING DECISION GATE (PRIMARY OUTPUT)
    # =========================================================================
    decision_color = DECISION_COLORS.get(result.trading_decision, "white")
    decision_icon = DECISION_ICONS.get(result.trading_decision, "")
    env_color = ENV_COLORS.get(result.environment, "white")
    
    decision_content = f"[{decision_color}]{decision_icon} {result.trading_decision.value.upper()}[/{decision_color}]"
    if result.trading_decision == TradingDecision.RESTRICT:
        decision_content += f" (Max Position: {result.max_position_pct:.0f}%)"
    
    gate_content = f"[bold]Decision:[/bold] {decision_content}\n"
    gate_content += f"[bold]Environment:[/bold] [{env_color}]{result.environment.value.upper()}[/{env_color}]\n"
    gate_content += f"[bold]Danger Score:[/bold] {result.danger_score:.0f}/100\n"
    
    if result.decision_reasons:
        gate_content += "\n[bold]Reasons:[/bold]"
        for reason in result.decision_reasons:
            gate_content += f"\n  - {reason}"
    
    gate_panel = Panel(
        gate_content,
        title="🚦 TRADING GATE",
        border_style=decision_color.split()[-1] if " " in decision_color else decision_color
    )
    console.print(gate_panel)
    console.print()
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    regime_color = REGIME_COLORS.get(result.regime, "white")
    regime_icon = REGIME_ICONS.get(result.regime, "")
    
    regime_panel = Panel(
        f"[bold]Current Regime:[/bold] [{regime_color}]{regime_icon} {result.regime.name}[/{regime_color}]\n"
        f"[bold]Confidence:[/bold] {result.regime_confidence:.0%}\n"
        f"[bold]Regime Break Alert:[/bold] {'🚨 YES' if result.regime_break_alert else 'No'}",
        title="Market Regime",
        border_style=regime_color
    )
    console.print(regime_panel)
    console.print()
    
    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================
    vol_table = Table(title="📊 Volatility Analysis", show_header=True)
    vol_table.add_column("Metric", style="cyan", width=25)
    vol_table.add_column("Value", style="green", width=20)
    vol_table.add_column("Interpretation", style="yellow", width=25)
    
    vol_color = "red" if result.volatility_regime == "extreme" else "magenta" if result.volatility_regime == "high" else "green" if result.volatility_regime == "low" else "white"
    vol_table.add_row("Realized Volatility", f"{result.realized_volatility:.1%}", f"Annualized")
    vol_table.add_row("Implied Volatility", f"{result.implied_volatility:.1%}", "")
    vol_table.add_row("Volatility Regime", f"[{vol_color}]{result.volatility_regime.upper()}[/{vol_color}]", f"Z-score: {result.volatility_zscore:.2f}")
    
    console.print(vol_table)
    console.print()
    
    # =========================================================================
    # EXPECTED PRICE RANGES
    # =========================================================================
    range_table = Table(title="🎯 Expected Price Ranges (95% Confidence)", show_header=True)
    range_table.add_column("Timeframe", style="cyan", width=15)
    range_table.add_column("Low", style="red", width=18)
    range_table.add_column("High", style="green", width=18)
    range_table.add_column("Range %", style="yellow", width=15)
    
    for tf, (low, high) in [("1 Hour", result.expected_range_1h), 
                             ("4 Hours", result.expected_range_4h),
                             ("24 Hours", result.expected_range_24h)]:
        if low > 0 and high > 0:
            range_pct = (high - low) / current_price * 100
            range_table.add_row(tf, format_price(low), format_price(high), f"±{range_pct/2:.2f}%")
        else:
            range_table.add_row(tf, "N/A", "N/A", "N/A")
    
    console.print(range_table)
    console.print()
    
    # =========================================================================
    # ABNORMAL MOVE DETECTION
    # =========================================================================
    abnormal_table = Table(title="⚠️ Abnormal Move Detection", show_header=True)
    abnormal_table.add_column("Metric", style="cyan", width=25)
    abnormal_table.add_column("Value", style="green", width=20)
    abnormal_table.add_column("Status", style="yellow", width=25)
    
    abnormal_color = "red" if result.is_abnormal else "green"
    abnormal_table.add_row("Move Z-Score", f"{result.abnormal_move_score:.2f}", 
                          f"[{abnormal_color}]{'🚨 ABNORMAL' if result.is_abnormal else '✅ Normal'}[/{abnormal_color}]")
    
    if abs(result.abnormal_move_score) > 2:
        direction = "UP" if result.abnormal_move_score > 0 else "DOWN"
        abnormal_table.add_row("Direction", direction, f"{abs(result.abnormal_move_score):.1f}σ move")
    
    console.print(abnormal_table)
    console.print()
    
    # =========================================================================
    # JUMP & CASCADE RISK
    # =========================================================================
    risk_table = Table(title="🔥 Jump & Cascade Risk", show_header=True)
    risk_table.add_column("Metric", style="cyan", width=25)
    risk_table.add_column("Value", style="green", width=20)
    risk_table.add_column("Risk Level", style="yellow", width=25)
    
    # Jump probability
    jump_color = "red" if result.jump_probability > 0.3 else "yellow" if result.jump_probability > 0.1 else "green"
    risk_table.add_row("Jump Probability (1h)", f"[{jump_color}]{result.jump_probability:.1%}[/{jump_color}]",
                      "HIGH" if result.jump_probability > 0.3 else "MEDIUM" if result.jump_probability > 0.1 else "LOW")
    
    risk_table.add_row("Expected Jump Size", f"{result.expected_jump_size:.2%}", "If jump occurs")
    risk_table.add_row("Jump Intensity", f"{result.jump_intensity:.3f}", "Hawkes λ(t)")
    
    # Cascade probability
    cascade_color = "red" if result.cascade_probability > 0.3 else "yellow" if result.cascade_probability > 0.1 else "green"
    risk_table.add_row("Cascade Probability", f"[{cascade_color}]{result.cascade_probability:.1%}[/{cascade_color}]",
                      "🚨 HIGH RISK" if result.cascade_probability > 0.3 else "⚠️ ELEVATED" if result.cascade_probability > 0.1 else "✅ LOW")
    
    risk_table.add_row("Liquidation Velocity", f"{result.liquidation_velocity:.3f}", "Events/period")
    
    console.print(risk_table)
    console.print()
    
    # =========================================================================
    # DISTRIBUTION METRICS
    # =========================================================================
    dist_table = Table(title="📈 Distribution Metrics", show_header=True)
    dist_table.add_column("Metric", style="cyan", width=25)
    dist_table.add_column("Value", style="green", width=20)
    dist_table.add_column("Interpretation", style="yellow", width=25)
    
    # Skewness
    skew_interp = "Left tail (crash risk)" if result.skewness < -0.5 else "Right tail (squeeze risk)" if result.skewness > 0.5 else "Symmetric"
    dist_table.add_row("Skewness", f"{result.skewness:.3f}", skew_interp)
    
    # Kurtosis
    kurt_interp = "Fat tails (extreme moves likely)" if result.kurtosis > 3 else "Thin tails" if result.kurtosis < 0 else "Normal tails"
    dist_table.add_row("Excess Kurtosis", f"{result.kurtosis:.3f}", kurt_interp)
    
    # Tail risk
    tail_color = "red" if result.tail_risk_score > 0.03 else "yellow" if result.tail_risk_score > 0.01 else "green"
    dist_table.add_row("Tail Risk Score", f"[{tail_color}]{result.tail_risk_score:.4f}[/{tail_color}]", 
                      "CVaR proxy")
    
    console.print(dist_table)
    console.print()
    
    # =========================================================================
    # SUMMARY PANEL
    # =========================================================================
    # Calculate overall danger score
    danger_score = 0
    danger_factors = []
    
    if result.regime == Regime.CASCADE_RISK:
        danger_score += 40
        danger_factors.append("CASCADE RISK regime")
    elif result.regime == Regime.HIGH_VOLATILITY:
        danger_score += 20
        danger_factors.append("High volatility regime")
    
    if result.is_abnormal:
        danger_score += 25
        danger_factors.append(f"Abnormal move ({result.abnormal_move_score:.1f}σ)")
    
    if result.jump_probability > 0.3:
        danger_score += 15
        danger_factors.append(f"High jump probability ({result.jump_probability:.0%})")
    
    if result.cascade_probability > 0.3:
        danger_score += 20
        danger_factors.append(f"Cascade risk ({result.cascade_probability:.0%})")
    
    if result.volatility_regime == "extreme":
        danger_score += 15
        danger_factors.append("Extreme volatility")
    
    if result.kurtosis > 5:
        danger_score += 10
        danger_factors.append("Fat tails detected")
    
    danger_color = "red" if danger_score >= 50 else "yellow" if danger_score >= 25 else "green"
    
    summary_content = f"[bold]Danger Score:[/bold] [{danger_color}]{danger_score}/100[/{danger_color}]\n"
    summary_content += f"[bold]Status:[/bold] "
    
    if danger_score >= 50:
        summary_content += "[bold red]⛔ HIGH DANGER - Reduce exposure[/bold red]"
    elif danger_score >= 25:
        summary_content += "[yellow]⚠️ ELEVATED RISK - Trade with caution[/yellow]"
    else:
        summary_content += "[green]✅ NORMAL CONDITIONS[/green]"
    
    if danger_factors:
        summary_content += "\n\n[bold]Risk Factors:[/bold]"
        for factor in danger_factors:
            summary_content += f"\n  - {factor}"
    
    summary_panel = Panel(
        summary_content,
        title="Statistical Reality Summary",
        border_style=danger_color
    )
    console.print(summary_panel)


async def main():
    parser = argparse.ArgumentParser(description="Test Layer 2 Statistical Reality Engine")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    parser.add_argument("--all", action="store_true", help="Test all pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Layer 2 Test[/bold cyan]\n"
        "Statistical Reality Engine\n"
        "Testing: Regime, Volatility, Ranges, Jumps, Cascades",
        border_style="cyan"
    ))
    
    # Load config
    config = HydraConfig.load()
    
    # Initialize Layer 1 (for market data)
    console.print("\n[cyan]Initializing Layer 1 (Market Intelligence)...[/cyan]")
    layer1 = MarketIntelligenceLayer(config)
    await layer1.setup()
    console.print("[green]✓ Layer 1 ready[/green]")
    
    # Initialize Layer 2
    console.print("[cyan]Initializing Layer 2 (Statistical Reality Engine)...[/cyan]")
    layer2 = StatisticalRealityEngine(config)
    await layer2.setup()
    console.print("[green]✓ Layer 2 ready[/green]\n")
    
    # Determine which symbols to test
    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = list(PERMITTED_PAIRS)
    else:
        symbols = ["cmt_btcusdt"]  # Default
    
    # Test each symbol
    results = []
    for symbol in symbols:
        console.print(f"[dim]Analyzing {PAIR_DISPLAY_NAMES.get(symbol, symbol)}...[/dim]")
        
        try:
            # Get market state from Layer 1
            market_state = await layer1.get_market_state(symbol)
            
            if market_state:
                # Run Layer 2 analysis
                result = await layer2.analyze(market_state)
                results.append((result, market_state.price))
                print_statistical_result(result, market_state.price)
            else:
                console.print(f"[red]Failed to get market state for {symbol}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
            logger.exception(f"Layer 2 analysis error for {symbol}")
    
    # Summary table for all pairs
    if len(results) > 1:
        console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        console.print("[bold white]Summary: All Pairs - Trading Gate Status[/bold white]")
        console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
        
        summary_table = Table(show_header=True)
        summary_table.add_column("Symbol", style="cyan", width=10)
        summary_table.add_column("Decision", style="white", width=12)
        summary_table.add_column("Environment", style="white", width=12)
        summary_table.add_column("Max Pos", style="white", width=8)
        summary_table.add_column("Danger", style="white", width=8)
        summary_table.add_column("Regime", style="white", width=14)
        
        for result, price in results:
            decision_color = DECISION_COLORS.get(result.trading_decision, "white")
            decision_icon = DECISION_ICONS.get(result.trading_decision, "")
            env_color = ENV_COLORS.get(result.environment, "white")
            regime_color = REGIME_COLORS.get(result.regime, "white")
            danger_color = "red" if result.danger_score >= 70 else "yellow" if result.danger_score >= 40 else "green"
            
            symbol_display = PAIR_DISPLAY_NAMES.get(result.symbol, result.symbol).split("/")[0]
            
            summary_table.add_row(
                symbol_display,
                f"[{decision_color}]{decision_icon} {result.trading_decision.value.upper()}[/{decision_color}]",
                f"[{env_color}]{result.environment.value.upper()}[/{env_color}]",
                f"{result.max_position_pct:.0f}%",
                f"[{danger_color}]{result.danger_score:.0f}[/{danger_color}]",
                f"[{regime_color}]{result.regime.name}[/{regime_color}]",
            )
        
        console.print(summary_table)
    
    # Cleanup
    await layer1.stop_feeds()
    
    console.print("\n[green]✓ Layer 2 test complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
