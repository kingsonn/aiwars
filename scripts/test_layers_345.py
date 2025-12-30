#!/usr/bin/env python3
"""
Test Script for Layers 3, 4, 5 - Alpha, Risk, and Execution

This tests the complete decision pipeline:
- Layer 3: Alpha & Behavior Modeling (with Groq LLM)
- Layer 4: Risk, Leverage & Capital Brain
- Layer 5: Decision & Execution Engine

Uses Layer 1 data (including news) and Layer 2 trading gate.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.core.types import Side, Position, Regime
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
from hydra.layers.layer2_statistical import (
    StatisticalRealityEngine, TradingDecision, MarketEnvironment
)
from hydra.layers.layer3_alpha import AlphaBehaviorEngine
from hydra.layers.layer4_risk import RiskCapitalBrain, RiskDecision
from hydra.layers.layer5_execution import DecisionExecutionEngine

console = Console()

# Colors for decisions
DECISION_COLORS = {
    TradingDecision.ALLOW: "bold green",
    TradingDecision.RESTRICT: "bold yellow",
    TradingDecision.BLOCK: "bold red",
}

SIDE_COLORS = {
    Side.LONG: "green",
    Side.SHORT: "red",
    Side.FLAT: "yellow",
}


async def test_full_pipeline():
    """Test the complete Layer 3-4-5 pipeline."""
    console.print("\n[bold cyan]=" * 70)
    console.print("[bold white]HYDRA Layer 3-4-5 Integration Test")
    console.print("[bold cyan]=" * 70)
    console.print()
    
    config = HydraConfig()
    
    # Check for Groq API key
    if config.llm.groq_api_key:
        console.print(f"[green]✓ Groq API key configured[/green]")
        console.print(f"  Model: {config.llm.llm_model}")
    else:
        console.print("[yellow]⚠ No Groq API key - LLM will use fallback heuristics[/yellow]")
    
    console.print()
    
    # Initialize all layers
    console.print("[bold]Initializing layers...[/bold]")
    
    layer1 = MarketIntelligenceLayer(config)
    layer2 = StatisticalRealityEngine(config)
    layer3 = AlphaBehaviorEngine(config)
    layer4 = RiskCapitalBrain(config)
    layer5 = DecisionExecutionEngine(config)
    
    await layer1.setup()  # Uses setup() not start_feeds()
    await layer2.setup()
    await layer3.setup()
    await layer4.setup()
    await layer5.setup()
    
    console.print("[green]✓ All layers initialized[/green]\n")
    
    # Process each pair
    results = []
    
    for symbol in PERMITTED_PAIRS:
        display = PAIR_DISPLAY_NAMES.get(symbol, symbol)
        console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        console.print(f"[bold white]Processing: {display}[/bold white]")
        console.print(f"[bold cyan]{'='*70}[/bold cyan]")
        
        try:
            # Layer 1: Get market state
            market_state = await layer1.get_market_state(symbol)
            if not market_state:
                console.print(f"[red]Failed to get market state[/red]")
                continue
            
            console.print(f"[dim]Price: ${market_state.price:,.2f}[/dim]")
            
            # Get news from Layer 1 for LLM context
            news_data = layer1.get_news_sentiment(symbol)
            recent_news = []
            if news_data and hasattr(news_data, 'recent_headlines'):
                recent_news = news_data.recent_headlines[:5]
                console.print(f"[dim]News: {len(recent_news)} recent headlines[/dim]")
            
            # Layer 2: Statistical analysis + Trading Gate
            stat_result = await layer2.analyze(market_state)
            
            decision_color = DECISION_COLORS.get(stat_result.trading_decision, "white")
            console.print(f"\n[bold]Layer 2 Trading Gate:[/bold] [{decision_color}]{stat_result.trading_decision.value.upper()}[/{decision_color}]")
            console.print(f"  Danger Score: {stat_result.danger_score:.0f}/100")
            console.print(f"  Environment: {stat_result.environment.value}")
            
            # Check if Layer 2 blocks trading
            if stat_result.trading_decision == TradingDecision.BLOCK:
                console.print(f"[red]⛔ Layer 2 BLOCKED trading - skipping Layers 3-5[/red]")
                results.append({
                    'symbol': symbol,
                    'l2_decision': stat_result.trading_decision,
                    'l3_signal': None,
                    'l4_decision': None,
                    'l5_approved': False,
                    'final_action': "BLOCKED",
                })
                continue
            
            # Layer 3: Alpha & Behavior (with news from Layer 1)
            console.print(f"\n[bold]Layer 3: Alpha Generation (Groq LLM)...[/bold]")
            
            signals = await layer3.generate_signals(
                market_state=market_state,
                stat_result=stat_result,
                current_position=None,
                recent_news=recent_news,
            )
            
            if not signals:
                console.print("[yellow]No signals generated - HOLD[/yellow]")
                results.append({
                    'symbol': symbol,
                    'l2_decision': stat_result.trading_decision,
                    'l3_signal': None,
                    'l4_decision': None,
                    'l5_approved': False,
                    'final_action': "HOLD",
                })
                continue
            
            signal = signals[0]  # Best signal
            side_color = SIDE_COLORS.get(signal.side, "white")
            
            console.print(f"  Direction: [{side_color}]{signal.side.value.upper()}[/{side_color}]")
            console.print(f"  Confidence: {signal.confidence:.1%}")
            
            if signal.metadata:
                thesis = signal.metadata.get('thesis', '')[:80]
                if thesis:
                    console.print(f"  Thesis: {thesis}...")
                
                if signal.metadata.get('squeeze_opportunity'):
                    console.print(f"  [magenta]🔥 Squeeze opportunity detected![/magenta]")
                if signal.metadata.get('trap_play'):
                    console.print(f"  [cyan]🪤 Trap play detected![/cyan]")
            
            # Layer 4: Risk Assessment
            console.print(f"\n[bold]Layer 4: Risk Assessment...[/bold]")
            
            risk_decision = await layer4.evaluate(
                signal=signal,
                market_state=market_state,
                stat_result=stat_result,
                current_position=None,
                all_positions={},
            )
            
            if risk_decision.veto:
                console.print(f"[red]❌ VETOED: {risk_decision.veto_reason}[/red]")
                results.append({
                    'symbol': symbol,
                    'l2_decision': stat_result.trading_decision,
                    'l3_signal': signal,
                    'l4_decision': risk_decision,
                    'l5_approved': False,
                    'final_action': "VETOED",
                })
                continue
            
            console.print(f"  [green]✓ Approved[/green]")
            console.print(f"  Size: ${risk_decision.recommended_size_usd:,.0f}")
            console.print(f"  Leverage: {risk_decision.recommended_leverage:.1f}x")
            console.print(f"  Stop Loss: ${risk_decision.stop_loss_price:,.2f}")
            console.print(f"  Take Profit: ${risk_decision.take_profit_price:,.2f}")
            console.print(f"  Risk Score: {risk_decision.risk_score:.1%}")
            
            # Layer 5: Multi-Agent Vote
            console.print(f"\n[bold]Layer 5: Multi-Agent Voting...[/bold]")
            
            votes = await layer5.collect_votes(
                signal=signal,
                market_state=market_state,
                stat_result=stat_result,
                risk_decision=risk_decision,
            )
            
            # Display votes
            vote_tree = Tree("Agent Votes")
            for vote in votes:
                status = "🚫 VETO" if vote.veto else f"✓ {vote.action.value.upper()}"
                color = "red" if vote.veto else "green"
                vote_tree.add(f"[{color}]{vote.agent_name}: {status} ({vote.confidence:.0%})[/{color}]")
                if vote.veto:
                    vote_tree.add(f"  [dim]Reason: {vote.veto_reason}[/dim]")
            console.print(vote_tree)
            
            # Evaluate votes
            approved, plan = layer5.evaluate_votes(votes, signal, risk_decision)
            
            if approved and plan:
                console.print(f"\n[bold green]✅ TRADE APPROVED[/bold green]")
                console.print(f"  Action: {plan.target_side.value.upper()}")
                console.print(f"  Size: {plan.target_size:.6f} ({plan.num_slices} slices)")
                console.print(f"  Style: {plan.execution_style.upper()}")
                final_action = f"{plan.target_side.value.upper()}"
            else:
                console.print(f"\n[yellow]Trade not approved - HOLD[/yellow]")
                final_action = "HOLD"
            
            results.append({
                'symbol': symbol,
                'l2_decision': stat_result.trading_decision,
                'l3_signal': signal,
                'l4_decision': risk_decision,
                'l5_approved': approved,
                'final_action': final_action,
            })
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception(f"Error processing {symbol}")
    
    # Summary table
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print("[bold white]Summary: Complete Pipeline Results[/bold white]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    summary_table = Table(show_header=True)
    summary_table.add_column("Symbol", style="cyan", width=10)
    summary_table.add_column("L2 Gate", width=12)
    summary_table.add_column("L3 Signal", width=12)
    summary_table.add_column("L4 Risk", width=12)
    summary_table.add_column("L5 Vote", width=10)
    summary_table.add_column("Final", width=12)
    
    for r in results:
        symbol_display = PAIR_DISPLAY_NAMES.get(r['symbol'], r['symbol']).split("/")[0]
        
        # L2 Gate
        l2_color = DECISION_COLORS.get(r['l2_decision'], "white")
        l2_str = f"[{l2_color}]{r['l2_decision'].value.upper()}[/{l2_color}]"
        
        # L3 Signal
        if r['l3_signal']:
            side_color = SIDE_COLORS.get(r['l3_signal'].side, "white")
            l3_str = f"[{side_color}]{r['l3_signal'].side.value.upper()} {r['l3_signal'].confidence:.0%}[/{side_color}]"
        else:
            l3_str = "[dim]N/A[/dim]"
        
        # L4 Risk
        if r['l4_decision']:
            if r['l4_decision'].veto:
                l4_str = "[red]VETO[/red]"
            else:
                l4_str = f"[green]OK ${r['l4_decision'].recommended_size_usd:.0f}[/green]"
        else:
            l4_str = "[dim]N/A[/dim]"
        
        # L5 Vote
        l5_str = "[green]✓[/green]" if r['l5_approved'] else "[red]✗[/red]"
        
        # Final
        if r['final_action'] in ["LONG", "SHORT"]:
            final_color = "green" if r['final_action'] == "LONG" else "red"
            final_str = f"[bold {final_color}]{r['final_action']}[/bold {final_color}]"
        elif r['final_action'] == "BLOCKED":
            final_str = "[bold red]BLOCKED[/bold red]"
        elif r['final_action'] == "VETOED":
            final_str = "[red]VETOED[/red]"
        else:
            final_str = "[yellow]HOLD[/yellow]"
        
        summary_table.add_row(
            symbol_display,
            l2_str,
            l3_str,
            l4_str,
            l5_str,
            final_str,
        )
    
    console.print(summary_table)
    
    # Stats
    total = len(results)
    blocked = sum(1 for r in results if r['final_action'] == "BLOCKED")
    vetoed = sum(1 for r in results if r['final_action'] == "VETOED")
    holding = sum(1 for r in results if r['final_action'] == "HOLD")
    trading = sum(1 for r in results if r['final_action'] in ["LONG", "SHORT"])
    
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Total pairs: {total}")
    console.print(f"  Blocked by L2: {blocked}")
    console.print(f"  Vetoed by L4/L5: {vetoed}")
    console.print(f"  Holding: {holding}")
    console.print(f"  Trading: {trading}")
    
    # Cleanup
    await layer1.stop_feeds()
    
    console.print("\n[green]✓ Layer 3-4-5 test complete![/green]")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
