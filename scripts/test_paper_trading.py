"""
HYDRA Paper Trading Test Script

Tests the enhanced paper trading system with:
- Dual mode operation (Entry vs Position Management)
- Dynamic leverage (up to 20x)
- Thesis tracking and exit logic
- Position management (HOLD/EXIT only for held positions)

Usage:
    python scripts/test_paper_trading.py --balance 1000 --cycles 5
"""

import asyncio
import argparse
from datetime import datetime, timezone
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

# Configure logging
import sys
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

console = Console()


async def test_paper_trading(initial_balance: float, num_cycles: int, cycle_interval: int):
    """Run paper trading test."""
    from hydra.core.config import HydraConfig, PAIR_DISPLAY_NAMES
    from hydra.core.types import Side
    from hydra.paper_trading.engine import PaperTradingEngine
    from hydra.core.position_manager import OperatingMode
    from hydra.layers.layer2_statistical import TradingDecision
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Paper Trading Test[/bold cyan]\n"
        f"Initial Balance: ${initial_balance:,.2f}\n"
        f"Cycles: {num_cycles}\n"
        f"Interval: {cycle_interval}s",
        title="🐍 HYDRA"
    ))
    
    # Load config
    config = HydraConfig.load()
    console.print(f"\n[green]✓[/green] Config loaded")
    console.print(f"  Max Leverage: {config.risk.max_leverage}x")
    console.print(f"  Max Position Size: ${config.trading.max_position_size_usd:,.0f}")
    console.print(f"  Risk Per Trade: {config.risk.risk_per_trade_pct}%")
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(
        config=config,
        initial_balance=initial_balance,
        data_dir="./data/paper_trading_test",
    )
    
    # Setup HYDRA
    console.print("\n[yellow]Setting up HYDRA layers...[/yellow]")
    await engine.hydra.setup()
    console.print("[green]✓[/green] All layers initialized")
    
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
    
    # Run cycles
    console.print(f"\n[bold]Running {num_cycles} trading cycles...[/bold]\n")
    
    for cycle in range(1, num_cycles + 1):
        cycle_start = datetime.now(timezone.utc)
        console.rule(f"[bold]Cycle {cycle}/{num_cycles}[/bold]")
        
        # Collect pipeline results for summary table
        cycle_results = []
        
        # Process each symbol
        for symbol in config.trading.symbols:
            display_name = PAIR_DISPLAY_NAMES.get(symbol, symbol)
            result = {
                'symbol': symbol,
                'display': display_name.split("/")[0],
                'mode': 'ENTRY',
                'l2_decision': None,
                'l3_signal': None,
                'l3_confidence': 0.0,
                'l4_approved': False,
                'l4_size': 0.0,
                'l4_leverage': 0.0,
                'l4_veto_reason': '',
                'l5_approved': False,
                'final_action': 'FLAT',
                'pnl_pct': 0.0,
            }
            
            try:
                # Get current position
                position = engine._get_position_as_type(symbol)
                mode = engine.position_manager.get_operating_mode(symbol, position)
                result['mode'] = mode.value.upper()
                
                if position:
                    result['pnl_pct'] = position.unrealized_pnl_pct
                
                # Get market state
                market_state = await engine.hydra.layer1.get_market_state(symbol)
                if not market_state:
                    result['final_action'] = 'NO DATA'
                    cycle_results.append(result)
                    continue
                
                # Update price cache
                engine._current_prices[symbol] = market_state.price
                
                # Layer 2: Statistical Analysis
                stat_result = await engine.hydra.layer2.analyze(market_state)
                result['l2_decision'] = stat_result.trading_decision
                
                # Check L2 safety for positions
                force_exit, force_reason = engine.position_manager.check_layer2_safety(
                    stat_result, position
                )
                
                if force_exit and position:
                    result['final_action'] = 'FORCE EXIT'
                    engine.portfolio.close_position(symbol, market_state.price, "force_exit_l2_block")
                    engine.position_manager.clear_position_context(symbol)
                    cycle_results.append(result)
                    continue
                
                # POSITION MANAGEMENT MODE
                if mode == OperatingMode.MANAGEMENT and position:
                    context = engine.position_manager.get_position_context(symbol)
                    thesis_status = engine.position_manager.check_thesis_validity(
                        position, market_state, stat_result, context
                    )
                    decision = engine.position_manager.evaluate_exit_decision(
                        position, market_state, stat_result, thesis_status, context
                    )
                    
                    result['l3_signal'] = Side.LONG if position.side == Side.LONG else Side.SHORT
                    result['l3_confidence'] = 0.0  # Thesis check, not signal
                    
                    from hydra.core.types import PositionAction
                    if decision.action in [PositionAction.EXIT, PositionAction.FORCE_EXIT]:
                        result['final_action'] = 'EXIT'
                        engine.portfolio.close_position(
                            symbol, market_state.price, decision.exit_reason,
                            partial_pct=decision.exit_pct
                        )
                        if decision.exit_pct >= 0.99:
                            engine.position_manager.clear_position_context(symbol)
                    else:
                        result['final_action'] = 'HOLD'
                    
                    cycle_results.append(result)
                    continue
                
                # ENTRY MODE
                if stat_result.trading_decision == TradingDecision.BLOCK:
                    result['final_action'] = 'BLOCKED'
                    cycle_results.append(result)
                    continue
                
                # Layer 3: Alpha Generation
                recent_news = []
                news_sentiment = engine.hydra.layer1.get_news_sentiment(symbol)
                if news_sentiment and news_sentiment.breaking_news:
                    recent_news = [n.title for n in news_sentiment.breaking_news[:5]]
                
                signals = await engine.hydra.layer3.generate_signals(
                    market_state, stat_result, None, recent_news
                )
                
                if not signals:
                    result['final_action'] = 'NO SIGNAL'
                    cycle_results.append(result)
                    continue
                
                best_signal = max(signals, key=lambda s: s.confidence)
                result['l3_signal'] = best_signal.side
                result['l3_confidence'] = best_signal.confidence
                
                if best_signal.confidence < config.risk.min_confidence_threshold:
                    result['final_action'] = 'LOW CONF'
                    cycle_results.append(result)
                    continue
                
                # Calculate leverage
                leverage_decision = engine.position_manager.calculate_leverage(
                    best_signal.confidence, stat_result, market_state,
                    engine.portfolio.current_drawdown
                )
                result['l4_leverage'] = leverage_decision.leverage
                
                # Calculate size
                all_positions = engine._get_all_positions_as_type()
                stop_loss_pct = best_signal.expected_adverse_excursion or 0.02
                size_usd = engine.position_manager.calculate_position_size(
                    total_equity=engine.portfolio.total_equity,
                    signal_confidence=best_signal.confidence,
                    leverage=leverage_decision.leverage,
                    stop_loss_pct=stop_loss_pct,
                    max_positions=config.trading.max_positions,
                    current_positions=len(all_positions),
                )
                result['l4_size'] = size_usd
                
                if size_usd < 50:
                    result['final_action'] = 'SIZE TOO SMALL'
                    cycle_results.append(result)
                    continue
                
                # Layer 4: Risk Evaluation
                from hydra.core.types import Signal
                signal = Signal(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    side=best_signal.side,
                    confidence=best_signal.confidence,
                    expected_return=0.02,
                    expected_adverse_excursion=best_signal.expected_adverse_excursion,
                    holding_period_minutes=int(best_signal.expected_holding_period_hours * 60),
                    source="hydra_alpha",
                    regime=stat_result.regime,
                    metadata={'thesis': best_signal.metadata.get('thesis', '') if best_signal.metadata else ''},
                )
                
                risk_decision = await engine.hydra.layer4.evaluate(
                    signal, market_state, stat_result, position, all_positions
                )
                
                if risk_decision.trigger_kill_switch:
                    result['final_action'] = 'KILL SWITCH'
                    cycle_results.append(result)
                    continue
                
                result['l4_approved'] = risk_decision.approved
                if not risk_decision.approved:
                    result['l4_veto_reason'] = risk_decision.veto_reason or ''
                    result['final_action'] = 'L4 VETO'
                    cycle_results.append(result)
                    continue
                
                # Layer 5: Multi-Agent Voting
                votes = await engine.hydra.layer5.collect_votes(
                    signal, market_state, stat_result, risk_decision
                )
                approved, plan = engine.hydra.layer5.evaluate_votes(votes, signal, risk_decision)
                result['l5_approved'] = approved
                
                if not approved:
                    result['final_action'] = 'L5 VETO'
                    cycle_results.append(result)
                    continue
                
                # Execute trade
                result['final_action'] = signal.side.value.upper()
                await engine._execute_trade(
                    symbol=symbol,
                    side=signal.side,
                    size_usd=size_usd,
                    price=market_state.price,
                    leverage=leverage_decision.leverage,
                    signal=signal,
                )
                
            except Exception as e:
                result['final_action'] = f'ERROR'
                logger.error(f"Error processing {symbol}: {e}")
            
            cycle_results.append(result)
        
        # Update portfolio prices
        engine.portfolio.update_prices(engine._current_prices)
        
        # Show pipeline summary table
        show_pipeline_summary(cycle_results, DECISION_COLORS, SIDE_COLORS)
        
        # Show portfolio status
        show_portfolio_status(engine)
        
        # Wait for next cycle (except on last cycle)
        if cycle < num_cycles:
            console.print(f"\n[dim]Waiting {cycle_interval}s for next cycle...[/dim]")
            await asyncio.sleep(cycle_interval)
    
    # Final summary
    console.print("\n")
    console.rule("[bold green]Final Summary[/bold green]")
    show_final_summary(engine)
    
    # Save state
    engine.portfolio.save()
    console.print("\n[green]✓[/green] Portfolio state saved")
    
    # Cleanup
    await engine.hydra.stop()
    console.print("[green]✓[/green] Paper trading test complete!")


def show_pipeline_summary(results: list, decision_colors: dict, side_colors: dict):
    """Display pipeline summary table for all symbols."""
    from hydra.layers.layer2_statistical import TradingDecision
    from hydra.core.types import Side
    
    console.print(f"\n[bold cyan]{'─'*80}[/bold cyan]")
    console.print("[bold white]Pipeline Summary[/bold white]")
    console.print(f"[bold cyan]{'─'*80}[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Symbol", style="cyan", width=8)
    table.add_column("Mode", width=6)
    table.add_column("L2 Gate", width=10)
    table.add_column("L3 Signal", width=12)
    table.add_column("L4 Risk", width=14)
    table.add_column("L5 Vote", width=8)
    table.add_column("Final", width=12)
    
    for r in results:
        # Mode
        mode_color = "yellow" if r['mode'] == "MANAGEMENT" else "blue"
        mode_str = f"[{mode_color}]{r['mode'][:4]}[/{mode_color}]"
        
        # L2 Gate
        if r['l2_decision']:
            l2_color = decision_colors.get(r['l2_decision'], "white")
            l2_str = f"[{l2_color}]{r['l2_decision'].value.upper()}[/{l2_color}]"
        else:
            l2_str = "[dim]N/A[/dim]"
        
        # L3 Signal
        if r['l3_signal']:
            side_color = side_colors.get(r['l3_signal'], "white")
            if r['l3_confidence'] > 0:
                l3_str = f"[{side_color}]{r['l3_signal'].value.upper()} {r['l3_confidence']:.0%}[/{side_color}]"
            else:
                l3_str = f"[{side_color}]{r['l3_signal'].value.upper()}[/{side_color}]"
        else:
            l3_str = "[dim]N/A[/dim]"
        
        # L4 Risk
        if r['l4_size'] > 0:
            l4_str = f"[green]${r['l4_size']:.0f} @ {r['l4_leverage']:.1f}x[/green]"
        elif r['l4_veto_reason']:
            l4_str = f"[red]VETO[/red]"
        elif r['l4_approved']:
            l4_str = f"[green]OK[/green]"
        else:
            l4_str = "[dim]N/A[/dim]"
        
        # L5 Vote
        if r['final_action'] in ['LONG', 'SHORT', 'HOLD', 'EXIT']:
            l5_str = "[green]✓[/green]" if r['l5_approved'] else "[dim]—[/dim]"
        elif r['final_action'] == 'L5 VETO':
            l5_str = "[red]✗[/red]"
        else:
            l5_str = "[dim]—[/dim]"
        
        # Final action with color
        action = r['final_action']
        if action == 'LONG':
            final_str = f"[bold green]📈 LONG[/bold green]"
        elif action == 'SHORT':
            final_str = f"[bold red]📉 SHORT[/bold red]"
        elif action == 'HOLD':
            pnl_color = "green" if r['pnl_pct'] >= 0 else "red"
            final_str = f"[yellow]HOLD[/yellow] [{pnl_color}]{r['pnl_pct']:+.1%}[/{pnl_color}]"
        elif action == 'EXIT':
            final_str = f"[magenta]📤 EXIT[/magenta]"
        elif action == 'FORCE EXIT':
            final_str = f"[bold red]🚨 FORCE EXIT[/bold red]"
        elif action == 'BLOCKED':
            final_str = f"[bold red]🚫 BLOCKED[/bold red]"
        elif action in ['L4 VETO', 'L5 VETO']:
            final_str = f"[red]VETOED[/red]"
        elif action == 'LOW CONF':
            final_str = f"[yellow]LOW CONF[/yellow]"
        else:
            final_str = f"[dim]{action}[/dim]"
        
        table.add_row(
            r['display'],
            mode_str,
            l2_str,
            l3_str,
            l4_str,
            l5_str,
            final_str,
        )
    
    console.print(table)
    
    # Stats
    total = len(results)
    blocked = sum(1 for r in results if r['final_action'] == 'BLOCKED')
    vetoed = sum(1 for r in results if 'VETO' in r['final_action'])
    holding = sum(1 for r in results if r['final_action'] == 'HOLD')
    trading = sum(1 for r in results if r['final_action'] in ['LONG', 'SHORT'])
    exiting = sum(1 for r in results if 'EXIT' in r['final_action'])
    
    console.print(f"\n[bold]Stats:[/bold] {total} pairs | "
                  f"[red]Blocked: {blocked}[/red] | "
                  f"[yellow]Vetoed: {vetoed}[/yellow] | "
                  f"Holding: {holding} | "
                  f"[green]Trading: {trading}[/green] | "
                  f"[magenta]Exiting: {exiting}[/magenta]")


def show_portfolio_status(engine):
    """Display current portfolio status."""
    portfolio = engine.portfolio
    
    # Create positions table
    positions = portfolio.get_open_positions()
    
    if positions:
        table = Table(title="Open Positions", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Leverage", justify="right")
        
        for pos in positions:
            pnl_color = "green" if pos["unrealized_pnl"] >= 0 else "red"
            pnl_str = f"[{pnl_color}]{pos['unrealized_pnl']:+.2f} ({pos['unrealized_pnl_pct']:+.1%})[/{pnl_color}]"
            
            table.add_row(
                pos["symbol"],
                pos["side"].upper(),
                f"${pos['size_usd']:,.0f}",
                f"${pos['entry_price']:,.2f}",
                f"${pos['current_price']:,.2f}",
                pnl_str,
                f"{pos['leverage']:.1f}x",
            )
        
        console.print(table)
    
    # Portfolio summary
    summary = portfolio.get_summary()
    pnl = summary["total_pnl"]
    pnl_pct = summary["total_pnl_pct"]
    pnl_color = "green" if pnl >= 0 else "red"
    
    console.print(
        f"\n  💰 Equity: ${summary['total_equity']:,.2f} | "
        f"[{pnl_color}]P&L: {pnl:+.2f} ({pnl_pct:+.2%})[/{pnl_color}] | "
        f"Positions: {summary['num_positions']} | "
        f"Leverage: {summary['gross_leverage']:.1f}x"
    )


def show_final_summary(engine):
    """Display final trading summary."""
    summary = engine.portfolio.get_summary()
    trades = engine.portfolio.get_trade_history(limit=10)
    
    # Summary panel
    pnl = summary["total_pnl"]
    pnl_pct = summary["total_pnl_pct"]
    pnl_color = "green" if pnl >= 0 else "red"
    
    summary_text = f"""
[bold]Portfolio Performance[/bold]
━━━━━━━━━━━━━━━━━━━━━━
Initial Balance:  ${summary['initial_balance']:,.2f}
Final Equity:     ${summary['total_equity']:,.2f}
[{pnl_color}]Total P&L:        {pnl:+.2f} ({pnl_pct:+.2%})[/{pnl_color}]

[bold]Trading Stats[/bold]
━━━━━━━━━━━━━━━━━━━━━━
Total Trades:     {summary['total_trades']}
Win Rate:         {summary['win_rate']:.1%}
Max Drawdown:     {summary['current_drawdown']:.2%}

[bold]Exposure[/bold]
━━━━━━━━━━━━━━━━━━━━━━
Gross Exposure:   ${summary['gross_exposure']:,.2f}
Net Exposure:     ${summary['net_exposure']:+,.2f}
Gross Leverage:   {summary['gross_leverage']:.1f}x
"""
    console.print(Panel(summary_text, title="📊 Summary"))
    
    # Recent trades
    if trades:
        table = Table(title="Recent Trades", show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Time")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Action")
        table.add_column("Price", justify="right")
        table.add_column("P&L", justify="right")
        
        for t in trades[:10]:
            pnl_color = "green" if t["pnl"] >= 0 else "red" if t["pnl"] < 0 else "dim"
            pnl_str = f"[{pnl_color}]{t['pnl']:+.2f}[/{pnl_color}]" if t["pnl"] != 0 else "-"
            
            table.add_row(
                t["id"],
                t["time"][:19],
                t["symbol"],
                t["side"].upper(),
                t["action"],
                f"${t['price']:,.2f}",
                pnl_str,
            )
        
        console.print(table)


async def test_position_management():
    """Test position management mode specifically."""
    from hydra.core.config import HydraConfig
    from hydra.core.position_manager import PositionManager, OperatingMode
    from hydra.core.types import Position, Side, MarketState, Regime
    from hydra.layers.layer2_statistical import StatisticalResult, TradingDecision, MarketEnvironment
    
    console.print(Panel.fit(
        "[bold cyan]Position Management Mode Test[/bold cyan]",
        title="🧪 Test"
    ))
    
    config = HydraConfig.load()
    pm = PositionManager(config)
    
    # Create mock position
    position = Position(
        symbol="cmt_btcusdt",
        side=Side.LONG,
        size=0.01,
        size_usd=870.0,
        entry_price=87000.0,
        current_price=87500.0,
        leverage=5.0,
        unrealized_pnl=5.0,
        unrealized_pnl_pct=0.0057,
        entry_time=datetime.now(timezone.utc),
    )
    
    # Store context
    pm.store_position_context(
        symbol="cmt_btcusdt",
        position=position,
        thesis="LONG: Trap play - shorts trapped",
        signal_source="hydra_alpha",
        confidence=0.65,
        funding_rate=0.0001,
    )
    
    # Test mode detection
    mode = pm.get_operating_mode("cmt_btcusdt", position)
    console.print(f"Operating Mode: {mode.value}")
    assert mode == OperatingMode.MANAGEMENT, "Should be in MANAGEMENT mode"
    
    # Create mock stat result
    stat_result = StatisticalResult(
        timestamp=datetime.now(timezone.utc),
        symbol="cmt_btcusdt",
        trading_decision=TradingDecision.ALLOW,
        environment=MarketEnvironment.NORMAL,
        regime=Regime.RANGING,
        regime_confidence=0.7,
        volatility_regime="normal",
        volatility_zscore=0.5,
        realized_volatility=0.02,
        is_abnormal=False,
        abnormal_move_score=0.0,
        cascade_probability=0.1,
        liquidation_velocity=0.0,
        regime_break_alert=False,
    )
    
    # Create mock market state
    market_state = MarketState(
        timestamp=datetime.now(timezone.utc),
        symbol="cmt_btcusdt",
        price=87500.0,
    )
    
    # Test thesis check
    context = pm.get_position_context("cmt_btcusdt")
    thesis_status = pm.check_thesis_validity(position, market_state, stat_result, context)
    console.print(f"Thesis Status: {thesis_status.value}")
    
    # Test exit decision
    decision = pm.evaluate_exit_decision(position, market_state, stat_result, thesis_status, context)
    console.print(f"Action: {decision.action.value}")
    console.print(f"Reasoning: {decision.reasoning}")
    
    # Test leverage calculation
    leverage = pm.calculate_leverage(
        signal_confidence=0.7,
        stat_result=stat_result,
        market_state=market_state,
        portfolio_drawdown=0.02,
    )
    console.print(f"Calculated Leverage: {leverage.leverage:.1f}x")
    console.print(f"Reasoning: {leverage.reasoning}")
    
    console.print("\n[green]✓[/green] Position management tests passed!")


def main():
    parser = argparse.ArgumentParser(description="HYDRA Paper Trading Test")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance in USDT")
    parser.add_argument("--cycles", type=int, default=3, help="Number of trading cycles")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    parser.add_argument("--test-pm", action="store_true", help="Test position management only")
    
    args = parser.parse_args()
    
    if args.test_pm:
        asyncio.run(test_position_management())
    else:
        asyncio.run(test_paper_trading(args.balance, args.cycles, args.interval))


if __name__ == "__main__":
    main()
