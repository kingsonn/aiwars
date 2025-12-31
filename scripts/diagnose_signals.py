"""
Diagnostic script to understand why no signals are firing.
Checks thresholds, market conditions, and ML filtering.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

async def diagnose():
    from hydra.core.config import PERMITTED_PAIRS
    from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
    from hydra.layers.layer2_statistical import StatisticalRealityEngine
    from hydra.layers.layer3_alpha.signals import (
        BehavioralSignalGenerator,
        FUNDING_SQUEEZE_THRESHOLD,
        FUNDING_EXTREME_THRESHOLD,
        FUNDING_CARRY_MIN,
        OI_DIVERGENCE_PRICE_THRESHOLD,
        OI_DIVERGENCE_OI_THRESHOLD,
        CROWDING_LS_RATIO_HIGH,
        CROWDING_LS_RATIO_LOW,
        LIQ_IMBALANCE_THRESHOLD,
        MIN_SIGNAL_CONFIDENCE,
        ML_SCORE_THRESHOLD,
    )
    from hydra.core.config import HydraConfig
    
    console.print(Panel("[bold cyan]HYDRA Signal Diagnostic[/bold cyan]"))
    
    # Show thresholds
    console.print("\n[bold yellow]Current Thresholds:[/bold yellow]")
    thresholds = Table(title="Signal Thresholds")
    thresholds.add_column("Parameter", style="cyan")
    thresholds.add_column("Value", style="green")
    thresholds.add_column("Description")
    
    thresholds.add_row("FUNDING_SQUEEZE_THRESHOLD", f"{FUNDING_SQUEEZE_THRESHOLD*100:.3f}%", "Min funding for squeeze signal")
    thresholds.add_row("FUNDING_EXTREME_THRESHOLD", f"{FUNDING_EXTREME_THRESHOLD*100:.3f}%", "Extreme funding boost")
    thresholds.add_row("FUNDING_CARRY_MIN", f"{FUNDING_CARRY_MIN*100:.3f}%", "Min funding for carry trade")
    thresholds.add_row("OI_DIVERGENCE_PRICE", f"{OI_DIVERGENCE_PRICE_THRESHOLD*100:.1f}%", "Price move threshold")
    thresholds.add_row("OI_DIVERGENCE_OI", f"{OI_DIVERGENCE_OI_THRESHOLD*100:.1f}%", "OI change threshold")
    thresholds.add_row("CROWDING_LS_HIGH", f"{CROWDING_LS_RATIO_HIGH:.1f}", "Long crowding L/S ratio")
    thresholds.add_row("CROWDING_LS_LOW", f"{CROWDING_LS_RATIO_LOW:.1f}", "Short crowding L/S ratio")
    thresholds.add_row("MIN_SIGNAL_CONFIDENCE", f"{MIN_SIGNAL_CONFIDENCE:.0%}", "Min confidence to pass")
    thresholds.add_row("ML_SCORE_THRESHOLD", f"{ML_SCORE_THRESHOLD:.0%}", "Min ML score to pass")
    
    console.print(thresholds)
    
    # Initialize layers
    console.print("\n[bold]Initializing layers...[/bold]")
    
    config = HydraConfig()
    layer1 = MarketIntelligenceLayer()
    await layer1.initialize()
    
    layer2 = StatisticalRealityEngine(config, use_ml_regime=True)
    await layer2.setup()
    
    layer3 = BehavioralSignalGenerator()
    ml_status = "LOADED" if layer3.ml_scorer else "NOT LOADED"
    console.print(f"ML Signal Scorer: [{'green' if layer3.ml_scorer else 'red'}]{ml_status}[/]")
    
    # Check each pair
    results = Table(title="Market Conditions Analysis")
    results.add_column("Pair", style="cyan")
    results.add_column("Price", style="white")
    results.add_column("Funding", style="yellow")
    results.add_column("Fund vs Thresh", style="magenta")
    results.add_column("OI Δ%", style="blue")
    results.add_column("Regime", style="green")
    results.add_column("Tradability", style="red")
    results.add_column("Cascade%", style="red")
    results.add_column("Signals", style="cyan")
    
    for symbol in PERMITTED_PAIRS:
        console.print(f"\nChecking [cyan]{symbol}[/cyan]...")
        
        try:
            await layer1.refresh_symbol(symbol)
            market_state = layer1.get_market_state(symbol)
            
            if not market_state or market_state.price == 0:
                results.add_row(symbol, "NO DATA", "-", "-", "-", "-", "-", "-", "-")
                continue
            
            stat_result = await layer2.analyze(market_state)
            
            # Get funding info
            funding = market_state.funding_rate.rate if market_state.funding_rate else 0
            funding_pct = funding * 100
            
            # Compare to threshold
            if abs(funding) >= FUNDING_SQUEEZE_THRESHOLD:
                fund_status = f"[green]✓ ABOVE ({abs(funding)/FUNDING_SQUEEZE_THRESHOLD:.1f}x)[/green]"
            elif abs(funding) >= FUNDING_CARRY_MIN:
                fund_status = f"[yellow]~ CARRY ({abs(funding)/FUNDING_CARRY_MIN:.1f}x)[/yellow]"
            else:
                fund_status = f"[red]✗ BELOW ({abs(funding)/FUNDING_SQUEEZE_THRESHOLD:.1f}x)[/red]"
            
            # OI delta
            oi_delta = market_state.open_interest.delta_pct if market_state.open_interest else 0
            
            # Generate signals (without ML filtering first)
            signals_no_ml = []
            from hydra.layers.layer3_alpha.signals import (
                funding_squeeze, liquidation_reversal, oi_divergence, 
                crowding_fade, funding_carry
            )
            
            for gen_name, gen_func in [
                ("FUNDING_SQUEEZE", lambda: funding_squeeze(market_state, stat_result)),
                ("LIQUIDATION_REVERSAL", lambda: liquidation_reversal(market_state, stat_result)),
                ("OI_DIVERGENCE", lambda: oi_divergence(market_state, stat_result)),
                ("CROWDING_FADE", lambda: crowding_fade(market_state, stat_result, 1.0)),
                ("FUNDING_CARRY", lambda: funding_carry(market_state, stat_result)),
            ]:
                try:
                    sig = gen_func()
                    if sig:
                        signals_no_ml.append(f"{gen_name[:8]}({sig.confidence:.0%})")
                except Exception as e:
                    pass
            
            # With ML filtering
            signals_with_ml = layer3.generate_signals(market_state, stat_result, 1.0)
            
            sig_str = f"{len(signals_no_ml)} raw"
            if signals_no_ml:
                sig_str += f" → {len(signals_with_ml)} ML"
            
            results.add_row(
                symbol.replace("cmt_", "").upper(),
                f"${market_state.price:,.2f}",
                f"{funding_pct:+.4f}%",
                fund_status,
                f"{oi_delta*100:+.2f}%",
                stat_result.regime.name[:6],
                stat_result.trading_decision.value,
                f"{stat_result.cascade_probability:.0%}",
                sig_str,
            )
            
            # Show detailed signal info
            if signals_no_ml:
                console.print(f"  Raw signals: {', '.join(signals_no_ml)}")
            if signals_with_ml:
                for sig in signals_with_ml:
                    ml_score = sig.metadata.get('ml_score', 'N/A')
                    console.print(f"  [green]✓ PASSED ML:[/green] {sig.source} {sig.side.value} conf={sig.confidence:.0%} ml={ml_score}")
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            results.add_row(symbol, "ERROR", "-", "-", "-", "-", "-", "-", str(e)[:20])
    
    console.print("\n")
    console.print(results)
    
    # Recommendations
    console.print("\n[bold yellow]Analysis & Recommendations:[/bold yellow]")
    console.print("""
1. If funding rates are below thresholds, market is "normal" - no squeeze signals
2. If regime is not RANGING, no FUNDING_CARRY signals
3. ML filter may reject signals with score < 0.6

[bold]To increase signal frequency, consider:[/bold]
- Lower FUNDING_SQUEEZE_THRESHOLD from 0.05% to 0.03%
- Lower FUNDING_CARRY_MIN from 0.03% to 0.02%
- Lower ML_SCORE_THRESHOLD from 0.6 to 0.5
- Add momentum-based signals (not just behavioral)
""")
    
    await layer1.close()

if __name__ == "__main__":
    asyncio.run(diagnose())
