#!/usr/bin/env python3
"""
Test Layer 1: Market Intelligence Layer

Shows all data that Layer 1 can collect for the 8 permitted pairs.

Usage:
    python scripts/test_layer1.py
    python scripts/test_layer1.py --symbol cmt_btcusdt
    python scripts/test_layer1.py --live  # Start live feeds
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer

console = Console()


def to_exchange_symbol(internal: str) -> str:
    """Convert cmt_btcusdt -> BTC/USDT:USDT"""
    display = PAIR_DISPLAY_NAMES.get(internal, internal)
    return f"{display}:USDT"


async def test_single_fetch(layer: MarketIntelligenceLayer, symbol: str) -> dict:
    """Test fetching data for a single symbol."""
    exchange_symbol = to_exchange_symbol(symbol)
    # Layer 1 stores data using internal symbol format
    internal_symbol = symbol  # Already in internal format
    
    result = {
        "symbol": symbol,
        "exchange_symbol": exchange_symbol,
        "ohlcv": {},
        "funding": None,
        "open_interest": None,
        "orderbook": None,
        "sentiment": None,
        "errors": [],
    }
    
    try:
        # Test OHLCV for each timeframe (use internal symbol for lookups)
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        for tf in timeframes:
            candles = layer.get_candles(internal_symbol, tf, limit=10)
            result["ohlcv"][tf] = {
                "count": len(candles),
                "latest": candles[-1] if candles else None,
            }
        
        # Test funding rate (use internal symbol)
        funding = layer.get_funding_rate(internal_symbol)
        result["funding"] = funding
        
        # Test open interest (use internal symbol)
        oi = layer.get_open_interest(internal_symbol)
        result["open_interest"] = oi
        
        # Test orderbook (use internal symbol)
        orderbook = layer.get_orderbook(internal_symbol)
        result["orderbook"] = orderbook
        
        # Test sentiment (use internal symbol)
        sentiment = await layer.fetch_sentiment(internal_symbol)
        result["sentiment"] = sentiment
        
        # Test full market state (use internal symbol)
        market_state = await layer.get_market_state(internal_symbol)
        result["market_state"] = market_state
        
    except Exception as e:
        result["errors"].append(str(e))
    
    return result


def print_symbol_data(result: dict):
    """Print data for a single symbol."""
    symbol = result["symbol"]
    exchange_symbol = result["exchange_symbol"]
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]{symbol}[/bold] ({exchange_symbol})")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    if result["errors"]:
        console.print(f"[red]Errors: {result['errors']}[/red]")
    
    # OHLCV Table
    ohlcv_table = Table(title="OHLCV Data")
    ohlcv_table.add_column("Timeframe", style="cyan")
    ohlcv_table.add_column("Candles", style="green")
    ohlcv_table.add_column("Latest Close", style="yellow")
    ohlcv_table.add_column("Volume", style="magenta")
    
    for tf, data in result["ohlcv"].items():
        if data["latest"]:
            ohlcv_table.add_row(
                tf,
                str(data["count"]),
                f"${data['latest'].close:,.2f}",
                f"{data['latest'].volume:,.0f}"
            )
        else:
            ohlcv_table.add_row(tf, str(data["count"]), "N/A", "N/A")
    
    console.print(ohlcv_table)
    
    # Funding Rate
    funding = result["funding"]
    if funding:
        funding_table = Table(title="Funding Rate")
        funding_table.add_column("Metric", style="cyan")
        funding_table.add_column("Value", style="green")
        
        funding_table.add_row("Current Rate", f"{funding.rate:.6f} ({funding.rate*100:.4f}%)")
        if funding.predicted_rate:
            funding_table.add_row("Predicted Rate", f"{funding.predicted_rate:.6f}")
        if funding.next_funding_time:
            funding_table.add_row("Next Funding", str(funding.next_funding_time))
        
        # Funding interpretation
        if funding.rate > 0.0001:
            funding_table.add_row("Interpretation", "[red]Longs paying shorts (bullish crowd)[/red]")
        elif funding.rate < -0.0001:
            funding_table.add_row("Interpretation", "[green]Shorts paying longs (bearish crowd)[/green]")
        else:
            funding_table.add_row("Interpretation", "[yellow]Neutral[/yellow]")
        
        console.print(funding_table)
    else:
        console.print("[yellow]No funding data available[/yellow]")
    
    # Open Interest
    oi = result["open_interest"]
    if oi:
        oi_table = Table(title="Open Interest")
        oi_table.add_column("Metric", style="cyan")
        oi_table.add_column("Value", style="green")
        
        oi_table.add_row("OI (contracts)", f"{oi.open_interest:,.0f}")
        oi_table.add_row("OI (USD)", f"${oi.open_interest_usd:,.0f}")
        oi_table.add_row("Delta", f"${oi.delta:,.0f}")
        oi_table.add_row("Delta %", f"{oi.delta_pct:.2f}%")
        
        console.print(oi_table)
    else:
        console.print("[yellow]No open interest data available[/yellow]")
    
    # Order Book
    orderbook = result["orderbook"]
    if orderbook and orderbook.bids and orderbook.asks:
        ob_table = Table(title="Order Book (Top 5)")
        ob_table.add_column("Bid Price", style="green")
        ob_table.add_column("Bid Size", style="green")
        ob_table.add_column("Ask Price", style="red")
        ob_table.add_column("Ask Size", style="red")
        
        for i in range(min(5, len(orderbook.bids), len(orderbook.asks))):
            bid = orderbook.bids[i]
            ask = orderbook.asks[i]
            ob_table.add_row(
                f"${bid[0]:,.2f}",
                f"{bid[1]:,.4f}",
                f"${ask[0]:,.2f}",
                f"{ask[1]:,.4f}"
            )
        
        # Calculate spread and imbalance
        if orderbook.bids and orderbook.asks:
            spread = orderbook.asks[0][0] - orderbook.bids[0][0]
            spread_pct = spread / orderbook.bids[0][0] * 100
            
            bid_depth = sum(b[1] * b[0] for b in orderbook.bids[:10])
            ask_depth = sum(a[1] * a[0] for a in orderbook.asks[:10])
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
            
            ob_table.add_row("---", "---", "---", "---")
            ob_table.add_row("Spread", f"${spread:.2f} ({spread_pct:.4f}%)", "", "")
            ob_table.add_row("Bid Depth", f"${bid_depth:,.0f}", "Ask Depth", f"${ask_depth:,.0f}")
            ob_table.add_row("Imbalance", f"{imbalance:.2%}", "", "")
        
        console.print(ob_table)
    else:
        console.print("[yellow]No orderbook data available[/yellow]")
    
    # Market State Summary
    ms = result.get("market_state")
    if ms:
        ms_table = Table(title="Market State Summary")
        ms_table.add_column("Metric", style="cyan")
        ms_table.add_column("Value", style="green")
        
        ms_table.add_row("Current Price", f"${ms.price:,.2f}")
        ms_table.add_row("Mark Price", f"${ms.mark_price:,.2f}")
        ms_table.add_row("Index Price", f"${ms.index_price:,.2f}")
        ms_table.add_row("Basis", f"{ms.basis:.4%}")
        ms_table.add_row("24h Change", f"{ms.price_change_24h:.2%}")
        ms_table.add_row("24h Volume", f"${ms.volume_24h:,.0f}")
        ms_table.add_row("Volatility (ann.)", f"{ms.volatility:.2%}")
        
        console.print(ms_table)
    
    # Sentiment
    sentiment = result["sentiment"]
    if sentiment:
        sent_table = Table(title="Derived Sentiment")
        sent_table.add_column("Metric", style="cyan")
        sent_table.add_column("Value", style="green")
        
        sent_table.add_row("Funding Sentiment", f"{sentiment.funding_sentiment:.2f}")
        sent_table.add_row("Crowd Positioning", f"{sentiment.crowd_positioning:.2f}")
        sent_table.add_row("Fear/Greed Index", f"{sentiment.fear_greed_index:.0f}")
        
        console.print(sent_table)


async def run_live_dashboard(layer: MarketIntelligenceLayer):
    """Run live updating dashboard."""
    console.print("[cyan]Starting live feeds... Press Ctrl+C to stop[/cyan]")
    
    await layer.start_feeds()
    
    try:
        while True:
            console.clear()
            console.print(Panel.fit(
                f"[bold cyan]HYDRA Layer 1 - Live Market Intelligence[/bold cyan]\n"
                f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="cyan"
            ))
            
            # Quick summary table
            summary_table = Table(title="All Pairs Summary")
            summary_table.add_column("Pair", style="cyan")
            summary_table.add_column("Price", style="green")
            summary_table.add_column("24h %", style="yellow")
            summary_table.add_column("Funding", style="magenta")
            summary_table.add_column("OI Δ%", style="blue")
            summary_table.add_column("Spread", style="white")
            
            for pair in PERMITTED_PAIRS:
                exchange_symbol = to_exchange_symbol(pair)
                
                try:
                    ms = await layer.get_market_state(exchange_symbol)
                    if ms:
                        funding_str = f"{ms.funding_rate.rate*100:.4f}%" if ms.funding_rate else "N/A"
                        oi_str = f"{ms.open_interest.delta_pct:.2f}%" if ms.open_interest else "N/A"
                        
                        spread = "N/A"
                        if ms.order_book and ms.order_book.bids and ms.order_book.asks:
                            spread_val = (ms.order_book.asks[0][0] - ms.order_book.bids[0][0]) / ms.order_book.bids[0][0] * 100
                            spread = f"{spread_val:.4f}%"
                        
                        change_color = "green" if ms.price_change_24h >= 0 else "red"
                        
                        summary_table.add_row(
                            PAIR_DISPLAY_NAMES.get(pair, pair),
                            f"${ms.price:,.2f}",
                            f"[{change_color}]{ms.price_change_24h:+.2%}[/{change_color}]",
                            funding_str,
                            oi_str,
                            spread
                        )
                except Exception as e:
                    summary_table.add_row(
                        PAIR_DISPLAY_NAMES.get(pair, pair),
                        "Error", str(e)[:20], "", "", ""
                    )
            
            console.print(summary_table)
            console.print("\n[dim]Press Ctrl+C to stop[/dim]")
            
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping feeds...[/yellow]")
    finally:
        await layer.stop_feeds()


async def main():
    parser = argparse.ArgumentParser(description="Test HYDRA Layer 1")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    parser.add_argument("--live", action="store_true", help="Run live dashboard")
    parser.add_argument("--all", action="store_true", help="Test all pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Layer 1 Test[/bold cyan]\n"
        "Market Intelligence Layer",
        border_style="cyan"
    ))
    
    # Load config and initialize layer
    config = HydraConfig.load()
    layer = MarketIntelligenceLayer(config)
    
    console.print("[cyan]Initializing exchange connection...[/cyan]")
    await layer.setup()
    console.print("[green]✓ Exchange connected[/green]")
    
    if args.live:
        await run_live_dashboard(layer)
    else:
        # Test specific symbol or all
        if args.symbol:
            symbols = [args.symbol]
        elif args.all:
            symbols = list(PERMITTED_PAIRS)
        else:
            symbols = ["cmt_btcusdt"]  # Default to BTC
        
        for symbol in symbols:
            console.print(f"\n[cyan]Testing {symbol}...[/cyan]")
            result = await test_single_fetch(layer, symbol)
            print_symbol_data(result)
        
        await layer.stop_feeds()
    
    console.print("\n[green]✓ Layer 1 test complete[/green]")


if __name__ == "__main__":
    asyncio.run(main())
