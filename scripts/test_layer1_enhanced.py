#!/usr/bin/env python3
"""
Test Enhanced Layer 1: Market Intelligence Layer

Tests all new features:
- Immediate funding/OI/orderbook fetch
- Liquidation tracking (Binance API)
- CVD tracking
- On-chain data (exchange flows, whale activity)
- News sentiment (CryptoPanic, Google News)
- Social sentiment (Reddit, CoinGecko)
- Market positioning (L/S ratio, top traders)
- Stablecoin metrics

Usage:
    python scripts/test_layer1_enhanced.py
    python scripts/test_layer1_enhanced.py --symbol cmt_btcusdt
    python scripts/test_layer1_enhanced.py --all
"""

import argparse
import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer

console = Console()

# Signal colors
SIGNAL_COLORS = {
    "strong_buy": "bold green",
    "buy": "green", 
    "neutral": "white",
    "sell": "red",
    "strong_sell": "bold red",
}


def format_usd(value) -> str:
    """Format USD value."""
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_pct(value: float) -> str:
    """Format percentage."""
    return f"{value*100:.2f}%"


def print_symbol_intelligence(layer: MarketIntelligenceLayer, symbol: str):
    """Print comprehensive intelligence for a symbol."""
    display = PAIR_DISPLAY_NAMES.get(symbol, symbol)
    
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold white]{display}[/bold white] - Market Intelligence Summary")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    # =========================================================================
    # PRICE DATA (Funding, OI, Orderbook)
    # =========================================================================
    price_table = Table(title="📊 Price Data", show_header=True)
    price_table.add_column("Metric", style="cyan", width=25)
    price_table.add_column("Value", style="green", width=20)
    price_table.add_column("Analysis", style="yellow", width=25)
    
    # Funding
    funding = layer.get_funding_rate(symbol)
    if funding:
        rate_pct = funding.rate * 100
        analysis = "🔴 Longs paying" if rate_pct > 0.01 else "🟢 Shorts paying" if rate_pct < -0.01 else "⚪ Neutral"
        price_table.add_row("Funding Rate", f"{rate_pct:.4f}%", analysis)
        if funding.predicted_rate:
            price_table.add_row("Predicted Rate", f"{funding.predicted_rate*100:.4f}%", "")
    else:
        price_table.add_row("Funding Rate", "N/A", "")
    
    # Open Interest
    oi = layer.get_open_interest(symbol)
    if oi and oi.open_interest_usd:
        delta_str = f"Δ {oi.delta_pct:.2f}%" if oi.delta_pct else ""
        price_table.add_row("Open Interest", format_usd(oi.open_interest_usd), delta_str)
    else:
        price_table.add_row("Open Interest", "N/A", "")
    
    # Orderbook
    orderbook = layer.get_orderbook(symbol)
    if orderbook and orderbook.bids and orderbook.asks:
        spread = orderbook.asks[0][0] - orderbook.bids[0][0]
        spread_pct = spread / orderbook.bids[0][0] * 100
        
        bid_depth = sum(b[0] * b[1] for b in orderbook.bids[:10])
        ask_depth = sum(a[0] * a[1] for a in orderbook.asks[:10])
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        price_table.add_row("Spread", f"{spread_pct:.4f}%", "")
        price_table.add_row("Bid Depth (10)", format_usd(bid_depth), "")
        price_table.add_row("Ask Depth (10)", format_usd(ask_depth), "")
        imb_color = "green" if imbalance > 0.1 else "red" if imbalance < -0.1 else "white"
        price_table.add_row("Book Imbalance", f"[{imb_color}]{imbalance:.2%}[/{imb_color}]", "Bid > Ask" if imbalance > 0 else "Ask > Bid")
    else:
        price_table.add_row("Orderbook", "N/A", "")
    
    console.print(price_table)
    console.print()
    
    # =========================================================================
    # LIQUIDATION DATA
    # =========================================================================
    liq_table = Table(title="💥 Liquidations", show_header=True)
    liq_table.add_column("Metric", style="cyan", width=25)
    liq_table.add_column("Value", style="green", width=20)
    liq_table.add_column("Analysis", style="yellow", width=25)
    
    liq = layer.get_liquidation_data(symbol)
    if liq:
        liq_table.add_row("Long Liqs (1h)", format_usd(liq.long_liquidations_1h), "")
        liq_table.add_row("Short Liqs (1h)", format_usd(liq.short_liquidations_1h), "")
        liq_table.add_row("Long Liqs (24h)", format_usd(liq.long_liquidations_24h), "")
        liq_table.add_row("Short Liqs (24h)", format_usd(liq.short_liquidations_24h), "")
        liq_table.add_row("Largest Liq", format_usd(liq.largest_liquidation), "")
        
        imb = liq.liquidation_imbalance
        imb_analysis = "🔴 Longs crushed" if imb > 0.3 else "🟢 Shorts crushed" if imb < -0.3 else "⚪ Balanced"
        liq_table.add_row("Liq Imbalance", f"{imb:.2%}", imb_analysis)
    else:
        liq_table.add_row("Liquidations", "N/A", "No data")
    
    console.print(liq_table)
    console.print()
    
    # =========================================================================
    # POSITIONING DATA
    # =========================================================================
    pos_table = Table(title="📈 Market Positioning", show_header=True)
    pos_table.add_column("Metric", style="cyan", width=25)
    pos_table.add_column("Value", style="green", width=20)
    pos_table.add_column("Analysis", style="yellow", width=25)
    
    positioning = layer.get_positioning_data(symbol)
    if positioning:
        ls_ratio = positioning.long_short_ratio
        ls_analysis = "🔴 Crowded long" if ls_ratio > 1.5 else "🟢 Crowded short" if ls_ratio < 0.67 else "⚪ Balanced"
        pos_table.add_row("Long/Short Ratio", f"{ls_ratio:.2f}", ls_analysis)
        pos_table.add_row("Top Trader Long %", f"{positioning.top_trader_long_ratio:.1%}", "")
        pos_table.add_row("Top Trader Short %", f"{positioning.top_trader_short_ratio:.1%}", "")
    else:
        pos_table.add_row("Positioning", "N/A", "No data")
    
    # CVD
    cvd = layer.get_cvd(symbol)
    cvd_analysis = "🟢 Buyers dominant" if cvd > 1000 else "🔴 Sellers dominant" if cvd < -1000 else "⚪ Balanced"
    pos_table.add_row("CVD", f"{cvd:,.0f}", cvd_analysis)
    
    console.print(pos_table)
    console.print()
    
    # =========================================================================
    # NEWS SENTIMENT
    # =========================================================================
    news_table = Table(title="📰 News Sentiment", show_header=True)
    news_table.add_column("Metric", style="cyan", width=25)
    news_table.add_column("Value", style="green", width=20)
    news_table.add_column("Analysis", style="yellow", width=25)
    
    news = layer.get_news_sentiment(symbol)
    if news:
        news_table.add_row("Articles (1h)", str(news.news_count_1h), "")
        news_table.add_row("Articles (24h)", str(news.news_count_24h), "")
        
        sent_color = "green" if news.avg_sentiment_24h > 0.2 else "red" if news.avg_sentiment_24h < -0.2 else "white"
        news_table.add_row("Sentiment (24h)", f"[{sent_color}]{news.avg_sentiment_24h:.2f}[/{sent_color}]", 
                          "Bullish" if news.avg_sentiment_24h > 0.2 else "Bearish" if news.avg_sentiment_24h < -0.2 else "Neutral")
        news_table.add_row("Sentiment Momentum", f"{news.sentiment_momentum:.2f}", "")
        
        # Show breaking news
        if news.breaking_news:
            console.print(news_table)
            console.print("\n[bold yellow]Breaking News:[/bold yellow]")
            for item in news.breaking_news[:3]:
                sent_icon = "🟢" if item.sentiment > 0.2 else "🔴" if item.sentiment < -0.2 else "⚪"
                console.print(f"  {sent_icon} {item.title[:60]}...")
    else:
        news_table.add_row("News", "N/A", "No data")
        console.print(news_table)
    
    console.print()
    
    # =========================================================================
    # SOCIAL SENTIMENT
    # =========================================================================
    social_table = Table(title="🐦 Social Sentiment", show_header=True)
    social_table.add_column("Metric", style="cyan", width=25)
    social_table.add_column("Value", style="green", width=20)
    social_table.add_column("Analysis", style="yellow", width=25)
    
    social = layer.get_social_sentiment(symbol)
    if social:
        social_table.add_row("Reddit Posts (24h)", str(social.reddit_posts_24h), "")
        social_table.add_row("Reddit Comments (24h)", str(social.reddit_comments_24h), "")
        
        sent_color = "green" if social.reddit_sentiment > 0.2 else "red" if social.reddit_sentiment < -0.2 else "white"
        social_table.add_row("Reddit Sentiment", f"[{sent_color}]{social.reddit_sentiment:.2f}[/{sent_color}]", "")
        
        social_table.add_row("Social Volume", f"{social.social_volume_24h:,}", "")
        social_table.add_row("Social Dominance", f"{social.social_dominance:.2%}", "")
        
        trend_color = "green" if social.sentiment_trend == "bullish" else "red" if social.sentiment_trend == "bearish" else "white"
        social_table.add_row("Trend", f"[{trend_color}]{social.sentiment_trend.upper()}[/{trend_color}]", "")
    else:
        social_table.add_row("Social", "N/A", "No data")
    
    console.print(social_table)
    console.print()
    
    # =========================================================================
    # ON-CHAIN DATA
    # =========================================================================
    onchain_table = Table(title="⛓️ On-Chain Data", show_header=True)
    onchain_table.add_column("Metric", style="cyan", width=25)
    onchain_table.add_column("Value", style="green", width=20)
    onchain_table.add_column("Analysis", style="yellow", width=25)
    
    onchain = layer.get_onchain_data(symbol)
    if onchain:
        onchain_table.add_row("Exchange Inflow", format_usd(onchain.exchange_inflow_usd), "")
        onchain_table.add_row("Exchange Outflow", format_usd(onchain.exchange_outflow_usd), "")
        
        flow = onchain.net_exchange_flow
        flow_analysis = "🔴 Sell pressure" if flow > 0 else "🟢 Accumulation" if flow < 0 else "⚪ Neutral"
        onchain_table.add_row("Net Flow", format_usd(flow), flow_analysis)
        
        onchain_table.add_row("Large Txs (24h)", str(onchain.large_tx_count_24h), "")
        onchain_table.add_row("Large Tx Volume", format_usd(onchain.large_tx_volume_usd), "")
        
        whale = onchain.whale_accumulation_score
        whale_analysis = "🐋 Accumulating" if whale > 0.3 else "🦈 Distributing" if whale < -0.3 else "⚪ Neutral"
        onchain_table.add_row("Whale Score", f"{whale:.2f}", whale_analysis)
        
        onchain_table.add_row("Active Addresses", f"{onchain.active_addresses_24h:,}", "")
    else:
        onchain_table.add_row("On-Chain", "N/A", "No data")
    
    console.print(onchain_table)
    console.print()
    
    # =========================================================================
    # FEAR & GREED INDEX
    # =========================================================================
    fear_greed = layer.get_fear_greed()
    if fear_greed:
        fg_color = "green" if fear_greed.value >= 60 else "red" if fear_greed.value <= 40 else "yellow"
        fg_panel = Panel(
            f"[bold]Value:[/bold] [{fg_color}]{fear_greed.value}[/{fg_color}] ({fear_greed.classification})\n"
            f"Yesterday: {fear_greed.value_yesterday} | Week Ago: {fear_greed.value_week_ago} | Month Ago: {fear_greed.value_month_ago}\n"
            f"Trend: {fear_greed.trend.upper()}",
            title="Fear & Greed Index (Market-Wide)",
            border_style=fg_color
        )
        console.print(fg_panel)
        console.print()
    
    # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================
    technical = layer.get_technical_sentiment(symbol)
    if technical:
        tech_table = Table(title="Technical Analysis", show_header=True)
        tech_table.add_column("Metric", style="cyan", width=20)
        tech_table.add_column("Value", style="green", width=15)
        tech_table.add_column("Signal", style="yellow", width=20)
        
        trend_colors = {"bullish": "green", "bearish": "red", "neutral": "white"}
        tech_table.add_row("Trend 1H", technical.trend_1h, f"[{trend_colors.get(technical.trend_1h, 'white')}]{technical.trend_1h.upper()}[/]")
        tech_table.add_row("Trend 4H", technical.trend_4h, f"[{trend_colors.get(technical.trend_4h, 'white')}]{technical.trend_4h.upper()}[/]")
        tech_table.add_row("Trend 1D", technical.trend_1d, f"[{trend_colors.get(technical.trend_1d, 'white')}]{technical.trend_1d.upper()}[/]")
        
        rsi_color = "red" if technical.rsi_signal == "overbought" else "green" if technical.rsi_signal == "oversold" else "white"
        tech_table.add_row("RSI (14)", f"{technical.rsi_14:.1f}", f"[{rsi_color}]{technical.rsi_signal.upper()}[/]")
        
        tech_table.add_row("Price vs EMA20", f"{technical.price_vs_ema20:.2f}%", "Above" if technical.price_vs_ema20 > 0 else "Below")
        tech_table.add_row("Volatility", f"{technical.volatility_percentile:.0f}%ile", technical.volatility_regime.upper())
        tech_table.add_row("Tech Score", f"{technical.technical_score:.2f}", "")
        
        console.print(tech_table)
        console.print()
    
    # =========================================================================
    # ENHANCED COMBINED SIGNAL
    # =========================================================================
    enhanced = layer.get_enhanced_sentiment(symbol)
    if enhanced:
        signal_color = SIGNAL_COLORS.get(enhanced.signal, "white")
        border_color = signal_color.split()[-1] if " " in signal_color else signal_color
        
        signal_content = f"[bold]Overall Score:[/bold] {enhanced.overall_score:.2f} | [bold]Confidence:[/bold] {enhanced.confidence:.0%}\n"
        signal_content += f"[bold]Signal:[/bold] [{signal_color}]{enhanced.signal.upper().replace('_', ' ')}[/{signal_color}] | [bold]Regime:[/bold] {enhanced.regime.upper()}\n\n"
        signal_content += f"Fear/Greed: {enhanced.fear_greed_score:.2f} | Technical: {enhanced.technical_score:.2f}\n"
        signal_content += f"News: {enhanced.news_score:.2f} | Social: {enhanced.social_score:.2f}\n"
        signal_content += f"Funding: {enhanced.funding_score:.2f} | Positioning: {enhanced.positioning_score:.2f}"
        
        if enhanced.key_factors:
            signal_content += "\n\n[bold]Key Factors:[/bold]"
            for factor in enhanced.key_factors[:4]:
                signal_content += f"\n  - {factor}"
        
        if enhanced.warnings:
            signal_content += "\n\n[bold yellow]Warnings:[/bold yellow]"
            for warning in enhanced.warnings[:3]:
                signal_content += f"\n  - {warning}"
        
        signal_panel = Panel(
            signal_content,
            title="Enhanced Intelligence Signal",
            border_style=border_color
        )
        console.print(signal_panel)
    else:
        # Fallback to basic combined sentiment
        combined = layer.get_combined_sentiment(symbol)
        sentiment_panel = Panel(
            f"[bold]Overall Score:[/bold] {combined['overall_score']:.2f}\n"
            f"[bold]Signal:[/bold] {combined['signal'].upper()}\n\n"
            f"News: {combined['news_score']:.2f} | Social: {combined['social_score']:.2f}\n"
            f"Funding: {combined['funding_score']:.2f} | Positioning: {combined['positioning_score']:.2f}",
            title="Combined Intelligence Signal",
            border_style="green" if combined['signal'] == 'bullish' else "red" if combined['signal'] == 'bearish' else "white"
        )
        console.print(sentiment_panel)


def print_stablecoin_metrics(layer: MarketIntelligenceLayer):
    """Print global stablecoin metrics."""
    console.print("\n[bold cyan]{'='*70}[/bold cyan]")
    console.print("[bold white]💵 Global Stablecoin Metrics[/bold white]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    metrics = layer.get_stablecoin_metrics()
    
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=25)
    
    table.add_row("Total Supply", format_usd(metrics.get("total_stablecoin_supply", 0)))
    table.add_row("USDT Dominance", f"{metrics.get('usdt_dominance', 0):.2%}")
    table.add_row("Exchange Reserve", format_usd(metrics.get("stablecoin_exchange_reserve", 0)))
    
    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Test Enhanced Layer 1")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    parser.add_argument("--all", action="store_true", help="Test all pairs")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]HYDRA Enhanced Layer 1 Test[/bold cyan]\n"
        "Testing: Funding, OI, Orderbook, Liquidations, CVD,\n"
        "On-Chain, News, Social, Positioning",
        border_style="cyan"
    ))
    
    # Load config and initialize layer
    config = HydraConfig.load()
    layer = MarketIntelligenceLayer(config)
    
    console.print("\n[cyan]Initializing Layer 1 (this may take 30-60 seconds)...[/cyan]")
    console.print("[dim]Loading: OHLCV -> Funding -> OI -> Orderbook -> News -> Social -> On-Chain[/dim]\n")
    
    await layer.setup()
    console.print("[green]✓ Layer 1 initialized with all enhanced data[/green]\n")
    
    # Print stablecoin metrics
    print_stablecoin_metrics(layer)
    
    # Determine which symbols to test
    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = list(PERMITTED_PAIRS)
    else:
        symbols = ["cmt_btcusdt"]  # Default
    
    # Test each symbol
    for symbol in symbols:
        print_symbol_intelligence(layer, symbol)
    
    # Cleanup
    await layer.stop_feeds()
    
    console.print("\n[green]✓ Enhanced Layer 1 test complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
