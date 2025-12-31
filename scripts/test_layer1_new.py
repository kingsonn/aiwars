"""
Test script for the revamped Layer 1: Market Intelligence

Tests:
1. Binance Futures API - OHLCV, Funding, OI, Order Book
2. Coinalyse API - Liquidation history (requires API key)
3. Full MarketState generation

Usage:
    python scripts/test_layer1_new.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from hydra.layers.layer1_market_intel import (
    BinanceFuturesClient,
    CoinalyseLiquidationClient,
    MarketIntelligenceLayer,
    PERMITTED_PAIRS,
    TIMEFRAMES,
)


async def test_binance_client():
    """Test Binance Futures API client."""
    print("\n" + "="*60)
    print("TESTING BINANCE FUTURES CLIENT")
    print("="*60)
    
    client = BinanceFuturesClient()
    symbol = "cmt_btcusdt"
    
    try:
        # Test OHLCV
        print(f"\n[1] Fetching OHLCV for {symbol}...")
        candles = await client.fetch_ohlcv(symbol, "5m", limit=10)
        if candles:
            print(f"    ✓ Got {len(candles)} candles")
            print(f"    Latest: {candles[-1].timestamp} | O:{candles[-1].open:.2f} H:{candles[-1].high:.2f} L:{candles[-1].low:.2f} C:{candles[-1].close:.2f}")
        else:
            print("    ✗ No candles returned")
        
        # Test Funding Rate
        print(f"\n[2] Fetching Funding Rate for {symbol}...")
        funding = await client.fetch_funding_rate(symbol)
        if funding:
            print(f"    ✓ Rate: {funding.rate:.6f} ({funding.rate*100:.4f}%)")
            print(f"    Annualized: {funding.annualized:.2f}%")
            print(f"    Next funding: {funding.next_funding_time}")
        else:
            print("    ✗ No funding data returned")
        
        # Test Open Interest
        print(f"\n[3] Fetching Open Interest for {symbol}...")
        oi = await client.fetch_open_interest(symbol)
        if oi:
            print(f"    ✓ OI: {oi.open_interest:,.2f} contracts")
            print(f"    OI USD: ${oi.open_interest_usd:,.0f}")
        else:
            print("    ✗ No OI data returned")
        
        # Test Order Book
        print(f"\n[4] Fetching Order Book for {symbol}...")
        ob = await client.fetch_order_book(symbol, limit=5)
        if ob:
            print(f"    ✓ Spread: {ob.spread:.4%}")
            print(f"    Imbalance: {ob.imbalance:+.3f}")
            print(f"    Best Bid: ${ob.bids[0][0]:,.2f} ({ob.bids[0][1]:.4f})")
            print(f"    Best Ask: ${ob.asks[0][0]:,.2f} ({ob.asks[0][1]:.4f})")
        else:
            print("    ✗ No order book returned")
        
        # Test Long/Short Ratio
        print(f"\n[5] Fetching Long/Short Ratio for {symbol}...")
        ls_ratio = await client.fetch_long_short_ratio(symbol)
        if ls_ratio:
            print(f"    ✓ L/S Ratio: {ls_ratio:.3f}")
        else:
            print("    ✗ No L/S ratio returned")
        
        # Test Positioning
        print(f"\n[6] Fetching Full Positioning Data for {symbol}...")
        positioning = await client.fetch_positioning(symbol)
        print(f"    L/S Ratio: {positioning.long_short_ratio:.3f}")
        print(f"    Top Trader Long: {positioning.top_trader_long_ratio:.2%}")
        print(f"    Top Trader Short: {positioning.top_trader_short_ratio:.2%}")
        print(f"    Taker Buy: {positioning.taker_buy_ratio:.2%}")
        print(f"    Taker Sell: {positioning.taker_sell_ratio:.2%}")
        
    finally:
        await client.close()
    
    print("\n✓ Binance client tests completed")


async def test_coinalyse_client():
    """Test Coinalyse Liquidation API client."""
    print("\n" + "="*60)
    print("TESTING COINALYSE LIQUIDATION CLIENT")
    print("="*60)
    
    api_key = os.getenv("COINALYSE_API_KEY", "")
    
    if not api_key or api_key == "your_coinalyse_api_key":
        print("\n⚠ COINALYSE_API_KEY not set in .env")
        print("  Liquidation data will not be available")
        print("  Get your API key from: https://coinalyze.net/api/")
        return
    
    client = CoinalyseLiquidationClient(api_key=api_key)
    symbol = "cmt_btcusdt"
    
    try:
        print(f"\n[1] Fetching Liquidations for {symbol}...")
        liqs = await client.fetch_liquidations(symbol, interval="5min", hours_back=1)
        
        if liqs:
            print(f"    ✓ Got {len(liqs)} liquidation entries")
            
            long_total = sum(l.usd_value for l in liqs if l.side.value == "long")
            short_total = sum(l.usd_value for l in liqs if l.side.value == "short")
            
            print(f"    Long Liquidations: ${long_total:,.0f}")
            print(f"    Short Liquidations: ${short_total:,.0f}")
            
            if liqs:
                print(f"    Most recent: {liqs[0].timestamp} | ${liqs[0].usd_value:,.0f} ({liqs[0].side.value})")
        else:
            print("    ✗ No liquidation data returned")
        
        # Test batch fetch
        print(f"\n[2] Fetching Liquidations for ALL pairs (batch)...")
        batch = await client.fetch_liquidations_batch(PERMITTED_PAIRS[:4])  # First 4 pairs
        
        for sym, liq_list in batch.items():
            if liq_list:
                total = sum(l.usd_value for l in liq_list)
                print(f"    {sym}: {len(liq_list)} entries, ${total:,.0f} total")
            else:
                print(f"    {sym}: No data")
        
    finally:
        await client.close()
    
    print("\n✓ Coinalyse client tests completed")


async def test_market_intel_layer():
    """Test full Market Intelligence Layer."""
    print("\n" + "="*60)
    print("TESTING MARKET INTELLIGENCE LAYER")
    print("="*60)
    
    coinalyse_key = os.getenv("COINALYSE_API_KEY", "")
    layer = MarketIntelligenceLayer(coinalyse_api_key=coinalyse_key)
    
    try:
        print("\n[1] Initializing Layer (fetching initial data)...")
        await layer.initialize()
        print("    ✓ Layer initialized")
        
        # Test single symbol refresh
        symbol = "cmt_btcusdt"
        print(f"\n[2] Getting MarketState for {symbol}...")
        state = layer.get_market_state(symbol)
        
        print(f"    Timestamp: {state.timestamp}")
        print(f"    Price: ${state.price:,.2f}")
        print(f"    24h Change: {state.price_change_24h:+.2%}")
        print(f"    Volatility: {state.volatility:.2%}")
        print(f"    24h Volume: ${state.volume_24h:,.0f}")
        
        if state.funding_rate:
            print(f"    Funding: {state.funding_rate.rate:.6f}")
        
        if state.open_interest:
            print(f"    OI: ${state.open_interest.open_interest_usd:,.0f}")
            print(f"    OI Delta: {state.open_interest.delta_pct:+.2%}")
        
        if state.order_book:
            print(f"    Spread: {state.order_book.spread:.4%}")
            print(f"    OB Imbalance: {state.order_book.imbalance:+.3f}")
        
        print(f"    Liquidations: {len(state.recent_liquidations)} entries")
        
        # OHLCV data
        print(f"\n[3] OHLCV Data:")
        for tf in TIMEFRAMES:
            candles = state.ohlcv.get(tf, [])
            print(f"    {tf}: {len(candles)} candles")
        
        # Positioning
        print(f"\n[4] Positioning Data:")
        pos = layer.get_positioning(symbol)
        if pos:
            print(f"    L/S Ratio: {pos.long_short_ratio:.3f}")
            print(f"    Top Trader Long: {pos.top_trader_long_ratio:.2%}")
        
        # Liquidation metrics
        print(f"\n[5] Liquidation Metrics:")
        imbalance = layer.get_liquidation_imbalance(symbol)
        velocity = layer.get_liquidation_velocity(symbol)
        print(f"    Imbalance: {imbalance:+.3f}")
        print(f"    Velocity: {velocity:.4f}")
        
        # Test all symbols
        print(f"\n[6] Getting MarketState for ALL symbols...")
        all_states = layer.get_all_market_states()
        
        for sym, s in all_states.items():
            price_str = f"${s.price:,.2f}" if s.price else "N/A"
            funding_str = f"{s.funding_rate.rate:.5f}" if s.funding_rate else "N/A"
            oi_str = f"${s.open_interest.open_interest_usd/1e6:.1f}M" if s.open_interest else "N/A"
            print(f"    {sym}: {price_str} | Funding: {funding_str} | OI: {oi_str}")
        
    finally:
        await layer.close()
    
    print("\n✓ Market Intelligence Layer tests completed")


async def main():
    print("="*60)
    print("HYDRA LAYER 1: MARKET INTELLIGENCE - TEST SUITE")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*60)
    
    # Run tests
    await test_binance_client()
    await test_coinalyse_client()
    await test_market_intel_layer()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
