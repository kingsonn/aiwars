"""Test script to check OI and liquidation data from Binance."""

import asyncio
import ccxt.async_support as ccxt
import aiohttp


async def test_ccxt_oi():
    """Test OI via ccxt."""
    print("=" * 50)
    print("Testing OI via CCXT")
    print("=" * 50)
    
    ex = ccxt.binanceusdm()
    await ex.load_markets()
    
    try:
        oi = await ex.fetch_open_interest('BTC/USDT:USDT')
        print(f"CCXT OI Response: {oi}")
        print(f"  openInterestAmount: {oi.get('openInterestAmount')}")
        print(f"  openInterestValue: {oi.get('openInterestValue')}")
        print(f"  info: {oi.get('info')}")
    except Exception as e:
        print(f"Error: {e}")
    
    await ex.close()


async def test_binance_direct_oi():
    """Test OI via direct Binance API."""
    print("\n" + "=" * 50)
    print("Testing OI via Direct Binance API")
    print("=" * 50)
    
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    
    async with aiohttp.ClientSession() as session:
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            params = {"symbol": symbol}
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                print(f"\n{symbol}:")
                print(f"  Response: {data}")
                if 'openInterest' in data:
                    oi = float(data['openInterest'])
                    print(f"  Open Interest: {oi:,.2f} contracts")


async def test_binance_liquidations():
    """Test liquidations via direct Binance API."""
    print("\n" + "=" * 50)
    print("Testing Liquidations via Direct Binance API")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test Long/Short Ratio (public endpoint - works)
        print("\n--- Testing Long/Short Ratio endpoint ---")
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {"symbol": "BTCUSDT", "period": "5m", "limit": 3}
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            print(f"Long/Short Ratio: {data}")
        
        # Test Top Trader Long/Short Ratio (public)
        print("\n--- Testing Top Trader Positions ---")
        url2 = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        params2 = {"symbol": "BTCUSDT", "period": "5m", "limit": 3}
        async with session.get(url2, params=params2) as resp:
            data = await resp.json()
            print(f"Top Trader Position Ratio: {data}")
        
        # Test Taker Buy/Sell Volume (public)
        print("\n--- Testing Taker Buy/Sell Volume ---")
        url3 = "https://fapi.binance.com/futures/data/takerlongshortRatio"
        params3 = {"symbol": "BTCUSDT", "period": "5m", "limit": 3}
        async with session.get(url3, params=params3) as resp:
            data = await resp.json()
            print(f"Taker Long/Short: {data}")


async def test_oi_calculation():
    """Debug OI calculation."""
    print("\n" + "=" * 50)
    print("Debugging OI Calculation")
    print("=" * 50)
    
    ex = ccxt.binanceusdm()
    await ex.load_markets()
    
    ticker = await ex.fetch_ticker('BTC/USDT:USDT')
    oi = await ex.fetch_open_interest('BTC/USDT:USDT')
    
    price = float(ticker.get('last', 0))
    oi_amount = float(oi.get('openInterestAmount', 0))
    oi_value_api = oi.get('openInterestValue')
    oi_value_calc = oi_amount * price
    
    print(f"BTC Price: ${price:,.2f}")
    print(f"OI Amount (contracts): {oi_amount:,.2f}")
    print(f"OI Value (from API): {oi_value_api}")
    print(f"OI Value (calculated): ${oi_value_calc:,.0f}")
    print(f"OI Value in Billions: ${oi_value_calc/1e9:.2f}B")
    
    await ex.close()


async def test_layer1_data():
    """Test OI and L/S ratio via Layer 1."""
    print("\n" + "=" * 50)
    print("Testing Layer 1 Data Fetching")
    print("=" * 50)
    
    from hydra.core.config import HydraConfig
    from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
    
    config = HydraConfig.load()
    layer1 = MarketIntelligenceLayer(config)
    await layer1.setup()
    
    for sym in config.trading.symbols[:3]:
        print(f"\n{sym}:")
        oi = layer1.get_open_interest(sym)
        ls = layer1.get_long_short_ratio(sym)
        
        if oi:
            print(f"  OI: {oi.open_interest:,.0f} contracts = ${oi.open_interest_usd/1e9:.2f}B")
        else:
            print(f"  OI: None")
        
        if ls:
            print(f"  L/S Ratio: {ls['ratio']:.2f} (L:{ls['longAccount']:.1%} S:{ls['shortAccount']:.1%})")
        else:
            print(f"  L/S Ratio: None")
    
    await layer1.stop_feeds()


async def main():
    await test_oi_calculation()
    await test_layer1_data()


if __name__ == "__main__":
    asyncio.run(main())
