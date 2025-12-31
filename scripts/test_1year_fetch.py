"""Quick test for 1-year data fetching."""
import asyncio
from datetime import datetime, timezone, timedelta
from hydra.training.historical_data import BinanceHistoricalFetcher

async def main():
    f = BinanceHistoricalFetcher()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)
    
    print(f"Testing 1-year fetch (15min): {start.date()} to {end.date()}")
    df = await f.fetch_ohlcv("cmt_btcusdt", "15min", start, end)
    await f.close()
    
    print(f"\nResult: {len(df):,} candles")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nExpected ~35,040 candles (365 days * 24 hours * 4 per hour)")

if __name__ == "__main__":
    asyncio.run(main())
