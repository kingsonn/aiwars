"""Test Coinalyze data fetching with rate limit handling."""
import asyncio
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()

from hydra.training.historical_data import HistoricalDataCollector

async def main():
    # Test with 90 days of data
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=90)
    
    print(f"Testing Coinalyze fetch: {start.date()} to {end.date()}")
    print(f"API Key set: {bool(os.getenv('COINALYSE_API_KEY'))}")
    
    collector = HistoricalDataCollector(
        coinalyze_api_key=os.getenv("COINALYSE_API_KEY", "")
    )
    
    try:
        # Test single symbol
        df = await collector.collect_symbol_data(
            "cmt_btcusdt",
            "15min",
            start,
            end,
            use_cache=False  # Force fresh fetch
        )
        
        print(f"\n✓ Success! Fetched {len(df):,} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData completeness:")
        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df) * 100
            print(f"  {col}: {100-null_pct:.1f}% complete")
        
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main())
