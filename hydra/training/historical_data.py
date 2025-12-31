"""
Historical Data Fetcher for ML Training

Fetches historical data from:
- Binance API: OHLCV candles
- Coinalyze API: Open Interest, Funding Rate, Liquidation history

All data uses consistent intervals and date ranges for training.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import aiohttp
import pandas as pd
import numpy as np
from loguru import logger


# =============================================================================
# CONSTANTS
# =============================================================================

PERMITTED_PAIRS = [
    "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_bnbusdt",
    "cmt_adausdt", "cmt_xrpusdt", "cmt_ltcusdt", "cmt_dogeusdt"
]

# Coinalyze symbol mapping
COINALYZE_SYMBOL_MAP = {
    "cmt_btcusdt": "BTCUSDT_PERP.A",
    "cmt_ethusdt": "ETHUSDT_PERP.A",
    "cmt_solusdt": "SOLUSDT_PERP.A",
    "cmt_bnbusdt": "BNBUSDT_PERP.A",
    "cmt_adausdt": "ADAUSDT_PERP.A",
    "cmt_xrpusdt": "XRPUSDT_PERP.A",
    "cmt_ltcusdt": "LTCUSDT_PERP.A",
    "cmt_dogeusdt": "DOGEUSDT_PERP.A",
}

# Binance symbol mapping
BINANCE_SYMBOL_MAP = {
    "cmt_btcusdt": "BTCUSDT",
    "cmt_ethusdt": "ETHUSDT",
    "cmt_solusdt": "SOLUSDT",
    "cmt_bnbusdt": "BNBUSDT",
    "cmt_adausdt": "ADAUSDT",
    "cmt_xrpusdt": "XRPUSDT",
    "cmt_ltcusdt": "LTCUSDT",
    "cmt_dogeusdt": "DOGEUSDT",
}

# Interval mapping
INTERVAL_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "4hour": 240,
    "daily": 1440,
}

BINANCE_INTERVAL_MAP = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1hour": "1h",
    "4hour": "4h",
    "daily": "1d",
}


# =============================================================================
# BINANCE HISTORICAL OHLCV FETCHER
# =============================================================================

class BinanceHistoricalFetcher:
    """Fetches historical OHLCV data from Binance Futures API."""
    
    BASE_URL = "https://fapi.binance.com"
    MAX_LIMIT = 1500  # Binance max per request
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Internal symbol (cmt_btcusdt)
            interval: Interval string (5min, 1hour, etc.)
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        binance_symbol = BINANCE_SYMBOL_MAP.get(symbol)
        binance_interval = BINANCE_INTERVAL_MAP.get(interval)
        
        if not binance_symbol or not binance_interval:
            raise ValueError(f"Unknown symbol or interval: {symbol}, {interval}")
        
        session = await self._get_session()
        all_data = []
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        current_ts = start_ts
        
        interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
        
        total_intervals = (end_ts - start_ts) // interval_ms
        logger.info(f"Fetching OHLCV for {symbol} from {start_date.date()} to {end_date.date()} (~{total_intervals} candles)")
        
        while current_ts < end_ts:
            try:
                url = f"{self.BASE_URL}/fapi/v1/klines"
                params = {
                    "symbol": binance_symbol,
                    "interval": binance_interval,
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": self.MAX_LIMIT,
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data:
                            break
                        
                        all_data.extend(data)
                        
                        # Move to next batch
                        last_ts = data[-1][0]
                        current_ts = last_ts + interval_ms
                        
                        # Progress logging every 10k candles
                        if len(all_data) % 10000 < self.MAX_LIMIT:
                            pct = min(100, len(all_data) / total_intervals * 100) if total_intervals > 0 else 100
                            logger.info(f"  Progress: {len(all_data):,} candles ({pct:.1f}%)")
                    elif resp.status == 429:
                        logger.warning("Rate limited, waiting 60s...")
                        await asyncio.sleep(60)
                    else:
                        text = await resp.text()
                        logger.error(f"Binance error {resp.status}: {text[:200]}")
                        break
                
                await asyncio.sleep(0.1)  # Rate limit
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        
        logger.info(f"Fetched {len(df)} OHLCV candles for {symbol}")
        
        return df


# =============================================================================
# COINALYZE HISTORICAL DATA FETCHER
# =============================================================================

class CoinalyzeHistoricalFetcher:
    """
    Fetches historical data from Coinalyze API:
    - Open Interest history
    - Funding Rate history
    - Liquidation history
    """
    
    BASE_URL = "https://api.coinalyze.net/v1"
    
    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.getenv("COINALYSE_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {"api_key": self._api_key}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def fetch_open_interest(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical open interest from Coinalyze.
        
        Response format: {"t": timestamp, "o": open, "h": high, "l": low, "c": close}
        We use "c" (close) as the OI value at that interval.
        """
        if not self._api_key:
            logger.warning("COINALYSE_API_KEY not set, OI data unavailable")
            return pd.DataFrame()
        
        coinalyze_symbol = COINALYZE_SYMBOL_MAP.get(symbol)
        if not coinalyze_symbol:
            return pd.DataFrame()
        
        session = await self._get_session()
        all_data = []
        
        # Use smaller chunks to avoid rate limits - 7 days per chunk for reliability
        chunk_days = 7
        current_start = start_date
        retry_count = 0
        max_retries = 3
        
        total_days = (end_date - start_date).days
        logger.info(f"Fetching OI history for {symbol} from {start_date.date()} to {end_date.date()} (~{total_days} days)")
        
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            try:
                url = f"{self.BASE_URL}/open-interest-history"
                params = {
                    "symbols": coinalyze_symbol,
                    "interval": interval,
                    "from": int(current_start.timestamp()),
                    "to": int(chunk_end.timestamp()),
                    "convert_to_usd": "true",
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data and len(data) > 0:
                            history = data[0].get("history", [])
                            all_data.extend(history)
                            days_done = (chunk_end - start_date).days
                            pct = min(100, days_done / total_days * 100) if total_days > 0 else 100
                            logger.info(f"  OI Progress: {days_done}/{total_days} days ({pct:.1f}%) - {len(all_data)} entries")
                        retry_count = 0  # Reset on success
                    elif resp.status == 429:
                        # Rate limited - exponential backoff
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Max retries exceeded for OI, skipping remaining data")
                            break
                        wait_time = min(60, 5 * (2 ** retry_count))
                        logger.warning(f"Rate limited (429), waiting {wait_time}s... (retry {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue  # Retry same chunk
                    elif resp.status == 401:
                        logger.error("Coinalyze API key invalid")
                        break
                    else:
                        text = await resp.text()
                        logger.warning(f"Coinalyze OI error {resp.status}: {text[:200]}")
                        retry_count += 1
                        if retry_count > max_retries:
                            break
                        await asyncio.sleep(5)
                        continue
                
                current_start = chunk_end
                await asyncio.sleep(2.0)  # Conservative rate limit - 2 seconds between requests
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching OI, retrying...")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error fetching OI: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["open_interest"] = df["c"].astype(float)  # Use close as OI value
        df = df[["timestamp", "open_interest"]]
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        
        logger.info(f"Fetched {len(df)} OI history entries for {symbol}")
        
        return df
    
    async def fetch_funding_rate(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Coinalyze.
        
        Response format: {"t": timestamp, "o": open, "h": high, "l": low, "c": close}
        We use "c" (close) as the funding rate at that interval.
        """
        if not self._api_key:
            logger.warning("COINALYSE_API_KEY not set, funding data unavailable")
            return pd.DataFrame()
        
        coinalyze_symbol = COINALYZE_SYMBOL_MAP.get(symbol)
        if not coinalyze_symbol:
            return pd.DataFrame()
        
        session = await self._get_session()
        all_data = []
        
        chunk_days = 7  # Smaller chunks for reliability
        current_start = start_date
        retry_count = 0
        max_retries = 3
        
        total_days = (end_date - start_date).days
        logger.info(f"Fetching funding history for {symbol} from {start_date.date()} to {end_date.date()} (~{total_days} days)")
        
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            try:
                url = f"{self.BASE_URL}/funding-rate-history"
                params = {
                    "symbols": coinalyze_symbol,
                    "interval": interval,
                    "from": int(current_start.timestamp()),
                    "to": int(chunk_end.timestamp()),
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data and len(data) > 0:
                            history = data[0].get("history", [])
                            all_data.extend(history)
                            days_done = (chunk_end - start_date).days
                            pct = min(100, days_done / total_days * 100) if total_days > 0 else 100
                            logger.info(f"  Funding Progress: {days_done}/{total_days} days ({pct:.1f}%) - {len(all_data)} entries")
                        retry_count = 0
                    elif resp.status == 429:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Max retries exceeded for funding, skipping remaining data")
                            break
                        wait_time = min(60, 5 * (2 ** retry_count))
                        logger.warning(f"Rate limited (429), waiting {wait_time}s... (retry {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    elif resp.status == 401:
                        logger.error("Coinalyze API key invalid")
                        break
                    else:
                        text = await resp.text()
                        logger.warning(f"Coinalyze funding error {resp.status}: {text[:200]}")
                        retry_count += 1
                        if retry_count > max_retries:
                            break
                        await asyncio.sleep(5)
                        continue
                
                current_start = chunk_end
                await asyncio.sleep(1.0)
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching funding, retrying...")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error fetching funding: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["funding_rate"] = df["c"].astype(float)  # Use close as funding value
        df = df[["timestamp", "funding_rate"]]
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        
        logger.info(f"Fetched {len(df)} funding history entries for {symbol}")
        
        return df
    
    async def fetch_liquidations(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical liquidation data from Coinalyze.
        
        Response format: {"t": timestamp, "l": long_liq, "s": short_liq}
        """
        if not self._api_key:
            logger.warning("COINALYSE_API_KEY not set, liquidation data unavailable")
            return pd.DataFrame()
        
        coinalyze_symbol = COINALYZE_SYMBOL_MAP.get(symbol)
        if not coinalyze_symbol:
            return pd.DataFrame()
        
        session = await self._get_session()
        all_data = []
        
        chunk_days = 7  # Smaller chunks for reliability
        current_start = start_date
        retry_count = 0
        max_retries = 3
        
        total_days = (end_date - start_date).days
        logger.info(f"Fetching liquidation history for {symbol} from {start_date.date()} to {end_date.date()} (~{total_days} days)")
        
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            try:
                url = f"{self.BASE_URL}/liquidation-history"
                params = {
                    "symbols": coinalyze_symbol,
                    "interval": interval,
                    "from": int(current_start.timestamp()),
                    "to": int(chunk_end.timestamp()),
                    "convert_to_usd": "true",
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data and len(data) > 0:
                            history = data[0].get("history", [])
                            all_data.extend(history)
                            days_done = (chunk_end - start_date).days
                            pct = min(100, days_done / total_days * 100) if total_days > 0 else 100
                            logger.info(f"  Liquidation Progress: {days_done}/{total_days} days ({pct:.1f}%) - {len(all_data)} entries")
                        retry_count = 0
                    elif resp.status == 429:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Max retries exceeded for liquidations, skipping remaining data")
                            break
                        wait_time = min(60, 5 * (2 ** retry_count))
                        logger.warning(f"Rate limited (429), waiting {wait_time}s... (retry {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    elif resp.status == 401:
                        logger.error("Coinalyze API key invalid")
                        break
                    else:
                        text = await resp.text()
                        logger.warning(f"Coinalyze liq error {resp.status}: {text[:200]}")
                        retry_count += 1
                        if retry_count > max_retries:
                            break
                        await asyncio.sleep(5)
                        continue
                
                current_start = chunk_end
                await asyncio.sleep(1.0)
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching liquidations, retrying...")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error fetching liquidations: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    break
                await asyncio.sleep(5)
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["long_liquidations"] = df["l"].astype(float)
        df["short_liquidations"] = df["s"].astype(float)
        df["total_liquidations"] = df["long_liquidations"] + df["short_liquidations"]
        df = df[["timestamp", "long_liquidations", "short_liquidations", "total_liquidations"]]
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        
        logger.info(f"Fetched {len(df)} liquidation history entries for {symbol}")
        
        return df


# =============================================================================
# COMBINED HISTORICAL DATA COLLECTOR
# =============================================================================

class HistoricalDataCollector:
    """
    Collects all historical data needed for ML training.
    
    Combines:
    - Binance: OHLCV
    - Coinalyze: OI, Funding, Liquidations
    
    Ensures consistent intervals and date ranges.
    """
    
    def __init__(self, coinalyze_api_key: str = ""):
        self.binance = BinanceHistoricalFetcher()
        self.coinalyze = CoinalyzeHistoricalFetcher(api_key=coinalyze_api_key)
        self._cache_dir = Path("./data/historical")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def close(self):
        await self.binance.close()
        await self.coinalyze.close()
    
    async def collect_symbol_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Collect all historical data for a single symbol.
        
        Returns merged DataFrame with columns:
        - open, high, low, close, volume (OHLCV)
        - open_interest
        - funding_rate
        - long_liquidations, short_liquidations, total_liquidations
        """
        cache_file = self._cache_dir / f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}.parquet"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Collecting data for {symbol} ({interval}) from {start_date.date()} to {end_date.date()}")
        
        # Fetch OHLCV first (fast, no rate limits)
        ohlcv_df = await self.binance.fetch_ohlcv(symbol, interval, start_date, end_date)
        
        # Fetch Coinalyze data at hourly interval (API limits prevent 5min for long ranges)
        # We'll resample/interpolate to match OHLCV frequency later
        coinalyze_interval = "1hour" if interval in ["1min", "5min", "15min"] else interval
        logger.info(f"Fetching Coinalyze data for {symbol} at {coinalyze_interval} (will resample to {interval})...")
        oi_df = await self.coinalyze.fetch_open_interest(symbol, coinalyze_interval, start_date, end_date)
        funding_df = await self.coinalyze.fetch_funding_rate(symbol, coinalyze_interval, start_date, end_date)
        liq_df = await self.coinalyze.fetch_liquidations(symbol, coinalyze_interval, start_date, end_date)
        
        if ohlcv_df.empty:
            logger.error(f"No OHLCV data for {symbol}")
            return pd.DataFrame()
        
        # Merge all data on timestamp index
        merged = ohlcv_df.copy()
        
        if not oi_df.empty:
            merged = merged.join(oi_df, how="left")
        else:
            merged["open_interest"] = np.nan
        
        if not funding_df.empty:
            merged = merged.join(funding_df, how="left")
        else:
            merged["funding_rate"] = np.nan
        
        if not liq_df.empty:
            merged = merged.join(liq_df, how="left")
        else:
            merged["long_liquidations"] = 0.0
            merged["short_liquidations"] = 0.0
            merged["total_liquidations"] = 0.0
        
        # Forward fill OI and funding (they update less frequently)
        merged["open_interest"] = merged["open_interest"].ffill()
        merged["funding_rate"] = merged["funding_rate"].ffill()
        
        # Fill liquidations with 0 (no liquidations in that period)
        merged["long_liquidations"] = merged["long_liquidations"].fillna(0)
        merged["short_liquidations"] = merged["short_liquidations"].fillna(0)
        merged["total_liquidations"] = merged["total_liquidations"].fillna(0)
        
        # Add symbol metadata
        merged.attrs["symbol"] = symbol
        
        # Cache result
        merged.to_parquet(cache_file)
        logger.info(f"Cached data for {symbol}: {len(merged)} rows")
        
        return merged
    
    async def collect_all_symbols(
        self,
        symbols: list[str],
        interval: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Collect data for all symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                df = await self.collect_symbol_data(
                    symbol, interval, start_date, end_date, use_cache
                )
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def collect_training_data(
    symbols: list[str] = None,
    interval: str = "5min",
    days_back: int = 90,
    end_date: datetime = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience function to collect training data.
    
    Args:
        symbols: List of symbols (default: all 8 pairs)
        interval: Data interval (default: 5min)
        days_back: Days of history (default: 90)
        end_date: End date (default: now)
    
    Returns:
        Dict of symbol -> DataFrame
    """
    symbols = symbols or PERMITTED_PAIRS
    end_date = end_date or datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    coinalyze_key = os.getenv("COINALYSE_API_KEY", "")
    collector = HistoricalDataCollector(coinalyze_api_key=coinalyze_key)
    
    try:
        data = await collector.collect_all_symbols(
            symbols, interval, start_date, end_date
        )
        return data
    finally:
        await collector.close()
