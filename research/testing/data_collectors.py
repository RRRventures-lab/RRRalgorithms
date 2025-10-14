from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import numpy as np
import pandas as pd
import time


"""
Data collection utilities for hypothesis testing.

This module provides unified interfaces for collecting data from various
free-tier APIs including Coinbase, Binance, Etherscan, DeFiLlama, etc.
"""



class DataCollector(ABC):
    """Base class for all data collectors."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize data collector.

        Args:
            api_key: Optional API key for authenticated endpoints
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    @abstractmethod
    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect historical data.

        Args:
            start_date: Start of collection period
            end_date: End of collection period
            **kwargs: Additional parameters specific to data source

        Returns:
            DataFrame with collected data
        """
        pass


class CoinbaseDataCollector(DataCollector):
    """Collect data from Coinbase Pro (free tier)."""

    BASE_URL = "https://api.exchange.coinbase.com"

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "BTC-USD",
        granularity: int = 3600  # 1 hour in seconds
    ) -> pd.DataFrame:
        """
        Collect OHLCV data from Coinbase Pro.

        Args:
            start_date: Start date
            end_date: End date
            symbol: Trading pair (e.g., "BTC-USD")
            granularity: Candle size in seconds (60, 300, 900, 3600, 21600, 86400)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        all_candles = []

        async with aiohttp.ClientSession() as session:
            current_end = end_date
            current_start = max(start_date, current_end - timedelta(hours=300))

            while current_start > start_date:
                await self._rate_limit()

                url = f"{self.BASE_URL}/products/{symbol}/candles"
                params = {
                    "start": current_start.isoformat(),
                    "end": current_end.isoformat(),
                    "granularity": granularity
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        candles = await response.json()
                        all_candles.extend(candles)
                    else:
                        print(f"Error fetching Coinbase data: {response.status}")

                current_end = current_start
                current_start = max(start_date, current_end - timedelta(hours=300))

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'low', 'high', 'open', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df


class BinanceDataCollector(DataCollector):
    """Collect data from Binance (free tier)."""

    BASE_URL = "https://api.binance.com/api/v3"

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "BTCUSDT",
        interval: str = "1h"  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    ) -> pd.DataFrame:
        """
        Collect OHLCV data from Binance.

        Args:
            start_date: Start date
            end_date: End date
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval

        Returns:
            DataFrame with OHLCV data
        """
        all_klines = []

        async with aiohttp.ClientSession() as session:
            current_start = start_date

            while current_start < end_date:
                await self._rate_limit()

                url = f"{self.BASE_URL}/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(end_date.timestamp() * 1000),
                    "limit": 1000
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        if not klines:
                            break
                        all_klines.extend(klines)
                        # Update current_start to last timestamp
                        current_start = datetime.fromtimestamp(klines[-1][0] / 1000)
                    else:
                        print(f"Error fetching Binance data: {response.status}")
                        break

        # Convert to DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                     'taker_buy_quote', 'ignore']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        return df


class EtherscanDataCollector(DataCollector):
    """Collect on-chain data from Etherscan (free tier)."""

    BASE_URL = "https://api.etherscan.io/api"

    def __init__(self, api_key: str):
        """
        Initialize Etherscan collector.

        Args:
            api_key: Etherscan API key (get free at https://etherscan.io/apis)
        """
        super().__init__(api_key=api_key, rate_limit_delay=0.2)  # 5 req/sec free tier

    async def get_whale_transfers(
        self,
        start_date: datetime,
        end_date: datetime,
        min_value_usd: float = 10_000_000,  # $10M minimum
        token: str = "ETH"
    ) -> pd.DataFrame:
        """
        Get large transfers (whale movements).

        Args:
            start_date: Start date
            end_date: End date
            min_value_usd: Minimum transfer value in USD
            token: Token symbol (ETH or ERC20 address)

        Returns:
            DataFrame with whale transfers
        """
        # Note: This is a simplified implementation
        # Real implementation would query blocks and filter transactions
        transfers = []

        # For demo purposes, create sample data structure
        # In production, you'd query actual blockchain data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start_date, end_date, freq='6H'),
            'from_address': ['0x' + 'a' * 40] * len(pd.date_range(start_date, end_date, freq='6H')),
            'to_address': ['0x' + 'b' * 40] * len(pd.date_range(start_date, end_date, freq='6H')),
            'value_usd': np.random.uniform(min_value_usd, min_value_usd * 5, len(pd.date_range(start_date, end_date, freq='6H'))),
            'token': token
        })

        return df

    async def get_stablecoin_supply(
        self,
        start_date: datetime,
        end_date: datetime,
        token_addresses: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Get daily stablecoin total supply.

        Args:
            start_date: Start date
            end_date: End date
            token_addresses: Dict of {symbol: contract_address}

        Returns:
            DataFrame with daily supply data
        """
        if token_addresses is None:
            token_addresses = {
                "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
            }

        # Simplified implementation - would need to query historical supply
        # For now, create sample data
        dates = pd.date_range(start_date, end_date, freq='D')
        data = {}

        for symbol in token_addresses.keys():
            data[f'{symbol}_supply'] = np.cumsum(np.random.randint(-1000000, 2000000, len(dates))) + 50_000_000_000

        df = pd.DataFrame(data, index=dates)
        df.index.name = 'timestamp'
        df = df.reset_index()

        return df


class DeFiLlamaDataCollector(DataCollector):
    """Collect DeFi data from DeFiLlama (free API)."""

    BASE_URL = "https://api.llama.fi"

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        protocol: str = "uniswap"
    ) -> pd.DataFrame:
        """
        Collect TVL data for a DeFi protocol.

        Args:
            start_date: Start date
            end_date: End date
            protocol: Protocol slug (e.g., "uniswap", "aave", "curve")

        Returns:
            DataFrame with TVL data
        """
        async with aiohttp.ClientSession() as session:
            await self._rate_limit()

            url = f"{self.BASE_URL}/protocol/{protocol}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract TVL timeseries
                    if 'tvl' in data:
                        tvl_data = data['tvl']
                        df = pd.DataFrame(tvl_data)
                        df['timestamp'] = pd.to_datetime(df['date'], unit='s')
                        df = df[['timestamp', 'totalLiquidityUSD']].copy()
                        df = df.rename(columns={'totalLiquidityUSD': 'tvl'})

                        # Filter by date range
                        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

                        return df
                else:
                    print(f"Error fetching DeFiLlama data: {response.status}")
                    return pd.DataFrame()


class UniswapDataCollector(DataCollector):
    """Collect DEX data from Uniswap subgraph (free)."""

    SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        pool_address: str = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"  # USDC/ETH 0.3%
    ) -> pd.DataFrame:
        """
        Collect DEX price and liquidity data.

        Args:
            start_date: Start date
            end_date: End date
            pool_address: Uniswap pool address

        Returns:
            DataFrame with price and liquidity data
        """
        # GraphQL query
        query = """
        query ($pool: String!, $startTime: Int!, $endTime: Int!) {
          poolHourDatas(
            where: {
              pool: $pool,
              periodStartUnix_gte: $startTime,
              periodStartUnix_lte: $endTime
            },
            orderBy: periodStartUnix,
            orderDirection: asc
          ) {
            periodStartUnix
            token0Price
            token1Price
            liquidity
            volumeUSD
          }
        }
        """

        variables = {
            "pool": pool_address.lower(),
            "startTime": int(start_date.timestamp()),
            "endTime": int(end_date.timestamp())
        }

        async with aiohttp.ClientSession() as session:
            await self._rate_limit()

            async with session.post(
                self.SUBGRAPH_URL,
                json={"query": query, "variables": variables}
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'data' in data and 'poolHourDatas' in data['data']:
                        pool_data = data['data']['poolHourDatas']
                        df = pd.DataFrame(pool_data)

                        if not df.empty:
                            df['timestamp'] = pd.to_datetime(df['periodStartUnix'], unit='s')
                            df['price'] = df['token0Price'].astype(float)
                            df['liquidity'] = df['liquidity'].astype(float)
                            df['volume'] = df['volumeUSD'].astype(float)

                            df = df[['timestamp', 'price', 'liquidity', 'volume']].copy()
                            return df

                print(f"Error fetching Uniswap data: {response.status}")
                return pd.DataFrame()


class CoingeckoDataCollector(DataCollector):
    """Collect price data from Coingecko (free tier)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        coin_id: str = "bitcoin"
    ) -> pd.DataFrame:
        """
        Collect historical price data.

        Args:
            start_date: Start date
            end_date: End date
            coin_id: Coingecko coin ID (e.g., "bitcoin", "ethereum")

        Returns:
            DataFrame with price data
        """
        async with aiohttp.ClientSession() as session:
            await self._rate_limit()

            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": int(start_date.timestamp()),
                "to": int(end_date.timestamp())
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract prices
                    prices = data.get('prices', [])
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Add volume if available
                    if 'total_volumes' in data:
                        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
                        df = df.merge(volumes, on='timestamp', how='left')

                    return df
                else:
                    print(f"Error fetching Coingecko data: {response.status}")
                    return pd.DataFrame()


class MultiSourceCollector:
    """Collect data from multiple sources and merge."""

    def __init__(self):
        """Initialize multi-source collector."""
        self.collectors = {
            'coinbase': CoinbaseDataCollector(),
            'binance': BinanceDataCollector(),
            'coingecko': CoingeckoDataCollector(),
            'defillama': DeFiLlamaDataCollector(),
            'uniswap': UniswapDataCollector()
        }

    async def collect_price_data(
        self,
        start_date: datetime,
        end_date: datetime,
        sources: List[str] = None
    ) -> pd.DataFrame:
        """
        Collect price data from multiple sources and merge.

        Args:
            start_date: Start date
            end_date: End date
            sources: List of source names to use (default: all)

        Returns:
            Merged DataFrame with price data
        """
        if sources is None:
            sources = ['coinbase', 'binance', 'coingecko']

        tasks = []
        for source in sources:
            if source in self.collectors:
                if source == 'coinbase':
                    tasks.append(self.collectors[source].collect(start_date, end_date, symbol="BTC-USD"))
                elif source == 'binance':
                    tasks.append(self.collectors[source].collect(start_date, end_date, symbol="BTCUSDT"))
                elif source == 'coingecko':
                    tasks.append(self.collectors[source].collect(start_date, end_date, coin_id="bitcoin"))

        results = await asyncio.gather(*tasks)

        # Merge results
        # For simplicity, just return the first valid result
        for df in results:
            if not df.empty:
                return df

        return pd.DataFrame()

    def add_collector(self, name: str, collector: DataCollector):
        """Add a custom data collector."""
        self.collectors[name] = collector


# Utility functions

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    **kwargs
) -> Optional[Any]:
    """
    Fetch data with exponential backoff retry logic.

    Args:
        session: aiohttp session
        url: URL to fetch
        max_retries: Maximum number of retries
        **kwargs: Additional arguments to session.get()

    Returns:
        Response JSON or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            async with session.get(url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error {response.status} fetching {url}")
                    return None
        except Exception as e:
            print(f"Exception fetching {url}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return None


def resample_to_common_timeframe(
    dfs: List[pd.DataFrame],
    timeframe: str = "1H"
) -> List[pd.DataFrame]:
    """
    Resample multiple DataFrames to a common timeframe.

    Args:
        dfs: List of DataFrames with 'timestamp' column
        timeframe: Target timeframe (e.g., "1H", "1D")

    Returns:
        List of resampled DataFrames
    """
    resampled = []

    for df in dfs:
        if 'timestamp' in df.columns:
            df_copy = df.set_index('timestamp')
            df_resampled = df_copy.resample(timeframe).agg({
                col: 'last' if col != 'volume' else 'sum'
                for col in df_copy.columns
            })
            df_resampled = df_resampled.reset_index()
            resampled.append(df_resampled)
        else:
            resampled.append(df)

    return resampled
