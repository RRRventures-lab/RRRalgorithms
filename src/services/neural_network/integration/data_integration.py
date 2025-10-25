"""
Data Pipeline Integration

Connects neural network models with real-time data sources.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataPipelineConnector:
    """
    Connects neural network system to data pipeline.

    Fetches data from Polygon.io and other sources.
    """

    def __init__(
        self,
        polygon_client=None,
        supabase_client=None
    ):
        """
        Initialize data pipeline connector.

        Args:
            polygon_client: Polygon.io client
            supabase_client: Supabase client for database access
        """
        self.polygon_client = polygon_client
        self.supabase_client = supabase_client

        logger.info("Initialized DataPipelineConnector")

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1min, 5min, etc.)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        if self.polygon_client:
            # Use Polygon.io client
            try:
                # This would call the actual Polygon.io API
                logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

                # Placeholder - would use actual polygon_client
                data = pd.DataFrame({
                    'timestamp': pd.date_range(start_date, end_date, freq='1min'),
                    'open': np.random.randn(1000) + 100,
                    'high': np.random.randn(1000) + 101,
                    'low': np.random.randn(1000) + 99,
                    'close': np.random.randn(1000) + 100,
                    'volume': np.random.randint(1000000, 10000000, 1000)
                })

                return data

            except Exception as e:
                logger.error(f"Error fetching data from Polygon.io: {e}")
                raise

        elif self.supabase_client:
            # Fetch from Supabase
            try:
                logger.info(f"Fetching {symbol} data from Supabase")

                # Query Supabase
                # result = self.supabase_client.table('market_data')...

                # Placeholder
                data = pd.DataFrame({
                    'timestamp': pd.date_range(start_date, end_date, freq='1min'),
                    'open': np.random.randn(1000) + 100,
                    'high': np.random.randn(1000) + 101,
                    'low': np.random.randn(1000) + 99,
                    'close': np.random.randn(1000) + 100,
                    'volume': np.random.randint(1000000, 10000000, 1000)
                })

                return data

            except Exception as e:
                logger.error(f"Error fetching data from Supabase: {e}")
                raise

        else:
            raise ValueError("No data source configured")

    def prepare_features(
        self,
        data: pd.DataFrame,
        seq_len: int = 100
    ) -> np.ndarray:
        """
        Prepare features for model input.

        Args:
            data: OHLCV data
            seq_len: Sequence length

        Returns:
            Feature array [n_sequences, seq_len, n_features]
        """
        # Extract OHLCV features
        features = data[['open', 'high', 'low', 'close', 'volume']].values

        # Add VWAP if not present
        if 'vwap' not in data.columns:
            vwap = (data['high'] + data['low'] + data['close']).values / 3
            features = np.column_stack([features, vwap])

        # Normalize
        features = self._normalize_features(features)

        # Create sequences
        sequences = []
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])

        return np.array(sequences, dtype=np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        # Robust normalization
        median = np.median(features, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        iqr = q75 - q25 + 1e-8

        return (features - median) / iqr


class RealtimeDataStream:
    """
    Real-time data streaming for live predictions.

    Maintains connection to data sources and streams updates.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = '1min',
        buffer_size: int = 100
    ):
        """
        Initialize real-time data stream.

        Args:
            symbols: Symbols to stream
            timeframe: Timeframe for aggregation
            buffer_size: Size of data buffer
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.buffer_size = buffer_size

        # Data buffers for each symbol
        self.buffers = {symbol: [] for symbol in symbols}

        # Callbacks
        self.on_data_callbacks: List[Callable] = []

        logger.info(f"Initialized RealtimeDataStream for {symbols}")

    def subscribe(self, callback: Callable):
        """
        Subscribe to data updates.

        Args:
            callback: Function to call on new data
        """
        self.on_data_callbacks.append(callback)

    async def start(self):
        """Start streaming data."""
        logger.info("Starting real-time data stream...")

        # This would connect to WebSocket or similar
        # For now, simulate with periodic updates
        while True:
            # Simulate receiving new data
            await asyncio.sleep(1)  # 1 second intervals

            for symbol in self.symbols:
                # Generate fake tick
                new_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'open': np.random.randn() + 100,
                    'high': np.random.randn() + 101,
                    'low': np.random.randn() + 99,
                    'close': np.random.randn() + 100,
                    'volume': np.random.randint(1000, 10000)
                }

                # Update buffer
                self.buffers[symbol].append(new_data)
                if len(self.buffers[symbol]) > self.buffer_size:
                    self.buffers[symbol].pop(0)

                # Notify callbacks
                for callback in self.on_data_callbacks:
                    try:
                        await callback(symbol, new_data, self.get_buffer(symbol))
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")

    def get_buffer(self, symbol: str) -> pd.DataFrame:
        """
        Get current buffer for symbol.

        Args:
            symbol: Symbol

        Returns:
            DataFrame with buffered data
        """
        if symbol not in self.buffers or not self.buffers[symbol]:
            return pd.DataFrame()

        return pd.DataFrame(self.buffers[symbol])

    async def stop(self):
        """Stop streaming."""
        logger.info("Stopping real-time data stream...")
        # Would close WebSocket connection


if __name__ == "__main__":
    # Test data integration
    print("Testing Data Integration...")

    # Test DataPipelineConnector
    print("\n1. Testing DataPipelineConnector...")
    connector = DataPipelineConnector()

    # Fetch historical data (simulated)
    async def test_historical():
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        data = await connector.fetch_historical_data(
            symbol="X:BTCUSD",
            timeframe="1min",
            start_date=start_date,
            end_date=end_date
        )

        print(f"   Fetched {len(data)} data points")
        print(f"   Columns: {list(data.columns)}")

        # Prepare features
        features = connector.prepare_features(data, seq_len=100)
        print(f"   Features shape: {features.shape}")

    asyncio.run(test_historical())

    # Test RealtimeDataStream
    print("\n2. Testing RealtimeDataStream...")

    async def test_streaming():
        stream = RealtimeDataStream(symbols=["X:BTCUSD", "X:ETHUSD"])

        # Subscribe to updates
        async def on_data(symbol, data, buffer):
            if len(buffer) >= 10:
                print(f"   {symbol}: Buffer size = {len(buffer)}, Latest close = {data['close']:.2f}")

        stream.subscribe(on_data)

        # Run for a few seconds
        task = asyncio.create_task(stream.start())
        await asyncio.sleep(5)
        task.cancel()

        print("   Streaming test completed")

    asyncio.run(test_streaming())

    print("\nData integration tests completed!")
