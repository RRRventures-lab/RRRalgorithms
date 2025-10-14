from datetime import datetime, timedelta
from ingestion.market_data_ingestor import MarketDataIngestor, OHLCVBar
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
import os
import pytest
import sys

"""
Unit Tests for Market Data Ingestor

Tests data ingestion, WebSocket connections, and data validation.
Critical for maintaining data quality and availability.
"""


# Add data pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/data-pipeline/src'))



class TestMarketDataIngestorInitialization:
    """Test ingestor initialization"""

    def test_initialization_default(self):
        """Test default initialization"""
        ingestor = MarketDataIngestor()
        assert ingestor is not None
        assert hasattr(ingestor, 'subscriptions')

    def test_initialization_with_symbols(self):
        """Test initialization with symbol list"""
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        ingestor = MarketDataIngestor(symbols=symbols)
        assert len(ingestor.symbols) == 3
        assert 'BTC-USD' in ingestor.symbols


class TestWebSocketConnection:
    """Test WebSocket connection management"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor(symbols=['BTC-USD'])

    @pytest.mark.asyncio
    async def test_connect_websocket(self, ingestor):
        """Test WebSocket connection"""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            await ingestor.connect()

            assert ingestor.connected is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, ingestor):
        """Test WebSocket disconnection"""
        with patch('websockets.connect', new_callable=AsyncMock):
            await ingestor.connect()
            await ingestor.disconnect()

            assert ingestor.connected is False

    @pytest.mark.asyncio
    async def test_reconnect_on_failure(self, ingestor):
        """Test automatic reconnection on failure"""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            # First attempt fails, second succeeds
            mock_connect.side_effect = [
                Exception("Connection failed"),
                AsyncMock()
            ]

            await ingestor.connect_with_retry(max_retries=2)

            assert mock_connect.call_count == 2


class TestDataSubscription:
    """Test data subscription management"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    def test_subscribe_to_symbol(self, ingestor):
        """Test subscribing to a symbol"""
        ingestor.subscribe('BTC-USD')

        assert 'BTC-USD' in ingestor.subscriptions

    def test_subscribe_multiple_symbols(self, ingestor):
        """Test subscribing to multiple symbols"""
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        ingestor.subscribe_batch(symbols)

        assert len(ingestor.subscriptions) == 3
        for symbol in symbols:
            assert symbol in ingestor.subscriptions

    def test_unsubscribe_from_symbol(self, ingestor):
        """Test unsubscribing from a symbol"""
        ingestor.subscribe('BTC-USD')
        ingestor.unsubscribe('BTC-USD')

        assert 'BTC-USD' not in ingestor.subscriptions

    def test_subscribe_duplicate(self, ingestor):
        """Test subscribing to same symbol twice"""
        ingestor.subscribe('BTC-USD')
        ingestor.subscribe('BTC-USD')

        # Should not create duplicate subscriptions
        assert ingestor.subscriptions.count('BTC-USD') == 1


class TestMessageProcessing:
    """Test message processing and parsing"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    def test_parse_tick_message(self, ingestor):
        """Test parsing tick message"""
        raw_message = json.dumps({
            'type': 'ticker',
            'symbol': 'BTC-USD',
            'price': 50000.00,
            'volume': 0.5,
            'timestamp': datetime.now().isoformat()
        })

        tick = ingestor.parse_message(raw_message)

        assert tick is not None
        assert tick['symbol'] == 'BTC-USD'
        assert tick['price'] == 50000.00

    def test_parse_trade_message(self, ingestor):
        """Test parsing trade message"""
        raw_message = json.dumps({
            'type': 'trade',
            'symbol': 'ETH-USD',
            'price': 3000.00,
            'size': 1.5,
            'side': 'BUY',
            'timestamp': datetime.now().isoformat()
        })

        trade = ingestor.parse_message(raw_message)

        assert trade['type'] == 'trade'
        assert trade['price'] == 3000.00
        assert trade['side'] == 'BUY'

    def test_parse_invalid_message(self, ingestor):
        """Test handling invalid message"""
        invalid_message = "not a valid json"

        result = ingestor.parse_message(invalid_message)

        assert result is None or 'error' in result

    def test_handle_malformed_data(self, ingestor):
        """Test handling malformed data"""
        malformed = json.dumps({
            'type': 'ticker',
            # Missing required fields
            'symbol': 'BTC-USD'
        })

        result = ingestor.parse_message(malformed)

        # Should handle gracefully without crashing
        assert result is not None


class TestOHLCVAggregation:
    """Test OHLCV candle aggregation"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    def test_aggregate_1min_candle(self, ingestor):
        """Test 1-minute candle aggregation"""
        # Send multiple ticks within 1 minute
        ticks = [
            {'price': 50000, 'volume': 0.1, 'timestamp': datetime.now()},
            {'price': 50100, 'volume': 0.2, 'timestamp': datetime.now()},
            {'price': 49900, 'volume': 0.15, 'timestamp': datetime.now()},
            {'price': 50050, 'volume': 0.25, 'timestamp': datetime.now()},
        ]

        candle = ingestor.aggregate_candle(ticks, timeframe='1m')

        assert candle.open == 50000  # First price
        assert candle.high == 50100  # Highest price
        assert candle.low == 49900   # Lowest price
        assert candle.close == 50050 # Last price
        assert candle.volume == 0.7  # Sum of volumes

    def test_aggregate_5min_candle(self, ingestor):
        """Test 5-minute candle aggregation"""
        base_time = datetime.now()
        ticks = []

        # Generate ticks over 5 minutes
        for i in range(10):
            ticks.append({
                'price': 50000 + i * 10,
                'volume': 0.1,
                'timestamp': base_time + timedelta(seconds=i * 30)
            })

        candle = ingestor.aggregate_candle(ticks, timeframe='5m')

        assert candle.open == 50000
        assert candle.close == 50090

    def test_candle_completion_callback(self, ingestor):
        """Test callback when candle completes"""
        callback_called = False
        completed_candle = None

        def on_candle_complete(candle):
            nonlocal callback_called, completed_candle
            callback_called = True
            completed_candle = candle

        ingestor.on_candle_complete = on_candle_complete

        # Simulate candle completion
        ticks = [{'price': 50000, 'volume': 0.1, 'timestamp': datetime.now()}]
        ingestor.aggregate_candle(ticks, timeframe='1m')
        ingestor.finalize_current_candle()

        assert callback_called is True
        assert completed_candle is not None


class TestDataValidation:
    """Test data validation and quality checks"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    def test_validate_price_data(self, ingestor):
        """Test price data validation"""
        valid_tick = {'symbol': 'BTC-USD', 'price': 50000.00, 'volume': 0.5}
        assert ingestor.validate_tick(valid_tick) is True

        # Invalid: negative price
        invalid_tick = {'symbol': 'BTC-USD', 'price': -100, 'volume': 0.5}
        assert ingestor.validate_tick(invalid_tick) is False

        # Invalid: zero volume
        invalid_tick2 = {'symbol': 'BTC-USD', 'price': 50000, 'volume': 0}
        assert ingestor.validate_tick(invalid_tick2) is False

    def test_detect_stale_data(self, ingestor):
        """Test detection of stale data"""
        # Recent data
        recent_tick = {
            'price': 50000,
            'timestamp': datetime.now()
        }
        assert ingestor.is_stale(recent_tick, max_age_seconds=60) is False

        # Stale data
        stale_tick = {
            'price': 50000,
            'timestamp': datetime.now() - timedelta(minutes=5)
        }
        assert ingestor.is_stale(stale_tick, max_age_seconds=60) is True

    def test_detect_price_spike(self, ingestor):
        """Test detection of abnormal price spikes"""
        # Set baseline
        ingestor.last_price = 50000

        # Normal price movement (1%)
        normal_tick = {'price': 50500}
        assert ingestor.is_spike(normal_tick, threshold=0.05) is False

        # Spike (10%)
        spike_tick = {'price': 55000}
        assert ingestor.is_spike(spike_tick, threshold=0.05) is True


class TestCacheManagement:
    """Test data caching"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    def test_cache_latest_price(self, ingestor):
        """Test caching latest price"""
        ingestor.cache_price('BTC-USD', 50000.00)

        cached_price = ingestor.get_cached_price('BTC-USD')
        assert cached_price == 50000.00

    def test_cache_expiration(self, ingestor):
        """Test cache expiration"""
        ingestor.cache_price('BTC-USD', 50000.00, ttl=1)  # 1 second TTL

        # Immediate retrieval should work
        assert ingestor.get_cached_price('BTC-USD') == 50000.00

        # After expiration
        import time
        time.sleep(1.1)
        assert ingestor.get_cached_price('BTC-USD') is None

    def test_clear_cache(self, ingestor):
        """Test clearing cache"""
        ingestor.cache_price('BTC-USD', 50000.00)
        ingestor.cache_price('ETH-USD', 3000.00)

        ingestor.clear_cache()

        assert ingestor.get_cached_price('BTC-USD') is None
        assert ingestor.get_cached_price('ETH-USD') is None


class TestErrorHandling:
    """Test error handling and recovery"""

    @pytest.fixture
    def ingestor(self):
        return MarketDataIngestor()

    @pytest.mark.asyncio
    async def test_handle_connection_timeout(self, ingestor):
        """Test handling connection timeout"""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await ingestor.connect(timeout=1)

    def test_handle_rate_limit(self, ingestor):
        """Test handling rate limit errors"""
        ingestor.is_rate_limited = True

        # Should not process messages when rate limited
        result = ingestor.process_message({'price': 50000})
        assert result is None or 'rate_limited' in result

    def test_message_queue_overflow(self, ingestor):
        """Test handling message queue overflow"""
        # Fill queue to capacity
        for i in range(ingestor.max_queue_size + 10):
            ingestor.enqueue_message({'id': i})

        # Queue should not exceed max size
        assert ingestor.queue.qsize() <= ingestor.max_queue_size


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
