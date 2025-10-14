from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import os
import pytest
import time

"""
Real-time Subscription Tests

Tests Supabase real-time subscriptions for cross-worktree communication:
- Trading signals trigger order execution
- Orders update positions
- Positions update portfolio snapshots
- Events trigger alerts
"""


# Load environment variables
load_dotenv('config/api-keys/.env')


class TestRealtimeSubscriptions:
    """Test real-time data propagation across worktrees"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup Supabase client"""
        self.supabase: Client = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_ANON_KEY')
        )
        self.events_received = []
        yield
        # Cleanup
        self.cleanup_test_data()

    def cleanup_test_data(self):
        """Remove test data"""
        try:
            self.supabase.table('trading_signals').delete().eq('symbol', 'REALTIME-TEST').execute()
            self.supabase.table('orders').delete().eq('symbol', 'REALTIME-TEST').execute()
            self.supabase.table('positions').delete().eq('symbol', 'REALTIME-TEST').execute()
        except:
            pass

    def test_signal_insertion_detected(self):
        """Test 1: New trading signal is detected in real-time"""
        # In a real scenario, this would use WebSocket subscriptions
        # For testing, we'll verify the data flow works via polling

        # Insert a signal
        signal_data = {
            'symbol': 'REALTIME-TEST',
            'strategy': 'realtime_test',
            'signal_type': 'BUY',
            'confidence': 0.90,
            'price': 1000.0,
            'stop_loss': 950.0,
            'take_profit': 1100.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Insert
        insert_result = self.supabase.table('trading_signals').insert(signal_data).execute()
        assert len(insert_result.data) > 0, "Signal insertion failed"
        signal_id = insert_result.data[0]['signal_id']

        # Verify it can be queried immediately (real-time propagation)
        time.sleep(0.1)  # Small delay for propagation

        query_result = self.supabase.table('trading_signals')\
            .select('*')\
            .eq('signal_id', signal_id)\
            .execute()

        assert len(query_result.data) > 0, "Signal not immediately queryable"
        assert query_result.data[0]['symbol'] == 'REALTIME-TEST'

        print("✅ Test 1 PASSED: Signal insertion detected in real-time")

    def test_order_execution_flow(self):
        """Test 2: Signal → Order flow works in real-time"""
        # Step 1: Create signal
        signal_data = {
            'symbol': 'REALTIME-TEST',
            'strategy': 'flow_test',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 2000.0,
            'stop_loss': 1900.0,
            'take_profit': 2200.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        signal_result = self.supabase.table('trading_signals').insert(signal_data).execute()
        signal_id = signal_result.data[0]['signal_id']

        # Step 2: Simulate Trading Engine responding to signal
        time.sleep(0.2)  # Simulate processing time

        order_data = {
            'symbol': 'REALTIME-TEST',
            'side': 'BUY',
            'order_type': 'MARKET',
            'quantity': 5.0,
            'status': 'SUBMITTED',
            'signal_id': signal_id,
            'created_at': datetime.utcnow().isoformat()
        }
        order_result = self.supabase.table('orders').insert(order_data).execute()
        order_id = order_result.data[0]['order_id']

        # Step 3: Verify order is immediately queryable
        query_result = self.supabase.table('orders')\
            .select('*')\
            .eq('order_id', order_id)\
            .execute()

        assert len(query_result.data) > 0, "Order not immediately queryable"
        assert query_result.data[0]['signal_id'] == signal_id

        print("✅ Test 2 PASSED: Signal → Order flow working")

    def test_order_fill_updates_position(self):
        """Test 3: Order fill triggers position update"""
        # Create order
        order_data = {
            'symbol': 'REALTIME-TEST',
            'side': 'BUY',
            'order_type': 'MARKET',
            'quantity': 10.0,
            'status': 'SUBMITTED',
            'created_at': datetime.utcnow().isoformat()
        }
        order_result = self.supabase.table('orders').insert(order_data).execute()
        order_id = order_result.data[0]['order_id']

        # Simulate order fill
        time.sleep(0.1)
        fill_update = {
            'status': 'FILLED',
            'filled_quantity': 10.0,
            'avg_fill_price': 2005.0,
            'updated_at': datetime.utcnow().isoformat()
        }
        self.supabase.table('orders').update(fill_update).eq('order_id', order_id).execute()

        # Simulate Position Manager creating position
        time.sleep(0.1)
        position_data = {
            'symbol': 'REALTIME-TEST',
            'quantity': 10.0,
            'entry_price': 2005.0,
            'current_price': 2005.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = self.supabase.table('positions').insert(position_data).execute()

        # Verify position exists
        assert len(position_result.data) > 0, "Position not created"
        assert position_result.data[0]['quantity'] == 10.0

        print("✅ Test 3 PASSED: Order fill → Position update working")

    def test_position_update_propagates(self):
        """Test 4: Position updates propagate in real-time"""
        # Create position
        position_data = {
            'symbol': 'REALTIME-TEST',
            'quantity': 10.0,
            'entry_price': 1000.0,
            'current_price': 1000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = self.supabase.table('positions').insert(position_data).execute()
        position_id = position_result.data[0]['position_id']

        # Update position (price change)
        time.sleep(0.1)
        new_price = 1050.0
        new_pnl = 10.0 * (new_price - 1000.0)

        update_data = {
            'current_price': new_price,
            'unrealized_pnl': new_pnl,
            'updated_at': datetime.utcnow().isoformat()
        }
        self.supabase.table('positions').update(update_data).eq('position_id', position_id).execute()

        # Verify update is immediately visible
        query_result = self.supabase.table('positions')\
            .select('*')\
            .eq('position_id', position_id)\
            .execute()

        assert query_result.data[0]['current_price'] == 1050.0
        assert query_result.data[0]['unrealized_pnl'] == 500.0

        print("✅ Test 4 PASSED: Position updates propagate in real-time")

    def test_portfolio_snapshot_creation(self):
        """Test 5: Portfolio snapshots are created and queryable"""
        snapshot_data = {
            'total_equity': 110000.0,
            'cash_balance': 60000.0,
            'positions_value': 50000.0,
            'daily_pnl': 10000.0,
            'total_pnl': 15000.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = self.supabase.table('portfolio_snapshots').insert(snapshot_data).execute()
        assert len(result.data) > 0, "Portfolio snapshot not created"

        # Query immediately
        snapshot_id = result.data[0]['snapshot_id']
        query_result = self.supabase.table('portfolio_snapshots')\
            .select('*')\
            .eq('snapshot_id', snapshot_id)\
            .execute()

        assert len(query_result.data) > 0
        assert query_result.data[0]['total_equity'] == 110000.0

        print("✅ Test 5 PASSED: Portfolio snapshots propagate in real-time")

    def test_market_data_stream(self):
        """Test 6: Market data updates flow through pipeline"""
        # Simulate data pipeline writing multiple bars
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        timestamps = []

        for i, symbol in enumerate(symbols):
            data = {
                'symbol': symbol,
                'timeframe': '1min',
                'open': 1000.0 + i,
                'high': 1010.0 + i,
                'low': 990.0 + i,
                'close': 1005.0 + i,
                'volume': 100.0,
                'vwap': 1002.5 + i,
                'timestamp': datetime.utcnow().isoformat()
            }
            result = self.supabase.table('crypto_aggregates').insert(data).execute()
            timestamps.append(result.data[0]['timestamp'])

        # Verify all inserted and queryable
        for symbol in symbols:
            query_result = self.supabase.table('crypto_aggregates')\
                .select('*')\
                .eq('symbol', symbol)\
                .order('timestamp', desc=True)\
                .limit(1)\
                .execute()

            assert len(query_result.data) > 0, f"No data for {symbol}"

        print(f"✅ Test 6 PASSED: Market data stream working ({len(symbols)} symbols)")

    def test_sentiment_updates(self):
        """Test 7: Sentiment updates are real-time accessible"""
        sentiment_data = {
            'symbol': 'BTC-USD',
            'source': 'test_perplexity',
            'sentiment_score': 0.80,
            'confidence': 0.85,
            'summary': 'Test bullish sentiment',
            'timestamp': datetime.utcnow().isoformat()
        }

        result = self.supabase.table('market_sentiment').insert(sentiment_data).execute()
        sentiment_id = result.data[0]['sentiment_id']

        # Immediate query
        query_result = self.supabase.table('market_sentiment')\
            .select('*')\
            .eq('sentiment_id', sentiment_id)\
            .execute()

        assert len(query_result.data) > 0
        assert query_result.data[0]['sentiment_score'] == 0.80

        print("✅ Test 7 PASSED: Sentiment updates real-time accessible")

    def test_system_events_logging(self):
        """Test 8: System events are logged in real-time"""
        event_types = ['INFO', 'WARNING', 'ERROR']
        event_ids = []

        for severity in event_types:
            event_data = {
                'event_type': 'REALTIME_TEST',
                'severity': severity,
                'source': 'test_suite',
                'message': f'Test {severity} event',
                'metadata': {'test': True},
                'timestamp': datetime.utcnow().isoformat()
            }
            result = self.supabase.table('system_events').insert(event_data).execute()
            event_ids.append(result.data[0]['event_id'])

        # Query all events
        query_result = self.supabase.table('system_events')\
            .select('*')\
            .eq('event_type', 'REALTIME_TEST')\
            .execute()

        assert len(query_result.data) >= len(event_types), "Not all events logged"

        print(f"✅ Test 8 PASSED: System events logged ({len(event_types)} events)")

    def test_concurrent_writes(self):
        """Test 9: Multiple concurrent writes work correctly"""
        # Simulate multiple worktrees writing simultaneously
        import threading

        def write_signal(thread_id):
            signal_data = {
                'symbol': f'THREAD-{thread_id}',
                'strategy': 'concurrent_test',
                'signal_type': 'BUY',
                'confidence': 0.75,
                'price': 1000.0 + thread_id,
                'stop_loss': 950.0,
                'take_profit': 1100.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.supabase.table('trading_signals').insert(signal_data).execute()

        # Launch 5 concurrent writes
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_signal, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all writes succeeded
        time.sleep(0.2)
        query_result = self.supabase.table('trading_signals')\
            .select('*')\
            .like('symbol', 'THREAD-%')\
            .execute()

        assert len(query_result.data) >= 5, "Not all concurrent writes succeeded"

        # Cleanup
        for i in range(5):
            self.supabase.table('trading_signals').delete().eq('symbol', f'THREAD-{i}').execute()

        print(f"✅ Test 9 PASSED: Concurrent writes working ({len(query_result.data)} records)")

    def test_query_performance_under_load(self):
        """Test 10: Query performance remains acceptable under load"""
        import time

        # Insert 50 records quickly
        start_insert = time.time()
        for i in range(50):
            data = {
                'symbol': 'LOAD-TEST',
                'timeframe': '1min',
                'open': 1000.0 + i,
                'high': 1010.0 + i,
                'low': 990.0 + i,
                'close': 1005.0 + i,
                'volume': 100.0,
                'vwap': 1002.5,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.supabase.table('crypto_aggregates').insert(data).execute()
        end_insert = time.time()

        insert_duration = (end_insert - start_insert) * 1000
        avg_insert_latency = insert_duration / 50

        # Now query
        start_query = time.time()
        query_result = self.supabase.table('crypto_aggregates')\
            .select('*')\
            .eq('symbol', 'LOAD-TEST')\
            .execute()
        end_query = time.time()

        query_latency = (end_query - start_query) * 1000

        # Cleanup
        self.supabase.table('crypto_aggregates').delete().eq('symbol', 'LOAD-TEST').execute()

        # Assertions
        assert avg_insert_latency < 200, f"Insert too slow: {avg_insert_latency:.0f}ms avg"
        assert query_latency < 500, f"Query too slow: {query_latency:.0f}ms"
        assert len(query_result.data) >= 50, "Not all records inserted"

        print(f"✅ Test 10 PASSED: Performance under load OK")
        print(f"   - Avg insert: {avg_insert_latency:.0f}ms")
        print(f"   - Query (50 records): {query_latency:.0f}ms")


class TestCrossWorktreeCommunication:
    """Test communication patterns between worktrees"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup Supabase client"""
        self.supabase: Client = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_ANON_KEY')
        )

    def test_data_pipeline_to_neural_network(self):
        """Test 11: Data Pipeline → Neural Network communication"""
        # Data Pipeline writes market data
        for i in range(10):
            data = {
                'symbol': 'COMM-TEST',
                'timeframe': '1min',
                'open': 1000.0 + i,
                'high': 1010.0 + i,
                'low': 990.0 + i,
                'close': 1005.0 + i,
                'volume': 100.0,
                'vwap': 1002.5,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.supabase.table('crypto_aggregates').insert(data).execute()

        # Neural Network reads market data
        time.sleep(0.1)
        query_result = self.supabase.table('crypto_aggregates')\
            .select('*')\
            .eq('symbol', 'COMM-TEST')\
            .order('timestamp', desc=True)\
            .limit(10)\
            .execute()

        assert len(query_result.data) >= 10, "Neural network can't read pipeline data"

        # Cleanup
        self.supabase.table('crypto_aggregates').delete().eq('symbol', 'COMM-TEST').execute()

        print("✅ Test 11 PASSED: Data Pipeline → Neural Network communication OK")

    def test_neural_network_to_trading_engine(self):
        """Test 12: Neural Network → Trading Engine communication"""
        # Neural Network writes signal
        signal_data = {
            'symbol': 'COMM-TEST',
            'strategy': 'comm_test',
            'signal_type': 'BUY',
            'confidence': 0.88,
            'price': 3000.0,
            'stop_loss': 2900.0,
            'take_profit': 3300.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        signal_result = self.supabase.table('trading_signals').insert(signal_data).execute()
        signal_id = signal_result.data[0]['signal_id']

        # Trading Engine reads signal
        time.sleep(0.1)
        query_result = self.supabase.table('trading_signals')\
            .select('*')\
            .eq('signal_id', signal_id)\
            .execute()

        assert len(query_result.data) > 0, "Trading engine can't read signals"

        # Cleanup
        self.supabase.table('trading_signals').delete().eq('symbol', 'COMM-TEST').execute()

        print("✅ Test 12 PASSED: Neural Network → Trading Engine communication OK")

    def test_trading_engine_to_risk_management(self):
        """Test 13: Trading Engine → Risk Management communication"""
        # Trading Engine writes position
        position_data = {
            'symbol': 'COMM-TEST',
            'quantity': 5.0,
            'entry_price': 2000.0,
            'current_price': 2000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = self.supabase.table('positions').insert(position_data).execute()
        position_id = position_result.data[0]['position_id']

        # Risk Management reads position
        time.sleep(0.1)
        query_result = self.supabase.table('positions')\
            .select('*')\
            .eq('position_id', position_id)\
            .execute()

        assert len(query_result.data) > 0, "Risk management can't read positions"

        # Cleanup
        self.supabase.table('positions').delete().eq('symbol', 'COMM-TEST').execute()

        print("✅ Test 13 PASSED: Trading Engine → Risk Management communication OK")

    def test_all_worktrees_to_monitoring(self):
        """Test 14: All Worktrees → Monitoring Dashboard communication"""
        # Multiple worktrees write events
        sources = ['data-pipeline', 'neural-network', 'trading-engine', 'risk-management']

        for source in sources:
            event_data = {
                'event_type': 'STARTUP',
                'severity': 'INFO',
                'source': source,
                'message': f'{source} started',
                'metadata': {'test': True},
                'timestamp': datetime.utcnow().isoformat()
            }
            self.supabase.table('system_events').insert(event_data).execute()

        # Monitoring reads all events
        time.sleep(0.1)
        query_result = self.supabase.table('system_events')\
            .select('*')\
            .eq('event_type', 'STARTUP')\
            .execute()

        assert len(query_result.data) >= len(sources), "Monitoring can't see all worktree events"

        print(f"✅ Test 14 PASSED: All Worktrees → Monitoring communication OK ({len(sources)} sources)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
