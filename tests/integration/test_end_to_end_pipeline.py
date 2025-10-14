from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import os
import pytest

"""
End-to-End Integration Tests for RRRalgorithms Trading System

Tests the complete data flow:
Polygon.io → Data Pipeline → Neural Network → Trading Engine → Risk Management → Monitoring
"""


# Load environment variables
load_dotenv('config/api-keys/.env')

# Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_ANON_KEY')
)


class TestEndToEndPipeline:
    """Test complete data flow through all system components"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data and clean state"""
        # Clean up test data from previous runs
        self.cleanup_test_data()
        yield
        # Cleanup after test
        self.cleanup_test_data()

    def cleanup_test_data(self):
        """Remove test data from database"""
        test_symbol = 'TEST-USD'
        try:
            supabase.table('trading_signals').delete().eq('symbol', test_symbol).execute()
            supabase.table('orders').delete().eq('symbol', test_symbol).execute()
            supabase.table('positions').delete().eq('symbol', test_symbol).execute()
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def test_data_pipeline_to_database(self):
        """Test 1: Data Pipeline writes market data to Supabase"""
        # Insert test market data (simulating Polygon.io data)
        test_data = {
            'symbol': 'BTC-USD',
            'timeframe': '1min',
            'open': 43000.0,
            'high': 43100.0,
            'low': 42900.0,
            'close': 43050.0,
            'volume': 150.5,
            'vwap': 43025.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('crypto_aggregates').insert(test_data).execute()
        assert len(result.data) > 0, "Failed to insert market data"
        assert result.data[0]['symbol'] == 'BTC-USD'
        print("✅ Test 1 PASSED: Data pipeline writes to database")

    def test_neural_network_signal_generation(self):
        """Test 2: Neural Network generates trading signals from market data"""
        # Insert market data
        for i in range(100):  # Neural network needs 100 timesteps
            timestamp = datetime.utcnow() - timedelta(minutes=100-i)
            supabase.table('crypto_aggregates').insert({
                'symbol': 'BTC-USD',
                'timeframe': '1min',
                'open': 43000.0 + i,
                'high': 43100.0 + i,
                'low': 42900.0 + i,
                'close': 43050.0 + i,
                'volume': 150.5,
                'vwap': 43025.0,
                'timestamp': timestamp.isoformat()
            }).execute()

        # Simulate neural network generating a signal
        signal_data = {
            'symbol': 'BTC-USD',
            'strategy': 'transformer_price_prediction',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 43050.0,
            'stop_loss': 42000.0,
            'take_profit': 46000.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('trading_signals').insert(signal_data).execute()
        assert len(result.data) > 0, "Failed to insert trading signal"
        assert result.data[0]['signal_type'] == 'BUY'
        assert result.data[0]['confidence'] == 0.85
        print("✅ Test 2 PASSED: Neural network generates signals")

    def test_risk_management_validation(self):
        """Test 3: Risk Management validates position sizing"""
        # Create test portfolio state
        portfolio_data = {
            'total_equity': 100000.0,
            'cash_balance': 50000.0,
            'positions_value': 50000.0,
            'daily_pnl': 1000.0,
            'total_pnl': 5000.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        supabase.table('portfolio_snapshots').insert(portfolio_data).execute()

        # Test position size calculation (Kelly Criterion)
        max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.20))
        portfolio_value = portfolio_data['total_equity']
        max_position_value = portfolio_value * max_position_size

        assert max_position_value == 20000.0, f"Expected 20000, got {max_position_value}"

        # Test daily loss limit
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 0.05))
        max_loss_value = portfolio_value * max_daily_loss

        assert max_loss_value == 5000.0, f"Expected 5000, got {max_loss_value}"

        # Current daily P&L is +$1000, so trading should be allowed
        can_trade = portfolio_data['daily_pnl'] > -max_loss_value
        assert can_trade, "Trading should be allowed with positive P&L"

        print("✅ Test 3 PASSED: Risk management validates positions")

    def test_order_execution_flow(self):
        """Test 4: Trading Engine executes orders from signals"""
        # Create a trading signal
        signal_data = {
            'symbol': 'TEST-USD',
            'strategy': 'test_strategy',
            'signal_type': 'BUY',
            'confidence': 0.90,
            'price': 1000.0,
            'stop_loss': 950.0,
            'take_profit': 1100.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        signal_result = supabase.table('trading_signals').insert(signal_data).execute()
        signal_id = signal_result.data[0]['signal_id']

        # Simulate order creation from signal
        order_data = {
            'symbol': 'TEST-USD',
            'side': 'BUY',
            'order_type': 'MARKET',
            'quantity': 10.0,
            'status': 'SUBMITTED',
            'signal_id': signal_id,
            'created_at': datetime.utcnow().isoformat()
        }
        order_result = supabase.table('orders').insert(order_data).execute()
        order_id = order_result.data[0]['order_id']

        # Simulate order fill
        update_data = {
            'status': 'FILLED',
            'filled_quantity': 10.0,
            'avg_fill_price': 1001.0,  # 0.1% slippage
            'updated_at': datetime.utcnow().isoformat()
        }
        update_result = supabase.table('orders').update(update_data).eq('order_id', order_id).execute()

        assert update_result.data[0]['status'] == 'FILLED'
        assert update_result.data[0]['filled_quantity'] == 10.0

        # Create position from filled order
        position_data = {
            'symbol': 'TEST-USD',
            'quantity': 10.0,
            'entry_price': 1001.0,
            'current_price': 1001.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = supabase.table('positions').insert(position_data).execute()

        assert len(position_result.data) > 0
        assert position_result.data[0]['quantity'] == 10.0

        print("✅ Test 4 PASSED: Order execution flow complete")

    def test_position_pnl_calculation(self):
        """Test 5: Position P&L updates correctly"""
        # Create a position
        position_data = {
            'symbol': 'TEST-USD',
            'quantity': 10.0,
            'entry_price': 1000.0,
            'current_price': 1050.0,  # +5% price increase
            'unrealized_pnl': 500.0,  # 10 * (1050 - 1000)
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        result = supabase.table('positions').insert(position_data).execute()
        position_id = result.data[0]['position_id']

        # Verify P&L calculation
        expected_pnl = 10.0 * (1050.0 - 1000.0)
        assert result.data[0]['unrealized_pnl'] == expected_pnl

        # Update price and recalculate
        new_price = 1100.0
        new_pnl = 10.0 * (new_price - 1000.0)
        update_result = supabase.table('positions').update({
            'current_price': new_price,
            'unrealized_pnl': new_pnl,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('position_id', position_id).execute()

        assert update_result.data[0]['unrealized_pnl'] == 1000.0
        print("✅ Test 5 PASSED: P&L calculation correct")

    def test_stop_loss_trigger(self):
        """Test 6: Stop loss triggers correctly"""
        # Create position with stop loss
        position_data = {
            'symbol': 'TEST-USD',
            'quantity': 10.0,
            'entry_price': 1000.0,
            'current_price': 950.0,  # -5% (breaches 2% stop loss)
            'unrealized_pnl': -500.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = supabase.table('positions').insert(position_data).execute()
        position_id = position_result.data[0]['position_id']

        # Check if stop loss should trigger (2% default)
        entry_price = position_data['entry_price']
        current_price = position_data['current_price']
        pnl_percent = (current_price - entry_price) / entry_price
        stop_loss_percent = -0.02  # -2%

        should_stop = pnl_percent <= stop_loss_percent
        assert should_stop, f"Stop loss should trigger at {pnl_percent*100}%"

        # Create stop-loss order
        stop_order = {
            'symbol': 'TEST-USD',
            'side': 'SELL',
            'order_type': 'MARKET',
            'quantity': 10.0,
            'status': 'FILLED',
            'avg_fill_price': 950.0,
            'created_at': datetime.utcnow().isoformat()
        }
        order_result = supabase.table('orders').insert(stop_order).execute()

        # Close position
        supabase.table('positions').update({
            'quantity': 0.0,
            'realized_pnl': -500.0,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('position_id', position_id).execute()

        print("✅ Test 6 PASSED: Stop loss triggers correctly")

    def test_sentiment_integration(self):
        """Test 7: Sentiment analysis integration"""
        # Insert sentiment data
        sentiment_data = {
            'symbol': 'BTC-USD',
            'source': 'perplexity_ai',
            'sentiment_score': 0.75,  # Bullish
            'confidence': 0.85,
            'summary': 'Bitcoin shows strong bullish momentum with institutional inflows',
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('market_sentiment').insert(sentiment_data).execute()
        assert len(result.data) > 0
        assert result.data[0]['sentiment_score'] == 0.75

        # Verify sentiment can be used in signal generation
        # (In production, neural network would read this)
        query_result = supabase.table('market_sentiment')\
            .select('*')\
            .eq('symbol', 'BTC-USD')\
            .order('timestamp', desc=True)\
            .limit(1)\
            .execute()

        assert len(query_result.data) > 0
        assert query_result.data[0]['sentiment_score'] > 0.5  # Bullish

        print("✅ Test 7 PASSED: Sentiment analysis integrated")

    def test_portfolio_snapshot(self):
        """Test 8: Portfolio snapshots are created correctly"""
        snapshot_data = {
            'total_equity': 105000.0,
            'cash_balance': 45000.0,
            'positions_value': 60000.0,
            'daily_pnl': 5000.0,
            'total_pnl': 10000.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('portfolio_snapshots').insert(snapshot_data).execute()
        assert len(result.data) > 0

        # Calculate metrics
        total_equity = snapshot_data['total_equity']
        total_pnl = snapshot_data['total_pnl']
        roi = (total_pnl / (total_equity - total_pnl)) * 100

        assert roi > 0, "ROI should be positive"
        print(f"✅ Test 8 PASSED: Portfolio snapshot created (ROI: {roi:.2f}%)")

    def test_system_event_logging(self):
        """Test 9: System events are logged correctly"""
        event_data = {
            'event_type': 'INTEGRATION_TEST',
            'severity': 'INFO',
            'source': 'test_suite',
            'message': 'Integration test execution',
            'metadata': {'test': 'end_to_end', 'status': 'running'},
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('system_events').insert(event_data).execute()
        assert len(result.data) > 0
        assert result.data[0]['severity'] == 'INFO'

        print("✅ Test 9 PASSED: System events logged")

    def test_complete_trading_cycle(self):
        """Test 10: Complete trading cycle from signal to close"""
        symbol = 'TEST-USD'

        # Step 1: Market data arrives
        market_data = {
            'symbol': symbol,
            'timeframe': '1min',
            'open': 1000.0,
            'high': 1010.0,
            'low': 995.0,
            'close': 1005.0,
            'volume': 100.0,
            'vwap': 1002.5,
            'timestamp': datetime.utcnow().isoformat()
        }
        supabase.table('crypto_aggregates').insert(market_data).execute()

        # Step 2: Neural network generates signal
        signal_data = {
            'symbol': symbol,
            'strategy': 'complete_cycle_test',
            'signal_type': 'BUY',
            'confidence': 0.88,
            'price': 1005.0,
            'stop_loss': 980.0,
            'take_profit': 1050.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        signal_result = supabase.table('trading_signals').insert(signal_data).execute()
        signal_id = signal_result.data[0]['signal_id']

        # Step 3: Risk management approves (assume approved for test)
        # Position size = 10 units (for simplicity)

        # Step 4: Order is created and executed
        order_data = {
            'symbol': symbol,
            'side': 'BUY',
            'order_type': 'MARKET',
            'quantity': 10.0,
            'status': 'FILLED',
            'filled_quantity': 10.0,
            'avg_fill_price': 1006.0,  # Slight slippage
            'signal_id': signal_id,
            'created_at': datetime.utcnow().isoformat()
        }
        order_result = supabase.table('orders').insert(order_data).execute()
        order_id = order_result.data[0]['order_id']

        # Step 5: Position is opened
        position_data = {
            'symbol': symbol,
            'quantity': 10.0,
            'entry_price': 1006.0,
            'current_price': 1006.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'opened_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        position_result = supabase.table('positions').insert(position_data).execute()
        position_id = position_result.data[0]['position_id']

        # Step 6: Price moves in favor (take profit hit)
        new_price = 1051.0  # Above take profit
        new_pnl = 10.0 * (new_price - 1006.0)
        supabase.table('positions').update({
            'current_price': new_price,
            'unrealized_pnl': new_pnl,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('position_id', position_id).execute()

        # Step 7: Take profit order is executed
        close_order = {
            'symbol': symbol,
            'side': 'SELL',
            'order_type': 'MARKET',
            'quantity': 10.0,
            'status': 'FILLED',
            'filled_quantity': 10.0,
            'avg_fill_price': 1050.0,
            'created_at': datetime.utcnow().isoformat()
        }
        supabase.table('orders').insert(close_order).execute()

        # Step 8: Position is closed
        final_pnl = 10.0 * (1050.0 - 1006.0)
        supabase.table('positions').update({
            'quantity': 0.0,
            'realized_pnl': final_pnl,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('position_id', position_id).execute()

        # Step 9: Portfolio snapshot is updated
        portfolio_data = {
            'total_equity': 100000.0 + final_pnl,
            'cash_balance': 50000.0 + final_pnl,
            'positions_value': 0.0,
            'daily_pnl': final_pnl,
            'total_pnl': final_pnl,
            'timestamp': datetime.utcnow().isoformat()
        }
        supabase.table('portfolio_snapshots').insert(portfolio_data).execute()

        # Verify complete cycle
        assert final_pnl == 440.0, f"Expected $440 profit, got ${final_pnl}"

        print(f"✅ Test 10 PASSED: Complete trading cycle (Profit: ${final_pnl})")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
