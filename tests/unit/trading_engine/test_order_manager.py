from datetime import datetime
from src.trading.engine.engine.oms.order_manager import OrderManager
from unittest.mock import Mock, patch, MagicMock
import os
import pytest
import sys

"""
Unit Tests for Order Manager

Tests order creation, modification, cancellation, and lifecycle management.
Critical for production trading system.
"""


# Add trading engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/trading-engine/src'))



class TestOrderManagerInitialization:
    """Test OrderManager initialization"""

    def test_initialization_default(self):
        """Test default initialization"""
        om = OrderManager()
        assert om is not None
        assert hasattr(om, 'orders')
        assert hasattr(om, 'order_history')

    def test_initialization_with_database(self):
        """Test initialization with database connection"""
        with patch('engine.oms.order_manager.create_client') as mock_client:
            mock_client.return_value = Mock()
            om = OrderManager(database_url="postgresql://test")
            assert om.database_url == "postgresql://test"


class TestOrderCreation:
    """Test order creation with various types"""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager fixture"""
        return OrderManager()

    def test_create_market_order_buy(self, order_manager):
        """Test market buy order creation"""
        order = order_manager.create_market_order(
            symbol='BTC-USD',
            side='BUY',
            quantity=0.001,
            exchange='paper'
        )

        assert order is not None
        assert order['symbol'] == 'BTC-USD'
        assert order['side'] == 'BUY'
        assert order['quantity'] == 0.001
        assert order['order_type'] == 'MARKET'
        assert order['status'] in ['PENDING', 'FILLED']
        assert 'order_id' in order
        assert 'timestamp' in order

    def test_create_market_order_sell(self, order_manager):
        """Test market sell order creation"""
        order = order_manager.create_market_order(
            symbol='ETH-USD',
            side='SELL',
            quantity=0.01,
            exchange='paper'
        )

        assert order['symbol'] == 'ETH-USD'
        assert order['side'] == 'SELL'
        assert order['quantity'] == 0.01

    def test_create_limit_order_buy(self, order_manager):
        """Test limit buy order creation"""
        order = order_manager.create_limit_order(
            symbol='BTC-USD',
            side='BUY',
            quantity=0.001,
            limit_price=50000.00,
            exchange='paper'
        )

        assert order['order_type'] == 'LIMIT'
        assert order['limit_price'] == 50000.00
        assert order['status'] == 'OPEN'

    def test_create_limit_order_sell(self, order_manager):
        """Test limit sell order creation"""
        order = order_manager.create_limit_order(
            symbol='SOL-USD',
            side='SELL',
            quantity=1.0,
            limit_price=150.00,
            exchange='paper'
        )

        assert order['symbol'] == 'SOL-USD'
        assert order['side'] == 'SELL'
        assert order['limit_price'] == 150.00

    def test_create_stop_loss_order(self, order_manager):
        """Test stop-loss order creation"""
        order = order_manager.create_stop_loss_order(
            symbol='BTC-USD',
            side='SELL',
            quantity=0.001,
            stop_price=48000.00,
            exchange='paper'
        )

        assert order['order_type'] == 'STOP_LOSS'
        assert order['stop_price'] == 48000.00
        assert order['status'] == 'OPEN'

    def test_create_order_invalid_side(self, order_manager):
        """Test order creation with invalid side"""
        with pytest.raises(ValueError):
            order_manager.create_market_order(
                symbol='BTC-USD',
                side='INVALID',
                quantity=0.001,
                exchange='paper'
            )

    def test_create_order_negative_quantity(self, order_manager):
        """Test order creation with negative quantity"""
        with pytest.raises(ValueError):
            order_manager.create_market_order(
                symbol='BTC-USD',
                side='BUY',
                quantity=-0.001,
                exchange='paper'
            )

    def test_create_order_zero_quantity(self, order_manager):
        """Test order creation with zero quantity"""
        with pytest.raises(ValueError):
            order_manager.create_market_order(
                symbol='BTC-USD',
                side='BUY',
                quantity=0.0,
                exchange='paper'
            )


class TestOrderModification:
    """Test order modification operations"""

    @pytest.fixture
    def order_manager_with_order(self):
        """Create OrderManager with an existing order"""
        om = OrderManager()
        order = om.create_limit_order(
            symbol='BTC-USD',
            side='BUY',
            quantity=0.001,
            limit_price=50000.00,
            exchange='paper'
        )
        return om, order

    def test_modify_order_price(self, order_manager_with_order):
        """Test modifying order price"""
        om, order = order_manager_with_order
        order_id = order['order_id']

        result = om.modify_order(
            order_id=order_id,
            new_price=51000.00
        )

        assert result is True
        modified_order = om.get_order(order_id)
        assert modified_order['limit_price'] == 51000.00

    def test_modify_order_quantity(self, order_manager_with_order):
        """Test modifying order quantity"""
        om, order = order_manager_with_order
        order_id = order['order_id']

        result = om.modify_order(
            order_id=order_id,
            new_quantity=0.002
        )

        assert result is True
        modified_order = om.get_order(order_id)
        assert modified_order['quantity'] == 0.002

    def test_modify_filled_order(self, order_manager_with_order):
        """Test modifying a filled order (should fail)"""
        om, order = order_manager_with_order
        order_id = order['order_id']

        # Manually fill the order
        om.update_order_status(order_id, 'FILLED')

        with pytest.raises(ValueError):
            om.modify_order(order_id=order_id, new_price=52000.00)

    def test_modify_nonexistent_order(self, order_manager_with_order):
        """Test modifying non-existent order"""
        om, _ = order_manager_with_order

        with pytest.raises(KeyError):
            om.modify_order(order_id='nonexistent', new_price=50000.00)


class TestOrderCancellation:
    """Test order cancellation"""

    @pytest.fixture
    def order_manager_with_orders(self):
        """Create OrderManager with multiple orders"""
        om = OrderManager()
        order1 = om.create_limit_order(
            symbol='BTC-USD', side='BUY', quantity=0.001,
            limit_price=50000.00, exchange='paper'
        )
        order2 = om.create_stop_loss_order(
            symbol='ETH-USD', side='SELL', quantity=0.01,
            stop_price=3000.00, exchange='paper'
        )
        return om, [order1, order2]

    def test_cancel_single_order(self, order_manager_with_orders):
        """Test cancelling a single order"""
        om, orders = order_manager_with_orders
        order_id = orders[0]['order_id']

        result = om.cancel_order(order_id)

        assert result is True
        cancelled_order = om.get_order(order_id)
        assert cancelled_order['status'] == 'CANCELLED'

    def test_cancel_all_orders(self, order_manager_with_orders):
        """Test cancelling all orders"""
        om, orders = order_manager_with_orders

        result = om.cancel_all_orders()

        assert result is True
        for order in orders:
            cancelled_order = om.get_order(order['order_id'])
            assert cancelled_order['status'] == 'CANCELLED'

    def test_cancel_filled_order(self, order_manager_with_orders):
        """Test cancelling a filled order (should fail)"""
        om, orders = order_manager_with_orders
        order_id = orders[0]['order_id']

        # Fill the order
        om.update_order_status(order_id, 'FILLED')

        with pytest.raises(ValueError):
            om.cancel_order(order_id)

    def test_cancel_nonexistent_order(self, order_manager_with_orders):
        """Test cancelling non-existent order"""
        om, _ = order_manager_with_orders

        with pytest.raises(KeyError):
            om.cancel_order('nonexistent')


class TestOrderStatusTracking:
    """Test order status tracking and querying"""

    @pytest.fixture
    def order_manager_with_varied_orders(self):
        """Create OrderManager with orders in various states"""
        om = OrderManager()

        # Create and fill some orders
        filled_order = om.create_market_order(
            symbol='BTC-USD', side='BUY', quantity=0.001, exchange='paper'
        )
        om.update_order_status(filled_order['order_id'], 'FILLED')

        # Create open orders
        open_order1 = om.create_limit_order(
            symbol='BTC-USD', side='BUY', quantity=0.001,
            limit_price=50000.00, exchange='paper'
        )
        open_order2 = om.create_limit_order(
            symbol='ETH-USD', side='SELL', quantity=0.01,
            limit_price=3500.00, exchange='paper'
        )

        # Create and cancel an order
        cancelled_order = om.create_stop_loss_order(
            symbol='SOL-USD', side='SELL', quantity=1.0,
            stop_price=140.00, exchange='paper'
        )
        om.cancel_order(cancelled_order['order_id'])

        return om

    def test_list_open_orders(self, order_manager_with_varied_orders):
        """Test listing all open orders"""
        om = order_manager_with_varied_orders

        open_orders = om.list_open_orders()

        assert len(open_orders) == 2
        for order in open_orders:
            assert order['status'] == 'OPEN'

    def test_list_filled_orders(self, order_manager_with_varied_orders):
        """Test listing filled orders"""
        om = order_manager_with_varied_orders

        filled_orders = om.list_filled_orders()

        assert len(filled_orders) >= 1
        for order in filled_orders:
            assert order['status'] == 'FILLED'

    def test_list_orders_by_symbol(self, order_manager_with_varied_orders):
        """Test listing orders for specific symbol"""
        om = order_manager_with_varied_orders

        btc_orders = om.list_orders_by_symbol('BTC-USD')

        assert len(btc_orders) >= 2  # 1 filled, 1 open
        for order in btc_orders:
            assert order['symbol'] == 'BTC-USD'

    def test_get_order_by_id(self, order_manager_with_varied_orders):
        """Test getting specific order by ID"""
        om = order_manager_with_varied_orders

        open_orders = om.list_open_orders()
        order_id = open_orders[0]['order_id']

        order = om.get_order(order_id)

        assert order is not None
        assert order['order_id'] == order_id

    def test_get_nonexistent_order(self, order_manager_with_varied_orders):
        """Test getting non-existent order"""
        om = order_manager_with_varied_orders

        with pytest.raises(KeyError):
            om.get_order('nonexistent')


class TestOrderExecution:
    """Test order execution logic"""

    @pytest.fixture
    def order_manager_with_mock_exchange(self):
        """Create OrderManager with mocked exchange"""
        om = OrderManager()
        om.exchange = Mock()
        return om

    def test_execute_market_order(self, order_manager_with_mock_exchange):
        """Test market order execution"""
        om = order_manager_with_mock_exchange
        om.exchange.execute_market_order.return_value = {
            'status': 'FILLED',
            'fill_price': 50000.00,
            'filled_quantity': 0.001
        }

        order = om.create_market_order(
            symbol='BTC-USD', side='BUY', quantity=0.001, exchange='mock'
        )

        om.execute_order(order['order_id'])

        executed_order = om.get_order(order['order_id'])
        assert executed_order['status'] == 'FILLED'
        assert 'fill_price' in executed_order

    def test_execute_order_with_exchange_error(self, order_manager_with_mock_exchange):
        """Test order execution with exchange error"""
        om = order_manager_with_mock_exchange
        om.exchange.execute_market_order.side_effect = Exception("Exchange error")

        order = om.create_market_order(
            symbol='BTC-USD', side='BUY', quantity=0.001, exchange='mock'
        )

        with pytest.raises(Exception):
            om.execute_order(order['order_id'])

        failed_order = om.get_order(order['order_id'])
        assert failed_order['status'] in ['REJECTED', 'FAILED']


class TestOrderDatabaseIntegration:
    """Test database persistence for orders"""

    @pytest.fixture
    def order_manager_with_mock_db(self):
        """Create OrderManager with mocked database"""
        with patch('engine.oms.order_manager.create_client') as mock_client:
            mock_db = Mock()
            mock_client.return_value = mock_db
            om = OrderManager(database_url="postgresql://test")
            om.db = mock_db
            return om

    def test_save_order_to_database(self, order_manager_with_mock_db):
        """Test saving order to database"""
        om = order_manager_with_mock_db
        om.db.table.return_value.insert.return_value.execute.return_value = Mock()

        order = om.create_market_order(
            symbol='BTC-USD', side='BUY', quantity=0.001, exchange='paper'
        )

        om.save_order_to_database(order)

        om.db.table.assert_called_with('orders')
        om.db.table.return_value.insert.assert_called_once()

    def test_load_order_from_database(self, order_manager_with_mock_db):
        """Test loading order from database"""
        om = order_manager_with_mock_db
        mock_order = {
            'order_id': 'test-123',
            'symbol': 'BTC-USD',
            'side': 'BUY',
            'quantity': 0.001,
            'status': 'FILLED'
        }
        om.db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [mock_order]

        loaded_order = om.load_order_from_database('test-123')

        assert loaded_order == mock_order
        om.db.table.assert_called_with('orders')


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
