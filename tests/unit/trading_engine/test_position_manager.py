from datetime import datetime
from decimal import Decimal
from engine.portfolio.position_manager import PositionManager, Position
from unittest.mock import Mock, patch
import os
import pytest
import sys

"""
Unit Tests for Position Manager

Tests position tracking, P&L calculation, and portfolio management.
Critical for maintaining accurate position state.
"""


# Add trading engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/trading-engine/src'))



class TestPositionManagerInitialization:
    """Test PositionManager initialization"""

    def test_initialization_default(self):
        """Test default initialization"""
        pm = PositionManager()
        assert pm is not None
        assert hasattr(pm, 'positions')
        assert len(pm.positions) == 0

    def test_initialization_with_initial_capital(self):
        """Test initialization with capital"""
        pm = PositionManager(initial_capital=100000)
        assert pm.initial_capital == 100000
        assert pm.available_capital == 100000


class TestPositionOpening:
    """Test opening new positions"""

    @pytest.fixture
    def position_manager(self):
        """Create PositionManager fixture"""
        return PositionManager(initial_capital=100000)

    def test_open_long_position(self, position_manager):
        """Test opening a long position"""
        position = position_manager.open_position(
            symbol='BTC-USD',
            side='LONG',
            quantity=0.001,
            entry_price=50000.00,
            timestamp=datetime.now()
        )

        assert position is not None
        assert position['symbol'] == 'BTC-USD'
        assert position['side'] == 'LONG'
        assert position['quantity'] == 0.001
        assert position['entry_price'] == 50000.00
        assert position['status'] == 'OPEN'

    def test_open_short_position(self, position_manager):
        """Test opening a short position"""
        position = position_manager.open_position(
            symbol='ETH-USD',
            side='SHORT',
            quantity=0.01,
            entry_price=3000.00,
            timestamp=datetime.now()
        )

        assert position['side'] == 'SHORT'
        assert position['quantity'] == 0.01

    def test_open_multiple_positions(self, position_manager):
        """Test opening multiple positions"""
        pos1 = position_manager.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )
        pos2 = position_manager.open_position(
            symbol='ETH-USD', side='LONG', quantity=0.01,
            entry_price=3000.00, timestamp=datetime.now()
        )

        assert len(position_manager.list_open_positions()) == 2

    def test_open_position_with_stop_loss(self, position_manager):
        """Test opening position with stop-loss"""
        position = position_manager.open_position(
            symbol='BTC-USD',
            side='LONG',
            quantity=0.001,
            entry_price=50000.00,
            stop_loss=48000.00,
            timestamp=datetime.now()
        )

        assert position['stop_loss'] == 48000.00

    def test_open_position_with_take_profit(self, position_manager):
        """Test opening position with take-profit"""
        position = position_manager.open_position(
            symbol='BTC-USD',
            side='LONG',
            quantity=0.001,
            entry_price=50000.00,
            take_profit=55000.00,
            timestamp=datetime.now()
        )

        assert position['take_profit'] == 55000.00

    def test_open_position_insufficient_capital(self, position_manager):
        """Test opening position with insufficient capital"""
        with pytest.raises(ValueError):
            position_manager.open_position(
                symbol='BTC-USD',
                side='LONG',
                quantity=10.0,  # Too large
                entry_price=50000.00,
                timestamp=datetime.now()
            )


class TestPositionClosing:
    """Test closing positions"""

    @pytest.fixture
    def position_manager_with_position(self):
        """Create PositionManager with an open position"""
        pm = PositionManager(initial_capital=100000)
        position = pm.open_position(
            symbol='BTC-USD',
            side='LONG',
            quantity=0.001,
            entry_price=50000.00,
            timestamp=datetime.now()
        )
        return pm, position

    def test_close_position_with_profit(self, position_manager_with_position):
        """Test closing position with profit"""
        pm, position = position_manager_with_position
        position_id = position['position_id']

        closed_position = pm.close_position(
            position_id=position_id,
            exit_price=55000.00,
            timestamp=datetime.now()
        )

        assert closed_position['status'] == 'CLOSED'
        assert closed_position['exit_price'] == 55000.00
        assert closed_position['pnl'] > 0  # Profitable trade

    def test_close_position_with_loss(self, position_manager_with_position):
        """Test closing position with loss"""
        pm, position = position_manager_with_position
        position_id = position['position_id']

        closed_position = pm.close_position(
            position_id=position_id,
            exit_price=48000.00,
            timestamp=datetime.now()
        )

        assert closed_position['pnl'] < 0  # Losing trade

    def test_close_nonexistent_position(self, position_manager_with_position):
        """Test closing non-existent position"""
        pm, _ = position_manager_with_position

        with pytest.raises(KeyError):
            pm.close_position(
                position_id='nonexistent',
                exit_price=50000.00,
                timestamp=datetime.now()
            )

    def test_close_all_positions(self, position_manager_with_position):
        """Test closing all open positions"""
        pm, _ = position_manager_with_position

        # Open second position
        pm.open_position(
            symbol='ETH-USD', side='LONG', quantity=0.01,
            entry_price=3000.00, timestamp=datetime.now()
        )

        # Close all
        pm.close_all_positions(current_prices={'BTC-USD': 51000, 'ETH-USD': 3100})

        assert len(pm.list_open_positions()) == 0


class TestPnLCalculation:
    """Test profit and loss calculations"""

    @pytest.fixture
    def position_manager(self):
        return PositionManager(initial_capital=100000)

    def test_calculate_pnl_long_profit(self, position_manager):
        """Test P&L calculation for profitable long"""
        position = position_manager.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )

        pnl = position_manager.calculate_unrealized_pnl(
            position['position_id'],
            current_price=55000.00
        )

        expected_pnl = 0.001 * (55000 - 50000)
        assert abs(pnl - expected_pnl) < 0.01

    def test_calculate_pnl_long_loss(self, position_manager):
        """Test P&L calculation for losing long"""
        position = position_manager.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )

        pnl = position_manager.calculate_unrealized_pnl(
            position['position_id'],
            current_price=48000.00
        )

        expected_pnl = 0.001 * (48000 - 50000)
        assert abs(pnl - expected_pnl) < 0.01

    def test_calculate_pnl_short_profit(self, position_manager):
        """Test P&L calculation for profitable short"""
        position = position_manager.open_position(
            symbol='BTC-USD', side='SHORT', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )

        pnl = position_manager.calculate_unrealized_pnl(
            position['position_id'],
            current_price=48000.00
        )

        expected_pnl = 0.001 * (50000 - 48000)
        assert abs(pnl - expected_pnl) < 0.01

    def test_calculate_pnl_short_loss(self, position_manager):
        """Test P&L calculation for losing short"""
        position = position_manager.open_position(
            symbol='BTC-USD', side='SHORT', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )

        pnl = position_manager.calculate_unrealized_pnl(
            position['position_id'],
            current_price=52000.00
        )

        expected_pnl = 0.001 * (50000 - 52000)
        assert abs(pnl - expected_pnl) < 0.01

    def test_total_unrealized_pnl(self, position_manager):
        """Test total unrealized P&L across all positions"""
        # Open multiple positions
        position_manager.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )
        position_manager.open_position(
            symbol='ETH-USD', side='LONG', quantity=0.01,
            entry_price=3000.00, timestamp=datetime.now()
        )

        total_pnl = position_manager.get_total_unrealized_pnl(
            current_prices={'BTC-USD': 51000, 'ETH-USD': 3100}
        )

        expected = (0.001 * 1000) + (0.01 * 100)  # BTC + ETH
        assert abs(total_pnl - expected) < 0.01


class TestPositionModification:
    """Test position modification (add to position, reduce position)"""

    @pytest.fixture
    def position_manager_with_position(self):
        pm = PositionManager(initial_capital=100000)
        position = pm.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )
        return pm, position

    def test_increase_position_size(self, position_manager_with_position):
        """Test adding to existing position"""
        pm, position = position_manager_with_position
        position_id = position['position_id']

        pm.increase_position(
            position_id=position_id,
            additional_quantity=0.0005,
            entry_price=51000.00,
            timestamp=datetime.now()
        )

        modified = pm.get_position(position_id)
        assert modified['quantity'] == 0.0015  # 0.001 + 0.0005

    def test_reduce_position_size(self, position_manager_with_position):
        """Test reducing existing position"""
        pm, position = position_manager_with_position
        position_id = position['position_id']

        pm.reduce_position(
            position_id=position_id,
            reduce_quantity=0.0005,
            exit_price=51000.00,
            timestamp=datetime.now()
        )

        modified = pm.get_position(position_id)
        assert modified['quantity'] == 0.0005  # 0.001 - 0.0005

    def test_update_stop_loss(self, position_manager_with_position):
        """Test updating stop-loss"""
        pm, position = position_manager_with_position
        position_id = position['position_id']

        pm.update_stop_loss(position_id, new_stop_loss=49000.00)

        modified = pm.get_position(position_id)
        assert modified['stop_loss'] == 49000.00


class TestPortfolioMetrics:
    """Test portfolio-level metrics"""

    @pytest.fixture
    def position_manager_with_multiple_positions(self):
        pm = PositionManager(initial_capital=100000)
        pm.open_position(
            symbol='BTC-USD', side='LONG', quantity=0.001,
            entry_price=50000.00, timestamp=datetime.now()
        )
        pm.open_position(
            symbol='ETH-USD', side='LONG', quantity=0.01,
            entry_price=3000.00, timestamp=datetime.now()
        )
        pm.open_position(
            symbol='SOL-USD', side='LONG', quantity=1.0,
            entry_price=150.00, timestamp=datetime.now()
        )
        return pm

    def test_get_portfolio_value(self, position_manager_with_multiple_positions):
        """Test total portfolio value calculation"""
        pm = position_manager_with_multiple_positions

        portfolio_value = pm.get_portfolio_value(
            current_prices={'BTC-USD': 51000, 'ETH-USD': 3100, 'SOL-USD': 155}
        )

        # Initial capital + unrealized P&L
        assert portfolio_value > 100000

    def test_get_position_allocation(self, position_manager_with_multiple_positions):
        """Test position allocation percentages"""
        pm = position_manager_with_multiple_positions

        allocation = pm.get_position_allocation(
            current_prices={'BTC-USD': 51000, 'ETH-USD': 3100, 'SOL-USD': 155}
        )

        # All allocations should sum to ~100%
        total_allocation = sum(allocation.values())
        assert 0.99 < total_allocation < 1.01

    def test_get_largest_position(self, position_manager_with_multiple_positions):
        """Test identifying largest position"""
        pm = position_manager_with_multiple_positions

        largest = pm.get_largest_position(
            current_prices={'BTC-USD': 51000, 'ETH-USD': 3100, 'SOL-USD': 155}
        )

        assert largest is not None
        assert 'symbol' in largest


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
