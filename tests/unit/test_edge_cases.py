from decimal import Decimal
from src.core.constants import ValidationConstants, TradingConstants
from src.core.exceptions import InputValidationError, DataValidationError
from src.core.validation import TradeRequest, OHLCVData, validate_market_data
from src.monitoring.local_monitor import LocalMonitor
from src.neural_network.mock_predictor import MockPredictor, EnsemblePredictor
import pytest
import time

"""
Edge Case Tests
================

Comprehensive edge case testing for critical system components.
Tests boundary conditions, error handling, and unusual scenarios.

Author: RRR Ventures
Date: 2025-10-12
"""




class TestValidationEdgeCases:
    """Test validation edge cases and boundary conditions"""
    
    def test_minimum_price_boundary(self):
        """Test minimum price validation"""
        # Just above minimum
        data = OHLCVData(
            open=ValidationConstants.MIN_PRICE + 0.000001,
            high=ValidationConstants.MIN_PRICE + 0.000002,
            low=ValidationConstants.MIN_PRICE,
            close=ValidationConstants.MIN_PRICE + 0.000001,
            volume=1000
        )
        assert data.low == ValidationConstants.MIN_PRICE
        
        # Below minimum (should fail)
        with pytest.raises(Exception):
            OHLCVData(
                open=0,
                high=0.0000001,
                low=0,
                close=0.000001,
                volume=1000
            )
    
    def test_maximum_price_boundary(self):
        """Test maximum price validation"""
        # Just below maximum
        data = OHLCVData(
            open=ValidationConstants.MAX_PRICE - 1,
            high=ValidationConstants.MAX_PRICE - 0.5,
            low=ValidationConstants.MAX_PRICE - 2,
            close=ValidationConstants.MAX_PRICE - 1,
            volume=1000
        )
        assert data.high < ValidationConstants.MAX_PRICE
        
        # Above maximum (should fail)
        with pytest.raises(Exception):
            OHLCVData(
                open=ValidationConstants.MAX_PRICE + 1,
                high=ValidationConstants.MAX_PRICE + 1,
                low=ValidationConstants.MAX_PRICE,
                close=ValidationConstants.MAX_PRICE,
                volume=1000
            )
    
    def test_ohlcv_relationship_violations(self):
        """Test OHLCV data relationship validation"""
        # High < Low (should fail)
        with pytest.raises(Exception):
            OHLCVData(
                open=50000,
                high=49000,  # High < Low!
                low=51000,
                close=50000,
                volume=1000
            )
        
        # All values equal (should pass)
        data = OHLCVData(
            open=50000,
            high=50000,
            low=50000,
            close=50000,
            volume=0
        )
        assert data.open == data.high == data.low == data.close
    
    def test_negative_volume(self):
        """Test negative volume is rejected"""
        with pytest.raises(Exception):
            OHLCVData(
                open=50000,
                high=51000,
                low=49000,
                close=50500,
                volume=-1000  # Negative!
            )
    
    def test_zero_volume(self):
        """Test zero volume is accepted"""
        data = OHLCVData(
            open=50000,
            high=51000,
            low=49000,
            close=50500,
            volume=0  # Zero is valid
        )
        assert data.volume == 0
    
    def test_very_small_quantity(self):
        """Test micro quantities"""
        trade = TradeRequest(
            symbol='BTC-USD',
            side='buy',
            order_type='market',
            quantity=ValidationConstants.MIN_QUANTITY  # Smallest allowed
        )
        assert trade.quantity == ValidationConstants.MIN_QUANTITY
    
    def test_very_large_quantity(self):
        """Test maximum quantity boundary"""
        trade = TradeRequest(
            symbol='BTC-USD',
            side='buy',
            order_type='market',
            quantity=ValidationConstants.MAX_QUANTITY - 1
        )
        assert trade.quantity < ValidationConstants.MAX_QUANTITY
    
    def test_symbol_case_normalization(self):
        """Test symbols are normalized to uppercase"""
        trade = TradeRequest(
            symbol='btc-usd',  # Lowercase
            side='buy',
            order_type='market',
            quantity=1.0
        )
        assert trade.symbol == 'BTC-USD'  # Uppercase
    
    def test_invalid_symbol_characters(self):
        """Test invalid symbol characters are rejected"""
        invalid_symbols = [
            'BTC_USD',  # Underscore
            'BTC/USD',  # Slash
            'BTC USD',  # Space
            'BTC@USD',  # Special char
            'btc usd',  # Lowercase with space
        ]
        
        for symbol in invalid_symbols:
            with pytest.raises(Exception):
                TradeRequest(
                    symbol=symbol,
                    side='buy',
                    order_type='market',
                    quantity=1.0
                )
    
    def test_empty_symbol(self):
        """Test empty symbol is rejected"""
        with pytest.raises(Exception):
            TradeRequest(
                symbol='',
                side='buy',
                order_type='market',
                quantity=1.0
            )
    
    def test_very_long_symbol(self):
        """Test symbol length limit"""
        with pytest.raises(Exception):
            TradeRequest(
                symbol='A' * (ValidationConstants.MAX_SYMBOL_LENGTH + 1),
                side='buy',
                order_type='market',
                quantity=1.0
            )


class TestPredictorEdgeCases:
    """Test ML predictor edge cases"""
    
    def test_prediction_with_no_history(self):
        """Test prediction works with no price history"""
        predictor = MockPredictor()
        prediction = predictor.predict('NEW-SYMBOL', 50000.0)
        
        assert prediction is not None
        assert 'predicted_price' in prediction
        assert prediction['confidence'] >= 0
    
    def test_prediction_with_extreme_prices(self):
        """Test prediction handles extreme prices"""
        predictor = MockPredictor()
        
        # Very high price (Bitcoin at $1M)
        pred_high = predictor.predict('BTC-USD', 1000000.0)
        assert pred_high['predicted_price'] > 0
        
        # Very low price (micro-cap at $0.0001)
        pred_low = predictor.predict('SHIB-USD', 0.0001)
        assert pred_low['predicted_price'] > 0
    
    def test_prediction_with_zero_price(self):
        """Test prediction handles zero price gracefully"""
        predictor = MockPredictor()
        # Should not crash, though practically invalid
        prediction = predictor.predict('TEST-USD', 0.01)
        assert prediction is not None
    
    def test_ensemble_with_single_predictor(self):
        """Test ensemble with only one predictor"""
        ensemble = EnsemblePredictor(strategies=['trend_following'])
        prediction = ensemble.predict('BTC-USD', 50000.0)
        
        assert prediction is not None
        assert prediction['model_type'] == 'ensemble'
    
    def test_ensemble_with_empty_strategies(self):
        """Test ensemble with no strategies falls back to default"""
        ensemble = EnsemblePredictor(strategies=[])
        # Should use default strategies
        assert len(ensemble.predictors) > 0
    
    def test_confidence_bounds(self):
        """Test confidence always stays in [0, 1]"""
        predictor = MockPredictor()
        
        for _ in range(100):
            prediction = predictor.predict('BTC-USD', 50000.0)
            assert 0 <= prediction['confidence'] <= 1


class TestMonitorEdgeCases:
    """Test monitoring system edge cases"""
    
    def test_win_rate_with_no_trades(self):
        """Test win rate calculation with zero trades"""
        monitor = LocalMonitor(use_rich=False)
        win_rate = monitor.get_win_rate()
        
        assert win_rate == 0.0  # Not undefined or NaN
    
    def test_win_rate_with_only_winning_trades(self):
        """Test win rate with 100% wins"""
        monitor = LocalMonitor(use_rich=False)
        
        for _ in range(10):
            monitor.record_trade('buy', profit=100.0)
        
        assert monitor.get_win_rate() == 1.0
    
    def test_win_rate_with_only_losing_trades(self):
        """Test win rate with 0% wins"""
        monitor = LocalMonitor(use_rich=False)
        
        for _ in range(10):
            monitor.record_trade('sell', profit=-50.0)
        
        assert monitor.get_win_rate() == 0.0
    
    def test_win_rate_with_neutral_trades(self):
        """Test trades with zero profit don't affect win rate"""
        monitor = LocalMonitor(use_rich=False)
        
        monitor.record_trade('buy', profit=100.0)  # Win
        monitor.record_trade('sell', profit=0.0)   # Neutral (not counted)
        monitor.record_trade('buy', profit=-50.0)  # Loss
        
        # Win rate should be 50% (1 win, 1 loss, neutral ignored)
        assert monitor.get_win_rate() == 0.5
    
    def test_negative_portfolio_value(self):
        """Test monitor handles negative portfolio value"""
        monitor = LocalMonitor(use_rich=False)
        
        # Lost all capital and more (margin call scenario)
        monitor.update_portfolio(
            value=-1000.0,  # Negative!
            cash=-1000.0,
            pnl=-11000.0,
            daily_pnl=-11000.0
        )
        
        assert monitor.portfolio_value == -1000.0
        assert monitor.pnl == -11000.0
    
    def test_extreme_pnl_values(self):
        """Test very large profit/loss values"""
        monitor = LocalMonitor(use_rich=False)
        
        # Massive profit (10x return)
        monitor.update_portfolio(
            value=100000.0,
            cash=50000.0,
            pnl=90000.0,
            daily_pnl=90000.0
        )
        
        assert monitor.portfolio_value == 100000.0


class TestDatabaseEdgeCases:
    """Test database operation edge cases"""
    
    def test_concurrent_inserts(self, tmp_path):
        """Test database handles concurrent writes"""
        from src.core.database.local_db import LocalDatabase
        import threading
        
        db = LocalDatabase(str(tmp_path / "test.db"))
        results = []
        
        def insert_trade(trade_id):
            try:
                db.insert_trade({
                    'symbol': f'TEST-{trade_id}',
                    'side': 'buy',
                    'order_type': 'market',
                    'quantity': 1.0,
                    'price': 50000.0,
                    'timestamp': time.time()
                })
                results.append(True)
            except Exception as e:
                results.append(False)
        
        # Create 20 concurrent threads
        threads = [
            threading.Thread(target=insert_trade, args=(i,))
            for i in range(20)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert all(results)
        assert len(db.get_trades()) == 20
        
        db.close()
    
    def test_transaction_rollback_on_error(self, tmp_path):
        """Test transaction rolls back on error"""
        from src.core.database.local_db import LocalDatabase
        
        db = LocalDatabase(str(tmp_path / "test.db"))
        
        initial_count = len(db.get_trades())
        
        try:
            with db.transaction():
                db.insert_trade({
                    'symbol': 'BTC-USD',
                    'side': 'buy',
                    'order_type': 'market',
                    'quantity': 1.0,
                    'price': 50000.0,
                    'timestamp': time.time()
                })
                # Simulate error
                raise Exception("Simulated error")
        except:
            pass
        
        # Count should not have changed (rolled back)
        final_count = len(db.get_trades())
        assert final_count == initial_count
        
        db.close()
    
    def test_empty_result_handling(self, tmp_path):
        """Test queries on empty tables"""
        from src.core.database.local_db import LocalDatabase
        
        db = LocalDatabase(str(tmp_path / "test.db"))
        
        # Empty table queries
        assert db.get_trades() == []
        assert db.get_positions() == []
        assert db.get_latest_portfolio_metrics() is None
        
        db.close()
    
    def test_update_nonexistent_trade(self, tmp_path):
        """Test updating non-existent trade"""
        from src.core.database.local_db import LocalDatabase
        
        db = LocalDatabase(str(tmp_path / "test.db"))
        
        # Update trade that doesn't exist
        db.update_trade(99999, {'status': 'executed'})
        
        # Should not crash, just do nothing
        trades = db.get_trades()
        assert len(trades) == 0
        
        db.close()


class TestRateLimiterEdgeCases:
    """Test rate limiter edge cases"""
    
    def test_rate_limiter_with_zero_calls(self):
        """Test rate limiter with 0 calls remaining"""
        from src.core.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_calls=2, period=1.0)
        
        # Use up all calls
        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert not limiter.try_acquire()  # Should fail
        
        assert limiter.get_remaining_calls() == 0
    
    def test_rate_limiter_reset_time(self):
        """Test rate limit reset time calculation"""
        from src.core.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_calls=1, period=2.0)
        
        # Use the one call
        limiter.acquire()
        
        # Check reset time
        reset_time = limiter.get_reset_time()
        assert 0 < reset_time <= 2.0
    
    def test_rate_limiter_thread_safety(self):
        """Test rate limiter is thread-safe"""
        from src.core.rate_limiter import RateLimiter
        import threading
        
        limiter = RateLimiter(max_calls=10, period=1.0)
        successful_calls = []
        
        def try_call():
            if limiter.try_acquire():
                successful_calls.append(True)
        
        # Create 20 threads competing for 10 slots
        threads = [threading.Thread(target=try_call) for _ in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Exactly 10 should succeed
        assert len(successful_calls) == 10
    
    def test_rate_limiter_blocking(self):
        """Test rate limiter blocks correctly"""
        from src.core.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_calls=1, period=0.5)
        
        start = time.time()
        
        # First call immediate
        limiter.acquire()
        
        # Second call should wait ~0.5s
        limiter.acquire()
        
        elapsed = time.time() - start
        assert 0.4 < elapsed < 0.7  # Allow some timing variance


class TestConstantsEdgeCases:
    """Test constants module edge cases"""
    
    def test_all_constants_are_numbers_or_enums(self):
        """Test constants have appropriate types"""
        from src.core.constants import TradingConstants, RiskConstants
        
        # All numeric constants should be float or int
        for attr in dir(TradingConstants):
            if not attr.startswith('_'):
                value = getattr(TradingConstants, attr)
                assert isinstance(value, (int, float)), f"{attr} is not numeric"
    
    def test_percentage_constants_in_valid_range(self):
        """Test percentage constants are between 0 and 1"""
        from src.core.constants import TradingConstants, RiskConstants
        
        pct_constants = [
            TradingConstants.MAX_POSITION_SIZE_PCT,
            TradingConstants.MIN_POSITION_SIZE_PCT,
            RiskConstants.MAX_DAILY_LOSS_PCT,
            RiskConstants.MAX_WEEKLY_LOSS_PCT,
            RiskConstants.DEFAULT_STOP_LOSS_PCT,
        ]
        
        for const in pct_constants:
            assert 0 < const < 1, f"Percentage {const} out of range"
    
    def test_enum_values_are_strings(self):
        """Test all enums have string values"""
        from src.core.constants import OrderSide, OrderType, OrderStatus
        
        for side in OrderSide:
            assert isinstance(side.value, str)
        
        for order_type in OrderType:
            assert isinstance(order_type.value, str)
        
        for status in OrderStatus:
            assert isinstance(status.value, str)


class TestAsyncUtilsEdgeCases:
    """Test async utilities edge cases"""
    
    @pytest.mark.asyncio
    async def test_retry_async_max_attempts(self):
        """Test retry stops after max attempts"""
        from src.core.async_utils import retry_async
        
        attempt_count = []
        
        async def failing_func():
            attempt_count.append(1)
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await retry_async(failing_func, max_attempts=3, delay=0.01)
        
        assert len(attempt_count) == 3  # Exactly 3 attempts
    
    @pytest.mark.asyncio
    async def test_retry_async_success_on_second_attempt(self):
        """Test retry succeeds after initial failure"""
        from src.core.async_utils import retry_async
        
        attempt_count = []
        
        async def sometimes_fails():
            attempt_count.append(1)
            if len(attempt_count) < 2:
                raise ValueError("First attempt fails")
            return "success"
        
        result = await retry_async(sometimes_fails, max_attempts=3, delay=0.01)
        
        assert result == "success"
        assert len(attempt_count) == 2  # Failed once, succeeded second time
    
    @pytest.mark.asyncio
    async def test_timeout_async_exceeds(self):
        """Test async timeout raises TimeoutError"""
        from src.core.async_utils import timeout_async
        import asyncio
        
        async def slow_function():
            await asyncio.sleep(1.0)
            return "done"
        
        with pytest.raises(asyncio.TimeoutError):
            await timeout_async(slow_function(), timeout_seconds=0.1)
    
    @pytest.mark.asyncio
    async def test_gather_with_concurrency_limit(self):
        """Test concurrency limit is enforced"""
        from src.core.async_utils import gather_with_concurrency
        import asyncio
        
        concurrent_count = []
        max_concurrent = 0
        
        async def track_concurrency():
            concurrent_count.append(1)
            current = len(concurrent_count)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, current)
            
            await asyncio.sleep(0.1)
            concurrent_count.pop()
            return True
        
        # Create 10 tasks with max 3 concurrent
        tasks = [track_concurrency() for _ in range(10)]
        results = await gather_with_concurrency(3, *tasks)
        
        assert len(results) == 10
        assert max_concurrent <= 3  # Never exceeded limit


class TestDataPipelineEdgeCases:
    """Test data pipeline edge cases"""
    
    def test_mock_data_source_with_zero_symbols(self):
        """Test data source with no symbols"""
        from src.data_pipeline.mock_data_source import MockDataSource
        
        source = MockDataSource(symbols=[])
        data = source.get_latest_data()
        
        assert data == {}  # Empty dict, not None
    
    def test_mock_data_source_with_many_symbols(self):
        """Test data source with many symbols"""
        from src.data_pipeline.mock_data_source import MockDataSource
        
        symbols = [f'SYM-{i}' for i in range(100)]
        source = MockDataSource(symbols=symbols)
        data = source.get_latest_data()
        
        assert len(data) == 100
    
    def test_mock_data_source_extreme_volatility(self):
        """Test data source with extreme volatility"""
        from src.data_pipeline.mock_data_source import MockDataSource
        
        # Very high volatility (100%)
        source = MockDataSource(symbols=['BTC-USD'], volatility=1.0)
        data = source.get_latest_data()
        
        assert data is not None
        assert 'BTC-USD' in data
    
    def test_mock_data_source_zero_volatility(self):
        """Test data source with zero volatility"""
        from src.data_pipeline.mock_data_source import MockDataSource
        
        source = MockDataSource(symbols=['BTC-USD'], volatility=0.0)
        data1 = source.get_latest_data()
        data2 = source.get_latest_data()
        
        # Prices should be very similar (not identical due to rounding)
        price1 = data1['BTC-USD']['close']
        price2 = data2['BTC-USD']['close']
        assert abs(price1 - price2) / price1 < 0.001  # < 0.1% difference


@pytest.mark.parametrize("price,quantity,expected_value", [
    (50000.0, 1.0, 50000.0),
    (0.0001, 1000000.0, 100.0),  # Micro-cap with large quantity
    (1000000.0, 0.001, 1000.0),  # Expensive asset with small quantity
    (50000.0, 0, 0),  # Zero quantity
])
def test_position_value_calculation(price, quantity, expected_value):
    """Test position value calculation for various scenarios"""
    position_value = price * quantity
    assert abs(position_value - expected_value) < 0.01


@pytest.mark.parametrize("entry_price,exit_price,quantity,expected_pnl", [
    (50000, 51000, 1.0, 1000.0),      # Profit
    (50000, 49000, 1.0, -1000.0),     # Loss
    (50000, 50000, 1.0, 0.0),         # Break even
    (50000, 55000, 0.1, 500.0),       # Small quantity profit
    (100, 110, 1000, 10000.0),        # Low price, high quantity
])
def test_pnl_calculation_scenarios(entry_price, exit_price, quantity, expected_pnl):
    """Test PnL calculation for various trading scenarios"""
    pnl = (exit_price - entry_price) * quantity
    assert abs(pnl - expected_pnl) < 0.01

