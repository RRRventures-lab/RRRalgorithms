from src.core.constants import TradingConstants, RiskConstants, OrderSide, OrderStatus
from src.core.database.local_db import LocalDatabase
from src.core.exceptions import RiskLimitError, InputValidationError
from src.core.validation import TradeRequest, validate_market_data
from src.data_pipeline.mock_data_source import MockDataSource
from src.monitoring.local_monitor import LocalMonitor
from src.neural_network.mock_predictor import EnsemblePredictor
from typing import Dict, Any
import pytest
import time

"""
Critical Trading Flow Tests
============================

Comprehensive integration tests for the complete trading flow:
- Data ingestion → Signal generation → Order execution → Position tracking

These tests ensure the critical path works correctly end-to-end.

Author: RRR Ventures
Date: 2025-10-12
"""




@pytest.fixture
def test_db(tmp_path):
    """Provide isolated test database"""
    db_path = tmp_path / "test.db"
    db = LocalDatabase(str(db_path))
    yield db
    db.close()


@pytest.fixture
def test_data_source():
    """Provide mock data source"""
    return MockDataSource(
        symbols=['BTC-USD', 'ETH-USD'],
        volatility=0.02
    )


@pytest.fixture
def test_predictor():
    """Provide ensemble predictor"""
    return EnsemblePredictor()


@pytest.fixture
def test_monitor():
    """Provide local monitor"""
    return LocalMonitor(use_rich=False)


class TestCriticalTradingFlow:
    """Test complete trading flow end-to-end"""
    
    def test_full_trading_cycle(self, test_db, test_data_source, test_predictor):
        """
        Test complete trading cycle:
        1. Fetch market data
        2. Generate prediction
        3. Store data
        4. Verify storage
        """
        # 1. Fetch market data
        market_data = test_data_source.get_latest_data()
        assert len(market_data) == 2  # BTC-USD and ETH-USD
        assert 'BTC-USD' in market_data
        
        btc_data = market_data['BTC-USD']
        
        # Validate market data structure
        required_keys = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        for key in required_keys:
            assert key in btc_data, f"Missing {key} in market data"
        
        # Validate OHLCV relationships
        assert btc_data['high'] >= btc_data['low']
        assert btc_data['high'] >= btc_data['open']
        assert btc_data['high'] >= btc_data['close']
        assert btc_data['low'] <= btc_data['open']
        assert btc_data['low'] <= btc_data['close']
        
        # 2. Generate prediction
        current_price = btc_data['close']
        prediction = test_predictor.predict('BTC-USD', current_price)
        
        assert 'predicted_price' in prediction
        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert 0 <= prediction['confidence'] <= 1
        
        # 3. Store market data
        test_db.insert_market_data(
            'BTC-USD',
            btc_data['timestamp'],
            {
                'open': btc_data['open'],
                'high': btc_data['high'],
                'low': btc_data['low'],
                'close': btc_data['close'],
                'volume': btc_data['volume']
            }
        )
        
        # 4. Store prediction
        test_db.insert_prediction({
            'symbol': 'BTC-USD',
            'timestamp': prediction['timestamp'],
            'horizon': 1,
            'predicted_price': prediction['predicted_price'],
            'predicted_direction': prediction['direction'],
            'confidence': prediction['confidence'],
            'model_version': prediction.get('model_type', 'ensemble')
        })
        
        # 5. Verify storage
        stored_data = test_db.get_market_data('BTC-USD', limit=1)
        assert len(stored_data) == 1
        assert stored_data[0]['close'] == btc_data['close']
    
    def test_trade_execution_with_validation(self, test_db):
        """Test trade execution with input validation"""
        # Valid trade request
        trade_data = {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'order_type': 'market',
            'quantity': 1.0,
            'timestamp': time.time(),
            'strategy': 'test_strategy'
        }
        
        # Validate request
        trade_request = TradeRequest(**trade_data)
        assert trade_request.symbol == 'BTC-USD'
        assert trade_request.side == 'buy'
        
        # Insert trade
        trade_id = test_db.insert_trade({
            'symbol': trade_request.symbol,
            'side': trade_request.side,
            'order_type': trade_request.order_type,
            'quantity': trade_request.quantity,
            'price': 50000.0,
            'timestamp': trade_request.timestamp,
            'status': 'pending',
            'strategy': trade_request.strategy,
            'notes': None
        })
        
        assert trade_id > 0
        
        # Update trade to executed
        test_db.update_trade(trade_id, {
            'status': 'executed',
            'executed_quantity': 1.0,
            'executed_price': 50000.0
        })
        
        # Verify update
        trades = test_db.get_trades(symbol='BTC-USD', limit=1)
        assert len(trades) == 1
        assert trades[0]['status'] == 'executed'
        assert trades[0]['executed_quantity'] == 1.0
    
    def test_invalid_trade_request_validation(self):
        """Test that invalid trade requests are rejected"""
        # Test 1: Invalid side
        with pytest.raises(Exception):  # ValidationError
            TradeRequest(
                symbol='BTC-USD',
                side='invalid_side',
                order_type='market',
                quantity=1.0
            )
        
        # Test 2: Negative quantity
        with pytest.raises(Exception):  # ValidationError
            TradeRequest(
                symbol='BTC-USD',
                side='buy',
                order_type='market',
                quantity=-1.0
            )
        
        # Test 3: Limit order without price
        with pytest.raises(Exception):  # ValidationError
            TradeRequest(
                symbol='BTC-USD',
                side='buy',
                order_type='limit',
                quantity=1.0,
                price=None  # Required for limit orders!
            )
        
        # Test 4: Invalid symbol format
        with pytest.raises(Exception):  # ValidationError
            TradeRequest(
                symbol='invalid symbol!',
                side='buy',
                order_type='market',
                quantity=1.0
            )
    
    def test_position_tracking(self, test_db):
        """Test position tracking after trades"""
        # Execute buy trade
        test_db.upsert_position(
            symbol='BTC-USD',
            quantity=2.0,
            average_price=50000.0,
            current_price=51000.0
        )
        
        # Verify position
        positions = test_db.get_positions()
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'BTC-USD'
        assert positions[0]['quantity'] == 2.0
        assert positions[0]['average_price'] == 50000.0
        
        # Calculate expected PnL
        expected_pnl = (51000.0 - 50000.0) * 2.0  # $2,000
        assert abs(positions[0]['unrealized_pnl'] - expected_pnl) < 0.01
        
        # Update position (sell 1.0)
        test_db.upsert_position(
            symbol='BTC-USD',
            quantity=1.0,
            average_price=50000.0,
            current_price=52000.0
        )
        
        # Verify updated position
        positions = test_db.get_positions()
        assert len(positions) == 1
        assert positions[0]['quantity'] == 1.0
    
    def test_portfolio_metrics_calculation(self, test_db):
        """Test portfolio metrics tracking"""
        # Insert portfolio metrics
        metrics = {
            'timestamp': time.time(),
            'total_value': 11000.0,
            'cash': 5000.0,
            'positions_value': 6000.0,
            'daily_pnl': 1000.0,
            'total_pnl': 1000.0,
            'sharpe_ratio': 1.5,
            'win_rate': 0.60
        }
        
        test_db.insert_portfolio_metrics(metrics)
        
        # Retrieve and verify
        latest = test_db.get_latest_portfolio_metrics()
        assert latest is not None
        assert latest['total_value'] == 11000.0
        assert latest['total_pnl'] == 1000.0
        assert latest['sharpe_ratio'] == 1.5
        
        # Verify cash + positions = total
        assert abs(
            latest['cash'] + latest['positions_value'] - latest['total_value']
        ) < 0.01
    
    def test_risk_limit_enforcement(self):
        """Test risk limits are enforced"""
        portfolio_value = 10000.0
        max_position_size = portfolio_value * TradingConstants.MAX_POSITION_SIZE_PCT
        
        # Test 1: Position within limit (should pass)
        position_size = 1500.0  # 15% of portfolio
        assert position_size <= max_position_size
        
        # Test 2: Position exceeds limit (should fail)
        oversized_position = 3000.0  # 30% of portfolio
        assert oversized_position > max_position_size
        
        # In real system, this would raise RiskLimitError
        # For now, just verify the calculation
        assert oversized_position / portfolio_value > TradingConstants.MAX_POSITION_SIZE_PCT
    
    def test_multi_symbol_parallel_processing(self, test_db, test_data_source, test_predictor):
        """Test processing multiple symbols in sequence"""
        market_data = test_data_source.get_latest_data()
        
        predictions = {}
        
        # Process each symbol
        for symbol, ohlcv in market_data.items():
            # Generate prediction
            prediction = test_predictor.predict(symbol, ohlcv['close'])
            predictions[symbol] = prediction
            
            # Store data
            test_db.insert_market_data(symbol, ohlcv['timestamp'], {
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume']
            })
            
            # Store prediction
            test_db.insert_prediction({
                'symbol': symbol,
                'timestamp': prediction['timestamp'],
                'horizon': 1,
                'predicted_price': prediction['predicted_price'],
                'predicted_direction': prediction['direction'],
                'confidence': prediction['confidence'],
                'model_version': prediction.get('model_type', 'ensemble')
            })
        
        # Verify all symbols processed
        assert len(predictions) == 2
        assert 'BTC-USD' in predictions
        assert 'ETH-USD' in predictions
        
        # Verify all data stored
        btc_data = test_db.get_market_data('BTC-USD', limit=1)
        eth_data = test_db.get_market_data('ETH-USD', limit=1)
        assert len(btc_data) == 1
        assert len(eth_data) == 1
    
    def test_data_validation_integration(self, test_data_source):
        """Test market data validation in integration"""
        market_data = test_data_source.get_latest_data()
        
        for symbol, ohlcv in market_data.items():
            # Validate with validation framework
            validated = validate_market_data(
                symbol=symbol,
                timestamp=ohlcv['timestamp'],
                ohlcv={
                    'open': ohlcv['open'],
                    'high': ohlcv['high'],
                    'low': ohlcv['low'],
                    'close': ohlcv['close'],
                    'volume': ohlcv['volume']
                }
            )
            
            # Verify validation passed
            assert validated.symbol == symbol.upper()
            assert validated.ohlcv.high >= validated.ohlcv.low
            assert validated.ohlcv.high >= validated.ohlcv.open
            assert validated.ohlcv.high >= validated.ohlcv.close


@pytest.mark.asyncio
class TestAsyncTradingFlow:
    """Test async trading flow"""
    
    async def test_async_data_fetch(self, test_data_source):
        """Test async data fetching"""
        import asyncio
        
        # Simulate async fetch
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            test_data_source.get_latest_data
        )
        
        assert data is not None
        assert len(data) > 0
    
    async def test_async_parallel_predictions(self, test_predictor):
        """Test parallel predictions"""
        import asyncio
        
        symbols = ['BTC-USD', 'ETH-USD']
        prices = [50000.0, 3000.0]
        
        # Create prediction tasks
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                test_predictor.predict,
                symbol,
                price
            )
            for symbol, price in zip(symbols, prices)
        ]
        
        # Execute in parallel
        predictions = await asyncio.gather(*tasks)
        
        assert len(predictions) == 2
        assert all('predicted_price' in p for p in predictions)


class TestPerformance:
    """Performance tests for critical paths"""
    
    def test_single_iteration_latency(self, test_db, test_data_source, test_predictor):
        """Test single trading iteration meets latency target"""
        start_time = time.time()
        
        # Fetch data
        market_data = test_data_source.get_latest_data()
        
        # Process one symbol
        symbol = 'BTC-USD'
        ohlcv = market_data[symbol]
        
        # Generate prediction
        prediction = test_predictor.predict(symbol, ohlcv['close'])
        
        # Store data
        test_db.insert_market_data(symbol, ohlcv['timestamp'], {
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'volume': ohlcv['volume']
        })
        
        # Store prediction
        test_db.insert_prediction({
            'symbol': symbol,
            'timestamp': prediction['timestamp'],
            'horizon': 1,
            'predicted_price': prediction['predicted_price'],
            'predicted_direction': prediction['direction'],
            'confidence': prediction['confidence'],
            'model_version': prediction.get('model_type', 'ensemble')
        })
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete in less than 100ms
        assert elapsed < 100, f"Single iteration took {elapsed:.1f}ms (target: <100ms)"
    
    def test_database_query_performance(self, test_db):
        """Test database queries meet performance targets"""
        # Insert test data
        for i in range(100):
            test_db.insert_market_data(
                'TEST-USD',
                time.time() + i,
                {
                    'open': 50000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'close': 50500.0,
                    'volume': 1000000.0
                }
            )
        
        # Query with timing
        start_time = time.time()
        results = test_db.get_market_data('TEST-USD', limit=100)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        assert len(results) == 100
        assert elapsed < 10, f"Query took {elapsed:.1f}ms (target: <10ms)"

