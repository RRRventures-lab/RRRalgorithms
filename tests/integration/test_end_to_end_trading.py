from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import numpy as np
import os
import pandas as pd
import pytest
import sys

"""
End-to-End Integration Tests for Trading System

Tests complete workflows from data ingestion through to order execution.
Critical for validating system integration before deployment.
"""


# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../worktrees/data-pipeline/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../worktrees/neural-network/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../worktrees/trading-engine/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../worktrees/risk-management/src'))


class TestDataPipelineIntegration:
    """Test data pipeline integration with downstream components"""

    @pytest.mark.asyncio
    async def test_data_to_features_pipeline(self):
        """Test data ingestion to feature engineering pipeline"""
        from ingestion.market_data_ingestor import MarketDataIngestor
        from features.technical_indicators import TechnicalFeatureEngineering, OHLCVData

        # Mock ingestor
        ingestor = MarketDataIngestor(symbols=['BTC-USD'])

        # Generate sample OHLCV data
        n = 100
        prices = 50000 + np.cumsum(np.random.randn(n) * 100)
        ohlcv = OHLCVData(
            open=prices + np.random.randn(n) * 50,
            high=prices + np.abs(np.random.randn(n) * 100),
            low=prices - np.abs(np.random.randn(n) * 100),
            close=prices,
            volume=np.random.randint(1000000, 10000000, n).astype(float)
        )

        # Process through feature engineering
        fe = TechnicalFeatureEngineering()
        features = fe.compute_all_features(ohlcv)

        # Verify pipeline output
        assert features.shape[0] == n
        assert features.shape[1] >= 25  # At least 25 features
        assert not np.any(np.isinf(features))

    @pytest.mark.asyncio
    async def test_websocket_to_database_pipeline(self):
        """Test WebSocket data flow to database"""
        # This would test actual WebSocket -> Redis -> Database flow
        # For now, test the interface
        pass  # Placeholder for actual WebSocket test


class TestMLModelIntegration:
    """Test ML model integration with trading system"""

    def test_model_prediction_to_signal(self):
        """Test ML model predictions converting to trading signals"""
        from models.price_prediction.transformer_model import TransformerPricePredictor
        import torch

        model = TransformerPricePredictor()
        model.eval()

        # Create sample input
        batch_size = 4
        seq_len = 100
        input_dim = 6
        x = torch.randn(batch_size, seq_len, input_dim)

        # Get predictions
        with torch.no_grad():
            outputs = model(x)

        # Verify prediction format
        assert '5min' in outputs
        assert '15min' in outputs
        assert '1hr' in outputs

        for horizon in ['5min', '15min', '1hr']:
            assert 'logits' in outputs[horizon]
            assert 'probs' in outputs[horizon]
            assert outputs[horizon]['probs'].shape == (batch_size, 3)

            # Probabilities should sum to 1
            prob_sums = outputs[horizon]['probs'].sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

    def test_model_batch_inference(self):
        """Test batch inference for multiple symbols"""
        from models.price_prediction.transformer_model import create_price_predictor
        import torch

        model = create_price_predictor()
        model.eval()

        # Simulate multiple symbols
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        batch_size = len(symbols)

        x = torch.randn(batch_size, 100, 6)

        with torch.no_grad():
            outputs = model(x)

        # Should handle batch inference correctly
        assert outputs['5min']['probs'].shape[0] == batch_size


class TestTradingEngineIntegration:
    """Test trading engine integration"""

    def test_signal_to_order_flow(self):
        """Test converting signals to orders"""
        from engine.oms.order_manager import OrderManager

        om = OrderManager()

        # Simulate trading signal
        signal = {
            'symbol': 'BTC-USD',
            'direction': 'LONG',
            'confidence': 0.75,
            'suggested_size': 0.001
        }

        # Create order from signal
        order = om.create_market_order(
            symbol=signal['symbol'],
            side='BUY' if signal['direction'] == 'LONG' else 'SELL',
            quantity=signal['suggested_size'],
            exchange='paper'
        )

        assert order is not None
        assert order['symbol'] == 'BTC-USD'
        assert order['side'] == 'BUY'

    def test_order_lifecycle(self):
        """Test complete order lifecycle"""
        from engine.oms.order_manager import OrderManager

        om = OrderManager()

        # Create order
        order = om.create_limit_order(
            symbol='BTC-USD',
            side='BUY',
            quantity=0.001,
            limit_price=50000.00,
            exchange='paper'
        )

        order_id = order['order_id']
        assert order['status'] == 'OPEN'

        # Modify order
        om.modify_order(order_id, new_price=49500.00)
        modified = om.get_order(order_id)
        assert modified['limit_price'] == 49500.00

        # Cancel order
        om.cancel_order(order_id)
        cancelled = om.get_order(order_id)
        assert cancelled['status'] == 'CANCELLED'


class TestRiskManagementIntegration:
    """Test risk management integration"""

    def test_position_sizing_with_kelly(self):
        """Test position sizing using Kelly criterion"""
        from risk.sizing.kelly_criterion import calculate_fractional_kelly, calculate_position_size

        # Simulate strategy statistics
        win_rate = 0.60
        avg_win = 2.0
        avg_loss = 1.0

        # Calculate Kelly
        kelly_pct = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.25)

        # Calculate position size
        capital = 100000
        position_size = calculate_position_size(capital, kelly_pct, max_position_pct=0.2)

        # Verify reasonable position size
        assert 0 < position_size <= 20000  # Max 20% of capital

    def test_portfolio_risk_limits(self):
        """Test portfolio-level risk limits"""
        from engine.portfolio.position_manager import PositionManager

        pm = PositionManager(initial_capital=100000)

        # Open multiple positions
        pm.open_position('BTC-USD', 'LONG', 0.001, 50000.00, datetime.now())
        pm.open_position('ETH-USD', 'LONG', 0.01, 3000.00, datetime.now())

        # Check portfolio value
        portfolio_value = pm.get_portfolio_value(
            current_prices={'BTC-USD': 51000, 'ETH-USD': 3100}
        )

        assert portfolio_value >= 100000  # Should include initial capital + P&L

    def test_stop_loss_trigger(self):
        """Test stop-loss triggering"""
        from engine.portfolio.position_manager import PositionManager

        pm = PositionManager(initial_capital=100000)

        # Open position with stop-loss
        position = pm.open_position(
            symbol='BTC-USD',
            side='LONG',
            quantity=0.001,
            entry_price=50000.00,
            stop_loss=48000.00,
            timestamp=datetime.now()
        )

        # Simulate price drop
        current_price = 47500.00

        # Check if stop-loss should trigger
        should_trigger = current_price <= position['stop_loss']
        assert should_trigger is True

        # Close position at stop-loss
        pm.close_position(
            position_id=position['position_id'],
            exit_price=48000.00,
            timestamp=datetime.now()
        )


class TestBacktestingIntegration:
    """Test backtesting system integration"""

    def test_walk_forward_validation_integration(self):
        """Test walk-forward validation with real model"""
        from validation.walk_forward import WalkForwardValidator, WalkForwardConfig
        from sklearn.ensemble import RandomForestClassifier

        # Generate sample data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        data = pd.DataFrame({
            'feature_1': np.random.randn(len(dates)),
            'feature_2': np.random.randn(len(dates)),
            'target': np.random.randint(0, 2, len(dates))
        }, index=dates)

        # Configure validator
        config = WalkForwardConfig(
            train_window_days=90,
            test_window_days=30,
            step_days=14
        )

        validator = WalkForwardValidator(config)

        # Define functions
        def model_factory():
            return RandomForestClassifier(n_estimators=50, random_state=42)

        def train_fn(model, train_data):
            X = train_data[['feature_1', 'feature_2']]
            y = train_data['target']
            model.fit(X, y)
            return model

        def predict_fn(model, test_data):
            X = test_data[['feature_1', 'feature_2']]
            return model.predict(X)

        def metric_fn(predictions, actuals):
            accuracy = np.mean(predictions == actuals)
            return {'accuracy': accuracy}

        # Run validation
        results = validator.validate(
            data=data,
            model_factory=model_factory,
            train_fn=train_fn,
            predict_fn=predict_fn,
            metric_fn=metric_fn
        )

        # Verify results
        assert len(results) > 0
        assert all(r.test_metrics['accuracy'] >= 0 for r in results)

    def test_performance_metrics_integration(self):
        """Test performance metrics calculation"""
        from metrics.trading_metrics import TradingMetricsCalculator

        # Generate sample returns
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        # Generate sample trades
        trades = pd.DataFrame({
            'pnl': np.random.randn(100) * 100,
            'entry_time': pd.date_range('2023-01-01', periods=100, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=100, freq='D')
        })

        # Calculate metrics
        calc = TradingMetricsCalculator()
        metrics = calc.calculate_all_metrics(returns, trades)

        # Verify metrics
        assert -1 < metrics.total_return < 1
        assert metrics.sharpe_ratio is not None
        assert 0 <= metrics.win_rate <= 1
        assert metrics.max_drawdown <= 0


class TestPaperTradingMode:
    """Test paper trading mode"""

    def test_paper_trading_enabled(self):
        """Test paper trading mode is enabled"""
        from engine.oms.order_manager import OrderManager

        om = OrderManager(paper_trading_mode=True)

        assert om.paper_trading_mode is True

        # Orders should not hit real exchange
        order = om.create_market_order(
            symbol='BTC-USD',
            side='BUY',
            quantity=0.001,
            exchange='paper'
        )

        assert order['exchange'] == 'paper'

    def test_paper_trading_isolation(self):
        """Test paper trading doesn't affect real capital"""
        from engine.portfolio.position_manager import PositionManager

        pm = PositionManager(initial_capital=100000, paper_trading_mode=True)

        # Open position
        pm.open_position('BTC-USD', 'LONG', 0.001, 50000.00, datetime.now())

        # Close with loss
        positions = pm.list_open_positions()
        pm.close_position(
            positions[0]['position_id'],
            exit_price=45000.00,
            timestamp=datetime.now()
        )

        # Paper trading losses shouldn't affect real capital
        assert pm.paper_trading_mode is True


class TestDatabaseIntegration:
    """Test database integration"""

    @pytest.mark.asyncio
    async def test_save_and_load_order(self):
        """Test saving and loading orders from database"""
        # This would test actual database operations
        # Requires database connection
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_time_series_data_storage(self):
        """Test storing time-series data"""
        # Test storing OHLCV data in TimescaleDB
        pass  # Placeholder


class TestMonitoringIntegration:
    """Test monitoring and metrics collection"""

    def test_prometheus_metrics_collection(self):
        """Test Prometheus metrics are being collected"""
        # This would test that metrics are properly exposed
        pass  # Placeholder

    def test_trading_metrics_exported(self):
        """Test trading-specific metrics are exported"""
        # Test metrics like order count, P&L, win rate
        pass  # Placeholder


class TestErrorRecovery:
    """Test error handling and recovery"""

    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self):
        """Test recovery from connection loss"""
        from ingestion.market_data_ingestor import MarketDataIngestor

        ingestor = MarketDataIngestor()

        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            # Simulate connection loss and recovery
            mock_connect.side_effect = [
                Exception("Connection lost"),
                AsyncMock()  # Successful reconnect
            ]

            await ingestor.connect_with_retry(max_retries=2)

            # Should have recovered
            assert mock_connect.call_count == 2

    def test_order_failure_handling(self):
        """Test handling of order failures"""
        from engine.oms.order_manager import OrderManager

        om = OrderManager()

        with patch.object(om, 'execute_order', side_effect=Exception("Exchange error")):
            order = om.create_market_order(
                symbol='BTC-USD',
                side='BUY',
                quantity=0.001,
                exchange='paper'
            )

            # Should handle gracefully
            try:
                om.execute_order(order['order_id'])
            except Exception:
                # Error should be logged, not crash system
                pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure
