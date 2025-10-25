"""
Integration Tests for Live Trading System

Tests the complete live trading flow in paper trading mode:
- Order placement and execution
- Position management
- Risk controls and circuit breakers
- Audit logging
- Error handling

IMPORTANT: These tests use PAPER TRADING mode by default.
DO NOT run with live trading enabled unless explicitly intended.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Add trading engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "services" / "trading_engine"))


class TestLiveTradingSystem:
    """Test suite for live trading system"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Force paper trading mode for safety
        os.environ["PAPER_TRADING"] = "true"
        os.environ["LIVE_TRADING_ENABLED"] = "false"

        yield

        # Cleanup
        pass

    def test_credentials_manager_initialization(self):
        """Test credentials manager loads configuration"""
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()

        assert manager is not None
        assert manager.is_paper_trading() is True
        assert manager.is_live_trading_enabled() is False

    def test_coinbase_credentials_loading(self):
        """Test Coinbase credentials can be loaded"""
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()

        try:
            creds = manager.get_coinbase_credentials()
            assert "api_key" in creds
            assert "private_key" in creds
            assert creds["api_key"] is not None
        except ValueError:
            pytest.skip("Coinbase credentials not configured")

    def test_risk_limits_configuration(self):
        """Test risk limits are properly configured"""
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()
        limits = manager.get_risk_limits()

        assert "max_order_size_usd" in limits
        assert "max_daily_volume_usd" in limits
        assert "max_open_positions" in limits
        assert limits["max_order_size_usd"] > 0

    def test_live_trading_safety_checks(self):
        """Test live trading safety validation"""
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()

        is_safe, warnings = manager.validate_live_trading_safety()

        # Should fail in test environment (paper trading enabled)
        assert is_safe is False
        assert len(warnings) > 0
        assert any("PAPER_TRADING" in w for w in warnings)

    def test_coinbase_exchange_paper_mode(self):
        """Test Coinbase exchange initialization in paper mode"""
        from exchanges.coinbase_exchange import CoinbaseExchange

        exchange = CoinbaseExchange(paper_trading=True)

        assert exchange is not None
        assert exchange.paper_trading is True
        assert exchange.name == "Coinbase"

    def test_paper_trading_market_order(self):
        """Test market order execution in paper mode"""
        from exchanges.coinbase_exchange import CoinbaseExchange

        exchange = CoinbaseExchange(paper_trading=True)

        # Place paper market order
        order = exchange.create_market_order(
            product_id="BTC-USD",
            side="BUY",
            size=0.001
        )

        assert order["success"] is True
        assert order["order_id"] is not None
        assert order["status"] == "FILLED"
        assert order["paper_trading"] is True

    def test_paper_trading_limit_order(self):
        """Test limit order creation in paper mode"""
        from exchanges.coinbase_exchange import CoinbaseExchange

        exchange = CoinbaseExchange(paper_trading=True)

        # Place paper limit order
        order = exchange.create_limit_order(
            product_id="BTC-USD",
            side="BUY",
            size=0.001,
            price=45000.0
        )

        assert order["success"] is True
        assert order["order_id"] is not None
        assert order["status"] == "OPEN"
        assert order["price"] == 45000.0

    def test_paper_trading_order_cancellation(self):
        """Test order cancellation in paper mode"""
        from exchanges.coinbase_exchange import CoinbaseExchange

        exchange = CoinbaseExchange(paper_trading=True)

        # Place and cancel order
        order = exchange.create_limit_order(
            product_id="BTC-USD",
            side="BUY",
            size=0.001,
            price=45000.0
        )

        order_id = order["order_id"]
        cancelled = exchange.cancel_order(order_id)

        assert cancelled is True

    @pytest.mark.asyncio
    async def test_order_manager_integration(self):
        """Test order manager with paper trading"""
        from oms.order_manager import OrderManager
        from exchanges.coinbase_exchange import CoinbaseExchange

        # Setup
        exchange = CoinbaseExchange(paper_trading=True)
        await exchange.connect()

        # Skip if no database credentials
        if not os.getenv("DATABASE_PATH"):
            pytest.skip("Database credentials not configured")

        order_mgr = OrderManager(
            exchange=exchange,
            supabase_url=os.getenv("DATABASE_PATH"),
            supabase_key=os.getenv("SUPABASE_ANON_KEY")
        )

        # Create order
        order = await order_mgr.create_order(
            symbol="BTC-USD",
            side="buy",
            order_type="market",
            quantity=0.001
        )

        assert order is not None
        assert order["order_id"] is not None

    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal conditions"""
        from risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.10
        )

        breaker = CircuitBreaker(config)

        # Normal operation
        allowed = breaker.check_and_update(
            portfolio_value=100000,
            daily_pnl=500,
            open_positions=2,
            largest_position_pct=0.20,
            portfolio_volatility=0.25
        )

        assert allowed is True
        assert breaker.is_trading_allowed() is True

    def test_circuit_breaker_excessive_loss(self):
        """Test circuit breaker opens on excessive loss"""
        from risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(max_daily_loss_pct=0.05)

        breaker = CircuitBreaker(config)

        # Excessive loss triggers circuit
        allowed = breaker.check_and_update(
            portfolio_value=94000,
            daily_pnl=-6000,  # 6% loss
            open_positions=2,
            largest_position_pct=0.20,
            portfolio_volatility=0.25
        )

        assert allowed is False
        assert breaker.is_trading_allowed() is False

    def test_order_validator_valid_order(self):
        """Test order validator approves valid orders"""
        from risk.order_validator import OrderValidator

        validator = OrderValidator(
            max_order_size_usd=1000.0,
            max_position_size_pct=0.30
        )

        response = validator.validate_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.01,
            order_type="market",
            price=None,
            current_price=50000.0,
            portfolio_value=10000.0,
            current_position_size=0.0,
            open_positions=2,
            available_balance=5000.0
        )

        assert response.allowed is True
        assert response.result.value == "approved"

    def test_order_validator_rejects_oversized_order(self):
        """Test order validator rejects oversized orders"""
        from risk.order_validator import OrderValidator

        validator = OrderValidator(max_order_size_usd=1000.0)

        response = validator.validate_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.1,  # $5000 order
            order_type="market",
            price=None,
            current_price=50000.0,
            portfolio_value=10000.0,
            current_position_size=0.0,
            open_positions=2,
            available_balance=10000.0
        )

        assert response.allowed is False
        assert response.result.value == "rejected"

    def test_audit_integration_order_placement(self):
        """Test audit logging for order placement"""
        from audit_integration import AuditedTradingEngine

        engine = AuditedTradingEngine(engine_id="test-engine")

        # Place order (will be audited)
        order = engine.place_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.001,
            order_type="market",
            user_id="test-user"
        )

        assert order is not None
        assert "order_id" in order

    @pytest.mark.asyncio
    async def test_position_manager_lifecycle(self):
        """Test position opening and closing"""
        from positions.position_manager import PositionManager

        # Skip if no database credentials
        if not os.getenv("DATABASE_PATH"):
            pytest.skip("Database credentials not configured")

        mgr = PositionManager(
            supabase_url=os.getenv("DATABASE_PATH"),
            supabase_key=os.getenv("SUPABASE_ANON_KEY")
        )

        # Open position
        position = await mgr.open_position(
            symbol="BTC-USD",
            side="long",
            quantity=0.001,
            entry_price=50000.0,
            order_id="test-order-1"
        )

        assert position is not None
        assert position["status"] == "open"

        # Close position
        closed = await mgr.close_position(
            position_id=position["position_id"],
            exit_price=51000.0,
            order_id="test-order-2"
        )

        assert closed["status"] == "closed"
        assert closed["realized_pnl"] > 0  # Profit

    def test_rest_client_authentication(self):
        """Test Coinbase REST client authentication"""
        from exchanges.rest_client import CoinbaseRestClient

        try:
            client = CoinbaseRestClient()
            assert client.api_key is not None
            assert client.private_key is not None
        except ValueError:
            pytest.skip("Coinbase credentials not configured")


class TestLiveTradingSafety:
    """Safety-focused tests to prevent accidental live trading"""

    def test_environment_defaults_to_paper_trading(self):
        """Verify system defaults to paper trading"""
        from security.credentials_manager import get_credentials_manager

        # Clear any existing environment
        os.environ.pop("PAPER_TRADING", None)
        os.environ.pop("LIVE_TRADING_ENABLED", None)

        manager = get_credentials_manager()

        # Should default to paper trading
        assert manager.is_paper_trading() is True
        assert manager.is_live_trading_enabled() is False

    def test_live_trading_requires_explicit_enable(self):
        """Verify live trading requires explicit configuration"""
        from security.credentials_manager import get_credentials_manager

        os.environ["PAPER_TRADING"] = "false"
        os.environ["LIVE_TRADING_ENABLED"] = "false"

        manager = get_credentials_manager()

        is_safe, warnings = manager.validate_live_trading_safety()

        # Should fail - LIVE_TRADING_ENABLED still false
        assert is_safe is False

    def test_trading_engine_rejects_unsafe_live_mode(self):
        """Verify trading engine rejects unsafe live trading attempts"""
        # This test verifies the engine properly validates before going live
        # In CI/CD, this should always fail unless explicitly configured

        with pytest.raises(RuntimeError, match="safety"):
            # Attempt to initialize in live mode should fail
            from main import TradingEngine

            engine = TradingEngine(mode="live")

            # This should raise before reaching here
            asyncio.run(engine.initialize())


def run_smoke_tests():
    """Run quick smoke tests for deployment validation"""
    print("=" * 70)
    print("Live Trading System - Smoke Tests")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Paper trading mode verification
    print("\n1. Verifying paper trading mode...")
    try:
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()
        assert manager.is_paper_trading() is True
        print("   ✓ Paper trading enabled")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        tests_failed += 1

    # Test 2: Risk limits configured
    print("\n2. Checking risk limits...")
    try:
        from security.credentials_manager import get_credentials_manager

        manager = get_credentials_manager()
        limits = manager.get_risk_limits()
        assert limits["max_order_size_usd"] > 0
        print(f"   ✓ Risk limits configured: {limits}")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        tests_failed += 1

    # Test 3: Coinbase exchange in paper mode
    print("\n3. Testing Coinbase exchange (paper mode)...")
    try:
        from exchanges.coinbase_exchange import CoinbaseExchange

        exchange = CoinbaseExchange(paper_trading=True)
        price = exchange.get_current_price("BTC-USD")
        assert price is not None
        print(f"   ✓ Coinbase exchange operational (BTC: ${price:,.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        tests_failed += 1

    # Test 4: Circuit breaker operational
    print("\n4. Testing circuit breaker...")
    try:
        from risk.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        allowed = breaker.check_and_update(
            portfolio_value=100000,
            daily_pnl=0,
            open_positions=0,
            largest_position_pct=0,
            portfolio_volatility=0.2
        )
        assert allowed is True
        print("   ✓ Circuit breaker operational")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        tests_failed += 1

    # Test 5: Order validator
    print("\n5. Testing order validator...")
    try:
        from risk.order_validator import OrderValidator

        validator = OrderValidator()
        response = validator.validate_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.001,
            order_type="market",
            price=None,
            current_price=50000.0,
            portfolio_value=10000.0,
            current_position_size=0.0,
            open_positions=0,
            available_balance=5000.0
        )
        assert response.allowed is True
        print("   ✓ Order validator operational")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Smoke Tests Complete: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run smoke tests
    success = run_smoke_tests()

    sys.exit(0 if success else 1)
