from pathlib import Path
from src.core.config.loader import ConfigLoader
from src.core.database.local_db import LocalDatabase
from src.data_pipeline.mock_data_source import MockDataSource
from src.neural_network.mock_predictor import MockPredictor
import os
import pytest
import shutil
import sys
import tempfile

"""
Pytest configuration and fixtures for local testing.
Provides SQLite database, mock services, and test utilities.
"""


# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="rrr_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(test_data_dir):
    """
    Provide a clean SQLite database for each test.
    Database is created fresh for each test function.
    """
    db_path = test_data_dir / "test.db"
    
    # Remove if exists from previous test
    if db_path.exists():
        db_path.unlink()
    
    # Create database
    db = LocalDatabase(str(db_path))
    
    yield db
    
    # Cleanup
    db.close()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="session")
def test_config(test_data_dir):
    """
    Provide test configuration.
    Uses local configuration with test overrides.
    """
    # Set environment to local for testing
    os.environ['ENVIRONMENT'] = 'local'
    
    config = ConfigLoader()
    
    # Override database path to test directory
    config.config['database']['path'] = str(test_data_dir / "test.db")
    
    return config


@pytest.fixture(scope="function")
def mock_data_source():
    """Provide mock data source for testing."""
    return MockDataSource(
        symbols=['BTC-USD', 'ETH-USD'],
        volatility=0.02,
        update_interval=0.1  # Fast updates for testing
    )


@pytest.fixture(scope="function")
def mock_predictor():
    """Provide mock ML predictor for testing."""
    return MockPredictor(
        model_type="trend_following",
        random_seed=42  # Fixed seed for reproducibility
    )


@pytest.fixture(scope="function")
def test_env(monkeypatch):
    """Set common environment variables expected by config utilities."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("PAPER_TRADING", "true")
    monkeypatch.setenv("LIVE_TRADING", "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///tmp/test.db")
    return {
        "ENVIRONMENT": "test",
        "PAPER_TRADING": "true",
        "LIVE_TRADING": "false",
        "DATABASE_URL": "sqlite:///tmp/test.db",
    }


@pytest.fixture
def sample_market_data():
    """Provide sample market data for testing."""
    return {
        'symbol': 'BTC-USD',
        'timestamp': 1704067200.0,
        'open': 50000.0,
        'high': 51000.0,
        'low': 49500.0,
        'close': 50500.0,
        'volume': 1000000.0
    }


@pytest.fixture
def sample_trade():
    """Provide sample trade data for testing."""
    return {
        'symbol': 'BTC-USD',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 50000.0,
        'timestamp': 1704067200.0,
        'status': 'pending',
        'strategy': 'test_strategy'
    }


@pytest.fixture
def sample_prediction():
    """Provide sample prediction data for testing."""
    return {
        'symbol': 'BTC-USD',
        'timestamp': 1704067200.0,
        'horizon': 1,
        'predicted_price': 50500.0,
        'predicted_direction': 'up',
        'confidence': 0.75,
        'model_version': 'test-v1'
    }


@pytest.fixture(scope="function")
def populated_db(test_db, sample_market_data, sample_trade):
    """Provide database pre-populated with test data."""
    # Insert some market data
    for i in range(10):
        data = sample_market_data.copy()
        data['timestamp'] += i * 60
        data['close'] = data['close'] * (1 + (i - 5) * 0.001)
        test_db.insert_market_data(data['symbol'], data['timestamp'], {
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })
    
    # Insert some trades
    for i in range(5):
        trade = sample_trade.copy()
        trade['timestamp'] += i * 60
        test_db.insert_trade(trade)
    
    return test_db


@pytest.fixture
def temp_config_file(test_data_dir):
    """Create temporary config file for testing."""
    config_content = """
environment: test
debug: true

database:
  type: sqlite
  path: data/test.db

trading:
  mode: paper
  initial_capital: 10000

neural_network:
  mode: mock
"""
    
    config_file = test_data_dir / "test_config.yml"
    config_file.write_text(config_content)
    
    return config_file


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may be slower)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (skip with -m 'not slow')"
    )


# Helper functions for tests
class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def assert_dict_contains(actual, expected):
        """Assert that actual dict contains all keys/values from expected."""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in {actual}"
            assert actual[key] == value, f"Value mismatch for '{key}': {actual[key]} != {value}"
    
    @staticmethod
    def generate_price_series(start_price, periods, trend=0.001, volatility=0.02):
        """Generate realistic price series for testing."""
        import numpy as np
        
        prices = [start_price]
        for _ in range(periods - 1):
            change = trend + np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        
        return prices


@pytest.fixture
def test_helpers():
    """Provide test helper functions."""
    return TestHelpers()


# Async fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
