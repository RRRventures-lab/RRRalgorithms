from enum import Enum

"""
Trading System Constants
========================

Centralized constants for the entire trading system.
Eliminates magic numbers and improves maintainability.

Author: RRR Ventures
Date: 2025-10-12
"""



# =============================================================================
# Trading Constants
# =============================================================================

class TradingConstants:
    """Core trading system constants"""
    
    # Price thresholds
    TREND_THRESHOLD_PCT = 0.01          # 1% price change defines trend
    MEAN_REVERSION_THRESHOLD_PCT = 0.02  # 2% deviation from mean
    VOLATILITY_THRESHOLD = 0.02         # 2% volatility threshold
    
    # Position sizing
    MAX_POSITION_SIZE_PCT = 0.20        # Maximum 20% of portfolio per position
    MIN_POSITION_SIZE_PCT = 0.01        # Minimum 1% of portfolio
    DEFAULT_POSITION_SIZE_PCT = 0.10    # Default 10% of portfolio
    
    # Performance metrics
    MIN_SHARPE_RATIO = 2.0              # Minimum Sharpe ratio for strategies
    MIN_WIN_RATE = 0.55                 # Minimum 55% win rate
    MIN_PROFIT_FACTOR = 1.5             # Minimum profit factor
    MAX_DRAWDOWN_PCT = 0.20             # Maximum 20% drawdown allowed
    
    # Trading frequency
    DEFAULT_UPDATE_INTERVAL_SEC = 1.0   # 1 second between updates
    MIN_UPDATE_INTERVAL_SEC = 0.1       # 100ms minimum interval
    MAX_UPDATE_INTERVAL_SEC = 60.0      # 60 seconds maximum interval


class RiskConstants:
    """Risk management constants"""
    
    # Loss limits
    MAX_DAILY_LOSS_PCT = 0.05           # Maximum 5% daily loss
    MAX_WEEKLY_LOSS_PCT = 0.10          # Maximum 10% weekly loss
    MAX_MONTHLY_LOSS_PCT = 0.20         # Maximum 20% monthly loss
    
    # Stop loss / Take profit
    DEFAULT_STOP_LOSS_PCT = 0.02        # 2% stop loss
    DEFAULT_TAKE_PROFIT_PCT = 0.04      # 4% take profit (2:1 ratio)
    TRAILING_STOP_DISTANCE_PCT = 0.01   # 1% trailing stop distance
    
    # Position limits
    MAX_OPEN_POSITIONS = 10             # Maximum concurrent positions
    MAX_LEVERAGE = 1.0                  # No leverage by default
    MIN_ACCOUNT_BALANCE = 1000.0        # Minimum $1,000 account balance
    
    # Risk scoring
    LOW_RISK_THRESHOLD = 0.3            # Risk score < 0.3 = low risk
    MEDIUM_RISK_THRESHOLD = 0.6         # Risk score < 0.6 = medium risk
    HIGH_RISK_THRESHOLD = 1.0           # Risk score >= 0.6 = high risk


class MLConstants:
    """Machine learning model constants"""
    
    # Price history
    PRICE_HISTORY_SIZE = 10             # Keep last 10 prices
    PRICE_HISTORY_SIZE_EXTENDED = 50    # Extended history for ML models
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.5                # Minimum 50% confidence
    HIGH_CONFIDENCE = 0.75              # 75%+ is high confidence
    VERY_HIGH_CONFIDENCE = 0.90         # 90%+ is very high confidence
    
    # Prediction horizons (minutes)
    HORIZON_1MIN = 1
    HORIZON_5MIN = 5
    HORIZON_15MIN = 15
    HORIZON_1HOUR = 60
    HORIZON_4HOUR = 240
    HORIZON_1DAY = 1440
    
    # Feature importance threshold
    MIN_FEATURE_IMPORTANCE = 0.01       # Features below 1% importance ignored
    
    # Model performance
    MIN_VALIDATION_ACCURACY = 0.55      # Minimum 55% accuracy
    MIN_BACKTESTING_SHARPE = 1.0        # Minimum Sharpe 1.0 in backtest


class DataConstants:
    """Data processing constants"""
    
    # Database
    DEFAULT_DB_PATH = "data/local.db"
    MAX_DB_QUERY_LIMIT = 10000          # Maximum rows per query
    DEFAULT_QUERY_LIMIT = 100           # Default query limit
    DB_TIMEOUT_SECONDS = 30.0           # Database timeout
    
    # Data retention
    MAX_MARKET_DATA_DAYS = 365          # Keep 1 year of market data
    MAX_TRADE_HISTORY_DAYS = 730        # Keep 2 years of trade history
    MAX_LOG_RETENTION_DAYS = 90         # Keep 90 days of logs
    
    # Caching
    CACHE_TTL_SECONDS = 5               # 5 second cache TTL
    POSITION_CACHE_TTL = 2              # 2 second position cache
    PORTFOLIO_CACHE_TTL = 5             # 5 second portfolio cache
    
    # Batch sizes
    BATCH_INSERT_SIZE = 100             # Insert 100 rows at a time
    BATCH_UPDATE_SIZE = 50              # Update 50 rows at a time


class APIConstants:
    """External API constants"""
    
    # Rate limiting (calls per second)
    POLYGON_RATE_LIMIT = 5              # 5 calls/sec for free tier
    COINBASE_RATE_LIMIT = 3             # 3 calls/sec
    PERPLEXITY_RATE_LIMIT = 1           # 1 call/sec
    
    # Timeouts (seconds)
    API_TIMEOUT_SECONDS = 10.0          # Default API timeout
    API_RETRY_ATTEMPTS = 3              # Retry failed requests 3 times
    API_RETRY_DELAY_SECONDS = 1.0       # Wait 1 second between retries
    
    # Response limits
    MAX_API_RESPONSE_SIZE_MB = 10       # Max 10MB response
    MAX_WEBSOCKET_MESSAGE_SIZE_KB = 256 # Max 256KB WebSocket message


class MonitoringConstants:
    """Monitoring and alerting constants"""
    
    # Performance targets
    TARGET_SIGNAL_LATENCY_MS = 100      # Target <100ms signal latency
    TARGET_ORDER_LATENCY_MS = 50        # Target <50ms order latency
    TARGET_DATA_DELAY_MS = 1000         # Target <1s data delay
    MAX_STARTUP_TIME_SECONDS = 5        # Max 5s startup time
    
    # Resource limits
    MAX_MEMORY_GB = 4.0                 # Maximum 4GB memory
    MAX_CPU_PERCENT = 80                # Alert at 80% CPU
    MAX_DISK_USAGE_PERCENT = 90         # Alert at 90% disk
    
    # Alert thresholds
    ERROR_RATE_THRESHOLD = 0.01         # Alert at 1% error rate
    LATENCY_P99_THRESHOLD_MS = 200      # Alert if P99 latency >200ms
    
    # Logging
    LOG_ROTATION_SIZE_MB = 100          # Rotate logs at 100MB
    LOG_RETENTION_COUNT = 10            # Keep 10 log files


class TestConstants:
    """Testing constants"""
    
    # Test data
    TEST_SYMBOL = "BTC-USD"
    TEST_PRICE = 50000.0
    TEST_QUANTITY = 1.0
    TEST_INITIAL_CAPITAL = 10000.0
    
    # Test thresholds
    FLOAT_COMPARISON_EPSILON = 0.01     # Allow 0.01 float comparison error
    PERFORMANCE_TEST_TIMEOUT_SEC = 5    # Performance tests must complete in 5s
    
    # Coverage targets
    MIN_UNIT_TEST_COVERAGE = 0.80       # 80% unit test coverage
    MIN_INTEGRATION_TEST_COVERAGE = 0.60 # 60% integration test coverage
    MIN_OVERALL_COVERAGE = 0.75         # 75% overall coverage


# =============================================================================
# String Constants
# =============================================================================

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIAL = "partial"


class SignalDirection(Enum):
    """Trading signal direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    HOLD = "hold"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service status"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# =============================================================================
# Validation Constants
# =============================================================================

class ValidationConstants:
    """Input validation constants"""
    
    # Symbol validation
    MAX_SYMBOL_LENGTH = 20
    MIN_SYMBOL_LENGTH = 3
    VALID_SYMBOL_PATTERN = r"^[A-Z0-9\-]+$"
    
    # Price validation
    MIN_PRICE = 0.000001                # Minimum $0.000001 (for micro-cap)
    MAX_PRICE = 10000000.0              # Maximum $10M per unit
    
    # Quantity validation
    MIN_QUANTITY = 0.000001             # Minimum quantity
    MAX_QUANTITY = 1000000.0            # Maximum quantity
    
    # String lengths
    MAX_STRATEGY_NAME_LENGTH = 50
    MAX_NOTES_LENGTH = 1000
    MAX_ERROR_MESSAGE_LENGTH = 500


# =============================================================================
# Environment Constants
# =============================================================================

class EnvironmentType(Enum):
    """Environment type"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(Enum):
    """Trading mode"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


# =============================================================================
# Export all constants
# =============================================================================

__all__ = [
    'TradingConstants',
    'RiskConstants',
    'MLConstants',
    'DataConstants',
    'APIConstants',
    'MonitoringConstants',
    'TestConstants',
    'ValidationConstants',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'SignalDirection',
    'RiskLevel',
    'ServiceStatus',
    'EnvironmentType',
    'TradingMode',
]

