"""
Core utilities and shared functionality for RRRalgorithms trading system.
"""

__version__ = "0.1.0"

# Import available modules
try:
    from .config import get_project_root, get_env_file, load_config
except ImportError:
    pass

try:
    from .exceptions import (
        RRRAlgorithmsError,
        ConfigurationError,
        SecretError,
        DatabaseConnectionError,
        DataPipelineError,
        DataIngestionError,
        DataValidationError,
        APIError,
        RateLimitError,
        TradingError,
        OrderError,
        InvalidOrderError,
        OrderRejectedError,
        InsufficientFundsError,
        PositionError,
        ExchangeError,
        RiskError,
        RiskLimitExceededError,
        MaxDrawdownExceededError,
        MaxPositionSizeExceededError,
        MaxDailyLossExceededError,
        ModelError,
        ModelNotFoundError,
        ModelLoadError,
        InferenceError,
        InvalidInputError,
        ValidationError,
        AIValidationError,
        HallucinationDetectedError,
        SecurityError,
        AuthenticationError,
        AuthorizationError,
        AuditError,
        BacktestError,
        InvalidBacktestPeriodError,
        InsufficientDataError,
        MonitoringError,
        MetricError,
        AlertError,
        get_exception_details,
        is_retryable_error,
    )
except ImportError:
    pass

try:
    from .retry import async_retry, sync_retry
except ImportError:
    pass

try:
    from .settings import (
        Settings,
        get_settings,
        reload_settings,
        is_production,
        is_development,
        is_testing,
    )
except ImportError:
    pass

# Database imports disabled - using local SQLite database instead
# from .database import (
#     DatabasePool,
#     get_database_pool,
#     close_database_pool,
#     execute_query,
#     execute_many,
#     get_supabase,
#     check_database_health,
# )

__all__ = [
    # Config & Settings
    "get_project_root",
    "get_env_file",
    "load_config",
    "Settings",
    "get_settings",
    "reload_settings",
    "is_production",
    "is_development",
    "is_testing",
    # Retry utilities
    "async_retry",
    "sync_retry",
    # Database - disabled, using local SQLite database instead
    # "DatabasePool",
    # "get_database_pool",
    # "close_database_pool",
    # "execute_query",
    # "execute_many",
    # "get_supabase",
    # "check_database_health",
    # Exceptions
    "RRRAlgorithmsError",
    "ConfigurationError",
    "SecretError",
    "DatabaseConnectionError",
    "DataPipelineError",
    "DataIngestionError",
    "DataValidationError",
    "APIError",
    "RateLimitError",
    "TradingError",
    "OrderError",
    "InvalidOrderError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "PositionError",
    "ExchangeError",
    "RiskError",
    "RiskLimitExceededError",
    "MaxDrawdownExceededError",
    "MaxPositionSizeExceededError",
    "MaxDailyLossExceededError",
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "InvalidInputError",
    "ValidationError",
    "AIValidationError",
    "HallucinationDetectedError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "AuditError",
    "BacktestError",
    "InvalidBacktestPeriodError",
    "InsufficientDataError",
    "MonitoringError",
    "MetricError",
    "AlertError",
    "get_exception_details",
    "is_retryable_error",
]
