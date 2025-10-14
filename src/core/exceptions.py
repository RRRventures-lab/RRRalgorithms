from functools import lru_cache
from typing import Optional, Dict, Any


"""
Core Exception Classes for RRRalgorithms Trading System
Provides a hierarchy of custom exceptions for better error handling
"""



class RRRAlgorithmsError(Exception):
    """Base exception for all RRRalgorithms errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Configuration & Setup Errors
# =============================================================================


class ConfigurationError(RRRAlgorithmsError):
    """Raised when configuration is invalid or missing"""

    pass


class SecretError(RRRAlgorithmsError):
    """Raised when secrets/API keys are missing or invalid"""

    pass


class DatabaseConnectionError(RRRAlgorithmsError):
    """Raised when database connection fails"""

    pass


# =============================================================================
# Data Pipeline Errors
# =============================================================================


class DataPipelineError(RRRAlgorithmsError):
    """Base class for data pipeline errors"""

    pass


class DataIngestionError(DataPipelineError):
    """Raised when data ingestion fails"""

    pass


class DataValidationError(DataPipelineError):
    """Raised when data validation fails"""

    pass


class APIError(DataPipelineError):
    """Raised when external API calls fail"""

    def __init__(
        self,
        message: str,
        api_name: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.api_name = api_name
        self.status_code = status_code
        super().__init__(message, details)

    def __str__(self) -> str:
        base = f"{self.api_name} API Error: {self.message}"
        if self.status_code:
            base += f" (Status: {self.status_code})"
        if self.details:
            base += f" | Details: {self.details}"
        return base


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded"""

    pass


# =============================================================================
# Trading Engine Errors
# =============================================================================


class TradingError(RRRAlgorithmsError):
    """Base class for trading-related errors"""

    pass


class OrderError(TradingError):
    """Raised when order operations fail"""

    pass


class InvalidOrderError(OrderError):
    """Raised when order parameters are invalid"""

    pass


class OrderRejectedError(OrderError):
    """Raised when exchange rejects an order"""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.order_id = order_id
        self.reason = reason
        super().__init__(message, details)


class InsufficientFundsError(OrderError):
    """Raised when account has insufficient funds"""

    pass


class PositionError(TradingError):
    """Raised when position operations fail"""

    pass


class ExchangeError(TradingError):
    """Raised when exchange operations fail"""

    pass


# =============================================================================
# Risk Management Errors
# =============================================================================


class RiskError(RRRAlgorithmsError):
    """Base class for risk management errors"""

    pass


class RiskLimitExceededError(RiskError):
    """Raised when risk limits are exceeded"""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: float,
        limit_value: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(message, details)

    def __str__(self) -> str:
        return (
            f"Risk Limit Exceeded ({self.limit_type}): "
            f"{self.current_value} > {self.limit_value} | {self.message}"
        )


class MaxDrawdownExceededError(RiskLimitExceededError):
    """Raised when maximum drawdown is exceeded"""

    pass


class MaxPositionSizeExceededError(RiskLimitExceededError):
    """Raised when position size exceeds limits"""

    pass


class MaxDailyLossExceededError(RiskLimitExceededError):
    """Raised when daily loss limit is exceeded"""

    pass


# =============================================================================
# ML/Model Errors
# =============================================================================


class ModelError(RRRAlgorithmsError):
    """Base class for ML model errors"""

    pass


class ModelNotFoundError(ModelError):
    """Raised when model file is not found"""

    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails"""

    pass


class InferenceError(ModelError):
    """Raised when model inference fails"""

    pass


class InvalidInputError(ModelError):
    """Raised when model input is invalid"""

    pass


# =============================================================================
# Validation & Security Errors
# =============================================================================


class ValidationError(RRRAlgorithmsError):
    """Base class for validation errors"""

    pass


class AIValidationError(ValidationError):
    """Raised when AI validation fails"""

    def __init__(
        self,
        message: str,
        decision_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.decision_id = decision_id
        self.failure_reason = failure_reason
        super().__init__(message, details)


class HallucinationDetectedError(AIValidationError):
    """Raised when AI hallucination is detected"""

    pass


class SecurityError(RRRAlgorithmsError):
    """Base class for security errors"""

    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails"""

    pass


class AuthorizationError(SecurityError):
    """Raised when authorization check fails"""

    pass


class AuditError(SecurityError):
    """Raised when audit logging fails"""

    pass


# =============================================================================
# Backtest Errors
# =============================================================================


class BacktestError(RRRAlgorithmsError):
    """Base class for backtesting errors"""

    pass


class InvalidBacktestPeriodError(BacktestError):
    """Raised when backtest period is invalid"""

    pass


class InsufficientDataError(BacktestError):
    """Raised when insufficient data for backtest"""

    pass


# =============================================================================
# Monitoring & Observability Errors
# =============================================================================


class MonitoringError(RRRAlgorithmsError):
    """Base class for monitoring errors"""

    pass


class MetricError(MonitoringError):
    """Raised when metric collection/export fails"""

    pass


class AlertError(MonitoringError):
    """Raised when alert system fails"""

    pass


# =============================================================================
# Utility Functions
# =============================================================================


@lru_cache(maxsize=128)


def get_exception_details(exception: Exception) -> Dict[str, Any]:
    """
    Extract details from an exception for logging/debugging

    Args:
        exception: Exception instance

    Returns:
        Dictionary with exception details
    """
    details = {
        "type": type(exception).__name__,
        "message": str(exception),
        "module": exception.__class__.__module__,
    }

    # Add custom attributes if it's an RRRAlgorithmsError
    if isinstance(exception, RRRAlgorithmsError):
        if hasattr(exception, "details") and exception.details:
            details["custom_details"] = exception.details

    return details


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable

    Args:
        exception: Exception instance

    Returns:
        True if the operation should be retried
    """
    # Network/API errors are typically retryable
    retryable_types = (
        APIError,
        DatabaseConnectionError,
        ExchangeError,
    )

    # Don't retry rate limits or authentication issues
    non_retryable_types = (
        RateLimitError,
        AuthenticationError,
        AuthorizationError,
        InvalidOrderError,
        RiskLimitExceededError,
    )

    if isinstance(exception, non_retryable_types):
        return False

    if isinstance(exception, retryable_types):
        return True

    # For non-RRRAlgorithms exceptions, be conservative
    return False


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Aliases for tests and external code
RiskLimitError = RiskLimitExceededError
InputValidationError = InvalidInputError

