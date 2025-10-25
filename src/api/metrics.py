"""
Prometheus Metrics for API
Exports detailed metrics for monitoring and observability
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable

# ============================================================================
# API Metrics
# ============================================================================

# Request counters
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Database metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table']
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections'
)

db_connections_idle = Gauge(
    'db_connections_idle',
    'Idle database connections'
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_key_prefix']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_key_prefix']
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Cache size in bytes'
)

# Rate limiting metrics
rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['client_ip']
)

# Authentication metrics
auth_attempts_total = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['status']
)

auth_token_validations_total = Counter(
    'auth_token_validations_total',
    'Total token validations',
    ['status']
)

# Trading metrics
trades_executed_total = Counter(
    'trades_executed_total',
    'Total trades executed',
    ['symbol', 'side']
)

portfolio_equity = Gauge(
    'portfolio_equity',
    'Current portfolio equity'
)

portfolio_pnl = Gauge(
    'portfolio_pnl',
    'Current portfolio P&L'
)

active_positions = Gauge(
    'active_positions',
    'Number of active positions'
)

# AI/ML metrics
ai_predictions_total = Counter(
    'ai_predictions_total',
    'Total AI predictions',
    ['model_name', 'outcome']
)

ai_prediction_confidence = Histogram(
    'ai_prediction_confidence',
    'AI prediction confidence scores',
    ['model_name'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# System metrics
api_info = Info('api_info', 'API version and information')
api_info.info({
    'version': '1.0.0',
    'name': 'RRRalgorithms Transparency API',
    'environment': 'production'
})


# ============================================================================
# Metric Decorators
# ============================================================================

def track_request_metrics(endpoint: str):
    """
    Decorator to track request metrics

    Usage:
        @track_request_metrics("portfolio")
        async def get_portfolio():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                api_request_duration_seconds.labels(
                    method="GET",
                    endpoint=endpoint
                ).observe(duration)
                api_requests_total.labels(
                    method="GET",
                    endpoint=endpoint,
                    status_code=status_code
                ).inc()

        return wrapper
    return decorator


def track_db_query(operation: str, table: str):
    """
    Decorator to track database query metrics

    Usage:
        @track_db_query("select", "trades")
        async def get_trades():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                db_query_duration_seconds.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                db_queries_total.labels(
                    operation=operation,
                    table=table
                ).inc()

        return wrapper
    return decorator


# ============================================================================
# Metric Update Functions
# ============================================================================

def record_cache_hit(key_prefix: str):
    """Record a cache hit"""
    cache_hits_total.labels(cache_key_prefix=key_prefix).inc()


def record_cache_miss(key_prefix: str):
    """Record a cache miss"""
    cache_misses_total.labels(cache_key_prefix=key_prefix).inc()


def record_rate_limit_violation(client_ip: str):
    """Record a rate limit violation"""
    rate_limit_exceeded_total.labels(client_ip=client_ip).inc()


def record_auth_attempt(success: bool):
    """Record an authentication attempt"""
    status = "success" if success else "failure"
    auth_attempts_total.labels(status=status).inc()


def record_token_validation(success: bool):
    """Record a token validation"""
    status = "success" if success else "failure"
    auth_token_validations_total.labels(status=status).inc()


def update_portfolio_metrics(equity: float, pnl: float, positions: int):
    """Update portfolio metrics"""
    portfolio_equity.set(equity)
    portfolio_pnl.set(pnl)
    active_positions.set(positions)


def record_trade(symbol: str, side: str):
    """Record a trade execution"""
    trades_executed_total.labels(symbol=symbol, side=side).inc()


def record_ai_prediction(model_name: str, confidence: float, outcome: str):
    """Record an AI prediction"""
    ai_predictions_total.labels(model_name=model_name, outcome=outcome).inc()
    ai_prediction_confidence.labels(model_name=model_name).observe(confidence)


def update_db_connection_metrics(active: int, idle: int):
    """Update database connection metrics"""
    db_connections_active.set(active)
    db_connections_idle.set(idle)
