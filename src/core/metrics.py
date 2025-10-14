from .settings import get_settings
from collections import defaultdict
from functools import lru_cache
from prometheus_client import (
from typing import Dict, Any, Optional
import logging
import threading
import time


"""
Custom Metrics Collection and Export
Provides business and technical metrics for Prometheus/Grafana monitoring
"""

    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    push_to_gateway,
    REGISTRY,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Trading Metrics
# =============================================================================

# Orders
orders_total = Counter(
    "rrr_orders_total",
    "Total number of orders",
    ["symbol", "side", "order_type", "status"],
)

orders_rejected = Counter(
    "rrr_orders_rejected",
    "Total number of rejected orders",
    ["symbol", "reason"],
)

order_fill_time = Histogram(
    "rrr_order_fill_time_seconds",
    "Time to fill orders",
    ["symbol", "order_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Positions
open_positions = Gauge(
    "rrr_open_positions",
    "Number of open positions",
    ["symbol"],
)

position_pnl = Gauge(
    "rrr_position_pnl_usd",
    "Position PnL in USD",
    ["symbol"],
)

# Portfolio
portfolio_value = Gauge(
    "rrr_portfolio_value_usd",
    "Total portfolio value in USD",
)

portfolio_cash = Gauge(
    "rrr_portfolio_cash_usd",
    "Available cash in USD",
)

portfolio_return_pct = Gauge(
    "rrr_portfolio_return_percent",
    "Portfolio return percentage",
)

# Risk Metrics
portfolio_drawdown = Gauge(
    "rrr_portfolio_drawdown_percent",
    "Current drawdown percentage",
)

sharpe_ratio = Gauge(
    "rrr_sharpe_ratio",
    "Sharpe ratio (rolling 30 days)",
)

sortino_ratio = Gauge(
    "rrr_sortino_ratio",
    "Sortino ratio (rolling 30 days)",
)

# =============================================================================
# ML Model Metrics
# =============================================================================

model_inference_time = Histogram(
    "rrr_model_inference_seconds",
    "ML model inference time",
    ["model_name", "model_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

model_predictions_total = Counter(
    "rrr_model_predictions_total",
    "Total number of model predictions",
    ["model_name", "model_type"],
)

model_cache_hits = Counter(
    "rrr_model_cache_hits",
    "Model cache hits",
    ["model_name"],
)

model_cache_misses = Counter(
    "rrr_model_cache_misses",
    "Model cache misses",
    ["model_name"],
)

model_accuracy = Gauge(
    "rrr_model_accuracy",
    "Model prediction accuracy",
    ["model_name"],
)

# =============================================================================
# Data Pipeline Metrics
# =============================================================================

data_ingestion_total = Counter(
    "rrr_data_ingestion_total",
    "Total data records ingested",
    ["source", "symbol"],
)

data_ingestion_errors = Counter(
    "rrr_data_ingestion_errors",
    "Data ingestion errors",
    ["source", "error_type"],
)

data_processing_time = Histogram(
    "rrr_data_processing_seconds",
    "Data processing time",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

api_requests_total = Counter(
    "rrr_api_requests_total",
    "Total API requests",
    ["api_name", "endpoint", "status_code"],
)

api_latency = Histogram(
    "rrr_api_latency_seconds",
    "API request latency",
    ["api_name", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# =============================================================================
# AI Validation Metrics
# =============================================================================

ai_validations_total = Counter(
    "rrr_ai_validations_total",
    "Total AI validations",
    ["decision_type", "result"],
)

ai_validation_latency = Histogram(
    "rrr_ai_validation_latency_seconds",
    "AI validation latency",
    ["decision_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)

hallucinations_detected = Counter(
    "rrr_hallucinations_detected",
    "Hallucinations detected by AI validator",
    ["severity"],
)

# =============================================================================
# Database Metrics
# =============================================================================

db_queries_total = Counter(
    "rrr_db_queries_total",
    "Total database queries",
    ["operation", "table"],
)

db_query_duration = Histogram(
    "rrr_db_query_duration_seconds",
    "Database query duration",
    ["operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

db_connections_active = Gauge(
    "rrr_db_connections_active",
    "Active database connections",
    ["pool"],
)

# =============================================================================
# System Metrics
# =============================================================================

errors_total = Counter(
    "rrr_errors_total",
    "Total errors",
    ["component", "error_type"],
)

uptime_seconds = Gauge(
    "rrr_uptime_seconds",
    "Service uptime in seconds",
    ["service"],
)


# =============================================================================
# Helper Functions
# =============================================================================


class MetricsCollector:
    """
    Helper class to collect and aggregate metrics
    """

    def __init__(self):
        self._start_time = time.time()
        self._metrics_buffer = defaultdict(list)
        self._lock = threading.Lock()

    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        fill_time: Optional[float] = None,
    ):
        """Record order metrics"""
        orders_total.labels(symbol=symbol, side=side, order_type=order_type, status=status).inc()

        if fill_time is not None:
            order_fill_time.labels(symbol=symbol, order_type=order_type).observe(fill_time)

    def record_rejected_order(self, symbol: str, reason: str):
        """Record rejected order"""
        orders_rejected.labels(symbol=symbol, reason=reason).inc()

    def update_position(self, symbol: str, quantity: float, pnl: float):
        """Update position metrics"""
        is_open = quantity != 0
        open_positions.labels(symbol=symbol).set(1 if is_open else 0)
        position_pnl.labels(symbol=symbol).set(pnl)

    def update_portfolio(
        self,
        total_value: float,
        cash: float,
        return_pct: float,
        drawdown_pct: float,
    ):
        """Update portfolio metrics"""
        portfolio_value.set(total_value)
        portfolio_cash.set(cash)
        portfolio_return_pct.set(return_pct)
        portfolio_drawdown.set(drawdown_pct)

    def update_risk_metrics(self, sharpe: float, sortino: float):
        """Update risk metrics"""
        sharpe_ratio.set(sharpe)
        sortino_ratio.set(sortino)

    def record_model_inference(
        self,
        model_name: str,
        model_type: str,
        inference_time: float,
        cache_hit: bool = False,
    ):
        """Record model inference metrics"""
        model_inference_time.labels(model_name=model_name, model_type=model_type).observe(
            inference_time
        )
        model_predictions_total.labels(model_name=model_name, model_type=model_type).inc()

        if cache_hit:
            model_cache_hits.labels(model_name=model_name).inc()
        else:
            model_cache_misses.labels(model_name=model_name).inc()

    def record_data_ingestion(
        self,
        source: str,
        symbol: str,
        count: int = 1,
        error_type: Optional[str] = None,
    ):
        """Record data ingestion metrics"""
        if error_type:
            data_ingestion_errors.labels(source=source, error_type=error_type).inc()
        else:
            data_ingestion_total.labels(source=source, symbol=symbol).inc(count)

    def record_api_request(
        self,
        api_name: str,
        endpoint: str,
        status_code: int,
        latency: float,
    ):
        """Record API request metrics"""
        api_requests_total.labels(
            api_name=api_name, endpoint=endpoint, status_code=status_code
        ).inc()
        api_latency.labels(api_name=api_name, endpoint=endpoint).observe(latency)

    def record_ai_validation(
        self,
        decision_type: str,
        result: str,
        latency: float,
        hallucination_severity: Optional[str] = None,
    ):
        """Record AI validation metrics"""
        ai_validations_total.labels(decision_type=decision_type, result=result).inc()
        ai_validation_latency.labels(decision_type=decision_type).observe(latency)

        if hallucination_severity:
            hallucinations_detected.labels(severity=hallucination_severity).inc()

    def record_db_query(
        self,
        operation: str,
        table: str,
        duration: float,
    ):
        """Record database query metrics"""
        db_queries_total.labels(operation=operation, table=table).inc()
        db_query_duration.labels(operation=operation, table=table).observe(duration)

    def record_error(self, component: str, error_type: str):
        """Record error"""
        errors_total.labels(component=component, error_type=error_type).inc()

    @lru_cache(maxsize=128)

    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self._start_time


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


@lru_cache(maxsize=128)


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance

    Returns:
        MetricsCollector singleton
    """
    global _metrics_collector
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector

