from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import FastAPI, Response
from functools import lru_cache
from prometheus_client import (
from src.core.exceptions import MonitoringError
from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import logging
import time


"""
Prometheus Metrics Collection
============================

Comprehensive metrics collection for RRRalgorithms trading system.
Provides detailed observability and monitoring capabilities.

Author: RRR Ventures
Date: 2025-10-12
"""

    Counter, Histogram, Gauge, Summary, Info,
    start_http_server, generate_latest, CONTENT_TYPE_LATEST
)



@dataclass
class MetricConfig:
    """Metric configuration."""
    name: str
    description: str
    labels: List[str] = None
    metric_type: str = "counter"  # counter, gauge, histogram, summary


class PrometheusMetrics:
    """
    Prometheus metrics collector for trading system.
    
    Features:
    - Trading-specific metrics
    - Performance monitoring
    - Error tracking
    - Custom business metrics
    - Real-time dashboards
    """
    
    def __init__(self, port: int = 9090):
        """
        Initialize Prometheus metrics.
        
        Args:
            port: Metrics server port
        """
        self.port = port
        self.metrics: Dict[str, Any] = {}
        self.collectors: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup all Prometheus metrics."""
        
        # Trading Metrics
        self.metrics['trades_total'] = Counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'status']
        )
        
        self.metrics['orders_total'] = Counter(
            'orders_total',
            'Total number of orders placed',
            ['symbol', 'side', 'type', 'status']
        )
        
        self.metrics['order_value_total'] = Counter(
            'order_value_total',
            'Total value of orders',
            ['symbol', 'side']
        )
        
        self.metrics['position_pnl'] = Gauge(
            'position_pnl',
            'Current position PnL',
            ['symbol']
        )
        
        self.metrics['position_quantity'] = Gauge(
            'position_quantity',
            'Current position quantity',
            ['symbol']
        )
        
        self.metrics['portfolio_value'] = Gauge(
            'portfolio_value',
            'Total portfolio value'
        )
        
        # Data Pipeline Metrics
        self.metrics['market_data_points'] = Counter(
            'market_data_points_total',
            'Total market data points processed',
            ['symbol', 'source']
        )
        
        self.metrics['data_processing_duration'] = Histogram(
            'data_processing_duration_seconds',
            'Time spent processing market data',
            ['symbol', 'source'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.metrics['websocket_connections'] = Gauge(
            'websocket_connections_active',
            'Number of active WebSocket connections'
        )
        
        self.metrics['websocket_reconnects'] = Counter(
            'websocket_reconnects_total',
            'Total WebSocket reconnections',
            ['exchange']
        )
        
        # ML Metrics
        self.metrics['predictions_total'] = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['symbol', 'model_version', 'status']
        )
        
        self.metrics['prediction_duration'] = Histogram(
            'prediction_duration_seconds',
            'Time spent making predictions',
            ['symbol', 'model_version'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.metrics['prediction_accuracy'] = Gauge(
            'prediction_accuracy',
            'Prediction accuracy percentage',
            ['symbol', 'model_version']
        )
        
        self.metrics['model_confidence'] = Histogram(
            'model_confidence',
            'Model prediction confidence',
            ['symbol', 'model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Database Metrics
        self.metrics['database_queries'] = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table', 'status']
        )
        
        self.metrics['database_query_duration'] = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.metrics['database_connections'] = Gauge(
            'database_connections_active',
            'Number of active database connections'
        )
        
        # Cache Metrics
        self.metrics['cache_operations'] = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_type', 'status']
        )
        
        self.metrics['cache_hit_ratio'] = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache_type']
        )
        
        self.metrics['cache_size'] = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['cache_type']
        )
        
        # System Metrics
        self.metrics['api_requests'] = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.metrics['api_request_duration'] = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.metrics['active_connections'] = Gauge(
            'active_connections',
            'Number of active connections',
            ['service']
        )
        
        # Error Metrics
        self.metrics['errors_total'] = Counter(
            'errors_total',
            'Total number of errors',
            ['service', 'error_type', 'severity']
        )
        
        self.metrics['error_rate'] = Gauge(
            'error_rate',
            'Error rate percentage',
            ['service']
        )
        
        # Business Metrics
        self.metrics['profit_loss'] = Gauge(
            'profit_loss_total',
            'Total profit/loss',
            ['type']  # realized, unrealized
        )
        
        self.metrics['trading_volume'] = Counter(
            'trading_volume_total',
            'Total trading volume',
            ['symbol']
        )
        
        self.metrics['risk_metrics'] = Gauge(
            'risk_metrics',
            'Risk management metrics',
            ['metric_type']  # var, max_drawdown, sharpe_ratio
        )
        
        # Custom Info Metric
        self.metrics['system_info'] = Info(
            'system_info',
            'System information'
        )
        
        # Set system info
        self.metrics['system_info'].info({
            'version': '1.0.0',
            'environment': 'production',
            'start_time': datetime.now().isoformat()
        })
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        status: str,
        value: float = 0.0
    ) -> None:
        """Record trade metrics."""
        self.metrics['trades_total'].labels(
            symbol=symbol,
            side=side,
            status=status
        ).inc()
        
        if value > 0:
            self.metrics['order_value_total'].labels(
                symbol=symbol,
                side=side
            ).inc(value)
    
    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str
    ) -> None:
        """Record order metrics."""
        self.metrics['orders_total'].labels(
            symbol=symbol,
            side=side,
            type=order_type,
            status=status
        ).inc()
    
    def record_position(
        self,
        symbol: str,
        quantity: float,
        pnl: float
    ) -> None:
        """Record position metrics."""
        self.metrics['position_quantity'].labels(symbol=symbol).set(quantity)
        self.metrics['position_pnl'].labels(symbol=symbol).set(pnl)
    
    def record_portfolio_value(self, value: float) -> None:
        """Record portfolio value."""
        self.metrics['portfolio_value'].set(value)
    
    def record_market_data(
        self,
        symbol: str,
        source: str,
        processing_time: float
    ) -> None:
        """Record market data processing metrics."""
        self.metrics['market_data_points'].labels(
            symbol=symbol,
            source=source
        ).inc()
        
        self.metrics['data_processing_duration'].labels(
            symbol=symbol,
            source=source
        ).observe(processing_time)
    
    def record_websocket_connection(self, active: int) -> None:
        """Record WebSocket connection metrics."""
        self.metrics['websocket_connections'].set(active)
    
    def record_websocket_reconnect(self, exchange: str) -> None:
        """Record WebSocket reconnection."""
        self.metrics['websocket_reconnects'].labels(exchange=exchange).inc()
    
    def record_prediction(
        self,
        symbol: str,
        model_version: str,
        duration: float,
        confidence: float,
        status: str = "success"
    ) -> None:
        """Record prediction metrics."""
        self.metrics['predictions_total'].labels(
            symbol=symbol,
            model_version=model_version,
            status=status
        ).inc()
        
        self.metrics['prediction_duration'].labels(
            symbol=symbol,
            model_version=model_version
        ).observe(duration)
        
        self.metrics['model_confidence'].labels(
            symbol=symbol,
            model_version=model_version
        ).observe(confidence)
    
    def record_database_query(
        self,
        operation: str,
        table: str,
        duration: float,
        status: str = "success"
    ) -> None:
        """Record database query metrics."""
        self.metrics['database_queries'].labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        
        self.metrics['database_query_duration'].labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def record_database_connections(self, active: int) -> None:
        """Record database connection metrics."""
        self.metrics['database_connections'].set(active)
    
    def record_cache_operation(
        self,
        operation: str,
        cache_type: str,
        status: str = "success"
    ) -> None:
        """Record cache operation metrics."""
        self.metrics['cache_operations'].labels(
            operation=operation,
            cache_type=cache_type,
            status=status
        ).inc()
    
    def record_cache_hit_ratio(self, cache_type: str, ratio: float) -> None:
        """Record cache hit ratio."""
        self.metrics['cache_hit_ratio'].labels(cache_type=cache_type).set(ratio)
    
    def record_cache_size(self, cache_type: str, size: int) -> None:
        """Record cache size."""
        self.metrics['cache_size'].labels(cache_type=cache_type).set(size)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        duration: float,
        status_code: int
    ) -> None:
        """Record API request metrics."""
        self.metrics['api_requests'].labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.metrics['api_request_duration'].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_active_connections(self, service: str, count: int) -> None:
        """Record active connections."""
        self.metrics['active_connections'].labels(service=service).set(count)
    
    def record_error(
        self,
        service: str,
        error_type: str,
        severity: str = "error"
    ) -> None:
        """Record error metrics."""
        self.metrics['errors_total'].labels(
            service=service,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_error_rate(self, service: str, rate: float) -> None:
        """Record error rate."""
        self.metrics['error_rate'].labels(service=service).set(rate)
    
    def record_profit_loss(self, pnl_type: str, value: float) -> None:
        """Record profit/loss metrics."""
        self.metrics['profit_loss'].labels(type=pnl_type).set(value)
    
    def record_trading_volume(self, symbol: str, volume: float) -> None:
        """Record trading volume."""
        self.metrics['trading_volume'].labels(symbol=symbol).inc(volume)
    
    def record_risk_metric(self, metric_type: str, value: float) -> None:
        """Record risk management metrics."""
        self.metrics['risk_metrics'].labels(metric_type=metric_type).set(value)
    
    def add_collector(self, collector: Callable) -> None:
        """Add custom metric collector."""
        self.collectors.append(collector)
    
    async def collect_custom_metrics(self) -> None:
        """Collect custom metrics from registered collectors."""
        for collector in self.collectors:
            try:
                await collector(self)
            except Exception as e:
                self.logger.error(f"Custom metric collection failed: {e}")
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest().decode('utf-8')
    
    def start_server(self) -> None:
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.port)
            self.logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            raise MonitoringError(f"Metrics server startup failed: {e}")
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get metrics collector status."""
        return {
            'port': self.port,
            'metrics_count': len(self.metrics),
            'collectors_count': len(self.collectors),
            'status': 'running'
        }


class MetricsCollector:
    """Metrics collector for automatic metric gathering."""
    
    def __init__(self, metrics: PrometheusMetrics):
        """
        Initialize metrics collector.
        
        Args:
            metrics: Prometheus metrics instance
        """
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
    
    async def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # This would collect actual system metrics
            # For now, we'll simulate some metrics
            
            # Simulate active connections
            self.metrics.record_active_connections('data-service', 5)
            self.metrics.record_active_connections('ml-service', 3)
            self.metrics.record_active_connections('trading-service', 2)
            
            # Simulate database connections
            self.metrics.record_database_connections(10)
            
            # Simulate cache metrics
            self.metrics.record_cache_hit_ratio('memory', 0.95)
            self.metrics.record_cache_hit_ratio('redis', 0.88)
            self.metrics.record_cache_size('memory', 1024 * 1024)  # 1MB
            self.metrics.record_cache_size('redis', 10 * 1024 * 1024)  # 10MB
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    async def collect_trading_metrics(self) -> None:
        """Collect trading-specific metrics."""
        try:
            # This would collect actual trading metrics
            # For now, we'll simulate some metrics
            
            # Simulate portfolio value
            self.metrics.record_portfolio_value(100000.0)
            
            # Simulate profit/loss
            self.metrics.record_profit_loss('realized', 5000.0)
            self.metrics.record_profit_loss('unrealized', 2500.0)
            
            # Simulate risk metrics
            self.metrics.record_risk_metric('var', 0.05)
            self.metrics.record_risk_metric('max_drawdown', 0.02)
            self.metrics.record_risk_metric('sharpe_ratio', 1.5)
            
        except Exception as e:
            self.logger.error(f"Trading metrics collection failed: {e}")


# Global metrics instance
_metrics: Optional[PrometheusMetrics] = None


@lru_cache(maxsize=128)


def get_metrics() -> PrometheusMetrics:
    """Get the global metrics instance."""
    global _metrics
    
    if _metrics is None:
        _metrics = PrometheusMetrics()
    
    return _metrics


def start_metrics_server() -> None:
    """Start the global metrics server."""
    metrics = get_metrics()
    metrics.start_server()
    
    # Add collectors
    collector = MetricsCollector(metrics)
    metrics.add_collector(collector.collect_system_metrics)
    metrics.add_collector(collector.collect_trading_metrics)


__all__ = [
    'PrometheusMetrics',
    'MetricsCollector',
    'get_metrics',
    'start_metrics_server',
]