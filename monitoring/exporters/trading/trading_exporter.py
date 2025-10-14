#!/usr/bin/env python3
"""
Custom Trading System Metrics Exporter for Prometheus
Exposes real-time trading metrics for monitoring
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any
from datetime import datetime
import json

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    Enum,
    CollectorRegistry
)
import aiosqlite

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMetricsExporter:
    """Export trading system metrics to Prometheus."""
    
    def __init__(self, db_path: str = "/data/db/trading.db"):
        self.db_path = db_path
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info("Trading Metrics Exporter initialized")
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # System Info
        self.info = Info(
            'trading_system',
            'Trading system information',
            registry=self.registry
        )
        
        # Trading Metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders placed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.order_latency = Histogram(
            'trading_order_latency_seconds',
            'Order execution latency in seconds',
            ['exchange'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry
        )
        
        self.active_positions = Gauge(
            'trading_active_positions',
            'Number of active positions',
            ['symbol'],
            registry=self.registry
        )
        
        self.position_size = Gauge(
            'trading_position_size',
            'Current position size in USD',
            ['symbol'],
            registry=self.registry
        )
        
        self.pnl_total = Gauge(
            'trading_pnl_total',
            'Total profit and loss in USD',
            registry=self.registry
        )
        
        self.pnl_unrealized = Gauge(
            'trading_pnl_unrealized',
            'Unrealized profit and loss in USD',
            ['symbol'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'trading_portfolio_value',
            'Total portfolio value in USD',
            registry=self.registry
        )
        
        self.portfolio_drawdown = Gauge(
            'trading_portfolio_drawdown',
            'Current portfolio drawdown percentage',
            registry=self.registry
        )
        
        # Market Data Metrics
        self.market_data_received = Counter(
            'trading_market_data_received_total',
            'Total market data points received',
            ['source', 'symbol'],
            registry=self.registry
        )
        
        self.market_data_lag = Histogram(
            'trading_market_data_lag_seconds',
            'Market data lag in seconds',
            ['source'],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self.current_price = Gauge(
            'trading_current_price',
            'Current market price',
            ['symbol'],
            registry=self.registry
        )
        
        # Model Metrics
        self.model_predictions = Counter(
            'trading_model_predictions_total',
            'Total model predictions made',
            ['model', 'symbol'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'trading_model_accuracy',
            'Model prediction accuracy',
            ['model'],
            registry=self.registry
        )
        
        self.model_prediction_time = Histogram(
            'trading_model_prediction_duration_seconds',
            'Model prediction duration',
            ['model'],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        # Risk Metrics
        self.risk_exposure = Gauge(
            'trading_risk_exposure',
            'Current risk exposure in USD',
            registry=self.registry
        )
        
        self.var_95 = Gauge(
            'trading_var_95',
            'Value at Risk (95% confidence)',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'trading_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        # API Metrics
        self.api_calls = Counter(
            'trading_api_calls_total',
            'Total API calls made',
            ['api', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_rate_limit_remaining = Gauge(
            'trading_api_rate_limit_remaining',
            'API rate limit remaining',
            ['api'],
            registry=self.registry
        )
        
        self.api_rate_limit_total = Gauge(
            'trading_api_rate_limit_total',
            'API rate limit total',
            ['api'],
            registry=self.registry
        )
        
        # Database Metrics
        self.db_connections_active = Gauge(
            'trading_db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_connections_max = Gauge(
            'trading_db_connections_max',
            'Maximum database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'trading_db_query_duration_seconds',
            'Database query duration',
            ['query_type'],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        # System Health
        self.system_status = Enum(
            'trading_system_status',
            'Overall system status',
            states=['healthy', 'degraded', 'critical'],
            registry=self.registry
        )
    
    async def update_metrics(self):
        """Update all metrics from database and system state."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Update system info
                self.info.info({
                    'version': '1.0.0',
                    'environment': 'production',
                    'started_at': str(datetime.now())
                })
                
                # Update trading metrics
                await self._update_trading_metrics(db)
                
                # Update portfolio metrics
                await self._update_portfolio_metrics(db)
                
                # Update market data metrics
                await self._update_market_data_metrics(db)
                
                # Update model metrics
                await self._update_model_metrics(db)
                
                # Update risk metrics
                await self._update_risk_metrics(db)
                
                # Update system health
                await self._update_system_health(db)
                
                logger.debug("Metrics updated successfully")
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            self.system_status.state('critical')
    
    async def _update_trading_metrics(self, db):
        """Update trading-related metrics."""
        # Get recent orders
        cursor = await db.execute("""
            SELECT side, status, COUNT(*) as count
            FROM orders
            WHERE timestamp > strftime('%s', 'now', '-1 hour')
            GROUP BY side, status
        """)
        
        orders = await cursor.fetchall()
        for side, status, count in orders:
            self.orders_total.labels(
                symbol='ALL',
                side=side,
                status=status
            )._value._value = count
        
        # Get active positions
        cursor = await db.execute("""
            SELECT symbol, quantity, current_value
            FROM positions
            WHERE quantity != 0
        """)
        
        positions = await cursor.fetchall()
        total_position_value = 0
        
        for symbol, quantity, value in positions:
            self.active_positions.labels(symbol=symbol).set(1 if quantity != 0 else 0)
            self.position_size.labels(symbol=symbol).set(abs(value))
            total_position_value += value
        
        # Update total position gauge
        self.position_size.labels(symbol='TOTAL').set(total_position_value)
    
    async def _update_portfolio_metrics(self, db):
        """Update portfolio-related metrics."""
        # Get latest portfolio snapshot
        cursor = await db.execute("""
            SELECT total_value, total_pnl, unrealized_pnl, drawdown
            FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        snapshot = await cursor.fetchone()
        if snapshot:
            value, pnl, unrealized, drawdown = snapshot
            self.portfolio_value.set(value)
            self.pnl_total.set(pnl)
            self.pnl_unrealized.labels(symbol='ALL').set(unrealized)
            self.portfolio_drawdown.set(drawdown)
    
    async def _update_market_data_metrics(self, db):
        """Update market data metrics."""
        # Get market data stats
        cursor = await db.execute("""
            SELECT symbol, source, COUNT(*) as count
            FROM market_data
            WHERE timestamp > strftime('%s', 'now', '-5 minutes')
            GROUP BY symbol, source
        """)
        
        data_counts = await cursor.fetchall()
        for symbol, source, count in data_counts:
            # Approximate rate per second
            rate = count / 300  # 5 minutes = 300 seconds
            self.market_data_received.labels(
                source=source or 'unknown',
                symbol=symbol
            )._value._value = count
        
        # Get current prices
        cursor = await db.execute("""
            SELECT DISTINCT symbol, close
            FROM (
                SELECT symbol, close, 
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                FROM market_data
            )
            WHERE rn = 1
        """)
        
        prices = await cursor.fetchall()
        for symbol, price in prices:
            self.current_price.labels(symbol=symbol).set(price)
    
    async def _update_model_metrics(self, db):
        """Update ML model metrics."""
        # Get model performance stats
        cursor = await db.execute("""
            SELECT model_name, accuracy, avg_prediction_time, prediction_count
            FROM model_performance
            WHERE timestamp > strftime('%s', 'now', '-1 hour')
        """)
        
        model_stats = await cursor.fetchall()
        for model, accuracy, avg_time, count in model_stats:
            self.model_accuracy.labels(model=model).set(accuracy)
            self.model_predictions.labels(
                model=model,
                symbol='ALL'
            )._value._value = count
            
            # Update prediction time histogram
            if avg_time:
                self.model_prediction_time.labels(model=model).observe(avg_time)
    
    async def _update_risk_metrics(self, db):
        """Update risk management metrics."""
        # Get risk metrics
        cursor = await db.execute("""
            SELECT metric_name, value
            FROM risk_metrics
            WHERE timestamp > strftime('%s', 'now', '-5 minutes')
            ORDER BY timestamp DESC
        """)
        
        metrics = await cursor.fetchall()
        risk_dict = {name: value for name, value in metrics}
        
        if 'total_exposure' in risk_dict:
            self.risk_exposure.set(risk_dict['total_exposure'])
        
        if 'var_95' in risk_dict:
            self.var_95.set(risk_dict['var_95'])
        
        if 'sharpe_ratio' in risk_dict:
            self.sharpe_ratio.set(risk_dict['sharpe_ratio'])
    
    async def _update_system_health(self, db):
        """Update overall system health status."""
        # Check various health indicators
        health_score = 100
        
        # Check if we have recent data
        cursor = await db.execute("""
            SELECT COUNT(*) 
            FROM market_data 
            WHERE timestamp > strftime('%s', 'now', '-5 minutes')
        """)
        recent_data_count = (await cursor.fetchone())[0]
        
        if recent_data_count == 0:
            health_score -= 50  # No recent data is critical
        
        # Check for recent errors
        cursor = await db.execute("""
            SELECT COUNT(*) 
            FROM system_events 
            WHERE severity = 'ERROR' 
            AND timestamp > strftime('%s', 'now', '-30 minutes')
        """)
        error_count = (await cursor.fetchone())[0]
        
        if error_count > 10:
            health_score -= 30
        elif error_count > 5:
            health_score -= 15
        
        # Determine overall status
        if health_score >= 90:
            self.system_status.state('healthy')
        elif health_score >= 70:
            self.system_status.state('degraded')
        else:
            self.system_status.state('critical')
    
    async def run_forever(self, update_interval: int = 15):
        """Run the exporter forever, updating metrics periodically."""
        logger.info(f"Starting metrics update loop (interval: {update_interval}s)")
        
        while True:
            try:
                await self.update_metrics()
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                logger.info("Metrics update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(update_interval)


async def main():
    """Main entry point."""
    # Configuration
    metrics_port = int(os.getenv('METRICS_PORT', 8000))
    db_path = os.getenv('DATABASE_PATH', '/data/db/trading.db')
    update_interval = int(os.getenv('UPDATE_INTERVAL', 15))
    
    # Initialize exporter
    exporter = TradingMetricsExporter(db_path)
    
    # Start HTTP server
    start_http_server(metrics_port, registry=exporter.registry)
    logger.info(f"Metrics server started on port {metrics_port}")
    
    # Run update loop
    await exporter.run_forever(update_interval)


if __name__ == "__main__":
    asyncio.run(main())
