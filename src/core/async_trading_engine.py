from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from src.core.async_utils import gather_with_concurrency, create_task_safe, run_periodic
from src.core.constants import TradingConstants, MonitoringConstants
from src.core.exceptions import TradingError
from src.core.validation import MarketDataInput, PredictionRequest
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import time


"""
Async Trading Engine
===================

High-performance async trading engine supporting parallel processing.
Replaces synchronous main loop for production deployment.

Performance improvements (targeted):
- 10x throughput improvement over synchronous version
- Sub-100ms latency target
- Parallel symbol processing for multiple assets
- Non-blocking I/O for database and API calls

Author: RRR Ventures
Date: 2025-10-12
"""




class AsyncTradingEngine:
    """
    High-performance async trading engine with parallel processing.
    
    Features:
    - Parallel symbol processing (10+ symbols simultaneously)
    - Async database operations with connection pooling
    - Non-blocking ML predictions
    - Real-time monitoring and health checks
    - Graceful shutdown and error recovery
    """
    
    def __init__(
        self,
        symbols: List[str],
        data_source,
        predictor,
        db,
        monitor,
        update_interval: float = 1.0,
        max_concurrency: int = 10
    ):
        """
        Initialize async trading engine.
        
        Args:
            symbols: List of trading symbols
            data_source: Data source instance (async or sync)
            predictor: ML predictor instance (async or sync)
            db: Database instance (async or sync)
            monitor: Monitor instance
            update_interval: Update interval in seconds
            max_concurrency: Maximum concurrent symbol processing
        """
        self.symbols = symbols
        self.data_source = data_source
        self.predictor = predictor
        self.db = db
        self.monitor = monitor
        self.update_interval = update_interval
        self.max_concurrency = max_concurrency
        
        # State management
        self.running = False
        self.iteration = 0
        self.tasks: List[asyncio.Task] = []
        self.performance_metrics = {
            'total_iterations': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': float('inf'),
            'symbols_processed': 0,
            'errors_count': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    @asynccontextmanager
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the async trading engine."""
        if self.running:
            self.logger.warning("Trading engine already running")
            return
        
        self.running = True
        self.logger.info("Starting async trading engine...")
        self.monitor.log('INFO', 'Starting async trading engine...')
        
        # Start background tasks
        self.tasks = [
            create_task_safe(self._trading_loop(), name="trading_loop"),
            create_task_safe(self._monitoring_loop(), name="monitoring_loop"),
            create_task_safe(self._health_check_loop(), name="health_check"),
            create_task_safe(self._performance_tracker(), name="performance_tracker"),
        ]
        
        self.logger.info(f"Started {len(self.tasks)} background tasks")
        self.monitor.log('SUCCESS', '✓ Async trading engine started')
    
    async def stop(self):
        """Stop the async trading engine gracefully."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping async trading engine...")
        self.monitor.log('INFO', 'Stopping async trading engine...')
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some tasks did not stop gracefully")
        
        self.tasks.clear()
        self.logger.info("Async trading engine stopped")
        self.monitor.log('SUCCESS', '✓ Async trading engine stopped')
    
    async def _trading_loop(self):
        """Main trading loop - processes all symbols in parallel."""
        self.logger.info("Trading loop started")
        
        while self.running:
            iteration_start = time.time()
            self.iteration += 1
            
            try:
                # Fetch market data for all symbols (parallel)
                market_data = await self._fetch_all_market_data()
                
                if not market_data:
                    self.logger.warning("No market data received, skipping iteration")
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # Process all symbols in parallel with concurrency limit
                await self._process_symbols_parallel(market_data)
                
                # Calculate iteration time
                iteration_time = time.time() - iteration_start
                latency_ms = iteration_time * 1000
                
                # Update performance metrics
                self._update_performance_metrics(latency_ms, len(market_data))
                
                # Log performance (every 10 iterations)
                if self.iteration % 10 == 0:
                    self.logger.info(
                        f"Iteration {self.iteration}: {latency_ms:.1f}ms "
                        f"({len(market_data)} symbols, "
                        f"avg: {self.performance_metrics['avg_latency_ms']:.1f}ms)"
                    )
                    self.monitor.log(
                        'INFO',
                        f'Iteration {self.iteration}: {latency_ms:.1f}ms '
                        f'({len(market_data)} symbols)'
                    )
                
                # Check if we met performance target
                if latency_ms > MonitoringConstants.TARGET_SIGNAL_LATENCY_MS:
                    self.logger.warning(
                        f"Slow iteration: {latency_ms:.1f}ms "
                        f"(target: <{MonitoringConstants.TARGET_SIGNAL_LATENCY_MS}ms)"
                    )
                    self.monitor.log(
                        'WARNING',
                        f'Slow iteration: {latency_ms:.1f}ms '
                        f'(target: <{MonitoringConstants.TARGET_SIGNAL_LATENCY_MS}ms)'
                    )
                
                # Sleep until next update
                sleep_time = max(0, self.update_interval - iteration_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Trading loop iteration failed: {e}")
                self.monitor.log('ERROR', f'Trading loop iteration failed: {e}')
                self.performance_metrics['errors_count'] += 1
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _fetch_all_market_data(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch market data for all symbols in parallel.
        
        Returns:
            Dictionary mapping symbol to OHLCV data
        """
        # Create tasks for each symbol
        tasks = [
            self._fetch_symbol_data(symbol)
            for symbol in self.symbols
        ]
        
        # Execute in parallel with concurrency limit
        results = await gather_with_concurrency(
            self.max_concurrency, 
            *tasks
        )
        
        # Build dictionary (filter out None results)
        market_data = {
            symbol: data
            for symbol, data in zip(self.symbols, results)
            if data is not None
        }
        
        return market_data
    
    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch data for single symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OHLCV data or None if error
        """
        try:
            # Check if data source is async
            if hasattr(self.data_source, 'get_latest_data_async'):
                data = await self.data_source.get_latest_data_async([symbol])
            else:
                # Wrap sync call in executor
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    self.data_source.get_latest_data,
                    [symbol]
                )
            
            return data.get(symbol) if data else None
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol}: {e}")
            return None
    
    async def _process_symbols_parallel(self, market_data: Dict[str, Dict[str, float]]):
        """
        Process all symbols in parallel with concurrency control.
        
        Args:
            market_data: Dictionary of symbol -> OHLCV
        """
        # Create tasks for each symbol
        tasks = [
            self._process_single_symbol(symbol, ohlcv)
            for symbol, ohlcv in market_data.items()
        ]
        
        # Execute all in parallel with concurrency limit
        await gather_with_concurrency(
            self.max_concurrency,
            *tasks
        )
    
    async def _process_single_symbol(self, symbol: str, ohlcv: Dict[str, float]):
        """
        Process single symbol: predict + store.
        
        Args:
            symbol: Trading symbol
            ohlcv: OHLCV data
        """
        try:
            current_price = ohlcv['close']
            timestamp = ohlcv['timestamp']
            
            # Validate market data
            try:
                MarketDataInput(
                    symbol=symbol,
                    timestamp=timestamp,
                    ohlcv=ohlcv
                )
            except Exception as e:
                self.logger.warning(f"Invalid market data for {symbol}: {e}")
                return
            
            # Generate prediction (async)
            prediction = await self._generate_prediction_async(symbol, current_price)
            
            # Store data and prediction (async)
            await asyncio.gather(
                self._store_market_data_async(symbol, timestamp, ohlcv),
                self._store_prediction_async(prediction),
                return_exceptions=True
            )
            
            # Update symbols processed count
            self.performance_metrics['symbols_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process {symbol}: {e}")
            self.performance_metrics['errors_count'] += 1
    
    async def _generate_prediction_async(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate prediction asynchronously.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            Prediction dictionary
        """
        try:
            # Check if predictor is async
            if hasattr(self.predictor, 'predict_async'):
                prediction = await self.predictor.predict_async(symbol, current_price)
            else:
                # Run prediction in executor (sync to async)
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None,
                    self.predictor.predict,
                    symbol,
                    current_price
                )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            # Return a default prediction to keep system running
            return {
                'symbol': symbol,
                'predicted_price': current_price,
                'direction': 'neutral',
                'confidence': 0.5,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    async def _store_market_data_async(
        self,
        symbol: str,
        timestamp: float,
        ohlcv: Dict[str, float]
    ):
        """Store market data asynchronously."""
        try:
            # Check if database is async
            if hasattr(self.db, 'insert_market_data_async'):
                await self.db.insert_market_data_async(symbol, timestamp, ohlcv)
            else:
                # Run in executor (sync to async)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.db.insert_market_data,
                    symbol,
                    timestamp,
                    ohlcv
                )
        except Exception as e:
            self.logger.error(f"Failed to store market data for {symbol}: {e}")
    
    async def _store_prediction_async(self, prediction: Dict[str, Any]):
        """Store prediction asynchronously."""
        try:
            # Check if database is async
            if hasattr(self.db, 'insert_prediction_async'):
                await self.db.insert_prediction_async(prediction)
            else:
                # Run in executor (sync to async)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.db.insert_prediction,
                    prediction
                )
        except Exception as e:
            self.logger.error(f"Failed to store prediction: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        await run_periodic(
            5.0,  # Every 5 seconds
            self._update_monitoring
        )
    
    async def _update_monitoring(self):
        """Update monitoring metrics."""
        try:
            # Get latest portfolio metrics
            loop = asyncio.get_event_loop()
            
            # Check if database has async method
            if hasattr(self.db, 'get_latest_portfolio_metrics_async'):
                metrics = await self.db.get_latest_portfolio_metrics_async()
            else:
                metrics = await loop.run_in_executor(
                    None,
                    self.db.get_latest_portfolio_metrics
                )
            
            if metrics:
                self.monitor.update_portfolio(
                    value=metrics['total_value'],
                    cash=metrics['cash'],
                    pnl=metrics['total_pnl'],
                    daily_pnl=metrics['daily_pnl']
                )
                
        except Exception as e:
            self.logger.error(f"Monitoring update failed: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        await run_periodic(
            30.0,  # Every 30 seconds
            self._perform_health_check
        )
    
    async def _perform_health_check(self):
        """Perform system health check."""
        health = {
            'iteration': self.iteration,
            'running': self.running,
            'symbols': len(self.symbols),
            'tasks': len([t for t in self.tasks if not t.done()]),
            'performance': self.performance_metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.debug(f"Health check: {health}")
        
        # Check for performance issues
        if self.performance_metrics['avg_latency_ms'] > 1000:
            self.logger.warning("High average latency detected")
            self.monitor.log('WARNING', 'High average latency detected')
    
    async def _performance_tracker(self):
        """Background performance tracking."""
        await run_periodic(
            60.0,  # Every 60 seconds
            self._log_performance_summary
        )
    
    async def _log_performance_summary(self):
        """Log performance summary."""
        metrics = self.performance_metrics
        
        self.logger.info(
            f"Performance Summary - "
            f"Iterations: {metrics['total_iterations']}, "
            f"Avg Latency: {metrics['avg_latency_ms']:.1f}ms, "
            f"Symbols Processed: {metrics['symbols_processed']}, "
            f"Errors: {metrics['errors_count']}"
        )
    
    def _update_performance_metrics(self, latency_ms: float, symbols_count: int):
        """Update performance metrics."""
        metrics = self.performance_metrics
        
        # Update iteration count
        metrics['total_iterations'] += 1
        
        # Update latency metrics
        if metrics['total_iterations'] == 1:
            metrics['avg_latency_ms'] = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            metrics['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * metrics['avg_latency_ms']
            )
        
        # Update min/max latency
        metrics['min_latency_ms'] = min(metrics['min_latency_ms'], latency_ms)
        metrics['max_latency_ms'] = max(metrics['max_latency_ms'], latency_ms)
    
    @lru_cache(maxsize=128)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.running,
            'iteration': self.iteration,
            'symbols': self.symbols,
            'update_interval': self.update_interval,
            'max_concurrency': self.max_concurrency,
            'active_tasks': len([t for t in self.tasks if not t.done()]),
            'performance': self.performance_metrics
        }


async def run_async_trading_system(
    symbols: List[str],
    data_source,
    predictor,
    db,
    monitor,
    update_interval: float = 1.0,
    max_concurrency: int = 10
):
    """
    Run async trading system.
    
    This is the async entry point for the trading system.
    
    Args:
        symbols: List of trading symbols
        data_source: Data source instance
        predictor: ML predictor instance
        db: Database instance
        monitor: Monitor instance
        update_interval: Update interval in seconds
        max_concurrency: Maximum concurrent symbol processing
        
    Example:
        # Run async trading system
        await run_async_trading_system(
            symbols=['BTC-USD', 'ETH-USD'],
            data_source=data_source,
            predictor=predictor,
            db=db,
            monitor=monitor,
            max_concurrency=10
        )
    """
    async with AsyncTradingEngine(
        symbols=symbols,
        data_source=data_source,
        predictor=predictor,
        db=db,
        monitor=monitor,
        update_interval=update_interval,
        max_concurrency=max_concurrency
    ) as engine:
        # Engine will run until cancelled
        await asyncio.Event().wait()


__all__ = [
    'AsyncTradingEngine',
    'run_async_trading_system',
]