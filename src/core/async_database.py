from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from src.core.exceptions import DatabaseConnectionError, DataValidationError
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import sqlite3


"""
Async Database Wrapper
=====================

High-performance async database wrapper with connection pooling.
Provides async interface for database operations with batch processing.

Author: RRR Ventures
Date: 2025-10-12
"""




class AsyncDatabase:
    """
    Async database wrapper with connection pooling and batch operations.
    
    Features:
    - Connection pooling for concurrent operations
    - Batch insert/update operations
    - Async/await interface
    - Error handling and retry logic
    - Performance monitoring
    """
    
    def __init__(
        self,
        database_path: str,
        max_connections: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize async database wrapper.
        
        Args:
            database_path: Path to SQLite database file
            max_connections: Maximum number of concurrent connections
            timeout: Connection timeout in seconds
        """
        self.database_path = database_path
        self.max_connections = max_connections
        self.timeout = timeout
        
        # Connection pool
        self._pool: Optional[asyncio.Queue] = None
        self._connections: List[sqlite3.Connection] = []
        self._pool_initialized = False
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'total_errors': 0,
            'avg_query_time': 0.0,
            'max_query_time': 0.0,
            'min_query_time': float('inf'),
            'pool_utilization': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool_initialized:
            return
        
        self.logger.info(f"Initializing database pool with {self.max_connections} connections")
        
        # Create connection pool
        self._pool = asyncio.Queue(maxsize=self.max_connections)
        
        # Create connections
        for _ in range(self.max_connections):
            conn = sqlite3.connect(
                self.database_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            self._connections.append(conn)
            await self._pool.put(conn)
        
        self._pool_initialized = True
        self.logger.info("Database pool initialized successfully")
    
    async def close(self) -> None:
        """Close all connections and cleanup."""
        if not self._pool_initialized:
            return
        
        self.logger.info("Closing database pool...")
        
        # Close all connections
        for conn in self._connections:
            conn.close()
        
        self._connections.clear()
        self._pool = None
        self._pool_initialized = False
        
        self.logger.info("Database pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._pool_initialized:
            await self.initialize()
        
        conn = None
        try:
            # Get connection from pool
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=self.timeout
            )
            yield conn
        finally:
            # Return connection to pool
            if conn and self._pool:
                await self._pool.put(conn)
    
    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a database query asynchronously.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch:
                    results = [dict(row) for row in cursor.fetchall()]
                else:
                    results = None
                
                conn.commit()
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                return results
                
        except Exception as e:
            self.metrics['total_errors'] += 1
            self.logger.error(f"Query failed: {e}")
            raise DatabaseConnectionError(f"Query execution failed: {e}")
    
    async def batch_insert_market_data(
        self,
        data_batch: List[Dict[str, Any]]
    ) -> None:
        """
        Batch insert market data for better performance.
        
        Args:
            data_batch: List of market data dictionaries
        """
        if not data_batch:
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare batch data
                batch_data = []
                for data in data_batch:
                    batch_data.append((
                        data['symbol'],
                        data['timestamp'],
                        data['open'],
                        data['high'],
                        data['low'],
                        data['close'],
                        data['volume']
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    """
                    INSERT INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_data
                )
                
                conn.commit()
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                self.logger.debug(f"Batch inserted {len(data_batch)} market data records")
                
        except Exception as e:
            self.metrics['total_errors'] += 1
            self.logger.error(f"Batch insert failed: {e}")
            raise DatabaseConnectionError(f"Batch insert failed: {e}")
    
    async def batch_insert_predictions(
        self,
        predictions_batch: List[Dict[str, Any]]
    ) -> None:
        """
        Batch insert predictions for better performance.
        
        Args:
            predictions_batch: List of prediction dictionaries
        """
        if not predictions_batch:
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare batch data
                batch_data = []
                for pred in predictions_batch:
                    batch_data.append((
                        pred['symbol'],
                        pred['timestamp'],
                        pred['horizon'],
                        pred['predicted_price'],
                        pred['predicted_direction'],
                        pred['confidence'],
                        pred.get('model_version', 'unknown')
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    """
                    INSERT INTO predictions 
                    (symbol, timestamp, horizon, predicted_price, predicted_direction, confidence, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_data
                )
                
                conn.commit()
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                self.logger.debug(f"Batch inserted {len(predictions_batch)} prediction records")
                
        except Exception as e:
            self.metrics['total_errors'] += 1
            self.logger.error(f"Batch insert predictions failed: {e}")
            raise DatabaseConnectionError(f"Batch insert predictions failed: {e}")
    
    async def get_latest_market_data(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get latest market data.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of records
            
        Returns:
            List of market data records
        """
        if symbol:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (symbol, limit)
        else:
            query = """
                SELECT * FROM market_data 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (limit,)
        
        results = await self.execute_query(query, params, fetch=True)
        return results or []
    
    async def get_latest_predictions(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get latest predictions.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of records
            
        Returns:
            List of prediction records
        """
        if symbol:
            query = """
                SELECT * FROM predictions 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (symbol, limit)
        else:
            query = """
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (limit,)
        
        results = await self.execute_query(query, params, fetch=True)
        return results or []
    
    async def get_portfolio_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get latest portfolio metrics.
        
        Returns:
            Portfolio metrics dictionary or None
        """
        query = """
            SELECT 
                SUM(cash) as total_value,
                SUM(cash) as cash,
                SUM(pnl) as total_pnl,
                SUM(daily_pnl) as daily_pnl
            FROM portfolio_metrics 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        results = await self.execute_query(query, fetch=True)
        return results[0] if results else None
    
    def _update_metrics(self, query_time: float) -> None:
        """Update performance metrics."""
        self.metrics['total_queries'] += 1
        
        # Update timing metrics
        if self.metrics['total_queries'] == 1:
            self.metrics['avg_query_time'] = query_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_query_time'] = (
                alpha * query_time + 
                (1 - alpha) * self.metrics['avg_query_time']
            )
        
        self.metrics['max_query_time'] = max(self.metrics['max_query_time'], query_time)
        self.metrics['min_query_time'] = min(self.metrics['min_query_time'], query_time)
        
        # Update pool utilization
        if self._pool:
            available = self._pool.qsize()
            total = self.max_connections
            self.metrics['pool_utilization'] = (total - available) / total
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get database status."""
        return {
            'pool_initialized': self._pool_initialized,
            'max_connections': self.max_connections,
            'available_connections': self._pool.qsize() if self._pool else 0,
            'database_path': self.database_path,
            'metrics': self.metrics
        }


# Global async database instance
_async_db: Optional[AsyncDatabase] = None


async def get_async_db() -> AsyncDatabase:
    """Get the global async database instance."""
    global _async_db
    
    if _async_db is None:
        from src.core.config.loader import config_get
        
        database_path = config_get('database.path', 'data/database/trading.db')
        max_connections = config_get('database.max_connections', 10)
        
        _async_db = AsyncDatabase(
            database_path=database_path,
            max_connections=max_connections
        )
        await _async_db.initialize()
    
    return _async_db


async def close_async_db() -> None:
    """Close the global async database instance."""
    global _async_db
    
    if _async_db:
        await _async_db.close()
        _async_db = None


__all__ = [
    'AsyncDatabase',
    'get_async_db',
    'close_async_db',
]