from asyncpg import Pool, Connection
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from src.core.exceptions import DatabaseConnectionError, DataValidationError
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import asyncpg
import json
import logging


"""
Async PostgreSQL Database
========================

High-performance async PostgreSQL database with connection pooling,
time-series optimization, and advanced querying capabilities.

Author: RRR Ventures
Date: 2025-10-12
"""




class AsyncPostgreSQL:
    """
    Async PostgreSQL database with advanced features.
    
    Features:
    - Connection pooling
    - Time-series optimization
    - Batch operations
    - Transaction support
    - Performance monitoring
    - Automatic reconnection
    """
    
    def __init__(
        self,
        database_url: str,
        min_connections: int = 5,
        max_connections: int = 20,
        command_timeout: float = 30.0,
        server_settings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize async PostgreSQL database.
        
        Args:
            database_url: PostgreSQL connection URL
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            command_timeout: Command timeout in seconds
            server_settings: Additional server settings
        """
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.server_settings = server_settings or {}
        
        # Connection pool
        self.pool: Optional[Pool] = None
        self.initialized = False
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_query_time': 0.0,
            'max_query_time': 0.0,
            'min_query_time': float('inf'),
            'pool_size': 0,
            'active_connections': 0,
            'idle_connections': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        if self.initialized:
            return
        
        try:
            self.logger.info("Initializing PostgreSQL connection pool...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.command_timeout,
                server_settings=self.server_settings
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            self.initialized = True
            self.logger.info("PostgreSQL connection pool initialized successfully")
            
            # Create tables if they don't exist
            await self._create_tables()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise DatabaseConnectionError(f"PostgreSQL initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            self.logger.info("Closing PostgreSQL connection pool...")
            await self.pool.close()
            self.pool = None
            self.initialized = False
            self.logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.initialized:
            await self.initialize()
        
        if not self.pool:
            raise DatabaseConnectionError("Database pool not initialized")
        
        conn = None
        try:
            conn = await self.pool.acquire()
            yield conn
        finally:
            if conn:
                await self.pool.release(conn)
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            async with self.get_connection() as conn:
                # Market data table (time-series optimized)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        open DECIMAL(20, 8) NOT NULL,
                        high DECIMAL(20, 8) NOT NULL,
                        low DECIMAL(20, 8) NOT NULL,
                        close DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        source VARCHAR(50) DEFAULT 'websocket',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(symbol, timestamp)
                    )
                """)
                
                # Create time-series index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                    ON market_data (symbol, timestamp DESC)
                """)
                
                # Create time-based partition index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
                    ON market_data (timestamp DESC)
                """)
                
                # Predictions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        horizon INTEGER NOT NULL,
                        predicted_price DECIMAL(20, 8) NOT NULL,
                        predicted_direction VARCHAR(20) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        model_version VARCHAR(50) NOT NULL,
                        features_used JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create prediction indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time 
                    ON predictions (symbol, timestamp DESC)
                """)
                
                # Portfolio metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        total_value DECIMAL(20, 8) NOT NULL,
                        cash DECIMAL(20, 8) NOT NULL,
                        total_pnl DECIMAL(20, 8) NOT NULL,
                        daily_pnl DECIMAL(20, 8) NOT NULL,
                        positions JSONB,
                        risk_metrics JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create portfolio metrics index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp 
                    ON portfolio_metrics (timestamp DESC)
                """)
                
                # Trades table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        total_value DECIMAL(20, 8) NOT NULL,
                        fees DECIMAL(20, 8) DEFAULT 0,
                        order_id VARCHAR(100),
                        status VARCHAR(20) DEFAULT 'filled',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create trades indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
                    ON trades (symbol, timestamp DESC)
                """)
                
                self.logger.info("Database tables created successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise DatabaseConnectionError(f"Table creation failed: {e}")
    
    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = False,
        fetch_one: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Execute a database query.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            fetch_one: Whether to fetch only one result
            
        Returns:
            Query results or None
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                if params:
                    if fetch_one:
                        result = await conn.fetchrow(query, *params)
                        if result:
                            return dict(result)
                        return None
                    elif fetch:
                        rows = await conn.fetch(query, *params)
                        return [dict(row) for row in rows]
                    else:
                        await conn.execute(query, *params)
                else:
                    if fetch_one:
                        result = await conn.fetchrow(query)
                        if result:
                            return dict(result)
                        return None
                    elif fetch:
                        rows = await conn.fetch(query)
                        return [dict(row) for row in rows]
                    else:
                        await conn.execute(query)
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                return None
                
        except Exception as e:
            self.metrics['failed_queries'] += 1
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
                        data['volume'],
                        data.get('source', 'websocket')
                    ))
                
                # Execute batch insert with ON CONFLICT handling
                await conn.executemany("""
                    INSERT INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source
                """, batch_data)
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                self.logger.debug(f"Batch inserted {len(data_batch)} market data records")
                
        except Exception as e:
            self.metrics['failed_queries'] += 1
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
                        pred.get('model_version', 'unknown'),
                        json.dumps(pred.get('features_used', []))
                    ))
                
                # Execute batch insert
                await conn.executemany("""
                    INSERT INTO predictions 
                    (symbol, timestamp, horizon, predicted_price, predicted_direction, 
                     confidence, model_version, features_used)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, batch_data)
                
                # Update metrics
                self._update_metrics(asyncio.get_event_loop().time() - start_time)
                
                self.logger.debug(f"Batch inserted {len(predictions_batch)} prediction records")
                
        except Exception as e:
            self.metrics['failed_queries'] += 1
            self.logger.error(f"Batch insert predictions failed: {e}")
            raise DatabaseConnectionError(f"Batch insert predictions failed: {e}")
    
    async def get_latest_market_data(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get latest market data.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of records
            hours_back: Hours of data to retrieve
            
        Returns:
            List of market data records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        if symbol:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = $1 AND timestamp >= $2
                ORDER BY timestamp DESC 
                LIMIT $3
            """
            params = (symbol, cutoff_time, limit)
        else:
            query = """
                SELECT * FROM market_data 
                WHERE timestamp >= $1
                ORDER BY timestamp DESC 
                LIMIT $2
            """
            params = (cutoff_time, limit)
        
        results = await self.execute_query(query, params, fetch=True)
        return results or []
    
    async def get_latest_predictions(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get latest predictions.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of records
            hours_back: Hours of data to retrieve
            
        Returns:
            List of prediction records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        if symbol:
            query = """
                SELECT * FROM predictions 
                WHERE symbol = $1 AND timestamp >= $2
                ORDER BY timestamp DESC 
                LIMIT $3
            """
            params = (symbol, cutoff_time, limit)
        else:
            query = """
                SELECT * FROM predictions 
                WHERE timestamp >= $2
                ORDER BY timestamp DESC 
                LIMIT $3
            """
            params = (cutoff_time, limit)
        
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
                total_value,
                cash,
                total_pnl,
                daily_pnl,
                positions,
                risk_metrics
            FROM portfolio_metrics 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        result = await self.execute_query(query, fetch_one=True)
        return result
    
    async def get_performance_analytics(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance analytics.
        
        Args:
            symbol: Optional symbol filter
            days_back: Days of data to analyze
            
        Returns:
            Performance analytics dictionary
        """
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        if symbol:
            query = """
                SELECT 
                    symbol,
                    COUNT(*) as total_records,
                    AVG(close) as avg_price,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    STDDEV(close) as price_volatility,
                    AVG(volume) as avg_volume,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp
                FROM market_data 
                WHERE symbol = $1 AND timestamp >= $2
                GROUP BY symbol
            """
            params = (symbol, cutoff_time)
        else:
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    AVG(close) as avg_price,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    STDDEV(close) as price_volatility,
                    AVG(volume) as avg_volume,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp
                FROM market_data 
                WHERE timestamp >= $1
            """
            params = (cutoff_time,)
        
        result = await self.execute_query(query, params, fetch_one=True)
        return result or {}
    
    def _update_metrics(self, query_time: float) -> None:
        """Update performance metrics."""
        self.metrics['total_queries'] += 1
        self.metrics['successful_queries'] += 1
        
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
        
        # Update pool metrics
        if self.pool:
            self.metrics['pool_size'] = self.pool.get_size()
            self.metrics['active_connections'] = self.pool.get_idle_size()
            self.metrics['idle_connections'] = self.pool.get_idle_size()
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get database status."""
        return {
            'initialized': self.initialized,
            'pool_size': self.pool.get_size() if self.pool else 0,
            'active_connections': self.pool.get_idle_size() if self.pool else 0,
            'database_url': self.database_url,
            'metrics': self.metrics
        }


# Global async PostgreSQL instance
_async_postgres: Optional[AsyncPostgreSQL] = None


async def get_async_postgres() -> AsyncPostgreSQL:
    """Get the global async PostgreSQL instance."""
    global _async_postgres
    
    if _async_postgres is None:
        from src.core.config.loader import config_get
        
        database_url = config_get('database.postgresql.url', 'postgresql://user:password@localhost:5432/trading')
        
        _async_postgres = AsyncPostgreSQL(
            database_url=database_url,
            min_connections=5,
            max_connections=20
        )
        await _async_postgres.initialize()
    
    return _async_postgres


async def close_async_postgres() -> None:
    """Close the global async PostgreSQL instance."""
    global _async_postgres
    
    if _async_postgres:
        await _async_postgres.close()
        _async_postgres = None


__all__ = [
    'AsyncPostgreSQL',
    'get_async_postgres',
    'close_async_postgres',
]