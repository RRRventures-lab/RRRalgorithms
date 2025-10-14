from asyncpg.pool import Pool
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
import asyncpg
import json
import logging
import numpy as np
import os
import pandas as pd


"""
Database writer for Polygon.io WebSocket data.
Handles batch inserts, data validation, and performance optimization.
"""


logger = logging.getLogger(__name__)

class PolygonDatabaseWriter:
    """
    Async database writer for Polygon.io market data.
    Uses PostgreSQL with connection pooling and batch inserts.
    """

    def __init__(self, database_url: str = None):
        """Initialize database writer."""
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://localhost/trading_data')
        self.pool: Optional[Pool] = None
        self.batch_size = 1000
        self.flush_interval = 5  # seconds

        # Performance counters
        self.stats = {
            'trades_inserted': 0,
            'quotes_inserted': 0,
            'aggregates_inserted': 0,
            'batches_processed': 0,
            'errors': 0
        }

    async def initialize(self):
        """Initialize database connection pool and create tables."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            # Create tables if they don't exist
            await self.create_tables()

            logger.info("Database initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    async def create_tables(self):
        """Create necessary database tables."""
        async with self.pool.acquire() as conn:
            # Crypto trades table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS crypto_trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    size DECIMAL(20, 8) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    exchange INTEGER,
                    conditions INTEGER[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, exchange)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
                ON crypto_trades(symbol, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON crypto_trades(timestamp DESC);
            ''')

            # Crypto quotes table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS crypto_quotes (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    bid_price DECIMAL(20, 8) NOT NULL,
                    bid_size DECIMAL(20, 8) NOT NULL,
                    ask_price DECIMAL(20, 8) NOT NULL,
                    ask_size DECIMAL(20, 8) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    exchange INTEGER,
                    spread DECIMAL(20, 8) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
                    mid_price DECIMAL(20, 8) GENERATED ALWAYS AS ((ask_price + bid_price) / 2) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, exchange)
                );

                CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp
                ON crypto_quotes(symbol, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_quotes_timestamp
                ON crypto_quotes(timestamp DESC);
            ''')

            # Crypto aggregates (OHLCV) table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS crypto_aggregates (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timespan VARCHAR(10) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    open DECIMAL(20, 8) NOT NULL,
                    high DECIMAL(20, 8) NOT NULL,
                    low DECIMAL(20, 8) NOT NULL,
                    close DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) NOT NULL,
                    vwap DECIMAL(20, 8),
                    transactions INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timespan, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_aggregates_symbol_timestamp
                ON crypto_aggregates(symbol, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_aggregates_timestamp
                ON crypto_aggregates(timestamp DESC);
            ''')

            # Data quality metrics table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    metric_value DECIMAL(20, 8),
                    timestamp TIMESTAMP NOT NULL,
                    details JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_quality_symbol_timestamp
                ON data_quality_metrics(symbol, timestamp DESC);
            ''')

            logger.info("Database tables created/verified")

    async def insert_trades(self, trades: List[Dict[str, Any]]):
        """Batch insert trade data."""
        if not trades:
            return

        try:
            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                values = [
                    (
                        trade['symbol'],
                        trade['price'],
                        trade['size'],
                        trade['timestamp'],
                        trade.get('exchange'),
                        trade.get('conditions', [])
                    )
                    for trade in trades
                ]

                # Batch insert with ON CONFLICT DO NOTHING
                result = await conn.executemany(
                    '''
                    INSERT INTO crypto_trades (symbol, price, size, timestamp, exchange, conditions)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (symbol, timestamp, exchange) DO NOTHING
                    ''',
                    values
                )

                inserted_count = int(result.split(' ')[-1]) if result else 0
                self.stats['trades_inserted'] += inserted_count
                logger.debug(f"Inserted {inserted_count} trades")

        except Exception as e:
            logger.error(f"Failed to insert trades: {e}")
            self.stats['errors'] += 1

    async def insert_quotes(self, quotes: List[Dict[str, Any]]):
        """Batch insert quote data."""
        if not quotes:
            return

        try:
            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                values = [
                    (
                        quote['symbol'],
                        quote['bid_price'],
                        quote['bid_size'],
                        quote['ask_price'],
                        quote['ask_size'],
                        quote['timestamp'],
                        quote.get('exchange')
                    )
                    for quote in quotes
                ]

                # Batch insert with ON CONFLICT DO NOTHING
                result = await conn.executemany(
                    '''
                    INSERT INTO crypto_quotes
                    (symbol, bid_price, bid_size, ask_price, ask_size, timestamp, exchange)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, timestamp, exchange) DO NOTHING
                    ''',
                    values
                )

                inserted_count = int(result.split(' ')[-1]) if result else 0
                self.stats['quotes_inserted'] += inserted_count
                logger.debug(f"Inserted {inserted_count} quotes")

        except Exception as e:
            logger.error(f"Failed to insert quotes: {e}")
            self.stats['errors'] += 1

    async def insert_aggregates(self, aggregates: List[Dict[str, Any]]):
        """Batch insert aggregate bar data."""
        if not aggregates:
            return

        try:
            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                values = [
                    (
                        agg['symbol'],
                        agg.get('timespan', 'minute'),
                        agg['timestamp'],
                        agg['open'],
                        agg['high'],
                        agg['low'],
                        agg['close'],
                        agg['volume'],
                        agg.get('vwap'),
                        agg.get('transactions')
                    )
                    for agg in aggregates
                ]

                # Batch insert with ON CONFLICT DO UPDATE for latest values
                result = await conn.executemany(
                    '''
                    INSERT INTO crypto_aggregates
                    (symbol, timespan, timestamp, open, high, low, close, volume, vwap, transactions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol, timespan, timestamp)
                    DO UPDATE SET
                        high = GREATEST(EXCLUDED.high, crypto_aggregates.high),
                        low = LEAST(EXCLUDED.low, crypto_aggregates.low),
                        close = EXCLUDED.close,
                        volume = crypto_aggregates.volume + EXCLUDED.volume
                    ''',
                    values
                )

                inserted_count = int(result.split(' ')[-1]) if result else 0
                self.stats['aggregates_inserted'] += inserted_count
                logger.debug(f"Inserted/Updated {inserted_count} aggregates")

        except Exception as e:
            logger.error(f"Failed to insert aggregates: {e}")
            self.stats['errors'] += 1

    async def validate_and_insert(self, data_type: str, data: List[Dict[str, Any]]):
        """Validate data quality before insertion."""
        if not data:
            return

        # Perform data quality checks
        quality_issues = []

        for item in data:
            # Check for required fields
            if not item.get('symbol') or not item.get('timestamp'):
                quality_issues.append({
                    'type': 'missing_required_field',
                    'data': item
                })
                continue

            # Check for reasonable price ranges (prevent bad data)
            if data_type in ['trade', 'aggregate']:
                price = item.get('price') or item.get('close')
                if price and (price <= 0 or price > 1000000):
                    quality_issues.append({
                        'type': 'invalid_price',
                        'symbol': item['symbol'],
                        'price': price
                    })

            # Check for timestamp validity (not too far in future/past)
            timestamp = item.get('timestamp')
            if timestamp:
                current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                if abs(timestamp - current_time) > 86400000:  # More than 1 day off
                    quality_issues.append({
                        'type': 'invalid_timestamp',
                        'symbol': item['symbol'],
                        'timestamp': timestamp
                    })

        # Log quality issues if any
        if quality_issues:
            logger.warning(f"Data quality issues found: {len(quality_issues)}")
            await self.log_quality_metrics(quality_issues)

        # Insert data based on type
        if data_type == 'trade':
            await self.insert_trades(data)
        elif data_type == 'quote':
            await self.insert_quotes(data)
        elif data_type == 'aggregate':
            await self.insert_aggregates(data)

        self.stats['batches_processed'] += 1

    async def log_quality_metrics(self, issues: List[Dict[str, Any]]):
        """Log data quality metrics to database."""
        try:
            async with self.pool.acquire() as conn:
                for issue in issues:
                    await conn.execute(
                        '''
                        INSERT INTO data_quality_metrics
                        (symbol, metric_type, metric_value, timestamp, details)
                        VALUES ($1, $2, $3, $4, $5)
                        ''',
                        issue.get('symbol', 'UNKNOWN'),
                        issue['type'],
                        1.0,  # Count of issues
                        datetime.now(timezone.utc),
                        json.dumps(issue)
                    )
        except Exception as e:
            logger.error(f"Failed to log quality metrics: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self.stats.copy()

        try:
            async with self.pool.acquire() as conn:
                # Get record counts
                trades_count = await conn.fetchval("SELECT COUNT(*) FROM crypto_trades")
                quotes_count = await conn.fetchval("SELECT COUNT(*) FROM crypto_quotes")
                aggregates_count = await conn.fetchval("SELECT COUNT(*) FROM crypto_aggregates")

                stats.update({
                    'total_trades': trades_count,
                    'total_quotes': quotes_count,
                    'total_aggregates': aggregates_count
                })

                # Get latest timestamps
                latest_trade = await conn.fetchrow(
                    "SELECT symbol, MAX(timestamp) as latest FROM crypto_trades GROUP BY symbol"
                )
                if latest_trade:
                    stats['latest_trade'] = dict(latest_trade)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")

        return stats

    async def cleanup(self):
        """Clean up database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def optimize_tables(self):
        """Run maintenance operations on tables."""
        try:
            async with self.pool.acquire() as conn:
                # Analyze tables for query optimization
                await conn.execute("ANALYZE crypto_trades")
                await conn.execute("ANALYZE crypto_quotes")
                await conn.execute("ANALYZE crypto_aggregates")

                # Optional: VACUUM to reclaim space (be careful in production)
                # await conn.execute("VACUUM crypto_trades")

                logger.info("Database tables optimized")

        except Exception as e:
            logger.error(f"Failed to optimize tables: {e}")