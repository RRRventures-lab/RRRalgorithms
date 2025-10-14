"""
Optimized Database Module
=========================

High-performance database operations with:
- Batch inserts/updates
- Connection pooling
- Query optimization
- Prepared statements

Author: RRR Ventures
Date: 2025-10-12
"""

import sqlite3
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import threading
from queue import Queue
import time
import logging

logger = logging.getLogger(__name__)


class OptimizedDatabase:
    """Optimized database with batch operations and connection pooling."""

    def __init__(self, db_path: str, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connection_pool = Queue(maxsize=pool_size)
        self._lock = threading.RLock()

        # Initialize connection pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self.connection_pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized database connection."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )

        # Optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 30000000000")

        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = self.connection_pool.get()
        try:
            yield conn
        finally:
            self.connection_pool.put(conn)

    def insert_batch(self, table: str, records: List[Dict[str, Any]]):
        """
        Batch insert records.

        Args:
            table: Table name
            records: List of dictionaries to insert
        """
        if not records:
            return

        # Get column names from first record
        columns = list(records[0].keys())
        placeholders = ','.join(['?' for _ in columns])
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Use executemany for batch insert
            cursor.executemany(
                query,
                [tuple(r[col] for col in columns) for r in records]
            )

            conn.commit()

        logger.info(f"Batch inserted {len(records)} records into {table}")

    def update_batch(self, table: str, updates: List[Dict[str, Any]], key_column: str = 'id'):
        """
        Batch update records.

        Args:
            table: Table name
            updates: List of dictionaries with updates
            key_column: Column to use for WHERE clause
        """
        if not updates:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for update in updates:
                # Build SET clause
                set_columns = [f"{k} = ?" for k in update.keys() if k != key_column]
                set_clause = ', '.join(set_columns)

                # Build query
                query = f"UPDATE {table} SET {set_clause} WHERE {key_column} = ?"

                # Get values
                values = [v for k, v in update.items() if k != key_column]
                values.append(update[key_column])

                cursor.execute(query, values)

            conn.commit()

        logger.info(f"Batch updated {len(updates)} records in {table}")

    def execute_transaction(self, operations: List[Tuple[str, tuple]]):
        """
        Execute multiple operations in a single transaction.

        Args:
            operations: List of (query, params) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")

                for query, params in operations:
                    cursor.execute(query, params)

                cursor.execute("COMMIT")
                logger.info(f"Executed transaction with {len(operations)} operations")

            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Transaction failed: {e}")
                raise

    def bulk_fetch(self, query: str, params: tuple = (), chunk_size: int = 1000):
        """
        Fetch large result sets in chunks.

        Args:
            query: SQL query
            params: Query parameters
            chunk_size: Number of rows per chunk

        Yields:
            Chunks of results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield [dict(row) for row in rows]


# Global optimized database instance
_optimized_db: Optional[OptimizedDatabase] = None


def get_optimized_db(db_path: str = "data/local.db") -> OptimizedDatabase:
    """Get optimized database instance."""
    global _optimized_db
    if _optimized_db is None:
        _optimized_db = OptimizedDatabase(db_path)
    return _optimized_db
