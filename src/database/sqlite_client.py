"""
SQLite database client implementation.
Optimized for local storage on Lexar 2TB drive.
"""

import aiosqlite
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base import DatabaseClient


logger = logging.getLogger(__name__)


class SQLiteClient(DatabaseClient):
    """SQLite database client with async support."""
    
    def __init__(
        self,
        db_path: str = None,
        cache_size_mb: int = 64,
        mmap_size_gb: int = 30,
        timeout: float = 30.0
    ):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file
            cache_size_mb: Cache size in MB (default: 64MB)
            mmap_size_gb: Memory-map size in GB (default: 30GB)
            timeout: Connection timeout in seconds
        """
        if db_path is None:
            # Default to data/db/trading.db
            base_path = os.getenv('TRADING_HOME', os.getcwd())
            db_path = os.path.join(base_path, 'data', 'db', 'trading.db')
        
        self.db_path = db_path
        self.cache_size_mb = cache_size_mb
        self.mmap_size_gb = mmap_size_gb
        self.timeout = timeout
        self.connection: Optional[aiosqlite.Connection] = None
        self._initialized = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        logger.info(f"SQLiteClient initialized with db_path: {self.db_path}")
    
    async def connect(self) -> None:
        """Establish database connection with optimizations."""
        if self.connection is not None:
            logger.warning("Connection already established")
            return
        
        logger.info(f"Connecting to SQLite database: {self.db_path}")
        
        # Create connection with optimizations
        self.connection = await aiosqlite.connect(
            self.db_path,
            timeout=self.timeout,
            isolation_level=None  # Autocommit mode for better concurrency with WAL
        )
        
        # Enable row factory for dict results
        self.connection.row_factory = aiosqlite.Row
        
        # Apply SQLite optimizations
        await self._apply_optimizations()
        
        # Initialize schema if needed
        if not self._initialized:
            await self._initialize_schema()
            self._initialized = True
        
        logger.info("SQLite connection established and optimized")
    
    async def _apply_optimizations(self) -> None:
        """Apply SQLite performance optimizations."""
        optimizations = [
            "PRAGMA journal_mode = WAL",  # Write-Ahead Logging
            "PRAGMA synchronous = NORMAL",  # Balance speed/safety
            f"PRAGMA cache_size = -{self.cache_size_mb * 1024}",  # Negative = KB
            "PRAGMA temp_store = MEMORY",  # Temp tables in RAM
            f"PRAGMA mmap_size = {self.mmap_size_gb * 1024 * 1024 * 1024}",  # Memory mapping
            "PRAGMA foreign_keys = ON",  # Enforce foreign keys
            "PRAGMA auto_vacuum = INCREMENTAL",  # Space management
            "PRAGMA busy_timeout = 30000",  # 30 second busy timeout
        ]
        
        for pragma in optimizations:
            await self.connection.execute(pragma)
        
        await self.connection.commit()
        logger.debug("Applied SQLite optimizations")
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema from schema.sql file."""
        schema_path = os.path.join(
            os.path.dirname(__file__),
            'schema.sql'
        )
        
        if not os.path.exists(schema_path):
            logger.warning(f"Schema file not found: {schema_path}")
            return
        
        logger.info("Initializing database schema...")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema (split by semicolons)
        await self.connection.executescript(schema_sql)
        await self.connection.commit()
        
        logger.info("Database schema initialized successfully")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("SQLite connection closed")
    
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> aiosqlite.Cursor:
        """Execute a query and return cursor."""
        if not self.connection:
            await self.connect()
        
        if params:
            cursor = await self.connection.execute(query, params)
        else:
            cursor = await self.connection.execute(query)
        
        await self.connection.commit()
        return cursor
    
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """Execute same query with multiple parameter sets."""
        if not self.connection:
            await self.connect()
        
        await self.connection.executemany(query, params_list)
        await self.connection.commit()
    
    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch single row as dictionary."""
        if not self.connection:
            await self.connect()
        
        if params:
            cursor = await self.connection.execute(query, params)
        else:
            cursor = await self.connection.execute(query)
        
        row = await cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        if not self.connection:
            await self.connect()
        
        if params:
            cursor = await self.connection.execute(query, params)
        else:
            cursor = await self.connection.execute(query)
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> int:
        """Insert single row, return inserted ID."""
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        
        # Filter out None values in keys
        data = {k: v for k, v in data.items() if v is not None or k in data}
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        cursor = await self.execute(query, tuple(data.values()))
        return cursor.lastrowid
    
    async def insert_many(
        self,
        table: str,
        data_list: List[Dict[str, Any]]
    ) -> None:
        """Insert multiple rows."""
        if not data_list:
            return
        
        # Use first row to determine columns
        columns = list(data_list[0].keys())
        placeholders = ', '.join(['?' for _ in columns])
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Convert dicts to tuples in same column order
        params_list = [
            tuple(row.get(col) for col in columns)
            for row in data_list
        ]
        
        await self.execute_many(query, params_list)
    
    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> int:
        """Update rows matching where clause, return count."""
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        if not where:
            raise ValueError("Where clause cannot be empty (prevents accidental full table update)")
        
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = tuple(data.values()) + tuple(where.values())
        
        cursor = await self.execute(query, params)
        return cursor.rowcount
    
    async def delete(
        self,
        table: str,
        where: Dict[str, Any]
    ) -> int:
        """Delete rows matching where clause, return count."""
        if not where:
            raise ValueError("Where clause cannot be empty (prevents accidental full table delete)")
        
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        
        cursor = await self.execute(query, tuple(where.values()))
        return cursor.rowcount
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        result = await self.fetch_one(query, (table_name,))
        return result is not None
    
    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        query = f"PRAGMA table_info({table_name})"
        return await self.fetch_all(query)
    
    async def vacuum(self) -> None:
        """Run VACUUM to optimize database file."""
        logger.info("Running VACUUM on database...")
        await self.connection.execute("VACUUM")
        await self.connection.commit()
        logger.info("VACUUM completed")
    
    async def analyze(self) -> None:
        """Update query optimizer statistics."""
        logger.info("Running ANALYZE on database...")
        await self.connection.execute("ANALYZE")
        await self.connection.commit()
        logger.info("ANALYZE completed")
    
    async def get_db_size(self) -> Dict[str, Any]:
        """Get database size information."""
        db_size = os.path.getsize(self.db_path)
        
        page_count = await self.fetch_one("PRAGMA page_count")
        page_size = await self.fetch_one("PRAGMA page_size")
        freelist_count = await self.fetch_one("PRAGMA freelist_count")
        
        return {
            'file_size_bytes': db_size,
            'file_size_mb': db_size / (1024 * 1024),
            'page_count': page_count['page_count'] if page_count else 0,
            'page_size': page_size['page_size'] if page_size else 0,
            'free_pages': freelist_count['freelist_count'] if freelist_count else 0,
        }
    
    async def backup(self, backup_path: str) -> None:
        """Create backup of database."""
        logger.info(f"Creating database backup: {backup_path}")
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Use SQLite backup API for safe backup
        def backup_db():
            source_conn = sqlite3.connect(self.db_path)
            dest_conn = sqlite3.connect(backup_path)
            source_conn.backup(dest_conn)
            source_conn.close()
            dest_conn.close()
        
        # Run in thread pool to avoid blocking
        import asyncio
        await asyncio.get_event_loop().run_in_executor(None, backup_db)
        
        logger.info(f"Database backup completed: {backup_path}")
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()

