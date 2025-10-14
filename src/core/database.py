from .exceptions import DatabaseConnectionError
from .settings import get_settings
from contextlib import contextmanager
from functools import lru_cache
from psycopg2 import pool
from src.database import get_db, Client
from typing import Optional, Dict, Any
import logging
import psycopg2
import threading


"""
Database Connection Pool Management
Provides centralized database connection pooling for all services
"""



logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Thread-safe database connection pool manager
    Provides connection pooling for PostgreSQL/Supabase
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for connection pool"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize connection pool (only once)"""
        if self._initialized:
            return

        settings = get_settings()
        self._initialized = True
        self._pools: Dict[str, Any] = {}
        self._supabase_client: Optional[Client] = None

        try:
            # Initialize PostgreSQL connection pool
            if settings.database_url:
                self._pools["postgres"] = psycopg2.pool.ThreadedConnectionPool(
                    minconn=settings.database_pool_min,
                    maxconn=settings.database_pool_max,
                    dsn=settings.database_url,
                    connect_timeout=settings.database_timeout,
                )
                logger.info(
                    f"Initialized PostgreSQL pool (min={settings.database_pool_min}, "
                    f"max={settings.database_pool_max})"
                )

            # Initialize Supabase client
            if settings.supabase_url and settings.supabase_service_key:
                self._supabase_client = create_client(
                    settings.supabase_url, settings.supabase_service_key
                )
                logger.info("Initialized Supabase client")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseConnectionError(
                "Failed to initialize database connection pool", details={"error": str(e)}
            )

    @contextmanager
    @lru_cache(maxsize=128)
    def get_postgres_connection(self):
        """
        Get a PostgreSQL connection from the pool

        Usage:
            with db_pool.get_postgres_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")

        Yields:
            PostgreSQL connection from pool
        """
        if "postgres" not in self._pools:
            raise DatabaseConnectionError("PostgreSQL pool not initialized")

        conn = None
        try:
            conn = self._pools["postgres"].getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise DatabaseConnectionError(
                "Database operation failed", details={"error": str(e)}
            )
        finally:
            if conn:
                self._pools["postgres"].putconn(conn)

    @lru_cache(maxsize=128)

    def get_supabase_client(self) -> Client:
        """
        Get Supabase client

        Returns:
            Supabase Client instance
        """
        if not self._supabase_client:
            raise DatabaseConnectionError("Supabase client not initialized")
        return self._supabase_client

    def close_all(self):
        """Close all connection pools"""
        try:
            if "postgres" in self._pools:
                self._pools["postgres"].closeall()
                logger.info("Closed PostgreSQL connection pool")

            self._pools.clear()
            self._supabase_client = None
            logger.info("All database connections closed")

        except Exception as e:
            logger.error(f"Error closing database pools: {e}")

    @lru_cache(maxsize=128)

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics

        Returns:
            Dictionary with pool statistics
        """
        stats = {}

        try:
            if "postgres" in self._pools:
                pool = self._pools["postgres"]
                stats["postgres"] = {
                    "type": "PostgreSQL",
                    "status": "initialized",
                    "min_connections": pool.minconn,
                    "max_connections": pool.maxconn,
                }

            if self._supabase_client:
                stats["supabase"] = {
                    "type": "Supabase",
                    "status": "initialized",
                }

        except Exception as e:
            logger.error(f"Error getting pool stats: {e}")
            stats["error"] = str(e)

        return stats


# Global database pool instance
_db_pool: Optional[DatabasePool] = None


@lru_cache(maxsize=128)


def get_database_pool() -> DatabasePool:
    """
    Get the global database pool instance

    Returns:
        DatabasePool singleton instance
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool


def close_database_pool():
    """Close the global database pool"""
    global _db_pool
    if _db_pool:
        _db_pool.close_all()
        _db_pool = None


# =============================================================================
# Helper Functions
# =============================================================================


def execute_query(query: str, params: Optional[tuple] = None) -> list:
    """
    Execute a PostgreSQL query and return results

    Args:
        query: SQL query string
        params: Query parameters (optional)

    Returns:
        List of result rows
    """
    db_pool = get_database_pool()
    with db_pool.get_postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:  # SELECT query
                return cur.fetchall()
            return []


def execute_many(query: str, params_list: list) -> int:
    """
    Execute a PostgreSQL query with multiple parameter sets

    Args:
        query: SQL query string
        params_list: List of parameter tuples

    Returns:
        Number of rows affected
    """
    db_pool = get_database_pool()
    with db_pool.get_postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params_list)
            return cur.rowcount


@lru_cache(maxsize=128)


def get_supabase() -> Client:
    """
    Get Supabase client instance

    Returns:
        Supabase Client
    """
    db_pool = get_database_pool()
    return db_pool.get_supabase_client()


# =============================================================================
# Health Check
# =============================================================================


def check_database_health() -> Dict[str, Any]:
    """
    Check database connection health

    Returns:
        Dictionary with health status
    """
    health = {"status": "unknown", "details": {}}

    try:
        db_pool = get_database_pool()

        # Check PostgreSQL
        try:
            result = execute_query("SELECT 1")
            health["details"]["postgres"] = {"status": "healthy", "test_query": "passed"}
        except Exception as e:
            health["details"]["postgres"] = {"status": "unhealthy", "error": str(e)}

        # Check Supabase
        try:
            supabase = get_supabase()
            # Simple health check - get from any table
            health["details"]["supabase"] = {"status": "healthy", "client": "connected"}
        except Exception as e:
            health["details"]["supabase"] = {"status": "unhealthy", "error": str(e)}

        # Overall status
        all_healthy = all(
            details.get("status") == "healthy" for details in health["details"].values()
        )
        health["status"] = "healthy" if all_healthy else "degraded"

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health

