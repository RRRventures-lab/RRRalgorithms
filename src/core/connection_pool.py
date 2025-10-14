"""
Connection Pool Implementation
==============================

Generic connection pooling for various resources.

Author: RRR Ventures
Date: 2025-10-12
"""

import asyncio
from typing import Any, Optional, Callable, Dict
from contextlib import asynccontextmanager
import time
import logging

logger = logging.getLogger(__name__)


class AsyncConnectionPool:
    """Async connection pool for any resource."""

    def __init__(self,
                 create_connection: Callable,
                 min_size: int = 5,
                 max_size: int = 20,
                 timeout: float = 10.0):
        """
        Initialize connection pool.

        Args:
            create_connection: Async function to create connections
            min_size: Minimum pool size
            max_size: Maximum pool size
            timeout: Connection acquisition timeout
        """
        self.create_connection = create_connection
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout

        self._pool = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._stats = {
            'acquisitions': 0,
            'releases': 0,
            'creates': 0,
            'timeouts': 0
        }

    async def initialize(self):
        """Initialize the pool with minimum connections."""
        async with self._lock:
            for _ in range(self.min_size):
                conn = await self.create_connection()
                await self._pool.put(conn)
                self._size += 1
                self._stats['creates'] += 1

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        start_time = time.time()
        conn = None

        try:
            # Try to get from pool
            try:
                conn = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=self.timeout
                )
                self._stats['acquisitions'] += 1

            except asyncio.TimeoutError:
                # Create new connection if under max size
                async with self._lock:
                    if self._size < self.max_size:
                        conn = await self.create_connection()
                        self._size += 1
                        self._stats['creates'] += 1
                        self._stats['acquisitions'] += 1
                    else:
                        self._stats['timeouts'] += 1
                        raise asyncio.TimeoutError("Connection pool timeout")

            yield conn

        finally:
            # Return connection to pool
            if conn is not None:
                await self._pool.put(conn)
                self._stats['releases'] += 1

            # Log if acquisition was slow
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                logger.warning(f"Slow connection acquisition: {elapsed:.2f}s")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            'current_size': self._size,
            'available': self._pool.qsize()
        }

    async def close(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                conn = await self._pool.get()
                if hasattr(conn, 'close'):
                    await conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")


# Example usage for different resources
class PolygonConnectionPool(AsyncConnectionPool):
    """Connection pool for Polygon.io API."""

    def __init__(self):
        super().__init__(
            create_connection=self._create_polygon_connection,
            min_size=2,
            max_size=10
        )

    async def _create_polygon_connection(self):
        """Create Polygon API connection."""
        import aiohttp
        session = aiohttp.ClientSession(
            headers={'Authorization': 'Bearer YOUR_API_KEY'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return session


class RedisConnectionPool(AsyncConnectionPool):
    """Connection pool for Redis."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        super().__init__(
            create_connection=self._create_redis_connection,
            min_size=5,
            max_size=50
        )

    async def _create_redis_connection(self):
        """Create Redis connection."""
        import aioredis
        return await aioredis.create_redis_pool(self.redis_url)
