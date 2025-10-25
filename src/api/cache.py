"""
API Response Caching Layer
Redis-based caching for API responses to improve performance
"""

import json
import logging
import hashlib
from typing import Optional, Any, Callable
from functools import wraps
from datetime import timedelta

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class APICache:
    """
    Redis-based API response cache

    Features:
    - Automatic cache key generation from function args
    - TTL-based expiration
    - Cache invalidation
    - Cache statistics
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize cache

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None
        self.enabled = redis is not None

        if not self.enabled:
            logger.warning("Redis not available - caching disabled")

    async def connect(self):
        """Connect to Redis"""
        if not self.enabled:
            return

        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            await self.client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments

        Args:
            prefix: Cache key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create a string representation of args and kwargs
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"

        # Hash it to keep key length reasonable
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return f"api_cache:{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.enabled or not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        if not self.enabled or not self.client:
            return False

        try:
            value_json = json.dumps(value)
            await self.client.setex(key, ttl, value_json)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return False

        try:
            await self.client.delete(key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern

        Args:
            pattern: Key pattern (e.g., "api_cache:trades:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False}

        try:
            info = await self.client.info("stats")
            return {
                "enabled": True,
                "total_keys": await self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": False, "error": str(e)}


# Global cache instance
_cache: Optional[APICache] = None


def get_cache() -> APICache:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        from src.security.secrets_manager import get_secrets_manager
        secrets = get_secrets_manager()
        redis_url = secrets.get_secret("REDIS_URL", "redis://localhost:6379/0")
        _cache = APICache(redis_url)
    return _cache


def cached(prefix: str, ttl: int = 300):
    """
    Decorator to cache function results

    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds

    Usage:
        @cached(prefix="portfolio", ttl=60)
        async def get_portfolio():
            # expensive operation
            return data
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key
            cache_key = cache._generate_cache_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator
