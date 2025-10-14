from collections import OrderedDict
from functools import lru_cache
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union
import asyncio
import logging
import threading
import time


"""
Memory Cache System
==================

High-performance in-memory caching system with TTL and LRU eviction.
Provides fast data access with automatic cache management.

Author: RRR Ventures
Date: 2025-10-12
"""



class MemoryCache:
    """
    High-performance in-memory cache with TTL and LRU eviction.
    
    Features:
    - Time-to-live (TTL) support
    - LRU eviction policy
    - Thread-safe operations
    - Async/await support
    - Performance metrics
    - Automatic cleanup
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0  # 1 minute
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Cache storage (OrderedDict for LRU)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0,
            'total_operations': 0,
            'cache_size': 0,
            'hit_rate': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the cache cleanup task."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Memory cache started")
    
    async def stop(self) -> None:
        """Stop the cache cleanup task."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Memory cache stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            self.metrics['total_operations'] += 1
            
            if key not in self._cache:
                self.metrics['misses'] += 1
                return None
            
            item = self._cache[key]
            
            # Check TTL
            if time.time() > item['expires_at']:
                del self._cache[key]
                self.metrics['misses'] += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            self.metrics['hits'] += 1
            self._update_hit_rate()
            
            return item['value']
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            self.metrics['total_operations'] += 1
            
            # Remove existing key if present
            if key in self._cache:
                del self._cache[key]
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            expires_at = time.time() + (ttl or self.default_ttl)
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            
            self.metrics['cache_size'] = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.metrics['cache_size'] = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self.metrics['cache_size'] = 0
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            item = self._cache[key]
            
            # Check TTL
            if time.time() > item['expires_at']:
                del self._cache[key]
                return False
            
            return True
    
    @lru_cache(maxsize=128)
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """
        Get value from cache or set it using factory function.
        
        Args:
            key: Cache key
            factory: Function to generate value if not in cache
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or newly generated value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Generate new value
        value = factory()
        self.set(key, value, ttl)
        return value
    
    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """
        Async version of get_or_set.
        
        Args:
            key: Cache key
            factory: Async function to generate value if not in cache
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or newly generated value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Generate new value (await if it's a coroutine)
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        self.set(key, value, ttl)
        return value
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._cache:
            # Remove first item (oldest)
            self._cache.popitem(last=False)
            self.metrics['evictions'] += 1
            self.metrics['cache_size'] = len(self._cache)
    
    def _update_hit_rate(self) -> None:
        """Update hit rate metric."""
        total = self.metrics['hits'] + self.metrics['misses']
        if total > 0:
            self.metrics['hit_rate'] = self.metrics['hits'] / total
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop to remove expired items."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, item in self._cache.items():
                if current_time > item['expires_at']:
                    expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
        
        if expired_keys:
            self.metrics['cleanups'] += 1
            self.metrics['cache_size'] = len(self._cache)
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired items")
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self.metrics.copy()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            'max_size': self.max_size,
            'current_size': len(self._cache),
            'utilization': len(self._cache) / self.max_size,
            'running': self._running,
            'metrics': self.metrics
        }


# Global cache instance
_cache: Optional[MemoryCache] = None


@lru_cache(maxsize=128)


def get_cache() -> MemoryCache:
    """Get the global cache instance."""
    global _cache
    
    if _cache is None:
        _cache = MemoryCache()
    
    return _cache


def cached(
    ttl: Optional[float] = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Get cache instance
            cache = get_cache()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def async_cached(
    ttl: Optional[float] = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching async function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Get cache instance
            cache = get_cache()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


__all__ = [
    'MemoryCache',
    'get_cache',
    'cached',
    'async_cached',
]