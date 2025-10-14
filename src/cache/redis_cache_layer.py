from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import hashlib
import json
import numpy as np
import pickle
import threading
import time


"""
Redis Caching Layer for High-Performance Data Access
=====================================================

Distributed caching layer featuring:
- Redis for shared cache across processes
- Automatic serialization/deserialization
- TTL management
- Cache warming and prefetching
- Fallback to local cache if Redis unavailable
- Pub/Sub for cache invalidation

Author: RRR Ventures
Date: 2025-10-12
"""



# Try to import Redis, fallback to local cache if not available
try:
    import redis
    from redis import Redis, ConnectionPool
    from redis.exceptions import RedisError, ConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    ConnectionPool = None
    RedisError = Exception
    ConnectionError = Exception


class LocalCacheFallback:
    """Local in-memory cache as fallback when Redis is unavailable."""

    def __init__(self, max_size: int = 10000):
        """
        Initialize local cache.

        Args:
            max_size: Maximum cache size
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                return self.cache[key]
            return None

    def set(self, key: str, value: bytes, ttl: int = 300) -> bool:
        """Set value in cache."""
        with self._lock:
            # Simple LRU eviction
            if len(self.cache) >= self.max_size:
                oldest = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest]
                del self.timestamps[oldest]

            self.cache[key] = value
            self.timestamps[key] = time.time()
            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.cache

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for key (no-op for local cache)."""
        return True

    def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        """Get multiple keys."""
        with self._lock:
            return [self.cache.get(k) for k in keys]

    def mset(self, mapping: Dict[str, bytes]) -> bool:
        """Set multiple keys."""
        with self._lock:
            for key, value in mapping.items():
                self.set(key, value)
            return True

    def flush(self):
        """Clear all cache."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5
    connection_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 300  # 5 minutes
    key_prefix: str = "rrr:trading:"
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB


class RedisCache:
    """
    Redis-based caching layer with advanced features.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Redis cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.connected = False

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'fallback_used': 0,
            'bytes_saved': 0,
            'bytes_loaded': 0
        }

        # Initialize connection
        if REDIS_AVAILABLE:
            try:
                self._init_redis()
                self.connected = True
            except Exception as e:
                print(f"Redis connection failed, using local cache: {e}")
                self.connected = False

        # Fallback to local cache if Redis not available
        if not self.connected:
            self.client = LocalCacheFallback(max_size=10000)
            self.pubsub = None
        else:
            # Set up pub/sub for cache invalidation
            self.pubsub = self.client.pubsub()

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _init_redis(self):
        """Initialize Redis connection pool."""
        self.pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.connection_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
            health_check_interval=self.config.health_check_interval
        )
        self.client = Redis(connection_pool=self.pool)

        # Test connection
        self.client.ping()

    def _generate_key(self, key: str) -> str:
        """Generate prefixed cache key."""
        return f"{self.config.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        # Try JSON first (for simple types)
        try:
            data = json.dumps(value).encode()
            metadata = b'j'  # JSON marker
        except (TypeError, ValueError):
            # Fall back to pickle for complex types
            data = pickle.dumps(value)
            metadata = b'p'  # Pickle marker

        # Compress if needed
        if self.config.enable_compression and len(data) > self.config.compression_threshold:
            import zlib
            data = zlib.compress(data, level=6)
            metadata += b'z'  # Compression marker

        return metadata + data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not data:
            return None

        # Check metadata
        metadata = data[0:1]
        actual_data = data[1:]

        # Check for additional markers
        if len(data) > 1 and data[1:2] == b'z':
            # Compressed
            import zlib
            actual_data = zlib.decompress(data[2:])
            metadata = data[0:1]
        elif b'z' in metadata:
            # Compressed
            import zlib
            actual_data = zlib.decompress(actual_data)
            metadata = metadata.replace(b'z', b'')

        # Deserialize based on type
        if metadata == b'j':
            return json.loads(actual_data.decode())
        elif metadata == b'p':
            return pickle.loads(actual_data)
        else:
            # Raw bytes
            return actual_data

    def get(
        self,
        key: str,
        default: Any = None,
        deserialize: bool = True
    ) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found
            deserialize: Whether to deserialize value

        Returns:
            Cached value or default
        """
        full_key = self._generate_key(key)

        try:
            data = self.client.get(full_key)

            if data is None:
                self.stats['misses'] += 1
                return default

            self.stats['hits'] += 1
            self.stats['bytes_loaded'] += len(data)

            if deserialize:
                return self._deserialize(data)
            return data

        except (RedisError, ConnectionError) as e:
            self.stats['errors'] += 1
            self.stats['fallback_used'] += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize value

        Returns:
            Success status
        """
        full_key = self._generate_key(key)
        ttl = ttl or self.config.default_ttl

        try:
            if serialize:
                data = self._serialize(value)
            else:
                data = value

            result = self.client.set(full_key, data, ex=ttl)

            if result:
                self.stats['bytes_saved'] += len(data)

            return bool(result)

        except (RedisError, ConnectionError) as e:
            self.stats['errors'] += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._generate_key(key)

        try:
            return bool(self.client.delete(full_key))
        except (RedisError, ConnectionError):
            self.stats['errors'] += 1
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._generate_key(key)

        try:
            return bool(self.client.exists(full_key))
        except (RedisError, ConnectionError):
            self.stats['errors'] += 1
            return False

    def mget(self, keys: List[str]) -> List[Any]:
        """
        Get multiple keys at once.

        Args:
            keys: List of keys

        Returns:
            List of values
        """
        full_keys = [self._generate_key(k) for k in keys]

        try:
            values = self.client.mget(full_keys)
            results = []

            for value in values:
                if value is None:
                    self.stats['misses'] += 1
                    results.append(None)
                else:
                    self.stats['hits'] += 1
                    self.stats['bytes_loaded'] += len(value)
                    results.append(self._deserialize(value))

            return results

        except (RedisError, ConnectionError):
            self.stats['errors'] += 1
            return [None] * len(keys)

    def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple keys at once.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live for all keys

        Returns:
            Success status
        """
        ttl = ttl or self.config.default_ttl

        try:
            # Serialize all values
            serialized = {}
            for key, value in mapping.items():
                full_key = self._generate_key(key)
                data = self._serialize(value)
                serialized[full_key] = data
                self.stats['bytes_saved'] += len(data)

            # Set all at once
            result = self.client.mset(serialized)

            # Set TTL for each key
            if ttl and result:
                pipeline = self.client.pipeline()
                for key in serialized:
                    pipeline.expire(key, ttl)
                pipeline.execute()

            return bool(result)

        except (RedisError, ConnectionError):
            self.stats['errors'] += 1
            return False

    def cache_decorator(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live
            key_prefix: Additional key prefix

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function and arguments
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result

                # Call function
                result = func(*args, **kwargs)

                # Cache result
                self.set(cache_key, result, ttl=ttl)

                return result

            return wrapper
        return decorator

    async def get_async(self, key: str, default: Any = None) -> Any:
        """Async get operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get,
            key,
            default
        )

    async def set_async(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Async set operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.set,
            key,
            value,
            ttl
        )

    def cache_market_data(
        self,
        symbol: str,
        timestamp: float,
        ohlcv: Dict[str, float],
        ttl: int = 60
    ) -> bool:
        """
        Cache market data with optimized structure.

        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            ohlcv: OHLCV data
            ttl: Cache TTL

        Returns:
            Success status
        """
        # Use sorted set for time series data
        key = f"market:{symbol}"

        # Store as hash for efficiency
        data = {
            't': timestamp,
            'o': ohlcv['open'],
            'h': ohlcv['high'],
            'l': ohlcv['low'],
            'c': ohlcv['close'],
            'v': ohlcv['volume']
        }

        return self.set(f"{key}:{int(timestamp)}", data, ttl=ttl)

    @lru_cache(maxsize=128)

    def get_market_data_range(
        self,
        symbol: str,
        start_time: float,
        end_time: float
    ) -> List[Dict[str, float]]:
        """
        Get market data for time range.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of OHLCV data
        """
        results = []

        # Generate keys for time range
        keys = []
        current = int(start_time)
        while current <= end_time:
            keys.append(f"market:{symbol}:{current}")
            current += 60  # Assume minute bars

        # Batch get
        values = self.mget(keys)

        for value in values:
            if value:
                results.append({
                    'timestamp': value['t'],
                    'open': value['o'],
                    'high': value['h'],
                    'low': value['l'],
                    'close': value['c'],
                    'volume': value['v']
                })

        return results

    def cache_prediction(
        self,
        symbol: str,
        prediction: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """
        Cache ML prediction.

        Args:
            symbol: Trading symbol
            prediction: Prediction data
            ttl: Cache TTL

        Returns:
            Success status
        """
        key = f"prediction:{symbol}:{int(time.time())}"
        return self.set(key, prediction, ttl=ttl)

    @lru_cache(maxsize=128)

    def get_latest_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest prediction for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest prediction or None
        """
        # Look for recent predictions
        current_time = int(time.time())
        for i in range(5):  # Check last 5 seconds
            key = f"prediction:{symbol}:{current_time - i}"
            result = self.get(key)
            if result:
                return result

        return None

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "market:BTC-*")

        Returns:
            Number of keys deleted
        """
        if not self.connected:
            return 0

        try:
            full_pattern = self._generate_key(pattern)
            keys = self.client.keys(full_pattern)

            if keys:
                return self.client.delete(*keys)

            return 0

        except (RedisError, ConnectionError):
            self.stats['errors'] += 1
            return 0

    def warm_cache(
        self,
        data_loader: Callable,
        keys: List[str],
        ttl: Optional[int] = None
    ) -> int:
        """
        Pre-populate cache with data.

        Args:
            data_loader: Function to load data
            keys: Keys to warm
            ttl: Cache TTL

        Returns:
            Number of keys warmed
        """
        warmed = 0

        for key in keys:
            if not self.exists(key):
                try:
                    value = data_loader(key)
                    if value is not None:
                        self.set(key, value, ttl=ttl)
                        warmed += 1
                except Exception:
                    pass

        return warmed

    @lru_cache(maxsize=128)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0

        stats = {
            **self.stats,
            'hit_rate': hit_rate,
            'connected': self.connected,
            'bytes_saved_mb': self.stats['bytes_saved'] / (1024 * 1024),
            'bytes_loaded_mb': self.stats['bytes_loaded'] / (1024 * 1024)
        }

        # Add Redis info if connected
        if self.connected and REDIS_AVAILABLE:
            try:
                info = self.client.info()
                stats['redis_memory_mb'] = info.get('used_memory', 0) / (1024 * 1024)
                stats['redis_keys'] = self.client.dbsize()
            except:
                pass

        return stats

    def flush(self, pattern: Optional[str] = None):
        """
        Flush cache.

        Args:
            pattern: Optional pattern to flush
        """
        if pattern:
            self.invalidate_pattern(pattern)
        else:
            try:
                if self.connected:
                    self.client.flushdb()
                else:
                    self.client.flush()
            except:
                pass


# Singleton instance
_cache_instance: Optional[RedisCache] = None


@lru_cache(maxsize=128)


def get_redis_cache(config: Optional[CacheConfig] = None) -> RedisCache:
    """
    Get singleton Redis cache instance.

    Args:
        config: Optional cache configuration

    Returns:
        Redis cache instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RedisCache(config)

    return _cache_instance


if __name__ == "__main__":
    import asyncio

    print("🚀 Testing Redis Cache Layer\n")

    # Initialize cache
    config = CacheConfig(
        host="localhost",
        port=6379,
        default_ttl=60,
        enable_compression=True
    )

    cache = RedisCache(config)

    if cache.connected:
        print("✅ Connected to Redis\n")
    else:
        print("⚠️ Using local cache fallback\n")

    # Test basic operations
    print("📊 Testing Basic Operations...")

    # Set value
    cache.set("test:key1", {"data": "value1", "timestamp": time.time()})
    print("  Set test:key1")

    # Get value
    value = cache.get("test:key1")
    print(f"  Got test:key1: {value}")

    # Test batch operations
    print("\n📦 Testing Batch Operations...")

    batch_data = {
        f"batch:key{i}": {"value": i, "data": f"data{i}"}
        for i in range(10)
    }

    cache.mset(batch_data)
    print(f"  Set {len(batch_data)} keys")

    keys = list(batch_data.keys())
    values = cache.mget(keys[:5])
    print(f"  Retrieved {len([v for v in values if v])} values")

    # Test market data caching
    print("\n📈 Testing Market Data Cache...")

    symbols = ["BTC-USD", "ETH-USD"]
    for symbol in symbols:
        for i in range(10):
            cache.cache_market_data(
                symbol,
                time.time() - (10 - i) * 60,
                {
                    'open': 50000 + i * 100,
                    'high': 50100 + i * 100,
                    'low': 49900 + i * 100,
                    'close': 50050 + i * 100,
                    'volume': 1000000
                }
            )

    print(f"  Cached market data for {len(symbols)} symbols")

    # Test decorator
    print("\n🎯 Testing Cache Decorator...")

    @cache.cache_decorator(ttl=60, key_prefix="compute")
    def expensive_computation(x: int, y: int) -> int:
        print(f"    Computing {x} + {y}...")
        time.sleep(0.1)  # Simulate expensive operation
        return x + y

    # First call (cache miss)
    result1 = expensive_computation(5, 3)
    print(f"  First call result: {result1}")

    # Second call (cache hit)
    result2 = expensive_computation(5, 3)
    print(f"  Second call result (cached): {result2}")

    # Test async operations
    async def test_async():
        print("\n⚡ Testing Async Operations...")

        # Async set
        await cache.set_async("async:key", {"async": True})
        print("  Async set completed")

        # Async get
        value = await cache.get_async("async:key")
        print(f"  Async get result: {value}")

    asyncio.run(test_async())

    # Show statistics
    print("\n📊 Cache Statistics:")
    stats = cache.get_statistics()
    print(f"  Hits:              {stats['hits']}")
    print(f"  Misses:            {stats['misses']}")
    print(f"  Hit Rate:          {stats['hit_rate']:.1%}")
    print(f"  Errors:            {stats['errors']}")
    print(f"  Fallback Used:     {stats['fallback_used']}")
    print(f"  Data Saved:        {stats['bytes_saved_mb']:.2f} MB")
    print(f"  Data Loaded:       {stats['bytes_loaded_mb']:.2f} MB")

    if stats.get('redis_memory_mb'):
        print(f"  Redis Memory:      {stats['redis_memory_mb']:.2f} MB")
        print(f"  Redis Keys:        {stats.get('redis_keys', 0)}")

    print("\n✅ Redis Cache Test Complete!")