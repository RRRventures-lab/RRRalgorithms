from datetime import datetime, timedelta
from functools import lru_cache
from redis.asyncio import Redis, ConnectionPool
from src.core.exceptions import DatabaseConnectionError
from typing import Dict, List, Optional, Any, Union, Callable
import asyncio
import json
import logging
import pickle
import redis.asyncio as redis


"""
Redis Cache System
=================

High-performance Redis caching system with clustering support,
pub/sub messaging, and advanced data structures.

Author: RRR Ventures
Date: 2025-10-12
"""




class RedisCache:
    """
    High-performance Redis cache with advanced features.
    
    Features:
    - Connection pooling
    - Pub/sub messaging
    - Data serialization
    - TTL management
    - Clustering support
    - Performance monitoring
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        decode_responses: bool = True
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connections in pool
            decode_responses: Whether to decode responses
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        
        # Redis connection
        self.redis: Optional[Redis] = None
        self.pool: Optional[ConnectionPool] = None
        self.initialized = False
        
        # Performance metrics
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_operation_time': 0.0,
            'max_operation_time': 0.0,
            'min_operation_time': float('inf')
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self.initialized:
            return
        
        try:
            self.logger.info("Initializing Redis connection pool...")
            
            # Create connection pool
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses
            )
            
            # Create Redis client
            self.redis = Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis.ping()
            
            self.initialized = True
            self.logger.info("Redis connection pool initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            raise DatabaseConnectionError(f"Redis initialization failed: {e}")
    
    async def close(self) -> None:
        """Close Redis connection pool."""
        if self.redis:
            self.logger.info("Closing Redis connection pool...")
            await self.redis.close()
            self.redis = None
            self.pool = None
            self.initialized = False
            self.logger.info("Redis connection pool closed")
    
    async def _execute_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute Redis operation with metrics tracking."""
        if not self.initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await operation(*args, **kwargs)
            
            # Update metrics
            self.metrics['total_operations'] += 1
            self.metrics['successful_operations'] += 1
            
            operation_time = asyncio.get_event_loop().time() - start_time
            self._update_timing_metrics(operation_time)
            
            return result
            
        except Exception as e:
            self.metrics['total_operations'] += 1
            self.metrics['failed_operations'] += 1
            self.logger.error(f"Redis operation failed: {e}")
            raise DatabaseConnectionError(f"Redis operation failed: {e}")
    
    def _update_timing_metrics(self, operation_time: float) -> None:
        """Update timing metrics."""
        if self.metrics['total_operations'] == 1:
            self.metrics['avg_operation_time'] = operation_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_operation_time'] = (
                alpha * operation_time + 
                (1 - alpha) * self.metrics['avg_operation_time']
            )
        
        self.metrics['max_operation_time'] = max(self.metrics['max_operation_time'], operation_time)
        self.metrics['min_operation_time'] = min(self.metrics['min_operation_time'], operation_time)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time-to-live in seconds
            serialize: Whether to serialize the value
            
        Returns:
            True if successful
        """
        try:
            # Serialize value if needed
            if serialize and not isinstance(value, (str, int, float, bytes)):
                value = json.dumps(value, default=str)
            
            # Set with TTL if provided
            if ttl:
                result = await self._execute_operation(
                    self.redis.setex, key, ttl, value
                )
            else:
                result = await self._execute_operation(
                    self.redis.set, key, value
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to set key {key}: {e}")
            return False
    
    async def get(
        self,
        key: str,
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize the value
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        try:
            value = await self._execute_operation(self.redis.get, key)
            
            if value is None:
                self.metrics['cache_misses'] += 1
                return default
            
            self.metrics['cache_hits'] += 1
            
            # Deserialize value if needed
            if deserialize and isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get key {key}: {e}")
            self.metrics['cache_misses'] += 1
            return default
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key was deleted
        """
        try:
            result = await self._execute_operation(self.redis.delete, key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key exists
        """
        try:
            result = await self._execute_operation(self.redis.exists, key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for a key.
        
        Args:
            key: Redis key
            ttl: Time-to-live in seconds
            
        Returns:
            True if TTL was set
        """
        try:
            result = await self._execute_operation(self.redis.expire, key, ttl)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to set TTL for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get TTL for a key.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            result = await self._execute_operation(self.redis.ttl, key)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get TTL for key {key}: {e}")
            return -2
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            List of matching keys
        """
        try:
            result = await self._execute_operation(self.redis.keys, pattern)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []
    
    async def mget(self, keys: List[str]) -> List[Any]:
        """
        Get multiple values at once.
        
        Args:
            keys: List of Redis keys
            
        Returns:
            List of values (None for missing keys)
        """
        try:
            result = await self._execute_operation(self.redis.mget, keys)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get multiple keys: {e}")
            return [None] * len(keys)
    
    async def mset(self, mapping: Dict[str, Any]) -> bool:
        """
        Set multiple key-value pairs at once.
        
        Args:
            mapping: Dictionary of key-value pairs
            
        Returns:
            True if successful
        """
        try:
            # Serialize values
            serialized_mapping = {}
            for key, value in mapping.items():
                if not isinstance(value, (str, int, float, bytes)):
                    serialized_mapping[key] = json.dumps(value, default=str)
                else:
                    serialized_mapping[key] = value
            
            result = await self._execute_operation(self.redis.mset, serialized_mapping)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to set multiple keys: {e}")
            return False
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """
        Set hash fields.
        
        Args:
            name: Hash name
            mapping: Dictionary of field-value pairs
            
        Returns:
            Number of fields set
        """
        try:
            # Serialize values
            serialized_mapping = {}
            for field, value in mapping.items():
                if not isinstance(value, (str, int, float, bytes)):
                    serialized_mapping[field] = json.dumps(value, default=str)
                else:
                    serialized_mapping[field] = value
            
            result = await self._execute_operation(self.redis.hset, name, mapping=serialized_mapping)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to set hash {name}: {e}")
            return 0
    
    async def hget(self, name: str, key: str) -> Any:
        """
        Get hash field value.
        
        Args:
            name: Hash name
            key: Field key
            
        Returns:
            Field value or None
        """
        try:
            result = await self._execute_operation(self.redis.hget, name, key)
            
            if result and isinstance(result, str):
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get hash field {name}:{key}: {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Get all hash fields and values.
        
        Args:
            name: Hash name
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            result = await self._execute_operation(self.redis.hgetall, name)
            
            # Deserialize values
            deserialized = {}
            for field, value in result.items():
                if isinstance(value, str):
                    try:
                        deserialized[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        deserialized[field] = value
                else:
                    deserialized[field] = value
            
            return deserialized
            
        except Exception as e:
            self.logger.error(f"Failed to get all hash fields {name}: {e}")
            return {}
    
    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of subscribers that received the message
        """
        try:
            # Serialize message if needed
            if not isinstance(message, (str, int, float, bytes)):
                message = json.dumps(message, default=str)
            
            result = await self._execute_operation(self.redis.publish, channel, message)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to publish to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, channels: List[str]) -> Any:
        """
        Subscribe to channels.
        
        Args:
            channels: List of channel names
            
        Returns:
            PubSub object
        """
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channels {channels}: {e}")
            raise
    
    async def cache_market_data(
        self,
        symbol: str,
        data: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """
        Cache market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Market data dictionary
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        key = f"market_data:{symbol}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary or None
        """
        key = f"market_data:{symbol}"
        return await self.get(key, default=None)
    
    async def cache_prediction(
        self,
        symbol: str,
        prediction: Dict[str, Any],
        ttl: int = 600
    ) -> bool:
        """
        Cache prediction for a symbol.
        
        Args:
            symbol: Trading symbol
            prediction: Prediction dictionary
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        key = f"prediction:{symbol}"
        return await self.set(key, prediction, ttl=ttl)
    
    async def get_cached_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Prediction dictionary or None
        """
        key = f"prediction:{symbol}"
        return await self.get(key, default=None)
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        hit_rate = 0.0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'hit_rate': hit_rate,
            'success_rate': self.metrics['successful_operations'] / max(self.metrics['total_operations'], 1)
        }
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            'initialized': self.initialized,
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'max_connections': self.max_connections,
            'metrics': self.metrics
        }


# Global Redis cache instance
_redis_cache: Optional[RedisCache] = None


async def get_redis_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    global _redis_cache
    
    if _redis_cache is None:
        from src.core.config.loader import config_get
        
        host = config_get('redis.host', 'localhost')
        port = config_get('redis.port', 6379)
        db = config_get('redis.db', 0)
        password = config_get('redis.password', None)
        max_connections = config_get('redis.max_connections', 20)
        
        _redis_cache = RedisCache(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections
        )
        await _redis_cache.initialize()
    
    return _redis_cache


async def close_redis_cache() -> None:
    """Close the global Redis cache instance."""
    global _redis_cache
    
    if _redis_cache:
        await _redis_cache.close()
        _redis_cache = None


__all__ = [
    'RedisCache',
    'get_redis_cache',
    'close_redis_cache',
]