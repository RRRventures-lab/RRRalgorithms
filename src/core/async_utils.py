"""
Async Utilities
===============

Optimized async operations and utilities.

Author: RRR Ventures
Date: 2025-10-12
"""

import asyncio
from typing import Any, List, Callable, TypeVar, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import functools
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def gather_with_limit(
    *coroutines,
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Similar to asyncio.gather but with concurrency limit.

    Args:
        *coroutines: Coroutines to execute
        limit: Maximum concurrent executions
        return_exceptions: Whether to return exceptions as results

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited_coro(coro) for coro in coroutines],
        return_exceptions=return_exceptions
    )


async def retry_async(
    func: Callable,
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Function arguments
        max_retries: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries} attempts failed")

    raise last_exception


class AsyncBatcher:
    """Batch async operations for efficiency."""

    def __init__(self,
                 batch_func: Callable,
                 batch_size: int = 100,
                 batch_timeout: float = 1.0):
        """
        Initialize async batcher.

        Args:
            batch_func: Async function to process batches
            batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch
        """
        self.batch_func = batch_func
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._batch: List[Any] = []
        self._futures: List[asyncio.Future] = []
        self._lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None

    async def add(self, item: Any) -> Any:
        """Add item to batch and get result."""
        future = asyncio.Future()

        async with self._lock:
            self._batch.append(item)
            self._futures.append(future)

            # Start timer if not running
            if self._timer_task is None:
                self._timer_task = asyncio.create_task(self._timer())

            # Process if batch is full
            if len(self._batch) >= self.batch_size:
                await self._process_batch()

        return await future

    async def _timer(self):
        """Timer to process batch after timeout."""
        await asyncio.sleep(self.batch_timeout)
        async with self._lock:
            if self._batch:
                await self._process_batch()
            self._timer_task = None

    async def _process_batch(self):
        """Process current batch."""
        if not self._batch:
            return

        batch = self._batch.copy()
        futures = self._futures.copy()

        self._batch.clear()
        self._futures.clear()

        try:
            # Process batch
            results = await self.batch_func(batch)

            # Set results
            for future, result in zip(futures, results):
                future.set_result(result)

        except Exception as e:
            # Set exceptions
            for future in futures:
                future.set_exception(e)


class AsyncCache:
    """Async LRU cache with TTL."""

    def __init__(self, maxsize: int = 128, ttl: float = 60.0):
        """
        Initialize async cache.

        Args:
            maxsize: Maximum cache size
            ttl: Time to live in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[Any, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]

                # Check TTL
                if time.time() - timestamp < self.ttl:
                    # Move to end (LRU)
                    del self._cache[key]
                    self._cache[key] = (value, timestamp)
                    return value
                else:
                    # Expired
                    del self._cache[key]

        return None

    async def set(self, key: Any, value: Any):
        """Set value in cache."""
        async with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.maxsize and key not in self._cache:
                # Remove first (oldest) item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = (value, time.time())

    def cache_async(self, ttl: Optional[float] = None):
        """Decorator for async functions."""
        cache_ttl = ttl or self.ttl

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                key = (func.__name__, args, tuple(sorted(kwargs.items())))

                # Check cache
                result = await self.get(key)
                if result is not None:
                    return result

                # Call function
                result = await func(*args, **kwargs)

                # Store in cache
                await self.set(key, result)

                return result

            return wrapper

        return decorator


# Thread pool for CPU-bound operations
_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run CPU-bound function in thread pool.

    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _thread_pool,
        functools.partial(func, *args, **kwargs)
    )
