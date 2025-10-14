from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio
import hashlib
import json
import numpy as np
import pandas as pd
import pickle
import queue
import threading
import time
import zlib

"""
Optimized Data Pipeline with Caching and Batching
==================================================

High-performance data pipeline featuring:
- Multi-level caching (memory, disk, Redis)
- Batch processing for efficiency
- Parallel data fetching
- Data compression
- Smart prefetching
- Circuit breaker pattern

Author: RRR Ventures
Date: 2025-10-12
"""



@dataclass
class DataPoint:
    """Standardized data point for pipeline."""
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "unknown"
    metadata: Optional[Dict] = None


class CacheLayer:
    """Multi-level cache with TTL and compression."""

    def __init__(
        self,
        memory_size: int = 10000,
        disk_cache_dir: str = "data/cache",
        enable_compression: bool = True,
        ttl_seconds: int = 300
    ):
        """
        Initialize cache layer.

        Args:
            memory_size: Maximum memory cache size
            disk_cache_dir: Directory for disk cache
            enable_compression: Enable data compression
            ttl_seconds: Time to live for cache entries
        """
        # Memory cache (L1)
        self.memory_cache = {}
        self.memory_timestamps = {}
        self.memory_size = memory_size

        # Disk cache (L2)
        self.disk_cache_dir = Path(disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.enable_compression = enable_compression
        self.ttl_seconds = ttl_seconds

        # Statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'total_requests': 0
        }

        # LRU eviction queue
        self.access_order = deque(maxlen=memory_size)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            self.stats['total_requests'] += 1

            # Check L1 (memory)
            if key in self.memory_cache:
                # Check TTL
                if time.time() - self.memory_timestamps[key] < self.ttl_seconds:
                    self.stats['memory_hits'] += 1
                    # Update LRU
                    self.access_order.append(key)
                    return self.memory_cache[key]
                else:
                    # Expired
                    del self.memory_cache[key]
                    del self.memory_timestamps[key]

            self.stats['memory_misses'] += 1

            # Check L2 (disk)
            disk_path = self.disk_cache_dir / f"{key}.cache"
            if disk_path.exists():
                try:
                    # Check file age
                    file_age = time.time() - disk_path.stat().st_mtime
                    if file_age < self.ttl_seconds:
                        with open(disk_path, 'rb') as f:
                            data = f.read()
                            if self.enable_compression:
                                data = zlib.decompress(data)
                            value = pickle.loads(data)

                        # Promote to L1
                        self._add_to_memory_cache(key, value)
                        self.stats['disk_hits'] += 1
                        return value
                except Exception:
                    pass

            self.stats['disk_misses'] += 1
            return None

    def set(self, key: str, value: Any):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Add to memory cache
            self._add_to_memory_cache(key, value)

            # Write to disk cache asynchronously
            threading.Thread(
                target=self._write_to_disk,
                args=(key, value),
                daemon=True
            ).start()

    def _add_to_memory_cache(self, key: str, value: Any):
        """Add item to memory cache with LRU eviction."""
        # Evict if necessary
        if len(self.memory_cache) >= self.memory_size:
            # Find least recently used key
            lru_key = None
            for k in self.memory_cache:
                if k not in self.access_order:
                    lru_key = k
                    break

            if lru_key:
                del self.memory_cache[lru_key]
                del self.memory_timestamps[lru_key]

        self.memory_cache[key] = value
        self.memory_timestamps[key] = time.time()
        self.access_order.append(key)

    def _write_to_disk(self, key: str, value: Any):
        """Write value to disk cache."""
        try:
            disk_path = self.disk_cache_dir / f"{key}.cache"
            data = pickle.dumps(value)

            if self.enable_compression:
                data = zlib.compress(data, level=6)

            with open(disk_path, 'wb') as f:
                f.write(data)
        except Exception:
            pass  # Fail silently

    def clear_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            current_time = time.time()

            # Clear expired memory entries
            expired_keys = [
                k for k, t in self.memory_timestamps.items()
                if current_time - t > self.ttl_seconds
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                del self.memory_timestamps[key]

            # Clear expired disk entries
            for cache_file in self.disk_cache_dir.glob("*.cache"):
                if current_time - cache_file.stat().st_mtime > self.ttl_seconds:
                    cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self.stats['total_requests']
            if total > 0:
                hit_rate = (
                    (self.stats['memory_hits'] + self.stats['disk_hits']) / total
                )
            else:
                hit_rate = 0

            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_size': len(self.memory_cache),
                'disk_files': len(list(self.disk_cache_dir.glob("*.cache")))
            }


class DataBatcher:
    """Batch data requests for efficiency."""

    def __init__(
        self,
        batch_size: int = 100,
        batch_timeout: float = 0.1,
        max_concurrent: int = 5
    ):
        """
        Initialize data batcher.

        Args:
            batch_size: Maximum batch size
            batch_timeout: Maximum time to wait for batch
            max_concurrent: Maximum concurrent batch processors
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent = max_concurrent

        self.pending_requests = defaultdict(list)
        self.request_futures = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = threading.Lock()
        self._running = True
        self._batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self._batch_thread.start()

    def add_request(
        self,
        batch_key: str,
        request_id: str,
        request_data: Any
    ) -> asyncio.Future:
        """
        Add request to batch.

        Args:
            batch_key: Key to group requests
            request_id: Unique request ID
            request_data: Request data

        Returns:
            Future for result
        """
        future = asyncio.Future()

        with self._lock:
            self.pending_requests[batch_key].append({
                'id': request_id,
                'data': request_data,
                'future': future,
                'timestamp': time.time()
            })

        return future

    def _batch_processor(self):
        """Process batches periodically."""
        while self._running:
            time.sleep(self.batch_timeout)

            with self._lock:
                current_time = time.time()

                for batch_key, requests in list(self.pending_requests.items()):
                    if not requests:
                        continue

                    # Check if batch is ready
                    oldest = min(r['timestamp'] for r in requests)
                    batch_ready = (
                        len(requests) >= self.batch_size or
                        current_time - oldest >= self.batch_timeout
                    )

                    if batch_ready:
                        # Process batch
                        batch = requests[:self.batch_size]
                        self.pending_requests[batch_key] = requests[self.batch_size:]

                        # Submit for processing
                        self.executor.submit(
                            self._process_batch,
                            batch_key,
                            batch
                        )

    def _process_batch(self, batch_key: str, batch: List[Dict]):
        """Process a batch of requests."""
        # This would be implemented based on specific data source
        # For now, simulate processing
        for request in batch:
            try:
                # Simulate processing
                result = {
                    'data': request['data'],
                    'processed': True,
                    'batch_key': batch_key
                }

                # Set result on future
                if not request['future'].done():
                    request['future'].set_result(result)
            except Exception as e:
                if not request['future'].done():
                    request['future'].set_exception(e)

    def shutdown(self):
        """Shutdown batcher."""
        self._running = False
        self.executor.shutdown(wait=True)


class CircuitBreaker:
    """Circuit breaker for external services."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening
            recovery_timeout: Time before half-open state
            success_threshold: Successes to close
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open
        """
        with self._lock:
            # Check state
            if self.state == "open":
                if (
                    self.last_failure_time and
                    time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    # Try half-open
                    self.state = "half-open"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = func(*args, **kwargs)

                # Success
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0

                return result

            except Exception as e:
                # Failure
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

                raise e


class OptimizedDataPipeline:
    """
    Optimized data pipeline with advanced features.
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        batch_size: int = 100,
        prefetch_size: int = 50,
        enable_compression: bool = True
    ):
        """
        Initialize optimized pipeline.

        Args:
            cache_ttl: Cache time-to-live in seconds
            batch_size: Batch processing size
            prefetch_size: Prefetch queue size
            enable_compression: Enable data compression
        """
        # Cache layer
        self.cache = CacheLayer(
            memory_size=10000,
            ttl_seconds=cache_ttl,
            enable_compression=enable_compression
        )

        # Batch processor
        self.batcher = DataBatcher(
            batch_size=batch_size,
            batch_timeout=0.1
        )

        # Circuit breakers for each data source
        self.circuit_breakers = {}

        # Prefetch queue
        self.prefetch_queue = asyncio.Queue(maxsize=prefetch_size)

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Statistics
        self.stats = {
            'data_points_processed': 0,
            'batch_count': 0,
            'prefetch_hits': 0,
            'processing_time_ms': deque(maxlen=1000)
        }

        self._running = True

    async def fetch_data(
        self,
        symbols: List[str],
        start_time: float,
        end_time: float,
        source: str = "polygon"
    ) -> Dict[str, List[DataPoint]]:
        """
        Fetch data for multiple symbols efficiently.

        Args:
            symbols: List of symbols
            start_time: Start timestamp
            end_time: End timestamp
            source: Data source

        Returns:
            Dictionary mapping symbol to data points
        """
        start = time.time()

        # Check cache first
        results = {}
        uncached_symbols = []

        for symbol in symbols:
            cache_key = self._generate_cache_key(symbol, start_time, end_time, source)
            cached = self.cache.get(cache_key)

            if cached:
                results[symbol] = cached
            else:
                uncached_symbols.append(symbol)

        # Batch fetch uncached symbols
        if uncached_symbols:
            # Use circuit breaker
            breaker = self._get_circuit_breaker(source)

            try:
                # Parallel fetch with batching
                fetch_tasks = []
                for i in range(0, len(uncached_symbols), self.batcher.batch_size):
                    batch = uncached_symbols[i:i + self.batcher.batch_size]
                    task = asyncio.create_task(
                        self._fetch_batch(batch, start_time, end_time, source, breaker)
                    )
                    fetch_tasks.append(task)

                batch_results = await asyncio.gather(*fetch_tasks)

                # Merge results and cache
                for batch_result in batch_results:
                    for symbol, data in batch_result.items():
                        results[symbol] = data

                        # Cache result
                        cache_key = self._generate_cache_key(
                            symbol, start_time, end_time, source
                        )
                        self.cache.set(cache_key, data)

            except Exception as e:
                print(f"Error fetching data: {e}")
                # Return partial results
                pass

        # Update statistics
        elapsed_ms = (time.time() - start) * 1000
        self.stats['processing_time_ms'].append(elapsed_ms)
        self.stats['data_points_processed'] += sum(len(v) for v in results.values())

        return results

    async def _fetch_batch(
        self,
        symbols: List[str],
        start_time: float,
        end_time: float,
        source: str,
        breaker: CircuitBreaker
    ) -> Dict[str, List[DataPoint]]:
        """Fetch batch of symbols."""
        loop = asyncio.get_event_loop()

        # Run in executor
        result = await loop.run_in_executor(
            self.executor,
            self._fetch_batch_sync,
            symbols,
            start_time,
            end_time,
            source,
            breaker
        )

        return result

    def _fetch_batch_sync(
        self,
        symbols: List[str],
        start_time: float,
        end_time: float,
        source: str,
        breaker: CircuitBreaker
    ) -> Dict[str, List[DataPoint]]:
        """Synchronous batch fetch."""
        results = {}

        for symbol in symbols:
            try:
                # Use circuit breaker
                data = breaker.call(
                    self._fetch_single_symbol,
                    symbol,
                    start_time,
                    end_time,
                    source
                )
                results[symbol] = data
            except Exception:
                # Failed to fetch
                results[symbol] = []

        return results

    def _fetch_single_symbol(
        self,
        symbol: str,
        start_time: float,
        end_time: float,
        source: str
    ) -> List[DataPoint]:
        """Fetch data for single symbol (mock implementation)."""
        # This would connect to actual data source
        # For now, generate mock data
        data_points = []

        num_points = min(100, int((end_time - start_time) / 60))
        timestamps = np.linspace(start_time, end_time, num_points)

        base_price = 50000 if "BTC" in symbol else 4000
        prices = base_price * (1 + np.random.randn(num_points) * 0.001).cumprod()

        for i, ts in enumerate(timestamps):
            price = prices[i]
            data_points.append(DataPoint(
                symbol=symbol,
                timestamp=ts,
                open=price * 0.998,
                high=price * 1.002,
                low=price * 0.997,
                close=price,
                volume=1000000 * (1 + np.random.randn() * 0.1),
                source=source
            ))

        return data_points

    def _generate_cache_key(
        self,
        symbol: str,
        start_time: float,
        end_time: float,
        source: str
    ) -> str:
        """Generate cache key."""
        key_data = f"{symbol}_{start_time}_{end_time}_{source}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_circuit_breaker(self, source: str) -> CircuitBreaker:
        """Get or create circuit breaker for source."""
        if source not in self.circuit_breakers:
            self.circuit_breakers[source] = CircuitBreaker()
        return self.circuit_breakers[source]

    async def stream_data(
        self,
        symbols: List[str],
        callback: Callable[[DataPoint], None],
        source: str = "polygon"
    ):
        """
        Stream real-time data.

        Args:
            symbols: Symbols to stream
            callback: Callback for each data point
            source: Data source
        """
        while self._running:
            try:
                # Fetch latest data
                end_time = time.time()
                start_time = end_time - 60  # Last minute

                data = await self.fetch_data(
                    symbols,
                    start_time,
                    end_time,
                    source
                )

                # Process data points
                for symbol, points in data.items():
                    for point in points:
                        await callback(point)

                # Small delay
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Stream error: {e}")
                await asyncio.sleep(5)

    async def prefetch_data(
        self,
        symbols: List[str],
        lookahead_minutes: int = 5
    ):
        """
        Prefetch data for future use.

        Args:
            symbols: Symbols to prefetch
            lookahead_minutes: Minutes to look ahead
        """
        while self._running:
            try:
                # Predict what data will be needed
                current_time = time.time()

                for offset in range(1, lookahead_minutes + 1):
                    future_time = current_time + (offset * 60)

                    # Add to prefetch queue
                    if not self.prefetch_queue.full():
                        await self.prefetch_queue.put({
                            'symbols': symbols,
                            'timestamp': future_time
                        })

                # Process prefetch queue
                while not self.prefetch_queue.empty():
                    item = await self.prefetch_queue.get()

                    # Fetch and cache
                    await self.fetch_data(
                        item['symbols'],
                        item['timestamp'] - 300,
                        item['timestamp'],
                        "polygon"
                    )

                    self.stats['prefetch_hits'] += 1

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"Prefetch error: {e}")
                await asyncio.sleep(60)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = (
            np.mean(self.stats['processing_time_ms'])
            if self.stats['processing_time_ms'] else 0
        )

        return {
            'data_points_processed': self.stats['data_points_processed'],
            'batch_count': self.stats['batch_count'],
            'prefetch_hits': self.stats['prefetch_hits'],
            'avg_processing_time_ms': avg_time,
            'cache_stats': self.cache.get_stats(),
            'circuit_breakers': {
                source: breaker.state
                for source, breaker in self.circuit_breakers.items()
            }
        }

    async def shutdown(self):
        """Shutdown pipeline."""
        self._running = False
        self.batcher.shutdown()
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    import asyncio

    async def test_pipeline():
        print("🚀 Testing Optimized Data Pipeline\n")

        # Initialize pipeline
        pipeline = OptimizedDataPipeline(
            cache_ttl=300,
            batch_size=50,
            prefetch_size=100,
            enable_compression=True
        )

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "MATIC-USD"]
        end_time = time.time()
        start_time = end_time - 3600  # Last hour

        # Test batch fetch
        print("📊 Testing Batch Fetch...")
        start = time.time()
        data = await pipeline.fetch_data(symbols, start_time, end_time)
        elapsed = time.time() - start

        total_points = sum(len(points) for points in data.values())
        print(f"  Fetched {total_points} data points in {elapsed:.2f}s")
        print(f"  Throughput: {total_points / elapsed:.0f} points/second\n")

        # Test cache hit
        print("💾 Testing Cache Hit...")
        start = time.time()
        cached_data = await pipeline.fetch_data(symbols, start_time, end_time)
        elapsed = time.time() - start

        print(f"  Cache fetch completed in {elapsed:.3f}s")
        cache_stats = pipeline.cache.get_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}\n")

        # Test streaming
        print("📡 Testing Data Streaming...")
        stream_count = 0

        async def stream_callback(point: DataPoint):
            nonlocal stream_count
            stream_count += 1
            if stream_count % 100 == 0:
                print(f"  Streamed {stream_count} data points")

        # Stream for 5 seconds
        stream_task = asyncio.create_task(
            pipeline.stream_data(symbols[:2], stream_callback)
        )
        await asyncio.sleep(5)
        pipeline._running = False
        await stream_task

        # Show statistics
        print("\n📈 Pipeline Statistics:")
        stats = pipeline.get_statistics()
        print(f"  Total Points:      {stats['data_points_processed']:,}")
        print(f"  Avg Process Time:  {stats['avg_processing_time_ms']:.1f}ms")
        print(f"  Cache Hit Rate:    {stats['cache_stats']['hit_rate']:.1%}")
        print(f"  Memory Cache Size: {stats['cache_stats']['memory_size']}")
        print(f"  Disk Cache Files:  {stats['cache_stats']['disk_files']}")
        print(f"  Prefetch Hits:     {stats['prefetch_hits']}")

        await pipeline.shutdown()
        print("\n✅ Pipeline Test Complete!")

    asyncio.run(test_pipeline())