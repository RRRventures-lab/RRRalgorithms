from collections import deque
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from src.database import get_db, Client
from typing import Dict, List, Any, Optional, Callable
import functools
import os
import threading
import time


"""
Performance Monitoring Service

Tracks system performance metrics including:
- Trading signal latency
- API response times
- Database query performance
- System resource utilization
"""


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)


class PerformanceMonitor:
    """
    Monitor system performance and track metrics

    Features:
    - Measure function execution time
    - Track API call latency
    - Monitor throughput
    - Alert on performance degradation
    """

    def __init__(self):
        """Initialize performance monitor"""
        supabase_url = os.getenv("DATABASE_PATH")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment")

        self.supabase: Client = get_db()

        # Performance thresholds
        self.signal_latency_threshold_ms = 100  # Target <100ms for trading signals
        self.api_latency_threshold_ms = 1000  # Alert if API calls >1s
        self.db_query_threshold_ms = 500  # Alert if DB queries >500ms

        # In-memory metrics store (rolling window)
        self._metrics_lock = threading.Lock()
        self._recent_metrics = deque(maxlen=1000)  # Keep last 1000 measurements

        # Batch insert buffer
        self._batch_buffer: List[Dict[str, Any]] = []
        self._batch_size = 20

    def measure_execution_time(self, func: Callable) -> Callable:
        """
        Decorator to measure function execution time

        Usage:
            @monitor.measure_execution_time
            def my_function():
                pass
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000

            # Log the metric
            self.log_metric(
                endpoint=f"{func.__module__}.{func.__name__}",
                latency_ms=latency_ms,
                status_code=200,
                metadata={"function": func.__name__}
            )

            return result

        return wrapper

    def measure_latency(self, operation_name: str) -> 'LatencyContext':
        """
        Context manager to measure operation latency

        Usage:
            with monitor.measure_latency("api_call"):
                # Your code here
                pass
        """
        return LatencyContext(self, operation_name)

    def log_metric(
        self,
        endpoint: str,
        latency_ms: float,
        status_code: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a performance metric

        Args:
            endpoint: API endpoint or operation name
            latency_ms: Latency in milliseconds
            status_code: HTTP status code or custom status
            metadata: Additional metadata
        """
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "status_code": status_code,
            "metadata": metadata or {}
        }

        # Store in memory
        with self._metrics_lock:
            self._recent_metrics.append(metric)
            self._batch_buffer.append(metric)

            # Batch insert to Supabase
            if len(self._batch_buffer) >= self._batch_size:
                self._flush_batch()

    def _flush_batch(self):
        """Flush batch of metrics to Supabase"""
        if not self._batch_buffer:
            return

        try:
            self.supabase.table("api_usage").insert(self._batch_buffer).execute()
            self._batch_buffer = []
        except Exception as e:
            print(f"Error flushing metrics batch: {e}")
            self._batch_buffer = []

    @lru_cache(maxsize=128)

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics from memory

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of recent metrics
        """
        with self._metrics_lock:
            return list(self._recent_metrics)[-limit:]

    @lru_cache(maxsize=128)

    def get_average_latency(self, endpoint: Optional[str] = None, minutes: int = 5) -> float:
        """
        Calculate average latency for an endpoint

        Args:
            endpoint: Specific endpoint (None for all)
            minutes: Time window in minutes

        Returns:
            Average latency in milliseconds
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._metrics_lock:
            relevant_metrics = [
                m for m in self._recent_metrics
                if (endpoint is None or m["endpoint"] == endpoint)
                and datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]

        if not relevant_metrics:
            return 0.0

        total_latency = sum(m["latency_ms"] for m in relevant_metrics)
        return total_latency / len(relevant_metrics)

    @lru_cache(maxsize=128)

    def get_throughput(self, endpoint: Optional[str] = None, minutes: int = 1) -> float:
        """
        Calculate throughput (requests per second)

        Args:
            endpoint: Specific endpoint (None for all)
            minutes: Time window in minutes

        Returns:
            Requests per second
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._metrics_lock:
            relevant_metrics = [
                m for m in self._recent_metrics
                if (endpoint is None or m["endpoint"] == endpoint)
                and datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]

        if not relevant_metrics:
            return 0.0

        return len(relevant_metrics) / (minutes * 60)

    @lru_cache(maxsize=128)

    def get_error_rate(self, endpoint: Optional[str] = None, minutes: int = 5) -> float:
        """
        Calculate error rate (percentage)

        Args:
            endpoint: Specific endpoint (None for all)
            minutes: Time window in minutes

        Returns:
            Error rate as percentage (0-100)
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._metrics_lock:
            relevant_metrics = [
                m for m in self._recent_metrics
                if (endpoint is None or m["endpoint"] == endpoint)
                and datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]

        if not relevant_metrics:
            return 0.0

        error_count = sum(1 for m in relevant_metrics if m["status_code"] >= 400)
        return (error_count / len(relevant_metrics)) * 100

    @lru_cache(maxsize=128)

    def get_percentiles(
        self,
        endpoint: Optional[str] = None,
        minutes: int = 5,
        percentiles: List[int] = [50, 95, 99]
    ) -> Dict[int, float]:
        """
        Calculate latency percentiles

        Args:
            endpoint: Specific endpoint (None for all)
            minutes: Time window in minutes
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary of percentile to latency
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._metrics_lock:
            relevant_metrics = [
                m for m in self._recent_metrics
                if (endpoint is None or m["endpoint"] == endpoint)
                and datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]

        if not relevant_metrics:
            return {p: 0.0 for p in percentiles}

        latencies = sorted([m["latency_ms"] for m in relevant_metrics])
        result = {}

        for p in percentiles:
            idx = int(len(latencies) * p / 100)
            result[p] = latencies[min(idx, len(latencies) - 1)]

        return result

    def check_performance_health(self) -> Dict[str, Any]:
        """
        Check overall performance health

        Returns:
            Health status with metrics and issues
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }

        # Calculate key metrics
        avg_latency = self.get_average_latency(minutes=5)
        throughput = self.get_throughput(minutes=1)
        error_rate = self.get_error_rate(minutes=5)
        percentiles = self.get_percentiles(minutes=5)

        health_status["metrics"]["avg_latency_ms"] = avg_latency
        health_status["metrics"]["throughput_rps"] = throughput
        health_status["metrics"]["error_rate_pct"] = error_rate
        health_status["metrics"]["p50_latency_ms"] = percentiles[50]
        health_status["metrics"]["p95_latency_ms"] = percentiles[95]
        health_status["metrics"]["p99_latency_ms"] = percentiles[99]

        # Check thresholds
        if avg_latency > self.api_latency_threshold_ms:
            health_status["status"] = "degraded"
            health_status["issues"].append(
                f"High average latency: {avg_latency:.2f}ms"
            )

        if error_rate > 10:  # >10% error rate
            health_status["status"] = "degraded"
            health_status["issues"].append(
                f"High error rate: {error_rate:.2f}%"
            )

        if percentiles[99] > self.api_latency_threshold_ms * 2:
            health_status["status"] = "degraded"
            health_status["issues"].append(
                f"High P99 latency: {percentiles[99]:.2f}ms"
            )

        return health_status

    def flush(self):
        """Flush any pending metrics to Supabase"""
        with self._metrics_lock:
            self._flush_batch()


class LatencyContext:
    """Context manager for measuring latency"""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.metadata = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        latency_ms = (end_time - self.start_time) * 1000

        status_code = 200 if exc_type is None else 500

        if exc_type:
            self.metadata["error"] = str(exc_val)

        self.monitor.log_metric(
            endpoint=self.operation_name,
            latency_ms=latency_ms,
            status_code=status_code,
            metadata=self.metadata
        )

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the measurement"""
        self.metadata[key] = value


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


@lru_cache(maxsize=128)


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def main():
    """Test performance monitoring"""
    monitor = get_performance_monitor()

    print("=" * 60)
    print("Performance Monitoring Test")
    print("=" * 60)

    # Test function decorator
    @monitor.measure_execution_time
    def slow_function():
        time.sleep(0.1)
        return "done"

    print("\nTesting function decorator...")
    slow_function()

    # Test context manager
    print("Testing context manager...")
    with monitor.measure_latency("test_operation") as ctx:
        time.sleep(0.05)
        ctx.add_metadata("test_key", "test_value")

    # Log some test metrics
    print("Logging test metrics...")
    for i in range(10):
        monitor.log_metric(
            endpoint="test_endpoint",
            latency_ms=50 + i * 10,
            status_code=200 if i < 8 else 500
        )

    # Get performance health
    time.sleep(0.1)
    health = monitor.check_performance_health()

    print("\nPerformance Health:")
    print(f"Status: {health['status'].upper()}")
    print(f"\nMetrics:")
    for key, value in health['metrics'].items():
        print(f"  {key}: {value:.2f}")

    if health['issues']:
        print(f"\nIssues:")
        for issue in health['issues']:
            print(f"  - {issue}")

    # Flush remaining metrics
    monitor.flush()
    print("\nMetrics flushed to Supabase.")


if __name__ == "__main__":
    main()
