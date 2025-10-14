from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from functools import wraps
from typing import Dict, List, Optional, Any, Callable
import asyncio
import cProfile
import io
import json
import numpy as np
import os
import pstats
import psutil
import threading
import time
import tracemalloc


"""
Advanced Performance Monitoring and Profiling System
=====================================================

Comprehensive performance monitoring featuring:
- Real-time metrics collection
- Performance profiling
- Bottleneck detection
- Memory tracking
- Resource utilization monitoring
- Automated alerting
- Grafana/Prometheus integration

Author: RRR Ventures
Date: 2025-10-12
"""



@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int
    processes: int


@dataclass
class OperationMetrics:
    """Individual operation metrics."""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentMetrics:
    """Component-level metrics."""
    name: str
    operations_count: int = 0
    errors_count: int = 0
    avg_latency_ms: float = 0
    p50_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    throughput_per_sec: float = 0
    error_rate: float = 0

    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Advanced performance monitoring system.
    """

    def __init__(
        self,
        enable_profiling: bool = False,
        enable_memory_tracking: bool = False,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.

        Args:
            enable_profiling: Enable CPU profiling
            enable_memory_tracking: Enable memory tracking
            alert_thresholds: Alert threshold configuration
        """
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80,
            'memory_percent': 80,
            'error_rate': 0.05,
            'p99_latency_ms': 1000
        }

        # Metrics storage
        self.system_metrics = deque(maxlen=1000)
        self.component_metrics = defaultdict(ComponentMetrics)
        self.operation_history = deque(maxlen=10000)

        # Profiling
        self.profiler = cProfile.Profile() if enable_profiling else None

        # Memory tracking
        if enable_memory_tracking:
            tracemalloc.start()

        # Background monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            'monitoring_started': time.time(),
            'total_operations': 0,
            'total_errors': 0,
            'alerts_triggered': 0
        }

    @contextmanager
    def measure_operation(
        self,
        operation_name: str,
        component: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Context manager to measure operation performance.

        Args:
            operation_name: Name of operation
            component: Component name
            metadata: Additional metadata

        Example:
            with monitor.measure_operation("fetch_data", "data_pipeline"):
                # Operation code
                pass
        """
        start_time = time.perf_counter()
        error = None
        success = True

        try:
            yield

        except Exception as e:
            error = str(e)
            success = False
            raise

        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Record operation
            operation = OperationMetrics(
                name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error=error,
                metadata=metadata or {}
            )

            self._record_operation(operation, component)

    def measure_async(
        self,
        operation_name: str,
        component: Optional[str] = None
    ):
        """
        Decorator for measuring async operations.

        Args:
            operation_name: Name of operation
            component: Component name

        Example:
            @monitor.measure_async("process_symbol", "trading_engine")
            async def process_symbol():
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.measure_operation(operation_name, component):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def measure_sync(
        self,
        operation_name: str,
        component: Optional[str] = None
    ):
        """
        Decorator for measuring sync operations.

        Args:
            operation_name: Name of operation
            component: Component name
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure_operation(operation_name, component):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _record_operation(self, operation: OperationMetrics, component: Optional[str]):
        """Record operation metrics."""
        self.operation_history.append(operation)
        self.stats['total_operations'] += 1

        if not operation.success:
            self.stats['total_errors'] += 1

        # Update component metrics
        if component:
            comp_metrics = self.component_metrics[component]
            comp_metrics.operations_count += 1

            if not operation.success:
                comp_metrics.errors_count += 1

            # Update latencies
            comp_metrics.latencies.append(operation.duration_ms)

            # Recalculate statistics
            self._update_component_stats(comp_metrics)

    def _update_component_stats(self, metrics: ComponentMetrics):
        """Update component statistics."""
        if not metrics.latencies:
            return

        latencies = list(metrics.latencies)

        # Calculate percentiles
        metrics.avg_latency_ms = np.mean(latencies)
        metrics.p50_latency_ms = np.percentile(latencies, 50)
        metrics.p95_latency_ms = np.percentile(latencies, 95)
        metrics.p99_latency_ms = np.percentile(latencies, 99)

        # Calculate throughput
        time_window = time.time() - metrics.last_update
        if time_window > 0:
            metrics.throughput_per_sec = metrics.operations_count / time_window

        # Calculate error rate
        if metrics.operations_count > 0:
            metrics.error_rate = metrics.errors_count / metrics.operations_count

        # Check alerts
        self._check_component_alerts(metrics)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)

                # Check system alerts
                self._check_system_alerts(metrics)

                # Sleep for 1 second
                time.sleep(1)

            except Exception as e:
                print(f"Monitor error: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        process = psutil.Process()

        # Get network and disk I/O
        net_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            memory_mb=process.memory_info().rss / (1024 * 1024),
            disk_io_read_mb=disk_io.read_bytes / (1024 * 1024),
            disk_io_write_mb=disk_io.write_bytes / (1024 * 1024),
            network_sent_mb=net_io.bytes_sent / (1024 * 1024),
            network_recv_mb=net_io.bytes_recv / (1024 * 1024),
            open_files=len(process.open_files()),
            threads=process.num_threads(),
            processes=len(psutil.pids())
        )

    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system-level alerts."""
        alerts = []

        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'system',
                'severity': 'high',
                'metric': 'cpu_percent',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%'
            })

        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'system',
                'severity': 'high',
                'metric': 'memory_percent',
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f'High memory usage: {metrics.memory_percent:.1f}%'
            })

        for alert in alerts:
            self._trigger_alert(alert)

    def _check_component_alerts(self, metrics: ComponentMetrics):
        """Check component-level alerts."""
        alerts = []

        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'component',
                'severity': 'high',
                'component': metrics.name,
                'metric': 'error_rate',
                'value': metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate'],
                'message': f'High error rate in {metrics.name}: {metrics.error_rate:.1%}'
            })

        if metrics.p99_latency_ms > self.alert_thresholds['p99_latency_ms']:
            alerts.append({
                'type': 'component',
                'severity': 'medium',
                'component': metrics.name,
                'metric': 'p99_latency_ms',
                'value': metrics.p99_latency_ms,
                'threshold': self.alert_thresholds['p99_latency_ms'],
                'message': f'High P99 latency in {metrics.name}: {metrics.p99_latency_ms:.1f}ms'
            })

        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert to callbacks."""
        self.stats['alerts_triggered'] += 1

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def start_profiling(self):
        """Start CPU profiling."""
        if self.profiler:
            self.profiler.enable()

    def stop_profiling(self) -> str:
        """Stop profiling and return results."""
        if not self.profiler:
            return "Profiling not enabled"

        self.profiler.disable()

        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions

        return s.getvalue()

    @lru_cache(maxsize=128)

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get memory usage snapshot."""
        if not self.enable_memory_tracking:
            return {}

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        memory_info = {
            'total_mb': sum(stat.size for stat in top_stats) / (1024 * 1024),
            'top_allocations': []
        }

        for stat in top_stats[:10]:
            memory_info['top_allocations'].append({
                'file': stat.traceback.format()[0],
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            })

        return memory_info

    @lru_cache(maxsize=128)

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        if not self.system_metrics:
            return {}

        recent_metrics = list(self.system_metrics)[-60:]  # Last minute

        return {
            'cpu_percent_avg': np.mean([m.cpu_percent for m in recent_metrics]),
            'cpu_percent_max': np.max([m.cpu_percent for m in recent_metrics]),
            'memory_percent_avg': np.mean([m.memory_percent for m in recent_metrics]),
            'memory_mb_current': recent_metrics[-1].memory_mb if recent_metrics else 0,
            'threads_current': recent_metrics[-1].threads if recent_metrics else 0,
            'open_files_current': recent_metrics[-1].open_files if recent_metrics else 0
        }

    @lru_cache(maxsize=128)

    def get_component_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get component metrics summary."""
        summary = {}

        for name, metrics in self.component_metrics.items():
            summary[name] = {
                'operations_count': metrics.operations_count,
                'errors_count': metrics.errors_count,
                'error_rate': metrics.error_rate,
                'avg_latency_ms': metrics.avg_latency_ms,
                'p50_latency_ms': metrics.p50_latency_ms,
                'p95_latency_ms': metrics.p95_latency_ms,
                'p99_latency_ms': metrics.p99_latency_ms,
                'throughput_per_sec': metrics.throughput_per_sec
            }

        return summary

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # System metrics
        system = self.get_system_summary()
        for key, value in system.items():
            lines.append(f"system_{key} {value}")

        # Component metrics
        components = self.get_component_summary()
        for comp_name, metrics in components.items():
            for metric_name, value in metrics.items():
                lines.append(f'component_{metric_name}{{component="{comp_name}"}} {value}')

        # Overall stats
        lines.append(f"total_operations {self.stats['total_operations']}")
        lines.append(f"total_errors {self.stats['total_errors']}")
        lines.append(f"alerts_triggered {self.stats['alerts_triggered']}")

        return "\n".join(lines)

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE MONITORING REPORT")
        report.append("=" * 60)

        # Monitoring duration
        duration = (time.time() - self.stats['monitoring_started']) / 60
        report.append(f"\nMonitoring Duration: {duration:.1f} minutes")

        # Overall statistics
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"  Total Operations:  {self.stats['total_operations']:,}")
        report.append(f"  Total Errors:      {self.stats['total_errors']:,}")
        error_rate = (
            self.stats['total_errors'] / self.stats['total_operations']
            if self.stats['total_operations'] > 0 else 0
        )
        report.append(f"  Overall Error Rate: {error_rate:.2%}")
        report.append(f"  Alerts Triggered:  {self.stats['alerts_triggered']}")

        # System metrics
        system = self.get_system_summary()
        report.append(f"\n💻 System Resources:")
        report.append(f"  CPU Average:       {system.get('cpu_percent_avg', 0):.1f}%")
        report.append(f"  CPU Peak:          {system.get('cpu_percent_max', 0):.1f}%")
        report.append(f"  Memory Average:    {system.get('memory_percent_avg', 0):.1f}%")
        report.append(f"  Memory Current:    {system.get('memory_mb_current', 0):.1f} MB")
        report.append(f"  Threads:           {system.get('threads_current', 0)}")
        report.append(f"  Open Files:        {system.get('open_files_current', 0)}")

        # Component performance
        components = self.get_component_summary()
        if components:
            report.append(f"\n🔧 Component Performance:")
            for comp_name, metrics in components.items():
                report.append(f"\n  {comp_name}:")
                report.append(f"    Operations:      {metrics['operations_count']:,}")
                report.append(f"    Error Rate:      {metrics['error_rate']:.2%}")
                report.append(f"    Avg Latency:     {metrics['avg_latency_ms']:.1f}ms")
                report.append(f"    P95 Latency:     {metrics['p95_latency_ms']:.1f}ms")
                report.append(f"    P99 Latency:     {metrics['p99_latency_ms']:.1f}ms")
                report.append(f"    Throughput:      {metrics['throughput_per_sec']:.1f} ops/sec")

        # Memory snapshot
        if self.enable_memory_tracking:
            memory = self.get_memory_snapshot()
            if memory:
                report.append(f"\n💾 Memory Analysis:")
                report.append(f"  Total Allocated:   {memory['total_mb']:.1f} MB")
                report.append(f"  Top Allocations:")
                for alloc in memory['top_allocations'][:5]:
                    report.append(f"    {alloc['size_mb']:.1f} MB - {alloc['file']}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def shutdown(self):
        """Shutdown monitor."""
        self._monitoring = False

        if self.enable_memory_tracking:
            tracemalloc.stop()


# Global monitor instance
_monitor_instance: Optional[PerformanceMonitor] = None


@lru_cache(maxsize=128)


def get_performance_monitor(
    enable_profiling: bool = False,
    enable_memory: bool = False
) -> PerformanceMonitor:
    """
    Get singleton performance monitor.

    Args:
        enable_profiling: Enable CPU profiling
        enable_memory: Enable memory tracking

    Returns:
        Performance monitor instance
    """
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor(
            enable_profiling=enable_profiling,
            enable_memory_tracking=enable_memory
        )

    return _monitor_instance


if __name__ == "__main__":
    import asyncio

    print("🚀 Testing Performance Monitor\n")

    # Initialize monitor
    monitor = PerformanceMonitor(
        enable_profiling=True,
        enable_memory_tracking=True
    )

    # Add alert callback
    def alert_handler(alert):
        print(f"⚠️ ALERT: {alert['message']}")

    monitor.add_alert_callback(alert_handler)

    # Test synchronous operations
    @monitor.measure_sync("compute_heavy", "cpu_tasks")
    def heavy_computation():
        total = 0
        for i in range(1000000):
            total += i ** 2
        return total

    print("📊 Testing Sync Operations...")
    for _ in range(5):
        with monitor.measure_operation("test_operation", "test_component"):
            time.sleep(0.01)
            result = heavy_computation()

    # Test async operations
    @monitor.measure_async("async_fetch", "network_tasks")
    async def fetch_data():
        await asyncio.sleep(0.01)
        return {"data": "test"}

    async def test_async():
        print("\n⚡ Testing Async Operations...")
        tasks = []
        for _ in range(10):
            tasks.append(fetch_data())
        await asyncio.gather(*tasks)

    asyncio.run(test_async())

    # Generate some errors
    print("\n❌ Testing Error Tracking...")
    for _ in range(3):
        try:
            with monitor.measure_operation("error_operation", "test_component"):
                raise ValueError("Test error")
        except ValueError:
            pass

    # Wait for metrics collection
    time.sleep(2)

    # Show component summary
    print("\n📈 Component Summary:")
    components = monitor.get_component_summary()
    for comp_name, metrics in components.items():
        print(f"\n{comp_name}:")
        print(f"  Operations:    {metrics['operations_count']}")
        print(f"  Errors:        {metrics['errors_count']}")
        print(f"  Error Rate:    {metrics['error_rate']:.1%}")
        print(f"  Avg Latency:   {metrics['avg_latency_ms']:.1f}ms")
        print(f"  P99 Latency:   {metrics['p99_latency_ms']:.1f}ms")

    # Show system summary
    print("\n💻 System Summary:")
    system = monitor.get_system_summary()
    print(f"  CPU Average:   {system.get('cpu_percent_avg', 0):.1f}%")
    print(f"  Memory:        {system.get('memory_mb_current', 0):.1f}MB")

    # Generate report
    print("\n" + monitor.generate_report())

    monitor.shutdown()
    print("\n✅ Performance Monitor Test Complete!")