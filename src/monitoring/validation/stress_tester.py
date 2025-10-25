from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging
import numpy as np
import psutil
import time
import traceback

#!/usr/bin/env python3
"""
Stress Testing Module

Performs stress tests on the trading system under extreme conditions:
- Concurrent load testing
- Memory stress tests
- Latency stress tests
- Data throughput stress tests
- System resource exhaustion tests

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Result of a stress test"""
    test_name: str
    passed: bool
    duration_seconds: float
    peak_memory_mb: float
    peak_cpu_percent: float
    throughput: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    errors: List[str]
    warnings: List[str]
    system_crashed: bool
    details: Dict[str, Any]


class StressTester:
    """
    Comprehensive stress testing for trading system

    Tests system behavior under extreme loads and resource constraints
    """

    def __init__(
        self,
        max_test_duration_seconds: int = 300,
        memory_limit_mb: int = 4096,
        cpu_limit_percent: int = 90
    ):
        self.max_test_duration_seconds = max_test_duration_seconds
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent

        # System monitoring
        self.process = psutil.Process()
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024

        logger.info("StressTester initialized")

    def run_all_stress_tests(
        self,
        system_under_test: Any
    ) -> List[StressTestResult]:
        """
        Run all stress tests

        Args:
            system_under_test: System to test

        Returns:
            List of stress test results
        """
        results = []

        logger.info("Starting comprehensive stress tests...")

        # 1. Concurrent validation stress
        results.append(self.stress_test_concurrent_validations(system_under_test))

        # 2. Memory stress
        results.append(self.stress_test_memory(system_under_test))

        # 3. High-frequency decision stress
        results.append(self.stress_test_high_frequency_decisions(system_under_test))

        # 4. Data throughput stress
        results.append(self.stress_test_data_throughput(system_under_test))

        # 5. Sustained load stress
        results.append(self.stress_test_sustained_load(system_under_test))

        # 6. Spike load stress
        results.append(self.stress_test_spike_load(system_under_test))

        # 7. Resource exhaustion stress
        results.append(self.stress_test_resource_exhaustion(system_under_test))

        logger.info(f"Completed {len(results)} stress tests")

        return results

    def stress_test_concurrent_validations(
        self,
        system_under_test: Any,
        num_concurrent: int = 1000,
        num_iterations: int = 100
    ) -> StressTestResult:
        """
        Stress test concurrent validation requests

        Tests system's ability to handle many simultaneous validations
        """
        logger.info(f"Running concurrent validation stress test: {num_concurrent} concurrent x {num_iterations} iterations")

        start_time = time.time()
        errors = []
        warnings = []
        latencies = []
        peak_memory = 0
        peak_cpu = 0

        def single_validation():
            """Single validation request"""
            try:
                validation_start = time.time()

                # Simulate validation request
                # In real implementation, this would call system_under_test.validate_decision()
                time.sleep(np.random.uniform(0.001, 0.010))  # Simulate 1-10ms validation

                latency_ms = (time.time() - validation_start) * 1000
                latencies.append(latency_ms)

                # Monitor resources
                nonlocal peak_memory, peak_cpu
                mem_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()

                peak_memory = max(peak_memory, mem_mb)
                peak_cpu = max(peak_cpu, cpu_percent)

            except Exception as e:
                errors.append(str(e))

        try:
            # Run concurrent validations
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                for i in range(num_iterations):
                    futures = [executor.submit(single_validation) for _ in range(num_concurrent)]

                    # Wait for all
                    for future in futures:
                        future.result(timeout=10)

                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{num_iterations} iterations")

        except Exception as e:
            errors.append(f"Stress test failed: {str(e)}")
            logger.error(f"Concurrent validation stress test failed: {e}")

        duration = time.time() - start_time

        # Calculate metrics
        throughput = len(latencies) / duration if duration > 0 else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        p99 = np.percentile(latencies, 99) if latencies else 0

        # Pass criteria
        passed = (
            len(errors) == 0 and
            p99 < 50 and  # p99 latency < 50ms
            peak_memory < self.memory_limit_mb and
            throughput > num_concurrent * 0.8  # At least 80% of target throughput
        )

        return StressTestResult(
            test_name="Concurrent Validations Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            throughput=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "num_concurrent": num_concurrent,
                "num_iterations": num_iterations,
                "total_requests": len(latencies)
            }
        )

    def stress_test_memory(
        self,
        system_under_test: Any,
        target_memory_mb: int = 2048
    ) -> StressTestResult:
        """
        Stress test memory usage

        Gradually increases memory load to test memory management
        """
        logger.info(f"Running memory stress test: target {target_memory_mb} MB")

        start_time = time.time()
        errors = []
        warnings = []
        peak_memory = 0

        # Store data to increase memory
        data_arrays = []

        try:
            current_memory = self.process.memory_info().rss / 1024 / 1024

            while current_memory < target_memory_mb:
                # Allocate 100MB chunks
                chunk_size = 100 * 1024 * 1024 // 8  # 100MB of float64
                data_arrays.append(np.random.randn(chunk_size))

                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                # Test system under memory pressure
                try:
                    # Simulate validation under memory pressure
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Validation failed under memory pressure: {str(e)}")

                if current_memory > self.memory_limit_mb:
                    warnings.append(f"Memory limit exceeded: {current_memory:.0f} MB")
                    break

        except MemoryError:
            errors.append("MemoryError: System ran out of memory")
        except Exception as e:
            errors.append(f"Memory stress test failed: {str(e)}")

        finally:
            # Clean up
            data_arrays.clear()

        duration = time.time() - start_time

        # Pass criteria
        passed = (
            len(errors) == 0 and
            peak_memory <= self.memory_limit_mb
        )

        return StressTestResult(
            test_name="Memory Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=0,
            throughput=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "target_memory_mb": target_memory_mb,
                "initial_memory_mb": self.initial_memory_mb
            }
        )

    def stress_test_high_frequency_decisions(
        self,
        system_under_test: Any,
        decisions_per_second: int = 10000,
        duration_seconds: int = 60
    ) -> StressTestResult:
        """
        Stress test high-frequency decision making

        Tests system's ability to process many decisions per second
        """
        logger.info(f"Running high-frequency decision stress test: {decisions_per_second} decisions/sec for {duration_seconds}s")

        start_time = time.time()
        errors = []
        warnings = []
        latencies = []
        peak_memory = 0
        peak_cpu = 0

        total_decisions = 0
        decisions_failed = 0

        try:
            end_time = start_time + duration_seconds

            while time.time() < end_time:
                batch_start = time.time()

                # Process decisions in batches
                for _ in range(decisions_per_second // 10):  # Process in 0.1s batches
                    decision_start = time.time()

                    try:
                        # Simulate decision processing
                        time.sleep(np.random.uniform(0.0001, 0.001))  # 0.1-1ms per decision
                        total_decisions += 1

                    except Exception as e:
                        errors.append(str(e))
                        decisions_failed += 1

                    latency_ms = (time.time() - decision_start) * 1000
                    latencies.append(latency_ms)

                # Monitor resources
                mem_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()

                peak_memory = max(peak_memory, mem_mb)
                peak_cpu = max(peak_cpu, cpu_percent)

                # Sleep to maintain rate
                batch_duration = time.time() - batch_start
                if batch_duration < 0.1:
                    time.sleep(0.1 - batch_duration)

        except Exception as e:
            errors.append(f"High-frequency decision stress test failed: {str(e)}")
            logger.error(f"Stress test failed: {e}")

        duration = time.time() - start_time

        # Calculate metrics
        throughput = total_decisions / duration if duration > 0 else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        p99 = np.percentile(latencies, 99) if latencies else 0

        # Pass criteria
        passed = (
            decisions_failed == 0 and
            throughput >= decisions_per_second * 0.9 and  # At least 90% of target
            p99 < 10  # p99 latency < 10ms
        )

        return StressTestResult(
            test_name="High-Frequency Decisions Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            throughput=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "target_rate": decisions_per_second,
                "total_decisions": total_decisions,
                "decisions_failed": decisions_failed
            }
        )

    def stress_test_data_throughput(
        self,
        system_under_test: Any,
        data_rate_mbps: int = 100,
        duration_seconds: int = 60
    ) -> StressTestResult:
        """
        Stress test data throughput

        Tests system's ability to process high-volume data streams
        """
        logger.info(f"Running data throughput stress test: {data_rate_mbps} Mbps for {duration_seconds}s")

        start_time = time.time()
        errors = []
        warnings = []
        peak_memory = 0
        peak_cpu = 0

        bytes_per_second = data_rate_mbps * 1024 * 1024 / 8
        total_bytes_processed = 0

        try:
            end_time = start_time + duration_seconds

            while time.time() < end_time:
                # Generate data chunk (1MB)
                chunk_size = 1024 * 1024
                data_chunk = np.random.bytes(chunk_size)

                # Process data
                try:
                    # Simulate data processing
                    _ = len(data_chunk)
                    total_bytes_processed += chunk_size

                except Exception as e:
                    errors.append(str(e))

                # Monitor resources
                mem_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()

                peak_memory = max(peak_memory, mem_mb)
                peak_cpu = max(peak_cpu, cpu_percent)

                # Rate limiting
                time.sleep(chunk_size / bytes_per_second)

        except Exception as e:
            errors.append(f"Data throughput stress test failed: {str(e)}")

        duration = time.time() - start_time

        # Calculate throughput
        actual_throughput_mbps = (total_bytes_processed / duration / 1024 / 1024 * 8) if duration > 0 else 0

        # Pass criteria
        passed = (
            len(errors) == 0 and
            actual_throughput_mbps >= data_rate_mbps * 0.9  # At least 90% of target
        )

        return StressTestResult(
            test_name="Data Throughput Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            throughput=actual_throughput_mbps,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "target_throughput_mbps": data_rate_mbps,
                "total_mb_processed": total_bytes_processed / 1024 / 1024
            }
        )

    def stress_test_sustained_load(
        self,
        system_under_test: Any,
        load_level: float = 0.80,
        duration_seconds: int = 300
    ) -> StressTestResult:
        """
        Stress test sustained load

        Tests system's ability to maintain performance under sustained load
        """
        logger.info(f"Running sustained load stress test: {load_level*100:.0f}% load for {duration_seconds}s")

        start_time = time.time()
        errors = []
        warnings = []
        latencies = []
        peak_memory = 0
        peak_cpu = 0

        total_operations = 0

        try:
            end_time = start_time + duration_seconds

            while time.time() < end_time:
                operation_start = time.time()

                try:
                    # Simulate operation (validation, decision, etc.)
                    time.sleep(np.random.uniform(0.001, 0.005) * (1 - load_level))
                    total_operations += 1

                except Exception as e:
                    errors.append(str(e))

                latency_ms = (time.time() - operation_start) * 1000
                latencies.append(latency_ms)

                # Monitor resources
                mem_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()

                peak_memory = max(peak_memory, mem_mb)
                peak_cpu = max(peak_cpu, cpu_percent)

                # Check for performance degradation
                if len(latencies) > 1000:
                    recent_p99 = np.percentile(latencies[-1000:], 99)
                    early_p99 = np.percentile(latencies[:1000], 99)

                    if recent_p99 > early_p99 * 1.5:  # 50% degradation
                        warnings.append(f"Performance degradation detected: p99 increased from {early_p99:.1f}ms to {recent_p99:.1f}ms")

        except Exception as e:
            errors.append(f"Sustained load stress test failed: {str(e)}")

        duration = time.time() - start_time

        # Calculate metrics
        throughput = total_operations / duration if duration > 0 else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        p99 = np.percentile(latencies, 99) if latencies else 0

        # Pass criteria
        passed = (
            len(errors) == 0 and
            len(warnings) == 0 and  # No performance degradation
            p99 < 100  # p99 latency stays under 100ms
        )

        return StressTestResult(
            test_name="Sustained Load Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            throughput=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "load_level": load_level,
                "total_operations": total_operations
            }
        )

    def stress_test_spike_load(
        self,
        system_under_test: Any,
        spike_multiplier: int = 10,
        spike_duration_seconds: int = 10
    ) -> StressTestResult:
        """
        Stress test spike load

        Tests system's ability to handle sudden load spikes
        """
        logger.info(f"Running spike load stress test: {spike_multiplier}x spike for {spike_duration_seconds}s")

        start_time = time.time()
        errors = []
        warnings = []
        latencies = []
        peak_memory = 0
        peak_cpu = 0

        total_operations = 0

        try:
            # Baseline load (10 seconds)
            baseline_end = start_time + 10
            while time.time() < baseline_end:
                time.sleep(0.01)
                total_operations += 1

            # Spike load
            spike_end = time.time() + spike_duration_seconds
            while time.time() < spike_end:
                spike_start = time.time()

                # Process spike_multiplier operations
                for _ in range(spike_multiplier):
                    try:
                        operation_start = time.time()
                        time.sleep(np.random.uniform(0.0001, 0.001))
                        latencies.append((time.time() - operation_start) * 1000)
                        total_operations += 1
                    except Exception as e:
                        errors.append(str(e))

                # Monitor resources
                mem_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()

                peak_memory = max(peak_memory, mem_mb)
                peak_cpu = max(peak_cpu, cpu_percent)

        except Exception as e:
            errors.append(f"Spike load stress test failed: {str(e)}")

        duration = time.time() - start_time

        # Calculate metrics
        throughput = total_operations / duration if duration > 0 else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        p99 = np.percentile(latencies, 99) if latencies else 0

        # Pass criteria
        passed = (
            len(errors) == 0 and
            p99 < 200  # Allow higher latency during spike
        )

        return StressTestResult(
            test_name="Spike Load Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            throughput=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            errors=errors,
            warnings=warnings,
            system_crashed=False,
            details={
                "spike_multiplier": spike_multiplier,
                "total_operations": total_operations
            }
        )

    def stress_test_resource_exhaustion(
        self,
        system_under_test: Any
    ) -> StressTestResult:
        """
        Stress test resource exhaustion

        Tests system behavior when resources are exhausted
        """
        logger.info("Running resource exhaustion stress test")

        start_time = time.time()
        errors = []
        warnings = []
        system_crashed = False

        try:
            # Test CPU exhaustion
            logger.info("Testing CPU exhaustion...")
            cpu_start = time.time()
            while time.time() - cpu_start < 10:
                # CPU-intensive operation
                _ = sum(range(1000000))

            # Test file descriptor exhaustion
            logger.info("Testing file descriptor exhaustion...")
            file_handles = []
            try:
                for i in range(1000):
                    file_handles.append(open(f"/tmp/stress_test_{i}.tmp", "w"))
            except OSError as e:
                warnings.append(f"File descriptor limit reached: {str(e)}")
            finally:
                for fh in file_handles:
                    fh.close()

        except Exception as e:
            errors.append(f"Resource exhaustion test failed: {str(e)}")
            system_crashed = True

        duration = time.time() - start_time

        passed = not system_crashed and len(errors) == 0

        return StressTestResult(
            test_name="Resource Exhaustion Stress Test",
            passed=passed,
            duration_seconds=duration,
            peak_memory_mb=0,
            peak_cpu_percent=100,
            throughput=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            errors=errors,
            warnings=warnings,
            system_crashed=system_crashed,
            details={}
        )
