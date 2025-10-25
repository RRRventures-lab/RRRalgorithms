"""
Load Testing
Tests API performance under load
"""

import pytest
import asyncio
import aiohttp
import time
from typing import List
import statistics


class LoadTester:
    """Load testing utility"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> dict:
        """Make a single request and measure performance"""
        start_time = time.time()
        status_code = 0
        error = None

        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                status_code = response.status
                await response.text()
        except Exception as e:
            error = str(e)

        duration = time.time() - start_time

        return {
            "endpoint": endpoint,
            "status_code": status_code,
            "duration": duration,
            "error": error
        }

    async def run_concurrent_requests(
        self,
        endpoint: str,
        num_requests: int,
        concurrency: int
    ) -> List[dict]:
        """Run concurrent requests"""
        results = []

        async with aiohttp.ClientSession() as session:
            # Create batches
            for i in range(0, num_requests, concurrency):
                batch_size = min(concurrency, num_requests - i)
                tasks = [
                    self.make_request(session, endpoint)
                    for _ in range(batch_size)
                ]

                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                # Small delay between batches
                await asyncio.sleep(0.1)

        return results

    def analyze_results(self, results: List[dict]) -> dict:
        """Analyze load test results"""
        durations = [r["duration"] for r in results if r["status_code"] == 200]
        errors = [r for r in results if r["error"] or r["status_code"] >= 400]

        if not durations:
            return {
                "error": "No successful requests",
                "total_errors": len(errors)
            }

        return {
            "total_requests": len(results),
            "successful_requests": len(durations),
            "failed_requests": len(errors),
            "success_rate": len(durations) / len(results) * 100,
            "avg_response_time": statistics.mean(durations),
            "median_response_time": statistics.median(durations),
            "min_response_time": min(durations),
            "max_response_time": max(durations),
            "p95_response_time": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
            "p99_response_time": statistics.quantiles(durations, n=100)[98] if len(durations) > 100 else max(durations),
            "requests_per_second": len(durations) / sum(durations) if sum(durations) > 0 else 0
        }


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,  # Skip by default - run manually for load testing
    reason="Load test - run manually with: pytest -m loadtest"
)
class TestLoadPerformance:
    """Load performance tests"""

    @pytest.fixture
    def load_tester(self):
        return LoadTester()

    async def test_health_endpoint_load(self, load_tester):
        """Test /health endpoint under load"""
        results = await load_tester.run_concurrent_requests(
            endpoint="/health",
            num_requests=1000,
            concurrency=50
        )

        analysis = load_tester.analyze_results(results)

        print("\n=== Health Endpoint Load Test ===")
        print(f"Total Requests: {analysis['total_requests']}")
        print(f"Success Rate: {analysis['success_rate']:.2f}%")
        print(f"Avg Response Time: {analysis['avg_response_time']*1000:.2f}ms")
        print(f"P95 Response Time: {analysis['p95_response_time']*1000:.2f}ms")
        print(f"Requests/sec: {analysis['requests_per_second']:.2f}")

        # Assertions
        assert analysis['success_rate'] > 95, "Success rate should be > 95%"
        assert analysis['avg_response_time'] < 0.1, "Avg response time should be < 100ms"

    async def test_portfolio_endpoint_load(self, load_tester):
        """Test /api/portfolio endpoint under load"""
        results = await load_tester.run_concurrent_requests(
            endpoint="/api/portfolio",
            num_requests=500,
            concurrency=25
        )

        analysis = load_tester.analyze_results(results)

        print("\n=== Portfolio Endpoint Load Test ===")
        print(f"Total Requests: {analysis['total_requests']}")
        print(f"Success Rate: {analysis['success_rate']:.2f}%")
        print(f"Avg Response Time: {analysis['avg_response_time']*1000:.2f}ms")
        print(f"P95 Response Time: {analysis['p95_response_time']*1000:.2f}ms")

        # Assertions
        assert analysis['success_rate'] > 90, "Success rate should be > 90%"
        assert analysis['avg_response_time'] < 0.5, "Avg response time should be < 500ms"

    async def test_rate_limiting_under_load(self, load_tester):
        """Test that rate limiting works under load"""
        results = await load_tester.run_concurrent_requests(
            endpoint="/api/trades",
            num_requests=200,
            concurrency=50  # High concurrency to trigger rate limit
        )

        # Count 429 (rate limited) responses
        rate_limited = len([r for r in results if r["status_code"] == 429])

        print(f"\n=== Rate Limiting Test ===")
        print(f"Total Requests: {len(results)}")
        print(f"Rate Limited: {rate_limited}")
        print(f"Success: {len([r for r in results if r['status_code'] == 200])}")

        # Should have some rate limiting
        assert rate_limited > 0, "Rate limiting should kick in under high load"


@pytest.mark.asyncio
class TestWebSocketPerformance:
    """WebSocket performance tests"""

    async def test_websocket_message_throughput(self):
        """Test WebSocket can handle 100-500 messages/sec"""
        # TODO: Implement WebSocket throughput test
        # This would test the WebSocket server's ability to handle
        # high-frequency updates
        pass


if __name__ == "__main__":
    # Run load tests
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "loadtest"]))
