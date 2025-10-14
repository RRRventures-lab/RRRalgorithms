from functools import lru_cache
from typing import List, Dict
import asyncio
import random
import time


"""
Benchmark: Async Trading Loop Performance
==========================================

Measures ACTUAL async vs sync trading loop performance.

Key metric: Throughput (trades/second) improvement

Author: RRR Ventures
Date: 2025-10-12
"""



# Mock classes to avoid import issues
class MockDataSource:
    """Mock data source for benchmarking"""
    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    @lru_cache(maxsize=128)

    def get_latest_data(self):
        """Sync data fetch"""
        # Simulate API call latency
        time.sleep(0.001)  # 1ms per symbol
        return {
            symbol: {
                'close': 50000 + random.gauss(0, 500),
                'timestamp': time.time()
            }
            for symbol in self.symbols
        }

    async def get_latest_data_async(self):
        """Async data fetch"""
        # Simulate async API call
        await asyncio.sleep(0.001)  # 1ms per symbol
        return {
            symbol: {
                'close': 50000 + random.gauss(0, 500),
                'timestamp': time.time()
            }
            for symbol in self.symbols
        }


class MockPredictor:
    """Mock predictor for benchmarking"""
    def predict(self, symbol: str, price: float):
        """Sync prediction"""
        # Simulate ML inference
        time.sleep(0.0005)  # 0.5ms
        return {
            'predicted_price': price * (1 + random.gauss(0, 0.01)),
            'direction': 'up' if random.random() > 0.5 else 'down',
            'confidence': random.random()
        }

    async def predict_async(self, symbol: str, price: float):
        """Async prediction"""
        await asyncio.sleep(0.0005)  # 0.5ms
        return {
            'predicted_price': price * (1 + random.gauss(0, 0.01)),
            'direction': 'up' if random.random() > 0.5 else 'down',
            'confidence': random.random()
        }


class MockDatabase:
    """Mock database for benchmarking"""
    def __init__(self):
        self.data = []

    def insert_data(self, symbol: str, data: Dict):
        """Sync insert"""
        time.sleep(0.0002)  # 0.2ms
        self.data.append((symbol, data))

    async def insert_data_async(self, symbol: str, data: Dict):
        """Async insert"""
        await asyncio.sleep(0.0002)  # 0.2ms
        self.data.append((symbol, data))


# ============================================================================
# SYNCHRONOUS LOOP (Baseline)
# ============================================================================

def sync_trading_loop(symbols: List[str], iterations: int = 100):
    """
    Synchronous trading loop (baseline).
    Processes symbols sequentially.
    """
    data_source = MockDataSource(symbols)
    predictor = MockPredictor()
    database = MockDatabase()

    start = time.time()

    for i in range(iterations):
        # Fetch data (sequential)
        market_data = data_source.get_latest_data()

        # Process each symbol (sequential)
        for symbol, ohlcv in market_data.items():
            # Generate prediction
            prediction = predictor.predict(symbol, ohlcv['close'])

            # Store data
            database.insert_data(symbol, {
                'price': ohlcv['close'],
                'prediction': prediction['predicted_price']
            })

    elapsed = time.time() - start
    throughput = (iterations * len(symbols)) / elapsed

    return {
        'elapsed': elapsed,
        'iterations': iterations,
        'symbols': len(symbols),
        'total_ops': iterations * len(symbols),
        'throughput': throughput,
        'latency_per_op': elapsed / (iterations * len(symbols)) * 1000  # ms
    }


# ============================================================================
# ASYNCHRONOUS LOOP (Optimized)
# ============================================================================

async def async_trading_loop(symbols: List[str], iterations: int = 100):
    """
    Asynchronous trading loop (optimized).
    Processes symbols in parallel.
    """
    data_source = MockDataSource(symbols)
    predictor = MockPredictor()
    database = MockDatabase()

    start = time.time()

    for i in range(iterations):
        # Fetch data (async)
        market_data = await data_source.get_latest_data_async()

        # Process all symbols in parallel
        tasks = []
        for symbol, ohlcv in market_data.items():
            tasks.append(process_symbol_async(
                symbol, ohlcv, predictor, database
            ))

        # Wait for all to complete
        await asyncio.gather(*tasks)

    elapsed = time.time() - start
    throughput = (iterations * len(symbols)) / elapsed

    return {
        'elapsed': elapsed,
        'iterations': iterations,
        'symbols': len(symbols),
        'total_ops': iterations * len(symbols),
        'throughput': throughput,
        'latency_per_op': elapsed / (iterations * len(symbols)) * 1000  # ms
    }


async def process_symbol_async(symbol: str, ohlcv: Dict, predictor, database):
    """Process single symbol asynchronously"""
    # Generate prediction
    prediction = await predictor.predict_async(symbol, ohlcv['close'])

    # Store data
    await database.insert_data_async(symbol, {
        'price': ohlcv['close'],
        'prediction': prediction['predicted_price']
    })


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(symbols: List[str], iterations: int = 100):
    """Run both sync and async benchmarks and compare"""

    print("=" * 80)
    print("BENCHMARK: Async Trading Loop Performance")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Iterations: {iterations}")
    print(f"  Total operations: {iterations * len(symbols)}")
    print()

    # Run synchronous benchmark
    print("Running SYNCHRONOUS loop...")
    sync_results = sync_trading_loop(symbols, iterations)

    print(f"  Elapsed: {sync_results['elapsed']:.2f}s")
    print(f"  Throughput: {sync_results['throughput']:.1f} ops/sec")
    print(f"  Latency: {sync_results['latency_per_op']:.2f}ms per operation")
    print()

    # Run asynchronous benchmark
    print("Running ASYNCHRONOUS loop...")
    async_results = asyncio.run(async_trading_loop(symbols, iterations))

    print(f"  Elapsed: {async_results['elapsed']:.2f}s")
    print(f"  Throughput: {async_results['throughput']:.1f} ops/sec")
    print(f"  Latency: {async_results['latency_per_op']:.2f}ms per operation")
    print()

    # Calculate improvement
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)

    speedup = sync_results['elapsed'] / async_results['elapsed']
    throughput_improvement = async_results['throughput'] / sync_results['throughput']
    latency_improvement = sync_results['latency_per_op'] / async_results['latency_per_op']

    print(f"Speedup: {speedup:.1f}x FASTER")
    print(f"  Sync:  {sync_results['elapsed']:.2f}s")
    print(f"  Async: {async_results['elapsed']:.2f}s")
    print()

    print(f"Throughput Improvement: {throughput_improvement:.1f}x")
    print(f"  Sync:  {sync_results['throughput']:.1f} ops/sec")
    print(f"  Async: {async_results['throughput']:.1f} ops/sec")
    print()

    print(f"Latency Improvement: {latency_improvement:.1f}x FASTER")
    print(f"  Sync:  {sync_results['latency_per_op']:.2f}ms per operation")
    print(f"  Async: {async_results['latency_per_op']:.2f}ms per operation")
    print()

    # Verdict
    if speedup >= 10:
        print("✅ SUCCESS: Async loop delivers 10x+ improvement (CLAIM VERIFIED)")
    elif speedup >= 5:
        print("⚡ GOOD: Async loop delivers 5-10x improvement (solid performance)")
    elif speedup >= 2:
        print("⚠️  MODERATE: Async loop delivers 2-5x improvement (some benefit)")
    else:
        print("❌ MINIMAL: Async loop delivers <2x improvement (claim not verified)")

    print()

    return {
        'sync': sync_results,
        'async': async_results,
        'speedup': speedup,
        'throughput_improvement': throughput_improvement,
        'claim_verified': speedup >= 10
    }


if __name__ == "__main__":
    # Test configurations
    configs = [
        (['BTC-USD'], 200, "Single symbol"),
        (['BTC-USD', 'ETH-USD'], 200, "Two symbols"),
        (['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'MATIC-USD'], 100, "Five symbols"),
    ]

    all_results = []

    for symbols, iterations, description in configs:
        print(f"\n{'='*80}")
        print(f"TEST: {description}")
        print(f"{'='*80}\n")

        results = run_benchmark(symbols, iterations)
        all_results.append((description, results))

        time.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Async Trading Loop Performance")
    print("=" * 80)
    print()

    for description, results in all_results:
        print(f"{description}:")
        print(f"  Speedup: {results['speedup']:.1f}x")
        print(f"  Throughput improvement: {results['throughput_improvement']:.1f}x")
        print(f"  Claim verified: {'✅ YES' if results['claim_verified'] else '❌ NO'}")
        print()

    # Overall verdict
    avg_speedup = sum(r['speedup'] for _, r in all_results) / len(all_results)
    print(f"Average speedup: {avg_speedup:.1f}x")

    if avg_speedup >= 10:
        print("✅ OVERALL: 10-20x claim VERIFIED!")
    else:
        print(f"⚠️  OVERALL: Actual speedup is {avg_speedup:.1f}x (claim of 10-20x is optimistic)")
