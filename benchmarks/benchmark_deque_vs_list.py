from collections import deque
import time

"""
Benchmark: Deque vs List Performance
=====================================

Measures ACTUAL performance improvement of deque vs list
for the price history tracking optimization.

Author: RRR Ventures
Date: 2025-10-12
"""



def benchmark_list_operations(iterations=1000):
    """Benchmark list with pop(0) - O(n) operation"""
    prices = []
    
    start = time.time()
    for i in range(iterations):
        prices.append(i)
        if len(prices) > 10:
            prices.pop(0)  # O(n) operation
    elapsed = time.time() - start
    
    return elapsed


def benchmark_deque_operations(iterations=1000):
    """Benchmark deque with maxlen - O(1) operation"""
    prices = deque(maxlen=10)
    
    start = time.time()
    for i in range(iterations):
        prices.append(i)  # O(1) operation, auto-removes old
    elapsed = time.time() - start
    
    return elapsed


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK: Deque vs List Performance")
    print("="*70)
    print()
    
    iterations = 10000
    runs = 5
    
    print(f"Running {iterations} operations, {runs} times each...\n")
    
    # Benchmark list
    list_times = []
    for run in range(runs):
        t = benchmark_list_operations(iterations)
        list_times.append(t)
    list_avg = sum(list_times) / len(list_times)
    
    print(f"List (with pop(0)):")
    print(f"  Average: {list_avg*1000:.2f}ms")
    print(f"  Best:    {min(list_times)*1000:.2f}ms")
    print(f"  Worst:   {max(list_times)*1000:.2f}ms")
    print()
    
    # Benchmark deque
    deque_times = []
    for run in range(runs):
        t = benchmark_deque_operations(iterations)
        deque_times.append(t)
    deque_avg = sum(deque_times) / len(deque_times)
    
    print(f"Deque (with maxlen):")
    print(f"  Average: {deque_avg*1000:.2f}ms")
    print(f"  Best:    {min(deque_times)*1000:.2f}ms")
    print(f"  Worst:   {max(deque_times)*1000:.2f}ms")
    print()
    
    # Calculate improvement
    improvement = list_avg / deque_avg if deque_avg > 0 else 0
    
    print("="*70)
    print(f"RESULT: Deque is {improvement:.1f}x FASTER than list")
    print("="*70)
    print()
    print(f"✅ Optimization saves {(list_avg - deque_avg)*1000:.2f}ms per {iterations} operations")
    print()

