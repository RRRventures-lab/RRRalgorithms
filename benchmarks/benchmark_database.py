from pathlib import Path
import sqlite3
import sys
import time

"""
Benchmark: Database Performance
================================

Measures ACTUAL database query performance with and without indexes.

Author: RRR Ventures
Date: 2025-10-12
"""


sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_db_without_indexes(db_path):
    """Create test database WITHOUT timestamp indexes"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp REAL NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        )
    """)
    
    # Only symbol+timestamp composite index (original)
    cursor.execute("""
        CREATE INDEX idx_market_data_symbol_timestamp 
        ON market_data(symbol, timestamp DESC)
    """)
    
    conn.commit()
    return conn


def create_test_db_with_indexes(db_path):
    """Create test database WITH timestamp indexes"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp REAL NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        )
    """)
    
    # Original composite index
    cursor.execute("""
        CREATE INDEX idx_market_data_symbol_timestamp 
        ON market_data(symbol, timestamp DESC)
    """)
    
    # NEW: Timestamp-only index
    cursor.execute("""
        CREATE INDEX idx_market_data_timestamp 
        ON market_data(timestamp DESC)
    """)
    
    conn.commit()
    return conn


def insert_test_data(conn, rows=10000):
    """Insert test data"""
    cursor = conn.cursor()
    
    start = time.time()
    for i in range(rows):
        cursor.execute("""
            INSERT INTO market_data 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f"SYM-{i % 100}", float(i), 50000.0, 51000.0, 49000.0, 50500.0, 1000000.0))
    
    conn.commit()
    elapsed = time.time() - start
    
    return elapsed


def benchmark_timestamp_queries(conn, queries=100):
    """Benchmark timestamp-based queries"""
    cursor = conn.cursor()
    
    start = time.time()
    for _ in range(queries):
        cursor.execute("""
            SELECT * FROM market_data 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        results = cursor.fetchall()
    elapsed = time.time() - start
    
    return elapsed


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK: Database Index Performance")
    print("="*70)
    print()
    
    # Test WITHOUT timestamp index
    print("Creating database WITHOUT timestamp index...")
    conn_without = create_test_db_without_indexes("/tmp/test_without_idx.db")
    
    print("Inserting 10,000 rows...")
    insert_time_without = insert_test_data(conn_without, 10000)
    print(f"  Insert time: {insert_time_without*1000:.1f}ms")
    
    print("Benchmarking queries (100 queries)...")
    query_time_without = benchmark_timestamp_queries(conn_without, 100)
    print(f"  Query time: {query_time_without*1000:.1f}ms")
    print(f"  Per query: {query_time_without*1000/100:.2f}ms")
    
    conn_without.close()
    
    print()
    
    # Test WITH timestamp index
    print("Creating database WITH timestamp index...")
    conn_with = create_test_db_with_indexes("/tmp/test_with_idx.db")
    
    print("Inserting 10,000 rows...")
    insert_time_with = insert_test_data(conn_with, 10000)
    print(f"  Insert time: {insert_time_with*1000:.1f}ms")
    
    print("Benchmarking queries (100 queries)...")
    query_time_with = benchmark_timestamp_queries(conn_with, 100)
    print(f"  Query time: {query_time_with*1000:.1f}ms")
    print(f"  Per query: {query_time_with*1000/100:.2f}ms")
    
    conn_with.close()
    
    print()
    print("="*70)
    print("RESULTS:")
    print("="*70)
    
    # Calculate improvements
    if query_time_with > 0:
        query_improvement = query_time_without / query_time_with
        print(f"Query Speed:  {query_improvement:.1f}x FASTER with index")
        print(f"  Without index: {query_time_without*1000/100:.2f}ms per query")
        print(f"  With index:    {query_time_with*1000/100:.2f}ms per query")
    
    if insert_time_with > 0:
        insert_overhead = (insert_time_with / insert_time_without - 1) * 100
        print(f"\nInsert Overhead: {insert_overhead:.1f}% slower (index maintenance)")
    
    print()
    print("✅ Database optimization benchmark complete!")
    print()

