from pathlib import Path
from src.core.async_database import AsyncDatabase
from src.core.memory_cache import MemoryCache
from src.neural_network.production_predictor import ProductionPredictor, PredictionResult
from typing import Dict, List, Any
import asyncio
import sys
import time

#!/usr/bin/env python3
"""
Phase 2B Simple Test
===================

Simplified test of Phase 2B components without external dependencies.
Tests core functionality and performance improvements.

Usage:
    python scripts/test_phase_2b_simple.py

Author: RRR Ventures
Date: 2025-10-12
"""


# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



class Phase2BSimpleTest:
    """Simplified Phase 2B test without external dependencies."""
    
    def __init__(self):
        """Initialize test components."""
        self.ml_predictor = None
        self.memory_cache = None
        self.async_db = None
        
        self.test_results = {
            'ml_predictor': False,
            'memory_cache': False,
            'async_database': False,
            'integration': False
        }
        
        self.performance_metrics = {}
    
    async def setup(self) -> None:
        """Setup test components."""
        print("🔧 Setting up Phase 2B components...")
        
        try:
            # Initialize production ML predictor
            print("  - Initializing production ML predictor...")
            self.ml_predictor = ProductionPredictor()
            await self.ml_predictor.initialize()
            print("  ✅ Production ML predictor ready")
            
            # Initialize memory cache
            print("  - Initializing memory cache...")
            self.memory_cache = MemoryCache(max_size=1000, default_ttl=60.0)
            await self.memory_cache.start()
            print("  ✅ Memory cache ready")
            
            # Initialize async database (SQLite for testing)
            print("  - Initializing async database...")
            self.async_db = AsyncDatabase(
                database_path="data/database/test_trading.db",
                max_connections=5
            )
            await self.async_db.initialize()
            print("  ✅ Async database ready")
            
            print("✅ All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise
    
    async def test_ml_predictor(self) -> bool:
        """Test production ML predictor."""
        print("\n🤖 Testing Production ML Predictor...")
        
        try:
            # Test predictor initialization
            status = self.ml_predictor.get_status()
            print(f"  - Status: {status}")
            
            # Test prediction
            mock_market_data = {
                'close': 50000.0,
                'volume': 1000.0,
                'high': 51000.0,
                'low': 49000.0,
                'open': 49500.0,
                'price': 50000.0
            }
            
            start_time = time.time()
            prediction = await self.ml_predictor.predict(
                symbol='BTC-USD',
                market_data=mock_market_data,
                horizon_minutes=60
            )
            inference_time = time.time() - start_time
            
            print(f"  - Prediction: {prediction.predicted_direction} @ ${prediction.predicted_price:.2f}")
            print(f"  - Confidence: {prediction.confidence:.2f}")
            print(f"  - Inference time: {inference_time*1000:.1f}ms")
            
            # Test metrics
            metrics = self.ml_predictor.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ ML predictor test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ ML predictor test failed: {e}")
            return False
    
    async def test_memory_cache(self) -> bool:
        """Test memory cache."""
        print("\n🧠 Testing Memory Cache...")
        
        try:
            # Test cache status
            status = self.memory_cache.get_status()
            print(f"  - Status: {status}")
            
            # Test operations
            test_data = {f'key_{i}': f'value_{i}' for i in range(100)}
            
            # Test set operations
            start_time = time.time()
            for key, value in test_data.items():
                self.memory_cache.set(key, value, ttl=60)
            set_time = time.time() - start_time
            
            print(f"  - Set operations: {len(test_data)} items in {set_time*1000:.1f}ms")
            
            # Test get operations
            start_time = time.time()
            hits = 0
            for key in test_data.keys():
                if self.memory_cache.get(key):
                    hits += 1
            get_time = time.time() - start_time
            
            print(f"  - Get operations: {hits}/{len(test_data)} hits in {get_time*1000:.1f}ms")
            
            # Test async operations
            async def async_operation(key: str) -> str:
                await asyncio.sleep(0.001)  # Simulate async work
                return f"async_value_{key}"
            
            start_time = time.time()
            async_results = []
            for i in range(10):
                result = await self.memory_cache.get_or_set_async(
                    f"async_key_{i}",
                    lambda: async_operation(f"async_key_{i}"),
                    ttl=60
                )
                async_results.append(result)
            async_time = time.time() - start_time
            
            print(f"  - Async operations: {len(async_results)} items in {async_time*1000:.1f}ms")
            
            # Test metrics
            metrics = self.memory_cache.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ Memory cache test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Memory cache test failed: {e}")
            return False
    
    async def test_async_database(self) -> bool:
        """Test async database."""
        print("\n🗄️ Testing Async Database...")
        
        try:
            # Test database status
            status = self.async_db.get_status()
            print(f"  - Status: {status}")
            
            # Test batch insert
            test_data = [
                {
                    'symbol': 'BTC-USD',
                    'timestamp': time.time(),
                    'open': 49500.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'close': 50000.0,
                    'volume': 1000.0
                },
                {
                    'symbol': 'ETH-USD',
                    'timestamp': time.time(),
                    'open': 3000.0,
                    'high': 3100.0,
                    'low': 2900.0,
                    'close': 3000.0,
                    'volume': 5000.0
                }
            ]
            
            start_time = time.time()
            await self.async_db.batch_insert_market_data(test_data)
            insert_time = time.time() - start_time
            
            print(f"  - Batch insert: {len(test_data)} records in {insert_time*1000:.1f}ms")
            
            # Test query
            start_time = time.time()
            results = await self.async_db.get_latest_market_data(limit=10)
            query_time = time.time() - start_time
            
            print(f"  - Query: {len(results)} records in {query_time*1000:.1f}ms")
            
            # Test predictions insert
            predictions_data = [
                {
                    'symbol': 'BTC-USD',
                    'timestamp': time.time(),
                    'horizon': 60,
                    'predicted_price': 51000.0,
                    'predicted_direction': 'up',
                    'confidence': 0.8,
                    'model_version': 'test_v1.0'
                }
            ]
            
            start_time = time.time()
            await self.async_db.batch_insert_predictions(predictions_data)
            pred_insert_time = time.time() - start_time
            
            print(f"  - Predictions insert: {len(predictions_data)} records in {pred_insert_time*1000:.1f}ms")
            
            # Test metrics
            metrics = self.async_db.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ Async database test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Async database test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test end-to-end integration."""
        print("\n🔗 Testing End-to-End Integration...")
        
        try:
            # Simulate complete trading pipeline
            print("  - Simulating trading pipeline...")
            
            # 1. Generate market data
            market_data = {
                'timestamp': time.time(),
                'price': 50000.0,
                'volume': 1000.0,
                'high': 51000.0,
                'low': 49000.0,
                'open': 49500.0,
                'close': 50000.0
            }
            
            # 2. Cache market data
            self.memory_cache.set("market_data:BTC-USD", market_data, ttl=60)
            
            # 3. Make prediction
            prediction = await self.ml_predictor.predict(
                symbol='BTC-USD',
                market_data=market_data,
                horizon_minutes=60
            )
            
            # 4. Cache prediction
            self.memory_cache.set("prediction:BTC-USD", {
                'predicted_price': prediction.predicted_price,
                'predicted_direction': prediction.predicted_direction,
                'confidence': prediction.confidence
            }, ttl=300)
            
            # 5. Store in database
            await self.async_db.batch_insert_market_data([{
                'symbol': 'BTC-USD',
                'timestamp': market_data['timestamp'],
                'open': market_data['open'],
                'high': market_data['high'],
                'low': market_data['low'],
                'close': market_data['close'],
                'volume': market_data['volume']
            }])
            
            await self.async_db.batch_insert_predictions([{
                'symbol': 'BTC-USD',
                'timestamp': prediction.timestamp,
                'horizon': prediction.prediction_horizon,
                'predicted_price': prediction.predicted_price,
                'predicted_direction': prediction.predicted_direction,
                'confidence': prediction.confidence,
                'model_version': prediction.model_version
            }])
            
            print(f"  - Market data processed and cached")
            print(f"  - Prediction made: {prediction.predicted_direction} @ ${prediction.predicted_price:.2f}")
            print(f"  - Data stored in database successfully")
            
            print("  ✅ Integration test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Integration test failed: {e}")
            return False
    
    async def run_performance_benchmark(self) -> None:
        """Run performance benchmark."""
        print("\n📊 Running Performance Benchmark...")
        
        # Benchmark data processing
        print("  - Benchmarking data processing...")
        start_time = time.time()
        
        # Process 100 data points
        for i in range(100):
            market_data = {
                'symbol': f'TEST-{i}',
                'timestamp': time.time(),
                'price': 50000.0 + i,
                'volume': 1000.0,
                'high': 51000.0 + i,
                'low': 49000.0 + i,
                'open': 49500.0 + i,
                'close': 50000.0 + i
            }
            
            # Cache in memory
            self.memory_cache.set(f"test_{i}", market_data, ttl=60)
        
        processing_time = time.time() - start_time
        print(f"    - Processed 100 data points in {processing_time*1000:.1f}ms")
        print(f"    - Rate: {100/processing_time:.1f} data points/second")
        
        # Benchmark predictions
        print("  - Benchmarking predictions...")
        start_time = time.time()
        
        # Make 10 predictions
        for i in range(10):
            await self.ml_predictor.predict(
                symbol=f'TEST-{i}',
                market_data={'close': 50000.0 + i, 'volume': 1000.0},
                horizon_minutes=60
            )
        
        prediction_time = time.time() - start_time
        print(f"    - Made 10 predictions in {prediction_time*1000:.1f}ms")
        print(f"    - Rate: {10/prediction_time:.1f} predictions/second")
        
        # Benchmark database operations
        print("  - Benchmarking database operations...")
        start_time = time.time()
        
        # Insert 50 records
        batch_data = []
        for i in range(50):
            batch_data.append({
                'symbol': f'BENCH-{i}',
                'timestamp': time.time(),
                'open': 49500.0 + i,
                'high': 51000.0 + i,
                'low': 49000.0 + i,
                'close': 50000.0 + i,
                'volume': 1000.0 + i
            })
        
        await self.async_db.batch_insert_market_data(batch_data)
        db_time = time.time() - start_time
        print(f"    - Inserted 50 records in {db_time*1000:.1f}ms")
        print(f"    - Rate: {50/db_time:.1f} records/second")
        
        # Store performance metrics
        self.performance_metrics = {
            'data_processing_rate': 100/processing_time,
            'prediction_rate': 10/prediction_time,
            'database_insert_rate': 50/db_time,
            'total_time': processing_time + prediction_time + db_time
        }
    
    async def cleanup(self) -> None:
        """Cleanup test components."""
        print("\n🧹 Cleaning up test components...")
        
        try:
            if self.ml_predictor:
                await self.ml_predictor.close()
            
            if self.memory_cache:
                await self.memory_cache.stop()
            
            if self.async_db:
                await self.async_db.close()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    async def run_all_tests(self) -> None:
        """Run all Phase 2B tests."""
        print("🚀 Phase 2B Simple Test Suite")
        print("=" * 50)
        
        try:
            # Setup
            await self.setup()
            
            # Run individual component tests
            self.test_results['ml_predictor'] = await self.test_ml_predictor()
            self.test_results['memory_cache'] = await self.test_memory_cache()
            self.test_results['async_database'] = await self.test_async_database()
            
            # Run integration test
            self.test_results['integration'] = await self.test_integration()
            
            # Run performance benchmark
            await self.run_performance_benchmark()
            
            # Print results
            self.print_results()
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
        finally:
            await self.cleanup()
    
    def print_results(self) -> None:
        """Print test results."""
        print("\n📋 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {component.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if self.performance_metrics:
            print("\n📊 PERFORMANCE METRICS")
            print("=" * 50)
            for metric, value in self.performance_metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
        
        if passed_tests == total_tests:
            print("\n🎉 All Phase 2B tests passed! System ready for production!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review and fix issues.")


async def main():
    """Main test function."""
    test_suite = Phase2BSimpleTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())