from pathlib import Path
from src.core.async_postgresql import AsyncPostgreSQL
from src.core.memory_cache import MemoryCache
from src.core.redis_cache import RedisCache
from src.data_pipeline.websocket_pipeline import WebSocketDataSource, MarketData
from src.neural_network.production_predictor import ProductionPredictor, PredictionResult
from typing import Dict, List, Any
import asyncio
import sys
import time

#!/usr/bin/env python3
"""
Phase 2B Integration Test
========================

Comprehensive test of Phase 2B components:
- WebSocket data pipeline
- Production ML models
- PostgreSQL database
- Redis caching
- Performance validation

Usage:
    python scripts/test_phase_2b_integration.py

Author: RRR Ventures
Date: 2025-10-12
"""


# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



class Phase2BIntegrationTest:
    """Comprehensive Phase 2B integration test."""
    
    def __init__(self):
        """Initialize test components."""
        self.websocket_source = None
        self.ml_predictor = None
        self.postgres_db = None
        self.redis_cache = None
        self.memory_cache = None
        
        self.test_results = {
            'websocket_pipeline': False,
            'ml_predictor': False,
            'postgresql_db': False,
            'redis_cache': False,
            'memory_cache': False,
            'integration': False
        }
        
        self.performance_metrics = {}
    
    async def setup(self) -> None:
        """Setup all test components."""
        print("🔧 Setting up Phase 2B components...")
        
        try:
            # Initialize WebSocket data source (mock mode for testing)
            print("  - Initializing WebSocket data source...")
            self.websocket_source = WebSocketDataSource(
                symbols=['BTC-USD', 'ETH-USD'],
                exchanges=['polygon']  # Use polygon for testing
            )
            print("  ✅ WebSocket data source ready")
            
            # Initialize production ML predictor
            print("  - Initializing production ML predictor...")
            self.ml_predictor = ProductionPredictor()
            await self.ml_predictor.initialize()
            print("  ✅ Production ML predictor ready")
            
            # Initialize PostgreSQL database
            print("  - Initializing PostgreSQL database...")
            self.postgres_db = AsyncPostgreSQL(
                database_url="postgresql://user:password@localhost:5432/trading_test",
                min_connections=2,
                max_connections=5
            )
            await self.postgres_db.initialize()
            print("  ✅ PostgreSQL database ready")
            
            # Initialize Redis cache
            print("  - Initializing Redis cache...")
            self.redis_cache = RedisCache(
                host='localhost',
                port=6379,
                db=1,  # Use different DB for testing
                max_connections=5
            )
            await self.redis_cache.initialize()
            print("  ✅ Redis cache ready")
            
            # Initialize memory cache
            print("  - Initializing memory cache...")
            self.memory_cache = MemoryCache(max_size=1000, default_ttl=60.0)
            await self.memory_cache.start()
            print("  ✅ Memory cache ready")
            
            print("✅ All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise
    
    async def test_websocket_pipeline(self) -> bool:
        """Test WebSocket data pipeline."""
        print("\n📡 Testing WebSocket Data Pipeline...")
        
        try:
            # Test WebSocket source initialization
            status = self.websocket_source.get_status()
            print(f"  - Status: {status}")
            
            # Test data callbacks
            received_data = []
            
            def data_callback(market_data: MarketData):
                received_data.append(market_data)
                print(f"  - Received data: {market_data.symbol} @ ${market_data.price:.2f}")
            
            self.websocket_source.add_data_callback(data_callback)
            
            # Test mock data generation
            mock_data = {
                'BTC-USD': {
                    'timestamp': time.time(),
                    'price': 50000.0,
                    'volume': 1000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'open': 49500.0,
                    'close': 50000.0
                }
            }
            
            # Simulate data processing
            for symbol, data in mock_data.items():
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=data['timestamp'],
                    price=data['price'],
                    volume=data['volume'],
                    high=data['high'],
                    low=data['low'],
                    open=data['open'],
                    close=data['close'],
                    source='test'
                )
                await self.websocket_source._process_market_data(market_data)
            
            print(f"  ✅ WebSocket pipeline test passed - {len(received_data)} data points processed")
            return True
            
        except Exception as e:
            print(f"  ❌ WebSocket pipeline test failed: {e}")
            return False
    
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
    
    async def test_postgresql_database(self) -> bool:
        """Test PostgreSQL database."""
        print("\n🗄️ Testing PostgreSQL Database...")
        
        try:
            # Test database status
            status = self.postgres_db.get_status()
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
                    'volume': 1000.0,
                    'source': 'test'
                },
                {
                    'symbol': 'ETH-USD',
                    'timestamp': time.time(),
                    'open': 3000.0,
                    'high': 3100.0,
                    'low': 2900.0,
                    'close': 3000.0,
                    'volume': 5000.0,
                    'source': 'test'
                }
            ]
            
            start_time = time.time()
            await self.postgres_db.batch_insert_market_data(test_data)
            insert_time = time.time() - start_time
            
            print(f"  - Batch insert: {len(test_data)} records in {insert_time*1000:.1f}ms")
            
            # Test query
            start_time = time.time()
            results = await self.postgres_db.get_latest_market_data(limit=10)
            query_time = time.time() - start_time
            
            print(f"  - Query: {len(results)} records in {query_time*1000:.1f}ms")
            
            # Test analytics
            analytics = await self.postgres_db.get_performance_analytics(days_back=1)
            print(f"  - Analytics: {analytics}")
            
            # Test metrics
            metrics = self.postgres_db.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ PostgreSQL database test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ PostgreSQL database test failed: {e}")
            return False
    
    async def test_redis_cache(self) -> bool:
        """Test Redis cache."""
        print("\n💾 Testing Redis Cache...")
        
        try:
            # Test cache status
            status = self.redis_cache.get_status()
            print(f"  - Status: {status}")
            
            # Test basic operations
            test_data = {
                'BTC-USD': {
                    'price': 50000.0,
                    'volume': 1000.0,
                    'timestamp': time.time()
                },
                'ETH-USD': {
                    'price': 3000.0,
                    'volume': 5000.0,
                    'timestamp': time.time()
                }
            }
            
            # Test set operations
            start_time = time.time()
            for symbol, data in test_data.items():
                await self.redis_cache.cache_market_data(symbol, data, ttl=300)
            set_time = time.time() - start_time
            
            print(f"  - Set operations: {len(test_data)} items in {set_time*1000:.1f}ms")
            
            # Test get operations
            start_time = time.time()
            for symbol in test_data.keys():
                cached_data = await self.redis_cache.get_cached_market_data(symbol)
                if cached_data:
                    print(f"  - Retrieved {symbol}: ${cached_data['price']:.2f}")
            get_time = time.time() - start_time
            
            print(f"  - Get operations: {len(test_data)} items in {get_time*1000:.1f}ms")
            
            # Test hash operations
            await self.redis_cache.hset('test_hash', {'field1': 'value1', 'field2': 'value2'})
            hash_data = await self.redis_cache.hgetall('test_hash')
            print(f"  - Hash operations: {hash_data}")
            
            # Test pub/sub
            pubsub = await self.redis_cache.subscribe(['test_channel'])
            await self.redis_cache.publish('test_channel', {'message': 'test'})
            await pubsub.close()
            print("  - Pub/sub operations: OK")
            
            # Test metrics
            metrics = self.redis_cache.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ Redis cache test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Redis cache test failed: {e}")
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
            
            # Test metrics
            metrics = self.memory_cache.get_metrics()
            print(f"  - Metrics: {metrics}")
            
            print("  ✅ Memory cache test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Memory cache test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test end-to-end integration."""
        print("\n🔗 Testing End-to-End Integration...")
        
        try:
            # Simulate complete trading pipeline
            print("  - Simulating trading pipeline...")
            
            # 1. Generate market data
            market_data = {
                'BTC-USD': {
                    'timestamp': time.time(),
                    'price': 50000.0,
                    'volume': 1000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'open': 49500.0,
                    'close': 50000.0
                }
            }
            
            # 2. Cache market data
            for symbol, data in market_data.items():
                await self.redis_cache.cache_market_data(symbol, data, ttl=300)
                self.memory_cache.set(f"market_data:{symbol}", data, ttl=60)
            
            # 3. Make prediction
            prediction = await self.ml_predictor.predict(
                symbol='BTC-USD',
                market_data=market_data['BTC-USD'],
                horizon_minutes=60
            )
            
            # 4. Cache prediction
            await self.redis_cache.cache_prediction('BTC-USD', {
                'predicted_price': prediction.predicted_price,
                'predicted_direction': prediction.predicted_direction,
                'confidence': prediction.confidence
            }, ttl=600)
            
            # 5. Store in database
            await self.postgres_db.batch_insert_market_data([
                {
                    'symbol': 'BTC-USD',
                    'timestamp': market_data['BTC-USD']['timestamp'],
                    'open': market_data['BTC-USD']['open'],
                    'high': market_data['BTC-USD']['high'],
                    'low': market_data['BTC-USD']['low'],
                    'close': market_data['BTC-USD']['close'],
                    'volume': market_data['BTC-USD']['volume'],
                    'source': 'integration_test'
                }
            ])
            
            await self.postgres_db.batch_insert_predictions([{
                'symbol': 'BTC-USD',
                'timestamp': prediction.timestamp,
                'horizon': prediction.prediction_horizon,
                'predicted_price': prediction.predicted_price,
                'predicted_direction': prediction.predicted_direction,
                'confidence': prediction.confidence,
                'model_version': prediction.model_version,
                'features_used': prediction.features_used
            }])
            
            print(f"  - Market data processed: {len(market_data)} symbols")
            print(f"  - Prediction made: {prediction.predicted_direction} @ ${prediction.predicted_price:.2f}")
            print(f"  - Data cached and stored successfully")
            
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
            
            # Cache in Redis
            await self.redis_cache.cache_market_data(f'TEST-{i}', market_data, ttl=300)
        
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
        
        # Store performance metrics
        self.performance_metrics = {
            'data_processing_rate': 100/processing_time,
            'prediction_rate': 10/prediction_time,
            'total_time': processing_time + prediction_time
        }
    
    async def cleanup(self) -> None:
        """Cleanup test components."""
        print("\n🧹 Cleaning up test components...")
        
        try:
            if self.websocket_source:
                await self.websocket_source.stop()
            
            if self.ml_predictor:
                await self.ml_predictor.close()
            
            if self.postgres_db:
                await self.postgres_db.close()
            
            if self.redis_cache:
                await self.redis_cache.close()
            
            if self.memory_cache:
                await self.memory_cache.stop()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    async def run_all_tests(self) -> None:
        """Run all Phase 2B tests."""
        print("🚀 Phase 2B Integration Test Suite")
        print("=" * 50)
        
        try:
            # Setup
            await self.setup()
            
            # Run individual component tests
            self.test_results['websocket_pipeline'] = await self.test_websocket_pipeline()
            self.test_results['ml_predictor'] = await self.test_ml_predictor()
            self.test_results['postgresql_db'] = await self.test_postgresql_database()
            self.test_results['redis_cache'] = await self.test_redis_cache()
            self.test_results['memory_cache'] = await self.test_memory_cache()
            
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
    test_suite = Phase2BIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())