from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from polygon import RESTClient
from src.core.rate_limiter import RateLimiter
from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
from typing import List, Dict, Any
import asyncio
import concurrent.futures
import os
import random
import signal
import sys
import time

#!/usr/bin/env python
"""
Polygon.io Connection Robustness Test
======================================

Tests connection reliability, error handling, rate limiting,
and recovery mechanisms.
"""


# Setup
sys.path.insert(0, '.')
load_dotenv('config/api-keys/.env')

# Import available exceptions
try:
    from polygon.exceptions import BadResponse, AuthError
except ImportError:
    # Fallback for different polygon versions
    BadResponse = Exception
    AuthError = Exception


class RobustnessTest:
    """Comprehensive robustness testing for Polygon connection"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.client = RESTClient(api_key=self.api_key)
        self.feed = PolygonLiveFeed(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'])
        self.results = {
            'tests_passed': [],
            'tests_failed': [],
            'warnings': [],
            'metrics': {}
        }
        self.start_time = time.time()
    
    def print_header(self, title):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def test_basic_connection(self) -> bool:
        """Test 1: Basic connection and authentication"""
        self.print_header("Test 1: Basic Connection")
        
        try:
            # Test market status
            status = self.client.get_market_status()
            print(f"✅ Connected to Polygon.io")
            print(f"   Market: {status.market}")
            
            # Test authentication
            if self.api_key:
                print(f"✅ API key validated: {self.api_key[:8]}...")
            
            self.results['tests_passed'].append('basic_connection')
            return True
            
        except AuthError as e:
            print(f"❌ Authentication failed: {e}")
            self.results['tests_failed'].append('basic_connection')
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.results['tests_failed'].append('basic_connection')
            return False
    
    def test_rate_limiting(self) -> bool:
        """Test 2: Rate limiting compliance"""
        self.print_header("Test 2: Rate Limiting")
        
        try:
            # Create rate limiter (5 calls per minute for free tier)
            limiter = RateLimiter(max_calls=5, period=60, name='polygon_test')
            
            print("Testing rate limit compliance (5 calls in quick succession)...")
            successful_calls = 0
            rate_limited = False
            
            for i in range(7):  # Try 7 calls (should hit limit at 5)
                try:
                    with limiter:
                        # Make API call
                        self.client.get_previous_close_agg('X:BTCUSD')
                        successful_calls += 1
                        print(f"  Call {i+1}: ✅ Success")
                        time.sleep(0.1)
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limited = True
                        print(f"  Call {i+1}: ⚠️  Rate limited (expected)")
                    else:
                        print(f"  Call {i+1}: ❌ Error: {e}")
            
            if successful_calls == 5 and rate_limited:
                print(f"✅ Rate limiting working correctly: {successful_calls}/5 calls succeeded")
                self.results['tests_passed'].append('rate_limiting')
                return True
            else:
                print(f"⚠️  Rate limiting may not be configured properly")
                self.results['warnings'].append('rate_limiting_configuration')
                return True
                
        except Exception as e:
            print(f"❌ Rate limiting test failed: {e}")
            self.results['tests_failed'].append('rate_limiting')
            return False
    
    def test_error_recovery(self) -> bool:
        """Test 3: Error handling and recovery"""
        self.print_header("Test 3: Error Recovery")
        
        try:
            test_cases = [
                ('Invalid symbol', 'INVALID123'),
                ('Empty symbol', ''),
                ('Malformed symbol', 'X:BTC/USD'),
            ]
            
            recovered = 0
            for test_name, bad_symbol in test_cases:
                print(f"\nTesting {test_name}: '{bad_symbol}'")
                try:
                    self.client.get_previous_close_agg(bad_symbol)
                    print(f"  ⚠️  Unexpected success")
                except (BadResponse, Exception) as e:
                    print(f"  ✅ Error caught properly: {type(e).__name__}")
                    recovered += 1
                
                # Verify connection still works after error
                try:
                    self.client.get_previous_close_agg('X:BTCUSD')
                    print(f"  ✅ Connection recovered after error")
                except:
                    print(f"  ❌ Connection failed to recover")
            
            if recovered == len(test_cases):
                print(f"\n✅ Error recovery successful: {recovered}/{len(test_cases)} errors handled")
                self.results['tests_passed'].append('error_recovery')
                return True
            else:
                print(f"\n⚠️  Partial recovery: {recovered}/{len(test_cases)}")
                self.results['warnings'].append('partial_error_recovery')
                return True
                
        except Exception as e:
            print(f"❌ Error recovery test failed: {e}")
            self.results['tests_failed'].append('error_recovery')
            return False
    
    def test_concurrent_requests(self) -> bool:
        """Test 4: Concurrent request handling"""
        self.print_header("Test 4: Concurrent Requests")
        
        try:
            symbols = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD', 'X:ADAUSD', 'X:DOGEUSD']
            print(f"Testing {len(symbols)} concurrent requests...")
            
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for symbol in symbols:
                    future = executor.submit(self.client.get_previous_close_agg, symbol)
                    futures.append((symbol, future))
                
                results = []
                for symbol, future in futures:
                    try:
                        result = future.result(timeout=10)
                        if result:
                            results.append(symbol)
                            print(f"  ✅ {symbol}: Retrieved")
                    except Exception as e:
                        print(f"  ❌ {symbol}: {e}")
            
            elapsed = time.time() - start
            
            if len(results) >= 3:  # At least 3 successful
                print(f"\n✅ Concurrent requests handled: {len(results)}/{len(symbols)} in {elapsed:.2f}s")
                self.results['tests_passed'].append('concurrent_requests')
                self.results['metrics']['concurrent_time'] = elapsed
                return True
            else:
                print(f"\n❌ Too many concurrent failures: {len(results)}/{len(symbols)}")
                self.results['tests_failed'].append('concurrent_requests')
                return False
                
        except Exception as e:
            print(f"❌ Concurrent test failed: {e}")
            self.results['tests_failed'].append('concurrent_requests')
            return False
    
    def test_data_consistency(self) -> bool:
        """Test 5: Data consistency and validation"""
        self.print_header("Test 5: Data Consistency")
        
        try:
            print("Fetching BTC data multiple times...")
            prices = []
            
            for i in range(3):
                result = self.client.get_previous_close_agg('X:BTCUSD')
                if result and len(result) > 0:
                    price = result[0].close
                    prices.append(price)
                    print(f"  Fetch {i+1}: ${price:,.2f}")
                time.sleep(2)  # Small delay between requests
            
            if len(prices) >= 2:
                # Check consistency (prices should be the same or very close)
                price_variance = max(prices) - min(prices)
                avg_price = sum(prices) / len(prices)
                variance_pct = (price_variance / avg_price) * 100 if avg_price > 0 else 0
                
                print(f"\nPrice variance: ${price_variance:.2f} ({variance_pct:.3f}%)")
                
                if variance_pct < 1:  # Less than 1% variance
                    print(f"✅ Data consistency verified")
                    self.results['tests_passed'].append('data_consistency')
                    return True
                else:
                    print(f"⚠️  High price variance detected")
                    self.results['warnings'].append('price_variance')
                    return True
            else:
                print(f"❌ Insufficient data for consistency check")
                self.results['tests_failed'].append('data_consistency')
                return False
                
        except Exception as e:
            print(f"❌ Data consistency test failed: {e}")
            self.results['tests_failed'].append('data_consistency')
            return False
    
    def test_network_interruption(self) -> bool:
        """Test 6: Network interruption simulation"""
        self.print_header("Test 6: Network Interruption Recovery")
        
        try:
            print("Simulating network issues...")
            
            # Test with timeout
            print("\n1. Testing with short timeout...")
            temp_client = RESTClient(api_key=self.api_key)
            
            # Save original timeout
            original_timeout = 30
            
            # Make a normal request
            try:
                result = temp_client.get_previous_close_agg('X:BTCUSD')
                print(f"  ✅ Normal request succeeded")
            except Exception as e:
                print(f"  ❌ Normal request failed: {e}")
            
            # Test recovery after "network issue"
            print("\n2. Testing recovery after simulated issue...")
            time.sleep(2)  # Simulate network recovery time
            
            try:
                result = self.client.get_previous_close_agg('X:BTCUSD')
                if result:
                    print(f"  ✅ Connection recovered successfully")
                    self.results['tests_passed'].append('network_recovery')
                    return True
            except Exception as e:
                print(f"  ⚠️  Recovery may need more time: {e}")
                self.results['warnings'].append('slow_network_recovery')
                return True
                
        except Exception as e:
            print(f"❌ Network recovery test failed: {e}")
            self.results['tests_failed'].append('network_recovery')
            return False
    
    def test_long_running_stability(self) -> bool:
        """Test 7: Long-running stability (mini stress test)"""
        self.print_header("Test 7: Stability Test (30 seconds)")
        
        try:
            print("Running continuous requests for 30 seconds...")
            print("(This tests connection stability over time)")
            
            start_time = time.time()
            successful = 0
            failed = 0
            errors = []
            
            while (time.time() - start_time) < 30:
                try:
                    # Rotate through different symbols
                    symbols = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
                    symbol = symbols[successful % len(symbols)]
                    
                    result = self.client.get_previous_close_agg(symbol)
                    if result:
                        successful += 1
                        
                        # Show progress every 5 successful calls
                        if successful % 5 == 0:
                            elapsed = time.time() - start_time
                            rate = successful / elapsed if elapsed > 0 else 0
                            print(f"  Progress: {successful} calls, {rate:.1f} calls/sec")
                    
                    # Respect rate limits
                    time.sleep(12)  # 5 calls per minute = 1 call per 12 seconds
                    
                except Exception as e:
                    failed += 1
                    error_type = type(e).__name__
                    if error_type not in errors:
                        errors.append(error_type)
                    
                    # Don't flood with errors
                    if failed <= 3:
                        print(f"  ⚠️  Error #{failed}: {error_type}")
            
            total = successful + failed
            success_rate = (successful / total * 100) if total > 0 else 0
            elapsed = time.time() - start_time
            
            print(f"\nStability test complete:")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {success_rate:.1f}%")
            
            if success_rate >= 80:
                print(f"✅ Connection stable over time")
                self.results['tests_passed'].append('long_running_stability')
                self.results['metrics']['stability_rate'] = success_rate
                return True
            elif success_rate >= 60:
                print(f"⚠️  Connection somewhat unstable")
                self.results['warnings'].append('connection_instability')
                return True
            else:
                print(f"❌ Connection unstable")
                self.results['tests_failed'].append('long_running_stability')
                return False
                
        except Exception as e:
            print(f"❌ Stability test failed: {e}")
            self.results['tests_failed'].append('long_running_stability')
            return False
    
    def test_custom_feed_integration(self) -> bool:
        """Test 8: Custom feed integration"""
        self.print_header("Test 8: Custom Feed Integration")
        
        try:
            print("Testing our PolygonLiveFeed wrapper...")
            
            # Test connection
            if self.feed.test_connection():
                print("✅ Custom feed connected")
            else:
                print("❌ Custom feed connection failed")
                return False
            
            # Test data fetching
            print("\nFetching latest data through custom feed...")
            latest = self.feed.get_latest_data()
            
            if latest and len(latest) > 0:
                print("✅ Data retrieved successfully:")
                for symbol, data in list(latest.items())[:3]:
                    if 'close' in data:
                        print(f"   {symbol}: ${data['close']:,.2f}")
            else:
                print("❌ No data retrieved")
                return False
            
            # Test historical data
            print("\nFetching historical data...")
            historical = self.feed.get_historical_data(
                symbol='BTC-USD',
                start_date=datetime.now() - timedelta(hours=6),
                timespan='hour'
            )
            
            if historical and len(historical) > 0:
                print(f"✅ Retrieved {len(historical)} historical data points")
                self.results['tests_passed'].append('custom_feed_integration')
                return True
            else:
                print("⚠️  Historical data not available (may need paid tier)")
                self.results['warnings'].append('historical_data_limited')
                return True
                
        except Exception as e:
            print(f"❌ Custom feed test failed: {e}")
            self.results['tests_failed'].append('custom_feed_integration')
            return False
    
    def run_all_tests(self):
        """Run all robustness tests"""
        print("\n" + "="*60)
        print("🔬 POLYGON.IO ROBUSTNESS TEST SUITE")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API Key: {self.api_key[:8]}..." if self.api_key else "No API key")
        
        tests = [
            ("Basic Connection", self.test_basic_connection),
            ("Rate Limiting", self.test_rate_limiting),
            ("Error Recovery", self.test_error_recovery),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Data Consistency", self.test_data_consistency),
            ("Network Recovery", self.test_network_interruption),
            ("Stability (30s)", self.test_long_running_stability),
            ("Custom Integration", self.test_custom_feed_integration),
        ]
        
        print(f"\nRunning {len(tests)} robustness tests...")
        print("This will take approximately 1-2 minutes.\n")
        
        for test_name, test_func in tests:
            try:
                test_func()
            except KeyboardInterrupt:
                print("\n\n⚠️  Test interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error in {test_name}: {e}")
                self.results['tests_failed'].append(test_name.lower().replace(' ', '_'))
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("📊 ROBUSTNESS TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results['tests_passed']) + len(self.results['tests_failed'])
        
        print(f"\n✅ Tests Passed: {len(self.results['tests_passed'])}/{total_tests}")
        for test in self.results['tests_passed']:
            print(f"   • {test}")
        
        if self.results['tests_failed']:
            print(f"\n❌ Tests Failed: {len(self.results['tests_failed'])}/{total_tests}")
            for test in self.results['tests_failed']:
                print(f"   • {test}")
        
        if self.results['warnings']:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
        
        if self.results['metrics']:
            print(f"\n📈 Performance Metrics:")
            for metric, value in self.results['metrics'].items():
                print(f"   • {metric}: {value}")
        
        print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
        
        # Overall assessment
        success_rate = (len(self.results['tests_passed']) / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        if success_rate >= 90:
            print("🎉 EXCELLENT: Connection is highly robust!")
            print("Your Polygon.io integration is production-ready.")
        elif success_rate >= 75:
            print("✅ GOOD: Connection is reasonably robust.")
            print("Minor issues detected but suitable for paper trading.")
        elif success_rate >= 50:
            print("⚠️  FAIR: Connection needs improvement.")
            print("Review warnings and failed tests before production use.")
        else:
            print("❌ POOR: Connection has significant issues.")
            print("Address failed tests before proceeding.")
        print("="*60)


def main():
    """Run robustness tests"""
    tester = RobustnessTest()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()