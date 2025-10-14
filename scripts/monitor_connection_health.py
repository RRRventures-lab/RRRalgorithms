from collections import deque
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from polygon import RESTClient
from src.core.audit_logger import get_audit_logger
from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
from typing import Dict, List, Optional
import json
import os
import sys
import time

#!/usr/bin/env python

"""
Polygon Connection Health Monitor
==================================

Continuously monitors the health and performance of the Polygon.io connection.
Tracks metrics, logs issues, and alerts on problems.
"""


# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv('config/api-keys/.env')



class ConnectionHealthMonitor:
    """Monitor Polygon.io connection health"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.client = RESTClient(api_key=self.api_key)
        self.feed = PolygonLiveFeed(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'])
        self.audit_logger = get_audit_logger()
        
        # Health metrics
        self.metrics = {
            'uptime_seconds': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0,
            'last_error': None,
            'last_success': None,
            'current_status': 'UNKNOWN'
        }
        
        # Rolling windows for tracking
        self.response_times = deque(maxlen=100)  # Last 100 response times
        self.error_log = deque(maxlen=50)  # Last 50 errors
        self.success_rate_window = deque(maxlen=100)  # Last 100 requests (True/False)
        
        self.start_time = time.time()
        self.running = False
    
    def check_connection(self) -> bool:
        """Perform single health check"""
        start = time.time()
        
        try:
            # Try to get market status
            status = self.client.get_market_status()
            
            # Calculate response time
            response_time = (time.time() - start) * 1000  # ms
            self.response_times.append(response_time)
            
            # Update metrics
            self.metrics['successful_requests'] += 1
            self.metrics['last_success'] = datetime.now().isoformat()
            self.success_rate_window.append(True)
            
            return True
            
        except Exception as e:
            # Log error
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': type(e).__name__
            }
            self.error_log.append(error_info)
            
            # Update metrics
            self.metrics['failed_requests'] += 1
            self.metrics['last_error'] = error_info
            self.success_rate_window.append(False)
            
            return False
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        # Success rate (40% weight)
        if len(self.success_rate_window) > 0:
            success_rate = sum(self.success_rate_window) / len(self.success_rate_window)
            score -= (1 - success_rate) * 40
        
        # Response time (30% weight)
        if self.response_times:
            avg_response = sum(self.response_times) / len(self.response_times)
            if avg_response > 1000:  # Over 1 second
                score -= 20
            elif avg_response > 500:  # Over 500ms
                score -= 10
        
        # Recent errors (20% weight)
        recent_errors = sum(1 for e in self.error_log 
                          if datetime.fromisoformat(e['timestamp']) > 
                          datetime.now() - timedelta(minutes=5))
        if recent_errors > 5:
            score -= 20
        elif recent_errors > 2:
            score -= 10
        
        # Uptime (10% weight)
        if self.metrics['uptime_seconds'] < 60:
            score -= 10  # Penalty for just started
        
        return max(0, min(100, score))
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> str:
        """Get current connection status"""
        score = self.calculate_health_score()
        
        if score >= 90:
            return "🟢 HEALTHY"
        elif score >= 70:
            return "🟡 DEGRADED"
        elif score >= 50:
            return "🟠 UNSTABLE"
        else:
            return "🔴 CRITICAL"
    
    def print_dashboard(self):
        """Print health dashboard"""
        # Clear screen (optional)
        # os.system('clear')
        
        uptime = time.time() - self.start_time
        self.metrics['uptime_seconds'] = uptime
        
        # Calculate metrics
        total = self.metrics['successful_requests'] + self.metrics['failed_requests']
        success_rate = (self.metrics['successful_requests'] / total * 100) if total > 0 else 0
        avg_response = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        health_score = self.calculate_health_score()
        status = self.get_status()
        
        print(f"\n{'='*60}")
        print(f"📊 POLYGON CONNECTION HEALTH MONITOR")
        print(f"{'='*60}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Uptime: {uptime:.0f} seconds")
        print(f"\nStatus: {status}")
        print(f"Health Score: {health_score:.0f}/100")
        print(f"\n{'─'*60}")
        print(f"📈 Metrics:")
        print(f"  Total Requests: {total}")
        print(f"  Successful: {self.metrics['successful_requests']}")
        print(f"  Failed: {self.metrics['failed_requests']}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Response Time: {avg_response:.1f}ms")
        
        if self.metrics['last_success']:
            print(f"\n✅ Last Success: {self.metrics['last_success']}")
        
        if self.metrics['last_error']:
            print(f"❌ Last Error: {self.metrics['last_error']['timestamp']}")
            print(f"   Type: {self.metrics['last_error']['type']}")
        
        # Show recent errors if any
        recent_errors = [e for e in self.error_log 
                        if datetime.fromisoformat(e['timestamp']) > 
                        datetime.now() - timedelta(minutes=5)]
        if recent_errors:
            print(f"\n⚠️  Recent Errors (last 5 min): {len(recent_errors)}")
            for err in recent_errors[-3:]:  # Show last 3
                print(f"   • {err['timestamp']}: {err['type']}")
        
        print(f"{'─'*60}")
        
        # Recommendations based on health
        if health_score < 50:
            print("\n⚠️  RECOMMENDATIONS:")
            print("  • Check API key validity")
            print("  • Verify internet connection")
            print("  • Check Polygon service status")
            print("  • Review rate limits")
        elif health_score < 80:
            print("\n💡 SUGGESTIONS:")
            print("  • Monitor for patterns in errors")
            print("  • Consider upgrading API tier if rate limited")
        else:
            print("\n✅ Connection is healthy and stable")
        
        print(f"{'='*60}\n")
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = Path('logs/connection_health.json')
        metrics_file.parent.mkdir(exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'health_score': self.calculate_health_score(),
            'status': self.get_status(),
            'metrics': self.metrics,
            'recent_errors': list(self.error_log)[-10:]  # Last 10 errors
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(self, check_interval=30):
        """Run continuous health monitoring"""
        self.running = True
        print("Starting Polygon Connection Health Monitor...")
        print(f"Checking every {check_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Perform health check
                success = self.check_connection()
                self.metrics['total_requests'] += 1
                
                # Update status
                self.metrics['current_status'] = "UP" if success else "DOWN"
                
                # Print dashboard
                self.print_dashboard()
                
                # Save metrics
                self.save_metrics()
                
                # Log to audit
                if not success and self.metrics['last_error']:
                    self.audit_logger.log_error(
                        component='ConnectionMonitor',
                        error_type=self.metrics['last_error']['type'],
                        error_message=self.metrics['last_error']['error']
                    )
                
                # Alert on critical issues
                health = self.calculate_health_score()
                if health < 50 and self.metrics['total_requests'] > 10:
                    print("\n🚨 ALERT: Connection health critical! Check logs.")
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        print("\nFinal Statistics:")
        print(f"  Total uptime: {self.metrics['uptime_seconds']:.0f} seconds")
        print(f"  Total requests: {self.metrics['total_requests']}")
        print(f"  Success rate: {(self.metrics['successful_requests'] / max(1, self.metrics['total_requests']) * 100):.1f}%")
        print(f"  Final health score: {self.calculate_health_score():.0f}/100")
        
        # Save final metrics
        self.save_metrics()
        
        # Shutdown audit logger
        self.audit_logger.shutdown()


def main():
    """Run health monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Polygon connection health')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--duration', type=int, 
                       help='Run duration in seconds (optional)')
    
    args = parser.parse_args()
    
    monitor = ConnectionHealthMonitor()
    
    if args.duration:
        # Run for specific duration
        import threading
        timer = threading.Timer(args.duration, lambda: setattr(monitor, 'running', False))
        timer.start()
    
    monitor.run(check_interval=args.interval)


if __name__ == "__main__":
    main()