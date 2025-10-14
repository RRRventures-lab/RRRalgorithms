from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from src.core.audit_logger import get_audit_logger
from src.core.database.local_db import LocalDatabase
from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
import asyncio
import os
import sys
import time

#!/usr/bin/env python
"""
Start Live Market Data Feed
============================

Connects to Polygon.io and starts monitoring real-time crypto prices.
Updates database with latest market data.
"""


# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv('config/api-keys/.env')

# Simple monitor class for now
class Monitor:
    def log(self, level, message):
        print(f"[{level}] {message}")


class LiveTradingSystem:
    """Live trading system with real market data"""
    
    def __init__(self, symbols=None):
        """Initialize live trading system"""
        self.symbols = symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD']
        self.running = False
        
        # Initialize components
        print("🚀 Initializing Live Trading System...")
        
        # Polygon feed
        print("  📡 Connecting to Polygon.io...")
        self.feed = PolygonLiveFeed(symbols=self.symbols)
        if not self.feed.test_connection():
            raise ConnectionError("Failed to connect to Polygon.io")
        print("  ✅ Polygon connected")
        
        # Database
        print("  💾 Initializing database...")
        self.db = LocalDatabase()
        print("  ✅ Database ready")
        
        # Audit logger
        print("  📝 Starting audit logger...")
        self.audit_logger = get_audit_logger()
        print("  ✅ Audit logging active")
        
        # Monitor
        print("  📊 Starting monitor...")
        self.monitor = Monitor()
        print("  ✅ Monitor ready")
        
        print("\n✅ System initialized successfully!\n")
    
    def display_prices(self):
        """Display current prices"""
        latest = self.feed.get_latest_data()
        
        if not latest:
            print("No data available")
            return
        
        # Clear screen (optional)
        # os.system('clear')
        
        print(f"\n{'='*60}")
        print(f"📈 Live Market Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"{'Symbol':<10} {'Price':>12} {'24h Change':>12} {'Volume':>15}")
        print(f"{'-'*60}")
        
        for symbol, data in latest.items():
            if 'close' in data:
                price = data['close']
                # Calculate 24h change if we have open price
                change_str = "N/A"
                if 'open' in data and data['open'] > 0:
                    change = ((price - data['open']) / data['open']) * 100
                    change_str = f"{change:+.2f}%"
                    if change > 0:
                        change_str = f"🟢 {change_str}"
                    else:
                        change_str = f"🔴 {change_str}"
                
                volume = data.get('volume', 0)
                print(f"{symbol:<10} ${price:>11,.2f} {change_str:>12} {volume:>15,.0f}")
        
        print(f"{'-'*60}\n")
    
    def save_to_database(self, data):
        """Save market data to database"""
        for symbol, ohlcv in data.items():
            if 'close' in ohlcv:
                try:
                    self.db.insert_market_data(
                        symbol=symbol,
                        timestamp=ohlcv.get('timestamp', time.time()),
                        ohlcv=ohlcv
                    )
                except Exception as e:
                    print(f"Error saving {symbol}: {e}")
    
    def run_sync(self, duration=None):
        """Run synchronous monitoring loop"""
        self.running = True
        start_time = time.time()
        iteration = 0
        
        print("Starting live market monitoring...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                iteration += 1
                
                # Fetch latest data
                latest = self.feed.get_latest_data()
                
                if latest:
                    # Display prices
                    self.display_prices()
                    
                    # Save to database
                    self.save_to_database(latest)
                    
                    # Log to audit
                    self.audit_logger.log_event(
                        event_type='DATA_RECEIVED',
                        severity='INFO',
                        component='LiveFeed',
                        action='price_update',
                        details={'symbols': list(latest.keys())}
                    )
                    
                    # Update monitor
                    self.monitor.log('INFO', f'Iteration {iteration}: Updated {len(latest)} symbols')
                
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Wait before next update (respect rate limits)
                time.sleep(15)  # Update every 15 seconds
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            self.stop()
    
    async def run_async(self):
        """Run asynchronous monitoring with WebSocket (if available)"""
        print("Starting async monitoring (WebSocket)...")
        print("Note: WebSocket requires paid Polygon subscription")
        
        try:
            # Start WebSocket
            await self.feed.start_websocket()
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
            print("Falling back to REST API polling...")
            self.run_sync()
    
    def stop(self):
        """Stop the system"""
        self.running = False
        print("Shutting down...")
        
        # Close connections
        self.audit_logger.shutdown()
        self.db.close()
        
        print("✅ System stopped successfully")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Market Data Feed')
    parser.add_argument('--symbols', nargs='+', help='Symbols to monitor')
    parser.add_argument('--async', action='store_true', help='Use async WebSocket mode')
    parser.add_argument('--duration', type=int, help='Run duration in seconds')
    
    args = parser.parse_args()
    
    # Default symbols
    symbols = args.symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Create system
    system = LiveTradingSystem(symbols=symbols)
    
    # Run
    if args.__dict__.get('async'):
        asyncio.run(system.run_async())
    else:
        system.run_sync(duration=args.duration)


if __name__ == "__main__":
    main()