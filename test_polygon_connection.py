from datetime import datetime, timedelta
from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
import os
import sys

#!/usr/bin/env python
"""
Test Polygon.io Connection
==========================

Quick test script to verify Polygon API connectivity.
"""


# Add project to path
sys.path.insert(0, '.')

# Set a test API key if not present (you'll need to add your real key)
if not os.getenv('POLYGON_API_KEY'):
    print("⚠️  No POLYGON_API_KEY found in environment")
    print("Please set your Polygon API key:")
    print("  export POLYGON_API_KEY='your_key_here'")
    print("\nOr add to config/api-keys/.env:")
    print("  POLYGON_API_KEY=your_key_here")
    sys.exit(1)


def test_polygon():
    """Test Polygon connection and data fetching"""
    
    print("🔍 Testing Polygon.io Connection...")
    print("=" * 50)
    
    try:
        # Initialize feed
        feed = PolygonLiveFeed(
            symbols=['BTC-USD', 'ETH-USD', 'SOL-USD']
        )
        
        # Test connection
        print("\n1. Testing API connection...")
        if feed.test_connection():
            print("✅ Connected to Polygon.io successfully!")
        else:
            print("❌ Failed to connect to Polygon.io")
            return False
        
        # Get latest data
        print("\n2. Fetching latest prices...")
        latest = feed.get_latest_data()
        
        if latest:
            print("✅ Latest prices:")
            for symbol, data in latest.items():
                print(f"   {symbol}: ${data['close']:,.2f}")
        else:
            print("⚠️  No latest data available")
        
        # Get historical data
        print("\n3. Fetching 24h historical data...")
        historical = feed.get_historical_data(
            symbol='BTC-USD',
            start_date=datetime.now() - timedelta(days=1),
            timespan='hour'
        )
        
        if historical:
            print(f"✅ Got {len(historical)} hourly data points")
            latest_point = historical[-1]
            print(f"   Latest: ${latest_point.close:,.2f} at {latest_point.timestamp}")
        else:
            print("⚠️  No historical data available")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Polygon.io is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Verify Polygon subscription level")
        return False


if __name__ == "__main__":
    success = test_polygon()
    sys.exit(0 if success else 1)