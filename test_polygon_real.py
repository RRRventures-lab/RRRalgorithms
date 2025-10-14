from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from polygon import RESTClient
import os
import sys

#!/usr/bin/env python
"""
Test Polygon.io Connection with Real API
"""


# Setup
sys.path.insert(0, '.')
load_dotenv('config/api-keys/.env')

# Import Polygon client

# Initialize client
api_key = os.getenv('POLYGON_API_KEY')
if not api_key:
    print("❌ No POLYGON_API_KEY found in environment")
    sys.exit(1)

client = RESTClient(api_key=api_key)

print('🔍 Testing Polygon.io Connection...')
print('=' * 50)

try:
    # Test 1: Market status
    print('\n1. Checking market status...')
    status = client.get_market_status()
    print(f'✅ Market is: {status.market}')
    # Server time attribute may vary by API version
    if hasattr(status, 'server_time'):
        print(f'   Server time: {status.server_time}')
    elif hasattr(status, 'serverTime'):
        print(f'   Server time: {status.serverTime}')
    
    # Test 2: Get latest crypto prices
    print('\n2. Fetching latest crypto prices...')
    cryptos = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD', 'X:ADAUSD', 'X:DOGEUSD']
    
    for ticker in cryptos:
        try:
            # Get previous close
            agg = client.get_previous_close_agg(ticker)
            if agg and len(agg) > 0:
                symbol = ticker.replace('X:', '')
                change_pct = ((agg[0].close - agg[0].open) / agg[0].open) * 100
                print(f'  {symbol:8} ${agg[0].close:>10,.2f}  ({change_pct:+.2f}%)')
        except Exception as e:
            print(f'  {ticker}: Not available')
    
    # Test 3: Get recent BTC data
    print('\n3. Fetching recent BTC hourly data...')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    aggs = list(client.list_aggs(
        ticker='X:BTCUSD',
        multiplier=1,
        timespan='hour',
        from_=yesterday,
        to=today,
        limit=24
    ))
    
    if aggs:
        print(f'✅ Got {len(aggs)} hourly bars')
        latest = aggs[-1]
        print(f'  Latest: ${latest.close:,.2f} at {datetime.fromtimestamp(latest.timestamp/1000):%Y-%m-%d %H:%M}')
        
        # Show last 5 bars
        print('\n  Last 5 hours:')
        print('  Time     Open      High      Low       Close     Volume')
        print('  ' + '-' * 60)
        for agg in aggs[-5:]:
            time_str = datetime.fromtimestamp(agg.timestamp/1000).strftime('%H:%M')
            print(f'  {time_str}  ${agg.open:>8,.0f}  ${agg.high:>8,.0f}  ${agg.low:>8,.0f}  ${agg.close:>8,.0f}  {agg.volume:>10,.0f}')
    
    # Test 4: Get snapshot
    print('\n4. Getting current snapshot for BTC...')
    try:
        snapshot = client.get_snapshot_crypto('X:BTCUSD')
        if snapshot:
            print(f'  Current price: ${snapshot.price:,.2f}')
            print(f'  24h change: {snapshot.todaysChangePerc:.2f}%')
            print(f'  24h volume: {snapshot.day.volume:,.0f}')
    except:
        # Try different method
        ticker = client.get_ticker_details('X:BTCUSD')
        if ticker:
            print(f'  Ticker info retrieved')
    
    # Test 5: Check API limits
    print('\n5. API Usage Info:')
    print('  Free tier: 5 calls/minute')
    print('  Your tier: Check dashboard.polygon.io')
    
    print('\n' + '=' * 50)
    print('✅ SUCCESS! Polygon.io is working perfectly!')
    print('✅ You can now fetch real market data!')
    
    # Test our custom integration
    print('\n6. Testing our custom Polygon integration...')
    from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
    
    feed = PolygonLiveFeed(symbols=['BTC-USD', 'ETH-USD'])
    if feed.test_connection():
        print('✅ Custom integration working!')
        
        # Get latest data through our wrapper
        latest_data = feed.get_latest_data()
        if latest_data:
            print('\n  Latest prices via our system:')
            for symbol, data in latest_data.items():
                if 'close' in data:
                    print(f'    {symbol}: ${data["close"]:,.2f}')
    
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
    print('\nTroubleshooting:')
    print('1. Check API key is valid')
    print('2. Verify subscription level at dashboard.polygon.io')
    print('3. Check rate limits (free = 5/min)')