from data_pipeline.polygon import PolygonRESTClient
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

"""
Demo script showing how to use the Polygon.io REST client.

Run with:
    python examples/polygon_demo.py
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Load environment variables
load_dotenv("../../config/api-keys/.env")


def main():
    print("=" * 60)
    print("Polygon.io REST API Demo")
    print("=" * 60)
    print()

    # Initialize client
    print("📡 Initializing Polygon REST client...")
    client = PolygonRESTClient(
        rate_limit=5,  # Free tier: 5 requests/second
        cache_ttl=300,  # Cache for 5 minutes
        enable_cache=True,
    )
    print("✅ Client initialized")
    print()

    # Example 1: Get latest BTC price
    print("=" * 60)
    print("Example 1: Get Latest BTC Price")
    print("=" * 60)
    try:
        trade = client.get_last_trade("X:BTCUSD")
        print(f"Ticker: {trade.ticker}")
        print(f"Price: ${trade.price:,.2f}")
        print(f"Size: {trade.size}")
        print(f"Time: {trade.datetime}")
        print(f"Exchange: {trade.exchange}")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 2: Get BTC bid/ask quote
    print("=" * 60)
    print("Example 2: Get Latest BTC Quote (Bid/Ask)")
    print("=" * 60)
    try:
        quote = client.get_last_quote("X:BTCUSD")
        print(f"Bid: ${quote.bid_price:,.2f} (size: {quote.bid_size})")
        print(f"Ask: ${quote.ask_price:,.2f} (size: {quote.ask_size})")
        print(f"Spread: ${quote.spread:,.2f}")
        print(f"Mid Price: ${quote.mid_price:,.2f}")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 3: Get historical daily bars
    print("=" * 60)
    print("Example 3: Get Last 7 Days of BTC Daily Bars")
    print("=" * 60)
    try:
        bars = client.get_daily_bars("X:BTCUSD", days_back=7)
        print(f"Retrieved {len(bars)} daily bars")
        print()
        print(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15}")
        print("-" * 70)
        for bar in bars[-7:]:  # Last 7 days
            date_str = bar.datetime.strftime("%Y-%m-%d")
            print(
                f"{date_str:<12} "
                f"{bar.open:>10,.2f} "
                f"{bar.high:>10,.2f} "
                f"{bar.low:>10,.2f} "
                f"{bar.close:>10,.2f} "
                f"{bar.volume:>15,.0f}"
            )
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 4: Get minute bars (last 2 hours)
    print("=" * 60)
    print("Example 4: Get 1-Minute BTC Bars (Last 2 Hours)")
    print("=" * 60)
    try:
        from datetime import datetime, timedelta

        to_date = datetime.now()
        from_date = to_date - timedelta(hours=2)

        bars = client.get_aggregates(
            ticker="X:BTCUSD",
            multiplier=1,
            timespan="minute",
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            limit=120,  # 2 hours = 120 minutes
        )

        print(f"Retrieved {len(bars)} 1-minute bars")
        if bars:
            print()
            print("Last 5 bars:")
            print(f"{'Time':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
            print("-" * 80)
            for bar in bars[-5:]:
                time_str = bar.datetime.strftime("%Y-%m-%d %H:%M")
                print(
                    f"{time_str:<20} "
                    f"{bar.open:>10,.2f} "
                    f"{bar.high:>10,.2f} "
                    f"{bar.low:>10,.2f} "
                    f"{bar.close:>10,.2f} "
                    f"{bar.volume:>12,.2f}"
                )
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 5: Get ticker details
    print("=" * 60)
    print("Example 5: Get BTC Ticker Details")
    print("=" * 60)
    try:
        details = client.get_ticker_details("X:BTCUSD")
        print(f"Ticker: {details.ticker}")
        print(f"Name: {details.name}")
        print(f"Market: {details.market}")
        print(f"Type: {details.type}")
        print(f"Active: {details.active}")
        print(f"Base Currency: {details.base_currency_name}")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 6: List crypto tickers
    print("=" * 60)
    print("Example 6: List Available Crypto Tickers")
    print("=" * 60)
    try:
        tickers = client.list_crypto_tickers(active=True)
        print(f"Found {len(tickers)} active crypto tickers")
        print()
        print("Top 10 crypto pairs:")
        for ticker in tickers[:10]:
            print(f"  - {ticker.ticker}: {ticker.name}")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Example 7: Market status
    print("=" * 60)
    print("Example 7: Get Market Status")
    print("=" * 60)
    try:
        status = client.get_market_status()
        print(f"Market: {status.market}")
        print(f"Server Time: {status.server_time}")
        print(f"Exchanges: {list(status.exchanges.keys())[:5]}...")  # Show first 5
        print(f"Currencies: {list(status.currencies.keys())[:5]}...")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

    # Show client statistics
    print("=" * 60)
    print("Client Statistics")
    print("=" * 60)
    stats = client.get_stats()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Errors: {stats['errors']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run this script with your Polygon API key")
    print("2. Test WebSocket client for real-time data")
    print("3. Store data in PostgreSQL database")
    print("4. Build feature engineering pipeline")
    print()


if __name__ == "__main__":
    main()
