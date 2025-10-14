from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from pathlib import Path
import sys

#!/usr/bin/env python3
"""
Verify Solana backfill statistics from Supabase database
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Query database for backfill statistics."""

    print("=" * 80)
    print("SOLANA (X:SOLUSD) BACKFILL VERIFICATION")
    print("=" * 80)

    # Initialize client
    print("\nConnecting to Supabase...")
    supabase = SupabaseClient()

    ticker = "X:SOLUSD"

    # Calculate date range (24 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=24 * 30)

    print(f"Ticker: {ticker}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"\nQuerying database for bar counts...\n")

    # Query total count
    try:
        result = (
            supabase.client
            .table("crypto_aggregates")
            .select("*", count="exact")
            .eq("ticker", ticker)
            .gte("event_time", start_date.isoformat())
            .lte("event_time", end_date.isoformat())
            .limit(1)
            .execute()
        )

        total_bars = result.count

        print(f"✅ Total bars in database: {total_bars:,}")

        # Get date range of actual data
        earliest = (
            supabase.client
            .table("crypto_aggregates")
            .select("event_time")
            .eq("ticker", ticker)
            .order("event_time", desc=False)
            .limit(1)
            .execute()
        )

        latest = (
            supabase.client
            .table("crypto_aggregates")
            .select("event_time")
            .eq("ticker", ticker)
            .order("event_time", desc=True)
            .limit(1)
            .execute()
        )

        if earliest.data and latest.data:
            print(f"Earliest bar: {earliest.data[0]['event_time']}")
            print(f"Latest bar: {latest.data[0]['event_time']}")

        # Get sample of recent data
        print("\nRecent bars (last 5):")
        print("-" * 80)
        recent = supabase.get_latest_prices(ticker, limit=5)
        for bar in recent:
            print(f"  {bar['event_time']}: O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']}")

        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)

        # Check system events log
        print("\nChecking system events for backfill completion...")
        events = (
            supabase.client
            .table("system_events")
            .select("*")
            .eq("event_type", "backfill_complete")
            .eq("component", "historical_backfill")
            .order("event_time", desc=True)
            .limit(10)
            .execute()
        )

        if events.data:
            print(f"\nFound {len(events.data)} recent backfill events:")
            for event in events.data:
                metadata = event.get('metadata', {})
                print(f"  - {event['event_time']}: {event['message']}")
                if 'bars_count' in metadata:
                    print(f"    Bars: {metadata['bars_count']:,}")

        return 0

    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
