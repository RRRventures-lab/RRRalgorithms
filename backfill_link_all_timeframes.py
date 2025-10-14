from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import os
import sys

#!/usr/bin/env python3
"""
Backfill 2 Years of Chainlink (X:LINKUSD) Historical Data
==========================================================

Downloads 24 months of historical data for Chainlink across all timeframes:
- 1 minute
- 5 minute
- 15 minute
- 1 hour
- 4 hour
- 1 day

This script will:
1. Initialize Supabase and Polygon clients
2. Download data for each timeframe sequentially
3. Track progress and handle rate limiting
4. Generate detailed statistics report
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))



# Timeframe configurations
TIMEFRAMES = [
    {"multiplier": 1, "timespan": "minute", "name": "1min"},
    {"multiplier": 5, "timespan": "minute", "name": "5min"},
    {"multiplier": 15, "timespan": "minute", "name": "15min"},
    {"multiplier": 1, "timespan": "hour", "name": "1hr"},
    {"multiplier": 4, "timespan": "hour", "name": "4hr"},
    {"multiplier": 1, "timespan": "day", "name": "1day"},
]


async def backfill_all_timeframes():
    """
    Backfill all timeframes for Chainlink (X:LINKUSD).
    """
    print("=" * 80)
    print("CHAINLINK (X:LINKUSD) HISTORICAL DATA BACKFILL")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: 24 months (2 years)")
    print(f"Ticker: X:LINKUSD")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print("=" * 80)
    print()

    # Initialize clients
    print("Initializing clients...")
    try:
        supabase = SupabaseClient()
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase: {e}")
        return

    try:
        polygon = PolygonRESTClient()
        print("✅ Polygon client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Polygon: {e}")
        return

    print()

    # Overall statistics
    overall_start = datetime.now()
    total_bars = 0
    total_errors = 0
    timeframe_results = []

    # Process each timeframe
    for i, tf in enumerate(TIMEFRAMES, 1):
        print("=" * 80)
        print(f"TIMEFRAME {i}/{len(TIMEFRAMES)}: {tf['name']}")
        print("=" * 80)

        tf_start = datetime.now()

        try:
            # Create backfill instance for this specific ticker and timeframe
            backfill = HistoricalDataBackfill(
                polygon_client=polygon,
                supabase_client=supabase,
                tickers=["X:LINKUSD"],
                progress_dir=f"/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/progress/{tf['name']}"
            )

            # Run backfill for 24 months
            bars = await backfill.backfill_aggregates(
                months=24,
                multiplier=tf["multiplier"],
                timespan=tf["timespan"],
            )

            tf_duration = datetime.now() - tf_start
            stats = backfill.get_stats()

            # Record results
            result = {
                "timeframe": tf["name"],
                "bars_fetched": stats["bars_fetched"],
                "bars_stored": stats["bars_stored"],
                "errors": stats["errors"],
                "duration": str(tf_duration),
                "duration_seconds": tf_duration.total_seconds(),
            }
            timeframe_results.append(result)

            total_bars += stats["bars_stored"]
            total_errors += stats["errors"]

            print()
            print(f"✅ {tf['name']} COMPLETE")
            print(f"   Bars fetched: {stats['bars_fetched']:,}")
            print(f"   Bars stored: {stats['bars_stored']:,}")
            print(f"   Errors: {stats['errors']}")
            print(f"   Duration: {tf_duration}")
            print()

        except Exception as e:
            print(f"❌ ERROR processing {tf['name']}: {e}")
            total_errors += 1
            timeframe_results.append({
                "timeframe": tf["name"],
                "bars_fetched": 0,
                "bars_stored": 0,
                "errors": 1,
                "duration": "N/A",
                "error_message": str(e)
            })

        # Small delay between timeframes to avoid rate limiting
        if i < len(TIMEFRAMES):
            print("Waiting 5 seconds before next timeframe...")
            await asyncio.sleep(5)
            print()

    # Final summary
    overall_duration = datetime.now() - overall_start

    print()
    print("=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Ticker: X:LINKUSD")
    print(f"Total Duration: {overall_duration}")
    print(f"Total Bars Downloaded: {total_bars:,}")
    print(f"Total Errors: {total_errors}")
    print()
    print("BREAKDOWN BY TIMEFRAME:")
    print("-" * 80)
    print(f"{'Timeframe':<12} {'Bars Fetched':<15} {'Bars Stored':<15} {'Errors':<10} {'Duration':<15}")
    print("-" * 80)

    for result in timeframe_results:
        print(
            f"{result['timeframe']:<12} "
            f"{result['bars_fetched']:>14,} "
            f"{result['bars_stored']:>14,} "
            f"{result['errors']:>9} "
            f"{result['duration']:<15}"
        )

    print("-" * 80)
    print()

    # Success metrics
    success_rate = ((total_bars / sum(r['bars_fetched'] for r in timeframe_results if r['bars_fetched'] > 0)) * 100) if total_bars > 0 else 0
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Time per Timeframe: {overall_duration.total_seconds() / len(TIMEFRAMES):.1f} seconds")
    print()

    # Estimated data coverage
    print("ESTIMATED DATA COVERAGE:")
    print("-" * 80)
    for result in timeframe_results:
        if result['bars_stored'] > 0:
            # Calculate expected bars for 24 months
            tf_name = result['timeframe']
            if "min" in tf_name:
                mins = int(tf_name.replace("min", ""))
                # Market is 24/7 for crypto
                expected_bars = (24 * 60 / mins) * 730  # 730 days in 24 months
            elif "hr" in tf_name:
                hrs = int(tf_name.replace("hr", ""))
                expected_bars = (24 / hrs) * 730
            elif "day" in tf_name:
                expected_bars = 730

            coverage = (result['bars_stored'] / expected_bars) * 100
            print(f"{tf_name:<12}: {coverage:.1f}% coverage ({result['bars_stored']:,} / ~{expected_bars:,.0f} expected bars)")

    print()
    print("=" * 80)
    print("BACKFILL COMPLETE!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Save detailed report
    report_path = "/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/backfill_report.txt"
    try:
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CHAINLINK (X:LINKUSD) HISTORICAL DATA BACKFILL REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {overall_duration}\n")
            f.write(f"Total Bars: {total_bars:,}\n")
            f.write(f"Total Errors: {total_errors}\n")
            f.write("\n")
            f.write("TIMEFRAME BREAKDOWN:\n")
            f.write("-" * 80 + "\n")
            for result in timeframe_results:
                f.write(f"\n{result['timeframe']}:\n")
                f.write(f"  Bars Fetched: {result['bars_fetched']:,}\n")
                f.write(f"  Bars Stored: {result['bars_stored']:,}\n")
                f.write(f"  Errors: {result['errors']}\n")
                f.write(f"  Duration: {result['duration']}\n")
                if 'error_message' in result:
                    f.write(f"  Error Message: {result['error_message']}\n")
        print(f"\n📄 Detailed report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(backfill_all_timeframes())
    except KeyboardInterrupt:
        print("\n\n⚠️  Backfill interrupted by user. Progress has been saved.")
        print("You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
