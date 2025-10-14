from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import csv
import json
import os
import sys

#!/usr/bin/env python3
"""
Simple Chainlink Historical Data Downloader
============================================

Downloads 24 months of historical data for Chainlink and saves to local files.
This version bypasses Supabase and saves data directly to CSV/Parquet files.
"""


# Load environment variables
load_dotenv("/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env")

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

DATA_DIR = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/data/linkusd")


async def download_timeframe(polygon, ticker, timeframe_config):
    """Download data for a single timeframe."""

    tf_name = timeframe_config["name"]
    multiplier = timeframe_config["multiplier"]
    timespan = timeframe_config["timespan"]

    print(f"\n{'='*80}")
    print(f"DOWNLOADING: {tf_name} timeframe")
    print(f"{'='*80}")

    # Calculate date range (24 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~24 months

    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Ticker: {ticker}")
    print()

    # Create output directory
    output_dir = DATA_DIR / tf_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    csv_file = output_dir / f"{ticker.replace(':', '_')}_{tf_name}.csv"

    all_bars = []
    total_bars = 0
    chunks_fetched = 0
    errors = 0

    # Fetch data in 30-day chunks
    chunk_days = 30
    current_date = start_date

    start_time = datetime.now()

    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

        print(f"  Fetching: {current_date.date()} to {chunk_end.date()}", end="")

        try:
            aggregates = polygon.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=current_date.strftime("%Y-%m-%d"),
                to_date=chunk_end.strftime("%Y-%m-%d"),
                limit=50000,
            )

            if aggregates:
                # Convert to dict format
                for agg in aggregates:
                    all_bars.append({
                        "timestamp": agg.datetime.isoformat(),
                        "open": float(agg.open),
                        "high": float(agg.high),
                        "low": float(agg.low),
                        "close": float(agg.close),
                        "volume": float(agg.volume),
                        "vwap": float(agg.vwap) if agg.vwap else None,
                        "trade_count": agg.trade_count,
                    })

                total_bars += len(aggregates)
                chunks_fetched += 1
                print(f" ✓ {len(aggregates)} bars")
            else:
                print(f" ✗ No data")

        except Exception as e:
            print(f" ✗ Error: {str(e)[:50]}")
            errors += 1

        current_date = chunk_end + timedelta(days=1)
        await asyncio.sleep(0.2)  # Rate limiting

    # Save to CSV
    if all_bars:
        print(f"\n  Saving to {csv_file}...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_bars[0].keys())
            writer.writeheader()
            writer.writerows(all_bars)
        print(f"  ✓ Saved {total_bars:,} bars to CSV")

        # Also save a JSON summary
        summary_file = output_dir / f"{ticker.replace(':', '_')}_{tf_name}_summary.json"
        duration = datetime.now() - start_time
        summary = {
            "ticker": ticker,
            "timeframe": tf_name,
            "total_bars": total_bars,
            "chunks_fetched": chunks_fetched,
            "errors": errors,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "download_duration": str(duration),
            "csv_file": str(csv_file),
            "file_size_mb": round(csv_file.stat().st_size / 1024 / 1024, 2)
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Summary saved to {summary_file}")

    duration = datetime.now() - start_time

    return {
        "timeframe": tf_name,
        "bars": total_bars,
        "chunks": chunks_fetched,
        "errors": errors,
        "duration": duration,
        "csv_file": str(csv_file) if all_bars else None
    }


async def main():
    """Main download function."""

    ticker = "X:LINKUSD"

    print("="*80)
    print("CHAINLINK (X:LINKUSD) HISTORICAL DATA DOWNLOAD")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: 24 months (2 years)")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"Output Directory: {DATA_DIR}")
    print("="*80)

    # Initialize Polygon client
    try:
        polygon = PolygonRESTClient()
        print("✓ Polygon client initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize Polygon: {e}")
        return

    # Download each timeframe
    overall_start = datetime.now()
    results = []

    for i, tf in enumerate(TIMEFRAMES, 1):
        print(f"\n[{i}/{len(TIMEFRAMES)}]")
        result = await download_timeframe(polygon, ticker, tf)
        results.append(result)

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            print(f"\nWaiting 3 seconds before next timeframe...")
            await asyncio.sleep(3)

    # Final summary
    overall_duration = datetime.now() - overall_start
    total_bars = sum(r["bars"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Total Duration: {overall_duration}")
    print(f"Total Bars Downloaded: {total_bars:,}")
    print(f"Total Errors: {total_errors}")
    print()
    print("BREAKDOWN BY TIMEFRAME:")
    print("-"*80)
    print(f"{'Timeframe':<12} {'Bars':>15} {'Chunks':>10} {'Errors':>10} {'Duration':<20}")
    print("-"*80)

    for result in results:
        print(
            f"{result['timeframe']:<12} "
            f"{result['bars']:>15,} "
            f"{result['chunks']:>10} "
            f"{result['errors']:>10} "
            f"{str(result['duration']):<20}"
        )

    print("-"*80)
    print()
    print("DATA FILES:")
    for result in results:
        if result['csv_file']:
            print(f"  {result['timeframe']}: {result['csv_file']}")

    print()
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
