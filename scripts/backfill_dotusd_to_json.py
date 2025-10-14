from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import json
import logging
import os
import sys

#!/usr/bin/env python3
"""
Backfill 2 years of Polkadot (X:DOTUSD) data and save to JSON files.

This script downloads historical cryptocurrency data for X:DOTUSD across all timeframes
and saves them as JSON files (as backup/fallback when Supabase has issues).

Usage:
    python scripts/backfill_dotusd_to_json.py
"""


# Load environment variables
env_path = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env")
load_dotenv(env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Timeframe configurations
TIMEFRAMES = [
    {"multiplier": 1, "timespan": "minute", "name": "1min"},
    {"multiplier": 5, "timespan": "minute", "name": "5min"},
    {"multiplier": 15, "timespan": "minute", "name": "15min"},
    {"multiplier": 1, "timespan": "hour", "name": "1hr"},
    {"multiplier": 4, "timespan": "hour", "name": "4hr"},
    {"multiplier": 1, "timespan": "day", "name": "1day"},
]

TICKER = "X:DOTUSD"
MONTHS = 24  # 2 years
OUTPUT_DIR = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/data/json_backfill")


async def fetch_and_save_timeframe(
    polygon_client: PolygonRESTClient,
    timeframe: dict,
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """
    Fetch data for a single timeframe and save to JSON.

    Returns:
        dict with results including bars_count, file_path, duration
    """
    tf_name = timeframe["name"]
    logger.info(f"\n{'='*70}")
    logger.info(f"Fetching {TICKER} - {tf_name}")
    logger.info(f"{'='*70}")

    start_time = datetime.now()
    all_bars = []
    errors = 0

    try:
        # Fetch data in 30-day chunks
        chunk_days = 30
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

            logger.info(f"Fetching {current_date.date()} to {chunk_end.date()}...")

            try:
                aggregates = polygon_client.get_aggregates(
                    ticker=TICKER,
                    multiplier=timeframe["multiplier"],
                    timespan=timeframe["timespan"],
                    from_date=current_date.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                    limit=50000,
                )

                # Convert to dict format
                for agg in aggregates:
                    all_bars.append({
                        "ticker": TICKER,
                        "timestamp": agg.timestamp,
                        "datetime": agg.datetime.isoformat(),
                        "open": float(agg.open),
                        "high": float(agg.high),
                        "low": float(agg.low),
                        "close": float(agg.close),
                        "volume": float(agg.volume),
                        "vwap": float(agg.vwap) if agg.vwap else None,
                        "trade_count": agg.trade_count,
                    })

                logger.info(f"  Fetched {len(aggregates)} bars (total: {len(all_bars)})")

            except Exception as e:
                logger.error(f"Error fetching chunk: {e}")
                errors += 1

            current_date = chunk_end + timedelta(days=1)
            await asyncio.sleep(1)  # Rate limiting

        # Save to JSON file
        output_file = OUTPUT_DIR / f"{TICKER}_{tf_name}_{start_date.date()}_{end_date.date()}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "ticker": TICKER,
                "timeframe": tf_name,
                "multiplier": timeframe["multiplier"],
                "timespan": timeframe["timespan"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "bars_count": len(all_bars),
                "bars": all_bars,
                "downloaded_at": datetime.now().isoformat(),
            }, f, indent=2)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            "timeframe": tf_name,
            "status": "success",
            "bars_fetched": len(all_bars),
            "errors": errors,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "file_path": str(output_file),
            "file_size_mb": output_file.stat().st_size / (1024 * 1024),
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"Completed {tf_name}:")
        logger.info(f"  Bars fetched: {result['bars_fetched']:,}")
        logger.info(f"  File: {output_file.name}")
        logger.info(f"  Size: {result['file_size_mb']:.2f} MB")
        logger.info(f"  Errors: {result['errors']}")
        logger.info(f"  Duration: {result['duration_minutes']:.2f} minutes")
        logger.info(f"{'='*70}\n")

        return result

    except Exception as e:
        logger.error(f"Error processing {tf_name}: {e}")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "timeframe": tf_name,
            "status": "failed",
            "bars_fetched": len(all_bars),
            "errors": errors + 1,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "error_message": str(e),
        }


async def main():
    """Main function to fetch all timeframes and save to JSON."""

    logger.info("\n" + "="*70)
    logger.info("POLKADOT (X:DOTUSD) JSON BACKUP DOWNLOAD")
    logger.info("="*70)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Time range: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*70 + "\n")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}\n")

    # Initialize Polygon client
    logger.info("Initializing Polygon client...")
    try:
        polygon = PolygonRESTClient()
        logger.info("Polygon client initialized successfully\n")
    except Exception as e:
        logger.error(f"Failed to initialize Polygon client: {e}")
        return

    # Track overall stats
    overall_start = datetime.now()
    results = []

    # Fetch each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"[{i}/{len(TIMEFRAMES)}] Processing {timeframe['name']}...")

        result = await fetch_and_save_timeframe(
            polygon_client=polygon,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        results.append(result)

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("Waiting 5 seconds before next timeframe...")
            await asyncio.sleep(5)

    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()

    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY - ALL TIMEFRAMES")
    logger.info("="*70)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Total duration: {overall_duration / 60:.2f} minutes ({overall_duration / 3600:.2f} hours)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("\nResults by timeframe:")
    logger.info("-" * 70)

    total_bars = 0
    total_size_mb = 0
    total_errors = 0

    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"\n{status_icon} {result['timeframe']}:")
        logger.info(f"    Status: {result['status']}")
        logger.info(f"    Bars: {result['bars_fetched']:,}")

        if result.get("file_path"):
            logger.info(f"    File: {Path(result['file_path']).name}")
            logger.info(f"    Size: {result.get('file_size_mb', 0):.2f} MB")

        logger.info(f"    Errors: {result['errors']}")
        logger.info(f"    Duration: {result['duration_minutes']:.2f} minutes")

        if result.get("error_message"):
            logger.info(f"    Error: {result['error_message']}")

        total_bars += result['bars_fetched']
        total_size_mb += result.get('file_size_mb', 0)
        total_errors += result['errors']

    logger.info("\n" + "-" * 70)
    logger.info("TOTALS:")
    logger.info(f"  Total bars: {total_bars:,}")
    logger.info(f"  Total size: {total_size_mb:.2f} MB")
    logger.info(f"  Total errors: {total_errors}")
    logger.info(f"  Success rate: {(len([r for r in results if r['status'] == 'success']) / len(results) * 100):.1f}%")
    logger.info("="*70 + "\n")

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "ticker": TICKER,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_bars": total_bars,
            "total_size_mb": total_size_mb,
            "total_errors": total_errors,
            "duration_minutes": overall_duration / 60,
            "timeframes": results,
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
