from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import logging
import os
import sys
import time

#!/usr/bin/env python3
"""
Simple Cardano Data Backfill - Download Only (No Database Storage)
===================================================================

This script downloads 24 months of historical data for Cardano across all timeframes
and reports statistics WITHOUT storing to Supabase.

Usage:
    python backfill_cardano_simple.py
"""


# Load environment variables
env_path = Path(__file__).parent.parent.parent / "config" / "api-keys" / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cardano_simple_backfill.log'),
        logging.StreamHandler()
    ]
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

TICKER = "X:ADAUSD"
MONTHS = 24  # 2 years


async def backfill_timeframe(
    polygon_client: PolygonRESTClient,
    ticker: str,
    timeframe: dict,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """
    Backfill data for a single timeframe.

    Returns:
        dict: Statistics for this timeframe
    """
    timeframe_name = timeframe["name"]
    multiplier = timeframe["multiplier"]
    timespan = timeframe["timespan"]

    logger.info("=" * 80)
    logger.info(f"BACKFILLING: {ticker} - {timeframe_name}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 80)

    start_time = time.time()
    total_bars = 0
    total_chunks = 0
    errors = 0

    try:
        # Split into 30-day chunks to avoid API limits
        chunk_days = 30
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
            total_chunks += 1

            logger.info(f"Chunk {total_chunks}: {current_date.date()} to {chunk_end.date()}")

            try:
                # Fetch aggregates from Polygon
                aggregates = polygon_client.get_aggregates(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=current_date.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                    limit=50000,
                )

                bars_count = len(aggregates)
                total_bars += bars_count

                logger.info(f"  Downloaded {bars_count:,} bars")

                # Small delay to respect rate limits
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"  Error fetching chunk: {e}")
                errors += 1

            # Move to next chunk
            current_date = chunk_end + timedelta(days=1)

        elapsed_time = time.time() - start_time

        result = {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "total_bars": total_bars,
            "total_chunks": total_chunks,
            "errors": errors,
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_minutes": round(elapsed_time / 60, 2),
            "status": "success" if errors == 0 else "completed_with_errors",
            "bars_per_second": round(total_bars / elapsed_time, 2) if elapsed_time > 0 else 0
        }

        logger.info("")
        logger.info(f"COMPLETED {timeframe_name}:")
        logger.info(f"  Total Bars: {total_bars:,}")
        logger.info(f"  Time: {result['elapsed_minutes']:.2f} minutes")
        logger.info(f"  Rate: {result['bars_per_second']:.2f} bars/sec")
        logger.info("")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"FAILED {timeframe_name}: {e}")

        return {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "total_bars": total_bars,
            "total_chunks": total_chunks,
            "errors": errors + 1,
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_minutes": round(elapsed_time / 60, 2),
            "status": "failed",
            "error": str(e)
        }


async def main():
    """Main execution function."""

    logger.info("\n" + "=" * 80)
    logger.info("CARDANO (X:ADAUSD) HISTORICAL DATA DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    for tf in TIMEFRAMES:
        logger.info(f"  - {tf['name']}: {tf['multiplier']} {tf['timespan']}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize Polygon client
    logger.info("Initializing Polygon.io client...")
    try:
        polygon_client = PolygonRESTClient()
        logger.info("Polygon client initialized successfully")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to initialize Polygon client: {e}")
        return

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    # Track overall stats
    overall_start = time.time()
    all_results = []

    # Process each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"[{i}/{len(TIMEFRAMES)}] Processing {timeframe['name']}...")
        logger.info("")

        result = await backfill_timeframe(
            polygon_client=polygon_client,
            ticker=TICKER,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        all_results.append(result)

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("Waiting 5 seconds before next timeframe...")
            logger.info("")
            await asyncio.sleep(5)

    # Calculate overall statistics
    overall_elapsed = time.time() - overall_start
    total_bars = sum(r["total_bars"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)
    successful_timeframes = sum(1 for r in all_results if r["status"] == "success")

    # Print final report
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE - FINAL REPORT")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months ({start_date.date()} to {end_date.date()})")
    logger.info(f"Total Time: {overall_elapsed / 60:.2f} minutes ({overall_elapsed / 3600:.2f} hours)")
    logger.info("")
    logger.info("OVERALL STATISTICS:")
    logger.info(f"  Total Bars Downloaded: {total_bars:,}")
    logger.info(f"  Total Errors: {total_errors}")
    logger.info(f"  Successful Timeframes: {successful_timeframes}/{len(TIMEFRAMES)}")
    logger.info(f"  Average Download Rate: {total_bars / overall_elapsed:.2f} bars/sec")
    logger.info("")
    logger.info("TIMEFRAME BREAKDOWN:")
    logger.info("-" * 80)
    logger.info(f"{'Timeframe':<12} {'Status':<20} {'Total Bars':<15} {'Time (min)':<12} {'Rate (bars/s)':<15} {'Errors':<8}")
    logger.info("-" * 80)

    for result in all_results:
        logger.info(
            f"{result['timeframe']:<12} "
            f"{result['status']:<20} "
            f"{result['total_bars']:>14,} "
            f"{result['elapsed_minutes']:>11.2f} "
            f"{result.get('bars_per_second', 0):>14.2f} "
            f"{result['errors']:>7}"
        )

    logger.info("-" * 80)
    logger.info("")

    # Show any errors
    failed_timeframes = [r for r in all_results if r["status"] == "failed"]
    if failed_timeframes:
        logger.info("ERRORS:")
        for result in failed_timeframes:
            logger.info(f"  {result['timeframe']}: {result.get('error', 'Unknown error')}")
        logger.info("")

    logger.info("Log file: cardano_simple_backfill.log")
    logger.info("=" * 80)

    # Save results to JSON
    import json
    results_file = "cardano_download_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "ticker": TICKER,
            "months": MONTHS,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "overall_elapsed_minutes": round(overall_elapsed / 60, 2),
            "overall_elapsed_hours": round(overall_elapsed / 3600, 2),
            "total_bars_downloaded": total_bars,
            "total_errors": total_errors,
            "successful_timeframes": successful_timeframes,
            "average_download_rate_bars_per_sec": round(total_bars / overall_elapsed, 2) if overall_elapsed > 0 else 0,
            "timeframe_results": all_results,
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Results saved to: {results_file}")
    logger.info("")

    return all_results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
