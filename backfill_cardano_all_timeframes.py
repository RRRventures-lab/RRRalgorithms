from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import logging
import os
import sys
import time

#!/usr/bin/env python3
"""
Backfill 2 Years of Cardano (X:ADAUSD) Data Across All Timeframes
==================================================================

This script downloads 24 months of historical data for Cardano across:
- 1-minute bars
- 5-minute bars
- 15-minute bars
- 1-hour bars
- 4-hour bars
- 1-day bars

Usage:
    python backfill_cardano_all_timeframes.py
"""


# Load environment variables from config
env_path = Path(__file__).parent.parent.parent / "config" / "api-keys" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded environment from: {env_path}")
else:
    logging.warning(f"Environment file not found at: {env_path}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cardano_backfill.log'),
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


async def backfill_single_timeframe(
    polygon_client: PolygonRESTClient,
    supabase_client: SupabaseClient,
    ticker: str,
    timeframe: dict,
    months: int
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
    logger.info(f"STARTING BACKFILL: {ticker} - {timeframe_name}")
    logger.info("=" * 80)

    start_time = time.time()

    # Create backfill instance for this timeframe
    # Use a unique progress file for each timeframe
    progress_dir = Path.cwd() / "progress"
    progress_dir.mkdir(exist_ok=True)

    backfill = HistoricalDataBackfill(
        polygon_client=polygon_client,
        supabase_client=supabase_client,
        tickers=[ticker],
        progress_dir=str(progress_dir / timeframe_name)
    )

    try:
        # Run backfill
        total_bars = await backfill.backfill_aggregates(
            months=months,
            multiplier=multiplier,
            timespan=timespan,
        )

        elapsed_time = time.time() - start_time
        stats = backfill.get_stats()

        result = {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "bars_fetched": stats["bars_fetched"],
            "bars_stored": stats["bars_stored"],
            "errors": stats["errors"],
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_minutes": round(elapsed_time / 60, 2),
            "status": "success" if stats["errors"] == 0 else "completed_with_errors"
        }

        logger.info(f"COMPLETED {timeframe_name}: {result['bars_stored']} bars in {result['elapsed_minutes']:.2f} minutes")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"FAILED {timeframe_name}: {e}")

        return {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "bars_fetched": 0,
            "bars_stored": 0,
            "errors": 1,
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_minutes": round(elapsed_time / 60, 2),
            "status": "failed",
            "error": str(e)
        }


async def main():
    """Main execution function."""

    logger.info("\n" + "=" * 80)
    logger.info("CARDANO (X:ADAUSD) HISTORICAL DATA BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    for tf in TIMEFRAMES:
        logger.info(f"  - {tf['name']}: {tf['multiplier']} {tf['timespan']}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize clients
    logger.info("Initializing Polygon.io and Supabase clients...")
    try:
        polygon_client = PolygonRESTClient()
        try:
            supabase_client = SupabaseClient()
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.warning(f"Supabase client initialization failed: {e}")
            logger.warning("Continuing without Supabase storage (data will be fetched but not stored)")
            supabase_client = None
        logger.info("Polygon client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Polygon client: {e}")
        return

    # Track overall stats
    overall_start = time.time()
    all_results = []

    # Process each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n[{i}/{len(TIMEFRAMES)}] Processing {timeframe['name']}...")

        result = await backfill_single_timeframe(
            polygon_client=polygon_client,
            supabase_client=supabase_client,
            ticker=TICKER,
            timeframe=timeframe,
            months=MONTHS
        )

        all_results.append(result)

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("Waiting 5 seconds before next timeframe...")
            await asyncio.sleep(5)

    # Calculate overall statistics
    overall_elapsed = time.time() - overall_start
    total_bars_fetched = sum(r["bars_fetched"] for r in all_results)
    total_bars_stored = sum(r["bars_stored"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)
    successful_timeframes = sum(1 for r in all_results if r["status"] == "success")

    # Print final report
    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE - FINAL REPORT")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months")
    logger.info(f"Total Time: {overall_elapsed / 60:.2f} minutes ({overall_elapsed / 3600:.2f} hours)")
    logger.info("")
    logger.info("OVERALL STATISTICS:")
    logger.info(f"  Total Bars Fetched: {total_bars_fetched:,}")
    logger.info(f"  Total Bars Stored: {total_bars_stored:,}")
    logger.info(f"  Total Errors: {total_errors}")
    logger.info(f"  Successful Timeframes: {successful_timeframes}/{len(TIMEFRAMES)}")
    logger.info("")
    logger.info("TIMEFRAME BREAKDOWN:")
    logger.info("-" * 80)
    logger.info(f"{'Timeframe':<12} {'Status':<20} {'Bars Stored':<15} {'Time (min)':<12} {'Errors':<8}")
    logger.info("-" * 80)

    for result in all_results:
        logger.info(
            f"{result['timeframe']:<12} "
            f"{result['status']:<20} "
            f"{result['bars_stored']:>14,} "
            f"{result['elapsed_minutes']:>11.2f} "
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

    logger.info("Log file: cardano_backfill.log")
    logger.info("=" * 80)

    # Save results to JSON
    import json
    results_file = "cardano_backfill_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "ticker": TICKER,
            "months": MONTHS,
            "overall_elapsed_minutes": round(overall_elapsed / 60, 2),
            "total_bars_fetched": total_bars_fetched,
            "total_bars_stored": total_bars_stored,
            "total_errors": total_errors,
            "successful_timeframes": successful_timeframes,
            "timeframe_results": all_results,
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nBackfill interrupted by user. Progress has been saved.")
        logger.info("You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
