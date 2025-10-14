from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import logging
import os
import sys

#!/usr/bin/env python3
"""
Backfill 24 months of Ethereum (X:ETHUSD) data across all timeframes.

This script downloads historical data for the following timeframes:
- 1 minute bars
- 5 minute bars
- 15 minute bars
- 1 hour bars
- 4 hour bars
- 1 day bars

Usage:
    python backfill_ethereum_all_timeframes.py
"""


# Load environment variables from .env file
env_path = "/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env"
load_dotenv(env_path)
logger_init = logging.getLogger(__name__)
logger_init.info(f"Loaded environment from {env_path}")

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backfill_ethereum.log')
    ]
)
logger = logging.getLogger(__name__)


# Define all timeframes to backfill
TIMEFRAMES = [
    {"name": "1min", "multiplier": 1, "timespan": "minute"},
    {"name": "5min", "multiplier": 5, "timespan": "minute"},
    {"name": "15min", "multiplier": 15, "timespan": "minute"},
    {"name": "1hr", "multiplier": 1, "timespan": "hour"},
    {"name": "4hr", "multiplier": 4, "timespan": "hour"},
    {"name": "1day", "multiplier": 1, "timespan": "day"},
]

TICKER = "X:ETHUSD"
MONTHS = 24  # 2 years


async def backfill_single_timeframe(
    polygon_client: PolygonRESTClient,
    supabase_client: SupabaseClient,
    timeframe: dict,
    progress_dir: str
) -> dict:
    """
    Backfill data for a single timeframe.

    Args:
        polygon_client: Polygon API client
        supabase_client: Supabase database client
        timeframe: Timeframe configuration dict
        progress_dir: Directory for progress files

    Returns:
        Dict with statistics (bars_fetched, errors, time_taken)
    """
    timeframe_name = timeframe["name"]
    multiplier = timeframe["multiplier"]
    timespan = timeframe["timespan"]

    logger.info("=" * 80)
    logger.info(f"Starting backfill for {TICKER} - {timeframe_name}")
    logger.info("=" * 80)

    start_time = datetime.now()

    # Create timeframe-specific progress directory
    tf_progress_dir = os.path.join(progress_dir, timeframe_name)
    os.makedirs(tf_progress_dir, exist_ok=True)

    # Initialize backfill with timeframe-specific progress tracking
    backfill = HistoricalDataBackfill(
        polygon_client=polygon_client,
        supabase_client=supabase_client,
        tickers=[TICKER],
        progress_dir=tf_progress_dir,
    )

    try:
        # Run backfill
        total_bars = await backfill.backfill_aggregates(
            months=MONTHS,
            multiplier=multiplier,
            timespan=timespan,
        )

        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        stats = {
            "timeframe": timeframe_name,
            "bars_fetched": backfill.bars_fetched,
            "bars_stored": backfill.bars_stored,
            "errors": backfill.errors,
            "time_taken_seconds": time_taken,
            "success": True,
        }

        logger.info(f"\n✅ Completed {timeframe_name}:")
        logger.info(f"   Bars fetched: {stats['bars_fetched']:,}")
        logger.info(f"   Bars stored: {stats['bars_stored']:,}")
        logger.info(f"   Errors: {stats['errors']}")
        logger.info(f"   Time taken: {time_taken:.1f}s ({time_taken/60:.1f} minutes)")

        return stats

    except Exception as e:
        logger.error(f"❌ Error backfilling {timeframe_name}: {e}")
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        return {
            "timeframe": timeframe_name,
            "bars_fetched": 0,
            "bars_stored": 0,
            "errors": 1,
            "time_taken_seconds": time_taken,
            "success": False,
            "error_message": str(e),
        }


async def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("ETHEREUM HISTORICAL DATA BACKFILL - ALL TIMEFRAMES")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    for tf in TIMEFRAMES:
        logger.info(f"  - {tf['name']}: {tf['multiplier']} {tf['timespan']}")
    logger.info("=" * 80)
    logger.info("")

    overall_start = datetime.now()

    # Initialize clients
    logger.info("Initializing API clients...")
    try:
        polygon_client = PolygonRESTClient()
        supabase_client = SupabaseClient()
        logger.info("✅ Clients initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize clients: {e}")
        return 1

    # Create progress directory
    progress_dir = os.path.join(os.getcwd(), "backfill_progress")
    os.makedirs(progress_dir, exist_ok=True)
    logger.info(f"Progress files will be saved to: {progress_dir}")
    logger.info("")

    # Backfill each timeframe sequentially
    all_stats = []

    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing timeframe {i}/{len(TIMEFRAMES)}: {timeframe['name']}")
        logger.info(f"{'='*80}\n")

        stats = await backfill_single_timeframe(
            polygon_client=polygon_client,
            supabase_client=supabase_client,
            timeframe=timeframe,
            progress_dir=progress_dir,
        )

        all_stats.append(stats)

        # Brief pause between timeframes to be respectful of API
        if i < len(TIMEFRAMES):
            logger.info("\nPausing for 5 seconds before next timeframe...\n")
            await asyncio.sleep(5)

    # Calculate overall statistics
    overall_end = datetime.now()
    total_time = (overall_end - overall_start).total_seconds()

    total_bars_fetched = sum(s['bars_fetched'] for s in all_stats)
    total_bars_stored = sum(s['bars_stored'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    successful_timeframes = sum(1 for s in all_stats if s['success'])

    # Print final summary
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY - ETHEREUM BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nTicker: {TICKER}")
    logger.info(f"Period: {MONTHS} months")
    logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes, {total_time/3600:.1f} hours)")
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total bars fetched: {total_bars_fetched:,}")
    logger.info(f"  Total bars stored: {total_bars_stored:,}")
    logger.info(f"  Total errors: {total_errors}")
    logger.info(f"  Successful timeframes: {successful_timeframes}/{len(TIMEFRAMES)}")

    logger.info(f"\nPer-Timeframe Breakdown:")
    logger.info("-" * 80)
    logger.info(f"{'Timeframe':<12} {'Bars Fetched':>15} {'Bars Stored':>15} {'Errors':>8} {'Time (min)':>12} {'Status':>10}")
    logger.info("-" * 80)

    for stats in all_stats:
        status_icon = "✅" if stats['success'] else "❌"
        logger.info(
            f"{stats['timeframe']:<12} "
            f"{stats['bars_fetched']:>15,} "
            f"{stats['bars_stored']:>15,} "
            f"{stats['errors']:>8} "
            f"{stats['time_taken_seconds']/60:>12.1f} "
            f"{status_icon:>10}"
        )

    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<12} "
                f"{total_bars_fetched:>15,} "
                f"{total_bars_stored:>15,} "
                f"{total_errors:>8} "
                f"{total_time/60:>12.1f}")
    logger.info("=" * 80)

    # Log any errors
    if total_errors > 0:
        logger.warning(f"\n⚠️  {total_errors} errors encountered during backfill")
        for stats in all_stats:
            if not stats['success']:
                logger.error(f"  - {stats['timeframe']}: {stats.get('error_message', 'Unknown error')}")
    else:
        logger.info("\n✅ All timeframes completed successfully with no errors!")

    logger.info(f"\nProgress files saved to: {progress_dir}")
    logger.info("Log file: backfill_ethereum.log")
    logger.info("\n" + "=" * 80)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n\n❌ Backfill interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved. Run again to resume.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
