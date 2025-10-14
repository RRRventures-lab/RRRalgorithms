from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime
from pathlib import Path
import asyncio
import logging
import os
import sys
import time

#!/usr/bin/env python3
"""
Backfill 24 months of Uniswap (X:UNIUSD) data across all timeframes
===================================================================

This script downloads historical data for X:UNIUSD from Polygon.io
and stores it in Supabase for the following timeframes:
- 1 minute
- 5 minutes
- 15 minutes
- 1 hour
- 4 hours
- 1 day

Usage:
    cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
    python backfill_uniswap.py
"""


# Add project root to path
project_root = "/Volumes/Lexar/RRRVentures/RRRalgorithms"
sys.path.insert(0, str(Path(project_root) / "worktrees" / "data-pipeline" / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backfill_uniswap.log')
    ]
)
logger = logging.getLogger(__name__)


# Define timeframes to backfill
TIMEFRAMES = [
    {"name": "1min", "multiplier": 1, "timespan": "minute"},
    {"name": "5min", "multiplier": 5, "timespan": "minute"},
    {"name": "15min", "multiplier": 15, "timespan": "minute"},
    {"name": "1hr", "multiplier": 1, "timespan": "hour"},
    {"name": "4hr", "multiplier": 4, "timespan": "hour"},
    {"name": "1day", "multiplier": 1, "timespan": "day"},
]

TICKER = "X:UNIUSD"
MONTHS = 24


async def backfill_timeframe(
    timeframe: dict,
    supabase_client: SupabaseClient,
    polygon_client: PolygonRESTClient,
) -> dict:
    """
    Backfill data for a single timeframe.

    Args:
        timeframe: Dictionary with multiplier and timespan
        supabase_client: Supabase client instance
        polygon_client: Polygon REST client instance

    Returns:
        Dictionary with statistics for this timeframe
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info(f"Starting backfill for {TICKER} - {timeframe['name']}")
    logger.info("=" * 80)

    # Create progress directory for this timeframe
    progress_dir = Path.cwd() / "progress" / timeframe['name']
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backfill
    backfill = HistoricalDataBackfill(
        polygon_client=polygon_client,
        supabase_client=supabase_client,
        tickers=[TICKER],
        progress_dir=str(progress_dir),
    )

    try:
        # Run backfill
        total_bars = await backfill.backfill_aggregates(
            months=MONTHS,
            multiplier=timeframe["multiplier"],
            timespan=timeframe["timespan"],
        )

        elapsed_time = time.time() - start_time
        stats = backfill.get_stats()

        result = {
            "timeframe": timeframe['name'],
            "status": "success",
            "total_bars": total_bars,
            "bars_fetched": stats['bars_fetched'],
            "bars_stored": stats['bars_stored'],
            "errors": stats['errors'],
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_minutes": round(elapsed_time / 60, 2),
        }

        logger.info(f"\n{'=' * 80}")
        logger.info(f"COMPLETED: {timeframe['name']}")
        logger.info(f"Total bars: {total_bars}")
        logger.info(f"Time taken: {result['elapsed_minutes']:.2f} minutes")
        logger.info(f"{'=' * 80}\n")

        return result

    except Exception as e:
        logger.error(f"Error backfilling {timeframe['name']}: {e}", exc_info=True)
        elapsed_time = time.time() - start_time

        return {
            "timeframe": timeframe['name'],
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": round(elapsed_time, 2),
        }


async def main():
    """Main function to orchestrate backfill across all timeframes."""
    overall_start = time.time()

    logger.info("\n" + "=" * 80)
    logger.info("UNISWAP (X:UNIUSD) HISTORICAL DATA BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    # Initialize clients
    try:
        logger.info("Initializing Supabase client...")
        supabase_client = SupabaseClient()
        logger.info("✓ Supabase client initialized")

        logger.info("Initializing Polygon.io client...")
        polygon_client = PolygonRESTClient()
        logger.info("✓ Polygon.io client initialized")

    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return

    # Track results
    results = []

    # Process each timeframe sequentially to avoid rate limiting
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n[{i}/{len(TIMEFRAMES)}] Processing {timeframe['name']}...")

        result = await backfill_timeframe(
            timeframe=timeframe,
            supabase_client=supabase_client,
            polygon_client=polygon_client,
        )

        results.append(result)

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("Pausing 5 seconds before next timeframe...")
            await asyncio.sleep(5)

    # Calculate overall statistics
    overall_elapsed = time.time() - overall_start
    total_bars = sum(r.get('total_bars', 0) for r in results)
    total_errors = sum(r.get('errors', 0) for r in results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months")
    logger.info(f"Total timeframes: {len(TIMEFRAMES)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total bars downloaded: {total_bars:,}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total time: {overall_elapsed / 60:.2f} minutes ({overall_elapsed / 3600:.2f} hours)")
    logger.info("=" * 80)

    # Print detailed results per timeframe
    logger.info("\nDETAILED RESULTS BY TIMEFRAME:")
    logger.info("-" * 80)

    for result in results:
        logger.info(f"\n{result['timeframe']}:")
        logger.info(f"  Status: {result['status']}")

        if result['status'] == 'success':
            logger.info(f"  Total bars: {result['total_bars']:,}")
            logger.info(f"  Bars fetched: {result['bars_fetched']:,}")
            logger.info(f"  Bars stored: {result['bars_stored']:,}")
            logger.info(f"  Errors: {result['errors']}")
            logger.info(f"  Time: {result['elapsed_minutes']:.2f} minutes")
        else:
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
            logger.info(f"  Time: {result.get('elapsed_seconds', 0):.2f} seconds")

    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    # Log to Supabase
    try:
        supabase_client.log_system_event(
            event_type="backfill_complete",
            severity="info",
            message=f"Completed {MONTHS}-month backfill for {TICKER} across {len(TIMEFRAMES)} timeframes",
            component="historical_backfill",
            metadata={
                "ticker": TICKER,
                "months": MONTHS,
                "total_bars": total_bars,
                "total_errors": total_errors,
                "timeframes": [r['timeframe'] for r in results if r['status'] == 'success'],
                "elapsed_hours": round(overall_elapsed / 3600, 2),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log to Supabase: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nBackfill interrupted by user. Progress has been saved.")
        logger.info("Run the script again to resume from where you left off.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
