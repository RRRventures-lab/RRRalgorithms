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
Backfill Bitcoin (X:BTCUSD) Historical Data - All Timeframes
============================================================

Downloads 2 years (24 months) of historical data for Bitcoin across all timeframes:
- 1 minute bars
- 5 minute bars
- 15 minute bars
- 1 hour bars
- 4 hour bars
- 1 day bars

This script will:
1. Initialize Supabase and Polygon clients
2. Run backfill for each timeframe sequentially
3. Track progress and handle errors
4. Report statistics for each timeframe
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / "config" / "api-keys" / ".env"
load_dotenv(env_path)

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

TICKER = "X:BTCUSD"
MONTHS = 24  # 2 years


async def backfill_timeframe(
    ticker: str,
    multiplier: int,
    timespan: str,
    name: str,
    months: int,
    supabase: SupabaseClient,
    polygon: PolygonRESTClient,
):
    """Backfill a single timeframe."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting backfill for {ticker} - {name}")
    logger.info(f"{'='*80}")

    start_time = datetime.now()

    # Create backfill instance for this timeframe
    backfill = HistoricalDataBackfill(
        polygon_client=polygon,
        supabase_client=supabase,
        tickers=[ticker],
        progress_dir=f"./progress/{name}",  # Separate progress for each timeframe
    )

    try:
        # Run backfill
        total_bars = await backfill.backfill_aggregates(
            months=months,
            multiplier=multiplier,
            timespan=timespan,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # Get stats
        stats = backfill.get_stats()

        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED: {ticker} - {name}")
        logger.info(f"{'='*80}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Total bars: {total_bars}")
        logger.info(f"Bars fetched: {stats['bars_fetched']}")
        logger.info(f"Bars stored: {stats['bars_stored']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"{'='*80}\n")

        return {
            "name": name,
            "ticker": ticker,
            "multiplier": multiplier,
            "timespan": timespan,
            "bars_fetched": stats['bars_fetched'],
            "bars_stored": stats['bars_stored'],
            "errors": stats['errors'],
            "duration": str(duration),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error backfilling {name}: {e}")
        end_time = datetime.now()
        duration = end_time - start_time

        return {
            "name": name,
            "ticker": ticker,
            "multiplier": multiplier,
            "timespan": timespan,
            "bars_fetched": 0,
            "bars_stored": 0,
            "errors": 1,
            "duration": str(duration),
            "success": False,
            "error": str(e),
        }


async def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("BITCOIN HISTORICAL DATA BACKFILL - ALL TIMEFRAMES")
    logger.info("="*80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*80)

    overall_start = datetime.now()

    # Initialize clients
    logger.info("\nInitializing clients...")
    supabase = SupabaseClient()
    polygon = PolygonRESTClient()
    logger.info("Clients initialized successfully")

    # Results tracking
    results = []
    total_bars = 0
    total_errors = 0

    # Backfill each timeframe
    for i, tf in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"TIMEFRAME {i}/{len(TIMEFRAMES)}: {tf['name']}")
        logger.info(f"{'#'*80}")

        result = await backfill_timeframe(
            ticker=TICKER,
            multiplier=tf["multiplier"],
            timespan=tf["timespan"],
            name=tf["name"],
            months=MONTHS,
            supabase=supabase,
            polygon=polygon,
        )

        results.append(result)
        total_bars += result["bars_stored"]
        total_errors += result["errors"]

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("\nWaiting 5 seconds before next timeframe...")
            await asyncio.sleep(5)

    overall_end = datetime.now()
    overall_duration = overall_end - overall_start

    # Final summary
    logger.info("\n\n" + "="*80)
    logger.info("FINAL SUMMARY - ALL TIMEFRAMES")
    logger.info("="*80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months")
    logger.info(f"Total duration: {overall_duration}")
    logger.info(f"Total bars downloaded: {total_bars:,}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("="*80)

    logger.info("\nDetailed Results:")
    logger.info("-"*80)
    for result in results:
        status = "✓" if result["success"] else "✗"
        logger.info(
            f"{status} {result['name']:8} | "
            f"Bars: {result['bars_stored']:>10,} | "
            f"Duration: {result['duration']:>12} | "
            f"Errors: {result['errors']}"
        )
        if not result["success"]:
            logger.info(f"  Error: {result.get('error', 'Unknown')}")

    logger.info("-"*80)
    logger.info(f"TOTAL: {total_bars:,} bars downloaded")
    logger.info(f"Overall duration: {overall_duration}")
    logger.info("="*80)

    # Log to Supabase system events
    try:
        supabase.log_system_event(
            event_type="backfill_complete_all_timeframes",
            severity="info",
            message=f"Completed backfill for {TICKER} across all timeframes",
            component="historical_backfill",
            metadata={
                "ticker": TICKER,
                "months": MONTHS,
                "total_bars": total_bars,
                "total_errors": total_errors,
                "duration": str(overall_duration),
                "timeframes": results,
            }
        )
    except Exception as e:
        logger.warning(f"Could not log to Supabase events: {e}")

    logger.info("\n✓ All timeframes completed!")

    # Return stats
    return {
        "ticker": TICKER,
        "months": MONTHS,
        "total_bars": total_bars,
        "total_errors": total_errors,
        "duration": str(overall_duration),
        "results": results,
    }


if __name__ == "__main__":
    try:
        stats = asyncio.run(main())
        sys.exit(0 if stats["total_errors"] == 0 else 1)
    except KeyboardInterrupt:
        logger.info("\n\nBackfill interrupted by user")
        logger.info("Progress has been saved. Run again to resume.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
