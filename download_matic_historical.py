from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
import sys
import time

#!/usr/bin/env python3
"""
Download 2 years of historical data for Polygon (X:MATICUSD)
across all timeframes: 1min, 5min, 15min, 1hr, 4hr, 1day
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Timeframes configuration
TIMEFRAMES = [
    {"multiplier": 1, "timespan": "minute", "label": "1min"},
    {"multiplier": 5, "timespan": "minute", "label": "5min"},
    {"multiplier": 15, "timespan": "minute", "label": "15min"},
    {"multiplier": 1, "timespan": "hour", "label": "1hr"},
    {"multiplier": 4, "timespan": "hour", "label": "4hr"},
    {"multiplier": 1, "timespan": "day", "label": "1day"},
]

TICKER = "X:MATICUSD"
MONTHS = 24  # 2 years


async def download_timeframe(
    backfill: HistoricalDataBackfill,
    ticker: str,
    multiplier: int,
    timespan: str,
    label: str,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """
    Download data for a single timeframe.

    Returns:
        Dictionary with results
    """
    logger.info("=" * 80)
    logger.info(f"Starting download for {ticker} - {label}")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        bars = await backfill.backfill_ticker_aggregates(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            multiplier=multiplier,
            timespan=timespan,
        )

        elapsed = time.time() - start_time

        result = {
            "timeframe": label,
            "bars": bars,
            "elapsed_seconds": round(elapsed, 2),
            "success": True,
            "error": None
        }

        logger.info(f"Completed {label}: {bars} bars in {elapsed:.2f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed to download {label}: {e}")
        return {
            "timeframe": label,
            "bars": 0,
            "elapsed_seconds": round(elapsed, 2),
            "success": False,
            "error": str(e)
        }


async def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("POLYGON (MATIC) HISTORICAL DATA DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Duration: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {', '.join([tf['label'] for tf in TIMEFRAMES])}")
    logger.info("=" * 80)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 80)

    # Initialize clients
    try:
        logger.info("Initializing Supabase client...")
        supabase = SupabaseClient()
        logger.info("Supabase client initialized")

        logger.info("Initializing Polygon client...")
        polygon = PolygonRESTClient()
        logger.info("Polygon client initialized")

    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return 1

    # Results tracking
    results = []
    total_bars = 0
    total_errors = 0
    overall_start = time.time()

    # Download each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n[{i}/{len(TIMEFRAMES)}] Processing {timeframe['label']}...")

        # Create a new backfill instance for each timeframe
        # This ensures progress tracking is separate
        backfill = HistoricalDataBackfill(
            polygon_client=polygon,
            supabase_client=supabase,
            tickers=[TICKER],
            progress_dir=f"/tmp/backfill_{timeframe['label']}"
        )

        result = await download_timeframe(
            backfill=backfill,
            ticker=TICKER,
            multiplier=timeframe["multiplier"],
            timespan=timeframe["timespan"],
            label=timeframe["label"],
            start_date=start_date,
            end_date=end_date
        )

        results.append(result)
        total_bars += result["bars"]

        if not result["success"]:
            total_errors += 1

        # Small delay between timeframes to respect rate limits
        if i < len(TIMEFRAMES):
            logger.info("Pausing 5 seconds before next timeframe...")
            await asyncio.sleep(5)

    overall_elapsed = time.time() - overall_start

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Total Duration: {overall_elapsed / 60:.2f} minutes")
    logger.info(f"Total Bars Downloaded: {total_bars:,}")
    logger.info(f"Total Errors: {total_errors}")
    logger.info("=" * 80)
    logger.info("\nBreakdown by Timeframe:")
    logger.info("-" * 80)

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        logger.info(
            f"  {result['timeframe']:>6} | {status:>7} | "
            f"{result['bars']:>8,} bars | "
            f"{result['elapsed_seconds']:>6.2f}s"
        )
        if result["error"]:
            logger.info(f"         Error: {result['error']}")

    logger.info("=" * 80)

    # Calculate statistics
    if results:
        avg_time = sum(r["elapsed_seconds"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100

        logger.info("\nStatistics:")
        logger.info(f"  Average time per timeframe: {avg_time:.2f}s")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Total bars/second: {total_bars / overall_elapsed:.2f}")

    logger.info("=" * 80)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
