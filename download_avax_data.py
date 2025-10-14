from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
import os
import sys

#!/usr/bin/env python3
"""
Download 24 months of AVAX historical data across all timeframes.

This script downloads 2 years of historical cryptocurrency data for Avalanche (X:AVAXUSD)
across 6 different timeframes: 1min, 5min, 15min, 1hr, 4hr, and 1day.

Usage:
    python download_avax_data.py
"""


# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


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

TICKER = "X:AVAXUSD"
MONTHS = 24  # 2 years


async def download_timeframe(
    backfill: HistoricalDataBackfill,
    ticker: str,
    multiplier: int,
    timespan: str,
    timeframe_name: str,
    months: int,
) -> dict:
    """
    Download data for a single timeframe.

    Args:
        backfill: HistoricalDataBackfill instance
        ticker: Ticker symbol
        multiplier: Bar size multiplier
        timespan: Bar timespan (minute, hour, day)
        timeframe_name: Human-readable timeframe name
        months: Number of months to download

    Returns:
        Dictionary with download statistics
    """
    logger.info("=" * 80)
    logger.info(f"Starting download for {ticker} - {timeframe_name}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Reset progress for this specific ticker/timeframe combination
        # Create a unique progress file for each timeframe
        original_progress_path = backfill.progress_path
        backfill.progress_path = original_progress_path.parent / f"backfill_progress_{timeframe_name}.json"
        backfill.progress = backfill._load_progress()

        # Set tickers to only AVAX
        backfill.tickers = [ticker]

        # Run backfill
        bars_downloaded = await backfill.backfill_aggregates(
            months=months,
            multiplier=multiplier,
            timespan=timespan,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "bars_downloaded": bars_downloaded,
            "bars_stored": backfill.bars_stored,
            "errors": backfill.errors,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "success": True,
        }

        logger.info(f"\n{'=' * 80}")
        logger.info(f"COMPLETED: {ticker} - {timeframe_name}")
        logger.info(f"Bars downloaded: {bars_downloaded}")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"{'=' * 80}\n")

        # Restore original progress path
        backfill.progress_path = original_progress_path

        return stats

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.error(f"Error downloading {timeframe_name}: {e}", exc_info=True)

        stats = {
            "timeframe": timeframe_name,
            "ticker": ticker,
            "bars_downloaded": 0,
            "bars_stored": 0,
            "errors": 1,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "success": False,
            "error_message": str(e),
        }

        return stats


async def main():
    """Main execution function."""

    logger.info("\n" + "=" * 80)
    logger.info("AVALANCHE (X:AVAXUSD) HISTORICAL DATA DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    for tf in TIMEFRAMES:
        logger.info(f"  - {tf['name']}: {tf['multiplier']}{tf['timespan']}")
    logger.info("=" * 80)
    logger.info("")

    overall_start_time = datetime.now()

    try:
        # Initialize clients
        logger.info("Initializing Supabase client...")
        supabase_client = SupabaseClient()
        logger.info("Supabase client initialized successfully")

        logger.info("Initializing Polygon REST client...")
        polygon_client = PolygonRESTClient()
        logger.info("Polygon REST client initialized successfully")

        # Initialize backfill handler
        logger.info("Initializing Historical Data Backfill handler...")
        backfill = HistoricalDataBackfill(
            polygon_client=polygon_client,
            supabase_client=supabase_client,
            tickers=[TICKER],
        )
        logger.info("Backfill handler initialized successfully\n")

        # Download each timeframe sequentially
        all_stats = []

        for i, timeframe in enumerate(TIMEFRAMES, 1):
            logger.info(f"\n{'#' * 80}")
            logger.info(f"TIMEFRAME {i}/{len(TIMEFRAMES)}: {timeframe['name']}")
            logger.info(f"{'#' * 80}\n")

            stats = await download_timeframe(
                backfill=backfill,
                ticker=TICKER,
                multiplier=timeframe["multiplier"],
                timespan=timeframe["timespan"],
                timeframe_name=timeframe["name"],
                months=MONTHS,
            )

            all_stats.append(stats)

            # Small delay between timeframes
            if i < len(TIMEFRAMES):
                logger.info("Waiting 5 seconds before next timeframe...\n")
                await asyncio.sleep(5)

        overall_end_time = datetime.now()
        overall_duration = (overall_end_time - overall_start_time).total_seconds()

        # Final summary report
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"Ticker: {TICKER}")
        logger.info(f"Total duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes)")
        logger.info("")

        total_bars = sum(s["bars_downloaded"] for s in all_stats)
        total_stored = sum(s["bars_stored"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)
        successful_downloads = sum(1 for s in all_stats if s["success"])

        logger.info("OVERALL STATISTICS:")
        logger.info(f"  Total bars downloaded: {total_bars:,}")
        logger.info(f"  Total bars stored: {total_stored:,}")
        logger.info(f"  Total errors: {total_errors}")
        logger.info(f"  Successful timeframes: {successful_downloads}/{len(TIMEFRAMES)}")
        logger.info("")

        logger.info("BREAKDOWN BY TIMEFRAME:")
        logger.info("-" * 80)
        logger.info(f"{'Timeframe':<12} {'Bars':<15} {'Duration':<20} {'Status':<10}")
        logger.info("-" * 80)

        for stats in all_stats:
            status = "SUCCESS" if stats["success"] else "FAILED"
            duration_str = f"{stats['duration_seconds']:.1f}s ({stats['duration_seconds']/60:.1f}m)"
            logger.info(
                f"{stats['timeframe']:<12} {stats['bars_downloaded']:>14,} "
                f"{duration_str:<20} {status:<10}"
            )

        logger.info("-" * 80)
        logger.info("")

        # Error details if any
        failed_downloads = [s for s in all_stats if not s["success"]]
        if failed_downloads:
            logger.error("ERRORS ENCOUNTERED:")
            for stats in failed_downloads:
                logger.error(f"  {stats['timeframe']}: {stats.get('error_message', 'Unknown error')}")
            logger.error("")

        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("=" * 80)

        # Return summary
        return {
            "total_bars": total_bars,
            "total_stored": total_stored,
            "total_errors": total_errors,
            "successful_timeframes": successful_downloads,
            "total_timeframes": len(TIMEFRAMES),
            "overall_duration_seconds": overall_duration,
            "timeframe_stats": all_stats,
        }

    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user. Progress has been saved.")
        logger.info("You can resume by running this script again.")
        return None

    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        result = asyncio.run(main())

        if result:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
