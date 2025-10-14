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
Download 2 years (24 months) of historical Solana (X:SOLUSD) data
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


async def main():
    """Download Solana data for all timeframes."""

    # Configuration
    TICKER = "X:SOLUSD"
    MONTHS = 24  # 2 years

    # Define all timeframes
    TIMEFRAMES = [
        {"name": "1min", "multiplier": 1, "timespan": "minute"},
        {"name": "5min", "multiplier": 5, "timespan": "minute"},
        {"name": "15min", "multiplier": 15, "timespan": "minute"},
        {"name": "1hr", "multiplier": 1, "timespan": "hour"},
        {"name": "4hr", "multiplier": 4, "timespan": "hour"},
        {"name": "1day", "multiplier": 1, "timespan": "day"},
    ]

    logger.info("=" * 80)
    logger.info("SOLANA (X:SOLUSD) HISTORICAL DATA BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {', '.join([tf['name'] for tf in TIMEFRAMES])}")
    logger.info("=" * 80)

    # Initialize clients
    logger.info("\nInitializing clients...")
    try:
        supabase = SupabaseClient()
        polygon = PolygonRESTClient()
        logger.info("✅ Clients initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize clients: {e}")
        return 1

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    logger.info(f"\nDate range: {start_date.date()} to {end_date.date()}")

    # Track overall statistics
    overall_stats = {
        "total_bars": 0,
        "total_errors": 0,
        "start_time": datetime.now(),
        "timeframes": {}
    }

    # Process each timeframe
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        tf_name = timeframe["name"]
        multiplier = timeframe["multiplier"]
        timespan = timeframe["timespan"]

        logger.info("\n" + "=" * 80)
        logger.info(f"TIMEFRAME {i}/{len(TIMEFRAMES)}: {tf_name}")
        logger.info("=" * 80)

        # Create dedicated backfill instance for this timeframe
        # Use separate progress files for each timeframe
        progress_dir = Path(__file__).parent / "backfill_progress"
        progress_dir.mkdir(exist_ok=True)

        # Custom progress file per timeframe
        custom_progress_file = f"backfill_progress_{TICKER.replace(':', '_')}_{tf_name}.json"

        backfill = HistoricalDataBackfill(
            polygon_client=polygon,
            supabase_client=supabase,
            tickers=[TICKER],
            progress_dir=str(progress_dir)
        )

        # Override progress file name
        backfill.PROGRESS_FILE = custom_progress_file
        backfill.progress_path = progress_dir / custom_progress_file
        backfill.progress = backfill._load_progress()

        try:
            tf_start_time = datetime.now()

            # Run backfill for this timeframe
            logger.info(f"Starting backfill for {tf_name}...")
            bars_count = await backfill.backfill_ticker_aggregates(
                ticker=TICKER,
                start_date=start_date,
                end_date=end_date,
                multiplier=multiplier,
                timespan=timespan,
            )

            tf_end_time = datetime.now()
            tf_duration = (tf_end_time - tf_start_time).total_seconds()

            # Record stats
            overall_stats["timeframes"][tf_name] = {
                "bars": bars_count,
                "errors": backfill.errors,
                "duration_seconds": tf_duration,
                "bars_per_second": bars_count / tf_duration if tf_duration > 0 else 0
            }
            overall_stats["total_bars"] += bars_count
            overall_stats["total_errors"] += backfill.errors

            logger.info(f"✅ Completed {tf_name}: {bars_count} bars in {tf_duration:.1f}s")

        except Exception as e:
            logger.error(f"❌ Failed to backfill {tf_name}: {e}")
            overall_stats["timeframes"][tf_name] = {
                "bars": 0,
                "errors": 1,
                "error_message": str(e)
            }
            overall_stats["total_errors"] += 1
            continue

    # Calculate final statistics
    overall_stats["end_time"] = datetime.now()
    overall_stats["total_duration_seconds"] = (
        overall_stats["end_time"] - overall_stats["start_time"]
    ).total_seconds()
    overall_stats["total_duration_minutes"] = overall_stats["total_duration_seconds"] / 60

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total bars downloaded: {overall_stats['total_bars']:,}")
    logger.info(f"Total errors: {overall_stats['total_errors']}")
    logger.info(f"Total duration: {overall_stats['total_duration_minutes']:.1f} minutes")
    logger.info("")
    logger.info("Breakdown by timeframe:")
    logger.info("-" * 80)

    for tf_name, stats in overall_stats["timeframes"].items():
        if "error_message" in stats:
            logger.info(f"  {tf_name:8s}: ❌ ERROR - {stats['error_message']}")
        else:
            logger.info(
                f"  {tf_name:8s}: {stats['bars']:,} bars | "
                f"{stats['duration_seconds']:.1f}s | "
                f"{stats['bars_per_second']:.1f} bars/s"
            )

    logger.info("=" * 80)

    # Log to Supabase
    try:
        supabase.log_system_event(
            event_type="backfill_complete_all_timeframes",
            severity="info",
            message=f"Completed 24-month backfill for {TICKER} across all timeframes",
            component="historical_backfill",
            metadata={
                "ticker": TICKER,
                "months": MONTHS,
                "total_bars": overall_stats["total_bars"],
                "total_errors": overall_stats["total_errors"],
                "duration_minutes": overall_stats["total_duration_minutes"],
                "timeframes": overall_stats["timeframes"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
        )
    except Exception as e:
        logger.warning(f"Could not log to Supabase: {e}")

    return 0 if overall_stats["total_errors"] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Backfill interrupted by user.")
        logger.info("Progress has been saved. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
