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
Backfill 2 years of Polkadot (X:DOTUSD) data across all timeframes.

This script downloads historical cryptocurrency data for X:DOTUSD across:
- 1-minute bars
- 5-minute bars
- 15-minute bars
- 1-hour bars
- 4-hour bars
- 1-day bars

Usage:
    python scripts/backfill_dotusd_all_timeframes.py
"""


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


async def backfill_single_timeframe(
    backfill: HistoricalDataBackfill,
    timeframe: dict,
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """
    Backfill data for a single timeframe.

    Returns:
        dict with results including bars_count, errors, duration
    """
    tf_name = timeframe["name"]
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting backfill for {TICKER} - {tf_name}")
    logger.info(f"{'='*70}")

    start_time = datetime.now()

    try:
        # Reset stats for this timeframe
        backfill.bars_fetched = 0
        backfill.bars_stored = 0
        backfill.errors = 0

        # Run backfill
        bars = await backfill.backfill_ticker_aggregates(
            ticker=TICKER,
            start_date=start_date,
            end_date=end_date,
            multiplier=timeframe["multiplier"],
            timespan=timeframe["timespan"],
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            "timeframe": tf_name,
            "status": "success",
            "bars_fetched": backfill.bars_fetched,
            "bars_stored": backfill.bars_stored,
            "errors": backfill.errors,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"Completed {tf_name}:")
        logger.info(f"  Bars fetched: {result['bars_fetched']}")
        logger.info(f"  Bars stored: {result['bars_stored']}")
        logger.info(f"  Errors: {result['errors']}")
        logger.info(f"  Duration: {result['duration_minutes']:.2f} minutes")
        logger.info(f"{'='*70}\n")

        return result

    except Exception as e:
        logger.error(f"Error backfilling {tf_name}: {e}")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "timeframe": tf_name,
            "status": "failed",
            "bars_fetched": 0,
            "bars_stored": 0,
            "errors": 1,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "error_message": str(e),
        }


async def main():
    """Main function to backfill all timeframes."""

    logger.info("\n" + "="*70)
    logger.info("POLKADOT (X:DOTUSD) MULTI-TIMEFRAME BACKFILL")
    logger.info("="*70)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Time range: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {len(TIMEFRAMES)}")
    for tf in TIMEFRAMES:
        logger.info(f"  - {tf['name']}: {tf['multiplier']}{tf['timespan']}")
    logger.info("="*70 + "\n")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Initialize clients
    logger.info("\nInitializing Supabase and Polygon clients...")
    try:
        supabase = SupabaseClient()
        polygon = PolygonRESTClient()
        logger.info("Clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return

    # Initialize backfill with only X:DOTUSD
    backfill = HistoricalDataBackfill(
        polygon_client=polygon,
        supabase_client=supabase,
        tickers=[TICKER],
        progress_dir="/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/data",
    )

    # Track overall stats
    overall_start = datetime.now()
    results = []

    # Backfill each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n[{i}/{len(TIMEFRAMES)}] Processing {timeframe['name']}...")

        result = await backfill_single_timeframe(
            backfill=backfill,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        results.append(result)

        # Small delay between timeframes to respect rate limits
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
    logger.info("\nResults by timeframe:")
    logger.info("-" * 70)

    total_bars_fetched = 0
    total_bars_stored = 0
    total_errors = 0

    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"\n{status_icon} {result['timeframe']}:")
        logger.info(f"    Status: {result['status']}")
        logger.info(f"    Bars fetched: {result['bars_fetched']:,}")
        logger.info(f"    Bars stored: {result['bars_stored']:,}")
        logger.info(f"    Errors: {result['errors']}")
        logger.info(f"    Duration: {result['duration_minutes']:.2f} minutes")

        if result.get("error_message"):
            logger.info(f"    Error: {result['error_message']}")

        total_bars_fetched += result['bars_fetched']
        total_bars_stored += result['bars_stored']
        total_errors += result['errors']

    logger.info("\n" + "-" * 70)
    logger.info("TOTALS:")
    logger.info(f"  Total bars fetched: {total_bars_fetched:,}")
    logger.info(f"  Total bars stored: {total_bars_stored:,}")
    logger.info(f"  Total errors: {total_errors}")
    logger.info(f"  Success rate: {(len([r for r in results if r['status'] == 'success']) / len(results) * 100):.1f}%")
    logger.info("="*70 + "\n")

    # Save summary to file
    summary_file = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/data/backfill_summary.txt")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        f.write(f"Polkadot (X:DOTUSD) Backfill Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Ticker: {TICKER}\n")
        f.write(f"Date range: {start_date.date()} to {end_date.date()}\n")
        f.write(f"Duration: {overall_duration / 60:.2f} minutes\n\n")

        for result in results:
            f.write(f"\n{result['timeframe']}:\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Bars fetched: {result['bars_fetched']:,}\n")
            f.write(f"  Bars stored: {result['bars_stored']:,}\n")
            f.write(f"  Errors: {result['errors']}\n")
            f.write(f"  Duration: {result['duration_minutes']:.2f} min\n")

        f.write(f"\nTOTALS:\n")
        f.write(f"  Total bars fetched: {total_bars_fetched:,}\n")
        f.write(f"  Total bars stored: {total_bars_stored:,}\n")
        f.write(f"  Total errors: {total_errors}\n")

    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nBackfill interrupted by user. Progress has been saved.")
        logger.info("Run again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
