from data_pipeline.backfill.historical import HistoricalDataBackfill
from data_pipeline.polygon.rest_client import PolygonRESTClient
from data_pipeline.supabase_client import SupabaseClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any
import asyncio
import logging
import os
import sys

#!/usr/bin/env python3
"""
Backfill 2 years of Cosmos (X:ATOMUSD) data across all timeframes.

This script downloads historical data for:
- 1 minute bars
- 5 minute bars
- 15 minute bars
- 1 hour bars
- 4 hour bars
- 1 day bars

Usage:
    python run_atom_backfill.py
"""


# Load environment variables from config/api-keys/.env
project_root = os.getenv("PROJECT_ROOT", "/Volumes/Lexar/RRRVentures/RRRalgorithms")
env_file = os.path.join(project_root, "config/api-keys/.env")
load_dotenv(env_file)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('atom_backfill.log')
    ]
)
logger = logging.getLogger(__name__)


class MultiTimeframeBackfill:
    """Run backfill across multiple timeframes for a single ticker."""

    TIMEFRAMES = [
        {"multiplier": 1, "timespan": "minute", "name": "1min"},
        {"multiplier": 5, "timespan": "minute", "name": "5min"},
        {"multiplier": 15, "timespan": "minute", "name": "15min"},
        {"multiplier": 1, "timespan": "hour", "name": "1hr"},
        {"multiplier": 4, "timespan": "hour", "name": "4hr"},
        {"multiplier": 1, "timespan": "day", "name": "1day"},
    ]

    def __init__(self, ticker: str = "X:ATOMUSD", months: int = 24):
        """
        Initialize multi-timeframe backfill.

        Args:
            ticker: Ticker symbol (default: X:ATOMUSD)
            months: Number of months to backfill (default: 24)
        """
        self.ticker = ticker
        self.months = months

        # Initialize clients
        logger.info("Initializing Polygon and Supabase clients...")
        self.polygon_client = PolygonRESTClient()
        self.supabase_client = SupabaseClient()

        # Statistics
        self.results: Dict[str, Dict[str, Any]] = {}
        self.total_bars = 0
        self.total_errors = 0
        self.start_time = None
        self.end_time = None

    async def run(self):
        """Run backfill for all timeframes."""
        self.start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("MULTI-TIMEFRAME BACKFILL FOR COSMOS (X:ATOMUSD)")
        logger.info("=" * 80)
        logger.info(f"Ticker: {self.ticker}")
        logger.info(f"Period: {self.months} months ({self.months * 30} days)")
        logger.info(f"Timeframes: {len(self.TIMEFRAMES)}")
        logger.info(f"Start time: {self.start_time}")
        logger.info("=" * 80)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months * 30)

        logger.info(f"\nDate range: {start_date.date()} to {end_date.date()}")

        # Process each timeframe
        for i, timeframe in enumerate(self.TIMEFRAMES, 1):
            tf_name = timeframe["name"]
            multiplier = timeframe["multiplier"]
            timespan = timeframe["timespan"]

            logger.info("\n" + "=" * 80)
            logger.info(f"TIMEFRAME {i}/{len(self.TIMEFRAMES)}: {tf_name}")
            logger.info("=" * 80)

            tf_start = datetime.now()

            try:
                # Create backfill instance for this timeframe
                backfill = HistoricalDataBackfill(
                    polygon_client=self.polygon_client,
                    supabase_client=self.supabase_client,
                    tickers=[self.ticker],
                    progress_dir=f"./progress_{tf_name}"
                )

                # Run backfill
                bars = await backfill.backfill_ticker_aggregates(
                    ticker=self.ticker,
                    start_date=start_date,
                    end_date=end_date,
                    multiplier=multiplier,
                    timespan=timespan,
                )

                tf_end = datetime.now()
                tf_duration = (tf_end - tf_start).total_seconds()

                # Store results
                self.results[tf_name] = {
                    "bars": bars,
                    "errors": backfill.errors,
                    "duration_seconds": tf_duration,
                    "bars_per_second": bars / tf_duration if tf_duration > 0 else 0,
                    "status": "completed"
                }

                self.total_bars += bars
                self.total_errors += backfill.errors

                logger.info(f"\n✅ {tf_name} completed:")
                logger.info(f"   Bars downloaded: {bars:,}")
                logger.info(f"   Errors: {backfill.errors}")
                logger.info(f"   Duration: {tf_duration:.1f}s")
                logger.info(f"   Speed: {bars/tf_duration:.1f} bars/sec")

            except Exception as e:
                logger.error(f"\n❌ {tf_name} failed: {e}")
                self.results[tf_name] = {
                    "bars": 0,
                    "errors": 1,
                    "duration_seconds": 0,
                    "bars_per_second": 0,
                    "status": "failed",
                    "error": str(e)
                }
                self.total_errors += 1

        self.end_time = datetime.now()
        self._print_summary()

    def _print_summary(self):
        """Print final summary report."""
        total_duration = (self.end_time - self.start_time).total_seconds()

        logger.info("\n\n" + "=" * 80)
        logger.info("BACKFILL COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Ticker: {self.ticker}")
        logger.info(f"Period: {self.months} months")
        logger.info(f"Total duration: {total_duration/60:.1f} minutes ({total_duration:.0f}s)")
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"End time: {self.end_time}")
        logger.info("\n" + "-" * 80)
        logger.info("TIMEFRAME BREAKDOWN:")
        logger.info("-" * 80)

        for tf_name, result in self.results.items():
            status_emoji = "✅" if result["status"] == "completed" else "❌"
            logger.info(f"\n{status_emoji} {tf_name}:")
            logger.info(f"   Bars downloaded: {result['bars']:,}")
            logger.info(f"   Errors: {result['errors']}")
            logger.info(f"   Duration: {result['duration_seconds']:.1f}s")
            logger.info(f"   Speed: {result['bars_per_second']:.1f} bars/sec")

            if result["status"] == "failed":
                logger.info(f"   Error: {result.get('error', 'Unknown error')}")

        logger.info("\n" + "-" * 80)
        logger.info("TOTALS:")
        logger.info("-" * 80)
        logger.info(f"Total bars downloaded: {self.total_bars:,}")
        logger.info(f"Total errors: {self.total_errors}")
        logger.info(f"Average speed: {self.total_bars/total_duration:.1f} bars/sec")
        logger.info(f"Success rate: {len([r for r in self.results.values() if r['status'] == 'completed'])}/{len(self.TIMEFRAMES)} timeframes")
        logger.info("=" * 80)

        # Log to Supabase
        try:
            self.supabase_client.log_system_event(
                event_type="multi_timeframe_backfill_complete",
                severity="info",
                message=f"Completed backfill for {self.ticker} across {len(self.TIMEFRAMES)} timeframes",
                component="atom_backfill",
                metadata={
                    "ticker": self.ticker,
                    "months": self.months,
                    "total_bars": self.total_bars,
                    "total_errors": self.total_errors,
                    "duration_seconds": total_duration,
                    "timeframes": self.results
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log to Supabase: {e}")


async def main():
    """Main entry point."""
    backfill = MultiTimeframeBackfill(ticker="X:ATOMUSD", months=24)

    try:
        await backfill.run()

        # Return exit code based on errors
        if backfill.total_errors > 0:
            logger.warning(f"\n⚠️  Completed with {backfill.total_errors} errors")
            sys.exit(1)
        else:
            logger.info("\n✅ All timeframes completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Backfill interrupted by user. Progress has been saved.")
        logger.info("Run again to resume from where you left off.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
