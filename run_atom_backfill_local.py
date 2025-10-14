from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import csv
import logging
import os
import sys

#!/usr/bin/env python3
"""
Backfill 2 years of Cosmos (X:ATOMUSD) data across all timeframes.
Saves data locally as CSV files (bypassing Supabase due to API key issues).

This script downloads historical data for:
- 1 minute bars
- 5 minute bars
- 15 minute bars
- 1 hour bars
- 4 hour bars
- 1 day bars

Usage:
    python run_atom_backfill_local.py
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
        logging.FileHandler('atom_backfill_local.log')
    ]
)
logger = logging.getLogger(__name__)


class LocalCSVBackfill:
    """Run backfill across multiple timeframes for a single ticker, save to CSV."""

    TIMEFRAMES = [
        {"multiplier": 1, "timespan": "minute", "name": "1min"},
        {"multiplier": 5, "timespan": "minute", "name": "5min"},
        {"multiplier": 15, "timespan": "minute", "name": "15min"},
        {"multiplier": 1, "timespan": "hour", "name": "1hr"},
        {"multiplier": 4, "timespan": "hour", "name": "4hr"},
        {"multiplier": 1, "timespan": "day", "name": "1day"},
    ]

    def __init__(self, ticker: str = "X:ATOMUSD", months: int = 24, output_dir: str = "./data"):
        """
        Initialize multi-timeframe backfill.

        Args:
            ticker: Ticker symbol (default: X:ATOMUSD)
            months: Number of months to backfill (default: 24)
            output_dir: Directory to save CSV files
        """
        self.ticker = ticker
        self.months = months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Polygon client
        logger.info("Initializing Polygon client...")
        self.polygon_client = PolygonRESTClient()

        # Statistics
        self.results: Dict[str, Dict[str, Any]] = {}
        self.total_bars = 0
        self.total_errors = 0
        self.start_time = None
        self.end_time = None

    def save_to_csv(self, aggregates: List, timeframe: str):
        """Save aggregates to CSV file."""
        if not aggregates:
            return 0

        filename = self.output_dir / f"{self.ticker.replace(':', '_')}_{timeframe}.csv"

        # Check if file exists to determine if we need headers
        file_exists = filename.exists()

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for agg in aggregates:
                writer.writerow({
                    'ticker': agg.ticker,
                    'datetime': agg.datetime.isoformat() if hasattr(agg.datetime, 'isoformat') else str(agg.datetime),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap if hasattr(agg, 'vwap') else None,
                    'trade_count': agg.trade_count if hasattr(agg, 'trade_count') else None,
                })

        return len(aggregates)

    async def backfill_timeframe(
        self,
        start_date: datetime,
        end_date: datetime,
        multiplier: int,
        timespan: str,
        timeframe_name: str
    ) -> int:
        """Backfill single timeframe."""
        logger.info(f"Backfilling {self.ticker} {timeframe_name} from {start_date.date()} to {end_date.date()}")

        total_bars = 0
        errors = 0

        # Split into 30-day chunks
        chunk_days = 30
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

            logger.info(f"  Fetching {timeframe_name}: {current_date.date()} to {chunk_end.date()}")

            try:
                aggregates = self.polygon_client.get_aggregates(
                    ticker=self.ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=current_date.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                    limit=50000,
                )

                if aggregates:
                    saved = self.save_to_csv(aggregates, timeframe_name)
                    total_bars += saved
                    logger.info(f"  Saved {saved} bars to CSV")
                else:
                    logger.warning(f"  No data returned for this period")

            except Exception as e:
                logger.error(f"  Error fetching chunk: {e}")
                errors += 1

            # Move to next chunk
            current_date = chunk_end + timedelta(days=1)

            # Small delay to respect rate limits
            await asyncio.sleep(0.5)

        return total_bars, errors

    async def run(self):
        """Run backfill for all timeframes."""
        self.start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("MULTI-TIMEFRAME BACKFILL FOR COSMOS (X:ATOMUSD)")
        logger.info("=" * 80)
        logger.info(f"Ticker: {self.ticker}")
        logger.info(f"Period: {self.months} months ({self.months * 30} days)")
        logger.info(f"Timeframes: {len(self.TIMEFRAMES)}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
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
                bars, errors = await self.backfill_timeframe(
                    start_date=start_date,
                    end_date=end_date,
                    multiplier=multiplier,
                    timespan=timespan,
                    timeframe_name=tf_name
                )

                tf_end = datetime.now()
                tf_duration = (tf_end - tf_start).total_seconds()

                # Store results
                self.results[tf_name] = {
                    "bars": bars,
                    "errors": errors,
                    "duration_seconds": tf_duration,
                    "bars_per_second": bars / tf_duration if tf_duration > 0 else 0,
                    "status": "completed",
                    "csv_file": f"{self.ticker.replace(':', '_')}_{tf_name}.csv"
                }

                self.total_bars += bars
                self.total_errors += errors

                logger.info(f"\n✅ {tf_name} completed:")
                logger.info(f"   Bars downloaded: {bars:,}")
                logger.info(f"   Errors: {errors}")
                logger.info(f"   Duration: {tf_duration:.1f}s")
                logger.info(f"   Speed: {bars/tf_duration:.1f} bars/sec" if tf_duration > 0 else "   Speed: N/A")
                logger.info(f"   CSV file: {self.results[tf_name]['csv_file']}")

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
        logger.info(f"Output directory: {self.output_dir.absolute()}")
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
            if result['duration_seconds'] > 0:
                logger.info(f"   Speed: {result['bars_per_second']:.1f} bars/sec")
            if result["status"] == "completed":
                logger.info(f"   CSV file: {result.get('csv_file', 'N/A')}")
            else:
                logger.info(f"   Error: {result.get('error', 'Unknown error')}")

        logger.info("\n" + "-" * 80)
        logger.info("TOTALS:")
        logger.info("-" * 80)
        logger.info(f"Total bars downloaded: {self.total_bars:,}")
        logger.info(f"Total errors: {self.total_errors}")
        if total_duration > 0:
            logger.info(f"Average speed: {self.total_bars/total_duration:.1f} bars/sec")
        logger.info(f"Success rate: {len([r for r in self.results.values() if r['status'] == 'completed'])}/{len(self.TIMEFRAMES)} timeframes")
        logger.info("=" * 80)

        # List CSV files
        logger.info("\nCSV FILES CREATED:")
        logger.info("-" * 80)
        for file in sorted(self.output_dir.glob(f"{self.ticker.replace(':', '_')}_*.csv")):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {file.name} ({size_mb:.2f} MB)")


async def main():
    """Main entry point."""
    backfill = LocalCSVBackfill(ticker="X:ATOMUSD", months=24, output_dir="./atom_data")

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
        logger.info("\n\n⚠️  Backfill interrupted by user.")
        logger.info(f"Partial data saved to: {backfill.output_dir.absolute()}")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
