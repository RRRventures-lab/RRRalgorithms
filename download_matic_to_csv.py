from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import csv
import logging
import os
import sys
import time

#!/usr/bin/env python3
"""
Download 2 years of historical data for Polygon (X:MATICUSD) and save to CSV files
Since Supabase keys need to be updated, this saves data locally for safekeeping
"""


# Load environment variables
load_dotenv("/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env")

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
OUTPUT_DIR = Path("./data/historical/maticusd")


async def download_and_save_timeframe(
    polygon: PolygonRESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    label: str,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """
    Download data for a single timeframe and save to CSV.

    Returns:
        Dictionary with results
    """
    logger.info("=" * 80)
    logger.info(f"Starting download for {ticker} - {label}")
    logger.info("=" * 80)

    start_time = time.time()
    total_bars = 0
    all_data = []

    try:
        # Split into 30-day chunks
        chunk_days = 30
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

            logger.info(f"Fetching {ticker}: {current_date.date()} to {chunk_end.date()}")

            try:
                aggregates = polygon.get_aggregates(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=current_date.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                    limit=50000,
                )

                if aggregates:
                    for agg in aggregates:
                        all_data.append({
                            "ticker": ticker,
                            "event_time": agg.datetime.isoformat(),
                            "open": float(agg.open),
                            "high": float(agg.high),
                            "low": float(agg.low),
                            "close": float(agg.close),
                            "volume": float(agg.volume),
                            "vwap": float(agg.vwap) if agg.vwap else None,
                            "trade_count": agg.trade_count,
                        })

                    total_bars += len(aggregates)
                    logger.info(f"Retrieved {len(aggregates)} bars (total: {total_bars})")

            except Exception as e:
                logger.error(f"Error fetching chunk: {e}")

            # Move to next chunk
            current_date = chunk_end + timedelta(days=1)

            # Rate limit delay
            await asyncio.sleep(0.2)  # 5 requests per second

        # Save to CSV
        if all_data:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            csv_file = OUTPUT_DIR / f"maticusd_{label}.csv"

            with open(csv_file, 'w', newline='') as f:
                fieldnames = ["ticker", "event_time", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)

            logger.info(f"Saved {len(all_data)} bars to {csv_file}")

        elapsed = time.time() - start_time

        result = {
            "timeframe": label,
            "bars": total_bars,
            "elapsed_seconds": round(elapsed, 2),
            "success": True,
            "error": None,
            "csv_file": str(csv_file) if all_data else None
        }

        logger.info(f"Completed {label}: {total_bars} bars in {elapsed:.2f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed to download {label}: {e}")
        return {
            "timeframe": label,
            "bars": total_bars,
            "elapsed_seconds": round(elapsed, 2),
            "success": False,
            "error": str(e),
            "csv_file": None
        }


async def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("POLYGON (MATIC) HISTORICAL DATA DOWNLOAD TO CSV")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Duration: {MONTHS} months (2 years)")
    logger.info(f"Timeframes: {', '.join([tf['label'] for tf in TIMEFRAMES])}")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    logger.info("=" * 80)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS * 30)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 80)

    # Initialize Polygon client
    try:
        logger.info("Initializing Polygon client...")
        polygon = PolygonRESTClient()
        logger.info("Polygon client initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Polygon client: {e}")
        return 1

    # Results tracking
    results = []
    total_bars = 0
    total_errors = 0
    overall_start = time.time()

    # Download each timeframe sequentially
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        logger.info(f"\n[{i}/{len(TIMEFRAMES)}] Processing {timeframe['label']}...")

        result = await download_and_save_timeframe(
            polygon=polygon,
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

        # Small delay between timeframes
        if i < len(TIMEFRAMES):
            logger.info("Pausing 3 seconds before next timeframe...")
            await asyncio.sleep(3)

    overall_elapsed = time.time() - overall_start

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Ticker: {TICKER}")
    logger.info(f"Total Duration: {overall_elapsed / 60:.2f} minutes")
    logger.info(f"Total Bars Downloaded: {total_bars:,}")
    logger.info(f"Total Errors: {total_errors}")
    logger.info(f"Data saved to: {OUTPUT_DIR.absolute()}")
    logger.info("=" * 80)
    logger.info("\nBreakdown by Timeframe:")
    logger.info("-" * 80)

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        csv_info = f" | File: {result['csv_file']}" if result['csv_file'] else ""
        logger.info(
            f"  {result['timeframe']:>6} | {status:>7} | "
            f"{result['bars']:>8,} bars | "
            f"{result['elapsed_seconds']:>6.2f}s{csv_info}"
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
        logger.info(f"  Files created: {sum(1 for r in results if r['csv_file'])}")

    logger.info("=" * 80)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
