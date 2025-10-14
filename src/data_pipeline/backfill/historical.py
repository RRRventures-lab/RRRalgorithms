from data_pipeline.polygon.rest_client import PolygonRESTClient
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import os


"""
Historical Data Backfill for Cryptocurrency Trading System
===========================================================

This module backfills historical cryptocurrency market data from Polygon.io
into Supabase for backtesting and model training.

Features:
- Backfill up to 2 years of historical data
- Resumable (tracks progress to handle interruptions)
- Rate-limited to respect API quotas
- Bulk inserts for efficiency
- Progress tracking and reporting

Usage:
    from data_pipeline.backfill.historical import HistoricalDataBackfill
    from src.database import SQLiteClient as DatabaseClient

    supabase = get_db()
    backfill = HistoricalDataBackfill(supabase_client=supabase)

    # Backfill 6 months of data
    await backfill.backfill_aggregates(months=6)
"""



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataBackfill:
    """
    Backfill historical market data from Polygon.io to Supabase.

    This class handles:
    - Fetching historical aggregate (OHLCV) data
    - Storing data in Supabase
    - Tracking progress for resumability
    - Rate limiting and error handling
    """

    # Default tickers to backfill
    DEFAULT_TICKERS = [
        "X:BTCUSD",   # Bitcoin
        "X:ETHUSD",   # Ethereum
        "X:SOLUSD",   # Solana
        "X:ADAUSD",   # Cardano
        "X:DOTUSD",   # Polkadot
        "X:MATICUSD", # Polygon
        "X:AVAXUSD",  # Avalanche
        "X:ATOMUSD",  # Cosmos
        "X:LINKUSD",  # Chainlink
        "X:UNIUSD",   # Uniswap
    ]

    # Progress tracking file
    PROGRESS_FILE = "backfill_progress.json"

    def __init__(
        self,
        polygon_client: Optional[PolygonRESTClient] = None,
        supabase_client=None,
        tickers: Optional[List[str]] = None,
        progress_dir: Optional[str] = None,
    ):
        """
        Initialize historical data backfill.

        Args:
            polygon_client: PolygonRESTClient instance (or create new)
            supabase_client: SupabaseClient instance
            tickers: List of tickers to backfill (default: major cryptos)
            progress_dir: Directory to store progress file
        """
        self.polygon_client = polygon_client or PolygonRESTClient()
        self.supabase_client = supabase_client

        if not self.supabase_client:
            logger.warning("No Supabase client provided. Data will not be stored.")

        self.tickers = tickers or self.DEFAULT_TICKERS

        # Progress tracking
        if progress_dir:
            self.progress_path = Path(progress_dir) / self.PROGRESS_FILE
        else:
            self.progress_path = Path.cwd() / self.PROGRESS_FILE

        self.progress = self._load_progress()

        # Statistics
        self.bars_fetched = 0
        self.bars_stored = 0
        self.errors = 0

        logger.info(f"Historical backfill initialized for {len(self.tickers)} tickers")

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Loaded progress from {self.progress_path}")
                return progress
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
                return {}
        return {}

    def _save_progress(self):
        """Save progress to file."""
        try:
            with open(self.progress_path, 'w') as f:
                json.dump(self.progress, f, indent=2)
            logger.debug(f"Progress saved to {self.progress_path}")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def _mark_ticker_complete(self, ticker: str, end_date: str):
        """Mark ticker as complete in progress."""
        if ticker not in self.progress:
            self.progress[ticker] = {}

        self.progress[ticker]["completed"] = True
        self.progress[ticker]["end_date"] = end_date
        self.progress[ticker]["completed_at"] = datetime.now().isoformat()

        self._save_progress()

    def _get_last_backfilled_date(self, ticker: str) -> Optional[str]:
        """Get the last date that was backfilled for a ticker."""
        if ticker in self.progress:
            return self.progress[ticker].get("last_date")
        return None

    def _update_ticker_progress(self, ticker: str, last_date: str, bars_count: int):
        """Update progress for a ticker."""
        if ticker not in self.progress:
            self.progress[ticker] = {}

        self.progress[ticker]["last_date"] = last_date
        self.progress[ticker]["bars_count"] = self.progress[ticker].get("bars_count", 0) + bars_count
        self.progress[ticker]["last_updated"] = datetime.now().isoformat()

        self._save_progress()

    async def backfill_ticker_aggregates(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        multiplier: int = 1,
        timespan: str = "minute",
        batch_size: int = 50000,
    ) -> int:
        """
        Backfill aggregate data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date for backfill
            end_date: End date for backfill
            multiplier: Bar size multiplier (e.g., 1, 5, 15)
            timespan: Bar timespan (minute, hour, day)
            batch_size: Number of bars per request

        Returns:
            Number of bars fetched and stored
        """
        logger.info(
            f"Backfilling {ticker} from {start_date.date()} to {end_date.date()} "
            f"({multiplier}{timespan} bars)"
        )

        total_bars = 0

        try:
            # Check if already completed
            if self.progress.get(ticker, {}).get("completed"):
                logger.info(f"{ticker} already completed, skipping")
                return 0

            # Get last backfilled date or start from beginning
            last_date = self._get_last_backfilled_date(ticker)
            if last_date:
                current_start = datetime.fromisoformat(last_date) + timedelta(days=1)
                logger.info(f"Resuming {ticker} from {current_start.date()}")
            else:
                current_start = start_date

            # Fetch data in chunks (Polygon has limits)
            # Split into 30-day chunks to avoid hitting limits
            chunk_days = 30
            current_date = current_start

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

                logger.info(
                    f"Fetching {ticker}: {current_date.date()} to {chunk_end.date()}"
                )

                # Fetch aggregates
                try:
                    aggregates = self.polygon_client.get_aggregates(
                        ticker=ticker,
                        multiplier=multiplier,
                        timespan=timespan,
                        from_date=current_date.strftime("%Y-%m-%d"),
                        to_date=chunk_end.strftime("%Y-%m-%d"),
                        limit=batch_size,
                    )

                    self.bars_fetched += len(aggregates)
                    total_bars += len(aggregates)

                    if aggregates:
                        # Convert to Supabase format
                        db_records = []
                        for agg in aggregates:
                            db_records.append({
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

                        # Bulk insert to Supabase
                        if self.supabase_client and db_records:
                            try:
                                self.supabase_client.insert_crypto_aggregates_bulk(db_records)
                                self.bars_stored += len(db_records)
                                logger.info(f"Stored {len(db_records)} bars for {ticker}")
                            except Exception as e:
                                logger.error(f"Error storing data: {e}")
                                self.errors += 1

                        # Update progress
                        self._update_ticker_progress(
                            ticker,
                            chunk_end.strftime("%Y-%m-%d"),
                            len(aggregates)
                        )

                    else:
                        logger.warning(f"No data returned for {ticker} in this period")

                except Exception as e:
                    logger.error(f"Error fetching {ticker} chunk: {e}")
                    self.errors += 1

                # Move to next chunk
                current_date = chunk_end + timedelta(days=1)

                # Small delay to respect rate limits
                await asyncio.sleep(1)

            # Mark as complete
            self._mark_ticker_complete(ticker, end_date.strftime("%Y-%m-%d"))
            logger.info(f"Completed backfill for {ticker}: {total_bars} bars")

        except Exception as e:
            logger.error(f"Error backfilling {ticker}: {e}")
            self.errors += 1

        return total_bars

    async def backfill_aggregates(
        self,
        months: int = 6,
        multiplier: int = 1,
        timespan: str = "minute",
    ):
        """
        Backfill aggregate data for all configured tickers.

        Args:
            months: Number of months to backfill
            multiplier: Bar size multiplier (e.g., 1, 5, 15)
            timespan: Bar timespan (minute, hour, day)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)

        logger.info("=" * 70)
        logger.info("HISTORICAL DATA BACKFILL")
        logger.info("=" * 70)
        logger.info(f"Tickers: {len(self.tickers)}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Timeframe: {multiplier}{timespan}")
        logger.info("=" * 70)

        total_bars = 0

        # Backfill each ticker sequentially
        for i, ticker in enumerate(self.tickers, 1):
            try:
                logger.info(f"\n[{i}/{len(self.tickers)}] Processing {ticker}...")

                bars = await self.backfill_ticker_aggregates(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    multiplier=multiplier,
                    timespan=timespan,
                )

                total_bars += bars

                logger.info(f"Completed {ticker}: {bars} bars")

                # Log to system events
                if self.supabase_client:
                    self.supabase_client.log_system_event(
                        event_type="backfill_complete",
                        severity="info",
                        message=f"Completed backfill for {ticker}",
                        component="historical_backfill",
                        metadata={
                            "ticker": ticker,
                            "bars_count": bars,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to backfill {ticker}: {e}")
                self.errors += 1

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Bars fetched: {self.bars_fetched}")
        logger.info(f"Bars stored: {self.bars_stored}")
        logger.info(f"Errors: {self.errors}")
        logger.info("=" * 70)

        return total_bars

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict[str, Any]:
        """Get backfill statistics."""
        return {
            "bars_fetched": self.bars_fetched,
            "bars_stored": self.bars_stored,
            "errors": self.errors,
            "tickers": self.tickers,
            "progress": self.progress,
        }

    def reset_progress(self, ticker: Optional[str] = None):
        """
        Reset progress tracking.

        Args:
            ticker: Reset specific ticker, or all if None
        """
        if ticker:
            if ticker in self.progress:
                del self.progress[ticker]
                logger.info(f"Reset progress for {ticker}")
        else:
            self.progress = {}
            logger.info("Reset all progress")

        self._save_progress()


# =============================================================================
# Example Usage & CLI
# =============================================================================

async def main():
    """CLI for running historical backfill."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill historical cryptocurrency data")
    parser.add_argument("--months", type=int, default=6, help="Months to backfill (default: 6)")
    parser.add_argument("--timespan", default="minute", choices=["minute", "hour", "day"], help="Bar timespan")
    parser.add_argument("--multiplier", type=int, default=1, help="Bar multiplier (e.g., 5 for 5-minute bars)")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to backfill")
    parser.add_argument("--reset", action="store_true", help="Reset progress before starting")

    args = parser.parse_args()

    # Initialize clients
    from src.database import SQLiteClient as DatabaseClient

    supabase = get_db()
    polygon = PolygonRESTClient()

    # Initialize backfill
    backfill = HistoricalDataBackfill(
        polygon_client=polygon,
        supabase_client=supabase,
        tickers=args.tickers,
    )

    # Reset if requested
    if args.reset:
        backfill.reset_progress()

    # Run backfill
    try:
        await backfill.backfill_aggregates(
            months=args.months,
            multiplier=args.multiplier,
            timespan=args.timespan,
        )
    except KeyboardInterrupt:
        logger.info("\nBackfill interrupted. Progress has been saved.")
        logger.info("Run again to resume from where you left off.")


if __name__ == "__main__":
    asyncio.run(main())
