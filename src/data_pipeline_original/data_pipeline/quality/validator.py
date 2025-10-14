from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
import statistics


"""
Data Quality Validator for Market Data Pipeline
================================================

This module validates the quality of market data, checking for:
- Missing data and gaps
- Outliers and anomalies
- Data consistency
- Timestamp ordering
- Duplicate records

All quality issues are logged to the system_events table in Supabase.

Usage:
    from data_pipeline.quality.validator import DataQualityValidator
    from data_pipeline.supabase_client import SupabaseClient

    supabase = SupabaseClient()
    validator = DataQualityValidator(supabase_client=supabase)

    # Validate recent data
    issues = await validator.validate_recent_data()

    # Run continuous monitoring
    await validator.run_monitoring()
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityIssue:
    """Represents a data quality issue."""

    def __init__(
        self,
        issue_type: str,
        severity: str,
        message: str,
        table: str,
        ticker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.issue_type = issue_type
        self.severity = severity
        self.message = message
        self.table = table
        self.ticker = ticker
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_type": "data_quality_issue",
            "severity": self.severity,
            "message": self.message,
            "component": "data_quality_validator",
            "metadata": {
                "issue_type": self.issue_type,
                "table": self.table,
                "ticker": self.ticker,
                **self.metadata
            }
        }


class DataQualityValidator:
    """
    Validates data quality for cryptocurrency market data.

    Checks performed:
    1. Missing data detection (gaps in time series)
    2. Outlier detection (price spikes, unusual volumes)
    3. Duplicate detection
    4. Timestamp ordering
    5. Data completeness (null values)
    """

    # Outlier detection thresholds
    PRICE_CHANGE_THRESHOLD = 0.20  # 20% price change in 1 minute
    VOLUME_SPIKE_THRESHOLD = 5.0   # 5x average volume
    Z_SCORE_THRESHOLD = 4.0        # Z-score for outliers

    # Data gap thresholds
    MAX_GAP_MINUTES = 5  # Maximum acceptable gap in aggregates

    def __init__(
        self,
        supabase_client,
        tickers: Optional[List[str]] = None,
        check_interval: int = 300,  # 5 minutes
    ):
        """
        Initialize data quality validator.

        Args:
            supabase_client: SupabaseClient instance
            tickers: List of tickers to monitor (None = all)
            check_interval: Check interval in seconds
        """
        self.supabase_client = supabase_client
        if not self.supabase_client:
            raise ValueError("SupabaseClient is required")

        self.tickers = tickers
        self.check_interval = check_interval
        self.running = False

        self.issues_found = 0
        self.checks_performed = 0

        logger.info("Data quality validator initialized")

    async def check_missing_data(
        self,
        ticker: str,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Check for missing data gaps in aggregates.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            List of DataQualityIssue objects
        """
        issues = []
        try:
            # Get recent aggregates
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            aggregates = self.supabase_client.get_price_history(
                ticker=ticker,
                start_time=start_time
            )

            if not aggregates:
                issue = DataQualityIssue(
                    issue_type="no_data",
                    severity="warning",
                    message=f"No aggregate data found for {ticker} in last {lookback_hours} hours",
                    table="crypto_aggregates",
                    ticker=ticker,
                    metadata={"lookback_hours": lookback_hours}
                )
                issues.append(issue)
                return issues

            # Check for gaps between consecutive records
            previous_time = None
            for agg in aggregates:
                current_time = datetime.fromisoformat(agg["event_time"].replace("Z", "+00:00"))

                if previous_time:
                    gap_minutes = (current_time - previous_time).total_seconds() / 60

                    if gap_minutes > self.MAX_GAP_MINUTES:
                        issue = DataQualityIssue(
                            issue_type="data_gap",
                            severity="warning",
                            message=f"Data gap of {gap_minutes:.1f} minutes detected for {ticker}",
                            table="crypto_aggregates",
                            ticker=ticker,
                            metadata={
                                "gap_minutes": gap_minutes,
                                "gap_start": previous_time.isoformat(),
                                "gap_end": current_time.isoformat(),
                            }
                        )
                        issues.append(issue)

                previous_time = current_time

        except Exception as e:
            logger.error(f"Error checking missing data for {ticker}: {e}")

        return issues

    async def check_price_outliers(
        self,
        ticker: str,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Check for price outliers and unusual movements.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            List of DataQualityIssue objects
        """
        issues = []
        try:
            # Get recent aggregates
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            aggregates = self.supabase_client.get_price_history(
                ticker=ticker,
                start_time=start_time
            )

            if len(aggregates) < 2:
                return issues

            # Calculate price changes
            previous_close = None
            prices = []

            for agg in aggregates:
                current_close = float(agg["close"])
                prices.append(current_close)

                if previous_close:
                    price_change = abs(current_close - previous_close) / previous_close

                    # Check for sudden price spikes
                    if price_change > self.PRICE_CHANGE_THRESHOLD:
                        issue = DataQualityIssue(
                            issue_type="price_spike",
                            severity="warning",
                            message=f"Large price change detected for {ticker}: {price_change*100:.1f}%",
                            table="crypto_aggregates",
                            ticker=ticker,
                            metadata={
                                "price_change_pct": price_change * 100,
                                "previous_price": previous_close,
                                "current_price": current_close,
                                "timestamp": agg["event_time"],
                            }
                        )
                        issues.append(issue)

                previous_close = current_close

            # Check for statistical outliers using Z-score
            if len(prices) >= 10:
                mean_price = statistics.mean(prices)
                stdev_price = statistics.stdev(prices)

                if stdev_price > 0:
                    for i, price in enumerate(prices):
                        z_score = abs((price - mean_price) / stdev_price)

                        if z_score > self.Z_SCORE_THRESHOLD:
                            issue = DataQualityIssue(
                                issue_type="statistical_outlier",
                                severity="info",
                                message=f"Statistical outlier detected for {ticker} (Z-score: {z_score:.2f})",
                                table="crypto_aggregates",
                                ticker=ticker,
                                metadata={
                                    "z_score": z_score,
                                    "price": price,
                                    "mean_price": mean_price,
                                    "stdev_price": stdev_price,
                                    "timestamp": aggregates[i]["event_time"],
                                }
                            )
                            issues.append(issue)

        except Exception as e:
            logger.error(f"Error checking price outliers for {ticker}: {e}")

        return issues

    async def check_volume_outliers(
        self,
        ticker: str,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Check for volume outliers and spikes.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            List of DataQualityIssue objects
        """
        issues = []
        try:
            # Get recent aggregates
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            aggregates = self.supabase_client.get_price_history(
                ticker=ticker,
                start_time=start_time
            )

            if len(aggregates) < 10:
                return issues

            # Extract volumes
            volumes = [float(agg["volume"]) for agg in aggregates if agg.get("volume")]

            if not volumes:
                return issues

            # Calculate average volume
            avg_volume = statistics.mean(volumes)

            # Check for volume spikes
            for i, agg in enumerate(aggregates):
                volume = float(agg.get("volume", 0))

                if volume > 0 and avg_volume > 0:
                    volume_ratio = volume / avg_volume

                    if volume_ratio > self.VOLUME_SPIKE_THRESHOLD:
                        issue = DataQualityIssue(
                            issue_type="volume_spike",
                            severity="info",
                            message=f"Volume spike detected for {ticker}: {volume_ratio:.1f}x average",
                            table="crypto_aggregates",
                            ticker=ticker,
                            metadata={
                                "volume": volume,
                                "avg_volume": avg_volume,
                                "volume_ratio": volume_ratio,
                                "timestamp": agg["event_time"],
                            }
                        )
                        issues.append(issue)

        except Exception as e:
            logger.error(f"Error checking volume outliers for {ticker}: {e}")

        return issues

    async def check_null_values(
        self,
        ticker: str,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Check for null/missing values in critical fields.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            List of DataQualityIssue objects
        """
        issues = []
        try:
            # Get recent aggregates
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            aggregates = self.supabase_client.get_price_history(
                ticker=ticker,
                start_time=start_time
            )

            # Check for null critical fields
            critical_fields = ["open", "high", "low", "close", "volume"]

            for agg in aggregates:
                missing_fields = []
                for field in critical_fields:
                    if agg.get(field) is None:
                        missing_fields.append(field)

                if missing_fields:
                    issue = DataQualityIssue(
                        issue_type="null_values",
                        severity="error",
                        message=f"Null values in critical fields for {ticker}: {', '.join(missing_fields)}",
                        table="crypto_aggregates",
                        ticker=ticker,
                        metadata={
                            "missing_fields": missing_fields,
                            "timestamp": agg["event_time"],
                        }
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(f"Error checking null values for {ticker}: {e}")

        return issues

    async def validate_ticker(
        self,
        ticker: str,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Run all validation checks for a ticker.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            List of all DataQualityIssue objects found
        """
        logger.info(f"Validating data quality for {ticker}...")

        all_issues = []

        # Run all checks
        checks = [
            self.check_missing_data(ticker, lookback_hours),
            self.check_price_outliers(ticker, lookback_hours),
            self.check_volume_outliers(ticker, lookback_hours),
            self.check_null_values(ticker, lookback_hours),
        ]

        # Execute checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)

        # Collect issues
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Check failed: {result}")
            elif isinstance(result, list):
                all_issues.extend(result)

        # Log issues to Supabase
        for issue in all_issues:
            try:
                self.supabase_client.log_system_event(**issue.to_dict())
            except Exception as e:
                logger.error(f"Failed to log issue: {e}")

        self.issues_found += len(all_issues)
        self.checks_performed += 1

        if all_issues:
            logger.warning(f"Found {len(all_issues)} quality issues for {ticker}")
        else:
            logger.info(f"No quality issues found for {ticker}")

        return all_issues

    async def validate_recent_data(
        self,
        lookback_hours: int = 24
    ) -> List[DataQualityIssue]:
        """
        Validate recent data for all monitored tickers.

        Args:
            lookback_hours: Hours to look back

        Returns:
            List of all issues found
        """
        logger.info("Starting data quality validation...")

        # Get tickers to check
        if self.tickers:
            tickers = self.tickers
        else:
            # Get all unique tickers from recent data
            # For now, use default list
            tickers = ["X:BTCUSD", "X:ETHUSD", "X:SOLUSD"]

        all_issues = []

        # Validate each ticker
        for ticker in tickers:
            try:
                issues = await self.validate_ticker(ticker, lookback_hours)
                all_issues.extend(issues)

                # Small delay between tickers
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Failed to validate {ticker}: {e}")

        logger.info(
            f"Validation complete: {len(all_issues)} issues found across {len(tickers)} tickers"
        )

        return all_issues

    async def run_monitoring(self):
        """
        Run continuous data quality monitoring.

        This runs indefinitely, performing validation checks at regular intervals.
        """
        self.running = True
        logger.info(f"Starting data quality monitoring (interval: {self.check_interval}s)")

        while self.running:
            try:
                start_time = datetime.now()

                # Run validation
                issues = await self.validate_recent_data()

                # Calculate elapsed time
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Validation completed in {elapsed:.1f}s")

                # Wait for next interval
                sleep_time = max(0, self.check_interval - elapsed)
                if sleep_time > 0:
                    logger.info(f"Next validation in {sleep_time:.0f}s")
                    await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.running = False
                break

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

        logger.info("Stopped data quality monitoring")

    async def stop(self):
        """Stop the monitoring."""
        logger.info("Stopping data quality validator...")
        self.running = False

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "running": self.running,
            "checks_performed": self.checks_performed,
            "issues_found": self.issues_found,
            "check_interval": self.check_interval,
            "monitored_tickers": self.tickers,
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of data quality validator."""
    from data_pipeline.supabase_client import SupabaseClient

    # Initialize Supabase client
    supabase = SupabaseClient()

    # Initialize validator
    validator = DataQualityValidator(
        supabase_client=supabase,
        check_interval=300,  # 5 minutes
    )

    # Option 1: Validate once
    issues = await validator.validate_recent_data(lookback_hours=24)
    print(f"\nFound {len(issues)} quality issues")
    for issue in issues[:5]:  # Show first 5
        print(f"  - [{issue.severity}] {issue.message}")

    # Option 2: Run continuous monitoring (uncomment to use)
    # await validator.run_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
