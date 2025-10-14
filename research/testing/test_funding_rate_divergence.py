from datetime import datetime, timedelta
from hypothesis_tester import HypothesisTester, HypothesisReport
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
from typing import Dict, List, Optional
import asyncio
import numpy as np
import pandas as pd
import sys

"""
Hypothesis H005: Funding Rate Divergence Strategy

Tests whether extreme perpetual futures funding rates predict mean reversion
opportunities in cryptocurrency markets.

Hypothesis: When funding rates become extreme (> 0.1% for 8h funding),
prices tend to mean-revert within 24-48 hours, creating profitable short/long opportunities.

Author: Research Agent H5
Created: 2025-10-12
"""

sys.path.append(str(Path(__file__).parent))




class FundingRateDivergenceHypothesis(HypothesisTester):
    """
    Test funding rate divergence mean reversion strategy.

    Strategy:
    1. Simulate realistic 8-hour funding rates from price momentum & volatility
    2. When funding rate > 0.1% (extreme long bias) → SHORT (expect correction)
    3. When funding rate < -0.1% (extreme short bias) → LONG (expect rally)
    4. Hold 24-48 hours for mean reversion
    """

    def __init__(self):
        """Initialize funding rate divergence tester."""
        super().__init__(
            hypothesis_id="H005",
            title="Funding Rate Divergence Mean Reversion",
            category="microstructure",
            priority_score=900  # High priority - funding rates are powerful signal
        )

        # Strategy parameters
        self.funding_window = 8  # 8-hour funding period (standard for perps)
        self.extreme_positive_threshold = 0.001  # 0.1% per 8h
        self.extreme_negative_threshold = -0.001  # -0.1% per 8h
        self.hold_period = 24  # Hours to hold position

        # Set primary feature
        self.primary_feature = 'funding_rate'

    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect BTC data and simulate funding rates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with prices and simulated funding rates
        """
        print(f"[H005] Collecting BTC data from database...")
        btc_data = await professional_collector.collect_crypto_data(
            "BTC",
            start_date,
            end_date,
            include_sentiment=False
        )

        price_df = btc_data['price'].set_index('timestamp').sort_index()

        # Add close column for base class
        price_df['close'] = price_df['close']

        # Simulate funding rates
        price_df = self._simulate_funding_rates(price_df)

        print(f"[H005] Data ready: {len(price_df)} rows with simulated funding rates")
        return price_df

    def _simulate_funding_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate realistic funding rates based on price momentum and volatility.

        In real perp markets:
        - High positive funding = longs pay shorts = crowded long = correction risk
        - High negative funding = shorts pay longs = crowded short = rally risk
        - Funding correlates with momentum but mean-reverts

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added funding rate column
        """
        df = data.copy()

        # Calculate 8-hour returns (funding period)
        df['return_8h'] = df['close'].pct_change(8)

        # Calculate rolling volatility (24h)
        df['volatility_24h'] = df['close'].pct_change().rolling(24).std()

        # Base funding rate: momentum * volatility factor
        # High momentum + high vol = extreme funding
        df['funding_base'] = df['return_8h'] * df['volatility_24h'] * 3

        # Add mean reversion component
        # Funding should revert to 0 over time
        df['funding_ma_24h'] = df['funding_base'].rolling(24).mean()
        mean_reversion = -df['funding_ma_24h'] * 0.3

        # Add realistic noise
        noise = np.random.normal(0, 0.0002, len(df))  # Small random component

        # Combine components
        df['funding_rate'] = df['funding_base'] + mean_reversion + noise

        # Clip to realistic bounds (-0.3% to +0.3% per 8h)
        df['funding_rate'] = df['funding_rate'].clip(-0.003, 0.003)

        # Fill NaN with 0 (first few rows)
        df['funding_rate'] = df['funding_rate'].fillna(0)

        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for funding rate analysis.

        Features:
        - funding_rate: Current 8h funding rate
        - funding_percentile: Historical percentile ranking
        - funding_zscore: Z-score vs 30-day mean
        - price_momentum: Recent price changes
        - volatility metrics: For risk assessment

        Args:
            data: Raw data with funding rates

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # Funding rate metrics
        df['funding_ma_8h'] = df['funding_rate'].rolling(8).mean()
        df['funding_ma_24h'] = df['funding_rate'].rolling(24).mean()
        df['funding_std_30d'] = df['funding_rate'].rolling(720).std()  # 30 days

        # Z-score (how many std devs from mean)
        df['funding_zscore'] = (
            (df['funding_rate'] - df['funding_ma_24h']) / df['funding_std_30d']
        )
        df['funding_zscore'] = df['funding_zscore'].fillna(0)

        # Percentile ranking (0-100)
        df['funding_percentile'] = df['funding_rate'].rolling(720).rank(pct=True) * 100
        df['funding_percentile'] = df['funding_percentile'].fillna(50)

        # Price momentum (for context)
        df['price_momentum_8h'] = df['close'].pct_change(8)
        df['price_momentum_24h'] = df['close'].pct_change(24)

        # Volatility (for risk management)
        df['volatility_24h'] = df['close'].pct_change().rolling(24).std()

        # Volume surge (if we had volume data)
        if 'volume' in df.columns:
            df['volume_ma_24h'] = df['volume'].rolling(24).mean()
            df['volume_surge'] = df['volume'] / df['volume_ma_24h']
        else:
            df['volume_surge'] = 1.0

        # Forward returns (for validation - not used in signals)
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        df['forward_return_48h'] = df['close'].pct_change(48).shift(-48)

        # Drop NaN rows
        df = df.dropna()

        print(f"[H005] Features engineered: {df.shape[1]} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on extreme funding rates.

        Logic:
        - LONG (1): Extreme negative funding (shorts crowded → expect rally)
        - SHORT (-1): Extreme positive funding (longs crowded → expect correction)
        - HOLD (0): Normal funding levels

        Args:
            features: DataFrame with engineered features

        Returns:
            Series with trading signals {-1, 0, 1}
        """
        signals = pd.Series(0, index=features.index)

        # Extreme positive funding → SHORT signal
        short_conditions = (
            (features['funding_rate'] > self.extreme_positive_threshold) &
            (features['funding_percentile'] > 90) &  # In top 10% historically
            (features['funding_zscore'] > 1.5)  # > 1.5 std devs above mean
        )

        # Extreme negative funding → LONG signal
        long_conditions = (
            (features['funding_rate'] < self.extreme_negative_threshold) &
            (features['funding_percentile'] < 10) &  # In bottom 10% historically
            (features['funding_zscore'] < -1.5)  # > 1.5 std devs below mean
        )

        signals[short_conditions] = -1  # SHORT
        signals[long_conditions] = 1   # LONG

        trade_count = (signals != 0).sum()
        print(f"[H005] Generated {trade_count} signals ({(signals == 1).sum()} LONG, {(signals == -1).sum()} SHORT)")

        return signals


async def main():
    """Run the funding rate divergence hypothesis test."""
    print("=" * 80)
    print("HYPOTHESIS H005: Funding Rate Divergence Strategy")
    print("=" * 80)
    print()

    # Initialize tester
    tester = FundingRateDivergenceHypothesis()

    # Define test period (6 months of data)
    end_date = datetime(2025, 10, 11)
    start_date = end_date - timedelta(days=180)  # 6 months

    print(f"Test Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Mean reversion when funding rates are extreme")
    print(f"Expected Sharpe: 1.0-1.8 if hypothesis holds")
    print()

    # Execute full testing pipeline
    report = await tester.execute_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        lookback_months=6
    )

    # Print summary
    print()
    print("=" * 80)
    print("TEST COMPLETE - H005: Funding Rate Divergence")
    print("=" * 80)
    print(f"Decision: {report.decision.decision}")
    print(f"Confidence: {report.decision.confidence * 100:.1f}%")
    print(f"Sharpe Ratio: {report.backtest_results.sharpe_ratio:.2f}")
    print(f"Win Rate: {report.backtest_results.win_rate * 100:.1f}%")
    print(f"Max Drawdown: {report.backtest_results.max_drawdown * 100:.1f}%")
    print(f"Total Trades: {report.backtest_results.total_trades}")
    print()
    print("Reasoning:")
    for reason in report.decision.reasoning:
        print(f"  - {reason}")
    print()

    # Generate report files
    results_dir = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H005")
    generator = HypothesisReportGenerator(results_dir)
    report_path = generator.generate_report(report)
    print(f"Full report saved to: {report_path}")
    print()

    return report


if __name__ == "__main__":
    asyncio.run(main())
