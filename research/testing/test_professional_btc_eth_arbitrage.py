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
Professional hypothesis test: BTC-ETH Correlation Arbitrage

Uses REAL data from Polygon.io to test if BTC-ETH correlation breaks
create profitable arbitrage opportunities.

This is a more realistic arbitrage strategy than CEX-DEX, as we're using:
- Real Polygon.io data (not simulated)
- Both BTC and ETH hourly prices
- Correlation-based signals
"""

sys.path.append(str(Path(__file__).parent))




class BTCETHArbitrageHypothesis(HypothesisTester):
    """
    Test BTC-ETH correlation arbitrage.

    Strategy:
    1. Monitor BTC-ETH price correlation (usually 0.8+)
    2. When correlation breaks (BTC moves but ETH doesn't), trade the laggard
    3. Mean reversion play - expect correlation to restore
    """

    def __init__(self):
        """Initialize BTC-ETH arbitrage tester."""
        super().__init__(
            hypothesis_id="H004",
            title="BTC-ETH Correlation Arbitrage (Real Polygon Data)",
            category="arbitrage",
            priority_score=850
        )

        # Strategy parameters
        self.lookback_window = 24  # Hours for correlation calculation
        self.correlation_threshold = 0.7  # Normal correlation level
        self.breakout_threshold = 0.5  # Trade when correlation drops below this

        # Set primary feature
        self.primary_feature = 'correlation_break'

    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect BTC and ETH data from Polygon.io.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with BTC and ETH prices
        """
        print(f"[H004] Collecting BTC data from Polygon.io...")
        btc_data = await professional_collector.collect_crypto_data(
            "BTC",
            start_date,
            end_date,
            include_sentiment=False
        )

        print(f"[H004] Collecting ETH data from Polygon.io...")
        eth_data = await professional_collector.collect_crypto_data(
            "ETH",
            start_date,
            end_date,
            include_sentiment=False
        )

        # Merge BTC and ETH data
        btc_price = btc_data['price'][['timestamp', 'close']].rename(columns={'close': 'btc_close'})
        eth_price = eth_data['price'][['timestamp', 'close']].rename(columns={'close': 'eth_close'})

        merged = pd.merge(btc_price, eth_price, on='timestamp', how='inner')
        merged = merged.set_index('timestamp').sort_index()

        # Add a 'close' column for the base class (use BTC as reference)
        merged['close'] = merged['btc_close']

        print(f"[H004] Merged data: {len(merged)} rows")

        return merged

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer correlation-based features.

        Args:
            data: Raw data with btc_close and eth_close

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)

        # Calculate returns
        features['btc_return'] = data['btc_close'].pct_change()
        features['eth_return'] = data['eth_close'].pct_change()

        # Rolling correlation
        features['correlation'] = features['btc_return'].rolling(self.lookback_window).corr(
            features['eth_return']
        )

        # Correlation break signal
        features['correlation_break'] = (
            (features['correlation'] < self.breakout_threshold) &
            (features['correlation'].shift(1) > self.correlation_threshold)
        ).astype(int)

        # Price ratio (BTC / ETH)
        features['price_ratio'] = data['btc_close'] / data['eth_close']
        features['price_ratio_ma'] = features['price_ratio'].rolling(self.lookback_window).mean()
        features['price_ratio_zscore'] = (
            (features['price_ratio'] - features['price_ratio_ma']) /
            features['price_ratio'].rolling(self.lookback_window).std()
        )

        # Relative strength
        features['btc_stronger'] = features['btc_return'] > features['eth_return']

        # Volatility
        features['btc_vol'] = features['btc_return'].rolling(self.lookback_window).std()
        features['eth_vol'] = features['eth_return'].rolling(self.lookback_window).std()
        features['vol_ratio'] = features['btc_vol'] / features['eth_vol']

        return features

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate arbitrage signals.

        Args:
            features: Engineered features

        Returns:
            Series of trading signals (1=long correlation restore, 0=neutral)
        """
        signals = pd.Series(0, index=features.index)

        # When correlation breaks and BTC is stronger, expect ETH to catch up
        long_eth_condition = (
            (features['correlation'] < self.breakout_threshold) &
            (features['btc_stronger']) &
            (features['price_ratio_zscore'] > 0.5)  # BTC overextended vs ETH
        )

        # When correlation breaks and ETH is stronger, expect BTC to catch up
        long_btc_condition = (
            (features['correlation'] < self.breakout_threshold) &
            (~features['btc_stronger']) &
            (features['price_ratio_zscore'] < -0.5)  # ETH overextended vs BTC
        )

        # For simplicity, we'll trade on BTC (long when BTC needs to catch up)
        signals[long_btc_condition] = 1

        return signals

    async def execute_full_pipeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_months: int = 3
    ) -> HypothesisReport:
        """Execute full test pipeline."""
        # Call parent implementation
        report = await super().execute_full_pipeline(
            start_date,
            end_date,
            lookback_months
        )

        # Generate custom visualizations
        self._generate_custom_charts(report)

        # Generate markdown report
        report_gen = HypothesisReportGenerator(self.results_dir)
        report_gen.generate_report(
            hypothesis_id=self.metadata.hypothesis_id,
            title=self.metadata.title,
            backtest_results=report.backtest_results,
            statistical_validation=report.statistical_validation,
            decision=report.decision,
            metadata={
                'category': self.metadata.category,
                'priority_score': self.metadata.priority_score,
                'execution_time': report.execution_time,
                'data_source': 'Polygon.io (Real Data)'
            }
        )

        return report

    def _generate_custom_charts(self, report: HypothesisReport):
        """Generate custom charts."""
        import matplotlib.pyplot as plt

        if self.raw_data is None or self.features is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. BTC vs ETH price (normalized)
        btc_norm = self.raw_data['btc_close'] / self.raw_data['btc_close'].iloc[0]
        eth_norm = self.raw_data['eth_close'] / self.raw_data['eth_close'].iloc[0]

        axes[0, 0].plot(btc_norm.index, btc_norm.values, label='BTC', alpha=0.7)
        axes[0, 0].plot(eth_norm.index, eth_norm.values, label='ETH', alpha=0.7)
        axes[0, 0].set_title('BTC vs ETH Price (Normalized)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Normalized Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Rolling correlation
        axes[0, 1].plot(self.features.index, self.features['correlation'], alpha=0.7)
        axes[0, 1].axhline(y=self.correlation_threshold, color='g', linestyle='--', label='Normal')
        axes[0, 1].axhline(y=self.breakout_threshold, color='r', linestyle='--', label='Breakout')
        axes[0, 1].set_title(f'BTC-ETH Correlation ({self.lookback_window}h rolling)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Price ratio z-score
        axes[1, 0].plot(self.features.index, self.features['price_ratio_zscore'], alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('BTC/ETH Price Ratio Z-Score')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Z-Score')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Returns comparison
        axes[1, 1].scatter(
            self.features['btc_return'],
            self.features['eth_return'],
            alpha=0.3,
            s=10
        )
        axes[1, 1].plot([-0.05, 0.05], [-0.05, 0.05], 'r--', alpha=0.5, label='Perfect Correlation')
        axes[1, 1].set_title('BTC vs ETH Returns')
        axes[1, 1].set_xlabel('BTC Return')
        axes[1, 1].set_ylabel('ETH Return')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = self.results_dir / "btc_eth_arbitrage_analysis.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[H004] Custom charts saved to {chart_file}")


async def main():
    """Run the BTC-ETH arbitrage hypothesis test."""
    print("=" * 70)
    print("HYPOTHESIS TEST: BTC-ETH Correlation Arbitrage (Real Polygon Data)")
    print("=" * 70)

    # Initialize tester
    tester = BTCETHArbitrageHypothesis()

    # Define test period (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    print(f"\nTest Period: {start_date.date()} to {end_date.date()}")
    print(f"Data Source: Polygon.io (100% real data)")
    print(f"Assets: BTC-USD, ETH-USD")

    # Run test
    report = await tester.execute_full_pipeline(
        start_date=start_date,
        end_date=end_date
    )

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Decision: {report.decision.decision}")
    print(f"Confidence: {report.decision.confidence:.0%}")
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {report.backtest_results.sharpe_ratio:.2f}")
    print(f"  Win Rate: {report.backtest_results.win_rate:.1%}")
    print(f"  Total Return: {report.backtest_results.total_return:.2%}")
    print(f"  Max Drawdown: {report.backtest_results.max_drawdown:.2%}")
    print(f"  Total Trades: {report.backtest_results.total_trades}")
    print(f"\nStatistical Validation:")
    print(f"  P-value: {report.statistical_validation.p_value:.4f}")
    print(f"  Significant: {report.statistical_validation.significant}")
    print(f"\nReasoning:")
    for reason in report.decision.reasoning:
        print(f"  - {reason}")
    print(f"\nNext Steps:")
    for step in report.decision.next_steps:
        print(f"  - {step}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = asyncio.run(main())
