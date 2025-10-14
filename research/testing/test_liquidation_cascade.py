from datetime import datetime, timedelta
from hypothesis_tester import HypothesisTester
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
import asyncio
import numpy as np
import pandas as pd
import sys

"""
Hypothesis H007: Liquidation Cascade Prediction & Defense

Defensive strategy that reduces risk when price approaches major liquidation clusters
and takes contrarian positions after cascades.

Author: Research Agent H7
Created: 2025-10-12
"""

sys.path.append(str(Path(__file__).parent))




class LiquidationCascadeHypothesis(HypothesisTester):
    """Test liquidation cascade defense strategy."""

    def __init__(self):
        super().__init__(
            hypothesis_id="H007",
            title="Liquidation Cascade Defense Strategy",
            category="microstructure",
            priority_score=550
        )
        self.primary_feature = 'cascade_risk_score'

    async def collect_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        print(f"[H007] Collecting BTC data...")
        btc_data = await professional_collector.collect_crypto_data("BTC", start_date, end_date, include_sentiment=False)
        df = btc_data['price'].set_index('timestamp').sort_index()
        df['close'] = df['close']

        # Simulate liquidation clusters at support/resistance levels
        df = self._simulate_liquidation_clusters(df)
        print(f"[H007] Data ready: {len(df)} rows with liquidation risk scores")
        return df

    def _simulate_liquidation_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Find support/resistance (local min/max)
        df['rolling_high'] = df['close'].rolling(48).max()
        df['rolling_low'] = df['close'].rolling(48).min()

        # Distance to nearest cluster (%)
        df['distance_to_high_pct'] = (df['rolling_high'] - df['close']) / df['close']
        df['distance_to_low_pct'] = (df['close'] - df['rolling_low']) / df['close']
        df['distance_to_cluster'] = df[['distance_to_high_pct', 'distance_to_low_pct']].min(axis=1)

        # Volatility and volume surge
        df['volatility'] = df['close'].pct_change().rolling(24).std()
        df['vol_surge'] = df.get('volume', pd.Series(1, index=df.index)) / df.get('volume', pd.Series(1, index=df.index)).rolling(24).mean()

        # Cascade risk score (0-1)
        risk_from_distance = 1 / (1 + df['distance_to_cluster'] * 50)  # Higher when close to cluster
        risk_from_vol = df['volatility'] / df['volatility'].rolling(168).mean()  # Higher in volatile periods
        df['cascade_risk_score'] = (risk_from_distance * 0.6 + risk_from_vol.fillna(1) * 0.4).clip(0, 1)

        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        df = df.dropna()
        print(f"[H007] Features: {df.shape[1]} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features.index)

        # Exit/reduce when risk score > 0.75
        # (Defensive: cash is a position)
        exit_condition = features['cascade_risk_score'] > 0.75
        signals[exit_condition] = 0  # Flat

        # Contrarian long after detected cascade (risk was high, price dropped)
        cascade_detected = (features['cascade_risk_score'].shift(1) > 0.75) & (features['close'].pct_change() < -0.03)
        signals[cascade_detected] = 1  # LONG after cascade

        # Default trend following when risk is low
        low_risk = features['cascade_risk_score'] < 0.3
        uptrend = features['close'] > features['close'].rolling(48).mean()
        signals[low_risk & uptrend] = 1  # LONG in low risk uptrends

        print(f"[H007] Generated {(signals != 0).sum()} signals")
        return signals


async def main():
    print("=" * 80)
    print("HYPOTHESIS H007: Liquidation Cascade Defense")
    print("=" * 80)

    tester = LiquidationCascadeHypothesis()
    end_date = datetime(2025, 10, 11)
    start_date = end_date - timedelta(days=180)

    report = await tester.execute_full_pipeline(start_date, end_date, 6)

    print(f"\nDecision: {report.decision.decision}")
    print(f"Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
    print(f"Win Rate: {report.backtest_results.win_rate * 100:.1f}%")

    results_dir = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H007")
    generator = HypothesisReportGenerator(results_dir)
    generator.generate_report(report)

    return report

if __name__ == "__main__":
    asyncio.run(main())
