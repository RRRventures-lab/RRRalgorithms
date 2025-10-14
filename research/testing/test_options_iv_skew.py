from datetime import datetime
from hypothesis_tester import HypothesisTester
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
import asyncio
import numpy as np
import pandas as pd
import sys

"""H011: Options IV Skew - Implied volatility skew predicts directional moves"""
sys.path.append(str(Path(__file__).parent))




class OptionsIVSkewHypothesis(HypothesisTester):
    def __init__(self):
        super().__init__(hypothesis_id="H011", title="Options IV Skew Strategy", category="derivatives", priority_score=710)
        self.primary_feature = 'iv_skew'

    async def collect_historical_data(self, start_date, end_date) -> pd.DataFrame:
        btc_data = await professional_collector.collect_crypto_data("BTC", start_date, end_date, include_sentiment=False)
        df = btc_data['price'].set_index('timestamp').sort_index()
        df['close'] = df['close']

        # Simulate IV skew (put/call vol difference)
        df['realized_vol'] = df['close'].pct_change().rolling(48).std() * np.sqrt(365 * 24)
        df['put_iv'] = df['realized_vol'] * 1.2 + np.random.normal(0, 0.05, len(df))
        df['call_iv'] = df['realized_vol'] * 0.9 + np.random.normal(0, 0.05, len(df))
        df['iv_skew'] = df['put_iv'] - df['call_iv']  # Positive = fear (put demand)
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['skew_zscore'] = (df['iv_skew'] - df['iv_skew'].rolling(168).mean()) / df['iv_skew'].rolling(168).std()
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        return df.dropna()

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features.index)
        # Extreme positive skew (fear) → contrarian LONG
        extreme_fear = features['skew_zscore'] > 2
        signals[extreme_fear] = 1  # LONG
        # Extreme negative skew (greed) → contrarian SHORT
        extreme_greed = features['skew_zscore'] < -2
        signals[extreme_greed] = -1  # SHORT
        return signals


async def main():
    tester = OptionsIVSkewHypothesis()
    report = await tester.execute_full_pipeline(datetime(2025, 4, 14), datetime(2025, 10, 11), 6)
    print(f"H011 Decision: {report.decision.decision}, Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
    HypothesisReportGenerator(Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H011")).generate_report(report)
    return report

if __name__ == "__main__":
    asyncio.run(main())
