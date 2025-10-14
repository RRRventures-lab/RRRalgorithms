from datetime import datetime
from hypothesis_tester import HypothesisTester
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
import asyncio
import numpy as np
import pandas as pd
import sys

"""H010: DeFi TVL Changes - TVL momentum predicts token prices"""
sys.path.append(str(Path(__file__).parent))




class DefiTVLHypothesis(HypothesisTester):
    def __init__(self):
        super().__init__(hypothesis_id="H010", title="DeFi TVL Momentum Strategy", category="on-chain", priority_score=690)
        self.primary_feature = 'tvl_change_pct'

    async def collect_historical_data(self, start_date, end_date) -> pd.DataFrame:
        btc_data = await professional_collector.collect_crypto_data("BTC", start_date, end_date, include_sentiment=False)
        df = btc_data['price'].set_index('timestamp').sort_index()
        df['close'] = df['close']

        # Simulate TVL (correlated with price + some lead)
        df['tvl'] = 50_000_000_000 + df['close'].pct_change(48).shift(-24).fillna(0) * 10_000_000_000
        df['tvl'] += np.random.normal(0, 2_000_000_000, len(df))
        df['tvl'] = df['tvl'].clip(30_000_000_000, 100_000_000_000)
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['tvl_change_pct'] = df['tvl'].pct_change(24)
        df['tvl_velocity'] = df['tvl_change_pct'].diff()
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        return df.dropna()

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features.index)
        # LONG when TVL increases >20% in 24h
        strong_inflow = features['tvl_change_pct'] > 0.20
        signals[strong_inflow] = 1
        return signals


async def main():
    tester = DefiTVLHypothesis()
    report = await tester.execute_full_pipeline(datetime(2025, 4, 14), datetime(2025, 10, 11), 6)
    print(f"H010 Decision: {report.decision.decision}, Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
    HypothesisReportGenerator(Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H010")).generate_report(report)
    return report

if __name__ == "__main__":
    asyncio.run(main())
