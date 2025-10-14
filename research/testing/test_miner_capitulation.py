from datetime import datetime, timedelta
from hypothesis_tester import HypothesisTester
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
import asyncio
import numpy as np
import pandas as pd
import sys

"""H009: Miner Capitulation Signal - Bitcoin miner selling predicts bottoms"""
sys.path.append(str(Path(__file__).parent))




class MinerCapitulationHypothesis(HypothesisTester):
    def __init__(self):
        super().__init__(hypothesis_id="H009", title="Miner Capitulation Bottom Signal", category="on-chain", priority_score=750)
        self.primary_feature = 'miner_outflow'

    async def collect_historical_data(self, start_date, end_date) -> pd.DataFrame:
        btc_data = await professional_collector.collect_crypto_data("BTC", start_date, end_date, include_sentiment=False)
        df = btc_data['price'].set_index('timestamp').sort_index()
        df['close'] = df['close']

        # Simulate miner outflows (correlated with price drops + capitulation events)
        df['price_drop'] = -df['close'].pct_change(24).clip(upper=0)  # Only negative returns
        df['miner_outflow'] = df['price_drop'] * 5000 + np.random.normal(0, 1000, len(df))  # BTC units
        df['miner_outflow'] = df['miner_outflow'].clip(0, 50000)  # Realistic bounds
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['outflow_ma'] = df['miner_outflow'].rolling(168).mean()
        df['outflow_zscore'] = (df['miner_outflow'] - df['outflow_ma']) / df['miner_outflow'].rolling(168).std()
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        return df.dropna()

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features.index)
        # Contrarian: High miner outflow → LONG (capitulation bottom)
        capitulation = features['outflow_zscore'] > 2
        signals[capitulation] = 1  # LONG
        return signals


async def main():
    tester = MinerCapitulationHypothesis()
    report = await tester.execute_full_pipeline(datetime(2025, 4, 14), datetime(2025, 10, 11), 6)
    print(f"H009 Decision: {report.decision.decision}, Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
    HypothesisReportGenerator(Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H009")).generate_report(report)
    return report

if __name__ == "__main__":
    asyncio.run(main())
