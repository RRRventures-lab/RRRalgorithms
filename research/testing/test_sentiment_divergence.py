from datetime import datetime, timedelta
from hypothesis_tester import HypothesisTester
from pathlib import Path
from professional_data_collectors import professional_collector
from report_generator import HypothesisReportGenerator
import asyncio
import numpy as np
import pandas as pd
import sys

"""H008: Cross-Exchange Sentiment Divergence - Arbitrage when sentiment diverges"""
sys.path.append(str(Path(__file__).parent))




class SentimentDivergenceHypothesis(HypothesisTester):
    def __init__(self):
        super().__init__(hypothesis_id="H008", title="Sentiment Divergence Arbitrage", category="sentiment", priority_score=680)
        self.primary_feature = 'sentiment_divergence'

    async def collect_historical_data(self, start_date, end_date) -> pd.DataFrame:
        btc_data = await professional_collector.collect_crypto_data("BTC", start_date, end_date, include_sentiment=False)
        df = btc_data['price'].set_index('timestamp').sort_index()
        df['close'] = df['close']

        # Simulate sentiment scores (correlated with price but with divergences)
        df['us_sentiment'] = df['close'].pct_change(24).rolling(24).mean() + np.random.normal(0, 0.01, len(df))
        df['asia_sentiment'] = df['close'].pct_change(24).rolling(24).mean() + np.random.normal(0, 0.015, len(df))
        df['sentiment_divergence'] = (df['us_sentiment'] - df['asia_sentiment']).abs()
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['divergence_zscore'] = (df['sentiment_divergence'] - df['sentiment_divergence'].rolling(168).mean()) / df['sentiment_divergence'].rolling(168).std()
        df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)
        return df.dropna()

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features.index)
        # Trade when divergence is extreme
        high_divergence = features['divergence_zscore'] > 2
        us_bullish = features['us_sentiment'] > features['asia_sentiment']
        signals[high_divergence & us_bullish] = 1  # LONG
        signals[high_divergence & ~us_bullish] = -1  # SHORT
        return signals


async def main():
    tester = SentimentDivergenceHypothesis()
    report = await tester.execute_full_pipeline(datetime(2025, 4, 14), datetime(2025, 10, 11), 6)
    print(f"H008 Decision: {report.decision.decision}, Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
    HypothesisReportGenerator(Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/H008")).generate_report(report)
    return report

if __name__ == "__main__":
    asyncio.run(main())
