from datetime import datetime, timedelta
from pathlib import Path
from research.testing.hypothesis_tester import HypothesisTester
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import numpy as np
import pandas as pd
import sys

"""
H015: MACD Crossover Strategy
==============================

Classical technical analysis strategy using Moving Average Convergence Divergence (MACD).

Strategy Logic:
- LONG when MACD crosses above signal line (bullish crossover)
- SHORT when MACD crosses below signal line (bearish crossover)
- Additional confirmation from histogram and trend

MACD Parameters:
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal line: 9-period EMA of MACD
- Histogram: MACD - Signal line

Expected Performance:
- Sharpe Ratio: 0.6-0.9
- Win Rate: 45-55%
- Trade Frequency: 3-8%

Reference: Gerald Appel (1979) - "Technical Analysis: Power Tools for Active Investors"

Author: RRR Ventures
Date: 2025-10-12
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class MACDCrossoverTester(HypothesisTester):
    """
    Test MACD crossover strategy with real data.

    Uses standard MACD (12, 26, 9) to identify trend changes
    and momentum shifts.
    """

    def __init__(self):
        super().__init__(
            hypothesis_id="H015",
            title="MACD Crossover Strategy (12,26,9)",
            category="momentum",
            priority_score=85
        )

        self.collector = ProfessionalDataCollector()

        # Strategy parameters
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

        # Primary feature for statistical validation
        self.primary_feature = 'macd'

    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect historical price data for MACD calculation"""
        print(f"[{self.metadata.hypothesis_id}] Collecting BTC data from database...")

        price_data = await self.collector.polygon.get_crypto_aggregates(
            "X:BTCUSD",
            start_date,
            end_date,
            "hour",
            1
        )

        if price_data.empty:
            raise ValueError("No price data collected")

        df = price_data.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"[{self.metadata.hypothesis_id}] Data ready: {len(df)} rows")
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD and related features.

        Features:
        1. MACD line (12 EMA - 26 EMA)
        2. Signal line (9 EMA of MACD)
        3. Histogram (MACD - Signal)
        4. Crossover events
        5. Trend strength

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # ========================================
        # 1. Calculate EMAs
        # ========================================
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # ========================================
        # 2. MACD Line
        # ========================================
        df['macd'] = df['ema_fast'] - df['ema_slow']

        # ========================================
        # 3. Signal Line
        # ========================================
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()

        # ========================================
        # 4. Histogram
        # ========================================
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # ========================================
        # 5. Crossover Detection
        # ========================================
        # Bullish crossover: MACD crosses above signal
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_above_signal_prev'] = df['macd_above_signal'].shift(1)

        df['bullish_crossover'] = (
            (df['macd_above_signal'] == 1) &
            (df['macd_above_signal_prev'] == 0)
        ).astype(int)

        # Bearish crossover: MACD crosses below signal
        df['bearish_crossover'] = (
            (df['macd_above_signal'] == 0) &
            (df['macd_above_signal_prev'] == 1)
        ).astype(int)

        # ========================================
        # 6. Histogram Momentum
        # ========================================
        # Histogram growing = strengthening trend
        df['histogram_change'] = df['macd_histogram'].diff()
        df['histogram_growing'] = (df['histogram_change'] > 0).astype(int)

        # Histogram acceleration
        df['histogram_accel'] = df['histogram_change'].diff()

        # ========================================
        # 7. Zero Line Crosses
        # ========================================
        # MACD above/below zero indicates overall trend
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)
        df['macd_above_zero_prev'] = df['macd_above_zero'].shift(1)

        df['macd_cross_above_zero'] = (
            (df['macd_above_zero'] == 1) &
            (df['macd_above_zero_prev'] == 0)
        ).astype(int)

        df['macd_cross_below_zero'] = (
            (df['macd_above_zero'] == 0) &
            (df['macd_above_zero_prev'] == 1)
        ).astype(int)

        # ========================================
        # 8. Price Momentum Confirmation
        # ========================================
        df['returns'] = df['close'].pct_change()
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # Price trend
        df['price_trend_20'] = df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )

        # ========================================
        # 9. Volatility Context
        # ========================================
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ========================================
        # 10. Signal Strength
        # ========================================
        # How strong is the crossover signal?
        df['signal_strength'] = (
            # Histogram size
            (np.abs(df['macd_histogram']) / df['close'] * 1000).clip(0, 1) * 0.3 +
            # Histogram momentum
            (df['histogram_growing']) * 0.2 +
            # MACD absolute value
            (np.abs(df['macd']) / df['close'] * 1000).clip(0, 1) * 0.2 +
            # Price momentum alignment
            ((df['macd'] * df['momentum_10'] > 0).astype(int)) * 0.3
        )

        # ========================================
        # 11. Trend Context (Higher Timeframe)
        # ========================================
        # Use 200-period SMA for overall trend
        df['sma_200'] = df['close'].rolling(200).mean()
        df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)

        # ========================================
        # 12. Volume Confirmation (if available)
        # ========================================
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_surge'] = (df['volume_ratio'] > 1.3).astype(int)
        else:
            df['volume_ratio'] = 1.0
            df['volume_surge'] = 0

        # Drop NaN
        df = df.dropna()

        print(f"[{self.metadata.hypothesis_id}] Features engineered: {len(df.columns)} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossovers.

        Signal Logic:
        - LONG (1): Bullish crossover + strong histogram + trend confirmation
        - SHORT (-1): Bearish crossover + strong histogram + trend confirmation
        - NEUTRAL (0): No crossover or weak signal

        Args:
            features: DataFrame with engineered features

        Returns:
            Series of trading signals (1=LONG, -1=SHORT, 0=NEUTRAL)
        """
        signals = pd.Series(0, index=features.index)

        # ========================================
        # LONG Conditions (Bullish Crossover)
        # ========================================
        long_conditions = (
            # Primary: Bullish crossover
            (features['bullish_crossover'] == 1) &

            # Confirmation 1: Histogram growing
            (features['histogram_growing'] == 1) &

            # Confirmation 2: Positive price momentum
            (features['momentum_10'] > -0.02) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.4) &

            # Optional: Above 200 SMA (uptrend)
            ((features['above_sma_200'] == 1) | (features['signal_strength'] > 0.6))
        )

        # ========================================
        # SHORT Conditions (Bearish Crossover)
        # ========================================
        short_conditions = (
            # Primary: Bearish crossover
            (features['bearish_crossover'] == 1) &

            # Confirmation 1: Histogram shrinking
            (features['histogram_growing'] == 0) &

            # Confirmation 2: Negative price momentum
            (features['momentum_10'] < 0.02) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.4) &

            # Optional: Below 200 SMA (downtrend)
            ((features['above_sma_200'] == 0) | (features['signal_strength'] > 0.6))
        )

        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # ========================================
        # Exit Logic (Opposite Crossover)
        # ========================================
        # Exit longs on bearish crossover or MACD crosses below zero
        exit_long = (
            (features['bearish_crossover'] == 1) |
            (features['macd_cross_below_zero'] == 1)
        )

        # Exit shorts on bullish crossover or MACD crosses above zero
        exit_short = (
            (features['bullish_crossover'] == 1) |
            (features['macd_cross_above_zero'] == 1)
        )

        signals = self._apply_exits(signals, exit_long, exit_short)

        print(f"[{self.metadata.hypothesis_id}] Generated {(signals != 0).sum()} signals "
              f"({(signals == 1).sum()} LONG, {(signals == -1).sum()} SHORT)")

        return signals

    def _apply_exits(
        self,
        signals: pd.Series,
        exit_long: pd.Series,
        exit_short: pd.Series
    ) -> pd.Series:
        """Apply exit logic to signals"""
        result = signals.copy()

        position = 0
        for i in range(len(result)):
            current_signal = result.iloc[i]

            if current_signal != 0:
                position = current_signal

            if position == 1 and exit_long.iloc[i]:
                result.iloc[i] = 0
                position = 0
            elif position == -1 and exit_short.iloc[i]:
                result.iloc[i] = 0
                position = 0

        return result


async def main():
    """Run H015 MACD Crossover test"""

    print("=" * 80)
    print("HYPOTHESIS H015: MACD Crossover Strategy")
    print("=" * 80)
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: MACD (12,26,9) crossover with histogram confirmation")
    print(f"Expected Sharpe: 0.6-0.9 if hypothesis holds")
    print()

    tester = MACDCrossoverTester()

    report = await tester.execute_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        
    )

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Decision: {report.decision}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"Win Rate: {report.win_rate:.1%}")
    print(f"Total Return: {report.total_return:.1%}")
    print(f"Max Drawdown: {report.max_drawdown:.1%}")
    print(f"Trade Count: {report.trade_count}")
    print()

    if report.decision == "SCALE":
        print("✅ SUCCESS! Strategy is profitable!")
    elif report.decision == "ITERATE":
        print("⚡ ITERATE: Strategy shows promise")
    else:
        print("❌ KILL: Strategy not profitable")

    return report


if __name__ == "__main__":
    asyncio.run(main())
