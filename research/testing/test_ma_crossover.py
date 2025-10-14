from datetime import datetime, timedelta
from pathlib import Path
from research.testing.hypothesis_tester import HypothesisTester
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import numpy as np
import pandas as pd
import sys

"""
H017: Moving Average Crossover Strategy
========================================

Classical trend-following strategy using moving average crossovers.

Strategy Logic:
- LONG when fast MA crosses above slow MA (Golden Cross)
- SHORT when fast MA crosses below slow MA (Death Cross)
- Hold until opposite crossover

MA Parameters:
- Fast MA: 50-period SMA
- Slow MA: 200-period SMA
- Classic "Golden Cross / Death Cross" setup

Expected Performance:
- Sharpe Ratio: 0.5-0.8
- Win Rate: 40-50%
- Trade Frequency: 1-3% (low frequency, trend-following)

Reference: Richard Donchian (1960s) - Moving Average Systems

Author: RRR Ventures
Date: 2025-10-12
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class MACrossoverTester(HypothesisTester):
    """
    Test Moving Average crossover strategy (50/200).

    Classic trend-following system using Golden Cross and Death Cross.
    """

    def __init__(self):
        super().__init__(
            hypothesis_id="H017",
            title="MA Crossover Strategy (50/200 Golden Cross)",
            category="trend_following",
            priority_score=80
        )

        self.collector = ProfessionalDataCollector()

        # Strategy parameters
        self.fast_period = 50
        self.slow_period = 200

        # Primary feature
        self.primary_feature = 'ma_diff'

    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect historical price data"""
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
        Calculate moving averages and trend features.

        Features:
        1. Fast MA (50-period SMA)
        2. Slow MA (200-period SMA)
        3. MA crossovers
        4. Trend strength
        5. Momentum alignment

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # ========================================
        # 1. Moving Averages
        # ========================================
        df['ma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_period).mean()

        # MA difference (normalized)
        df['ma_diff'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']

        # ========================================
        # 2. Crossover Detection
        # ========================================
        df['fast_above_slow'] = (df['ma_fast'] > df['ma_slow']).astype(int)
        df['fast_above_slow_prev'] = df['fast_above_slow'].shift(1)

        # Golden Cross: Fast crosses above slow (bullish)
        df['golden_cross'] = (
            (df['fast_above_slow'] == 1) &
            (df['fast_above_slow_prev'] == 0)
        ).astype(int)

        # Death Cross: Fast crosses below slow (bearish)
        df['death_cross'] = (
            (df['fast_above_slow'] == 0) &
            (df['fast_above_slow_prev'] == 1)
        ).astype(int)

        # ========================================
        # 3. MA Slope (Trend Direction)
        # ========================================
        df['ma_fast_slope'] = df['ma_fast'].diff(10) / df['ma_fast']
        df['ma_slow_slope'] = df['ma_slow'].diff(10) / df['ma_slow']

        # Both MAs rising = strong uptrend
        df['both_mas_rising'] = (
            (df['ma_fast_slope'] > 0) &
            (df['ma_slow_slope'] > 0)
        ).astype(int)

        # Both MAs falling = strong downtrend
        df['both_mas_falling'] = (
            (df['ma_fast_slope'] < 0) &
            (df['ma_slow_slope'] < 0)
        ).astype(int)

        # ========================================
        # 4. Price Position
        # ========================================
        # Price above both MAs = very bullish
        df['price_above_both'] = (
            (df['close'] > df['ma_fast']) &
            (df['close'] > df['ma_slow'])
        ).astype(int)

        # Price below both MAs = very bearish
        df['price_below_both'] = (
            (df['close'] < df['ma_fast']) &
            (df['close'] < df['ma_slow'])
        ).astype(int)

        # Distance from fast MA (pullback opportunity)
        df['distance_from_fast'] = (df['close'] - df['ma_fast']) / df['ma_fast']

        # ========================================
        # 5. Trend Strength
        # ========================================
        # MA separation (wider = stronger trend)
        df['ma_separation'] = np.abs(df['ma_diff'])

        # MA separation percentile
        df['ma_separation_percentile'] = df['ma_separation'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Strong trend when MAs well separated
        df['strong_trend'] = (df['ma_separation_percentile'] > 0.6).astype(int)

        # ========================================
        # 6. Momentum Confirmation
        # ========================================
        df['returns'] = df['close'].pct_change()
        df['momentum_20'] = df['close'].pct_change(20)
        df['momentum_50'] = df['close'].pct_change(50)

        # Price momentum aligned with MA signal
        df['momentum_aligned_bullish'] = (
            (df['fast_above_slow'] == 1) &
            (df['momentum_20'] > 0)
        ).astype(int)

        df['momentum_aligned_bearish'] = (
            (df['fast_above_slow'] == 0) &
            (df['momentum_20'] < 0)
        ).astype(int)

        # ========================================
        # 7. Volatility Context
        # ========================================
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ========================================
        # 8. Crossover Quality Score
        # ========================================
        # How strong is the crossover signal?
        df['crossover_quality'] = (
            # MA separation (stronger trend)
            (df['ma_separation_percentile']) * 0.30 +
            # Both MAs aligned with trend
            ((df['both_mas_rising'] | df['both_mas_falling'])) * 0.25 +
            # Momentum confirmation
            ((df['momentum_aligned_bullish'] | df['momentum_aligned_bearish'])) * 0.25 +
            # Low volatility (cleaner signal)
            ((1 - df['volatility_percentile'])) * 0.20
        )

        # ========================================
        # 9. Additional Indicators
        # ========================================
        # RSI for confirmation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume
        if 'volume' in df.columns:
            df['volume_ma_50'] = df['volume'].rolling(50).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_50']
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
        Generate trading signals based on MA crossovers.

        Signal Logic:
        - LONG (1): Golden Cross + trend confirmation
        - SHORT (-1): Death Cross + trend confirmation
        - NEUTRAL (0): No crossover

        Args:
            features: DataFrame with engineered features

        Returns:
            Series of trading signals
        """
        signals = pd.Series(0, index=features.index)

        # ========================================
        # LONG Conditions (Golden Cross)
        # ========================================
        long_conditions = (
            # Primary: Golden Cross
            (features['golden_cross'] == 1) &

            # Confirmation 1: Both MAs rising or strong trend
            ((features['both_mas_rising'] == 1) | (features['strong_trend'] == 1)) &

            # Confirmation 2: Momentum aligned
            (features['momentum_20'] > -0.05) &

            # Quality: High crossover quality
            (features['crossover_quality'] > 0.4)
        )

        # ========================================
        # SHORT Conditions (Death Cross)
        # ========================================
        short_conditions = (
            # Primary: Death Cross
            (features['death_cross'] == 1) &

            # Confirmation 1: Both MAs falling or strong trend
            ((features['both_mas_falling'] == 1) | (features['strong_trend'] == 1)) &

            # Confirmation 2: Momentum aligned
            (features['momentum_20'] < 0.05) &

            # Quality: High crossover quality
            (features['crossover_quality'] > 0.4)
        )

        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # ========================================
        # Exit Logic (Opposite Crossover)
        # ========================================
        # Exit longs on death cross or trend weakness
        exit_long = (
            (features['death_cross'] == 1) |
            ((features['fast_above_slow'] == 0) & (features['momentum_50'] < -0.1))
        )

        # Exit shorts on golden cross or trend weakness
        exit_short = (
            (features['golden_cross'] == 1) |
            ((features['fast_above_slow'] == 1) & (features['momentum_50'] > 0.1))
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
    """Run H017 MA Crossover test"""

    print("=" * 80)
    print("HYPOTHESIS H017: Moving Average Crossover Strategy")
    print("=" * 80)
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: 50/200 MA crossover (Golden/Death Cross)")
    print(f"Expected Sharpe: 0.5-0.8 if hypothesis holds")
    print()

    tester = MACrossoverTester()

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
