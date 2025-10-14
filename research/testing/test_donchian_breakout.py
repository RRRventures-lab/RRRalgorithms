from datetime import datetime, timedelta
from pathlib import Path
from research.testing.hypothesis_tester import HypothesisTester
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import numpy as np
import pandas as pd
import sys

"""
H018: Donchian Channel Breakout Strategy
=========================================

Classical breakout strategy using Donchian Channels.

Strategy Logic:
- LONG when price breaks above upper channel (20-period high)
- SHORT when price breaks below lower channel (20-period low)
- Exit on opposite breakout or middle channel touch

Donchian Channel Parameters:
- Period: 20 (standard)
- Upper channel: Highest high of last 20 periods
- Lower channel: Lowest low of last 20 periods
- Middle channel: Average of upper and lower

Expected Performance:
- Sharpe Ratio: 0.9-1.3
- Win Rate: 40-55%
- Trade Frequency: 5-12%

Reference: Richard Donchian (1960s) - "Turtle Trading System"

Author: RRR Ventures
Date: 2025-10-12
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class DonchianBreakoutTester(HypothesisTester):
    """
    Test Donchian Channel breakout strategy.

    Uses 20-period high/low channels to identify breakouts
    that signal strong trends.
    """

    def __init__(self):
        super().__init__(
            hypothesis_id="H018",
            title="Donchian Channel Breakout (20-period)",
            category="breakout",
            priority_score=92
        )

        self.collector = ProfessionalDataCollector()

        # Strategy parameters
        self.channel_period = 20
        self.exit_period = 10  # Shorter period for exits

        # Primary feature
        self.primary_feature = 'channel_position'

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
        Calculate Donchian Channels and breakout features.

        Features:
        1. Upper/Lower/Middle channels
        2. Breakout detection
        3. Channel width (volatility)
        4. Price position
        5. Momentum confirmation

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # ========================================
        # 1. Donchian Channels
        # ========================================
        # Upper channel (20-period high)
        df['channel_upper'] = df['high'].rolling(self.channel_period).max()

        # Lower channel (20-period low)
        df['channel_lower'] = df['low'].rolling(self.channel_period).min()

        # Middle channel (average)
        df['channel_middle'] = (df['channel_upper'] + df['channel_lower']) / 2

        # Shorter exit channels
        df['exit_channel_upper'] = df['high'].rolling(self.exit_period).max()
        df['exit_channel_lower'] = df['low'].rolling(self.exit_period).min()

        # ========================================
        # 2. Price Position in Channel
        # ========================================
        # Position = (close - lower) / (upper - lower)
        df['channel_position'] = (
            (df['close'] - df['channel_lower']) /
            (df['channel_upper'] - df['channel_lower'])
        )

        # ========================================
        # 3. Breakout Detection
        # ========================================
        # Upper breakout: Close above upper channel
        df['breakout_upper'] = (df['close'] > df['channel_upper'].shift(1)).astype(int)

        # Lower breakout: Close below lower channel
        df['breakout_lower'] = (df['close'] < df['channel_lower'].shift(1)).astype(int)

        # Strong breakout: High/low also breaks channel
        df['strong_breakout_upper'] = (
            (df['breakout_upper'] == 1) &
            (df['high'] > df['channel_upper'].shift(1))
        ).astype(int)

        df['strong_breakout_lower'] = (
            (df['breakout_lower'] == 1) &
            (df['low'] < df['channel_lower'].shift(1))
        ).astype(int)

        # ========================================
        # 4. Channel Width (Volatility Measure)
        # ========================================
        df['channel_width'] = (
            (df['channel_upper'] - df['channel_lower']) /
            df['channel_middle']
        )

        # Channel width percentile
        df['channel_width_percentile'] = df['channel_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Channel squeeze (consolidation)
        df['channel_squeeze'] = (df['channel_width_percentile'] < 0.3).astype(int)

        # Channel expansion (high volatility)
        df['channel_expansion'] = (df['channel_width_percentile'] > 0.7).astype(int)

        # ========================================
        # 5. Breakout Strength
        # ========================================
        # How far beyond channel did price break?
        df['breakout_distance_upper'] = (
            (df['close'] - df['channel_upper'].shift(1)) /
            df['channel_upper'].shift(1)
        ).clip(lower=0)

        df['breakout_distance_lower'] = (
            (df['channel_lower'].shift(1) - df['close']) /
            df['channel_lower'].shift(1)
        ).clip(lower=0)

        # ========================================
        # 6. Momentum Confirmation
        # ========================================
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # Acceleration
        df['momentum_accel'] = df['momentum_5'].diff(3)

        # ========================================
        # 7. Trend Context
        # ========================================
        # Use 50 SMA for trend filter
        df['sma_50'] = df['close'].rolling(50).mean()
        df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)

        # Channel middle slope
        df['channel_slope'] = df['channel_middle'].diff(5) / df['channel_middle']

        # ========================================
        # 8. Volatility Context
        # ========================================
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ========================================
        # 9. Consolidation Duration
        # ========================================
        # How long was price in channel before breakout?
        df['bars_since_upper'] = 0
        df['bars_since_lower'] = 0

        for i in range(1, len(df)):
            if df['channel_position'].iloc[i] > 0.95:
                df.iloc[i, df.columns.get_loc('bars_since_upper')] = 0
            else:
                df.iloc[i, df.columns.get_loc('bars_since_upper')] = (
                    df['bars_since_upper'].iloc[i-1] + 1
                )

            if df['channel_position'].iloc[i] < 0.05:
                df.iloc[i, df.columns.get_loc('bars_since_lower')] = 0
            else:
                df.iloc[i, df.columns.get_loc('bars_since_lower')] = (
                    df['bars_since_lower'].iloc[i-1] + 1
                )

        # Consolidation = been away from channels for a while
        df['consolidation'] = (
            (df['bars_since_upper'] > 5) &
            (df['bars_since_lower'] > 5)
        ).astype(int)

        # ========================================
        # 10. Volume Confirmation
        # ========================================
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)

            # Volume increasing = strong breakout
            df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume_ma_20']
        else:
            df['volume_ratio'] = 1.0
            df['volume_surge'] = 0
            df['volume_trend'] = 1.0

        # ========================================
        # 11. Breakout Quality Score
        # ========================================
        df['breakout_quality'] = (
            # Breakout distance (stronger = better)
            ((df['breakout_distance_upper'] + df['breakout_distance_lower']) * 100).clip(0, 1) * 0.25 +
            # Consolidation before breakout
            (df['consolidation']) * 0.20 +
            # Momentum acceleration
            ((df['momentum_accel'] > 0).astype(int)) * 0.20 +
            # Volume surge
            (df['volume_surge']) * 0.20 +
            # Low volatility (cleaner breakout)
            ((1 - df['volatility_percentile'])) * 0.15
        )

        # ========================================
        # 12. Additional Indicators
        # ========================================
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()

        # Drop NaN
        df = df.dropna()

        print(f"[{self.metadata.hypothesis_id}] Features engineered: {len(df.columns)} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Donchian breakouts.

        Signal Logic:
        - LONG (1): Upper channel breakout + confirmation
        - SHORT (-1): Lower channel breakout + confirmation
        - NEUTRAL (0): No breakout

        Args:
            features: DataFrame with engineered features

        Returns:
            Series of trading signals
        """
        signals = pd.Series(0, index=features.index)

        # ========================================
        # LONG Conditions (Upper Breakout)
        # ========================================
        long_conditions = (
            # Primary: Upper channel breakout
            (features['breakout_upper'] == 1) &

            # Confirmation 1: Positive momentum
            (features['momentum_5'] > 0) &

            # Confirmation 2: Above 50 SMA or strong quality
            ((features['above_sma_50'] == 1) | (features['breakout_quality'] > 0.6)) &

            # Quality: High breakout quality
            (features['breakout_quality'] > 0.4) &

            # Optional: RSI not extremely overbought
            (features['rsi'] < 80)
        )

        # ========================================
        # SHORT Conditions (Lower Breakout)
        # ========================================
        short_conditions = (
            # Primary: Lower channel breakout
            (features['breakout_lower'] == 1) &

            # Confirmation 1: Negative momentum
            (features['momentum_5'] < 0) &

            # Confirmation 2: Below 50 SMA or strong quality
            ((features['above_sma_50'] == 0) | (features['breakout_quality'] > 0.6)) &

            # Quality: High breakout quality
            (features['breakout_quality'] > 0.4) &

            # Optional: RSI not extremely oversold
            (features['rsi'] > 20)
        )

        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # ========================================
        # Exit Logic (Opposite Breakout or Exit Channel)
        # ========================================
        # Exit longs on lower breakout or exit channel
        exit_long = (
            (features['breakout_lower'] == 1) |
            (features['close'] < features['exit_channel_lower'])
        )

        # Exit shorts on upper breakout or exit channel
        exit_short = (
            (features['breakout_upper'] == 1) |
            (features['close'] > features['exit_channel_upper'])
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
    """Run H018 Donchian Breakout test"""

    print("=" * 80)
    print("HYPOTHESIS H018: Donchian Channel Breakout Strategy")
    print("=" * 80)
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: Donchian (20-period) breakout with confirmation")
    print(f"Expected Sharpe: 0.9-1.3 if hypothesis holds")
    print()

    tester = DonchianBreakoutTester()

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
