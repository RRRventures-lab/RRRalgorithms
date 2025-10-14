from datetime import datetime, timedelta
from pathlib import Path
from research.testing.hypothesis_tester import HypothesisTester
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import numpy as np
import pandas as pd
import sys

"""
H016: Bollinger Bands Mean Reversion Strategy
==============================================

Classical mean reversion strategy using Bollinger Bands.

Strategy Logic:
- LONG when price touches/breaks below lower band (oversold)
- SHORT when price touches/breaks above upper band (overbought)
- Exit when price returns to middle band

Bollinger Bands Parameters:
- Period: 20 (standard)
- Standard Deviation: 2.0 (covers ~95% of price action)
- Basis: Simple Moving Average

Expected Performance:
- Sharpe Ratio: 0.7-1.1
- Win Rate: 50-60%
- Trade Frequency: 8-15%

Reference: John Bollinger (1983) - "Bollinger on Bollinger Bands"

Author: RRR Ventures
Date: 2025-10-12
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class BollingerMeanReversionTester(HypothesisTester):
    """
    Test Bollinger Bands mean reversion strategy.

    Uses 20-period SMA with 2 standard deviations to identify
    extreme price movements that are likely to revert.
    """

    def __init__(self):
        super().__init__(
            hypothesis_id="H016",
            title="Bollinger Bands Mean Reversion (20, 2σ)",
            category="mean_reversion",
            priority_score=88
        )

        self.collector = ProfessionalDataCollector()

        # Strategy parameters
        self.bb_period = 20
        self.bb_std = 2.0

        # Primary feature
        self.primary_feature = 'bb_position'

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
        Calculate Bollinger Bands and mean reversion features.

        Features:
        1. Middle band (20 SMA)
        2. Upper/Lower bands (±2σ)
        3. Band width (volatility measure)
        4. Price position within bands
        5. Mean reversion signals

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # ========================================
        # 1. Bollinger Bands
        # ========================================
        # Middle band (SMA)
        df['bb_middle'] = df['close'].rolling(self.bb_period).mean()

        # Standard deviation
        df['bb_std'] = df['close'].rolling(self.bb_period).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])

        # ========================================
        # 2. Price Position (Bollinger %B)
        # ========================================
        # %B = (close - lower) / (upper - lower)
        # %B = 0: at lower band
        # %B = 0.5: at middle band
        # %B = 1: at upper band
        df['bb_position'] = (
            (df['close'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower'])
        )

        # ========================================
        # 3. Band Width (Volatility Indicator)
        # ========================================
        # Width = (upper - lower) / middle
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Width percentile (relative volatility)
        df['bb_width_percentile'] = df['bb_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Band squeeze (low volatility)
        df['bb_squeeze'] = (df['bb_width_percentile'] < 0.2).astype(int)

        # Band expansion (high volatility)
        df['bb_expansion'] = (df['bb_width_percentile'] > 0.8).astype(int)

        # ========================================
        # 4. Band Touches
        # ========================================
        # Price touching/breaking bands
        df['touch_upper'] = (df['high'] >= df['bb_upper']).astype(int)
        df['touch_lower'] = (df['low'] <= df['bb_lower']).astype(int)

        # Break above/below (close outside bands)
        df['break_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['break_lower'] = (df['close'] < df['bb_lower']).astype(int)

        # ========================================
        # 5. Mean Reversion Signals
        # ========================================
        # Distance from middle (in std devs)
        df['distance_from_middle'] = (df['close'] - df['bb_middle']) / df['bb_std']

        # Is price reverting? (moving back toward middle)
        df['reverting_from_upper'] = (
            (df['bb_position'].shift(1) > 0.9) &
            (df['bb_position'] < df['bb_position'].shift(1))
        ).astype(int)

        df['reverting_from_lower'] = (
            (df['bb_position'].shift(1) < 0.1) &
            (df['bb_position'] > df['bb_position'].shift(1))
        ).astype(int)

        # ========================================
        # 6. Momentum Context
        # ========================================
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # RSI for additional confirmation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ========================================
        # 7. Volume Confirmation
        # ========================================
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
        else:
            df['volume_ratio'] = 1.0
            df['volume_surge'] = 0

        # ========================================
        # 8. Signal Strength Score
        # ========================================
        # How strong is the mean reversion signal?
        df['signal_strength'] = (
            # Distance from bands
            (np.abs(df['bb_position'] - 0.5) * 2).clip(0, 1) * 0.35 +
            # Band width (prefer squeezes)
            ((1 - df['bb_width_percentile'])) * 0.20 +
            # RSI confirmation
            ((df['rsi'] < 30).astype(int) | (df['rsi'] > 70).astype(int)) * 0.25 +
            # Volume surge
            (df['volume_surge']) * 0.20
        )

        # ========================================
        # 9. Trend Filter
        # ========================================
        # Use 200 SMA for overall trend
        df['sma_200'] = df['close'].rolling(200).mean()
        df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)

        # Middle band slope (band trend)
        df['bb_middle_slope'] = df['bb_middle'].diff(5) / df['bb_middle']

        # Drop NaN
        df = df.dropna()

        print(f"[{self.metadata.hypothesis_id}] Features engineered: {len(df.columns)} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands.

        Signal Logic:
        - LONG (1): Price at/below lower band + reverting
        - SHORT (-1): Price at/above upper band + reverting
        - NEUTRAL (0): Price within bands

        Args:
            features: DataFrame with engineered features

        Returns:
            Series of trading signals
        """
        signals = pd.Series(0, index=features.index)

        # ========================================
        # LONG Conditions (Buy at lower band)
        # ========================================
        long_conditions = (
            # Primary: Price at or below lower band
            (features['bb_position'] < 0.1) &

            # Confirmation 1: RSI oversold
            (features['rsi'] < 35) &

            # Confirmation 2: Not in strong downtrend
            (features['bb_middle_slope'] > -0.01) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.5) &

            # Optional: Above 200 SMA or high quality
            ((features['above_sma_200'] == 1) | (features['signal_strength'] > 0.65))
        )

        # ========================================
        # SHORT Conditions (Sell at upper band)
        # ========================================
        short_conditions = (
            # Primary: Price at or above upper band
            (features['bb_position'] > 0.9) &

            # Confirmation 1: RSI overbought
            (features['rsi'] > 65) &

            # Confirmation 2: Not in strong uptrend
            (features['bb_middle_slope'] < 0.01) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.5) &

            # Optional: Below 200 SMA or high quality
            ((features['above_sma_200'] == 0) | (features['signal_strength'] > 0.65))
        )

        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # ========================================
        # Exit Logic (Return to middle band)
        # ========================================
        # Exit longs when price reaches middle or upper band
        exit_long = (
            (features['bb_position'] > 0.5) |  # Reached middle
            (features['reverting_from_upper'] == 1)  # Rejection from upper
        )

        # Exit shorts when price reaches middle or lower band
        exit_short = (
            (features['bb_position'] < 0.5) |  # Reached middle
            (features['reverting_from_lower'] == 1)  # Rejection from lower
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
    """Run H016 Bollinger Bands Mean Reversion test"""

    print("=" * 80)
    print("HYPOTHESIS H016: Bollinger Bands Mean Reversion Strategy")
    print("=" * 80)
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: Bollinger Bands (20, 2σ) mean reversion")
    print(f"Expected Sharpe: 0.7-1.1 if hypothesis holds")
    print()

    tester = BollingerMeanReversionTester()

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
