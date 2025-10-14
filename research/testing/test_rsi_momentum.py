from datetime import datetime, timedelta
from pathlib import Path
from research.testing.hypothesis_tester import HypothesisTester
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import numpy as np
import pandas as pd
import sys

"""
H014: RSI Momentum Strategy
============================

Classical technical analysis strategy using Relative Strength Index (RSI).

Strategy Logic:
- LONG when RSI < 30 (oversold) + rising momentum
- SHORT when RSI > 70 (overbought) + falling momentum
- Exit when RSI returns to neutral zone (40-60)

RSI Parameters:
- Period: 14 (standard)
- Overbought: 70
- Oversold: 30

Expected Performance:
- Sharpe Ratio: 0.8-1.0
- Win Rate: 50-60%
- Trade Frequency: 5-10%

Reference: J. Welles Wilder (1978) - "New Concepts in Technical Trading Systems"

Author: RRR Ventures
Date: 2025-10-12
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class RSIMomentumTester(HypothesisTester):
    """
    Test RSI momentum strategy with real data.

    Uses 14-period RSI to identify overbought/oversold conditions
    combined with price momentum confirmation.
    """

    def __init__(self):
        super().__init__(
            hypothesis_id="H014",
            title="RSI Momentum Strategy (14-period)",
            category="momentum",
            priority_score=90
        )

        self.collector = ProfessionalDataCollector()

        # Strategy parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_neutral_low = 40
        self.rsi_neutral_high = 60

        # Primary feature for statistical validation
        self.primary_feature = 'rsi'

    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect historical price data for RSI calculation.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        print(f"[{self.metadata.hypothesis_id}] Collecting BTC data from database...")

        # Get data from Polygon (cached)
        price_data = await self.collector.polygon.get_crypto_aggregates(
            "X:BTCUSD",
            start_date,
            end_date,
            "hour",
            1
        )

        if price_data.empty:
            raise ValueError("No price data collected")

        # Ensure required columns
        df = price_data.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"[{self.metadata.hypothesis_id}] Data ready: {len(df)} rows")
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and related features.

        Features:
        1. RSI (14-period)
        2. RSI momentum (rate of change)
        3. Price momentum
        4. RSI divergence (price vs RSI)
        5. Volatility adjustment

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # ========================================
        # 1. Calculate RSI (Wilder's method)
        # ========================================
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's smoothing (exponential moving average)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ========================================
        # 2. RSI Momentum (rate of change)
        # ========================================
        df['rsi_momentum'] = df['rsi'].diff(5)  # 5-period change
        df['rsi_acceleration'] = df['rsi_momentum'].diff(3)  # Second derivative

        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < self.rsi_oversold).astype(int)
        df['rsi_overbought'] = (df['rsi'] > self.rsi_overbought).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= self.rsi_neutral_low) &
                             (df['rsi'] <= self.rsi_neutral_high)).astype(int)

        # ========================================
        # 3. Price Momentum
        # ========================================
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # Momentum confirmation
        df['price_rising'] = (df['momentum_5'] > 0).astype(int)
        df['price_falling'] = (df['momentum_5'] < 0).astype(int)

        # ========================================
        # 4. RSI Divergence
        # ========================================
        # Bullish divergence: Price falling but RSI rising
        df['price_trend_20'] = df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )
        df['rsi_trend_20'] = df['rsi'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )

        df['bullish_divergence'] = (
            (df['price_trend_20'] < 0) &
            (df['rsi_trend_20'] > 0) &
            (df['rsi'] < 40)
        ).astype(int)

        df['bearish_divergence'] = (
            (df['price_trend_20'] > 0) &
            (df['rsi_trend_20'] < 0) &
            (df['rsi'] > 60)
        ).astype(int)

        # ========================================
        # 5. Volatility Adjustment
        # ========================================
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # High volatility = wider RSI thresholds
        df['rsi_oversold_adj'] = self.rsi_oversold + (df['volatility_percentile'] - 0.5) * 10
        df['rsi_overbought_adj'] = self.rsi_overbought + (df['volatility_percentile'] - 0.5) * 10

        # ========================================
        # 6. Signal Quality Metrics
        # ========================================
        # How long has RSI been in extreme zone?
        df['rsi_extreme_duration'] = 0
        in_extreme = (df['rsi'] < 35) | (df['rsi'] > 65)
        df.loc[in_extreme, 'rsi_extreme_duration'] = (
            in_extreme.groupby((~in_extreme).cumsum()).cumcount() + 1
        )

        # Volume confirmation (if available)
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
        else:
            df['volume_ratio'] = 1.0
            df['volume_surge'] = 0

        # ========================================
        # 7. Signal Strength Score
        # ========================================
        # Composite signal quality (0-1 scale)
        df['signal_strength'] = (
            (np.abs(df['rsi'] - 50) / 50) * 0.3 +  # Distance from neutral
            (np.abs(df['rsi_momentum']) / 10) * 0.2 +  # RSI momentum
            (df['bullish_divergence'] | df['bearish_divergence']) * 0.2 +  # Divergence
            (df['volume_surge']) * 0.15 +  # Volume confirmation
            (df['rsi_extreme_duration'] / 10).clip(0, 1) * 0.15  # Duration in extreme
        )

        # Drop NaN from rolling calculations
        df = df.dropna()

        print(f"[{self.metadata.hypothesis_id}] Features engineered: {len(df.columns)} columns, {len(df)} rows")
        return df

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI.

        Signal Logic:
        - LONG (1): RSI oversold + rising momentum + high signal strength
        - SHORT (-1): RSI overbought + falling momentum + high signal strength
        - NEUTRAL (0): RSI in neutral zone or low signal strength

        Args:
            features: DataFrame with engineered features

        Returns:
            Series of trading signals (1=LONG, -1=SHORT, 0=NEUTRAL)
        """
        signals = pd.Series(0, index=features.index)

        # ========================================
        # LONG Conditions (Buy oversold)
        # ========================================
        long_conditions = (
            # Primary: RSI oversold
            (features['rsi'] < features['rsi_oversold_adj']) &

            # Confirmation 1: RSI starting to rise
            (features['rsi_momentum'] > 0) &

            # Confirmation 2: Not in prolonged downtrend
            (features['momentum_20'] > -0.1) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.4) &

            # Optional: Bullish divergence bonus
            ((features['bullish_divergence'] == 1) | (features['signal_strength'] > 0.5))
        )

        # ========================================
        # SHORT Conditions (Sell overbought)
        # ========================================
        short_conditions = (
            # Primary: RSI overbought
            (features['rsi'] > features['rsi_overbought_adj']) &

            # Confirmation 1: RSI starting to fall
            (features['rsi_momentum'] < 0) &

            # Confirmation 2: Not in prolonged uptrend
            (features['momentum_20'] < 0.1) &

            # Quality: High signal strength
            (features['signal_strength'] > 0.4) &

            # Optional: Bearish divergence bonus
            ((features['bearish_divergence'] == 1) | (features['signal_strength'] > 0.5))
        )

        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # ========================================
        # Exit Logic (return to neutral)
        # ========================================
        # Exit longs when RSI > 60 or momentum reverses
        exit_long = (features['rsi'] > 60) | (features['rsi_momentum'] < -2)

        # Exit shorts when RSI < 40 or momentum reverses
        exit_short = (features['rsi'] < 40) | (features['rsi_momentum'] > 2)

        # Apply exits (force to neutral)
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

        # Track current position
        position = 0
        for i in range(len(result)):
            current_signal = result.iloc[i]

            # Update position
            if current_signal != 0:
                position = current_signal

            # Check exits
            if position == 1 and exit_long.iloc[i]:
                result.iloc[i] = 0
                position = 0
            elif position == -1 and exit_short.iloc[i]:
                result.iloc[i] = 0
                position = 0

        return result


async def main():
    """Run H014 RSI Momentum test"""

    print("=" * 80)
    print("HYPOTHESIS H014: RSI Momentum Strategy")
    print("=" * 80)
    print()

    # Test period: 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy: RSI (14-period) momentum with oversold/overbought signals")
    print(f"Expected Sharpe: 0.8-1.0 if hypothesis holds")
    print()

    # Create tester
    tester = RSIMomentumTester()

    # Run full pipeline
    report = await tester.execute_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        
    )

    # Display results
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
    print(f"Avg Trade Duration: {report.avg_trade_duration_hours:.1f} hours")
    print()

    if report.decision == "SCALE":
        print("✅ SUCCESS! First profitable strategy found!")
    elif report.decision == "ITERATE":
        print("⚡ ITERATE: Strategy shows promise, needs refinement")
    else:
        print("❌ KILL: Strategy not profitable")

    return report


if __name__ == "__main__":
    asyncio.run(main())
