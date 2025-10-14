from .strategy_base import StrategyBase
from datetime import datetime
import logging
import numpy as np
import pandas as pd

"""
Simple Momentum Strategy

A basic momentum strategy that buys when price crosses above moving average
and sells when it crosses below.
"""


logger = logging.getLogger(__name__)


class SimpleMomentumStrategy(StrategyBase):
    """
    Simple momentum strategy using moving average crossover.

    Parameters:
    - fast_period: Fast MA period (default: 20)
    - slow_period: Slow MA period (default: 50)
    - risk_per_trade: Risk per trade as fraction (default: 0.02)
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize Simple Momentum Strategy.

        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            risk_per_trade: Risk per trade as fraction of capital
        """
        super().__init__(name="SimpleMomentum")

        self.set_parameter('fast_period', fast_period)
        self.set_parameter('slow_period', slow_period)
        self.set_parameter('risk_per_trade', risk_per_trade)

        # State variables
        self.prices = []
        self.fast_ma = None
        self.slow_ma = None
        self.previous_signal = None

    def initialize(self, engine, symbol: str):
        """Initialize strategy."""
        super().initialize(engine, symbol)
        self.prices = []
        self.fast_ma = None
        self.slow_ma = None
        self.previous_signal = None

    def on_bar(self, timestamp: datetime, bar: pd.Series):
        """
        Process new bar and generate trading signals.

        Args:
            timestamp: Bar timestamp
            bar: Bar data with OHLCV
        """
        close_price = bar['close']
        self.prices.append(close_price)

        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')

        # Need enough data for slow MA
        if len(self.prices) < slow_period:
            return

        # Calculate moving averages
        self.fast_ma = np.mean(self.prices[-fast_period:])
        self.slow_ma = np.mean(self.prices[-slow_period:])

        # Generate signal
        current_signal = 'buy' if self.fast_ma > self.slow_ma else 'sell'

        # Execute on signal change
        if current_signal != self.previous_signal:
            if current_signal == 'buy' and not self.has_position():
                # Buy signal - open long position
                signal_strength = abs(self.fast_ma - self.slow_ma) / self.slow_ma
                signal_strength = min(1.0, signal_strength * 10)  # Normalize

                logger.info(
                    f"{timestamp}: BUY signal (Fast MA: {self.fast_ma:.2f}, "
                    f"Slow MA: {self.slow_ma:.2f}, Strength: {signal_strength:.2f})"
                )

                self.buy(
                    timestamp=timestamp,
                    price=close_price,
                    signal_strength=signal_strength
                )

            elif current_signal == 'sell' and self.has_position():
                # Sell signal - close position
                logger.info(
                    f"{timestamp}: SELL signal (Fast MA: {self.fast_ma:.2f}, "
                    f"Slow MA: {self.slow_ma:.2f})"
                )

                self.sell(
                    timestamp=timestamp,
                    price=close_price
                )

        self.previous_signal = current_signal
