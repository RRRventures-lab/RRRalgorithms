from .strategy_base import StrategyBase
from datetime import datetime
import logging
import numpy as np
import pandas as pd

"""
Mean Reversion Strategy

A strategy that buys when price is oversold (below lower Bollinger Band)
and sells when price is overbought (above upper Bollinger Band).
"""


logger = logging.getLogger(__name__)


class MeanReversionStrategy(StrategyBase):
    """
    Mean reversion strategy using Bollinger Bands.

    Parameters:
    - period: Bollinger Band period (default: 20)
    - std_dev: Standard deviations for bands (default: 2.0)
    - risk_per_trade: Risk per trade as fraction (default: 0.02)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize Mean Reversion Strategy.

        Args:
            period: Moving average period for Bollinger Bands
            std_dev: Number of standard deviations for bands
            risk_per_trade: Risk per trade as fraction of capital
        """
        super().__init__(name="MeanReversion")

        self.set_parameter('period', period)
        self.set_parameter('std_dev', std_dev)
        self.set_parameter('risk_per_trade', risk_per_trade)

        # State variables
        self.prices = []
        self.middle_band = None
        self.upper_band = None
        self.lower_band = None

    def initialize(self, engine, symbol: str):
        """Initialize strategy."""
        super().initialize(engine, symbol)
        self.prices = []
        self.middle_band = None
        self.upper_band = None
        self.lower_band = None

    def on_bar(self, timestamp: datetime, bar: pd.Series):
        """
        Process new bar and generate trading signals.

        Args:
            timestamp: Bar timestamp
            bar: Bar data with OHLCV
        """
        close_price = bar['close']
        self.prices.append(close_price)

        period = self.get_parameter('period')
        std_dev = self.get_parameter('std_dev')

        # Need enough data for calculation
        if len(self.prices) < period:
            return

        # Calculate Bollinger Bands
        recent_prices = self.prices[-period:]
        self.middle_band = np.mean(recent_prices)
        std = np.std(recent_prices)
        self.upper_band = self.middle_band + (std_dev * std)
        self.lower_band = self.middle_band - (std_dev * std)

        # Generate signals
        if not self.has_position():
            # Look for oversold condition (buy signal)
            if close_price < self.lower_band:
                # Calculate signal strength based on distance from lower band
                distance = (self.lower_band - close_price) / self.middle_band
                signal_strength = min(1.0, distance * 20)  # Normalize

                logger.info(
                    f"{timestamp}: BUY signal (Price: {close_price:.2f}, "
                    f"Lower Band: {self.lower_band:.2f}, "
                    f"Strength: {signal_strength:.2f})"
                )

                self.buy(
                    timestamp=timestamp,
                    price=close_price,
                    signal_strength=signal_strength
                )

        else:
            # Look for exit conditions
            entry_price = self.get_position_entry_price()

            # Exit if price reaches middle band (mean reversion complete)
            if close_price >= self.middle_band:
                logger.info(
                    f"{timestamp}: SELL signal - Mean reversion (Price: {close_price:.2f}, "
                    f"Middle Band: {self.middle_band:.2f}, "
                    f"Entry: {entry_price:.2f})"
                )

                self.sell(
                    timestamp=timestamp,
                    price=close_price
                )

            # Stop loss if price continues below lower band
            elif close_price < self.lower_band * 0.98:  # 2% below lower band
                logger.info(
                    f"{timestamp}: SELL signal - Stop loss (Price: {close_price:.2f}, "
                    f"Lower Band: {self.lower_band:.2f})"
                )

                self.sell(
                    timestamp=timestamp,
                    price=close_price
                )

            # Take profit if price reaches upper band
            elif close_price > self.upper_band:
                logger.info(
                    f"{timestamp}: SELL signal - Take profit (Price: {close_price:.2f}, "
                    f"Upper Band: {self.upper_band:.2f})"
                )

                self.sell(
                    timestamp=timestamp,
                    price=close_price
                )
