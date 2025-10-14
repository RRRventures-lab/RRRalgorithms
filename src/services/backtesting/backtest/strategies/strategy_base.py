from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Optional, Dict, Any
import logging
import pandas as pd


"""
Base Strategy Class

Abstract base class for all trading strategies.
"""


logger = logging.getLogger(__name__)


class StrategyBase(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - on_bar(): Process a new bar of data
    - calculate_position_size(): Determine position size
    """

    def __init__(self, name: str = "Strategy"):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.engine = None
        self.symbol = None
        self.parameters: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        logger.info(f"Initialized {self.name}")

    def initialize(self, engine, symbol: str):
        """
        Initialize strategy with backtesting engine.

        Args:
            engine: BacktestEngine instance
            symbol: Trading symbol
        """
        self.engine = engine
        self.symbol = symbol
        self.state = {}
        logger.info(f"{self.name} initialized for {symbol}")

    @abstractmethod
    def on_bar(self, timestamp: datetime, bar: pd.Series):
        """
        Process a new bar of data.

        Args:
            timestamp: Bar timestamp
            bar: Bar data (open, high, low, close, volume)
        """
        pass

    def calculate_position_size(
        self,
        signal_strength: float = 1.0,
        current_price: float = None,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk management rules.

        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size in number of units
        """
        if not self.engine or not current_price:
            return 0.0

        # Kelly Criterion approximation
        capital_at_risk = self.engine.current_capital * risk_per_trade
        position_value = capital_at_risk * signal_strength

        # Account for commission and slippage
        commission = position_value * self.engine.commission_rate
        slippage = position_value * (self.engine.slippage_bps / 10000.0)
        adjusted_value = position_value - commission - slippage

        # Calculate number of units
        quantity = adjusted_value / current_price

        return max(0.0, quantity)

    def buy(
        self,
        timestamp: datetime,
        price: float,
        quantity: Optional[float] = None,
        signal_strength: float = 1.0
    ):
        """
        Execute buy order.

        Args:
            timestamp: Order timestamp
            price: Limit price
            quantity: Order quantity (calculated if None)
            signal_strength: Signal strength for position sizing
        """
        if quantity is None:
            quantity = self.calculate_position_size(signal_strength, price)

        if quantity > 0:
            self.engine.execute_order(
                symbol=self.symbol,
                side='buy',
                quantity=quantity,
                price=price,
                timestamp=timestamp
            )

    def sell(
        self,
        timestamp: datetime,
        price: float,
        quantity: Optional[float] = None
    ):
        """
        Execute sell order.

        Args:
            timestamp: Order timestamp
            price: Limit price
            quantity: Order quantity (None = sell all)
        """
        # If no quantity specified, sell entire position
        if quantity is None:
            if self.symbol in self.engine.positions:
                quantity = self.engine.positions[self.symbol].quantity
            else:
                logger.warning(f"No position to sell for {self.symbol}")
                return

        if quantity > 0:
            self.engine.execute_order(
                symbol=self.symbol,
                side='sell',
                quantity=quantity,
                price=price,
                timestamp=timestamp
            )

    def has_position(self) -> bool:
        """Check if strategy has an open position."""
        return self.symbol in self.engine.positions

    @lru_cache(maxsize=128)

    def get_position_quantity(self) -> float:
        """Get current position quantity."""
        if self.has_position():
            return self.engine.positions[self.symbol].quantity
        return 0.0

    @lru_cache(maxsize=128)

    def get_position_entry_price(self) -> Optional[float]:
        """Get position entry price."""
        if self.has_position():
            return self.engine.positions[self.symbol].entry_price
        return None

    def set_parameter(self, name: str, value: Any):
        """
        Set strategy parameter.

        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        logger.debug(f"{self.name}: Set {name} = {value}")

    @lru_cache(maxsize=128)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get strategy parameter.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)

    def __repr__(self):
        return f"{self.name}({self.parameters})"
