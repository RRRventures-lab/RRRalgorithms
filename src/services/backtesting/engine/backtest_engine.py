from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import logging
import numpy as np
import pandas as pd

"""
Backtesting Engine

Core engine for simulating trading strategies on historical data.
"""


logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    side: str  # 'long' or 'short'


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions: List[Position]
    final_equity: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    metadata: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Core backtesting engine that replays historical data and simulates trading.

    Features:
    - Realistic order execution with slippage
    - Commission/fee modeling
    - Position tracking
    - Equity curve calculation
    - Support for multiple strategies
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage_model: str = 'fixed',  # 'fixed', 'volumetric', 'volatility'
        slippage_bps: float = 5.0,  # basis points
        max_position_size: float = 0.95,  # 95% of capital
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital in USD
            commission_rate: Commission as decimal (0.001 = 0.1%)
            slippage_model: Model for calculating slippage
            slippage_bps: Slippage in basis points
            max_position_size: Maximum position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size

        # State variables
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_timestamp: Optional[datetime] = None

        logger.info(f"Initialized BacktestEngine with ${initial_capital:,.2f}")

    def reset(self):
        """Reset the engine to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_timestamp = None
        logger.info("Engine reset to initial state")

    def calculate_slippage(
        self,
        price: float,
        side: str,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate slippage based on the configured model.

        Args:
            price: Order price
            side: 'buy' or 'sell'
            volume: Trading volume (for volumetric model)
            volatility: Price volatility (for volatility model)

        Returns:
            Slippage amount in dollars
        """
        if self.slippage_model == 'fixed':
            slippage_pct = self.slippage_bps / 10000.0
            slippage = price * slippage_pct
            return slippage if side == 'buy' else -slippage

        elif self.slippage_model == 'volumetric':
            # More volume = less slippage
            if volume and volume > 0:
                base_slippage = self.slippage_bps / 10000.0
                volume_factor = max(0.5, min(2.0, 1.0 / np.log10(volume + 10)))
                slippage = price * base_slippage * volume_factor
                return slippage if side == 'buy' else -slippage
            else:
                return self.calculate_slippage(price, side)

        elif self.slippage_model == 'volatility':
            # Higher volatility = more slippage
            if volatility and volatility > 0:
                base_slippage = self.slippage_bps / 10000.0
                volatility_factor = max(0.5, min(3.0, 1.0 + volatility * 10))
                slippage = price * base_slippage * volatility_factor
                return slippage if side == 'buy' else -slippage
            else:
                return self.calculate_slippage(price, side)

        return 0.0

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trading order with realistic slippage and commissions.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price
            timestamp: Order timestamp
            volume: Market volume (optional)
            volatility: Price volatility (optional)

        Returns:
            Trade object if successful, None otherwise
        """
        # Calculate slippage
        slippage = self.calculate_slippage(price, side, volume, volatility)
        execution_price = price + slippage

        # Calculate commission
        order_value = quantity * execution_price
        commission = order_value * self.commission_rate

        # Check if we have enough capital for buy orders
        if side == 'buy':
            total_cost = order_value + commission
            if total_cost > self.current_capital * self.max_position_size:
                logger.warning(
                    f"Insufficient capital for {symbol} buy: "
                    f"${total_cost:,.2f} > ${self.current_capital * self.max_position_size:,.2f}"
                )
                return None

            # Create position
            if symbol in self.positions:
                # Add to existing position
                existing = self.positions[symbol]
                total_quantity = existing.quantity + quantity
                avg_price = (
                    (existing.quantity * existing.entry_price + quantity * execution_price)
                    / total_quantity
                )
                existing.quantity = total_quantity
                existing.entry_price = avg_price
            else:
                # Open new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    entry_timestamp=timestamp,
                    side='long'
                )

            self.current_capital -= total_cost

        elif side == 'sell':
            # Check if we have the position
            if symbol not in self.positions:
                logger.warning(f"Cannot sell {symbol}: no open position")
                return None

            position = self.positions[symbol]
            if quantity > position.quantity:
                logger.warning(
                    f"Cannot sell {quantity} {symbol}: only {position.quantity} available"
                )
                return None

            # Calculate P&L
            pnl = (execution_price - position.entry_price) * quantity - commission

            # Update position
            if quantity == position.quantity:
                # Close entire position
                del self.positions[symbol]
            else:
                # Reduce position
                position.quantity -= quantity

            self.current_capital += order_value - commission

        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=abs(slippage),
            pnl=pnl if side == 'sell' else None
        )

        self.trades.append(trade)
        logger.debug(
            f"Executed {side} {quantity} {symbol} @ ${execution_price:.2f} "
            f"(slippage: ${slippage:.4f}, commission: ${commission:.2f})"
        )

        return trade

    def update_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        """
        Update equity curve based on current prices.

        Args:
            timestamp: Current timestamp
            current_prices: Dictionary of symbol -> current price
        """
        self.current_timestamp = timestamp

        # Calculate position value
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_price = current_prices[symbol]
                position_value += position.quantity * market_price
            else:
                # Use entry price if no current price available
                position_value += position.quantity * position.entry_price

        total_equity = self.current_capital + position_value
        self.equity_curve.append((timestamp, total_equity))

    def run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        symbol: str = 'BTC-USD'
    ) -> BacktestResult:
        """
        Run backtest with a given strategy on historical data.

        Args:
            strategy: Strategy instance with on_bar() method
            data: Historical OHLCV data
            symbol: Trading symbol

        Returns:
            BacktestResult with complete backtest results
        """
        logger.info(f"Starting backtest for {symbol} with {len(data)} bars")
        self.reset()

        strategy.initialize(self, symbol)

        # Iterate through each bar
        for timestamp, bar in data.iterrows():
            # Update equity before processing bar
            current_prices = {symbol: bar['close']}
            self.update_equity(timestamp, current_prices)

            # Let strategy process the bar
            strategy.on_bar(timestamp, bar)

        # Final equity update
        if len(data) > 0:
            last_bar = data.iloc[-1]
            final_prices = {symbol: last_bar['close']}
            self.update_equity(data.index[-1], final_prices)

        # Build result
        result = self._build_result(strategy)
        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"${result.final_equity:,.2f} final equity "
            f"({result.total_return:.2f}% return)"
        )

        return result

    def _build_result(self, strategy) -> BacktestResult:
        """Build BacktestResult from current state."""
        # Convert equity curve to series
        if len(self.equity_curve) > 0:
            equity_series = pd.Series(
                data=[eq for _, eq in self.equity_curve],
                index=[ts for ts, _ in self.equity_curve]
            )
        else:
            equity_series = pd.Series()

        # Calculate drawdown
        if len(equity_series) > 0:
            rolling_max = equity_series.expanding().max()
            drawdown_series = (equity_series - rolling_max) / rolling_max
        else:
            drawdown_series = pd.Series()

        # Calculate metrics
        final_equity = self.current_capital
        if len(equity_series) > 0:
            final_equity = equity_series.iloc[-1]

        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        winning_trades = len([t for t in self.trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl and t.pnl < 0])
        total_trades = len(self.trades)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        max_drawdown = drawdown_series.min() if len(drawdown_series) > 0 else 0.0

        # Calculate Sharpe ratio
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Calculate Sortino ratio
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            sortino_ratio = (
                returns.mean() / downside_std * np.sqrt(252)
                if downside_std > 0 else 0.0
            )
        else:
            sortino_ratio = 0.0

        # Calculate profit factor
        gross_profit = sum([t.pnl for t in self.trades if t.pnl and t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in self.trades if t.pnl and t.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            positions=list(self.positions.values()),
            final_equity=final_equity,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            metadata={
                'initial_capital': self.initial_capital,
                'strategy': strategy.__class__.__name__,
                'commission_rate': self.commission_rate,
                'slippage_bps': self.slippage_bps,
            }
        )
