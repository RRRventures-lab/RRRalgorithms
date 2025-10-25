from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd
import logging

"""
High-Performance Backtesting Engine
====================================

Vectorized backtesting engine supporting:
- Multiple strategies
- Walk-forward analysis
- Transaction costs
- Slippage modeling
- Position sizing
- Risk management
"""

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    position_size: float
    pnl: float
    pnl_percent: float
    fees: float
    slippage: float

    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds"""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Profit metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Statistical tests
    t_statistic: float = 0.0
    p_value: float = 1.0

    # Additional
    avg_trade_duration: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0

    def is_viable(self,
                  min_sharpe: float = 2.0,
                  min_win_rate: float = 0.55,
                  min_profit_factor: float = 1.5,
                  max_dd: float = 0.20,
                  min_trades: int = 30) -> bool:
        """Check if strategy meets viability criteria"""
        return (
            self.sharpe_ratio >= min_sharpe and
            self.win_rate >= min_win_rate and
            self.profit_factor >= min_profit_factor and
            abs(self.max_drawdown) <= max_dd and
            self.total_trades >= min_trades and
            self.p_value < 0.01
        )


class BacktestEngine:
    """
    High-performance vectorized backtesting engine

    Features:
    - Vectorized operations for speed
    - Walk-forward analysis
    - Multiple time frames
    - Transaction costs and slippage
    - Position sizing
    - Risk management
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_pct: float = 0.001,  # 0.1%
                 slippage_pct: float = 0.0005,   # 0.05%
                 max_position_size: float = 0.1,  # 10% per position
                 leverage: float = 1.0):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital
            commission_pct: Commission as percentage of trade value
            slippage_pct: Slippage as percentage of trade value
            max_position_size: Maximum position size as fraction of capital
            leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size
        self.leverage = leverage

        # Results storage
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None
        self.metrics: Optional[BacktestMetrics] = None

    def backtest_strategy(self,
                         data: pd.DataFrame,
                         signal_func: Callable[[pd.DataFrame], pd.Series],
                         strategy_name: str = "Strategy") -> BacktestMetrics:
        """
        Backtest a trading strategy

        Args:
            data: DataFrame with OHLCV data
            signal_func: Function that returns trading signals (1=long, -1=short, 0=neutral)
            strategy_name: Name of strategy

        Returns:
            BacktestMetrics with performance statistics
        """
        logger.info(f"Backtesting {strategy_name} on {len(data)} bars")

        # Generate signals
        signals = signal_func(data)

        # Validate signals
        if len(signals) != len(data):
            raise ValueError("Signal length must match data length")

        # Calculate returns
        returns = data['close'].pct_change()

        # Calculate strategy returns (vectorized)
        position = signals.shift(1)  # Execute signal on next bar
        strategy_returns = position * returns

        # Apply transaction costs
        trades_mask = position.diff().abs() > 0
        transaction_costs = trades_mask * (self.commission_pct + self.slippage_pct)
        strategy_returns = strategy_returns - transaction_costs

        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        self.equity_curve = equity_curve

        # Extract individual trades
        self._extract_trades(data, signals, returns)

        # Calculate comprehensive metrics
        self.metrics = self._calculate_metrics(strategy_returns, equity_curve)

        logger.info(f"Backtest complete: {self.metrics.total_trades} trades, "
                   f"Sharpe: {self.metrics.sharpe_ratio:.2f}, "
                   f"Win rate: {self.metrics.win_rate:.1%}")

        return self.metrics

    def _extract_trades(self, data: pd.DataFrame, signals: pd.Series, returns: pd.Series):
        """Extract individual trades from signals"""
        self.trades = []

        position = 0
        entry_idx = None
        entry_price = None

        for i in range(1, len(signals)):
            current_signal = signals.iloc[i]
            prev_signal = signals.iloc[i-1]

            # Entry: signal changes from 0 to non-zero
            if prev_signal == 0 and current_signal != 0:
                position = current_signal
                entry_idx = i
                entry_price = data.iloc[i]['close']

            # Exit: signal changes or goes to 0
            elif position != 0 and (current_signal != position or current_signal == 0):
                exit_idx = i
                exit_price = data.iloc[i]['close']

                # Calculate P&L
                if position > 0:  # Long
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:  # Short
                    pnl_pct = (entry_price - exit_price) / entry_price

                # Apply costs
                fees = self.commission_pct * 2  # Entry + exit
                slippage = self.slippage_pct * 2
                pnl_pct = pnl_pct - fees - slippage

                # Calculate position size and absolute P&L
                position_value = self.initial_capital * self.max_position_size
                pnl = position_value * pnl_pct

                # Create trade record
                trade = Trade(
                    entry_time=data.index[entry_idx],
                    exit_time=data.index[exit_idx],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction='long' if position > 0 else 'short',
                    position_size=position_value,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    fees=position_value * fees,
                    slippage=position_value * slippage
                )

                self.trades.append(trade)

                # Reset for next trade
                position = current_signal
                if current_signal != 0:
                    entry_idx = i
                    entry_price = data.iloc[i]['close']
                else:
                    entry_idx = None
                    entry_price = None

    def _calculate_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = BacktestMetrics()

        # Basic trade statistics
        metrics.total_trades = len(self.trades)

        if metrics.total_trades == 0:
            return metrics

        winning_trades = [t for t in self.trades if t.is_winner]
        losing_trades = [t for t in self.trades if not t.is_winner]

        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # Average wins/losses
        if winning_trades:
            metrics.avg_win = np.mean([t.pnl_percent for t in winning_trades])
            metrics.largest_win = max([t.pnl_percent for t in winning_trades])
            metrics.gross_profit = sum([t.pnl for t in winning_trades])

        if losing_trades:
            metrics.avg_loss = np.mean([t.pnl_percent for t in losing_trades])
            metrics.largest_loss = min([t.pnl_percent for t in losing_trades])
            metrics.gross_loss = abs(sum([t.pnl for t in losing_trades]))

        metrics.avg_trade = np.mean([t.pnl_percent for t in self.trades])

        # Profit factor
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss

        # Returns
        metrics.total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        metrics.cumulative_return = metrics.total_return

        # Annualized return (assuming daily data)
        trading_days = len(returns)
        years = trading_days / 252
        if years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1

        # Volatility
        metrics.volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        if metrics.volatility > 0:
            metrics.sharpe_ratio = (metrics.annualized_return - 0.02) / metrics.volatility  # Assuming 2% risk-free rate

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics.sortino_ratio = (metrics.annualized_return - 0.02) / downside_std

        # Maximum drawdown
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = drawdown.min()

        # Max drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(cumulative)
        metrics.max_drawdown_duration = drawdown_duration

        # Calmar ratio
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = abs(metrics.annualized_return / metrics.max_drawdown)

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        metrics.var_95 = np.percentile(returns.dropna(), 5)
        metrics.cvar_95 = returns[returns <= metrics.var_95].mean()

        # Statistical significance (t-test)
        if len(returns) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
            metrics.t_statistic = t_stat
            metrics.p_value = p_value

        # Additional metrics
        metrics.avg_trade_duration = np.mean([t.duration_seconds for t in self.trades])
        metrics.total_fees = sum([t.fees for t in self.trades])
        metrics.total_slippage = sum([t.slippage for t in self.trades])

        return metrics

    def _calculate_drawdown_duration(self, cumulative: pd.Series) -> int:
        """Calculate maximum drawdown duration in bars"""
        running_max = cumulative.expanding().max()
        is_drawdown = cumulative < running_max

        max_duration = 0
        current_duration = 0

        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def walk_forward_analysis(self,
                            data: pd.DataFrame,
                            signal_func: Callable[[pd.DataFrame], pd.Series],
                            train_size: int = 252,  # 1 year
                            test_size: int = 63,    # 3 months
                            n_splits: int = 5) -> List[BacktestMetrics]:
        """
        Perform walk-forward analysis

        Args:
            data: Full dataset
            signal_func: Signal generation function
            train_size: Training window size
            test_size: Testing window size
            n_splits: Number of walk-forward splits

        Returns:
            List of metrics for each test period
        """
        logger.info(f"Starting walk-forward analysis with {n_splits} splits")

        results = []
        step_size = len(data) // (n_splits + 1)

        for i in range(n_splits):
            start_idx = i * step_size
            train_end = start_idx + train_size
            test_end = train_end + test_size

            if test_end > len(data):
                break

            # Split data
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]

            logger.info(f"Split {i+1}: Train={len(train_data)} bars, Test={len(test_data)} bars")

            # Backtest on out-of-sample data
            metrics = self.backtest_strategy(test_data, signal_func, f"WF_Split_{i+1}")
            results.append(metrics)

        logger.info(f"Walk-forward analysis complete: {len(results)} periods tested")

        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'duration_seconds': t.duration_seconds,
                'fees': t.fees,
                'slippage': t.slippage
            }
            for t in self.trades
        ])
