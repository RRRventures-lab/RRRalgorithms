from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import numpy as np
import pandas as pd

"""
Performance Metrics Calculator

Calculates comprehensive performance metrics for backtesting results.
"""


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Return metrics
    total_return: float
    cagr: float
    daily_return_mean: float
    daily_return_std: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float

    # Position metrics
    avg_holding_period: float  # hours
    avg_bars_in_trade: float

    # Additional metrics
    expectancy: float
    sqn: float  # System Quality Number
    kelly_criterion: float
    recovery_factor: float

    # Time-based metrics
    start_date: datetime
    end_date: datetime
    total_days: int
    trading_days: int


class PerformanceCalculator:
    """Calculate comprehensive performance metrics from backtest results."""

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize performance calculator.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List,
        start_date: datetime,
        end_date: datetime
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Series of equity values over time
            trades: List of Trade objects
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        logger.info("Calculating performance metrics")

        # Time metrics
        total_days = (end_date - start_date).days
        trading_days = len(equity_curve)

        # Return metrics
        final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else self.initial_capital
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Calculate CAGR
        years = total_days / 365.25
        cagr = (((final_equity / self.initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0.0

        # Daily returns
        returns = equity_curve.pct_change().dropna()
        daily_return_mean = returns.mean() * 100
        daily_return_std = returns.std() * 100

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0

        # Trade metrics
        total_trades = len([t for t in trades if t.pnl is not None])
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl and t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade = np.mean([t.pnl for t in trades if t.pnl]) if trades else 0.0

        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Position metrics
        holding_periods = []
        for i, trade in enumerate(trades):
            if trade.side == 'sell' and i > 0:
                # Find corresponding buy
                for j in range(i - 1, -1, -1):
                    if trades[j].side == 'buy' and trades[j].symbol == trade.symbol:
                        duration = (trade.timestamp - trades[j].timestamp).total_seconds() / 3600
                        holding_periods.append(duration)
                        break

        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        avg_bars_in_trade = len(holding_periods) / total_trades if total_trades > 0 else 0.0

        # Advanced metrics
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # System Quality Number (SQN)
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]
        if len(trade_pnls) > 1:
            sqn = (np.mean(trade_pnls) / np.std(trade_pnls)) * np.sqrt(len(trade_pnls))
        else:
            sqn = 0.0

        # Kelly Criterion
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            kelly_criterion = win_rate - ((1 - win_rate) / win_loss_ratio)
        else:
            kelly_criterion = 0.0

        # Recovery Factor
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            daily_return_mean=daily_return_mean,
            daily_return_std=daily_return_std,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_period=avg_holding_period,
            avg_bars_in_trade=avg_bars_in_trade,
            expectancy=expectancy,
            sqn=sqn,
            kelly_criterion=kelly_criterion,
            recovery_factor=recovery_factor,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            trading_days=trading_days
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (uses only downside deviation).

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return (default: 0)

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> tuple:
        """
        Calculate maximum drawdown and its duration.

        Args:
            equity_curve: Series of equity values

        Returns:
            Tuple of (max_drawdown_pct, duration_in_days)
        """
        if len(equity_curve) == 0:
            return 0.0, 0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100

        max_drawdown = drawdown.min()

        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                current_duration = 0

        return max_drawdown, max_duration

    def print_summary(self, metrics: PerformanceMetrics):
        """
        Print formatted summary of performance metrics.

        Args:
            metrics: PerformanceMetrics object
        """
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)

        print(f"\n{'Period':<30} {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}")
        print(f"{'Total Days':<30} {metrics.total_days}")
        print(f"{'Trading Days':<30} {metrics.trading_days}")

        print("\n" + "-" * 70)
        print("RETURN METRICS")
        print("-" * 70)
        print(f"{'Total Return':<30} {metrics.total_return:>12.2f}%")
        print(f"{'CAGR':<30} {metrics.cagr:>12.2f}%")
        print(f"{'Daily Return (Mean)':<30} {metrics.daily_return_mean:>12.4f}%")
        print(f"{'Daily Return (Std)':<30} {metrics.daily_return_std:>12.4f}%")

        print("\n" + "-" * 70)
        print("RISK METRICS")
        print("-" * 70)
        print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio:>12.2f}")
        print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio:>12.2f}")
        print(f"{'Calmar Ratio':<30} {metrics.calmar_ratio:>12.2f}")
        print(f"{'Max Drawdown':<30} {metrics.max_drawdown:>12.2f}%")
        print(f"{'Max DD Duration (days)':<30} {metrics.max_drawdown_duration:>12}")

        print("\n" + "-" * 70)
        print("TRADE METRICS")
        print("-" * 70)
        print(f"{'Total Trades':<30} {metrics.total_trades:>12}")
        print(f"{'Winning Trades':<30} {metrics.winning_trades:>12}")
        print(f"{'Losing Trades':<30} {metrics.losing_trades:>12}")
        print(f"{'Win Rate':<30} {metrics.win_rate * 100:>12.2f}%")
        print(f"{'Profit Factor':<30} {metrics.profit_factor:>12.2f}")
        print(f"{'Average Win':<30} ${metrics.avg_win:>11.2f}")
        print(f"{'Average Loss':<30} ${metrics.avg_loss:>11.2f}")
        print(f"{'Average Trade':<30} ${metrics.avg_trade:>11.2f}")
        print(f"{'Largest Win':<30} ${metrics.largest_win:>11.2f}")
        print(f"{'Largest Loss':<30} ${metrics.largest_loss:>11.2f}")

        print("\n" + "-" * 70)
        print("POSITION METRICS")
        print("-" * 70)
        print(f"{'Avg Holding Period (hours)':<30} {metrics.avg_holding_period:>12.1f}")
        print(f"{'Avg Bars in Trade':<30} {metrics.avg_bars_in_trade:>12.1f}")

        print("\n" + "-" * 70)
        print("ADVANCED METRICS")
        print("-" * 70)
        print(f"{'Expectancy':<30} ${metrics.expectancy:>11.2f}")
        print(f"{'System Quality Number':<30} {metrics.sqn:>12.2f}")
        print(f"{'Kelly Criterion':<30} {metrics.kelly_criterion:>12.4f}")
        print(f"{'Recovery Factor':<30} {metrics.recovery_factor:>12.2f}")

        print("\n" + "=" * 70 + "\n")
