from dataclasses import dataclass
from typing import Union, Optional, Dict
import logging
import numpy as np
import pandas as pd

"""
Trading-Specific Performance Metrics

Comprehensive metrics for evaluating trading strategies:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Win Rate, Profit Factor
- Risk-adjusted returns and more
"""


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all trading performance metrics"""
    # Returns metrics
    total_return: float
    annual_return: float
    daily_return_mean: float
    daily_return_std: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    recovery_factor: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_days: float

    # Other metrics
    expectancy: float
    kelly_criterion: float
    tail_ratio: float
    var_95: float
    cvar_95: float


class TradingMetricsCalculator:
    """
    Calculate comprehensive trading performance metrics

    Supports both returns-based and trade-based calculations
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        trades: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics

        Args:
            returns: Daily returns series
            trades: DataFrame with trade details (optional)
            equity_curve: Equity curve series (optional, will compute from returns)

        Returns:
            PerformanceMetrics object with all metrics
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        if equity_curve is None:
            equity_curve = (1 + returns).cumprod()

        # Returns metrics
        total_return = self.total_return(returns)
        annual_return = self.annual_return(returns)
        daily_return_mean = returns.mean()
        daily_return_std = returns.std()

        # Risk-adjusted metrics
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns)
        omega = self.omega_ratio(returns)

        # Drawdown metrics
        max_dd = self.max_drawdown(equity_curve)
        max_dd_duration = self.max_drawdown_duration(equity_curve)
        avg_dd = self.average_drawdown(equity_curve)
        recovery = self.recovery_factor(returns, equity_curve)

        # Trade metrics (if trades provided)
        if trades is not None and len(trades) > 0:
            total_trades = len(trades)
            win_rate = self.win_rate(trades)
            profit_factor = self.profit_factor(trades)
            avg_win = self.average_win(trades)
            avg_loss = self.average_loss(trades)
            largest_win = self.largest_win(trades)
            largest_loss = self.largest_loss(trades)
            avg_duration = self.average_trade_duration(trades)
            expectancy = self.expectancy(trades)
            kelly = self.kelly_criterion(trades)
        else:
            total_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            largest_win = 0.0
            largest_loss = 0.0
            avg_duration = 0.0
            expectancy = 0.0
            kelly = 0.0

        # Additional metrics
        tail = self.tail_ratio(returns)
        var = self.value_at_risk(returns, confidence=0.95)
        cvar = self.conditional_value_at_risk(returns, confidence=0.95)

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            daily_return_mean=daily_return_mean,
            daily_return_std=daily_return_std,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown=avg_dd,
            recovery_factor=recovery,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_days=avg_duration,
            expectancy=expectancy,
            kelly_criterion=kelly,
            tail_ratio=tail,
            var_95=var,
            cvar_95=cvar
        )

    # ========== Returns-Based Metrics ==========

    def total_return(self, returns: pd.Series) -> float:
        """Calculate total cumulative return"""
        return (1 + returns).prod() - 1

    def annual_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        total_ret = self.total_return(returns)
        n_periods = len(returns)
        years = n_periods / periods_per_year
        if years > 0:
            return (1 + total_ret) ** (1 / years) - 1
        return 0.0

    def sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / returns.std()
        return sharpe * np.sqrt(periods_per_year)

    def sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio

        Like Sharpe but only penalizes downside volatility
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std()
        return sortino * np.sqrt(periods_per_year)

    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio

        Calmar = Annual Return / Max Drawdown
        """
        annual_ret = self.annual_return(returns)
        equity_curve = (1 + returns).cumprod()
        max_dd = abs(self.max_drawdown(equity_curve))

        if max_dd == 0:
            return 0.0

        return annual_ret / max_dd

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio

        Ratio of probability-weighted gains to losses above/below threshold
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return np.inf if gains.sum() > 0 else 0.0

        return gains.sum() / losses.sum()

    # ========== Drawdown Metrics ==========

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Returns negative value (e.g., -0.15 for 15% drawdown)
        """
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """
        Calculate longest drawdown duration in days

        Returns number of periods underwater
        """
        if len(equity_curve) == 0:
            return 0

        running_max = equity_curve.expanding().max()
        is_drawdown = equity_curve < running_max

        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0

        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        return max(drawdown_periods) if drawdown_periods else 0

    def average_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown magnitude"""
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        drawdowns = drawdowns[drawdowns < 0]

        return drawdowns.mean() if len(drawdowns) > 0 else 0.0

    def recovery_factor(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """
        Calculate recovery factor

        Recovery Factor = Total Return / Max Drawdown
        """
        total_ret = self.total_return(returns)
        max_dd = abs(self.max_drawdown(equity_curve))

        if max_dd == 0:
            return 0.0

        return total_ret / max_dd

    # ========== Trade-Based Metrics ==========

    def win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate (% of profitable trades)"""
        if len(trades) == 0:
            return 0.0

        winning_trades = (trades['pnl'] > 0).sum()
        return winning_trades / len(trades)

    def profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Calculate profit factor

        Profit Factor = Gross Profit / Gross Loss
        """
        if len(trades) == 0:
            return 0.0

        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_win(self, trades: pd.DataFrame) -> float:
        """Calculate average winning trade"""
        winning_trades = trades[trades['pnl'] > 0]
        return winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0

    def average_loss(self, trades: pd.DataFrame) -> float:
        """Calculate average losing trade"""
        losing_trades = trades[trades['pnl'] < 0]
        return losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0

    def largest_win(self, trades: pd.DataFrame) -> float:
        """Calculate largest winning trade"""
        return trades['pnl'].max() if len(trades) > 0 else 0.0

    def largest_loss(self, trades: pd.DataFrame) -> float:
        """Calculate largest losing trade"""
        return trades['pnl'].min() if len(trades) > 0 else 0.0

    def average_trade_duration(self, trades: pd.DataFrame) -> float:
        """Calculate average trade duration in days"""
        if len(trades) == 0 or 'entry_time' not in trades.columns:
            return 0.0

        durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 86400
        return durations.mean()

    def expectancy(self, trades: pd.DataFrame) -> float:
        """
        Calculate expectancy (expected value per trade)

        Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
        """
        if len(trades) == 0:
            return 0.0

        win_rate = self.win_rate(trades)
        loss_rate = 1 - win_rate
        avg_win = self.average_win(trades)
        avg_loss = self.average_loss(trades)

        return (win_rate * avg_win) + (loss_rate * avg_loss)

    def kelly_criterion(self, trades: pd.DataFrame) -> float:
        """
        Calculate Kelly Criterion optimal bet size

        Kelly % = (Win Rate * Avg Win - Loss Rate * Avg Loss) / Avg Win
        """
        if len(trades) == 0:
            return 0.0

        win_rate = self.win_rate(trades)
        loss_rate = 1 - win_rate
        avg_win = abs(self.average_win(trades))
        avg_loss = abs(self.average_loss(trades))

        if avg_win == 0:
            return 0.0

        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        return max(0.0, kelly)  # Kelly can't be negative

    # ========== Risk Metrics ==========

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        VaR_95 = 5th percentile of returns distribution
        """
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_value_at_risk(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)

        CVaR = Average of losses beyond VaR threshold
        """
        if len(returns) == 0:
            return 0.0

        var = self.value_at_risk(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar

    def tail_ratio(self, returns: pd.Series) -> float:
        """
        Calculate tail ratio

        Tail Ratio = 95th percentile / abs(5th percentile)
        """
        if len(returns) == 0:
            return 0.0

        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))

        if p5 == 0:
            return 0.0

        return p95 / p5


# Convenience functions
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Quick Sharpe ratio calculation"""
    calc = TradingMetricsCalculator(risk_free_rate=risk_free_rate)
    return calc.sharpe_ratio(returns)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Quick Sortino ratio calculation"""
    calc = TradingMetricsCalculator(risk_free_rate=risk_free_rate)
    return calc.sortino_ratio(returns)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Quick max drawdown calculation"""
    calc = TradingMetricsCalculator()
    return calc.max_drawdown(equity_curve)


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Daily returns

    # Generate sample trades
    trades = pd.DataFrame({
        'pnl': np.random.randn(100) * 100,
        'entry_time': pd.date_range('2023-01-01', periods=100, freq='D'),
        'exit_time': pd.date_range('2023-01-02', periods=100, freq='D')
    })

    # Calculate metrics
    calc = TradingMetricsCalculator(risk_free_rate=0.02)
    metrics = calc.calculate_all_metrics(returns, trades)

    # Print results
    print("=== Performance Metrics ===")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annual Return: {metrics.annual_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Kelly Criterion: {metrics.kelly_criterion:.2%}")
