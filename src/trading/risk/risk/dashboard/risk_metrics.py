from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from supabase import create_client, Client
from typing import Dict, Optional, List
import logging
import numpy as np
import os
import pandas as pd


"""
Risk Dashboard

Calculates and displays comprehensive risk metrics:
- Current leverage
- Portfolio beta
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Risk-adjusted returns

Queries from Supabase tables and provides a unified view of system risk.
"""


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class RiskDashboardMetrics:
    """Complete risk dashboard metrics"""
    timestamp: datetime

    # Portfolio metrics
    total_value: float
    num_positions: int
    cash_balance: float
    invested_capital: float

    # Leverage and exposure
    gross_exposure: float
    net_exposure: float
    leverage: float

    # Risk metrics
    portfolio_volatility: float
    var_95: float
    max_drawdown: float
    current_drawdown: float

    # Performance metrics
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    beta: Optional[float]

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Daily metrics
    daily_pnl: float
    daily_pnl_pct: float

    # Risk status
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    warnings: List[str]


class RiskDashboard:
    """
    Comprehensive risk metrics dashboard

    Aggregates data from all risk monitors and calculates
    unified risk metrics for the entire system.
    """

    def __init__(
        self,
        lookback_days: int = 30,
        benchmark_symbol: str = "BTC-USD",
        risk_free_rate: float = 0.04
    ):
        """
        Initialize risk dashboard

        Args:
            lookback_days: Days of historical data for calculations
            benchmark_symbol: Benchmark for beta calculation
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.lookback_days = lookback_days
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate

        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

        self.supabase: Client = create_client(supabase_url, supabase_key)

        logger.info(f"Risk Dashboard initialized: lookback={lookback_days}d")

    @lru_cache(maxsize=128)

    def get_portfolio_value(self) -> Dict[str, float]:
        """Get current portfolio value breakdown"""
        try:
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if not positions:
                return {
                    "total_value": 0.0,
                    "invested_capital": 0.0,
                    "cash_balance": 0.0,
                    "num_positions": 0
                }

            total_value = sum(
                float(pos["quantity"]) * float(pos["current_price"])
                for pos in positions
            )

            invested_capital = sum(
                float(pos["quantity"]) * float(pos["entry_price"])
                for pos in positions
            )

            # Get cash balance from account
            cash_response = self.supabase.table("account").select("cash_balance").execute()
            cash_balance = float(cash_response.data[0]["cash_balance"]) if cash_response.data else 0.0

            return {
                "total_value": total_value + cash_balance,
                "invested_capital": invested_capital,
                "cash_balance": cash_balance,
                "num_positions": len(positions)
            }

        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return {
                "total_value": 0.0,
                "invested_capital": 0.0,
                "cash_balance": 0.0,
                "num_positions": 0
            }

    def calculate_leverage(self) -> Dict[str, float]:
        """Calculate portfolio leverage metrics"""
        try:
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if not positions:
                return {
                    "gross_exposure": 0.0,
                    "net_exposure": 0.0,
                    "leverage": 0.0
                }

            portfolio_value = self.get_portfolio_value()["total_value"]

            # Calculate exposures
            long_exposure = sum(
                float(pos["quantity"]) * float(pos["current_price"])
                for pos in positions
                if pos["side"] == "buy"
            )

            short_exposure = sum(
                float(pos["quantity"]) * float(pos["current_price"])
                for pos in positions
                if pos["side"] == "sell"
            )

            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure

            leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0.0

            return {
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "leverage": leverage
            }

        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            return {
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "leverage": 0.0
            }

    def calculate_performance_metrics(self) -> Dict[str, Optional[float]]:
        """Calculate Sharpe, Sortino, and Calmar ratios"""
        try:
            # Get portfolio snapshots
            start_date = datetime.now() - timedelta(days=self.lookback_days)
            response = (
                self.supabase.table("portfolio_snapshots")
                .select("timestamp, total_value")
                .gte("timestamp", start_date.isoformat())
                .order("timestamp")
                .execute()
            )

            snapshots = response.data

            if len(snapshots) < 2:
                return {
                    "sharpe_ratio": None,
                    "sortino_ratio": None,
                    "calmar_ratio": None
                }

            # Calculate daily returns
            values = [float(s["total_value"]) for s in snapshots]
            returns = np.diff(values) / values[:-1]

            if len(returns) == 0:
                return {
                    "sharpe_ratio": None,
                    "sortino_ratio": None,
                    "calmar_ratio": None
                }

            # Sharpe ratio
            daily_rf = self.risk_free_rate / 365
            excess_returns = returns - daily_rf
            sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            sharpe_annualized = sharpe * np.sqrt(365)

            # Sortino ratio (only downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
            sortino = np.mean(excess_returns) / downside_std
            sortino_annualized = sortino * np.sqrt(365)

            # Calmar ratio (return / max drawdown)
            cumulative_returns = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))

            annual_return = ((values[-1] / values[0]) ** (365 / len(returns))) - 1
            calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

            return {
                "sharpe_ratio": sharpe_annualized,
                "sortino_ratio": sortino_annualized,
                "calmar_ratio": calmar
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "calmar_ratio": None
            }

    def calculate_drawdown(self) -> Dict[str, float]:
        """Calculate maximum and current drawdown"""
        try:
            start_date = datetime.now() - timedelta(days=self.lookback_days)
            response = (
                self.supabase.table("portfolio_snapshots")
                .select("timestamp, total_value")
                .gte("timestamp", start_date.isoformat())
                .order("timestamp")
                .execute()
            )

            snapshots = response.data

            if len(snapshots) < 2:
                return {
                    "max_drawdown": 0.0,
                    "current_drawdown": 0.0
                }

            values = np.array([float(s["total_value"]) for s in snapshots])

            # Calculate running maximum
            running_max = np.maximum.accumulate(values)

            # Calculate drawdowns
            drawdowns = (values - running_max) / running_max

            max_drawdown = abs(np.min(drawdowns))
            current_drawdown = abs(drawdowns[-1])

            return {
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown
            }

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return {
                "max_drawdown": 0.0,
                "current_drawdown": 0.0
            }

    def calculate_trading_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        try:
            # Get closed trades
            response = (
                self.supabase.table("trades")
                .select("pnl")
                .eq("status", "closed")
                .execute()
            )

            trades = response.data

            if not trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0
                }

            pnl_values = [float(t["pnl"]) for t in trades]

            winning_trades = [p for p in pnl_values if p > 0]
            losing_trades = [abs(p) for p in pnl_values if p < 0]

            total_trades = len(pnl_values)
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)

            win_rate = num_wins / total_trades if total_trades > 0 else 0.0

            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = sum(losing_trades) if losing_trades else 1

            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = np.mean(losing_trades) if losing_trades else 0.0

            return {
                "total_trades": total_trades,
                "winning_trades": num_wins,
                "losing_trades": num_losses,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            }

        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }

    def calculate_daily_pnl(self) -> Dict[str, float]:
        """Calculate today's P&L"""
        try:
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())

            # Get starting balance
            start_response = (
                self.supabase.table("portfolio_snapshots")
                .select("total_value")
                .gte("timestamp", today_start.isoformat())
                .order("timestamp")
                .limit(1)
                .execute()
            )

            if not start_response.data:
                return {"daily_pnl": 0.0, "daily_pnl_pct": 0.0}

            starting_value = float(start_response.data[0]["total_value"])
            current_value = self.get_portfolio_value()["total_value"]

            daily_pnl = current_value - starting_value
            daily_pnl_pct = (daily_pnl / starting_value) if starting_value > 0 else 0.0

            return {
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct
            }

        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return {"daily_pnl": 0.0, "daily_pnl_pct": 0.0}

    def assess_risk_level(self, metrics: Dict) -> tuple[str, List[str]]:
        """
        Assess overall risk level and generate warnings

        Returns:
            Tuple of (risk_level, warnings)
        """
        warnings = []
        risk_score = 0

        # Check leverage
        if metrics["leverage"] > 2.0:
            warnings.append(f"High leverage: {metrics['leverage']:.2f}x")
            risk_score += 2
        elif metrics["leverage"] > 1.5:
            warnings.append(f"Elevated leverage: {metrics['leverage']:.2f}x")
            risk_score += 1

        # Check drawdown
        if metrics["current_drawdown"] > 0.15:
            warnings.append(f"Significant drawdown: {metrics['current_drawdown']:.1%}")
            risk_score += 2
        elif metrics["current_drawdown"] > 0.10:
            warnings.append(f"Moderate drawdown: {metrics['current_drawdown']:.1%}")
            risk_score += 1

        # Check volatility
        if metrics["portfolio_volatility"] > 0.30:
            warnings.append(f"High volatility: {metrics['portfolio_volatility']:.1%}")
            risk_score += 2
        elif metrics["portfolio_volatility"] > 0.20:
            warnings.append(f"Elevated volatility: {metrics['portfolio_volatility']:.1%}")
            risk_score += 1

        # Check daily loss
        if metrics["daily_pnl_pct"] < -0.05:
            warnings.append(f"Large daily loss: {metrics['daily_pnl_pct']:.1%}")
            risk_score += 2
        elif metrics["daily_pnl_pct"] < -0.03:
            warnings.append(f"Moderate daily loss: {metrics['daily_pnl_pct']:.1%}")
            risk_score += 1

        # Check win rate
        if metrics["win_rate"] < 0.40 and metrics["total_trades"] > 10:
            warnings.append(f"Low win rate: {metrics['win_rate']:.1%}")
            risk_score += 1

        # Determine risk level
        if risk_score >= 5:
            risk_level = "CRITICAL"
        elif risk_score >= 3:
            risk_level = "HIGH"
        elif risk_score >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return risk_level, warnings

    @lru_cache(maxsize=128)

    def get_dashboard_metrics(self) -> RiskDashboardMetrics:
        """
        Get complete risk dashboard metrics

        Returns:
            RiskDashboardMetrics object with all metrics
        """
        # Import here to avoid circular dependency
        from ..monitors.portfolio_risk import PortfolioRiskMonitor

        # Get portfolio value
        portfolio = self.get_portfolio_value()

        # Get leverage
        leverage_metrics = self.calculate_leverage()

        # Get risk metrics from monitor
        monitor = PortfolioRiskMonitor(lookback_days=self.lookback_days)
        risk_metrics = monitor.get_portfolio_metrics()

        # Get performance metrics
        performance = self.calculate_performance_metrics()

        # Get drawdown
        drawdown = self.calculate_drawdown()

        # Get trading metrics
        trading = self.calculate_trading_metrics()

        # Get daily P&L
        daily = self.calculate_daily_pnl()

        # Combine all metrics
        combined_metrics = {
            **portfolio,
            **leverage_metrics,
            "portfolio_volatility": risk_metrics.volatility_30d,
            "var_95": risk_metrics.var_95,
            "max_drawdown": drawdown["max_drawdown"],
            "current_drawdown": drawdown["current_drawdown"],
            **performance,
            "beta": risk_metrics.beta,
            **trading,
            **daily
        }

        # Assess risk level
        risk_level, warnings = self.assess_risk_level(combined_metrics)

        return RiskDashboardMetrics(
            timestamp=datetime.now(),
            total_value=portfolio["total_value"],
            num_positions=portfolio["num_positions"],
            cash_balance=portfolio["cash_balance"],
            invested_capital=portfolio["invested_capital"],
            gross_exposure=leverage_metrics["gross_exposure"],
            net_exposure=leverage_metrics["net_exposure"],
            leverage=leverage_metrics["leverage"],
            portfolio_volatility=risk_metrics.volatility_30d,
            var_95=risk_metrics.var_95,
            max_drawdown=drawdown["max_drawdown"],
            current_drawdown=drawdown["current_drawdown"],
            sharpe_ratio=performance["sharpe_ratio"],
            sortino_ratio=performance["sortino_ratio"],
            calmar_ratio=performance["calmar_ratio"],
            beta=risk_metrics.beta,
            total_trades=trading["total_trades"],
            winning_trades=trading["winning_trades"],
            losing_trades=trading["losing_trades"],
            win_rate=trading["win_rate"],
            profit_factor=trading["profit_factor"],
            avg_win=trading["avg_win"],
            avg_loss=trading["avg_loss"],
            daily_pnl=daily["daily_pnl"],
            daily_pnl_pct=daily["daily_pnl_pct"],
            risk_level=risk_level,
            warnings=warnings
        )

    def print_dashboard(self):
        """Print formatted risk dashboard"""
        metrics = self.get_dashboard_metrics()

        print("\n" + "=" * 70)
        print(f"{'RISK DASHBOARD':^70}")
        print("=" * 70)
        print(f"Timestamp: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Risk Level: {metrics.risk_level}")

        print("\n" + "-" * 70)
        print("PORTFOLIO")
        print("-" * 70)
        print(f"Total Value:        ${metrics.total_value:>15,.2f}")
        print(f"Invested Capital:   ${metrics.invested_capital:>15,.2f}")
        print(f"Cash Balance:       ${metrics.cash_balance:>15,.2f}")
        print(f"Positions:          {metrics.num_positions:>15}")

        print("\n" + "-" * 70)
        print("EXPOSURE & LEVERAGE")
        print("-" * 70)
        print(f"Gross Exposure:     ${metrics.gross_exposure:>15,.2f}")
        print(f"Net Exposure:       ${metrics.net_exposure:>15,.2f}")
        print(f"Leverage:           {metrics.leverage:>15.2f}x")

        print("\n" + "-" * 70)
        print("RISK METRICS")
        print("-" * 70)
        print(f"Portfolio Vol:      {metrics.portfolio_volatility:>15.1%}")
        print(f"VaR (95%):          ${metrics.var_95:>15,.2f}")
        print(f"Max Drawdown:       {metrics.max_drawdown:>15.1%}")
        print(f"Current Drawdown:   {metrics.current_drawdown:>15.1%}")

        print("\n" + "-" * 70)
        print("PERFORMANCE")
        print("-" * 70)
        print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>15.2f}" if metrics.sharpe_ratio else "Sharpe Ratio:       N/A")
        print(f"Sortino Ratio:      {metrics.sortino_ratio:>15.2f}" if metrics.sortino_ratio else "Sortino Ratio:      N/A")
        print(f"Calmar Ratio:       {metrics.calmar_ratio:>15.2f}" if metrics.calmar_ratio else "Calmar Ratio:       N/A")
        print(f"Beta:               {metrics.beta:>15.2f}" if metrics.beta else "Beta:               N/A")

        print("\n" + "-" * 70)
        print("TRADING STATISTICS")
        print("-" * 70)
        print(f"Total Trades:       {metrics.total_trades:>15}")
        print(f"Win Rate:           {metrics.win_rate:>15.1%}")
        print(f"Profit Factor:      {metrics.profit_factor:>15.2f}")
        print(f"Avg Win:            ${metrics.avg_win:>15,.2f}")
        print(f"Avg Loss:           ${metrics.avg_loss:>15,.2f}")

        print("\n" + "-" * 70)
        print("TODAY'S PERFORMANCE")
        print("-" * 70)
        print(f"Daily P&L:          ${metrics.daily_pnl:>15,.2f}")
        print(f"Daily P&L %:        {metrics.daily_pnl_pct:>15.2%}")

        if metrics.warnings:
            print("\n" + "-" * 70)
            print("WARNINGS")
            print("-" * 70)
            for warning in metrics.warnings:
                print(f"  ⚠ {warning}")

        print("\n" + "=" * 70)


def main():
    """Example usage"""
    try:
        dashboard = RiskDashboard(
            lookback_days=30,
            benchmark_symbol="BTC-USD"
        )

        dashboard.print_dashboard()

    except Exception as e:
        logger.error(f"Error displaying risk dashboard: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
