from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from supabase import create_client, Client
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import os
import pandas as pd


"""
Portfolio Risk Monitor

Calculates and tracks portfolio-level risk metrics:
- Value at Risk (VaR) at 95% confidence
- Portfolio volatility (rolling 30-day)
- Asset correlations
- Beta to benchmark
- Maximum drawdown

Integrates with Supabase for real-time position and price data.
"""


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics snapshot"""
    timestamp: datetime
    total_value: float
    var_95: float  # Value at Risk at 95% confidence
    volatility_30d: float  # 30-day rolling volatility
    max_drawdown: float  # Maximum drawdown from peak
    sharpe_ratio: Optional[float]  # Sharpe ratio (if enough data)
    beta: Optional[float]  # Beta to benchmark
    correlation_matrix: Optional[pd.DataFrame]  # Asset correlations
    largest_position_pct: float  # Largest single position as % of portfolio
    num_positions: int
    leverage: float  # Total exposure / portfolio value
    notes: str


class PortfolioRiskMonitor:
    """
    Monitor portfolio-level risk metrics

    Queries Supabase for position and price data to calculate
    real-time risk metrics and check against limits.
    """

    def __init__(
        self,
        max_portfolio_volatility: float = 0.25,
        var_confidence_level: float = 0.95,
        lookback_days: int = 30,
        benchmark_symbol: str = "BTC-USD"
    ):
        """
        Initialize portfolio risk monitor

        Args:
            max_portfolio_volatility: Maximum acceptable portfolio volatility
            var_confidence_level: VaR confidence level (default 95%)
            lookback_days: Days of historical data for calculations
            benchmark_symbol: Benchmark symbol for beta calculation
        """
        self.max_portfolio_volatility = max_portfolio_volatility
        self.var_confidence_level = var_confidence_level
        self.lookback_days = lookback_days
        self.benchmark_symbol = benchmark_symbol

        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment")

        self.supabase: Client = create_client(supabase_url, supabase_key)

        logger.info(
            f"Portfolio Risk Monitor initialized: max_vol={max_portfolio_volatility:.1%}, "
            f"VaR={var_confidence_level:.0%}, lookback={lookback_days}d"
        )

    @lru_cache(maxsize=128)

    def get_current_positions(self) -> pd.DataFrame:
        """
        Get current portfolio positions from Supabase

        Returns:
            DataFrame with columns: symbol, quantity, current_price, value, pct_of_portfolio
        """
        try:
            # Query positions table
            response = self.supabase.table("positions").select("*").execute()
            positions = pd.DataFrame(response.data)

            if positions.empty:
                logger.warning("No positions found in database")
                return pd.DataFrame(columns=["symbol", "quantity", "current_price", "value", "pct_of_portfolio"])

            # Calculate position values
            positions["value"] = positions["quantity"] * positions["current_price"]
            total_value = positions["value"].sum()
            positions["pct_of_portfolio"] = positions["value"] / total_value

            return positions

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    @lru_cache(maxsize=128)

    def get_historical_prices(
        self,
        symbols: List[str],
        days: int
    ) -> pd.DataFrame:
        """
        Get historical price data from Supabase

        Args:
            symbols: List of symbols to fetch
            days: Number of days of historical data

        Returns:
            DataFrame with columns: timestamp, symbol, close
        """
        try:
            start_date = datetime.now() - timedelta(days=days)

            # Query crypto_aggregates table
            response = (
                self.supabase.table("crypto_aggregates")
                .select("timestamp, symbol, close")
                .in_("symbol", symbols)
                .gte("timestamp", start_date.isoformat())
                .order("timestamp")
                .execute()
            )

            prices = pd.DataFrame(response.data)

            if prices.empty:
                logger.warning(f"No price data found for symbols: {symbols}")
                return pd.DataFrame(columns=["timestamp", "symbol", "close"])

            prices["timestamp"] = pd.to_datetime(prices["timestamp"])

            return prices

        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            raise

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data

        Args:
            prices: DataFrame with timestamp, symbol, close

        Returns:
            DataFrame with daily returns for each symbol
        """
        # Pivot to wide format
        price_pivot = prices.pivot(index="timestamp", columns="symbol", values="close")

        # Calculate daily returns
        returns = price_pivot.pct_change().dropna()

        return returns

    def calculate_var(
        self,
        portfolio_returns: np.ndarray,
        portfolio_value: float,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation

        Args:
            portfolio_returns: Array of portfolio returns
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (default 95%)

        Returns:
            VaR in dollars (potential loss)
        """
        if len(portfolio_returns) == 0:
            return 0.0

        # Sort returns
        sorted_returns = np.sort(portfolio_returns)

        # Find the percentile
        percentile_idx = int((1 - confidence_level) * len(sorted_returns))
        var_return = sorted_returns[percentile_idx]

        # Convert to dollar amount (negative value)
        var_dollars = abs(var_return * portfolio_value)

        return var_dollars

    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            returns: Array of returns
            annualize: If True, annualize the volatility

        Returns:
            Volatility (annualized if requested)
        """
        if len(returns) == 0:
            return 0.0

        vol = np.std(returns)

        if annualize:
            # Assume 365 days per year for crypto
            vol = vol * np.sqrt(365)

        return vol

    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown

        Args:
            portfolio_values: Array of portfolio values over time

        Returns:
            Maximum drawdown as a fraction (e.g., 0.15 = 15% drawdown)
        """
        if len(portfolio_values) == 0:
            return 0.0

        # Calculate cumulative maximum
        cummax = np.maximum.accumulate(portfolio_values)

        # Calculate drawdowns
        drawdowns = (portfolio_values - cummax) / cummax

        # Return maximum drawdown (most negative)
        max_dd = abs(np.min(drawdowns))

        return max_dd

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.04
    ) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default 4%)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        # Calculate excess returns
        daily_rf = risk_free_rate / 365
        excess_returns = returns - daily_rf

        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)

        # Annualize
        sharpe = sharpe * np.sqrt(365)

        return sharpe

    def calculate_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Optional[float]:
        """
        Calculate portfolio beta to benchmark

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta (None if insufficient data)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return None

        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return None

        beta = covariance / benchmark_variance

        return beta

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate asset correlation matrix

        Args:
            returns: DataFrame of returns for each asset

        Returns:
            Correlation matrix
        """
        if returns.empty:
            return pd.DataFrame()

        return returns.corr()

    @lru_cache(maxsize=128)

    def get_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """
        Calculate all portfolio risk metrics

        Returns:
            PortfolioRiskMetrics object with current metrics
        """
        notes = []

        # Get current positions
        positions = self.get_current_positions()

        if positions.empty:
            logger.warning("No positions to calculate risk metrics")
            return PortfolioRiskMetrics(
                timestamp=datetime.now(),
                total_value=0.0,
                var_95=0.0,
                volatility_30d=0.0,
                max_drawdown=0.0,
                sharpe_ratio=None,
                beta=None,
                correlation_matrix=None,
                largest_position_pct=0.0,
                num_positions=0,
                leverage=0.0,
                notes="No positions"
            )

        # Calculate total portfolio value
        total_value = positions["value"].sum()
        num_positions = len(positions)
        largest_position_pct = positions["pct_of_portfolio"].max()

        # Get historical prices
        symbols = positions["symbol"].tolist()
        prices = self.get_historical_prices(symbols, self.lookback_days)

        if prices.empty:
            notes.append("Insufficient price history for risk calculations")
            return PortfolioRiskMetrics(
                timestamp=datetime.now(),
                total_value=total_value,
                var_95=0.0,
                volatility_30d=0.0,
                max_drawdown=0.0,
                sharpe_ratio=None,
                beta=None,
                correlation_matrix=None,
                largest_position_pct=largest_position_pct,
                num_positions=num_positions,
                leverage=1.0,
                notes="; ".join(notes)
            )

        # Calculate returns
        returns = self.calculate_returns(prices)

        # Calculate portfolio returns (weighted by position)
        portfolio_returns = np.zeros(len(returns))
        for symbol in symbols:
            if symbol in returns.columns:
                weight = positions[positions["symbol"] == symbol]["pct_of_portfolio"].iloc[0]
                portfolio_returns += returns[symbol].values * weight

        # Calculate VaR
        var_95 = self.calculate_var(portfolio_returns, total_value, self.var_confidence_level)

        # Calculate volatility
        volatility = self.calculate_volatility(portfolio_returns)

        # Check against limit
        if volatility > self.max_portfolio_volatility:
            notes.append(
                f"WARNING: Portfolio volatility ({volatility:.1%}) exceeds limit ({self.max_portfolio_volatility:.1%})"
            )

        # Calculate Sharpe ratio
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)

        # Calculate max drawdown (need portfolio value history)
        # For now, estimate from returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = cumulative_returns * total_value
        max_dd = self.calculate_max_drawdown(portfolio_values)

        # Calculate beta (if benchmark data available)
        beta = None
        try:
            benchmark_prices = self.get_historical_prices([self.benchmark_symbol], self.lookback_days)
            if not benchmark_prices.empty:
                benchmark_returns = self.calculate_returns(benchmark_prices)
                if self.benchmark_symbol in benchmark_returns.columns:
                    beta = self.calculate_beta(portfolio_returns, benchmark_returns[self.benchmark_symbol].values)
        except Exception as e:
            logger.warning(f"Could not calculate beta: {e}")

        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)

        # Check for high concentration
        if largest_position_pct > 0.30:
            notes.append(f"High concentration: largest position is {largest_position_pct:.1%} of portfolio")

        # Calculate leverage (for now, assume no leverage)
        leverage = 1.0

        return PortfolioRiskMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            var_95=var_95,
            volatility_30d=volatility,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            beta=beta,
            correlation_matrix=corr_matrix,
            largest_position_pct=largest_position_pct,
            num_positions=num_positions,
            leverage=leverage,
            notes="; ".join(notes) if notes else "All risk metrics within limits"
        )

    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits

        Returns:
            Tuple of (is_within_limits, list_of_violations)
        """
        metrics = self.get_portfolio_metrics()
        violations = []

        # Check volatility
        if metrics.volatility_30d > self.max_portfolio_volatility:
            violations.append(
                f"Portfolio volatility ({metrics.volatility_30d:.1%}) exceeds limit ({self.max_portfolio_volatility:.1%})"
            )

        # Check concentration
        if metrics.largest_position_pct > 0.30:
            violations.append(
                f"Largest position ({metrics.largest_position_pct:.1%}) exceeds recommended 30% limit"
            )

        # Check VaR relative to portfolio
        var_pct = metrics.var_95 / metrics.total_value if metrics.total_value > 0 else 0
        if var_pct > 0.10:  # 10% VaR threshold
            violations.append(
                f"VaR ({var_pct:.1%} of portfolio) exceeds 10% threshold"
            )

        is_within_limits = len(violations) == 0

        return is_within_limits, violations


def main():
    """Example usage"""
    print("=" * 60)
    print("Portfolio Risk Monitor")
    print("=" * 60)

    try:
        # Initialize monitor
        monitor = PortfolioRiskMonitor(
            max_portfolio_volatility=0.25,
            var_confidence_level=0.95,
            lookback_days=30
        )

        # Get risk metrics
        print("\nCalculating portfolio risk metrics...")
        metrics = monitor.get_portfolio_metrics()

        print(f"\nTimestamp: {metrics.timestamp}")
        print(f"Portfolio Value: ${metrics.total_value:,.2f}")
        print(f"Number of Positions: {metrics.num_positions}")
        print(f"Largest Position: {metrics.largest_position_pct:.1%}")
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%): ${metrics.var_95:,.2f} ({metrics.var_95/metrics.total_value:.1%} of portfolio)")
        print(f"  30-Day Volatility: {metrics.volatility_30d:.1%}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "  Sharpe Ratio: N/A")
        print(f"  Beta: {metrics.beta:.2f}" if metrics.beta else "  Beta: N/A")
        print(f"  Leverage: {metrics.leverage:.2f}x")
        print(f"\nNotes: {metrics.notes}")

        # Check risk limits
        print("\n" + "=" * 60)
        print("Risk Limit Check")
        print("=" * 60)

        is_within_limits, violations = monitor.check_risk_limits()

        if is_within_limits:
            print("\n✓ All risk limits are within acceptable ranges")
        else:
            print("\n✗ Risk limit violations detected:")
            for violation in violations:
                print(f"  - {violation}")

        # Display correlation matrix if available
        if metrics.correlation_matrix is not None and not metrics.correlation_matrix.empty:
            print("\n" + "=" * 60)
            print("Asset Correlation Matrix")
            print("=" * 60)
            print(metrics.correlation_matrix.round(2))

    except Exception as e:
        logger.error(f"Error running portfolio risk monitor: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
