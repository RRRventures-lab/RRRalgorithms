from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import numpy as np
import pandas as pd

"""
Monte Carlo Simulation

Tests strategy robustness by:
1. Shuffling trade order
2. Running thousands of simulations
3. Calculating probability distributions
4. Estimating risk of ruin
"""


logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    final_equity_dist: np.ndarray
    total_return_dist: np.ndarray
    max_drawdown_dist: np.ndarray
    sharpe_ratio_dist: np.ndarray

    # Statistics
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    # Risk metrics
    probability_of_profit: float
    probability_of_ruin: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)

    # Confidence intervals
    return_ci_95: tuple
    sharpe_ci_95: tuple
    drawdown_ci_95: tuple


class MonteCarloSimulator:
    """
    Monte Carlo simulator for strategy robustness testing.

    Tests strategy by randomly reordering trades to:
    - Assess impact of trade sequence
    - Calculate probability distributions
    - Estimate risk of ruin
    - Generate confidence intervals
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        initial_capital: float = 100000.0,
        ruin_threshold: float = 0.5,  # 50% drawdown = ruin
        n_jobs: int = -1  # -1 = use all cores
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run
            initial_capital: Starting capital
            ruin_threshold: Drawdown threshold for ruin (0.5 = 50%)
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.n_simulations = n_simulations
        self.initial_capital = initial_capital
        self.ruin_threshold = ruin_threshold
        self.n_jobs = n_jobs if n_jobs > 0 else None

        logger.info(
            f"Initialized MonteCarloSimulator: "
            f"{n_simulations} simulations, ${initial_capital:,.2f} initial capital"
        )

    def run_simulation(
        self,
        trades: List,
        method: str = 'bootstrap'
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on trade results.

        Args:
            trades: List of Trade objects from backtest
            method: Simulation method ('bootstrap', 'block_bootstrap')

        Returns:
            MonteCarloResult with simulation statistics
        """
        logger.info(f"Starting Monte Carlo simulation with {self.n_simulations} iterations")

        # Extract trade P&Ls
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]

        if len(trade_pnls) < 10:
            logger.warning(f"Only {len(trade_pnls)} trades available - results may not be reliable")

        logger.info(f"Running simulation with {len(trade_pnls)} trades")

        # Run simulations
        if method == 'bootstrap':
            results = self._run_bootstrap_simulations(trade_pnls)
        elif method == 'block_bootstrap':
            results = self._run_block_bootstrap_simulations(trade_pnls)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Analyze results
        monte_carlo_result = self._analyze_results(results)

        logger.info("Monte Carlo simulation complete")
        return monte_carlo_result

    def _run_bootstrap_simulations(self, trade_pnls: List[float]) -> List[Dict]:
        """
        Run bootstrap simulations (sample with replacement).

        Args:
            trade_pnls: List of trade P&Ls

        Returns:
            List of simulation results
        """
        results = []

        for i in range(self.n_simulations):
            # Sample trades with replacement
            sampled_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)

            # Calculate equity curve
            equity_curve = np.zeros(len(sampled_pnls) + 1)
            equity_curve[0] = self.initial_capital

            for j, pnl in enumerate(sampled_pnls):
                equity_curve[j + 1] = equity_curve[j] + pnl

            # Calculate metrics
            result = self._calculate_simulation_metrics(equity_curve, sampled_pnls)
            results.append(result)

            if (i + 1) % 100 == 0:
                logger.debug(f"Completed {i + 1}/{self.n_simulations} simulations")

        return results

    def _run_block_bootstrap_simulations(
        self,
        trade_pnls: List[float],
        block_size: int = 5
    ) -> List[Dict]:
        """
        Run block bootstrap simulations (preserve sequential patterns).

        Args:
            trade_pnls: List of trade P&Ls
            block_size: Size of blocks to sample

        Returns:
            List of simulation results
        """
        results = []
        n_trades = len(trade_pnls)

        for i in range(self.n_simulations):
            # Create blocks
            sampled_pnls = []
            while len(sampled_pnls) < n_trades:
                # Random starting point
                start_idx = np.random.randint(0, max(1, n_trades - block_size))
                end_idx = min(start_idx + block_size, n_trades)

                # Add block
                sampled_pnls.extend(trade_pnls[start_idx:end_idx])

            # Trim to correct length
            sampled_pnls = sampled_pnls[:n_trades]

            # Calculate equity curve
            equity_curve = np.zeros(len(sampled_pnls) + 1)
            equity_curve[0] = self.initial_capital

            for j, pnl in enumerate(sampled_pnls):
                equity_curve[j + 1] = equity_curve[j] + pnl

            # Calculate metrics
            result = self._calculate_simulation_metrics(equity_curve, sampled_pnls)
            results.append(result)

            if (i + 1) % 100 == 0:
                logger.debug(f"Completed {i + 1}/{self.n_simulations} simulations")

        return results

    def _calculate_simulation_metrics(
        self,
        equity_curve: np.ndarray,
        trade_pnls: np.ndarray
    ) -> Dict:
        """
        Calculate metrics for a single simulation.

        Args:
            equity_curve: Array of equity values
            trade_pnls: Array of trade P&Ls

        Returns:
            Dictionary of metrics
        """
        final_equity = equity_curve[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Calculate max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calculate Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Check for ruin
        is_ruined = max_drawdown < (-self.ruin_threshold * 100)

        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'is_ruined': is_ruined
        }

    def _analyze_results(self, results: List[Dict]) -> MonteCarloResult:
        """
        Analyze simulation results and calculate statistics.

        Args:
            results: List of simulation result dictionaries

        Returns:
            MonteCarloResult with statistics
        """
        # Extract distributions
        final_equity_dist = np.array([r['final_equity'] for r in results])
        total_return_dist = np.array([r['total_return'] for r in results])
        max_drawdown_dist = np.array([r['max_drawdown'] for r in results])
        sharpe_ratio_dist = np.array([r['sharpe_ratio'] for r in results])

        # Calculate statistics
        mean_return = np.mean(total_return_dist)
        median_return = np.median(total_return_dist)
        std_return = np.std(total_return_dist)

        # Percentiles
        percentile_5 = np.percentile(total_return_dist, 5)
        percentile_25 = np.percentile(total_return_dist, 25)
        percentile_75 = np.percentile(total_return_dist, 75)
        percentile_95 = np.percentile(total_return_dist, 95)

        # Risk metrics
        probability_of_profit = np.mean(total_return_dist > 0)
        probability_of_ruin = np.mean([r['is_ruined'] for r in results])

        # Value at Risk (95%)
        var_95 = np.percentile(total_return_dist, 5)  # 5th percentile

        # Conditional VaR (95%) - average of worst 5%
        worst_5pct = total_return_dist[total_return_dist <= var_95]
        cvar_95 = np.mean(worst_5pct) if len(worst_5pct) > 0 else var_95

        # Confidence intervals (95%)
        return_ci_95 = (np.percentile(total_return_dist, 2.5), np.percentile(total_return_dist, 97.5))
        sharpe_ci_95 = (np.percentile(sharpe_ratio_dist, 2.5), np.percentile(sharpe_ratio_dist, 97.5))
        drawdown_ci_95 = (np.percentile(max_drawdown_dist, 2.5), np.percentile(max_drawdown_dist, 97.5))

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            final_equity_dist=final_equity_dist,
            total_return_dist=total_return_dist,
            max_drawdown_dist=max_drawdown_dist,
            sharpe_ratio_dist=sharpe_ratio_dist,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            probability_of_profit=probability_of_profit,
            probability_of_ruin=probability_of_ruin,
            var_95=var_95,
            cvar_95=cvar_95,
            return_ci_95=return_ci_95,
            sharpe_ci_95=sharpe_ci_95,
            drawdown_ci_95=drawdown_ci_95
        )

    def print_summary(self, result: MonteCarloResult):
        """
        Print formatted summary of Monte Carlo results.

        Args:
            result: MonteCarloResult object
        """
        print("\n" + "=" * 70)
        print("MONTE CARLO SIMULATION SUMMARY")
        print("=" * 70)
        print(f"\nNumber of simulations: {result.n_simulations}")

        print("\n" + "-" * 70)
        print("RETURN DISTRIBUTION")
        print("-" * 70)
        print(f"{'Mean Return':<30} {result.mean_return:>12.2f}%")
        print(f"{'Median Return':<30} {result.median_return:>12.2f}%")
        print(f"{'Std Deviation':<30} {result.std_return:>12.2f}%")
        print(f"{'5th Percentile':<30} {result.percentile_5:>12.2f}%")
        print(f"{'25th Percentile':<30} {result.percentile_25:>12.2f}%")
        print(f"{'75th Percentile':<30} {result.percentile_75:>12.2f}%")
        print(f"{'95th Percentile':<30} {result.percentile_95:>12.2f}%")

        print("\n" + "-" * 70)
        print("RISK METRICS")
        print("-" * 70)
        print(f"{'Probability of Profit':<30} {result.probability_of_profit * 100:>12.2f}%")
        print(f"{'Probability of Ruin':<30} {result.probability_of_ruin * 100:>12.2f}%")
        print(f"{'Value at Risk (95%)':<30} {result.var_95:>12.2f}%")
        print(f"{'Conditional VaR (95%)':<30} {result.cvar_95:>12.2f}%")

        print("\n" + "-" * 70)
        print("CONFIDENCE INTERVALS (95%)")
        print("-" * 70)
        print(
            f"{'Return':<30} [{result.return_ci_95[0]:>7.2f}%, {result.return_ci_95[1]:>7.2f}%]"
        )
        print(
            f"{'Sharpe Ratio':<30} [{result.sharpe_ci_95[0]:>7.2f}, {result.sharpe_ci_95[1]:>7.2f}]"
        )
        print(
            f"{'Max Drawdown':<30} [{result.drawdown_ci_95[0]:>7.2f}%, {result.drawdown_ci_95[1]:>7.2f}%]"
        )

        print("\n" + "=" * 70 + "\n")
