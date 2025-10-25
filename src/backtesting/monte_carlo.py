from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Monte Carlo Simulation Framework
=================================

Runs thousands of Monte Carlo simulations to:
- Estimate probability distributions of returns
- Calculate risk of ruin
- Validate strategy robustness
- Generate confidence intervals
"""

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    strategy_id: str
    n_simulations: int

    # Return distribution
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    # Risk metrics
    probability_of_profit: float  # P(return > 0)
    probability_of_target: float  # P(return > target)
    risk_of_ruin: float           # P(drawdown > threshold)
    expected_max_drawdown: float

    # Sharpe distribution
    mean_sharpe: float
    std_sharpe: float
    percentile_5_sharpe: float
    percentile_95_sharpe: float

    # Confidence intervals
    return_ci_95: tuple  # 95% confidence interval for returns
    sharpe_ci_95: tuple  # 95% confidence interval for Sharpe

    # All simulation results
    all_returns: np.ndarray
    all_sharpes: np.ndarray
    all_max_drawdowns: np.ndarray

    def get_summary(self) -> str:
        """Get human-readable summary"""
        return f"""
Monte Carlo Simulation Results ({self.n_simulations:,} runs)
{'='*60}
Returns:
  Mean: {self.mean_return:.2%}
  Median: {self.median_return:.2%}
  95% CI: [{self.return_ci_95[0]:.2%}, {self.return_ci_95[1]:.2%}]
  5th percentile: {self.percentile_5:.2%}
  95th percentile: {self.percentile_95:.2%}

Sharpe Ratio:
  Mean: {self.mean_sharpe:.2f}
  95% CI: [{self.sharpe_ci_95[0]:.2f}, {self.sharpe_ci_95[1]:.2f}]

Risk:
  Probability of Profit: {self.probability_of_profit:.1%}
  Risk of Ruin: {self.risk_of_ruin:.2%}
  Expected Max Drawdown: {self.expected_max_drawdown:.1%}
"""


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for strategy validation

    Simulation methods:
    1. Bootstrap resampling of historical returns
    2. Parametric simulation (assuming distribution)
    3. Walk-forward Monte Carlo
    """

    def __init__(self,
                 n_simulations: int = 10000,
                 simulation_length: int = 252,  # 1 year of trading days
                 n_parallel_jobs: int = 4):
        """
        Initialize Monte Carlo simulator

        Args:
            n_simulations: Number of Monte Carlo runs
            simulation_length: Number of periods per simulation
            n_parallel_jobs: Number of parallel processes
        """
        self.n_simulations = n_simulations
        self.simulation_length = simulation_length
        self.n_parallel_jobs = n_parallel_jobs

    def run_bootstrap_simulation(self,
                                 historical_returns: pd.Series,
                                 strategy_id: str,
                                 target_return: float = 0.20,  # 20% target
                                 ruin_threshold: float = 0.50) -> MonteCarloResult:
        """
        Run Monte Carlo simulation using bootstrap resampling

        Args:
            historical_returns: Historical strategy returns
            strategy_id: Strategy identifier
            target_return: Target annual return for probability calculation
            ruin_threshold: Drawdown threshold for risk of ruin

        Returns:
            MonteCarloResult with simulation statistics
        """
        logger.info(f"Running {self.n_simulations:,} Monte Carlo simulations for {strategy_id}")

        # Remove NaN values
        returns_clean = historical_returns.dropna().values

        if len(returns_clean) < 30:
            logger.warning(f"Insufficient data for Monte Carlo: {len(returns_clean)} returns")
            return self._empty_result(strategy_id)

        # Run simulations in parallel
        chunk_size = self.n_simulations // self.n_parallel_jobs
        futures = []

        with ProcessPoolExecutor(max_workers=self.n_parallel_jobs) as executor:
            for i in range(self.n_parallel_jobs):
                n_sims = chunk_size if i < self.n_parallel_jobs - 1 else (self.n_simulations - chunk_size * i)

                future = executor.submit(
                    self._run_bootstrap_chunk,
                    returns_clean,
                    n_sims,
                    self.simulation_length
                )
                futures.append(future)

            # Collect results
            all_sim_returns = []
            all_sim_sharpes = []
            all_sim_drawdowns = []

            for future in as_completed(futures):
                chunk_returns, chunk_sharpes, chunk_drawdowns = future.result()
                all_sim_returns.extend(chunk_returns)
                all_sim_sharpes.extend(chunk_sharpes)
                all_sim_drawdowns.extend(chunk_drawdowns)

        # Convert to numpy arrays
        sim_returns = np.array(all_sim_returns)
        sim_sharpes = np.array(all_sim_sharpes)
        sim_drawdowns = np.array(all_sim_drawdowns)

        # Calculate statistics
        result = self._calculate_mc_statistics(
            strategy_id=strategy_id,
            sim_returns=sim_returns,
            sim_sharpes=sim_sharpes,
            sim_drawdowns=sim_drawdowns,
            target_return=target_return,
            ruin_threshold=ruin_threshold
        )

        logger.info(f"Monte Carlo complete:")
        logger.info(f"  Mean return: {result.mean_return:.2%}")
        logger.info(f"  Mean Sharpe: {result.mean_sharpe:.2f}")
        logger.info(f"  P(profit): {result.probability_of_profit:.1%}")
        logger.info(f"  Risk of ruin: {result.risk_of_ruin:.2%}")

        return result

    @staticmethod
    def _run_bootstrap_chunk(returns: np.ndarray,
                            n_simulations: int,
                            simulation_length: int) -> tuple:
        """Run a chunk of bootstrap simulations (for parallel execution)"""
        chunk_returns = []
        chunk_sharpes = []
        chunk_drawdowns = []

        for _ in range(n_simulations):
            # Bootstrap resample
            sim_returns = np.random.choice(returns, size=simulation_length, replace=True)

            # Calculate cumulative return
            cumulative_return = np.prod(1 + sim_returns) - 1

            # Calculate Sharpe ratio
            if len(sim_returns) > 1 and np.std(sim_returns) > 0:
                sharpe = (np.mean(sim_returns) / np.std(sim_returns)) * np.sqrt(252)
            else:
                sharpe = 0

            # Calculate max drawdown
            cumulative = np.cumprod(1 + sim_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            chunk_returns.append(cumulative_return)
            chunk_sharpes.append(sharpe)
            chunk_drawdowns.append(max_drawdown)

        return chunk_returns, chunk_sharpes, chunk_drawdowns

    def run_parametric_simulation(self,
                                 mean_return: float,
                                 std_return: float,
                                 strategy_id: str,
                                 distribution: str = 'normal',
                                 target_return: float = 0.20,
                                 ruin_threshold: float = 0.50) -> MonteCarloResult:
        """
        Run parametric Monte Carlo simulation

        Assumes returns follow a specific distribution (normal, t-distribution, etc.)

        Args:
            mean_return: Expected daily return
            std_return: Daily return standard deviation
            strategy_id: Strategy identifier
            distribution: 'normal' or 't' (Student's t)
            target_return: Target annual return
            ruin_threshold: Drawdown threshold for risk of ruin

        Returns:
            MonteCarloResult
        """
        logger.info(f"Running parametric Monte Carlo ({distribution} distribution)")

        sim_returns = []
        sim_sharpes = []
        sim_drawdowns = []

        for _ in range(self.n_simulations):
            # Generate returns from distribution
            if distribution == 'normal':
                returns = np.random.normal(mean_return, std_return, self.simulation_length)
            elif distribution == 't':
                # Use t-distribution with df=5 (heavier tails)
                returns = stats.t.rvs(df=5, loc=mean_return, scale=std_return, size=self.simulation_length)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            # Calculate metrics
            cumulative_return = np.prod(1 + returns) - 1

            if np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0

            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            sim_returns.append(cumulative_return)
            sim_sharpes.append(sharpe)
            sim_drawdowns.append(max_drawdown)

        # Calculate statistics
        result = self._calculate_mc_statistics(
            strategy_id=strategy_id,
            sim_returns=np.array(sim_returns),
            sim_sharpes=np.array(sim_sharpes),
            sim_drawdowns=np.array(sim_drawdowns),
            target_return=target_return,
            ruin_threshold=ruin_threshold
        )

        return result

    def _calculate_mc_statistics(self,
                                strategy_id: str,
                                sim_returns: np.ndarray,
                                sim_sharpes: np.ndarray,
                                sim_drawdowns: np.ndarray,
                                target_return: float,
                                ruin_threshold: float) -> MonteCarloResult:
        """Calculate statistics from Monte Carlo results"""

        # Return statistics
        mean_return = np.mean(sim_returns)
        median_return = np.median(sim_returns)
        std_return = np.std(sim_returns)

        percentiles = np.percentile(sim_returns, [5, 25, 75, 95])
        percentile_5, percentile_25, percentile_75, percentile_95 = percentiles

        # Probabilities
        probability_of_profit = np.mean(sim_returns > 0)
        probability_of_target = np.mean(sim_returns > target_return)
        risk_of_ruin = np.mean(sim_drawdowns < -ruin_threshold)

        # Sharpe statistics
        mean_sharpe = np.mean(sim_sharpes)
        std_sharpe = np.std(sim_sharpes)
        sharpe_percentiles = np.percentile(sim_sharpes, [5, 95])
        percentile_5_sharpe, percentile_95_sharpe = sharpe_percentiles

        # Drawdown statistics
        expected_max_drawdown = np.mean(sim_drawdowns)

        # Confidence intervals (95%)
        return_ci_95 = tuple(np.percentile(sim_returns, [2.5, 97.5]))
        sharpe_ci_95 = tuple(np.percentile(sim_sharpes, [2.5, 97.5]))

        return MonteCarloResult(
            strategy_id=strategy_id,
            n_simulations=len(sim_returns),
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            probability_of_profit=probability_of_profit,
            probability_of_target=probability_of_target,
            risk_of_ruin=risk_of_ruin,
            expected_max_drawdown=expected_max_drawdown,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            percentile_5_sharpe=percentile_5_sharpe,
            percentile_95_sharpe=percentile_95_sharpe,
            return_ci_95=return_ci_95,
            sharpe_ci_95=sharpe_ci_95,
            all_returns=sim_returns,
            all_sharpes=sim_sharpes,
            all_max_drawdowns=sim_drawdowns
        )

    def _empty_result(self, strategy_id: str) -> MonteCarloResult:
        """Return empty result when simulation cannot be run"""
        return MonteCarloResult(
            strategy_id=strategy_id,
            n_simulations=0,
            mean_return=0.0,
            median_return=0.0,
            std_return=0.0,
            percentile_5=0.0,
            percentile_25=0.0,
            percentile_75=0.0,
            percentile_95=0.0,
            probability_of_profit=0.0,
            probability_of_target=0.0,
            risk_of_ruin=1.0,
            expected_max_drawdown=0.0,
            mean_sharpe=0.0,
            std_sharpe=0.0,
            percentile_5_sharpe=0.0,
            percentile_95_sharpe=0.0,
            return_ci_95=(0.0, 0.0),
            sharpe_ci_95=(0.0, 0.0),
            all_returns=np.array([]),
            all_sharpes=np.array([]),
            all_max_drawdowns=np.array([])
        )

    def compare_strategies(self,
                          results: List[MonteCarloResult]) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            results: List of MonteCarloResult

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for result in results:
            comparison_data.append({
                'strategy_id': result.strategy_id,
                'mean_return': result.mean_return,
                'median_return': result.median_return,
                'return_95_ci_lower': result.return_ci_95[0],
                'return_95_ci_upper': result.return_ci_95[1],
                'mean_sharpe': result.mean_sharpe,
                'sharpe_95_ci_lower': result.sharpe_ci_95[0],
                'sharpe_95_ci_upper': result.sharpe_ci_95[1],
                'prob_profit': result.probability_of_profit,
                'risk_of_ruin': result.risk_of_ruin,
                'expected_max_dd': result.expected_max_drawdown
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('mean_sharpe', ascending=False)

        return df
