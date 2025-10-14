from dataclasses import dataclass
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

"""
QAOA-Inspired Portfolio Optimizer

This module implements a Quantum Approximate Optimization Algorithm (QAOA) simulation
for portfolio optimization. It runs on classical hardware but uses quantum-inspired
techniques to find optimal portfolio weights.

The algorithm is based on:
- Alternating application of problem Hamiltonian and mixer Hamiltonian
- Variational optimization of parameters
- Classical simulation of quantum circuits

Author: Quantum Optimization Team
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    sum_to_one: bool = True  # Weights must sum to 1
    long_only: bool = True   # No short positions
    risk_tolerance: float = 1.0  # Risk aversion parameter


@dataclass
class OptimizationResult:
    """Results from portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    iterations: int
    converged: bool
    execution_time: float


class QAOAPortfolioOptimizer:
    """
    Quantum-Inspired Portfolio Optimizer using QAOA principles

    This optimizer uses variational quantum-inspired techniques to find
    optimal portfolio allocations that maximize risk-adjusted returns.
    """

    def __init__(self,
                 n_layers: int = 3,
                 n_iterations: int = 100,
                 learning_rate: float = 0.1):
        """
        Initialize QAOA optimizer

        Args:
            n_layers: Number of QAOA layers (depth)
            n_iterations: Maximum optimization iterations
            learning_rate: Learning rate for parameter updates
        """
        self.n_layers = n_layers
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.optimization_history = []

    def _problem_hamiltonian(self,
                            weights: np.ndarray,
                            expected_returns: np.ndarray,
                            covariance: np.ndarray,
                            risk_aversion: float) -> float:
        """
        Problem Hamiltonian: encodes the portfolio optimization objective

        H_P = -μ^T w + λ w^T Σ w

        Where:
        - w: portfolio weights
        - μ: expected returns
        - Σ: covariance matrix
        - λ: risk aversion parameter
        """
        return_term = -np.dot(expected_returns, weights)
        risk_term = risk_aversion * np.dot(weights, np.dot(covariance, weights))
        return return_term + risk_term

    def _mixer_hamiltonian(self, weights: np.ndarray) -> np.ndarray:
        """
        Mixer Hamiltonian: explores solution space

        Applies mixing operations to explore different weight configurations
        """
        n_assets = len(weights)
        # Random mixing with small perturbations
        mixing_matrix = np.random.randn(n_assets, n_assets) * 0.1
        mixing_matrix = (mixing_matrix + mixing_matrix.T) / 2  # Make symmetric
        return np.dot(mixing_matrix, weights)

    def _apply_qaoa_layer(self,
                         weights: np.ndarray,
                         gamma: float,
                         beta: float,
                         expected_returns: np.ndarray,
                         covariance: np.ndarray,
                         risk_aversion: float) -> np.ndarray:
        """
        Apply one QAOA layer: problem Hamiltonian + mixer Hamiltonian

        Args:
            weights: Current portfolio weights
            gamma: Problem Hamiltonian parameter
            beta: Mixer Hamiltonian parameter
            expected_returns: Expected returns vector
            covariance: Covariance matrix
            risk_aversion: Risk aversion parameter

        Returns:
            Updated weights after QAOA layer
        """
        # Apply problem Hamiltonian (gradient descent step)
        gradient = -expected_returns + 2 * risk_aversion * np.dot(covariance, weights)
        weights_after_problem = weights - gamma * gradient

        # Apply mixer Hamiltonian (exploration step)
        mixing = self._mixer_hamiltonian(weights_after_problem)
        weights_after_mixer = weights_after_problem + beta * mixing

        return weights_after_mixer

    def _project_to_constraints(self,
                                weights: np.ndarray,
                                constraints: PortfolioConstraints) -> np.ndarray:
        """
        Project weights to satisfy constraints

        Args:
            weights: Raw portfolio weights
            constraints: Portfolio constraints

        Returns:
            Constrained weights
        """
        # Apply bounds
        if constraints.long_only:
            weights = np.maximum(weights, 0)
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)

        # Normalize to sum to 1
        if constraints.sum_to_one:
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Equal weights if all zero
                weights = np.ones_like(weights) / len(weights)

        return weights

    def optimize(self,
                expected_returns: np.ndarray,
                covariance: np.ndarray,
                constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio using QAOA-inspired algorithm

        Args:
            expected_returns: Expected returns for each asset (n_assets,)
            covariance: Covariance matrix (n_assets, n_assets)
            constraints: Portfolio constraints

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        import time
        start_time = time.time()

        if constraints is None:
            constraints = PortfolioConstraints()

        n_assets = len(expected_returns)

        # Initialize weights (equal weight)
        weights = np.ones(n_assets) / n_assets

        # Initialize QAOA parameters
        gamma_params = np.random.uniform(0, 2*np.pi, self.n_layers)
        beta_params = np.random.uniform(0, np.pi, self.n_layers)

        def objective(params):
            """Objective function for parameter optimization"""
            gamma = params[:self.n_layers]
            beta = params[self.n_layers:]

            # Apply QAOA circuit
            current_weights = weights.copy()
            for i in range(self.n_layers):
                current_weights = self._apply_qaoa_layer(
                    current_weights,
                    gamma[i],
                    beta[i],
                    expected_returns,
                    covariance,
                    constraints.risk_tolerance
                )
                current_weights = self._project_to_constraints(current_weights, constraints)

            # Evaluate objective (negative for minimization)
            return self._problem_hamiltonian(
                current_weights,
                expected_returns,
                covariance,
                constraints.risk_tolerance
            )

        # Optimize QAOA parameters
        initial_params = np.concatenate([gamma_params, beta_params])
        result = minimize(
            objective,
            initial_params,
            method='COBYLA',
            options={'maxiter': self.n_iterations}
        )

        # Get optimal weights using optimized parameters
        optimal_gamma = result.x[:self.n_layers]
        optimal_beta = result.x[self.n_layers:]

        optimal_weights = weights.copy()
        for i in range(self.n_layers):
            optimal_weights = self._apply_qaoa_layer(
                optimal_weights,
                optimal_gamma[i],
                optimal_beta[i],
                expected_returns,
                covariance,
                constraints.risk_tolerance
            )
            optimal_weights = self._project_to_constraints(optimal_weights, constraints)

        # Calculate performance metrics
        expected_return = np.dot(optimal_weights, expected_returns)
        volatility = np.sqrt(np.dot(optimal_weights, np.dot(covariance, optimal_weights)))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        execution_time = time.time() - start_time

        logger.info(f"QAOA optimization completed in {execution_time:.2f}s")
        logger.info(f"Expected return: {expected_return:.4f}, Volatility: {volatility:.4f}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")

        return OptimizationResult(
            weights=optimal_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            iterations=result.nit,
            converged=result.success,
            execution_time=execution_time
        )


class ClassicalMarkowitzOptimizer:
    """
    Classical Markowitz Mean-Variance Optimizer for comparison
    """

    def optimize(self,
                expected_returns: np.ndarray,
                covariance: np.ndarray,
                constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio using classical Markowitz approach

        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix
            constraints: Portfolio constraints

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        import time
        start_time = time.time()

        if constraints is None:
            constraints = PortfolioConstraints()

        n_assets = len(expected_returns)

        def objective(weights):
            """Mean-variance objective"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            # Negative Sharpe ratio (for minimization)
            return -(portfolio_return - constraints.risk_tolerance * portfolio_variance)

        # Constraints
        cons = []
        if constraints.sum_to_one:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        if constraints.long_only:
            bounds = [(max(0, b[0]), b[1]) for b in bounds]

        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )

        optimal_weights = result.x

        # Calculate metrics
        expected_return = np.dot(optimal_weights, expected_returns)
        volatility = np.sqrt(np.dot(optimal_weights, np.dot(covariance, optimal_weights)))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        execution_time = time.time() - start_time

        logger.info(f"Classical optimization completed in {execution_time:.2f}s")
        logger.info(f"Expected return: {expected_return:.4f}, Volatility: {volatility:.4f}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")

        return OptimizationResult(
            weights=optimal_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            iterations=result.nit,
            converged=result.success,
            execution_time=execution_time
        )


def compare_optimizers(expected_returns: np.ndarray,
                      covariance: np.ndarray,
                      constraints: Optional[PortfolioConstraints] = None) -> Dict:
    """
    Compare QAOA vs Classical optimizer performance

    Args:
        expected_returns: Expected returns vector
        covariance: Covariance matrix
        constraints: Portfolio constraints

    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing QAOA vs Classical Markowitz optimizer...")

    qaoa = QAOAPortfolioOptimizer(n_layers=3, n_iterations=100)
    classical = ClassicalMarkowitzOptimizer()

    logger.info("\n=== QAOA Optimizer ===")
    qaoa_result = qaoa.optimize(expected_returns, covariance, constraints)

    logger.info("\n=== Classical Optimizer ===")
    classical_result = classical.optimize(expected_returns, covariance, constraints)

    comparison = {
        'qaoa': {
            'weights': qaoa_result.weights,
            'expected_return': qaoa_result.expected_return,
            'volatility': qaoa_result.volatility,
            'sharpe_ratio': qaoa_result.sharpe_ratio,
            'execution_time': qaoa_result.execution_time,
            'converged': qaoa_result.converged
        },
        'classical': {
            'weights': classical_result.weights,
            'expected_return': classical_result.expected_return,
            'volatility': classical_result.volatility,
            'sharpe_ratio': classical_result.sharpe_ratio,
            'execution_time': classical_result.execution_time,
            'converged': classical_result.converged
        },
        'performance_ratio': {
            'sharpe_improvement': (qaoa_result.sharpe_ratio / classical_result.sharpe_ratio - 1) * 100
                                   if classical_result.sharpe_ratio != 0 else 0,
            'speed_ratio': classical_result.execution_time / qaoa_result.execution_time
                          if qaoa_result.execution_time > 0 else 0
        }
    }

    logger.info("\n=== Comparison Summary ===")
    logger.info(f"Sharpe improvement (QAOA vs Classical): {comparison['performance_ratio']['sharpe_improvement']:.2f}%")
    logger.info(f"Speed ratio (Classical/QAOA): {comparison['performance_ratio']['speed_ratio']:.2f}x")

    return comparison


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data (5 assets)
    n_assets = 5
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)

    # Generate random covariance matrix
    random_matrix = np.random.randn(n_assets, n_assets)
    covariance = np.dot(random_matrix, random_matrix.T) * 0.01

    # Set constraints
    constraints = PortfolioConstraints(
        min_weight=0.05,
        max_weight=0.5,
        long_only=True,
        risk_tolerance=1.0
    )

    # Compare optimizers
    results = compare_optimizers(expected_returns, covariance, constraints)

    print("\nOptimal Weights (QAOA):")
    print(results['qaoa']['weights'])
    print("\nOptimal Weights (Classical):")
    print(results['classical']['weights'])
