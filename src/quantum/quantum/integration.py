from .features import (
from .hyperparameter import (
from .portfolio import (
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import numpy as np


"""
Quantum Optimization Integration Module

This module provides a unified interface for quantum-inspired optimization
algorithms that can be easily used by other components of the trading system.

Usage Examples:

    # Portfolio optimization
    from quantum.integration import optimize_portfolio
    weights = optimize_portfolio(expected_returns, covariance_matrix)

    # Hyperparameter tuning
    from quantum.integration import tune_hyperparameters
    best_params = tune_hyperparameters(param_spaces, objective_function)

    # Feature selection
    from quantum.integration import select_features
    selected_features = select_features(X, y)

Author: Quantum Optimization Team
Date: 2025-10-11
"""


    QAOAPortfolioOptimizer,
    ClassicalMarkowitzOptimizer,
    PortfolioConstraints,
    OptimizationResult
)
    QuantumAnnealingTuner,
    HyperparameterSpace,
    TuningResult
)
    QuantumFeatureSelector,
    ClassicalFeatureSelector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Portfolio Optimization Interface
# ============================================================================

def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    risk_tolerance: float = 1.0,
    use_quantum: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize portfolio allocation using quantum-inspired or classical methods

    Args:
        expected_returns: Expected returns for each asset (n_assets,)
        covariance_matrix: Covariance matrix (n_assets, n_assets)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        risk_tolerance: Risk aversion parameter (higher = more risk-averse)
        use_quantum: If True, use QAOA; if False, use classical Markowitz
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Dictionary containing:
            - weights: Optimal portfolio weights
            - expected_return: Expected portfolio return
            - volatility: Portfolio volatility (risk)
            - sharpe_ratio: Risk-adjusted return metric
            - execution_time: Optimization time
            - method: Optimization method used

    Example:
        >>> returns = np.array([0.10, 0.12, 0.08, 0.15])
        >>> cov = np.array([[0.01, 0, 0, 0],
        ...                 [0, 0.02, 0, 0],
        ...                 [0, 0, 0.015, 0],
        ...                 [0, 0, 0, 0.025]])
        >>> result = optimize_portfolio(returns, cov, risk_tolerance=1.0)
        >>> print(result['weights'])
        >>> print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
    """
    constraints = PortfolioConstraints(
        min_weight=min_weight,
        max_weight=max_weight,
        sum_to_one=True,
        long_only=True,
        risk_tolerance=risk_tolerance
    )

    if use_quantum:
        logger.info("Using QAOA portfolio optimizer")
        optimizer = QAOAPortfolioOptimizer(
            n_layers=kwargs.get('n_layers', 3),
            n_iterations=kwargs.get('n_iterations', 100)
        )
    else:
        logger.info("Using classical Markowitz optimizer")
        optimizer = ClassicalMarkowitzOptimizer()

    result = optimizer.optimize(expected_returns, covariance_matrix, constraints)

    return {
        'weights': result.weights,
        'expected_return': result.expected_return,
        'volatility': result.volatility,
        'sharpe_ratio': result.sharpe_ratio,
        'execution_time': result.execution_time,
        'converged': result.converged,
        'method': 'QAOA' if use_quantum else 'Classical_Markowitz'
    }


def compare_portfolio_optimizers(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare quantum and classical portfolio optimizers

    Args:
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from both optimizers and comparison metrics

    Example:
        >>> comparison = compare_portfolio_optimizers(returns, cov)
        >>> print(f"Quantum Sharpe: {comparison['qaoa']['sharpe_ratio']:.4f}")
        >>> print(f"Classical Sharpe: {comparison['classical']['sharpe_ratio']:.4f}")
        >>> print(f"Speedup: {comparison['speedup']:.2f}x")
    """
    quantum_result = optimize_portfolio(expected_returns, covariance_matrix, use_quantum=True, **kwargs)
    classical_result = optimize_portfolio(expected_returns, covariance_matrix, use_quantum=False, **kwargs)

    speedup = classical_result['execution_time'] / quantum_result['execution_time']
    sharpe_improvement = (quantum_result['sharpe_ratio'] / classical_result['sharpe_ratio'] - 1) * 100

    return {
        'qaoa': quantum_result,
        'classical': classical_result,
        'speedup': speedup,
        'sharpe_improvement_percent': sharpe_improvement
    }


# ============================================================================
# Hyperparameter Tuning Interface
# ============================================================================

def tune_hyperparameters(
    param_spaces: List[Dict[str, Any]],
    objective_function: Callable[[Dict[str, Any]], float],
    maximize: bool = True,
    use_quantum: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Tune hyperparameters using quantum-inspired or classical methods

    Args:
        param_spaces: List of parameter space definitions, each with keys:
            - 'name': Parameter name
            - 'type': 'continuous', 'discrete', or 'categorical'
            - 'bounds': (min, max) for continuous/discrete
            - 'values': List for categorical
            - 'log_scale': (optional) Use log scale for continuous
        objective_function: Function to optimize (params -> score)
        maximize: Whether to maximize (True) or minimize (False)
        use_quantum: If True, use quantum annealing; if False, use grid search
        **kwargs: Additional tuner-specific parameters

    Returns:
        Dictionary containing:
            - best_params: Best parameter configuration
            - best_score: Best objective value
            - optimization_time: Time taken
            - n_evaluations: Number of function evaluations
            - method: Tuning method used

    Example:
        >>> param_spaces = [
        ...     {'name': 'learning_rate', 'type': 'continuous',
        ...      'bounds': (0.001, 0.1), 'log_scale': True},
        ...     {'name': 'n_layers', 'type': 'discrete', 'bounds': (1, 5)},
        ...     {'name': 'activation', 'type': 'categorical',
        ...      'values': ['relu', 'tanh', 'sigmoid']}
        ... ]
        >>> def objective(params):
        ...     # Train model and return validation score
        ...     return model.score(params)
        >>> result = tune_hyperparameters(param_spaces, objective)
        >>> print(result['best_params'])
    """
    # Convert param_spaces dict format to HyperparameterSpace objects
    spaces = []
    for space_dict in param_spaces:
        space = HyperparameterSpace(
            name=space_dict['name'],
            param_type=space_dict['type'],
            bounds=space_dict.get('bounds'),
            values=space_dict.get('values'),
            log_scale=space_dict.get('log_scale', False)
        )
        spaces.append(space)

    if use_quantum:
        logger.info("Using Quantum Annealing tuner")
        tuner = QuantumAnnealingTuner(
            param_spaces=spaces,
            n_qubits=kwargs.get('n_qubits', 10),
            n_iterations=kwargs.get('n_iterations', 100),
            n_parallel=kwargs.get('n_parallel', 4)
        )
    else:
        logger.info("Using Grid Search tuner")
        from .hyperparameter import GridSearchTuner
        tuner = GridSearchTuner(
            param_spaces=spaces,
            n_points=kwargs.get('n_points', 5)
        )

    result = tuner.tune(objective_function, maximize=maximize)

    return {
        'best_params': result.best_params,
        'best_score': result.best_score,
        'optimization_time': result.optimization_time,
        'n_evaluations': result.n_evaluations,
        'convergence_history': result.convergence_history,
        'method': 'Quantum_Annealing' if use_quantum else 'Grid_Search'
    }


# ============================================================================
# Feature Selection Interface
# ============================================================================

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    n_features: Optional[int] = None,
    estimator: Optional[Any] = None,
    use_quantum: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Select optimal features using quantum-inspired or classical methods

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        n_features: Number of features to select (None = auto)
        estimator: Optional ML estimator for fitness evaluation
        use_quantum: If True, use quantum selector; if False, use classical
        **kwargs: Additional selector-specific parameters

    Returns:
        Dictionary containing:
            - selected_features: Indices of selected features
            - feature_scores: Importance score for each feature
            - X_transformed: Transformed feature matrix (optional)
            - optimization_time: Time taken
            - method: Selection method used

    Example:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> X, y = make_classification(n_samples=200, n_features=50)
        >>> result = select_features(X, y, n_features=10,
        ...                          estimator=RandomForestClassifier())
        >>> print(f"Selected features: {result['selected_features']}")
        >>> X_reduced = result['X_transformed']
    """
    if use_quantum:
        logger.info("Using Quantum Feature Selector")
        selector = QuantumFeatureSelector(
            n_features_to_select=n_features,
            n_iterations=kwargs.get('n_iterations', 50),
            population_size=kwargs.get('population_size', 20)
        )
    else:
        logger.info("Using Classical Feature Selector")
        selector = ClassicalFeatureSelector(n_features_to_select=n_features)

    selector.fit(X, y, estimator)
    X_transformed = selector.transform(X)

    return {
        'selected_features': selector.selected_features_,
        'feature_scores': selector.feature_importance_,
        'X_transformed': X_transformed,
        'optimization_time': selector.optimization_time_,
        'n_features_selected': len(selector.selected_features_),
        'method': 'Quantum' if use_quantum else 'Classical'
    }


# ============================================================================
# Unified Optimization Interface
# ============================================================================

class QuantumOptimizer:
    """
    Unified interface for all quantum-inspired optimization algorithms

    This class provides a single entry point for portfolio optimization,
    hyperparameter tuning, and feature selection.
    """

    def __init__(self, use_quantum: bool = True):
        """
        Initialize quantum optimizer

        Args:
            use_quantum: If True, use quantum-inspired algorithms;
                        if False, use classical algorithms
        """
        self.use_quantum = use_quantum
        logger.info(f"Initialized QuantumOptimizer (mode: {'quantum' if use_quantum else 'classical'})")

    def optimize_portfolio(self, *args, **kwargs) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        return optimize_portfolio(*args, use_quantum=self.use_quantum, **kwargs)

    def tune_hyperparameters(self, *args, **kwargs) -> Dict[str, Any]:
        """Tune model hyperparameters"""
        return tune_hyperparameters(*args, use_quantum=self.use_quantum, **kwargs)

    def select_features(self, *args, **kwargs) -> Dict[str, Any]:
        """Select optimal features"""
        return select_features(*args, use_quantum=self.use_quantum, **kwargs)


# ============================================================================
# Utilities
# ============================================================================

@lru_cache(maxsize=128)

def get_default_portfolio_constraints(
    risk_profile: str = 'balanced'
) -> Dict[str, float]:
    """
    Get default portfolio constraints based on risk profile

    Args:
        risk_profile: 'conservative', 'balanced', or 'aggressive'

    Returns:
        Dictionary with constraint parameters

    Example:
        >>> constraints = get_default_portfolio_constraints('conservative')
        >>> result = optimize_portfolio(returns, cov, **constraints)
    """
    profiles = {
        'conservative': {
            'min_weight': 0.05,
            'max_weight': 0.30,
            'risk_tolerance': 2.0
        },
        'balanced': {
            'min_weight': 0.02,
            'max_weight': 0.50,
            'risk_tolerance': 1.0
        },
        'aggressive': {
            'min_weight': 0.0,
            'max_weight': 0.80,
            'risk_tolerance': 0.5
        }
    }

    if risk_profile not in profiles:
        logger.warning(f"Unknown risk profile '{risk_profile}', using 'balanced'")
        risk_profile = 'balanced'

    return profiles[risk_profile]


# ============================================================================
# Module Info
# ============================================================================

__all__ = [
    'optimize_portfolio',
    'compare_portfolio_optimizers',
    'tune_hyperparameters',
    'select_features',
    'QuantumOptimizer',
    'get_default_portfolio_constraints'
]

__version__ = '1.0.0'
__author__ = 'Quantum Optimization Team'


if __name__ == "__main__":
    # Example usage demonstrations
    logger.info("="*60)
    logger.info("QUANTUM OPTIMIZATION INTEGRATION MODULE EXAMPLES")
    logger.info("="*60)

    # Example 1: Portfolio Optimization
    logger.info("\n### Example 1: Portfolio Optimization ###")
    np.random.seed(42)
    n_assets = 5
    returns = np.random.uniform(0.05, 0.15, n_assets)
    random_matrix = np.random.randn(n_assets, n_assets)
    cov = np.dot(random_matrix, random_matrix.T) * 0.01

    result = optimize_portfolio(returns, cov, risk_tolerance=1.0)
    print(f"\nOptimal weights: {result['weights']}")
    print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
    print(f"Method: {result['method']}")

    # Example 2: Hyperparameter Tuning
    logger.info("\n### Example 2: Hyperparameter Tuning ###")
    param_spaces = [
        {'name': 'x', 'type': 'continuous', 'bounds': (-5, 5)},
        {'name': 'y', 'type': 'continuous', 'bounds': (-5, 5)}
    ]

    def simple_objective(params):
        """Simple quadratic function"""
        return -(params['x']**2 + params['y']**2)

    result = tune_hyperparameters(
        param_spaces,
        simple_objective,
        maximize=True,
        n_iterations=20
    )
    print(f"\nBest params: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Method: {result['method']}")

    # Example 3: Feature Selection
    logger.info("\n### Example 3: Feature Selection ###")
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)

    result = select_features(X, y, n_features=10, n_iterations=10)
    print(f"\nSelected {result['n_features_selected']} features: {result['selected_features']}")
    print(f"Method: {result['method']}")

    # Example 4: Unified Interface
    logger.info("\n### Example 4: Unified QuantumOptimizer Interface ###")
    optimizer = QuantumOptimizer(use_quantum=True)

    portfolio_result = optimizer.optimize_portfolio(returns, cov)
    print(f"\nPortfolio Sharpe ratio: {portfolio_result['sharpe_ratio']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("EXAMPLES COMPLETE")
    logger.info("="*60)
