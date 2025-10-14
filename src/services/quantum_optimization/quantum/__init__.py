from .benchmarks import (
from .features import (
from .hyperparameter import (
from .integration import (
from .portfolio import (

"""
Quantum-Inspired Optimization Module

This module provides quantum-inspired optimization algorithms for:
- Portfolio optimization (QAOA)
- Hyperparameter tuning (Quantum Annealing)
- Feature selection (Quantum Genetic Algorithm)

All algorithms run on classical hardware using quantum-inspired techniques.

Author: Quantum Optimization Team
Date: 2025-10-11
Version: 1.0.0
"""

# Import main integration interface
    optimize_portfolio,
    compare_portfolio_optimizers,
    tune_hyperparameters,
    select_features,
    QuantumOptimizer,
    get_default_portfolio_constraints
)

# Import specific optimizers for advanced usage
    QAOAPortfolioOptimizer,
    ClassicalMarkowitzOptimizer,
    PortfolioConstraints,
    OptimizationResult
)

    QuantumAnnealingTuner,
    GridSearchTuner,
    HyperparameterSpace,
    TuningResult
)

    QuantumFeatureSelector,
    ClassicalFeatureSelector,
    FeatureSelectionResult
)

    OptimizerBenchmark,
    BenchmarkResult,
    BenchmarkSummary
)

__version__ = '1.0.0'
__author__ = 'Quantum Optimization Team'

__all__ = [
    # Main interface
    'optimize_portfolio',
    'compare_portfolio_optimizers',
    'tune_hyperparameters',
    'select_features',
    'QuantumOptimizer',
    'get_default_portfolio_constraints',

    # Portfolio optimization
    'QAOAPortfolioOptimizer',
    'ClassicalMarkowitzOptimizer',
    'PortfolioConstraints',
    'OptimizationResult',

    # Hyperparameter tuning
    'QuantumAnnealingTuner',
    'GridSearchTuner',
    'HyperparameterSpace',
    'TuningResult',

    # Feature selection
    'QuantumFeatureSelector',
    'ClassicalFeatureSelector',
    'FeatureSelectionResult',

    # Benchmarks
    'OptimizerBenchmark',
    'BenchmarkResult',
    'BenchmarkSummary',
]
