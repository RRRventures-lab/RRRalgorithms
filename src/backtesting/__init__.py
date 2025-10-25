"""
Backtesting Infrastructure
===========================

Production-grade backtesting system for cryptocurrency trading strategies.
"""

from .engine import BacktestEngine
from .strategy_generator import StrategyGenerator
from .statistical_validator import StatisticalValidator
from .monte_carlo import MonteCarloSimulator
from .results_aggregator import ResultsAggregator

__all__ = [
    'BacktestEngine',
    'StrategyGenerator',
    'StatisticalValidator',
    'MonteCarloSimulator',
    'ResultsAggregator',
]
