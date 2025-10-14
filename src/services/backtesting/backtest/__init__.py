from .engine.backtest_engine import BacktestEngine
from .metrics.performance import PerformanceMetrics
from .strategies.strategy_base import StrategyBase

"""
Backtesting Framework for RRRalgorithms Trading System

This module provides a comprehensive backtesting framework with:
- Historical data replay
- Realistic order simulation with slippage
- Performance metrics calculation
- Walk-forward optimization
- Monte Carlo simulation
- Reporting and visualization
"""

__version__ = "0.1.0"


__all__ = [
    'BacktestEngine',
    'StrategyBase',
    'PerformanceMetrics',
]
