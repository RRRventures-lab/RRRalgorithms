from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import pandas as pd

"""
Walk-Forward Optimization

Implements walk-forward analysis to prevent overfitting by:
1. Training on in-sample data
2. Testing on out-of-sample data
3. Rolling the window forward
"""


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    best_parameters: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    all_results: List[Dict]
    optimization_history: List[Dict]


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for strategy parameters.

    Prevents overfitting by:
    - Training on in-sample period
    - Validating on out-of-sample period
    - Rolling the window forward through time
    """

    def __init__(
        self,
        in_sample_ratio: float = 0.7,
        out_of_sample_ratio: float = 0.3,
        rolling_window: bool = True,
        n_splits: int = 5
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            in_sample_ratio: Fraction of data for training (0.7 = 70%)
            out_of_sample_ratio: Fraction of data for testing (0.3 = 30%)
            rolling_window: Use rolling window (True) or expanding window (False)
            n_splits: Number of walk-forward splits
        """
        self.in_sample_ratio = in_sample_ratio
        self.out_of_sample_ratio = out_of_sample_ratio
        self.rolling_window = rolling_window
        self.n_splits = n_splits

        if abs(in_sample_ratio + out_of_sample_ratio - 1.0) > 0.01:
            raise ValueError("in_sample_ratio + out_of_sample_ratio must equal 1.0")

        logger.info(
            f"Initialized WalkForwardOptimizer: "
            f"{in_sample_ratio:.0%} in-sample, {out_of_sample_ratio:.0%} out-of-sample, "
            f"{n_splits} splits"
        )

    def split_data(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into walk-forward windows.

        Args:
            data: Full dataset with DatetimeIndex

        Returns:
            List of (in_sample, out_of_sample) DataFrame tuples
        """
        total_length = len(data)
        in_sample_size = int(total_length * self.in_sample_ratio / self.n_splits)
        out_of_sample_size = int(total_length * self.out_of_sample_ratio / self.n_splits)

        splits = []

        for i in range(self.n_splits):
            if self.rolling_window:
                # Rolling window: fixed size window moves forward
                in_sample_start = i * (in_sample_size + out_of_sample_size)
                in_sample_end = in_sample_start + in_sample_size
            else:
                # Expanding window: start stays fixed, end moves forward
                in_sample_start = 0
                in_sample_end = in_sample_size + i * (in_sample_size + out_of_sample_size)

            out_of_sample_start = in_sample_end
            out_of_sample_end = out_of_sample_start + out_of_sample_size

            # Check bounds
            if out_of_sample_end > total_length:
                break

            in_sample_data = data.iloc[in_sample_start:in_sample_end]
            out_of_sample_data = data.iloc[out_of_sample_start:out_of_sample_end]

            splits.append((in_sample_data, out_of_sample_data))

            logger.debug(
                f"Split {i+1}: In-sample {len(in_sample_data)} bars, "
                f"Out-of-sample {len(out_of_sample_data)} bars"
            )

        return splits

    def optimize_parameters(
        self,
        engine,
        strategy_class,
        data: pd.DataFrame,
        parameter_grid: Dict[str, List[Any]],
        symbol: str = 'BTC-USD',
        optimization_metric: str = 'sharpe_ratio'
    ) -> WalkForwardResult:
        """
        Optimize strategy parameters using walk-forward analysis.

        Args:
            engine: BacktestEngine instance
            strategy_class: Strategy class to optimize
            data: Historical data
            parameter_grid: Dictionary of parameter names to lists of values
            symbol: Trading symbol
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            WalkForwardResult with optimization results
        """
        logger.info(f"Starting walk-forward optimization for {strategy_class.__name__}")
        logger.info(f"Parameter grid: {parameter_grid}")
        logger.info(f"Optimizing for: {optimization_metric}")

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Split data into walk-forward windows
        splits = self.split_data(data)
        logger.info(f"Created {len(splits)} walk-forward splits")

        optimization_history = []
        all_results = []

        # For each split
        for split_idx, (in_sample_data, out_of_sample_data) in enumerate(splits):
            logger.info(f"\nProcessing split {split_idx + 1}/{len(splits)}")

            split_results = []

            # Test each parameter combination on in-sample data
            for param_combo in param_combinations:
                params = dict(zip(param_names, param_combo))

                try:
                    # Create strategy with parameters
                    strategy = strategy_class(**params)

                    # Run backtest on in-sample data
                    engine.reset()
                    result = engine.run_backtest(strategy, in_sample_data, symbol)

                    # Extract optimization metric
                    metric_value = getattr(result, optimization_metric)

                    split_results.append({
                        'parameters': params,
                        'metric_value': metric_value,
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'total_trades': result.total_trades
                    })

                except Exception as e:
                    logger.error(f"Error testing parameters {params}: {e}")
                    continue

            # Find best parameters for this split
            if not split_results:
                logger.warning(f"No valid results for split {split_idx + 1}")
                continue

            best_result = max(split_results, key=lambda x: x['metric_value'])
            best_params = best_result['parameters']

            logger.info(f"Best parameters for split {split_idx + 1}: {best_params}")
            logger.info(
                f"In-sample {optimization_metric}: {best_result['metric_value']:.4f}"
            )

            # Test best parameters on out-of-sample data
            try:
                strategy = strategy_class(**best_params)
                engine.reset()
                oos_result = engine.run_backtest(strategy, out_of_sample_data, symbol)

                oos_metric_value = getattr(oos_result, optimization_metric)

                logger.info(
                    f"Out-of-sample {optimization_metric}: {oos_metric_value:.4f}"
                )

                optimization_history.append({
                    'split': split_idx + 1,
                    'best_parameters': best_params,
                    'in_sample_metric': best_result['metric_value'],
                    'out_of_sample_metric': oos_metric_value,
                    'in_sample_return': best_result['total_return'],
                    'out_of_sample_return': oos_result.total_return,
                    'in_sample_sharpe': best_result['sharpe_ratio'],
                    'out_of_sample_sharpe': oos_result.sharpe_ratio
                })

                all_results.extend([{
                    'split': split_idx + 1,
                    'type': 'in_sample',
                    **best_result
                }, {
                    'split': split_idx + 1,
                    'type': 'out_of_sample',
                    'parameters': best_params,
                    'metric_value': oos_metric_value,
                    'total_return': oos_result.total_return,
                    'sharpe_ratio': oos_result.sharpe_ratio,
                    'max_drawdown': oos_result.max_drawdown,
                    'total_trades': oos_result.total_trades
                }])

            except Exception as e:
                logger.error(f"Error testing out-of-sample: {e}")
                continue

        # Aggregate results across all splits
        if not optimization_history:
            raise ValueError("No valid optimization results")

        # Calculate average metrics
        avg_in_sample = {
            'metric': np.mean([h['in_sample_metric'] for h in optimization_history]),
            'return': np.mean([h['in_sample_return'] for h in optimization_history]),
            'sharpe': np.mean([h['in_sample_sharpe'] for h in optimization_history])
        }

        avg_out_of_sample = {
            'metric': np.mean([h['out_of_sample_metric'] for h in optimization_history]),
            'return': np.mean([h['out_of_sample_return'] for h in optimization_history]),
            'sharpe': np.mean([h['out_of_sample_sharpe'] for h in optimization_history])
        }

        # Find most frequently optimal parameters
        param_counts = {}
        for h in optimization_history:
            param_tuple = tuple(sorted(h['best_parameters'].items()))
            param_counts[param_tuple] = param_counts.get(param_tuple, 0) + 1

        most_common_params = max(param_counts.items(), key=lambda x: x[1])
        best_parameters = dict(most_common_params[0])

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best parameters: {best_parameters}")
        logger.info(f"Frequency: {most_common_params[1]}/{len(splits)} splits")
        logger.info(f"\nAverage In-Sample {optimization_metric}: {avg_in_sample['metric']:.4f}")
        logger.info(f"Average Out-of-Sample {optimization_metric}: {avg_out_of_sample['metric']:.4f}")
        logger.info(f"Robustness Ratio: {avg_out_of_sample['metric'] / avg_in_sample['metric']:.2%}")
        logger.info("=" * 70 + "\n")

        return WalkForwardResult(
            best_parameters=best_parameters,
            in_sample_metrics=avg_in_sample,
            out_of_sample_metrics=avg_out_of_sample,
            all_results=all_results,
            optimization_history=optimization_history
        )

    def analyze_stability(self, result: WalkForwardResult) -> Dict[str, float]:
        """
        Analyze parameter stability across splits.

        Args:
            result: WalkForwardResult object

        Returns:
            Dictionary of stability metrics
        """
        history = result.optimization_history

        # Calculate performance degradation
        is_metrics = [h['in_sample_metric'] for h in history]
        oos_metrics = [h['out_of_sample_metric'] for h in history]

        degradation = np.mean([
            (is_m - oos_m) / is_m if is_m != 0 else 0
            for is_m, oos_m in zip(is_metrics, oos_metrics)
        ])

        # Calculate consistency
        oos_std = np.std(oos_metrics)
        oos_mean = np.mean(oos_metrics)
        coefficient_of_variation = oos_std / oos_mean if oos_mean != 0 else float('inf')

        return {
            'performance_degradation': degradation * 100,  # as percentage
            'oos_consistency': 1 / (1 + coefficient_of_variation),  # 0-1 scale
            'oos_mean': oos_mean,
            'oos_std': oos_std
        }
