from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Callable
import logging
import numpy as np
import pandas as pd


"""
Walk-Forward Validation for Trading Strategies

Implements rolling window validation to prevent look-ahead bias and
evaluate strategy robustness over time.
"""


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    train_window_days: int = 90  # Training window size
    test_window_days: int = 30   # Test window size
    step_days: int = 7            # Step size between windows
    min_train_samples: int = 1000  # Minimum training samples
    refit_frequency: int = 1      # Refit model every N windows


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    model_params: Optional[Dict] = None


class WalkForwardValidator:
    """
    Walk-Forward Validation Engine

    Performs rolling window validation with:
    - Fixed or expanding training windows
    - Customizable test window sizes
    - Model retraining at specified intervals
    - Gap periods to prevent data leakage
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results: List[WalkForwardResult] = []

    def validate(
        self,
        data: pd.DataFrame,
        model_factory: Callable,
        train_fn: Callable,
        predict_fn: Callable,
        metric_fn: Callable,
        expanding_window: bool = False,
        gap_days: int = 0
    ) -> List[WalkForwardResult]:
        """
        Perform walk-forward validation

        Args:
            data: DataFrame with datetime index and feature columns
            model_factory: Function that returns a new model instance
            train_fn: Function(model, train_data) -> trained_model
            predict_fn: Function(model, test_data) -> predictions
            metric_fn: Function(predictions, actuals) -> metrics_dict
            expanding_window: If True, training window expands over time
            gap_days: Days between train and test to prevent leakage

        Returns:
            List of WalkForwardResult objects
        """
        logger.info("Starting walk-forward validation")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Total samples: {len(data)}")

        results = []
        window_id = 0
        model = None

        # Calculate total windows
        data_start = data.index[0]
        data_end = data.index[-1]
        total_days = (data_end - data_start).days

        # Iterate through windows
        current_date = data_start + timedelta(days=self.config.train_window_days)

        while current_date + timedelta(days=self.config.test_window_days) <= data_end:
            # Define train window
            if expanding_window:
                train_start = data_start
            else:
                train_start = current_date - timedelta(days=self.config.train_window_days)

            train_end = current_date

            # Apply gap period
            test_start = train_end + timedelta(days=gap_days)
            test_end = test_start + timedelta(days=self.config.test_window_days)

            # Extract train and test data
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

            # Check minimum samples
            if len(train_data) < self.config.min_train_samples:
                logger.warning(f"Window {window_id}: Insufficient training samples ({len(train_data)})")
                current_date += timedelta(days=self.config.step_days)
                continue

            if len(test_data) == 0:
                logger.warning(f"Window {window_id}: No test data")
                break

            logger.info(f"\nWindow {window_id}:")
            logger.info(f"  Train: {train_start} to {train_end} ({len(train_data)} samples)")
            logger.info(f"  Test:  {test_start} to {test_end} ({len(test_data)} samples)")

            # Train model (or reuse if not refit time)
            if model is None or window_id % self.config.refit_frequency == 0:
                logger.info(f"  Training new model...")
                model = model_factory()
                model = train_fn(model, train_data)

                # Calculate training metrics
                train_predictions = predict_fn(model, train_data)
                train_actuals = self._extract_targets(train_data)
                train_metrics = metric_fn(train_predictions, train_actuals)
                logger.info(f"  Train metrics: {train_metrics}")
            else:
                logger.info(f"  Reusing model from window {window_id - 1}")
                train_metrics = {}

            # Make predictions on test set
            test_predictions = predict_fn(model, test_data)
            test_actuals = self._extract_targets(test_data)

            # Calculate test metrics
            test_metrics = metric_fn(test_predictions, test_actuals)
            logger.info(f"  Test metrics: {test_metrics}")

            # Store results
            result = WalkForwardResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                predictions=test_predictions,
                actuals=test_actuals,
                model_params=getattr(model, 'get_params', lambda: None)()
            )

            results.append(result)

            # Move to next window
            window_id += 1
            current_date += timedelta(days=self.config.step_days)

        logger.info(f"\nWalk-forward validation complete: {len(results)} windows")
        self.results = results
        return results

    def _extract_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Extract target values from data"""
        if 'target' in data.columns:
            return data['target'].values
        elif 'returns' in data.columns:
            return data['returns'].values
        else:
            # Assume last column is target
            return data.iloc[:, -1].values

    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all windows

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.results:
            return {}

        # Collect all test metrics
        all_metrics = {}
        for result in self.results:
            for metric_name, metric_value in result.test_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        # Aggregate
        aggregated = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            aggregated[f"{metric_name}_mean"] = np.mean(values_array)
            aggregated[f"{metric_name}_std"] = np.std(values_array)
            aggregated[f"{metric_name}_min"] = np.min(values_array)
            aggregated[f"{metric_name}_max"] = np.max(values_array)
            aggregated[f"{metric_name}_median"] = np.median(values_array)

        return aggregated

    @lru_cache(maxsize=128)

    def get_stability_score(self, metric_name: str = 'sharpe_ratio') -> float:
        """
        Calculate stability score based on metric consistency

        Lower variance in metric across windows indicates more stable strategy.

        Returns:
            Stability score (higher is better, range 0-1)
        """
        if not self.results:
            return 0.0

        values = [r.test_metrics.get(metric_name, 0) for r in self.results]
        if len(values) == 0 or all(v == 0 for v in values):
            return 0.0

        mean_value = np.mean(values)
        std_value = np.std(values)

        # Calculate coefficient of variation (inverse for stability)
        if mean_value != 0:
            cv = std_value / abs(mean_value)
            stability = 1 / (1 + cv)  # Map to 0-1 range
        else:
            stability = 0.0

        return stability

    def detect_overfitting(self, threshold: float = 0.2) -> bool:
        """
        Detect potential overfitting by comparing train vs test performance

        Args:
            threshold: Maximum acceptable degradation (test_metric / train_metric)

        Returns:
            True if overfitting detected
        """
        if not self.results:
            return False

        # Calculate average train vs test performance
        train_scores = []
        test_scores = []

        for result in self.results:
            # Use first available metric
            if result.train_metrics and result.test_metrics:
                metric_name = list(result.test_metrics.keys())[0]
                train_val = result.train_metrics.get(metric_name, 0)
                test_val = result.test_metrics.get(metric_name, 0)

                if train_val > 0:  # Avoid division by zero
                    train_scores.append(train_val)
                    test_scores.append(test_val)

        if not train_scores:
            return False

        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)

        degradation = (avg_train - avg_test) / avg_train if avg_train != 0 else 0

        logger.info(f"Performance degradation: {degradation:.2%}")

        return degradation > threshold

    def plot_results(self, metric_name: str = 'sharpe_ratio', save_path: Optional[str] = None):
        """
        Plot walk-forward results over time

        Args:
            metric_name: Metric to plot
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        if not self.results:
            logger.warning("No results to plot")
            return

        # Extract data
        window_ids = [r.window_id for r in self.results]
        test_dates = [r.test_start for r in self.results]
        train_values = [r.train_metrics.get(metric_name, 0) for r in self.results]
        test_values = [r.test_metrics.get(metric_name, 0) for r in self.results]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Metric over time
        ax1.plot(test_dates, train_values, 'g-', label='Train', alpha=0.7)
        ax1.plot(test_dates, test_values, 'b-', label='Test', linewidth=2)
        ax1.axhline(y=np.mean(test_values), color='r', linestyle='--',
                    label=f'Test Mean: {np.mean(test_values):.3f}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(metric_name)
        ax1.set_title(f'Walk-Forward Validation: {metric_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Rolling window performance
        window_size = min(5, len(test_values))
        rolling_mean = pd.Series(test_values).rolling(window=window_size).mean()

        ax2.plot(test_dates, test_values, 'b-', alpha=0.3, label='Test')
        ax2.plot(test_dates, rolling_mean, 'r-', linewidth=2,
                 label=f'{window_size}-window MA')
        ax2.set_xlabel('Date')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Rolling Average Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def export_results(self, output_path: str):
        """Export results to CSV"""
        if not self.results:
            logger.warning("No results to export")
            return

        # Convert results to DataFrame
        rows = []
        for result in self.results:
            row = {
                'window_id': result.window_id,
                'train_start': result.train_start,
                'train_end': result.train_end,
                'test_start': result.test_start,
                'test_end': result.test_end,
            }

            # Add train metrics
            for k, v in result.train_metrics.items():
                row[f'train_{k}'] = v

            # Add test metrics
            for k, v in result.test_metrics.items():
                row[f'test_{k}'] = v

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    data = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'target': np.random.randint(0, 2, len(dates))
    }, index=dates)

    # Configure validator
    config = WalkForwardConfig(
        train_window_days=90,
        test_window_days=30,
        step_days=7
    )

    validator = WalkForwardValidator(config)

    # Define model factory and functions
    def model_factory():
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def train_fn(model, train_data):
        X = train_data[['feature1', 'feature2']]
        y = train_data['target']
        model.fit(X, y)
        return model

    def predict_fn(model, test_data):
        X = test_data[['feature1', 'feature2']]
        return model.predict(X)

    def metric_fn(predictions, actuals):
        accuracy = np.mean(predictions == actuals)
        return {'accuracy': accuracy}

    # Run validation
    results = validator.validate(
        data=data,
        model_factory=model_factory,
        train_fn=train_fn,
        predict_fn=predict_fn,
        metric_fn=metric_fn,
        gap_days=1
    )

    # Print results
    print("\nAggregated Metrics:")
    for k, v in validator.aggregate_metrics().items():
        print(f"  {k}: {v:.4f}")

    print(f"\nStability Score: {validator.get_stability_score('accuracy'):.4f}")
    print(f"Overfitting Detected: {validator.detect_overfitting()}")
