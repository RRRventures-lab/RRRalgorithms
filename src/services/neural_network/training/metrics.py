"""
Metrics for Neural Network Training and Evaluation

This module implements comprehensive metrics for evaluating cryptocurrency
price prediction models, including both classification and trading metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and compute metrics during training and evaluation.
    """

    def __init__(self, horizons: List[str] = None):
        """
        Initialize metrics tracker.

        Args:
            horizons: List of prediction horizons
        """
        self.horizons = horizons or ['5min', '15min', '1hr']
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = {h: [] for h in self.horizons}
        self.targets = {h: [] for h in self.horizons}
        self.confidences = {h: [] for h in self.horizons}
        self.losses = []

    def update(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        loss: float
    ):
        """
        Update metrics with new batch.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            loss: Batch loss
        """
        self.losses.append(loss)

        for horizon in self.horizons:
            if horizon in predictions and horizon in targets:
                probs = predictions[horizon]['probs'].detach().cpu()
                preds = torch.argmax(probs, dim=-1).numpy()
                targs = targets[horizon].cpu().numpy()

                self.predictions[horizon].extend(preds)
                self.targets[horizon].extend(targs)

                if 'confidence' in predictions[horizon]:
                    conf = predictions[horizon]['confidence'].detach().cpu().numpy()
                    self.confidences[horizon].extend(conf.flatten())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric values
        """
        metrics = {
            'loss': np.mean(self.losses) if self.losses else 0.0
        }

        # Compute per-horizon metrics
        for horizon in self.horizons:
            if not self.predictions[horizon]:
                continue

            preds = np.array(self.predictions[horizon])
            targs = np.array(self.targets[horizon])

            # Accuracy
            accuracy = (preds == targs).mean()
            metrics[f'{horizon}_accuracy'] = accuracy

            # Per-class accuracy
            for class_idx in range(3):
                class_mask = targs == class_idx
                if class_mask.sum() > 0:
                    class_acc = (preds[class_mask] == targs[class_mask]).mean()
                    class_name = ['down', 'flat', 'up'][class_idx]
                    metrics[f'{horizon}_{class_name}_accuracy'] = class_acc

            # Precision, Recall, F1 for each class
            for class_idx in range(3):
                class_name = ['down', 'flat', 'up'][class_idx]

                # True positives
                tp = ((preds == class_idx) & (targs == class_idx)).sum()
                # False positives
                fp = ((preds == class_idx) & (targs != class_idx)).sum()
                # False negatives
                fn = ((preds != class_idx) & (targs == class_idx)).sum()

                # Precision
                precision = tp / (tp + fp + 1e-8)
                metrics[f'{horizon}_{class_name}_precision'] = precision

                # Recall
                recall = tp / (tp + fn + 1e-8)
                metrics[f'{horizon}_{class_name}_recall'] = recall

                # F1
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                metrics[f'{horizon}_{class_name}_f1'] = f1

            # Macro averages
            precisions = [metrics[f'{horizon}_{c}_precision'] for c in ['down', 'flat', 'up']]
            recalls = [metrics[f'{horizon}_{c}_recall'] for c in ['down', 'flat', 'up']]
            f1s = [metrics[f'{horizon}_{c}_f1'] for c in ['down', 'flat', 'up']]

            metrics[f'{horizon}_precision'] = np.mean(precisions)
            metrics[f'{horizon}_recall'] = np.mean(recalls)
            metrics[f'{horizon}_f1'] = np.mean(f1s)

            # Confidence calibration (if available)
            if self.confidences[horizon]:
                confs = np.array(self.confidences[horizon])
                correct = (preds == targs).astype(float)

                # Expected Calibration Error (ECE)
                ece = np.abs(confs - correct).mean()
                metrics[f'{horizon}_ece'] = ece

                # Confidence on correct predictions
                metrics[f'{horizon}_confidence_correct'] = confs[correct == 1].mean() if (correct == 1).any() else 0
                metrics[f'{horizon}_confidence_incorrect'] = confs[correct == 0].mean() if (correct == 0).any() else 0

        # Overall metrics (average across horizons)
        if self.horizons:
            metrics['accuracy'] = np.mean([metrics.get(f'{h}_accuracy', 0) for h in self.horizons])
            metrics['precision'] = np.mean([metrics.get(f'{h}_precision', 0) for h in self.horizons])
            metrics['recall'] = np.mean([metrics.get(f'{h}_recall', 0) for h in self.horizons])
            metrics['f1'] = np.mean([metrics.get(f'{h}_f1', 0) for h in self.horizons])

        return metrics

    def get_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrices for each horizon.

        Returns:
            Dictionary of confusion matrices
        """
        confusion_matrices = {}

        for horizon in self.horizons:
            if not self.predictions[horizon]:
                continue

            preds = np.array(self.predictions[horizon])
            targs = np.array(self.targets[horizon])

            # Compute confusion matrix
            cm = np.zeros((3, 3), dtype=int)
            for true_label in range(3):
                for pred_label in range(3):
                    cm[true_label, pred_label] = ((targs == true_label) & (preds == pred_label)).sum()

            confusion_matrices[horizon] = cm

        return confusion_matrices


class TradingMetrics:
    """
    Trading-specific metrics for evaluating model profitability.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        initial_capital: float = 10000.0
    ):
        """
        Initialize trading metrics.

        Args:
            transaction_cost: Transaction cost (0.1% = 0.001)
            initial_capital: Initial trading capital
        """
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        """Reset trading metrics."""
        self.trades = []
        self.returns = []
        self.positions = []
        self.capital_history = [self.initial_capital]

    def update(
        self,
        prediction: int,
        actual_return: float,
        confidence: Optional[float] = None
    ):
        """
        Update with new trading decision.

        Args:
            prediction: Predicted direction (0=down, 1=flat, 2=up)
            actual_return: Actual price return
            confidence: Prediction confidence
        """
        # Map prediction to position: 0=short, 1=neutral, 2=long
        position = prediction

        if position == 0:  # Short
            trade_return = -actual_return - self.transaction_cost
        elif position == 2:  # Long
            trade_return = actual_return - self.transaction_cost
        else:  # Neutral/Hold
            trade_return = 0.0

        self.trades.append({
            'prediction': prediction,
            'actual_return': actual_return,
            'trade_return': trade_return,
            'confidence': confidence
        })

        self.returns.append(trade_return)
        self.positions.append(position)

        # Update capital
        new_capital = self.capital_history[-1] * (1 + trade_return)
        self.capital_history.append(new_capital)

    def compute(self) -> Dict[str, float]:
        """
        Compute trading metrics.

        Returns:
            Dictionary of trading metrics
        """
        if not self.returns:
            return {}

        returns = np.array(self.returns)

        metrics = {
            # Return metrics
            'total_return': (self.capital_history[-1] / self.initial_capital - 1) * 100,
            'mean_return': returns.mean() * 100,
            'std_return': returns.std() * 100,

            # Risk metrics
            'sharpe_ratio': self._compute_sharpe_ratio(returns),
            'sortino_ratio': self._compute_sortino_ratio(returns),
            'max_drawdown': self._compute_max_drawdown() * 100,

            # Trading activity
            'num_trades': len(self.trades),
            'win_rate': (returns > 0).mean() * 100,
            'avg_win': returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() * 100 if (returns < 0).any() else 0,

            # Position distribution
            'pct_long': (np.array(self.positions) == 2).mean() * 100,
            'pct_short': (np.array(self.positions) == 0).mean() * 100,
            'pct_neutral': (np.array(self.positions) == 1).mean() * 100,
        }

        # Profit factor
        total_wins = returns[returns > 0].sum() if (returns > 0).any() else 0
        total_losses = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-8
        metrics['profit_factor'] = total_wins / total_losses

        return metrics

    def _compute_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Compute Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)  # Annualized

    def _compute_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Compute Sortino ratio (uses downside deviation).

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    def _compute_max_drawdown(self) -> float:
        """
        Compute maximum drawdown.

        Returns:
            Maximum drawdown as fraction
        """
        if len(self.capital_history) < 2:
            return 0.0

        capital = np.array(self.capital_history)
        running_max = np.maximum.accumulate(capital)
        drawdown = (capital - running_max) / running_max
        return abs(drawdown.min())


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics...")

    # Create dummy data
    batch_size = 100
    num_classes = 3

    # Test MetricsTracker
    print("\n1. Testing MetricsTracker...")
    tracker = MetricsTracker(horizons=['5min', '15min', '1hr'])

    for _ in range(10):  # Simulate 10 batches
        predictions = {
            '5min': {
                'probs': torch.softmax(torch.randn(batch_size, num_classes), dim=-1),
                'confidence': torch.rand(batch_size, 1)
            },
            '15min': {
                'probs': torch.softmax(torch.randn(batch_size, num_classes), dim=-1),
                'confidence': torch.rand(batch_size, 1)
            },
            '1hr': {
                'probs': torch.softmax(torch.randn(batch_size, num_classes), dim=-1),
                'confidence': torch.rand(batch_size, 1)
            }
        }
        targets = {
            '5min': torch.randint(0, num_classes, (batch_size,)),
            '15min': torch.randint(0, num_classes, (batch_size,)),
            '1hr': torch.randint(0, num_classes, (batch_size,))
        }
        loss = np.random.rand()

        tracker.update(predictions, targets, loss)

    metrics = tracker.compute()
    print(f"   Computed {len(metrics)} metrics")
    print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Overall F1: {metrics['f1']:.4f}")

    confusion_matrices = tracker.get_confusion_matrices()
    print(f"\n   Confusion Matrix (5min):")
    print(f"   {confusion_matrices['5min']}")

    # Test TradingMetrics
    print("\n2. Testing TradingMetrics...")
    trading_metrics = TradingMetrics(
        transaction_cost=0.001,
        initial_capital=10000.0
    )

    # Simulate 100 trades
    for _ in range(100):
        prediction = np.random.randint(0, 3)
        actual_return = np.random.randn() * 0.01  # 1% std
        confidence = np.random.rand()

        trading_metrics.update(prediction, actual_return, confidence)

    trading_results = trading_metrics.compute()
    print(f"   Total Return: {trading_results['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {trading_results['max_drawdown']:.2f}%")
    print(f"   Win Rate: {trading_results['win_rate']:.2f}%")
    print(f"   Profit Factor: {trading_results['profit_factor']:.2f}")

    print("\nMetrics tests completed successfully!")
