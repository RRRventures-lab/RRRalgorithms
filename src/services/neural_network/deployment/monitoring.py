"""
Model Monitoring and Drift Detection

Monitors model performance in production and detects distribution drift
and performance degradation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics."""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_mean: float
    confidence_std: float
    latency_mean: float
    latency_p95: float
    drift_score: float
    alert_triggered: bool


class ModelMonitor:
    """
    Monitor model performance in production.

    Tracks metrics, detects anomalies, and triggers alerts.
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        horizons: List[str] = None
    ):
        """
        Initialize model monitor.

        Args:
            window_size: Size of sliding window for metrics
            alert_thresholds: Thresholds for triggering alerts
            horizons: Prediction horizons to monitor
        """
        self.window_size = window_size
        self.horizons = horizons or ['5min', '15min', '1hr']

        # Default thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.1,  # 10% drop from baseline
            'latency_increase': 2.0,  # 2x increase
            'confidence_drop': 0.15,  # 15% drop
            'drift_score': 0.1  # Drift threshold
        }

        # Metrics history
        self.predictions_history: Deque = deque(maxlen=window_size)
        self.targets_history: Deque = deque(maxlen=window_size)
        self.confidence_history: Deque = deque(maxlen=window_size)
        self.latency_history: Deque = deque(maxlen=window_size)

        # Baseline metrics (set during initial period)
        self.baseline_accuracy: Optional[float] = None
        self.baseline_latency: Optional[float] = None
        self.baseline_confidence: Optional[float] = None

        # Alert state
        self.alerts: List[Dict] = []

        logger.info("Initialized ModelMonitor")

    def update(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Optional[Dict[str, np.ndarray]] = None,
        confidence: Optional[Dict[str, float]] = None,
        latency: Optional[float] = None
    ):
        """
        Update monitoring with new predictions.

        Args:
            predictions: Model predictions {horizon: array}
            targets: Ground truth labels (if available)
            confidence: Confidence scores
            latency: Inference latency
        """
        # Store predictions
        self.predictions_history.append(predictions)

        if targets is not None:
            self.targets_history.append(targets)

        if confidence is not None:
            self.confidence_history.append(confidence)

        if latency is not None:
            self.latency_history.append(latency)

        # Set baseline if not set and have enough data
        if self.baseline_accuracy is None and len(self.targets_history) >= 100:
            self._set_baseline()

    def _set_baseline(self):
        """Set baseline metrics from initial data."""
        if len(self.targets_history) < 100:
            return

        # Compute baseline accuracy
        accuracies = []
        for preds, targets in zip(
            list(self.predictions_history)[-100:],
            list(self.targets_history)[-100:]
        ):
            for horizon in self.horizons:
                if horizon in preds and horizon in targets:
                    acc = (preds[horizon] == targets[horizon]).mean()
                    accuracies.append(acc)

        self.baseline_accuracy = np.mean(accuracies) if accuracies else None

        # Baseline latency
        if self.latency_history:
            self.baseline_latency = np.mean(list(self.latency_history)[-100:])

        # Baseline confidence
        if self.confidence_history:
            confs = []
            for conf_dict in list(self.confidence_history)[-100:]:
                confs.extend(conf_dict.values())
            self.baseline_confidence = np.mean(confs) if confs else None

        logger.info(
            f"Baseline set - Accuracy: {self.baseline_accuracy:.4f}, "
            f"Latency: {self.baseline_latency:.2f}ms, "
            f"Confidence: {self.baseline_confidence:.4f}"
        )

    def compute_metrics(self) -> MonitoringMetrics:
        """
        Compute current monitoring metrics.

        Returns:
            Current metrics
        """
        # Compute accuracy metrics
        if len(self.targets_history) > 0:
            predictions = list(self.predictions_history)[-len(self.targets_history):]
            targets = list(self.targets_history)

            all_correct = []
            all_tp = {h: 0 for h in self.horizons}
            all_fp = {h: 0 for h in self.horizons}
            all_fn = {h: 0 for h in self.horizons}

            for pred, target in zip(predictions, targets):
                for horizon in self.horizons:
                    if horizon in pred and horizon in target:
                        correct = pred[horizon] == target[horizon]
                        all_correct.extend(correct)

                        # Precision/Recall components
                        for c in range(3):  # 3 classes
                            tp = ((pred[horizon] == c) & (target[horizon] == c)).sum()
                            fp = ((pred[horizon] == c) & (target[horizon] != c)).sum()
                            fn = ((pred[horizon] != c) & (target[horizon] == c)).sum()

                            all_tp[horizon] += tp
                            all_fp[horizon] += fp
                            all_fn[horizon] += fn

            accuracy = np.mean(all_correct) if all_correct else 0.0

            # Average precision/recall
            precisions = []
            recalls = []
            for h in self.horizons:
                prec = all_tp[h] / (all_tp[h] + all_fp[h] + 1e-8)
                rec = all_tp[h] / (all_tp[h] + all_fn[h] + 1e-8)
                precisions.append(prec)
                recalls.append(rec)

            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        else:
            accuracy = precision = recall = f1_score = 0.0

        # Confidence metrics
        if self.confidence_history:
            recent_confs = []
            for conf_dict in list(self.confidence_history)[-100:]:
                recent_confs.extend(conf_dict.values())
            confidence_mean = np.mean(recent_confs)
            confidence_std = np.std(recent_confs)
        else:
            confidence_mean = confidence_std = 0.0

        # Latency metrics
        if self.latency_history:
            recent_latencies = list(self.latency_history)[-100:]
            latency_mean = np.mean(recent_latencies)
            latency_p95 = np.percentile(recent_latencies, 95)
        else:
            latency_mean = latency_p95 = 0.0

        # Drift detection (placeholder)
        drift_score = 0.0

        # Check for alerts
        alert_triggered = self._check_alerts(
            accuracy, latency_mean, confidence_mean
        )

        return MonitoringMetrics(
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            latency_mean=latency_mean,
            latency_p95=latency_p95,
            drift_score=drift_score,
            alert_triggered=alert_triggered
        )

    def _check_alerts(
        self,
        accuracy: float,
        latency: float,
        confidence: float
    ) -> bool:
        """
        Check if any alert conditions are met.

        Args:
            accuracy: Current accuracy
            latency: Current latency
            confidence: Current confidence

        Returns:
            Whether alert was triggered
        """
        alert_triggered = False

        # Accuracy drop
        if self.baseline_accuracy is not None:
            accuracy_drop = self.baseline_accuracy - accuracy
            if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                self._trigger_alert(
                    'accuracy_drop',
                    f"Accuracy dropped by {accuracy_drop:.2%} from baseline"
                )
                alert_triggered = True

        # Latency increase
        if self.baseline_latency is not None:
            latency_ratio = latency / self.baseline_latency
            if latency_ratio > self.alert_thresholds['latency_increase']:
                self._trigger_alert(
                    'latency_increase',
                    f"Latency increased {latency_ratio:.1f}x from baseline"
                )
                alert_triggered = True

        # Confidence drop
        if self.baseline_confidence is not None:
            confidence_drop = self.baseline_confidence - confidence
            if confidence_drop > self.alert_thresholds['confidence_drop']:
                self._trigger_alert(
                    'confidence_drop',
                    f"Confidence dropped by {confidence_drop:.2%} from baseline"
                )
                alert_triggered = True

        return alert_triggered

    def _trigger_alert(self, alert_type: str, message: str):
        """
        Trigger an alert.

        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now()
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            since: Only return alerts after this time

        Returns:
            List of alerts
        """
        if since is None:
            return self.alerts

        return [a for a in self.alerts if a['timestamp'] >= since]


class DriftDetector:
    """
    Detect distribution drift in input features and predictions.

    Uses statistical tests to identify when data distribution changes.
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        test_method: str = 'ks'
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Size of reference window
            drift_threshold: P-value threshold for drift detection
            test_method: Statistical test ('ks', 'chi2', 'psi')
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.test_method = test_method

        # Reference distribution
        self.reference_features: Optional[np.ndarray] = None
        self.reference_predictions: Optional[Dict[str, np.ndarray]] = None

        # Current window
        self.current_features: Deque = deque(maxlen=window_size)
        self.current_predictions: Deque = deque(maxlen=window_size)

        # Drift history
        self.drift_scores: List[float] = []
        self.drift_detected: List[datetime] = []

        logger.info(f"Initialized DriftDetector (method: {test_method})")

    def set_reference(
        self,
        features: np.ndarray,
        predictions: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Set reference distribution.

        Args:
            features: Reference features
            predictions: Reference predictions
        """
        self.reference_features = features
        self.reference_predictions = predictions

        logger.info(f"Reference distribution set ({len(features)} samples)")

    def update(
        self,
        features: np.ndarray,
        predictions: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Update current window.

        Args:
            features: New features
            predictions: New predictions
        """
        self.current_features.append(features)

        if predictions is not None:
            self.current_predictions.append(predictions)

    def detect_feature_drift(self) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect drift in input features.

        Returns:
            (drift_detected, overall_score, per_feature_scores)
        """
        if self.reference_features is None or len(self.current_features) < 100:
            return False, 0.0, {}

        # Convert current window to array
        current = np.vstack(list(self.current_features))

        # Test each feature
        feature_scores = {}
        for i in range(current.shape[1]):
            ref_feature = self.reference_features[:, i]
            cur_feature = current[:, i]

            if self.test_method == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_feature, cur_feature)
                score = 1 - p_value
            elif self.test_method == 'chi2':
                # Chi-square test (for categorical)
                # Bin continuous features
                ref_binned, bins = np.histogram(ref_feature, bins=10)
                cur_binned, _ = np.histogram(cur_feature, bins=bins)

                # Chi-square test
                statistic, p_value = stats.chisquare(cur_binned + 1, ref_binned + 1)
                score = 1 - p_value
            else:
                score = 0.0

            feature_scores[f'feature_{i}'] = score

        # Overall drift score (max of feature scores)
        overall_score = max(feature_scores.values()) if feature_scores else 0.0

        # Detect drift
        drift_detected = overall_score > (1 - self.drift_threshold)

        if drift_detected:
            self.drift_detected.append(datetime.now())
            logger.warning(f"Feature drift detected! Score: {overall_score:.4f}")

        self.drift_scores.append(overall_score)

        return drift_detected, overall_score, feature_scores

    def detect_prediction_drift(self) -> Tuple[bool, float]:
        """
        Detect drift in prediction distribution.

        Returns:
            (drift_detected, score)
        """
        if self.reference_predictions is None or len(self.current_predictions) < 100:
            return False, 0.0

        # Get current predictions
        current_preds = list(self.current_predictions)

        # Compare distribution for each horizon
        scores = []
        for horizon in self.reference_predictions.keys():
            ref_dist = self.reference_predictions[horizon]

            # Get current distribution
            cur_preds = [p[horizon] for p in current_preds if horizon in p]
            if not cur_preds:
                continue

            cur_dist = np.concatenate(cur_preds)

            # Compare distributions
            if self.test_method == 'ks':
                _, p_value = stats.ks_2samp(ref_dist, cur_dist)
                score = 1 - p_value
            else:
                # Chi-square test on class frequencies
                ref_counts = np.bincount(ref_dist.astype(int), minlength=3)
                cur_counts = np.bincount(cur_dist.astype(int), minlength=3)

                _, p_value = stats.chisquare(cur_counts + 1, ref_counts + 1)
                score = 1 - p_value

            scores.append(score)

        overall_score = max(scores) if scores else 0.0
        drift_detected = overall_score > (1 - self.drift_threshold)

        if drift_detected:
            logger.warning(f"Prediction drift detected! Score: {overall_score:.4f}")

        return drift_detected, overall_score

    def get_drift_report(self) -> Dict:
        """
        Get comprehensive drift report.

        Returns:
            Drift report dictionary
        """
        feature_drift, feature_score, feature_scores = self.detect_feature_drift()
        pred_drift, pred_score = self.detect_prediction_drift()

        return {
            'feature_drift_detected': feature_drift,
            'feature_drift_score': feature_score,
            'feature_scores': feature_scores,
            'prediction_drift_detected': pred_drift,
            'prediction_drift_score': pred_score,
            'total_drift_events': len(self.drift_detected),
            'recent_drift_scores': self.drift_scores[-10:],
            'last_drift_detected': self.drift_detected[-1] if self.drift_detected else None
        }


if __name__ == "__main__":
    # Test monitoring
    print("Testing Model Monitoring...")

    # Test ModelMonitor
    print("\n1. Testing ModelMonitor...")
    monitor = ModelMonitor(window_size=1000)

    # Simulate predictions
    for i in range(200):
        predictions = {
            '5min': np.random.randint(0, 3, 10),
            '15min': np.random.randint(0, 3, 10),
            '1hr': np.random.randint(0, 3, 10)
        }
        targets = {
            '5min': np.random.randint(0, 3, 10),
            '15min': np.random.randint(0, 3, 10),
            '1hr': np.random.randint(0, 3, 10)
        }
        confidence = {
            '5min': np.random.rand(),
            '15min': np.random.rand(),
            '1hr': np.random.rand()
        }
        latency = np.random.rand() * 100

        monitor.update(predictions, targets, confidence, latency)

    metrics = monitor.compute_metrics()
    print(f"   Accuracy: {metrics.accuracy:.4f}")
    print(f"   F1 Score: {metrics.f1_score:.4f}")
    print(f"   Latency: {metrics.latency_mean:.2f}ms")
    print(f"   Confidence: {metrics.confidence_mean:.4f}")

    # Test DriftDetector
    print("\n2. Testing DriftDetector...")
    detector = DriftDetector(window_size=1000, drift_threshold=0.05)

    # Set reference
    ref_features = np.random.randn(1000, 10)
    detector.set_reference(ref_features)

    # Simulate drift
    for i in range(150):
        if i < 100:
            # Similar distribution
            features = np.random.randn(10) + 0.1
        else:
            # Drifted distribution
            features = np.random.randn(10) + 2.0

        detector.update(features)

        if i % 50 == 0:
            drift_detected, score, _ = detector.detect_feature_drift()
            print(f"   Step {i}: Drift={drift_detected}, Score={score:.4f}")

    report = detector.get_drift_report()
    print(f"\n   Drift Report:")
    print(f"   Feature Drift: {report['feature_drift_detected']}")
    print(f"   Score: {report['feature_drift_score']:.4f}")
    print(f"   Total Events: {report['total_drift_events']}")

    print("\nMonitoring tests completed successfully!")
