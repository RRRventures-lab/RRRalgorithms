from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import numpy as np

#!/usr/bin/env python3

"""
Robustness Testing Module

Tests system robustness across multiple dimensions:
- Input perturbation robustness
- Adversarial robustness
- Noise injection robustness
- Edge case handling
- Distribution shift robustness

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


@dataclass
class RobustnessTestResult:
    """Result of a robustness test"""
    test_name: str
    perturbation_type: str
    perturbation_magnitude: float
    original_accuracy: float
    perturbed_accuracy: float
    accuracy_degradation_percent: float
    passed: bool
    num_samples_tested: int
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class RobustnessTester:
    """
    Comprehensive robustness testing for trading system

    Tests system's ability to maintain performance under various perturbations
    """

    def __init__(
        self,
        robustness_threshold: float = 0.10,  # Allow up to 10% accuracy degradation
        num_test_samples: int = 1000
    ):
        self.robustness_threshold = robustness_threshold
        self.num_test_samples = num_test_samples

        # Test results
        self.test_results: List[RobustnessTestResult] = []

        logger.info("RobustnessTester initialized")

    def test_all_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> List[RobustnessTestResult]:
        """
        Run all robustness tests

        Args:
            system_under_test: System to test
            test_data: Test data
            test_labels: True labels

        Returns:
            List of robustness test results
        """
        logger.info("Running comprehensive robustness tests...")

        results = []

        # Get baseline accuracy
        baseline_accuracy = self._get_accuracy(system_under_test, test_data, test_labels)
        logger.info(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")

        # 1. Gaussian noise robustness
        results.extend(self.test_gaussian_noise_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        # 2. Salt-and-pepper noise robustness
        results.extend(self.test_salt_pepper_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        # 3. Feature dropout robustness
        results.extend(self.test_feature_dropout_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        # 4. Feature scaling robustness
        results.extend(self.test_feature_scaling_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        # 5. Distribution shift robustness
        results.extend(self.test_distribution_shift_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        # 6. Temporal shift robustness
        results.extend(self.test_temporal_shift_robustness(
            system_under_test, test_data, test_labels, baseline_accuracy
        ))

        self.test_results = results

        # Summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(f"Robustness tests completed: {passed}/{total} passed")

        return results

    def test_gaussian_noise_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to Gaussian noise

        Tests system's ability to handle noisy inputs
        """
        results = []

        noise_levels = [0.01, 0.05, 0.10, 0.25, 0.50]

        for noise_std in noise_levels:
            logger.info(f"Testing Gaussian noise with std={noise_std}")

            # Add Gaussian noise
            noisy_data = test_data + np.random.normal(0, noise_std, test_data.shape)

            # Get accuracy on noisy data
            noisy_accuracy = self._get_accuracy(system_under_test, noisy_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - noisy_accuracy) / baseline_accuracy * 100

            # Pass if degradation is below threshold
            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Gaussian Noise Robustness",
                perturbation_type="gaussian_noise",
                perturbation_magnitude=noise_std,
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=noisy_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"noise_std": noise_std}
            ))

        return results

    def test_salt_pepper_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to salt-and-pepper noise

        Tests system's ability to handle random feature corruption
        """
        results = []

        corruption_rates = [0.01, 0.05, 0.10, 0.20]

        for corruption_rate in corruption_rates:
            logger.info(f"Testing salt-and-pepper noise with rate={corruption_rate}")

            # Add salt-and-pepper noise
            corrupted_data = test_data.copy()
            num_features = test_data.shape[1]

            for i in range(len(test_data)):
                # Randomly corrupt features
                corrupt_mask = np.random.rand(num_features) < corruption_rate
                corrupted_data[i, corrupt_mask] = np.random.choice([0, 1], size=np.sum(corrupt_mask))

            # Get accuracy
            corrupted_accuracy = self._get_accuracy(system_under_test, corrupted_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - corrupted_accuracy) / baseline_accuracy * 100

            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Salt-and-Pepper Noise Robustness",
                perturbation_type="salt_pepper_noise",
                perturbation_magnitude=corruption_rate,
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=corrupted_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"corruption_rate": corruption_rate}
            ))

        return results

    def test_feature_dropout_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to feature dropout

        Tests system's ability to handle missing features
        """
        results = []

        dropout_rates = [0.10, 0.25, 0.50]

        for dropout_rate in dropout_rates:
            logger.info(f"Testing feature dropout with rate={dropout_rate}")

            # Drop features
            dropout_data = test_data.copy()
            num_features = test_data.shape[1]

            for i in range(len(test_data)):
                # Randomly drop features (set to 0)
                dropout_mask = np.random.rand(num_features) < dropout_rate
                dropout_data[i, dropout_mask] = 0

            # Get accuracy
            dropout_accuracy = self._get_accuracy(system_under_test, dropout_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - dropout_accuracy) / baseline_accuracy * 100

            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Feature Dropout Robustness",
                perturbation_type="feature_dropout",
                perturbation_magnitude=dropout_rate,
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=dropout_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"dropout_rate": dropout_rate}
            ))

        return results

    def test_feature_scaling_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to feature scaling

        Tests system's ability to handle scaled features
        """
        results = []

        scale_factors = [0.5, 0.75, 1.25, 2.0]

        for scale_factor in scale_factors:
            logger.info(f"Testing feature scaling with factor={scale_factor}")

            # Scale features
            scaled_data = test_data * scale_factor

            # Get accuracy
            scaled_accuracy = self._get_accuracy(system_under_test, scaled_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - scaled_accuracy) / baseline_accuracy * 100

            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Feature Scaling Robustness",
                perturbation_type="feature_scaling",
                perturbation_magnitude=scale_factor,
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=scaled_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"scale_factor": scale_factor}
            ))

        return results

    def test_distribution_shift_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to distribution shift

        Tests system's ability to handle data from shifted distribution
        """
        results = []

        shift_magnitudes = [0.5, 1.0, 2.0]

        for shift_magnitude in shift_magnitudes:
            logger.info(f"Testing distribution shift with magnitude={shift_magnitude}")

            # Shift distribution (add bias to all features)
            shifted_data = test_data + shift_magnitude

            # Get accuracy
            shifted_accuracy = self._get_accuracy(system_under_test, shifted_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - shifted_accuracy) / baseline_accuracy * 100

            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Distribution Shift Robustness",
                perturbation_type="distribution_shift",
                perturbation_magnitude=shift_magnitude,
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=shifted_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"shift_magnitude": shift_magnitude}
            ))

        return results

    def test_temporal_shift_robustness(
        self,
        system_under_test: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_accuracy: float
    ) -> List[RobustnessTestResult]:
        """
        Test robustness to temporal shifts

        Tests system's ability to handle time-shifted data (e.g., lag)
        """
        results = []

        # Assume data has temporal dimension
        if len(test_data.shape) < 2:
            return results

        lag_amounts = [1, 5, 10]

        for lag in lag_amounts:
            logger.info(f"Testing temporal shift with lag={lag}")

            # Apply temporal lag (shift features)
            lagged_data = np.roll(test_data, lag, axis=1)

            # Get accuracy
            lagged_accuracy = self._get_accuracy(system_under_test, lagged_data, test_labels)

            # Calculate degradation
            degradation = (baseline_accuracy - lagged_accuracy) / baseline_accuracy * 100

            passed = degradation <= self.robustness_threshold * 100

            results.append(RobustnessTestResult(
                test_name="Temporal Shift Robustness",
                perturbation_type="temporal_shift",
                perturbation_magnitude=float(lag),
                original_accuracy=baseline_accuracy,
                perturbed_accuracy=lagged_accuracy,
                accuracy_degradation_percent=degradation,
                passed=passed,
                num_samples_tested=len(test_data),
                errors=[],
                warnings=[] if passed else [f"Accuracy degraded by {degradation:.1f}%"],
                details={"lag": lag}
            ))

        return results

    def _get_accuracy(
        self,
        system_under_test: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Get accuracy of system on given data

        This is a placeholder - in production, this would call the actual system
        """
        # Placeholder: simulate prediction
        # In real implementation: predictions = system_under_test.predict(data)

        # For testing purposes, return random accuracy between 0.6 and 0.9
        return np.random.uniform(0.6, 0.9)

    @lru_cache(maxsize=128)

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get summary of robustness test results"""
        if not self.test_results:
            return {"error": "No robustness tests run yet"}

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        pass_rate = passed_tests / total_tests

        # Group by perturbation type
        by_type = {}
        for result in self.test_results:
            ptype = result.perturbation_type
            if ptype not in by_type:
                by_type[ptype] = {"passed": 0, "failed": 0}

            if result.passed:
                by_type[ptype]["passed"] += 1
            else:
                by_type[ptype]["failed"] += 1

        # Find worst degradation
        worst_result = max(self.test_results, key=lambda r: r.accuracy_degradation_percent)

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": pass_rate,
            "by_perturbation_type": by_type,
            "worst_degradation": {
                "test_name": worst_result.test_name,
                "perturbation_type": worst_result.perturbation_type,
                "degradation_percent": worst_result.accuracy_degradation_percent,
                "perturbation_magnitude": worst_result.perturbation_magnitude
            },
            "average_degradation": np.mean([r.accuracy_degradation_percent for r in self.test_results])
        }
