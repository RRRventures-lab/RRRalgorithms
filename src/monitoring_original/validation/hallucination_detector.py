from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

#!/usr/bin/env python3

"""
Hallucination Detection System

Multi-layer detection system for identifying AI hallucinations in trading decisions.
Implements 5 detection layers:
1. Statistical Plausibility
2. Historical Consistency
3. Cross-Validation Ensemble
4. Logical Coherence
5. Source Attribution

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


class HallucinationSeverity(Enum):
    """Severity levels for detected hallucinations"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HallucinationType(Enum):
    """Types of hallucinations"""
    STATISTICAL_IMPOSSIBLE = "statistical_impossible"
    STATISTICAL_OUTLIER = "statistical_outlier"
    HISTORICAL_INCONSISTENT = "historical_inconsistent"
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    CAUSALITY_VIOLATION = "causality_violation"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    UNSOURCED_DATA = "unsourced_data"
    CORRUPTED_DATA = "corrupted_data"
    IMPOSSIBLE_TRANSITION = "impossible_transition"


@dataclass
class DataPoint:
    """Data point with source attribution"""
    value: Any
    source_url: Optional[str]
    timestamp: datetime
    checksum: Optional[str]
    confidence: float


@dataclass
class HallucinationReport:
    """Report of detected hallucination"""
    detected: bool
    severity: HallucinationSeverity
    hallucination_type: HallucinationType
    layer: str
    reason: str
    evidence: Dict[str, Any]
    confidence: float
    timestamp: datetime
    suggested_action: str


@dataclass
class HistoricalContext:
    """Historical context for consistency checks"""
    prices: List[float]
    volumes: List[float]
    timestamps: List[datetime]
    volatility: float
    trend: str  # 'upward', 'downward', 'sideways'
    regime: str  # 'bull', 'bear', 'ranging'


class HallucinationDetector:
    """
    Multi-layer hallucination detection system

    Detects when AI systems generate outputs that are not grounded in reality,
    data, or sound reasoning.
    """

    def __init__(
        self,
        statistical_outlier_threshold: float = 5.0,  # 5-sigma
        ensemble_disagreement_threshold: float = 0.3,  # 30% CoV
        volatility_multiplier: float = 10.0,
        min_historical_samples: int = 100,
        enable_all_layers: bool = True
    ):
        self.statistical_outlier_threshold = statistical_outlier_threshold
        self.ensemble_disagreement_threshold = ensemble_disagreement_threshold
        self.volatility_multiplier = volatility_multiplier
        self.min_historical_samples = min_historical_samples
        self.enable_all_layers = enable_all_layers

        # Tracking
        self.detections_by_type: Dict[HallucinationType, int] = {}
        self.detections_by_layer: Dict[str, int] = {}

        logger.info("HallucinationDetector initialized with strict mode")

    def detect(
        self,
        prediction: float,
        current_price: float,
        historical_context: HistoricalContext,
        ensemble_predictions: Optional[List[float]] = None,
        data_sources: Optional[List[DataPoint]] = None,
        decision_reasoning: Optional[Dict[str, Any]] = None
    ) -> List[HallucinationReport]:
        """
        Detect hallucinations using all 5 layers

        Args:
            prediction: Predicted value (price, direction, etc.)
            current_price: Current market price
            historical_context: Historical market data
            ensemble_predictions: Predictions from multiple models
            data_sources: Data points used for prediction
            decision_reasoning: Reasoning provided by AI

        Returns:
            List of hallucination reports (empty if none detected)
        """
        reports = []

        # Layer 1: Statistical Plausibility
        report = self._layer1_statistical_plausibility(
            prediction, current_price, historical_context
        )
        if report:
            reports.append(report)

        # Layer 2: Historical Consistency
        report = self._layer2_historical_consistency(
            prediction, current_price, historical_context
        )
        if report:
            reports.append(report)

        # Layer 3: Cross-Validation Ensemble
        if ensemble_predictions:
            report = self._layer3_ensemble_agreement(
                prediction, ensemble_predictions
            )
            if report:
                reports.append(report)

        # Layer 4: Logical Coherence
        if decision_reasoning:
            report = self._layer4_logical_coherence(
                prediction, current_price, decision_reasoning, historical_context
            )
            if report:
                reports.append(report)

        # Layer 5: Source Attribution
        if data_sources:
            report = self._layer5_source_attribution(data_sources)
            if report:
                reports.append(report)

        # Track detections
        for report in reports:
            self.detections_by_type[report.hallucination_type] = \
                self.detections_by_type.get(report.hallucination_type, 0) + 1
            self.detections_by_layer[report.layer] = \
                self.detections_by_layer.get(report.layer, 0) + 1

        return reports

    def _layer1_statistical_plausibility(
        self,
        prediction: float,
        current_price: float,
        historical_context: HistoricalContext
    ) -> Optional[HallucinationReport]:
        """
        Layer 1: Check statistical plausibility

        Detects:
        - Impossible values (negative prices, probabilities > 1.0)
        - Extreme outliers (beyond N-sigma from mean)
        - Magnitude sanity checks
        """
        # Check for impossible values
        if prediction <= 0:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                hallucination_type=HallucinationType.STATISTICAL_IMPOSSIBLE,
                layer="statistical_plausibility",
                reason=f"Impossible prediction: {prediction} (must be positive)",
                evidence={"prediction": prediction, "current_price": current_price},
                confidence=1.0,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION"
            )

        # Check for extreme outliers
        if len(historical_context.prices) >= self.min_historical_samples:
            mean_price = np.mean(historical_context.prices)
            std_price = np.std(historical_context.prices)

            z_score = abs((prediction - mean_price) / (std_price + 1e-10))

            if z_score > self.statistical_outlier_threshold:
                severity = HallucinationSeverity.HIGH if z_score > 7 else HallucinationSeverity.MEDIUM

                return HallucinationReport(
                    detected=True,
                    severity=severity,
                    hallucination_type=HallucinationType.STATISTICAL_OUTLIER,
                    layer="statistical_plausibility",
                    reason=f"Prediction is {z_score:.2f}-sigma outlier (threshold: {self.statistical_outlier_threshold})",
                    evidence={
                        "prediction": prediction,
                        "mean": mean_price,
                        "std": std_price,
                        "z_score": z_score
                    },
                    confidence=min(z_score / 10.0, 1.0),
                    timestamp=datetime.utcnow(),
                    suggested_action="REVIEW_REQUIRED" if severity == HallucinationSeverity.MEDIUM else "REJECT_DECISION"
                )

        return None

    def _layer2_historical_consistency(
        self,
        prediction: float,
        current_price: float,
        historical_context: HistoricalContext
    ) -> Optional[HallucinationReport]:
        """
        Layer 2: Check historical consistency

        Detects:
        - Predictions that violate historical volatility patterns
        - Trend contradictions
        - Impossible transitions
        - Regime-inconsistent predictions
        """
        if len(historical_context.prices) < self.min_historical_samples:
            return None

        # Check volatility consistency
        prediction_change = abs(prediction - current_price) / current_price
        max_expected_change = historical_context.volatility * self.volatility_multiplier

        if prediction_change > max_expected_change:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                hallucination_type=HallucinationType.HISTORICAL_INCONSISTENT,
                layer="historical_consistency",
                reason=f"Prediction implies {prediction_change*100:.1f}% change, exceeds {max_expected_change*100:.1f}% historical volatility threshold",
                evidence={
                    "prediction_change": prediction_change,
                    "historical_volatility": historical_context.volatility,
                    "max_expected_change": max_expected_change,
                    "current_price": current_price,
                    "prediction": prediction
                },
                confidence=0.85,
                timestamp=datetime.utcnow(),
                suggested_action="REVIEW_REQUIRED"
            )

        # Check trend consistency
        if historical_context.trend == 'upward':
            # Upward trend: prediction significantly below current should be flagged
            if prediction < current_price * 0.90:
                return HallucinationReport(
                    detected=True,
                    severity=HallucinationSeverity.MEDIUM,
                    hallucination_type=HallucinationType.HISTORICAL_INCONSISTENT,
                    layer="historical_consistency",
                    reason=f"Prediction contradicts upward trend: {prediction} vs {current_price}",
                    evidence={
                        "trend": historical_context.trend,
                        "prediction": prediction,
                        "current_price": current_price,
                        "drop_percentage": (current_price - prediction) / current_price * 100
                    },
                    confidence=0.70,
                    timestamp=datetime.utcnow(),
                    suggested_action="WARNING"
                )

        elif historical_context.trend == 'downward':
            # Downward trend: prediction significantly above current should be flagged
            if prediction > current_price * 1.10:
                return HallucinationReport(
                    detected=True,
                    severity=HallucinationSeverity.MEDIUM,
                    hallucination_type=HallucinationType.HISTORICAL_INCONSISTENT,
                    layer="historical_consistency",
                    reason=f"Prediction contradicts downward trend: {prediction} vs {current_price}",
                    evidence={
                        "trend": historical_context.trend,
                        "prediction": prediction,
                        "current_price": current_price,
                        "rise_percentage": (prediction - current_price) / current_price * 100
                    },
                    confidence=0.70,
                    timestamp=datetime.utcnow(),
                    suggested_action="WARNING"
                )

        return None

    def _layer3_ensemble_agreement(
        self,
        prediction: float,
        ensemble_predictions: List[float]
    ) -> Optional[HallucinationReport]:
        """
        Layer 3: Check ensemble agreement

        Detects:
        - High disagreement among models
        - Outlier predictions within ensemble
        """
        if len(ensemble_predictions) < 3:
            return None

        mean_prediction = np.mean(ensemble_predictions)
        std_prediction = np.std(ensemble_predictions)

        # Calculate coefficient of variation
        cov = std_prediction / (mean_prediction + 1e-10)

        if cov > self.ensemble_disagreement_threshold:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.MEDIUM,
                hallucination_type=HallucinationType.ENSEMBLE_DISAGREEMENT,
                layer="ensemble_agreement",
                reason=f"Models disagree significantly: CoV={cov:.2%} (threshold: {self.ensemble_disagreement_threshold:.2%})",
                evidence={
                    "ensemble_predictions": ensemble_predictions,
                    "mean": mean_prediction,
                    "std": std_prediction,
                    "cov": cov,
                    "prediction": prediction
                },
                confidence=0.80,
                timestamp=datetime.utcnow(),
                suggested_action="REVIEW_REQUIRED"
            )

        # Check if our prediction is an outlier within the ensemble
        if abs(prediction - mean_prediction) > 2 * std_prediction:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.MEDIUM,
                hallucination_type=HallucinationType.ENSEMBLE_DISAGREEMENT,
                layer="ensemble_agreement",
                reason=f"Prediction is outlier in ensemble: {prediction} vs mean {mean_prediction}",
                evidence={
                    "prediction": prediction,
                    "ensemble_mean": mean_prediction,
                    "ensemble_std": std_prediction,
                    "z_score": abs(prediction - mean_prediction) / (std_prediction + 1e-10)
                },
                confidence=0.75,
                timestamp=datetime.utcnow(),
                suggested_action="REVIEW_REQUIRED"
            )

        return None

    def _layer4_logical_coherence(
        self,
        prediction: float,
        current_price: float,
        decision_reasoning: Dict[str, Any],
        historical_context: HistoricalContext
    ) -> Optional[HallucinationReport]:
        """
        Layer 4: Check logical coherence

        Detects:
        - Contradictions between decision and reasoning
        - Causality violations
        - Temporal inconsistencies
        """
        # Extract decision direction
        decision_direction = decision_reasoning.get('direction', None)
        decision_action = decision_reasoning.get('action', None)

        # Check contradiction: BUY decision but prediction is down
        if decision_action == 'BUY' and decision_direction == 'DOWN':
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                hallucination_type=HallucinationType.LOGICAL_CONTRADICTION,
                layer="logical_coherence",
                reason="Decision contradicts prediction: BUY action with DOWN direction",
                evidence={
                    "action": decision_action,
                    "direction": decision_direction,
                    "prediction": prediction,
                    "current_price": current_price
                },
                confidence=0.95,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION"
            )

        # Check contradiction: SELL decision but prediction is up
        if decision_action == 'SELL' and decision_direction == 'UP':
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                hallucination_type=HallucinationType.LOGICAL_CONTRADICTION,
                layer="logical_coherence",
                reason="Decision contradicts prediction: SELL action with UP direction",
                evidence={
                    "action": decision_action,
                    "direction": decision_direction,
                    "prediction": prediction,
                    "current_price": current_price
                },
                confidence=0.95,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION"
            )

        # Check causality: timestamps must be ordered correctly
        reasoning_events = decision_reasoning.get('events', [])
        for i in range(len(reasoning_events) - 1):
            if reasoning_events[i].get('caused_by') == reasoning_events[i+1].get('event_id'):
                # Event A caused by Event B, but Event A is earlier
                if reasoning_events[i].get('timestamp', datetime.min) < reasoning_events[i+1].get('timestamp', datetime.min):
                    return HallucinationReport(
                        detected=True,
                        severity=HallucinationSeverity.CRITICAL,
                        hallucination_type=HallucinationType.CAUSALITY_VIOLATION,
                        layer="logical_coherence",
                        reason="Causality violation: effect occurred before cause",
                        evidence={
                            "cause_event": reasoning_events[i+1],
                            "effect_event": reasoning_events[i]
                        },
                        confidence=1.0,
                        timestamp=datetime.utcnow(),
                        suggested_action="REJECT_DECISION"
                    )

        return None

    def _layer5_source_attribution(
        self,
        data_sources: List[DataPoint]
    ) -> Optional[HallucinationReport]:
        """
        Layer 5: Check source attribution

        Detects:
        - Unsourced data
        - Corrupted data (invalid checksums)
        - Future data (timestamps in future)
        """
        now = datetime.utcnow()
        unsourced_count = 0
        corrupted_count = 0
        future_data_count = 0

        for data_point in data_sources:
            # Check for missing source
            if data_point.source_url is None:
                unsourced_count += 1

            # Check for future timestamps (data leakage)
            if data_point.timestamp > now:
                future_data_count += 1

            # Check checksum integrity
            if data_point.checksum and not self._verify_checksum(data_point):
                corrupted_count += 1

        # Report unsourced data
        if unsourced_count > 0:
            severity = HallucinationSeverity.CRITICAL if unsourced_count == len(data_sources) else HallucinationSeverity.HIGH

            return HallucinationReport(
                detected=True,
                severity=severity,
                hallucination_type=HallucinationType.UNSOURCED_DATA,
                layer="source_attribution",
                reason=f"{unsourced_count}/{len(data_sources)} data points have no source attribution",
                evidence={
                    "unsourced_count": unsourced_count,
                    "total_data_points": len(data_sources)
                },
                confidence=0.90,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION" if severity == HallucinationSeverity.CRITICAL else "REVIEW_REQUIRED"
            )

        # Report future data (most critical - indicates data leakage)
        if future_data_count > 0:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                hallucination_type=HallucinationType.TEMPORAL_ANOMALY,
                layer="source_attribution",
                reason=f"Future data detected: {future_data_count}/{len(data_sources)} data points have timestamps in the future",
                evidence={
                    "future_data_count": future_data_count,
                    "total_data_points": len(data_sources),
                    "current_time": now
                },
                confidence=1.0,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION"
            )

        # Report corrupted data
        if corrupted_count > 0:
            return HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                hallucination_type=HallucinationType.CORRUPTED_DATA,
                layer="source_attribution",
                reason=f"Data corruption detected: {corrupted_count}/{len(data_sources)} data points failed checksum verification",
                evidence={
                    "corrupted_count": corrupted_count,
                    "total_data_points": len(data_sources)
                },
                confidence=1.0,
                timestamp=datetime.utcnow(),
                suggested_action="REJECT_DECISION"
            )

        return None

    def _verify_checksum(self, data_point: DataPoint) -> bool:
        """Verify data point checksum"""
        # Placeholder - implement actual checksum verification
        # In production, this would calculate checksum and compare
        return True

    @lru_cache(maxsize=128)

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            "detections_by_type": {k.value: v for k, v in self.detections_by_type.items()},
            "detections_by_layer": self.detections_by_layer,
            "total_detections": sum(self.detections_by_type.values())
        }
