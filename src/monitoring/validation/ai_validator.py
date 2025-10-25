from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import json
import logging
import numpy as np
import pandas as pd


"""
AI Validator - Core Validation Engine

Comprehensive validation framework to prevent hallucinations, validate decisions,
ensure data authenticity, and maintain AI system integrity.

"""


logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    WARNING = "WARNING"


class HallucinationSeverity(Enum):
    """Hallucination severity levels"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class HallucinationReport:
    """Report of hallucination detection"""
    detected: bool
    severity: HallucinationSeverity
    layer: str  # Which detection layer found it
    reason: str
    evidence: Dict[str, Any]
    confidence: float  # 0-1


@dataclass
class ValidationReport:
    """Complete validation report"""
    request_id: str
    timestamp: datetime
    validation_status: ValidationStatus
    overall_confidence: float

    # Individual check results
    hallucination_check: HallucinationReport
    data_authenticity_passed: bool
    decision_logic_valid: bool
    adversarial_robust: bool
    confidence_calibrated: bool

    # Additional info
    concerns: List[str]
    recommendations: List[str]
    audit_logged: bool
    execution_allowed: bool

    # Performance
    validation_latency_ms: float


@dataclass
class DecisionContext:
    """Context for a trading decision to be validated"""
    decision_id: str
    timestamp: datetime
    decision_type: str  # 'TRADE', 'POSITION_SIZE', 'STOP_LOSS', etc.

    # Inputs
    symbol: str
    current_price: float
    features: np.ndarray
    historical_data: pd.DataFrame

    # Outputs
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    confidence: float

    # Reasoning
    reasoning: List[str]
    model_version: str
    data_sources: List[str]

    # Risk analysis
    expected_value: float
    max_loss: float
    probability_success: float


class AIValidator:
    """
    Core AI Validation Engine

    Implements multi-layer validation to prevent hallucinations, ensure data integrity,
    validate decision logic, and maintain system robustness.
    """

    def __init__(
        self,
        enable_strict_mode: bool = True,
        validation_timeout_ms: float = 50.0,
        hallucination_threshold: float = 0.01,
        ensemble_disagreement_threshold: float = 0.3
    ):
        """
        Args:
            enable_strict_mode: If True, reject on any concerns
            validation_timeout_ms: Maximum time for validation
            hallucination_threshold: Probability threshold for hallucination
            ensemble_disagreement_threshold: Max disagreement between models
        """
        self.strict_mode = enable_strict_mode
        self.timeout_ms = validation_timeout_ms
        self.hallucination_threshold = hallucination_threshold
        self.ensemble_disagreement_threshold = ensemble_disagreement_threshold

        # Historical tracking
        self.validation_history: deque = deque(maxlen=10000)
        self.hallucination_history: deque = deque(maxlen=1000)

        # Statistics
        self.total_validations = 0
        self.total_rejections = 0
        self.total_hallucinations = 0

        logger.info(f"AIValidator initialized (strict_mode={enable_strict_mode})")

    # ===== Main Validation Entry Point =====

    def validate_decision(
        self,
        context: DecisionContext,
        ensemble_predictions: Optional[List[float]] = None
    ) -> ValidationReport:
        """
        Main validation entry point

        Performs comprehensive validation of a trading decision.

        Args:
            context: Decision context with all relevant information
            ensemble_predictions: Optional predictions from multiple models

        Returns:
            ValidationReport with detailed results
        """
        start_time = datetime.now()

        logger.info(f"Validating decision {context.decision_id} ({context.action} {context.symbol})")

        try:
            # Layer 1: Hallucination detection
            hallucination_report = self.detect_hallucination(
                context, ensemble_predictions
            )

            # Layer 2: Data authenticity
            data_authentic = self.validate_data_authenticity(context)

            # Layer 3: Decision logic
            logic_valid = self.validate_decision_logic(context)

            # Layer 4: Adversarial robustness
            adversarial_robust = self.check_adversarial_robustness(context)

            # Layer 5: Confidence calibration
            confidence_calibrated = self.validate_confidence_calibration(context)

            # Build report
            concerns = []
            recommendations = []

            if hallucination_report.detected:
                concerns.append(f"Hallucination detected: {hallucination_report.reason}")

            if not data_authentic:
                concerns.append("Data authenticity check failed")

            if not logic_valid:
                concerns.append("Decision logic validation failed")

            if not adversarial_robust:
                concerns.append("Adversarial robustness concerns")

            if not confidence_calibrated:
                recommendations.append("Confidence may be miscalibrated")

            # Determine overall status
            if hallucination_report.severity == HallucinationSeverity.CRITICAL:
                status = ValidationStatus.REJECTED
                execution_allowed = False
            elif not data_authentic or not logic_valid:
                status = ValidationStatus.REJECTED
                execution_allowed = False
            elif len(concerns) > 0:
                if self.strict_mode:
                    status = ValidationStatus.REJECTED
                    execution_allowed = False
                else:
                    status = ValidationStatus.WARNING
                    execution_allowed = True
            else:
                status = ValidationStatus.APPROVED
                execution_allowed = True

            # Calculate overall confidence
            check_scores = [
                1.0 - hallucination_report.confidence if hallucination_report.detected else 1.0,
                1.0 if data_authentic else 0.0,
                1.0 if logic_valid else 0.0,
                1.0 if adversarial_robust else 0.7,
                1.0 if confidence_calibrated else 0.9
            ]
            overall_confidence = np.mean(check_scores)

            # Measure latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Create report
            report = ValidationReport(
                request_id=context.decision_id,
                timestamp=datetime.now(),
                validation_status=status,
                overall_confidence=overall_confidence,
                hallucination_check=hallucination_report,
                data_authenticity_passed=data_authentic,
                decision_logic_valid=logic_valid,
                adversarial_robust=adversarial_robust,
                confidence_calibrated=confidence_calibrated,
                concerns=concerns,
                recommendations=recommendations,
                audit_logged=True,
                execution_allowed=execution_allowed,
                validation_latency_ms=latency_ms
            )

            # Record in history
            self._record_validation(report)

            logger.info(
                f"Validation complete: {status.value} "
                f"(confidence={overall_confidence:.3f}, latency={latency_ms:.1f}ms)"
            )

            return report

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            # Fail closed - reject on error
            return self._create_error_report(context, str(e))

    # ===== Layer 1: Hallucination Detection =====

    def detect_hallucination(
        self,
        context: DecisionContext,
        ensemble_predictions: Optional[List[float]] = None
    ) -> HallucinationReport:
        """
        Multi-layer hallucination detection

        Checks:
        1. Statistical plausibility
        2. Historical consistency
        3. Ensemble agreement
        4. Logical coherence
        5. Source attribution
        """

        # Sub-layer 1: Statistical plausibility
        stat_check = self._check_statistical_plausibility(context)
        if stat_check[0]:
            return stat_check[1]

        # Sub-layer 2: Historical consistency
        hist_check = self._check_historical_consistency(context)
        if hist_check[0]:
            return hist_check[1]

        # Sub-layer 3: Ensemble agreement
        if ensemble_predictions is not None:
            ensemble_check = self._check_ensemble_agreement(
                context, ensemble_predictions
            )
            if ensemble_check[0]:
                return ensemble_check[1]

        # Sub-layer 4: Logical coherence
        logic_check = self._check_logical_coherence(context)
        if logic_check[0]:
            return logic_check[1]

        # Sub-layer 5: Source attribution
        source_check = self._check_source_attribution(context)
        if source_check[0]:
            return source_check[1]

        # No hallucination detected
        return HallucinationReport(
            detected=False,
            severity=HallucinationSeverity.NONE,
            layer="all",
            reason="All checks passed",
            evidence={},
            confidence=0.99
        )

    def _check_statistical_plausibility(
        self, context: DecisionContext
    ) -> Tuple[bool, Optional[HallucinationReport]]:
        """Check if values are statistically plausible"""

        # Check for impossible values
        if context.current_price <= 0:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                layer="statistical",
                reason="Negative or zero price is impossible",
                evidence={"price": context.current_price},
                confidence=1.0
            )

        if context.confidence < 0 or context.confidence > 1:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                layer="statistical",
                reason="Confidence outside [0, 1] range",
                evidence={"confidence": context.confidence},
                confidence=1.0
            )

        if context.quantity < 0:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                layer="statistical",
                reason="Negative quantity is impossible",
                evidence={"quantity": context.quantity},
                confidence=1.0
            )

        # Check for extreme outliers
        if len(context.historical_data) > 0:
            hist_prices = context.historical_data['close'].values
            mean_price = np.mean(hist_prices)
            std_price = np.std(hist_prices)

            z_score = abs((context.current_price - mean_price) / (std_price + 1e-10))

            if z_score > 5:
                return True, HallucinationReport(
                    detected=True,
                    severity=HallucinationSeverity.HIGH,
                    layer="statistical",
                    reason=f"Price is {z_score:.1f}-sigma outlier",
                    evidence={"z_score": z_score, "price": context.current_price},
                    confidence=0.95
                )

        return False, None

    def _check_historical_consistency(
        self, context: DecisionContext
    ) -> Tuple[bool, Optional[HallucinationReport]]:
        """Check consistency with historical patterns"""

        if len(context.historical_data) < 50:
            return False, None  # Insufficient history

        hist_returns = context.historical_data['close'].pct_change().dropna()
        hist_volatility = hist_returns.std()

        # Calculate implied return from decision
        if context.action == 'BUY':
            implied_return = context.expected_value / (context.current_price * context.quantity)
        elif context.action == 'SELL':
            implied_return = -context.expected_value / (context.current_price * context.quantity)
        else:
            return False, None

        # Check if implied return is realistic
        if abs(implied_return) > 10 * hist_volatility:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.MEDIUM,
                layer="historical",
                reason=f"Implied return ({implied_return:.2%}) exceeds 10x historical volatility",
                evidence={
                    "implied_return": implied_return,
                    "historical_volatility": hist_volatility
                },
                confidence=0.85
            )

        return False, None

    def _check_ensemble_agreement(
        self,
        context: DecisionContext,
        ensemble_predictions: List[float]
    ) -> Tuple[bool, Optional[HallucinationReport]]:
        """Check agreement among ensemble models"""

        if len(ensemble_predictions) < 2:
            return False, None

        mean_pred = np.mean(ensemble_predictions)
        std_pred = np.std(ensemble_predictions)

        # Calculate coefficient of variation
        if abs(mean_pred) > 1e-10:
            cv = std_pred / abs(mean_pred)
        else:
            cv = std_pred

        if cv > self.ensemble_disagreement_threshold:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.MEDIUM,
                layer="ensemble",
                reason=f"High disagreement among models (CV={cv:.2f})",
                evidence={
                    "predictions": ensemble_predictions,
                    "mean": mean_pred,
                    "std": std_pred,
                    "cv": cv
                },
                confidence=0.80
            )

        return False, None

    def _check_logical_coherence(
        self, context: DecisionContext
    ) -> Tuple[bool, Optional[HallucinationReport]]:
        """Check for logical contradictions"""

        # Check action matches expected value
        if context.action == 'BUY' and context.expected_value < 0:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                layer="logical",
                reason="BUY decision has negative expected value",
                evidence={
                    "action": context.action,
                    "expected_value": context.expected_value
                },
                confidence=0.98
            )

        if context.action == 'SELL' and context.expected_value < 0:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                layer="logical",
                reason="SELL decision has negative expected value",
                evidence={
                    "action": context.action,
                    "expected_value": context.expected_value
                },
                confidence=0.98
            )

        # Check risk-reward makes sense
        if context.expected_value < abs(context.max_loss):
            # This is actually fine for asymmetric bets, so just flag as low severity
            if context.probability_success < 0.5:
                return True, HallucinationReport(
                    detected=True,
                    severity=HallucinationSeverity.LOW,
                    layer="logical",
                    reason="Negative risk-reward with low probability",
                    evidence={
                        "expected_value": context.expected_value,
                        "max_loss": context.max_loss,
                        "probability": context.probability_success
                    },
                    confidence=0.70
                )

        return False, None

    def _check_source_attribution(
        self, context: DecisionContext
    ) -> Tuple[bool, Optional[HallucinationReport]]:
        """Check that all data has verifiable sources"""

        if not context.data_sources or len(context.data_sources) == 0:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.HIGH,
                layer="source",
                reason="No data sources provided",
                evidence={"sources": context.data_sources},
                confidence=0.95
            )

        # Check for known trusted sources
        trusted_sources = ['coinbase', 'polygon', 'tradingview', 'binance', 'kraken']
        has_trusted_source = any(
            source.lower() in trusted_sources
            for source in context.data_sources
        )

        if not has_trusted_source:
            return True, HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.MEDIUM,
                layer="source",
                reason="No trusted data sources found",
                evidence={"sources": context.data_sources},
                confidence=0.80
            )

        return False, None

    # ===== Layer 2: Data Authenticity =====

    def validate_data_authenticity(self, context: DecisionContext) -> bool:
        """Validate data authenticity and integrity"""

        # Check timestamp is realistic
        now = datetime.now()
        if context.timestamp > now + timedelta(minutes=1):
            logger.warning("Future timestamp detected")
            return False

        if context.timestamp < now - timedelta(days=7):
            logger.warning("Very old timestamp (>7 days)")
            return False

        # Check data sources
        if not context.data_sources:
            logger.warning("No data sources provided")
            return False

        # Check historical data is valid
        if len(context.historical_data) > 0:
            if context.historical_data.isnull().sum().sum() > len(context.historical_data) * 0.1:
                logger.warning("Too many null values in historical data")
                return False

        return True

    # ===== Layer 3: Decision Logic =====

    def validate_decision_logic(self, context: DecisionContext) -> bool:
        """Validate decision logic is sound"""

        # Check reasoning is provided
        if not context.reasoning or len(context.reasoning) == 0:
            logger.warning("No reasoning provided for decision")
            return False

        # Check model version is specified
        if not context.model_version:
            logger.warning("No model version specified")
            return False

        # Check confidence is reasonable
        if context.confidence < 0.3:
            logger.warning(f"Very low confidence: {context.confidence}")
            return False

        # Check expected value calculation makes sense
        if context.action != 'HOLD':
            if context.expected_value == 0 and context.max_loss == 0:
                logger.warning("Zero expected value and loss - unrealistic")
                return False

        return True

    # ===== Layer 4: Adversarial Robustness =====

    def check_adversarial_robustness(self, context: DecisionContext) -> bool:
        """Check for adversarial patterns"""

        # Check for suspiciously perfect predictions
        if context.confidence > 0.99 and context.probability_success > 0.99:
            logger.warning("Suspiciously high confidence - possible overfit")
            return False

        # Check for unusual feature patterns
        if len(context.features) > 0:
            if np.any(np.isnan(context.features)) or np.any(np.isinf(context.features)):
                logger.warning("NaN or Inf in features")
                return False

            # Check for extreme feature values
            if np.any(np.abs(context.features) > 100):
                logger.warning("Extreme feature values detected")
                return False

        return True

    # ===== Layer 5: Confidence Calibration =====

    def validate_confidence_calibration(self, context: DecisionContext) -> bool:
        """Check if confidence is properly calibrated"""

        # This requires historical data of predictions vs outcomes
        # For now, just check if confidence is reasonable

        if context.confidence > 0.95:
            # Very high confidence - should be rare
            if len(self.validation_history) > 100:
                recent_high_conf = sum(
                    1 for v in list(self.validation_history)[-100:]
                    if hasattr(v, 'confidence') and v.confidence > 0.95
                )
                if recent_high_conf > 10:
                    logger.warning("Too many high-confidence predictions")
                    return False

        return True

    # ===== Utility Methods =====

    def _record_validation(self, report: ValidationReport):
        """Record validation in history"""
        self.validation_history.append(report)
        self.total_validations += 1

        if report.validation_status == ValidationStatus.REJECTED:
            self.total_rejections += 1

        if report.hallucination_check.detected:
            self.hallucination_history.append(report.hallucination_check)
            self.total_hallucinations += 1

    def _create_error_report(
        self, context: DecisionContext, error_msg: str
    ) -> ValidationReport:
        """Create error report when validation fails"""
        return ValidationReport(
            request_id=context.decision_id,
            timestamp=datetime.now(),
            validation_status=ValidationStatus.REJECTED,
            overall_confidence=0.0,
            hallucination_check=HallucinationReport(
                detected=True,
                severity=HallucinationSeverity.CRITICAL,
                layer="error",
                reason=f"Validation error: {error_msg}",
                evidence={"error": error_msg},
                confidence=1.0
            ),
            data_authenticity_passed=False,
            decision_logic_valid=False,
            adversarial_robust=False,
            confidence_calibrated=False,
            concerns=[f"Validation error: {error_msg}"],
            recommendations=["Review system logs"],
            audit_logged=True,
            execution_allowed=False,
            validation_latency_ms=0.0
        )

    # ===== Statistics and Monitoring =====

    @lru_cache(maxsize=128)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "total_validations": self.total_validations,
            "total_rejections": self.total_rejections,
            "total_hallucinations": self.total_hallucinations,
            "rejection_rate": self.total_rejections / max(self.total_validations, 1),
            "hallucination_rate": self.total_hallucinations / max(self.total_validations, 1),
            "recent_validations": len(self.validation_history),
            "recent_hallucinations": len(self.hallucination_history)
        }

    @lru_cache(maxsize=128)

    def get_recent_hallucinations(self, n: int = 10) -> List[HallucinationReport]:
        """Get recent hallucination reports"""
        return list(self.hallucination_history)[-n:]

    def reset_statistics(self):
        """Reset statistics (for testing)"""
        self.total_validations = 0
        self.total_rejections = 0
        self.total_hallucinations = 0
        logger.info("Statistics reset")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create validator
    validator = AIValidator(enable_strict_mode=True)

    # Create sample context
    context = DecisionContext(
        decision_id="test-001",
        timestamp=datetime.now(),
        decision_type="TRADE",
        symbol="BTC-USD",
        current_price=50000.0,
        features=np.array([0.23, -0.45, 0.67, -0.12]),
        historical_data=pd.DataFrame({
            'close': np.random.randn(100) * 1000 + 50000
        }),
        action="BUY",
        quantity=0.001,
        confidence=0.72,
        reasoning=["RSI oversold", "Bullish divergence"],
        model_version="v1.0.0",
        data_sources=["coinbase", "tradingview"],
        expected_value=150.0,
        max_loss=-2000.0,
        probability_success=0.72
    )

    # Validate
    report = validator.validate_decision(context)

    print("\n=== Validation Report ===")
    print(f"Status: {report.validation_status.value}")
    print(f"Confidence: {report.overall_confidence:.3f}")
    print(f"Execution Allowed: {report.execution_allowed}")
    print(f"Latency: {report.validation_latency_ms:.1f}ms")

    if report.concerns:
        print("\nConcerns:")
        for concern in report.concerns:
            print(f"  - {concern}")

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    print(f"\nStatistics: {validator.get_statistics()}")
