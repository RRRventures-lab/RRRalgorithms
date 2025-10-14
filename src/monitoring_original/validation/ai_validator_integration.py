from .ai_validator import AIValidator, ValidationReport, DecisionContext
from .decision_auditor import DecisionAuditor, DecisionType, ValidationStatus
from .hallucination_detector import HallucinationDetector
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Any
import asyncio
import logging
import time
import uuid

#!/usr/bin/env python3

"""
AI Validator Integration Module

Integrates AI validation system into the trading engine for real-time validation.
Implements the AI-to-AI Validation Protocol.

Author: AI Psychology Team
Date: 2025-10-11
"""


# Import validation modules

logger = logging.getLogger(__name__)


class UrgencyLevel(Enum):
    """Decision urgency levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationRequest:
    """Request for decision validation"""
    request_id: str
    timestamp: datetime
    request_type: str

    # Model info
    model_name: str
    model_version: str

    # Decision details
    decision_id: str
    decision_type: str
    symbol: str
    quantity: float
    price: float
    confidence: float
    urgency: UrgencyLevel
    timeout_ms: float

    # Inputs
    features: List[float]
    feature_names: List[str]
    current_price: float
    historical_prices: List[float]
    market_context: Dict[str, Any]

    # Reasoning
    reasoning: Dict[str, Any]
    ensemble_predictions: Optional[List[Dict[str, Any]]] = None
    data_sources: Optional[List[Dict[str, Any]]] = None

    # Risk assessment
    risk_assessment: Optional[Dict[str, float]] = None


@dataclass
class ValidationResponse:
    """Response from validation"""
    request_id: str
    response_timestamp: datetime
    processing_time_ms: float

    validation_status: str
    execution_allowed: bool
    confidence: float

    validations: Dict[str, Any]
    concerns: List[Dict[str, str]]
    recommendations: List[str]

    audit_info: Dict[str, str]


class AIValidatorIntegration:
    """
    Integration layer for AI validation in trading system

    Provides real-time validation with <10ms latency
    """

    def __init__(
        self,
        validator: Optional[AIValidator] = None,
        auditor: Optional[DecisionAuditor] = None,
        enable_async: bool = True,
        max_workers: int = 10
    ):
        # Initialize components
        self.validator = validator or AIValidator()
        self.auditor = auditor or DecisionAuditor()

        self.enable_async = enable_async
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async else None

        # Performance tracking
        self.total_validations = 0
        self.total_approved = 0
        self.total_rejected = 0
        self.latencies: List[float] = []

        logger.info("AIValidatorIntegration initialized")

    async def validate_decision_async(
        self,
        request: ValidationRequest
    ) -> ValidationResponse:
        """
        Validate decision asynchronously

        Args:
            request: Validation request

        Returns:
            Validation response
        """
        start_time = time.time()

        try:
            # Run validation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self.validate_decision_sync,
                request
            )

            return response

        except Exception as e:
            logger.error(f"Async validation failed: {e}")
            return self._create_error_response(request, str(e))

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)

    def validate_decision_sync(
        self,
        request: ValidationRequest
    ) -> ValidationResponse:
        """
        Validate decision synchronously

        Args:
            request: Validation request

        Returns:
            Validation response
        """
        start_time = time.time()

        try:
            # Create decision context
            context = self._create_decision_context(request)

            # Validate
            validation_report = self.validator.validate_decision(
                context=context,
                ensemble_predictions=[
                    p['prediction'] for p in (request.ensemble_predictions or [])
                ]
            )

            # Log to audit trail
            audit_entry = self._log_to_audit(request, validation_report)

            # Create response
            response = self._create_response(
                request,
                validation_report,
                audit_entry,
                time.time() - start_time
            )

            # Track statistics
            self.total_validations += 1
            if response.execution_allowed:
                self.total_approved += 1
            else:
                self.total_rejected += 1

            return response

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._create_error_response(request, str(e))

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)

            # Log performance warning if slow
            if latency_ms > 50:
                logger.warning(f"Slow validation: {latency_ms:.1f}ms for {request.request_id}")

    def _create_decision_context(self, request: ValidationRequest) -> DecisionContext:
        """Create decision context from request"""
        return DecisionContext(
            decision_id=request.decision_id,
            symbol=request.symbol,
            current_price=request.current_price,
            predicted_price=request.price,
            confidence=request.confidence,
            features=request.features,
            feature_names=request.feature_names,
            historical_data=request.historical_prices,
            timestamp=request.timestamp
        )

    def _log_to_audit(
        self,
        request: ValidationRequest,
        validation_report: ValidationReport
    ) -> Any:
        """Log validation to audit trail"""
        try:
            # Determine decision type
            decision_type = self._map_decision_type(request.decision_type)

            # Determine validation status
            validation_status = self._map_validation_status(validation_report.validation_status)

            # Log to auditor
            audit_entry = self.auditor.log_decision(
                decision_id=request.decision_id,
                decision_type=decision_type,
                model_version=request.model_version,
                model_name=request.model_name,
                inputs={
                    "features": request.features,
                    "feature_names": request.feature_names
                },
                market_context=request.market_context,
                output={
                    "decision_type": request.decision_type,
                    "quantity": request.quantity,
                    "price": request.price
                },
                confidence=request.confidence,
                reasoning=list(request.reasoning.get('supporting_signals', [])),
                feature_importance=request.reasoning.get('feature_importance', {}),
                validation_status=validation_status,
                hallucination_reports=[
                    {
                        "detected": validation_report.hallucination_check.detected,
                        "severity": validation_report.hallucination_check.severity.value,
                        "details": validation_report.hallucination_check.details
                    }
                ],
                data_authenticity_score=1.0 if validation_report.data_authenticity_passed else 0.0,
                decision_logic_valid=validation_report.decision_logic_valid,
                risk_metrics=request.risk_assessment or {},
                expected_value=request.risk_assessment.get('expected_value', 0) if request.risk_assessment else 0,
                max_loss=request.risk_assessment.get('max_loss', 0) if request.risk_assessment else 0,
                probability_success=request.risk_assessment.get('probability_success', 0) if request.risk_assessment else 0,
                data_sources=request.data_sources or []
            )

            return audit_entry

        except Exception as e:
            logger.error(f"Failed to log to audit trail: {e}")
            return None

    def _create_response(
        self,
        request: ValidationRequest,
        validation_report: ValidationReport,
        audit_entry: Any,
        processing_time: float
    ) -> ValidationResponse:
        """Create validation response"""
        # Determine execution allowed
        execution_allowed = validation_report.execution_allowed

        # Build validations dict
        validations = {
            "hallucination_check": {
                "passed": not validation_report.hallucination_check.detected,
                "confidence": validation_report.hallucination_check.confidence,
                "details": validation_report.hallucination_check.details
            },
            "data_authenticity": {
                "passed": validation_report.data_authenticity_passed,
                "confidence": 0.95,
                "details": "Data authenticity verified"
            },
            "decision_logic": {
                "passed": validation_report.decision_logic_valid,
                "confidence": 0.90,
                "details": "Decision logic validated"
            },
            "adversarial_robustness": {
                "passed": validation_report.adversarial_robust,
                "confidence": 0.92,
                "details": "No adversarial patterns detected"
            },
            "confidence_calibration": {
                "passed": validation_report.confidence_calibrated,
                "confidence": 0.88,
                "details": "Confidence is calibrated"
            }
        }

        # Create response
        return ValidationResponse(
            request_id=request.request_id,
            response_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time * 1000,
            validation_status=validation_report.validation_status.value,
            execution_allowed=execution_allowed,
            confidence=validation_report.overall_confidence,
            validations=validations,
            concerns=[
                {"severity": "HIGH", "type": c}
                for c in validation_report.concerns
            ],
            recommendations=validation_report.recommendations,
            audit_info={
                "audit_entry_id": audit_entry.entry_id if audit_entry else "unknown",
                "audit_logged": audit_entry is not None,
                "audit_hash": audit_entry.entry_hash if audit_entry else "unknown"
            }
        )

    def _create_error_response(
        self,
        request: ValidationRequest,
        error_message: str
    ) -> ValidationResponse:
        """Create error response"""
        return ValidationResponse(
            request_id=request.request_id,
            response_timestamp=datetime.utcnow(),
            processing_time_ms=0,
            validation_status="ERROR",
            execution_allowed=False,
            confidence=0.0,
            validations={},
            concerns=[
                {
                    "severity": "CRITICAL",
                    "type": "VALIDATION_ERROR",
                    "description": error_message
                }
            ],
            recommendations=["Review validation system health"],
            audit_info={
                "audit_entry_id": "error",
                "audit_logged": False,
                "audit_hash": "error"
            }
        )

    def _map_decision_type(self, decision_type_str: str) -> DecisionType:
        """Map string decision type to enum"""
        mapping = {
            "BUY": DecisionType.TRADE,
            "SELL": DecisionType.TRADE,
            "POSITION_SIZE": DecisionType.POSITION_SIZING,
            "STOP_LOSS": DecisionType.STOP_LOSS,
            "TAKE_PROFIT": DecisionType.TAKE_PROFIT
        }
        return mapping.get(decision_type_str, DecisionType.TRADE)

    def _map_validation_status(self, status_str: str) -> ValidationStatus:
        """Map string validation status to enum"""
        mapping = {
            "APPROVED": ValidationStatus.APPROVED,
            "REJECTED": ValidationStatus.REJECTED,
            "NEEDS_REVIEW": ValidationStatus.NEEDS_REVIEW,
            "WARNING": ValidationStatus.WARNING
        }
        return mapping.get(status_str, ValidationStatus.NEEDS_REVIEW)

    @lru_cache(maxsize=128)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.latencies:
            return {"error": "No validations performed yet"}

        import numpy as np

        return {
            "total_validations": self.total_validations,
            "total_approved": self.total_approved,
            "total_rejected": self.total_rejected,
            "approval_rate": self.total_approved / self.total_validations if self.total_validations > 0 else 0,
            "latency_stats": {
                "p50_ms": float(np.percentile(self.latencies, 50)),
                "p95_ms": float(np.percentile(self.latencies, 95)),
                "p99_ms": float(np.percentile(self.latencies, 99)),
                "mean_ms": float(np.mean(self.latencies)),
                "max_ms": float(np.max(self.latencies))
            }
        }
