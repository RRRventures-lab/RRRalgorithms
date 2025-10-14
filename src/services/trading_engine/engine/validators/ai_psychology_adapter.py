from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional
from validation.ai_validator_integration import (
import logging
import sys

#!/usr/bin/env python3

"""
AI Psychology Team Integration Adapter

Integrates the AI Psychology Team validation system into the trading engine.
Provides a clean interface for order validation before execution.

Author: AI Psychology Team
Date: 2025-10-11
"""


# Add monitoring worktree to path
monitoring_path = Path(__file__).parent.parent.parent.parent.parent / "monitoring" / "src"
sys.path.insert(0, str(monitoring_path))

    AIValidatorIntegration,
    ValidationRequest,
    ValidationResponse,
    UrgencyLevel
)

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    """Order action types"""
    BUY = "BUY"
    SELL = "SELL"
    CANCEL = "CANCEL"


class AIPsychologyAdapter:
    """
    Adapter for integrating AI Psychology Team validation into trading engine

    Converts trading engine orders into validation requests and processes responses.
    """

    def __init__(
        self,
        enable_validation: bool = True,
        fail_open: bool = False,  # Allow trading if validator fails
        timeout_ms: float = 50.0
    ):
        """
        Initialize adapter

        Args:
            enable_validation: Enable/disable validation
            fail_open: If True, allow trading when validator fails
            timeout_ms: Validation timeout in milliseconds
        """
        self.enable_validation = enable_validation
        self.fail_open = fail_open
        self.timeout_ms = timeout_ms

        # Initialize validator integration
        if self.enable_validation:
            self.validator = AIValidatorIntegration()
            logger.info("AI Psychology validator initialized")
        else:
            self.validator = None
            logger.warning("AI Psychology validation DISABLED")

        # Statistics
        self.total_validations = 0
        self.approved = 0
        self.rejected = 0
        self.errors = 0

    async def validate_order_async(
        self,
        order: Dict[str, Any],
        model_info: Dict[str, str],
        market_context: Dict[str, Any],
        reasoning: Dict[str, Any],
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> tuple[bool, Optional[ValidationResponse]]:
        """
        Validate an order before execution (async)

        Args:
            order: Order details (symbol, side, quantity, price, etc.)
            model_info: Model name and version
            market_context: Current market conditions
            reasoning: Decision reasoning
            risk_metrics: Risk assessment metrics

        Returns:
            Tuple of (execution_allowed, validation_response)
        """
        if not self.enable_validation:
            return True, None

        try:
            # Create validation request
            request = self._create_validation_request(
                order, model_info, market_context, reasoning, risk_metrics
            )

            # Validate
            response = await self.validator.validate_decision_async(request)

            # Update statistics
            self.total_validations += 1
            if response.execution_allowed:
                self.approved += 1
            else:
                self.rejected += 1

            # Log result
            if response.execution_allowed:
                logger.info(
                    f"Order {order.get('order_id')} APPROVED "
                    f"(confidence: {response.confidence:.2%}, "
                    f"latency: {response.processing_time_ms:.1f}ms)"
                )
            else:
                logger.warning(
                    f"Order {order.get('order_id')} REJECTED: "
                    f"{response.concerns}"
                )

            return response.execution_allowed, response

        except Exception as e:
            self.errors += 1
            logger.error(f"Validation error: {e}")

            # Fail-open or fail-closed
            if self.fail_open:
                logger.warning("Validation failed, proceeding anyway (fail-open)")
                return True, None
            else:
                logger.error("Validation failed, blocking order (fail-closed)")
                return False, None

    def validate_order_sync(
        self,
        order: Dict[str, Any],
        model_info: Dict[str, str],
        market_context: Dict[str, Any],
        reasoning: Dict[str, Any],
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> tuple[bool, Optional[ValidationResponse]]:
        """
        Validate an order before execution (sync)

        See validate_order_async for parameters.
        """
        if not self.enable_validation:
            return True, None

        try:
            # Create validation request
            request = self._create_validation_request(
                order, model_info, market_context, reasoning, risk_metrics
            )

            # Validate
            response = self.validator.validate_decision_sync(request)

            # Update statistics
            self.total_validations += 1
            if response.execution_allowed:
                self.approved += 1
            else:
                self.rejected += 1

            # Log result
            if response.execution_allowed:
                logger.info(
                    f"Order {order.get('order_id')} APPROVED "
                    f"(confidence: {response.confidence:.2%}, "
                    f"latency: {response.processing_time_ms:.1f}ms)"
                )
            else:
                logger.warning(
                    f"Order {order.get('order_id')} REJECTED: "
                    f"{response.concerns}"
                )

            return response.execution_allowed, response

        except Exception as e:
            self.errors += 1
            logger.error(f"Validation error: {e}")

            # Fail-open or fail-closed
            if self.fail_open:
                logger.warning("Validation failed, proceeding anyway (fail-open)")
                return True, None
            else:
                logger.error("Validation failed, blocking order (fail-closed)")
                return False, None

    def _create_validation_request(
        self,
        order: Dict[str, Any],
        model_info: Dict[str, str],
        market_context: Dict[str, Any],
        reasoning: Dict[str, Any],
        risk_metrics: Optional[Dict[str, float]]
    ) -> ValidationRequest:
        """Create validation request from order"""
        import uuid

        # Map order action to decision type
        action = order.get('side', 'BUY').upper()

        # Determine urgency from order type
        order_type = order.get('type', 'LIMIT')
        if order_type == 'MARKET':
            urgency = UrgencyLevel.HIGH
        elif order_type == 'STOP_LOSS':
            urgency = UrgencyLevel.CRITICAL
        else:
            urgency = UrgencyLevel.NORMAL

        # Build request
        request = ValidationRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            request_type="TRADE_DECISION",

            # Model info
            model_name=model_info.get('model_name', 'unknown'),
            model_version=model_info.get('model_version', '1.0.0'),

            # Decision details
            decision_id=order.get('order_id', str(uuid.uuid4())),
            decision_type=action,
            symbol=order.get('symbol', 'UNKNOWN'),
            quantity=float(order.get('quantity', 0)),
            price=float(order.get('price', 0)),
            confidence=float(order.get('confidence', 0.5)),
            urgency=urgency,
            timeout_ms=self.timeout_ms,

            # Inputs
            features=market_context.get('features', []),
            feature_names=market_context.get('feature_names', []),
            current_price=float(market_context.get('current_price', 0)),
            historical_prices=market_context.get('historical_prices', []),
            market_context=market_context,

            # Reasoning
            reasoning=reasoning,
            ensemble_predictions=reasoning.get('ensemble_predictions'),
            data_sources=reasoning.get('data_sources'),

            # Risk assessment
            risk_assessment=risk_metrics
        )

        return request

    @lru_cache(maxsize=128)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.total_validations
        approval_rate = self.approved / total if total > 0 else 0
        rejection_rate = self.rejected / total if total > 0 else 0
        error_rate = self.errors / total if total > 0 else 0

        return {
            "total_validations": total,
            "approved": self.approved,
            "rejected": self.rejected,
            "errors": self.errors,
            "approval_rate": approval_rate,
            "rejection_rate": rejection_rate,
            "error_rate": error_rate,
            "validation_enabled": self.enable_validation,
            "fail_open_mode": self.fail_open
        }


# Singleton instance
_global_adapter: Optional[AIPsychologyAdapter] = None


@lru_cache(maxsize=128)


def get_ai_psychology_adapter(
    enable_validation: bool = True,
    fail_open: bool = False,
    timeout_ms: float = 50.0
) -> AIPsychologyAdapter:
    """
    Get global AI Psychology adapter instance

    Args:
        enable_validation: Enable validation
        fail_open: Fail-open mode
        timeout_ms: Validation timeout

    Returns:
        AIPsychologyAdapter instance
    """
    global _global_adapter

    if _global_adapter is None:
        _global_adapter = AIPsychologyAdapter(
            enable_validation=enable_validation,
            fail_open=fail_open,
            timeout_ms=timeout_ms
        )

    return _global_adapter
