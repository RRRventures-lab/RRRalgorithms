"""
Circuit Breaker for Trading Engine

Implements circuit breaker pattern to halt trading during anomalous conditions:
- Excessive losses
- High volatility
- API errors
- Position limit breaches
- Drawdown thresholds

Acts as a safety mechanism to prevent cascade failures.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing if conditions improved


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker thresholds"""
    # Loss limits
    max_daily_loss_pct: float = 0.05  # 5% daily loss
    max_drawdown_pct: float = 0.10  # 10% drawdown from peak

    # Position limits
    max_open_positions: int = 5
    max_position_size_pct: float = 0.30  # 30% of portfolio

    # Volatility limits
    max_portfolio_volatility: float = 0.50  # 50% annualized volatility

    # API error limits
    max_consecutive_errors: int = 5
    max_error_rate: float = 0.20  # 20% error rate

    # Circuit breaker timing
    cooldown_period_minutes: int = 30  # Time before attempting half-open
    test_period_minutes: int = 5  # Time in half-open before closing


@dataclass
class CircuitBreakerMetrics:
    """Current metrics tracked by circuit breaker"""
    # Loss tracking
    daily_pnl: float = 0.0
    peak_portfolio_value: float = 0.0
    current_portfolio_value: float = 0.0
    current_drawdown_pct: float = 0.0

    # Position tracking
    open_positions: int = 0
    largest_position_pct: float = 0.0

    # Error tracking
    consecutive_errors: int = 0
    total_requests: int = 0
    total_errors: int = 0

    # Volatility
    portfolio_volatility: float = 0.0

    # Timestamps
    last_error_time: Optional[datetime] = None
    circuit_opened_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """
    Circuit breaker for trading engine

    Monitors trading conditions and halts trading when risk limits are exceeded.
    Implements three-state pattern: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker

        Args:
            config: Configuration for thresholds (uses defaults if None)
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()

        # Violation history
        self.violations: List[Dict[str, Any]] = []

        # Reset daily metrics
        self._last_reset = datetime.utcnow()

        logger.info(f"Circuit breaker initialized in {self.state.value.upper()} state")

    def check_and_update(
        self,
        portfolio_value: float,
        daily_pnl: float,
        open_positions: int,
        largest_position_pct: float,
        portfolio_volatility: float,
    ) -> bool:
        """
        Check conditions and update circuit breaker state

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Profit/loss for the day
            open_positions: Number of open positions
            largest_position_pct: Largest position as % of portfolio
            portfolio_volatility: Annualized portfolio volatility

        Returns:
            True if trading is allowed, False if circuit is open
        """
        # Update metrics
        self.metrics.current_portfolio_value = portfolio_value
        self.metrics.daily_pnl = daily_pnl
        self.metrics.open_positions = open_positions
        self.metrics.largest_position_pct = largest_position_pct
        self.metrics.portfolio_volatility = portfolio_volatility
        self.metrics.last_update_time = datetime.utcnow()

        # Update peak value
        if portfolio_value > self.metrics.peak_portfolio_value:
            self.metrics.peak_portfolio_value = portfolio_value

        # Calculate drawdown
        if self.metrics.peak_portfolio_value > 0:
            self.metrics.current_drawdown_pct = (
                (self.metrics.peak_portfolio_value - portfolio_value)
                / self.metrics.peak_portfolio_value
            )

        # Reset daily metrics if new day
        self._check_daily_reset()

        # Check violations based on current state
        violations = self._check_violations()

        if self.state == CircuitState.CLOSED:
            if violations:
                self._open_circuit(violations)
                return False
            return True

        elif self.state == CircuitState.OPEN:
            # Check if cooldown period has elapsed
            if self.metrics.circuit_opened_time:
                elapsed = datetime.utcnow() - self.metrics.circuit_opened_time
                if elapsed > timedelta(minutes=self.config.cooldown_period_minutes):
                    self._half_open_circuit()
            return False

        elif self.state == CircuitState.HALF_OPEN:
            # In test mode - allow limited trading
            if violations:
                # Violations persist, reopen circuit
                self._open_circuit(violations)
                return False
            else:
                # Check if test period completed successfully
                if self.metrics.circuit_opened_time:
                    elapsed = datetime.utcnow() - self.metrics.circuit_opened_time
                    if elapsed > timedelta(minutes=self.config.test_period_minutes):
                        self._close_circuit()
                        return True
                return True  # Allow trading during test period

        return False

    def _check_violations(self) -> List[str]:
        """
        Check for risk limit violations

        Returns:
            List of violation messages
        """
        violations = []

        # Check daily loss
        if self.metrics.current_portfolio_value > 0:
            daily_loss_pct = abs(self.metrics.daily_pnl / self.metrics.current_portfolio_value)
            if self.metrics.daily_pnl < 0 and daily_loss_pct > self.config.max_daily_loss_pct:
                violations.append(
                    f"Daily loss ({daily_loss_pct:.1%}) exceeds limit ({self.config.max_daily_loss_pct:.1%})"
                )

        # Check drawdown
        if self.metrics.current_drawdown_pct > self.config.max_drawdown_pct:
            violations.append(
                f"Drawdown ({self.metrics.current_drawdown_pct:.1%}) exceeds limit ({self.config.max_drawdown_pct:.1%})"
            )

        # Check position limits
        if self.metrics.open_positions > self.config.max_open_positions:
            violations.append(
                f"Open positions ({self.metrics.open_positions}) exceeds limit ({self.config.max_open_positions})"
            )

        if self.metrics.largest_position_pct > self.config.max_position_size_pct:
            violations.append(
                f"Largest position ({self.metrics.largest_position_pct:.1%}) exceeds limit ({self.config.max_position_size_pct:.1%})"
            )

        # Check volatility
        if self.metrics.portfolio_volatility > self.config.max_portfolio_volatility:
            violations.append(
                f"Portfolio volatility ({self.metrics.portfolio_volatility:.1%}) exceeds limit ({self.config.max_portfolio_volatility:.1%})"
            )

        # Check error rate
        if self.metrics.total_requests > 10:  # Only check after minimum requests
            error_rate = self.metrics.total_errors / self.metrics.total_requests
            if error_rate > self.config.max_error_rate:
                violations.append(
                    f"Error rate ({error_rate:.1%}) exceeds limit ({self.config.max_error_rate:.1%})"
                )

        # Check consecutive errors
        if self.metrics.consecutive_errors >= self.config.max_consecutive_errors:
            violations.append(
                f"Consecutive errors ({self.metrics.consecutive_errors}) exceeds limit ({self.config.max_consecutive_errors})"
            )

        return violations

    def _open_circuit(self, violations: List[str]):
        """
        Open the circuit (halt trading)

        Args:
            violations: List of violations that triggered opening
        """
        self.state = CircuitState.OPEN
        self.metrics.circuit_opened_time = datetime.utcnow()

        # Record violations
        violation_record = {
            "timestamp": self.metrics.circuit_opened_time.isoformat(),
            "violations": violations,
            "metrics": {
                "daily_pnl": self.metrics.daily_pnl,
                "drawdown_pct": self.metrics.current_drawdown_pct,
                "open_positions": self.metrics.open_positions,
                "portfolio_volatility": self.metrics.portfolio_volatility,
            },
        }
        self.violations.append(violation_record)

        logger.critical("=" * 70)
        logger.critical("CIRCUIT BREAKER OPENED - TRADING HALTED")
        logger.critical("=" * 70)
        for violation in violations:
            logger.critical(f"  VIOLATION: {violation}")
        logger.critical("=" * 70)

    def _half_open_circuit(self):
        """Move to half-open state (test if conditions improved)"""
        self.state = CircuitState.HALF_OPEN
        logger.warning("Circuit breaker entering HALF-OPEN state - testing conditions")

    def _close_circuit(self):
        """Close the circuit (resume normal trading)"""
        self.state = CircuitState.CLOSED
        self.metrics.circuit_opened_time = None
        logger.info("Circuit breaker CLOSED - resuming normal trading")

    def record_error(self):
        """Record an API or execution error"""
        self.metrics.total_errors += 1
        self.metrics.consecutive_errors += 1
        self.metrics.last_error_time = datetime.utcnow()

    def record_success(self):
        """Record a successful operation"""
        self.metrics.total_requests += 1
        self.metrics.consecutive_errors = 0  # Reset consecutive errors

    def _check_daily_reset(self):
        """Reset daily metrics if it's a new day"""
        now = datetime.utcnow()
        if now.date() > self._last_reset.date():
            logger.info("Resetting daily circuit breaker metrics")
            self.metrics.daily_pnl = 0.0
            self.metrics.total_requests = 0
            self.metrics.total_errors = 0
            self.metrics.consecutive_errors = 0
            self._last_reset = now

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed

        Returns:
            True if circuit is closed or half-open, False if open
        """
        return self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]

    def get_status(self) -> Dict[str, Any]:
        """
        Get current circuit breaker status

        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "trading_allowed": self.is_trading_allowed(),
            "metrics": {
                "daily_pnl": self.metrics.daily_pnl,
                "current_drawdown_pct": self.metrics.current_drawdown_pct,
                "open_positions": self.metrics.open_positions,
                "largest_position_pct": self.metrics.largest_position_pct,
                "portfolio_volatility": self.metrics.portfolio_volatility,
                "consecutive_errors": self.metrics.consecutive_errors,
                "error_rate": (
                    self.metrics.total_errors / self.metrics.total_requests
                    if self.metrics.total_requests > 0
                    else 0
                ),
            },
            "circuit_opened_time": (
                self.metrics.circuit_opened_time.isoformat()
                if self.metrics.circuit_opened_time
                else None
            ),
            "last_update": self.metrics.last_update_time.isoformat(),
        }

    def get_violation_history(self) -> List[Dict[str, Any]]:
        """Get history of circuit breaker violations"""
        return self.violations

    def manual_open(self, reason: str):
        """
        Manually open the circuit

        Args:
            reason: Reason for manual opening
        """
        self._open_circuit([f"Manual override: {reason}"])

    def manual_close(self):
        """Manually close the circuit (use with caution)"""
        logger.warning("Circuit breaker manually closed - ensure conditions are safe")
        self._close_circuit()


def main():
    """Example usage"""
    print("=" * 70)
    print("Circuit Breaker - Test Scenarios")
    print("=" * 70)

    # Create circuit breaker with test config
    config = CircuitBreakerConfig(
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.10,
        max_open_positions=3,
    )

    breaker = CircuitBreaker(config)

    # Scenario 1: Normal operation
    print("\nScenario 1: Normal operation")
    allowed = breaker.check_and_update(
        portfolio_value=100000,
        daily_pnl=500,
        open_positions=2,
        largest_position_pct=0.20,
        portfolio_volatility=0.25,
    )
    print(f"Trading allowed: {allowed}")
    print(f"Circuit state: {breaker.state.value}")

    # Scenario 2: Excessive daily loss
    print("\nScenario 2: Excessive daily loss")
    allowed = breaker.check_and_update(
        portfolio_value=94000,
        daily_pnl=-6000,  # 6% loss
        open_positions=2,
        largest_position_pct=0.20,
        portfolio_volatility=0.25,
    )
    print(f"Trading allowed: {allowed}")
    print(f"Circuit state: {breaker.state.value}")

    # Check status
    print("\nCircuit Breaker Status:")
    status = breaker.get_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    main()
