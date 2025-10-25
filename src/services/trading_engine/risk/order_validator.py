"""
Order Validator - Pre-Flight Checks for Trading Orders

Validates orders before submission to exchange:
- Risk limit compliance
- Position sizing
- Order parameters
- Account balance
- Market conditions

Prevents invalid orders from reaching the exchange.
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Order validation result"""
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"


@dataclass
class OrderValidationResponse:
    """Response from order validation"""
    result: ValidationResult
    allowed: bool
    messages: List[str]
    warnings: List[str]
    order_details: Dict


class OrderValidator:
    """
    Pre-flight order validation

    Checks orders against risk limits and account constraints before submission.
    """

    def __init__(
        self,
        max_order_size_usd: float = 100.0,
        max_position_size_pct: float = 0.30,
        max_open_positions: int = 5,
        min_order_size_usd: float = 10.0,
        max_leverage: float = 1.0,
    ):
        """
        Initialize order validator

        Args:
            max_order_size_usd: Maximum order size in USD
            max_position_size_pct: Maximum position as % of portfolio
            max_open_positions: Maximum number of open positions
            min_order_size_usd: Minimum order size in USD
            max_leverage: Maximum leverage allowed
        """
        self.max_order_size_usd = max_order_size_usd
        self.max_position_size_pct = max_position_size_pct
        self.max_open_positions = max_open_positions
        self.min_order_size_usd = min_order_size_usd
        self.max_leverage = max_leverage

        logger.info(
            f"Order validator initialized: max_order=${max_order_size_usd}, "
            f"max_position={max_position_size_pct:.0%}, max_positions={max_open_positions}"
        )

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        current_price: float,
        portfolio_value: float,
        current_position_size: float,
        open_positions: int,
        available_balance: float,
    ) -> OrderValidationResponse:
        """
        Validate an order before submission

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: "market", "limit", "stop_loss"
            price: Limit/stop price (None for market orders)
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_position_size: Current position size in this symbol
            open_positions: Number of open positions
            available_balance: Available cash balance

        Returns:
            OrderValidationResponse with validation result
        """
        messages = []
        warnings = []
        errors = []

        # Calculate order value
        execution_price = price if price else current_price
        order_value_usd = quantity * execution_price

        # Order details for response
        order_details = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "current_price": current_price,
            "estimated_value_usd": order_value_usd,
        }

        # 1. Check symbol validity
        if not symbol or not isinstance(symbol, str):
            errors.append("Invalid symbol")

        # 2. Check side
        if side.lower() not in ["buy", "sell"]:
            errors.append(f"Invalid order side: {side}")

        # 3. Check quantity
        if quantity <= 0:
            errors.append(f"Invalid quantity: {quantity}")

        # 4. Check order type
        valid_order_types = ["market", "limit", "stop_loss", "take_profit"]
        if order_type.lower() not in valid_order_types:
            errors.append(f"Invalid order type: {order_type}")

        # 5. Check price for limit orders
        if order_type.lower() in ["limit", "stop_loss"] and not price:
            errors.append(f"{order_type} orders require a price")

        # 6. Check minimum order size
        if order_value_usd < self.min_order_size_usd:
            errors.append(
                f"Order value ${order_value_usd:.2f} below minimum ${self.min_order_size_usd:.2f}"
            )

        # 7. Check maximum order size
        if order_value_usd > self.max_order_size_usd:
            errors.append(
                f"Order value ${order_value_usd:.2f} exceeds maximum ${self.max_order_size_usd:.2f}"
            )

        # 8. Check position size limits
        if portfolio_value > 0:
            # Calculate new position size after this order
            if side.lower() == "buy":
                new_position_size = current_position_size + order_value_usd
            else:
                new_position_size = max(0, current_position_size - order_value_usd)

            position_pct = new_position_size / portfolio_value

            if position_pct > self.max_position_size_pct:
                errors.append(
                    f"Position size after order ({position_pct:.1%}) exceeds limit ({self.max_position_size_pct:.1%})"
                )
            elif position_pct > self.max_position_size_pct * 0.8:
                warnings.append(
                    f"Position size ({position_pct:.1%}) approaching limit ({self.max_position_size_pct:.1%})"
                )

        # 9. Check maximum open positions (for new positions)
        if side.lower() == "buy" and current_position_size == 0:
            if open_positions >= self.max_open_positions:
                errors.append(
                    f"Open positions ({open_positions}) at maximum ({self.max_open_positions})"
                )

        # 10. Check available balance for buy orders
        if side.lower() == "buy":
            required_balance = order_value_usd * 1.01  # Add 1% buffer for fees

            if required_balance > available_balance:
                errors.append(
                    f"Insufficient balance: need ${required_balance:.2f}, have ${available_balance:.2f}"
                )

        # 11. Check price reasonableness for limit orders
        if order_type.lower() == "limit" and price:
            if side.lower() == "buy":
                # Buy limit should be below market
                if price > current_price * 1.05:
                    warnings.append(
                        f"Buy limit price ${price:.2f} is {((price/current_price - 1)*100):.1f}% above market ${current_price:.2f}"
                    )
            else:
                # Sell limit should be above market
                if price < current_price * 0.95:
                    warnings.append(
                        f"Sell limit price ${price:.2f} is {((1 - price/current_price)*100):.1f}% below market ${current_price:.2f}"
                    )

        # 12. Check for position size relative to selling
        if side.lower() == "sell":
            if order_value_usd > current_position_size * 1.01:  # 1% buffer
                errors.append(
                    f"Cannot sell ${order_value_usd:.2f} when position is only ${current_position_size:.2f}"
                )

        # 13. Leverage check (for futures, not applicable to spot)
        # This is a placeholder for future futures trading support
        leverage = 1.0  # Spot trading has no leverage
        if leverage > self.max_leverage:
            errors.append(f"Leverage {leverage}x exceeds maximum {self.max_leverage}x")

        # Determine result
        if errors:
            result = ValidationResult.REJECTED
            allowed = False
            messages = errors
        elif warnings:
            result = ValidationResult.WARNING
            allowed = True  # Allow with warnings
            messages = ["Order approved with warnings"] + warnings
        else:
            result = ValidationResult.APPROVED
            allowed = True
            messages = ["Order validation passed"]

        # Log result
        if result == ValidationResult.REJECTED:
            logger.warning(f"Order REJECTED: {symbol} {side} {quantity} - {', '.join(errors)}")
        elif result == ValidationResult.WARNING:
            logger.info(f"Order APPROVED with warnings: {symbol} {side} {quantity}")
        else:
            logger.info(f"Order APPROVED: {symbol} {side} {quantity}")

        return OrderValidationResponse(
            result=result,
            allowed=allowed,
            messages=messages,
            warnings=warnings,
            order_details=order_details,
        )

    def validate_batch_orders(
        self,
        orders: List[Dict],
        portfolio_value: float,
        current_positions: Dict[str, float],
        available_balance: float,
    ) -> List[OrderValidationResponse]:
        """
        Validate multiple orders as a batch

        Args:
            orders: List of order dictionaries
            portfolio_value: Total portfolio value
            current_positions: Dictionary of symbol -> position size
            available_balance: Available cash balance

        Returns:
            List of validation responses
        """
        responses = []

        # Track cumulative impact on portfolio
        cumulative_balance_used = 0.0
        cumulative_positions = current_positions.copy()
        open_positions_count = len([p for p in current_positions.values() if p > 0])

        for order in orders:
            # Get current position for this symbol
            current_position = cumulative_positions.get(order["symbol"], 0.0)

            # Validate individual order
            response = self.validate_order(
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
                order_type=order["order_type"],
                price=order.get("price"),
                current_price=order["current_price"],
                portfolio_value=portfolio_value,
                current_position_size=current_position,
                open_positions=open_positions_count,
                available_balance=available_balance - cumulative_balance_used,
            )

            responses.append(response)

            # Update cumulative tracking if order was approved
            if response.allowed:
                order_value = order["quantity"] * (order.get("price") or order["current_price"])

                if order["side"].lower() == "buy":
                    cumulative_balance_used += order_value
                    cumulative_positions[order["symbol"]] = current_position + order_value
                    if current_position == 0:
                        open_positions_count += 1
                else:
                    cumulative_positions[order["symbol"]] = max(0, current_position - order_value)

        return responses


def main():
    """Example usage"""
    print("=" * 70)
    print("Order Validator - Test Cases")
    print("=" * 70)

    # Create validator
    validator = OrderValidator(
        max_order_size_usd=1000.0,
        max_position_size_pct=0.30,
        max_open_positions=5,
        min_order_size_usd=10.0,
    )

    # Test case 1: Valid buy order
    print("\nTest 1: Valid buy order")
    response = validator.validate_order(
        symbol="BTC-USD",
        side="buy",
        quantity=0.01,
        order_type="market",
        price=None,
        current_price=50000.0,
        portfolio_value=10000.0,
        current_position_size=0.0,
        open_positions=2,
        available_balance=5000.0,
    )
    print(f"Result: {response.result.value}")
    print(f"Allowed: {response.allowed}")
    print(f"Messages: {response.messages}")

    # Test case 2: Order too large
    print("\nTest 2: Order exceeds maximum size")
    response = validator.validate_order(
        symbol="BTC-USD",
        side="buy",
        quantity=0.1,  # $5000 order
        order_type="market",
        price=None,
        current_price=50000.0,
        portfolio_value=10000.0,
        current_position_size=0.0,
        open_positions=2,
        available_balance=10000.0,
    )
    print(f"Result: {response.result.value}")
    print(f"Allowed: {response.allowed}")
    print(f"Messages: {response.messages}")

    # Test case 3: Insufficient balance
    print("\nTest 3: Insufficient balance")
    response = validator.validate_order(
        symbol="BTC-USD",
        side="buy",
        quantity=0.01,
        order_type="market",
        price=None,
        current_price=50000.0,
        portfolio_value=10000.0,
        current_position_size=0.0,
        open_positions=2,
        available_balance=400.0,  # Not enough for $500 order
    )
    print(f"Result: {response.result.value}")
    print(f"Allowed: {response.allowed}")
    print(f"Messages: {response.messages}")

    # Test case 4: Position size limit
    print("\nTest 4: Position size approaching limit")
    response = validator.validate_order(
        symbol="BTC-USD",
        side="buy",
        quantity=0.01,
        order_type="market",
        price=None,
        current_price=50000.0,
        portfolio_value=10000.0,
        current_position_size=2500.0,  # Already have $2500 position
        open_positions=3,
        available_balance=5000.0,
    )
    print(f"Result: {response.result.value}")
    print(f"Allowed: {response.allowed}")
    print(f"Messages: {response.messages}")
    print(f"Warnings: {response.warnings}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
