from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

"""
Audit Logging Integration for Trading Engine
Demonstrates how to integrate audit logging into trading operations
"""

# Add monitoring path to import audit logger
monitoring_path = Path(__file__).parent.parent.parent / "monitoring" / "logging"
if monitoring_path.exists():
    sys.path.insert(0, str(monitoring_path))

try:
    from audit_logger import get_audit_logger, AuditAction, AuditSeverity
except ImportError:
    # Fallback if audit_logger is not available
    import logging
    logger = logging.getLogger(__name__)

    class FallbackAuditLogger:
        """Fallback audit logger when real one is not available"""
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger("audit")

        def log(self, **kwargs):
            self.logger.info(f"AUDIT: {kwargs}")

        def log_order(self, **kwargs):
            self.logger.info(f"ORDER AUDIT: {kwargs}")

        def log_position(self, **kwargs):
            self.logger.info(f"POSITION AUDIT: {kwargs}")

        def log_risk_event(self, **kwargs):
            self.logger.warning(f"RISK AUDIT: {kwargs}")

        def log_config_change(self, **kwargs):
            self.logger.info(f"CONFIG AUDIT: {kwargs}")

        def get_statistics(self):
            return {"fallback_mode": True}

    class AuditAction:
        """Fallback AuditAction enum"""
        SYSTEM_STARTUP = "SYSTEM_STARTUP"
        SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
        ORDER_PLACED = "ORDER_PLACED"
        ORDER_REJECTED = "ORDER_REJECTED"
        ORDER_CANCELLED = "ORDER_CANCELLED"
        POSITION_OPENED = "POSITION_OPENED"
        POSITION_CLOSED = "POSITION_CLOSED"
        RISK_LIMIT_BREACHED = "RISK_LIMIT_BREACHED"
        EMERGENCY_STOP = "EMERGENCY_STOP"

    class AuditSeverity:
        """Fallback AuditSeverity enum"""
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    def get_audit_logger(*args, **kwargs):
        """Fallback audit logger factory"""
        return FallbackAuditLogger(*args, **kwargs)



class AuditedTradingEngine:
    """
    Trading Engine with integrated audit logging

    This is a template showing how to integrate audit logging
    into the actual trading engine implementation
    """

    def __init__(self, engine_id: str = "trading-engine-01"):
        """Initialize trading engine with audit logging"""
        self.engine_id = engine_id

        # Get audit logger instance with proper path resolution
        project_root = Path(__file__).parent.parent.parent.parent
        log_dir = project_root / "logs" / "audit"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.audit_logger = get_audit_logger(
            system_component="trading-engine",
            fallback_file=str(log_dir / "trading-engine.jsonl")
        )

        # Log system startup
        self.audit_logger.log(
            action_type=AuditAction.SYSTEM_STARTUP.value,
            action_category="SYSTEM",
            severity=AuditSeverity.INFO,
            action_details={"engine_id": engine_id}
        )

    # ========================================================================
    # Order Management with Audit Logging
    # ========================================================================

    def place_order(self,
                    symbol: str,
                    side: str,
                    quantity: float,
                    order_type: str,
                    price: Optional[float] = None,
                    user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order with audit logging

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            order_type: Order type ("market", "limit", "stop")
            price: Limit price (for limit orders)
            user_id: User placing the order

        Returns:
            dict: Order result
        """
        import uuid

        order_id = str(uuid.uuid4())

        order_details = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "timestamp": str(self._get_timestamp())
        }

        try:
            # Import Coinbase exchange for actual order placement
            from exchanges.coinbase_exchange import CoinbaseExchange
            from security.credentials_manager import get_credentials_manager

            # Get credentials and check trading mode
            creds_manager = get_credentials_manager()
            is_paper = creds_manager.is_paper_trading()

            # Initialize exchange
            exchange = CoinbaseExchange(paper_trading=is_paper)

            # Place order based on order type
            if order_type.lower() == "market":
                result = exchange.create_market_order(
                    product_id=symbol,
                    side=side.upper(),
                    size=quantity,
                    client_order_id=order_id
                )
            elif order_type.lower() == "limit":
                if not price:
                    raise ValueError("Limit orders require a price")
                result = exchange.create_limit_order(
                    product_id=symbol,
                    side=side.upper(),
                    size=quantity,
                    price=price,
                    client_order_id=order_id
                )
            elif order_type.lower() == "stop":
                if not price:
                    raise ValueError("Stop orders require a stop price")
                result = exchange.create_stop_loss_order(
                    product_id=symbol,
                    side=side.upper(),
                    size=quantity,
                    stop_price=price,
                    client_order_id=order_id
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            success = result.get('success', False)

            # Log the order placement
            self.audit_logger.log_order(
                action=AuditAction.ORDER_PLACED.value,
                order_id=order_id,
                order_details=order_details,
                success=success,
                user_id=user_id
            )

            return {
                "order_id": order_id,
                "status": "placed",
                "details": order_details
            }

        except Exception as e:
            # Log failed order
            self.audit_logger.log_order(
                action=AuditAction.ORDER_REJECTED.value,
                order_id=order_id,
                order_details=order_details,
                success=False,
                error=str(e),
                user_id=user_id
            )

            raise

    def cancel_order(self, order_id: str, user_id: Optional[str] = None) -> bool:
        """
        Cancel an order with audit logging

        Args:
            order_id: Order ID to cancel
            user_id: User cancelling the order

        Returns:
            bool: True if cancelled successfully
        """
        try:
            # Import Coinbase exchange for actual order cancellation
            from exchanges.coinbase_exchange import CoinbaseExchange
            from security.credentials_manager import get_credentials_manager

            # Get credentials and check trading mode
            creds_manager = get_credentials_manager()
            is_paper = creds_manager.is_paper_trading()

            # Initialize exchange
            exchange = CoinbaseExchange(paper_trading=is_paper)

            # Cancel the order
            success = exchange.cancel_order(order_id)

            # Log the cancellation
            self.audit_logger.log_order(
                action=AuditAction.ORDER_CANCELLED.value,
                order_id=order_id,
                order_details={"action": "cancel"},
                success=success,
                user_id=user_id
            )

            return success

        except Exception as e:
            self.audit_logger.log_order(
                action=AuditAction.ORDER_CANCELLED.value,
                order_id=order_id,
                order_details={"action": "cancel"},
                success=False,
                error=str(e),
                user_id=user_id
            )
            raise

    # ========================================================================
    # Position Management with Audit Logging
    # ========================================================================

    def open_position(self,
                      symbol: str,
                      side: str,
                      size: float,
                      entry_price: float,
                      user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Open a position with audit logging

        Args:
            symbol: Trading symbol
            side: Position side ("long" or "short")
            size: Position size
            entry_price: Entry price
            user_id: User opening position

        Returns:
            dict: Position details
        """
        import uuid

        position_id = str(uuid.uuid4())

        position_details = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "timestamp": str(self._get_timestamp())
        }

        try:
            # Import position manager for actual position management
            import asyncio
            from positions.position_manager import PositionManager
            from security.credentials_manager import get_credentials_manager

            # Get credentials
            creds_manager = get_credentials_manager()
            supabase_creds = creds_manager.get_supabase_credentials()

            # Initialize position manager
            position_mgr = PositionManager(
                supabase_url=supabase_creds['url'],
                supabase_key=supabase_creds['key']
            )

            # Open position in database (need to create a dummy order ID)
            import uuid
            order_id = str(uuid.uuid4())

            # Run async operation
            position = asyncio.run(position_mgr.open_position(
                symbol=symbol,
                side=side,
                quantity=size,
                entry_price=entry_price,
                order_id=order_id,
                strategy_id=None,
                metadata={"audit_position_id": position_id}
            ))

            success = position is not None

            # Log the position opening
            self.audit_logger.log_position(
                action=AuditAction.POSITION_OPENED.value,
                position_id=position_id,
                position_details=position_details,
                success=success,
                user_id=user_id
            )

            return {
                "position_id": position_id,
                "status": "open",
                "details": position_details
            }

        except Exception as e:
            self.audit_logger.log_position(
                action=AuditAction.POSITION_OPENED.value,
                position_id=position_id,
                position_details=position_details,
                success=False,
                user_id=user_id
            )
            raise

    def close_position(self,
                       position_id: str,
                       exit_price: float,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close a position with audit logging

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            user_id: User closing position

        Returns:
            dict: Closing details including P&L
        """
        # Get position details from database
        import asyncio
        from positions.position_manager import PositionManager
        from security.credentials_manager import get_credentials_manager

        # Get credentials
        creds_manager = get_credentials_manager()
        supabase_creds = creds_manager.get_supabase_credentials()

        # Initialize position manager
        position_mgr = PositionManager(
            supabase_url=supabase_creds['url'],
            supabase_key=supabase_creds['key']
        )

        # Get existing position
        position = asyncio.run(position_mgr.get_position(position_id))

        position_details = {
            "position_id": position_id,
            "exit_price": exit_price,
            "timestamp": str(self._get_timestamp())
        }

        try:
            # Calculate P&L and close position
            if position:
                # Create a dummy order ID for closing
                import uuid
                close_order_id = str(uuid.uuid4())

                # Close the position
                closed_position = asyncio.run(position_mgr.close_position(
                    position_id=position_id,
                    exit_price=exit_price,
                    order_id=close_order_id
                ))

                pnl = closed_position.get("realized_pnl", 0.0)
            else:
                # Position not found, use placeholder
                pnl = 0.0

            position_details["pnl"] = pnl

            # Log the position closing
            self.audit_logger.log_position(
                action=AuditAction.POSITION_CLOSED.value,
                position_id=position_id,
                position_details=position_details,
                success=True,
                user_id=user_id
            )

            return position_details

        except Exception as e:
            self.audit_logger.log_position(
                action=AuditAction.POSITION_CLOSED.value,
                position_id=position_id,
                position_details=position_details,
                success=False,
                user_id=user_id
            )
            raise

    # ========================================================================
    # Risk Management with Audit Logging
    # ========================================================================

    def check_risk_limits(self,
                          order_details: Dict[str, Any]) -> bool:
        """
        Check risk limits and log violations

        Args:
            order_details: Order details to check

        Returns:
            bool: True if within limits
        """
        # TODO: Actual risk checks
        position_size_ok = True
        exposure_ok = True
        daily_loss_ok = True

        if not (position_size_ok and exposure_ok and daily_loss_ok):
            # Log risk limit breach
            self.audit_logger.log_risk_event(
                action=AuditAction.RISK_LIMIT_BREACHED.value,
                risk_details={
                    "order": order_details,
                    "violations": {
                        "position_size": not position_size_ok,
                        "exposure": not exposure_ok,
                        "daily_loss": not daily_loss_ok
                    }
                },
                severity=AuditSeverity.CRITICAL,
                requires_review=True
            )
            return False

        return True

    def emergency_stop(self, reason: str, user_id: Optional[str] = None):
        """
        Emergency stop with audit logging

        Args:
            reason: Reason for emergency stop
            user_id: User triggering stop
        """
        # Log the emergency stop
        self.audit_logger.log(
            action_type=AuditAction.EMERGENCY_STOP.value,
            action_category="RISK",
            severity=AuditSeverity.CRITICAL,
            action_details={
                "reason": reason,
                "engine_id": self.engine_id
            },
            requires_review=True,
            user_id=user_id
        )

        # TODO: Actual emergency stop logic
        # - Cancel all open orders
        # - Close all positions
        # - Halt trading

    # ========================================================================
    # Configuration Changes with Audit Logging
    # ========================================================================

    def update_config(self,
                      parameter: str,
                      new_value: Any,
                      old_value: Any,
                      user_id: Optional[str] = None):
        """
        Update configuration with audit logging

        Args:
            parameter: Parameter name
            new_value: New value
            old_value: Old value
            user_id: User making change
        """
        self.audit_logger.log_config_change(
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            user_id=user_id
        )

        # TODO: Apply configuration change

    # ========================================================================
    # Helpers
    # ========================================================================

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow()

    def shutdown(self):
        """Shutdown trading engine with logging"""
        self.audit_logger.log(
            action_type=AuditAction.SYSTEM_SHUTDOWN.value,
            action_category="SYSTEM",
            severity=AuditSeverity.INFO,
            action_details={"engine_id": self.engine_id}
        )

        # Get statistics
        stats = self.audit_logger.get_statistics()
        print(f"Audit Logger Statistics: {stats}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: How to use the audited trading engine

    # Initialize engine
    engine = AuditedTradingEngine(engine_id="trading-engine-01")

    # Place an order (will be audited)
    try:
        order = engine.place_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.5,
            order_type="limit",
            price=50000.0,
            user_id="user-123"
        )
        print(f"Order placed: {order}")
    except Exception as e:
        print(f"Order failed: {e}")

    # Open a position (will be audited)
    position = engine.open_position(
        symbol="BTC-USD",
        side="long",
        size=0.5,
        entry_price=50000.0,
        user_id="user-123"
    )
    print(f"Position opened: {position}")

    # Close the position (will be audited)
    close_result = engine.close_position(
        position_id=position["position_id"],
        exit_price=51000.0,
        user_id="user-123"
    )
    print(f"Position closed: {close_result}")

    # Shutdown (will be audited)
    engine.shutdown()
