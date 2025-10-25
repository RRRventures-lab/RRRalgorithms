"""
Transparency Dashboard Reporter

Sends real-time trading data to the transparency dashboard:
- Live trade execution
- Order status updates
- Position changes
- P&L updates
- Risk metrics

Integrates with the monitoring dashboard for real-time visibility.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class TradeEvent:
    """Trade event for dashboard"""
    event_id: str
    timestamp: str
    event_type: str  # "order_placed", "order_filled", "position_opened", etc.
    symbol: str
    side: str
    quantity: float
    price: float
    value_usd: float
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    pnl: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class RiskMetricsUpdate:
    """Risk metrics update for dashboard"""
    timestamp: str
    portfolio_value: float
    daily_pnl: float
    daily_return_pct: float
    total_pnl: float
    total_return_pct: float
    open_positions: int
    cash_balance: float
    equity_value: float
    circuit_breaker_state: str
    violations: List[str]


class DashboardReporter:
    """
    Reports trading activity to transparency dashboard

    Sends real-time updates to dashboard database tables for visualization.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize dashboard reporter

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        from src.database import get_db

        self.db = get_db()
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

        logger.info("Dashboard reporter initialized")

    async def report_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        metadata: Optional[Dict] = None,
    ):
        """
        Report order placement to dashboard

        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type
            price: Limit/stop price
            metadata: Additional metadata
        """
        try:
            event = {
                "event_id": f"order_placed_{order_id}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "order_placed",
                "component": "trading_engine",
                "severity": "info",
                "message": f"Order placed: {side} {quantity} {symbol}",
                "metadata": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price,
                    **(metadata or {}),
                },
            }

            self.db.table("trading_events").insert(event).execute()

            logger.info(f"Reported order placement to dashboard: {order_id}")

        except Exception as e:
            logger.error(f"Failed to report order to dashboard: {e}", exc_info=True)

    async def report_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Report order fill to dashboard

        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: Buy or sell
            quantity: Filled quantity
            fill_price: Fill price
            metadata: Additional metadata
        """
        try:
            event = {
                "event_id": f"order_filled_{order_id}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "order_filled",
                "component": "trading_engine",
                "severity": "info",
                "message": f"Order filled: {side} {quantity} {symbol} @ ${fill_price:,.2f}",
                "metadata": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "fill_price": fill_price,
                    "value_usd": quantity * fill_price,
                    **(metadata or {}),
                },
            }

            self.db.table("trading_events").insert(event).execute()

            logger.info(f"Reported order fill to dashboard: {order_id}")

        except Exception as e:
            logger.error(f"Failed to report order fill to dashboard: {e}", exc_info=True)

    async def report_position_opened(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Report position opening to dashboard

        Args:
            position_id: Position ID
            symbol: Trading symbol
            side: Long or short
            quantity: Position size
            entry_price: Entry price
            metadata: Additional metadata
        """
        try:
            event = {
                "event_id": f"position_opened_{position_id}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "position_opened",
                "component": "trading_engine",
                "severity": "info",
                "message": f"Position opened: {side} {quantity} {symbol} @ ${entry_price:,.2f}",
                "metadata": {
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "value_usd": quantity * entry_price,
                    **(metadata or {}),
                },
            }

            self.db.table("trading_events").insert(event).execute()

            logger.info(f"Reported position opening to dashboard: {position_id}")

        except Exception as e:
            logger.error(f"Failed to report position to dashboard: {e}", exc_info=True)

    async def report_position_closed(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Report position closing to dashboard

        Args:
            position_id: Position ID
            symbol: Trading symbol
            side: Long or short
            quantity: Position size
            entry_price: Entry price
            exit_price: Exit price
            pnl: Realized P&L
            metadata: Additional metadata
        """
        try:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

            event = {
                "event_id": f"position_closed_{position_id}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "position_closed",
                "component": "trading_engine",
                "severity": "info",
                "message": f"Position closed: {side} {quantity} {symbol} - P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)",
                "metadata": {
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    **(metadata or {}),
                },
            }

            self.db.table("trading_events").insert(event).execute()

            logger.info(f"Reported position closing to dashboard: {position_id} (P&L: ${pnl:,.2f})")

        except Exception as e:
            logger.error(f"Failed to report position closing to dashboard: {e}", exc_info=True)

    async def report_risk_metrics(self, metrics: RiskMetricsUpdate):
        """
        Report risk metrics to dashboard

        Args:
            metrics: Risk metrics update
        """
        try:
            event = {
                "event_id": f"risk_metrics_{datetime.utcnow().isoformat()}",
                "timestamp": metrics.timestamp,
                "event_type": "risk_metrics_update",
                "component": "trading_engine",
                "severity": "info" if not metrics.violations else "warning",
                "message": f"Risk metrics update - Portfolio: ${metrics.portfolio_value:,.2f}",
                "metadata": asdict(metrics),
            }

            self.db.table("trading_events").insert(event).execute()

            logger.debug("Reported risk metrics to dashboard")

        except Exception as e:
            logger.error(f"Failed to report risk metrics to dashboard: {e}", exc_info=True)

    async def report_circuit_breaker_opened(self, violations: List[str], metadata: Optional[Dict] = None):
        """
        Report circuit breaker opening to dashboard

        Args:
            violations: List of violations that triggered opening
            metadata: Additional metadata
        """
        try:
            event = {
                "event_id": f"circuit_breaker_opened_{datetime.utcnow().isoformat()}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "circuit_breaker_opened",
                "component": "trading_engine",
                "severity": "critical",
                "message": f"CIRCUIT BREAKER OPENED - Trading halted due to: {', '.join(violations)}",
                "metadata": {
                    "violations": violations,
                    "requires_review": True,
                    **(metadata or {}),
                },
            }

            self.db.table("trading_events").insert(event).execute()

            logger.critical(f"Reported circuit breaker opening to dashboard")

        except Exception as e:
            logger.error(f"Failed to report circuit breaker to dashboard: {e}", exc_info=True)

    async def report_circuit_breaker_closed(self):
        """Report circuit breaker closing to dashboard"""
        try:
            event = {
                "event_id": f"circuit_breaker_closed_{datetime.utcnow().isoformat()}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "circuit_breaker_closed",
                "component": "trading_engine",
                "severity": "info",
                "message": "Circuit breaker closed - resuming normal trading",
                "metadata": {},
            }

            self.db.table("trading_events").insert(event).execute()

            logger.info("Reported circuit breaker closing to dashboard")

        except Exception as e:
            logger.error(f"Failed to report circuit breaker closing to dashboard: {e}", exc_info=True)

    async def report_error(self, error_type: str, message: str, metadata: Optional[Dict] = None):
        """
        Report error to dashboard

        Args:
            error_type: Type of error
            message: Error message
            metadata: Additional metadata
        """
        try:
            event = {
                "event_id": f"error_{datetime.utcnow().isoformat()}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": error_type,
                "component": "trading_engine",
                "severity": "error",
                "message": message,
                "metadata": metadata or {},
            }

            self.db.table("trading_events").insert(event).execute()

            logger.error(f"Reported error to dashboard: {message}")

        except Exception as e:
            logger.error(f"Failed to report error to dashboard: {e}", exc_info=True)


async def main():
    """Example usage"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("Dashboard Reporter - Test")
    print("=" * 70)

    supabase_url = os.getenv("DATABASE_PATH")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Database credentials not configured")
        return

    reporter = DashboardReporter(supabase_url, supabase_key)

    # Test: Report order placement
    print("\n1. Reporting order placement...")
    await reporter.report_order_placed(
        order_id="test-order-123",
        symbol="BTC-USD",
        side="buy",
        quantity=0.001,
        order_type="market",
        price=None
    )
    print("   ✓ Order reported")

    # Test: Report order fill
    print("\n2. Reporting order fill...")
    await reporter.report_order_filled(
        order_id="test-order-123",
        symbol="BTC-USD",
        side="buy",
        quantity=0.001,
        fill_price=50000.0
    )
    print("   ✓ Order fill reported")

    # Test: Report position opened
    print("\n3. Reporting position opened...")
    await reporter.report_position_opened(
        position_id="test-pos-456",
        symbol="BTC-USD",
        side="long",
        quantity=0.001,
        entry_price=50000.0
    )
    print("   ✓ Position opened reported")

    # Test: Report position closed
    print("\n4. Reporting position closed...")
    await reporter.report_position_closed(
        position_id="test-pos-456",
        symbol="BTC-USD",
        side="long",
        quantity=0.001,
        entry_price=50000.0,
        exit_price=51000.0,
        pnl=1.0
    )
    print("   ✓ Position closed reported")

    print("\n" + "=" * 70)
    print("Dashboard reporting test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
