from ..exchanges import (
from datetime import datetime
from functools import lru_cache
from supabase import create_client, Client
from typing import Dict, List, Optional
import asyncio
import logging
import uuid


"""
Order Management System (OMS)
Manages order lifecycle and integrates with exchange and database
"""


    ExchangeInterface,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
)


logger = logging.getLogger(__name__)


class OrderManager:
    """
    Order Management System (OMS)
    Handles order creation, modification, cancellation, and status tracking
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        supabase_url: str,
        supabase_key: str,
    ):
        """
        Initialize OMS

        Args:
            exchange: Exchange connector instance
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        self.exchange = exchange
        self.db: Client = create_client(supabase_url, supabase_key)

        # Local order cache
        self.orders = {}  # order_id -> order_dict

        logger.info(f"Initialized OrderManager with exchange: {exchange.exchange_id}")

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "gtc",
        strategy_id: Optional[str] = None,
        signal_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a new order

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "buy" or "sell"
            order_type: "market", "limit", "stop_loss", "take_profit"
            quantity: Amount to trade
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: "gtc", "ioc", "fok", "day"
            strategy_id: Associated strategy ID
            signal_id: Associated signal ID
            metadata: Additional metadata

        Returns:
            Order dictionary
        """
        try:
            # Generate order ID
            order_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()

            # Convert string enums to enum types
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type_enum = OrderType[order_type.upper()]
            tif_enum = TimeInForce[time_in_force.upper()]

            # Create order record
            order = {
                "order_id": order_id,
                "exchange_id": self.exchange.exchange_id,
                "symbol": symbol,
                "side": side.lower(),
                "order_type": order_type.lower(),
                "quantity": quantity,
                "price": price,
                "stop_price": stop_price,
                "time_in_force": time_in_force.lower(),
                "status": OrderStatus.PENDING.value,
                "filled_quantity": 0.0,
                "remaining_quantity": quantity,
                "average_fill_price": 0.0,
                "strategy_id": strategy_id,
                "signal_id": signal_id,
                "metadata": metadata or {},
                "created_at": timestamp.isoformat(),
                "updated_at": timestamp.isoformat(),
            }

            # Store in database
            result = self.db.table("orders").insert(order).execute()
            if not result.data:
                raise Exception("Failed to insert order into database")

            logger.info(
                f"Created order {order_id}: {side} {quantity} {symbol} "
                f"@ {order_type} {price or 'market'}"
            )

            # Submit to exchange
            exchange_result = await self.exchange.create_order(
                symbol=symbol,
                side=side_enum,
                order_type=order_type_enum,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=tif_enum,
                client_order_id=order_id,
            )

            if not exchange_result.get("success"):
                # Update order status to rejected
                order["status"] = OrderStatus.REJECTED.value
                order["metadata"]["error"] = exchange_result.get("error", "Unknown error")
                await self._update_order_in_db(order)
                logger.error(f"Order {order_id} rejected by exchange: {exchange_result.get('error')}")
                return order

            # Update order with exchange details
            order["exchange_order_id"] = exchange_result.get("exchange_order_id")
            order["status"] = exchange_result["status"]
            order["updated_at"] = datetime.utcnow().isoformat()

            # Update exchange order details if filled
            exchange_order = exchange_result.get("order", {})
            if exchange_order.get("filled_quantity"):
                order["filled_quantity"] = exchange_order["filled_quantity"]
                order["remaining_quantity"] = exchange_order["remaining_quantity"]
                order["average_fill_price"] = exchange_order["average_fill_price"]

            # Update in database
            await self._update_order_in_db(order)

            # Cache locally
            self.orders[order_id] = order

            # Log to system events
            await self._log_event(
                event_type="order_created",
                severity="info",
                message=f"Order {order_id} created and submitted to {self.exchange.exchange_id}",
                metadata={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": order["status"],
                },
            )

            return order

        except Exception as e:
            logger.error(f"Failed to create order: {e}", exc_info=True)
            await self._log_event(
                event_type="order_error",
                severity="error",
                message=f"Failed to create order: {str(e)}",
                metadata={"error": str(e)},
            )
            raise

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict:
        """
        Modify an existing order

        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New price (optional)

        Returns:
            Updated order dictionary
        """
        try:
            # Get order from DB
            order = await self.get_order(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            # Check if order can be modified
            if order["status"] not in [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value]:
                raise ValueError(f"Cannot modify order in status {order['status']}")

            # Modify on exchange
            exchange_result = await self.exchange.modify_order(
                exchange_order_id=order["exchange_order_id"],
                quantity=quantity,
                price=price,
            )

            if not exchange_result.get("success"):
                raise Exception(f"Exchange modification failed: {exchange_result.get('error')}")

            # Update order
            if quantity is not None:
                order["quantity"] = quantity
                order["remaining_quantity"] = quantity - order["filled_quantity"]

            if price is not None:
                order["price"] = price

            order["updated_at"] = datetime.utcnow().isoformat()

            # Update in database
            await self._update_order_in_db(order)

            logger.info(f"Modified order {order_id}")
            await self._log_event(
                event_type="order_modified",
                severity="info",
                message=f"Order {order_id} modified",
                metadata={"order_id": order_id, "quantity": quantity, "price": price},
            )

            return order

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}", exc_info=True)
            await self._log_event(
                event_type="order_error",
                severity="error",
                message=f"Failed to modify order {order_id}: {str(e)}",
                metadata={"order_id": order_id, "error": str(e)},
            )
            raise

    async def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancelled order dictionary
        """
        try:
            # Get order from DB
            order = await self.get_order(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            # Check if order can be cancelled
            if order["status"] not in [
                OrderStatus.PENDING.value,
                OrderStatus.SUBMITTED.value,
                OrderStatus.PARTIALLY_FILLED.value,
            ]:
                raise ValueError(f"Cannot cancel order in status {order['status']}")

            # Cancel on exchange
            exchange_result = await self.exchange.cancel_order(
                exchange_order_id=order["exchange_order_id"]
            )

            if not exchange_result.get("success"):
                raise Exception(f"Exchange cancellation failed: {exchange_result.get('error')}")

            # Update order status
            order["status"] = OrderStatus.CANCELLED.value
            order["updated_at"] = datetime.utcnow().isoformat()

            # Update in database
            await self._update_order_in_db(order)

            logger.info(f"Cancelled order {order_id}")
            await self._log_event(
                event_type="order_cancelled",
                severity="info",
                message=f"Order {order_id} cancelled",
                metadata={"order_id": order_id},
            )

            return order

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
            await self._log_event(
                event_type="order_error",
                severity="error",
                message=f"Failed to cancel order {order_id}: {str(e)}",
                metadata={"order_id": order_id, "error": str(e)},
            )
            raise

    async def get_order_status(self, order_id: str) -> Dict:
        """
        Get current status of an order

        Args:
            order_id: Order ID

        Returns:
            Order dictionary with current status
        """
        try:
            # Get from database
            order = await self.get_order(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            # If order is still active, sync with exchange
            if order["status"] in [
                OrderStatus.PENDING.value,
                OrderStatus.SUBMITTED.value,
                OrderStatus.PARTIALLY_FILLED.value,
            ]:
                exchange_result = await self.exchange.get_order_status(
                    exchange_order_id=order["exchange_order_id"]
                )

                if exchange_result.get("success"):
                    exchange_order = exchange_result["order"]
                    order["status"] = exchange_order["status"]
                    order["filled_quantity"] = exchange_order["filled_quantity"]
                    order["remaining_quantity"] = exchange_order["remaining_quantity"]
                    order["average_fill_price"] = exchange_order["average_fill_price"]
                    order["updated_at"] = datetime.utcnow().isoformat()

                    # Update in database
                    await self._update_order_in_db(order)

            return order

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}", exc_info=True)
            raise

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order from database

        Args:
            order_id: Order ID

        Returns:
            Order dictionary or None
        """
        try:
            result = self.db.table("orders").select("*").eq("order_id", order_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}", exc_info=True)
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        try:
            query = self.db.table("orders").select("*").in_(
                "status",
                [
                    OrderStatus.PENDING.value,
                    OrderStatus.SUBMITTED.value,
                    OrderStatus.PARTIALLY_FILLED.value,
                ],
            )

            if symbol:
                query = query.eq("symbol", symbol)

            result = query.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}", exc_info=True)
            return []

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get historical orders

        Args:
            symbol: Optional symbol filter
            start_time: Optional start time
            end_time: Optional end time
            limit: Maximum number of orders

        Returns:
            List of orders
        """
        try:
            query = self.db.table("orders").select("*")

            if symbol:
                query = query.eq("symbol", symbol)

            if start_time:
                query = query.gte("created_at", start_time.isoformat())

            if end_time:
                query = query.lte("created_at", end_time.isoformat())

            result = query.order("created_at", desc=True).limit(limit).execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get order history: {e}", exc_info=True)
            return []

    async def _update_order_in_db(self, order: Dict):
        """Update order in database"""
        try:
            order["updated_at"] = datetime.utcnow().isoformat()
            result = (
                self.db.table("orders")
                .update(order)
                .eq("order_id", order["order_id"])
                .execute()
            )
            if not result.data:
                logger.warning(f"Failed to update order {order['order_id']} in database")
        except Exception as e:
            logger.error(f"Failed to update order in database: {e}", exc_info=True)

    async def _log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None,
    ):
        """Log event to system_events table"""
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "component": "order_manager",
                "severity": severity,
                "message": message,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }
            self.db.table("system_events").insert(event).execute()
        except Exception as e:
            logger.error(f"Failed to log event: {e}", exc_info=True)
