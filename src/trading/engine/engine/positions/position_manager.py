from datetime import datetime
from functools import lru_cache
from supabase import create_client, Client
from typing import Dict, List, Optional
import logging
import uuid


"""
Position Manager
Tracks open positions, calculates P&L, and manages position lifecycle
"""



logger = logging.getLogger(__name__)


class PositionManager:
    """
    Position Manager
    Manages trading positions and P&L calculations
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Position Manager

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        self.db: Client = create_client(supabase_url, supabase_key)

        # Local position cache
        self.positions = {}  # position_id -> position_dict

        logger.info("Initialized PositionManager")

    async def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        order_id: str,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Open a new position

        Args:
            symbol: Trading pair
            side: "long" or "short"
            quantity: Position size
            entry_price: Entry price
            order_id: Associated order ID
            strategy_id: Strategy that opened this position
            metadata: Additional metadata

        Returns:
            Position dictionary
        """
        try:
            # Check if we already have an open position for this symbol
            existing = await self.get_position_by_symbol(symbol)

            if existing and existing["status"] == "open":
                # Update existing position
                return await self._update_existing_position(
                    existing, quantity, entry_price, side, order_id
                )

            # Create new position
            position_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()

            position = {
                "position_id": position_id,
                "symbol": symbol,
                "side": side.lower(),
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": entry_price,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_pnl": 0.0,
                "status": "open",
                "strategy_id": strategy_id,
                "open_order_id": order_id,
                "close_order_id": None,
                "metadata": metadata or {},
                "opened_at": timestamp.isoformat(),
                "closed_at": None,
                "updated_at": timestamp.isoformat(),
            }

            # Insert into database
            result = self.db.table("positions").insert(position).execute()
            if not result.data:
                raise Exception("Failed to insert position into database")

            # Cache locally
            self.positions[position_id] = position

            logger.info(
                f"Opened position {position_id}: {side} {quantity} {symbol} @ {entry_price}"
            )

            # Log event
            await self._log_event(
                event_type="position_opened",
                severity="info",
                message=f"Position {position_id} opened",
                metadata={
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                },
            )

            return position

        except Exception as e:
            logger.error(f"Failed to open position: {e}", exc_info=True)
            await self._log_event(
                event_type="position_error",
                severity="error",
                message=f"Failed to open position: {str(e)}",
                metadata={"error": str(e)},
            )
            raise

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        order_id: str,
        quantity: Optional[float] = None,
    ) -> Dict:
        """
        Close an existing position (full or partial)

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            order_id: Associated order ID
            quantity: Quantity to close (None = full close)

        Returns:
            Updated position dictionary
        """
        try:
            # Get position
            position = await self.get_position(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            if position["status"] != "open":
                raise ValueError(f"Position {position_id} is not open")

            # Determine close quantity
            close_qty = quantity if quantity is not None else position["quantity"]

            if close_qty > position["quantity"]:
                raise ValueError(
                    f"Close quantity {close_qty} exceeds position quantity {position['quantity']}"
                )

            # Calculate realized P&L
            realized_pnl = self._calculate_realized_pnl(
                position["side"],
                position["entry_price"],
                exit_price,
                close_qty,
            )

            position["realized_pnl"] += realized_pnl
            position["quantity"] -= close_qty

            # If fully closed
            if position["quantity"] <= 0.0001:  # Handle floating point precision
                position["status"] = "closed"
                position["quantity"] = 0.0
                position["closed_at"] = datetime.utcnow().isoformat()
                position["close_order_id"] = order_id

            position["total_pnl"] = position["realized_pnl"] + position["unrealized_pnl"]
            position["updated_at"] = datetime.utcnow().isoformat()

            # Update in database
            await self._update_position_in_db(position)

            logger.info(
                f"Closed {'full' if position['status'] == 'closed' else 'partial'} position "
                f"{position_id}: {close_qty} @ {exit_price}, P&L: {realized_pnl:.2f}"
            )

            await self._log_event(
                event_type="position_closed" if position["status"] == "closed" else "position_reduced",
                severity="info",
                message=f"Position {position_id} {'closed' if position['status'] == 'closed' else 'reduced'}",
                metadata={
                    "position_id": position_id,
                    "close_quantity": close_qty,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "total_pnl": position["total_pnl"],
                },
            )

            return position

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}", exc_info=True)
            await self._log_event(
                event_type="position_error",
                severity="error",
                message=f"Failed to close position {position_id}: {str(e)}",
                metadata={"position_id": position_id, "error": str(e)},
            )
            raise

    async def update_position_price(self, position_id: str, current_price: float) -> Dict:
        """
        Update position with current market price and recalculate unrealized P&L

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Updated position dictionary
        """
        try:
            position = await self.get_position(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            if position["status"] != "open":
                return position  # Don't update closed positions

            # Update current price
            position["current_price"] = current_price

            # Recalculate unrealized P&L
            position["unrealized_pnl"] = self._calculate_unrealized_pnl(
                position["side"],
                position["entry_price"],
                current_price,
                position["quantity"],
            )

            position["total_pnl"] = position["realized_pnl"] + position["unrealized_pnl"]
            position["updated_at"] = datetime.utcnow().isoformat()

            # Update in database
            await self._update_position_in_db(position)

            # Update cache
            self.positions[position_id] = position

            return position

        except Exception as e:
            logger.error(f"Failed to update position price: {e}", exc_info=True)
            raise

    async def calculate_pnl(self, position_id: str, current_price: Optional[float] = None) -> Dict:
        """
        Calculate P&L for a position

        Args:
            position_id: Position ID
            current_price: Optional current price (if not provided, uses cached price)

        Returns:
            Dict with realized_pnl, unrealized_pnl, total_pnl
        """
        try:
            position = await self.get_position(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            price = current_price if current_price is not None else position["current_price"]

            unrealized_pnl = 0.0
            if position["status"] == "open":
                unrealized_pnl = self._calculate_unrealized_pnl(
                    position["side"],
                    position["entry_price"],
                    price,
                    position["quantity"],
                )

            return {
                "position_id": position_id,
                "realized_pnl": position["realized_pnl"],
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": position["realized_pnl"] + unrealized_pnl,
                "entry_price": position["entry_price"],
                "current_price": price,
                "quantity": position["quantity"],
            }

        except Exception as e:
            logger.error(f"Failed to calculate P&L: {e}", exc_info=True)
            raise

    async def get_position(self, position_id: str) -> Optional[Dict]:
        """Get position by ID"""
        try:
            # Check cache first
            if position_id in self.positions:
                return self.positions[position_id]

            # Query database
            result = (
                self.db.table("positions")
                .select("*")
                .eq("position_id", position_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                position = result.data[0]
                self.positions[position_id] = position
                return position

            return None

        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}", exc_info=True)
            return None

    async def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get open position by symbol"""
        try:
            result = (
                self.db.table("positions")
                .select("*")
                .eq("symbol", symbol)
                .eq("status", "open")
                .order("opened_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data and len(result.data) > 0:
                return result.data[0]

            return None

        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}", exc_info=True)
            return None

    async def get_all_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            result = (
                self.db.table("positions")
                .select("*")
                .eq("status", "open")
                .order("opened_at", desc=True)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}", exc_info=True)
            return []

    async def get_position_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical positions"""
        try:
            query = self.db.table("positions").select("*")

            if symbol:
                query = query.eq("symbol", symbol)

            if start_time:
                query = query.gte("opened_at", start_time.isoformat())

            if end_time:
                query = query.lte("opened_at", end_time.isoformat())

            result = query.order("opened_at", desc=True).limit(limit).execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get position history: {e}", exc_info=True)
            return []

    def _calculate_unrealized_pnl(
        self, side: str, entry_price: float, current_price: float, quantity: float
    ) -> float:
        """Calculate unrealized P&L"""
        if side == "long":
            return (current_price - entry_price) * quantity
        else:  # short
            return (entry_price - current_price) * quantity

    def _calculate_realized_pnl(
        self, side: str, entry_price: float, exit_price: float, quantity: float
    ) -> float:
        """Calculate realized P&L"""
        if side == "long":
            return (exit_price - entry_price) * quantity
        else:  # short
            return (entry_price - exit_price) * quantity

    async def _update_existing_position(
        self,
        position: Dict,
        new_quantity: float,
        new_price: float,
        side: str,
        order_id: str,
    ) -> Dict:
        """Update existing position when adding to it"""
        # Calculate new average entry price
        total_cost = (
            position["entry_price"] * position["quantity"] + new_price * new_quantity
        )
        position["quantity"] += new_quantity
        position["entry_price"] = total_cost / position["quantity"]
        position["updated_at"] = datetime.utcnow().isoformat()

        # Update in database
        await self._update_position_in_db(position)

        logger.info(
            f"Updated position {position['position_id']}: "
            f"+{new_quantity} @ {new_price}, new avg: {position['entry_price']:.2f}"
        )

        return position

    async def _update_position_in_db(self, position: Dict):
        """Update position in database"""
        try:
            position["updated_at"] = datetime.utcnow().isoformat()
            result = (
                self.db.table("positions")
                .update(position)
                .eq("position_id", position["position_id"])
                .execute()
            )
            if not result.data:
                logger.warning(
                    f"Failed to update position {position['position_id']} in database"
                )
            else:
                # Update cache
                self.positions[position["position_id"]] = position

        except Exception as e:
            logger.error(f"Failed to update position in database: {e}", exc_info=True)

    async def _log_event(
        self, event_type: str, severity: str, message: str, metadata: Optional[Dict] = None
    ):
        """Log event to system_events table"""
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "component": "position_manager",
                "severity": severity,
                "message": message,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }
            self.db.table("system_events").insert(event).execute()
        except Exception as e:
            logger.error(f"Failed to log event: {e}", exc_info=True)
