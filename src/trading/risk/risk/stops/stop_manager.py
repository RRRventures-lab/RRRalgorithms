from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import os

"""
Stop-Loss and Take-Profit Manager

Auto-generates and manages stop-loss and take-profit orders for all positions.
Features:
- Fixed stop-loss and take-profit levels
- Trailing stops
- Risk/reward ratio calculation
- ATR-based stops
- Integration with Supabase orders table
"""


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class StopOrder:
    """Stop or take-profit order"""
    symbol: str
    order_type: str  # "stop_loss" or "take_profit"
    stop_price: float
    quantity: float
    side: str  # "sell" for long positions, "buy" for short positions
    trailing: bool
    trailing_amount: Optional[float]
    position_id: Optional[str]
    risk_reward_ratio: Optional[float]
    notes: str


class StopManager:
    """
    Manage stop-loss and take-profit orders

    Automatically generates stops for all positions and manages
    trailing stops based on price movement.
    """

    def __init__(
        self,
        default_stop_loss_pct: float = 0.02,  # 2% stop loss
        default_take_profit_pct: float = 0.06,  # 6% take profit (3:1 R/R)
        enable_trailing_stops: bool = True,
        trailing_activation_pct: float = 0.03,  # Activate trailing at 3% profit
        trailing_distance_pct: float = 0.015,  # Trail by 1.5%
        use_atr_stops: bool = False,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize stop manager

        Args:
            default_stop_loss_pct: Default stop loss as % of entry price
            default_take_profit_pct: Default take profit as % of entry price
            enable_trailing_stops: Enable trailing stops
            trailing_activation_pct: % profit at which to activate trailing stop
            trailing_distance_pct: Distance to trail behind price
            use_atr_stops: Use ATR-based stops instead of fixed %
            atr_multiplier: ATR multiplier for stop distance
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.enable_trailing_stops = enable_trailing_stops
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier

        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

        self.supabase: Client = create_client(supabase_url, supabase_key)

        logger.info(
            f"Stop Manager initialized: SL={default_stop_loss_pct:.1%}, "
            f"TP={default_take_profit_pct:.1%}, trailing={enable_trailing_stops}"
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price of position
            side: "buy" for long, "sell" for short
            atr: Average True Range (optional, for ATR-based stops)

        Returns:
            Stop loss price
        """
        if self.use_atr_stops and atr is not None:
            # ATR-based stop
            stop_distance = atr * self.atr_multiplier
        else:
            # Percentage-based stop
            stop_distance = entry_price * self.default_stop_loss_pct

        if side.lower() == "buy":
            # Long position: stop below entry
            stop_price = entry_price - stop_distance
        else:
            # Short position: stop above entry
            stop_price = entry_price + stop_distance

        return stop_price

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        risk_reward_ratio: float = 3.0
    ) -> float:
        """
        Calculate take profit price

        Args:
            entry_price: Entry price of position
            side: "buy" for long, "sell" for short
            risk_reward_ratio: Target risk/reward ratio

        Returns:
            Take profit price
        """
        # Calculate profit distance based on risk/reward ratio
        risk_distance = entry_price * self.default_stop_loss_pct
        profit_distance = risk_distance * risk_reward_ratio

        if side.lower() == "buy":
            # Long position: take profit above entry
            tp_price = entry_price + profit_distance
        else:
            # Short position: take profit below entry
            tp_price = entry_price - profit_distance

        return tp_price

    def create_stop_orders(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        position_id: Optional[str] = None,
        atr: Optional[float] = None,
        custom_stop_loss_pct: Optional[float] = None,
        custom_take_profit_pct: Optional[float] = None
    ) -> Tuple[StopOrder, StopOrder]:
        """
        Create stop-loss and take-profit orders for a position

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            side: Position side ("buy" or "sell")
            position_id: Position ID in database
            atr: Average True Range (optional)
            custom_stop_loss_pct: Custom stop loss % (overrides default)
            custom_take_profit_pct: Custom take profit % (overrides default)

        Returns:
            Tuple of (stop_loss_order, take_profit_order)
        """
        # Calculate stop loss
        stop_loss_price = self.calculate_stop_loss(entry_price, side, atr)

        # Calculate take profit
        risk_reward_ratio = (
            custom_take_profit_pct / custom_stop_loss_pct
            if custom_stop_loss_pct and custom_take_profit_pct
            else self.default_take_profit_pct / self.default_stop_loss_pct
        )
        take_profit_price = self.calculate_take_profit(entry_price, side, risk_reward_ratio)

        # Create stop loss order
        stop_loss_order = StopOrder(
            symbol=symbol,
            order_type="stop_loss",
            stop_price=stop_loss_price,
            quantity=quantity,
            side="sell" if side.lower() == "buy" else "buy",
            trailing=False,  # Start with fixed stop
            trailing_amount=None,
            position_id=position_id,
            risk_reward_ratio=risk_reward_ratio,
            notes=f"Stop loss at {stop_loss_price:.2f} ({self.default_stop_loss_pct:.1%} from entry)"
        )

        # Create take profit order
        take_profit_order = StopOrder(
            symbol=symbol,
            order_type="take_profit",
            stop_price=take_profit_price,
            quantity=quantity,
            side="sell" if side.lower() == "buy" else "buy",
            trailing=False,
            trailing_amount=None,
            position_id=position_id,
            risk_reward_ratio=risk_reward_ratio,
            notes=f"Take profit at {take_profit_price:.2f} (R/R={risk_reward_ratio:.1f}:1)"
        )

        return stop_loss_order, take_profit_order

    def update_trailing_stop(
        self,
        stop_order: StopOrder,
        current_price: float,
        entry_price: float,
        side: str
    ) -> StopOrder:
        """
        Update trailing stop based on current price

        Args:
            stop_order: Current stop order
            current_price: Current market price
            entry_price: Original entry price
            side: Position side ("buy" or "sell")

        Returns:
            Updated stop order
        """
        if not self.enable_trailing_stops:
            return stop_order

        # Calculate profit %
        if side.lower() == "buy":
            profit_pct = (current_price - entry_price) / entry_price
            trailing_stop_price = current_price * (1 - self.trailing_distance_pct)
        else:
            profit_pct = (entry_price - current_price) / entry_price
            trailing_stop_price = current_price * (1 + self.trailing_distance_pct)

        # Activate trailing stop if profit threshold reached
        if profit_pct >= self.trailing_activation_pct:
            if not stop_order.trailing:
                logger.info(
                    f"Activating trailing stop for {stop_order.symbol} at {current_price:.2f} "
                    f"(profit: {profit_pct:.1%})"
                )
                stop_order.trailing = True
                stop_order.trailing_amount = self.trailing_distance_pct

            # Update stop price if trailing and price moved favorably
            if side.lower() == "buy":
                # Long: only raise stop, never lower
                if trailing_stop_price > stop_order.stop_price:
                    old_stop = stop_order.stop_price
                    stop_order.stop_price = trailing_stop_price
                    logger.debug(
                        f"Trailing stop updated for {stop_order.symbol}: "
                        f"{old_stop:.2f} -> {trailing_stop_price:.2f}"
                    )
            else:
                # Short: only lower stop, never raise
                if trailing_stop_price < stop_order.stop_price:
                    old_stop = stop_order.stop_price
                    stop_order.stop_price = trailing_stop_price
                    logger.debug(
                        f"Trailing stop updated for {stop_order.symbol}: "
                        f"{old_stop:.2f} -> {trailing_stop_price:.2f}"
                    )

        return stop_order

    def submit_stop_order_to_db(self, stop_order: StopOrder) -> bool:
        """
        Submit stop order to Supabase orders table

        Args:
            stop_order: Stop order to submit

        Returns:
            True if successful, False otherwise
        """
        try:
            order_data = {
                "symbol": stop_order.symbol,
                "order_type": stop_order.order_type,
                "side": stop_order.side,
                "quantity": float(stop_order.quantity),
                "price": float(stop_order.stop_price),
                "status": "pending",
                "stop_price": float(stop_order.stop_price),
                "trailing": stop_order.trailing,
                "trailing_amount": float(stop_order.trailing_amount) if stop_order.trailing_amount else None,
                "position_id": stop_order.position_id,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "risk_reward_ratio": stop_order.risk_reward_ratio,
                    "notes": stop_order.notes
                }
            }

            response = self.supabase.table("orders").insert(order_data).execute()

            logger.info(
                f"Stop order submitted: {stop_order.symbol} {stop_order.order_type} "
                f"@ {stop_order.stop_price:.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error submitting stop order: {e}")
            return False

    def generate_stops_for_all_positions(self) -> List[Tuple[StopOrder, StopOrder]]:
        """
        Generate stop orders for all open positions

        Queries positions table and creates stop/TP orders for any
        positions that don't have them.

        Returns:
            List of (stop_loss_order, take_profit_order) tuples
        """
        try:
            # Query open positions
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if not positions:
                logger.info("No open positions found")
                return []

            stop_orders = []

            for position in positions:
                symbol = position["symbol"]
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                side = position["side"]
                position_id = position.get("id")

                logger.info(f"Generating stops for {symbol} position ({quantity} @ {entry_price})")

                # Create stop orders
                stop_loss, take_profit = self.create_stop_orders(
                    symbol=symbol,
                    entry_price=entry_price,
                    quantity=quantity,
                    side=side,
                    position_id=position_id
                )

                # Submit to database
                self.submit_stop_order_to_db(stop_loss)
                self.submit_stop_order_to_db(take_profit)

                stop_orders.append((stop_loss, take_profit))

            logger.info(f"Generated {len(stop_orders)} stop order pairs")
            return stop_orders

        except Exception as e:
            logger.error(f"Error generating stops for positions: {e}")
            return []

    def update_all_trailing_stops(self) -> int:
        """
        Update all trailing stops based on current prices

        Returns:
            Number of stops updated
        """
        try:
            # Query active stop orders with trailing enabled
            response = (
                self.supabase.table("orders")
                .select("*")
                .eq("order_type", "stop_loss")
                .eq("trailing", True)
                .eq("status", "pending")
                .execute()
            )

            orders = response.data
            updated_count = 0

            for order_data in orders:
                symbol = order_data["symbol"]
                position_id = order_data.get("position_id")

                # Get position details
                if position_id:
                    pos_response = (
                        self.supabase.table("positions")
                        .select("*")
                        .eq("id", position_id)
                        .execute()
                    )

                    if pos_response.data:
                        position = pos_response.data[0]
                        entry_price = position["entry_price"]
                        current_price = position["current_price"]
                        side = position["side"]

                        # Reconstruct stop order
                        stop_order = StopOrder(
                            symbol=symbol,
                            order_type=order_data["order_type"],
                            stop_price=order_data["stop_price"],
                            quantity=order_data["quantity"],
                            side=order_data["side"],
                            trailing=order_data["trailing"],
                            trailing_amount=order_data.get("trailing_amount"),
                            position_id=position_id,
                            risk_reward_ratio=None,
                            notes=""
                        )

                        # Update trailing stop
                        updated_stop = self.update_trailing_stop(
                            stop_order,
                            current_price,
                            entry_price,
                            side
                        )

                        # Update in database if changed
                        if updated_stop.stop_price != stop_order.stop_price:
                            self.supabase.table("orders").update({
                                "stop_price": float(updated_stop.stop_price),
                                "price": float(updated_stop.stop_price),
                                "updated_at": datetime.now().isoformat()
                            }).eq("id", order_data["id"]).execute()

                            updated_count += 1

            logger.info(f"Updated {updated_count} trailing stops")
            return updated_count

        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")
            return 0


def main():
    """Example usage"""
    print("=" * 60)
    print("Stop-Loss and Take-Profit Manager")
    print("=" * 60)

    try:
        # Initialize manager
        manager = StopManager(
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.06,
            enable_trailing_stops=True
        )

        # Example 1: Create stops for a new position
        print("\nExample 1: Creating stops for new position")
        print("-" * 60)

        stop_loss, take_profit = manager.create_stop_orders(
            symbol="BTC-USD",
            entry_price=50000.0,
            quantity=0.1,
            side="buy"
        )

        print(f"Position: BTC-USD")
        print(f"Entry: $50,000")
        print(f"Quantity: 0.1 BTC")
        print(f"\nStop Loss Order:")
        print(f"  Price: ${stop_loss.stop_price:,.2f}")
        print(f"  Notes: {stop_loss.notes}")
        print(f"\nTake Profit Order:")
        print(f"  Price: ${take_profit.stop_price:,.2f}")
        print(f"  Notes: {take_profit.notes}")
        print(f"  Risk/Reward: {take_profit.risk_reward_ratio:.1f}:1")

        # Example 2: Generate stops for all positions
        print("\n" + "=" * 60)
        print("Example 2: Generating stops for all positions")
        print("-" * 60)

        stop_orders = manager.generate_stops_for_all_positions()
        print(f"\nGenerated {len(stop_orders)} stop order pairs")

        # Example 3: Update trailing stops
        print("\n" + "=" * 60)
        print("Example 3: Updating trailing stops")
        print("-" * 60)

        updated = manager.update_all_trailing_stops()
        print(f"\nUpdated {updated} trailing stops")

    except Exception as e:
        logger.error(f"Error in stop manager demo: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
