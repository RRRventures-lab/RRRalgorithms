from .exchange_interface import (
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional
import asyncio
import logging
import random
import uuid


"""
Paper Exchange - Simulated Trading Environment
Simulates order execution without real money for testing and development
"""


    ExchangeInterface,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
)


logger = logging.getLogger(__name__)


class PaperExchange(ExchangeInterface):
    """
    Paper trading exchange simulator.
    Simulates realistic order execution with slippage and latency.
    """

    def __init__(
        self,
        exchange_id: str = "paper",
        initial_balance: float = 100000.0,
        base_currency: str = "USD",
        slippage_bps: float = 5.0,  # 5 basis points default slippage
        latency_ms: tuple = (10, 50),  # Min/max latency in milliseconds
    ):
        """
        Initialize paper exchange

        Args:
            exchange_id: Unique identifier
            initial_balance: Starting cash balance
            base_currency: Base currency (USD, USDT, etc.)
            slippage_bps: Slippage in basis points (0.01% = 1 bp)
            latency_ms: Tuple of (min, max) latency in milliseconds
        """
        super().__init__(exchange_id=exchange_id, paper_trading=True)

        # Account state
        self.balances = {base_currency: initial_balance}
        self.base_currency = base_currency

        # Order management
        self.orders = {}  # exchange_order_id -> order_dict
        self.order_history = []
        self.positions = {}  # symbol -> position_dict

        # Market simulation parameters
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms

        # Market data cache (symbol -> price)
        self.market_prices = {}
        self.order_books = {}

        logger.info(
            f"Initialized PaperExchange with {initial_balance} {base_currency}, "
            f"slippage={slippage_bps}bps, latency={latency_ms}ms"
        )

    async def connect(self) -> bool:
        """Connect to paper exchange (always succeeds)"""
        await asyncio.sleep(random.uniform(0.01, 0.05))
        self.connected = True
        logger.info(f"Connected to {self.exchange_id}")
        return True

    async def disconnect(self) -> bool:
        """Disconnect from paper exchange"""
        self.connected = False
        logger.info(f"Disconnected from {self.exchange_id}")
        return True

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Create and simulate order execution"""

        # Validate parameters
        is_valid, error_msg = self.validate_order_params(
            symbol, side, order_type, quantity, price, stop_price
        )
        if not is_valid:
            logger.error(f"Order validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "status": OrderStatus.REJECTED.value,
            }

        # Simulate network latency
        await asyncio.sleep(random.uniform(self.latency_ms[0] / 1000, self.latency_ms[1] / 1000))

        # Generate order ID
        exchange_order_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Get current market price
        market_price = await self.get_current_price(symbol)

        # Create order record
        order = {
            "exchange_order_id": exchange_order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side.value,
            "order_type": order_type.value,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "time_in_force": time_in_force.value,
            "status": OrderStatus.SUBMITTED.value,
            "filled_quantity": 0.0,
            "remaining_quantity": quantity,
            "average_fill_price": 0.0,
            "created_at": timestamp,
            "updated_at": timestamp,
            "fills": [],
        }

        # Store order
        self.orders[exchange_order_id] = order

        logger.info(
            f"Created order {exchange_order_id}: {side.value} {quantity} {symbol} "
            f"@ {order_type.value} {price or 'market'}"
        )

        # Simulate immediate execution for market orders
        if order_type == OrderType.MARKET:
            fill_result = await self._execute_market_order(order, market_price)
            if fill_result["success"]:
                order["status"] = OrderStatus.FILLED.value
                order["filled_quantity"] = quantity
                order["remaining_quantity"] = 0.0
                order["average_fill_price"] = fill_result["fill_price"]
                order["fills"].append(
                    {
                        "fill_id": str(uuid.uuid4()),
                        "price": fill_result["fill_price"],
                        "quantity": quantity,
                        "timestamp": datetime.utcnow(),
                        "fee": fill_result["fee"],
                    }
                )
                self.order_history.append(order.copy())
                del self.orders[exchange_order_id]

                # Update position
                await self._update_position(symbol, side, quantity, fill_result["fill_price"])

        return {
            "success": True,
            "exchange_order_id": exchange_order_id,
            "status": order["status"],
            "order": order,
        }

    async def _execute_market_order(self, order: Dict, market_price: float) -> Dict:
        """
        Simulate market order execution with realistic slippage

        Args:
            order: Order dictionary
            market_price: Current market price

        Returns:
            Dict with execution details
        """
        # Calculate slippage
        slippage_factor = 1 + (random.uniform(-self.slippage_bps, self.slippage_bps) / 10000)

        if order["side"] == OrderSide.BUY.value:
            # Buying: pay more due to slippage
            fill_price = market_price * slippage_factor
        else:
            # Selling: receive less due to slippage
            fill_price = market_price * slippage_factor

        # Calculate fee (0.1% taker fee)
        fee = order["quantity"] * fill_price * 0.001

        # Check if we have sufficient balance
        if order["side"] == OrderSide.BUY.value:
            cost = order["quantity"] * fill_price + fee
            if self.balances.get(self.base_currency, 0) < cost:
                logger.error(f"Insufficient balance: need {cost}, have {self.balances.get(self.base_currency, 0)}")
                return {"success": False, "error": "Insufficient balance"}

            # Deduct balance
            self.balances[self.base_currency] -= cost

        else:  # SELL
            # Check if we have the asset to sell
            base_asset = order["symbol"].split("-")[0]
            if self.balances.get(base_asset, 0) < order["quantity"]:
                logger.error(f"Insufficient {base_asset}: need {order['quantity']}, have {self.balances.get(base_asset, 0)}")
                return {"success": False, "error": f"Insufficient {base_asset}"}

            # Deduct asset and credit base currency
            self.balances[base_asset] -= order["quantity"]
            proceeds = order["quantity"] * fill_price - fee
            self.balances[self.base_currency] = self.balances.get(self.base_currency, 0) + proceeds

        logger.info(
            f"Executed market order: {order['side']} {order['quantity']} @ {fill_price:.2f} "
            f"(slippage: {(slippage_factor - 1) * 10000:.2f}bps, fee: {fee:.2f})"
        )

        return {"success": True, "fill_price": fill_price, "fee": fee}

    async def _update_position(
        self, symbol: str, side: OrderSide, quantity: float, price: float
    ):
        """Update position after order fill"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "average_entry_price": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "last_updated": datetime.utcnow(),
            }

        position = self.positions[symbol]

        if side == OrderSide.BUY:
            # Increase position
            total_cost = position["quantity"] * position["average_entry_price"] + quantity * price
            position["quantity"] += quantity
            position["average_entry_price"] = total_cost / position["quantity"] if position["quantity"] > 0 else 0.0
        else:  # SELL
            # Decrease position or reverse
            if position["quantity"] > 0:
                # Calculate realized P&L
                realized_pnl = (price - position["average_entry_price"]) * min(quantity, position["quantity"])
                position["realized_pnl"] += realized_pnl

            position["quantity"] -= quantity

        position["last_updated"] = datetime.utcnow()
        logger.info(f"Updated position for {symbol}: {position['quantity']} @ {position['average_entry_price']:.2f}")

    async def cancel_order(self, exchange_order_id: str) -> Dict:
        """Cancel an order"""
        await asyncio.sleep(random.uniform(0.01, 0.03))

        if exchange_order_id not in self.orders:
            return {"success": False, "error": "Order not found"}

        order = self.orders[exchange_order_id]
        order["status"] = OrderStatus.CANCELLED.value
        order["updated_at"] = datetime.utcnow()

        self.order_history.append(order.copy())
        del self.orders[exchange_order_id]

        logger.info(f"Cancelled order {exchange_order_id}")
        return {"success": True, "order": order}

    async def modify_order(
        self,
        exchange_order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict:
        """Modify an existing order"""
        await asyncio.sleep(random.uniform(0.01, 0.03))

        if exchange_order_id not in self.orders:
            return {"success": False, "error": "Order not found"}

        order = self.orders[exchange_order_id]

        if quantity is not None:
            order["quantity"] = quantity
            order["remaining_quantity"] = quantity - order["filled_quantity"]

        if price is not None:
            order["price"] = price

        order["updated_at"] = datetime.utcnow()

        logger.info(f"Modified order {exchange_order_id}")
        return {"success": True, "order": order}

    async def get_order_status(self, exchange_order_id: str) -> Dict:
        """Get order status"""
        if exchange_order_id in self.orders:
            return {"success": True, "order": self.orders[exchange_order_id]}

        # Check history
        for order in self.order_history:
            if order["exchange_order_id"] == exchange_order_id:
                return {"success": True, "order": order}

        return {"success": False, "error": "Order not found"}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        if symbol:
            return [o for o in self.orders.values() if o["symbol"] == symbol]
        return list(self.orders.values())

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical orders"""
        history = self.order_history

        if symbol:
            history = [o for o in history if o["symbol"] == symbol]

        if start_time:
            history = [o for o in history if o["created_at"] >= start_time]

        if end_time:
            history = [o for o in history if o["created_at"] <= end_time]

        return history[-limit:]

    async def get_current_price(self, symbol: str) -> float:
        """
        Get current market price (simulated).
        In real implementation, this would fetch from Polygon.io or other data source.
        """
        # For simulation, use cached price or generate random price
        if symbol not in self.market_prices:
            # Initialize with reasonable default prices
            if "BTC" in symbol:
                self.market_prices[symbol] = 50000.0
            elif "ETH" in symbol:
                self.market_prices[symbol] = 3000.0
            else:
                self.market_prices[symbol] = 100.0

        # Add small random walk
        price = self.market_prices[symbol]
        change = random.uniform(-0.002, 0.002)  # ±0.2% change
        self.market_prices[symbol] = price * (1 + change)

        return self.market_prices[symbol]

    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get simulated order book"""
        mid_price = await self.get_current_price(symbol)

        # Generate synthetic order book
        bids = []
        asks = []

        for i in range(depth):
            bid_price = mid_price * (1 - (i + 1) * 0.0001)
            ask_price = mid_price * (1 + (i + 1) * 0.0001)

            bids.append(
                {
                    "price": bid_price,
                    "quantity": random.uniform(0.1, 10.0),
                }
            )
            asks.append(
                {
                    "price": ask_price,
                    "quantity": random.uniform(0.1, 10.0),
                }
            )

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.utcnow(),
        }

    async def get_balance(self, currency: Optional[str] = None) -> Dict:
        """Get account balance"""
        if currency:
            return {
                currency: {
                    "available": self.balances.get(currency, 0.0),
                    "locked": 0.0,
                    "total": self.balances.get(currency, 0.0),
                }
            }

        return {
            cur: {
                "available": bal,
                "locked": 0.0,
                "total": bal,
            }
            for cur, bal in self.balances.items()
        }

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions"""
        if symbol:
            return [self.positions[symbol]] if symbol in self.positions else []

        return list(self.positions.values())

    async def set_market_price(self, symbol: str, price: float):
        """
        Manually set market price (for testing).
        In production, prices would come from real market data feed.
        """
        self.market_prices[symbol] = price
        logger.info(f"Set market price for {symbol}: {price}")
