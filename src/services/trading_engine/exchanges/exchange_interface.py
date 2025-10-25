from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


"""
Exchange Interface - Abstract Base Class
Defines the contract for all exchange connectors (paper and live trading)
"""



class OrderType(Enum):
    """Order types supported by the exchange"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status lifecycle"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options"""
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    DAY = "day"  # Day order


class ExchangeInterface(ABC):
    """
    Abstract base class for all exchange connectors.
    Implements common functionality and defines required methods.
    """

    def __init__(self, exchange_id: str, paper_trading: bool = True):
        """
        Initialize exchange connector

        Args:
            exchange_id: Unique identifier for this exchange instance
            paper_trading: Whether this is paper trading mode
        """
        self.exchange_id = exchange_id
        self.paper_trading = paper_trading
        self.connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to exchange API

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from exchange API

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
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
        """
        Create a new order on the exchange

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: Buy or sell
            order_type: Market, limit, stop, etc.
            quantity: Amount to buy/sell
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Order time in force
            client_order_id: Optional client-side order ID

        Returns:
            Dict containing order details including exchange_order_id
        """
        pass

    @abstractmethod
    async def cancel_order(self, exchange_order_id: str) -> Dict:
        """
        Cancel an existing order

        Args:
            exchange_order_id: Exchange's order ID

        Returns:
            Dict containing cancellation status
        """
        pass

    @abstractmethod
    async def modify_order(
        self,
        exchange_order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict:
        """
        Modify an existing order

        Args:
            exchange_order_id: Exchange's order ID
            quantity: New quantity (optional)
            price: New price (optional)

        Returns:
            Dict containing modified order details
        """
        pass

    @abstractmethod
    async def get_order_status(self, exchange_order_id: str) -> Dict:
        """
        Get current status of an order

        Args:
            exchange_order_id: Exchange's order ID

        Returns:
            Dict containing order status and details
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
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
            symbol: Optional symbol to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol

        Args:
            symbol: Trading pair

        Returns:
            Current market price
        """
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """
        Get order book for a symbol

        Args:
            symbol: Trading pair
            depth: Number of levels to return

        Returns:
            Dict with bids and asks
        """
        pass

    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Get account balance

        Args:
            currency: Optional currency to filter by

        Returns:
            Dict with balance information
        """
        pass

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get current positions

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List of current positions
        """
        pass

    def validate_order_params(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Validate order parameters before submission

        Args:
            symbol: Trading pair
            side: Buy or sell
            order_type: Order type
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price

        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None or price <= 0:
                return False, f"{order_type.value} orders require a positive price"

        if order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.STOP_LIMIT]:
            if stop_price is None or stop_price <= 0:
                return False, f"{order_type.value} orders require a positive stop_price"

        if not symbol or "/" not in symbol and "-" not in symbol:
            return False, "Invalid symbol format"

        return True, ""

    def __repr__(self):
        mode = "PAPER" if self.paper_trading else "LIVE"
        status = "CONNECTED" if self.connected else "DISCONNECTED"
        return f"<{self.__class__.__name__} id={self.exchange_id} mode={mode} status={status}>"
