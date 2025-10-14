from datetime import datetime
from functools import lru_cache
from rest_client import CoinbaseRestClient
from typing import Dict, Optional, Any, List
import os
import sys


"""
Coinbase Exchange Adapter

Implements the exchange interface for Coinbase Advanced Trade API.
Supports both paper trading (simulation) and live trading.
"""


# Add API integration path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../worktrees/api-integration/coinbase'))



class CoinbaseExchange:
    """
    Coinbase exchange adapter for trading engine

    Supports:
    - Market orders
    - Limit orders
    - Stop-loss orders
    - Order cancellation
    - Real-time order status
    - Position tracking
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize Coinbase exchange

        Args:
            paper_trading: If True, simulate orders without executing (default: True)
        """
        self.paper_trading = paper_trading
        self.client = CoinbaseRestClient()
        self.name = "Coinbase"

        # Paper trading state
        self.paper_orders = {}
        self.paper_fills = {}
        self.paper_order_id_counter = 1000

        # Validate paper trading mode
        env_paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        if not env_paper_trading and not paper_trading:
            print("⚠️  WARNING: Live trading enabled on Coinbase!")
            print("⚠️  Real money will be used for orders!")

        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
        print(f"✅ Coinbase exchange initialized ({mode})")

    # ====================
    # Order Execution
    # ====================

    def create_market_order(self, product_id: str, side: str, size: float,
                           client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a market order

        Args:
            product_id: Product ID (e.g., 'BTC-USD', 'ETH-USD')
            side: 'BUY' or 'SELL'
            size: Order size in base currency
            client_order_id: Custom order ID (optional)

        Returns:
            Order dictionary with order_id, status, price, etc.
        """
        if self.paper_trading:
            return self._simulate_market_order(product_id, side, size, client_order_id)

        # Live trading
        result = self.client.create_market_order(product_id, side, size, client_order_id)

        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'order_id': None
            }

        return {
            'success': True,
            'order_id': result.get('success_response', {}).get('order_id'),
            'product_id': product_id,
            'side': side,
            'size': size,
            'order_type': 'MARKET',
            'status': 'FILLED',  # Market orders fill immediately
            'timestamp': datetime.utcnow().isoformat()
        }

    def create_limit_order(self, product_id: str, side: str, size: float, price: float,
                          post_only: bool = False, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a limit order

        Args:
            product_id: Product ID
            side: 'BUY' or 'SELL'
            size: Order size
            price: Limit price
            post_only: Only post as maker (no immediate fill)
            client_order_id: Custom order ID (optional)

        Returns:
            Order dictionary
        """
        if self.paper_trading:
            return self._simulate_limit_order(product_id, side, size, price, post_only, client_order_id)

        # Live trading
        result = self.client.create_limit_order(product_id, side, size, price, post_only, client_order_id)

        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'order_id': None
            }

        return {
            'success': True,
            'order_id': result.get('success_response', {}).get('order_id'),
            'product_id': product_id,
            'side': side,
            'size': size,
            'price': price,
            'order_type': 'LIMIT',
            'status': 'OPEN',
            'timestamp': datetime.utcnow().isoformat()
        }

    def create_stop_loss_order(self, product_id: str, side: str, size: float, stop_price: float,
                               client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a stop-loss order

        Args:
            product_id: Product ID
            side: 'BUY' or 'SELL'
            size: Order size
            stop_price: Stop trigger price
            client_order_id: Custom order ID (optional)

        Returns:
            Order dictionary
        """
        if self.paper_trading:
            return self._simulate_stop_loss_order(product_id, side, size, stop_price, client_order_id)

        # Live trading
        result = self.client.create_stop_loss_order(product_id, side, size, stop_price, client_order_id)

        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'order_id': None
            }

        return {
            'success': True,
            'order_id': result.get('success_response', {}).get('order_id'),
            'product_id': product_id,
            'side': side,
            'size': size,
            'stop_price': stop_price,
            'order_type': 'STOP_LOSS',
            'status': 'OPEN',
            'timestamp': datetime.utcnow().isoformat()
        }

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        if self.paper_trading:
            if order_id in self.paper_orders:
                self.paper_orders[order_id]['status'] = 'CANCELLED'
                print(f"✅ Paper order {order_id} cancelled")
                return True
            return False

        # Live trading
        return self.client.cancel_order(order_id)

    @lru_cache(maxsize=128)

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific order

        Args:
            order_id: Order ID

        Returns:
            Order dictionary or None
        """
        if self.paper_trading:
            return self.paper_orders.get(order_id)

        # Live trading
        order = self.client.get_order(order_id)
        if not order:
            return None

        return {
            'order_id': order.get('order_id'),
            'product_id': order.get('product_id'),
            'side': order.get('side'),
            'size': order.get('order_configuration', {}).get('limit_limit_gtc', {}).get('base_size'),
            'price': order.get('order_configuration', {}).get('limit_limit_gtc', {}).get('limit_price'),
            'status': order.get('status'),
            'filled_size': order.get('filled_size'),
            'average_filled_price': order.get('average_filled_price'),
            'created_time': order.get('created_time')
        }

    def list_open_orders(self, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all open orders

        Args:
            product_id: Filter by product ID (optional)

        Returns:
            List of order dictionaries
        """
        if self.paper_trading:
            orders = [o for o in self.paper_orders.values() if o['status'] == 'OPEN']
            if product_id:
                orders = [o for o in orders if o['product_id'] == product_id]
            return orders

        # Live trading
        return self.client.list_orders(product_id=product_id, status='OPEN')

    # ====================
    # Market Data
    # ====================

    @lru_cache(maxsize=128)

    def get_current_price(self, product_id: str) -> Optional[float]:
        """
        Get current market price

        Args:
            product_id: Product ID

        Returns:
            Current price or None
        """
        return self.client.get_current_price(product_id)

    @lru_cache(maxsize=128)

    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get product information

        Args:
            product_id: Product ID

        Returns:
            Product dictionary or None
        """
        return self.client.get_product(product_id)

    # ====================
    # Account Management
    # ====================

    @lru_cache(maxsize=128)

    def get_account_balance(self, currency: str = 'USD') -> float:
        """
        Get account balance for a currency

        Args:
            currency: Currency symbol (default: USD)

        Returns:
            Available balance
        """
        if self.paper_trading:
            # Start with $100,000 paper money
            if currency == 'USD':
                return 100000.0
            return 0.0

        # Live trading
        return self.client.get_account_balance(currency)

    @lru_cache(maxsize=128)

    def get_portfolio(self) -> Dict[str, Any]:
        """
        Get complete portfolio breakdown

        Returns:
            Portfolio dictionary with positions and values
        """
        if self.paper_trading:
            return {
                'total_value_usd': 100000.0,
                'positions': [],
                'paper_trading': True
            }

        # Live trading
        return self.client.get_portfolio_breakdown()

    # ====================
    # Paper Trading Simulation
    # ====================

    def _simulate_market_order(self, product_id: str, side: str, size: float,
                               client_order_id: Optional[str]) -> Dict[str, Any]:
        """Simulate market order execution"""
        # Get current price
        current_price = self.get_current_price(product_id)
        if not current_price:
            return {'success': False, 'error': f'Could not get price for {product_id}'}

        # Add realistic slippage (0.05%)
        slippage = 0.0005
        fill_price = current_price * (1 + slippage) if side == 'BUY' else current_price * (1 - slippage)

        # Create order ID
        order_id = client_order_id or f"paper-{self.paper_order_id_counter}"
        self.paper_order_id_counter += 1

        # Store order
        order = {
            'success': True,
            'order_id': order_id,
            'product_id': product_id,
            'side': side,
            'size': size,
            'order_type': 'MARKET',
            'status': 'FILLED',
            'filled_size': size,
            'average_filled_price': fill_price,
            'timestamp': datetime.utcnow().isoformat(),
            'paper_trading': True
        }

        self.paper_orders[order_id] = order
        self.paper_fills[order_id] = {
            'fill_price': fill_price,
            'fill_size': size,
            'timestamp': datetime.utcnow().isoformat()
        }

        print(f"📝 Paper trade: {side} {size} {product_id} @ ${fill_price:,.2f}")

        return order

    def _simulate_limit_order(self, product_id: str, side: str, size: float, price: float,
                             post_only: bool, client_order_id: Optional[str]) -> Dict[str, Any]:
        """Simulate limit order creation"""
        # Create order ID
        order_id = client_order_id or f"paper-{self.paper_order_id_counter}"
        self.paper_order_id_counter += 1

        # Store order
        order = {
            'success': True,
            'order_id': order_id,
            'product_id': product_id,
            'side': side,
            'size': size,
            'price': price,
            'order_type': 'LIMIT',
            'status': 'OPEN',
            'filled_size': 0,
            'post_only': post_only,
            'timestamp': datetime.utcnow().isoformat(),
            'paper_trading': True
        }

        self.paper_orders[order_id] = order

        print(f"📝 Paper limit order: {side} {size} {product_id} @ ${price:,.2f}")

        return order

    def _simulate_stop_loss_order(self, product_id: str, side: str, size: float, stop_price: float,
                                  client_order_id: Optional[str]) -> Dict[str, Any]:
        """Simulate stop-loss order creation"""
        # Create order ID
        order_id = client_order_id or f"paper-{self.paper_order_id_counter}"
        self.paper_order_id_counter += 1

        # Store order
        order = {
            'success': True,
            'order_id': order_id,
            'product_id': product_id,
            'side': side,
            'size': size,
            'stop_price': stop_price,
            'order_type': 'STOP_LOSS',
            'status': 'OPEN',
            'filled_size': 0,
            'timestamp': datetime.utcnow().isoformat(),
            'paper_trading': True
        }

        self.paper_orders[order_id] = order

        print(f"📝 Paper stop-loss: {side} {size} {product_id} @ stop ${stop_price:,.2f}")

        return order


# ====================
# Example Usage
# ====================

if __name__ == '__main__':
    # Initialize exchange (paper trading)
    exchange = CoinbaseExchange(paper_trading=True)

    # Get Bitcoin price
    print("\n=== Current Prices ===")
    btc_price = exchange.get_current_price('BTC-USD')
    print(f"BTC-USD: ${btc_price:,.2f}")

    # Get account balance
    print("\n=== Account Balance ===")
    usd_balance = exchange.get_account_balance('USD')
    print(f"USD Balance: ${usd_balance:,.2f}")

    # Create market order
    print("\n=== Create Market Order ===")
    order = exchange.create_market_order('BTC-USD', 'BUY', 0.001)
    if order['success']:
        print(f"Order ID: {order['order_id']}")
        print(f"Status: {order['status']}")
        print(f"Fill Price: ${order.get('average_filled_price', 0):,.2f}")

    # Create limit order
    print("\n=== Create Limit Order ===")
    limit_price = btc_price * 0.95  # 5% below current price
    limit_order = exchange.create_limit_order('BTC-USD', 'BUY', 0.001, limit_price)
    if limit_order['success']:
        print(f"Order ID: {limit_order['order_id']}")
        print(f"Limit Price: ${limit_price:,.2f}")
        print(f"Status: {limit_order['status']}")

    # List open orders
    print("\n=== Open Orders ===")
    open_orders = exchange.list_open_orders()
    print(f"Total open orders: {len(open_orders)}")
    for o in open_orders:
        print(f"  {o['order_id']}: {o['side']} {o['size']} {o['product_id']} @ ${o.get('price', 0):,.2f}")

    # Cancel order
    if len(open_orders) > 0:
        print("\n=== Cancel Order ===")
        order_to_cancel = open_orders[0]['order_id']
        success = exchange.cancel_order(order_to_cancel)
        print(f"Cancel successful: {success}")

    # Get portfolio
    print("\n=== Portfolio ===")
    portfolio = exchange.get_portfolio()
    print(f"Total Value: ${portfolio['total_value_usd']:,.2f}")
    print(f"Paper Trading: {portfolio.get('paper_trading', False)}")
