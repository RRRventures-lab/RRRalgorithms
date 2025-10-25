"""
Coinbase Advanced Trade REST API Client

Provides synchronous API calls for order execution, account management, and market data.
Implements JWT authentication for Coinbase Advanced Trade API.
"""

import hashlib
import hmac
import json
import logging
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import lru_cache


logger = logging.getLogger(__name__)


class CoinbaseRestClient:
    """
    Coinbase Advanced Trade REST API Client

    Handles authentication, rate limiting, and all REST API operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key: Optional[str] = None,
        base_url: str = "https://api.coinbase.com",
        timeout: int = 30,
    ):
        """
        Initialize Coinbase REST client

        Args:
            api_key: Coinbase API key (or from COINBASE_API_KEY env var)
            private_key: Coinbase private key (or from COINBASE_PRIVATE_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.private_key = private_key or os.getenv("COINBASE_PRIVATE_KEY")
        self.base_url = base_url
        self.timeout = timeout

        if not self.api_key or not self.private_key:
            raise ValueError(
                "Coinbase API credentials not found. Set COINBASE_API_KEY and "
                "COINBASE_PRIVATE_KEY environment variables."
            )

        self.session = requests.Session()
        logger.info("Coinbase REST client initialized")

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Generate HMAC SHA256 signature for API request

        Args:
            timestamp: Unix timestamp
            method: HTTP method (GET, POST, etc.)
            path: API path
            body: Request body (empty for GET)

        Returns:
            Hex signature string
        """
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.private_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated API request

        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/api/v3/brokerage/orders")
            params: Query parameters
            data: Request body data

        Returns:
            Response JSON
        """
        timestamp = str(int(time.time()))
        url = f"{self.base_url}{endpoint}"

        # Prepare request body
        body = ""
        if data:
            body = json.dumps(data)

        # Generate signature
        signature = self._generate_signature(timestamp, method, endpoint, body)

        # Set headers
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                response = self.session.get(
                    url, headers=headers, params=params, timeout=self.timeout
                )
            elif method == "POST":
                response = self.session.post(
                    url, headers=headers, json=data, timeout=self.timeout
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, headers=headers, params=params, timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}, Response: {e.response.text}")
            return {"error": str(e), "response": e.response.text}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e)}

    # ====================
    # Account Management
    # ====================

    @lru_cache(maxsize=128)
    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all accounts

        Returns:
            List of account dictionaries
        """
        response = self._make_request("GET", "/api/v3/brokerage/accounts")

        if "error" in response:
            logger.error(f"Failed to get accounts: {response['error']}")
            return []

        return response.get("accounts", [])

    @lru_cache(maxsize=128)
    def get_account_balance(self, currency: str = "USD") -> float:
        """
        Get account balance for a specific currency

        Args:
            currency: Currency symbol (e.g., "USD", "BTC")

        Returns:
            Available balance
        """
        accounts = self.get_accounts()

        for account in accounts:
            if account.get("currency") == currency:
                available = account.get("available_balance", {}).get("value", "0")
                return float(available)

        return 0.0

    @lru_cache(maxsize=128)
    def get_portfolio_breakdown(self) -> Dict[str, Any]:
        """
        Get complete portfolio breakdown with USD values

        Returns:
            Portfolio dictionary with positions and total value
        """
        accounts = self.get_accounts()
        positions = []
        total_value_usd = 0.0

        for account in accounts:
            currency = account.get("currency")
            balance = float(account.get("available_balance", {}).get("value", "0"))

            if balance > 0:
                # Get current USD value
                if currency == "USD":
                    usd_value = balance
                else:
                    product_id = f"{currency}-USD"
                    price = self.get_current_price(product_id)
                    usd_value = balance * price if price else 0

                positions.append({
                    "currency": currency,
                    "balance": balance,
                    "usd_value": usd_value,
                })

                total_value_usd += usd_value

        return {
            "total_value_usd": total_value_usd,
            "positions": positions,
            "num_positions": len(positions),
        }

    # ====================
    # Order Management
    # ====================

    def create_market_order(
        self,
        product_id: str,
        side: str,
        size: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a market order

        Args:
            product_id: Product ID (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            size: Order size in base currency
            client_order_id: Custom order ID (optional)

        Returns:
            Order response
        """
        data = {
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(size),
                }
            },
        }

        if client_order_id:
            data["client_order_id"] = client_order_id

        response = self._make_request("POST", "/api/v3/brokerage/orders", data=data)

        if "error" in response:
            logger.error(f"Failed to create market order: {response['error']}")

        return response

    def create_limit_order(
        self,
        product_id: str,
        side: str,
        size: float,
        price: float,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a limit order

        Args:
            product_id: Product ID
            side: "BUY" or "SELL"
            size: Order size
            price: Limit price
            post_only: Only post as maker (no immediate fill)
            client_order_id: Custom order ID (optional)

        Returns:
            Order response
        """
        data = {
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                    "post_only": post_only,
                }
            },
        }

        if client_order_id:
            data["client_order_id"] = client_order_id

        response = self._make_request("POST", "/api/v3/brokerage/orders", data=data)

        if "error" in response:
            logger.error(f"Failed to create limit order: {response['error']}")

        return response

    def create_stop_loss_order(
        self,
        product_id: str,
        side: str,
        size: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a stop-loss order

        Args:
            product_id: Product ID
            side: "BUY" or "SELL"
            size: Order size
            stop_price: Stop trigger price
            client_order_id: Custom order ID (optional)

        Returns:
            Order response
        """
        data = {
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(stop_price),
                    "stop_price": str(stop_price),
                    "stop_direction": "STOP_DIRECTION_STOP_DOWN" if side == "SELL" else "STOP_DIRECTION_STOP_UP",
                }
            },
        }

        if client_order_id:
            data["client_order_id"] = client_order_id

        response = self._make_request("POST", "/api/v3/brokerage/orders", data=data)

        if "error" in response:
            logger.error(f"Failed to create stop-loss order: {response['error']}")

        return response

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        endpoint = f"/api/v3/brokerage/orders/batch_cancel"
        data = {"order_ids": [order_id]}

        response = self._make_request("POST", endpoint, data=data)

        if "error" in response:
            logger.error(f"Failed to cancel order {order_id}: {response['error']}")
            return False

        results = response.get("results", [])
        if len(results) > 0:
            return results[0].get("success", False)

        return False

    @lru_cache(maxsize=128)
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details

        Args:
            order_id: Order ID

        Returns:
            Order dictionary or None
        """
        endpoint = f"/api/v3/brokerage/orders/historical/{order_id}"
        response = self._make_request("GET", endpoint)

        if "error" in response:
            logger.error(f"Failed to get order {order_id}: {response['error']}")
            return None

        return response.get("order")

    def list_orders(
        self,
        product_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List orders

        Args:
            product_id: Filter by product ID (optional)
            status: Filter by status ("OPEN", "FILLED", "CANCELLED")
            limit: Maximum number of orders

        Returns:
            List of orders
        """
        params = {"limit": limit}

        if product_id:
            params["product_id"] = product_id

        if status:
            params["order_status"] = status

        response = self._make_request("GET", "/api/v3/brokerage/orders/historical/batch", params=params)

        if "error" in response:
            logger.error(f"Failed to list orders: {response['error']}")
            return []

        return response.get("orders", [])

    # ====================
    # Market Data
    # ====================

    @lru_cache(maxsize=128)
    def get_current_price(self, product_id: str) -> Optional[float]:
        """
        Get current market price

        Args:
            product_id: Product ID (e.g., "BTC-USD")

        Returns:
            Current price or None
        """
        endpoint = f"/api/v3/brokerage/products/{product_id}/ticker"
        response = self._make_request("GET", endpoint)

        if "error" in response:
            logger.error(f"Failed to get price for {product_id}: {response['error']}")
            return None

        price = response.get("price")
        return float(price) if price else None

    @lru_cache(maxsize=128)
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get product information

        Args:
            product_id: Product ID

        Returns:
            Product dictionary or None
        """
        endpoint = f"/api/v3/brokerage/products/{product_id}"
        response = self._make_request("GET", endpoint)

        if "error" in response:
            logger.error(f"Failed to get product {product_id}: {response['error']}")
            return None

        return response

    @lru_cache(maxsize=128)
    def list_products(self) -> List[Dict[str, Any]]:
        """
        List all available products

        Returns:
            List of products
        """
        response = self._make_request("GET", "/api/v3/brokerage/products")

        if "error" in response:
            logger.error(f"Failed to list products: {response['error']}")
            return []

        return response.get("products", [])


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        client = CoinbaseRestClient()

        # Get account balance
        print("\n=== Account Balance ===")
        usd_balance = client.get_account_balance("USD")
        print(f"USD Balance: ${usd_balance:,.2f}")

        # Get Bitcoin price
        print("\n=== Current Prices ===")
        btc_price = client.get_current_price("BTC-USD")
        if btc_price:
            print(f"BTC-USD: ${btc_price:,.2f}")

        # Get portfolio
        print("\n=== Portfolio ===")
        portfolio = client.get_portfolio_breakdown()
        print(f"Total Value: ${portfolio['total_value_usd']:,.2f}")
        print(f"Positions: {portfolio['num_positions']}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"ERROR: {e}")
