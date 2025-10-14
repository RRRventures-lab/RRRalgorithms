from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import List, Optional, Dict, Any, Callable
from websockets.exceptions import ConnectionClosed, WebSocketException
import asyncio
import json
import logging
import os
import websockets


"""
Polygon.io WebSocket Client for Real-time Crypto Data Streaming
================================================================

This module provides a WebSocket client for streaming real-time cryptocurrency
market data from Polygon.io, including trades, quotes, and aggregates.

Features:
- Async WebSocket streaming
- Auto-reconnection with exponential backoff
- Multiple data type subscriptions (trades, quotes, aggregates)
- Error handling and recovery
- Direct integration with Supabase for data storage
- Configurable crypto pairs

Usage:
    from data_pipeline.polygon.websocket_client import PolygonWebSocketClient
    from src.database import SQLiteClient as DatabaseClient

    supabase = get_db()
    ws_client = PolygonWebSocketClient(supabase_client=supabase)

    # Run the streaming client
    await ws_client.run()
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolygonWebSocketClient:
    """
    WebSocket client for real-time Polygon.io crypto market data.

    Streams:
    - XT: Crypto trades
    - XQ: Crypto quotes (bid/ask)
    - XA: Crypto aggregates (1-minute bars)

    Data is automatically stored in Supabase tables:
    - crypto_trades
    - crypto_quotes
    - crypto_aggregates
    """

    # Polygon WebSocket endpoint
    WEBSOCKET_URL = "wss://socket.polygon.io/crypto"

    # Default crypto pairs to track
    DEFAULT_PAIRS = [
        "X:BTCUSD",   # Bitcoin
        "X:ETHUSD",   # Ethereum
        "X:SOLUSD",   # Solana
        "X:ADAUSD",   # Cardano
        "X:DOTUSD",   # Polkadot
        "X:MATICUSD", # Polygon
        "X:AVAXUSD",  # Avalanche
        "X:ATOMUSD",  # Cosmos
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        supabase_client=None,
        pairs: Optional[List[str]] = None,
        enable_trades: bool = True,
        enable_quotes: bool = True,
        enable_aggregates: bool = True,
        reconnect_delay: int = 5,
        max_reconnect_delay: int = 60,
    ):
        """
        Initialize Polygon WebSocket client.

        Args:
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
            supabase_client: SupabaseClient instance for data storage
            pairs: List of crypto pairs to subscribe to (default: major coins)
            enable_trades: Subscribe to trade updates (XT)
            enable_quotes: Subscribe to quote updates (XQ)
            enable_aggregates: Subscribe to aggregate bars (XA)
            reconnect_delay: Initial reconnect delay in seconds
            max_reconnect_delay: Maximum reconnect delay in seconds
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var or pass api_key parameter."
            )

        self.supabase_client = supabase_client
        if not self.supabase_client:
            logger.warning("No Supabase client provided. Data will not be stored.")

        self.pairs = pairs or self.DEFAULT_PAIRS
        self.enable_trades = enable_trades
        self.enable_quotes = enable_quotes
        self.enable_aggregates = enable_aggregates

        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.current_reconnect_delay = reconnect_delay

        self.websocket = None
        self.running = False
        self.connected = False

        # Statistics
        self.message_count = 0
        self.trade_count = 0
        self.quote_count = 0
        self.aggregate_count = 0
        self.error_count = 0
        self.last_message_time = None

        logger.info(f"Polygon WebSocket client initialized for {len(self.pairs)} pairs")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection and authenticate.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.WEBSOCKET_URL}")
            self.websocket = await websockets.connect(
                self.WEBSOCKET_URL,
                ping_interval=20,
                ping_timeout=10,
            )

            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))

            # Wait for auth response
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data[0].get("status") == "auth_success":
                logger.info("Authentication successful")
                self.connected = True
                self.current_reconnect_delay = self.reconnect_delay
                return True
            else:
                logger.error(f"Authentication failed: {response_data}")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def subscribe(self):
        """Subscribe to market data streams."""
        if not self.websocket or not self.connected:
            logger.error("Not connected. Cannot subscribe.")
            return

        subscriptions = []

        # Build subscription list
        for pair in self.pairs:
            if self.enable_trades:
                subscriptions.append(f"XT.{pair}")
            if self.enable_quotes:
                subscriptions.append(f"XQ.{pair}")
            if self.enable_aggregates:
                subscriptions.append(f"XA.{pair}")

        subscribe_message = {
            "action": "subscribe",
            "params": ",".join(subscriptions)
        }

        logger.info(f"Subscribing to {len(subscriptions)} streams")
        await self.websocket.send(json.dumps(subscribe_message))

        # Wait for subscription confirmation
        response = await self.websocket.recv()
        response_data = json.loads(response)

        if response_data[0].get("status") == "success":
            logger.info(f"Subscription successful: {response_data[0].get('message')}")
        else:
            logger.warning(f"Subscription response: {response_data}")

    async def process_message(self, message_data: List[Dict[str, Any]]):
        """
        Process incoming WebSocket messages.

        Args:
            message_data: List of message objects from Polygon
        """
        for msg in message_data:
            try:
                msg_type = msg.get("ev")

                if msg_type == "XT":
                    # Trade event
                    await self.process_trade(msg)
                    self.trade_count += 1

                elif msg_type == "XQ":
                    # Quote event
                    await self.process_quote(msg)
                    self.quote_count += 1

                elif msg_type == "XA":
                    # Aggregate event
                    await self.process_aggregate(msg)
                    self.aggregate_count += 1

                elif msg_type == "status":
                    # Status message
                    logger.info(f"Status: {msg.get('message')}")

                else:
                    logger.debug(f"Unhandled message type: {msg_type}")

                self.message_count += 1
                self.last_message_time = datetime.now()

            except Exception as e:
                logger.error(f"Error processing message {msg.get('ev')}: {e}")
                self.error_count += 1

    async def process_trade(self, trade_data: Dict[str, Any]):
        """
        Process and store trade data.

        Args:
            trade_data: Trade message from Polygon
        """
        try:
            # Extract trade data
            ticker = trade_data.get("pair")
            timestamp_ms = trade_data.get("t")
            price = float(trade_data.get("p"))
            size = float(trade_data.get("s"))
            exchange = trade_data.get("x")
            conditions = trade_data.get("c", [])
            trade_id = trade_data.get("i")

            # Convert to datetime
            event_time = datetime.fromtimestamp(timestamp_ms / 1000.0)

            # Prepare data for Supabase
            db_data = {
                "ticker": ticker,
                "event_time": event_time.isoformat(),
                "price": price,
                "size": size,
                "exchange_id": exchange,
                "conditions": conditions,
                "trade_id": trade_id,
            }

            # Store in Supabase
            if self.supabase_client:
                self.supabase_client.insert_crypto_trade(db_data)
                logger.debug(f"Trade stored: {ticker} @ ${price}")

        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            raise

    async def process_quote(self, quote_data: Dict[str, Any]):
        """
        Process and store quote (bid/ask) data.

        Args:
            quote_data: Quote message from Polygon
        """
        try:
            # Extract quote data
            ticker = quote_data.get("pair")
            timestamp_ms = quote_data.get("t")
            bid_price = float(quote_data.get("bp"))
            bid_size = float(quote_data.get("bs"))
            ask_price = float(quote_data.get("ap"))
            ask_size = float(quote_data.get("as"))
            exchange = quote_data.get("x")

            # Convert to datetime
            event_time = datetime.fromtimestamp(timestamp_ms / 1000.0)

            # Calculate spread
            spread = ask_price - bid_price

            # Prepare data for Supabase
            db_data = {
                "ticker": ticker,
                "event_time": event_time.isoformat(),
                "bid_price": bid_price,
                "bid_size": bid_size,
                "ask_price": ask_price,
                "ask_size": ask_size,
                "spread": spread,
                "exchange_id": exchange,
            }

            # Store in Supabase
            if self.supabase_client:
                self.supabase_client.insert_crypto_quote(db_data)
                logger.debug(f"Quote stored: {ticker} bid=${bid_price} ask=${ask_price}")

        except Exception as e:
            logger.error(f"Error processing quote: {e}")
            raise

    async def process_aggregate(self, agg_data: Dict[str, Any]):
        """
        Process and store aggregate (OHLCV) bar data.

        Args:
            agg_data: Aggregate message from Polygon
        """
        try:
            # Extract aggregate data
            ticker = agg_data.get("pair")
            timestamp_ms = agg_data.get("s")  # Start time
            open_price = float(agg_data.get("o"))
            high_price = float(agg_data.get("h"))
            low_price = float(agg_data.get("l"))
            close_price = float(agg_data.get("c"))
            volume = float(agg_data.get("v"))
            vwap = float(agg_data.get("vw", 0.0)) if agg_data.get("vw") else None
            trade_count = agg_data.get("n")

            # Convert to datetime
            event_time = datetime.fromtimestamp(timestamp_ms / 1000.0)

            # Prepare data for Supabase
            db_data = {
                "ticker": ticker,
                "event_time": event_time.isoformat(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "vwap": vwap,
                "trade_count": trade_count,
            }

            # Store in Supabase
            if self.supabase_client:
                self.supabase_client.insert_crypto_aggregate(db_data)
                logger.info(f"Aggregate stored: {ticker} @ ${close_price} (vol: {volume:.2f})")

        except Exception as e:
            logger.error(f"Error processing aggregate: {e}")
            raise

    async def listen(self):
        """Listen for incoming WebSocket messages."""
        try:
            while self.running and self.connected:
                message = await self.websocket.recv()
                message_data = json.loads(message)
                await self.process_message(message_data)

        except ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            self.connected = False

        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.connected = False
            self.error_count += 1

    async def reconnect(self):
        """Reconnect with exponential backoff."""
        while self.running and not self.connected:
            logger.info(f"Attempting to reconnect in {self.current_reconnect_delay}s...")
            await asyncio.sleep(self.current_reconnect_delay)

            if await self.connect():
                await self.subscribe()
            else:
                # Exponential backoff
                self.current_reconnect_delay = min(
                    self.current_reconnect_delay * 2,
                    self.max_reconnect_delay
                )

    async def run(self):
        """
        Main run loop with auto-reconnection.

        This method will run indefinitely, maintaining the WebSocket connection
        and automatically reconnecting on disconnection.
        """
        self.running = True
        logger.info("Starting Polygon WebSocket client...")

        while self.running:
            try:
                if not self.connected:
                    if await self.connect():
                        await self.subscribe()

                if self.connected:
                    await self.listen()

                # If we get here, connection was lost
                if self.running:
                    await self.reconnect()

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.running = False

            except Exception as e:
                logger.error(f"Unexpected error in run loop: {e}")
                self.error_count += 1
                await asyncio.sleep(5)

        await self.stop()

    async def stop(self):
        """Stop the WebSocket client and close connection."""
        logger.info("Stopping Polygon WebSocket client...")
        self.running = False
        self.connected = False

        if self.websocket:
            await self.websocket.close()

        logger.info(f"Statistics:")
        logger.info(f"  Total messages: {self.message_count}")
        logger.info(f"  Trades: {self.trade_count}")
        logger.info(f"  Quotes: {self.quote_count}")
        logger.info(f"  Aggregates: {self.aggregate_count}")
        logger.info(f"  Errors: {self.error_count}")

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "running": self.running,
            "connected": self.connected,
            "message_count": self.message_count,
            "trade_count": self.trade_count,
            "quote_count": self.quote_count,
            "aggregate_count": self.aggregate_count,
            "error_count": self.error_count,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "subscribed_pairs": self.pairs,
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of Polygon WebSocket client."""
    from src.database import SQLiteClient as DatabaseClient

    # Initialize Supabase client
    supabase = get_db()

    # Initialize WebSocket client
    ws_client = PolygonWebSocketClient(
        supabase_client=supabase,
        pairs=["X:BTCUSD", "X:ETHUSD"],  # Track only BTC and ETH for demo
        enable_trades=True,
        enable_quotes=True,
        enable_aggregates=True,
    )

    # Run the client (will run indefinitely)
    try:
        await ws_client.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
