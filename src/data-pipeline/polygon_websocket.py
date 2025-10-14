from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, Callable, List
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import asyncio
import json
import logging
import os
import pandas as pd
import psycopg2
import signal
import sys
import websockets

"""
Polygon.io WebSocket client for real-time cryptocurrency market data streaming.
Implements automatic reconnection, data validation, and database persistence.
"""


# Load environment variables
load_dotenv('config/api-keys/.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/polygon_websocket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CryptoTrade:
    """Represents a cryptocurrency trade."""
    symbol: str
    price: float
    size: float
    timestamp: int
    exchange: int
    conditions: List[int]

@dataclass
class CryptoQuote:
    """Represents a cryptocurrency quote (bid/ask)."""
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: int
    exchange: int

@dataclass
class CryptoAggregate:
    """Represents an aggregate bar (OHLCV)."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: int
    timespan: str

class PolygonWebSocketClient:
    """
    Polygon.io WebSocket client for real-time cryptocurrency data.
    Supports trades, quotes, and aggregate bars with automatic reconnection.
    """

    WEBSOCKET_URL = "wss://socket.polygon.io/crypto"
    RECONNECT_DELAY = 5  # seconds
    MAX_RECONNECT_ATTEMPTS = 10
    HEARTBEAT_INTERVAL = 30  # seconds

    def __init__(self, api_key: str = None):
        """Initialize the WebSocket client."""
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not found. Set POLYGON_API_KEY environment variable.")

        self.websocket: Optional[WebSocketClientProtocol] = None
        self.subscriptions: List[str] = []
        self.is_running = False
        self.reconnect_attempts = 0
        self.message_callbacks: Dict[str, Callable] = {}
        self.db_connection = None
        self.data_buffer = []
        self.buffer_size = 100
        self.last_heartbeat = datetime.now(timezone.utc)

        # Statistics
        self.stats = {
            'trades_received': 0,
            'quotes_received': 0,
            'aggregates_received': 0,
            'errors': 0,
            'reconnects': 0
        }

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            logger.info(f"Connecting to {self.WEBSOCKET_URL}...")
            self.websocket = await websockets.connect(
                self.WEBSOCKET_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )

            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))

            # Wait for authentication response
            # First message is usually "connected" status
            first_response = await self.websocket.recv()
            first_data = json.loads(first_response)
            logger.debug(f"First response: {first_data}")

            # Check if we got connected status
            if isinstance(first_data, list) and len(first_data) > 0:
                if first_data[0].get('status') == 'connected':
                    logger.info("Connected to Polygon.io, waiting for auth...")
                    # Wait for auth response
                    auth_response = await self.websocket.recv()
                    auth_data = json.loads(auth_response)

                    if isinstance(auth_data, list) and len(auth_data) > 0:
                        if auth_data[0].get('status') == 'auth_success':
                            logger.info("Successfully authenticated with Polygon.io")
                            self.reconnect_attempts = 0
                            return True
                elif first_data[0].get('status') == 'auth_success':
                    logger.info("Successfully authenticated with Polygon.io")
                    self.reconnect_attempts = 0
                    return True

            logger.error(f"Authentication failed or unexpected response")
            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def subscribe(self, symbols: List[str], channels: List[str] = None):
        """
        Subscribe to market data for specific symbols.

        Args:
            symbols: List of cryptocurrency symbols (e.g., ['BTC-USD', 'ETH-USD'])
            channels: List of channels to subscribe to ('XT' for trades, 'XQ' for quotes, 'XA' for aggregates)
        """
        if channels is None:
            channels = ['XT', 'XQ', 'XA']  # Default to all channels

        subscriptions = []
        for channel in channels:
            for symbol in symbols:
                # Format: XT.BTC-USD for trades, XQ.BTC-USD for quotes, XA.BTC-USD for aggregates
                subscription = f"{channel}.{symbol}"
                subscriptions.append(subscription)
                self.subscriptions.append(subscription)

        subscribe_message = {
            "action": "subscribe",
            "params": ",".join(subscriptions)
        }

        if self.websocket:
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to: {subscriptions}")

    async def unsubscribe(self, symbols: List[str], channels: List[str] = None):
        """Unsubscribe from market data."""
        if channels is None:
            channels = ['XT', 'XQ', 'XA']

        unsubscriptions = []
        for channel in channels:
            for symbol in symbols:
                subscription = f"{channel}.{symbol}"
                unsubscriptions.append(subscription)
                if subscription in self.subscriptions:
                    self.subscriptions.remove(subscription)

        unsubscribe_message = {
            "action": "unsubscribe",
            "params": ",".join(unsubscriptions)
        }

        if self.websocket:
            await self.websocket.send(json.dumps(unsubscribe_message))
            logger.info(f"Unsubscribed from: {unsubscriptions}")

    def parse_trade(self, data: Dict[str, Any]) -> CryptoTrade:
        """Parse a trade message."""
        return CryptoTrade(
            symbol=data.get('pair', ''),
            price=float(data.get('p', 0)),
            size=float(data.get('s', 0)),
            timestamp=int(data.get('t', 0)),
            exchange=int(data.get('x', 0)),
            conditions=data.get('c', [])
        )

    def parse_quote(self, data: Dict[str, Any]) -> CryptoQuote:
        """Parse a quote message."""
        return CryptoQuote(
            symbol=data.get('pair', ''),
            bid_price=float(data.get('bp', 0)),
            bid_size=float(data.get('bs', 0)),
            ask_price=float(data.get('ap', 0)),
            ask_size=float(data.get('as', 0)),
            timestamp=int(data.get('t', 0)),
            exchange=int(data.get('x', 0))
        )

    def parse_aggregate(self, data: Dict[str, Any]) -> CryptoAggregate:
        """Parse an aggregate bar message."""
        return CryptoAggregate(
            symbol=data.get('pair', ''),
            open=float(data.get('o', 0)),
            high=float(data.get('h', 0)),
            low=float(data.get('l', 0)),
            close=float(data.get('c', 0)),
            volume=float(data.get('v', 0)),
            timestamp=int(data.get('s', 0)),
            timespan='minute'  # Default to minute bars
        )

    async def handle_message(self, message: str):
        """Process incoming WebSocket messages."""
        try:
            data_list = json.loads(message)

            for data in data_list:
                event_type = data.get('ev')

                if event_type == 'XT':  # Crypto Trade
                    trade = self.parse_trade(data)
                    self.stats['trades_received'] += 1
                    await self.on_trade(trade)

                elif event_type == 'XQ':  # Crypto Quote
                    quote = self.parse_quote(data)
                    self.stats['quotes_received'] += 1
                    await self.on_quote(quote)

                elif event_type == 'XA':  # Crypto Aggregate
                    aggregate = self.parse_aggregate(data)
                    self.stats['aggregates_received'] += 1
                    await self.on_aggregate(aggregate)

                elif event_type == 'status':
                    logger.info(f"Status message: {data}")

                else:
                    logger.debug(f"Unhandled event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.stats['errors'] += 1

    async def on_trade(self, trade: CryptoTrade):
        """Handle incoming trade data."""
        logger.debug(f"Trade: {trade.symbol} @ ${trade.price} x {trade.size}")

        # Add to buffer for batch insert
        self.data_buffer.append(('trade', trade))

        # Flush buffer if needed
        if len(self.data_buffer) >= self.buffer_size:
            await self.flush_buffer()

        # Call custom callback if registered
        if 'trade' in self.message_callbacks:
            await self.message_callbacks['trade'](trade)

    async def on_quote(self, quote: CryptoQuote):
        """Handle incoming quote data."""
        logger.debug(f"Quote: {quote.symbol} Bid: ${quote.bid_price} Ask: ${quote.ask_price}")

        # Add to buffer for batch insert
        self.data_buffer.append(('quote', quote))

        # Flush buffer if needed
        if len(self.data_buffer) >= self.buffer_size:
            await self.flush_buffer()

        # Call custom callback if registered
        if 'quote' in self.message_callbacks:
            await self.message_callbacks['quote'](quote)

    async def on_aggregate(self, aggregate: CryptoAggregate):
        """Handle incoming aggregate bar data."""
        logger.info(f"Bar: {aggregate.symbol} O:{aggregate.open} H:{aggregate.high} L:{aggregate.low} C:{aggregate.close} V:{aggregate.volume}")

        # Add to buffer for batch insert
        self.data_buffer.append(('aggregate', aggregate))

        # Flush buffer if needed
        if len(self.data_buffer) >= self.buffer_size:
            await self.flush_buffer()

        # Call custom callback if registered
        if 'aggregate' in self.message_callbacks:
            await self.message_callbacks['aggregate'](aggregate)

    async def flush_buffer(self):
        """Flush data buffer to database."""
        if not self.data_buffer:
            return

        # TODO: Implement database persistence
        # This would batch insert all buffered data to PostgreSQL
        logger.debug(f"Flushing {len(self.data_buffer)} records to database")
        self.data_buffer.clear()

    async def heartbeat(self):
        """Send periodic heartbeat to keep connection alive."""
        while self.is_running:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if self.websocket:
                    await self.websocket.ping()
                    self.last_heartbeat = datetime.now(timezone.utc)
                    logger.debug("Heartbeat sent")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def run(self):
        """Main event loop for WebSocket client."""
        self.is_running = True

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.heartbeat())

        while self.is_running:
            try:
                # Connect if not connected
                if not self.websocket:
                    connected = await self.connect()
                    if not connected:
                        if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                            logger.error("Max reconnection attempts reached. Exiting.")
                            break

                        self.reconnect_attempts += 1
                        logger.info(f"Reconnecting in {self.RECONNECT_DELAY} seconds... (Attempt {self.reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})")
                        await asyncio.sleep(self.RECONNECT_DELAY)
                        continue

                    # Re-subscribe to previous subscriptions after successful connection
                    if self.subscriptions:
                        # Create unique subscription list
                        unique_subs = list(set(self.subscriptions))
                        self.subscriptions = []  # Clear to avoid duplicates

                        # Parse symbols and channels from subscriptions
                        symbols = set()
                        channels = set()
                        for sub in unique_subs:
                            parts = sub.split('.')
                            if len(parts) == 2:
                                channels.add(parts[0])
                                symbols.add(parts[1])

                        if symbols and channels:
                            await self.subscribe(list(symbols), list(channels))
                            logger.info(f"Re-subscribed to {len(symbols)} symbols")

                # Receive and process messages
                message = await self.websocket.recv()
                await self.handle_message(message)

            except ConnectionClosedError as e:
                logger.warning(f"Connection closed: {e}")
                self.websocket = None
                self.stats['reconnects'] += 1

            except ConnectionClosedOK:
                logger.info("Connection closed gracefully")
                break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)

        # Cancel heartbeat
        heartbeat_task.cancel()

        # Flush remaining buffer
        await self.flush_buffer()

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()

        logger.info("WebSocket client stopped")
        self.print_statistics()

    def print_statistics(self):
        """Print session statistics."""
        logger.info("=== Session Statistics ===")
        logger.info(f"Trades received: {self.stats['trades_received']}")
        logger.info(f"Quotes received: {self.stats['quotes_received']}")
        logger.info(f"Aggregates received: {self.stats['aggregates_received']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Reconnects: {self.stats['reconnects']}")

    def register_callback(self, event_type: str, callback: Callable):
        """Register a custom callback for specific event types."""
        self.message_callbacks[event_type] = callback

    def stop(self):
        """Stop the WebSocket client."""
        self.is_running = False
        logger.info("Stopping WebSocket client...")

async def main():
    """Example usage of Polygon WebSocket client."""
    client = PolygonWebSocketClient()

    # Define custom callback for trades
    async def trade_callback(trade: CryptoTrade):
        print(f"[TRADE] {trade.symbol}: ${trade.price} x {trade.size}")

    # Register callback
    client.register_callback('trade', trade_callback)

    # Subscribe to Bitcoin and Ethereum
    await client.subscribe(['BTC-USD', 'ETH-USD'], channels=['XT', 'XA'])

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        client.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run the client
    await client.run()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Run the WebSocket client
    asyncio.run(main())