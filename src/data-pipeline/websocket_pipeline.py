from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from src.core.exceptions import DataPipelineError, APIError
from src.core.memory_cache import get_cache
from typing import Dict, List, Optional, Callable, Any
from websockets.exceptions import ConnectionClosed, WebSocketException
import asyncio
import json
import logging
import os
import time
import websockets


"""
WebSocket Data Pipeline
======================

Real-time market data pipeline using WebSocket connections.
Supports multiple exchanges and data sources with automatic reconnection.

Author: RRR Ventures
Date: 2025-10-12
"""




@dataclass
class MarketData:
    """Market data structure for real-time updates."""
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    source: str = "websocket"


class WebSocketDataSource:
    """
    WebSocket-based real-time data source.
    
    Features:
    - Multiple exchange support
    - Automatic reconnection
    - Data validation and normalization
    - Caching and batching
    - Error handling and recovery
    """
    
    def __init__(
        self,
        symbols: List[str],
        exchanges: List[str] = None,
        update_interval: float = 1.0,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize WebSocket data source.
        
        Args:
            symbols: List of trading symbols to monitor
            exchanges: List of exchanges to connect to
            update_interval: Update interval in seconds
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.symbols = symbols
        self.exchanges = exchanges or ['polygon', 'binance', 'coinbase']
        self.update_interval = update_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # WebSocket connections
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = False
        self.reconnect_attempts: Dict[str, int] = {}
        
        # Data processing
        self.data_callbacks: List[Callable[[MarketData], None]] = []
        self.cache = get_cache()
        self.data_buffer: Dict[str, List[MarketData]] = {}
        
        # Performance metrics
        self.metrics = {
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0,
            'reconnections': 0,
            'last_update': 0.0,
            'avg_latency_ms': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the WebSocket data pipeline."""
        self.logger.info("Starting WebSocket data pipeline...")
        self.running = True
        
        # Start connections for each exchange
        tasks = []
        for exchange in self.exchanges:
            task = asyncio.create_task(self._connect_exchange(exchange))
            tasks.append(task)
        
        # Wait for all connections to be established
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("WebSocket data pipeline started")
    
    async def stop(self) -> None:
        """Stop the WebSocket data pipeline."""
        self.logger.info("Stopping WebSocket data pipeline...")
        self.running = False
        
        # Close all connections
        for exchange, connection in self.connections.items():
            if connection and not connection.closed:
                await connection.close()
        
        self.connections.clear()
        self.logger.info("WebSocket data pipeline stopped")
    
    async def _connect_exchange(self, exchange: str) -> None:
        """Connect to a specific exchange."""
        self.reconnect_attempts[exchange] = 0
        
        while self.running and self.reconnect_attempts[exchange] < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Connecting to {exchange}...")
                
                # Get WebSocket URL for exchange
                ws_url = self._get_websocket_url(exchange)
                if not ws_url:
                    self.logger.error(f"No WebSocket URL configured for {exchange}")
                    break
                
                # Connect to WebSocket
                async with websockets.connect(ws_url) as websocket:
                    self.connections[exchange] = websocket
                    self.reconnect_attempts[exchange] = 0
                    
                    self.logger.info(f"Connected to {exchange}")
                    
                    # Subscribe to symbols
                    await self._subscribe_symbols(websocket, exchange)
                    
                    # Start listening for messages
                    await self._listen_messages(websocket, exchange)
                    
            except (ConnectionClosed, WebSocketException) as e:
                self.logger.warning(f"Connection to {exchange} lost: {e}")
                await self._handle_reconnection(exchange)
            except Exception as e:
                self.logger.error(f"Error connecting to {exchange}: {e}")
                await self._handle_reconnection(exchange)
    
    def _get_websocket_url(self, exchange: str) -> Optional[str]:
        """Get WebSocket URL for exchange."""
        urls = {
            'polygon': 'wss://socket.polygon.io/crypto',
            'binance': 'wss://stream.binance.com:9443/ws/btcusdt@ticker',
            'coinbase': 'wss://ws-feed.exchange.coinbase.com'
        }
        return urls.get(exchange)
    
    async def _subscribe_symbols(self, websocket, exchange: str) -> None:
        """Subscribe to symbols on the exchange."""
        try:
            if exchange == 'polygon':
                # Polygon.io subscription
                # Get API key from environment variable
                api_key = os.getenv('POLYGON_API_KEY')
                if not api_key:
                    raise ValueError("POLYGON_API_KEY environment variable not set")

                auth_message = {
                    "action": "auth",
                    "params": api_key
                }
                await websocket.send(json.dumps(auth_message))
                
                # Subscribe to crypto streams
                subscribe_message = {
                    "action": "subscribe",
                    "params": [f"XT.{symbol.replace('-', '')}" for symbol in self.symbols]
                }
                await websocket.send(json.dumps(subscribe_message))
                
            elif exchange == 'binance':
                # Binance subscription (simplified)
                for symbol in self.symbols:
                    stream_name = f"{symbol.lower().replace('-', '')}@ticker"
                    subscribe_message = {
                        "method": "SUBSCRIBE",
                        "params": [stream_name],
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscribe_message))
                    
            elif exchange == 'coinbase':
                # Coinbase subscription
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": self.symbols,
                    "channels": ["ticker"]
                }
                await websocket.send(json.dumps(subscribe_message))
                
        except Exception as e:
            self.logger.error(f"Error subscribing to {exchange}: {e}")
            raise DataPipelineError(f"Failed to subscribe to {exchange}: {e}")
    
    async def _listen_messages(self, websocket, exchange: str) -> None:
        """Listen for messages from the exchange."""
        try:
            async for message in websocket:
                if not self.running:
                    break
                
                try:
                    data = json.loads(message)
                    market_data = self._parse_message(data, exchange)
                    
                    if market_data:
                        await self._process_market_data(market_data)
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from {exchange}: {e}")
                    self.metrics['failed_messages'] += 1
                except Exception as e:
                    self.logger.error(f"Error processing message from {exchange}: {e}")
                    self.metrics['failed_messages'] += 1
                
                self.metrics['total_messages'] += 1
                
        except Exception as e:
            self.logger.error(f"Error listening to {exchange}: {e}")
            raise
    
    def _parse_message(self, data: Dict[str, Any], exchange: str) -> Optional[MarketData]:
        """Parse WebSocket message into MarketData."""
        try:
            if exchange == 'polygon':
                return self._parse_polygon_message(data)
            elif exchange == 'binance':
                return self._parse_binance_message(data)
            elif exchange == 'coinbase':
                return self._parse_coinbase_message(data)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing {exchange} message: {e}")
            return None
    
    def _parse_polygon_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Polygon.io message."""
        if data.get('ev') == 'XT':  # Crypto ticker
            return MarketData(
                symbol=data.get('pair', '').replace('USD', '-USD'),
                timestamp=data.get('t', time.time()),
                price=float(data.get('p', 0)),
                volume=float(data.get('v', 0)),
                high=float(data.get('h', 0)) if data.get('h') else None,
                low=float(data.get('l', 0)) if data.get('l') else None,
                source='polygon'
            )
        return None
    
    def _parse_binance_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Binance message."""
        if 'c' in data:  # Ticker data
            return MarketData(
                symbol=data.get('s', '').replace('USDT', '-USD'),
                timestamp=data.get('E', time.time()) / 1000,
                price=float(data.get('c', 0)),
                volume=float(data.get('v', 0)),
                high=float(data.get('h', 0)) if data.get('h') else None,
                low=float(data.get('l', 0)) if data.get('l') else None,
                source='binance'
            )
        return None
    
    def _parse_coinbase_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Coinbase message."""
        if data.get('type') == 'ticker':
            return MarketData(
                symbol=data.get('product_id', ''),
                timestamp=time.time(),
                price=float(data.get('price', 0)),
                volume=float(data.get('volume_24h', 0)),
                bid=float(data.get('best_bid', 0)) if data.get('best_bid') else None,
                ask=float(data.get('best_ask', 0)) if data.get('best_ask') else None,
                source='coinbase'
            )
        return None
    
    async def _process_market_data(self, market_data: MarketData) -> None:
        """Process incoming market data."""
        try:
            # Update cache
            cache_key = f"market_data:{market_data.symbol}"
            self.cache.set(cache_key, market_data, ttl=60.0)
            
            # Add to buffer
            if market_data.symbol not in self.data_buffer:
                self.data_buffer[market_data.symbol] = []
            
            self.data_buffer[market_data.symbol].append(market_data)
            
            # Keep only last 1000 entries per symbol
            if len(self.data_buffer[market_data.symbol]) > 1000:
                self.data_buffer[market_data.symbol] = self.data_buffer[market_data.symbol][-1000:]
            
            # Call registered callbacks
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
            
            # Update metrics
            self.metrics['successful_messages'] += 1
            self.metrics['last_update'] = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            self.metrics['failed_messages'] += 1
    
    async def _handle_reconnection(self, exchange: str) -> None:
        """Handle reconnection logic."""
        self.reconnect_attempts[exchange] += 1
        self.metrics['reconnections'] += 1
        
        if self.reconnect_attempts[exchange] >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts reached for {exchange}")
            return
        
        self.logger.info(f"Reconnecting to {exchange} in {self.reconnect_delay}s...")
        await asyncio.sleep(self.reconnect_delay)
        
        # Reconnect
        if self.running:
            asyncio.create_task(self._connect_exchange(exchange))
    
    def add_data_callback(self, callback: Callable[[MarketData], None]) -> None:
        """Add a callback for market data updates."""
        self.data_callbacks.append(callback)
    
    @lru_cache(maxsize=128)
    
    def get_latest_data(self, symbol: Optional[str] = None) -> Dict[str, MarketData]:
        """Get latest market data from cache."""
        if symbol:
            cache_key = f"market_data:{symbol}"
            data = self.cache.get(cache_key)
            return {symbol: data} if data else {}
        else:
            result = {}
            for sym in self.symbols:
                cache_key = f"market_data:{sym}"
                data = self.cache.get(cache_key)
                if data:
                    result[sym] = data
            return result
    
    @lru_cache(maxsize=128)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            'running': self.running,
            'connections': len(self.connections),
            'exchanges': list(self.connections.keys()),
            'symbols': self.symbols,
            'metrics': self.metrics
        }


# Global WebSocket data source instance
_websocket_source: Optional[WebSocketDataSource] = None


async def get_websocket_source() -> WebSocketDataSource:
    """Get the global WebSocket data source instance."""
    global _websocket_source
    
    if _websocket_source is None:
        from src.core.config.loader import config_get
        
        symbols = config_get('data_pipeline.websocket.symbols', ['BTC-USD', 'ETH-USD'])
        exchanges = config_get('data_pipeline.websocket.exchanges', ['polygon', 'binance'])
        
        _websocket_source = WebSocketDataSource(
            symbols=symbols,
            exchanges=exchanges
        )
        await _websocket_source.start()
    
    return _websocket_source


async def close_websocket_source() -> None:
    """Close the global WebSocket data source instance."""
    global _websocket_source
    
    if _websocket_source:
        await _websocket_source.stop()
        _websocket_source = None


__all__ = [
    'WebSocketDataSource',
    'MarketData',
    'get_websocket_source',
    'close_websocket_source',
]