from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from polygon import RESTClient
from polygon.websocket import WebSocketClient, Market
from src.core.audit_logger import get_audit_logger
from src.core.constants import TradingConstants, APIConstants
from src.core.database.optimized_db import OptimizedDatabase
from src.core.rate_limiter import get_rate_limiter
from src.core.validation import MarketDataInput
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import asyncio
import json
import logging
import os
import time
import websockets


"""
Polygon.io Live Feed Integration
=================================

Real-time market data feed from Polygon.io for cryptocurrency trading.
Supports REST API for historical data and WebSocket for real-time updates.

Features:
- Real-time price updates
- Historical data fetching
- Automatic reconnection
- Rate limiting compliance
- Data validation

Author: RRR Ventures
Date: 2025-10-12
"""




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Structured market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    def to_ohlcv(self) -> Dict[str, float]:
        """Convert to OHLCV format"""
        return {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp.timestamp()
        }


class PolygonLiveFeed:
    """
    Live market data feed from Polygon.io.
    
    Combines REST API for historical data and WebSocket for real-time updates.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        use_sandbox: bool = False
    ):
        """
        Initialize Polygon feed.
        
        Args:
            api_key: Polygon API key (or from env)
            symbols: List of symbols to track
            use_sandbox: Use sandbox environment for testing
        """
        # Get API key
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. "
                "Set POLYGON_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize REST client (simplified for compatibility)
        self.rest_client = RESTClient(api_key=self.api_key)
        
        # Symbol tracking
        self.symbols = symbols or ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
        self.symbol_map = self._create_symbol_map()
        
        # WebSocket client (will be initialized when started)
        self.ws_client = None
        self.ws_running = False
        
        # Data storage
        self.latest_prices: Dict[str, float] = {}
        self.latest_data: Dict[str, MarketDataPoint] = {}
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'trade': [],
            'quote': [],
            'aggregate': [],
            'error': []
        }
        
        # Rate limiter
        self.rate_limiter = get_rate_limiter('polygon')
        
        # Audit logger
        self.audit_logger = get_audit_logger()
        
        # Database
        self.db = OptimizedDatabase()
        
        logger.info(f"PolygonLiveFeed initialized with {len(self.symbols)} symbols")
    
    def _create_symbol_map(self) -> Dict[str, str]:
        """Create mapping between internal and Polygon symbols"""
        mapping = {}
        for symbol in self.symbols:
            # Convert from internal format (BTC-USD) to Polygon (X:BTCUSD)
            if '-' in symbol:
                base, quote = symbol.split('-')
                polygon_symbol = f"X:{base}{quote}"
            else:
                polygon_symbol = symbol
            
            mapping[symbol] = polygon_symbol
            mapping[polygon_symbol] = symbol
        
        return mapping
    
    @lru_cache(maxsize=128)
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timespan: str = 'minute',
        multiplier: int = 1
    ) -> List[MarketDataPoint]:
        """
        Get historical data from Polygon REST API.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date (default: now)
            timespan: minute, hour, day, week, month
            multiplier: Timespan multiplier (e.g., 5 for 5-minute bars)
            
        Returns:
            List of market data points
        """
        polygon_symbol = self.symbol_map.get(symbol, symbol)
        end_date = end_date or datetime.now(timezone.utc)
        
        # Rate limit check
        with self.rate_limiter:
            # Log API call
            self.audit_logger.log_api_call(
                api_name='Polygon',
                endpoint=f'/v2/aggs/ticker/{polygon_symbol}',
                method='GET'
            )
            
            try:
                # Fetch aggregates
                aggs = self.rest_client.get_aggs(
                    ticker=polygon_symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    adjusted=True,
                    sort='asc',
                    limit=50000
                )
                
                # Convert to MarketDataPoint
                data_points = []
                for agg in aggs:
                    point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                        open=agg.open,
                        high=agg.high,
                        low=agg.low,
                        close=agg.close,
                        volume=agg.volume,
                        vwap=agg.vwap if hasattr(agg, 'vwap') else None,
                        trades=agg.transactions if hasattr(agg, 'transactions') else None
                    )
                    data_points.append(point)
                
                logger.info(f"Fetched {len(data_points)} historical data points for {symbol}")
                return data_points
                
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                self.audit_logger.log_api_call(
                    api_name='Polygon',
                    endpoint=f'/v2/aggs/ticker/{polygon_symbol}',
                    method='GET',
                    error=str(e)
                )
                return []
    
    @lru_cache(maxsize=128)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price or None
        """
        return self.latest_prices.get(symbol)
    
    @lru_cache(maxsize=128)
    
    def get_latest_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get latest OHLCV data for symbols.
        
        Args:
            symbols: List of symbols (default: all tracked)
            
        Returns:
            Dictionary of symbol -> OHLCV data
        """
        symbols = symbols or self.symbols
        result = {}
        
        for symbol in symbols:
            if symbol in self.latest_data:
                result[symbol] = self.latest_data[symbol].to_ohlcv()
            else:
                # Fetch latest if not available
                try:
                    ticker = self.rest_client.get_previous_close_agg(
                        ticker=self.symbol_map.get(symbol, symbol)
                    )
                    
                    if ticker:
                        result[symbol] = {
                            'open': ticker[0].open,
                            'high': ticker[0].high,
                            'low': ticker[0].low,
                            'close': ticker[0].close,
                            'volume': ticker[0].volume,
                            'timestamp': ticker[0].timestamp / 1000
                        }
                except Exception as e:
                    logger.error(f"Error fetching latest data for {symbol}: {e}")
        
        return result
    
    async def start_websocket(self):
        """Start WebSocket connection for real-time updates"""
        if self.ws_running:
            logger.warning("WebSocket already running")
            return
        
        self.ws_running = True
        
        # Create WebSocket client
        def handle_msg(msg: List):
            """Handle incoming WebSocket messages"""
            try:
                for item in msg:
                    if item['ev'] == 'XA':  # Crypto aggregate
                        self._handle_aggregate(item)
                    elif item['ev'] == 'XT':  # Crypto trade
                        self._handle_trade(item)
                    elif item['ev'] == 'XQ':  # Crypto quote
                        self._handle_quote(item)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
        
        # Initialize client
        self.ws_client = WebSocketClient(
            api_key=self.api_key,
            market=Market.Crypto,
            on_message=handle_msg,
            on_error=lambda e: logger.error(f"WebSocket error: {e}"),
            on_close=lambda: logger.info("WebSocket closed")
        )
        
        # Subscribe to symbols
        for symbol in self.symbols:
            polygon_symbol = self.symbol_map.get(symbol, symbol)
            
            # Subscribe to trades and aggregates
            self.ws_client.subscribe(f"XT.{polygon_symbol}")  # Trades
            self.ws_client.subscribe(f"XA.{polygon_symbol}")  # Aggregates per second
            self.ws_client.subscribe(f"XQ.{polygon_symbol}")  # Quotes
        
        # Connect
        await self.ws_client.connect()
        
        logger.info(f"WebSocket connected, subscribed to {len(self.symbols)} symbols")
    
    def _handle_aggregate(self, msg: Dict[str, Any]):
        """Handle aggregate (bar) message"""
        try:
            symbol = self.symbol_map.get(msg['pair'], msg['pair'])
            
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(msg['s'] / 1000, tz=timezone.utc),
                open=msg['o'],
                high=msg['h'],
                low=msg['l'],
                close=msg['c'],
                volume=msg['v'],
                vwap=msg.get('vw')
            )
            
            # Update latest data
            self.latest_data[symbol] = data_point
            self.latest_prices[symbol] = data_point.close
            
            # Store in database
            self.db.insert_market_data(
                symbol=symbol,
                timestamp=data_point.timestamp.timestamp(),
                ohlcv=data_point.to_ohlcv()
            )
            
            # Trigger callbacks
            for callback in self.callbacks.get('aggregate', []):
                callback(data_point)
                
        except Exception as e:
            logger.error(f"Error handling aggregate: {e}")
    
    def _handle_trade(self, msg: Dict[str, Any]):
        """Handle trade message"""
        try:
            symbol = self.symbol_map.get(msg['pair'], msg['pair'])
            
            # Update latest price
            self.latest_prices[symbol] = msg['p']
            
            # Trigger callbacks
            for callback in self.callbacks.get('trade', []):
                callback({
                    'symbol': symbol,
                    'price': msg['p'],
                    'size': msg['s'],
                    'timestamp': msg['t'] / 1000
                })
                
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    def _handle_quote(self, msg: Dict[str, Any]):
        """Handle quote message"""
        try:
            symbol = self.symbol_map.get(msg['pair'], msg['pair'])
            
            # Trigger callbacks
            for callback in self.callbacks.get('quote', []):
                callback({
                    'symbol': symbol,
                    'bid': msg['bp'],
                    'ask': msg['ap'],
                    'bid_size': msg['bs'],
                    'ask_size': msg['as'],
                    'timestamp': msg['t'] / 1000
                })
                
        except Exception as e:
            logger.error(f"Error handling quote: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add callback for event type.
        
        Args:
            event_type: 'trade', 'quote', 'aggregate', or 'error'
            callback: Function to call with event data
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.ws_client:
            await self.ws_client.close()
            self.ws_running = False
            logger.info("WebSocket stopped")
    
    def test_connection(self) -> bool:
        """
        Test Polygon API connection.
        
        Returns:
            True if connection successful
        """
        try:
            # Try to get market status
            status = self.rest_client.get_market_status()
            
            logger.info(f"Polygon connection test successful. Market: {status.market}")
            return True
            
        except Exception as e:
            logger.error(f"Polygon connection test failed: {e}")
            return False


# Example usage and testing
async def main():
    """Test Polygon live feed"""
    
    # Initialize feed
    feed = PolygonLiveFeed(
        symbols=['BTC-USD', 'ETH-USD']
    )
    
    # Test connection
    if not feed.test_connection():
        logger.error("Failed to connect to Polygon")
        return
    
    # Get historical data
    historical = feed.get_historical_data(
        symbol='BTC-USD',
        start_date=datetime.now() - timedelta(days=1),
        timespan='hour'
    )
    
    if historical:
        logger.info(f"Got {len(historical)} historical data points")
        latest = historical[-1]
        logger.info(f"Latest: {latest.close} at {latest.timestamp}")
    
    # Get latest prices
    latest_data = feed.get_latest_data()
    for symbol, data in latest_data.items():
        logger.info(f"{symbol}: ${data['close']:.2f}")
    
    # Add callbacks
    feed.add_callback('aggregate', lambda d: logger.info(f"New bar: {d.symbol} ${d.close}"))
    feed.add_callback('trade', lambda t: logger.info(f"Trade: {t['symbol']} ${t['price']}"))
    
    # Start WebSocket (optional - requires valid subscription)
    try:
        await feed.start_websocket()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop
        await feed.stop_websocket()
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.info("Note: Real-time WebSocket requires paid Polygon subscription")


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    'PolygonLiveFeed',
    'MarketDataPoint'
]