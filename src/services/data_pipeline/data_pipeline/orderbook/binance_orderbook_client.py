from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional, Callable
import asyncio
import json
import logging
import requests
import websockets


"""
Binance Order Book WebSocket Client

Real-time order book monitoring using Binance WebSocket streams.
Maintains local order book state and calculates depth metrics.
"""


logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time"""
    symbol: str
    timestamp: datetime
    bids: List[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: List[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    
    # Calculated metrics
    mid_price: Decimal = field(init=False)
    spread_bps: Decimal = field(init=False)
    bid_depth_1pct: Decimal = field(init=False)
    ask_depth_1pct: Decimal = field(init=False)
    bid_ask_ratio: Decimal = field(init=False)
    depth_imbalance: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            
            # Mid price
            self.mid_price = (best_bid + best_ask) / 2
            
            # Spread in basis points
            self.spread_bps = ((best_ask - best_bid) / self.mid_price) * 10000
            
            # Calculate depth within 1% of mid price
            price_1pct_up = self.mid_price * Decimal('1.01')
            price_1pct_down = self.mid_price * Decimal('0.99')
            
            self.bid_depth_1pct = sum(
                qty for price, qty in self.bids 
                if price >= price_1pct_down
            )
            
            self.ask_depth_1pct = sum(
                qty for price, qty in self.asks 
                if price <= price_1pct_up
            )
            
            # Bid/ask ratio and imbalance
            if self.ask_depth_1pct > 0:
                self.bid_ask_ratio = self.bid_depth_1pct / self.ask_depth_1pct
            else:
                self.bid_ask_ratio = Decimal('999')  # Cap at very high value
            
            total_depth = self.bid_depth_1pct + self.ask_depth_1pct
            if total_depth > 0:
                self.depth_imbalance = (self.bid_depth_1pct - self.ask_depth_1pct) / total_depth
            else:
                self.depth_imbalance = Decimal('0')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'mid_price': float(self.mid_price),
            'spread_bps': float(self.spread_bps),
            'bid_depth_1pct': float(self.bid_depth_1pct),
            'ask_depth_1pct': float(self.ask_depth_1pct),
            'bid_ask_ratio': float(self.bid_ask_ratio),
            'depth_imbalance': float(self.depth_imbalance)
        }


class BinanceOrderBookClient:
    """
    Real-time Binance order book monitoring via WebSocket.
    
    Maintains local order book state and provides snapshots on demand.
    Calculates key microstructure metrics for trading signals.
    """
    
    BASE_WSS_URL = "wss://stream.binance.com:9443/ws"
    REST_API_URL = "https://api.binance.com/api/v3"
    
    def __init__(
        self,
        symbols: List[str] = None,
        update_speed: str = "100ms",  # "100ms" or "1000ms"
        on_snapshot: Optional[Callable[[OrderBookSnapshot], None]] = None
    ):
        """
        Initialize Binance order book client.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            update_speed: Update frequency ("100ms" or "1000ms")
            on_snapshot: Callback function for new snapshots
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.update_speed = update_speed
        self.on_snapshot = on_snapshot
        
        # Order book state for each symbol
        self.order_books: Dict[str, Dict] = {
            symbol: {'bids': {}, 'asks': {}, 'last_update_id': 0}
            for symbol in self.symbols
        }
        
        # WebSocket connection
        self.ws = None
        self.running = False
        
        # Statistics
        self.update_count = 0
        self.error_count = 0
    
    async def initialize_order_books(self):
        """Fetch initial order book snapshots via REST API"""
        for symbol in self.symbols:
            try:
                url = f"{self.REST_API_URL}/depth"
                params = {'symbol': symbol, 'limit': 1000}
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Initialize order book
                self.order_books[symbol]['bids'] = {
                    Decimal(price): Decimal(qty) 
                    for price, qty in data['bids']
                }
                self.order_books[symbol]['asks'] = {
                    Decimal(price): Decimal(qty) 
                    for price, qty in data['asks']
                }
                self.order_books[symbol]['last_update_id'] = data['lastUpdateId']
                
                logger.info(f"Initialized order book for {symbol}: "
                           f"{len(data['bids'])} bids, {len(data['asks'])} asks")
                
            except Exception as e:
                logger.error(f"Failed to initialize {symbol} order book: {e}")
                self.error_count += 1
    
    def _update_order_book(self, symbol: str, bids: List, asks: List):
        """Update local order book with new bids/asks"""
        book = self.order_books[symbol]
        
        # Update bids
        for price_str, qty_str in bids:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            
            if qty == 0:
                # Remove price level
                book['bids'].pop(price, None)
            else:
                # Update price level
                book['bids'][price] = qty
        
        # Update asks
        for price_str, qty_str in asks:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            
            if qty == 0:
                # Remove price level
                book['asks'].pop(price, None)
            else:
                # Update price level
                book['asks'][price] = qty
    
    @lru_cache(maxsize=128)
    
    def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        """Get current order book snapshot with metrics"""
        book = self.order_books[symbol]
        
        # Sort and get top levels
        sorted_bids = sorted(book['bids'].items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(book['asks'].items(), key=lambda x: x[0])
        
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted_bids[:100],  # Top 100 levels
            asks=sorted_asks[:100]
        )
        
        return snapshot
    
    async def handle_message(self, message: Dict):
        """Handle incoming WebSocket message"""
        try:
            # Check if this is a depth update
            if 'e' in message and message['e'] == 'depthUpdate':
                symbol = message['s']
                
                # Update order book
                self._update_order_book(
                    symbol=symbol,
                    bids=message['b'],
                    asks=message['a']
                )
                
                self.update_count += 1
                
                # Generate snapshot and call callback
                if self.on_snapshot and self.update_count % 50 == 0:  # Every 5 seconds at 100ms
                    snapshot = self.get_snapshot(symbol)
                    self.on_snapshot(snapshot)
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.error_count += 1
    
    async def run(self):
        """Main WebSocket loop"""
        # Initialize order books from REST API
        await self.initialize_order_books()
        
        # Build WebSocket URL
        streams = [f"{symbol.lower()}@depth@{self.update_speed}" for symbol in self.symbols]
        stream_names = '/'.join(streams)
        ws_url = f"{self.BASE_WSS_URL}/{stream_names}"
        
        logger.info(f"Connecting to Binance WebSocket: {ws_url}")
        
        self.running = True
        retry_count = 0
        max_retries = 10
        
        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws = websocket
                    retry_count = 0  # Reset on successful connection
                    logger.info("Connected to Binance WebSocket")
                    
                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await self.handle_message(data)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                retry_count += 1
                await asyncio.sleep(min(retry_count * 2, 30))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                retry_count += 1
                await asyncio.sleep(min(retry_count * 2, 30))
        
        if retry_count >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded, stopping")
        
        self.running = False
    
    async def stop(self):
        """Stop the WebSocket client"""
        logger.info("Stopping Binance order book client...")
        self.running = False
        if self.ws:
            await self.ws.close()
    
    @lru_cache(maxsize=128)
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            'symbols': self.symbols,
            'update_count': self.update_count,
            'error_count': self.error_count,
            'running': self.running,
            'order_book_sizes': {
                symbol: {
                    'bids': len(book['bids']),
                    'asks': len(book['asks'])
                }
                for symbol, book in self.order_books.items()
            }
        }


# Example usage and testing
async def main():
    """Test the order book client"""
    print("="*80)
    print("Binance Order Book Client Demo")
    print("="*80)
    print()
    
    # Callback to print snapshots
    def print_snapshot(snapshot: OrderBookSnapshot):
        print(f"\n📊 {snapshot.symbol} @ {snapshot.timestamp.strftime('%H:%M:%S')}")
        print(f"   Mid Price: ${snapshot.mid_price:,.2f}")
        print(f"   Spread: {snapshot.spread_bps:.2f} bps")
        print(f"   Bid Depth (1%): {snapshot.bid_depth_1pct:,.4f}")
        print(f"   Ask Depth (1%): {snapshot.ask_depth_1pct:,.4f}")
        print(f"   Bid/Ask Ratio: {snapshot.bid_ask_ratio:.3f}:1")
        print(f"   Imbalance: {snapshot.depth_imbalance:+.3f}")
        
        if snapshot.bid_ask_ratio > 2.0:
            print("   🟢 BULLISH IMBALANCE")
        elif snapshot.bid_ask_ratio < 0.5:
            print("   🔴 BEARISH IMBALANCE")
    
    # Create client
    client = BinanceOrderBookClient(
        symbols=['BTCUSDT', 'ETHUSDT'],
        update_speed="100ms",
        on_snapshot=print_snapshot
    )
    
    # Run for 30 seconds
    try:
        print("Starting order book monitoring (30 seconds)...")
        print("Press Ctrl+C to stop early")
        print()
        
        run_task = asyncio.create_task(client.run())
        await asyncio.sleep(30)
        await client.stop()
        await run_task
        
    except KeyboardInterrupt:
        print("\n\nStopping...")
        await client.stop()
    
    # Print statistics
    stats = client.get_stats()
    print("\n" + "="*80)
    print("Statistics:")
    print(f"  Updates received: {stats['update_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Order book sizes:")
    for symbol, sizes in stats['order_book_sizes'].items():
        print(f"    {symbol}: {sizes['bids']} bids, {sizes['asks']} asks")
    print("="*80)
    print("\n✅ Order book client demo complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())


