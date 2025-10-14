from ..base import BaseDataCollector
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from functools import lru_cache
from typing import Dict, List, Optional, Any, Callable
from websockets.client import WebSocketClientProtocol
import asyncio
import json
import logging
import numpy as np
import os
import pandas as pd
import websockets


"""
Enhanced Polygon.io collector with order flow analytics and microstructure metrics
"""



load_dotenv('config/api-keys/.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Enhanced tick data with microstructure features"""
    timestamp: int  # Unix timestamp in milliseconds
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    exchange: int
    conditions: List[int]
    
    # Derived features
    trade_sign: int = 0  # +1 for buy, -1 for sell
    dollar_volume: float = 0.0
    
    def __post_init__(self):
        self.dollar_volume = self.price * self.size
        # Use tick rule to determine trade direction
        self.trade_sign = 1 if self.side == 'buy' else -1


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with depth metrics"""
    timestamp: int
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    
    # Derived metrics
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    mid_price: float = 0.0
    
    # Depth metrics
    bid_depth_5: float = 0.0  # Total size in top 5 bid levels
    ask_depth_5: float = 0.0
    depth_imbalance: float = 0.0  # (bid - ask) / (bid + ask)
    
    def __post_init__(self):
        if self.bids and self.asks:
            self.bid_price = self.bids[0][0]
            self.ask_price = self.asks[0][0]
            self.spread = self.ask_price - self.bid_price
            self.mid_price = (self.bid_price + self.ask_price) / 2
            self.spread_bps = (self.spread / self.mid_price) * 10000
            
            # Calculate depth
            self.bid_depth_5 = sum(size for _, size in self.bids[:5])
            self.ask_depth_5 = sum(size for _, size in self.asks[:5])
            
            total_depth = self.bid_depth_5 + self.ask_depth_5
            if total_depth > 0:
                self.depth_imbalance = (self.bid_depth_5 - self.ask_depth_5) / total_depth


@dataclass
class MicrostructureMetrics:
    """Microstructure metrics calculated from tick data"""
    symbol: str
    timestamp: datetime
    window_seconds: int
    
    # Flow metrics
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    order_flow_imbalance: float = 0.0  # (buy - sell) / total
    
    # VPIN (Volume-synchronized Probability of Informed Trading)
    vpin: float = 0.0
    
    # Price impact
    kyle_lambda: float = 0.0  # Price impact coefficient
    amihud_illiquidity: float = 0.0
    
    # Spread metrics
    effective_spread: float = 0.0
    realized_spread: float = 0.0
    
    # Trade intensity
    trade_count: int = 0
    trades_per_second: float = 0.0
    avg_trade_size: float = 0.0
    
    # Volatility
    price_volatility: float = 0.0
    returns_std: float = 0.0


class EnhancedPolygonCollector(BaseDataCollector):
    """
    Enhanced Polygon.io WebSocket client with order flow analytics
    
    Features:
    - Real-time tick data with microsecond timestamps
    - Order book depth snapshots
    - Order flow imbalance calculation
    - VPIN (toxicity) estimation
    - Kyle's Lambda (price impact)
    - Multi-asset support
    """
    
    WEBSOCKET_URL = "wss://socket.polygon.io/crypto"
    
    def __init__(self, api_key: Optional[str] = None, buffer_size: int = 10000):
        super().__init__("EnhancedPolygonCollector")
        
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key required")
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.subscriptions: List[str] = []
        
        # Data buffers
        self.tick_buffer: Dict[str, deque] = {}  # symbol -> deque of TickData
        self.orderbook_buffer: Dict[str, deque] = {}  # symbol -> deque of OrderBookSnapshot
        self.buffer_size = buffer_size
        
        # Callbacks
        self.tick_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'ticks_received': 0,
            'quotes_received': 0,
            'trades_per_second': {},
            'current_prices': {}
        }
    
    async def connect(self) -> bool:
        """Connect to Polygon WebSocket"""
        try:
            logger.info(f"Connecting to {self.WEBSOCKET_URL}...")
            self.websocket = await websockets.connect(
                self.WEBSOCKET_URL,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await self.websocket.send(json.dumps(auth_msg))
            
            # Wait for auth response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if isinstance(data, list) and len(data) > 0:
                if data[0].get('status') in ['connected', 'auth_success']:
                    logger.info("✅ Connected and authenticated to Polygon.io")
                    return True
            
            logger.error("Authentication failed")
            return False
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def subscribe(self, symbols: List[str], channels: Optional[List[str]] = None) -> bool:
        """
        Subscribe to market data
        
        Args:
            symbols: List of symbols (e.g., ['BTC-USD', 'ETH-USD'])
            channels: List of channels ['XT' (trades), 'XQ' (quotes), 'XA' (aggregates)]
        """
        if channels is None:
            channels = ['XT', 'XQ']  # Trades and quotes for order flow
        
        subscriptions = []
        for channel in channels:
            for symbol in symbols:
                sub = f"{channel}.{symbol}"
                subscriptions.append(sub)
                self.subscriptions.append(sub)
                
                # Initialize buffers
                if symbol not in self.tick_buffer:
                    self.tick_buffer[symbol] = deque(maxlen=self.buffer_size)
                if symbol not in self.orderbook_buffer:
                    self.orderbook_buffer[symbol] = deque(maxlen=1000)
        
        if self.websocket:
            sub_msg = {"action": "subscribe", "params": ",".join(subscriptions)}
            await self.websocket.send(json.dumps(sub_msg))
            logger.info(f"📡 Subscribed to {len(subscriptions)} channels")
            return True
        
        return False
    
    def parse_trade(self, data: Dict[str, Any]) -> TickData:
        """Parse trade message into TickData"""
        # Determine side using tick rule (buy if at ask, sell if at bid)
        side = 'buy' if data.get('c', []) else 'sell'  # Simplified
        
        return TickData(
            timestamp=int(data.get('t', 0)),
            symbol=data.get('pair', ''),
            price=float(data.get('p', 0)),
            size=float(data.get('s', 0)),
            side=side,
            exchange=int(data.get('x', 0)),
            conditions=data.get('c', [])
        )
    
    def parse_quote(self, data: Dict[str, Any]) -> OrderBookSnapshot:
        """Parse quote message into OrderBookSnapshot"""
        bid_price = float(data.get('bp', 0))
        bid_size = float(data.get('bs', 0))
        ask_price = float(data.get('ap', 0))
        ask_size = float(data.get('as', 0))
        
        return OrderBookSnapshot(
            timestamp=int(data.get('t', 0)),
            symbol=data.get('pair', ''),
            bids=[(bid_price, bid_size)] if bid_price > 0 else [],
            asks=[(ask_price, ask_size)] if ask_price > 0 else []
        )
    
    async def handle_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            data_list = json.loads(message)
            
            for data in data_list:
                event_type = data.get('ev')
                
                if event_type == 'XT':  # Crypto Trade
                    tick = self.parse_trade(data)
                    self.tick_buffer[tick.symbol].append(tick)
                    self.stats['ticks_received'] += 1
                    self.stats['current_prices'][tick.symbol] = tick.price
                    
                    # Call callbacks
                    for callback in self.tick_callbacks:
                        await callback(tick)
                
                elif event_type == 'XQ':  # Crypto Quote
                    snapshot = self.parse_quote(data)
                    self.orderbook_buffer[snapshot.symbol].append(snapshot)
                    self.stats['quotes_received'] += 1
                    
                    # Call callbacks
                    for callback in self.orderbook_callbacks:
                        await callback(snapshot)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def collect(self) -> pd.DataFrame:
        """Collect data from buffer and return as DataFrame"""
        all_ticks = []
        for symbol, buffer in self.tick_buffer.items():
            all_ticks.extend(buffer)
        
        if not all_ticks:
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(tick) for tick in all_ticks])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_microstructure_metrics(self, symbol: str, window_seconds: int = 60) -> MicrostructureMetrics:
        """
        Calculate microstructure metrics for a symbol
        
        Args:
            symbol: Symbol to analyze
            window_seconds: Time window for calculation
            
        Returns:
            MicrostructureMetrics object
        """
        if symbol not in self.tick_buffer:
            return None
        
        ticks = list(self.tick_buffer[symbol])
        if len(ticks) < 10:
            return None
        
        # Filter to time window
        now = datetime.now(timezone.utc).timestamp() * 1000
        window_start = now - (window_seconds * 1000)
        recent_ticks = [t for t in ticks if t.timestamp >= window_start]
        
        if len(recent_ticks) < 5:
            return None
        
        # Calculate flow metrics
        buy_volume = sum(t.dollar_volume for t in recent_ticks if t.trade_sign == 1)
        sell_volume = sum(t.dollar_volume for t in recent_ticks if t.trade_sign == -1)
        total_volume = buy_volume + sell_volume
        
        ofi = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate VPIN (simplified version)
        vpin = self._calculate_vpin(recent_ticks)
        
        # Calculate Kyle's Lambda (price impact)
        kyle_lambda = self._calculate_kyle_lambda(recent_ticks)
        
        # Calculate Amihud illiquidity
        amihud = self._calculate_amihud_illiquidity(recent_ticks)
        
        # Trade statistics
        prices = np.array([t.price for t in recent_ticks])
        returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([])
        
        metrics = MicrostructureMetrics(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            window_seconds=window_seconds,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            order_flow_imbalance=ofi,
            vpin=vpin,
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud,
            trade_count=len(recent_ticks),
            trades_per_second=len(recent_ticks) / window_seconds,
            avg_trade_size=np.mean([t.size for t in recent_ticks]),
            price_volatility=np.std(prices) if len(prices) > 1 else 0,
            returns_std=np.std(returns) if len(returns) > 0 else 0
        )
        
        return metrics
    
    def _calculate_vpin(self, ticks: List[TickData], n_buckets: int = 50) -> float:
        """
        Calculate VPIN (Volume-synchronized Probability of Informed Trading)
        
        VPIN measures order flow toxicity - high values indicate informed trading
        """
        if len(ticks) < n_buckets:
            return 0.0
        
        # Calculate volume per bucket
        total_volume = sum(t.dollar_volume for t in ticks)
        bucket_volume = total_volume / n_buckets
        
        # Bucket trades by volume
        buckets = []
        current_bucket_buy = 0
        current_bucket_sell = 0
        current_bucket_vol = 0
        
        for tick in ticks:
            if tick.trade_sign == 1:
                current_bucket_buy += tick.dollar_volume
            else:
                current_bucket_sell += tick.dollar_volume
            
            current_bucket_vol += tick.dollar_volume
            
            if current_bucket_vol >= bucket_volume:
                imbalance = abs(current_bucket_buy - current_bucket_sell) / current_bucket_vol
                buckets.append(imbalance)
                current_bucket_buy = 0
                current_bucket_sell = 0
                current_bucket_vol = 0
        
        # VPIN is average absolute order imbalance
        return np.mean(buckets) if buckets else 0.0
    
    def _calculate_kyle_lambda(self, ticks: List[TickData]) -> float:
        """
        Calculate Kyle's Lambda (price impact coefficient)
        
        Lambda = Cov(ΔP, Q) / Var(Q)
        where ΔP is price change and Q is signed volume
        """
        if len(ticks) < 10:
            return 0.0
        
        prices = np.array([t.price for t in ticks])
        volumes = np.array([t.size * t.trade_sign for t in ticks])
        
        if len(prices) < 2:
            return 0.0
        
        price_changes = np.diff(prices)
        volumes = volumes[1:]  # Align with price changes
        
        if len(price_changes) != len(volumes):
            return 0.0
        
        if np.var(volumes) == 0:
            return 0.0
        
        lambda_coef = np.cov(price_changes, volumes)[0, 1] / np.var(volumes)
        return abs(lambda_coef)
    
    def _calculate_amihud_illiquidity(self, ticks: List[TickData]) -> float:
        """
        Calculate Amihud illiquidity measure
        
        Illiquidity = |return| / dollar_volume
        """
        if len(ticks) < 2:
            return 0.0
        
        prices = np.array([t.price for t in ticks])
        volumes = np.array([t.dollar_volume for t in ticks])
        
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(np.log(prices))
        volumes = volumes[1:]  # Align
        
        if len(returns) == 0 or np.sum(volumes) == 0:
            return 0.0
        
        illiquidity = np.mean(np.abs(returns) / (volumes + 1e-10))
        return illiquidity
    
    @lru_cache(maxsize=128)
    
    def get_order_flow_imbalance(self, symbol: str, window_seconds: int = 10) -> float:
        """Get current order flow imbalance for a symbol"""
        metrics = self.calculate_microstructure_metrics(symbol, window_seconds)
        return metrics.order_flow_imbalance if metrics else 0.0
    
    @lru_cache(maxsize=128)
    
    def get_vpin(self, symbol: str) -> float:
        """Get current VPIN (toxicity) for a symbol"""
        metrics = self.calculate_microstructure_metrics(symbol, 60)
        return metrics.vpin if metrics else 0.0
    
    async def run(self):
        """Main event loop"""
        self.is_running = True
        
        while self.is_running:
            try:
                if not self.websocket:
                    connected = await self.connect()
                    if not connected:
                        await asyncio.sleep(5)
                        continue
                    
                    # Re-subscribe
                    if self.subscriptions:
                        symbols = list(set(sub.split('.')[1] for sub in self.subscriptions))
                        channels = list(set(sub.split('.')[0] for sub in self.subscriptions))
                        await self.subscribe(symbols, channels)
                
                # Receive messages
                message = await self.websocket.recv()
                await self.handle_message(message)
                
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                self.websocket = None
                await asyncio.sleep(1)
        
        if self.websocket:
            await self.websocket.close()
    
    def register_tick_callback(self, callback: Callable):
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
    
    def register_orderbook_callback(self, callback: Callable):
        """Register callback for order book data"""
        self.orderbook_callbacks.append(callback)

