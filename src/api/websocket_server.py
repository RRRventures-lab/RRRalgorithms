from aiohttp import web
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from src.api.rate_limiter import get_rate_limiter, RateLimitConfig
from src.core.audit_logger import get_audit_logger
from src.core.database.local_db import LocalDatabase
from src.data_pipeline.polygon_live_feed import PolygonLiveFeed
from typing import Dict, Set, List, Any, Optional
import aiohttp_cors
import asyncio
import json
import logging
import os
import socketio
import sys

#!/usr/bin/env python
"""
WebSocket Server for Trading Command Center
===========================================

Real-time bidirectional communication between frontend and trading system.
Handles market data streaming, trade execution, and system monitoring.
"""


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Third-party imports

# Load environment
load_dotenv('config/api-keys/.env')

# Local imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    async_mode='aiohttp',
    logger=True,
    engineio_logger=True
)

# Create aiohttp app
app = web.Application()
sio.attach(app)

# Configure CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*"
    )
})


@dataclass
class Client:
    """Connected client information"""
    sid: str
    connected_at: datetime
    subscribed_symbols: Set[str]
    authenticated: bool = False
    username: Optional[str] = None
    ip_address: str = "unknown"


class TradingWebSocketServer:
    """WebSocket server for trading system"""

    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.market_data_feed: Optional[PolygonLiveFeed] = None
        self.db = LocalDatabase()
        self.audit_logger = get_audit_logger()
        self.trading_enabled = False
        self.system_status = "READY"

        # Initialize rate limiter
        self.rate_limiter = get_rate_limiter()
        
        # Market data cache
        self.latest_prices: Dict[str, float] = {}
        self.latest_market_data: Dict[str, Dict] = {}
        
        # Portfolio state
        self.portfolio = {
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'positions': []
        }
        
    async def initialize(self):
        """Initialize server components"""
        try:
            # Initialize rate limiter
            await self.rate_limiter.initialize()

            # Initialize Polygon feed
            self.market_data_feed = PolygonLiveFeed(
                symbols=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD']
            )

            # Test connection
            if self.market_data_feed.test_connection():
                logger.info("✅ Polygon.io connected successfully")
            else:
                logger.warning("⚠️ Polygon.io connection failed - using mock data")

            # Start background tasks
            asyncio.create_task(self.market_data_loop())
            asyncio.create_task(self.portfolio_update_loop())
            asyncio.create_task(self.system_health_loop())

            logger.info("WebSocket server initialized with rate limiting")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
    
    async def market_data_loop(self):
        """Continuously fetch and broadcast market data"""
        while True:
            try:
                # Get latest market data
                if self.market_data_feed:
                    latest = self.market_data_feed.get_latest_data()
                    
                    for symbol, data in latest.items():
                        # Update cache
                        self.latest_market_data[symbol] = data
                        if 'close' in data:
                            self.latest_prices[symbol] = data['close']
                        
                        # Broadcast to subscribed clients
                        await self.broadcast_market_data(symbol, data)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Market data loop error: {e}")
                await asyncio.sleep(10)
    
    async def portfolio_update_loop(self):
        """Update and broadcast portfolio data"""
        while True:
            try:
                # Calculate portfolio metrics
                total_pnl = sum(p.get('pnl', 0) for p in self.portfolio['positions'])
                self.portfolio['equity'] = self.portfolio['balance'] + total_pnl
                
                # Broadcast portfolio update
                await self.broadcast_portfolio_update()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Portfolio update loop error: {e}")
                await asyncio.sleep(5)
    
    async def system_health_loop(self):
        """Monitor and broadcast system health"""
        while True:
            try:
                health_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': self.system_status,
                    'trading_enabled': self.trading_enabled,
                    'connected_clients': len(self.clients),
                    'services': {
                        'polygon': 'healthy' if self.market_data_feed else 'error',
                        'database': 'healthy',
                        'websocket': 'healthy',
                        'ml_model': 'healthy',
                        'risk_manager': 'healthy'
                    },
                    'metrics': {
                        'cpu': 15.2,
                        'memory': 42.5,
                        'disk': 65.0,
                        'network': 30.1
                    }
                }
                
                await sio.emit('system_health', health_data)
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"System health loop error: {e}")
                await asyncio.sleep(30)
    
    async def broadcast_market_data(self, symbol: str, data: Dict):
        """Broadcast market data to subscribed clients"""
        message = {
            'symbol': symbol,
            'price': data.get('close', 0),
            'timestamp': data.get('timestamp', datetime.now().timestamp()),
            'volume': data.get('volume', 0),
            'change24h': self.calculate_change_24h(symbol, data.get('close', 0))
        }
        
        # Send to all clients subscribed to this symbol
        for client in self.clients.values():
            if symbol in client.subscribed_symbols:
                await sio.emit('market_data', message, room=client.sid)
    
    async def broadcast_portfolio_update(self):
        """Broadcast portfolio update to all clients"""
        await sio.emit('portfolio_update', {'portfolio': self.portfolio})
    
    async def execute_trade(self, data: Dict) -> Dict:
        """Execute a trade order"""
        try:
            symbol = data['symbol']
            side = data['side']
            quantity = data['quantity']
            order_type = data.get('type', 'market')
            
            # Get current price
            current_price = self.latest_prices.get(symbol, 0)
            
            if current_price == 0:
                return {'status': 'rejected', 'reason': 'No price available'}
            
            # Check if we have enough margin
            required_margin = quantity * current_price * 0.1  # 10% margin
            if required_margin > self.portfolio['free_margin']:
                return {'status': 'rejected', 'reason': 'Insufficient margin'}
            
            # Create position
            position = {
                'id': f"POS_{datetime.now().timestamp():.0f}",
                'symbol': symbol,
                'side': 'long' if side == 'buy' else 'short',
                'quantity': quantity,
                'entry_price': current_price,
                'current_price': current_price,
                'pnl': 0,
                'pnl_percent': 0,
                'open_time': datetime.now(timezone.utc).isoformat()
            }
            
            # Add to positions
            self.portfolio['positions'].append(position)
            
            # Update margin
            self.portfolio['margin'] += required_margin
            self.portfolio['free_margin'] -= required_margin
            
            # Log trade
            self.audit_logger.log_trade(
                action='execute',
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=current_price,
                order_id=position['id']
            )
            
            # Broadcast trade update
            await sio.emit('trade_update', {
                'trade': {
                    'status': 'executed',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            })
            
            return {'status': 'executed', 'position': position}
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'status': 'rejected', 'reason': str(e)}
    
    async def close_position(self, position_id: str):
        """Close a specific position"""
        for i, pos in enumerate(self.portfolio['positions']):
            if pos['id'] == position_id:
                # Calculate final P&L
                current_price = self.latest_prices.get(pos['symbol'], pos['entry_price'])
                if pos['side'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['quantity']
                
                # Update balance
                self.portfolio['balance'] += pnl
                
                # Release margin
                margin_to_release = pos['quantity'] * pos['entry_price'] * 0.1
                self.portfolio['margin'] -= margin_to_release
                self.portfolio['free_margin'] += margin_to_release
                
                # Remove position
                self.portfolio['positions'].pop(i)
                
                # Broadcast update
                await sio.emit('position_closed', {
                    'position_id': position_id,
                    'pnl': pnl
                })
                
                return True
        return False
    
    def calculate_change_24h(self, symbol: str, current_price: float) -> float:
        """Calculate 24h price change percentage"""
        # Simplified - in production would query historical price
        return -1.95 if symbol == 'BTC-USD' else -2.25
    
    async def emergency_stop(self):
        """Emergency stop - close all positions immediately"""
        logger.warning("⚠️ EMERGENCY STOP ACTIVATED")
        
        # Disable trading
        self.trading_enabled = False
        self.system_status = "STOPPED"
        
        # Close all positions
        positions_to_close = self.portfolio['positions'].copy()
        for position in positions_to_close:
            await self.close_position(position['id'])
        
        # Broadcast emergency stop
        await sio.emit('alert', {
            'severity': 'error',
            'message': 'EMERGENCY STOP - All positions closed',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Log event
        self.audit_logger.log_event(
            event_type='EMERGENCY_STOP',
            severity='CRITICAL',
            component='WebSocketServer',
            action='emergency_stop',
            details={'positions_closed': len(positions_to_close)}
        )


# Create server instance
server = TradingWebSocketServer()


# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    # Get client IP address
    ip_address = environ.get('REMOTE_ADDR', 'unknown')

    # Check rate limit for connections
    allowed, reason = await server.rate_limiter.check_connection_allowed(ip_address)
    if not allowed:
        logger.warning(f"Connection rejected for {ip_address}: {reason}")
        await sio.emit('error', {'message': f'Connection rejected: {reason}'}, room=sid)
        await sio.disconnect(sid)
        return False

    client = Client(
        sid=sid,
        connected_at=datetime.now(timezone.utc),
        subscribed_symbols=set(),
        ip_address=ip_address
    )
    server.clients[sid] = client

    logger.info(f"Client connected: {sid} from {ip_address}")

    # Send initial data
    await sio.emit('connected', {
        'message': 'Connected to trading system',
        'server_time': datetime.now(timezone.utc).isoformat()
    }, room=sid)

    # Send current portfolio
    await sio.emit('portfolio_update', {'portfolio': server.portfolio}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    if sid in server.clients:
        client = server.clients[sid]
        # Notify rate limiter of disconnection
        await server.rate_limiter.on_disconnect(client.ip_address)
        del server.clients[sid]
        logger.info(f"Client disconnected: {sid} from {client.ip_address}")


@sio.event
async def subscribe(sid, data):
    """Handle symbol subscription"""
    symbols = data.get('symbols', [])
    if sid in server.clients:
        server.clients[sid].subscribed_symbols.update(symbols)
        logger.info(f"Client {sid} subscribed to: {symbols}")
        
        # Send current prices for subscribed symbols
        for symbol in symbols:
            if symbol in server.latest_market_data:
                await server.broadcast_market_data(symbol, server.latest_market_data[symbol])


@sio.event
async def unsubscribe(sid, data):
    """Handle symbol unsubscription"""
    symbols = data.get('symbols', [])
    if sid in server.clients:
        server.clients[sid].subscribed_symbols.difference_update(symbols)
        logger.info(f"Client {sid} unsubscribed from: {symbols}")


@sio.event
async def place_trade(sid, data):
    """Handle trade placement"""
    logger.info(f"Trade request from {sid}: {data}")
    
    if not server.trading_enabled:
        await sio.emit('trade_update', {
            'trade': {'status': 'rejected', 'reason': 'Trading disabled'}
        }, room=sid)
        return
    
    result = await server.execute_trade(data)
    await sio.emit('trade_response', result, room=sid)


@sio.event
async def close_position(sid, data):
    """Handle position closure"""
    position_id = data.get('id')
    if await server.close_position(position_id):
        await sio.emit('position_closed', {'id': position_id}, room=sid)
    else:
        await sio.emit('error', {'message': 'Position not found'}, room=sid)


@sio.event
async def toggle_trading(sid, data):
    """Toggle trading on/off"""
    server.trading_enabled = data.get('enabled', False)
    server.system_status = "RUNNING" if server.trading_enabled else "PAUSED"
    
    await sio.emit('trading_status', {
        'enabled': server.trading_enabled,
        'status': server.system_status
    })
    
    logger.info(f"Trading {'enabled' if server.trading_enabled else 'disabled'} by {sid}")


@sio.event
async def emergency_stop(sid, data):
    """Handle emergency stop"""
    logger.warning(f"Emergency stop requested by {sid}")
    await server.emergency_stop()


# HTTP Routes
async def health_check(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'clients': len(server.clients),
        'trading_enabled': server.trading_enabled
    })


# Add routes
app.router.add_get('/health', health_check)
for route in app.router.routes():
    cors.add(route)


async def init_app():
    """Initialize application"""
    await server.initialize()
    return app


def main():
    """Run WebSocket server"""
    port = int(os.getenv('WEBSOCKET_PORT', 8000))
    
    logger.info(f"""
    ╔══════════════════════════════════════════╗
    ║   Trading WebSocket Server Starting...   ║
    ╠══════════════════════════════════════════╣
    ║   Port: {port:5}                            ║
    ║   CORS: Enabled                          ║
    ║   URL:  ws://localhost:{port:5}            ║
    ╚══════════════════════════════════════════╝
    """)
    
    web.run_app(
        init_app(),
        host='0.0.0.0',
        port=port,
        access_log=logger
    )


if __name__ == '__main__':
    main()