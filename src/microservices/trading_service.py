from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from src.core.async_database import get_async_db
from src.core.memory_cache import get_cache
from src.core.redis_cache import get_redis_cache
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging


"""
Trading Service
===============

Microservice for order management, position tracking, and trade execution.
Handles trading operations, portfolio management, and risk controls.

Author: RRR Ventures
Date: 2025-10-12
"""




class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderRequest(BaseModel):
    """Order request model."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled


class OrderResponse(BaseModel):
    """Order response model."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    timestamp: float
    created_at: str


class PositionResponse(BaseModel):
    """Position response model."""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float


class TradingService:
    """
    Trading service for order management and execution.
    
    Features:
    - Order management and execution
    - Position tracking
    - Portfolio management
    - Risk controls
    - Trade history
    """
    
    def __init__(self, port: int = 8003):
        """
        Initialize trading service.
        
        Args:
            port: Service port
        """
        self.port = port
        self.app = FastAPI(title="Trading Service", version="1.0.0")
        
        # Trading components
        self.db = None
        self.redis_cache = None
        self.memory_cache = None
        
        # Trading state
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0
        
        # Performance metrics
        self.metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_trades': 0,
            'total_volume': 0.0,
            'avg_execution_time': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check."""
            return {
                "status": "healthy",
                "service": "trading-service",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.metrics
        
        @self.app.post("/orders", response_model=OrderResponse)
        async def create_order(order_request: OrderRequest):
            """Create a new order."""
            try:
                # Generate order ID
                self.order_counter += 1
                order_id = f"ORD_{self.order_counter:06d}"
                
                # Validate order
                await self._validate_order(order_request)
                
                # Create order
                order = {
                    "order_id": order_id,
                    "symbol": order_request.symbol,
                    "side": order_request.side,
                    "order_type": order_request.order_type,
                    "quantity": order_request.quantity,
                    "price": order_request.price,
                    "stop_price": order_request.stop_price,
                    "time_in_force": order_request.time_in_force,
                    "status": OrderStatus.PENDING,
                    "filled_quantity": 0.0,
                    "remaining_quantity": order_request.quantity,
                    "timestamp": asyncio.get_event_loop().time(),
                    "created_at": datetime.now().isoformat()
                }
                
                # Store order
                self.orders[order_id] = order
                
                # Process order (simplified - in production, this would interface with exchange)
                await self._process_order(order)
                
                # Update metrics
                self.metrics['total_orders'] += 1
                self.metrics['successful_orders'] += 1
                
                return OrderResponse(**order)
                
            except Exception as e:
                self.logger.error(f"Order creation failed: {e}")
                self.metrics['total_orders'] += 1
                self.metrics['failed_orders'] += 1
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/orders/{order_id}", response_model=OrderResponse)
        async def get_order(order_id: str):
            """Get order by ID."""
            if order_id not in self.orders:
                raise HTTPException(status_code=404, detail="Order not found")
            
            return OrderResponse(**self.orders[order_id])
        
        @self.app.get("/orders")
        async def get_orders(symbol: Optional[str] = None, status: Optional[OrderStatus] = None):
            """Get orders with optional filters."""
            orders = list(self.orders.values())
            
            if symbol:
                orders = [o for o in orders if o['symbol'] == symbol]
            
            if status:
                orders = [o for o in orders if o['status'] == status]
            
            return {"orders": [OrderResponse(**order) for order in orders]}
        
        @self.app.delete("/orders/{order_id}")
        async def cancel_order(order_id: str):
            """Cancel an order."""
            if order_id not in self.orders:
                raise HTTPException(status_code=404, detail="Order not found")
            
            order = self.orders[order_id]
            if order['status'] in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                raise HTTPException(status_code=400, detail="Order cannot be cancelled")
            
            order['status'] = OrderStatus.CANCELLED
            order['remaining_quantity'] = 0.0
            
            return {"message": "Order cancelled successfully"}
        
        @self.app.get("/positions")
        async def get_positions():
            """Get all positions."""
            positions = []
            for symbol, position in self.positions.items():
                positions.append(PositionResponse(**position))
            
            return {"positions": positions}
        
        @self.app.get("/positions/{symbol}", response_model=PositionResponse)
        async def get_position(symbol: str):
            """Get position for a specific symbol."""
            if symbol not in self.positions:
                raise HTTPException(status_code=404, detail="Position not found")
            
            return PositionResponse(**self.positions[symbol])
        
        @self.app.get("/portfolio")
        async def get_portfolio():
            """Get portfolio summary."""
            total_value = 0.0
            total_pnl = 0.0
            
            for position in self.positions.values():
                total_value += position['market_value']
                total_pnl += position['unrealized_pnl']
            
            return {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "positions_count": len(self.positions),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/trades")
        async def get_trades(symbol: Optional[str] = None, limit: int = 100):
            """Get trade history."""
            try:
                # Get trades from database
                query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
                params = (limit,)
                
                if symbol:
                    query = "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
                    params = (symbol, limit)
                
                trades = await self.db.execute_query(query, params, fetch=True)
                return {"trades": trades or []}
                
            except Exception as e:
                self.logger.error(f"Error getting trades: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _validate_order(self, order_request: OrderRequest) -> None:
        """Validate order request."""
        # Check required fields
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not order_request.price:
            raise ValueError("Price is required for limit orders")
        
        if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order_request.stop_price:
            raise ValueError("Stop price is required for stop orders")
        
        # Check quantity
        if order_request.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Check price
        if order_request.price and order_request.price <= 0:
            raise ValueError("Price must be positive")
        
        # Check stop price
        if order_request.stop_price and order_request.stop_price <= 0:
            raise ValueError("Stop price must be positive")
    
    async def _process_order(self, order: Dict[str, Any]) -> None:
        """Process order execution (simplified)."""
        try:
            # Simulate order execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # For demo purposes, fill all market orders immediately
            if order['order_type'] == OrderType.MARKET:
                order['status'] = OrderStatus.FILLED
                order['filled_quantity'] = order['quantity']
                order['remaining_quantity'] = 0.0
                
                # Update position
                await self._update_position(order)
                
                # Record trade
                await self._record_trade(order)
                
                self.metrics['total_trades'] += 1
                self.metrics['total_volume'] += order['quantity']
            
            else:
                # For limit orders, keep as pending (in production, would check market price)
                order['status'] = OrderStatus.PENDING
            
        except Exception as e:
            self.logger.error(f"Order processing failed: {e}")
            order['status'] = OrderStatus.REJECTED
    
    async def _update_position(self, order: Dict[str, Any]) -> None:
        """Update position after order execution."""
        symbol = order['symbol']
        quantity = order['filled_quantity']
        price = order.get('price', 0.0)  # Use market price if not specified
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': 0.0,
                'average_price': 0.0,
                'current_price': price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'market_value': 0.0
            }
        
        position = self.positions[symbol]
        
        if order['side'] == OrderSide.BUY:
            # Add to position
            total_quantity = position['quantity'] + quantity
            if total_quantity > 0:
                position['average_price'] = (
                    (position['quantity'] * position['average_price'] + quantity * price) / total_quantity
                )
            position['quantity'] = total_quantity
        else:
            # Subtract from position
            position['quantity'] -= quantity
            if position['quantity'] < 0:
                position['quantity'] = 0
        
        # Update market value and PnL
        position['current_price'] = price
        position['market_value'] = position['quantity'] * price
        position['unrealized_pnl'] = position['quantity'] * (price - position['average_price'])
    
    async def _record_trade(self, order: Dict[str, Any]) -> None:
        """Record trade in database."""
        try:
            trade_data = {
                'symbol': order['symbol'],
                'timestamp': order['timestamp'],
                'side': order['side'],
                'quantity': order['filled_quantity'],
                'price': order.get('price', 0.0),
                'total_value': order['filled_quantity'] * order.get('price', 0.0),
                'fees': 0.0,  # Simplified
                'order_id': order['order_id'],
                'status': order['status']
            }
            
            # Store in database
            await self.db.batch_insert_trades([trade_data])
            
        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
    
    async def initialize(self) -> None:
        """Initialize trading service components."""
        try:
            # Initialize database
            self.db = await get_async_db()
            
            # Initialize caches
            self.redis_cache = await get_redis_cache()
            self.memory_cache = get_cache()
            
            self.logger.info("Trading service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading service: {e}")
            raise
    
    async def start(self) -> None:
        """Start the trading service."""
        self.logger.info(f"Starting Trading Service on port {self.port}")
        
        # Initialize components
        await self.initialize()
        
        # Start FastAPI server
        import uvicorn
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'service': 'trading-service',
            'port': self.port,
            'active_orders': len([o for o in self.orders.values() if o['status'] == OrderStatus.PENDING]),
            'total_positions': len(self.positions),
            'metrics': self.metrics
        }


# Global trading service instance
_trading_service: Optional[TradingService] = None


@lru_cache(maxsize=128)


def get_trading_service() -> TradingService:
    """Get the global trading service instance."""
    global _trading_service
    
    if _trading_service is None:
        _trading_service = TradingService()
    
    return _trading_service


if __name__ == "__main__":
    # Run trading service
    service = TradingService()
    asyncio.run(service.start())