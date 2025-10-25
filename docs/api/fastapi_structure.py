# FastAPI Backend Structure for Transparency Dashboard
# This is a reference implementation showing the recommended structure

"""
src/api/
├── main.py                 # FastAPI app entry point
├── websocket.py            # Socket.IO server
├── config.py               # Configuration
├── dependencies.py         # Dependency injection
├── middleware.py           # Custom middleware
├── routes/
│   ├── __init__.py
│   ├── trades.py          # Trade history endpoints
│   ├── performance.py     # Performance metrics
│   ├── ai_insights.py     # AI decision data
│   ├── backtest.py        # Backtest results
│   └── portfolio.py       # Portfolio data
├── models/
│   ├── __init__.py
│   ├── trade.py           # Pydantic models
│   ├── performance.py
│   ├── ai_decision.py
│   └── backtest.py
├── services/
│   ├── __init__.py
│   ├── trade_service.py
│   ├── performance_service.py
│   └── ai_service.py
└── database/
    ├── __init__.py
    ├── connection.py
    └── queries.py
"""

# ============================================================================
# main.py - FastAPI Application Entry Point
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import socketio
from contextlib import asynccontextmanager

from .config import settings
from .routes import trades, performance, ai_insights, backtest, portfolio
from .websocket import sio
from .database.connection import init_db, close_db


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await init_db()
    print("Database initialized")

    yield

    # Shutdown
    await close_db()
    print("Database closed")


# Create FastAPI app
app = FastAPI(
    title="RRRalgorithms Transparency API",
    description="Real-time trading transparency and analytics API",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount Socket.IO
sio_asgi_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path="/socket.io"
)

# Include routers
app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])
app.include_router(ai_insights.router, prefix="/api/v1/ai", tags=["ai"])
app.include_router(backtest.router, prefix="/api/v1/backtests", tags=["backtests"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RRRalgorithms Transparency API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# ============================================================================
# routes/trades.py - Trade Endpoints
# ============================================================================

from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime

from ..models.trade import Trade, TradeFilter, TradeResponse
from ..services.trade_service import TradeService
from ..dependencies import get_trade_service

router = APIRouter()


@router.get("/", response_model=TradeResponse)
@limiter.limit("100/minute")
async def get_trades(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    service: TradeService = Depends(get_trade_service)
):
    """
    Get trade history with pagination and filtering

    Args:
        limit: Number of trades to return (max 1000)
        offset: Offset for pagination
        symbol: Filter by symbol (e.g., BTC-USD)
        status: Filter by status (filled, pending, cancelled)
        start_date: Filter by start date
        end_date: Filter by end date

    Returns:
        Paginated trade history
    """
    filters = TradeFilter(
        symbol=symbol,
        status=status,
        start_date=start_date,
        end_date=end_date
    )

    trades = await service.get_trades(
        limit=limit,
        offset=offset,
        filters=filters
    )

    total = await service.count_trades(filters)

    return TradeResponse(
        trades=trades,
        total=total,
        page=offset // limit + 1,
        pages=(total + limit - 1) // limit
    )


@router.get("/{trade_id}", response_model=Trade)
async def get_trade(
    trade_id: str,
    service: TradeService = Depends(get_trade_service)
):
    """Get single trade details"""
    trade = await service.get_trade_by_id(trade_id)

    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade


@router.post("/export")
async def export_trades(
    format: str = Query("csv", regex="^(csv|json)$"),
    filters: TradeFilter = None,
    service: TradeService = Depends(get_trade_service)
):
    """Export trades to CSV or JSON"""
    if format == "csv":
        content = await service.export_trades_csv(filters)
        media_type = "text/csv"
        filename = "trades.csv"
    else:
        content = await service.export_trades_json(filters)
        media_type = "application/json"
        filename = "trades.json"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


# ============================================================================
# models/trade.py - Pydantic Models
# ============================================================================

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal


class Trade(BaseModel):
    """Trade model"""
    id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Decimal
    order_type: str
    status: str
    pnl: Optional[Decimal] = None
    strategy: str
    ai_confidence: Optional[float] = None
    ai_reasoning: Optional[str] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    fees: Optional[Decimal] = None
    slippage: Optional[Decimal] = None

    class Config:
        json_encoders = {
            Decimal: float,
            datetime: lambda v: v.isoformat()
        }


class TradeFilter(BaseModel):
    """Trade filter model"""
    symbol: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class TradeResponse(BaseModel):
    """Trade response with pagination"""
    trades: List[Trade]
    total: int
    page: int
    pages: int


class AIDecision(BaseModel):
    """AI decision model"""
    id: str
    timestamp: datetime
    symbol: str
    model_name: str
    prediction: Dict[str, Any]
    features: Dict[str, float]
    reasoning: str
    outcome: Optional[str] = None
    actual_return: Optional[Decimal] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    timestamp: datetime
    portfolio_value: Decimal
    daily_pnl: Decimal
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


# ============================================================================
# websocket.py - Socket.IO Server
# ============================================================================

import socketio
from typing import Set, Dict
import asyncio
import redis.asyncio as redis

from .config import settings


# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=True,
    engineio_logger=True
)

# Redis client for pub/sub
redis_client = None

# Track connected clients and their subscriptions
client_subscriptions: Dict[str, Set[str]] = {}


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    print(f"Client connected: {sid}")
    client_subscriptions[sid] = set()

    # Send connection confirmation
    await sio.emit('connected', {'status': 'ok'}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client disconnected: {sid}")

    # Clean up subscriptions
    if sid in client_subscriptions:
        del client_subscriptions[sid]


@sio.event
async def subscribe(sid, data):
    """Subscribe to channels"""
    channels = data.get('channels', [])

    for channel in channels:
        if channel in ['trades', 'performance', 'ai_decisions', 'positions']:
            client_subscriptions[sid].add(channel)
            await sio.enter_room(sid, channel)

    await sio.emit('subscribed', {
        'channels': list(client_subscriptions[sid])
    }, room=sid)


@sio.event
async def unsubscribe(sid, data):
    """Unsubscribe from channels"""
    channels = data.get('channels', [])

    for channel in channels:
        if channel in client_subscriptions[sid]:
            client_subscriptions[sid].remove(channel)
            await sio.leave_room(sid, channel)

    await sio.emit('unsubscribed', {
        'channels': channels
    }, room=sid)


async def broadcast_event(channel: str, event: str, data: dict):
    """Broadcast event to all clients subscribed to channel"""
    await sio.emit(f"{channel}:{event}", data, room=channel)


async def listen_to_redis():
    """Listen to Redis pub/sub for events from trading system"""
    global redis_client

    redis_client = await redis.from_url(settings.REDIS_URL)
    pubsub = redis_client.pubsub()

    # Subscribe to trading system events
    await pubsub.subscribe('trades', 'ai_decisions', 'performance', 'positions')

    async for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'])

            # Broadcast to WebSocket clients
            event_type = data.pop('event_type', 'update')
            await broadcast_event(channel, event_type, data)


# Start Redis listener on startup
@sio.on('startup')
async def startup():
    """Start Redis listener"""
    asyncio.create_task(listen_to_redis())


# ============================================================================
# services/trade_service.py - Trade Business Logic
# ============================================================================

from typing import List, Optional
from datetime import datetime
import csv
import json
from io import StringIO

from ..database.queries import TradeQueries
from ..models.trade import Trade, TradeFilter


class TradeService:
    """Trade service"""

    def __init__(self, queries: TradeQueries):
        self.queries = queries

    async def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[TradeFilter] = None
    ) -> List[Trade]:
        """Get trades with pagination and filtering"""
        rows = await self.queries.get_trades(
            limit=limit,
            offset=offset,
            filters=filters
        )

        return [Trade(**row) for row in rows]

    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get single trade by ID"""
        row = await self.queries.get_trade_by_id(trade_id)

        if row:
            return Trade(**row)

        return None

    async def count_trades(self, filters: Optional[TradeFilter] = None) -> int:
        """Count total trades"""
        return await self.queries.count_trades(filters)

    async def export_trades_csv(self, filters: Optional[TradeFilter] = None) -> str:
        """Export trades to CSV"""
        trades = await self.get_trades(limit=10000, filters=filters)

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'id', 'timestamp', 'symbol', 'side', 'quantity', 'price',
            'status', 'pnl', 'strategy', 'ai_confidence'
        ])

        writer.writeheader()
        for trade in trades:
            writer.writerow(trade.dict())

        return output.getvalue()

    async def export_trades_json(self, filters: Optional[TradeFilter] = None) -> str:
        """Export trades to JSON"""
        trades = await self.get_trades(limit=10000, filters=filters)

        return json.dumps([trade.dict() for trade in trades], indent=2, default=str)


# ============================================================================
# Example Usage
# ============================================================================

"""
# Start the server
uvicorn src.api.main:sio_asgi_app --host 0.0.0.0 --port 8000 --reload

# Connect from frontend
const socket = io('http://localhost:8000');

socket.emit('subscribe', {
  channels: ['trades', 'performance', 'ai_decisions']
});

socket.on('trades:new_trade', (data) => {
  console.log('New trade:', data);
});

socket.on('ai_decisions:new_prediction', (data) => {
  console.log('New AI prediction:', data);
});

socket.on('performance:metrics_update', (data) => {
  console.log('Performance update:', data);
});
"""
