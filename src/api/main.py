"""
RRRalgorithms Transparency Dashboard API
FastAPI backend for real-time trading transparency and analytics
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timedelta
import logging

# Import Prometheus metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import database client
from .transparency_db import get_db, close_db, TransparencyDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection status
database_connected = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global database_connected
    # Startup
    logger.info("Initializing transparency dashboard API...")
    try:
        # Initialize database connection
        await get_db()
        database_connected = True
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        database_connected = False

    logger.info("API initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    await close_db()
    database_connected = False
    logger.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RRRalgorithms Transparency API",
    description="Real-time trading transparency and analytics API - Production Ready",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware - restricted to specific origins for security
# Configure allowed origins via CORS_ORIGINS environment variable
from src.security.secrets_manager import get_secrets_manager
secrets = get_secrets_manager()
allowed_origins = secrets.get_secret("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins only - NEVER use "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security headers middleware
from src.security.middleware import SecurityHeadersMiddleware, RateLimitMiddleware, AuditLoggingMiddleware
app.add_middleware(SecurityHeadersMiddleware, enable_hsts=not secrets.is_development())

# Rate limiting middleware
rate_limit_config = {
    "requests_per_minute": int(secrets.get_secret("RATE_LIMIT_PER_MINUTE", "60")),
    "requests_per_hour": int(secrets.get_secret("RATE_LIMIT_PER_HOUR", "1000")),
    "burst_size": int(secrets.get_secret("RATE_LIMIT_BURST_SIZE", "10"))
}
app.add_middleware(RateLimitMiddleware, **rate_limit_config)

# Audit logging middleware
app.add_middleware(AuditLoggingMiddleware)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "api": "RRRalgorithms Transparency Dashboard",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if database_connected else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if database_connected else "disconnected",
        "components": {
            "api": "operational",
            "database": "connected" if database_connected else "disconnected",
            "websocket": "ready"
        }
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


# ============================================================================
# Portfolio Endpoints
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio(db: TransparencyDB = Depends(get_db)):
    """
    Get current portfolio overview

    Returns:
        Portfolio summary with positions, equity, and performance
    """
    return await db.get_portfolio_summary()


@app.get("/api/portfolio/positions")
async def get_positions():
    """
    Get all open positions

    Returns:
        List of current positions with P&L
    """
    # TODO: Connect to actual database
    return {
        "positions": [
            {
                "id": "pos-001",
                "symbol": "BTC-USD",
                "side": "long",
                "size": 0.5,
                "entry_price": 50000.00,
                "current_price": 51000.00,
                "unrealized_pnl": 500.00,
                "unrealized_pnl_percent": 2.00,
                "opened_at": (datetime.utcnow() - timedelta(hours=3)).isoformat()
            },
            {
                "id": "pos-002",
                "symbol": "ETH-USD",
                "side": "long",
                "size": 10.0,
                "entry_price": 3000.00,
                "current_price": 3050.00,
                "unrealized_pnl": 500.00,
                "unrealized_pnl_percent": 1.67,
                "opened_at": (datetime.utcnow() - timedelta(hours=5)).isoformat()
            }
        ],
        "total_count": 2,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Trading Feed Endpoints
# ============================================================================

@app.get("/api/trades")
async def get_trades(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    symbol: Optional[str] = None,
    db: TransparencyDB = Depends(get_db)
):
    """
    Get recent trade history

    Args:
        limit: Maximum number of trades to return (max 500)
        offset: Offset for pagination
        symbol: Filter by symbol (optional)

    Returns:
        List of recent trades with execution details
    """
    return await db.get_recent_trades(limit=limit, offset=offset, symbol=symbol)


# ============================================================================
# Performance Metrics Endpoints
# ============================================================================

@app.get("/api/performance")
async def get_performance(
    period: str = Query(default="1d", regex="^(1h|4h|1d|7d|30d|all)$"),
    db: TransparencyDB = Depends(get_db)
):
    """
    Get performance metrics for specified period

    Args:
        period: Time period (1h, 4h, 1d, 7d, 30d, all)

    Returns:
        Performance metrics including returns, Sharpe ratio, drawdown, etc.
    """
    return await db.get_performance_metrics(period=period)


@app.get("/api/performance/equity-curve")
async def get_equity_curve(
    period: str = Query(default="7d", regex="^(1d|7d|30d|90d|all)$"),
    interval: str = Query(default="1h", regex="^(5m|15m|1h|4h|1d)$"),
    db: TransparencyDB = Depends(get_db)
):
    """
    Get equity curve data points for charting

    Args:
        period: Time period to fetch
        interval: Data point interval

    Returns:
        Array of timestamp and equity value pairs
    """
    return await db.get_equity_curve(period=period, interval=interval)


# ============================================================================
# AI Insights Endpoints
# ============================================================================

@app.get("/api/ai/decisions")
async def get_ai_decisions(
    limit: int = Query(default=50, le=200),
    model: Optional[str] = None,
    db: TransparencyDB = Depends(get_db)
):
    """
    Get recent AI predictions and decisions

    Args:
        limit: Maximum number of decisions to return
        model: Filter by model name (optional)

    Returns:
        List of AI decisions with predictions and outcomes
    """
    return await db.get_ai_decisions(limit=limit, model=model)


@app.get("/api/ai/models")
async def get_ai_models(db: TransparencyDB = Depends(get_db)):
    """
    Get information about active AI models

    Returns:
        List of AI models with their performance stats
    """
    return await db.get_ai_models_performance()


# ============================================================================
# Backtest Results Endpoints
# ============================================================================

@app.get("/api/backtests")
async def get_backtests(limit: int = Query(default=20, le=100)):
    """
    Get recent backtest results

    Args:
        limit: Maximum number of backtests to return

    Returns:
        List of backtest results with performance metrics
    """
    # TODO: Connect to actual database
    return {
        "backtests": [
            {
                "id": "bt-001",
                "name": "Momentum Strategy v3",
                "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "period": "2024-01-01 to 2024-10-01",
                "total_return": 15.5,
                "sharpe_ratio": 1.92,
                "max_drawdown": -8.5,
                "win_rate": 67.2,
                "total_trades": 234,
                "status": "completed"
            },
            {
                "id": "bt-002",
                "name": "Mean Reversion v2",
                "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "period": "2024-01-01 to 2024-10-01",
                "total_return": 12.3,
                "sharpe_ratio": 1.75,
                "max_drawdown": -6.2,
                "win_rate": 58.5,
                "total_trades": 187,
                "status": "completed"
            }
        ],
        "total_count": 2,
        "limit": limit,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/backtests/{backtest_id}")
async def get_backtest_detail(backtest_id: str):
    """
    Get detailed backtest results

    Args:
        backtest_id: Backtest ID

    Returns:
        Detailed backtest metrics and trade log
    """
    # TODO: Connect to actual database
    return {
        "id": backtest_id,
        "name": "Momentum Strategy v3",
        "description": "RSI + MACD momentum strategy with dynamic position sizing",
        "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
        "period": {
            "start": "2024-01-01T00:00:00",
            "end": "2024-10-01T00:00:00",
            "days": 274
        },
        "performance": {
            "initial_capital": 100000.00,
            "final_equity": 115500.00,
            "total_return": 15.5,
            "total_return_percent": 15.5,
            "sharpe_ratio": 1.92,
            "sortino_ratio": 2.25,
            "max_drawdown": -8.5,
            "max_drawdown_percent": -8.5,
            "calmar_ratio": 1.82,
            "win_rate": 67.2,
            "profit_factor": 2.15,
            "total_trades": 234,
            "winning_trades": 157,
            "losing_trades": 77
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# System Stats Endpoints
# ============================================================================

@app.get("/api/stats")
async def get_system_stats(db: TransparencyDB = Depends(get_db)):
    """
    Get overall system statistics

    Returns:
        System-wide stats and metrics
    """
    return await db.get_system_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
