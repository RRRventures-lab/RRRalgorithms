"""
RRRalgorithms Transparency Dashboard API
FastAPI backend for real-time trading transparency and analytics
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database.client_factory import get_db
from database.repositories import (
    PortfolioRepository,
    TradingRepository,
    PerformanceRepository,
    AIRepository,
    BacktestRepository,
    SystemRepository
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database client and repositories
db_client = None
portfolio_repo = None
trading_repo = None
performance_repo = None
ai_repo = None
backtest_repo = None
system_repo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global db_client, portfolio_repo, trading_repo, performance_repo
    global ai_repo, backtest_repo, system_repo

    # Startup
    logger.info("Initializing transparency dashboard API...")

    # Initialize database connection
    try:
        db_client = get_db()
        await db_client.connect()
        logger.info("Database connection established")

        # Initialize repositories
        portfolio_repo = PortfolioRepository(db_client)
        trading_repo = TradingRepository(db_client)
        performance_repo = PerformanceRepository(db_client)
        ai_repo = AIRepository(db_client)
        backtest_repo = BacktestRepository(db_client)
        system_repo = SystemRepository(db_client)

        logger.info("All repositories initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    logger.info("API initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    if db_client:
        await db_client.disconnect()
        logger.info("Database connection closed")
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

# CORS middleware - allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


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
    db_connected = db_client is not None and db_client.connection is not None

    return {
        "status": "healthy" if db_connected else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db_connected else "disconnected",
        "components": {
            "api": "operational",
            "database": "connected" if db_connected else "disconnected",
            "websocket": "ready"
        }
    }


# ============================================================================
# Portfolio Endpoints
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio():
    """
    Get current portfolio overview

    Returns:
        Portfolio summary with positions, equity, and performance
    """
    if not portfolio_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await portfolio_repo.get_portfolio_overview()


@app.get("/api/portfolio/positions")
async def get_positions():
    """
    Get all open positions

    Returns:
        List of current positions with P&L
    """
    if not portfolio_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await portfolio_repo.get_positions()


# ============================================================================
# Trading Feed Endpoints
# ============================================================================

@app.get("/api/trades")
async def get_trades(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    symbol: Optional[str] = None
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
    if not trading_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await trading_repo.get_trades(limit=limit, offset=offset, symbol=symbol)


# ============================================================================
# Performance Metrics Endpoints
# ============================================================================

@app.get("/api/performance")
async def get_performance(
    period: str = Query(default="1d", regex="^(1h|4h|1d|7d|30d|all)$")
):
    """
    Get performance metrics for specified period

    Args:
        period: Time period (1h, 4h, 1d, 7d, 30d, all)

    Returns:
        Performance metrics including returns, Sharpe ratio, drawdown, etc.
    """
    if not performance_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await performance_repo.get_performance_metrics(period=period)


@app.get("/api/performance/equity-curve")
async def get_equity_curve(
    period: str = Query(default="7d", regex="^(1d|7d|30d|90d|all)$"),
    interval: str = Query(default="1h", regex="^(5m|15m|1h|4h|1d)$")
):
    """
    Get equity curve data points for charting

    Args:
        period: Time period to fetch
        interval: Data point interval

    Returns:
        Array of timestamp and equity value pairs
    """
    if not performance_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await performance_repo.get_equity_curve(period=period, interval=interval)


# ============================================================================
# AI Insights Endpoints
# ============================================================================

@app.get("/api/ai/decisions")
async def get_ai_decisions(
    limit: int = Query(default=50, le=200),
    model: Optional[str] = None
):
    """
    Get recent AI predictions and decisions

    Args:
        limit: Maximum number of decisions to return
        model: Filter by model name (optional)

    Returns:
        List of AI decisions with predictions and outcomes
    """
    if not ai_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await ai_repo.get_ai_decisions(limit=limit, model=model)


@app.get("/api/ai/models")
async def get_ai_models():
    """
    Get information about active AI models

    Returns:
        List of AI models with their performance stats
    """
    if not ai_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await ai_repo.get_ai_models()


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
    if not backtest_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await backtest_repo.get_backtests(limit=limit)


@app.get("/api/backtests/{backtest_id}")
async def get_backtest_detail(backtest_id: str):
    """
    Get detailed backtest results

    Args:
        backtest_id: Backtest ID

    Returns:
        Detailed backtest metrics and trade log
    """
    if not backtest_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    result = await backtest_repo.get_backtest_detail(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    return result


# ============================================================================
# System Stats Endpoints
# ============================================================================

@app.get("/api/stats")
async def get_system_stats():
    """
    Get overall system statistics

    Returns:
        System-wide stats and metrics
    """
    if not system_repo:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await system_repo.get_system_stats()


# ============================================================================
# Market Data Endpoints (Live Data from Polygon.io)
# ============================================================================

@app.get("/api/market/prices")
async def get_latest_prices(symbols: Optional[str] = Query(default=None)):
    """
    Get latest prices for symbols

    Args:
        symbols: Comma-separated list of symbols (e.g., "BTC-USD,ETH-USD")
                 If not provided, returns all symbols

    Returns:
        Latest price data for requested symbols
    """
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]

    # Query latest market data for each symbol
    query = """
        SELECT
            symbol,
            timestamp,
            close as price,
            volume,
            high,
            low,
            open,
            vwap
        FROM market_data
        WHERE symbol IN (
            SELECT symbol FROM (
                SELECT symbol, MAX(timestamp) as max_ts
                FROM market_data
                """ + ("WHERE symbol IN ({})".format(",".join(["?" for _ in symbol_list])) if symbol_list else "") + """
                GROUP BY symbol
            )
        )
        AND timestamp IN (
            SELECT MAX(timestamp)
            FROM market_data
            """ + ("WHERE symbol IN ({})".format(",".join(["?" for _ in symbol_list])) if symbol_list else "") + """
            GROUP BY symbol
        )
        ORDER BY symbol
    """

    params = tuple(symbol_list + symbol_list) if symbol_list else None
    results = await db_client.fetch_all(query, params)

    prices = []
    for row in results:
        prices.append({
            "symbol": row["symbol"],
            "price": round(row["price"], 2),
            "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "volume": round(row["volume"], 2),
            "vwap": round(row["vwap"], 2) if row["vwap"] else None
        })

    return {
        "prices": prices,
        "count": len(prices),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/market/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    interval: str = Query(default="1h", regex="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(default=100, le=1000)
):
    """
    Get OHLCV (candlestick) data for a symbol

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of bars to return (max 1000)

    Returns:
        OHLCV candlestick data
    """
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # For now, return 1-minute data (we can add aggregation later)
    query = """
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            trade_count
        FROM market_data
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    results = await db_client.fetch_all(query, (symbol, limit))

    candlesticks = []
    for row in results:
        candlesticks.append({
            "timestamp": row["timestamp"],
            "time": datetime.fromtimestamp(row["timestamp"]).isoformat(),
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "close": round(row["close"], 2),
            "volume": round(row["volume"], 2),
            "vwap": round(row["vwap"], 2) if row["vwap"] else None,
            "trades": row["trade_count"]
        })

    # Reverse to get chronological order
    candlesticks.reverse()

    return {
        "symbol": symbol,
        "interval": interval,
        "data": candlesticks,
        "count": len(candlesticks),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/market/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(default=100, le=1000)
):
    """
    Get recent trades for a symbol

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        limit: Number of trades to return (max 1000)

    Returns:
        Recent trade data
    """
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    query = """
        SELECT
            timestamp,
            price,
            size,
            side,
            exchange_id,
            trade_id
        FROM trades_data
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    results = await db_client.fetch_all(query, (symbol, limit))

    trades = []
    for row in results:
        trades.append({
            "timestamp": row["timestamp"],
            "time": datetime.fromtimestamp(row["timestamp"]).isoformat(),
            "price": round(row["price"], 2),
            "size": round(row["size"], 6),
            "side": row["side"],
            "exchange_id": row["exchange_id"],
            "trade_id": row["trade_id"],
            "value": round(row["price"] * row["size"], 2)
        })

    # Reverse to get chronological order
    trades.reverse()

    return {
        "symbol": symbol,
        "trades": trades,
        "count": len(trades),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/market/quotes/{symbol}")
async def get_recent_quotes(
    symbol: str,
    limit: int = Query(default=100, le=1000)
):
    """
    Get recent quotes (bid/ask) for a symbol

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        limit: Number of quotes to return (max 1000)

    Returns:
        Recent quote data
    """
    if not db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    query = """
        SELECT
            timestamp,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            exchange_id
        FROM quotes
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    results = await db_client.fetch_all(query, (symbol, limit))

    quotes = []
    for row in results:
        spread = row["ask_price"] - row["bid_price"]
        quotes.append({
            "timestamp": row["timestamp"],
            "time": datetime.fromtimestamp(row["timestamp"]).isoformat(),
            "bid_price": round(row["bid_price"], 2),
            "bid_size": round(row["bid_size"], 6),
            "ask_price": round(row["ask_price"], 2),
            "ask_size": round(row["ask_size"], 6),
            "spread": round(spread, 2),
            "exchange_id": row["exchange_id"]
        })

    # Reverse to get chronological order
    quotes.reverse()

    return {
        "symbol": symbol,
        "quotes": quotes,
        "count": len(quotes),
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
