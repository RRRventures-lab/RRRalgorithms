"""
Transparency Dashboard Database Client
Handles all database operations for the transparency API
"""

import aiosqlite
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TransparencyDB:
    """Database client for transparency dashboard operations"""

    def __init__(self, db_path: str = None):
        """
        Initialize transparency database client

        Args:
            db_path: Path to SQLite database (defaults to data/transparency.db)
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "transparency.db")

        self.db_path = db_path
        self.connection: Optional[aiosqlite.Connection] = None
        logger.info(f"TransparencyDB initialized with path: {self.db_path}")

    async def connect(self) -> None:
        """Establish database connection"""
        if self.connection is not None:
            logger.warning("Connection already established")
            return

        logger.info(f"Connecting to transparency database: {self.db_path}")

        self.connection = await aiosqlite.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None  # Autocommit mode
        )

        # Enable row factory for dict-like results
        self.connection.row_factory = aiosqlite.Row

        # Apply optimizations
        await self.connection.execute("PRAGMA journal_mode = WAL")
        await self.connection.execute("PRAGMA synchronous = NORMAL")
        await self.connection.execute("PRAGMA cache_size = -64000")  # 64MB
        await self.connection.execute("PRAGMA temp_store = MEMORY")

        logger.info("Transparency database connected successfully")

    async def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Transparency database disconnected")

    # ========================================================================
    # Portfolio Operations
    # ========================================================================

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio overview"""
        # Get latest performance snapshot
        query = """
            SELECT
                total_equity,
                cash_balance,
                invested_value,
                total_pnl,
                total_pnl_percent,
                daily_pnl,
                daily_pnl_percent
            FROM performance_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self.connection.execute(query) as cursor:
            row = await cursor.fetchone()

            if row:
                # Count positions (would need a positions table in real implementation)
                # For now, return with counts from trade feed
                positions_query = """
                    SELECT COUNT(DISTINCT symbol) as count
                    FROM trade_feed
                    WHERE event_type = 'order_filled'
                    AND timestamp > datetime('now', '-1 day')
                """

                async with self.connection.execute(positions_query) as pos_cursor:
                    pos_row = await pos_cursor.fetchone()
                    positions_count = pos_row['count'] if pos_row else 0

                return {
                    "total_equity": row['total_equity'],
                    "cash_balance": row['cash_balance'],
                    "invested": row['invested_value'],
                    "total_pnl": row['total_pnl'],
                    "total_pnl_percent": row['total_pnl_percent'],
                    "day_pnl": row['daily_pnl'],
                    "day_pnl_percent": row['daily_pnl_percent'],
                    "positions_count": positions_count,
                    "open_orders": 0,  # Would need orders table
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Return default values if no data
                return {
                    "total_equity": 100000.00,
                    "cash_balance": 100000.00,
                    "invested": 0.00,
                    "total_pnl": 0.00,
                    "total_pnl_percent": 0.00,
                    "day_pnl": 0.00,
                    "day_pnl_percent": 0.00,
                    "positions_count": 0,
                    "open_orders": 0,
                    "timestamp": datetime.utcnow().isoformat()
                }

    # ========================================================================
    # Trade Feed Operations
    # ========================================================================

    async def get_recent_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recent trades from trade feed"""

        where_clause = ""
        params = []

        if symbol:
            where_clause = "WHERE symbol = ?"
            params.append(symbol)

        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM trade_feed {where_clause}"
        async with self.connection.execute(count_query, params) as cursor:
            count_row = await cursor.fetchone()
            total_count = count_row['count'] if count_row else 0

        # Get trades
        query = f"""
            SELECT id, timestamp, event_type, symbol, data, source
            FROM trade_feed
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])

        trades = []
        async with self.connection.execute(query, params) as cursor:
            async for row in cursor:
                # Parse JSON data
                data = json.loads(row['data'])

                trades.append({
                    "id": row['id'],
                    "timestamp": row['timestamp'],
                    "symbol": row['symbol'],
                    "event_type": row['event_type'],
                    **data  # Merge data fields
                })

        return {
            "trades": trades,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # Performance Operations
    # ========================================================================

    async def get_performance_metrics(self, period: str = "1d") -> Dict[str, Any]:
        """Get performance metrics for specified period"""

        # Calculate time range based on period
        period_map = {
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "all": timedelta(days=365 * 10)  # 10 years for "all"
        }

        time_delta = period_map.get(period, timedelta(days=1))
        start_time = datetime.utcnow() - time_delta

        query = """
            SELECT
                total_pnl,
                total_pnl_percent,
                sharpe_ratio,
                sortino_ratio,
                max_drawdown,
                max_drawdown_percent,
                win_rate,
                profit_factor,
                total_trades
            FROM performance_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self.connection.execute(query, (start_time.isoformat(),)) as cursor:
            row = await cursor.fetchone()

            if row:
                return {
                    "period": period,
                    "total_return": row['total_pnl'],
                    "total_return_percent": row['total_pnl_percent'],
                    "sharpe_ratio": row['sharpe_ratio'],
                    "sortino_ratio": row['sortino_ratio'],
                    "max_drawdown": row['max_drawdown'],
                    "max_drawdown_percent": row['max_drawdown_percent'],
                    "win_rate": row['win_rate'],
                    "profit_factor": row['profit_factor'],
                    "total_trades": row['total_trades'],
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Return default metrics if no data
                return {
                    "period": period,
                    "total_return": 0.0,
                    "total_return_percent": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "max_drawdown_percent": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "timestamp": datetime.utcnow().isoformat()
                }

    async def get_equity_curve(
        self,
        period: str = "7d",
        interval: str = "1h"
    ) -> Dict[str, Any]:
        """Get equity curve data points"""

        # Calculate time range
        period_map = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "all": timedelta(days=365 * 10)
        }

        time_delta = period_map.get(period, timedelta(days=7))
        start_time = datetime.utcnow() - time_delta

        query = """
            SELECT timestamp, total_equity
            FROM performance_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """

        data_points = []
        async with self.connection.execute(query, (start_time.isoformat(),)) as cursor:
            async for row in cursor:
                data_points.append({
                    "timestamp": row['timestamp'],
                    "equity": row['total_equity']
                })

        # Get initial and current equity
        initial_equity = data_points[0]['equity'] if data_points else 100000.00
        current_equity = data_points[-1]['equity'] if data_points else 100000.00

        return {
            "period": period,
            "interval": interval,
            "data": data_points,
            "initial_equity": initial_equity,
            "current_equity": current_equity,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # AI Decision Operations
    # ========================================================================

    async def get_ai_decisions(
        self,
        limit: int = 50,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recent AI predictions and decisions"""

        where_clause = ""
        params = []

        if model:
            where_clause = "WHERE model_name = ?"
            params.append(model)

        query = f"""
            SELECT
                id, timestamp, symbol, model_name,
                prediction, features, reasoning,
                outcome, actual_return, confidence_score
            FROM ai_decisions
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """

        params.append(limit)

        decisions = []
        async with self.connection.execute(query, params) as cursor:
            async for row in cursor:
                decisions.append({
                    "id": row['id'],
                    "timestamp": row['timestamp'],
                    "model_name": row['model_name'],
                    "symbol": row['symbol'],
                    "prediction": json.loads(row['prediction']),
                    "reasoning": row['reasoning'],
                    "outcome": row['outcome'],
                    "actual_return": row['actual_return'],
                    "features": json.loads(row['features'])
                })

        return {
            "decisions": decisions,
            "total_count": len(decisions),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_ai_models_performance(self) -> Dict[str, Any]:
        """Get AI model performance statistics"""

        query = """
            SELECT
                model_name,
                COUNT(*) as predictions_today,
                AVG(confidence_score) as avg_confidence,
                SUM(CASE WHEN outcome = 'profitable' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                AVG(CASE WHEN outcome = 'profitable' THEN actual_return ELSE 0 END) as avg_return
            FROM ai_decisions
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY model_name
        """

        models = []
        async with self.connection.execute(query) as cursor:
            async for row in cursor:
                models.append({
                    "name": row['model_name'],
                    "type": "neural_network",  # Would need to track this
                    "status": "active",
                    "accuracy": row['win_rate'] or 0.0,
                    "predictions_today": row['predictions_today'],
                    "avg_confidence": row['avg_confidence'] or 0.0,
                    "win_rate": row['win_rate'] or 0.0
                })

        return {
            "models": models,
            "total_count": len(models),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # System Stats Operations
    # ========================================================================

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""

        # Trading stats
        trading_query = """
            SELECT
                COUNT(*) as total_trades,
                COUNT(DISTINCT symbol) as active_positions
            FROM trade_feed
            WHERE timestamp >= datetime('now', '-1 day')
            AND event_type = 'order_filled'
        """

        async with self.connection.execute(trading_query) as cursor:
            trade_row = await cursor.fetchone()

        # AI stats
        ai_query = """
            SELECT
                COUNT(*) as predictions_today,
                COUNT(DISTINCT model_name) as active_models,
                AVG(confidence_score) as avg_confidence
            FROM ai_decisions
            WHERE timestamp >= datetime('now', '-1 day')
        """

        async with self.connection.execute(ai_query) as cursor:
            ai_row = await cursor.fetchone()

        return {
            "trading": {
                "total_trades_today": trade_row['total_trades'] if trade_row else 0,
                "total_volume_today": 0.0,  # Would need to calculate from trades
                "active_positions": trade_row['active_positions'] if trade_row else 0,
                "open_orders": 0
            },
            "ai": {
                "predictions_today": ai_row['predictions_today'] if ai_row else 0,
                "active_models": ai_row['active_models'] if ai_row else 0,
                "avg_confidence": ai_row['avg_confidence'] if ai_row else 0.0,
                "accuracy_24h": 0.0  # Would need to calculate
            },
            "performance": {
                "uptime_percent": 99.95,
                "avg_api_latency_ms": 45,
                "database_queries_today": 0,
                "websocket_connections": 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # Data Writing Operations (for seeding/testing)
    # ========================================================================

    async def add_performance_snapshot(
        self,
        total_equity: float,
        cash_balance: float,
        invested_value: float,
        total_pnl: float,
        total_pnl_percent: float,
        **kwargs
    ) -> str:
        """Add a performance snapshot"""

        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        query = """
            INSERT INTO performance_snapshots (
                id, timestamp, total_equity, cash_balance, invested_value,
                total_pnl, total_pnl_percent, daily_pnl, daily_pnl_percent,
                sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_percent,
                total_trades, win_rate, profit_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        await self.connection.execute(query, (
            snapshot_id, timestamp, total_equity, cash_balance, invested_value,
            total_pnl, total_pnl_percent,
            kwargs.get('daily_pnl', 0.0),
            kwargs.get('daily_pnl_percent', 0.0),
            kwargs.get('sharpe_ratio', 0.0),
            kwargs.get('sortino_ratio', 0.0),
            kwargs.get('max_drawdown', 0.0),
            kwargs.get('max_drawdown_percent', 0.0),
            kwargs.get('total_trades', 0),
            kwargs.get('win_rate', 0.0),
            kwargs.get('profit_factor', 0.0)
        ))

        await self.connection.commit()
        return snapshot_id

    async def add_trade_event(
        self,
        event_type: str,
        symbol: str,
        data: Dict[str, Any],
        source: str = "trading_engine"
    ) -> str:
        """Add a trade event to the feed"""

        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        query = """
            INSERT INTO trade_feed (id, timestamp, event_type, symbol, data, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        await self.connection.execute(query, (
            event_id, timestamp, event_type, symbol,
            json.dumps(data), source
        ))

        await self.connection.commit()
        return event_id

    async def add_ai_decision(
        self,
        model_name: str,
        symbol: str,
        prediction: Dict[str, Any],
        features: Dict[str, Any],
        reasoning: str,
        confidence_score: float
    ) -> str:
        """Add an AI decision/prediction"""

        decision_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        query = """
            INSERT INTO ai_decisions (
                id, timestamp, symbol, model_name,
                prediction, features, reasoning, confidence_score,
                outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        await self.connection.execute(query, (
            decision_id, timestamp, symbol, model_name,
            json.dumps(prediction), json.dumps(features),
            reasoning, confidence_score, 'pending'
        ))

        await self.connection.commit()
        return decision_id


# Global database instance
_db_instance: Optional[TransparencyDB] = None


async def get_db() -> TransparencyDB:
    """Get or create database instance (dependency injection for FastAPI)"""
    global _db_instance

    if _db_instance is None:
        _db_instance = TransparencyDB()
        await _db_instance.connect()

    return _db_instance


async def close_db():
    """Close database connection"""
    global _db_instance

    if _db_instance is not None:
        await _db_instance.disconnect()
        _db_instance = None
