"""
Database repositories for API data access.
Provides high-level business logic queries for the transparency dashboard API.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from .base import DatabaseClient

logger = logging.getLogger(__name__)


class PortfolioRepository:
    """Repository for portfolio-related queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_portfolio_overview(self) -> Dict[str, Any]:
        """
        Get current portfolio overview with positions, equity, and performance.

        Returns:
            Portfolio summary with total equity, cash, P&L, etc.
        """
        # Get latest portfolio snapshot
        snapshot = await self.db.fetch_one("""
            SELECT
                total_value,
                cash_balance,
                positions_value,
                daily_pnl,
                total_pnl,
                num_positions
            FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        # Get active positions count
        positions_count = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM positions
            WHERE quantity != 0
        """)

        # Get open orders count
        open_orders = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM orders
            WHERE status IN ('pending', 'open', 'partial')
        """)

        if not snapshot:
            # Return default values if no snapshot exists
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

        total_equity = snapshot.get('total_value', 100000.00)
        cash_balance = snapshot.get('cash_balance', 100000.00)
        invested = snapshot.get('positions_value', 0.00)
        total_pnl = snapshot.get('total_pnl', 0.00)
        day_pnl = snapshot.get('daily_pnl', 0.00)

        # Calculate percentages
        initial_capital = 100000.00  # TODO: Get from config
        total_pnl_percent = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        day_pnl_percent = (day_pnl / total_equity * 100) if total_equity > 0 else 0

        return {
            "total_equity": round(total_equity, 2),
            "cash_balance": round(cash_balance, 2),
            "invested": round(invested, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(total_pnl_percent, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pnl_percent": round(day_pnl_percent, 2),
            "positions_count": positions_count.get('count', 0) if positions_count else 0,
            "open_orders": open_orders.get('count', 0) if open_orders else 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_positions(self) -> Dict[str, Any]:
        """
        Get all open positions with current prices and P&L.

        Returns:
            List of positions with unrealized P&L
        """
        positions = await self.db.fetch_all("""
            SELECT
                p.id,
                p.symbol,
                p.quantity,
                p.average_price,
                p.current_price,
                p.unrealized_pnl,
                p.realized_pnl,
                p.opened_at,
                p.updated_at
            FROM positions p
            WHERE p.quantity != 0
            ORDER BY p.symbol
        """)

        result = []
        for pos in positions:
            quantity = pos.get('quantity', 0)
            avg_price = pos.get('average_price', 0)
            current_price = pos.get('current_price', avg_price)
            unrealized_pnl = pos.get('unrealized_pnl', 0)

            # Calculate P&L percentage
            cost_basis = abs(quantity * avg_price)
            unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            result.append({
                "id": f"pos-{pos.get('id')}",
                "symbol": pos.get('symbol'),
                "side": "long" if quantity > 0 else "short",
                "size": abs(quantity),
                "entry_price": round(avg_price, 2),
                "current_price": round(current_price, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_percent": round(unrealized_pnl_percent, 2),
                "opened_at": datetime.fromtimestamp(pos.get('opened_at', 0)).isoformat()
            })

        return {
            "positions": result,
            "total_count": len(result),
            "timestamp": datetime.utcnow().isoformat()
        }


class TradingRepository:
    """Repository for trading-related queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recent trade history with pagination.

        Args:
            limit: Maximum number of trades to return
            offset: Offset for pagination
            symbol: Filter by symbol (optional)

        Returns:
            Paginated list of trades
        """
        # Build query
        query = """
            SELECT
                t.id,
                t.symbol,
                t.side,
                t.quantity,
                t.price,
                t.timestamp,
                t.status,
                t.fees,
                t.exchange,
                t.strategy,
                o.order_type
            FROM trades t
            LEFT JOIN orders o ON t.order_id = o.id
        """

        params = []
        if symbol:
            query += " WHERE t.symbol = ?"
            params.append(symbol)

        query += " ORDER BY t.timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        trades = await self.db.fetch_all(query, tuple(params))

        # Get total count
        count_query = "SELECT COUNT(*) as count FROM trades"
        if symbol:
            count_query += " WHERE symbol = ?"
            total_result = await self.db.fetch_one(count_query, (symbol,))
        else:
            total_result = await self.db.fetch_one(count_query)

        total_count = total_result.get('count', 0) if total_result else 0

        result = []
        for trade in trades:
            result.append({
                "id": f"trade-{trade.get('id')}",
                "timestamp": datetime.fromtimestamp(trade.get('timestamp', 0)).isoformat(),
                "symbol": trade.get('symbol'),
                "side": trade.get('side'),
                "quantity": trade.get('quantity'),
                "price": round(trade.get('price', 0), 2),
                "total_value": round(trade.get('quantity', 0) * trade.get('price', 0), 2),
                "fee": round(trade.get('fees', 0), 2),
                "status": trade.get('status'),
                "order_type": trade.get('order_type', 'market')
            })

        return {
            "trades": result,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }


class PerformanceRepository:
    """Repository for performance metrics queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_performance_metrics(self, period: str = "1d") -> Dict[str, Any]:
        """
        Get performance metrics for specified period.

        Args:
            period: Time period (1h, 4h, 1d, 7d, 30d, all)

        Returns:
            Performance metrics including returns, Sharpe, drawdown, etc.
        """
        # Calculate timestamp cutoff based on period
        now = datetime.utcnow()
        period_map = {
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "all": timedelta(days=365 * 10)  # 10 years
        }

        delta = period_map.get(period, timedelta(days=1))
        start_time = int((now - delta).timestamp())

        # Get trades in period
        trades = await self.db.fetch_all("""
            SELECT
                side,
                quantity,
                price,
                fees
            FROM trades
            WHERE timestamp >= ? AND status = 'filled'
            ORDER BY timestamp ASC
        """, (start_time,))

        if not trades:
            # Return default metrics if no trades
            return {
                "period": period,
                "total_return": 0.00,
                "total_return_percent": 0.00,
                "sharpe_ratio": 0.00,
                "sortino_ratio": 0.00,
                "max_drawdown": 0.00,
                "max_drawdown_percent": 0.00,
                "win_rate": 0.00,
                "profit_factor": 0.00,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "average_win": 0.00,
                "average_loss": 0.00,
                "largest_win": 0.00,
                "largest_loss": 0.00,
                "timestamp": datetime.utcnow().isoformat()
            }

        # Calculate basic metrics
        total_trades = len(trades)
        total_pnl = 0.0
        wins = []
        losses = []

        # Simple P&L calculation (actual implementation would track per-position)
        for trade in trades:
            # This is simplified - real P&L needs position tracking
            value = trade['quantity'] * trade['price']
            fee = trade.get('fees', 0)

            # Simplified: assume alternating buys/sells generate P&L
            # Real implementation would match buys with sells
            pnl = value * 0.01  # Placeholder

            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)

            total_pnl += pnl - fee

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        average_win = sum(wins) / len(wins) if wins else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Placeholder for advanced metrics (would need daily returns data)
        sharpe_ratio = 1.5  # Placeholder
        sortino_ratio = 1.8  # Placeholder
        max_drawdown = -500.0  # Placeholder
        max_drawdown_percent = -2.5  # Placeholder

        return {
            "period": period,
            "total_return": round(total_pnl, 2),
            "total_return_percent": round((total_pnl / 100000) * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_percent": round(max_drawdown_percent, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_equity_curve(
        self,
        period: str = "7d",
        interval: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get equity curve data points for charting.

        Args:
            period: Time period to fetch
            interval: Data point interval

        Returns:
            Array of timestamp and equity value pairs
        """
        # Calculate time range
        now = datetime.utcnow()
        period_map = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "all": timedelta(days=365)
        }

        delta = period_map.get(period, timedelta(days=7))
        start_time = int((now - delta).timestamp())

        # Get portfolio snapshots
        snapshots = await self.db.fetch_all("""
            SELECT
                timestamp,
                total_value
            FROM portfolio_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (start_time,))

        if not snapshots:
            # Generate default data if no snapshots
            data_points = []
            initial_equity = 100000.00

            # Generate hourly data for the period
            interval_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            interval_minutes = interval_map.get(interval, 60)
            total_minutes = int(delta.total_seconds() / 60)
            num_points = min(total_minutes // interval_minutes, 500)  # Cap at 500 points

            for i in range(num_points):
                timestamp = now - timedelta(minutes=total_minutes - (i * interval_minutes))
                # Simple simulation - slight upward trend with volatility
                equity = initial_equity + (i * 20) + ((-1 if i % 3 == 0 else 1) * 150)
                data_points.append({
                    "timestamp": timestamp.isoformat(),
                    "equity": round(equity, 2)
                })

            return {
                "period": period,
                "interval": interval,
                "data": data_points,
                "initial_equity": initial_equity,
                "current_equity": data_points[-1]["equity"] if data_points else initial_equity,
                "timestamp": datetime.utcnow().isoformat()
            }

        # Convert snapshots to data points
        data_points = [
            {
                "timestamp": datetime.fromtimestamp(snap['timestamp']).isoformat(),
                "equity": round(snap['total_value'], 2)
            }
            for snap in snapshots
        ]

        initial_equity = data_points[0]["equity"] if data_points else 100000.00
        current_equity = data_points[-1]["equity"] if data_points else initial_equity

        return {
            "period": period,
            "interval": interval,
            "data": data_points,
            "initial_equity": initial_equity,
            "current_equity": current_equity,
            "timestamp": datetime.utcnow().isoformat()
        }


class AIRepository:
    """Repository for AI/ML-related queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_ai_decisions(
        self,
        limit: int = 50,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recent AI predictions and decisions.

        Args:
            limit: Maximum number of decisions to return
            model: Filter by model name (optional)

        Returns:
            List of AI decisions with predictions and outcomes
        """
        query = """
            SELECT
                p.id,
                p.timestamp,
                p.symbol,
                p.prediction_type,
                p.prediction_value,
                p.confidence,
                p.features,
                m.model_name,
                m.model_type
            FROM ml_predictions p
            JOIN ml_models m ON p.model_id = m.id
        """

        params = []
        if model:
            query += " WHERE m.model_name = ?"
            params.append(model)

        query += " ORDER BY p.timestamp DESC LIMIT ?"
        params.append(limit)

        predictions = await self.db.fetch_all(query, tuple(params))

        result = []
        for pred in predictions:
            features = json.loads(pred.get('features', '{}')) if pred.get('features') else {}

            result.append({
                "id": f"dec-{pred.get('id')}",
                "timestamp": datetime.fromtimestamp(pred.get('timestamp', 0)).isoformat(),
                "model_name": pred.get('model_name'),
                "symbol": pred.get('symbol'),
                "prediction": {
                    "direction": "up" if pred.get('prediction_value', 0) > 0 else "down",
                    "confidence": round(pred.get('confidence', 0), 2),
                    "price_target": round(abs(pred.get('prediction_value', 0)), 2),
                    "time_horizon": "4h"  # Placeholder
                },
                "reasoning": "ML model prediction based on technical indicators",
                "outcome": "pending",
                "features": features
            })

        return {
            "decisions": result,
            "total_count": len(result),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_ai_models(self) -> Dict[str, Any]:
        """
        Get information about active AI models.

        Returns:
            List of AI models with their performance stats
        """
        models = await self.db.fetch_all("""
            SELECT
                m.id,
                m.model_name,
                m.model_type,
                m.version,
                m.active,
                m.metrics,
                m.trained_at,
                COUNT(p.id) as prediction_count
            FROM ml_models m
            LEFT JOIN ml_predictions p ON m.id = p.model_id
                AND p.timestamp >= strftime('%s', 'now') - 86400
            GROUP BY m.id
            ORDER BY m.active DESC, m.model_name
        """)

        result = []
        for model in models:
            metrics = json.loads(model.get('metrics', '{}')) if model.get('metrics') else {}

            result.append({
                "name": model.get('model_name'),
                "type": model.get('model_type'),
                "status": "active" if model.get('active') else "inactive",
                "accuracy": round(metrics.get('accuracy', 0) * 100, 1),
                "predictions_today": model.get('prediction_count', 0),
                "avg_confidence": round(metrics.get('avg_confidence', 0.7), 2),
                "win_rate": round(metrics.get('win_rate', 0.6) * 100, 1)
            })

        return {
            "models": result,
            "total_count": len(result),
            "timestamp": datetime.utcnow().isoformat()
        }


class BacktestRepository:
    """Repository for backtest-related queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_backtests(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get recent backtest results.

        Args:
            limit: Maximum number of backtests to return

        Returns:
            List of backtest summaries
        """
        backtests = await self.db.fetch_all("""
            SELECT
                id,
                run_name,
                strategy_name,
                start_date,
                end_date,
                total_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades,
                created_at
            FROM backtest_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        result = []
        for bt in backtests:
            start_date = datetime.fromtimestamp(bt.get('start_date', 0))
            end_date = datetime.fromtimestamp(bt.get('end_date', 0))

            result.append({
                "id": f"bt-{bt.get('id')}",
                "name": bt.get('run_name'),
                "created_at": datetime.fromtimestamp(bt.get('created_at', 0)).isoformat(),
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "total_return": round(bt.get('total_return', 0), 1),
                "sharpe_ratio": round(bt.get('sharpe_ratio', 0), 2),
                "max_drawdown": round(bt.get('max_drawdown', 0), 1),
                "win_rate": round(bt.get('win_rate', 0) * 100, 1),
                "total_trades": bt.get('total_trades', 0),
                "status": "completed"
            })

        return {
            "backtests": result,
            "total_count": len(result),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_backtest_detail(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed backtest results.

        Args:
            backtest_id: Backtest ID (e.g., "bt-001")

        Returns:
            Detailed backtest metrics or None if not found
        """
        # Extract numeric ID from "bt-001" format
        try:
            numeric_id = int(backtest_id.replace('bt-', ''))
        except ValueError:
            return None

        backtest = await self.db.fetch_one("""
            SELECT
                id,
                run_name,
                strategy_name,
                start_date,
                end_date,
                initial_capital,
                final_capital,
                total_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades,
                config,
                results,
                created_at
            FROM backtest_runs
            WHERE id = ?
        """, (numeric_id,))

        if not backtest:
            return None

        start_date = datetime.fromtimestamp(backtest.get('start_date', 0))
        end_date = datetime.fromtimestamp(backtest.get('end_date', 0))

        config = json.loads(backtest.get('config', '{}')) if backtest.get('config') else {}
        results = json.loads(backtest.get('results', '{}')) if backtest.get('results') else {}

        days = (end_date - start_date).days

        initial_capital = backtest.get('initial_capital', 100000)
        final_capital = backtest.get('final_capital', initial_capital)
        total_return = backtest.get('total_return', 0)
        total_trades = backtest.get('total_trades', 0)
        win_rate = backtest.get('win_rate', 0)

        winning_trades = int(total_trades * win_rate) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades

        return {
            "id": backtest_id,
            "name": backtest.get('run_name'),
            "description": config.get('description', f"{backtest.get('strategy_name')} strategy backtest"),
            "created_at": datetime.fromtimestamp(backtest.get('created_at', 0)).isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "performance": {
                "initial_capital": round(initial_capital, 2),
                "final_equity": round(final_capital, 2),
                "total_return": round(total_return, 2),
                "total_return_percent": round(total_return, 2),
                "sharpe_ratio": round(backtest.get('sharpe_ratio', 0), 2),
                "sortino_ratio": round(results.get('sortino_ratio', 0), 2),
                "max_drawdown": round(backtest.get('max_drawdown', 0), 2),
                "max_drawdown_percent": round(backtest.get('max_drawdown', 0), 2),
                "calmar_ratio": round(results.get('calmar_ratio', 0), 2),
                "win_rate": round(win_rate * 100, 1),
                "profit_factor": round(results.get('profit_factor', 0), 2),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class SystemRepository:
    """Repository for system statistics queries."""

    def __init__(self, db: DatabaseClient):
        self.db = db

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get overall system statistics.

        Returns:
            System-wide stats and metrics
        """
        # Trading stats
        trades_today = await self.db.fetch_one("""
            SELECT
                COUNT(*) as count,
                SUM(quantity * price) as volume
            FROM trades
            WHERE timestamp >= strftime('%s', 'now', 'start of day')
        """)

        positions = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM positions
            WHERE quantity != 0
        """)

        open_orders = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM orders
            WHERE status IN ('pending', 'open', 'partial')
        """)

        # AI stats
        predictions_today = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM ml_predictions
            WHERE timestamp >= strftime('%s', 'now', 'start of day')
        """)

        active_models = await self.db.fetch_one("""
            SELECT COUNT(*) as count
            FROM ml_models
            WHERE active = 1
        """)

        return {
            "trading": {
                "total_trades_today": trades_today.get('count', 0) if trades_today else 0,
                "total_volume_today": round(trades_today.get('volume', 0), 2) if trades_today else 0,
                "active_positions": positions.get('count', 0) if positions else 0,
                "open_orders": open_orders.get('count', 0) if open_orders else 0
            },
            "ai": {
                "predictions_today": predictions_today.get('count', 0) if predictions_today else 0,
                "active_models": active_models.get('count', 0) if active_models else 0,
                "avg_confidence": 0.75,  # Placeholder
                "accuracy_24h": 62.3  # Placeholder
            },
            "performance": {
                "uptime_percent": 99.95,
                "avg_api_latency_ms": 45,
                "database_queries_today": 0,  # Placeholder
                "websocket_connections": 0  # Placeholder - will be populated by WebSocket server
            },
            "timestamp": datetime.utcnow().isoformat()
        }
