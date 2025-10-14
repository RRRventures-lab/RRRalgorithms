from ..positions import PositionManager
from datetime import datetime, timedelta
from functools import lru_cache
from src.database import get_db, Client
from typing import Dict, List, Optional
import asyncio
import logging
import numpy as np
import uuid


"""
Portfolio Manager
Aggregates positions, calculates portfolio metrics, and manages snapshots
"""




logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Portfolio Manager
    Manages overall portfolio state, metrics, and risk analytics
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        position_manager: PositionManager,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize Portfolio Manager

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            position_manager: Position manager instance
            initial_capital: Initial portfolio capital
        """
        self.db: Client = get_db()
        self.position_manager = position_manager
        self.initial_capital = initial_capital

        # Portfolio state
        self.portfolio = {
            "total_value": initial_capital,
            "cash": initial_capital,
            "equity": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "total_return_pct": 0.0,
            "daily_return_pct": 0.0,
        }

        # Snapshot tracking
        self.last_snapshot_time = None
        self.snapshot_interval_seconds = 60  # Take snapshot every minute

        logger.info(f"Initialized PortfolioManager with initial capital: {initial_capital}")

    async def update_portfolio(self, market_prices: Dict[str, float]) -> Dict:
        """
        Update portfolio state with current market prices

        Args:
            market_prices: Dict of symbol -> current_price

        Returns:
            Updated portfolio dictionary
        """
        try:
            # Get all open positions
            positions = await self.position_manager.get_all_open_positions()

            # Calculate total equity from positions
            total_equity = 0.0
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0

            for position in positions:
                symbol = position["symbol"]
                current_price = market_prices.get(symbol)

                if current_price is None:
                    logger.warning(f"No market price for {symbol}, using cached price")
                    current_price = position["current_price"]

                # Update position price
                await self.position_manager.update_position_price(
                    position["position_id"], current_price
                )

                # Calculate position value
                position_value = position["quantity"] * current_price
                total_equity += position_value
                total_unrealized_pnl += position["unrealized_pnl"]
                total_realized_pnl += position["realized_pnl"]

            # Update portfolio metrics
            self.portfolio["equity"] = total_equity
            self.portfolio["total_value"] = self.portfolio["cash"] + total_equity
            self.portfolio["total_pnl"] = total_realized_pnl + total_unrealized_pnl
            self.portfolio["total_return_pct"] = (
                (self.portfolio["total_value"] - self.initial_capital)
                / self.initial_capital
                * 100
            )

            logger.debug(
                f"Portfolio updated: Value={self.portfolio['total_value']:.2f}, "
                f"P&L={self.portfolio['total_pnl']:.2f}, "
                f"Return={self.portfolio['total_return_pct']:.2f}%"
            )

            return self.portfolio

        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}", exc_info=True)
            raise

    async def take_snapshot(self) -> Dict:
        """
        Take a snapshot of current portfolio state and save to database

        Returns:
            Snapshot dictionary
        """
        try:
            timestamp = datetime.utcnow()

            # Get all positions
            positions = await self.position_manager.get_all_open_positions()

            # Build snapshot
            snapshot = {
                "snapshot_id": str(uuid.uuid4()),
                "timestamp": timestamp.isoformat(),
                "total_value": self.portfolio["total_value"],
                "cash": self.portfolio["cash"],
                "equity": self.portfolio["equity"],
                "total_pnl": self.portfolio["total_pnl"],
                "daily_pnl": self.portfolio["daily_pnl"],
                "total_return_pct": self.portfolio["total_return_pct"],
                "daily_return_pct": self.portfolio["daily_return_pct"],
                "num_positions": len(positions),
                "positions": [
                    {
                        "symbol": p["symbol"],
                        "quantity": p["quantity"],
                        "value": p["quantity"] * p["current_price"],
                        "pnl": p["total_pnl"],
                    }
                    for p in positions
                ],
                "metrics": await self.calculate_metrics(),
            }

            # Save to database
            result = self.db.table("portfolio_snapshots").insert(snapshot).execute()
            if not result.data:
                raise Exception("Failed to insert portfolio snapshot")

            self.last_snapshot_time = timestamp

            logger.info(
                f"Portfolio snapshot saved: Value={snapshot['total_value']:.2f}, "
                f"Positions={snapshot['num_positions']}"
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to take portfolio snapshot: {e}", exc_info=True)
            raise

    async def calculate_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics

        Returns:
            Dict with various metrics (Sharpe ratio, max drawdown, etc.)
        """
        try:
            # Get recent snapshots for calculations
            snapshots = await self.get_recent_snapshots(days=30)

            if len(snapshots) < 2:
                return {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                }

            # Calculate returns
            returns = []
            values = [s["total_value"] for s in snapshots]

            for i in range(1, len(values)):
                ret = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(ret)

            returns_array = np.array(returns)

            # Calculate Sharpe ratio (annualized, assuming daily snapshots)
            if len(returns) > 0 and np.std(returns_array) > 0:
                sharpe_ratio = (
                    np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0.0

            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            max_drawdown_pct = max_drawdown * 100

            # Calculate win rate from closed positions
            closed_positions = await self.position_manager.get_position_history(
                limit=100
            )
            closed_positions = [p for p in closed_positions if p["status"] == "closed"]

            wins = [p for p in closed_positions if p["realized_pnl"] > 0]
            losses = [p for p in closed_positions if p["realized_pnl"] < 0]

            win_rate = len(wins) / len(closed_positions) * 100 if closed_positions else 0.0
            avg_win = (
                np.mean([p["realized_pnl"] for p in wins]) if wins else 0.0
            )
            avg_loss = (
                np.mean([abs(p["realized_pnl"]) for p in losses]) if losses else 0.0
            )

            # Profit factor
            total_wins = sum([p["realized_pnl"] for p in wins])
            total_losses = sum([abs(p["realized_pnl"]) for p in losses])
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            metrics = {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "max_drawdown_pct": float(max_drawdown_pct),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
                "total_trades": len(closed_positions),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}", exc_info=True)
            return {}

    async def get_recent_snapshots(self, days: int = 7) -> List[Dict]:
        """
        Get recent portfolio snapshots

        Args:
            days: Number of days to look back

        Returns:
            List of snapshots
        """
        try:
            start_time = datetime.utcnow() - timedelta(days=days)

            result = (
                self.db.table("portfolio_snapshots")
                .select("*")
                .gte("timestamp", start_time.isoformat())
                .order("timestamp", desc=False)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get recent snapshots: {e}", exc_info=True)
            return []

    async def update_cash_balance(self, amount: float, reason: str = "trade"):
        """
        Update cash balance

        Args:
            amount: Amount to add (positive) or subtract (negative)
            reason: Reason for update
        """
        self.portfolio["cash"] += amount
        logger.info(f"Cash balance updated: {amount:+.2f} ({reason}), new balance: {self.portfolio['cash']:.2f}")

    async def check_risk_limits(self, max_position_size_pct: float = 0.20, max_daily_loss_pct: float = 0.05) -> Dict:
        """
        Check if portfolio is within risk limits

        Args:
            max_position_size_pct: Maximum position size as % of portfolio
            max_daily_loss_pct: Maximum daily loss as % of portfolio

        Returns:
            Dict with risk check results
        """
        try:
            positions = await self.position_manager.get_all_open_positions()

            # Check position size limits
            max_position_value = 0.0
            max_position_symbol = None

            for position in positions:
                position_value = position["quantity"] * position["current_price"]
                if position_value > max_position_value:
                    max_position_value = position_value
                    max_position_symbol = position["symbol"]

            max_position_pct = (
                max_position_value / self.portfolio["total_value"] * 100
                if self.portfolio["total_value"] > 0
                else 0.0
            )

            position_limit_exceeded = max_position_pct > (max_position_size_pct * 100)

            # Check daily loss limit
            daily_loss_pct = abs(self.portfolio["daily_return_pct"])
            daily_loss_limit_exceeded = (
                self.portfolio["daily_return_pct"] < 0
                and daily_loss_pct > (max_daily_loss_pct * 100)
            )

            return {
                "within_limits": not (position_limit_exceeded or daily_loss_limit_exceeded),
                "position_limit_exceeded": position_limit_exceeded,
                "max_position_pct": max_position_pct,
                "max_position_symbol": max_position_symbol,
                "max_allowed_position_pct": max_position_size_pct * 100,
                "daily_loss_limit_exceeded": daily_loss_limit_exceeded,
                "daily_loss_pct": daily_loss_pct,
                "max_allowed_daily_loss_pct": max_daily_loss_pct * 100,
            }

        except Exception as e:
            logger.error(f"Failed to check risk limits: {e}", exc_info=True)
            return {"within_limits": False, "error": str(e)}

    async def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary

        Returns:
            Dict with portfolio summary
        """
        try:
            # Optimized: Parallel execution
positions, metrics = await asyncio.gather(
    self.position_manager.get_all_open_positions(),
    self.calculate_metrics()
)

            return {
                "portfolio": self.portfolio,
                "positions": positions,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}", exc_info=True)
            return {}

    async def start_snapshot_loop(self):
        """Start background loop to take periodic snapshots"""
        logger.info(f"Starting snapshot loop (interval: {self.snapshot_interval_seconds}s)")

        while True:
            try:
                await asyncio.sleep(self.snapshot_interval_seconds)
                await self.take_snapshot()
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retrying
