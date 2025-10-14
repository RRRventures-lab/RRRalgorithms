from ..oms import OrderManager
from ..portfolio import PortfolioManager
from ..positions import PositionManager
from datetime import datetime, timedelta
from supabase import create_client, Client
from typing import Dict, List, Optional
import asyncio
import logging
import uuid

"""
Strategy Executor
Reads trading signals and executes trades via OMS
"""




logger = logging.getLogger(__name__)


class StrategyExecutor:
    """
    Strategy Executor
    Monitors trading signals and executes trades according to strategy rules
    """

    def __init__(
        self,
        order_manager: OrderManager,
        position_manager: PositionManager,
        portfolio_manager: PortfolioManager,
        supabase_url: str,
        supabase_key: str,
        max_position_size: float = 0.20,
        max_daily_loss: float = 0.05,
    ):
        """
        Initialize Strategy Executor

        Args:
            order_manager: Order manager instance
            position_manager: Position manager instance
            portfolio_manager: Portfolio manager instance
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            max_position_size: Max position size as % of portfolio
            max_daily_loss: Max daily loss as % of portfolio
        """
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.portfolio_manager = portfolio_manager
        self.db: Client = create_client(supabase_url, supabase_key)

        # Risk limits
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss

        # State
        self.last_processed_signal_time = None
        self.processed_signals = set()  # Track processed signal IDs

        logger.info(
            f"Initialized StrategyExecutor with risk limits: "
            f"max_position={max_position_size*100}%, max_daily_loss={max_daily_loss*100}%"
        )

    async def process_signals(self) -> List[Dict]:
        """
        Process new trading signals from database

        Returns:
            List of execution results
        """
        try:
            # Get unprocessed signals
            signals = await self._get_new_signals()

            if not signals:
                logger.debug("No new signals to process")
                return []

            logger.info(f"Processing {len(signals)} new signals")

            results = []

            for signal in signals:
                try:
                    result = await self._execute_signal(signal)
                    results.append(result)

                    # Mark signal as processed
                    await self._mark_signal_processed(signal["signal_id"], result)

                except Exception as e:
                    logger.error(
                        f"Failed to execute signal {signal['signal_id']}: {e}",
                        exc_info=True,
                    )
                    await self._mark_signal_failed(signal["signal_id"], str(e))

            return results

        except Exception as e:
            logger.error(f"Failed to process signals: {e}", exc_info=True)
            return []

    async def _execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a single trading signal

        Args:
            signal: Signal dictionary from database

        Returns:
            Execution result
        """
        signal_id = signal["signal_id"]
        symbol = signal["symbol"]
        action = signal["action"]  # "buy", "sell", "close"
        confidence = signal.get("confidence", 0.5)

        logger.info(
            f"Executing signal {signal_id}: {action} {symbol} (confidence: {confidence:.2f})"
        )

        # Check risk limits
        risk_check = await self.portfolio_manager.check_risk_limits(
            self.max_position_size, self.max_daily_loss
        )

        if not risk_check["within_limits"]:
            logger.warning(
                f"Signal {signal_id} blocked by risk limits: {risk_check}"
            )
            return {
                "signal_id": signal_id,
                "success": False,
                "reason": "risk_limits_exceeded",
                "risk_check": risk_check,
            }

        # Execute based on action
        if action == "buy":
            return await self._execute_buy(signal)
        elif action == "sell":
            return await self._execute_sell(signal)
        elif action == "close":
            return await self._execute_close(signal)
        else:
            logger.error(f"Unknown action: {action}")
            return {
                "signal_id": signal_id,
                "success": False,
                "reason": "unknown_action",
            }

    async def _execute_buy(self, signal: Dict) -> Dict:
        """Execute buy signal"""
        signal_id = signal["signal_id"]
        symbol = signal["symbol"]
        strategy_id = signal.get("strategy_id")

        # Calculate position size based on portfolio value and confidence
        portfolio_value = self.portfolio_manager.portfolio["total_value"]
        confidence = signal.get("confidence", 0.5)

        # Position size: min(max_position_size, confidence * max_position_size)
        position_size_pct = min(self.max_position_size, confidence * self.max_position_size)
        position_value = portfolio_value * position_size_pct

        # Get current price (from signal or fetch from market)
        current_price = signal.get("price")
        if not current_price:
            # In production, fetch from market data feed
            logger.warning(f"No price in signal, using placeholder")
            current_price = 50000.0  # Placeholder

        # Calculate quantity
        quantity = position_value / current_price

        # Create buy order
        try:
            order = await self.order_manager.create_order(
                symbol=symbol,
                side="buy",
                order_type="market",
                quantity=quantity,
                strategy_id=strategy_id,
                signal_id=signal_id,
                metadata={"signal": signal},
            )

            # If order filled, open position
            if order["status"] == "filled":
                position = await self.position_manager.open_position(
                    symbol=symbol,
                    side="long",
                    quantity=quantity,
                    entry_price=order["average_fill_price"],
                    order_id=order["order_id"],
                    strategy_id=strategy_id,
                    metadata={"signal_id": signal_id},
                )

                # Update cash balance
                cost = quantity * order["average_fill_price"]
                await self.portfolio_manager.update_cash_balance(-cost, "buy_order")

                logger.info(
                    f"Buy executed: {quantity:.6f} {symbol} @ {order['average_fill_price']:.2f}"
                )

                return {
                    "signal_id": signal_id,
                    "success": True,
                    "action": "buy",
                    "order": order,
                    "position": position,
                }

            else:
                logger.warning(
                    f"Buy order not filled: {order['status']}"
                )
                return {
                    "signal_id": signal_id,
                    "success": False,
                    "reason": "order_not_filled",
                    "order": order,
                }

        except Exception as e:
            logger.error(f"Failed to execute buy: {e}", exc_info=True)
            return {
                "signal_id": signal_id,
                "success": False,
                "reason": "execution_error",
                "error": str(e),
            }

    async def _execute_sell(self, signal: Dict) -> Dict:
        """Execute sell signal (short selling - for now just close longs)"""
        # For simplicity, treat sell as close position
        return await self._execute_close(signal)

    async def _execute_close(self, signal: Dict) -> Dict:
        """Execute close signal"""
        signal_id = signal["signal_id"]
        symbol = signal["symbol"]

        # Get open position for this symbol
        position = await self.position_manager.get_position_by_symbol(symbol)

        if not position:
            logger.warning(f"No open position for {symbol}, nothing to close")
            return {
                "signal_id": signal_id,
                "success": False,
                "reason": "no_open_position",
            }

        # Get current price
        current_price = signal.get("price", position["current_price"])

        # Create sell order
        try:
            order = await self.order_manager.create_order(
                symbol=symbol,
                side="sell",
                order_type="market",
                quantity=position["quantity"],
                strategy_id=position.get("strategy_id"),
                signal_id=signal_id,
                metadata={"signal": signal, "position_id": position["position_id"]},
            )

            # If order filled, close position
            if order["status"] == "filled":
                closed_position = await self.position_manager.close_position(
                    position_id=position["position_id"],
                    exit_price=order["average_fill_price"],
                    order_id=order["order_id"],
                )

                # Update cash balance
                proceeds = position["quantity"] * order["average_fill_price"]
                await self.portfolio_manager.update_cash_balance(proceeds, "sell_order")

                logger.info(
                    f"Position closed: {position['quantity']:.6f} {symbol} @ {order['average_fill_price']:.2f}, "
                    f"P&L: {closed_position['realized_pnl']:.2f}"
                )

                return {
                    "signal_id": signal_id,
                    "success": True,
                    "action": "close",
                    "order": order,
                    "position": closed_position,
                    "pnl": closed_position["realized_pnl"],
                }

            else:
                logger.warning(f"Sell order not filled: {order['status']}")
                return {
                    "signal_id": signal_id,
                    "success": False,
                    "reason": "order_not_filled",
                    "order": order,
                }

        except Exception as e:
            logger.error(f"Failed to execute close: {e}", exc_info=True)
            return {
                "signal_id": signal_id,
                "success": False,
                "reason": "execution_error",
                "error": str(e),
            }

    async def _get_new_signals(self, limit: int = 10) -> List[Dict]:
        """
        Get new unprocessed signals from database

        Args:
            limit: Maximum number of signals to fetch

        Returns:
            List of signal dictionaries
        """
        try:
            # Query for signals that haven't been processed
            # Status should be 'active' and not in processed_signals set
            query = (
                self.db.table("trading_signals")
                .select("*")
                .eq("status", "active")
                .order("created_at", desc=False)
                .limit(limit)
            )

            result = query.execute()
            signals = result.data or []

            # Filter out already processed signals
            new_signals = [
                s for s in signals if s["signal_id"] not in self.processed_signals
            ]

            return new_signals

        except Exception as e:
            logger.error(f"Failed to get new signals: {e}", exc_info=True)
            return []

    async def _mark_signal_processed(self, signal_id: str, result: Dict):
        """Mark signal as processed in database"""
        try:
            update = {
                "status": "processed",
                "processed_at": datetime.utcnow().isoformat(),
                "execution_result": result,
            }

            self.db.table("trading_signals").update(update).eq(
                "signal_id", signal_id
            ).execute()

            self.processed_signals.add(signal_id)
            logger.debug(f"Marked signal {signal_id} as processed")

        except Exception as e:
            logger.error(f"Failed to mark signal processed: {e}", exc_info=True)

    async def _mark_signal_failed(self, signal_id: str, error: str):
        """Mark signal as failed in database"""
        try:
            update = {
                "status": "failed",
                "processed_at": datetime.utcnow().isoformat(),
                "execution_result": {"success": False, "error": error},
            }

            self.db.table("trading_signals").update(update).eq(
                "signal_id", signal_id
            ).execute()

            logger.debug(f"Marked signal {signal_id} as failed")

        except Exception as e:
            logger.error(f"Failed to mark signal failed: {e}", exc_info=True)

    async def _log_event(
        self, event_type: str, severity: str, message: str, metadata: Optional[Dict] = None
    ):
        """Log event to system_events table"""
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "component": "strategy_executor",
                "severity": severity,
                "message": message,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }
            self.db.table("system_events").insert(event).execute()
        except Exception as e:
            logger.error(f"Failed to log event: {e}", exc_info=True)
