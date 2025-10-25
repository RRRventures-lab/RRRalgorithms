from dataclasses import dataclass
from datetime import datetime, time, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from src.database import get_db, Client
from typing import Dict, Optional, Tuple
import logging
import os


"""
Daily Loss Limiter

Tracks daily P&L and halts trading if maximum daily loss is reached.
This is a circuit breaker to prevent catastrophic losses.

Features:
- Track intraday P&L
- Compare against MAX_DAILY_LOSS limit
- Write critical alerts to system_events table
- Provide trading permission check
- Reset at market open
"""


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DailyPnL:
    """Daily P&L summary"""
    date: datetime
    starting_balance: float
    current_balance: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    pnl_pct: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    is_trading_halted: bool
    notes: str


class DailyLossLimiter:
    """
    Daily loss limiter with circuit breaker

    Monitors daily P&L and halts trading when loss limit is reached.
    Integrates with Supabase for position and trade data.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,  # 5% max daily loss
        warning_threshold_pct: float = 0.03,  # Warn at 3% loss
        enable_circuit_breaker: bool = True,
        market_open_time: time = time(9, 30),  # 9:30 AM
        market_close_time: time = time(16, 0)  # 4:00 PM
    ):
        """
        Initialize daily loss limiter

        Args:
            max_daily_loss_pct: Maximum daily loss as % of starting balance
            warning_threshold_pct: Warning threshold as % of starting balance
            enable_circuit_breaker: Enable automatic trading halt
            market_open_time: Market open time (for reset)
            market_close_time: Market close time
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.warning_threshold_pct = warning_threshold_pct
        self.enable_circuit_breaker = enable_circuit_breaker
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time

        # Trading halt flag
        self.is_trading_halted = False
        self.halt_reason = ""

        # Initialize Supabase client
        supabase_url = os.getenv("DATABASE_PATH")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("DATABASE_PATH and SUPABASE_ANON_KEY must be set")

        self.supabase: Client = get_db()

        # Cache for starting balance
        self._starting_balance: Optional[float] = None
        self._last_reset: Optional[datetime] = None

        logger.info(
            f"Daily Loss Limiter initialized: max_loss={max_daily_loss_pct:.1%}, "
            f"warning={warning_threshold_pct:.1%}, circuit_breaker={enable_circuit_breaker}"
        )

    @lru_cache(maxsize=128)

    def get_starting_balance(self) -> float:
        """
        Get starting balance for the day

        Uses portfolio_snapshots table to get balance at market open.
        Falls back to current balance if no snapshot available.

        Returns:
            Starting balance in dollars
        """
        try:
            # Check if we need to reset
            if self._should_reset():
                self._starting_balance = None
                self._last_reset = datetime.now()

            # Return cached value if available
            if self._starting_balance is not None:
                return self._starting_balance

            # Get today's date at market open
            today = datetime.now().date()
            market_open = datetime.combine(today, self.market_open_time)

            # Query portfolio snapshots
            response = (
                self.supabase.table("portfolio_snapshots")
                .select("total_value")
                .gte("timestamp", market_open.isoformat())
                .order("timestamp")
                .limit(1)
                .execute()
            )

            if response.data:
                self._starting_balance = float(response.data[0]["total_value"])
                logger.info(f"Starting balance for {today}: ${self._starting_balance:,.2f}")
            else:
                # Fallback to current portfolio value
                self._starting_balance = self.get_current_portfolio_value()
                logger.warning(
                    f"No snapshot found for {today}, using current value: ${self._starting_balance:,.2f}"
                )

            return self._starting_balance

        except Exception as e:
            logger.error(f"Error getting starting balance: {e}")
            # Return a safe default
            return 100000.0

    @lru_cache(maxsize=128)

    def get_current_portfolio_value(self) -> float:
        """
        Get current portfolio value

        Sums up all position values from positions table.

        Returns:
            Current portfolio value
        """
        try:
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if not positions:
                return 0.0

            total_value = sum(
                float(pos["quantity"]) * float(pos["current_price"])
                for pos in positions
            )

            return total_value

        except Exception as e:
            logger.error(f"Error getting current portfolio value: {e}")
            return 0.0

    @lru_cache(maxsize=128)

    def get_realized_pnl_today(self) -> Tuple[float, int, int]:
        """
        Get realized P&L from closed trades today

        Returns:
            Tuple of (realized_pnl, winning_trades, losing_trades)
        """
        try:
            # Get today's date at market open
            today = datetime.now().date()
            market_open = datetime.combine(today, self.market_open_time)

            # Query closed trades
            response = (
                self.supabase.table("trades")
                .select("pnl")
                .eq("status", "closed")
                .gte("closed_at", market_open.isoformat())
                .execute()
            )

            trades = response.data

            if not trades:
                return 0.0, 0, 0

            pnl_values = [float(t["pnl"]) for t in trades]
            realized_pnl = sum(pnl_values)
            winning_trades = len([p for p in pnl_values if p > 0])
            losing_trades = len([p for p in pnl_values if p < 0])

            return realized_pnl, winning_trades, losing_trades

        except Exception as e:
            logger.error(f"Error getting realized P&L: {e}")
            return 0.0, 0, 0

    @lru_cache(maxsize=128)

    def get_unrealized_pnl(self) -> float:
        """
        Get unrealized P&L from open positions

        Returns:
            Unrealized P&L in dollars
        """
        try:
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if not positions:
                return 0.0

            unrealized_pnl = sum(
                float(pos["unrealized_pnl"])
                for pos in positions
                if "unrealized_pnl" in pos
            )

            return unrealized_pnl

        except Exception as e:
            logger.error(f"Error getting unrealized P&L: {e}")
            return 0.0

    @lru_cache(maxsize=128)

    def get_daily_pnl(self) -> DailyPnL:
        """
        Calculate daily P&L summary

        Returns:
            DailyPnL object with all metrics
        """
        starting_balance = self.get_starting_balance()
        current_balance = self.get_current_portfolio_value()

        realized_pnl, winning_trades, losing_trades = self.get_realized_pnl_today()
        unrealized_pnl = self.get_unrealized_pnl()

        total_pnl = realized_pnl + unrealized_pnl
        pnl_pct = (total_pnl / starting_balance) if starting_balance > 0 else 0.0

        num_trades = winning_trades + losing_trades

        notes = []

        # Check against warning threshold
        if pnl_pct <= -self.warning_threshold_pct:
            notes.append(
                f"WARNING: Daily loss ({pnl_pct:.1%}) approaching limit ({self.max_daily_loss_pct:.1%})"
            )

        # Check against max loss
        if pnl_pct <= -self.max_daily_loss_pct:
            notes.append(
                f"CRITICAL: Daily loss limit ({self.max_daily_loss_pct:.1%}) EXCEEDED!"
            )

        return DailyPnL(
            date=datetime.now(),
            starting_balance=starting_balance,
            current_balance=current_balance,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            pnl_pct=pnl_pct,
            num_trades=num_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            is_trading_halted=self.is_trading_halted,
            notes="; ".join(notes) if notes else "P&L within limits"
        )

    def check_daily_loss(self) -> bool:
        """
        Check if trading can continue (not halted by daily loss limit)

        This is the main function called by the trading engine before each trade.

        Returns:
            True if trading is allowed, False if halted
        """
        # Check if we need to reset (new trading day)
        if self._should_reset():
            self.reset_daily_limits()

        # If already halted, stay halted until reset
        if self.is_trading_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return False

        # Get current P&L
        daily_pnl = self.get_daily_pnl()

        # Check against warning threshold
        if daily_pnl.pnl_pct <= -self.warning_threshold_pct:
            logger.warning(
                f"Daily loss warning: {daily_pnl.pnl_pct:.2%} "
                f"(${daily_pnl.total_pnl:,.2f})"
            )

            # Write warning event
            self._write_system_event(
                event_type="DAILY_LOSS_WARNING",
                severity="warning",
                message=f"Daily loss at {daily_pnl.pnl_pct:.2%} approaching limit",
                data={
                    "pnl_pct": daily_pnl.pnl_pct,
                    "total_pnl": daily_pnl.total_pnl,
                    "starting_balance": daily_pnl.starting_balance
                }
            )

        # Check against max loss
        if daily_pnl.pnl_pct <= -self.max_daily_loss_pct:
            if self.enable_circuit_breaker:
                self.halt_trading(
                    f"Daily loss limit exceeded: {daily_pnl.pnl_pct:.2%} "
                    f"(limit: {self.max_daily_loss_pct:.2%})"
                )
                return False
            else:
                logger.critical(
                    f"Daily loss limit exceeded but circuit breaker disabled: "
                    f"{daily_pnl.pnl_pct:.2%}"
                )

        return True

    def halt_trading(self, reason: str):
        """
        Halt all trading (circuit breaker)

        Args:
            reason: Reason for halt
        """
        self.is_trading_halted = True
        self.halt_reason = reason

        logger.critical(f"TRADING HALTED: {reason}")

        # Write critical event to system_events
        self._write_system_event(
            event_type="TRADING_HALTED",
            severity="critical",
            message=reason,
            data={
                "halted_at": datetime.now().isoformat(),
                "reason": reason
            }
        )

    def resume_trading(self):
        """Resume trading (manual override)"""
        self.is_trading_halted = False
        self.halt_reason = ""

        logger.info("Trading resumed (manual override)")

        self._write_system_event(
            event_type="TRADING_RESUMED",
            severity="info",
            message="Trading resumed by manual override",
            data={"resumed_at": datetime.now().isoformat()}
        )

    def reset_daily_limits(self):
        """Reset daily limits (called at market open)"""
        logger.info("Resetting daily limits for new trading day")

        self._starting_balance = None
        self.is_trading_halted = False
        self.halt_reason = ""
        self._last_reset = datetime.now()

        self._write_system_event(
            event_type="DAILY_LIMITS_RESET",
            severity="info",
            message="Daily loss limits reset for new trading day",
            data={"reset_at": datetime.now().isoformat()}
        )

    def _should_reset(self) -> bool:
        """Check if daily limits should be reset"""
        if self._last_reset is None:
            return True

        # Reset if it's a new trading day after market open
        now = datetime.now()
        last_reset_date = self._last_reset.date()
        current_date = now.date()

        if current_date > last_reset_date and now.time() >= self.market_open_time:
            return True

        return False

    def _write_system_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Write event to system_events table

        Args:
            event_type: Type of event
            severity: Event severity (info, warning, critical)
            message: Event message
            data: Additional event data
        """
        try:
            event = {
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "daily_loss_limiter",
                "data": data or {}
            }

            self.supabase.table("system_events").insert(event).execute()

            logger.debug(f"System event written: {event_type}")

        except Exception as e:
            logger.error(f"Error writing system event: {e}")


def main():
    """Example usage"""
    print("=" * 60)
    print("Daily Loss Limiter")
    print("=" * 60)

    try:
        # Initialize limiter
        limiter = DailyLossLimiter(
            max_daily_loss_pct=0.05,
            warning_threshold_pct=0.03,
            enable_circuit_breaker=True
        )

        # Get daily P&L
        print("\nCalculating daily P&L...")
        daily_pnl = limiter.get_daily_pnl()

        print(f"\nDate: {daily_pnl.date.strftime('%Y-%m-%d')}")
        print(f"Starting Balance: ${daily_pnl.starting_balance:,.2f}")
        print(f"Current Balance: ${daily_pnl.current_balance:,.2f}")
        print(f"\nP&L Breakdown:")
        print(f"  Realized P&L: ${daily_pnl.realized_pnl:,.2f}")
        print(f"  Unrealized P&L: ${daily_pnl.unrealized_pnl:,.2f}")
        print(f"  Total P&L: ${daily_pnl.total_pnl:,.2f} ({daily_pnl.pnl_pct:+.2%})")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {daily_pnl.num_trades}")
        print(f"  Winning Trades: {daily_pnl.winning_trades}")
        print(f"  Losing Trades: {daily_pnl.losing_trades}")
        print(f"\nTrading Status: {'HALTED' if daily_pnl.is_trading_halted else 'ACTIVE'}")
        print(f"Notes: {daily_pnl.notes}")

        # Check if trading is allowed
        print("\n" + "=" * 60)
        print("Trading Permission Check")
        print("=" * 60)

        can_trade = limiter.check_daily_loss()

        if can_trade:
            print("\n✓ Trading is ALLOWED - Within daily loss limits")
        else:
            print("\n✗ Trading is HALTED - Daily loss limit exceeded")
            print(f"   Reason: {limiter.halt_reason}")

        # Show thresholds
        print("\n" + "=" * 60)
        print("Risk Limits")
        print("=" * 60)
        print(f"Warning Threshold: {limiter.warning_threshold_pct:.1%}")
        print(f"Max Daily Loss: {limiter.max_daily_loss_pct:.1%}")
        print(f"Circuit Breaker: {'ENABLED' if limiter.enable_circuit_breaker else 'DISABLED'}")

    except Exception as e:
        logger.error(f"Error in daily loss limiter demo: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
