from .alerts.alert_manager import AlertManager
from .dashboard.risk_metrics import RiskDashboard
from .limits.daily_loss_limiter import DailyLossLimiter
from .monitors.portfolio_risk import PortfolioRiskMonitor
from .sizing.kelly_criterion import KellyCriterion
from .stops.stop_manager import StopManager
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import logging
import os
import signal
import sys
import time

"""
Main Risk Management Service

Orchestrates all risk management components:
- Portfolio risk monitoring
- Stop-loss management
- Daily loss limiting
- Risk alerts
- Position sizing

Runs continuously to monitor risk and enforce limits.
"""



load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskManagementService:
    """
    Main risk management service

    Coordinates all risk management components and runs
    continuous monitoring and enforcement.
    """

    def __init__(
        self,
        update_interval_seconds: int = 60,
        stop_update_interval_seconds: int = 300,
        alert_check_interval_seconds: int = 60
    ):
        """
        Initialize risk management service

        Args:
            update_interval_seconds: Seconds between risk metric updates
            stop_update_interval_seconds: Seconds between stop order updates
            alert_check_interval_seconds: Seconds between alert checks
        """
        self.update_interval = update_interval_seconds
        self.stop_update_interval = stop_update_interval_seconds
        self.alert_check_interval = alert_check_interval_seconds

        # Load configuration from environment
        max_position_size = float(os.getenv("MAX_POSITION_SIZE", "0.20"))
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
        max_volatility = float(os.getenv("MAX_PORTFOLIO_VOLATILITY", "0.25"))

        # Initialize components
        logger.info("Initializing risk management components...")

        self.portfolio_monitor = PortfolioRiskMonitor(
            max_portfolio_volatility=max_volatility
        )

        self.stop_manager = StopManager(
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.06,
            enable_trailing_stops=True
        )

        self.daily_loss_limiter = DailyLossLimiter(
            max_daily_loss_pct=max_daily_loss,
            enable_circuit_breaker=True
        )

        self.alert_manager = AlertManager(
            max_position_size=max_position_size,
            max_daily_loss=max_daily_loss,
            max_volatility=max_volatility
        )

        self.dashboard = RiskDashboard()

        self.kelly_calculator = KellyCriterion(
            max_position_size=max_position_size
        )

        # Service state
        self.is_running = False
        self.last_stop_update = datetime.now()
        self.last_alert_check = datetime.now()

        logger.info("Risk management service initialized")

    def check_trading_permission(self) -> tuple[bool, str]:
        """
        Check if trading is allowed

        This is called by the trading engine before executing trades.

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check daily loss limit
        can_trade = self.daily_loss_limiter.check_daily_loss()

        if not can_trade:
            reason = self.daily_loss_limiter.halt_reason
            logger.warning(f"Trading not allowed: {reason}")
            return False, reason

        # Check portfolio risk limits
        is_within_limits, violations = self.portfolio_monitor.check_risk_limits()

        if not is_within_limits:
            reason = f"Risk limits exceeded: {'; '.join(violations)}"
            logger.warning(f"Trading not allowed: {reason}")
            return False, reason

        return True, "All risk checks passed"

    def update_risk_metrics(self):
        """Update and log risk metrics"""
        try:
            logger.info("Updating risk metrics...")

            # Get portfolio risk metrics
            risk_metrics = self.portfolio_monitor.get_portfolio_metrics()

            logger.info(
                f"Portfolio Risk - Value: ${risk_metrics.total_value:,.2f}, "
                f"VaR: ${risk_metrics.var_95:,.2f}, "
                f"Volatility: {risk_metrics.volatility_30d:.1%}, "
                f"Drawdown: {risk_metrics.max_drawdown:.1%}"
            )

            # Get daily P&L
            daily_pnl = self.daily_loss_limiter.get_daily_pnl()

            logger.info(
                f"Daily P&L: ${daily_pnl.total_pnl:,.2f} ({daily_pnl.pnl_pct:+.2%}), "
                f"Trades: {daily_pnl.num_trades} "
                f"(W: {daily_pnl.winning_trades}, L: {daily_pnl.losing_trades})"
            )

            # Log to console every update
            if logger.level <= logging.INFO:
                print("\n" + "=" * 60)
                print(f"Risk Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                print(f"Portfolio Value: ${risk_metrics.total_value:,.2f}")
                print(f"Daily P&L: ${daily_pnl.total_pnl:,.2f} ({daily_pnl.pnl_pct:+.2%})")
                print(f"Volatility: {risk_metrics.volatility_30d:.1%}")
                print(f"VaR (95%): ${risk_metrics.var_95:,.2f}")
                print(f"Trading Status: {'ACTIVE' if not daily_pnl.is_trading_halted else 'HALTED'}")
                print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    def update_stops(self):
        """Update stop-loss and take-profit orders"""
        try:
            logger.info("Updating stop orders...")

            # Update trailing stops
            updated_count = self.stop_manager.update_all_trailing_stops()

            logger.info(f"Updated {updated_count} trailing stops")

        except Exception as e:
            logger.error(f"Error updating stops: {e}")

    def check_alerts(self):
        """Check for risk alerts"""
        try:
            logger.info("Checking for risk alerts...")

            # Check all risks
            alerts = self.alert_manager.check_all_risks()

            if alerts:
                logger.warning(f"Generated {len(alerts)} risk alerts")

                for alert in alerts:
                    if alert.severity.value == "critical":
                        logger.critical(f"CRITICAL ALERT: {alert.message}")
                    elif alert.severity.value == "warning":
                        logger.warning(f"Warning: {alert.message}")
                    else:
                        logger.info(f"Info: {alert.message}")

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def generate_stops_for_new_positions(self):
        """Generate stop orders for any positions without them"""
        try:
            logger.info("Checking for positions without stops...")

            # This would query positions without stop orders
            # For now, we'll just call the generate function
            stop_orders = self.stop_manager.generate_stops_for_all_positions()

            if stop_orders:
                logger.info(f"Generated {len(stop_orders)} new stop order pairs")

        except Exception as e:
            logger.error(f"Error generating stops: {e}")

    def run_once(self):
        """Run one iteration of risk monitoring"""
        # Update risk metrics
        self.update_risk_metrics()

        # Check alerts
        now = datetime.now()
        time_since_alert_check = (now - self.last_alert_check).total_seconds()

        if time_since_alert_check >= self.alert_check_interval:
            self.check_alerts()
            self.last_alert_check = now

        # Update stops
        time_since_stop_update = (now - self.last_stop_update).total_seconds()

        if time_since_stop_update >= self.stop_update_interval:
            self.update_stops()
            self.generate_stops_for_new_positions()
            self.last_stop_update = now

    def run(self):
        """Run continuous risk monitoring"""
        self.is_running = True

        logger.info("=" * 60)
        logger.info("Risk Management Service Starting")
        logger.info("=" * 60)
        logger.info(f"Update interval: {self.update_interval}s")
        logger.info(f"Stop update interval: {self.stop_update_interval}s")
        logger.info(f"Alert check interval: {self.alert_check_interval}s")
        logger.info("=" * 60)

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            while self.is_running:
                self.run_once()

                # Sleep until next update
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False

    def shutdown(self):
        """Shutdown service gracefully"""
        logger.info("Shutting down risk management service...")
        self.is_running = False
        logger.info("Risk management service stopped")

    def show_dashboard(self):
        """Display risk dashboard"""
        self.dashboard.print_dashboard()


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print(f"{'Risk Management Service':^70}")
    print("=" * 70)

    # Check if we should run continuously or just show dashboard
    mode = os.getenv("RISK_SERVICE_MODE", "continuous")

    if mode == "dashboard":
        # Just show dashboard once
        service = RiskManagementService()
        service.show_dashboard()

    elif mode == "once":
        # Run one iteration
        service = RiskManagementService()
        service.run_once()

    else:
        # Run continuously
        service = RiskManagementService(
            update_interval_seconds=60,
            stop_update_interval_seconds=300,
            alert_check_interval_seconds=60
        )

        service.run()


if __name__ == "__main__":
    main()
