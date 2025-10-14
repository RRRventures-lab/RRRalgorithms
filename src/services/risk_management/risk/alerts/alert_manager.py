from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from enum import Enum
from functools import lru_cache
from src.database import get_db, Client
from typing import Dict, List, Optional
import logging
import os


"""
Risk Alert Manager

Monitors for risk breaches and sends alerts:
- Position size violations
- Daily loss limit warnings
- High volatility alerts
- Drawdown alerts
- Leverage warnings

Logs all alerts to system_events table.
"""


load_dotenv()
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of risk alerts"""
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    VAR_BREACH = "var_breach"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"


@dataclass
class RiskAlert:
    """Risk alert"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    data: Dict
    acknowledged: bool = False


class AlertManager:
    """
    Manage risk alerts

    Monitors risk metrics and generates alerts when thresholds
    are breached. Logs all alerts to Supabase system_events table.
    """

    def __init__(
        self,
        max_position_size: float = 0.20,
        max_daily_loss: float = 0.05,
        max_volatility: float = 0.25,
        max_drawdown: float = 0.15,
        max_leverage: float = 2.0,
        max_concentration: float = 0.30,
        alert_cooldown_minutes: int = 15
    ):
        """
        Initialize alert manager

        Args:
            max_position_size: Max position as % of portfolio
            max_daily_loss: Max daily loss as % of portfolio
            max_volatility: Max portfolio volatility
            max_drawdown: Max acceptable drawdown
            max_leverage: Max leverage ratio
            max_concentration: Max single position concentration
            alert_cooldown_minutes: Minutes to wait before re-alerting same issue
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_volatility = max_volatility
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.alert_cooldown_minutes = alert_cooldown_minutes

        # Track recent alerts to avoid spam
        self._recent_alerts: Dict[str, datetime] = {}

        # Initialize Supabase client
        supabase_url = os.getenv("DATABASE_PATH")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("DATABASE_PATH and SUPABASE_ANON_KEY must be set")

        self.supabase: Client = get_db()

        logger.info(
            f"Alert Manager initialized: pos_size={max_position_size:.1%}, "
            f"daily_loss={max_daily_loss:.1%}, vol={max_volatility:.1%}"
        )

    def _should_alert(self, alert_key: str) -> bool:
        """
        Check if enough time has passed since last alert of this type

        Args:
            alert_key: Unique key for this alert type/symbol

        Returns:
            True if alert should be sent
        """
        if alert_key not in self._recent_alerts:
            return True

        last_alert_time = self._recent_alerts[alert_key]
        time_since_alert = (datetime.now() - last_alert_time).total_seconds() / 60

        return time_since_alert >= self.alert_cooldown_minutes

    def _log_alert(self, alert: RiskAlert):
        """
        Log alert to system_events table

        Args:
            alert: Alert to log
        """
        try:
            event_data = {
                "event_type": f"RISK_ALERT_{alert.alert_type.value.upper()}",
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": "alert_manager",
                "data": alert.data,
                "acknowledged": alert.acknowledged
            }

            self.supabase.table("system_events").insert(event_data).execute()

            # Update recent alerts
            alert_key = f"{alert.alert_type.value}_{alert.data.get('symbol', 'portfolio')}"
            self._recent_alerts[alert_key] = alert.timestamp

            logger.info(f"Alert logged: {alert.alert_type.value} - {alert.message}")

        except Exception as e:
            logger.error(f"Error logging alert: {e}")

    def check_position_size(self, symbol: str, position_size_pct: float) -> Optional[RiskAlert]:
        """
        Check if position size exceeds limit

        Args:
            symbol: Trading symbol
            position_size_pct: Position size as % of portfolio

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = f"{AlertType.POSITION_SIZE.value}_{symbol}"

        if position_size_pct > self.max_position_size and self._should_alert(alert_key):
            severity = AlertSeverity.CRITICAL if position_size_pct > self.max_position_size * 1.5 else AlertSeverity.WARNING

            alert = RiskAlert(
                alert_type=AlertType.POSITION_SIZE,
                severity=severity,
                message=f"Position size for {symbol} ({position_size_pct:.1%}) exceeds limit ({self.max_position_size:.1%})",
                timestamp=datetime.now(),
                data={
                    "symbol": symbol,
                    "position_size_pct": position_size_pct,
                    "limit": self.max_position_size
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_daily_loss(self, daily_loss_pct: float) -> Optional[RiskAlert]:
        """
        Check if daily loss exceeds limits

        Args:
            daily_loss_pct: Daily loss as % (negative value)

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = AlertType.DAILY_LOSS.value

        # Warning at 60% of limit
        warning_threshold = self.max_daily_loss * 0.6

        if abs(daily_loss_pct) >= self.max_daily_loss and self._should_alert(alert_key):
            alert = RiskAlert(
                alert_type=AlertType.DAILY_LOSS,
                severity=AlertSeverity.CRITICAL,
                message=f"Daily loss ({daily_loss_pct:.1%}) exceeds limit ({self.max_daily_loss:.1%})",
                timestamp=datetime.now(),
                data={
                    "daily_loss_pct": daily_loss_pct,
                    "limit": self.max_daily_loss
                }
            )

            self._log_alert(alert)
            return alert

        elif abs(daily_loss_pct) >= warning_threshold and self._should_alert(alert_key):
            alert = RiskAlert(
                alert_type=AlertType.DAILY_LOSS,
                severity=AlertSeverity.WARNING,
                message=f"Daily loss ({daily_loss_pct:.1%}) approaching limit ({self.max_daily_loss:.1%})",
                timestamp=datetime.now(),
                data={
                    "daily_loss_pct": daily_loss_pct,
                    "warning_threshold": warning_threshold,
                    "limit": self.max_daily_loss
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_volatility(self, volatility: float) -> Optional[RiskAlert]:
        """
        Check if portfolio volatility exceeds limit

        Args:
            volatility: Portfolio volatility (annualized)

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = AlertType.VOLATILITY.value

        if volatility > self.max_volatility and self._should_alert(alert_key):
            severity = AlertSeverity.CRITICAL if volatility > self.max_volatility * 1.5 else AlertSeverity.WARNING

            alert = RiskAlert(
                alert_type=AlertType.VOLATILITY,
                severity=severity,
                message=f"Portfolio volatility ({volatility:.1%}) exceeds limit ({self.max_volatility:.1%})",
                timestamp=datetime.now(),
                data={
                    "volatility": volatility,
                    "limit": self.max_volatility
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_drawdown(self, current_drawdown: float) -> Optional[RiskAlert]:
        """
        Check if drawdown exceeds limit

        Args:
            current_drawdown: Current drawdown as fraction

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = AlertType.DRAWDOWN.value

        if current_drawdown > self.max_drawdown and self._should_alert(alert_key):
            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.CRITICAL,
                message=f"Drawdown ({current_drawdown:.1%}) exceeds limit ({self.max_drawdown:.1%})",
                timestamp=datetime.now(),
                data={
                    "current_drawdown": current_drawdown,
                    "limit": self.max_drawdown
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_leverage(self, leverage: float) -> Optional[RiskAlert]:
        """
        Check if leverage exceeds limit

        Args:
            leverage: Current leverage ratio

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = AlertType.LEVERAGE.value

        if leverage > self.max_leverage and self._should_alert(alert_key):
            severity = AlertSeverity.CRITICAL if leverage > self.max_leverage * 1.5 else AlertSeverity.WARNING

            alert = RiskAlert(
                alert_type=AlertType.LEVERAGE,
                severity=severity,
                message=f"Leverage ({leverage:.2f}x) exceeds limit ({self.max_leverage:.2f}x)",
                timestamp=datetime.now(),
                data={
                    "leverage": leverage,
                    "limit": self.max_leverage
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_concentration(self, symbol: str, concentration: float) -> Optional[RiskAlert]:
        """
        Check if single position concentration is too high

        Args:
            symbol: Trading symbol
            concentration: Position as % of portfolio

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = f"{AlertType.CONCENTRATION.value}_{symbol}"

        if concentration > self.max_concentration and self._should_alert(alert_key):
            alert = RiskAlert(
                alert_type=AlertType.CONCENTRATION,
                severity=AlertSeverity.WARNING,
                message=f"High concentration in {symbol} ({concentration:.1%} of portfolio)",
                timestamp=datetime.now(),
                data={
                    "symbol": symbol,
                    "concentration": concentration,
                    "limit": self.max_concentration
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_var_breach(self, var_pct: float, var_threshold: float = 0.10) -> Optional[RiskAlert]:
        """
        Check if VaR exceeds acceptable threshold

        Args:
            var_pct: VaR as % of portfolio
            var_threshold: Acceptable VaR threshold

        Returns:
            RiskAlert if violation, None otherwise
        """
        alert_key = AlertType.VAR_BREACH.value

        if var_pct > var_threshold and self._should_alert(alert_key):
            alert = RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                severity=AlertSeverity.WARNING,
                message=f"VaR ({var_pct:.1%}) exceeds threshold ({var_threshold:.1%})",
                timestamp=datetime.now(),
                data={
                    "var_pct": var_pct,
                    "threshold": var_threshold
                }
            )

            self._log_alert(alert)
            return alert

        return None

    def check_all_risks(self) -> List[RiskAlert]:
        """
        Check all risk metrics and generate alerts

        Queries current portfolio state and checks against all thresholds.

        Returns:
            List of active risk alerts
        """
        alerts = []

        try:
            # Import dashboard to get metrics
            from ..dashboard.risk_metrics import RiskDashboard
            from ..limits.daily_loss_limiter import DailyLossLimiter

            # Get dashboard metrics
            dashboard = RiskDashboard()
            metrics = dashboard.get_dashboard_metrics()

            # Check leverage
            alert = self.check_leverage(metrics.leverage)
            if alert:
                alerts.append(alert)

            # Check volatility
            alert = self.check_volatility(metrics.portfolio_volatility)
            if alert:
                alerts.append(alert)

            # Check drawdown
            alert = self.check_drawdown(metrics.current_drawdown)
            if alert:
                alerts.append(alert)

            # Check daily loss
            limiter = DailyLossLimiter()
            daily_pnl = limiter.get_daily_pnl()
            alert = self.check_daily_loss(daily_pnl.pnl_pct)
            if alert:
                alerts.append(alert)

            # Check VaR
            var_pct = metrics.var_95 / metrics.total_value if metrics.total_value > 0 else 0
            alert = self.check_var_breach(var_pct)
            if alert:
                alerts.append(alert)

            # Check individual positions
            response = self.supabase.table("positions").select("*").execute()
            positions = response.data

            if positions:
                total_value = metrics.total_value

                for pos in positions:
                    symbol = pos["symbol"]
                    position_value = float(pos["quantity"]) * float(pos["current_price"])
                    position_pct = position_value / total_value if total_value > 0 else 0

                    # Check position size
                    alert = self.check_position_size(symbol, position_pct)
                    if alert:
                        alerts.append(alert)

                    # Check concentration
                    alert = self.check_concentration(symbol, position_pct)
                    if alert:
                        alerts.append(alert)

            logger.info(f"Risk check complete: {len(alerts)} alerts generated")

        except Exception as e:
            logger.error(f"Error checking risks: {e}")

        return alerts

    @lru_cache(maxsize=128)

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Get recent alerts from system_events table

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            response = (
                self.supabase.table("system_events")
                .select("*")
                .gte("timestamp", cutoff_time.isoformat())
                .like("event_type", "RISK_ALERT_%")
                .order("timestamp", desc=True)
                .execute()
            )

            return response.data

        except Exception as e:
            logger.error(f"Error fetching recent alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID in system_events table
        """
        try:
            self.supabase.table("system_events").update({
                "acknowledged": True,
                "acknowledged_at": datetime.now().isoformat()
            }).eq("id", alert_id).execute()

            logger.info(f"Alert {alert_id} acknowledged")

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")


def main():
    """Example usage"""
    print("=" * 60)
    print("Risk Alert Manager")
    print("=" * 60)

    try:
        # Initialize manager
        manager = AlertManager(
            max_position_size=0.20,
            max_daily_loss=0.05,
            max_volatility=0.25,
            max_drawdown=0.15,
            max_leverage=2.0
        )

        # Check all risks
        print("\nChecking all risk metrics...")
        alerts = manager.check_all_risks()

        if not alerts:
            print("\n✓ No risk alerts - all metrics within limits")
        else:
            print(f"\n⚠ {len(alerts)} risk alerts detected:")
            for i, alert in enumerate(alerts, 1):
                print(f"\n{i}. {alert.alert_type.value.upper()} - {alert.severity.value.upper()}")
                print(f"   {alert.message}")
                print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Get recent alerts
        print("\n" + "=" * 60)
        print("Recent Alerts (Last 24 Hours)")
        print("=" * 60)

        recent = manager.get_recent_alerts(hours=24)
        print(f"\nFound {len(recent)} alerts in the last 24 hours")

        for alert_data in recent[:5]:  # Show first 5
            print(f"\n- {alert_data['event_type']}")
            print(f"  Severity: {alert_data['severity']}")
            print(f"  Message: {alert_data['message']}")
            print(f"  Time: {alert_data['timestamp']}")

    except Exception as e:
        logger.error(f"Error in alert manager demo: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
