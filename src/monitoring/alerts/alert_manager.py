from datetime import datetime, timedelta
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from supabase import create_client, Client
from typing import Dict, List, Any, Optional
import json
import os
import requests
import smtplib

"""
Alert Management System

Monitors critical events and sends notifications via email and Slack.
Tracks:
- Large losses
- Risk limit breaches
- System errors
- API failures
"""


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)


class AlertManager:
    """
    Manage alerts and notifications for critical events

    Features:
    - Email notifications
    - Slack notifications (optional)
    - Alert deduplication
    - Severity levels
    - Alert history tracking
    """

    def __init__(self):
        """Initialize alert manager"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment")

        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Email configuration (using Gmail SMTP as example)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.alert_recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")

        # Slack configuration
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

        # Alert thresholds
        self.max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS", "0.05"))  # 5%
        self.max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE", "0.20"))  # 20%
        self.max_error_rate_pct = 10.0  # 10% error rate
        self.max_api_failure_count = 5  # consecutive failures

        # Alert cooldown (prevent spam)
        self.alert_cooldown_minutes = 15
        self._recent_alerts: Dict[str, datetime] = {}

    def _should_send_alert(self, alert_key: str) -> bool:
        """
        Check if alert should be sent based on cooldown

        Args:
            alert_key: Unique key for alert type

        Returns:
            True if alert should be sent
        """
        if alert_key not in self._recent_alerts:
            return True

        last_alert_time = self._recent_alerts[alert_key]
        cooldown_period = timedelta(minutes=self.alert_cooldown_minutes)

        return datetime.utcnow() - last_alert_time > cooldown_period

    def _record_alert(self, alert_key: str):
        """Record that an alert was sent"""
        self._recent_alerts[alert_key] = datetime.utcnow()

    def send_email_alert(
        self,
        subject: str,
        body: str,
        severity: str = "WARNING",
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send email alert

        Args:
            subject: Email subject
            body: Email body (plain text)
            severity: Alert severity (INFO, WARNING, ERROR, CRITICAL)
            html_body: Optional HTML body

        Returns:
            True if email sent successfully
        """
        if not self.email_user or not self.email_password:
            print("Email credentials not configured, skipping email alert")
            return False

        if not self.alert_recipients or not self.alert_recipients[0]:
            print("No alert recipients configured, skipping email alert")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity}] RRR Trading Alert: {subject}"
            msg['From'] = self.email_user
            msg['To'] = ", ".join(self.alert_recipients)

            # Add timestamp
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            body_with_timestamp = f"{body}\n\n---\nTimestamp: {timestamp}"

            # Attach plain text
            msg.attach(MIMEText(body_with_timestamp, 'plain'))

            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)

            print(f"Email alert sent: {subject}")
            return True

        except Exception as e:
            print(f"Error sending email alert: {e}")
            return False

    def send_slack_alert(
        self,
        message: str,
        severity: str = "WARNING",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send Slack alert via webhook

        Args:
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            True if Slack message sent successfully
        """
        if not self.slack_webhook_url or self.slack_webhook_url == "https://hooks.slack.com/services/YOUR/WEBHOOK/URL":
            print("Slack webhook not configured, skipping Slack alert")
            return False

        try:
            # Map severity to Slack color
            color_map = {
                "INFO": "#36a64f",      # green
                "WARNING": "#ff9900",   # orange
                "ERROR": "#ff0000",     # red
                "CRITICAL": "#990000"   # dark red
            }

            payload = {
                "attachments": [{
                    "color": color_map.get(severity, "#808080"),
                    "title": f"{severity}: RRR Trading Alert",
                    "text": message,
                    "footer": "RRR Trading System",
                    "ts": int(datetime.utcnow().timestamp())
                }]
            }

            # Add metadata fields
            if metadata:
                payload["attachments"][0]["fields"] = [
                    {"title": key, "value": str(value), "short": True}
                    for key, value in metadata.items()
                ]

            response = requests.post(
                self.slack_webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=5
            )

            if response.status_code == 200:
                print(f"Slack alert sent: {message}")
                return True
            else:
                print(f"Failed to send Slack alert: {response.status_code}")
                return False

        except Exception as e:
            print(f"Error sending Slack alert: {e}")
            return False

    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        metadata: Optional[Dict[str, Any]] = None,
        use_email: bool = True,
        use_slack: bool = True
    ):
        """
        Send alert via all configured channels

        Args:
            alert_type: Type of alert (used for deduplication)
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
            use_email: Send via email
            use_slack: Send via Slack
        """
        # Check cooldown
        alert_key = f"{alert_type}:{severity}"
        if not self._should_send_alert(alert_key):
            print(f"Alert {alert_key} is in cooldown period, skipping")
            return

        # Send alerts
        if use_email:
            self.send_email_alert(alert_type, message, severity)

        if use_slack:
            self.send_slack_alert(message, severity, metadata)

        # Record in database
        self._log_alert_to_database(alert_type, message, severity, metadata)

        # Update cooldown
        self._record_alert(alert_key)

    def _log_alert_to_database(
        self,
        alert_type: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Log alert to Supabase for tracking"""
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": f"alert_{alert_type}",
                "severity": severity,
                "component": "alert_manager",
                "message": message,
                "metadata": metadata or {}
            }

            self.supabase.table("system_events").insert(event).execute()
        except Exception as e:
            print(f"Error logging alert to database: {e}")

    def check_portfolio_losses(self) -> Optional[Dict[str, Any]]:
        """
        Check for large portfolio losses

        Returns:
            Alert data if loss threshold breached, None otherwise
        """
        try:
            # Get today's portfolio snapshots
            today = datetime.utcnow().date().isoformat()

            response = self.supabase.table("portfolio_snapshots") \
                .select("*") \
                .gte("timestamp", today) \
                .order("timestamp", desc=False) \
                .execute()

            if not response.data or len(response.data) < 2:
                return None

            # Calculate daily P&L
            start_value = response.data[0].get("total_value", 0)
            current_value = response.data[-1].get("total_value", 0)

            if start_value == 0:
                return None

            daily_return_pct = ((current_value - start_value) / start_value) * 100

            # Check if loss exceeds threshold
            if daily_return_pct < -(self.max_daily_loss_pct * 100):
                return {
                    "alert_type": "large_loss",
                    "message": f"Large daily loss detected: {daily_return_pct:.2f}%",
                    "severity": "CRITICAL",
                    "metadata": {
                        "start_value": start_value,
                        "current_value": current_value,
                        "daily_return_pct": daily_return_pct,
                        "threshold_pct": -(self.max_daily_loss_pct * 100)
                    }
                }

            return None

        except Exception as e:
            print(f"Error checking portfolio losses: {e}")
            return None

    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """
        Check for risk limit breaches

        Returns:
            List of alert data for any breaches
        """
        alerts = []

        try:
            # Get current positions
            response = self.supabase.table("positions") \
                .select("*") \
                .eq("status", "open") \
                .execute()

            if not response.data:
                return alerts

            # Get latest portfolio value
            portfolio_response = self.supabase.table("portfolio_snapshots") \
                .select("total_value") \
                .order("timestamp", desc=True) \
                .limit(1) \
                .execute()

            if not portfolio_response.data:
                return alerts

            portfolio_value = portfolio_response.data[0].get("total_value", 0)
            if portfolio_value == 0:
                return alerts

            # Check each position
            for position in response.data:
                position_value = position.get("current_value", 0)
                position_pct = (position_value / portfolio_value) * 100

                if position_pct > (self.max_position_size_pct * 100):
                    alerts.append({
                        "alert_type": "risk_limit_breach",
                        "message": f"Position size exceeds limit: {position['symbol']} at {position_pct:.2f}%",
                        "severity": "ERROR",
                        "metadata": {
                            "symbol": position['symbol'],
                            "position_value": position_value,
                            "position_pct": position_pct,
                            "threshold_pct": self.max_position_size_pct * 100
                        }
                    })

            return alerts

        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return alerts

    def check_system_errors(self, hours: int = 1) -> Optional[Dict[str, Any]]:
        """
        Check for high error rates

        Args:
            hours: Time window to check

        Returns:
            Alert data if error rate high, None otherwise
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            # Get total events
            total_response = self.supabase.table("system_events") \
                .select("*", count="exact") \
                .gte("timestamp", cutoff_time) \
                .limit(0) \
                .execute()

            total_count = total_response.count if total_response.count else 0

            if total_count == 0:
                return None

            # Get error events
            error_response = self.supabase.table("system_events") \
                .select("*", count="exact") \
                .gte("timestamp", cutoff_time) \
                .in_("severity", ["ERROR", "CRITICAL"]) \
                .limit(0) \
                .execute()

            error_count = error_response.count if error_response.count else 0

            # Calculate error rate
            error_rate_pct = (error_count / total_count) * 100

            if error_rate_pct > self.max_error_rate_pct:
                return {
                    "alert_type": "high_error_rate",
                    "message": f"High error rate detected: {error_rate_pct:.2f}% ({error_count}/{total_count})",
                    "severity": "ERROR",
                    "metadata": {
                        "error_count": error_count,
                        "total_count": total_count,
                        "error_rate_pct": error_rate_pct,
                        "threshold_pct": self.max_error_rate_pct,
                        "time_window_hours": hours
                    }
                }

            return None

        except Exception as e:
            print(f"Error checking system errors: {e}")
            return None

    def run_all_checks(self):
        """Run all monitoring checks and send alerts as needed"""
        print("Running alert checks...")

        # Check portfolio losses
        loss_alert = self.check_portfolio_losses()
        if loss_alert:
            self.send_alert(**loss_alert)

        # Check risk limits
        risk_alerts = self.check_risk_limits()
        for alert in risk_alerts:
            self.send_alert(**alert)

        # Check system errors
        error_alert = self.check_system_errors(hours=1)
        if error_alert:
            self.send_alert(**error_alert)

        print("Alert checks completed.")


def main():
    """Test alert manager"""
    manager = AlertManager()

    print("=" * 60)
    print("Alert Manager Test")
    print("=" * 60)

    # Test email alert (will only send if configured)
    print("\nTesting email alert...")
    manager.send_email_alert(
        subject="Test Alert",
        body="This is a test alert from the RRR Trading System",
        severity="INFO"
    )

    # Test Slack alert (will only send if configured)
    print("\nTesting Slack alert...")
    manager.send_slack_alert(
        message="This is a test alert from the RRR Trading System",
        severity="INFO",
        metadata={"test_key": "test_value"}
    )

    # Run all checks
    print("\nRunning all checks...")
    manager.run_all_checks()

    print("\nAlert manager test completed.")


if __name__ == "__main__":
    main()
