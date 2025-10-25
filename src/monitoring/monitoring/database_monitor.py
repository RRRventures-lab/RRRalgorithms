from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from supabase import create_client, Client
from typing import Dict, List, Any, Optional
import os
import time


"""
Database Monitoring Service

Monitors Supabase database health, performance, and growth patterns.
Tracks table sizes, row counts, and query performance.
"""


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)


class DatabaseMonitor:
    """
    Monitor database health and performance

    Features:
    - Table size tracking
    - Row count monitoring
    - Query performance analysis
    - Growth rate detection
    - Alert on anomalies
    """

    def __init__(self):
        """Initialize database monitor"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.tables = ["orders", "positions", "portfolio_snapshots", "system_events", "api_usage"]

        # Thresholds for alerts
        self.max_table_size_mb = 1000  # Alert if table exceeds 1GB
        self.max_daily_growth_rate = 2.0  # Alert if table doubles in size
        self.max_query_time_ms = 1000  # Alert if query takes >1s

    @lru_cache(maxsize=128)

    def get_table_row_counts(self) -> Dict[str, int]:
        """Get row count for each table"""
        row_counts = {}

        for table in self.tables:
            try:
                response = self.supabase.table(table).select("*", count="exact").limit(0).execute()
                row_counts[table] = response.count if response.count is not None else 0
            except Exception as e:
                row_counts[table] = -1  # Error indicator
                print(f"Error getting row count for {table}: {e}")

        return row_counts

    def measure_query_performance(self, table: str, limit: int = 100) -> float:
        """
        Measure query execution time

        Args:
            table: Table name
            limit: Number of rows to fetch

        Returns:
            Query execution time in milliseconds
        """
        try:
            start_time = time.time()
            self.supabase.table(table).select("*").limit(limit).execute()
            end_time = time.time()

            return (end_time - start_time) * 1000  # Convert to ms
        except Exception as e:
            print(f"Error measuring query performance for {table}: {e}")
            return -1

    @lru_cache(maxsize=128)

    def get_recent_events_count(self, hours: int = 24) -> Dict[str, int]:
        """
        Get count of events in the last N hours by type

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary of event_type to count
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            response = self.supabase.table("system_events") \
                .select("event_type") \
                .gte("timestamp", cutoff_time) \
                .execute()

            # Count by event type
            event_counts: Dict[str, int] = {}
            for row in response.data:
                event_type = row.get("event_type", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

            return event_counts
        except Exception as e:
            print(f"Error getting recent events count: {e}")
            return {}

    @lru_cache(maxsize=128)

    def get_error_count(self, hours: int = 24) -> int:
        """
        Get count of errors in the last N hours

        Args:
            hours: Number of hours to look back

        Returns:
            Error count
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            response = self.supabase.table("system_events") \
                .select("*", count="exact") \
                .gte("timestamp", cutoff_time) \
                .in_("severity", ["ERROR", "CRITICAL"]) \
                .limit(0) \
                .execute()

            return response.count if response.count is not None else 0
        except Exception as e:
            print(f"Error getting error count: {e}")
            return -1

    @lru_cache(maxsize=128)

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trade orders

        Args:
            limit: Maximum number of trades to fetch

        Returns:
            List of recent trades
        """
        try:
            response = self.supabase.table("orders") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()

            return response.data
        except Exception as e:
            print(f"Error getting recent trades: {e}")
            return []

    @lru_cache(maxsize=128)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get currently open positions

        Returns:
            List of open positions
        """
        try:
            response = self.supabase.table("positions") \
                .select("*") \
                .eq("status", "open") \
                .execute()

            return response.data
        except Exception as e:
            print(f"Error getting open positions: {e}")
            return []

    @lru_cache(maxsize=128)

    def get_latest_portfolio_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent portfolio snapshot

        Returns:
            Latest portfolio snapshot or None
        """
        try:
            response = self.supabase.table("portfolio_snapshots") \
                .select("*") \
                .order("timestamp", desc=True) \
                .limit(1) \
                .execute()

            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting latest portfolio snapshot: {e}")
            return None

    @lru_cache(maxsize=128)

    def get_api_usage_stats(self, hours: int = 24) -> Dict[str, int]:
        """
        Get API usage statistics

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary of endpoint to request count
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            response = self.supabase.table("api_usage") \
                .select("endpoint") \
                .gte("timestamp", cutoff_time) \
                .execute()

            # Count by endpoint
            usage_stats: Dict[str, int] = {}
            for row in response.data:
                endpoint = row.get("endpoint", "unknown")
                usage_stats[endpoint] = usage_stats.get(endpoint, 0) + 1

            return usage_stats
        except Exception as e:
            print(f"Error getting API usage stats: {e}")
            return {}

    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Dictionary with health status and metrics
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }

        # Check table row counts
        row_counts = self.get_table_row_counts()
        health_status["metrics"]["row_counts"] = row_counts

        # Check for errors in table access
        for table, count in row_counts.items():
            if count == -1:
                health_status["status"] = "unhealthy"
                health_status["issues"].append(f"Cannot access table: {table}")

        # Check query performance
        query_times = {}
        for table in self.tables:
            query_time = self.measure_query_performance(table)
            query_times[table] = query_time

            if query_time > self.max_query_time_ms:
                health_status["status"] = "degraded"
                health_status["issues"].append(
                    f"Slow query on {table}: {query_time:.2f}ms"
                )

        health_status["metrics"]["query_times_ms"] = query_times

        # Check recent error count
        error_count = self.get_error_count(hours=1)
        health_status["metrics"]["errors_last_hour"] = error_count

        if error_count > 100:  # More than 100 errors per hour
            health_status["status"] = "degraded"
            health_status["issues"].append(f"High error rate: {error_count} errors/hour")

        return health_status

    def log_metrics_to_supabase(self):
        """Log current metrics to api_usage table for tracking"""
        try:
            metrics = self.check_health()

            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "database_monitor",
                "latency_ms": sum(metrics["metrics"]["query_times_ms"].values()),
                "status_code": 200 if metrics["status"] == "healthy" else 500,
                "metadata": metrics
            }

            self.supabase.table("api_usage").insert(event).execute()
        except Exception as e:
            print(f"Error logging metrics to Supabase: {e}")


def main():
    """Run database monitoring checks"""
    monitor = DatabaseMonitor()

    print("=" * 60)
    print("Database Health Check")
    print("=" * 60)

    # Run health check
    health = monitor.check_health()

    print(f"\nStatus: {health['status'].upper()}")
    print(f"Timestamp: {health['timestamp']}")

    print("\nTable Row Counts:")
    for table, count in health['metrics']['row_counts'].items():
        print(f"  {table}: {count:,}")

    print("\nQuery Performance:")
    for table, time_ms in health['metrics']['query_times_ms'].items():
        print(f"  {table}: {time_ms:.2f}ms")

    print(f"\nErrors (last hour): {health['metrics']['errors_last_hour']}")

    if health['issues']:
        print("\nIssues Detected:")
        for issue in health['issues']:
            print(f"  - {issue}")

    # Get recent events
    print("\nRecent Events (last 24h):")
    events = monitor.get_recent_events_count(hours=24)
    for event_type, count in events.items():
        print(f"  {event_type}: {count}")

    # Log metrics
    monitor.log_metrics_to_supabase()
    print("\nMetrics logged to Supabase.")


if __name__ == "__main__":
    main()
