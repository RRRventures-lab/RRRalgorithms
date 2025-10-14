from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify
from functools import lru_cache
from monitoring.database_monitor import DatabaseMonitor
from monitoring.performance_monitor import get_performance_monitor
from pathlib import Path
from typing import Dict, Any
import os
import sys


"""
Health Check Service

Provides HTTP endpoint to check system health status.
Monitors all critical services and dependencies.
"""


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)

# Create Flask app
app = Flask(__name__)


class HealthChecker:
    """
    Check health of all system components

    Services checked:
    - Database connectivity
    - API availability
    - Performance metrics
    - Service status
    """

    def __init__(self):
        """Initialize health checker"""
        self.db_monitor = DatabaseMonitor()
        self.perf_monitor = get_performance_monitor()

    def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            health = self.db_monitor.check_health()
            return {
                "status": health["status"],
                "details": {
                    "row_counts": health["metrics"].get("row_counts", {}),
                    "errors_last_hour": health["metrics"].get("errors_last_hour", 0)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_performance_health(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            health = self.perf_monitor.check_performance_health()
            return {
                "status": health["status"],
                "details": {
                    "avg_latency_ms": health["metrics"].get("avg_latency_ms", 0),
                    "error_rate_pct": health["metrics"].get("error_rate_pct", 0)
                }
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }

    def check_supabase_connectivity(self) -> Dict[str, Any]:
        """Check if Supabase is accessible"""
        try:
            # Try to get row count from a table
            self.db_monitor.supabase.table("system_events").select("*", count="exact").limit(0).execute()
            return {
                "status": "healthy",
                "details": {"connection": "ok"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_api_keys(self) -> Dict[str, Any]:
        """Check if required API keys are configured"""
        required_keys = [
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY",
            "POLYGON_API_KEY"
        ]

        missing_keys = []
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)

        if missing_keys:
            return {
                "status": "unhealthy",
                "details": {"missing_keys": missing_keys}
            }

        return {
            "status": "healthy",
            "details": {"configured_keys": len(required_keys)}
        }

    @lru_cache(maxsize=128)

    def get_full_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        timestamp = datetime.utcnow().isoformat()

        # Check all services
        database_health = self.check_database_health()
        performance_health = self.check_performance_health()
        supabase_health = self.check_supabase_connectivity()
        api_keys_health = self.check_api_keys()

        # Determine overall status
        service_statuses = [
            database_health["status"],
            performance_health["status"],
            supabase_health["status"],
            api_keys_health["status"]
        ]

        if "unhealthy" in service_statuses:
            overall_status = "unhealthy"
        elif "degraded" in service_statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "timestamp": timestamp,
            "services": {
                "database": database_health,
                "performance": performance_health,
                "supabase": supabase_health,
                "api_keys": api_keys_health
            },
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }


# Create global health checker
health_checker = HealthChecker()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        health_status = health_checker.get_full_health_status()
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }), 500


@app.route('/health/database', methods=['GET'])
def health_database():
    """Database health check endpoint"""
    health_status = health_checker.check_database_health()
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@app.route('/health/performance', methods=['GET'])
def health_performance():
    """Performance health check endpoint"""
    health_status = health_checker.check_performance_health()
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@app.route('/health/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


def main():
    """Run health check server"""
    port = int(os.getenv("HEALTH_CHECK_PORT", 5001))
    host = os.getenv("HEALTH_CHECK_HOST", "0.0.0.0")

    print("=" * 60)
    print("Health Check Service")
    print("=" * 60)
    print(f"\nStarting health check server on {host}:{port}")
    print("\nEndpoints:")
    print(f"  GET http://{host}:{port}/health           - Full health check")
    print(f"  GET http://{host}:{port}/health/database  - Database health")
    print(f"  GET http://{host}:{port}/health/performance - Performance health")
    print(f"  GET http://{host}:{port}/health/ping      - Simple ping")
    print("\n" + "=" * 60)

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
