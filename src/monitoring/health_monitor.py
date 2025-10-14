from datetime import datetime
from pathlib import Path
from src.core.constants import MonitoringConstants
from src.core.database.local_db import get_db
from typing import Dict, Any, Optional
import logging
import psutil
import sys
import time

"""
System Health Monitor
=====================

Monitors system health and sends alerts when issues detected.
Runs continuously in background, checking every 5 minutes.

Monitors:
- Memory usage
- Disk space (Lexar drive)
- CPU usage
- Trading system alive
- Database connectivity
- Error rates

Author: RRR Ventures
Date: 2025-10-12
"""


# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    System health monitoring.
    
    Continuously checks system vitals and alerts on issues.
    """
    
    def __init__(self, check_interval: int = 300):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks (default: 5 minutes)
        """
        self.check_interval = check_interval
        self.last_check = None
        self.alert_callback = None
    
    def set_alert_callback(self, callback):
        """Set callback for sending alerts"""
        self.alert_callback = callback
    
    def send_alert(self, message: str, severity: str = "WARNING"):
        """Send alert via callback"""
        if self.alert_callback:
            self.alert_callback(message, severity)
        else:
            logger.warning(f"[{severity}] {message}")
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        mem = psutil.virtual_memory()
        
        status = {
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'total_gb': mem.total / (1024**3),
            'healthy': mem.percent < MonitoringConstants.MAX_MEMORY_GB * 25  # Conservative
        }
        
        if mem.percent > 85:
            self.send_alert(
                f"🔴 Memory critical: {mem.percent:.1f}% used ({mem.used / (1024**3):.1f}GB)",
                "CRITICAL"
            )
        elif mem.percent > 75:
            self.send_alert(
                f"⚠️ Memory high: {mem.percent:.1f}% used",
                "WARNING"
            )
        
        return status
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space on Lexar drive"""
        try:
            # Check Lexar drive
            lexar_path = "/Volumes/Lexar"
            if Path(lexar_path).exists():
                disk = psutil.disk_usage(lexar_path)
                
                status = {
                    'path': lexar_path,
                    'percent': disk.percent,
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'total_gb': disk.total / (1024**3),
                    'healthy': disk.percent < 85
                }
                
                if disk.percent > 95:
                    self.send_alert(
                        f"🔴 Lexar drive critical: {disk.percent:.1f}% full ({disk.free / (1024**3):.1f}GB free)",
                        "CRITICAL"
                    )
                elif disk.percent > 90:
                    self.send_alert(
                        f"⚠️ Lexar drive space low: {disk.percent:.1f}% full",
                        "WARNING"
                    )
                
                return status
            else:
                self.send_alert("❌ Lexar drive not mounted!", "CRITICAL")
                return {'healthy': False, 'error': 'Drive not mounted'}
                
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status = {
            'percent': cpu_percent,
            'cores': psutil.cpu_count(),
            'healthy': cpu_percent < 90
        }
        
        if cpu_percent > 95:
            self.send_alert(
                f"🔴 CPU critical: {cpu_percent:.1f}%",
                "CRITICAL"
            )
        elif cpu_percent > 85:
            self.send_alert(
                f"⚠️ CPU high: {cpu_percent:.1f}%",
                "WARNING"
            )
        
        return status
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and size"""
        try:
            db = get_db()
            
            # Test query
            trades = db.get_trades(limit=1)
            
            # Check database file size
            db_path = Path(db.db_path)
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024**2)
                
                status = {
                    'connected': True,
                    'size_mb': size_mb,
                    'path': str(db_path),
                    'healthy': size_mb < 5000  # Alert if >5GB
                }
                
                if size_mb > 10000:  # 10GB
                    self.send_alert(
                        f"⚠️ Database large: {size_mb:.1f}MB",
                        "WARNING"
                    )
                
                return status
            else:
                self.send_alert("❌ Database file not found!", "CRITICAL")
                return {'connected': False, 'healthy': False}
                
        except Exception as e:
            self.send_alert(f"❌ Database error: {e}", "CRITICAL")
            return {'connected': False, 'healthy': False, 'error': str(e)}
    
    def check_trading_system(self) -> Dict[str, Any]:
        """Check if trading system is responsive"""
        try:
            # Check if we can access recent data
            db = get_db()
            metrics = db.get_latest_portfolio_metrics()
            
            if metrics:
                # Check if data is recent (within last 5 minutes)
                last_update = metrics.get('timestamp', 0)
                age_seconds = time.time() - last_update
                
                status = {
                    'responsive': True,
                    'last_update_seconds': age_seconds,
                    'healthy': age_seconds < 600  # 10 minutes
                }
                
                if age_seconds > 600:
                    self.send_alert(
                        f"⚠️ Trading system stale: No updates for {age_seconds/60:.1f} minutes",
                        "WARNING"
                    )
                
                return status
            else:
                return {'responsive': True, 'healthy': True, 'note': 'No metrics yet'}
                
        except Exception as e:
            self.send_alert(f"❌ Trading system check failed: {e}", "ERROR")
            return {'responsive': False, 'healthy': False}
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run complete health check"""
        self.last_check = datetime.now()
        
        health = {
            'timestamp': self.last_check.isoformat(),
            'memory': self.check_memory(),
            'disk': self.check_disk_space(),
            'cpu': self.check_cpu(),
            'database': self.check_database(),
            'trading_system': self.check_trading_system()
        }
        
        # Overall health
        health['overall_healthy'] = all([
            health['memory'].get('healthy', False),
            health['disk'].get('healthy', False),
            health['cpu'].get('healthy', False),
            health['database'].get('healthy', False),
            health['trading_system'].get('healthy', False)
        ])
        
        return health
    
    def run_continuous(self):
        """Run health checks continuously"""
        print("="*70)
        print("System Health Monitor Started")
        print(f"Check interval: {self.check_interval} seconds")
        print("="*70)
        print()
        
        try:
            while True:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Running health check...")
                
                health = self.run_health_check()
                
                if health['overall_healthy']:
                    print("  ✅ All systems healthy")
                else:
                    print("  ⚠️  Issues detected (alerts sent)")
                
                print(f"     Memory: {health['memory']['percent']:.1f}%")
                print(f"     Disk: {health['disk'].get('percent', 'N/A')}%")
                print(f"     CPU: {health['cpu']['percent']:.1f}%")
                print()
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Health monitor stopped")


if __name__ == "__main__":
    """Run health monitor standalone"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Optional: Integrate with Telegram
    try:
        from src.monitoring.telegram_alerts import get_telegram_bot
        
        monitor = HealthMonitor(check_interval=300)  # 5 minutes
        
        # Set up Telegram alerts
        bot = get_telegram_bot()
        if bot:
            async def alert_callback(message, severity):
                await bot.send_system_alert(message, severity)
            
            monitor.set_alert_callback(alert_callback)
            print("✅ Telegram alerts enabled")
        else:
            print("⚠️  Telegram not configured (alerts will log only)")
        
    except Exception as e:
        monitor = HealthMonitor(check_interval=300)
        print(f"⚠️  Telegram not available: {e}")
    
    print()
    
    # Run continuous monitoring
    monitor.run_continuous()

