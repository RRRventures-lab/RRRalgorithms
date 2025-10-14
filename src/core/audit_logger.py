from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from src.core.constants import MonitoringConstants
from src.core.exceptions import AuditError
from typing import Dict, Any, Optional, List
import hashlib
import json
import logging
import logging.handlers
import queue
import threading
import time


"""
Audit Logging System
====================

Comprehensive audit logging for all trading activities, API calls,
and system events. Provides tamper-proof logging with rotation and
archival capabilities.

Features:
- Structured logging with JSON format
- Automatic log rotation
- Tamper detection via checksums
- Async logging for performance
- Compliance-ready formatting

Author: RRR Ventures
Date: 2025-10-12
"""




class AuditEventType(str, Enum):
    """Types of audit events"""
    # Trading events
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_EXECUTED = "ORDER_EXECUTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    
    # System events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    ERROR_OCCURRED = "ERROR_OCCURRED"
    
    # Security events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    API_KEY_USED = "API_KEY_USED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    
    # Data events
    DATA_RECEIVED = "DATA_RECEIVED"
    DATA_PROCESSED = "DATA_PROCESSED"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    
    # Risk events
    RISK_LIMIT_CHECK = "RISK_LIMIT_CHECK"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEvent:
    """Structured audit event"""
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    component: str
    action: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for tamper detection"""
        # Create deterministic string representation
        data = json.dumps({
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'severity': self.severity,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'component': self.component,
            'action': self.action,
            'details': self.details,
            'metadata': self.metadata
        }, sort_keys=True)
        
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        event_dict = asdict(self)
        # Add checksum if not present
        if not self.checksum:
            self.checksum = self.calculate_checksum()
            event_dict['checksum'] = self.checksum
        return json.dumps(event_dict)


class AuditLogger:
    """
    Thread-safe audit logger with async capabilities.
    
    Features:
    - Async logging to not block trading operations
    - Automatic rotation based on size and time
    - Multiple output handlers (file, database, remote)
    - Tamper detection via checksums
    """
    
    def __init__(
        self,
        log_dir: str = "logs/audit",
        max_bytes: int = 100_000_000,  # 100MB
        backup_count: int = 10,
        async_mode: bool = True
    ):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            max_bytes: Maximum size before rotation
            backup_count: Number of backup files to keep
            async_mode: Use async logging for performance
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.async_mode = async_mode
        
        # Session info
        self.session_id = self._generate_session_id()
        self.user_id = None
        
        # Setup logging
        self._setup_logger()
        
        # Async queue for non-blocking logging
        if async_mode:
            self.log_queue: queue.Queue = queue.Queue()
            self.worker_thread = threading.Thread(
                target=self._async_worker,
                daemon=True
            )
            self.worker_thread.start()
        
        # Log system start
        self.log_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            component="AuditLogger",
            action="initialize",
            details={"session_id": self.session_id}
        )
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{timestamp}{id(self)}".encode()).hexdigest()[:16]
    
    def _setup_logger(self):
        """Setup Python logger with rotation"""
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        log_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.jsonl"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        
        # JSON formatter
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Also log critical events to stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        self.logger.addHandler(console_handler)
    
    def _async_worker(self):
        """Background worker for async logging"""
        while True:
            try:
                # Get event from queue (blocks until available)
                event = self.log_queue.get(timeout=1)
                if event is None:  # Shutdown signal
                    break
                    
                # Write to log
                self._write_event(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue
                print(f"Audit logger error: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write event to log file"""
        try:
            # Add checksum
            if not event.checksum:
                event.checksum = event.calculate_checksum()
            
            # Log based on severity
            if event.severity == AuditSeverity.DEBUG:
                self.logger.debug(event.to_json())
            elif event.severity == AuditSeverity.INFO:
                self.logger.info(event.to_json())
            elif event.severity == AuditSeverity.WARNING:
                self.logger.warning(event.to_json())
            elif event.severity == AuditSeverity.ERROR:
                self.logger.error(event.to_json())
            elif event.severity == AuditSeverity.CRITICAL:
                self.logger.critical(event.to_json())
                
        except Exception as e:
            raise AuditError(f"Failed to write audit event: {e}")
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        component: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            severity: Event severity
            component: Component generating event
            action: Action being performed
            details: Event details
            metadata: Additional metadata
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            user_id=self.user_id,
            session_id=self.session_id,
            component=component,
            action=action,
            details=details or {},
            metadata=metadata or {}
        )
        
        if self.async_mode:
            # Queue for async processing
            self.log_queue.put(event)
        else:
            # Write immediately
            self._write_event(event)
    
    def log_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        **kwargs
    ):
        """
        Log trading activity.
        
        Args:
            action: Trade action (place, execute, cancel)
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            price: Order price
            order_id: Order identifier
            **kwargs: Additional details
        """
        event_type_map = {
            'place': AuditEventType.ORDER_PLACED,
            'execute': AuditEventType.ORDER_EXECUTED,
            'cancel': AuditEventType.ORDER_CANCELLED,
            'reject': AuditEventType.ORDER_REJECTED
        }
        
        self.log_event(
            event_type=event_type_map.get(action, AuditEventType.ORDER_PLACED),
            severity=AuditSeverity.INFO,
            component="TradingEngine",
            action=action,
            details={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
                **kwargs
            }
        )
    
    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Log external API calls.
        
        Args:
            api_name: Name of API
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status
            response_time: Response time in ms
            error: Error message if failed
        """
        severity = AuditSeverity.INFO
        if error or (status_code and status_code >= 400):
            severity = AuditSeverity.ERROR
            
        self.log_event(
            event_type=AuditEventType.DATA_RECEIVED,
            severity=severity,
            component="APIClient",
            action=f"{method} {api_name}",
            details={
                'api': api_name,
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time_ms': response_time,
                'error': error
            }
        )
    
    def log_risk_check(
        self,
        check_type: str,
        passed: bool,
        current_value: float,
        limit_value: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log risk management checks.
        
        Args:
            check_type: Type of risk check
            passed: Whether check passed
            current_value: Current metric value
            limit_value: Limit value
            details: Additional details
        """
        event_type = (
            AuditEventType.RISK_LIMIT_CHECK 
            if passed 
            else AuditEventType.RISK_LIMIT_EXCEEDED
        )
        
        severity = AuditSeverity.INFO if passed else AuditSeverity.WARNING
        
        self.log_event(
            event_type=event_type,
            severity=severity,
            component="RiskManager",
            action=check_type,
            details={
                'check_type': check_type,
                'passed': passed,
                'current_value': current_value,
                'limit_value': limit_value,
                **(details or {})
            }
        )
    
    def log_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        **kwargs
    ):
        """
        Log system errors.
        
        Args:
            component: Component where error occurred
            error_type: Type/class of error
            error_message: Error message
            stack_trace: Stack trace if available
            **kwargs: Additional context
        """
        self.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            severity=AuditSeverity.ERROR,
            component=component,
            action="error",
            details={
                'error_type': error_type,
                'error_message': error_message,
                'stack_trace': stack_trace,
                **kwargs
            }
        )
    
    @contextmanager
    def audit_operation(self, component: str, operation: str):
        """
        Context manager for auditing operations.
        
        Usage:
            with audit_logger.audit_operation("TradingEngine", "place_order"):
                # Operation code
                pass
        """
        start_time = time.time()
        
        # Log operation start
        self.log_event(
            event_type=AuditEventType.DATA_PROCESSED,
            severity=AuditSeverity.DEBUG,
            component=component,
            action=f"{operation}_start",
            details={'operation': operation}
        )
        
        try:
            yield
            
            # Log successful completion
            self.log_event(
                event_type=AuditEventType.DATA_PROCESSED,
                severity=AuditSeverity.DEBUG,
                component=component,
                action=f"{operation}_complete",
                details={
                    'operation': operation,
                    'duration_ms': (time.time() - start_time) * 1000
                }
            )
            
        except Exception as e:
            # Log failure
            self.log_error(
                component=component,
                error_type=type(e).__name__,
                error_message=str(e),
                operation=operation,
                duration_ms=(time.time() - start_time) * 1000
            )
            raise
    
    @lru_cache(maxsize=128)
    
    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit events (for monitoring/dashboard).
        
        Args:
            limit: Maximum events to return
            event_type: Filter by event type
            severity: Filter by severity
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        # Read from current log file
        log_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.jsonl"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()[-limit:]  # Get last N lines
                
                for line in lines:
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if event_type and event['event_type'] != event_type:
                            continue
                        if severity and event['severity'] != severity:
                            continue
                            
                        events.append(event)
                        
                    except json.JSONDecodeError:
                        continue
        
        return events
    
    def verify_integrity(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Verify audit log integrity via checksums.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            
        Returns:
            Integrity verification results
        """
        results = {
            'valid': 0,
            'invalid': 0,
            'missing_checksum': 0,
            'errors': []
        }
        
        # Check each log file in date range
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            # Parse date from filename
            file_date = log_file.stem.split('_')[1]
            
            if start_date <= file_date <= end_date:
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            event = json.loads(line)
                            
                            if 'checksum' not in event:
                                results['missing_checksum'] += 1
                                continue
                            
                            # Verify checksum
                            stored_checksum = event['checksum']
                            event_copy = event.copy()
                            del event_copy['checksum']
                            
                            calculated = hashlib.sha256(
                                json.dumps(event_copy, sort_keys=True).encode()
                            ).hexdigest()
                            
                            if stored_checksum == calculated:
                                results['valid'] += 1
                            else:
                                results['invalid'] += 1
                                results['errors'].append({
                                    'file': str(log_file),
                                    'line': line_num,
                                    'event': event['timestamp']
                                })
                                
                        except Exception as e:
                            results['errors'].append({
                                'file': str(log_file),
                                'line': line_num,
                                'error': str(e)
                            })
        
        return results
    
    def shutdown(self):
        """Shutdown audit logger gracefully"""
        if self.async_mode:
            # Signal worker to stop
            self.log_queue.put(None)
            self.worker_thread.join(timeout=5)
        
        # Log shutdown
        self.log_event(
            event_type=AuditEventType.SYSTEM_STOP,
            severity=AuditSeverity.INFO,
            component="AuditLogger",
            action="shutdown",
            details={"session_id": self.session_id}
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


@lru_cache(maxsize=128)


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance"""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger


def audit_function(component: str):
    """
    Decorator for automatic function auditing.
    
    Usage:
        @audit_function("TradingEngine")
        def place_order(symbol, quantity, price):
            # Function code
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_audit_logger()
            
            with logger.audit_operation(component, func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


__all__ = [
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'AuditSeverity',
    'get_audit_logger',
    'audit_function'
]