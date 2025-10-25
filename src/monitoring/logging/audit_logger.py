from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Optional, Dict, Any, List
import json
import logging
import os
import uuid


"""
Audit Logger
Comprehensive audit logging system for trading operations, compliance, and security
"""


try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase client not available - audit logs will be written to file only")


# ============================================================================
# Enums for Type Safety
# ============================================================================

class AuditSeverity(str, Enum):
    """Audit log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditCategory(str, Enum):
    """Categories of audit events"""
    TRADING = "TRADING"
    RISK = "RISK"
    CONFIG = "CONFIG"
    SECURITY = "SECURITY"
    DATA = "DATA"
    SYSTEM = "SYSTEM"
    COMPLIANCE = "COMPLIANCE"


class AuditAction(str, Enum):
    """Standard audit action types"""
    # Trading actions
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"

    # Position actions
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_MODIFIED = "POSITION_MODIFIED"

    # Risk actions
    RISK_LIMIT_BREACHED = "RISK_LIMIT_BREACHED"
    RISK_LIMIT_UPDATED = "RISK_LIMIT_UPDATED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"

    # Security actions
    API_KEY_ACCESSED = "API_KEY_ACCESSED"
    API_KEY_ROTATED = "API_KEY_ROTATED"
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"

    # Configuration actions
    CONFIG_UPDATED = "CONFIG_UPDATED"
    STRATEGY_ENABLED = "STRATEGY_ENABLED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    PARAMETER_CHANGED = "PARAMETER_CHANGED"

    # Data actions
    DATA_SYNC_STARTED = "DATA_SYNC_STARTED"
    DATA_SYNC_COMPLETED = "DATA_SYNC_COMPLETED"
    DATA_SYNC_FAILED = "DATA_SYNC_FAILED"

    # System actions
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    COMPONENT_STARTED = "COMPONENT_STARTED"
    COMPONENT_STOPPED = "COMPONENT_STOPPED"
    HEALTHCHECK_FAILED = "HEALTHCHECK_FAILED"


# ============================================================================
# Audit Event Data Class
# ============================================================================

@dataclass
class AuditEvent:
    """Represents a single audit event"""

    # Required fields
    system_component: str
    action_type: str
    action_category: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: AuditSeverity = AuditSeverity.INFO
    success: bool = True

    # User/Session
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Resource
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_details: Optional[Dict[str, Any]] = None

    # Action details
    action_details: Optional[Dict[str, Any]] = None
    previous_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Compliance
    compliance_level: Optional[str] = None
    requires_review: bool = False

    # Metadata
    environment: str = "development"
    version: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        # Convert enums to strings
        if isinstance(self.severity, AuditSeverity):
            data['severity'] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


# ============================================================================
# Audit Logger Class
# ============================================================================

class AuditLogger:
    """
    Comprehensive audit logging system

    Features:
    - Database persistence (Supabase)
    - File-based fallback
    - Async support (future)
    - Filtering and querying
    - Compliance reporting
    """

    def __init__(self,
                 system_component: str,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 fallback_file: Optional[str] = None,
                 environment: str = "development"):
        """
        Initialize Audit Logger

        Args:
            system_component: Name of the component (e.g., "trading-engine")
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            fallback_file: Path to fallback log file
            environment: Environment name (development, staging, production)
        """
        self.system_component = system_component
        self.environment = environment
        self.logger = logging.getLogger(f"audit.{system_component}")

        # Initialize Supabase client
        self.supabase: Optional[Client] = None
        if SUPABASE_AVAILABLE and supabase_url and supabase_key:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                self.logger.info(f"Audit logger initialized with Supabase for {system_component}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Supabase client: {e}")

        # Fallback file logging
        self.fallback_file = fallback_file
        if fallback_file:
            os.makedirs(os.path.dirname(fallback_file), exist_ok=True)
            self.logger.info(f"Fallback logging to: {fallback_file}")

        # Statistics
        self.events_logged = 0
        self.events_failed = 0

    def log(self,
            action_type: str,
            action_category: str,
            severity: AuditSeverity = AuditSeverity.INFO,
            success: bool = True,
            user_id: Optional[str] = None,
            resource_type: Optional[str] = None,
            resource_id: Optional[str] = None,
            resource_details: Optional[Dict[str, Any]] = None,
            action_details: Optional[Dict[str, Any]] = None,
            previous_value: Optional[Dict[str, Any]] = None,
            new_value: Optional[Dict[str, Any]] = None,
            error_message: Optional[str] = None,
            error_code: Optional[str] = None,
            requires_review: bool = False,
            correlation_id: Optional[str] = None,
            **kwargs) -> bool:
        """
        Log an audit event

        Args:
            action_type: Type of action (use AuditAction enum)
            action_category: Category (use AuditCategory enum)
            severity: Severity level
            success: Whether action succeeded
            user_id: User who performed action
            resource_type: Type of resource affected
            resource_id: ID of resource
            resource_details: Additional resource details
            action_details: Details about the action
            previous_value: Previous state (for updates)
            new_value: New state (for updates)
            error_message: Error message if failed
            error_code: Error code if failed
            requires_review: Flag for manual review
            correlation_id: ID to correlate related events
            **kwargs: Additional fields

        Returns:
            bool: True if logged successfully
        """
        try:
            # Create audit event
            event = AuditEvent(
                system_component=self.system_component,
                action_type=action_type,
                action_category=action_category,
                severity=severity,
                success=success,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_details=resource_details,
                action_details=action_details,
                previous_value=previous_value,
                new_value=new_value,
                error_message=error_message,
                error_code=error_code,
                requires_review=requires_review,
                correlation_id=correlation_id,
                environment=self.environment,
                **kwargs
            )

            # Log to Supabase
            if self.supabase:
                try:
                    self.supabase.table('audit_logs').insert(event.to_dict()).execute()
                    self.events_logged += 1
                    self.logger.debug(f"Logged audit event: {action_type}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to log to Supabase: {e}")
                    self.events_failed += 1
                    # Fall through to file logging

            # Fallback to file logging
            if self.fallback_file:
                try:
                    with open(self.fallback_file, 'a') as f:
                        f.write(event.to_json() + '\n')
                    self.events_logged += 1
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to log to file: {e}")
                    self.events_failed += 1

            return False

        except Exception as e:
            self.logger.error(f"Error in audit logging: {e}")
            self.events_failed += 1
            return False

    # ========================================================================
    # Convenience Methods for Common Actions
    # ========================================================================

    def log_order(self,
                  action: str,
                  order_id: str,
                  order_details: Dict[str, Any],
                  success: bool = True,
                  error: Optional[str] = None,
                  user_id: Optional[str] = None) -> bool:
        """Log an order-related action"""
        return self.log(
            action_type=action,
            action_category=AuditCategory.TRADING.value,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            success=success,
            user_id=user_id,
            resource_type="order",
            resource_id=order_id,
            resource_details=order_details,
            error_message=error
        )

    def log_position(self,
                     action: str,
                     position_id: str,
                     position_details: Dict[str, Any],
                     success: bool = True,
                     user_id: Optional[str] = None) -> bool:
        """Log a position-related action"""
        return self.log(
            action_type=action,
            action_category=AuditCategory.TRADING.value,
            severity=AuditSeverity.INFO,
            success=success,
            user_id=user_id,
            resource_type="position",
            resource_id=position_id,
            resource_details=position_details
        )

    def log_risk_event(self,
                       action: str,
                       risk_details: Dict[str, Any],
                       severity: AuditSeverity = AuditSeverity.WARNING,
                       requires_review: bool = True) -> bool:
        """Log a risk-related event"""
        return self.log(
            action_type=action,
            action_category=AuditCategory.RISK.value,
            severity=severity,
            resource_type="risk_limit",
            action_details=risk_details,
            requires_review=requires_review
        )

    def log_config_change(self,
                          parameter: str,
                          old_value: Any,
                          new_value: Any,
                          user_id: Optional[str] = None) -> bool:
        """Log a configuration change"""
        return self.log(
            action_type=AuditAction.CONFIG_UPDATED.value,
            action_category=AuditCategory.CONFIG.value,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            resource_type="config",
            resource_id=parameter,
            previous_value={"value": old_value},
            new_value={"value": new_value},
            requires_review=True
        )

    def log_security_event(self,
                           action: str,
                           details: Dict[str, Any],
                           severity: AuditSeverity = AuditSeverity.WARNING,
                           ip_address: Optional[str] = None) -> bool:
        """Log a security-related event"""
        return self.log(
            action_type=action,
            action_category=AuditCategory.SECURITY.value,
            severity=severity,
            action_details=details,
            ip_address=ip_address,
            requires_review=True
        )

    def log_api_key_access(self,
                           key_name: str,
                           accessed_by: str,
                           ip_address: Optional[str] = None) -> bool:
        """Log API key access"""
        return self.log(
            action_type=AuditAction.API_KEY_ACCESSED.value,
            action_category=AuditCategory.SECURITY.value,
            severity=AuditSeverity.INFO,
            resource_type="api_key",
            resource_id=key_name,
            action_details={"accessed_by": accessed_by},
            ip_address=ip_address
        )

    # ========================================================================
    # Query Methods
    # ========================================================================

    def query_recent(self,
                     limit: int = 100,
                     severity: Optional[AuditSeverity] = None) -> List[Dict[str, Any]]:
        """
        Query recent audit logs

        Args:
            limit: Maximum number of records
            severity: Filter by severity

        Returns:
            list: List of audit events
        """
        if not self.supabase:
            self.logger.warning("Supabase not available for queries")
            return []

        try:
            query = self.supabase.table('audit_logs').select('*')

            if severity:
                query = query.eq('severity', severity.value)

            query = query.order('timestamp', desc=True).limit(limit)

            result = query.execute()
            return result.data or []

        except Exception as e:
            self.logger.error(f"Failed to query audit logs: {e}")
            return []

    @lru_cache(maxsize=128)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        return {
            "system_component": self.system_component,
            "events_logged": self.events_logged,
            "events_failed": self.events_failed,
            "success_rate": (
                self.events_logged / (self.events_logged + self.events_failed)
                if (self.events_logged + self.events_failed) > 0
                else 0
            ),
            "supabase_enabled": self.supabase is not None,
            "file_logging_enabled": self.fallback_file is not None
        }

    def __repr__(self) -> str:
        return (f"AuditLogger(component={self.system_component}, "
                f"logged={self.events_logged}, failed={self.events_failed})")


# ============================================================================
# Singleton Factory
# ============================================================================

_audit_loggers: Dict[str, AuditLogger] = {}


@lru_cache(maxsize=128)


def get_audit_logger(system_component: str,
                     supabase_url: Optional[str] = None,
                     supabase_key: Optional[str] = None,
                     fallback_file: Optional[str] = None,
                     environment: Optional[str] = None) -> AuditLogger:
    """
    Get or create an audit logger for a component (singleton pattern)

    Args:
        system_component: Name of the component
        supabase_url: Supabase URL (uses env var if not provided)
        supabase_key: Supabase key (uses env var if not provided)
        fallback_file: Fallback file path
        environment: Environment name

    Returns:
        AuditLogger: Audit logger instance
    """
    if system_component not in _audit_loggers:
        # Use environment variables if not provided
        supabase_url = supabase_url or os.environ.get('SUPABASE_URL')
        supabase_key = supabase_key or os.environ.get('SUPABASE_SERVICE_KEY')
        environment = environment or os.environ.get('ENVIRONMENT', 'development')

        # Default fallback file
        if fallback_file is None:
            fallback_file = f"/Volumes/Lexar/RRRVentures/RRRalgorithms/logs/audit/{system_component}.jsonl"

        _audit_loggers[system_component] = AuditLogger(
            system_component=system_component,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            fallback_file=fallback_file,
            environment=environment
        )

    return _audit_loggers[system_component]
