from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from src.database import get_db, Client
from typing import Optional, Dict, Any
import json
import logging
import os
import traceback


"""
Centralized Logging Service for RRRalgorithms Trading System

This module provides a comprehensive logging infrastructure that:
- Writes logs to both local files and Supabase database
- Supports multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Provides structured logging with context
- Thread-safe for concurrent operations
"""


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)


class SupabaseLogHandler(logging.Handler):
    """Custom logging handler that writes logs to Supabase"""

    def __init__(self, supabase_client: Client):
        super().__init__()
        self.supabase = supabase_client
        self.batch = []
        self.batch_size = 10

    def emit(self, record: logging.LogRecord):
        """Emit a log record to Supabase"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "log",
                "severity": record.levelname,
                "component": record.name,
                "message": self.format(record),
                "metadata": {
                    "filename": record.filename,
                    "function": record.funcName,
                    "line_number": record.lineno,
                    "process_id": record.process,
                    "thread_id": record.thread
                }
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["metadata"]["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exception(*record.exc_info)
                }

            # Batch insert for performance
            self.batch.append(log_entry)
            if len(self.batch) >= self.batch_size:
                self._flush_batch()

        except Exception as e:
            # Fallback to stderr if Supabase fails
            print(f"Error writing log to Supabase: {e}", flush=True)

    def _flush_batch(self):
        """Flush batch of logs to Supabase"""
        if self.batch:
            try:
                self.supabase.table("system_events").insert(self.batch).execute()
                self.batch = []
            except Exception as e:
                print(f"Error flushing log batch to Supabase: {e}", flush=True)
                self.batch = []

    def close(self):
        """Flush remaining logs on close"""
        self._flush_batch()
        super().close()


class LoggerService:
    """
    Centralized logging service for the trading system

    Features:
    - Multi-destination logging (file + Supabase)
    - Structured logging with context
    - Automatic log rotation
    - Thread-safe operations
    """

    def __init__(
        self,
        component_name: str,
        log_level: str = None,
        enable_supabase: bool = True,
        enable_file: bool = True
    ):
        """
        Initialize logger service

        Args:
            component_name: Name of the component using this logger
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_supabase: Whether to log to Supabase
            enable_file: Whether to log to local files
        """
        self.component_name = component_name
        self.log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        self.enable_supabase = enable_supabase
        self.enable_file = enable_file

        # Initialize Supabase client
        self.supabase_client = None
        if enable_supabase:
            try:
                supabase_url = os.getenv("DATABASE_PATH")
                supabase_key = os.getenv("SUPABASE_ANON_KEY")
                if supabase_url and supabase_key:
                    self.supabase_client = get_db()
            except Exception as e:
                print(f"Warning: Failed to initialize Supabase client: {e}")

        # Set up logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger with handlers"""
        logger = logging.getLogger(self.component_name)
        logger.setLevel(getattr(logging, self.log_level))

        # Clear any existing handlers
        logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.enable_file:
            log_dir = Path(__file__).resolve().parents[2] / "logs"
            log_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(
                log_dir / f"{self.component_name}.log"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add Supabase handler if enabled
        if self.enable_supabase and self.supabase_client:
            supabase_handler = SupabaseLogHandler(self.supabase_client)
            supabase_handler.setFormatter(formatter)
            logger.addHandler(supabase_handler)

        return logger

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(self._format_message(message, context))

    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(self._format_message(message, context))

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(self._format_message(message, context))

    def error(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = True):
        """Log error message"""
        self.logger.error(self._format_message(message, context), exc_info=exc_info)

    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = True):
        """Log critical message"""
        self.logger.critical(self._format_message(message, context), exc_info=exc_info)

    def _format_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format message with optional context"""
        if context:
            return f"{message} | Context: {json.dumps(context)}"
        return message

    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log structured event directly to Supabase

        Args:
            event_type: Type of event (e.g., 'trade', 'signal', 'error')
            message: Event message
            severity: Severity level
            metadata: Additional metadata
        """
        if not self.supabase_client:
            return

        try:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "severity": severity,
                "component": self.component_name,
                "message": message,
                "metadata": metadata or {}
            }

            self.supabase_client.table("system_events").insert(event).execute()
        except Exception as e:
            self.logger.error(f"Failed to log event to Supabase: {e}")


# Global logger registry
_loggers: Dict[str, LoggerService] = {}


@lru_cache(maxsize=128)


def get_logger(component_name: str, **kwargs) -> LoggerService:
    """
    Get or create a logger for a component

    Args:
        component_name: Name of the component
        **kwargs: Additional arguments for LoggerService

    Returns:
        LoggerService instance
    """
    if component_name not in _loggers:
        _loggers[component_name] = LoggerService(component_name, **kwargs)
    return _loggers[component_name]


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_logger("test_component")

    # Test different log levels
    logger.debug("This is a debug message", {"user": "test", "value": 123})
    logger.info("System started successfully")
    logger.warning("High memory usage detected", {"memory_mb": 8192})

    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero error occurred", {"operation": "divide"})

    # Log structured event
    logger.log_event(
        event_type="trade_executed",
        message="Buy order filled",
        severity="INFO",
        metadata={
            "symbol": "BTC-USD",
            "quantity": 0.5,
            "price": 45000.0
        }
    )

    print("Logging test completed. Check logs/ directory and Supabase system_events table.")
