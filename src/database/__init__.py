"""
Database module for RRRalgorithms trading system.
Provides unified interface for SQLite database operations.
"""

from .sqlite_client import SQLiteClient
from .base import DatabaseClient
from .client_factory import get_database_client, get_db

__all__ = [
    'SQLiteClient',
    'DatabaseClient',
    'get_database_client',
    'get_db',
]

