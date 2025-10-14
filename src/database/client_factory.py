"""
Database client factory.
Creates appropriate database client based on configuration.
"""

import os
from typing import Optional
import logging

from .base import DatabaseClient
from .sqlite_client import SQLiteClient


logger = logging.getLogger(__name__)


# Global database client instance
_db_client: Optional[DatabaseClient] = None


def get_database_client(
    db_type: str = 'sqlite',
    db_path: Optional[str] = None,
    **kwargs
) -> DatabaseClient:
    """
    Create and return database client instance.
    
    Args:
        db_type: Type of database ('sqlite' or 'supabase')
        db_path: Path to database file (for SQLite)
        **kwargs: Additional arguments for specific client type
    
    Returns:
        DatabaseClient instance
    """
    db_type = db_type.lower()
    
    if db_type == 'sqlite':
        return SQLiteClient(db_path=db_path, **kwargs)
    elif db_type == 'supabase':
        # Legacy support - redirect to SQLite with warning
        logger.warning(
            "Supabase client requested but system now uses SQLite. "
            "Returning SQLite client instead."
        )
        return SQLiteClient(db_path=db_path, **kwargs)
    else:
        raise ValueError(f"Unknown database type: {db_type}")


def get_db() -> DatabaseClient:
    """
    Get global database client instance.
    Creates new instance if one doesn't exist.
    
    Returns:
        Shared DatabaseClient instance
    """
    global _db_client
    
    if _db_client is None:
        # Determine database type from environment
        db_type = os.getenv('DATABASE_TYPE', 'sqlite')
        
        # Get database path from environment or use default
        db_path = os.getenv('DATABASE_PATH')
        
        # Create client
        _db_client = get_database_client(db_type=db_type, db_path=db_path)
        
        logger.info(f"Created global database client: {db_type}")
    
    return _db_client


def reset_db_client() -> None:
    """Reset global database client (useful for testing)."""
    global _db_client
    if _db_client:
        _db_client = None
        logger.info("Global database client reset")


# Convenience aliases
get_client = get_db
create_client = get_database_client

