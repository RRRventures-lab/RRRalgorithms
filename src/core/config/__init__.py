from ..config_legacy import get_project_root, get_env_file, load_config
from .loader import get_config, config_get, ConfigLoader

"""
Configuration Module
====================

Configuration loading and management.

Provides both new (ConfigLoader-based) and legacy (dotenv-based) config functions.
"""

# Re-export legacy functions for backward compatibility
from ..config_legacy import (
    get_config_value,
    get_database_url,
    get_supabase_config,
    is_development,
    is_production,
    is_paper_trading,
    is_live_trading,
)

__all__ = [
    # New config system
    'get_config',
    'config_get',
    'ConfigLoader',
    # Legacy config system
    'get_project_root',
    'get_env_file',
    'load_config',
    'get_config_value',
    'get_database_url',
    'get_supabase_config',
    'is_development',
    'is_production',
    'is_paper_trading',
    'is_live_trading',
]
