from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
import os


"""
Configuration and environment management utilities.

Provides functions to discover project structure and load environment variables
without hardcoded paths.
"""



@lru_cache(maxsize=128)


def get_project_root() -> Path:
    """
    Find project root by looking for marker files.

    Searches up the directory tree for CLAUDE.md or .git directory.

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If project root cannot be found
    """
    # Start from this file's location
    current = Path(__file__).resolve()

    # Search up to 10 levels
    for parent in list(current.parents)[:10]:
        # Check for marker files/directories
        if (parent / "CLAUDE.md").exists() or (parent / ".git").exists():
            return parent

    # Fallback: check environment variable
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"])

    raise RuntimeError(
        "Could not find project root. Make sure CLAUDE.md exists in project root "
        "or set PROJECT_ROOT environment variable."
    )


@lru_cache(maxsize=128)


def get_env_file(worktree: Optional[str] = None) -> Path:
    """
    Get path to .env file.

    Args:
        worktree: Optional worktree name (e.g., 'data-pipeline')
                 If None, returns the main project .env file

    Returns:
        Path to .env file
    """
    root = get_project_root()

    if worktree:
        # Look for worktree-specific .env
        worktree_env = root / "worktrees" / worktree / ".env"
        if worktree_env.exists():
            return worktree_env

    # Return main .env file
    return root / "config" / "api-keys" / ".env"


def load_config(worktree: Optional[str] = None, override: bool = True) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        worktree: Optional worktree name
        override: Whether to override existing environment variables

    Returns:
        Dictionary of loaded environment variables
    """
    env_file = get_env_file(worktree)

    if not env_file.exists():
        raise FileNotFoundError(
            f"Environment file not found: {env_file}\n"
            f"Expected location: {env_file}"
        )

    # Load environment variables
    load_dotenv(env_file, override=override)

    # Return all environment variables that were in the file
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key = line.split('=', 1)[0]
                env_vars[key] = os.getenv(key, '')

    return env_vars


def get_config_value(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get a configuration value from environment.

    Args:
        key: Configuration key to retrieve
        default: Default value if not found
        required: If True, raises ValueError if not found

    Returns:
        Configuration value or default

    Raises:
        ValueError: If required=True and key not found
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(
            f"Required configuration '{key}' not found in environment. "
            f"Please set it in {get_env_file()}"
        )

    return value


def get_database_url(required: bool = True) -> str:
    """Get database URL from environment."""
    return get_config_value("DATABASE_URL", required=required) or \
           get_config_value("SUPABASE_DB_URL", required=required) or ""


def get_supabase_config() -> Dict[str, str]:
    """Get Supabase configuration."""
    return {
        "url": get_config_value("DATABASE_PATH", required=True),
        "anon_key": get_config_value("SUPABASE_ANON_KEY", required=False) or "",
        "service_key": get_config_value("DATABASE_TYPE", required=False) or "",
    }


def is_development() -> bool:
    """Check if running in development mode."""
    return get_config_value("ENVIRONMENT", "development").lower() in ["dev", "development"]


def is_production() -> bool:
    """Check if running in production mode."""
    return get_config_value("ENVIRONMENT", "development").lower() in ["prod", "production"]


def is_paper_trading() -> bool:
    """Check if paper trading is enabled."""
    return get_config_value("PAPER_TRADING", "true").lower() in ["true", "1", "yes"]


def is_live_trading() -> bool:
    """Check if live trading is enabled."""
    return get_config_value("LIVE_TRADING", "false").lower() in ["true", "1", "yes"]
