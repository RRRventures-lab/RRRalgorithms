from .keychain_manager import KeychainManager
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import os


"""
Secrets Manager
High-level interface for managing application secrets
Supports multiple backends: Keychain, encrypted files, environment variables
"""


logger = logging.getLogger(__name__)


class SecretsManager:
    """
    High-level secrets management interface

    Provides unified API for accessing secrets from multiple sources:
    1. macOS Keychain (primary, most secure)
    2. Environment variables (fallback)
    3. Encrypted .env file (fallback for non-macOS)
    """

    # Define all secret keys used in the application
    SECRET_KEYS = [
        # Market Data APIs
        "POLYGON_API_KEY",
        "PERPLEXITY_API_KEY",
        "ANTHROPIC_API_KEY",

        # Database
        "DATABASE_PATH",
        "SUPABASE_ANON_KEY",
        "DATABASE_TYPE",
        "SUPABASE_DB_URL",
        "DATABASE_URL",

        # Exchange APIs
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",

        # Development Tools
        "GITHUB_TOKEN",
        "WANDB_API_KEY",

        # Monitoring
        "SENTRY_DSN",
        "SLACK_WEBHOOK_URL",
        "PAGERDUTY_API_KEY",

        # Security
        "JWT_SECRET",
        "ENCRYPTION_KEY",
    ]

    # Non-secret configuration keys (can be in plaintext)
    CONFIG_KEYS = [
        "POLYGON_RATE_LIMIT",
        "CACHE_TTL",
        "REDIS_URL",
        "MLFLOW_TRACKING_URI",
        "ENVIRONMENT",
        "PROJECT_ROOT",
        "LOG_LEVEL",
        "DEBUG",
        "MAX_POSITION_SIZE",
        "MAX_DAILY_LOSS",
        "MAX_PORTFOLIO_VOLATILITY",
        "PAPER_TRADING",
        "LIVE_TRADING",
        "ENABLE_SENTIMENT",
        "ENABLE_QUANTUM_OPTIMIZATION",
        "ENABLE_MULTI_AGENT",
    ]

    def __init__(self,
                 service_name: str = "RRRalgorithms",
                 use_keychain: bool = True,
                 fallback_to_env: bool = True):
        """
        Initialize Secrets Manager

        Args:
            service_name: Service name for Keychain
            use_keychain: Whether to use Keychain as primary storage
            fallback_to_env: Whether to fallback to environment variables
        """
        self.service_name = service_name
        self.use_keychain = use_keychain and self._is_macos()
        self.fallback_to_env = fallback_to_env

        if self.use_keychain:
            self.keychain = KeychainManager(service_name)
            logger.info("Using macOS Keychain for secrets management")
        else:
            self.keychain = None
            logger.info("Keychain disabled - using environment variables only")

    @staticmethod
    def _is_macos() -> bool:
        """Check if running on macOS"""
        return os.uname().sysname == "Darwin"

    @lru_cache(maxsize=128)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value

        Priority order:
        1. macOS Keychain (if enabled)
        2. Environment variable (if fallback enabled)
        3. Default value

        Args:
            key: Secret key name
            default: Default value if not found

        Returns:
            str: Secret value or default
        """
        # Try Keychain first
        if self.use_keychain and self.keychain:
            value = self.keychain.get_secret(key)
            if value:
                logger.debug(f"Retrieved {key} from Keychain")
                return value

        # Fallback to environment variable
        if self.fallback_to_env:
            value = os.environ.get(key)
            if value:
                logger.debug(f"Retrieved {key} from environment")
                return value

        # Return default
        if default is not None:
            logger.debug(f"Using default value for {key}")
            return default

        logger.warning(f"Secret not found: {key}")
        return None

    def set_secret(self, key: str, value: str) -> bool:
        """
        Store a secret value

        Args:
            key: Secret key name
            value: Secret value

        Returns:
            bool: True if successful
        """
        if self.use_keychain and self.keychain:
            success = self.keychain.store_secret(key, value)
            if success:
                logger.info(f"Stored {key} in Keychain")
            return success
        else:
            logger.warning(f"Cannot store {key} - Keychain not available")
            return False

    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret

        Args:
            key: Secret key name

        Returns:
            bool: True if successful
        """
        if self.use_keychain and self.keychain:
            return self.keychain.delete_secret(key)
        return False

    @lru_cache(maxsize=128)

    def get_all_secrets(self) -> Dict[str, Optional[str]]:
        """
        Get all defined secrets

        Returns:
            dict: Dictionary of all secrets
        """
        secrets = {}
        for key in self.SECRET_KEYS:
            secrets[key] = self.get_secret(key)
        return secrets

    @lru_cache(maxsize=128)

    def get_all_config(self) -> Dict[str, Optional[str]]:
        """
        Get all configuration values (non-secret)

        Returns:
            dict: Dictionary of all config values
        """
        config = {}
        for key in self.CONFIG_KEYS:
            config[key] = os.environ.get(key)
        return config

    def verify_secrets(self) -> Dict[str, bool]:
        """
        Verify that all required secrets are available

        Returns:
            dict: Dictionary of {key: is_available}
        """
        results = {}
        for key in self.SECRET_KEYS:
            value = self.get_secret(key)
            results[key] = value is not None and len(value) > 0
        return results

    def migrate_from_env_file(self, env_file_path: str) -> Dict[str, bool]:
        """
        Migrate secrets from .env file to Keychain

        Args:
            env_file_path: Path to .env file

        Returns:
            dict: Migration results {key: success}
        """
        if not self.use_keychain:
            logger.error("Cannot migrate - Keychain not available")
            return {}

        logger.info(f"Migrating secrets from {env_file_path} to Keychain")

        results = {}
        env_path = Path(env_file_path)

        if not env_path.exists():
            logger.error(f"File not found: {env_file_path}")
            return {}

        # Read .env file
        with open(env_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Only migrate actual secrets (not config)
                    if key in self.SECRET_KEYS:
                        success = self.set_secret(key, value)
                        results[key] = success
                        logger.info(f"Migrated {key}: {'✓' if success else '✗'}")

        logger.info(f"Migration complete. Successful: {sum(results.values())}/{len(results)}")
        return results

    def export_to_env_file(self, env_file_path: str, include_config: bool = True) -> bool:
        """
        Export secrets to .env file (for backup or non-macOS systems)
        WARNING: This creates a plaintext file - use with caution

        Args:
            env_file_path: Path to output .env file
            include_config: Whether to include non-secret config values

        Returns:
            bool: True if successful
        """
        logger.warning(f"Exporting secrets to plaintext file: {env_file_path}")

        try:
            with open(env_file_path, 'w') as f:
                f.write("# RRRalgorithms Environment Configuration\n")
                f.write("# Generated by SecretsManager\n")
                f.write("# WARNING: Contains sensitive credentials\n\n")

                # Export secrets
                f.write("# ============================================================================\n")
                f.write("# API Keys and Secrets\n")
                f.write("# ============================================================================\n\n")

                secrets = self.get_all_secrets()
                for key, value in secrets.items():
                    if value:
                        f.write(f"{key}={value}\n")
                    else:
                        f.write(f"# {key}=<not set>\n")

                # Export config
                if include_config:
                    f.write("\n# ============================================================================\n")
                    f.write("# Configuration Values\n")
                    f.write("# ============================================================================\n\n")

                    config = self.get_all_config()
                    for key, value in config.items():
                        if value:
                            f.write(f"{key}={value}\n")

            logger.info(f"Exported configuration to {env_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return False

    def __repr__(self) -> str:
        return f"SecretsManager(service={self.service_name}, keychain={self.use_keychain})"
