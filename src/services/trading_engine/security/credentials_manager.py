"""
Credentials Manager for Trading Engine

Securely manages API credentials for exchanges and services.
Supports multiple backends: environment variables, encrypted files, system keychain.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from functools import lru_cache


logger = logging.getLogger(__name__)


class CredentialsManager:
    """
    Secure credentials management for trading engine

    Supports multiple storage backends:
    - Environment variables (default)
    - Encrypted configuration files
    - System keychain (future)
    - AWS Secrets Manager (future)
    - HashiCorp Vault (future)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize credentials manager

        Args:
            config_dir: Path to configuration directory (default: project root/config/api-keys)
        """
        if config_dir is None:
            # Default to project root/config/api-keys
            config_dir = Path(__file__).parent.parent.parent.parent.parent / "config" / "api-keys"

        self.config_dir = Path(config_dir)

        # Load environment variables from .env files
        self._load_env_files()

        # Encryption key for sensitive data
        self.encryption_key = os.getenv("ENCRYPTION_KEY")
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())
        else:
            self.cipher = None
            logger.warning("ENCRYPTION_KEY not set - encrypted storage unavailable")

        logger.info(f"Credentials Manager initialized with config_dir: {self.config_dir}")

    def _load_env_files(self):
        """Load environment variables from .env files"""
        # Load main .env file
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")

        # Load Coinbase-specific credentials
        coinbase_env = self.config_dir / ".env.coinbase"
        if coinbase_env.exists():
            load_dotenv(coinbase_env)
            logger.info(f"Loaded Coinbase credentials from {coinbase_env}")

    @lru_cache(maxsize=128)
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a credential by key

        Args:
            key: Credential key (e.g., "COINBASE_API_KEY")
            default: Default value if not found

        Returns:
            Credential value or default
        """
        # Try environment variable first
        value = os.getenv(key, default)

        if value is None:
            logger.warning(f"Credential not found: {key}")

        return value

    def get_coinbase_credentials(self) -> Dict[str, str]:
        """
        Get Coinbase API credentials

        Returns:
            Dictionary with api_key and private_key

        Raises:
            ValueError: If credentials are not found
        """
        api_key = self.get_credential("COINBASE_API_KEY")
        private_key = self.get_credential("COINBASE_PRIVATE_KEY")

        if not api_key or not private_key:
            raise ValueError(
                "Coinbase credentials not found. Please set COINBASE_API_KEY and "
                "COINBASE_PRIVATE_KEY in environment or .env.coinbase file"
            )

        return {
            "api_key": api_key,
            "private_key": private_key,
            "organization_id": self.get_credential("COINBASE_ORGANIZATION_ID", ""),
            "api_key_name": self.get_credential("COINBASE_API_KEY_NAME", ""),
        }

    def get_supabase_credentials(self) -> Dict[str, str]:
        """Get Supabase credentials"""
        url = self.get_credential("DATABASE_PATH") or self.get_credential("SUPABASE_URL")
        key = self.get_credential("SUPABASE_ANON_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase credentials not found. Please set DATABASE_PATH/SUPABASE_URL "
                "and SUPABASE_ANON_KEY in environment"
            )

        return {
            "url": url,
            "key": key,
            "service_key": self.get_credential("SUPABASE_SERVICE_KEY", ""),
        }

    def is_paper_trading(self) -> bool:
        """
        Check if paper trading mode is enabled

        Returns:
            True if paper trading, False for live trading
        """
        paper_trading = self.get_credential("PAPER_TRADING", "true").lower()
        return paper_trading in ["true", "1", "yes", "on"]

    def is_live_trading_enabled(self) -> bool:
        """
        Check if live trading is explicitly enabled

        Returns:
            True only if LIVE_TRADING_ENABLED=true
        """
        live_enabled = self.get_credential("LIVE_TRADING_ENABLED", "false").lower()
        return live_enabled in ["true", "1", "yes", "on"]

    def validate_live_trading_safety(self) -> tuple[bool, list[str]]:
        """
        Validate safety checks before enabling live trading

        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []

        # Check paper trading is disabled
        if self.is_paper_trading():
            warnings.append("PAPER_TRADING is still enabled - set to false for live trading")

        # Check live trading is explicitly enabled
        if not self.is_live_trading_enabled():
            warnings.append("LIVE_TRADING_ENABLED is not set to true")

        # Check environment
        env = self.get_credential("ENVIRONMENT", "development")
        if env != "production":
            warnings.append(f"ENVIRONMENT={env} - should be 'production' for live trading")

        # Check credentials exist
        try:
            self.get_coinbase_credentials()
        except ValueError as e:
            warnings.append(f"Coinbase credentials error: {e}")

        # Check risk limits are set
        risk_params = [
            "MAX_ORDER_SIZE_USD",
            "MAX_DAILY_VOLUME_USD",
            "MAX_OPEN_POSITIONS",
            "MAX_DAILY_LOSS_USD",
        ]
        for param in risk_params:
            if not self.get_credential(param):
                warnings.append(f"Risk parameter {param} not set")

        is_safe = len(warnings) == 0

        return is_safe, warnings

    def get_risk_limits(self) -> Dict[str, Any]:
        """
        Get risk management limits

        Returns:
            Dictionary of risk limits
        """
        return {
            "max_order_size_usd": float(self.get_credential("MAX_ORDER_SIZE_USD", "100.0")),
            "max_daily_volume_usd": float(self.get_credential("MAX_DAILY_VOLUME_USD", "500.0")),
            "max_open_positions": int(self.get_credential("MAX_OPEN_POSITIONS", "3")),
            "max_loss_per_trade_usd": float(self.get_credential("MAX_LOSS_PER_TRADE_USD", "50.0")),
            "max_daily_loss_usd": float(self.get_credential("MAX_DAILY_LOSS_USD", "100.0")),
            "max_position_size": float(self.get_credential("MAX_POSITION_SIZE", "0.20")),
            "max_portfolio_volatility": float(self.get_credential("MAX_PORTFOLIO_VOLATILITY", "0.25")),
        }

    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a value for secure storage

        Args:
            value: Plain text value

        Returns:
            Encrypted value (base64)
        """
        if not self.cipher:
            raise ValueError("Encryption not available - ENCRYPTION_KEY not set")

        return self.cipher.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt an encrypted value

        Args:
            encrypted_value: Encrypted value (base64)

        Returns:
            Decrypted plain text
        """
        if not self.cipher:
            raise ValueError("Encryption not available - ENCRYPTION_KEY not set")

        return self.cipher.decrypt(encrypted_value.encode()).decode()

    def save_encrypted_credentials(self, credentials: Dict[str, str], filename: str):
        """
        Save credentials to encrypted file

        Args:
            credentials: Dictionary of credentials
            filename: Output filename (e.g., "credentials.enc")
        """
        if not self.cipher:
            raise ValueError("Encryption not available - ENCRYPTION_KEY not set")

        # Encrypt the entire JSON
        json_data = json.dumps(credentials)
        encrypted_data = self.cipher.encrypt(json_data.encode())

        # Write to file
        output_path = self.config_dir / filename
        output_path.write_bytes(encrypted_data)

        logger.info(f"Saved encrypted credentials to {output_path}")

    def load_encrypted_credentials(self, filename: str) -> Dict[str, str]:
        """
        Load credentials from encrypted file

        Args:
            filename: Input filename

        Returns:
            Dictionary of credentials
        """
        if not self.cipher:
            raise ValueError("Encryption not available - ENCRYPTION_KEY not set")

        # Read encrypted file
        input_path = self.config_dir / filename
        encrypted_data = input_path.read_bytes()

        # Decrypt and parse JSON
        json_data = self.cipher.decrypt(encrypted_data).decode()
        credentials = json.loads(json_data)

        logger.info(f"Loaded encrypted credentials from {input_path}")

        return credentials


# Global credentials manager instance
_credentials_manager: Optional[CredentialsManager] = None


def get_credentials_manager() -> CredentialsManager:
    """
    Get global credentials manager instance

    Returns:
        CredentialsManager instance
    """
    global _credentials_manager

    if _credentials_manager is None:
        _credentials_manager = CredentialsManager()

    return _credentials_manager


def main():
    """Example usage and testing"""
    print("=" * 70)
    print("Credentials Manager - Configuration Check")
    print("=" * 70)

    try:
        manager = get_credentials_manager()

        # Check paper trading mode
        print(f"\nPaper Trading: {manager.is_paper_trading()}")
        print(f"Live Trading Enabled: {manager.is_live_trading_enabled()}")

        # Check Coinbase credentials
        print("\nCoinbase Credentials:")
        try:
            creds = manager.get_coinbase_credentials()
            print(f"  API Key: {creds['api_key'][:50]}...")
            print(f"  Private Key: {'*' * 20} (hidden)")
            print(f"  Organization: {creds['organization_id']}")
        except ValueError as e:
            print(f"  ERROR: {e}")

        # Check Supabase credentials
        print("\nSupabase Credentials:")
        try:
            creds = manager.get_supabase_credentials()
            print(f"  URL: {creds['url']}")
            print(f"  Key: {creds['key'][:20]}...")
        except ValueError as e:
            print(f"  ERROR: {e}")

        # Check risk limits
        print("\nRisk Limits:")
        limits = manager.get_risk_limits()
        for key, value in limits.items():
            print(f"  {key}: {value}")

        # Validate live trading safety
        print("\n" + "=" * 70)
        print("Live Trading Safety Check")
        print("=" * 70)

        is_safe, warnings = manager.validate_live_trading_safety()

        if is_safe:
            print("\n✓ All safety checks passed - ready for live trading")
        else:
            print("\n✗ Safety checks failed - live trading NOT recommended:")
            for warning in warnings:
                print(f"  - {warning}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
