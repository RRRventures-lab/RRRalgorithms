from functools import lru_cache
from typing import Optional, Dict, Any
import json
import logging
import subprocess


"""
macOS Keychain Manager
Provides secure storage and retrieval of API keys using macOS Keychain
"""


logger = logging.getLogger(__name__)


class KeychainManager:
    """
    Manages secrets in macOS Keychain

    Uses security command-line tool to interact with Keychain
    Stores secrets as generic passwords with service name and account
    """

    def __init__(self, service_name: str = "RRRalgorithms"):
        """
        Initialize Keychain Manager

        Args:
            service_name: Name of the service (used as keychain item identifier)
        """
        self.service_name = service_name
        logger.info(f"Initialized KeychainManager for service: {service_name}")

    def store_secret(self, account: str, secret: str) -> bool:
        """
        Store a secret in macOS Keychain

        Args:
            account: Account name (e.g., "POLYGON_API_KEY")
            secret: Secret value to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, try to delete existing item (if any)
            self.delete_secret(account)

            # Add new secret
            cmd = [
                "security",
                "add-generic-password",
                "-s", self.service_name,
                "-a", account,
                "-w", secret,
                "-U"  # Update if exists
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                logger.info(f"Successfully stored secret for account: {account}")
                return True
            else:
                logger.error(f"Failed to store secret for {account}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error storing secret for {account}: {e}")
            return False

    @lru_cache(maxsize=128)

    def get_secret(self, account: str) -> Optional[str]:
        """
        Retrieve a secret from macOS Keychain

        Args:
            account: Account name (e.g., "POLYGON_API_KEY")

        Returns:
            str: Secret value if found, None otherwise
        """
        try:
            cmd = [
                "security",
                "find-generic-password",
                "-s", self.service_name,
                "-a", account,
                "-w"  # Return password only
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                secret = result.stdout.strip()
                logger.debug(f"Successfully retrieved secret for account: {account}")
                return secret
            else:
                logger.warning(f"Secret not found for account: {account}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving secret for {account}: {e}")
            return None

    def delete_secret(self, account: str) -> bool:
        """
        Delete a secret from macOS Keychain

        Args:
            account: Account name

        Returns:
            bool: True if deleted or doesn't exist, False on error
        """
        try:
            cmd = [
                "security",
                "delete-generic-password",
                "-s", self.service_name,
                "-a", account
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            # Return code 0 = deleted, 44 = not found (both OK)
            if result.returncode in [0, 44]:
                logger.debug(f"Secret deleted or doesn't exist: {account}")
                return True
            else:
                logger.error(f"Failed to delete secret {account}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error deleting secret for {account}: {e}")
            return False

    def list_accounts(self) -> list:
        """
        List all accounts stored for this service

        Returns:
            list: List of account names
        """
        try:
            cmd = [
                "security",
                "dump-keychain"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            # Parse output to find matching service entries
            accounts = []
            lines = result.stdout.split('\n')

            for i, line in enumerate(lines):
                if f'svce<blob>="{self.service_name}"' in line:
                    # Look for account name in nearby lines
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if 'acct<blob>="' in lines[j]:
                            account = lines[j].split('acct<blob>="')[1].split('"')[0]
                            accounts.append(account)
                            break

            logger.info(f"Found {len(accounts)} accounts in keychain")
            return accounts

        except Exception as e:
            logger.error(f"Error listing accounts: {e}")
            return []

    def store_multiple(self, secrets: Dict[str, str]) -> Dict[str, bool]:
        """
        Store multiple secrets at once

        Args:
            secrets: Dictionary of {account: secret}

        Returns:
            dict: Dictionary of {account: success_bool}
        """
        results = {}
        for account, secret in secrets.items():
            results[account] = self.store_secret(account, secret)
        return results

    @lru_cache(maxsize=128)

    def get_multiple(self, accounts: list) -> Dict[str, Optional[str]]:
        """
        Retrieve multiple secrets at once

        Args:
            accounts: List of account names

        Returns:
            dict: Dictionary of {account: secret or None}
        """
        results = {}
        for account in accounts:
            results[account] = self.get_secret(account)
        return results

    def export_to_dict(self, accounts: list) -> Dict[str, str]:
        """
        Export secrets to dictionary format (for migration)
        WARNING: This exposes secrets in memory - use with caution

        Args:
            accounts: List of account names to export

        Returns:
            dict: Dictionary of {account: secret} (only successful retrievals)
        """
        logger.warning("Exporting secrets to dictionary - this exposes secrets in memory")
        results = self.get_multiple(accounts)
        return {k: v for k, v in results.items() if v is not None}
