from datetime import datetime
from pathlib import Path
from security.secrets_manager import SecretsManager
import argparse
import os
import shutil
import sys

#!/usr/bin/env python3
"""
Migrate secrets from plaintext .env file to macOS Keychain

This script:
1. Reads secrets from config/api-keys/.env
2. Stores them securely in macOS Keychain
3. Creates a backup of the original .env file
4. Optionally removes secrets from .env file

Usage:
    python scripts/security/migrate_secrets.py [--remove-plaintext] [--backup]
"""


# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))



def backup_env_file(env_path: Path) -> Path:
    """Create a timestamped backup of .env file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = env_path.parent / f".env.backup.{timestamp}"
    shutil.copy2(env_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def remove_secrets_from_env(env_path: Path, secrets_manager: SecretsManager):
    """Remove secret values from .env file, keeping config values"""
    print("\n🔒 Removing secrets from .env file...")

    lines = []
    secrets_removed = 0

    with open(env_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            # Keep comments and empty lines
            if not stripped or stripped.startswith('#'):
                lines.append(line)
                continue

            # Parse KEY=VALUE
            if '=' in stripped:
                key = stripped.split('=', 1)[0].strip()

                # If it's a secret, replace with placeholder
                if key in secrets_manager.SECRET_KEYS:
                    lines.append(f"# {key}=<stored in keychain>\n")
                    secrets_removed += 1
                else:
                    # Keep config values
                    lines.append(line)
            else:
                lines.append(line)

    # Write back
    with open(env_path, 'w') as f:
        f.writelines(lines)

    print(f"✓ Removed {secrets_removed} secrets from .env file")
    print(f"✓ Config values preserved in .env file")


def verify_migration(secrets_manager: SecretsManager):
    """Verify that secrets were migrated successfully"""
    print("\n🔍 Verifying migration...")

    results = secrets_manager.verify_secrets()
    available = [k for k, v in results.items() if v]
    missing = [k for k, v in results.items() if not v]

    print(f"\n✓ Successfully migrated: {len(available)} secrets")
    for key in available:
        print(f"  ✓ {key}")

    if missing:
        print(f"\n⚠ Missing secrets: {len(missing)}")
        for key in missing:
            print(f"  ✗ {key}")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Migrate secrets to macOS Keychain")
    parser.add_argument(
        "--remove-plaintext",
        action="store_true",
        help="Remove secrets from .env file after migration"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before modifying .env file (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="config/api-keys/.env",
        help="Path to .env file (relative to project root)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RRRalgorithms Secrets Migration Tool")
    print("=" * 80)

    # Check if running on macOS
    if os.uname().sysname != "Darwin":
        print("\n❌ ERROR: This tool requires macOS")
        print("For other platforms, use environment variables or encrypted .env files")
        sys.exit(1)

    # Resolve paths
    env_path = project_root / args.env_file

    if not env_path.exists():
        print(f"\n❌ ERROR: .env file not found: {env_path}")
        sys.exit(1)

    print(f"\n📁 Project root: {project_root}")
    print(f"📄 .env file: {env_path}")

    # Create backup if requested
    if args.backup and not args.no_backup:
        backup_path = backup_env_file(env_path)

    # Initialize secrets manager
    print("\n🔑 Initializing Secrets Manager...")
    secrets_manager = SecretsManager(
        service_name="RRRalgorithms",
        use_keychain=True,
        fallback_to_env=False
    )

    # Migrate secrets
    print("\n📦 Migrating secrets to Keychain...")
    results = secrets_manager.migrate_from_env_file(str(env_path))

    successful = sum(results.values())
    total = len(results)

    print(f"\n✓ Migration complete: {successful}/{total} secrets migrated")

    # Verify migration
    all_verified = verify_migration(secrets_manager)

    # Remove plaintext secrets if requested
    if args.remove_plaintext and all_verified:
        response = input("\n⚠️  Remove secrets from .env file? (y/N): ")
        if response.lower() == 'y':
            remove_secrets_from_env(env_path, secrets_manager)
        else:
            print("Skipped removing secrets from .env file")

    # Final instructions
    print("\n" + "=" * 80)
    print("✅ Migration Complete!")
    print("=" * 80)

    if all_verified:
        print("\n✓ All secrets successfully stored in Keychain")
        print("\nNext steps:")
        print("1. Update your code to use SecretsManager instead of python-dotenv")
        print("2. Test your application to ensure secrets are loading correctly")
        if args.remove_plaintext:
            print("3. Secrets removed from .env file - application will use Keychain")
        else:
            print("3. Consider removing secrets from .env file with --remove-plaintext")
        print("4. IMPORTANT: Rotate all exposed API keys (see docs/security/API_KEY_ROTATION_GUIDE.md)")
    else:
        print("\n⚠️  Some secrets were not migrated successfully")
        print("Review the errors above and try again")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
