# Secrets Management Guide

## Overview

RRRalgorithms uses a secure secrets management system to protect API keys, credentials, and sensitive configuration. This guide explains how to use the system effectively.

---

## Architecture

### Components

1. **macOS Keychain** (Primary Storage)
   - Native macOS secure storage
   - Hardware-backed encryption
   - System-level access control
   - Automatic backup with iCloud Keychain (optional)

2. **SecretsManager** (Application Interface)
   - High-level Python API
   - Automatic fallback to environment variables
   - Singleton pattern for efficiency
   - Logging and audit trail

3. **Environment Variables** (Fallback)
   - Used when Keychain unavailable
   - Development environments
   - CI/CD pipelines

### Data Flow

```
Application → SecretsManager → Keychain → Encrypted Storage
                    ↓
            Environment Variables (fallback)
```

---

## Setup

### 1. Initial Migration

Migrate existing secrets from `.env` to Keychain:

```bash
# From project root
python scripts/security/migrate_secrets.py --backup --remove-plaintext
```

**What this does**:
- Reads secrets from `config/api-keys/.env`
- Stores them securely in macOS Keychain
- Creates backup of original .env file
- Optionally removes secrets from .env (keeps config values)

### 2. Verify Migration

Check that secrets are accessible:

```python
from src.security.secrets_manager import SecretsManager

secrets = SecretsManager()
results = secrets.verify_secrets()

for key, available in results.items():
    print(f"{key}: {'✓' if available else '✗'}")
```

### 3. Update Application Code

Replace direct environment variable access with SecretsManager:

**Before**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('POLYGON_API_KEY')
```

**After**:
```python
from src.security.secrets_manager import SecretsManager

secrets = SecretsManager()
api_key = secrets.get_secret('POLYGON_API_KEY')
```

---

## Usage

### Basic Usage

```python
from src.security.secrets_manager import SecretsManager

# Initialize (singleton - same instance returned)
secrets = SecretsManager()

# Get a single secret
polygon_key = secrets.get_secret('POLYGON_API_KEY')

# Get secret with default value
wandb_key = secrets.get_secret('WANDB_API_KEY', default='not_configured')

# Get all secrets (for debugging - use with caution)
all_secrets = secrets.get_all_secrets()
```

### Advanced Usage

```python
from src.security.secrets_manager import SecretsManager

# Custom initialization
secrets = SecretsManager(
    service_name="RRRalgorithms",
    use_keychain=True,          # Use Keychain (default)
    fallback_to_env=True        # Fall back to env vars (default)
)

# Store a new secret
success = secrets.set_secret('NEW_API_KEY', 'secret_value_here')

# Delete a secret
success = secrets.delete_secret('OLD_API_KEY')

# Get configuration (non-secret values)
config = secrets.get_all_config()
```

### In Different Worktrees

Each worktree can use the same secrets:

```python
# In worktrees/trading-engine/src/main.py
import sys
from pathlib import Path

# Add main src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from security.secrets_manager import SecretsManager

secrets = SecretsManager()
coinbase_key = secrets.get_secret('COINBASE_API_KEY')
```

---

## Security Best Practices

### 1. Never Log Secrets

**Bad**:
```python
api_key = secrets.get_secret('API_KEY')
logger.info(f"Using API key: {api_key}")  # ❌ DON'T DO THIS
```

**Good**:
```python
api_key = secrets.get_secret('API_KEY')
logger.info(f"API key loaded: {'✓' if api_key else '✗'}")  # ✅ SAFE
```

### 2. Use Specific Permissions

Request only the permissions you need:

```python
# For read-only operations, use anon key
db_url = secrets.get_secret('SUPABASE_URL')
anon_key = secrets.get_secret('SUPABASE_ANON_KEY')

# For admin operations, use service key
service_key = secrets.get_secret('SUPABASE_SERVICE_KEY')
```

### 3. Handle Missing Secrets Gracefully

```python
api_key = secrets.get_secret('OPTIONAL_API_KEY')

if not api_key:
    logger.warning("Optional API key not configured - feature disabled")
    return

# Continue with feature...
```

### 4. Rotate Regularly

Set up automated rotation:

```python
from datetime import datetime, timedelta

def check_key_age(key_name: str, max_age_days: int = 90):
    """Check if key needs rotation"""
    # TODO: Track key creation date in metadata
    # For now, manual rotation required
    pass
```

---

## Keychain Management

### Using macOS Security Command

**View stored secrets**:
```bash
# List all RRRalgorithms keychain items
security find-generic-password -s "RRRalgorithms"
```

**View specific secret**:
```bash
# View POLYGON_API_KEY
security find-generic-password -s "RRRalgorithms" -a "POLYGON_API_KEY" -w
```

**Delete a secret**:
```bash
# Delete OLD_API_KEY
security delete-generic-password -s "RRRalgorithms" -a "OLD_API_KEY"
```

**Export all secrets** (DANGEROUS):
```bash
# Create encrypted backup
python -c "
from src.security.secrets_manager import SecretsManager
secrets = SecretsManager()
secrets.export_to_env_file('backup.env.encrypted')
"
```

### Keychain Access Control

Configure who can access secrets:

```bash
# Require authentication for access
security set-generic-password-partition-list \
  -s "RRRalgorithms" \
  -a "CRITICAL_API_KEY" \
  -S
```

---

## Environment-Specific Configuration

### Development

Use Keychain with fallback:

```python
secrets = SecretsManager(
    service_name="RRRalgorithms-Dev",
    use_keychain=True,
    fallback_to_env=True
)
```

### Staging

Use separate Keychain service:

```python
secrets = SecretsManager(
    service_name="RRRalgorithms-Staging",
    use_keychain=True,
    fallback_to_env=False  # Strict mode
)
```

### Production

Strict Keychain-only mode:

```python
secrets = SecretsManager(
    service_name="RRRalgorithms-Prod",
    use_keychain=True,
    fallback_to_env=False
)

# Verify all required secrets present
results = secrets.verify_secrets()
missing = [k for k, v in results.items() if not v]

if missing:
    raise RuntimeError(f"Missing required secrets: {missing}")
```

---

## CI/CD Integration

### GitHub Actions

Use encrypted secrets:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up environment
        env:
          POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
        run: |
          # Tests will use environment variables
          pytest tests/
```

### Docker

Use Docker secrets:

```dockerfile
# Dockerfile
FROM python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Secrets will be mounted at runtime
```

```bash
# Run with secrets
docker run \
  -e POLYGON_API_KEY="$POLYGON_API_KEY" \
  -e SUPABASE_URL="$SUPABASE_URL" \
  trading-engine:latest
```

---

## Troubleshooting

### Secret Not Found

```python
api_key = secrets.get_secret('API_KEY')
if not api_key:
    # Check Keychain
    # Check environment variables
    # Check spelling of key name
    # Verify migration completed
```

### Permission Denied

```bash
# Reset Keychain permissions
security unlock-keychain ~/Library/Keychains/login.keychain-db
```

### Migration Failed

```bash
# Re-run migration with verbose logging
python scripts/security/migrate_secrets.py --backup

# Check logs
tail -f logs/security/migration.log
```

### Keychain Not Available (Non-macOS)

Use environment variables only:

```python
secrets = SecretsManager(
    use_keychain=False,
    fallback_to_env=True
)
```

---

## API Reference

### SecretsManager

```python
class SecretsManager:
    def __init__(self,
                 service_name: str = "RRRalgorithms",
                 use_keychain: bool = True,
                 fallback_to_env: bool = True)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]
    def set_secret(self, key: str, value: str) -> bool
    def delete_secret(self, key: str) -> bool
    def get_all_secrets(self) -> Dict[str, Optional[str]]
    def get_all_config(self) -> Dict[str, Optional[str]]
    def verify_secrets(self) -> Dict[str, bool]
    def migrate_from_env_file(self, env_file_path: str) -> Dict[str, bool]
    def export_to_env_file(self, env_file_path: str, include_config: bool = True) -> bool
```

### KeychainManager

```python
class KeychainManager:
    def __init__(self, service_name: str = "RRRalgorithms")

    def store_secret(self, account: str, secret: str) -> bool
    def get_secret(self, account: str) -> Optional[str]
    def delete_secret(self, account: str) -> bool
    def list_accounts(self) -> list
    def store_multiple(self, secrets: Dict[str, str]) -> Dict[str, bool]
    def get_multiple(self, accounts: list) -> Dict[str, Optional[str]]
```

---

## Security Considerations

### Threat Model

**Protected Against**:
- ✅ Plaintext secrets in source code
- ✅ Secrets in git history
- ✅ Accidental secret logging
- ✅ Unauthorized access to secrets file

**Not Protected Against**:
- ⚠️ Memory dumps (secrets in RAM)
- ⚠️ Root/admin access to machine
- ⚠️ Keyloggers
- ⚠️ Social engineering

### Defense in Depth

1. **Keychain Encryption**: Secrets encrypted at rest
2. **Access Control**: OS-level authentication required
3. **Audit Logging**: All access logged
4. **Regular Rotation**: 90-day rotation policy
5. **Least Privilege**: Minimal permissions per key
6. **Monitoring**: Alert on unusual access patterns

---

## Future Enhancements

### Planned Features

1. **Automatic Rotation**: Scheduled key rotation
2. **Hardware Security Module (HSM)**: Enterprise-grade storage
3. **Multi-cloud Support**: AWS Secrets Manager, Azure Key Vault
4. **Secret Versioning**: Track and rollback secret changes
5. **Temporary Secrets**: Auto-expiring credentials
6. **Secret Sharing**: Secure team secret distribution

---

**Last Updated**: 2025-10-11
**Version**: 0.1.0
**Next Review**: 2025-11-11
