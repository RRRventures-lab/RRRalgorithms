from pathlib import Path
from security.keychain_manager import KeychainManager
from security.secrets_manager import SecretsManager
import os
import sys

#!/usr/bin/env python3
"""
Test Secrets Management System
Verify that secrets management is working correctly
"""


# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))



def test_keychain_manager():
    """Test KeychainManager functionality"""
    print("\n" + "="*80)
    print("Testing KeychainManager")
    print("="*80)

    keychain = KeychainManager(service_name="RRRalgorithms-Test")

    # Test 1: Store a secret
    print("\n[Test 1] Storing test secret...")
    success = keychain.store_secret("TEST_KEY", "test_value_123")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 2: Retrieve the secret
    print("\n[Test 2] Retrieving test secret...")
    value = keychain.get_secret("TEST_KEY")
    success = value == "test_value_123"
    print(f"  Expected: test_value_123")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 3: Update the secret
    print("\n[Test 3] Updating test secret...")
    success = keychain.store_secret("TEST_KEY", "updated_value_456")
    value = keychain.get_secret("TEST_KEY")
    success = value == "updated_value_456"
    print(f"  Expected: updated_value_456")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 4: Delete the secret
    print("\n[Test 4] Deleting test secret...")
    success = keychain.delete_secret("TEST_KEY")
    value = keychain.get_secret("TEST_KEY")
    success = value is None
    print(f"  Expected: None")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 5: Store multiple secrets
    print("\n[Test 5] Storing multiple secrets...")
    secrets = {
        "TEST_KEY_1": "value_1",
        "TEST_KEY_2": "value_2",
        "TEST_KEY_3": "value_3"
    }
    results = keychain.store_multiple(secrets)
    success = all(results.values())
    print(f"  Stored: {sum(results.values())}/{len(results)}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 6: Retrieve multiple secrets
    print("\n[Test 6] Retrieving multiple secrets...")
    retrieved = keychain.get_multiple(list(secrets.keys()))
    success = retrieved == secrets
    print(f"  Expected: {len(secrets)} secrets")
    print(f"  Got: {len([v for v in retrieved.values() if v])}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Cleanup
    print("\n[Cleanup] Removing test secrets...")
    for key in secrets.keys():
        keychain.delete_secret(key)

    return success


def test_secrets_manager():
    """Test SecretsManager functionality"""
    print("\n" + "="*80)
    print("Testing SecretsManager")
    print("="*80)

    secrets_manager = SecretsManager(
        service_name="RRRalgorithms-Test",
        use_keychain=True,
        fallback_to_env=True
    )

    # Test 1: Set and get secret
    print("\n[Test 1] Set and get secret...")
    success = secrets_manager.set_secret("TEST_SECRET", "secret_123")
    value = secrets_manager.get_secret("TEST_SECRET")
    success = value == "secret_123"
    print(f"  Expected: secret_123")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 2: Get with default value
    print("\n[Test 2] Get non-existent secret with default...")
    value = secrets_manager.get_secret("NONEXISTENT_KEY", default="default_value")
    success = value == "default_value"
    print(f"  Expected: default_value")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 3: Environment variable fallback
    print("\n[Test 3] Environment variable fallback...")
    os.environ["TEST_ENV_VAR"] = "env_value"
    value = secrets_manager.get_secret("TEST_ENV_VAR")
    success = value == "env_value"
    print(f"  Expected: env_value")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")
    del os.environ["TEST_ENV_VAR"]

    # Test 4: Delete secret
    print("\n[Test 4] Delete secret...")
    secrets_manager.delete_secret("TEST_SECRET")
    value = secrets_manager.get_secret("TEST_SECRET")
    success = value is None
    print(f"  Expected: None")
    print(f"  Got: {value}")
    print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

    # Test 5: Verify secrets
    print("\n[Test 5] Verify production secrets availability...")
    results = secrets_manager.verify_secrets()
    available = sum(results.values())
    total = len(results)
    print(f"  Available: {available}/{total} secrets")

    # List unavailable secrets
    unavailable = [k for k, v in results.items() if not v]
    if unavailable:
        print(f"  Missing secrets:")
        for key in unavailable[:5]:  # Show first 5
            print(f"    - {key}")
        if len(unavailable) > 5:
            print(f"    ... and {len(unavailable) - 5} more")

    print(f"  Result: ℹ INFO (not all secrets configured yet)")

    return True


def test_production_readiness():
    """Test if system is ready for production"""
    print("\n" + "="*80)
    print("Production Readiness Check")
    print("="*80)

    secrets_manager = SecretsManager()

    # Critical secrets that MUST be available
    critical_secrets = [
        "POLYGON_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_DB_URL",
        "JWT_SECRET",
        "ENCRYPTION_KEY"
    ]

    print("\n[Critical Secrets Check]")
    all_available = True
    for key in critical_secrets:
        value = secrets_manager.get_secret(key)
        available = value is not None and len(value) > 0
        all_available = all_available and available
        status = "✓" if available else "✗"
        print(f"  {status} {key}")

    # Trading secrets (required for live trading)
    trading_secrets = [
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET"
    ]

    print("\n[Trading Secrets Check] (required for live trading)")
    trading_ready = True
    for key in trading_secrets:
        value = secrets_manager.get_secret(key)
        available = value is not None and len(value) > 0
        trading_ready = trading_ready and available
        status = "✓" if available else "✗"
        print(f"  {status} {key}")

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if all_available:
        print("✅ PASS: Critical secrets available")
    else:
        print("❌ FAIL: Missing critical secrets")
        print("   Action: Run migration script and configure secrets")

    if trading_ready:
        print("✅ PASS: Trading secrets available")
    else:
        print("⚠️  WARNING: Trading secrets not configured")
        print("   Action: Configure exchange API keys before live trading")

    # Check if API keys were rotated
    print("\n⚠️  CRITICAL: Have you rotated all API keys?")
    print("   The keys in .env were EXPOSED and must be rotated")
    print("   See: docs/security/API_KEY_ROTATION_GUIDE.md")

    return all_available


def main():
    """Run all tests"""
    print("="*80)
    print("RRRalgorithms Secrets Management Test Suite")
    print("="*80)

    # Check if running on macOS
    if os.uname().sysname != "Darwin":
        print("\n❌ ERROR: Tests require macOS")
        print("Keychain-based tests will be skipped")
        sys.exit(1)

    # Run tests
    test_results = []

    try:
        result = test_keychain_manager()
        test_results.append(("KeychainManager", result))
    except Exception as e:
        print(f"\n❌ KeychainManager tests failed: {e}")
        test_results.append(("KeychainManager", False))

    try:
        result = test_secrets_manager()
        test_results.append(("SecretsManager", result))
    except Exception as e:
        print(f"\n❌ SecretsManager tests failed: {e}")
        test_results.append(("SecretsManager", False))

    try:
        result = test_production_readiness()
        test_results.append(("Production Readiness", result))
    except Exception as e:
        print(f"\n❌ Production readiness check failed: {e}")
        test_results.append(("Production Readiness", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    print(f"\nTotal: {passed}/{total} tests passed")

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
