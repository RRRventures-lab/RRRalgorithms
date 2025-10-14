from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import os
import sys

#!/usr/bin/env python3
"""
Deployment Readiness Check
Comprehensive security and system readiness assessment before live deployment
"""


# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class DeploymentReadinessChecker:
    """Comprehensive deployment readiness checker"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warning = 0
        self.blockers = []

    def print_header(self, title: str):
        """Print section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

    def check_item(self, name: str, passed: bool, critical: bool = False,
                   details: str = None) -> bool:
        """
        Check a single item and print result

        Args:
            name: Check name
            passed: Whether check passed
            critical: If True, adds to blockers list when failed
            details: Additional details to show

        Returns:
            bool: Check result
        """
        if passed:
            status = "✅ PASS"
            self.checks_passed += 1
        elif critical:
            status = "❌ FAIL (BLOCKER)"
            self.checks_failed += 1
            self.blockers.append(name)
        else:
            status = "⚠️  WARNING"
            self.checks_warning += 1

        print(f"{status}: {name}")
        if details:
            print(f"         {details}")

        return passed

    def check_secrets_management(self) -> bool:
        """Check secrets management system"""
        self.print_header("1. Secrets Management")

        all_passed = True

        # Check if running on macOS
        is_macos = os.uname().sysname == "Darwin"
        all_passed &= self.check_item(
            "macOS Keychain available",
            is_macos,
            critical=True,
            details="Keychain required for secure secret storage"
        )

        if not is_macos:
            return False

        try:
            from security.secrets_manager import SecretsManager

            secrets = SecretsManager()

            # Check critical secrets
            critical_secrets = [
                "POLYGON_API_KEY",
                "SUPABASE_URL",
                "SUPABASE_DB_URL",
                "JWT_SECRET",
                "ENCRYPTION_KEY"
            ]

            for secret_name in critical_secrets:
                value = secrets.get_secret(secret_name)
                has_value = value is not None and len(value) > 0
                all_passed &= self.check_item(
                    f"{secret_name} configured",
                    has_value,
                    critical=True,
                    details="Required for system operation"
                )

            # Check trading secrets (warning only)
            trading_secrets = ["COINBASE_API_KEY", "COINBASE_API_SECRET"]
            for secret_name in trading_secrets:
                value = secrets.get_secret(secret_name)
                has_value = value is not None and len(value) > 0
                self.check_item(
                    f"{secret_name} configured",
                    has_value,
                    critical=False,
                    details="Required for live trading"
                )

        except Exception as e:
            all_passed = False
            self.check_item(
                "Secrets management system",
                False,
                critical=True,
                details=f"Error: {e}"
            )

        return all_passed

    def check_api_keys_rotated(self) -> bool:
        """Check if exposed API keys have been rotated"""
        self.print_header("2. API Key Rotation")

        # This is a manual check - we can't automatically verify rotation
        print("\n⚠️  MANUAL VERIFICATION REQUIRED")
        print("\nThe following API keys were EXPOSED in plaintext:")
        print("  - Polygon.io API key")
        print("  - Perplexity AI API key")
        print("  - Anthropic Claude API key")
        print("  - Supabase credentials")
        print("  - Coinbase API keys and private key")
        print("  - GitHub Personal Access Token")
        print("  - JWT Secret")
        print("\nHave you rotated ALL of these keys?")
        print("See: docs/security/API_KEY_ROTATION_GUIDE.md")

        # For automated check, we assume keys need rotation
        self.check_item(
            "API keys rotated",
            False,
            critical=True,
            details="MUST rotate all exposed keys before deployment"
        )

        return False

    def check_audit_logging(self) -> bool:
        """Check audit logging system"""
        self.print_header("3. Audit Logging")

        all_passed = True

        # Check if audit logging module exists
        try:
            monitoring_path = project_root / "worktrees" / "monitoring" / "src"
            sys.path.insert(0, str(monitoring_path))
            from logging.audit_logger import get_audit_logger

            all_passed &= self.check_item(
                "Audit logger module available",
                True,
                critical=False
            )

            # Check if audit logs directory exists
            audit_log_dir = project_root / "logs" / "audit"
            audit_log_dir.mkdir(parents=True, exist_ok=True)
            all_passed &= self.check_item(
                "Audit log directory writable",
                audit_log_dir.exists(),
                critical=False
            )

        except Exception as e:
            all_passed = False
            self.check_item(
                "Audit logging system",
                False,
                critical=False,
                details=f"Error: {e}"
            )

        # Check database migration
        migration_file = project_root / "config" / "database" / "migrations" / "004_create_audit_logs.sql"
        all_passed &= self.check_item(
            "Audit logs SQL migration exists",
            migration_file.exists(),
            critical=False,
            details="Run migration to create audit_logs table"
        )

        return all_passed

    def check_environment_configuration(self) -> bool:
        """Check environment configuration"""
        self.print_header("4. Environment Configuration")

        all_passed = True

        # Check .env.example exists
        env_example = project_root / "config" / "api-keys" / ".env.example"
        all_passed &= self.check_item(
            ".env.example file exists",
            env_example.exists(),
            critical=False
        )

        # Check .env is in .gitignore
        gitignore = project_root / ".gitignore"
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                content = f.read()
                has_env = '.env' in content
                all_passed &= self.check_item(
                    ".env in .gitignore",
                    has_env,
                    critical=True,
                    details="Prevents accidental secret commits"
                )
        else:
            all_passed = False
            self.check_item(
                ".gitignore exists",
                False,
                critical=True
            )

        # Check ENVIRONMENT variable
        environment = os.environ.get('ENVIRONMENT', 'development')
        is_production = environment == 'production'
        self.check_item(
            f"Environment: {environment}",
            True,
            critical=False,
            details="Set ENVIRONMENT=production for live trading"
        )

        # Check PAPER_TRADING flag
        paper_trading = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'
        self.check_item(
            f"Paper trading mode: {paper_trading}",
            True,
            critical=False,
            details="Disable only after successful paper trading"
        )

        return all_passed

    def check_security_documentation(self) -> bool:
        """Check security documentation"""
        self.print_header("5. Security Documentation")

        all_passed = True

        docs = [
            ("SECURITY.md", "Security policy and vulnerability reporting"),
            ("docs/security/API_KEY_ROTATION_GUIDE.md", "API key rotation instructions"),
            ("docs/security/SECRETS_MANAGEMENT.md", "Secrets management guide")
        ]

        for doc_path, description in docs:
            full_path = project_root / doc_path
            all_passed &= self.check_item(
                f"{doc_path} exists",
                full_path.exists(),
                critical=False,
                details=description
            )

        return all_passed

    def check_database_security(self) -> bool:
        """Check database security"""
        self.print_header("6. Database Security")

        all_passed = True

        # Check if database migrations exist
        migrations_dir = project_root / "config" / "database" / "migrations"
        has_migrations = migrations_dir.exists() and any(migrations_dir.glob("*.sql"))

        all_passed &= self.check_item(
            "Database migrations exist",
            has_migrations,
            critical=False
        )

        # Check for RLS policies
        if has_migrations:
            migration_files = list(migrations_dir.glob("*.sql"))
            has_rls = False
            for mfile in migration_files:
                try:
                    with open(mfile, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'ROW LEVEL SECURITY' in content or 'ENABLE ROW LEVEL SECURITY' in content:
                            has_rls = True
                            break
                except Exception:
                    continue  # Skip files that can't be read

            all_passed &= self.check_item(
                "Row Level Security (RLS) policies defined",
                has_rls,
                critical=False,
                details="Ensures data isolation and security"
            )

        return all_passed

    def check_code_security(self) -> bool:
        """Check code security"""
        self.print_header("7. Code Security")

        all_passed = True

        # Check for common security issues in code
        src_dirs = [
            project_root / "src",
            project_root / "worktrees"
        ]

        # Simple check for hardcoded secrets
        suspicious_patterns = [
            'sk-ant-api',  # Anthropic keys
            'pplx-',       # Perplexity keys
            'ghp_',        # GitHub tokens
        ]

        issues_found = []
        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        for pattern in suspicious_patterns:
                            if pattern in content:
                                issues_found.append(f"{py_file.name}: {pattern}")
                except:
                    pass

        all_passed &= self.check_item(
            "No hardcoded secrets in code",
            len(issues_found) == 0,
            critical=True,
            details=f"Found {len(issues_found)} potential issues" if issues_found else None
        )

        if issues_found:
            print("\n  Potential hardcoded secrets found:")
            for issue in issues_found[:5]:
                print(f"    - {issue}")

        return all_passed

    def check_monitoring_alerting(self) -> bool:
        """Check monitoring and alerting setup"""
        self.print_header("8. Monitoring & Alerting")

        all_passed = True

        # Check monitoring worktree exists
        monitoring_dir = project_root / "worktrees" / "monitoring"
        all_passed &= self.check_item(
            "Monitoring worktree exists",
            monitoring_dir.exists(),
            critical=False
        )

        # Check if Sentry is configured
        from security.secrets_manager import SecretsManager
        secrets = SecretsManager()
        sentry_dsn = secrets.get_secret('SENTRY_DSN')

        self.check_item(
            "Sentry error tracking configured",
            sentry_dsn is not None and len(sentry_dsn) > 0,
            critical=False,
            details="Recommended for production error tracking"
        )

        # Check if Slack webhook is configured
        slack_webhook = secrets.get_secret('SLACK_WEBHOOK_URL')
        self.check_item(
            "Slack alerts configured",
            slack_webhook is not None and 'hooks.slack.com' in slack_webhook,
            critical=False,
            details="Recommended for critical alerts"
        )

        return all_passed

    def check_risk_management(self) -> bool:
        """Check risk management configuration"""
        self.print_header("9. Risk Management")

        all_passed = True

        # Check risk worktree exists
        risk_dir = project_root / "worktrees" / "risk-management"
        all_passed &= self.check_item(
            "Risk management worktree exists",
            risk_dir.exists(),
            critical=True,
            details="Required for safe trading"
        )

        # Check risk parameters
        max_position = os.environ.get('MAX_POSITION_SIZE')
        max_loss = os.environ.get('MAX_DAILY_LOSS')

        all_passed &= self.check_item(
            "MAX_POSITION_SIZE configured",
            max_position is not None,
            critical=True,
            details=f"Current: {max_position}"
        )

        all_passed &= self.check_item(
            "MAX_DAILY_LOSS configured",
            max_loss is not None,
            critical=True,
            details=f"Current: {max_loss}"
        )

        return all_passed

    def generate_report(self):
        """Generate final report"""
        self.print_header("DEPLOYMENT READINESS REPORT")

        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: RRRalgorithms")
        print(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")

        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)

        total = self.checks_passed + self.checks_failed + self.checks_warning
        print(f"✅ Passed:   {self.checks_passed:3d} / {total}")
        print(f"❌ Failed:   {self.checks_failed:3d} / {total}")
        print(f"⚠️  Warnings: {self.checks_warning:3d} / {total}")

        print("\n" + "-"*80)
        print("VERDICT")
        print("-"*80)

        if self.checks_failed == 0:
            print("\n✅ READY FOR DEPLOYMENT")
            print("\nAll critical checks passed!")
            if self.checks_warning > 0:
                print(f"\nNote: {self.checks_warning} warnings - review before deployment")
        else:
            print("\n❌ NOT READY FOR DEPLOYMENT")
            print(f"\n{self.checks_failed} critical checks failed:")
            for blocker in self.blockers:
                print(f"  ❌ {blocker}")

            print("\n⚠️  DO NOT DEPLOY until all blockers are resolved!")

        print("\n" + "-"*80)
        print("NEXT STEPS")
        print("-"*80)

        if self.checks_failed > 0:
            print("\n1. Fix all BLOCKER issues listed above")
            print("2. Run this check again: python scripts/security/deployment_readiness.py")
            print("3. Review all warnings")
            print("4. Complete manual verification steps")
            print("5. Run paper trading for 30+ days")
        else:
            print("\n1. Review all warnings")
            print("2. Complete manual verification steps")
            print("3. Test in staging environment")
            print("4. Run paper trading for 30+ days")
            print("5. Get legal/compliance approval")
            print("6. Deploy to production")

        print("\n" + "="*80)

        return self.checks_failed == 0


def main():
    """Run deployment readiness checks"""
    checker = DeploymentReadinessChecker()

    print("="*80)
    print("  RRRalgorithms Deployment Readiness Check")
    print("="*80)
    print("\nThis tool verifies that all security and system requirements")
    print("are met before live trading deployment.")

    # Run all checks
    checker.check_secrets_management()
    checker.check_api_keys_rotated()
    checker.check_audit_logging()
    checker.check_environment_configuration()
    checker.check_security_documentation()
    checker.check_database_security()
    checker.check_code_security()
    checker.check_monitoring_alerting()
    checker.check_risk_management()

    # Generate report
    ready = checker.generate_report()

    # Exit with appropriate code
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
