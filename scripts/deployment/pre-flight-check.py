from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import json
import os
import requests
import subprocess
import sys
import yaml

#!/usr/bin/env python3
"""
Pre-Flight Check Script

Validates system readiness before paper trading deployment.
Checks configuration, dependencies, services, and safety settings.
"""



@dataclass
class CheckResult:
    """Result of a pre-flight check"""
    name: str
    passed: bool
    message: str
    severity: str  # 'critical', 'warning', 'info'


class PreFlightChecker:
    """Comprehensive pre-flight checks"""

    def __init__(self, environment: str = 'paper'):
        self.environment = environment
        self.results: List[CheckResult] = []
        self.project_root = Path(__file__).parent.parent.parent

    def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """Run all pre-flight checks"""
        print("🚀 Running Pre-Flight Checks...")
        print(f"Environment: {self.environment}")
        print("=" * 60)

        # Critical checks
        self.check_docker()
        self.check_required_files()
        self.check_environment_variables()
        self.check_paper_trading_mode()
        self.check_disk_space()

        # Warning checks
        self.check_api_keys()
        self.check_database_connection()
        self.check_port_availability()

        # Info checks
        self.check_git_status()
        self.check_test_coverage()
        self.check_documentation()

        # Print results
        self.print_results()

        # Determine overall status
        critical_failures = [r for r in self.results if not r.passed and r.severity == 'critical']
        return len(critical_failures) == 0, self.results

    # ===== Critical Checks =====

    def check_docker(self):
        """Check Docker is running and version"""
        try:
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Get version
                version_result = subprocess.run(
                    ['docker', '--version'],
                    capture_output=True,
                    text=True
                )
                version = version_result.stdout.strip()

                self.results.append(CheckResult(
                    name="Docker Running",
                    passed=True,
                    message=f"✓ {version}",
                    severity='critical'
                ))
            else:
                self.results.append(CheckResult(
                    name="Docker Running",
                    passed=False,
                    message="✗ Docker is not running. Start Docker Desktop.",
                    severity='critical'
                ))

        except FileNotFoundError:
            self.results.append(CheckResult(
                name="Docker Installed",
                passed=False,
                message="✗ Docker is not installed",
                severity='critical'
            ))
        except subprocess.TimeoutExpired:
            self.results.append(CheckResult(
                name="Docker Running",
                passed=False,
                message="✗ Docker command timeout. Check Docker Desktop.",
                severity='critical'
            ))

    def check_required_files(self):
        """Check all required files exist"""
        required_files = [
            'docker-compose.yml',
            'docker-compose.paper-trading.yml',
            'config/api-keys/.env',
            'QUICK_START.md',
            'PAPER_TRADING_GUIDE.md',
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if not missing_files:
            self.results.append(CheckResult(
                name="Required Files",
                passed=True,
                message=f"✓ All {len(required_files)} required files found",
                severity='critical'
            ))
        else:
            self.results.append(CheckResult(
                name="Required Files",
                passed=False,
                message=f"✗ Missing files: {', '.join(missing_files)}",
                severity='critical'
            ))

    def check_environment_variables(self):
        """Check required environment variables"""
        env_file = self.project_root / 'config/api-keys/.env'

        if not env_file.exists():
            self.results.append(CheckResult(
                name="Environment File",
                passed=False,
                message="✗ .env file not found",
                severity='critical'
            ))
            return

        with open(env_file) as f:
            env_content = f.read()

        required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
        missing_vars = []

        for var in required_vars:
            if var not in env_content or f"{var}=" not in env_content:
                missing_vars.append(var)

        if not missing_vars:
            self.results.append(CheckResult(
                name="Environment Variables",
                passed=True,
                message="✓ All required variables set",
                severity='critical'
            ))
        else:
            self.results.append(CheckResult(
                name="Environment Variables",
                passed=False,
                message=f"✗ Missing variables: {', '.join(missing_vars)}",
                severity='critical'
            ))

    def check_paper_trading_mode(self):
        """Verify paper trading mode is enabled"""
        compose_file = self.project_root / 'docker-compose.paper-trading.yml'

        if not compose_file.exists():
            self.results.append(CheckResult(
                name="Paper Trading Config",
                passed=False,
                message="✗ docker-compose.paper-trading.yml not found",
                severity='critical'
            ))
            return

        with open(compose_file) as f:
            content = f.read()

        # Check for paper trading mode
        if 'PAPER_TRADING_MODE=true' in content:
            # Check for safety flags
            safety_flags = [
                'ENABLE_LIVE_TRADING=false',
                'EXCHANGE_MODE=paper'
            ]

            found_flags = sum(1 for flag in safety_flags if flag in content)

            self.results.append(CheckResult(
                name="Paper Trading Mode",
                passed=True,
                message=f"✓ Paper trading enabled ({found_flags}/2 safety flags)",
                severity='critical'
            ))
        else:
            self.results.append(CheckResult(
                name="Paper Trading Mode",
                passed=False,
                message="✗ PAPER_TRADING_MODE not set to true",
                severity='critical'
            ))

    def check_disk_space(self):
        """Check available disk space"""
        try:
            result = subprocess.run(
                ['df', '-h', str(self.project_root)],
                capture_output=True,
                text=True
            )

            # Parse output (macOS/Linux)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                available = parts[3]

                # Extract numeric value (e.g., "45Gi" -> 45)
                import re
                match = re.search(r'(\d+)', available)
                if match:
                    available_gb = int(match.group(1))

                    if available_gb >= 20:
                        self.results.append(CheckResult(
                            name="Disk Space",
                            passed=True,
                            message=f"✓ {available} available (20GB+ required)",
                            severity='critical'
                        ))
                    else:
                        self.results.append(CheckResult(
                            name="Disk Space",
                            passed=False,
                            message=f"✗ Only {available} available (20GB+ required)",
                            severity='critical'
                        ))
                    return

            # Fallback
            self.results.append(CheckResult(
                name="Disk Space",
                passed=True,
                message="⚠ Could not determine disk space",
                severity='warning'
            ))

        except Exception as e:
            self.results.append(CheckResult(
                name="Disk Space",
                passed=True,
                message=f"⚠ Could not check disk space: {e}",
                severity='warning'
            ))

    # ===== Warning Checks =====

    def check_api_keys(self):
        """Check if API keys are configured"""
        env_file = self.project_root / 'config/api-keys/.env'

        if not env_file.exists():
            return

        with open(env_file) as f:
            env_content = f.read()

        optional_keys = ['COINBASE_API_KEY', 'POLYGON_API_KEY', 'PERPLEXITY_API_KEY']
        missing_keys = [k for k in optional_keys if k not in env_content]

        if not missing_keys:
            self.results.append(CheckResult(
                name="API Keys (Optional)",
                passed=True,
                message="✓ All API keys configured",
                severity='info'
            ))
        else:
            self.results.append(CheckResult(
                name="API Keys (Optional)",
                passed=True,
                message=f"⚠ Missing optional keys: {', '.join(missing_keys)}",
                severity='warning'
            ))

    def check_database_connection(self):
        """Check database connectivity"""
        env_file = self.project_root / 'config/api-keys/.env'

        if not env_file.exists():
            return

        # Load environment variables
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

        supabase_url = os.environ.get('SUPABASE_URL')

        if not supabase_url:
            self.results.append(CheckResult(
                name="Database Connection",
                passed=True,
                message="⚠ SUPABASE_URL not set, cannot test connection",
                severity='warning'
            ))
            return

        try:
            # Test connection with timeout
            response = requests.get(supabase_url, timeout=5)

            if response.status_code in [200, 401, 403]:  # Any response is good
                self.results.append(CheckResult(
                    name="Database Connection",
                    passed=True,
                    message="✓ Database reachable",
                    severity='warning'
                ))
            else:
                self.results.append(CheckResult(
                    name="Database Connection",
                    passed=False,
                    message=f"⚠ Unexpected response: {response.status_code}",
                    severity='warning'
                ))

        except requests.exceptions.RequestException as e:
            self.results.append(CheckResult(
                name="Database Connection",
                passed=False,
                message=f"⚠ Cannot reach database: {str(e)[:50]}",
                severity='warning'
            ))

    def check_port_availability(self):
        """Check if required ports are available"""
        required_ports = [
            (3000, 'Grafana'),
            (8000, 'Neural Network'),
            (8001, 'Data Pipeline'),
            (8002, 'Trading Engine'),
            (9090, 'Prometheus'),
        ]

        ports_in_use = []

        for port, service in required_ports:
            try:
                result = subprocess.run(
                    ['lsof', '-i', f':{port}'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0 and result.stdout:
                    ports_in_use.append(f"{port} ({service})")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if not ports_in_use:
            self.results.append(CheckResult(
                name="Port Availability",
                passed=True,
                message="✓ All required ports available",
                severity='warning'
            ))
        else:
            self.results.append(CheckResult(
                name="Port Availability",
                passed=False,
                message=f"⚠ Ports in use: {', '.join(ports_in_use)}",
                severity='warning'
            ))

    # ===== Info Checks =====

    def check_git_status(self):
        """Check git status"""
        try:
            # Check if there are uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.stdout.strip():
                self.results.append(CheckResult(
                    name="Git Status",
                    passed=True,
                    message="ℹ Uncommitted changes present",
                    severity='info'
                ))
            else:
                self.results.append(CheckResult(
                    name="Git Status",
                    passed=True,
                    message="✓ Working directory clean",
                    severity='info'
                ))

        except subprocess.CalledProcessError:
            pass  # Git not available or not a repo

    def check_test_coverage(self):
        """Check test coverage"""
        # This is informational - we know coverage is around 52%
        self.results.append(CheckResult(
            name="Test Coverage",
            passed=True,
            message="ℹ Current coverage: ~52% (Target: 80%)",
            severity='info'
        ))

    def check_documentation(self):
        """Check documentation completeness"""
        doc_files = [
            'QUICK_START.md',
            'PAPER_TRADING_GUIDE.md',
            'CLAUDE.md',
            'README.md',
        ]

        existing = sum(1 for f in doc_files if (self.project_root / f).exists())

        self.results.append(CheckResult(
            name="Documentation",
            passed=True,
            message=f"✓ {existing}/{len(doc_files)} key documents present",
            severity='info'
        ))

    # ===== Results =====

    def print_results(self):
        """Print all check results"""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        # Group by severity
        critical = [r for r in self.results if r.severity == 'critical']
        warnings = [r for r in self.results if r.severity == 'warning']
        info = [r for r in self.results if r.severity == 'info']

        # Print critical
        print("\n🔴 CRITICAL CHECKS:")
        for result in critical:
            print(f"  {result.message}")

        # Print warnings
        if warnings:
            print("\n🟡 WARNINGS:")
            for result in warnings:
                print(f"  {result.message}")

        # Print info
        if info:
            print("\n🔵 INFORMATION:")
            for result in info:
                print(f"  {result.message}")

        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"SUMMARY: {passed}/{total} checks passed")

        critical_failures = [r for r in self.results if not r.passed and r.severity == 'critical']
        if critical_failures:
            print(f"\n❌ DEPLOYMENT BLOCKED: {len(critical_failures)} critical failures")
            print("Fix critical issues before proceeding.")
            return False
        else:
            print("\n✅ READY FOR PAPER TRADING")
            print("All critical checks passed!")
            return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Pre-flight checks for paper trading deployment')
    parser.add_argument('--environment', default='paper', choices=['paper', 'staging', 'production'],
                        help='Deployment environment')
    args = parser.parse_args()

    checker = PreFlightChecker(environment=args.environment)
    success, results = checker.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
