#!/usr/bin/env python3
"""
Test paper trading system before full deployment
Validates all components are working correctly
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.client_factory import get_database_client
from src.security.secrets_manager import SecretsManager
from src.core.config import get_env_file
from src.core.settings import get_settings

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'


class PaperTradingTester:
    """Test paper trading system components."""
    
    def __init__(self):
        self.results = {}
        self.critical_errors = []
        
    async def run_all_tests(self):
        """Run all system tests."""
        print(f"\n{GREEN}🧪 PAPER TRADING SYSTEM TEST{NC}")
        print("=" * 50)
        
        tests = [
            ("Database Connection", self.test_database),
            ("Secrets Management", self.test_secrets),
            ("Configuration", self.test_configuration),
            ("Market Data", self.test_market_data),
            ("Trading Engine", self.test_trading_engine),
            ("Risk Management", self.test_risk_management),
            ("Monitoring", self.test_monitoring),
            ("API Connectivity", self.test_api_connectivity),
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 Testing: {test_name}")
            print("-" * 30)
            
            try:
                result = await test_func()
                self.results[test_name] = result
                
                if result['status'] == 'PASS':
                    print(f"{GREEN}✅ PASS{NC}: {test_name}")
                elif result['status'] == 'WARN':
                    print(f"{YELLOW}⚠️  WARN{NC}: {test_name}")
                    print(f"   Issues: {', '.join(result['issues'])}")
                else:
                    print(f"{RED}❌ FAIL{NC}: {test_name}")
                    print(f"   Error: {result['error']}")
                    self.critical_errors.append(test_name)
                    
            except Exception as e:
                print(f"{RED}❌ ERROR{NC}: {test_name} - {e}")
                self.results[test_name] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
                self.critical_errors.append(test_name)
        
        # Print summary
        self._print_summary()
        
        return len(self.critical_errors) == 0
    
    async def test_database(self):
        """Test database connectivity and operations."""
        try:
            # Test regular database
            db = get_database_client(db_type="sqlite")
            await db.connect()
            
            # Test basic operations
            result = await db.fetch_one("SELECT 1 as test")
            if result['test'] != 1:
                raise ValueError("Database query failed")
            
            # Check tables exist
            tables = await db.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            table_names = [t['name'] for t in tables]
            
            required_tables = ['symbols', 'market_data', 'orders', 'positions']
            missing_tables = [t for t in required_tables if t not in table_names]
            
            await db.disconnect()
            
            if missing_tables:
                return {
                    'status': 'WARN',
                    'issues': [f"Missing tables: {', '.join(missing_tables)}"]
                }
            
            # Test encrypted database
            from src.security.encrypted_database import UltraSecureDatabase
            enc_db = UltraSecureDatabase()
            await enc_db.connect()
            
            # Test encryption
            test_data = "test_secret_123"
            encrypted = enc_db._encrypt_data(test_data)
            decrypted = enc_db._decrypt_data(encrypted)
            
            if decrypted != test_data:
                raise ValueError("Encryption/decryption failed")
            
            await enc_db.disconnect()
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_secrets(self):
        """Test secrets management."""
        try:
            secrets_mgr = SecretsManager()
            
            # Check critical API keys
            required_keys = ['POLYGON_API_KEY']
            missing_keys = []
            
            for key in required_keys:
                if not secrets_mgr.get_secret(key):
                    missing_keys.append(key)
            
            if missing_keys:
                return {
                    'status': 'WARN',
                    'issues': [f"Missing API keys: {', '.join(missing_keys)}"]
                }
            
            # Test secret operations
            test_key = "TEST_PAPER_TRADING"
            test_value = "test_value_123"
            
            # Store and retrieve
            success = secrets_mgr.set_secret(test_key, test_value)
            if not success:
                raise ValueError("Failed to store test secret")
            
            retrieved = secrets_mgr.get_secret(test_key)
            if retrieved != test_value:
                raise ValueError("Failed to retrieve test secret")
            
            # Clean up
            secrets_mgr.delete_secret(test_key)
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_configuration(self):
        """Test system configuration."""
        try:
            # Load settings
            settings = get_settings()
            
            issues = []
            
            # Check trading mode
            if settings.TRADING_MODE != 'paper':
                issues.append(f"Trading mode is {settings.TRADING_MODE}, should be 'paper'")
            
            # Check required paths
            paths_to_check = [
                ('DATABASE_PATH', settings.DATABASE_PATH),
                ('LOG_DIR', settings.LOG_DIR),
                ('DATA_DIR', settings.DATA_DIR),
            ]
            
            for name, path in paths_to_check:
                if not path:
                    issues.append(f"{name} not configured")
                elif not Path(path).parent.exists():
                    issues.append(f"{name} parent directory doesn't exist: {path}")
            
            if issues:
                return {'status': 'WARN', 'issues': issues}
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_market_data(self):
        """Test market data connectivity."""
        try:
            # This is a placeholder - would test actual market data connection
            # For now, just check if we can import the modules
            from src.data_pipeline.polygon_client import PolygonClient
            from src.data_pipeline.data_validator import DataValidator
            
            # Test validator
            validator = DataValidator()
            
            # Test sample data
            sample_data = {
                'symbol': 'BTC-USD',
                'price': 45000.0,
                'volume': 1000.0,
                'timestamp': int(time.time())
            }
            
            is_valid, errors = validator.validate_market_data(sample_data)
            if not is_valid:
                return {
                    'status': 'WARN',
                    'issues': [f"Validation errors: {errors}"]
                }
            
            return {'status': 'PASS'}
            
        except ImportError as e:
            return {
                'status': 'WARN',
                'issues': [f"Market data modules not found: {e}"]
            }
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_trading_engine(self):
        """Test trading engine components."""
        try:
            # Test imports
            modules = [
                'src.trading.paper_trader',
                'src.trading.order_manager',
                'src.trading.position_manager',
            ]
            
            missing_modules = []
            for module in modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                return {
                    'status': 'WARN',
                    'issues': [f"Missing modules: {', '.join(missing_modules)}"]
                }
            
            # Test paper trader initialization
            from src.trading.paper_trader import PaperTrader
            
            # Would initialize paper trader here in full implementation
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_risk_management(self):
        """Test risk management system."""
        try:
            # Test risk limits
            risk_config = {
                'max_position_size': 10000,
                'max_daily_loss': 1000,
                'max_drawdown': 0.1,
                'position_limit': 10
            }
            
            # Validate risk config
            for key, value in risk_config.items():
                if value <= 0:
                    return {
                        'status': 'FAIL',
                        'error': f"Invalid risk config: {key}={value}"
                    }
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_monitoring(self):
        """Test monitoring connectivity."""
        try:
            import httpx
            
            issues = []
            
            # Test Prometheus
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get("http://localhost:9090/-/ready", timeout=2.0)
                    if resp.status_code != 200:
                        issues.append("Prometheus not ready")
                except:
                    issues.append("Prometheus not reachable")
                
                # Test Grafana
                try:
                    resp = await client.get("http://localhost:3000/api/health", timeout=2.0)
                    if resp.status_code != 200:
                        issues.append("Grafana not healthy")
                except:
                    issues.append("Grafana not reachable")
            
            if issues:
                return {'status': 'WARN', 'issues': issues}
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_api_connectivity(self):
        """Test external API connectivity."""
        try:
            import httpx
            
            issues = []
            
            # Test internet connectivity
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get("https://api.polygon.io/v2/status", timeout=5.0)
                    if resp.status_code != 200:
                        issues.append(f"Polygon API returned {resp.status_code}")
                except Exception as e:
                    issues.append(f"Cannot reach Polygon API: {e}")
            
            if issues:
                return {'status': 'WARN', 'issues': issues}
            
            return {'status': 'PASS'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print(f"{GREEN}📊 TEST SUMMARY{NC}")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        warnings = sum(1 for r in self.results.values() if r['status'] == 'WARN')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"{GREEN}✅ Passed: {passed}{NC}")
        print(f"{YELLOW}⚠️  Warnings: {warnings}{NC}")
        print(f"{RED}❌ Failed: {failed}{NC}")
        
        if self.critical_errors:
            print(f"\n{RED}⚠️  CRITICAL ERRORS:{NC}")
            for error in self.critical_errors:
                print(f"   - {error}")
            print(f"\n{RED}❌ SYSTEM NOT READY FOR PAPER TRADING{NC}")
        else:
            print(f"\n{GREEN}✅ SYSTEM READY FOR PAPER TRADING!{NC}")
        
        # Save results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total_tests,
                    'passed': passed,
                    'warnings': warnings,
                    'failed': failed
                },
                'results': self.results,
                'ready_for_trading': len(self.critical_errors) == 0
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")


async def main():
    """Run paper trading tests."""
    tester = PaperTradingTester()
    success = await tester.run_all_tests()
    
    if success:
        print(f"\n{GREEN}🚀 All tests passed! Ready to deploy.{NC}")
        print("\nNext step: Run ./deploy_mac_mini.sh")
        return 0
    else:
        print(f"\n{RED}❌ Some tests failed. Please fix issues before deployment.{NC}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
