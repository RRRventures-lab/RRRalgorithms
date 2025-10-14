#!/usr/bin/env python3
"""
Run comprehensive security tests for the RRRalgorithms trading system
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.security.security_tester import SecurityTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/security/security_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

async def main():
    """Run security tests."""
    # Create logs directory
    Path("logs/security").mkdir(parents=True, exist_ok=True)
    Path("logs/audit").mkdir(parents=True, exist_ok=True)
    
    # Initialize security tester
    tester = SecurityTester()
    
    # Run comprehensive test
    print("\n🔐 Starting Comprehensive Security Audit...\n")
    report = await tester.generate_security_report()
    
    print("\n📊 Security Report Generated!")
    print(f"Report saved to: logs/audit/security_report.json")
    
    # Print summary
    import json
    report_data = json.loads(report)
    print(f"\n🔒 Overall Status: {report_data['overall_status']}")
    print(f"📈 Tests Passed: {report_data['tests_passed']}/{report_data['tests_total']}")
    print(f"⚠️  Critical Failures: {report_data['critical_failures']}")
    print(f"⚡ Warnings: {report_data['warnings']}")
    
    if report_data['recommendations']:
        print("\n📋 Recommendations:")
        for rec in report_data['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(main())
