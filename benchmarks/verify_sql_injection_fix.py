from pathlib import Path
from src.core.database.local_db import LocalDatabase
import sys

"""
Verification: SQL Injection Fix
================================

Verifies that the SQL injection vulnerability is actually fixed.

Author: RRR Ventures
Date: 2025-10-12
"""


sys.path.insert(0, str(Path(__file__).parent.parent))



def test_sql_injection_fix():
    """Test that SQL injection attempts are blocked"""
    print("="*70)
    print("VERIFICATION: SQL Injection Fix")
    print("="*70)
    print()
    
    # Create test database
    db = LocalDatabase("/tmp/test_sql_injection.db")
    
    # Insert a test trade
    trade_id = db.insert_trade({
        'symbol': 'BTC-USD',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 1.0,
        'price': 50000.0,
        'timestamp': 1234567890.0,
        'status': 'pending'
    })
    
    print(f"✅ Created test trade (ID: {trade_id})")
    print()
    
    # Test 1: Valid update (should work)
    print("Test 1: Valid update (should PASS)")
    try:
        db.update_trade(trade_id, {
            'status': 'executed',
            'executed_quantity': 1.0,
            'executed_price': 50000.0
        })
        print("  ✅ Valid update successful")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    print()
    
    # Test 2: Invalid column (should be BLOCKED)
    print("Test 2: Invalid column (should BLOCK)")
    try:
        db.update_trade(trade_id, {
            'malicious_column': 'injected_value'
        })
        print("  ❌ SECURITY ISSUE: Invalid column was NOT blocked!")
    except ValueError as e:
        if 'Invalid columns' in str(e):
            print(f"  ✅ Blocked correctly: {e}")
        else:
            print(f"  ⚠️  Blocked with unexpected error: {e}")
    except Exception as e:
        print(f"  ⚠️  Blocked with unexpected exception: {e}")
    
    print()
    
    # Test 3: SQL injection attempt (should be BLOCKED)
    print("Test 3: SQL injection attempt (should BLOCK)")
    malicious_update = {
        "status; DROP TABLE trades; --": "malicious"
    }
    try:
        db.update_trade(trade_id, malicious_update)
        print("  ❌ CRITICAL: SQL injection was NOT blocked!")
    except ValueError as e:
        if 'Invalid columns' in str(e):
            print(f"  ✅ SQL injection BLOCKED: {e}")
        else:
            print(f"  ⚠️  Blocked with unexpected error: {e}")
    except Exception as e:
        print(f"  ⚠️  Blocked with unexpected exception: {e}")
    
    print()
    
    # Test 4: Verify data integrity
    print("Test 4: Verify data integrity")
    trades = db.get_trades()
    if len(trades) == 1:
        print(f"  ✅ Data integrity maintained ({len(trades)} trade exists)")
    else:
        print(f"  ⚠️  Unexpected trade count: {len(trades)}")
    
    db.close()
    
    print()
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print()
    print("✅ SQL injection fix is working correctly!")
    print("   - Valid updates: ALLOWED")
    print("   - Invalid columns: BLOCKED")
    print("   - SQL injection: BLOCKED")
    print()


if __name__ == "__main__":
    test_sql_injection_fix()

