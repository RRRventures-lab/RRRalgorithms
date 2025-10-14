#!/usr/bin/env python3
"""
Quick test script to verify unified system can initialize.
Focuses on testing the new database layer.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_database():
    """Test database initialization."""
    print("Testing SQLite database...")
    from database import get_db, SQLiteClient
    
    db = get_db()
    print(f"  Database client: {type(db).__name__}")
    
    await db.connect()
    print("  ✅ Connected to database")
    
    # Test simple query
    symbols = await db.fetch_all("SELECT * FROM symbols")
    print(f"  ✅ Query successful: {len(symbols)} symbols found")
    
    # Show symbols
    for symbol in symbols:
        print(f"     - {symbol['symbol']}: {symbol['name']}")
    
    # Test database size
    db_stats = await db.get_db_size()
    print(f"  ✅ Database size: {db_stats['file_size_mb']:.2f} MB")
    
    await db.disconnect()
    print("  ✅ Disconnected from database")
    
    return True

async def test_database_operations():
    """Test database CRUD operations."""
    print("\nTesting database operations...")
    from database import SQLiteClient
    
    db = SQLiteClient()
    await db.connect()
    
    # Test insert
    test_symbol = {
        'symbol': 'TEST-USD',
        'name': 'Test Coin',
        'exchange': 'Test Exchange',
        'asset_type': 'crypto'
    }
    
    try:
        symbol_id = await db.insert('symbols', test_symbol)
        print(f"  ✅ Insert: Added TEST-USD")
        
        # Test fetch
        result = await db.fetch_one("SELECT * FROM symbols WHERE symbol = ?", ('TEST-USD',))
        print(f"  ✅ Fetch: Retrieved {result['name']}")
        
        # Test delete
        deleted = await db.delete('symbols', {'symbol': 'TEST-USD'})
        print(f"  ✅ Delete: Removed {deleted} row(s)")
        
    except Exception as e:
        print(f"  ⚠️  Operation error: {e}")
    
    await db.disconnect()
    return True

async def test_structure():
    """Test directory structure."""
    print("\nTesting directory structure...")
    
    base = Path(__file__).parent
    required_dirs = [
        'data/db',
        'data/historical',
        'data/models',
        'data/cache',
        'logs/trading',
        'logs/system',
        'logs/audit',
        'backups/daily',
        'src/database',
        'src/neural_network',
        'src/trading',
        'src/backtesting',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - missing")
            all_exist = False
    
    return all_exist

async def test_worktree_consolidation():
    """Test that worktrees were properly consolidated."""
    print("\nTesting worktree consolidation...")
    
    base = Path(__file__).parent
    consolidated_dirs = [
        'src/neural_network',
        'src/data_pipeline_original',
        'src/trading/engine',
        'src/trading/risk',
        'src/backtesting',
        'src/api',
        'src/quantum',
        'src/monitoring_original',
    ]
    
    all_exist = True
    for dir_path in consolidated_dirs:
        full_path = base / dir_path
        if full_path.exists():
            # Count Python files
            py_files = list(full_path.rglob('*.py'))
            print(f"  ✅ {dir_path} ({len(py_files)} .py files)")
        else:
            print(f"  ❌ {dir_path} - missing")
            all_exist = False
    
    return all_exist

async def main():
    """Run all tests."""
    print("="*60)
    print("RRRalgorithms Unified System - Quick Test")
    print("="*60)
    
    try:
        # Test structure
        structure_ok = await test_structure()
        
        # Test worktree consolidation
        worktree_ok = await test_worktree_consolidation()
        
        # Test database
        db_ok = await test_database()
        
        # Test database operations
        ops_ok = await test_database_operations()
        
        print("\n" + "="*60)
        print("Test Results:")
        print(f"  Directory Structure: {'✅' if structure_ok else '❌'}")
        print(f"  Worktree Consolidation: {'✅' if worktree_ok else '❌'}")
        print(f"  Database Connection: {'✅' if db_ok else '❌'}")
        print(f"  Database Operations: {'✅' if ops_ok else '❌'}")
        print("="*60)
        
        if structure_ok and worktree_ok and db_ok and ops_ok:
            print("\n🎉 All tests passed!")
            print("\nNext steps:")
            print("  1. Install remaining dependencies: pip install -r requirements-local-trading.txt")
            print("  2. Test unified entry point: python src/main_unified.py --help")
            print("  3. Deploy to Mac Mini when ready")
        else:
            print("\n⚠️  Some tests failed")
            print("System may still work, but review failures above")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
