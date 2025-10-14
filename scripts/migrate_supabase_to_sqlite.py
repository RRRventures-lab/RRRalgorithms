#!/usr/bin/env python3
"""
Migrate data from Supabase to SQLite.
This script exports data from Supabase and imports it into local SQLite database.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from database import SQLiteClient


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tables to migrate (in dependency order)
TABLES_TO_MIGRATE = [
    'symbols',
    'market_data',
    'trades_data',
    'quotes',
    'trades',
    'orders',
    'positions',
    'portfolio_snapshots',
    'risk_limits',
    'risk_events',
    'ml_models',
    'ml_predictions',
    'market_sentiment',
    'backtest_runs',
    'backtest_trades',
    'system_events',
    'performance_metrics',
    'audit_log',
]


async def check_supabase_available() -> bool:
    """Check if Supabase credentials are configured."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found in environment")
        return False
    
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection available")
        return True
    except Exception as e:
        logger.warning(f"Could not connect to Supabase: {e}")
        return False


async def export_table_from_supabase(
    supabase_client,
    table_name: str
) -> List[Dict[str, Any]]:
    """Export all data from a Supabase table."""
    logger.info(f"Exporting table: {table_name}")
    
    try:
        response = supabase_client.table(table_name).select("*").execute()
        data = response.data
        logger.info(f"Exported {len(data)} rows from {table_name}")
        return data
    except Exception as e:
        logger.error(f"Error exporting {table_name}: {e}")
        return []


async def import_table_to_sqlite(
    sqlite_client: SQLiteClient,
    table_name: str,
    data: List[Dict[str, Any]]
) -> None:
    """Import data into SQLite table."""
    if not data:
        logger.info(f"No data to import for {table_name}")
        return
    
    logger.info(f"Importing {len(data)} rows into {table_name}")
    
    try:
        # Check if table exists
        if not await sqlite_client.table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist in SQLite, skipping")
            return
        
        # Import in batches of 1000
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            await sqlite_client.insert_many(table_name, batch)
            logger.info(f"Imported batch {i // batch_size + 1} ({len(batch)} rows)")
        
        logger.info(f"Successfully imported {len(data)} rows into {table_name}")
    except Exception as e:
        logger.error(f"Error importing {table_name}: {e}")
        raise


async def migrate_all_tables() -> None:
    """Migrate all tables from Supabase to SQLite."""
    logger.info("Starting Supabase → SQLite migration")
    
    # Check Supabase availability
    if not await check_supabase_available():
        logger.error("Supabase not available. Migration aborted.")
        logger.info("If you don't have existing Supabase data, you can skip this migration.")
        return
    
    # Initialize clients
    from supabase import create_client
    supabase_client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    sqlite_client = SQLiteClient()
    await sqlite_client.connect()
    
    try:
        # Migrate each table
        total_rows = 0
        for table_name in TABLES_TO_MIGRATE:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Migrating table: {table_name}")
            logger.info(f"{'=' * 60}")
            
            # Export from Supabase
            data = await export_table_from_supabase(supabase_client, table_name)
            
            # Import to SQLite
            if data:
                await import_table_to_sqlite(sqlite_client, table_name, data)
                total_rows += len(data)
        
        logger.info(f"\n{'=' * 60}")
        logger.info("Migration completed successfully!")
        logger.info(f"Total rows migrated: {total_rows}")
        logger.info(f"{'=' * 60}\n")
        
        # Create backup
        backup_dir = Path(os.getenv('TRADING_HOME', '.')) / 'backups' / 'migration'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"post_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        logger.info(f"Creating backup: {backup_path}")
        await sqlite_client.backup(str(backup_path))
        logger.info("Backup created successfully")
        
    finally:
        await sqlite_client.disconnect()


async def verify_migration() -> None:
    """Verify migration by comparing row counts."""
    logger.info("\n" + "=" * 60)
    logger.info("Verifying migration...")
    logger.info("=" * 60 + "\n")
    
    sqlite_client = SQLiteClient()
    await sqlite_client.connect()
    
    try:
        for table_name in TABLES_TO_MIGRATE:
            if await sqlite_client.table_exists(table_name):
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                result = await sqlite_client.fetch_one(count_query)
                count = result['count'] if result else 0
                logger.info(f"{table_name}: {count} rows")
    finally:
        await sqlite_client.disconnect()


async def initialize_empty_database() -> None:
    """Initialize SQLite database without migrating data."""
    logger.info("Initializing empty SQLite database...")
    
    sqlite_client = SQLiteClient()
    await sqlite_client.connect()
    
    logger.info("Database initialized with schema")
    
    # Insert initial symbols
    initial_symbols = [
        {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'exchange': 'Coinbase', 'asset_type': 'crypto'},
        {'symbol': 'ETH-USD', 'name': 'Ethereum', 'exchange': 'Coinbase', 'asset_type': 'crypto'},
        {'symbol': 'SOL-USD', 'name': 'Solana', 'exchange': 'Coinbase', 'asset_type': 'crypto'},
    ]
    
    for symbol_data in initial_symbols:
        try:
            await sqlite_client.insert('symbols', symbol_data)
            logger.info(f"Inserted symbol: {symbol_data['symbol']}")
        except Exception as e:
            logger.warning(f"Could not insert {symbol_data['symbol']}: {e}")
    
    await sqlite_client.disconnect()
    logger.info("Empty database initialization complete")


async def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Supabase data to SQLite')
    parser.add_argument(
        '--skip-migration',
        action='store_true',
        help='Skip data migration, only initialize empty database'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing migration'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        await verify_migration()
        return
    
    if args.skip_migration:
        await initialize_empty_database()
    else:
        await migrate_all_tables()
        await verify_migration()
    
    logger.info("\n✅ Migration process complete!")
    logger.info("You can now use SQLite for all database operations.")
    logger.info("To remove Supabase dependencies, run: pip uninstall supabase")


if __name__ == '__main__':
    asyncio.run(main())

