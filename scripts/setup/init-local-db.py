from pathlib import Path
from src.core.config.loader import get_config
from src.core.database.local_db import LocalDatabase, get_db
import random
import sys
import time

#!/usr/bin/env python3
"""
Initialize local SQLite database for development.
Creates all necessary tables and optionally loads sample data.
"""


# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))



def init_database(with_sample_data: bool = True):
    """Initialize database and optionally load sample data."""
    print("🔧 Initializing local SQLite database...")
    
    # Get config
    config = get_config()
    db_path = config.get('database.path', 'data/local.db')
    
    print(f"📁 Database path: {db_path}")
    
    # Initialize database (creates tables automatically)
    db = get_db(db_path)
    print("✅ Database schema created successfully")
    
    if with_sample_data:
        print("\n📊 Loading sample data...")
        load_sample_data(db)
        print("✅ Sample data loaded successfully")
    
    print("\n🎉 Database initialization complete!")
    print(f"   Location: {db.db_path}")
    print(f"   Tables: market_data, trades, positions, portfolio_metrics,")
    print(f"           predictions, risk_metrics, system_logs")
    
    return db


def load_sample_data(db: LocalDatabase):
    """Load sample historical data for testing."""
    symbols = ["BTC-USD", "ETH-USD"]
    
    # Generate sample market data (last 100 periods)
    print("  - Generating sample market data...")
    current_time = time.time()
    
    for symbol in symbols:
        base_price = 50000 if symbol == "BTC-USD" else 3000
        price = base_price
        
        for i in range(100, 0, -1):
            # Random walk price
            change = random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            
            timestamp = current_time - (i * 60)  # 1 minute bars
            
            ohlcv = {
                'open': price * 0.998,
                'high': price * 1.005,
                'low': price * 0.995,
                'close': price,
                'volume': random.uniform(100000, 500000)
            }
            
            db.insert_market_data(symbol, timestamp, ohlcv)
    
    print(f"    ✓ Created {len(symbols) * 100} market data points")
    
    # Create sample trades
    print("  - Creating sample trades...")
    sample_trades = [
        {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'order_type': 'market',
            'quantity': 0.1,
            'price': 49800,
            'timestamp': current_time - 3600,
            'status': 'filled',
            'strategy': 'momentum'
        },
        {
            'symbol': 'ETH-USD',
            'side': 'buy',
            'order_type': 'limit',
            'quantity': 1.5,
            'price': 2980,
            'timestamp': current_time - 3000,
            'status': 'filled',
            'strategy': 'mean_reversion'
        }
    ]
    
    for trade in sample_trades:
        db.insert_trade(trade)
    
    print(f"    ✓ Created {len(sample_trades)} sample trades")
    
    # Create sample positions
    print("  - Creating sample positions...")
    db.upsert_position('BTC-USD', 0.1, 49800, 50500)
    db.upsert_position('ETH-USD', 1.5, 2980, 3020)
    print("    ✓ Created 2 sample positions")
    
    # Create portfolio metrics
    print("  - Creating portfolio metrics...")
    metrics = {
        'timestamp': current_time,
        'total_value': 10500,
        'cash': 5000,
        'positions_value': 5500,
        'daily_pnl': 150,
        'total_pnl': 500,
        'sharpe_ratio': 1.8,
        'sortino_ratio': 2.2,
        'max_drawdown': -0.05,
        'win_rate': 0.65
    }
    db.insert_portfolio_metrics(metrics)
    print("    ✓ Created initial portfolio metrics")
    
    # Log initialization
    db.log('INFO', 'database', 'Sample data loaded successfully', {
        'symbols': symbols,
        'data_points': len(symbols) * 100
    })


def reset_database():
    """Reset database by deleting and recreating."""
    config = get_config()
    db_path = Path(config.get('database.path', 'data/local.db'))
    
    if db_path.exists():
        print(f"⚠️  Deleting existing database: {db_path}")
        db_path.unlink()
    
    init_database(with_sample_data=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize local SQLite database')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset database (delete and recreate)')
    parser.add_argument('--no-sample-data', action='store_true',
                       help='Skip loading sample data')
    
    args = parser.parse_args()
    
    try:
        if args.reset:
            reset_database()
        else:
            init_database(with_sample_data=not args.no_sample_data)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

