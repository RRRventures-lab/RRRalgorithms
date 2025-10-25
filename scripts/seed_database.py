#!/usr/bin/env python3
"""
Database seeding script for RRRalgorithms transparency dashboard.
Populates the database with realistic sample data for testing and development.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import random
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.client_factory import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample symbols
SYMBOLS = [
    ('BTC-USD', 'Bitcoin', 'crypto'),
    ('ETH-USD', 'Ethereum', 'crypto'),
    ('SOL-USD', 'Solana', 'crypto'),
    ('MATIC-USD', 'Polygon', 'crypto'),
    ('AVAX-USD', 'Avalanche', 'crypto'),
]

# Sample model names
ML_MODELS = [
    ('Transformer-v1', 'neural_network', 'Price prediction transformer', 0.625),
    ('LSTM-v2', 'neural_network', 'LSTM sequence model', 0.583),
    ('QAOA-Portfolio', 'quantum', 'Quantum portfolio optimizer', 0.550),
]


async def seed_symbols(db):
    """Seed symbols table."""
    logger.info("Seeding symbols...")

    for symbol, name, asset_type in SYMBOLS:
        await db.insert('symbols', {
            'symbol': symbol,
            'name': name,
            'exchange': 'coinbase',
            'asset_type': asset_type,
            'active': 1
        })

    logger.info(f"Seeded {len(SYMBOLS)} symbols")


async def seed_market_data(db):
    """Seed market data table with historical OHLCV data."""
    logger.info("Seeding market data...")

    now = datetime.utcnow()
    base_prices = {
        'BTC-USD': 50000.0,
        'ETH-USD': 3000.0,
        'SOL-USD': 100.0,
        'MATIC-USD': 0.80,
        'AVAX-USD': 35.0,
    }

    records = []
    for symbol, _, _ in SYMBOLS:
        base_price = base_prices[symbol]
        price = base_price

        # Generate 7 days of hourly data
        for i in range(168):
            timestamp = now - timedelta(hours=168 - i)

            # Simulate price movement (simple random walk with mean reversion)
            change = random.uniform(-0.02, 0.02)  # ±2% max change
            price = price * (1 + change)

            # Add mean reversion
            if price > base_price * 1.1:
                price = price * 0.98
            elif price < base_price * 0.9:
                price = price * 1.02

            # Generate OHLCV
            high = price * random.uniform(1.0, 1.015)
            low = price * random.uniform(0.985, 1.0)
            open_price = random.uniform(low, high)
            close = random.uniform(low, high)
            volume = random.uniform(1000000, 5000000)

            records.append({
                'symbol': symbol,
                'timestamp': int(timestamp.timestamp()),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': round(volume, 2),
                'vwap': round(price, 2),
                'trade_count': random.randint(100, 1000)
            })

    await db.insert_many('market_data', records)
    logger.info(f"Seeded {len(records)} market data records")


async def seed_trades_and_orders(db):
    """Seed trades and orders tables."""
    logger.info("Seeding trades and orders...")

    now = datetime.utcnow()
    initial_capital = 100000.0
    cash_balance = initial_capital

    trades = []
    orders = []
    positions = {}

    # Generate 50 trades over the past 7 days
    for i in range(50):
        timestamp = now - timedelta(hours=random.randint(1, 168))
        symbol = random.choice([s[0] for s in SYMBOLS])
        side = random.choice(['buy', 'sell'])

        # Get approximate price for this symbol
        base_prices = {
            'BTC-USD': 50000.0, 'ETH-USD': 3000.0, 'SOL-USD': 100.0,
            'MATIC-USD': 0.80, 'AVAX-USD': 35.0
        }
        price = base_prices[symbol] * random.uniform(0.95, 1.05)

        # Calculate quantity based on position size (e.g., $5000 per trade)
        trade_size = 5000.0
        quantity = trade_size / price

        fees = trade_size * 0.001  # 0.1% fee

        # Create order first
        order_id = len(orders) + 1
        orders.append({
            'symbol': symbol,
            'side': side,
            'order_type': random.choice(['market', 'limit']),
            'quantity': round(quantity, 6),
            'price': round(price, 2) if random.random() > 0.5 else None,
            'status': 'filled',
            'exchange': 'coinbase',
            'filled_quantity': round(quantity, 6),
            'average_fill_price': round(price, 2),
            'timestamp': int(timestamp.timestamp()),
            'filled_at': int(timestamp.timestamp()),
            'strategy': random.choice(['momentum', 'mean_reversion', 'ml_signal'])
        })

        # Create trade
        trades.append({
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': round(quantity, 6),
            'price': round(price, 2),
            'timestamp': int(timestamp.timestamp()),
            'status': 'filled',
            'exchange': 'coinbase',
            'fees': round(fees, 2),
            'strategy': orders[-1]['strategy']
        })

        # Update positions
        if symbol not in positions:
            positions[symbol] = {
                'quantity': 0.0,
                'total_cost': 0.0,
                'opened_at': int(timestamp.timestamp())
            }

        if side == 'buy':
            positions[symbol]['quantity'] += quantity
            positions[symbol]['total_cost'] += (quantity * price + fees)
            cash_balance -= (quantity * price + fees)
        else:
            positions[symbol]['quantity'] -= quantity
            positions[symbol]['total_cost'] -= (quantity * price - fees)
            cash_balance += (quantity * price - fees)

    await db.insert_many('orders', orders)
    await db.insert_many('trades', trades)

    logger.info(f"Seeded {len(orders)} orders and {len(trades)} trades")

    # Seed positions
    logger.info("Seeding positions...")
    position_records = []
    for symbol, pos_data in positions.items():
        if abs(pos_data['quantity']) > 0.001:  # Only active positions
            avg_price = pos_data['total_cost'] / pos_data['quantity'] if pos_data['quantity'] != 0 else 0

            # Get current price
            base_prices = {
                'BTC-USD': 51000.0, 'ETH-USD': 3050.0, 'SOL-USD': 102.0,
                'MATIC-USD': 0.82, 'AVAX-USD': 36.0
            }
            current_price = base_prices.get(symbol, avg_price)
            unrealized_pnl = (current_price - avg_price) * pos_data['quantity']

            position_records.append({
                'symbol': symbol,
                'quantity': round(pos_data['quantity'], 6),
                'average_price': round(avg_price, 2),
                'current_price': round(current_price, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'realized_pnl': 0.0,
                'opened_at': pos_data['opened_at']
            })

    if position_records:
        await db.insert_many('positions', position_records)
        logger.info(f"Seeded {len(position_records)} positions")

    return cash_balance, sum(p['unrealized_pnl'] for p in position_records)


async def seed_portfolio_snapshots(db, cash_balance, unrealized_pnl):
    """Seed portfolio snapshots."""
    logger.info("Seeding portfolio snapshots...")

    now = datetime.utcnow()
    initial_capital = 100000.0
    current_equity = cash_balance + unrealized_pnl

    snapshots = []

    # Generate hourly snapshots for the past 7 days
    equity = initial_capital
    for i in range(168):
        timestamp = now - timedelta(hours=168 - i)

        # Simulate equity growth with volatility
        if i > 0:
            change = random.uniform(-0.005, 0.008)  # Slight upward bias
            equity = equity * (1 + change)

        daily_pnl = equity - (snapshots[-1]['total_value'] if snapshots else initial_capital)
        total_pnl = equity - initial_capital

        snapshots.append({
            'timestamp': int(timestamp.timestamp()),
            'total_value': round(equity, 2),
            'cash_balance': round(equity * random.uniform(0.3, 0.5), 2),
            'positions_value': round(equity * random.uniform(0.5, 0.7), 2),
            'daily_pnl': round(daily_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'num_positions': random.randint(2, 5)
        })

    await db.insert_many('portfolio_snapshots', snapshots)
    logger.info(f"Seeded {len(snapshots)} portfolio snapshots")


async def seed_ml_models(db):
    """Seed ML models table."""
    logger.info("Seeding ML models...")

    now = datetime.utcnow()
    model_records = []

    for i, (model_name, model_type, description, accuracy) in enumerate(ML_MODELS):
        metrics = {
            'accuracy': accuracy,
            'avg_confidence': random.uniform(0.65, 0.85),
            'win_rate': random.uniform(0.55, 0.70),
            'description': description
        }

        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'layers': random.randint(3, 8)
        }

        model_records.append({
            'model_name': model_name,
            'model_type': model_type,
            'version': f"v{i+1}.0",
            'file_path': f"/models/{model_name.lower().replace(' ', '_')}.pkl",
            'metrics': json.dumps(metrics),
            'hyperparameters': json.dumps(hyperparams),
            'trained_at': int((now - timedelta(days=random.randint(1, 30))).timestamp()),
            'active': 1
        })

    await db.insert_many('ml_models', model_records)
    logger.info(f"Seeded {len(model_records)} ML models")

    return len(model_records)


async def seed_ml_predictions(db, num_models):
    """Seed ML predictions table."""
    logger.info("Seeding ML predictions...")

    now = datetime.utcnow()
    predictions = []

    # Generate 100 predictions across all models
    for _ in range(100):
        timestamp = now - timedelta(hours=random.randint(1, 168))
        model_id = random.randint(1, num_models)
        symbol = random.choice([s[0] for s in SYMBOLS])

        features = {
            'rsi_14': random.uniform(20, 80),
            'macd': random.uniform(-100, 100),
            'volume_ratio': random.uniform(0.5, 2.0),
            'trend_strength': random.uniform(0, 1)
        }

        predictions.append({
            'model_id': model_id,
            'symbol': symbol,
            'timestamp': int(timestamp.timestamp()),
            'prediction_type': 'price_direction',
            'prediction_value': random.uniform(-5.0, 5.0),  # % change
            'confidence': random.uniform(0.6, 0.95),
            'features': json.dumps(features)
        })

    await db.insert_many('ml_predictions', predictions)
    logger.info(f"Seeded {len(predictions)} ML predictions")


async def seed_backtests(db):
    """Seed backtest runs and trades."""
    logger.info("Seeding backtests...")

    now = datetime.utcnow()
    backtest_runs = []

    strategies = [
        ('Momentum Strategy v3', 'momentum', 15.5, 1.92, -8.5, 0.672),
        ('Mean Reversion v2', 'mean_reversion', 12.3, 1.75, -6.2, 0.585),
        ('ML Ensemble v1', 'ml_ensemble', 18.7, 2.05, -9.1, 0.701),
    ]

    for i, (name, strategy, total_return, sharpe, drawdown, win_rate) in enumerate(strategies):
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 10, 1)

        config = {
            'description': f"{name} backtest",
            'indicators': ['RSI', 'MACD', 'Bollinger Bands'],
            'risk_per_trade': 0.02
        }

        results = {
            'sortino_ratio': sharpe * 1.15,
            'calmar_ratio': total_return / abs(drawdown) if drawdown != 0 else 0,
            'profit_factor': random.uniform(1.5, 2.5)
        }

        total_trades = random.randint(150, 300)

        backtest_runs.append({
            'run_name': name,
            'strategy_name': strategy,
            'start_date': int(start_date.timestamp()),
            'end_date': int(end_date.timestamp()),
            'initial_capital': 100000.0,
            'final_capital': 100000.0 * (1 + total_return / 100),
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'config': json.dumps(config),
            'results': json.dumps(results)
        })

    await db.insert_many('backtest_runs', backtest_runs)
    logger.info(f"Seeded {len(backtest_runs)} backtest runs")


async def main():
    """Main seeding function."""
    logger.info("=" * 60)
    logger.info("Starting database seeding process...")
    logger.info("=" * 60)

    # Initialize database
    db = get_db()
    await db.connect()

    try:
        # Seed all tables
        await seed_symbols(db)
        await seed_market_data(db)
        cash_balance, unrealized_pnl = await seed_trades_and_orders(db)
        await seed_portfolio_snapshots(db, cash_balance, unrealized_pnl)
        num_models = await seed_ml_models(db)
        await seed_ml_predictions(db, num_models)
        await seed_backtests(db)

        logger.info("=" * 60)
        logger.info("Database seeding completed successfully!")
        logger.info("=" * 60)

        # Print summary
        for table in ['symbols', 'market_data', 'trades', 'orders', 'positions',
                      'portfolio_snapshots', 'ml_models', 'ml_predictions', 'backtest_runs']:
            count = await db.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
            logger.info(f"  {table}: {count['count']} records")

    except Exception as e:
        logger.error(f"Error during seeding: {e}")
        raise
    finally:
        await db.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
