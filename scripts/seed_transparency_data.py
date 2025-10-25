#!/usr/bin/env python3
"""
Seed sample data into transparency database for testing

This script populates the transparency database with realistic sample data
for testing the API endpoints.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.transparency_db import TransparencyDB


async def seed_performance_snapshots(db: TransparencyDB, days: int = 7):
    """Seed performance snapshots for the last N days"""
    print(f"Seeding {days} days of performance snapshots...")

    initial_equity = 100000.00
    current_equity = initial_equity
    max_equity = initial_equity

    # Generate hourly snapshots
    hours = days * 24

    for i in range(hours):
        timestamp_offset = timedelta(hours=hours - i)

        # Simulate equity changes (random walk with slight upward bias)
        change_percent = random.uniform(-0.5, 0.7)  # Slight upward bias
        change_amount = current_equity * (change_percent / 100)
        current_equity += change_amount

        # Track max equity for drawdown calculation
        if current_equity > max_equity:
            max_equity = current_equity

        # Calculate metrics
        total_pnl = current_equity - initial_equity
        total_pnl_percent = (total_pnl / initial_equity) * 100

        # Daily P&L (compare to 24 hours ago)
        if i >= 24:
            daily_pnl = change_amount * 24  # Simplified
            daily_pnl_percent = (daily_pnl / current_equity) * 100
        else:
            daily_pnl = total_pnl
            daily_pnl_percent = total_pnl_percent

        # Risk metrics (simplified)
        drawdown = ((max_equity - current_equity) / max_equity) * 100
        sharpe_ratio = random.uniform(1.5, 2.5)
        sortino_ratio = random.uniform(1.8, 2.8)

        # Trading metrics
        total_trades = i * 2  # Roughly 2 trades per hour
        win_rate = random.uniform(58, 68)
        profit_factor = random.uniform(1.5, 2.2)

        await db.add_performance_snapshot(
            total_equity=round(current_equity, 2),
            cash_balance=round(current_equity * 0.3, 2),
            invested_value=round(current_equity * 0.7, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_percent, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_percent=round(daily_pnl_percent, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            max_drawdown=round(-drawdown, 2),
            max_drawdown_percent=round(-drawdown, 2),
            total_trades=total_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2)
        )

    print(f"✓ Seeded {hours} performance snapshots")
    print(f"  Initial equity: ${initial_equity:,.2f}")
    print(f"  Final equity: ${current_equity:,.2f}")
    print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl_percent:+.2f}%)")


async def seed_trade_events(db: TransparencyDB, count: int = 100):
    """Seed trade events"""
    print(f"Seeding {count} trade events...")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
    event_types = ["signal", "order_placed", "order_filled", "position_closed"]
    sides = ["buy", "sell"]

    for i in range(count):
        symbol = random.choice(symbols)
        event_type = random.choice(event_types)
        side = random.choice(sides)

        # Generate realistic price based on symbol
        if "BTC" in symbol:
            price = random.uniform(40000, 60000)
            quantity = random.uniform(0.01, 1.0)
        elif "ETH" in symbol:
            price = random.uniform(2000, 4000)
            quantity = random.uniform(0.1, 10.0)
        elif "SOL" in symbol:
            price = random.uniform(20, 200)
            quantity = random.uniform(1, 100)
        else:
            price = random.uniform(10, 100)
            quantity = random.uniform(1, 50)

        data = {
            "side": side,
            "quantity": round(quantity, 4),
            "price": round(price, 2),
            "total_value": round(quantity * price, 2),
            "fee": round(quantity * price * 0.001, 2),  # 0.1% fee
            "status": "filled" if event_type == "order_filled" else "pending",
            "order_type": random.choice(["market", "limit"])
        }

        await db.add_trade_event(
            event_type=event_type,
            symbol=symbol,
            data=data,
            source="trading_engine"
        )

    print(f"✓ Seeded {count} trade events")


async def seed_ai_decisions(db: TransparencyDB, count: int = 50):
    """Seed AI decisions"""
    print(f"Seeding {count} AI decisions...")

    models = [
        ("Transformer-v1", "neural_network"),
        ("LSTM-v2", "neural_network"),
        ("GRU-v1", "neural_network"),
        ("QAOA-Portfolio", "quantum"),
        ("Ensemble-v1", "ensemble")
    ]

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
    directions = ["up", "down", "neutral"]
    outcomes = ["profitable", "loss", "pending"]

    for i in range(count):
        model_name, model_type = random.choice(models)
        symbol = random.choice(symbols)
        direction = random.choice(directions)
        confidence = random.uniform(0.55, 0.95)

        # Generate price targets based on symbol
        if "BTC" in symbol:
            current_price = random.uniform(40000, 60000)
            target_change = random.uniform(-0.05, 0.05)
        elif "ETH" in symbol:
            current_price = random.uniform(2000, 4000)
            target_change = random.uniform(-0.06, 0.06)
        else:
            current_price = random.uniform(20, 200)
            target_change = random.uniform(-0.08, 0.08)

        price_target = current_price * (1 + target_change)

        prediction = {
            "direction": direction,
            "confidence": round(confidence, 3),
            "price_target": round(price_target, 2),
            "time_horizon": random.choice(["1h", "4h", "1d"])
        }

        features = {
            "rsi_14": random.uniform(30, 70),
            "macd": random.uniform(-100, 100),
            "volume_ratio": random.uniform(0.8, 2.0),
            "trend_strength": random.uniform(0.3, 0.9),
            "volatility": random.uniform(0.01, 0.05)
        }

        # Generate reasoning
        if direction == "up":
            reasoning = f"Bullish signals detected: RSI recovery, positive MACD, increasing volume. Target: ${price_target:.2f}"
        elif direction == "down":
            reasoning = f"Bearish signals: Overbought RSI, negative MACD divergence. Target: ${price_target:.2f}"
        else:
            reasoning = f"Neutral outlook: Mixed signals, consolidation expected around ${price_target:.2f}"

        await db.add_ai_decision(
            model_name=model_name,
            symbol=symbol,
            prediction=prediction,
            features=features,
            reasoning=reasoning,
            confidence_score=confidence
        )

    print(f"✓ Seeded {count} AI decisions")


async def main():
    """Main seeding function"""
    print("=" * 70)
    print("Transparency Database Seeding Script")
    print("=" * 70)
    print()

    # Connect to database
    db_path = Path(__file__).parent.parent / "data" / "transparency.db"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("Please run the migration script first:")
        print("  python scripts/migrate_transparency_schema.py --db-path data/transparency.db")
        return 1

    print(f"Database: {db_path}")
    print()

    db = TransparencyDB(str(db_path))

    try:
        await db.connect()
        print("✓ Connected to database")
        print()

        # Seed data
        await seed_performance_snapshots(db, days=7)
        print()

        await seed_trade_events(db, count=100)
        print()

        await seed_ai_decisions(db, count=50)
        print()

        print("=" * 70)
        print("✓ Database seeding completed successfully!")
        print("=" * 70)
        print()
        print("You can now test the API:")
        print("  python -m uvicorn src.api.main:app --reload")
        print()
        print("API will be available at:")
        print("  - http://localhost:8000")
        print("  - http://localhost:8000/docs (Swagger UI)")
        print()

        return 0

    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await db.disconnect()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
