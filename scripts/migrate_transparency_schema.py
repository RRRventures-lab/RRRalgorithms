#!/usr/bin/env python3
"""
Database Migration Script for Transparency Dashboard
Applies the transparency schema to the database

Usage:
    python scripts/migrate_transparency_schema.py --db-path /path/to/database.db
    python scripts/migrate_transparency_schema.py --supabase  # For Supabase/PostgreSQL
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime


# Transparency Schema for SQLite
SQLITE_SCHEMA = """
-- ============================================================================
-- RRRalgorithms Transparency Dashboard Schema (SQLite)
-- ============================================================================

-- AI Decision Log
CREATE TABLE IF NOT EXISTS ai_decisions (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,

    -- Prediction details (stored as JSON strings in SQLite)
    prediction TEXT NOT NULL,  -- JSON: {direction, confidence, price_target, time_horizon}
    features TEXT NOT NULL,     -- JSON: Input features used
    reasoning TEXT,             -- Human-readable explanation

    -- Outcome tracking
    outcome TEXT,               -- 'profitable', 'loss', 'pending', 'cancelled'
    actual_return REAL,         -- Actual return if prediction was acted upon
    prediction_error REAL,      -- Error between predicted and actual

    -- Metadata
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    time_horizon TEXT,          -- '1h', '4h', '1d', etc.
    strategy_name TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ai_decisions_timestamp ON ai_decisions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol ON ai_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_model ON ai_decisions(model_name);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_outcome ON ai_decisions(outcome);

-- Live Trading Feed
CREATE TABLE IF NOT EXISTS trade_feed (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,  -- 'signal', 'order_placed', 'order_filled', 'position_closed'
    symbol TEXT NOT NULL,

    -- Event data (stored as JSON)
    data TEXT NOT NULL,

    -- Metadata
    source TEXT,                -- 'trading_engine', 'risk_manager', etc.
    severity TEXT DEFAULT 'info',  -- 'info', 'warning', 'critical'

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trade_feed_timestamp ON trade_feed(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trade_feed_symbol ON trade_feed(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_feed_event_type ON trade_feed(event_type);

-- Performance Snapshots
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,

    -- Portfolio metrics
    total_equity REAL NOT NULL,
    cash_balance REAL NOT NULL,
    invested_value REAL NOT NULL,

    -- P&L metrics
    total_pnl REAL NOT NULL,
    total_pnl_percent REAL NOT NULL,
    daily_pnl REAL,
    daily_pnl_percent REAL,

    -- Risk metrics
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    max_drawdown_percent REAL,

    -- Trading metrics
    total_trades INTEGER DEFAULT 0,
    win_rate REAL,
    profit_factor REAL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_performance_snapshots_timestamp ON performance_snapshots(timestamp DESC);

-- AI Model Performance
CREATE TABLE IF NOT EXISTS ai_model_performance (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    date DATE NOT NULL,

    -- Performance metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy REAL,
    avg_confidence REAL,

    -- P&L metrics
    total_pnl REAL,
    win_rate REAL,
    sharpe_ratio REAL,

    -- Updated tracking
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_name, date)
);

CREATE INDEX IF NOT EXISTS idx_ai_model_performance_model ON ai_model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_ai_model_performance_date ON ai_model_performance(date DESC);

-- Backtest Results
CREATE TABLE IF NOT EXISTS backtest_results (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,

    -- Backtest configuration
    strategy_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital REAL NOT NULL,

    -- Performance metrics
    final_equity REAL NOT NULL,
    total_return REAL NOT NULL,
    total_return_percent REAL NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    max_drawdown_percent REAL,
    calmar_ratio REAL,

    -- Trading metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL,
    profit_factor REAL,
    avg_win REAL,
    avg_loss REAL,

    -- Status
    status TEXT DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_results_created ON backtest_results(created_at DESC);

-- Backtest Trades
CREATE TABLE IF NOT EXISTS backtest_trades (
    id TEXT PRIMARY KEY,
    backtest_id TEXT NOT NULL,

    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,

    -- Trade details
    order_type TEXT,
    pnl REAL,
    pnl_percent REAL,

    FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_timestamp ON backtest_trades(timestamp);

-- System Events
CREATE TABLE IF NOT EXISTS system_events (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,  -- 'startup', 'shutdown', 'error', 'warning', 'info'
    component TEXT NOT NULL,    -- 'api', 'trading_engine', 'data_pipeline', etc.

    -- Event details
    message TEXT NOT NULL,
    details TEXT,  -- JSON string
    severity TEXT DEFAULT 'info',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_component ON system_events(component);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events(severity);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    alert_type TEXT NOT NULL,  -- 'risk_limit', 'performance', 'system', 'data_quality'
    severity TEXT NOT NULL,     -- 'info', 'warning', 'critical'

    -- Alert content
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT,  -- JSON string

    -- Status
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at DATETIME,
    acknowledged_by TEXT,
    resolved BOOLEAN DEFAULT 0,
    resolved_at DATETIME,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
"""


def migrate_sqlite(db_path: str):
    """Apply transparency schema to SQLite database"""
    print(f"Applying transparency schema to SQLite database: {db_path}")

    # Check if file exists
    db_file = Path(db_path)
    if not db_file.parent.exists():
        print(f"Creating directory: {db_file.parent}")
        db_file.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Execute schema
        cursor.executescript(SQLITE_SCHEMA)

        # Commit changes
        conn.commit()

        # Verify tables were created
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE '%ai%' OR name LIKE '%trade%' OR name LIKE '%performance%'
            ORDER BY name
        """)
        tables = cursor.fetchall()

        print(f"\n✓ Successfully created {len(tables)} transparency tables:")
        for table in tables:
            print(f"  - {table[0]}")

        # Log migration
        cursor.execute("""
            INSERT INTO system_events (id, timestamp, event_type, component, message, severity)
            VALUES (?, ?, 'migration', 'database', 'Transparency schema applied successfully', 'info')
        """, (f"migration-{datetime.utcnow().isoformat()}", datetime.utcnow()))
        conn.commit()

        print(f"\n✓ Migration completed successfully at {datetime.utcnow().isoformat()}")

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate_supabase():
    """Print instructions for applying schema to Supabase/PostgreSQL"""
    schema_file = Path(__file__).parent.parent / "docs" / "database" / "transparency_schema.sql"

    print("\nTo apply the transparency schema to Supabase/PostgreSQL:")
    print(f"1. Open the SQL schema file: {schema_file}")
    print("2. Go to your Supabase dashboard: https://app.supabase.com")
    print("3. Navigate to SQL Editor")
    print("4. Copy and paste the SQL schema")
    print("5. Execute the query")
    print("\nAlternatively, use psql:")
    print(f"  psql <connection_string> -f {schema_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply transparency dashboard schema to database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--supabase",
        action="store_true",
        help="Show instructions for Supabase/PostgreSQL"
    )

    args = parser.parse_args()

    if args.supabase:
        migrate_supabase()
    elif args.db_path:
        migrate_sqlite(args.db_path)
    else:
        print("Error: Must specify either --db-path or --supabase")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
