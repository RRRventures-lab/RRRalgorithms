-- ============================================================================
-- RRRalgorithms SQLite Database Schema
-- Optimized for local storage on Lexar 2TB drive
-- ============================================================================

-- Enable performance optimizations
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging (faster, concurrent)
PRAGMA synchronous = NORMAL;         -- Balance speed/safety
PRAGMA cache_size = -64000;          -- 64MB cache
PRAGMA temp_store = MEMORY;          -- Temp tables in RAM
PRAGMA foreign_keys = ON;            -- Enforce foreign keys
PRAGMA auto_vacuum = INCREMENTAL;    -- Manage space efficiently

-- ============================================================================
-- Core Tables
-- ============================================================================

-- Symbols/Instruments
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    asset_type TEXT CHECK(asset_type IN ('crypto', 'stock', 'forex', 'option')),
    active INTEGER DEFAULT 1,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols(active, exchange);

-- ============================================================================
-- Market Data Tables
-- ============================================================================

-- OHLCV Market Data (1-minute bars)
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix timestamp
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    vwap REAL,
    trade_count INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_created ON market_data(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_unique ON market_data(symbol, timestamp);

-- Individual Trades
CREATE TABLE IF NOT EXISTS trades_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    side TEXT CHECK(side IN ('buy', 'sell', 'unknown')),
    exchange_id INTEGER,
    trade_id TEXT,
    conditions TEXT, -- JSON array of condition codes
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trades_data_symbol_time ON trades_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_data_exchange_id ON trades_data(exchange_id);

-- Quotes (Bid/Ask)
CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    bid_price REAL NOT NULL,
    bid_size REAL NOT NULL,
    ask_price REAL NOT NULL,
    ask_size REAL NOT NULL,
    exchange_id INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes(symbol, timestamp DESC);

-- ============================================================================
-- Trading Tables
-- ============================================================================

-- Trades (Executed Orders)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    status TEXT CHECK(status IN ('pending', 'filled', 'partial', 'cancelled', 'failed')),
    exchange TEXT NOT NULL,
    fees REAL DEFAULT 0,
    notes TEXT,
    strategy TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    order_type TEXT NOT NULL CHECK(order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity REAL NOT NULL,
    price REAL,
    stop_price REAL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'open', 'filled', 'partial', 'cancelled', 'failed')),
    exchange TEXT NOT NULL,
    exchange_order_id TEXT,
    filled_quantity REAL DEFAULT 0,
    average_fill_price REAL,
    timestamp INTEGER NOT NULL,
    filled_at INTEGER,
    cancelled_at INTEGER,
    notes TEXT,
    strategy TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_exchange_id ON orders(exchange_order_id);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    quantity REAL NOT NULL,
    average_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    realized_pnl REAL DEFAULT 0,
    opened_at INTEGER NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Portfolio Snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    total_value REAL NOT NULL,
    cash_balance REAL NOT NULL,
    positions_value REAL NOT NULL,
    daily_pnl REAL,
    total_pnl REAL,
    num_positions INTEGER,
    metadata TEXT, -- JSON
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_time ON portfolio_snapshots(timestamp DESC);

-- ============================================================================
-- Risk Management Tables
-- ============================================================================

-- Risk Limits
CREATE TABLE IF NOT EXISTS risk_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    limit_type TEXT NOT NULL CHECK(limit_type IN ('daily_loss', 'position_size', 'portfolio_exposure', 'drawdown')),
    limit_value REAL NOT NULL,
    current_value REAL DEFAULT 0,
    threshold_pct REAL DEFAULT 0.8,
    active INTEGER DEFAULT 1,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_risk_limits_type ON risk_limits(limit_type, active);

-- Risk Events
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('low', 'medium', 'high', 'critical')),
    symbol TEXT,
    description TEXT,
    action_taken TEXT,
    timestamp INTEGER NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_risk_events_time ON risk_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity);

-- ============================================================================
-- Machine Learning Tables
-- ============================================================================

-- ML Models Registry
CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metrics TEXT, -- JSON
    hyperparameters TEXT, -- JSON
    trained_at INTEGER NOT NULL,
    active INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(active, model_type);

-- ML Predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    prediction_type TEXT NOT NULL,
    prediction_value REAL NOT NULL,
    confidence REAL,
    features TEXT, -- JSON
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (model_id) REFERENCES ml_models(id),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_time ON ml_predictions(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time ON ml_predictions(symbol, timestamp DESC);

-- ============================================================================
-- Sentiment & Alternative Data Tables
-- ============================================================================

-- Market Sentiment
CREATE TABLE IF NOT EXISTS market_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    source TEXT NOT NULL,
    sentiment_label TEXT CHECK(sentiment_label IN ('bullish', 'neutral', 'bearish')),
    sentiment_score REAL CHECK(sentiment_score BETWEEN -1.0 AND 1.0),
    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    text TEXT,
    metadata TEXT, -- JSON
    timestamp INTEGER NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_market_sentiment_asset_time ON market_sentiment(asset, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_source ON market_sentiment(source);

-- ============================================================================
-- Backtesting Tables
-- ============================================================================

-- Backtest Runs
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    start_date INTEGER NOT NULL,
    end_date INTEGER NOT NULL,
    initial_capital REAL NOT NULL,
    final_capital REAL NOT NULL,
    total_return REAL NOT NULL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    total_trades INTEGER,
    config TEXT, -- JSON
    results TEXT, -- JSON
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_name, created_at DESC);

-- Backtest Trades
CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    pnl REAL,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(backtest_run_id);

-- ============================================================================
-- System & Monitoring Tables
-- ============================================================================

-- System Events
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    details TEXT, -- JSON
    timestamp INTEGER NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_system_events_time ON system_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events(severity);
CREATE INDEX IF NOT EXISTS idx_system_events_component ON system_events(component);

-- Performance Metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    component TEXT,
    timestamp INTEGER NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_time ON performance_metrics(metric_name, timestamp DESC);

-- Audit Log
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    user_id TEXT,
    changes TEXT, -- JSON
    ip_address TEXT,
    timestamp INTEGER NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_log_time ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_type, entity_id);

-- ============================================================================
-- Configuration Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
) WITHOUT ROWID;

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Active Positions with Current P&L
CREATE VIEW IF NOT EXISTS v_active_positions AS
SELECT 
    p.symbol,
    p.quantity,
    p.average_price,
    p.current_price,
    p.unrealized_pnl,
    p.realized_pnl,
    (p.unrealized_pnl + p.realized_pnl) as total_pnl,
    p.opened_at,
    p.updated_at
FROM positions p
WHERE p.quantity != 0;

-- Recent Trades Summary
CREATE VIEW IF NOT EXISTS v_recent_trades AS
SELECT 
    t.symbol,
    t.side,
    t.quantity,
    t.price,
    t.timestamp,
    t.status,
    t.strategy,
    (t.quantity * t.price) as trade_value
FROM trades t
ORDER BY t.timestamp DESC;

-- Daily Portfolio Performance
CREATE VIEW IF NOT EXISTS v_daily_performance AS
SELECT 
    DATE(timestamp, 'unixepoch') as date,
    MIN(total_value) as day_low,
    MAX(total_value) as day_high,
    AVG(total_value) as day_avg,
    SUM(daily_pnl) as total_daily_pnl
FROM portfolio_snapshots
GROUP BY DATE(timestamp, 'unixepoch')
ORDER BY date DESC;

-- ============================================================================
-- Triggers for Updated Timestamps
-- ============================================================================

CREATE TRIGGER IF NOT EXISTS update_symbols_timestamp 
AFTER UPDATE ON symbols
BEGIN
    UPDATE symbols SET updated_at = strftime('%s', 'now') WHERE symbol = NEW.symbol;
END;

CREATE TRIGGER IF NOT EXISTS update_positions_timestamp 
AFTER UPDATE ON positions
BEGIN
    UPDATE positions SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_risk_limits_timestamp 
AFTER UPDATE ON risk_limits
BEGIN
    UPDATE risk_limits SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_config_timestamp 
AFTER UPDATE ON config
BEGIN
    UPDATE config SET updated_at = strftime('%s', 'now') WHERE key = NEW.key;
END;

-- ============================================================================
-- Initial Configuration
-- ============================================================================

INSERT OR IGNORE INTO config (key, value, description) VALUES
    ('schema_version', '1.0.0', 'Database schema version'),
    ('initialized_at', strftime('%s', 'now'), 'Database initialization timestamp'),
    ('trading_mode', 'paper', 'Trading mode: paper or live'),
    ('max_daily_loss_pct', '0.03', 'Maximum daily loss percentage'),
    ('max_position_size_pct', '0.10', 'Maximum position size as % of portfolio');

-- ============================================================================
-- End of Schema
-- ============================================================================

