-- RRRalgorithms Database Schema
-- PostgreSQL with TimescaleDB extension for time-series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Crypto aggregate bars (OHLCV data)
CREATE TABLE IF NOT EXISTS crypto_aggregates (
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    vwap DECIMAL(20, 8),
    trade_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, timestamp)
);

-- Convert to TimescaleDB hypertable (must be done after table creation)
SELECT create_hypertable(
    'crypto_aggregates',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_crypto_agg_ticker_time
    ON crypto_aggregates (ticker, timestamp DESC);

-- Compression policy (compress data older than 7 days)
SELECT add_compression_policy(
    'crypto_aggregates',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Retention policy (keep data for 2 years)
SELECT add_retention_policy(
    'crypto_aggregates',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Order book metrics (microstructure data)
CREATE TABLE IF NOT EXISTS order_book_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    asset VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) DEFAULT 'binance',
    bid_ask_ratio DECIMAL(10, 4),
    depth_imbalance DECIMAL(10, 4),
    mid_price DECIMAL(20, 8),
    spread_bps DECIMAL(10, 4),
    bid_depth_1pct DECIMAL(20, 8),
    ask_depth_1pct DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, asset, exchange)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'order_book_metrics',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_order_book_asset_time
    ON order_book_metrics (asset, timestamp DESC);

-- Compression policy (compress data older than 7 days)
SELECT add_compression_policy(
    'order_book_metrics',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Retention policy (keep data for 30 days - order book data is short-term)
SELECT add_retention_policy(
    'order_book_metrics',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- Individual trades
CREATE TABLE IF NOT EXISTS crypto_trades (
    id BIGSERIAL,
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(30, 8) NOT NULL,
    exchange VARCHAR(50),
    conditions TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable(
    'crypto_trades',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_crypto_trades_ticker_time
    ON crypto_trades (ticker, timestamp DESC);

-- Compression for old trades
SELECT add_compression_policy(
    'crypto_trades',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- Quotes (bid/ask)
CREATE TABLE IF NOT EXISTS crypto_quotes (
    id BIGSERIAL,
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid_price DECIMAL(20, 8) NOT NULL,
    bid_size DECIMAL(30, 8) NOT NULL,
    ask_price DECIMAL(20, 8) NOT NULL,
    ask_size DECIMAL(30, 8) NOT NULL,
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'crypto_quotes',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_crypto_quotes_ticker_time
    ON crypto_quotes (ticker, timestamp DESC);

-- ============================================================================
-- SENTIMENT DATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_sentiment (
    id BIGSERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(50) NOT NULL, -- 'perplexity', 'twitter', 'reddit'
    sentiment_label VARCHAR(20), -- 'bullish', 'neutral', 'bearish'
    sentiment_score DECIMAL(5, 4), -- -1.0 to 1.0
    confidence DECIMAL(5, 4), -- 0.0 to 1.0
    text TEXT,
    metadata JSONB, -- Additional context
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable(
    'market_sentiment',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_sentiment_asset_time
    ON market_sentiment (asset, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_source
    ON market_sentiment (source);

-- ============================================================================
-- TRADING TABLES
-- ============================================================================

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop'
    quantity DECIMAL(30, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL, -- 'pending', 'filled', 'cancelled'
    filled_quantity DECIMAL(30, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8),
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker
    ON orders (ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_submitted
    ON orders (submitted_at DESC);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    average_entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed'
    UNIQUE (ticker, exchange, status)
);

CREATE INDEX IF NOT EXISTS idx_positions_status
    ON positions (status);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    positions_value DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8),
    total_pnl DECIMAL(20, 8),
    metadata JSONB,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'portfolio_snapshots',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================================
-- STRATEGY AND SIGNALS
-- ============================================================================

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL, -- 'price_prediction', 'sentiment', 'technical'
    signal VARCHAR(10) NOT NULL, -- 'buy', 'sell', 'hold'
    confidence DECIMAL(5, 4), -- 0.0 to 1.0
    price_target DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'trading_signals',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_time
    ON trading_signals (ticker, timestamp DESC);

-- ============================================================================
-- ML MODELS
-- ============================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS ml_models (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'price_prediction', 'sentiment', 'execution'
    architecture TEXT,
    hyperparameters JSONB,
    training_metrics JSONB,
    validation_metrics JSONB,
    deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_name, model_version)
);

-- Model predictions (for monitoring)
CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_id INTEGER REFERENCES ml_models(id),
    ticker VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50),
    predicted_value DECIMAL(20, 8),
    actual_value DECIMAL(20, 8),
    error DECIMAL(20, 8),
    metadata JSONB,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'model_predictions',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================================
-- SYSTEM LOGS AND MONITORING
-- ============================================================================

-- System events
CREATE TABLE IF NOT EXISTS system_events (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    component VARCHAR(50),
    message TEXT,
    metadata JSONB,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'system_events',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON system_events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity
    ON system_events (severity);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    api_name VARCHAR(50) NOT NULL, -- 'polygon', 'perplexity', 'tradingview'
    endpoint VARCHAR(200),
    request_count INTEGER DEFAULT 1,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'api_usage',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================================
-- CONTINUOUS AGGREGATES (for performance)
-- ============================================================================

-- Hourly crypto aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS crypto_aggregates_1h
WITH (timescaledb.continuous) AS
SELECT
    ticker,
    time_bucket('1 hour', timestamp) AS hour,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume,
    AVG(vwap) AS vwap,
    SUM(trade_count) AS trade_count
FROM crypto_aggregates
GROUP BY ticker, hour
WITH NO DATA;

-- Refresh policy
SELECT add_continuous_aggregate_policy(
    'crypto_aggregates_1h',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS crypto_aggregates_1d
WITH (timescaledb.continuous) AS
SELECT
    ticker,
    time_bucket('1 day', timestamp) AS day,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume,
    AVG(vwap) AS vwap,
    SUM(trade_count) AS trade_count
FROM crypto_aggregates
GROUP BY ticker, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'crypto_aggregates_1d',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get latest price
CREATE OR REPLACE FUNCTION get_latest_price(p_ticker VARCHAR)
RETURNS DECIMAL AS $$
    SELECT close
    FROM crypto_aggregates
    WHERE ticker = p_ticker
    ORDER BY timestamp DESC
    LIMIT 1;
$$ LANGUAGE SQL;

-- Function to calculate returns
CREATE OR REPLACE FUNCTION calculate_returns(
    p_ticker VARCHAR,
    p_periods INTEGER DEFAULT 1
)
RETURNS TABLE(
    timestamp TIMESTAMPTZ,
    price DECIMAL,
    returns DECIMAL
) AS $$
    SELECT
        timestamp,
        close AS price,
        (close - LAG(close, p_periods) OVER (ORDER BY timestamp))
            / LAG(close, p_periods) OVER (ORDER BY timestamp) AS returns
    FROM crypto_aggregates
    WHERE ticker = p_ticker
    ORDER BY timestamp DESC;
$$ LANGUAGE SQL;

-- ============================================================================
-- GRANTS (for application user)
-- ============================================================================

-- Grant permissions to trading_user
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- ============================================================================
-- INITIAL DATA / SEED
-- ============================================================================

-- Insert initial ML model entry (placeholder)
INSERT INTO ml_models (
    model_name,
    model_version,
    model_type,
    architecture,
    deployed
) VALUES (
    'price_prediction_transformer',
    'v0.1.0',
    'price_prediction',
    'Transformer with 6 encoder layers',
    FALSE
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE crypto_aggregates IS 'OHLCV bars for cryptocurrencies';
COMMENT ON TABLE crypto_trades IS 'Individual trade executions';
COMMENT ON TABLE crypto_quotes IS 'Bid/ask quotes';
COMMENT ON TABLE market_sentiment IS 'Sentiment analysis from various sources';
COMMENT ON TABLE orders IS 'All trading orders';
COMMENT ON TABLE positions IS 'Current and historical positions';
COMMENT ON TABLE trading_signals IS 'Trading signals from various strategies';
COMMENT ON TABLE ml_models IS 'ML model registry';
COMMENT ON TABLE model_predictions IS 'Model predictions for monitoring';
COMMENT ON TABLE system_events IS 'System events and logs';
COMMENT ON TABLE api_usage IS 'API usage and performance tracking';

-- ============================================================================
-- COMPLETE
-- ============================================================================

-- Refresh continuous aggregates
SELECT refresh_continuous_aggregate('crypto_aggregates_1h', NULL, NULL);
SELECT refresh_continuous_aggregate('crypto_aggregates_1d', NULL, NULL);

SELECT 'Database schema created successfully!' AS status;
