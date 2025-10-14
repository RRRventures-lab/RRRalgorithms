-- TimescaleDB Schema for Market Inefficiency Discovery System
-- High-frequency tick data and microstructure metrics
-- Author: RRR Ventures
-- Date: 2025-10-12

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- TICK DATA (Microsecond-level trade data)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tick_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    exchange TEXT,
    conditions JSONB,
    trade_sign INTEGER, -- +1 for buy, -1 for sell
    dollar_volume NUMERIC,
    PRIMARY KEY (timestamp, symbol, exchange)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('tick_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tick_exchange ON tick_data (exchange, timestamp DESC);

-- Compression policy (compress data older than 7 days)
ALTER TABLE tick_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'timestamp DESC',
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('tick_data', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- ORDER BOOK SNAPSHOTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid_price NUMERIC NOT NULL,
    bid_size NUMERIC NOT NULL,
    ask_price NUMERIC NOT NULL,
    ask_size NUMERIC NOT NULL,
    spread NUMERIC,
    spread_bps NUMERIC,
    mid_price NUMERIC,
    
    -- Depth metrics
    bid_depth_5 NUMERIC,  -- Top 5 levels
    ask_depth_5 NUMERIC,
    depth_imbalance NUMERIC,  -- (bid - ask) / (bid + ask)
    
    -- Full order book (JSONB for flexibility)
    bids JSONB,  -- Array of [price, size] tuples
    asks JSONB,
    
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('orderbook_snapshots', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_orderbook_symbol ON orderbook_snapshots (symbol, timestamp DESC);

-- Compression
ALTER TABLE orderbook_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'timestamp DESC',
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('orderbook_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- MICROSTRUCTURE METRICS (Pre-aggregated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS microstructure_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    window_seconds INTEGER NOT NULL,
    
    -- Flow metrics
    buy_volume NUMERIC,
    sell_volume NUMERIC,
    total_volume NUMERIC,
    order_flow_imbalance NUMERIC,
    
    -- Advanced metrics
    vpin NUMERIC,  -- Volume-synchronized Probability of Informed Trading
    kyle_lambda NUMERIC,  -- Price impact coefficient
    amihud_illiquidity NUMERIC,
    
    -- Spread metrics
    effective_spread NUMERIC,
    realized_spread NUMERIC,
    
    -- Trade statistics
    trade_count INTEGER,
    trades_per_second NUMERIC,
    avg_trade_size NUMERIC,
    
    -- Volatility
    price_volatility NUMERIC,
    returns_std NUMERIC,
    
    PRIMARY KEY (timestamp, symbol, window_seconds)
);

SELECT create_hypertable('microstructure_metrics', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_micro_symbol ON microstructure_metrics (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_micro_window ON microstructure_metrics (window_seconds, timestamp DESC);

-- ============================================================================
-- CONTINUOUS AGGREGATES (1-second bars)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1s
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 second', timestamp) AS bucket,
    symbol,
    first(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, timestamp) AS close,
    sum(dollar_volume) AS volume,
    sum(CASE WHEN trade_sign = 1 THEN dollar_volume ELSE 0 END) AS buy_volume,
    sum(CASE WHEN trade_sign = -1 THEN dollar_volume ELSE 0 END) AS sell_volume,
    count(*) AS tick_count
FROM tick_data
GROUP BY bucket, symbol;

-- Refresh policy (every 10 seconds)
SELECT add_continuous_aggregate_policy('bars_1s',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '10 seconds',
    schedule_interval => INTERVAL '10 seconds',
    if_not_exists => TRUE
);

-- ============================================================================
-- CONTINUOUS AGGREGATES (5-second bars)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5s
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 seconds', timestamp) AS bucket,
    symbol,
    first(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, timestamp) AS close,
    sum(dollar_volume) AS volume,
    sum(CASE WHEN trade_sign = 1 THEN dollar_volume ELSE 0 END) AS buy_volume,
    sum(CASE WHEN trade_sign = -1 THEN dollar_volume ELSE 0 END) AS sell_volume,
    count(*) AS tick_count
FROM tick_data
GROUP BY bucket, symbol;

SELECT add_continuous_aggregate_policy('bars_5s',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '30 seconds',
    schedule_interval => INTERVAL '30 seconds',
    if_not_exists => TRUE
);

-- ============================================================================
-- CONTINUOUS AGGREGATES (1-minute bars)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    symbol,
    first(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, timestamp) AS close,
    sum(dollar_volume) AS volume,
    sum(CASE WHEN trade_sign = 1 THEN dollar_volume ELSE 0 END) AS buy_volume,
    sum(CASE WHEN trade_sign = -1 THEN dollar_volume ELSE 0 END) AS sell_volume,
    count(*) AS tick_count,
    stddev(price) AS price_std,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
FROM tick_data
GROUP BY bucket, symbol;

SELECT add_continuous_aggregate_policy('bars_1m',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- ============================================================================
-- SENTIMENT DATA (From Perplexity)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sentiment_scores (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- '1h', '24h', '7d'
    
    score NUMERIC NOT NULL CHECK (score >= -1 AND score <= 1),
    confidence NUMERIC CHECK (confidence >= 0 AND confidence <= 1),
    magnitude NUMERIC CHECK (magnitude >= 0 AND magnitude <= 1),
    
    -- Source breakdown
    news_sentiment NUMERIC,
    social_sentiment NUMERIC,
    expert_sentiment NUMERIC,
    
    -- Metadata
    narrative TEXT,
    key_themes TEXT[],
    sources_count INTEGER,
    
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('sentiment_scores', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_scores (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_score ON sentiment_scores (score, timestamp DESC);

-- ============================================================================
-- NARRATIVE SHIFTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS narrative_shifts (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    old_narrative TEXT,
    new_narrative TEXT,
    shift_magnitude NUMERIC,
    confidence NUMERIC,
    related_events TEXT[],
    price_impact_estimate NUMERIC,
    
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('narrative_shifts', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- INEFFICIENCY SIGNALS
-- ============================================================================

CREATE TABLE IF NOT EXISTS inefficiency_signals (
    signal_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    inefficiency_type TEXT NOT NULL,
    
    -- Assets involved
    symbols TEXT[] NOT NULL,
    exchange TEXT,
    
    -- Signal metrics
    confidence NUMERIC NOT NULL,
    expected_return NUMERIC,
    expected_duration INTEGER,  -- seconds
    
    -- Statistical metrics
    p_value NUMERIC,
    z_score NUMERIC,
    sharpe_ratio NUMERIC,
    
    -- Trade parameters
    direction TEXT CHECK (direction IN ('long', 'short', 'neutral', 'pair')),
    entry_price NUMERIC,
    target_price NUMERIC,
    stop_loss NUMERIC,
    position_size NUMERIC,
    
    -- Context
    market_regime TEXT,
    volatility_regime TEXT,
    description TEXT,
    metadata JSONB,
    
    -- Status tracking
    status TEXT DEFAULT 'detected' CHECK (status IN ('detected', 'validated', 'executed', 'closed', 'expired')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON inefficiency_signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_type ON inefficiency_signals (inefficiency_type);
CREATE INDEX IF NOT EXISTS idx_signals_status ON inefficiency_signals (status);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON inefficiency_signals (confidence DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbols ON inefficiency_signals USING GIN (symbols);

-- ============================================================================
-- BACKTEST RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    inefficiency_type TEXT,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    
    -- Performance metrics
    total_return NUMERIC,
    annualized_return NUMERIC,
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    max_drawdown NUMERIC,
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate NUMERIC,
    avg_win NUMERIC,
    avg_loss NUMERIC,
    profit_factor NUMERIC,
    
    -- Risk metrics
    volatility NUMERIC,
    var_95 NUMERIC,
    cvar_95 NUMERIC,
    
    -- Statistical tests
    t_statistic NUMERIC,
    p_value NUMERIC,
    
    -- Additional metrics
    calmar_ratio NUMERIC,
    omega_ratio NUMERIC,
    
    -- Metadata
    parameters JSONB,
    is_viable BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results (strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_sharpe ON backtest_results (sharpe_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_viable ON backtest_results (is_viable) WHERE is_viable = TRUE;

-- ============================================================================
-- DATA RETENTION POLICIES
-- ============================================================================

-- Drop raw tick data older than 30 days (keep aggregates)
SELECT add_retention_policy('tick_data', INTERVAL '30 days', if_not_exists => TRUE);

-- Drop order book snapshots older than 14 days
SELECT add_retention_policy('orderbook_snapshots', INTERVAL '14 days', if_not_exists => TRUE);

-- Keep microstructure metrics for 60 days
SELECT add_retention_policy('microstructure_metrics', INTERVAL '60 days', if_not_exists => TRUE);

-- Keep sentiment scores for 90 days
SELECT add_retention_policy('sentiment_scores', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get recent order flow imbalance
CREATE OR REPLACE FUNCTION get_recent_ofi(
    p_symbol TEXT,
    p_window_seconds INTEGER DEFAULT 60
)
RETURNS NUMERIC AS $$
    SELECT order_flow_imbalance
    FROM microstructure_metrics
    WHERE symbol = p_symbol
      AND window_seconds = p_window_seconds
    ORDER BY timestamp DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to get VPIN (toxicity indicator)
CREATE OR REPLACE FUNCTION get_recent_vpin(
    p_symbol TEXT,
    p_window_seconds INTEGER DEFAULT 60
)
RETURNS NUMERIC AS $$
    SELECT vpin
    FROM microstructure_metrics
    WHERE symbol = p_symbol
      AND window_seconds = p_window_seconds
    ORDER BY timestamp DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to calculate spread percentile
CREATE OR REPLACE FUNCTION get_spread_percentile(
    p_symbol TEXT,
    p_current_spread NUMERIC,
    p_lookback_hours INTEGER DEFAULT 24
)
RETURNS NUMERIC AS $$
    WITH spreads AS (
        SELECT spread_bps
        FROM orderbook_snapshots
        WHERE symbol = p_symbol
          AND timestamp >= NOW() - (p_lookback_hours || ' hours')::INTERVAL
    )
    SELECT 
        100.0 * COUNT(*) / (SELECT COUNT(*) FROM spreads)
    FROM spreads
    WHERE spread_bps <= p_current_spread;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- GRANT PERMISSIONS (Adjust as needed)
-- ============================================================================

-- GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- ============================================================================
-- VACUUM AND ANALYZE
-- ============================================================================

VACUUM ANALYZE tick_data;
VACUUM ANALYZE orderbook_snapshots;
VACUUM ANALYZE microstructure_metrics;
VACUUM ANALYZE sentiment_scores;
VACUUM ANALYZE inefficiency_signals;
VACUUM ANALYZE backtest_results;

-- End of schema

