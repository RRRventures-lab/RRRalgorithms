-- ===========================================================================
-- RRRalgorithms Supabase Schema
-- Adapted for Supabase (PostgreSQL + Real-time + Edge Functions)
-- ===========================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_cron";

-- ===========================================================================
-- MARKET DATA TABLES
-- ===========================================================================

-- Crypto aggregate bars (OHLCV data)
CREATE TABLE IF NOT EXISTS crypto_aggregates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    vwap DECIMAL(20, 8),
    trade_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, event_time)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_crypto_agg_ticker_time
    ON crypto_aggregates (ticker, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_crypto_agg_created
    ON crypto_aggregates (created_at DESC);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE crypto_aggregates;

-- ===========================================================================
-- Individual trades
CREATE TABLE IF NOT EXISTS crypto_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(30, 8) NOT NULL,
    exchange VARCHAR(50),
    conditions TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crypto_trades_ticker_time
    ON crypto_trades (ticker, event_time DESC);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE crypto_trades;

-- ===========================================================================
-- Quotes (bid/ask)
CREATE TABLE IF NOT EXISTS crypto_quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    bid_price DECIMAL(20, 8) NOT NULL,
    bid_size DECIMAL(30, 8) NOT NULL,
    ask_price DECIMAL(20, 8) NOT NULL,
    ask_size DECIMAL(30, 8) NOT NULL,
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crypto_quotes_ticker_time
    ON crypto_quotes (ticker, event_time DESC);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE crypto_quotes;

-- ===========================================================================
-- SENTIMENT DATA
-- ===========================================================================

CREATE TABLE IF NOT EXISTS market_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(50) NOT NULL, -- 'perplexity', 'twitter', 'reddit'
    sentiment_label VARCHAR(20), -- 'bullish', 'neutral', 'bearish'
    sentiment_score DECIMAL(5, 4), -- -1.0 to 1.0
    confidence DECIMAL(5, 4), -- 0.0 to 1.0
    text TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_asset_time
    ON market_sentiment (asset, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_source
    ON market_sentiment (source);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE market_sentiment;

-- ===========================================================================
-- TRADING TABLES
-- ===========================================================================

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) UNIQUE NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity DECIMAL(30, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected')),
    filled_quantity DECIMAL(30, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8),
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders (ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_submitted ON orders (submitted_at DESC);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE orders;

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    average_entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, exchange, status)
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions (ticker);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE positions;

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    positions_value DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8),
    total_pnl DECIMAL(20, 8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_snapshots (event_time DESC);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE portfolio_snapshots;

-- ===========================================================================
-- STRATEGY AND SIGNALS
-- ===========================================================================

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    signal VARCHAR(10) NOT NULL CHECK (signal IN ('buy', 'sell', 'hold')),
    confidence DECIMAL(5, 4),
    price_target DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_time
    ON trading_signals (ticker, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_type
    ON trading_signals (signal_type);

-- Enable real-time
ALTER PUBLICATION supabase_realtime ADD TABLE trading_signals;

-- ===========================================================================
-- ML MODELS
-- ===========================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    architecture TEXT,
    hyperparameters JSONB,
    training_metrics JSONB,
    validation_metrics JSONB,
    deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_name, model_version)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL,
    model_id UUID REFERENCES ml_models(id),
    ticker VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50),
    predicted_value DECIMAL(20, 8),
    actual_value DECIMAL(20, 8),
    error DECIMAL(20, 8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_time
    ON model_predictions (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON model_predictions (model_id);

-- ===========================================================================
-- SYSTEM LOGS AND MONITORING
-- ===========================================================================

-- System events
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(50),
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_type ON system_events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events (severity);
CREATE INDEX IF NOT EXISTS idx_events_time ON system_events (event_time DESC);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200),
    request_count INTEGER DEFAULT 1,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_usage_name_time
    ON api_usage (api_name, event_time DESC);

-- ===========================================================================
-- ROW LEVEL SECURITY (RLS)
-- ===========================================================================

-- Enable RLS on all tables
ALTER TABLE crypto_aggregates ENABLE ROW LEVEL SECURITY;
ALTER TABLE crypto_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE crypto_quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_usage ENABLE ROW LEVEL SECURITY;

-- Create policies (allow all for service role, read-only for anon)
CREATE POLICY "Allow service role all access" ON crypto_aggregates
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Allow anon read" ON crypto_aggregates
    FOR SELECT USING (true);

-- Repeat for all tables (simplified - apply same pattern)
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename IN (
            'crypto_trades', 'crypto_quotes', 'market_sentiment',
            'orders', 'positions', 'portfolio_snapshots', 'trading_signals',
            'ml_models', 'model_predictions', 'system_events', 'api_usage'
        )
    LOOP
        EXECUTE format('CREATE POLICY "Allow service role all" ON %I FOR ALL USING (auth.role() = ''service_role'')', tbl);
        EXECUTE format('CREATE POLICY "Allow anon read" ON %I FOR SELECT USING (true)', tbl);
    END LOOP;
END;
$$;

-- ===========================================================================
-- HELPER FUNCTIONS
-- ===========================================================================

-- Function to get latest price
CREATE OR REPLACE FUNCTION get_latest_price(p_ticker VARCHAR)
RETURNS DECIMAL AS $$
    SELECT close
    FROM crypto_aggregates
    WHERE ticker = p_ticker
    ORDER BY event_time DESC
    LIMIT 1;
$$ LANGUAGE SQL;

-- Function to calculate returns
CREATE OR REPLACE FUNCTION calculate_returns(
    p_ticker VARCHAR,
    p_periods INTEGER DEFAULT 1
)
RETURNS TABLE(
    event_time TIMESTAMPTZ,
    price DECIMAL,
    returns DECIMAL
) AS $$
    SELECT
        event_time,
        close AS price,
        (close - LAG(close, p_periods) OVER (ORDER BY event_time))
            / NULLIF(LAG(close, p_periods) OVER (ORDER BY event_time), 0) AS returns
    FROM crypto_aggregates
    WHERE ticker = p_ticker
    ORDER BY event_time DESC;
$$ LANGUAGE SQL;

-- ===========================================================================
-- INITIAL DATA
-- ===========================================================================

-- Insert placeholder ML model
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

-- ===========================================================================
-- COMMENTS
-- ===========================================================================

COMMENT ON TABLE crypto_aggregates IS 'OHLCV bars for cryptocurrencies (real-time enabled)';
COMMENT ON TABLE crypto_trades IS 'Individual trade executions (real-time enabled)';
COMMENT ON TABLE crypto_quotes IS 'Bid/ask quotes (real-time enabled)';
COMMENT ON TABLE market_sentiment IS 'Sentiment analysis from various sources (real-time enabled)';
COMMENT ON TABLE orders IS 'All trading orders (real-time enabled)';
COMMENT ON TABLE positions IS 'Current and historical positions (real-time enabled)';
COMMENT ON TABLE trading_signals IS 'Trading signals from strategies (real-time enabled)';

SELECT 'Supabase schema created successfully with real-time enabled!' AS status;
