-- =============================================================================
-- RRRalgorithms Initial Database Schema
-- =============================================================================
-- Migration: 001_initial_schema
-- Created: 2025-01-11
-- Description: Initial schema for trading system with all core tables
-- =============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =============================================================================
-- Market Data Tables
-- =============================================================================

-- Crypto Aggregates (OHLCV Bars)
CREATE TABLE IF NOT EXISTS crypto_aggregates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(30, 8) NOT NULL,
    vwap NUMERIC(20, 8),
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_ticker_timestamp UNIQUE (ticker, timestamp),
    CONSTRAINT valid_ohlc CHECK (
        high >= low AND
        high >= open AND
        high >= close AND
        low <= open AND
        low <= close
    )
);

-- Create indexes for common queries
CREATE INDEX idx_crypto_agg_ticker ON crypto_aggregates(ticker);
CREATE INDEX idx_crypto_agg_timestamp ON crypto_aggregates(timestamp DESC);
CREATE INDEX idx_crypto_agg_ticker_time ON crypto_aggregates(ticker, timestamp DESC);

-- Crypto Trades (Tick Data)
CREATE TABLE IF NOT EXISTS crypto_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    size NUMERIC(30, 8) NOT NULL,
    exchange VARCHAR(50),
    conditions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT positive_price CHECK (price > 0),
    CONSTRAINT positive_size CHECK (size > 0)
);

CREATE INDEX idx_trades_ticker ON crypto_trades(ticker);
CREATE INDEX idx_trades_timestamp ON crypto_trades(timestamp DESC);
CREATE INDEX idx_trades_ticker_time ON crypto_trades(ticker, timestamp DESC);

-- Crypto Quotes (Bid/Ask)
CREATE TABLE IF NOT EXISTS crypto_quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bid_price NUMERIC(20, 8) NOT NULL,
    bid_size NUMERIC(30, 8) NOT NULL,
    ask_price NUMERIC(20, 8) NOT NULL,
    ask_size NUMERIC(30, 8) NOT NULL,
    exchange VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT positive_bid CHECK (bid_price > 0),
    CONSTRAINT positive_ask CHECK (ask_price > 0),
    CONSTRAINT positive_sizes CHECK (bid_size > 0 AND ask_size > 0),
    CONSTRAINT valid_spread CHECK (ask_price >= bid_price)
);

CREATE INDEX idx_quotes_ticker ON crypto_quotes(ticker);
CREATE INDEX idx_quotes_timestamp ON crypto_quotes(timestamp DESC);
CREATE INDEX idx_quotes_ticker_time ON crypto_quotes(ticker, timestamp DESC);

-- =============================================================================
-- Analysis & Signals Tables
-- =============================================================================

-- Market Sentiment
CREATE TABLE IF NOT EXISTS market_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset VARCHAR(20) NOT NULL,
    source VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sentiment_label VARCHAR(20) NOT NULL CHECK (sentiment_label IN ('bullish', 'bearish', 'neutral')),
    sentiment_score NUMERIC(5, 4) CHECK (sentiment_score BETWEEN -1 AND 1),
    confidence NUMERIC(5, 4) CHECK (confidence BETWEEN 0 AND 1),
    text TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sentiment_asset ON market_sentiment(asset);
CREATE INDEX idx_sentiment_timestamp ON market_sentiment(timestamp DESC);
CREATE INDEX idx_sentiment_asset_time ON market_sentiment(asset, timestamp DESC);
CREATE INDEX idx_sentiment_source ON market_sentiment(source);

-- Trading Signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    signal_type VARCHAR(50) NOT NULL,
    signal VARCHAR(10) NOT NULL CHECK (signal IN ('buy', 'sell', 'hold')),
    confidence NUMERIC(5, 4) CHECK (confidence BETWEEN 0 AND 1),
    price_target NUMERIC(20, 8),
    stop_loss NUMERIC(20, 8),
    metadata JSONB,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_signals_ticker ON trading_signals(ticker);
CREATE INDEX idx_signals_timestamp ON trading_signals(timestamp DESC);
CREATE INDEX idx_signals_ticker_time ON trading_signals(ticker, timestamp DESC);
CREATE INDEX idx_signals_processed ON trading_signals(processed, timestamp DESC);

-- =============================================================================
-- Trading Tables
-- =============================================================================

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) UNIQUE NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity NUMERIC(30, 8) NOT NULL CHECK (quantity > 0),
    price NUMERIC(20, 8),
    filled_quantity NUMERIC(30, 8) DEFAULT 0 CHECK (filled_quantity >= 0),
    average_fill_price NUMERIC(20, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected')
    ),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_orders_order_id ON orders(order_id);
CREATE INDEX idx_orders_ticker ON orders(ticker);
CREATE INDEX idx_orders_status ON orders(status, created_at DESC);
CREATE INDEX idx_orders_exchange ON orders(exchange);
CREATE INDEX idx_orders_created ON orders(created_at DESC);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    quantity NUMERIC(30, 8) NOT NULL,
    entry_price NUMERIC(20, 8) NOT NULL CHECK (entry_price > 0),
    current_price NUMERIC(20, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    realized_pnl NUMERIC(20, 8) DEFAULT 0,
    unrealized_pnl NUMERIC(20, 8) DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_open_position UNIQUE (ticker, exchange, status) WHERE status = 'open'
);

CREATE INDEX idx_positions_ticker ON positions(ticker);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_open ON positions(status, opened_at DESC) WHERE status = 'open';

-- Portfolio Snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_value NUMERIC(20, 8) NOT NULL,
    cash NUMERIC(20, 8) NOT NULL,
    equity NUMERIC(20, 8) NOT NULL,
    total_pnl NUMERIC(20, 8) NOT NULL,
    daily_pnl NUMERIC(20, 8),
    total_return_pct NUMERIC(8, 4),
    positions_count INTEGER DEFAULT 0,
    open_orders_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_portfolio_timestamp ON portfolio_snapshots(timestamp DESC);

-- =============================================================================
-- System Monitoring Tables
-- =============================================================================

-- System Events Log
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX idx_events_severity ON system_events(severity, timestamp DESC);
CREATE INDEX idx_events_component ON system_events(component, timestamp DESC);

-- API Usage Tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    response_time_ms INTEGER CHECK (response_time_ms >= 0),
    status_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_api_usage_timestamp ON api_usage(timestamp DESC);
CREATE INDEX idx_api_usage_api_name ON api_usage(api_name, timestamp DESC);

-- =============================================================================
-- Functions & Triggers
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to positions table
CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Row Level Security (RLS) Policies
-- =============================================================================
-- Note: These are placeholders. Adjust based on your authentication requirements

-- Enable RLS on sensitive tables
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;

-- Create policies (example - adjust as needed)
-- For now, allow authenticated users to access their own data

-- =============================================================================
-- Initial Data / Seed
-- =============================================================================

-- Insert initial system event
INSERT INTO system_events (event_type, severity, component, message)
VALUES ('system_init', 'info', 'database', 'Initial schema migration completed');

-- =============================================================================
-- Migration Complete
-- =============================================================================
-- Schema version: 001
-- Tables created: 11
-- Indexes created: 30+
-- =============================================================================
