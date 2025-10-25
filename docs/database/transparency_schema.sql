-- ============================================================================
-- RRRalgorithms Transparency Dashboard Database Schema
-- ============================================================================
-- This schema extends the existing database with tables needed for the
-- transparency dashboard. Run this migration after backing up your database.
--
-- Author: RRRVentures
-- Date: 2025-10-25
-- Version: 1.0.0
-- ============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- AI Decision Log
-- ============================================================================
-- Stores all AI predictions and their outcomes for transparency and analysis

CREATE TABLE IF NOT EXISTS ai_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,

    -- Prediction details
    prediction JSONB NOT NULL,  -- {direction, confidence, price_target, time_horizon}
    features JSONB NOT NULL,     -- Input features used {rsi, macd, volume_ratio, etc.}
    reasoning TEXT,              -- Human-readable explanation

    -- Outcome tracking
    outcome TEXT,                -- 'profitable', 'loss', 'pending', 'cancelled'
    actual_return NUMERIC,       -- Actual return if prediction was acted upon
    prediction_error NUMERIC,    -- Error between predicted and actual

    -- Metadata
    confidence_score NUMERIC CHECK (confidence_score >= 0 AND confidence_score <= 1),
    time_horizon TEXT,           -- '1h', '4h', '1d', etc.
    strategy_name TEXT,

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_ai_decisions_timestamp ON ai_decisions(timestamp DESC);
CREATE INDEX idx_ai_decisions_symbol ON ai_decisions(symbol);
CREATE INDEX idx_ai_decisions_model ON ai_decisions(model_name);
CREATE INDEX idx_ai_decisions_outcome ON ai_decisions(outcome);
CREATE INDEX idx_ai_decisions_created_at ON ai_decisions(created_at DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_ai_decisions_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ai_decisions_update_timestamp
    BEFORE UPDATE ON ai_decisions
    FOR EACH ROW
    EXECUTE FUNCTION update_ai_decisions_timestamp();


-- ============================================================================
-- Live Trading Feed
-- ============================================================================
-- Real-time feed of all trading events for the live dashboard

CREATE TABLE IF NOT EXISTS trade_feed (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,
    event_type TEXT NOT NULL,  -- 'signal', 'order_placed', 'order_filled', 'position_closed'
    symbol TEXT NOT NULL,

    -- Event data
    data JSONB NOT NULL,  -- All event-specific data

    -- Privacy
    visibility TEXT DEFAULT 'public' CHECK (visibility IN ('public', 'private')),

    -- Metadata
    user_id TEXT,  -- For multi-user support (future)
    strategy_name TEXT,

    created_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_trade_feed_timestamp ON trade_feed(timestamp DESC);
CREATE INDEX idx_trade_feed_symbol ON trade_feed(symbol);
CREATE INDEX idx_trade_feed_event_type ON trade_feed(event_type);
CREATE INDEX idx_trade_feed_visibility ON trade_feed(visibility);
CREATE INDEX idx_trade_feed_created_at ON trade_feed(created_at DESC);

-- Retention policy: Keep only last 30 days in trade_feed (archive older)
CREATE OR REPLACE FUNCTION cleanup_old_trade_feed()
RETURNS void AS $$
BEGIN
    DELETE FROM trade_feed
    WHERE created_at < now() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup (run daily)
-- Note: Use pg_cron or external scheduler to run this


-- ============================================================================
-- Performance Snapshots
-- ============================================================================
-- Regular snapshots of portfolio performance for charting

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,

    -- Portfolio values
    portfolio_value NUMERIC NOT NULL,
    cash NUMERIC NOT NULL,
    positions_value NUMERIC NOT NULL,

    -- Returns
    daily_return NUMERIC,
    total_return NUMERIC,

    -- Risk metrics
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    max_drawdown NUMERIC,
    current_drawdown NUMERIC,

    -- Trading stats
    win_rate NUMERIC,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,

    -- Additional metrics
    metrics JSONB,  -- {volatility, beta, alpha, etc.}

    created_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_performance_snapshots_timestamp ON performance_snapshots(timestamp DESC);
CREATE INDEX idx_performance_snapshots_created_at ON performance_snapshots(created_at DESC);

-- Unique constraint on timestamp (one snapshot per timestamp)
CREATE UNIQUE INDEX idx_performance_snapshots_unique_timestamp ON performance_snapshots(timestamp);


-- ============================================================================
-- Strategy Performance
-- ============================================================================
-- Aggregated performance metrics by strategy

CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- '1d', '1w', '1m', '3m', '1y', 'all'

    -- Performance metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate NUMERIC,

    -- Returns
    total_return NUMERIC,
    avg_return NUMERIC,
    best_trade NUMERIC,
    worst_trade NUMERIC,

    -- Risk metrics
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    calmar_ratio NUMERIC,
    max_drawdown NUMERIC,

    -- Time metrics
    avg_trade_duration INTERVAL,
    avg_winning_duration INTERVAL,
    avg_losing_duration INTERVAL,

    -- Additional metrics
    metrics JSONB,

    updated_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_strategy_performance_name ON strategy_performance(strategy_name);
CREATE INDEX idx_strategy_performance_timeframe ON strategy_performance(timeframe);
CREATE INDEX idx_strategy_performance_updated_at ON strategy_performance(updated_at DESC);

-- Unique constraint on strategy_name + timeframe
CREATE UNIQUE INDEX idx_strategy_performance_unique ON strategy_performance(strategy_name, timeframe);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_strategy_performance_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER strategy_performance_update_timestamp
    BEFORE UPDATE ON strategy_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_strategy_performance_timestamp();


-- ============================================================================
-- Backtest Results
-- ============================================================================
-- Complete backtest results with equity curves and all trades

CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name TEXT NOT NULL,
    backtest_id TEXT NOT NULL UNIQUE,

    -- Time period
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,

    -- Capital
    initial_capital NUMERIC NOT NULL,
    final_capital NUMERIC NOT NULL,

    -- Returns
    total_return NUMERIC,
    annualized_return NUMERIC,

    -- Risk metrics
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    calmar_ratio NUMERIC,
    max_drawdown NUMERIC,
    avg_drawdown NUMERIC,

    -- Trading statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate NUMERIC,
    avg_win NUMERIC,
    avg_loss NUMERIC,
    win_loss_ratio NUMERIC,
    profit_factor NUMERIC,

    -- Time metrics
    avg_trade_duration INTERVAL,
    max_trade_duration INTERVAL,

    -- Data
    equity_curve JSONB,  -- Time series of portfolio value
    trades JSONB,         -- All backtest trades
    metrics JSONB,        -- Additional metrics
    parameters JSONB,     -- Backtest parameters used

    -- Metadata
    status TEXT DEFAULT 'completed' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    error_message TEXT,

    created_at TIMESTAMP DEFAULT now(),
    completed_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_backtest_results_strategy_name ON backtest_results(strategy_name);
CREATE INDEX idx_backtest_results_start_date ON backtest_results(start_date);
CREATE INDEX idx_backtest_results_end_date ON backtest_results(end_date);
CREATE INDEX idx_backtest_results_created_at ON backtest_results(created_at DESC);
CREATE INDEX idx_backtest_results_status ON backtest_results(status);


-- ============================================================================
-- Trade Attribution
-- ============================================================================
-- Links trades to AI decisions for performance attribution

CREATE TABLE IF NOT EXISTS trade_attribution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID NOT NULL,
    ai_decision_id UUID REFERENCES ai_decisions(id),

    -- Attribution details
    prediction_accuracy NUMERIC,  -- How accurate was the prediction
    followed_prediction BOOLEAN,  -- Did we follow the AI recommendation
    override_reason TEXT,          -- If not followed, why?

    created_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_trade_attribution_trade_id ON trade_attribution(trade_id);
CREATE INDEX idx_trade_attribution_ai_decision_id ON trade_attribution(ai_decision_id);


-- ============================================================================
-- Feature Importance
-- ============================================================================
-- Track feature importance over time for different models

CREATE TABLE IF NOT EXISTS feature_importance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    importance_score NUMERIC NOT NULL,

    -- Time period
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,

    -- Additional metadata
    sample_size INTEGER,
    metadata JSONB,

    created_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_feature_importance_model ON feature_importance(model_name);
CREATE INDEX idx_feature_importance_feature ON feature_importance(feature_name);
CREATE INDEX idx_feature_importance_period ON feature_importance(period_start, period_end);


-- ============================================================================
-- Dashboard User Settings (for future multi-user support)
-- ============================================================================

CREATE TABLE IF NOT EXISTS dashboard_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL UNIQUE,

    -- Display preferences
    theme TEXT DEFAULT 'dark' CHECK (theme IN ('dark', 'light')),
    default_timeframe TEXT DEFAULT '1d',
    default_view TEXT DEFAULT 'dashboard',

    -- Privacy settings
    public_profile BOOLEAN DEFAULT false,
    show_trades BOOLEAN DEFAULT true,
    show_performance BOOLEAN DEFAULT true,

    -- Notification preferences
    email_notifications BOOLEAN DEFAULT true,
    trade_notifications BOOLEAN DEFAULT true,
    performance_notifications BOOLEAN DEFAULT false,

    -- Settings JSON
    settings JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_dashboard_settings_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER dashboard_settings_update_timestamp
    BEFORE UPDATE ON dashboard_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_dashboard_settings_timestamp();


-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Recent AI predictions with accuracy
CREATE OR REPLACE VIEW recent_ai_predictions AS
SELECT
    id,
    timestamp,
    symbol,
    model_name,
    prediction->>'direction' as predicted_direction,
    (prediction->>'confidence')::numeric as confidence,
    outcome,
    actual_return,
    prediction_error,
    reasoning
FROM ai_decisions
ORDER BY timestamp DESC
LIMIT 100;


-- Portfolio performance summary
CREATE OR REPLACE VIEW portfolio_performance_summary AS
SELECT
    DATE_TRUNC('day', timestamp) as date,
    AVG(portfolio_value) as avg_portfolio_value,
    MAX(portfolio_value) as max_portfolio_value,
    MIN(portfolio_value) as min_portfolio_value,
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    AVG(win_rate) as avg_win_rate,
    SUM(total_trades) as total_trades
FROM performance_snapshots
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY date DESC;


-- Strategy comparison
CREATE OR REPLACE VIEW strategy_comparison AS
SELECT
    strategy_name,
    total_trades,
    win_rate,
    total_return,
    sharpe_ratio,
    max_drawdown,
    RANK() OVER (ORDER BY sharpe_ratio DESC) as sharpe_rank,
    RANK() OVER (ORDER BY total_return DESC) as return_rank
FROM strategy_performance
WHERE timeframe = 'all'
ORDER BY sharpe_ratio DESC;


-- ============================================================================
-- Sample Data for Testing (Optional)
-- ============================================================================

-- Insert sample AI decision
INSERT INTO ai_decisions (timestamp, symbol, model_name, prediction, features, reasoning, confidence_score)
VALUES (
    now(),
    'BTC-USD',
    'transformer_v2',
    '{"direction": "up", "confidence": 0.85, "price_target": 51000, "time_horizon": "2h"}',
    '{"rsi": 65, "macd": 120, "volume_ratio": 1.8, "price_change_1h": 2.3}',
    'Strong bullish momentum with increasing volume. MACD crossover detected.',
    0.85
);

-- Insert sample performance snapshot
INSERT INTO performance_snapshots (timestamp, portfolio_value, cash, positions_value, daily_return, total_return, sharpe_ratio, win_rate, total_trades)
VALUES (
    now(),
    105234.50,
    21046.90,
    84187.60,
    0.0119,
    0.0523,
    1.85,
    0.68,
    142
);


-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to calculate AI model accuracy
CREATE OR REPLACE FUNCTION calculate_model_accuracy(
    p_model_name TEXT,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_predictions INTEGER,
    correct_predictions INTEGER,
    accuracy NUMERIC,
    avg_confidence NUMERIC,
    avg_error NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_predictions,
        COUNT(*) FILTER (WHERE outcome = 'profitable')::INTEGER as correct_predictions,
        (COUNT(*) FILTER (WHERE outcome = 'profitable')::NUMERIC / COUNT(*)) as accuracy,
        AVG(confidence_score) as avg_confidence,
        AVG(ABS(prediction_error)) as avg_error
    FROM ai_decisions
    WHERE model_name = p_model_name
        AND timestamp > now() - (p_days || ' days')::INTERVAL
        AND outcome IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Example usage:
-- SELECT * FROM calculate_model_accuracy('transformer_v2', 30);


-- Function to update strategy performance
CREATE OR REPLACE FUNCTION update_strategy_performance(
    p_strategy_name TEXT,
    p_timeframe TEXT
)
RETURNS void AS $$
DECLARE
    v_period_start TIMESTAMP;
    v_period_end TIMESTAMP;
BEGIN
    -- Determine time period based on timeframe
    v_period_end := now();

    CASE p_timeframe
        WHEN '1d' THEN v_period_start := now() - INTERVAL '1 day';
        WHEN '1w' THEN v_period_start := now() - INTERVAL '1 week';
        WHEN '1m' THEN v_period_start := now() - INTERVAL '1 month';
        WHEN '3m' THEN v_period_start := now() - INTERVAL '3 months';
        WHEN '1y' THEN v_period_start := now() - INTERVAL '1 year';
        WHEN 'all' THEN v_period_start := '1970-01-01'::TIMESTAMP;
        ELSE RAISE EXCEPTION 'Invalid timeframe: %', p_timeframe;
    END CASE;

    -- Update strategy performance
    -- Note: This is a template. Adjust based on your actual trades table structure
    INSERT INTO strategy_performance (
        strategy_name,
        timeframe,
        total_trades,
        winning_trades,
        losing_trades,
        win_rate,
        total_return
    )
    SELECT
        p_strategy_name,
        p_timeframe,
        COUNT(*) as total_trades,
        COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
        COUNT(*) FILTER (WHERE pnl < 0) as losing_trades,
        COUNT(*) FILTER (WHERE pnl > 0)::NUMERIC / COUNT(*) as win_rate,
        SUM(pnl) / 100000 as total_return  -- Assuming $100k starting capital
    FROM trades  -- Adjust table name as needed
    WHERE strategy = p_strategy_name
        AND timestamp BETWEEN v_period_start AND v_period_end
        AND status = 'closed'
    ON CONFLICT (strategy_name, timeframe) DO UPDATE SET
        total_trades = EXCLUDED.total_trades,
        winning_trades = EXCLUDED.winning_trades,
        losing_trades = EXCLUDED.losing_trades,
        win_rate = EXCLUDED.win_rate,
        total_return = EXCLUDED.total_return,
        updated_at = now();
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- Grants (adjust based on your user setup)
-- ============================================================================

-- Grant permissions to your application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;


-- ============================================================================
-- Migration Complete
-- ============================================================================

-- Verify tables created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
    AND table_name IN (
        'ai_decisions',
        'trade_feed',
        'performance_snapshots',
        'strategy_performance',
        'backtest_results',
        'trade_attribution',
        'feature_importance',
        'dashboard_settings'
    )
ORDER BY table_name;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Transparency dashboard schema migration completed successfully!';
    RAISE NOTICE 'Tables created: ai_decisions, trade_feed, performance_snapshots, strategy_performance, backtest_results';
    RAISE NOTICE 'Run sample queries to verify: SELECT * FROM recent_ai_predictions LIMIT 5;';
END $$;
