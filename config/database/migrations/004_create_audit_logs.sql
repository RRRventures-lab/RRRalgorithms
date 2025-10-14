-- ============================================================================
-- Audit Logs Table
-- ============================================================================
-- Tracks all critical actions in the trading system for compliance and security
-- Created: 2025-10-11
-- ============================================================================

-- Drop table if exists (for development only)
-- DROP TABLE IF EXISTS audit_logs CASCADE;

CREATE TABLE IF NOT EXISTS audit_logs (
    -- Primary Key
    id BIGSERIAL PRIMARY KEY,

    -- Timestamp (indexed for fast time-range queries)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- User/System Identification
    user_id VARCHAR(255),  -- User ID (if authenticated user)
    system_component VARCHAR(100) NOT NULL,  -- e.g., "trading-engine", "risk-manager"
    session_id VARCHAR(255),  -- Session identifier

    -- Action Details
    action_type VARCHAR(50) NOT NULL,  -- e.g., "ORDER_PLACED", "POSITION_CLOSED"
    action_category VARCHAR(50) NOT NULL,  -- e.g., "TRADING", "RISK", "CONFIG"
    severity VARCHAR(20) NOT NULL DEFAULT 'INFO',  -- INFO, WARNING, ERROR, CRITICAL

    -- Resource Information
    resource_type VARCHAR(50),  -- e.g., "order", "position", "api_key"
    resource_id VARCHAR(255),  -- ID of the affected resource
    resource_details JSONB,  -- Additional resource details

    -- Action Context
    action_details JSONB,  -- Detailed information about the action
    previous_value JSONB,  -- Previous state (for updates)
    new_value JSONB,  -- New state (for updates)

    -- Request Context
    ip_address INET,  -- IP address of requester
    user_agent TEXT,  -- User agent string
    api_endpoint VARCHAR(255),  -- API endpoint called (if applicable)

    -- Result
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,  -- Error message if failed
    error_code VARCHAR(50),  -- Error code if failed

    -- Compliance
    compliance_level VARCHAR(50),  -- e.g., "HIGH", "MEDIUM", "LOW"
    requires_review BOOLEAN DEFAULT FALSE,  -- Flag for manual review
    reviewed BOOLEAN DEFAULT FALSE,  -- Has been reviewed
    reviewed_by VARCHAR(255),  -- Who reviewed
    reviewed_at TIMESTAMPTZ,  -- When reviewed

    -- Metadata
    environment VARCHAR(50) NOT NULL DEFAULT 'development',  -- development, staging, production
    version VARCHAR(50),  -- Application version
    correlation_id VARCHAR(255),  -- For tracking related actions

    -- Indexes will be created below
    CONSTRAINT audit_logs_severity_check CHECK (severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    CONSTRAINT audit_logs_environment_check CHECK (environment IN ('development', 'staging', 'production'))
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Time-based queries (most common)
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp
    ON audit_logs (timestamp DESC);

-- Filter by action type
CREATE INDEX IF NOT EXISTS idx_audit_logs_action_type
    ON audit_logs (action_type);

-- Filter by user
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id
    ON audit_logs (user_id)
    WHERE user_id IS NOT NULL;

-- Filter by severity (for alerts)
CREATE INDEX IF NOT EXISTS idx_audit_logs_severity
    ON audit_logs (severity)
    WHERE severity IN ('ERROR', 'CRITICAL');

-- Filter by resource
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource
    ON audit_logs (resource_type, resource_id);

-- Filter by system component
CREATE INDEX IF NOT EXISTS idx_audit_logs_component
    ON audit_logs (system_component);

-- Find items requiring review
CREATE INDEX IF NOT EXISTS idx_audit_logs_review
    ON audit_logs (requires_review, reviewed)
    WHERE requires_review = TRUE;

-- Correlation ID for tracking related actions
CREATE INDEX IF NOT EXISTS idx_audit_logs_correlation
    ON audit_logs (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_audit_logs_time_severity_component
    ON audit_logs (timestamp DESC, severity, system_component);

-- JSONB indexes for searching details
CREATE INDEX IF NOT EXISTS idx_audit_logs_action_details
    ON audit_logs USING GIN (action_details);

CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_details
    ON audit_logs USING GIN (resource_details);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================

-- Enable RLS
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can do everything
CREATE POLICY "Service role has full access"
    ON audit_logs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Policy: Authenticated users can read their own logs
CREATE POLICY "Users can read their own logs"
    ON audit_logs
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid()::text);

-- Policy: Anon users cannot access logs
CREATE POLICY "Anonymous users cannot access logs"
    ON audit_logs
    FOR SELECT
    TO anon
    USING (false);

-- ============================================================================
-- Partitioning (for large scale deployments)
-- ============================================================================
-- Uncomment below to enable partitioning by month
-- This improves query performance and enables easy archival

/*
-- Convert to partitioned table
ALTER TABLE audit_logs RENAME TO audit_logs_old;

CREATE TABLE audit_logs (
    LIKE audit_logs_old INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create partitions for each month
CREATE TABLE audit_logs_2025_10 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE audit_logs_2025_11 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Add more partitions as needed
*/

-- ============================================================================
-- Retention Policy Function
-- ============================================================================
-- Automatically archive old logs to save space

CREATE OR REPLACE FUNCTION archive_old_audit_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete logs older than retention period
    -- Consider moving to archive table instead of deleting
    DELETE FROM audit_logs
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL
    AND severity NOT IN ('ERROR', 'CRITICAL')  -- Keep error logs longer
    AND requires_review = FALSE;  -- Don't delete logs requiring review

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to log an action (can be called from triggers)
CREATE OR REPLACE FUNCTION log_audit_event(
    p_system_component VARCHAR,
    p_action_type VARCHAR,
    p_action_category VARCHAR,
    p_severity VARCHAR DEFAULT 'INFO',
    p_resource_type VARCHAR DEFAULT NULL,
    p_resource_id VARCHAR DEFAULT NULL,
    p_action_details JSONB DEFAULT NULL,
    p_success BOOLEAN DEFAULT TRUE,
    p_user_id VARCHAR DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    new_id BIGINT;
BEGIN
    INSERT INTO audit_logs (
        system_component,
        action_type,
        action_category,
        severity,
        resource_type,
        resource_id,
        action_details,
        success,
        user_id,
        environment
    ) VALUES (
        p_system_component,
        p_action_type,
        p_action_category,
        p_severity,
        p_resource_type,
        p_resource_id,
        p_action_details,
        p_success,
        p_user_id,
        current_setting('app.environment', true)
    )
    RETURNING id INTO new_id;

    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Sample Queries
-- ============================================================================

-- Find all failed actions in last 24 hours
-- SELECT * FROM audit_logs
-- WHERE success = FALSE
-- AND timestamp > NOW() - INTERVAL '24 hours'
-- ORDER BY timestamp DESC;

-- Find all order placements by user
-- SELECT * FROM audit_logs
-- WHERE action_type = 'ORDER_PLACED'
-- AND user_id = 'user-123'
-- ORDER BY timestamp DESC;

-- Find all critical errors
-- SELECT * FROM audit_logs
-- WHERE severity = 'CRITICAL'
-- ORDER BY timestamp DESC
-- LIMIT 100;

-- Get audit summary by action type
-- SELECT action_type, COUNT(*),
--        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
--        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failures
-- FROM audit_logs
-- WHERE timestamp > NOW() - INTERVAL '7 days'
-- GROUP BY action_type
-- ORDER BY COUNT(*) DESC;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE audit_logs IS 'Audit trail for all critical system actions';
COMMENT ON COLUMN audit_logs.timestamp IS 'When the action occurred';
COMMENT ON COLUMN audit_logs.action_type IS 'Type of action (e.g., ORDER_PLACED)';
COMMENT ON COLUMN audit_logs.severity IS 'Severity level: DEBUG, INFO, WARNING, ERROR, CRITICAL';
COMMENT ON COLUMN audit_logs.resource_type IS 'Type of resource affected (order, position, etc.)';
COMMENT ON COLUMN audit_logs.compliance_level IS 'Compliance importance level';
COMMENT ON COLUMN audit_logs.requires_review IS 'Flags actions requiring manual review';

-- ============================================================================
-- Grants
-- ============================================================================

-- Grant appropriate permissions
GRANT SELECT ON audit_logs TO authenticated;
GRANT ALL ON audit_logs TO service_role;

-- ============================================================================
-- Complete
-- ============================================================================

-- Verify table was created
SELECT 'Audit logs table created successfully' AS status;
