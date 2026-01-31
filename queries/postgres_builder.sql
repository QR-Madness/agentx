-- init-scripts/01-init.sql
-- This script is idempotent - can be run multiple times safely

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- ============================================
-- SCHEMA VERSION TRACKING
-- ============================================
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert initial version if table is empty
INSERT INTO schema_version (version, description)
SELECT 1, 'Initial schema with channel support and audit logging'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 1);

-- ============================================
-- CONVERSATION LOGS
-- ============================================
-- Conversation logs (append-only time series)
CREATE TABLE IF NOT EXISTS conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    token_count INTEGER,
    model VARCHAR(100),
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- Adjust dimension as needed

    UNIQUE(conversation_id, turn_index)
);

-- BRIN index for time-range queries (very efficient for time-series)
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON conversation_logs USING BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_conversation ON conversation_logs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_logs_channel ON conversation_logs (channel);
CREATE INDEX IF NOT EXISTS idx_logs_embedding ON conversation_logs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================
-- MEMORY TIMELINE
-- ============================================
-- Memory timeline (unified temporal index)
CREATE TABLE IF NOT EXISTS memory_timeline (
    id BIGSERIAL PRIMARY KEY,
    memory_type VARCHAR(50) NOT NULL,  -- 'episode', 'fact', 'entity', 'reflection'
    neo4j_node_id VARCHAR(100),
    event_time TIMESTAMPTZ NOT NULL,
    summary TEXT,
    embedding vector(1536),
    importance_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    archived BOOLEAN DEFAULT FALSE,
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_timeline_time ON memory_timeline USING BRIN (event_time);
CREATE INDEX IF NOT EXISTS idx_timeline_type ON memory_timeline (memory_type);
CREATE INDEX IF NOT EXISTS idx_timeline_importance ON memory_timeline (importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_timeline_channel ON memory_timeline (channel);
CREATE INDEX IF NOT EXISTS idx_timeline_embedding ON memory_timeline USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================
-- TOOL INVOCATIONS
-- ============================================
-- Tool invocations audit
CREATE TABLE IF NOT EXISTS tool_invocations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name VARCHAR(100) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    success BOOLEAN,
    latency_ms INTEGER,
    error_message TEXT,
    channel VARCHAR(100) NOT NULL DEFAULT '_global'
);

CREATE INDEX IF NOT EXISTS idx_tools_conversation ON tool_invocations (conversation_id);
CREATE INDEX IF NOT EXISTS idx_tools_name ON tool_invocations (tool_name);
CREATE INDEX IF NOT EXISTS idx_tools_timestamp ON tool_invocations USING BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_tools_channel ON tool_invocations (channel);

-- ============================================
-- USER PROFILES
-- ============================================
-- User preferences and profiles (user-scoped, no channel needed)
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    expertise_areas JSONB DEFAULT '[]',
    communication_style JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- ============================================
-- MEMORY AUDIT LOG
-- ============================================
-- Audit log for memory operations (query tracing, cross-channel access)
-- Partitioned by day for efficient retention management
CREATE TABLE IF NOT EXISTS memory_audit_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    operation VARCHAR(50) NOT NULL,  -- 'store', 'retrieve', 'update', 'delete', 'promote'
    memory_type VARCHAR(50) NOT NULL,  -- 'episodic', 'semantic', 'procedural', 'working'
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    conversation_id UUID,
    source_channel VARCHAR(100),  -- Channel where operation originated
    target_channels TEXT[],  -- Channels queried/affected (for cross-channel ops)
    query_text TEXT,  -- For retrieval: the query used
    result_count INTEGER,  -- Number of results returned
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    -- Promotion tracking (for cross-channel fact promotion)
    promoted_from_channel VARCHAR(100),
    promotion_confidence FLOAT,
    promotion_access_count INTEGER,
    promotion_conversation_count INTEGER,
    -- Configuration snapshot (active thresholds at time of operation)
    config_snapshot JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for the next 30 days and 7 days of history
-- Additional partitions should be created by a maintenance job
DO $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_bound TEXT;
    end_bound TEXT;
BEGIN
    -- Create partitions from 7 days ago to 30 days ahead
    FOR i IN -7..30 LOOP
        partition_date := CURRENT_DATE + i;
        partition_name := 'memory_audit_log_' || TO_CHAR(partition_date, 'YYYYMMDD');
        start_bound := TO_CHAR(partition_date, 'YYYY-MM-DD');
        end_bound := TO_CHAR(partition_date + 1, 'YYYY-MM-DD');
        
        -- Check if partition exists before creating
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = partition_name AND n.nspname = 'public'
        ) THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF memory_audit_log FOR VALUES FROM (%L) TO (%L)',
                partition_name, start_bound, end_bound
            );
        END IF;
    END LOOP;
END $$;

-- Indexes on audit log (applied to all partitions)
CREATE INDEX IF NOT EXISTS idx_audit_user ON memory_audit_log (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_session ON memory_audit_log (session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON memory_audit_log (operation, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_memory_type ON memory_audit_log (memory_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_source_channel ON memory_audit_log (source_channel, timestamp);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================
-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger only if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'user_profiles_updated'
    ) THEN
        CREATE TRIGGER user_profiles_updated
            BEFORE UPDATE ON user_profiles
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at();
    END IF;
END $$;

-- Function to create audit log partition for a given date
CREATE OR REPLACE FUNCTION create_audit_partition(partition_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_bound TEXT;
    end_bound TEXT;
BEGIN
    partition_name := 'memory_audit_log_' || TO_CHAR(partition_date, 'YYYYMMDD');
    start_bound := TO_CHAR(partition_date, 'YYYY-MM-DD');
    end_bound := TO_CHAR(partition_date + 1, 'YYYY-MM-DD');
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF memory_audit_log FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_bound, end_bound
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to drop old audit log partitions (for retention cleanup)
CREATE OR REPLACE FUNCTION drop_old_audit_partitions(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    partition_record RECORD;
    dropped_count INTEGER := 0;
    cutoff_date DATE;
BEGIN
    cutoff_date := CURRENT_DATE - retention_days;
    
    FOR partition_record IN
        SELECT c.relname AS partition_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_inherits i ON i.inhrelid = c.oid
        JOIN pg_class parent ON parent.oid = i.inhparent
        WHERE parent.relname = 'memory_audit_log'
          AND n.nspname = 'public'
          AND c.relname ~ '^memory_audit_log_[0-9]{8}$'
    LOOP
        -- Extract date from partition name and check if older than cutoff
        IF TO_DATE(SUBSTRING(partition_record.partition_name FROM 18 FOR 8), 'YYYYMMDD') < cutoff_date THEN
            EXECUTE format('DROP TABLE %I', partition_record.partition_name);
            dropped_count := dropped_count + 1;
        END IF;
    END LOOP;
    
    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;