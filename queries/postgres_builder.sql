-- init-scripts/01-init.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Conversation logs (append-only time series)
CREATE TABLE conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    token_count INTEGER,
    model VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- Adjust dimension as needed

    UNIQUE(conversation_id, turn_index)
);

-- BRIN index for time-range queries (very efficient for time-series)
CREATE INDEX idx_logs_timestamp ON conversation_logs USING BRIN (timestamp);
CREATE INDEX idx_logs_conversation ON conversation_logs (conversation_id);
CREATE INDEX idx_logs_embedding ON conversation_logs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Memory timeline (unified temporal index)
CREATE TABLE memory_timeline (
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
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_timeline_time ON memory_timeline USING BRIN (event_time);
CREATE INDEX idx_timeline_type ON memory_timeline (memory_type);
CREATE INDEX idx_timeline_importance ON memory_timeline (importance_score DESC);
CREATE INDEX idx_timeline_embedding ON memory_timeline USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Tool invocations audit
CREATE TABLE tool_invocations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name VARCHAR(100) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    success BOOLEAN,
    latency_ms INTEGER,
    error_message TEXT
);

CREATE INDEX idx_tools_conversation ON tool_invocations (conversation_id);
CREATE INDEX idx_tools_name ON tool_invocations (tool_name);
CREATE INDEX idx_tools_timestamp ON tool_invocations USING BRIN (timestamp);

-- User preferences and profiles
CREATE TABLE user_profiles (
    user_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    expertise_areas JSONB DEFAULT '[]',
    communication_style JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_profiles_updated
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();