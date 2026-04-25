-- Migration 0002: User storage tables
-- Adds user_files, artifacts, and blob_cache tables.
-- Storage backend is decoupled via storage_backend + storage_path columns
-- so the destination (local/minio/s3) can change without a schema migration.

-- Add filename column to schema_version for migration tracking
ALTER TABLE schema_version ADD COLUMN IF NOT EXISTS filename VARCHAR(255);

-- ============================================
-- USER FILES
-- ============================================
-- Metadata for all user-uploaded files.
-- Actual bytes live at storage_path within the configured storage_backend.
CREATE TABLE IF NOT EXISTS user_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    filename VARCHAR(255) NOT NULL,           -- sanitized stored name
    original_filename VARCHAR(255) NOT NULL,  -- as provided by the user
    mime_type VARCHAR(100),
    size_bytes BIGINT,
    storage_backend VARCHAR(50) NOT NULL DEFAULT 'local',  -- 'local', 'minio', 's3'
    storage_path TEXT NOT NULL,               -- relative path or object key
    storage_bucket VARCHAR(100),              -- bucket name for minio/s3
    checksum_sha256 VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'active',  -- 'active', 'deleted', 'pending'
    conversation_id UUID,                     -- optional: which conversation uploaded this
    embedding vector(1024),                   -- for semantic search over file content
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_files_user ON user_files (user_id);
CREATE INDEX IF NOT EXISTS idx_files_channel ON user_files (user_id, channel);
CREATE INDEX IF NOT EXISTS idx_files_status ON user_files (status);
CREATE INDEX IF NOT EXISTS idx_files_conversation ON user_files (conversation_id);
CREATE INDEX IF NOT EXISTS idx_files_backend ON user_files (storage_backend);
CREATE INDEX IF NOT EXISTS idx_files_created ON user_files USING BRIN (created_at);
CREATE INDEX IF NOT EXISTS idx_files_embedding ON user_files USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'user_files_updated'
    ) THEN
        CREATE TRIGGER user_files_updated
            BEFORE UPDATE ON user_files
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at();
    END IF;
END $$;

-- ============================================
-- ARTIFACTS
-- ============================================
-- Agent-generated outputs (code, reports, exports, etc.) stored per conversation.
-- Small text artifacts can be stored inline; larger ones reference external storage.
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    turn_index INTEGER,
    user_id VARCHAR(100) NOT NULL,
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    name VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(100) NOT NULL,  -- 'code', 'document', 'report', 'data', 'image', 'other'
    mime_type VARCHAR(100),
    size_bytes BIGINT,
    storage_backend VARCHAR(50) NOT NULL DEFAULT 'local',
    storage_path TEXT,                    -- NULL when content is stored inline
    inline_content TEXT,                  -- for small text artifacts (avoids external I/O)
    checksum_sha256 VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding vector(1024),               -- for semantic search over artifact content
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_artifacts_conversation ON artifacts (conversation_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_user ON artifacts (user_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_channel ON artifacts (user_id, channel);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts (artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts USING BRIN (created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_embedding ON artifacts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================
-- BLOB CACHE
-- ============================================
-- Content-addressed cache for derived/processed binary data:
-- thumbnails, processed file chunks, embedding source material, etc.
-- storage_key is a hash-based primary key for deduplication.
-- expires_at = NULL means the entry never expires.
CREATE TABLE IF NOT EXISTS blob_cache (
    storage_key VARCHAR(255) PRIMARY KEY,   -- content-addressed key (hash-based)
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    source_file_id UUID REFERENCES user_files(id) ON DELETE SET NULL,
    cache_type VARCHAR(100) NOT NULL,       -- 'thumbnail', 'chunk', 'processed', 'embedding_source'
    mime_type VARCHAR(100),
    size_bytes BIGINT,
    storage_backend VARCHAR(50) NOT NULL DEFAULT 'local',
    storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,                 -- NULL = no expiry; set for TTL-based cleanup
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_blob_source ON blob_cache (source_file_id);
CREATE INDEX IF NOT EXISTS idx_blob_type ON blob_cache (cache_type);
CREATE INDEX IF NOT EXISTS idx_blob_expires ON blob_cache (expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_blob_created ON blob_cache USING BRIN (created_at);

-- ============================================
-- VERSION BUMP
-- ============================================
INSERT INTO schema_version (version, description, filename)
SELECT 2, 'User storage: user_files, artifacts, blob_cache', '0002_user_storage.sql'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 2);
