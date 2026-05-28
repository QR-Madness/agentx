-- Migration 0003: Per-turn agent attribution
-- Records which agent produced each turn so multi-agent (Phase 16) conversations
-- can attribute restored assistant messages to the right specialist. Historical
-- rows stay NULL and fall back to the generic agent name.

ALTER TABLE conversation_logs ADD COLUMN IF NOT EXISTS agent_id VARCHAR(100);

CREATE INDEX IF NOT EXISTS idx_logs_agent ON conversation_logs (agent_id);

-- ============================================
-- VERSION BUMP
-- ============================================
INSERT INTO schema_version (version, description, filename)
SELECT 3, 'Per-turn agent attribution: conversation_logs.agent_id', '0003_turn_agent_id.sql'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 3);
