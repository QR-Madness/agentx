// Migration 0003: AgentParticipant nodes (Phase 16.5)
// Makes conversation participation a first-class graph fact: one
// AgentParticipant per (agent, conversation), linked to its Conversation via
// PARTICIPATED_IN. Maintained live by EpisodicMemory.store_turn going forward;
// this migration declares the schema and backfills from existing turn-level
// attribution (Turn.agent_id, the Phase 16.1 source — captures every agent that
// spoke, not just the conversation's first agent).

// ============================================
// SCHEMA VERSION TRACKING
// ============================================
MERGE (m:_SchemaMeta {id: 'schema'})
ON CREATE SET m.version = 0, m.created_at = datetime()
SET m.version = 3, m.updated_at = datetime();

// ============================================
// AGENT PARTICIPANT NODES
// ============================================
// id is the deterministic "<conversation_id>:<agent_id>" so MERGE is idempotent
// across the live write path and re-runs of this backfill.
CREATE CONSTRAINT agent_participant_id IF NOT EXISTS
FOR (ap:AgentParticipant) REQUIRE ap.id IS UNIQUE;

CREATE INDEX agent_participant_agent IF NOT EXISTS
FOR (ap:AgentParticipant) ON (ap.agent_id);

// ============================================
// BACKFILL FROM EXISTING TURN ATTRIBUTION
// ============================================
MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
WHERE t.agent_id IS NOT NULL
WITH c, t.agent_id AS agent_id
MERGE (ap:AgentParticipant {id: c.id + ':' + agent_id})
ON CREATE SET ap.conversation_id = c.id,
              ap.agent_id = agent_id,
              ap.user_id = c.user_id,
              ap.channel = c.channel,
              ap.first_seen = datetime()
MERGE (ap)-[:PARTICIPATED_IN]->(c);

RETURN 1;
