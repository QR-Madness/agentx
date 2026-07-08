// Migration 0005: turn-level consolidation tracking (idempotency fix)
//
// Consolidation now marks each Turn (t.consolidated / t.self_consolidated) and
// discovers only turns not yet marked, instead of re-qualifying whole
// conversations by a 15-minute window (which re-poured every turn every sweep).
//
// Backfill existing turns from the conversation-level c.consolidated /
// c.self_consolidated so deploying this does NOT trigger one mass
// re-consolidation; index the turn markers so the `IS NULL` discovery scan stays
// fast. Idempotent: `IF NOT EXISTS` on indexes, `IS NULL` guards on the backfill,
// batched via CALL { … } IN TRANSACTIONS. Version bump is LAST so a failed
// backfill leaves the version un-bumped and the migration retries.

// ============================================
// INDEXES (fast `t.consolidated IS NULL` discovery)
// ============================================
CREATE INDEX turn_consolidated IF NOT EXISTS FOR (t:Turn) ON (t.consolidated);

CREATE INDEX turn_self_consolidated IF NOT EXISTS FOR (t:Turn) ON (t.self_consolidated);

// ============================================
// BACKFILL — baseline "everything up to now is consolidated"
// ============================================
MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
WHERE c.consolidated IS NOT NULL AND t.role = 'user' AND t.consolidated IS NULL
CALL { WITH c, t SET t.consolidated = c.consolidated } IN TRANSACTIONS OF 10000 ROWS;

MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
WHERE c.self_consolidated IS NOT NULL AND t.role = 'assistant' AND t.self_consolidated IS NULL
CALL { WITH c, t SET t.self_consolidated = c.self_consolidated } IN TRANSACTIONS OF 10000 ROWS;

// ============================================
// SCHEMA VERSION TRACKING (last — retry-safe for the backfill above)
// ============================================
MERGE (m:_SchemaMeta {id: 'schema'})
ON CREATE SET m.version = 0, m.created_at = datetime()
SET m.version = 5, m.updated_at = datetime();

RETURN 1;
