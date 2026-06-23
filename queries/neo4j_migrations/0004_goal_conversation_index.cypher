// Migration 0004: Goal.conversation_id index (Phase 16.7 Slice 4)
// Goals are now stamped with the conversation that opened them (GoalMemory.add_goal
// writes g.conversation_id; the value already flowed per-run via the memory facade —
// only the CREATE persisted it before). This index backs
// GoalMemory.get_goals_for_conversation, which the ambassador's survey_conversations
// tool uses to show each conversation's goals. Forward-looking: only goals created
// after this ships carry conversation_id, so there is nothing to backfill.

// ============================================
// SCHEMA VERSION TRACKING
// ============================================
MERGE (m:_SchemaMeta {id: 'schema'})
ON CREATE SET m.version = 0, m.created_at = datetime()
SET m.version = 4, m.updated_at = datetime();

// ============================================
// GOAL CONVERSATION INDEX
// ============================================
CREATE INDEX goal_conversation IF NOT EXISTS
FOR (g:Goal) ON (g.conversation_id);

RETURN 1;
