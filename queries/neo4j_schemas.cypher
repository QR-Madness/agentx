// ============================================
// CONSTRAINTS AND INDEXES
// ============================================

// Uniqueness constraints
CREATE CONSTRAINT conversation_id IF NOT EXISTS
FOR (c:Conversation) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT fact_id IF NOT EXISTS
FOR (f:Fact) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT goal_id IF NOT EXISTS
FOR (g:Goal) REQUIRE g.id IS UNIQUE;

CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT turn_id IF NOT EXISTS
FOR (t:Turn) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT strategy_id IF NOT EXISTS
FOR (s:Strategy) REQUIRE s.id IS UNIQUE;

// Property indexes for fast lookups
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX fact_confidence IF NOT EXISTS FOR (f:Fact) ON (f.confidence);
CREATE INDEX goal_status IF NOT EXISTS FOR (g:Goal) ON (g.status);
CREATE INDEX turn_timestamp IF NOT EXISTS FOR (t:Turn) ON (t.timestamp);

// ============================================
// CHANNEL INDEXES (Memory Scoping)
// ============================================
// Channels organize memory into traceable scopes:
// - '_global': User-wide memory (preferences, general facts)
// - '<project-name>': Project-specific memory containers
// Retrieval merges active channel + _global results

CREATE INDEX turn_channel IF NOT EXISTS FOR (t:Turn) ON (t.channel);
CREATE INDEX entity_channel IF NOT EXISTS FOR (e:Entity) ON (e.channel);
CREATE INDEX fact_channel IF NOT EXISTS FOR (f:Fact) ON (f.channel);
CREATE INDEX strategy_channel IF NOT EXISTS FOR (s:Strategy) ON (s.channel);
CREATE INDEX conversation_channel IF NOT EXISTS FOR (c:Conversation) ON (c.channel);
CREATE INDEX goal_channel IF NOT EXISTS FOR (g:Goal) ON (g.channel);

// Composite indexes for common query patterns
CREATE INDEX turn_user_timestamp IF NOT EXISTS FOR (t:Turn) ON (t.user_id, t.timestamp);
CREATE INDEX conversation_user_channel IF NOT EXISTS FOR (c:Conversation) ON (c.user_id, c.channel);
CREATE INDEX entity_user_channel IF NOT EXISTS FOR (e:Entity) ON (e.user_id, e.channel);
CREATE INDEX fact_user_channel IF NOT EXISTS FOR (f:Fact) ON (f.user_id, f.channel);

// Full-text search indexes
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases, e.description];

CREATE FULLTEXT INDEX fact_search IF NOT EXISTS
FOR (f:Fact) ON EACH [f.claim];

// ============================================
// VECTOR INDEXES
// ============================================

// Turn embeddings (episodic memory)
CREATE VECTOR INDEX turn_embeddings IF NOT EXISTS
FOR (t:Turn) ON (t.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Entity embeddings (semantic memory)
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Fact embeddings
CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
FOR (f:Fact) ON (f.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Strategy embeddings (procedural memory)
CREATE VECTOR INDEX strategy_embeddings IF NOT EXISTS
FOR (s:Strategy) ON (s.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// ============================================
// EPISODIC MEMORY SCHEMA
// ============================================

// Example: Create a conversation with turns
// MERGE (c:Conversation {id: $conv_id})
// SET c.started_at = datetime($started_at),
//     c.user_id = $user_id,
//     c.title = $title

// CREATE (t:Turn {
//     id: $turn_id,
//     index: $index,
//     timestamp: datetime(),
//     role: $role,
//     content: $content,
//     embedding: $embedding,
//     token_count: $tokens
// })
// MERGE (c)-[:HAS_TURN]->(t)

// ============================================
// SEMANTIC MEMORY SCHEMA
// ============================================

// Entity types: Person, Organization, Concept, Location, Event, Document, etc.
//
// Entity properties:
//   - id: UUID
//   - name: string
//   - type: string (label also applied as :Person, :Organization, etc.)
//   - aliases: list of strings
//   - description: string
//   - embedding: vector
//   - salience: float (0-1, importance score)
//   - first_seen: datetime
//   - last_accessed: datetime
//   - access_count: integer
//   - properties: map (flexible key-value)

// Relationship types between entities:
//   - RELATED_TO {type, weight, source}
//   - PART_OF
//   - LOCATED_IN
//   - WORKS_FOR
//   - KNOWS
//   - CREATED_BY
//   - REFERENCES
//   - CAUSED
//   - PRECEDES

// ============================================
// PROCEDURAL MEMORY SCHEMA
// ============================================

// Tools
// CREATE (tool:Tool {
//     name: 'web_search',
//     description: 'Search the web for information',
//     schema: $json_schema,
//     avg_latency_ms: 0,
//     success_rate: 1.0,
//     usage_count: 0
// })

// Strategies (learned patterns)
// CREATE (s:Strategy {
//     id: $strategy_id,
//     description: 'Use code interpreter for data analysis',
//     context_pattern: 'data analysis|csv|spreadsheet',
//     embedding: $embedding,
//     success_count: 0,
//     failure_count: 0,
//     last_used: datetime()
// })

// Link strategy to task types
// MERGE (s)-[:EFFECTIVE_FOR]->(:TaskType {name: 'data_analysis'})
// MERGE (s)-[:USES_TOOL]->(tool:Tool {name: 'code_interpreter'})

// ============================================
// GOAL TRACKING SCHEMA
// ============================================

// Goals
// CREATE (g:Goal {
//     id: $goal_id,
//     description: $description,
//     status: 'active',  // 'active', 'completed', 'abandoned', 'blocked'
//     priority: $priority,  // 1-5
//     created_at: datetime(),
//     deadline: datetime($deadline),
//     embedding: $embedding
// })

// Goal hierarchy
// MATCH (parent:Goal {id: $parent_id}), (child:Goal {id: $child_id})
// MERGE (child)-[:SUBGOAL_OF]->(parent)

// Goal dependencies
// MATCH (g1:Goal {id: $goal_id}), (g2:Goal {id: $blocker_id})
// MERGE (g1)-[:BLOCKED_BY]->(g2)

// ============================================
// USER MODEL SCHEMA
// ============================================

// User preferences and profile
// MERGE (u:User {id: $user_id})
// SET u.name = $name,
//     u.created_at = coalesce(u.created_at, datetime()),
//     u.last_active = datetime()

// MERGE (u)-[:HAS_PREFERENCE]->(:Preference {
//     domain: 'communication',
//     key: 'verbosity',
//     value: 'concise'
// })

// MERGE (u)-[:HAS_EXPERTISE {level: 'expert'}]->(:Topic {name: 'python'})
// MERGE (u)-[:INTERESTED_IN]->(:Topic {name: 'machine_learning'})

// ============================================
// REFLECTION / LEARNING SCHEMA
// ============================================

// Reflections on outcomes
// CREATE (r:Reflection {
//     id: $reflection_id,
//     type: 'failure_analysis',  // or 'success_pattern', 'insight'
//     content: $critique_text,
//     embedding: $embedding,
//     created_at: datetime()
// })
// MATCH (c:Conversation {id: $conv_id})
// MERGE (c)-[:HAS_REFLECTION]->(r)

// ============================================
// END OF SCHEMA (DO NOT DELETE)
// ============================================
// The RETURN statement below satisfies parsers that expect a final statement
RETURN 1;
