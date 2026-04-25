// Migration 0002: Document and Artifact node types
// Adds Document nodes (uploaded files) and Artifact nodes (agent-generated outputs)
// with appropriate constraints, indexes, and vector search support.

// ============================================
// SCHEMA VERSION TRACKING
// ============================================
// _SchemaMeta is a singleton node tracking the applied Neo4j migration version.
MERGE (m:_SchemaMeta {id: 'schema'})
ON CREATE SET m.version = 0, m.created_at = datetime()
SET m.version = 2, m.updated_at = datetime();

// ============================================
// DOCUMENT NODES (uploaded files)
// ============================================
// Document mirrors user_files in PostgreSQL at the graph layer.
// Enables fact/entity extraction lineage back to source files.
//
// Properties:
//   id, filename, original_filename, mime_type, size_bytes,
//   storage_backend, storage_path, description,
//   embedding (vector), created_at, status, user_id, channel

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE INDEX document_user_channel IF NOT EXISTS
FOR (d:Document) ON (d.user_id, d.channel);

CREATE INDEX document_status IF NOT EXISTS
FOR (d:Document) ON (d.status);

CREATE FULLTEXT INDEX document_search IF NOT EXISTS
FOR (d:Document) ON EACH [d.filename, d.description];

CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:Document) ON (d.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
    }
};

// ============================================
// ARTIFACT NODES (agent-generated outputs)
// ============================================
// Artifact mirrors the artifacts table in PostgreSQL at the graph layer.
// Linked to Conversation nodes via PRODUCED relationship.
//
// Properties:
//   id, name, artifact_type, mime_type, size_bytes,
//   storage_backend, storage_path, created_at, user_id, channel

CREATE CONSTRAINT artifact_id IF NOT EXISTS
FOR (a:Artifact) REQUIRE a.id IS UNIQUE;

CREATE INDEX artifact_conversation IF NOT EXISTS
FOR (a:Artifact) ON (a.conversation_id);

CREATE INDEX artifact_type IF NOT EXISTS
FOR (a:Artifact) ON (a.artifact_type);

CREATE INDEX artifact_user_channel IF NOT EXISTS
FOR (a:Artifact) ON (a.user_id, a.channel);

// ============================================
// RELATIONSHIP DOCUMENTATION
// ============================================
// The following relationships are created at runtime by the application:
//
//   (User)-[:HAS_DOCUMENT]->(Document)
//     Ownership: each uploaded file is owned by a user.
//
//   (Conversation)-[:PRODUCED]->(Artifact)
//     Provenance: each artifact was produced in a specific conversation.
//
//   (Fact)-[:DERIVED_FROM_FILE]->(Document)
//   (Entity)-[:DERIVED_FROM_FILE]->(Document)
//     Extraction lineage: facts/entities extracted from an uploaded document.

RETURN 1;
