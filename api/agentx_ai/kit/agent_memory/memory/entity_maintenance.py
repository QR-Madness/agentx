"""Entity dedupe and merge, inside one transaction the caller commits or rolls back.

``manage.py dedupe_entities`` finds duplicate entities by name (within a channel,
or across one user's channels) and by embedding similarity, then folds each group
into a survivor (``maintenance/entity_merge.py``): the survivor's fields, every
relationship re-homed, the duplicates deleted, parallel relationships collapsed.
Everything runs in one explicit transaction, so a dry run rolls the whole thing back.

The candidate scans span every user unless a user is given, so this class is
reachable only through ``CrossUserStore.entity_maintenance()``: the cross-user store
stays the one way into work across users (Decisions.md ADR-21).

⚠️  Never merge Agent entities (``agent_id`` is durable identity, INV-3). The
candidate scans exclude ``type = 'Agent'``; a caller that builds its own groups
must too.
"""

from typing import Any


class EntityMaintenance:
    """Dedupe scans and the node merge over one open transaction (see the module
    docstring). Obtain it from ``CrossUserStore.entity_maintenance()``."""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    def commit(self) -> None:
        """Make every change in the transaction permanent."""
        self._tx.commit()

    def rollback(self) -> None:
        """Discard every change in the transaction (a dry run ends here)."""
        self._tx.rollback()

    # -- candidate discovery -------------------------------------------------------

    def in_channel_groups(self, user_id: str | None, channel: str | None) -> list[Any]:
        """Non-Agent entities sharing a lowercased name within one user's channel,
        survivor first (highest salience, then access count, then earliest seen):
        ``user_id``, ``channel``, ``key``, ``ids``, ``names``, ``alias_lists``,
        ``descriptions``, ``access_counts``. A None filter spans every user or
        channel."""
        return list(self._tx.run("""
MATCH (e:Entity)
WHERE e.name IS NOT NULL
  AND coalesce(e.type, '') <> 'Agent'
  AND ($user_id IS NULL OR e.user_id = $user_id)
  AND ($channel IS NULL OR e.channel = $channel)
WITH e
ORDER BY coalesce(e.salience, 0.5) DESC,
         coalesce(e.access_count, 0) DESC,
         coalesce(e.first_seen, datetime('1970-01-01T00:00:00Z')) ASC
WITH e.user_id AS user_id,
     e.channel AS channel,
     toLower(e.name) AS key,
     collect(e) AS nodes
WHERE size(nodes) > 1
RETURN user_id, channel, key,
       [n IN nodes | n.id] AS ids,
       [n IN nodes | n.name] AS names,
       [n IN nodes | coalesce(n.aliases, [])] AS alias_lists,
       [n IN nodes | n.description] AS descriptions,
       [n IN nodes | coalesce(n.access_count, 0)] AS access_counts
""", user_id=user_id, channel=channel))

    def cross_channel_groups(self, user_id: str | None) -> list[Any]:
        """Non-Agent entities sharing a lowercased name across one user's channels,
        survivor first (``_global`` first, then as for in-channel groups): ``user_id``,
        ``key``, ``ids``, ``channels``, ``names``, ``alias_lists``, ``descriptions``,
        ``access_counts``. A None user spans every user."""
        return list(self._tx.run("""
MATCH (e:Entity)
WHERE e.name IS NOT NULL
  AND coalesce(e.type, '') <> 'Agent'
  AND ($user_id IS NULL OR e.user_id = $user_id)
WITH e
ORDER BY CASE WHEN e.channel = '_global' THEN 0 ELSE 1 END ASC,
         coalesce(e.salience, 0.5) DESC,
         coalesce(e.access_count, 0) DESC,
         coalesce(e.first_seen, datetime('1970-01-01T00:00:00Z')) ASC
WITH e.user_id AS user_id,
     toLower(e.name) AS key,
     collect(e) AS nodes
WHERE size(nodes) > 1
RETURN user_id, key,
       [n IN nodes | n.id] AS ids,
       [n IN nodes | n.channel] AS channels,
       [n IN nodes | n.name] AS names,
       [n IN nodes | coalesce(n.aliases, [])] AS alias_lists,
       [n IN nodes | n.description] AS descriptions,
       [n IN nodes | coalesce(n.access_count, 0)] AS access_counts
""", user_id=user_id))

    def count_unembedded(self, user_id: str | None) -> int:
        """Named entities without an embedding, which the semantic pass can't see."""
        record = self._tx.run("""
MATCH (e:Entity)
WHERE e.embedding IS NULL AND e.name IS NOT NULL
  AND ($user_id IS NULL OR e.user_id = $user_id)
RETURN count(e) AS n
""", user_id=user_id).single()
        return int(record["n"] or 0) if record else 0

    def embedded_entities(self, user_id: str | None, channel: str | None) -> list[Any]:
        """Embedded, named, non-Agent entities: ``id``, ``name``, ``type``,
        ``user_id``, ``embedding``. A None filter spans every user or channel."""
        return list(self._tx.run("""
MATCH (e:Entity)
WHERE e.embedding IS NOT NULL
  AND e.name IS NOT NULL
  AND coalesce(e.type, '') <> 'Agent'
  AND ($user_id IS NULL OR e.user_id = $user_id)
  AND ($channel IS NULL OR e.channel = $channel)
RETURN e.id AS id, e.name AS name, coalesce(e.type, '') AS type,
       e.user_id AS user_id, e.embedding AS embedding
""", user_id=user_id, channel=channel))

    def semantic_neighbors(self, entity_id: str, embedding: list[float], k: int) -> list[Any]:
        """The `k` nearest entities to `embedding` across every user, excluding the
        entity itself: ``id``, ``name``, ``type``, ``user_id``, ``score``. The caller
        filters by user, type and score."""
        return list(self._tx.run("""
CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
YIELD node AS n, score
WHERE n.id <> $id
RETURN n.id AS id, n.name AS name, coalesce(n.type, '') AS type,
       n.user_id AS user_id, score
""", k=k, embedding=embedding, id=entity_id))

    def group_fields(self, ids: list[str]) -> Any | None:
        """One merge group's fields, survivor first (as for in-channel groups):
        ``ids``, ``names``, ``alias_lists``, ``descriptions``, ``access_counts``."""
        return self._tx.run("""
MATCH (e:Entity) WHERE e.id IN $ids
WITH e
ORDER BY coalesce(e.salience, 0.5) DESC,
         coalesce(e.access_count, 0) DESC,
         coalesce(e.first_seen, datetime('1970-01-01T00:00:00Z')) ASC
RETURN collect(e.id) AS ids,
       collect(e.name) AS names,
       collect(coalesce(e.aliases, [])) AS alias_lists,
       collect(e.description) AS descriptions,
       collect(coalesce(e.access_count, 0)) AS access_counts
""", ids=ids).single()

    # -- the merge -----------------------------------------------------------------

    def apply_survivor_fields(
        self, survivor_id: str, merged_aliases: list[str], merged_description: Any,
        merged_access_count: int,
    ) -> None:
        """Fold the group's aliases, description (kept if the survivor has one) and
        access count into the survivor."""
        self._tx.run(
            """
MATCH (s:Entity {id: $survivor_id})
SET s.aliases = $merged_aliases,
    s.description = coalesce(s.description, $merged_description),
    s.access_count = $merged_access_count,
    s.last_accessed = datetime()
RETURN s.id AS id
""",
            survivor_id=survivor_id,
            merged_aliases=merged_aliases,
            merged_description=merged_description,
            merged_access_count=merged_access_count,
        ).consume()

    def rewrite_and_delete_duplicate(self, survivor_id: str, dup_id: str) -> int:
        """Re-home every relationship of the duplicate onto the survivor, then delete
        the duplicate. Returns how many relationships were rewritten. Uses APOC, a hard
        dependency of the Neo4j memory store (see init_memory_schema)."""
        record = self._tx.run(
            """
MATCH (s:Entity {id: $survivor_id})
MATCH (d:Entity {id: $dup_id})
WITH s, d
CALL {
    WITH s, d
    MATCH (src)-[r]->(d)
    WHERE id(src) <> id(s)
    WITH s, src, r, type(r) AS rtype, properties(r) AS rprops
    CALL apoc.create.relationship(src, rtype, rprops, s) YIELD rel
    DELETE r
    RETURN count(rel) AS in_count
}
CALL {
    WITH s, d
    MATCH (d)-[r]->(tgt)
    WHERE id(tgt) <> id(s)
    WITH s, tgt, r, type(r) AS rtype, properties(r) AS rprops
    CALL apoc.create.relationship(s, rtype, rprops, tgt) YIELD rel
    DELETE r
    RETURN count(rel) AS out_count
}
WITH s, d, in_count, out_count
DETACH DELETE d
RETURN in_count + out_count AS rewritten
""",
            survivor_id=survivor_id,
            dup_id=dup_id,
        ).single()
        return int(record["rewritten"] or 0) if record is not None else 0

    def collapse_parallel_relationships(self, survivor_id: str) -> int:
        """Keep one of each set of parallel relationships on the survivor (same type,
        same other end). Returns how many such sets were collapsed."""
        record = self._tx.run("""
MATCH (s:Entity {id: $survivor_id})
CALL {
    WITH s
    MATCH (src)-[r]->(s)
    WITH src, type(r) AS rtype, collect(r) AS rels
    WHERE size(rels) > 1
    FOREACH (rr IN rels[1..] | DELETE rr)
    RETURN count(*) AS in_groups
}
CALL {
    WITH s
    MATCH (s)-[r]->(tgt)
    WITH tgt, type(r) AS rtype, collect(r) AS rels
    WHERE size(rels) > 1
    FOREACH (rr IN rels[1..] | DELETE rr)
    RETURN count(*) AS out_groups
}
RETURN in_groups + out_groups AS collapsed_groups
""", survivor_id=survivor_id).single()
        return int(record["collapsed_groups"] or 0) if record is not None else 0
