"""True Entity node-merge: fold fields, re-home every relationship, delete dups.

Relocated verbatim from ``management/commands/dedupe_entities.py`` so both the
command and future write-path/janitor mergers share one implementation.
Functions take an explicit ``tx`` — the caller owns transaction/atomicity
(the dedupe command runs one explicit tx per invocation for dry-run rollback;
a future consolidation-time merger can enroll in the pipeline's session).

⚠️  Never merge Agent entities (``agent_id`` is durable identity — INV-3);
candidate *selection* is the caller's job and must exclude ``type = 'Agent'``.
"""

from __future__ import annotations

from typing import Any

# Cypher: fold scalar/list fields into the chosen survivor.
APPLY_SURVIVOR_FIELDS = """
MATCH (s:Entity {id: $survivor_id})
SET s.aliases = $merged_aliases,
    s.description = coalesce(s.description, $merged_description),
    s.access_count = $merged_access_count,
    s.last_accessed = datetime()
RETURN s.id AS id
"""


# Cypher: rewrite every relationship attached to a duplicate to point at the
# survivor, then detach-delete the duplicate. Returns the number of rels
# rewritten so we can report it.
#
# APOC is a hard dependency of the memory system (see init_memory_schema), so
# `apoc.create.relationship` is available.
REWRITE_AND_DELETE_DUP = """
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
"""


# Cypher: collapse parallel relationships on the survivor (same type + same
# other endpoint). Keeps the first occurrence, deletes the rest.
DEDUPE_PARALLEL_RELS = """
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
"""


def merge_alias_set(
    survivor_name: str,
    names: list[str],
    alias_lists: list[list[str]],
) -> list[str]:
    """
    Build the survivor's new aliases list: lowercase-dedup union of every
    duplicate's name + aliases, excluding anything that equals the survivor
    name (case-insensitive). Preserves first-seen casing.
    """
    survivor_key = (survivor_name or "").strip().lower()
    seen: set[str] = {survivor_key} if survivor_key else set()
    merged: list[str] = []

    def _push(value: str | None) -> None:
        if not value:
            return
        cleaned = value.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        merged.append(cleaned)

    # Duplicate names (excluding the survivor at index 0) become aliases.
    for name in names[1:]:
        _push(name)
    for alist in alias_lists:
        for alias in alist:
            _push(alias)
    return merged


def first_non_empty(values: list[Any]) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def merge_entity_group(
    tx,
    *,
    ids: list[str],
    names: list[str],
    alias_lists: list[list[str]],
    descriptions: list[Any],
    access_counts: list[int],
) -> dict[str, int]:
    """Merge a duplicate group into its survivor (``ids[0]``).

    Folds aliases/description/access counts, rewrites every relationship from
    each duplicate onto the survivor, deletes the duplicates, then collapses
    parallel rels. Returns ``{duplicates, rewritten, collapsed}``.
    """
    survivor_id = ids[0]
    dup_ids = ids[1:]
    survivor_name = names[0] if names else ""

    merged_aliases = merge_alias_set(survivor_name, names, alias_lists)
    merged_description = first_non_empty(descriptions)
    merged_access_count = int(sum(access_counts))

    tx.run(
        APPLY_SURVIVOR_FIELDS,
        survivor_id=survivor_id,
        merged_aliases=merged_aliases,
        merged_description=merged_description,
        merged_access_count=merged_access_count,
    ).consume()

    rewritten_total = 0
    for dup_id in dup_ids:
        record = tx.run(
            REWRITE_AND_DELETE_DUP,
            survivor_id=survivor_id,
            dup_id=dup_id,
        ).single()
        if record is not None:
            rewritten_total += int(record["rewritten"] or 0)

    collapsed = 0
    if dup_ids:
        record = tx.run(DEDUPE_PARALLEL_RELS, survivor_id=survivor_id).single()
        if record is not None:
            collapsed = int(record["collapsed_groups"] or 0)

    return {
        "duplicates": len(dup_ids),
        "rewritten": rewritten_total,
        "collapsed": collapsed,
    }
