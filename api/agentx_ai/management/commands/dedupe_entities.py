"""
Django management command to dedupe Entity nodes left over from before the
Phase 18.6 entity-resolution fix landed (commit dcefd7f).

The resolution code in ``consolidation/jobs.py`` keeps the live graph clean
going forward, but every entity written before that point still sits beside
whatever new turns resolve to today. This command brings the existing graph
in line with the write-path semantics:

  1. In-channel pass (always): group :Entity nodes by
     ``(user_id, channel, toLower(name))``; merge duplicates into the
     highest-salience survivor; rewrite incoming/outgoing relationships;
     dedupe parallel rels on the survivor.

  2. Cross-channel pass (``--cross-channel``): same shape but group by
     ``(user_id, toLower(name))``. Survivor selection prefers ``_global``
     over private channels (``_self_*``, ``_alloy_*``), then salience.
     This matches the recall semantics
     ``[active_channel, _self_{agent_id}, _global]``.

Default is ``--dry-run``: the command opens an explicit transaction, performs
the merge, prints the summary, and rolls back. Pass ``--apply`` to commit.

Usage:
    python manage.py dedupe_entities --dry-run
    python manage.py dedupe_entities --apply
    python manage.py dedupe_entities --user-id <uid> --apply
    python manage.py dedupe_entities --cross-channel --apply
"""

from __future__ import annotations

import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandError


logger = logging.getLogger(__name__)


# Cypher: enumerate in-channel duplicate groups.
#
# For each (user_id, channel, lower(name)) group we return the member ids
# ordered survivor-first (salience DESC, access_count DESC, first_seen ASC)
# plus the per-node fields we need to fold into the survivor.
ENUMERATE_IN_CHANNEL = """
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
"""


# Cypher: enumerate cross-channel duplicate groups.
#
# Survivor preference: `_global` first, then by salience. We surface
# channels for the summary report.
ENUMERATE_CROSS_CHANNEL = """
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
"""


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


def _merge_alias_set(
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


def _first_non_empty(values: list[Any]) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


class Command(BaseCommand):
    help = (
        "Dedupe :Entity nodes left over from before the 18.6 resolution fix. "
        "Default is dry-run (rolls back); pass --apply to commit."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Commit the merge. Without this flag the command rolls back.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Explicit dry-run (the default). Overridden by --apply.",
        )
        parser.add_argument(
            "--user-id",
            type=str,
            default=None,
            help="Restrict to a single user_id. Default: all users.",
        )
        parser.add_argument(
            "--channel",
            type=str,
            default=None,
            help="Restrict in-channel pass to one channel. Ignored when --cross-channel is set.",
        )
        parser.add_argument(
            "--cross-channel",
            action="store_true",
            help="Also collapse duplicates across channels (per user). Off by default.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show one line per merged group.",
        )

    def handle(self, *args, **opts):
        apply = bool(opts["apply"])
        user_id = opts.get("user_id")
        channel = opts.get("channel")
        cross_channel = bool(opts["cross_channel"])
        verbose = bool(opts["verbose"])

        mode = "APPLY (committing)" if apply else "DRY-RUN (rolling back)"
        self.stdout.write(f"dedupe_entities: {mode}")
        if user_id:
            self.stdout.write(f"  user_id filter: {user_id}")
        if channel:
            self.stdout.write(f"  channel filter: {channel}")
        if cross_channel:
            self.stdout.write("  cross-channel pass: ENABLED")

        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        except ImportError as e:
            raise CommandError(f"Could not import Neo4jConnection: {e}") from e

        driver = Neo4jConnection.get_driver()

        # Run everything inside a single explicit transaction so dry-run
        # rollback covers both passes atomically.
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                in_channel_summary = self._run_in_channel_pass(
                    tx,
                    user_id=user_id,
                    channel=channel,
                    verbose=verbose,
                )
                cross_summary = None
                if cross_channel:
                    cross_summary = self._run_cross_channel_pass(
                        tx,
                        user_id=user_id,
                        verbose=verbose,
                    )

                if apply:
                    tx.commit()
                    self.stdout.write(self.style.SUCCESS("Committed."))  # type: ignore[attr-defined]
                else:
                    tx.rollback()
                    self.stdout.write(self.style.WARNING("Rolled back (dry-run)."))  # type: ignore[attr-defined]
            except Exception:
                tx.rollback()
                raise

        self._print_summary("In-channel", in_channel_summary)
        if cross_summary is not None:
            self._print_summary("Cross-channel", cross_summary)

    # -- passes --------------------------------------------------------------

    def _run_in_channel_pass(
        self,
        tx,
        user_id: str | None,
        channel: str | None,
        verbose: bool,
    ) -> dict[str, int]:
        groups = list(
            tx.run(ENUMERATE_IN_CHANNEL, user_id=user_id, channel=channel)
        )
        summary = {
            "groups": 0,
            "duplicates": 0,
            "relationships_rewritten": 0,
            "parallel_groups_collapsed": 0,
        }
        for record in groups:
            summary["groups"] += 1
            merged = self._merge_one_group(
                tx,
                ids=record["ids"],
                names=record["names"],
                alias_lists=record["alias_lists"],
                descriptions=record["descriptions"],
                access_counts=record["access_counts"],
            )
            summary["duplicates"] += merged["duplicates"]
            summary["relationships_rewritten"] += merged["rewritten"]
            summary["parallel_groups_collapsed"] += merged["collapsed"]
            if verbose:
                self.stdout.write(
                    f"  in-channel: user={record['user_id']!s} "
                    f"channel={record['channel']!s} key={record['key']!r} "
                    f"merged {merged['duplicates']} dup(s) into {record['ids'][0]}"
                )
        return summary

    def _run_cross_channel_pass(
        self,
        tx,
        user_id: str | None,
        verbose: bool,
    ) -> dict[str, int]:
        groups = list(tx.run(ENUMERATE_CROSS_CHANNEL, user_id=user_id))
        summary = {
            "groups": 0,
            "duplicates": 0,
            "relationships_rewritten": 0,
            "parallel_groups_collapsed": 0,
        }
        for record in groups:
            summary["groups"] += 1
            merged = self._merge_one_group(
                tx,
                ids=record["ids"],
                names=record["names"],
                alias_lists=record["alias_lists"],
                descriptions=record["descriptions"],
                access_counts=record["access_counts"],
            )
            summary["duplicates"] += merged["duplicates"]
            summary["relationships_rewritten"] += merged["rewritten"]
            summary["parallel_groups_collapsed"] += merged["collapsed"]
            if verbose:
                self.stdout.write(
                    f"  cross-channel: user={record['user_id']!s} "
                    f"key={record['key']!r} channels={record['channels']} "
                    f"merged {merged['duplicates']} dup(s) into {record['ids'][0]}"
                )
        return summary

    # -- group merge ---------------------------------------------------------

    def _merge_one_group(
        self,
        tx,
        *,
        ids: list[str],
        names: list[str],
        alias_lists: list[list[str]],
        descriptions: list[Any],
        access_counts: list[int],
    ) -> dict[str, int]:
        survivor_id = ids[0]
        dup_ids = ids[1:]
        survivor_name = names[0] if names else ""

        merged_aliases = _merge_alias_set(survivor_name, names, alias_lists)
        merged_description = _first_non_empty(descriptions)
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

    # -- reporting -----------------------------------------------------------

    def _print_summary(self, label: str, summary: dict[str, int]) -> None:
        self.stdout.write(f"\n{label} pass:")
        self.stdout.write(f"  groups with duplicates: {summary['groups']}")
        self.stdout.write(f"  duplicate nodes merged: {summary['duplicates']}")
        self.stdout.write(
            f"  relationships rewritten: {summary['relationships_rewritten']}"
        )
        self.stdout.write(
            f"  parallel-rel groups collapsed: {summary['parallel_groups_collapsed']}"
        )
