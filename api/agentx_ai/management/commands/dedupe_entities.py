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

  3. Semantic pass (``--semantic``, §2.10 Slice 1): pair entities whose
     EMBEDDINGS score >= the auto threshold (same user_id + type, never
     Agent) — catches lexically disjoint duplicates like "RAG" vs
     "Retrieval-Augmented Generation" that name grouping can never see.
     Requires embeddings: run after the entity_linking job's backfill has
     converged (the pass reports the remaining NULL-embedding count).

Default is ``--dry-run``: the command opens an explicit transaction, performs
the merge, prints the summary, and rolls back. Pass ``--apply`` to commit.

Usage:
    python manage.py dedupe_entities --dry-run
    python manage.py dedupe_entities --apply
    python manage.py dedupe_entities --user-id <uid> --apply
    python manage.py dedupe_entities --cross-channel --apply
    python manage.py dedupe_entities --semantic --dry-run
    python manage.py dedupe_entities --semantic --threshold 0.93 --apply
"""

from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandError

# The merge machinery (survivor-field fold, relationship re-home, parallel-rel
# collapse) lives in the shared maintenance module so future write-path /
# janitor mergers reuse it. This command keeps candidate *discovery* only.
from agentx_ai.kit.agent_memory.maintenance.entity_merge import merge_entity_group


logger = logging.getLogger(__name__)


# Cypher: entities eligible for semantic pairing (embedded, non-Agent, scoped).
ENUMERATE_EMBEDDED = """
MATCH (e:Entity)
WHERE e.embedding IS NOT NULL
  AND e.name IS NOT NULL
  AND coalesce(e.type, '') <> 'Agent'
  AND ($user_id IS NULL OR e.user_id = $user_id)
  AND ($channel IS NULL OR e.channel = $channel)
RETURN e.id AS id, e.name AS name, coalesce(e.type, '') AS type,
       e.user_id AS user_id, e.embedding AS embedding
"""


# Cypher: count entities the semantic pass cannot see (embedding backlog).
COUNT_UNEMBEDDED = """
MATCH (e:Entity)
WHERE e.embedding IS NULL AND e.name IS NOT NULL
  AND ($user_id IS NULL OR e.user_id = $user_id)
RETURN count(e) AS n
"""


# Cypher: vector neighbors of one entity (filtered in Python: same user/type,
# non-Agent, score >= threshold).
SEMANTIC_NEIGHBORS = """
CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
YIELD node AS n, score
WHERE n.id <> $id
RETURN n.id AS id, n.name AS name, coalesce(n.type, '') AS type,
       n.user_id AS user_id, score
"""


# Cypher: per-group field fetch, survivor-first (same ordering as the name passes).
FETCH_GROUP_FIELDS = """
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
"""


def _group_semantic_pairs(pairs: list[tuple[str, str]]) -> list[set[str]]:
    """Union-find over similarity pairs → merge groups (order-independent)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[str, set[str]] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return [g for g in groups.values() if len(g) > 1]


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
            "--semantic",
            action="store_true",
            help="Also pair entities by EMBEDDING similarity (same user+type, "
                 "never Agent) above --threshold. Needs embeddings backfilled.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Similarity threshold for --semantic "
                 "(default: entity_linking_auto_threshold from settings).",
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
        semantic = bool(opts["semantic"])
        verbose = bool(opts["verbose"])
        if opts.get("threshold") is not None:
            threshold = float(opts["threshold"])
        else:
            from agentx_ai.kit.agent_memory.config import get_settings
            threshold = get_settings().entity_linking_auto_threshold

        mode = "APPLY (committing)" if apply else "DRY-RUN (rolling back)"
        self.stdout.write(f"dedupe_entities: {mode}")
        if user_id:
            self.stdout.write(f"  user_id filter: {user_id}")
        if channel:
            self.stdout.write(f"  channel filter: {channel}")
        if cross_channel:
            self.stdout.write("  cross-channel pass: ENABLED")
        if semantic:
            self.stdout.write(f"  semantic pass: ENABLED (threshold {threshold})")

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
                semantic_summary = None
                if semantic:
                    semantic_summary = self._run_semantic_pass(
                        tx,
                        user_id=user_id,
                        channel=channel,
                        threshold=threshold,
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
        if semantic_summary is not None:
            self._print_summary("Semantic", semantic_summary)
            if semantic_summary.get("unembedded"):
                self.stdout.write(self.style.WARNING(  # type: ignore[attr-defined]
                    f"  ⚠ {semantic_summary['unembedded']} entit(ies) still have no "
                    "embedding — the semantic pass cannot see them. The entity_linking "
                    "job backfills them over time; re-run once it converges."
                ))

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

    def _run_semantic_pass(
        self,
        tx,
        user_id: str | None,
        channel: str | None,
        threshold: float,
        verbose: bool,
    ) -> dict[str, int]:
        """Embedding-similarity pairing → union-find groups → shared merge.

        Pairs require: score >= threshold, same user_id, same non-Agent type.
        Neighbors come from the `entity_embeddings` vector index, so only
        embedded entities participate (the summary reports the blind spot).
        """
        unembedded_rec = tx.run(COUNT_UNEMBEDDED, user_id=user_id).single()
        unembedded = int(unembedded_rec["n"] or 0) if unembedded_rec else 0

        entities = list(tx.run(ENUMERATE_EMBEDDED, user_id=user_id, channel=channel))
        pairs: list[tuple[str, str]] = []
        pair_scores: dict[tuple[str, str], float] = {}
        for ent in entities:
            neighbors = tx.run(
                SEMANTIC_NEIGHBORS, k=6, embedding=ent["embedding"], id=ent["id"],
            )
            for nb in neighbors:
                if float(nb["score"] or 0.0) < threshold:
                    continue
                if nb["user_id"] != ent["user_id"]:
                    continue
                if nb["type"] != ent["type"] or nb["type"] == "Agent":
                    continue
                key = tuple(sorted((ent["id"], nb["id"])))
                if key not in pair_scores:
                    pair_scores[key] = float(nb["score"])
                    pairs.append((key[0], key[1]))
                    if verbose:
                        self.stdout.write(
                            f"  semantic pair: {ent['name']!r} ~ {nb['name']!r} "
                            f"score={float(nb['score']):.3f}"
                        )

        summary = {
            "groups": 0,
            "duplicates": 0,
            "relationships_rewritten": 0,
            "parallel_groups_collapsed": 0,
            "unembedded": unembedded,
        }
        for group in _group_semantic_pairs(pairs):
            fields = tx.run(FETCH_GROUP_FIELDS, ids=sorted(group)).single()
            if not fields or len(fields["ids"]) < 2:
                continue
            summary["groups"] += 1
            merged = self._merge_one_group(
                tx,
                ids=fields["ids"],
                names=fields["names"],
                alias_lists=fields["alias_lists"],
                descriptions=fields["descriptions"],
                access_counts=fields["access_counts"],
            )
            summary["duplicates"] += merged["duplicates"]
            summary["relationships_rewritten"] += merged["rewritten"]
            summary["parallel_groups_collapsed"] += merged["collapsed"]
            if verbose:
                self.stdout.write(
                    f"  semantic: merged {merged['duplicates']} dup(s) "
                    f"into {fields['ids'][0]} ({fields['names'][0]!r})"
                )
        return summary

    # -- group merge (delegates to maintenance.entity_merge) ------------------

    def _merge_one_group(self, tx, **group_fields) -> dict[str, int]:
        return merge_entity_group(tx, **group_fields)

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
