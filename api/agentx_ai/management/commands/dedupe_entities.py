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
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

# The merge machinery (survivor-field fold, relationship re-home, parallel-rel
# collapse) lives in the shared maintenance module so future write-path /
# janitor mergers reuse it. This command keeps candidate *discovery* only; every
# query runs through the store's EntityMaintenance transaction.
from agentx_ai.kit.agent_memory.maintenance.entity_merge import merge_entity_group

if TYPE_CHECKING:
    from agentx_ai.kit.agent_memory.memory.entity_maintenance import EntityMaintenance


logger = logging.getLogger(__name__)


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
            from agentx_ai.kit.agent_memory.memory.cross_user import CrossUserStore
        except ImportError as e:
            raise CommandError(f"Could not import the memory store: {e}") from e

        # Run everything inside a single explicit transaction so dry-run
        # rollback covers both passes atomically. Leaving the block without a
        # commit (an exception included) rolls it back.
        with CrossUserStore().entity_maintenance() as store:
            in_channel_summary = self._run_in_channel_pass(
                store,
                user_id=user_id,
                channel=channel,
                verbose=verbose,
            )
            cross_summary = None
            if cross_channel:
                cross_summary = self._run_cross_channel_pass(
                    store,
                    user_id=user_id,
                    verbose=verbose,
                )
            semantic_summary = None
            if semantic:
                semantic_summary = self._run_semantic_pass(
                    store,
                    user_id=user_id,
                    channel=channel,
                    threshold=threshold,
                    verbose=verbose,
                )

            if apply:
                store.commit()
                self.stdout.write(self.style.SUCCESS("Committed."))  # type: ignore[attr-defined]
            else:
                store.rollback()
                self.stdout.write(self.style.WARNING("Rolled back (dry-run)."))  # type: ignore[attr-defined]

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
        store: EntityMaintenance,
        user_id: str | None,
        channel: str | None,
        verbose: bool,
    ) -> dict[str, int]:
        groups = store.in_channel_groups(user_id, channel)
        summary = {
            "groups": 0,
            "duplicates": 0,
            "relationships_rewritten": 0,
            "parallel_groups_collapsed": 0,
        }
        for record in groups:
            summary["groups"] += 1
            merged = self._merge_one_group(
                store,
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
        store: EntityMaintenance,
        user_id: str | None,
        verbose: bool,
    ) -> dict[str, int]:
        groups = store.cross_channel_groups(user_id)
        summary = {
            "groups": 0,
            "duplicates": 0,
            "relationships_rewritten": 0,
            "parallel_groups_collapsed": 0,
        }
        for record in groups:
            summary["groups"] += 1
            merged = self._merge_one_group(
                store,
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
        store: EntityMaintenance,
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
        unembedded = store.count_unembedded(user_id)

        entities = store.embedded_entities(user_id, channel)
        pairs: list[tuple[str, str]] = []
        pair_scores: dict[tuple[str, str], float] = {}
        for ent in entities:
            neighbors = store.semantic_neighbors(ent["id"], ent["embedding"], 6)
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
            fields = store.group_fields(sorted(group))
            if not fields or len(fields["ids"]) < 2:
                continue
            summary["groups"] += 1
            merged = self._merge_one_group(
                store,
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

    def _merge_one_group(self, store: EntityMaintenance, **group_fields) -> dict[str, int]:
        return merge_entity_group(store, **group_fields)

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
