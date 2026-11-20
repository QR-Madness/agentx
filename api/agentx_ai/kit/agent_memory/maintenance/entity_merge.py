"""True Entity node-merge: fold fields, re-home every relationship, delete dups.

Relocated from ``management/commands/dedupe_entities.py`` so both the command and
future write-path/janitor mergers share one implementation. The queries live in the
store (``memory/entity_maintenance.py::EntityMaintenance``); this module decides
what the survivor keeps. ``merge_entity_group`` takes the store, whose transaction
the caller owns (the dedupe command runs one per invocation for dry-run rollback).

⚠️  Never merge Agent entities (``agent_id`` is durable identity — INV-3);
candidate *selection* is the caller's job and must exclude ``type = 'Agent'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.entity_maintenance import EntityMaintenance


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
    store: EntityMaintenance,
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

    store.apply_survivor_fields(
        survivor_id, merged_aliases, merged_description, merged_access_count,
    )

    rewritten_total = 0
    for dup_id in dup_ids:
        rewritten_total += store.rewrite_and_delete_duplicate(survivor_id, dup_id)

    collapsed = 0
    if dup_ids:
        collapsed = store.collapse_parallel_relationships(survivor_id)

    return {
        "duplicates": len(dup_ids),
        "rewritten": rewritten_total,
        "collapsed": collapsed,
    }
