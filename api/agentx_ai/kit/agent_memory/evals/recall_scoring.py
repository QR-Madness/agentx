"""Pure ranking metrics for the recall eval harness (no Django/Neo4j imports).

Conventions:
- ``ranked_ids`` is the ordered result list (best first) — for facts that is
  ``[f["id"] for f in bundle.facts]``; callback queries pass turn ids instead.
- ``relevant_ids`` is an unordered set of ids that count as hits for recall@k.
- MRR scores only the single *best* expected id (``expected_fact_keys[0]``).
- Negative/abstention queries are scored by :func:`score_negative` and are
  excluded from recall@k / MRR aggregates (reported as ``abstention_pass_rate``).
"""

from statistics import mean


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """|relevant ∩ ranked[:k]| / |relevant|. Empty relevant set → 0.0."""
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return sum(1 for rid in relevant_ids if rid in top) / len(relevant_ids)


def mrr(ranked_ids: list[str], best_id: str | None) -> float:
    """1/rank of the best expected id, 0.0 if absent (or no expectation)."""
    if not best_id:
        return 0.0
    try:
        return 1.0 / (ranked_ids.index(best_id) + 1)
    except ValueError:
        return 0.0


def rank_of(ranked_ids: list[str], best_id: str | None) -> int | None:
    """1-based rank of the best expected id, None if absent."""
    if not best_id:
        return None
    try:
        return ranked_ids.index(best_id) + 1
    except ValueError:
        return None


def score_negative(ranked_ids: list[str], forbidden_ids: list[str], k: int) -> bool:
    """PASS iff no forbidden id appears in the top-k."""
    top = set(ranked_ids[:k])
    return not any(fid in top for fid in forbidden_ids)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list (empty → 0.0)."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(pct / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _metric_block(rows: list[dict], ks: list[int]) -> dict:
    """Aggregate recall@k/MRR over positive rows + latency over all rows."""
    positive = [r for r in rows if not r.get("negative")]
    negative = [r for r in rows if r.get("negative")]
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    block: dict = {
        "queries": len(rows),
        "mrr": round(mean(r["mrr"] for r in positive), 4) if positive else None,
        "latency_ms": {
            "mean": round(mean(lat), 1) if lat else 0.0,
            "p50": round(_percentile(lat, 50), 1),
            "p95": round(_percentile(lat, 95), 1),
        },
    }
    for k in ks:
        block[f"recall@{k}"] = (
            round(mean(r[f"recall@{k}"] for r in positive), 4) if positive else None
        )
    if negative:
        block["abstention_pass_rate"] = round(
            sum(1 for r in negative if r["abstention_pass"]) / len(negative), 4
        )
    return block


def aggregate(per_query_rows: list[dict], ks: list[int]) -> dict:
    """Overall + per-category aggregate metrics.

    Each row: ``{name, category, negative: bool, mrr, recall@<k>...,
    abstention_pass (negative only), latency_ms}``.
    """
    by_category: dict[str, list[dict]] = {}
    for row in per_query_rows:
        by_category.setdefault(row["category"], []).append(row)
    return {
        **_metric_block(per_query_rows, ks),
        "by_category": {
            cat: _metric_block(rows, ks) for cat, rows in sorted(by_category.items())
        },
    }
