"""V5 M2 -- release-gate integration metrics (Phase 10, spec §40).

These metrics quantify **how much of a gold decision-level structure the Top-K
proposal set retains**, so "multiple proposals" is verifiably more informative
than a single root:

* **Node Recall@K** -- fraction of the gold *causal-envelope* operation nodes
  (the traced upstream path feeding the block) covered by the union of ``V_i``
  across the top-K proposals.
* **Edge Recall@K** -- fraction of gold structural edges covered by union of
  ``E_i``.
* **RootDecision Recall@K** -- fraction of gold root decision *sites* retained
  (verifies top-K + diversity never silently drops a legitimate localized root
  -- regression D's "no collapse" is a metric, not a vibe).
* **Edit Recall@K** -- fraction of gold *legal escape edits* surfaced (the
  hard-feasibility-enumerated opportunities actually carried into proposals).
* **Proposal Success Coverage@K** -- does *at least one* top-K proposal surface
  a gold *success* edit (a decision-time relief / accepted-style structure)?
  This is the spec's key number: "does the top-K contain a structure that would
  lead to an accepted intervention?"
* **Diversity@K** -- mean pairwise ``(1 - Jaccard(V_i))`` across the top-K set;
  1.0 = fully disjoint, 0 = all identical.

This module is **torch-free and training-free**: it measures the structural
chain.  Honesty boundary (project doctrine): these are integration/structural
metrics on the *first untrained* chain run -- they verify the pipeline produces
well-formed, non-collapsing proposals and that the harness measures correctly.
No number here is a validated performance claim; ``identified`` stays False.

Per family (A1/A2/A3/A4) and aggregate, at top-K in {1,3,5}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GoldBuffer:
    """Decision-level gold for one appearance block (hand-/deterministically
    built by the caller; torch-free).

    * ``root_site_ids`` -- gold root decision *site ids* (e.g.
      ``route:Ob->M5``).  Retention of these is RootDecision Recall.
    * ``causal_nodes`` -- gold causal-envelope operation ids feeding the block.
    * ``edges`` -- gold structural (u, v) pairs among those nodes.
    * ``legal_edit_ids`` -- gold hard-feasibility legal escape edits for the
      block's membership operations.
    * ``success_edit_ids`` -- subset of ``legal_edit_ids`` that qualify as a
      success structure (e.g. a decision-time relief edit).
    """

    root_site_ids: frozenset[str] = frozenset()
    causal_nodes: frozenset[str] = frozenset()
    edges: frozenset[tuple[str, str]] = frozenset()
    legal_edit_ids: frozenset[str] = frozenset()
    success_edit_ids: frozenset[str] = frozenset()


def _top_k(proposals: Sequence, k: int) -> list:
    return list(proposals[:k])


def _recall(covered: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & covered) / len(gold)


# -- per-metric helpers (proposal object is duck-typed to the schema) --------


def proposal_nodes_opset(p) -> set[str]:
    return {n for n in p.nodes if not n.startswith("machine:")}


def proposal_edges(p) -> set[tuple[str, str]]:
    return set(p.edges)


def proposal_root_site_ids(p) -> set[str]:
    return {d.site_id for d in p.root_decisions}


def proposal_edit_ids(p) -> set[str]:
    return {e.edit_id for e in p.edits}


def node_recall_at_k(proposals, gold: GoldBuffer, k: int) -> float:
    covered = set()
    for p in _top_k(proposals, k):
        covered |= proposal_nodes_opset(p)
    return _recall(covered, set(gold.causal_nodes))


def edge_recall_at_k(proposals, gold: GoldBuffer, k: int) -> float:
    covered = set()
    for p in _top_k(proposals, k):
        covered |= proposal_edges(p)
    return _recall(covered, set(gold.edges))


def root_decision_recall_at_k(proposals, gold: GoldBuffer, k: int) -> float:
    covered = set()
    for p in _top_k(proposals, k):
        covered |= proposal_root_site_ids(p)
    return _recall(covered, set(gold.root_site_ids))


def edit_recall_at_k(proposals, gold: GoldBuffer, k: int) -> float:
    covered = set()
    for p in _top_k(proposals, k):
        covered |= proposal_edit_ids(p)
    return _recall(covered, set(gold.legal_edit_ids))


def proposal_success_coverage_at_k(proposals, gold: GoldBuffer, k: int) -> float:
    """Top-K contains at least one success edit (spec's accepted-structure test)."""
    if not gold.success_edit_ids:
        return 0.0
    for p in _top_k(proposals, k):
        if proposal_edit_ids(p) & set(gold.success_edit_ids):
            return 1.0
    return 0.0


def diversity_at_k(proposals, k: int) -> float:
    top = _top_k(proposals, k)
    if len(top) < 2:
        return float(len(top))  # 1 proposal => 1.0 "diverse trivially", 0 => 0.0
    total = 0.0
    n = 0
    for i in range(len(top)):
        a = set(top[i].nodes)
        for j in range(i + 1, len(top)):
            b = set(top[j].nodes)
            denom = len(a | b)
            if denom:
                total += 1.0 - (len(a & b) / denom)
            n += 1
    return total / n if n else 0.0


# -- aggregation --------------------------------------------------------------


TOP_K_LEVELS = (1, 3, 5)
METRICS = (
    "node_recall",
    "edge_recall",
    "root_decision_recall",
    "edit_recall",
    "proposal_success_coverage",
    "diversity",
)


def _assess(proposals, gold: GoldBuffer, k: int) -> dict[str, float]:
    return {
        "node_recall": node_recall_at_k(proposals, gold, k),
        "edge_recall": edge_recall_at_k(proposals, gold, k),
        "root_decision_recall": root_decision_recall_at_k(proposals, gold, k),
        "edit_recall": edit_recall_at_k(proposals, gold, k),
        "proposal_success_coverage": proposal_success_coverage_at_k(proposals, gold, k),
        "diversity": diversity_at_k(proposals, k),
    }


def family_of(block_id: str) -> str:
    """Infer the appearance archetype from the block id prefix (e.g. ``A4:0001`` -> ``A4``)."""
    head = block_id.split(":", 1)[0]
    return head if head in ("A1", "A2", "A3", "A4") else "other"


def compute_release_metrics(
    blocks: Mapping[str, tuple],
    gold_by_block: Mapping[str, GoldBuffer],
    *,
    top_k_levels: Sequence[int] = TOP_K_LEVELS,
) -> dict:
    """Per-family + aggregate §40 metrics.

    ``blocks`` maps block_id -> (proposals, ...) sorted top-K first (rank order
    is the ordering convention).  Returns a nested dict::

        {
          "families": { "A4": { (1): {metric: value, ...}, (3): {...}, ... } },
          "aggregate": { (1): {...}, (3): {...}, (5): {...} },
          "n_blocks": int,
        }
    """
    families: dict[str, dict] = {}
    agg: dict[int, dict[str, float]] = {k: {} for k in top_k_levels}
    n_blocks = len(blocks)

    # accumulate per family and aggregate
    fam_accum: dict[str, dict[tuple[int, str], float]] = {}
    agg_accum: dict[tuple[int, str], float] = {}
    fam_counts: dict[str, int] = {}
    for block_id, proposals in blocks.items():
        fam = family_of(block_id)
        gold = gold_by_block.get(block_id, GoldBuffer())
        row = {k: _assess(proposals, gold, k) for k in top_k_levels}
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        bucket = fam_accum.get(fam)
        if bucket is None:
            bucket = {}
            fam_accum[fam] = bucket
        for k, vals in row.items():
            for m, v in vals.items():
                key = (k, m)
                agg_accum[key] = agg_accum.get(key, 0.0) + v
                bucket[key] = bucket.get(key, 0.0) + v

    def _mean(acc: dict[tuple[int, str], float], count: int) -> dict[int, dict[str, float]]:
        return {
            k: {m: (acc.get((k, m), 0.0) / count if count else 0.0) for m in METRICS}
            for k in top_k_levels
        }

    out = {
        "families": {
            fam: _mean(fam_accum[fam], cnt) for fam, cnt in fam_counts.items()
        },
        "aggregate": _mean(agg_accum, n_blocks),
        "n_blocks": n_blocks,
    }
    return out


__all__ = [
    "GoldBuffer",
    "compute_release_metrics",
    "node_recall_at_k",
    "edge_recall_at_k",
    "root_decision_recall_at_k",
    "edit_recall_at_k",
    "proposal_success_coverage_at_k",
    "diversity_at_k",
    "family_of",
    "TOP_K_LEVELS",
    "METRICS",
]