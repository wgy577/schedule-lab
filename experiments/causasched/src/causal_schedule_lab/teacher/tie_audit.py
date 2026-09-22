"""Tie-aware ranking audit (教师模型.md Phase2加强_2 §11/§13; 改A6.md §11-§13).

Max-Plus scores saturate heavily (zero-gap edges ``T=1`` give many operations
``R=1``), so ``Rel_seq = max(R(o), R(p))`` produces large tie groups.  The
deterministic ``_order`` tie-break (atom_id) is an evaluation artifact, not a
causal-model capability.  This module quantifies how much of the ranking is
held by ties, and reports strict / optimistic / expected Recall@K so we can
separate "model gave a low score" from "model gave a high score but too many
atoms tied" (改A6.md §15).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .phase2_5 import AtomRow

KS = (1, 3, 5, 10)


def _score(rows: Sequence[AtomRow], score_field: str) -> dict[str, float]:
    return {r.atom_id: float(getattr(r, score_field)) for r in rows}


def _ordered_ids(rows: Sequence[AtomRow], score_field: str) -> list[str]:
    return [r.atom_id for r in sorted(rows, key=lambda r: (-getattr(r, score_field), r.atom_id))]


def _positive_atoms(rows: Sequence[AtomRow], tau_ce: float) -> frozenset[str]:
    return frozenset(r.atom_id for r in rows if r.ce >= tau_ce)


def tie_report(rows: Sequence[AtomRow], score_field: str,
               ks: tuple[int, ...] = KS) -> dict[str, Any]:
    """Per-appearance tie statistics under ``score_field`` (§11)."""
    if not rows:
        return {"candidate_count": 0}
    scores = _score(rows, score_field)
    ordered = _ordered_ids(rows, score_field)
    candidates = len(ordered)
    score_values = [scores[a] for a in ordered]
    max_score = score_values[0]
    max_score_count = sum(1 for s in score_values if s == max_score)
    report: dict[str, Any] = {
        "candidate_count": candidates,
        "unique_score_count": len({round(s, 12) for s in score_values}),
        "max_score_count": max_score_count,
        "max_score_ratio": max_score_count / max(1, candidates),
    }
    for k in ks:
        if k > candidates:
            report[f"top{k}_cutoff_score"] = None
            report[f"top{k}_tie_group_size"] = 0
            continue
        cutoff = score_values[k - 1]
        report[f"top{k}_cutoff_score"] = round(cutoff, 6)
        report[f"top{k}_tie_group_size"] = sum(1 for s in score_values if s == cutoff)
    return report


def _cutoff_decomposition(
    rows: Sequence[AtomRow], score_field: str, k: int
) -> tuple[float, list[str], list[str], int]:
    """Decompose top-K into (cutoff_score, above, group, seats)."""
    scores = _score(rows, score_field)
    ordered = _ordered_ids(rows, score_field)
    if not ordered or k <= 0:
        return 0.0, [], [], 0
    cutoff = scores[ordered[min(k, len(ordered)) - 1]]
    above = [a for a in ordered if scores[a] > cutoff]
    group = [a for a in ordered if scores[a] == cutoff]
    seats = max(0, k - len(above))
    return cutoff, above, group, seats


def strict_recall_at_k(
    rows: Sequence[AtomRow], positive: frozenset[str], k: int, score_field: str
) -> float:
    """Deterministic top-K recall (the current, tie-break-sensitive metric)."""
    ids = _ordered_ids(rows, score_field)[:k]
    if not positive:
        return 0.0
    return len(set(ids) & positive) / len(positive)


def optimistic_recall_at_k(
    rows: Sequence[AtomRow], positive: frozenset[str], k: int, score_field: str
) -> float:
    """Best-case recall if the cutoff tie group could be reordered freely (§13).

    All positives strictly above the cutoff are counted; within the tie group we
    assume the ``seats`` most positive members could be placed in top-K.
    """
    if not positive:
        return 0.0
    _, above, group, seats = _cutoff_decomposition(rows, score_field, k)
    group_pos = len(set(group) & positive)
    hits = len(set(above) & positive) + min(seats, group_pos)
    return hits / len(positive)


def expected_recall_at_k(
    rows: Sequence[AtomRow], positive: frozenset[str], k: int, score_field: str
) -> float:
    """Expected recall if the cutoff tie group is filled uniformly at random."""
    if not positive:
        return 0.0
    _, above, group, seats = _cutoff_decomposition(rows, score_field, k)
    group_size = max(1, len(group))
    group_pos = len(set(group) & positive)
    hits = len(set(above) & positive) + seats * (group_pos / group_size)
    return hits / len(positive)


def tie_aware_recall(rows: Sequence[AtomRow], *, tau_ce: float = 0.30,
                     ks: tuple[int, ...] = KS, score_field: str = "causal_root",
                     group_by_appearance: bool = True) -> dict[str, Any]:
    """§13 strict / optimistic / expected Recall@K, averaged over appearances.

    Returns a nested dict ``{K: {strict, optimistic, expected}}``.  When
    ``group_by_appearance`` is True the per-appearance values are averaged;
    otherwise the flat atom set is used.
    """
    if group_by_appearance:
        groups: dict[str, list[AtomRow]] = defaultdict(list)
        for r in rows:
            groups[r.appearance_id].append(r)
        blocks = list(groups.values())
    else:
        blocks = [list(rows)]
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        strict = [strict_recall_at_k(g, _positive_atoms(g, tau_ce), k, score_field)
                  for g in blocks]
        optimistic = [optimistic_recall_at_k(g, _positive_atoms(g, tau_ce), k, score_field)
                      for g in blocks]
        expected = [expected_recall_at_k(g, _positive_atoms(g, tau_ce), k, score_field)
                    for g in blocks]
        out[str(k)] = {
            "strict": sum(strict) / len(strict) if strict else 0.0,
            "optimistic": sum(optimistic) / len(optimistic) if optimistic else 0.0,
            "expected": sum(expected) / len(expected) if expected else 0.0,
        }
    return out


def tie_audit_by_block(rows: Sequence[AtomRow], score_field: str = "causal_root",
                       ks: tuple[int, ...] = KS) -> list[dict[str, Any]]:
    """Per-appearance tie report rows (for ``tie_audit_by_block.csv``)."""
    groups: dict[str, list[AtomRow]] = defaultdict(list)
    for r in rows:
        groups[r.appearance_id].append(r)
    out = []
    for gid in sorted(groups):
        row: dict[str, Any] = {"appearance_id": gid}
        row.update(tie_report(groups[gid], score_field, ks))
        out.append(row)
    return out