"""共享原子集上的 Phase-2 排序指标（教师模型.md Phase2加强 §28）。"""

from __future__ import annotations

from typing import Sequence


def _top(ranked: Sequence[str], k: int) -> list[str]:
    return list(ranked[:k])


def recall_at_k(ranked: Sequence[str], positive: frozenset[str], k: int) -> float:
    """``|TopK ∩ R^+| / |R^+|`` — 0 if R^+ empty."""
    if not positive:
        return 0.0
    return len(set(_top(ranked, k)) & positive) / len(positive)


def precision_at_k(ranked: Sequence[str], positive: frozenset[str], k: int) -> float:
    """``|TopK ∩ R^+| / K`` — 0 if K == 0."""
    if k <= 0:
        return 0.0
    return len(set(_top(ranked, k)) & positive) / k


def best_ce_at_k(ranked: Sequence[str], ce_by_atom: dict[str, float], k: int) -> float:
    """``max_{r in TopK} CE(r)`` — 0 if TopK empty."""
    return max((ce_by_atom.get(a, 0.0) for a in _top(ranked, k)), default=0.0)


def mean_ce_at_k(ranked: Sequence[str], ce_by_atom: dict[str, float], k: int) -> float:
    """``(1/K) sum_{r in TopK} CE(r)`` — CE may be negative, not clipped."""
    ids = _top(ranked, k)
    if not ids:
        return 0.0
    return sum(ce_by_atom.get(a, 0.0) for a in ids) / len(ids)


def hit_at_k(ranked: Sequence[str], positive: frozenset[str], k: int) -> bool:
    """``1[exists r in TopK, CE(r) >= tau_CE]``."""
    return bool(set(_top(ranked, k)) & positive)


def success_rate_at_k(hits: Sequence[bool], n_blocks: int) -> float:
    """Share of blocks whose TopK hits at least one high-CE root."""
    if n_blocks <= 0:
        return 0.0
    return sum(1 for h in hits if h) / n_blocks


def lift_at_k(recall: float, random_recall: float, eps: float = 1e-9) -> float:
    """``Recall_method@K / (Recall_random@K + eps)``."""
    return recall / (random_recall + eps)


def spearman_score_ce(scores: Sequence[float], ces: Sequence[float]) -> float | None:
    """Spearman rank correlation between a source score and measured CE.

    Returns ``None`` when there are fewer than 2 points or zero variance in
    either series (no meaningful correlation).
    """
    n = len(scores)
    if n < 2 or len(ces) != n:
        return None
    if len({round(float(s), 12) for s in scores}) < 2:
        return None
    if len({round(float(c), 12) for c in ces}) < 2:
        return None
    rank_s = _rank(scores)
    rank_c = _rank(ces)
    d = sum((a - b) ** 2 for a, b in zip(rank_s, rank_c))
    denom = n * (n * n - 1)
    if denom == 0:
        return None
    return 1.0 - 6.0 * d / denom


def _rank(values: Sequence[float]) -> list[float]:
    """Average-method rank (1-based) for a sequence of values."""
    indexed = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[indexed[t]] = avg
        i = j + 1
    return ranks