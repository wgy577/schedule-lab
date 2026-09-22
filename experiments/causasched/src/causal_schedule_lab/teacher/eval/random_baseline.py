"""Random ranking baseline (教师模型.md Phase2加强 §10).

Random only re-ranks the shared atom set — it never re-runs the Solver.  Many
seeds are averaged so the random curve carries mean/std/min/max rather than a
single lucky draw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, median, pstdev

from .metrics import (
    best_ce_at_k,
    hit_at_k,
    mean_ce_at_k,
    precision_at_k,
    recall_at_k,
)

DEFAULT_SEEDS = 50
K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class RandSummary:
    mean: float
    std: float
    min: float
    max: float
    median: float

    def as_record(self) -> dict[str, float]:
        return {
            "mean": _r(self.mean),
            "std": _r(self.std),
            "min": _r(self.min),
            "max": _r(self.max),
            "median": _r(self.median),
        }


def _r(x: float) -> float:
    return round(float(x), 6)


def _summarize(values: list[float]) -> RandSummary:
    if not values:
        return RandSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    return RandSummary(
        mean=mean(values),
        std=pstdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        median=median(values),
    )


def evaluate_random_ranking(
    atom_ids: list[str],
    ce_by_atom: dict[str, float],
    positive: frozenset[str],
    *,
    k_values: tuple[int, ...] = K_VALUES,
    seeds: int = DEFAULT_SEEDS,
    rng_seed: int = 0,
) -> dict[str, dict[str, RandSummary]]:
    """Averaged random-priority metrics over ``seeds`` shuffles.

    Returns ``{metric: {str(k): RandSummary}}`` for metrics in
    ``recall / precision / best_ce / mean_ce / hit``.
    """
    rng = random.Random(rng_seed)
    per_k = {metric: {str(k): [] for k in k_values}
             for metric in ("recall", "precision", "best_ce", "mean_ce", "hit")}

    for _ in range(seeds):
        shuffled = atom_ids[:]
        rng.shuffle(shuffled)
        for k in k_values:
            per_k["recall"][str(k)].append(recall_at_k(shuffled, positive, k))
            per_k["precision"][str(k)].append(precision_at_k(shuffled, positive, k))
            per_k["best_ce"][str(k)].append(best_ce_at_k(shuffled, ce_by_atom, k))
            per_k["mean_ce"][str(k)].append(mean_ce_at_k(shuffled, ce_by_atom, k))
            per_k["hit"][str(k)].append(1.0 if hit_at_k(shuffled, positive, k) else 0.0)

    return {
        metric: {k: _summarize(vals) for k, vals in k_vals.items()}
        for metric, k_vals in per_k.items()
    }