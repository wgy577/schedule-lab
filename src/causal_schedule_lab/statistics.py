"""Reproducible statistical analysis for scheduling experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from scipy import stats


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_difference: float
    median_difference: float
    confidence_interval: tuple[float, float]
    wilcoxon_statistic: float
    p_value: float
    cliffs_delta: float


def bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    return (
        estimates[min(len(estimates) - 1, int(alpha * len(estimates)))],
        estimates[
            min(len(estimates) - 1, int((1.0 - alpha) * len(estimates)) - 1)
        ],
    )


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    wins = sum(a > b for a in left for b in right)
    losses = sum(a < b for a in left for b in right)
    return (wins - losses) / (len(left) * len(right))


def paired_comparison(
    baseline: Sequence[float],
    method: Sequence[float],
    *,
    seed: int = 0,
) -> PairedComparison:
    if len(baseline) != len(method) or not baseline:
        raise ValueError("paired samples need equal non-zero length")
    differences = [base - trial for base, trial in zip(baseline, method)]
    result = stats.wilcoxon(
        differences,
        zero_method="zsplit",
        alternative="two-sided",
    )
    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return PairedComparison(
        n=len(differences),
        mean_difference=sum(differences) / len(differences),
        median_difference=median,
        confidence_interval=bootstrap_interval(differences, seed=seed),
        wilcoxon_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        cliffs_delta=cliffs_delta(baseline, method),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def friedman_test(methods: Mapping[str, Sequence[float]]) -> tuple[float, float]:
    values = list(methods.values())
    if len(values) < 3 or len({len(item) for item in values}) != 1:
        raise ValueError("Friedman test needs at least 3 equal-length methods")
    result = stats.friedmanchisquare(*values)
    return float(result.statistic), float(result.pvalue)
