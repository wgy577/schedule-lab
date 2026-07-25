"""Multi-fidelity Bayesian evidence and budget-aware acquisition."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import ExperimentRecord


@dataclass(frozen=True)
class PosteriorEstimate:
    key: str
    observations: int
    full_observations: int
    validity_mean: float
    gain_mean: float
    gain_std: float
    cost_mean: float
    fidelity_bias: float

    def acquisition(
        self,
        *,
        beta: float = 0.35,
        risk_penalty: float = 1.0,
    ) -> float:
        uncertainty = self.gain_std / math.sqrt(max(1, self.observations))
        expected = max(0.0, self.gain_mean + self.fidelity_bias)
        return (
            expected * self.validity_mean
            + beta * uncertainty
            - risk_penalty * (1.0 - self.validity_mean)
        ) / max(1e-6, self.cost_mean)


class MultiFidelityPosterior:
    """Conjugate validity model plus shrinkage gain/cost estimates.

    Light evaluations update quickly.  Paired light/full observations estimate
    systematic proxy bias, so full-Oracle budget is spent where it matters.
    """

    def __init__(self, *, prior_valid: tuple[float, float] = (1.0, 1.0)) -> None:
        self.prior_valid = prior_valid
        self._records: dict[str, list[ExperimentRecord]] = defaultdict(list)

    @staticmethod
    def key(record: ExperimentRecord) -> str:
        return (
            f"{record.cip.diagnostic.type}|{record.action.operator}|"
            f"L{record.action.closure_level}"
        )

    def update(self, record: ExperimentRecord) -> None:
        self._records[self.key(record)].append(record)

    def fit(self, records: Iterable[ExperimentRecord]) -> "MultiFidelityPosterior":
        for record in records:
            self.update(record)
        return self

    def estimate(self, key: str) -> PosteriorEstimate:
        records = self._records.get(key, [])
        alpha, beta = self.prior_valid
        alpha += sum(record.accepted for record in records)
        beta += sum(not record.accepted for record in records)
        gains = [max(0.0, record.delta_objective) for record in records]
        costs = [max(1e-6, record.runtime_seconds) for record in records]
        full = [
            record
            for record in records
            if any(item.fidelity.value == "full" for item in record.verifications)
        ]
        paired_biases = []
        for record in full:
            light = next(
                (
                    item.proxy_delta
                    for item in record.verifications
                    if item.fidelity.value == "light" and item.proxy_delta is not None
                ),
                None,
            )
            truth = next(
                (
                    item.true_delta
                    for item in record.verifications
                    if item.fidelity.value == "full" and item.true_delta is not None
                ),
                None,
            )
            if light is not None and truth is not None:
                paired_biases.append(truth - light)
        gain_mean = sum(gains) / len(gains) if gains else 0.0
        variance = (
            sum((item - gain_mean) ** 2 for item in gains) / max(1, len(gains) - 1)
            if len(gains) > 1
            else 1.0
        )
        return PosteriorEstimate(
            key=key,
            observations=len(records),
            full_observations=len(full),
            validity_mean=alpha / (alpha + beta),
            gain_mean=gain_mean,
            gain_std=math.sqrt(variance),
            cost_mean=sum(costs) / len(costs) if costs else 1.0,
            fidelity_bias=(
                sum(paired_biases) / len(paired_biases) if paired_biases else 0.0
            ),
        )

    def rank(self) -> list[PosteriorEstimate]:
        return sorted(
            (self.estimate(key) for key in self._records),
            key=lambda item: (-item.acquisition(), item.key),
        )
