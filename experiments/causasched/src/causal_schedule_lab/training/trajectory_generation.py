"""Offline CP-SAT trajectory generation for Intervention Effect labels.

This is a teacher/data path, never the online value predictor.  Each supplied
macro proposal is executed once, and later steps are attached to the first
experience so a neutral first action can receive a delayed-success label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..counterfactual import CounterfactualEvaluator, CounterfactualResult
from ..m2_v5_schema_v1 import LegalEdit
from ..memory import (
    ExperienceStore,
    InterventionExperience,
    ProposalRecord,
    experience_final_gain,
    experience_future_success,
    experience_training_eligible,
)


@dataclass(frozen=True)
class TrajectoryEffectLabel:
    key: str
    immediate_delta_cmax: float
    final_delta_cmax: float
    number_future_steps: int
    future_success: float
    future_gain: float


@dataclass(frozen=True)
class OfflineTrajectoryArtifact:
    key: str
    results: tuple[CounterfactualResult, ...]
    label: TrajectoryEffectLabel


def trajectory_effect_label(experience: InterventionExperience) -> TrajectoryEffectLabel:
    """Derive predictor labels solely from observed Cmax trajectory outcomes."""
    if experience.outcome is None:
        raise ValueError("trajectory label requires a resolved immediate outcome")
    if not experience_training_eligible(experience):
        raise ValueError("trajectory label requires training-eligible frozen_local evidence")
    immediate = float(experience.outcome.delta_cmax)
    final = (
        float(experience.trajectory_final_delta_cmax)
        if experience.trajectory_final_delta_cmax is not None else immediate
    )
    return TrajectoryEffectLabel(
        key=experience.key,
        immediate_delta_cmax=immediate,
        final_delta_cmax=final,
        number_future_steps=int(experience.trajectory_future_steps),
        future_success=float(experience_future_success(experience)),
        future_gain=float(experience_final_gain(experience)),
    )


def generate_offline_trajectory(
    problem,
    initial_schedule,
    steps: Sequence[tuple[ProposalRecord, Sequence[LegalEdit]]],
    *,
    store: ExperienceStore,
    solver_time: float = 1.0,
    seed: int = 0,
) -> OfflineTrajectoryArtifact:
    """Execute a declared proposal sequence with CP-SAT and persist its labels."""
    if not steps:
        raise ValueError("offline trajectory requires at least one proposal")
    evaluator = CounterfactualEvaluator(store, solver_time=solver_time, seed=seed)
    schedule = initial_schedule
    key: str | None = None
    results: list[CounterfactualResult] = []
    for proposal, edits in steps:
        result = evaluator.evaluate(
            problem,
            schedule,
            proposal,
            edits,
            store_outcome=True,
            continue_trajectory_key=key,
        )
        results.append(result)
        if key is None:
            key = result.key
        if result.new_schedule is None:
            break
        schedule = result.new_schedule
    if key is None:
        raise RuntimeError("offline trajectory did not produce a memory key")
    experience = store.get(key)
    if experience is None:
        raise RuntimeError("offline trajectory was not persisted")
    return OfflineTrajectoryArtifact(
        key=key,
        results=tuple(results),
        label=trajectory_effect_label(experience),
    )


__all__ = [
    "TrajectoryEffectLabel",
    "OfflineTrajectoryArtifact",
    "trajectory_effect_label",
    "generate_offline_trajectory",
]
