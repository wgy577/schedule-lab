"""LEGACY TeacherScore ranking (教师模型.md §10/§11) — DEPRECATED for production.

**Phase 2.5 (教师模型.md Phase2加强_2):** the multiplicative TeacherScore

    TeacherScore_A(r) = Rel_A(r) [ lambda_T + (1 - lambda_T) D(r) ]

is retained ONLY as a paper/ablation baseline (``legacy_teacher_score``).  The
official root ranking is now ``causal_root_score = Rel_A`` (see
:mod:`causal_ranker`); the routing/sequencing ``D(r)`` has been downgraded to
:func:`routing_probe_utility` / :func:`sequencing_probe_utility`, used only for
``probe_priority`` ordering inside the causal Top-L (see :mod:`probe_ranker`).

Core principle (Phase2加强_2 §27, must be written into code+docs):

    Cause of Appearance != Ease of Intervention
    Root Ranking != Probe Ordering

For a single atom ``r``, ``Rel_A(r) = max_{v in V(r)} R_A(v)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ir import Problem, Schedule
from .atom_generator import DecisionAtom
from .propagation_graph import RealizedPropagationGraph

LAMBDA_T = 0.30


@dataclass(frozen=True)
class RankedAtom:
    atom: DecisionAtom
    relevance: float
    intervention_prior: float
    teacher_score: float

    def as_record(self) -> dict:
        return {
            **self.atom.as_record(),
            "propagation_relevance": self.relevance,
            "intervention_prior": self.intervention_prior,
            "teacher_score": self.teacher_score,
        }


def _machine_loads(
    problem: Problem,
    schedule: Schedule,
) -> dict[str, float]:
    """Per-machine total processing load (used for the routing load prior)."""
    mode_map = problem.mode_map()
    loads: dict[str, float] = {}
    for assignment in schedule.assignments:
        mode = mode_map[assignment.mode_id][1]
        duration = float(mode.duration) if mode.duration else 0.0
        for resource_id in mode.resources:
            loads[resource_id] = loads.get(resource_id, 0.0) + duration
    return loads


def routing_probe_utility(
    atom: DecisionAtom,
    problem: Problem,
    schedule: Schedule,
    loads: dict[str, float] | None = None,
) -> float:
    """Routing *probe* utility ``D^{route}(r)`` (Phase2加强_2 §6).

    How easy/cheap a routing counterfactual probe is (time + load relief).  It
    is NOT a root-causality score and must never enter ``causal_root_score``.
    """
    return _routing_utility(atom, problem, schedule, loads)


def sequencing_probe_utility(atom: DecisionAtom) -> float:
    """Sequencing *probe* utility ``D^{seq}(r)`` (Phase2加强_2 §7).

    v1: constant 1.0 (all sequencing probes are equally cheap/available).  Never
    enters ``causal_root_score``.
    """
    return 1.0


def legacy_teacher_score(
    atom: DecisionAtom,
    R: dict[str, float],
    problem: Problem,
    schedule: Schedule,
    *,
    lambda_t: float = LAMBDA_T,
    loads: dict[str, float] | None = None,
) -> RankedAtom:
    """LEGACY multiplicative TeacherScore — baseline / ablation only.

    Deprecated for production root ranking (Phase2加强_2 §4).  Use
    :func:`causal_ranker.causal_root_score` for the official ranking.
    """
    relevance = max(float(R.get(op, 0.0)) for op in atom.operations)
    if atom.atom_type == "routing":
        utility = routing_probe_utility(atom, problem, schedule, loads)
    else:
        utility = sequencing_probe_utility(atom)
    score = relevance * (lambda_t + (1.0 - lambda_t) * utility)
    return RankedAtom(
        atom=atom,
        relevance=relevance,
        intervention_prior=utility,
        teacher_score=score,
    )


# Backwards-compatible alias (Phase 1/2 code paths).
compute_teacher_score = legacy_teacher_score


def _routing_utility(
    atom: DecisionAtom,
    problem: Problem,
    schedule: Schedule,
    loads: dict[str, float] | None,
) -> float:
    operation = problem.operation_map().get(atom.operation)
    if operation is None or len(operation.modes) < 2:
        return 0.0
    assignment = schedule.assignment_map().get(atom.operation)
    if assignment is None:
        return 0.0
    mode_map = problem.mode_map()
    current_mode = mode_map[assignment.mode_id][1]
    current_duration = float(current_mode.duration) if current_mode.duration else 0.0
    if current_duration <= 0:
        return 0.0

    # D_r^time: relative processing-time saving to the best alternative.
    alternative_durations = [
        float(m.duration) for m in operation.modes
        if m.id != assignment.mode_id and m.duration is not None and m.duration > 0
    ]
    if not alternative_durations:
        d_time = 0.0
    else:
        best = min(alternative_durations)
        d_time = max(0.0, 1.0 - best / current_duration)

    # D_r^load: relief by moving to a lower-loaded eligible machine.
    if loads is None:
        loads = _machine_loads(problem, schedule)
    current_resources = current_mode.resources
    relief = 0.0
    if current_resources:
        current_load = max(loads.get(r, 0.0) for r in current_resources)
        for mode in operation.modes:
            if mode.id == assignment.mode_id:
                continue
            if not mode.resources:
                continue
            alt_load = max(loads.get(r, 0.0) for r in mode.resources)
            relief = max(relief, current_load - alt_load)
    # Normalize relief by the machine load scale (fall back to 1.0 cap).
    d_load = max(0.0, min(1.0, relief / max(1.0, current_load))) if current_load > 0 else 0.0

    return 1.0 - (1.0 - d_time) * (1.0 - d_load)


def rank_atoms(
    atoms: Iterable[DecisionAtom],
    R: dict[str, float],
    problem: Problem,
    schedule: Schedule,
    *,
    lambda_t: float = LAMBDA_T,
) -> tuple[RankedAtom, ...]:
    """Rank atoms by TeacherScore (descending, tie-break by atom id)."""
    loads = _machine_loads(problem, schedule)
    ranked = [
        compute_teacher_score(atom, R, problem, schedule, lambda_t=lambda_t, loads=loads)
        for atom in atoms
    ]
    return tuple(
        sorted(ranked, key=lambda item: (-item.teacher_score, item.atom.atom_id))
    )