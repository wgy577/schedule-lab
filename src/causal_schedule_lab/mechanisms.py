"""Deterministic mechanism measurements, eligibility gates and effect posteriors."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from enum import StrEnum
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .ir import Problem, Schedule
from .storage.graph_store import InterventionRecord


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateVariation(StrEnum):
    CONSTANT = "constant_across_candidates"
    DECISION_DEPENDENT = "decision_dependent"
    EXOGENOUS = "exogenous"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FactorControllability(StrEnum):
    DIRECT = "direct"
    HYPOTHESIZED_INDIRECT = "hypothesized_indirect"
    VERIFIED_INDIRECT = "verified_indirect"
    FIXED = "fixed"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class MechanismLink(StrEnum):
    DIRECT = "direct"
    MEDIATED = "mediated"
    SPECULATIVE = "speculative"
    NONE = "none"
    UNKNOWN = "unknown"


class MechanismDefinition(FrozenModel):
    id: str
    name: str
    aliases: tuple[str, ...]
    description: str
    calculator: str
    required_data: tuple[str, ...]
    objective_direction: str
    limitations: tuple[str, ...]


class MechanismMeasurement(FrozenModel):
    mechanism_id: str
    available: bool
    value: float | None = None
    unit: str
    scope: str
    components: dict[str, float | str | int] = Field(default_factory=dict)
    reason: str | None = None


class FactorQualification(FrozenModel):
    eligible: bool
    variation: CandidateVariation
    controllability: FactorControllability
    mechanism_link: MechanismLink
    replay_count: int = Field(ge=0)
    replay_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: tuple[str, ...]


class EffectPosteriorEstimate(FrozenModel):
    key: str
    observations: int
    full_observations: int
    valid_probability: float
    objective_gain_mean: float
    objective_gain_std: float
    conservative_gain: float
    mechanism_delta_mean: float
    runtime_mean: float
    token_mean: float
    acquisition: float
    scope_level: str


def load_mechanism_definitions(
    path: str | Path | None = None,
) -> tuple[MechanismDefinition, ...]:
    source = (
        Path(path).expanduser().resolve()
        if path
        else Path(__file__).resolve().parent / "knowledge" / "mechanism_targets.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    return tuple(MechanismDefinition.model_validate(item) for item in payload["mechanisms"])


def _resource_intervals(
    problem: Problem,
    schedule: Schedule,
) -> dict[str, list[tuple[int, int, str]]]:
    mode_map = problem.mode_map()
    result: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for assignment in schedule.assignments:
        _, mode = mode_map[assignment.mode_id]
        for resource_id in mode.resources:
            result[resource_id].append(
                (assignment.start, assignment.end, assignment.operation_id)
            )
    for intervals in result.values():
        intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    return result


def _critical_resource_idle_gap(
    problem: Problem,
    schedule: Schedule,
) -> MechanismMeasurement:
    intervals = _resource_intervals(problem, schedule)
    capacities = {item.id: item.capacity for item in problem.resources}
    candidates = []
    for resource_id, items in intervals.items():
        if capacities.get(resource_id, 1) != 1 or not items:
            continue
        busy = sum(end - start for start, end, _ in items)
        utilization = busy / max(1, schedule.makespan)
        candidates.append((utilization, busy, resource_id, items))
    if not candidates:
        return MechanismMeasurement(
            mechanism_id="critical_resource_idle_gap",
            available=False,
            unit="time",
            scope="schedule",
            reason="no used unit-capacity resource",
        )
    utilization, busy, resource_id, selected = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    gaps = [
        max(0, selected[index + 1][0] - selected[index][1])
        for index in range(len(selected) - 1)
    ]
    return MechanismMeasurement(
        mechanism_id="critical_resource_idle_gap",
        available=True,
        value=sum(gaps) / problem.time_scale,
        unit="time",
        scope=resource_id,
        components={
            "gap_count": sum(gap > 0 for gap in gaps),
            "max_gap": (max(gaps, default=0) / problem.time_scale),
            "busy": busy / problem.time_scale,
            "utilization": utilization,
        },
    )


def _critical_path_length(
    problem: Problem,
    schedule: Schedule,
) -> MechanismMeasurement:
    assignment_map = schedule.assignment_map()
    nodes = set(assignment_map)
    successors: dict[str, set[str]] = {item: set() for item in nodes}
    indegree = {item: 0 for item in nodes}
    for operation in problem.operations:
        if operation.id not in nodes:
            continue
        for predecessor in operation.predecessors:
            if predecessor in nodes and operation.id not in successors[predecessor]:
                successors[predecessor].add(operation.id)
                indegree[operation.id] += 1
    for items in _resource_intervals(problem, schedule).values():
        for (_, _, left), (_, _, right) in zip(items, items[1:], strict=False):
            if right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    queue = deque(sorted(item for item, degree in indegree.items() if degree == 0))
    topological = []
    while queue:
        current = queue.popleft()
        topological.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(topological) != len(nodes):
        return MechanismMeasurement(
            mechanism_id="critical_path_length",
            available=False,
            unit="time",
            scope="schedule",
            reason="precedence plus realized resource sequence contains a cycle",
        )
    distance = {
        item: assignment_map[item].end - assignment_map[item].start
        for item in nodes
    }
    parent: dict[str, str] = {}
    for current in topological:
        for successor in successors[current]:
            candidate = distance[current] + (
                assignment_map[successor].end - assignment_map[successor].start
            )
            if candidate > distance[successor]:
                distance[successor] = candidate
                parent[successor] = current
    terminal = max(distance, key=lambda item: (distance[item], item), default=None)
    if terminal is None:
        value = 0.0
        path_length = 0
    else:
        value = distance[terminal] / problem.time_scale
        path_length = 1
        while terminal in parent:
            terminal = parent[terminal]
            path_length += 1
    return MechanismMeasurement(
        mechanism_id="critical_path_length",
        available=True,
        value=value,
        unit="time",
        scope="schedule",
        components={"operation_count": path_length},
    )


def _critical_operation_waiting(
    problem: Problem,
    schedule: Schedule,
) -> MechanismMeasurement:
    assignments = schedule.assignment_map()
    jobs = problem.job_map()
    waits = []
    for operation in problem.operations:
        assignment = assignments.get(operation.id)
        if assignment is None:
            continue
        ready = max(operation.release, jobs[operation.job_id].release)
        ready = max(
            (
                assignments[predecessor].end
                for predecessor in operation.predecessors
                if predecessor in assignments
            ),
            default=ready,
        )
        waits.append(max(0, assignment.start - ready))
    return MechanismMeasurement(
        mechanism_id="critical_operation_waiting",
        available=bool(waits),
        value=(sum(waits) / problem.time_scale) if waits else None,
        unit="time",
        scope="schedule",
        components={
            "waiting_operations": sum(value > 0 for value in waits),
            "max_wait": max(waits, default=0) / problem.time_scale,
        },
        reason=None if waits else "schedule has no matched operations",
    )


def _bottleneck_load_imbalance(
    problem: Problem,
    schedule: Schedule,
) -> MechanismMeasurement:
    intervals = _resource_intervals(problem, schedule)
    resource_map = problem.resource_map()
    utilizations = []
    for resource_id, items in intervals.items():
        busy = sum(end - start for start, end, _ in items)
        capacity = resource_map[resource_id].capacity
        utilizations.append(busy / max(1, schedule.makespan * capacity))
    if len(utilizations) < 2:
        return MechanismMeasurement(
            mechanism_id="bottleneck_load_imbalance",
            available=False,
            unit="coefficient_of_variation",
            scope="schedule",
            reason="fewer than two used resources",
        )
    mean = fmean(utilizations)
    value = stdev(utilizations) / mean if mean > 0 else 0.0
    return MechanismMeasurement(
        mechanism_id="bottleneck_load_imbalance",
        available=True,
        value=value,
        unit="coefficient_of_variation",
        scope="schedule",
        components={
            "resource_count": len(utilizations),
            "mean_utilization": mean,
            "max_utilization": max(utilizations),
            "min_utilization": min(utilizations),
        },
    )


def _metadata_sum(
    problem: Problem,
    schedule: Schedule,
    *,
    mechanism_id: str,
    keys: tuple[str, ...],
) -> MechanismMeasurement:
    values = []
    for assignment in schedule.assignments:
        for key in keys:
            value = assignment.metadata.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return MechanismMeasurement(
            mechanism_id=mechanism_id,
            available=False,
            unit="time",
            scope="schedule",
            reason=f"none of metadata keys {keys} are present",
        )
    return MechanismMeasurement(
        mechanism_id=mechanism_id,
        available=True,
        value=sum(values) / problem.time_scale,
        unit="time",
        scope="schedule",
        components={
            "observations": len(values),
            "max_component": max(values) / problem.time_scale,
        },
    )


def _parallel_capacity_loss(
    problem: Problem,
    schedule: Schedule,
) -> MechanismMeasurement:
    mode_map = problem.mode_map()
    busy = 0
    for assignment in schedule.assignments:
        _, mode = mode_map[assignment.mode_id]
        busy += (assignment.end - assignment.start) * len(mode.resources)
    capacity = schedule.makespan * sum(item.capacity for item in problem.resources)
    if capacity <= 0:
        return MechanismMeasurement(
            mechanism_id="parallel_capacity_loss",
            available=False,
            unit="fraction",
            scope="schedule",
            reason="zero schedule capacity",
        )
    return MechanismMeasurement(
        mechanism_id="parallel_capacity_loss",
        available=True,
        value=max(0.0, capacity - busy) / capacity,
        unit="fraction",
        scope="schedule",
        components={
            "busy_capacity_time": busy / problem.time_scale,
            "available_capacity_time": capacity / problem.time_scale,
        },
    )


def measure_mechanisms(
    problem: Problem,
    schedule: Schedule,
) -> tuple[MechanismMeasurement, ...]:
    """Measure every approved makespan mechanism without inventing absent data."""

    return (
        _critical_resource_idle_gap(problem, schedule),
        _critical_path_length(problem, schedule),
        _critical_operation_waiting(problem, schedule),
        _bottleneck_load_imbalance(problem, schedule),
        _metadata_sum(
            problem,
            schedule,
            mechanism_id="critical_setup_overhead",
            keys=("setup_time", "setup_duration", "changeover_time"),
        ),
        _metadata_sum(
            problem,
            schedule,
            mechanism_id="transport_synchronization_delay",
            keys=(
                "transport_delay",
                "synchronization_delay",
                "vehicle_waiting",
            ),
        ),
        _metadata_sum(
            problem,
            schedule,
            mechanism_id="blocking_no_wait_loss",
            keys=("blocking_delay", "no_wait_penalty"),
        ),
        _parallel_capacity_loss(problem, schedule),
    )


def infer_candidate_variation(
    values: Iterable[float | int | str | bool | None],
    *,
    tolerance: float = 1e-9,
) -> CandidateVariation:
    observed = list(values)
    if len(observed) < 2 or any(value is None for value in observed):
        return CandidateVariation.UNKNOWN
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observed):
        numeric = [float(value) for value in observed]
        return (
            CandidateVariation.CONSTANT
            if max(numeric) - min(numeric) <= tolerance
            else CandidateVariation.DECISION_DEPENDENT
        )
    return (
        CandidateVariation.CONSTANT
        if len({json.dumps(value, sort_keys=True) for value in observed}) == 1
        else CandidateVariation.DECISION_DEPENDENT
    )


def qualify_factor(
    values: Iterable[float | int | str | bool | None],
    *,
    controllability: FactorControllability,
    mechanism_link: MechanismLink,
    replay_count: int = 0,
    replay_consistency: float | None = None,
    tolerance: float = 1e-9,
    minimum_replay_consistency: float = 0.8,
) -> FactorQualification:
    variation = infer_candidate_variation(values, tolerance=tolerance)
    reasons = []
    if variation in {CandidateVariation.CONSTANT, CandidateVariation.UNKNOWN}:
        reasons.append("factor does not currently distinguish candidates")
    if controllability == FactorControllability.HYPOTHESIZED_INDIRECT:
        reasons.append("indirect control has not been verified by replay")
    elif controllability in {
        FactorControllability.FIXED,
        FactorControllability.EXTERNAL,
        FactorControllability.UNKNOWN,
    }:
        reasons.append(f"controllability={controllability.value} is not an optimizer lever")
    if controllability == FactorControllability.VERIFIED_INDIRECT:
        if replay_count < 1:
            reasons.append("verified indirect control requires at least one replay")
        if replay_consistency is None or replay_consistency < minimum_replay_consistency:
            reasons.append("indirect replay consistency is below the current gate")
    if mechanism_link in {
        MechanismLink.NONE,
        MechanismLink.UNKNOWN,
        MechanismLink.SPECULATIVE,
    }:
        reasons.append("factor-to-mechanism path is not executable evidence")
    return FactorQualification(
        eligible=not reasons,
        variation=variation,
        controllability=controllability,
        mechanism_link=mechanism_link,
        replay_count=replay_count,
        replay_consistency=replay_consistency,
        reasons=tuple(reasons),
    )


def estimate_effect_posterior(
    records: Iterable[InterventionRecord],
    *,
    scope_level: str,
    confidence_z: float = 1.0,
    time_cost_weight: float = 1.0,
    token_cost_weight: float = 0.0001,
) -> EffectPosteriorEstimate:
    rows = list(records)
    if not rows:
        return EffectPosteriorEstimate(
            key="unobserved",
            observations=0,
            full_observations=0,
            valid_probability=0.5,
            objective_gain_mean=0.0,
            objective_gain_std=math.inf,
            conservative_gain=0.0,
            mechanism_delta_mean=0.0,
            runtime_mean=0.0,
            token_mean=0.0,
            acquisition=0.0,
            scope_level=scope_level,
        )
    key = (
        f"{rows[0].problem_family}|{rows[0].operator_id}|"
        f"{rows[0].factor_id}|{rows[0].mechanism_id}"
    )
    valid_probability = (1 + sum(item.valid for item in rows)) / (2 + len(rows))
    valid_rows = [item for item in rows if item.valid]
    gains = [item.objective_gain for item in valid_rows]
    gain_mean = fmean(gains) if gains else 0.0
    gain_std = stdev(gains) if len(gains) > 1 else (
        abs(gain_mean) if gains else math.inf
    )
    standard_error = (
        gain_std / math.sqrt(len(gains))
        if gains and math.isfinite(gain_std)
        else math.inf
    )
    conservative = max(
        0.0,
        gain_mean - confidence_z * standard_error
        if math.isfinite(standard_error)
        else 0.0,
    )
    runtime_mean = fmean(item.runtime_seconds for item in rows)
    token_mean = fmean(item.token_cost for item in rows)
    cost = max(
        1e-9,
        time_cost_weight * runtime_mean + token_cost_weight * token_mean,
    )
    return EffectPosteriorEstimate(
        key=key,
        observations=len(rows),
        full_observations=sum(
            item.validation_fidelity == "full" for item in rows
        ),
        valid_probability=valid_probability,
        objective_gain_mean=gain_mean,
        objective_gain_std=gain_std,
        conservative_gain=conservative,
        mechanism_delta_mean=fmean(item.mechanism_delta for item in rows),
        runtime_mean=runtime_mean,
        token_mean=token_mean,
        acquisition=conservative * valid_probability / cost,
        scope_level=scope_level,
    )
