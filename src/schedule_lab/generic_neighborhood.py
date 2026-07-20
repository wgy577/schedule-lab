from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

from .metrics import schedule_metrics
from .model import Problem, Schedule
from .solvers import solve_cp_sat
from .validation import validate_schedule


def _hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selected_resources(problem: Problem, schedule: Schedule) -> dict[str, tuple[str, ...]]:
    mode_map = problem.mode_map()
    return {
        assignment.operation_id: mode_map[assignment.mode_id][1].resources
        for assignment in schedule.assignments
    }


def diagnose_bottlenecks(problem: Problem, schedule: Schedule) -> list[dict[str, Any]]:
    """Return stable family-aware bottlenecks without domain-specific names."""

    validation = validate_schedule(problem, schedule)
    if not validation.feasible:
        raise ValueError(f"incumbent is infeasible: {validation.errors}")
    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    selected_resources = _selected_resources(problem, schedule)
    resource_map = problem.resource_map()
    by_resource: dict[str, list[Any]] = defaultdict(list)
    for assignment in schedule.assignments:
        for resource in selected_resources[assignment.operation_id]:
            by_resource[resource].append(assignment)
    for assignments in by_resource.values():
        assignments.sort(key=lambda item: (item.start, item.end, item.operation_id))

    candidates: list[dict[str, Any]] = []
    final_index = max((operation.index for operation in problem.operations), default=0)
    final_assignments = sorted(
        (
            assignment
            for assignment in schedule.assignments
            if operation_map[assignment.operation_id].index == final_index
        ),
        key=lambda item: (item.start, item.end, item.operation_id),
    )
    for left, right in zip(final_assignments, final_assignments[1:]):
        gap = right.start - left.end
        if gap <= 0:
            continue
        candidates.append(
            {
                "kind": "sink_stage_gap",
                "score": gap,
                "operations": [left.operation_id, right.operation_id],
                "window": [left.end, right.start],
                "resource": None,
                "recommendedOperator": (
                    "permutation_insertion"
                    if problem.kind == "FSP"
                    else "sink_gap_suffix_repair"
                ),
            }
        )

    for resource_id, assignments in sorted(by_resource.items()):
        if resource_map[resource_id].capacity != 1:
            continue
        for left, right in zip(assignments, assignments[1:]):
            gap = right.start - left.end
            block_score = (left.end - left.start) + (right.end - right.start) - max(0, gap)
            candidates.append(
                {
                    "kind": "critical_resource_block" if gap == 0 else "resource_idle_gap",
                    "score": block_score,
                    "operations": [left.operation_id, right.operation_id],
                    "window": [left.start, right.end],
                    "resource": resource_id,
                    "recommendedOperator": "critical_block_adjacent_swap",
                }
            )

    if problem.kind in {"FJSP", "HFSP"}:
        metrics = schedule_metrics(problem, schedule)
        utilization = {
            resource: values["utilization"]
            for resource, values in metrics["resource_metrics"].items()
        }
        for assignment in schedule.assignments:
            operation = operation_map[assignment.operation_id]
            if len(operation.modes) < 2:
                continue
            current_resources = selected_resources[operation.id]
            current_load = max(utilization.get(resource, 0.0) for resource in current_resources)
            alternatives = [
                mode.id
                for mode in operation.modes
                if mode.id != assignment.mode_id
                and max(utilization.get(resource, 0.0) for resource in mode.resources) + 1e-9 < current_load
            ]
            if alternatives:
                candidates.append(
                    {
                        "kind": "flexible_resource_imbalance",
                        "score": current_load,
                        "operations": [operation.id],
                        "window": [assignment.start, assignment.end],
                        "resource": current_resources[0],
                        "alternativeModes": alternatives,
                        "recommendedOperator": "alternative_resource_reassignment",
                    }
                )

    kind_priority = {
        "sink_stage_gap": 0 if problem.kind in {"FSP", "HFSP"} else 2,
        "critical_resource_block": 0 if problem.kind == "JSP" else 1,
        "flexible_resource_imbalance": 0 if problem.kind == "FJSP" else 1,
        "resource_idle_gap": 3,
    }
    return sorted(
        candidates,
        key=lambda item: (
            kind_priority.get(item["kind"], 9),
            -float(item["score"]),
            tuple(item["operations"]),
            item.get("resource") or "",
        ),
    )


def build_generic_neighborhood_plan(
    problem: Problem,
    schedule: Schedule,
    *,
    max_bottlenecks: int = 4,
    radii: Iterable[int] = (1, 2),
    include_pairs: bool = True,
    max_pair_fraction: float = 0.8,
) -> dict[str, Any]:
    bottlenecks = diagnose_bottlenecks(problem, schedule)[:max_bottlenecks]
    if not bottlenecks:
        raise ValueError("no generic bottleneck could be diagnosed")
    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    jobs: dict[str, list[Any]] = defaultdict(list)
    for operation in problem.operations:
        jobs[operation.job_id].append(operation)
    for operations in jobs.values():
        operations.sort(key=lambda item: (item.index, item.id))
    all_ids = {operation.id for operation in problem.operations}
    selected_resources = _selected_resources(problem, schedule)
    plans = []
    for bottleneck_rank, bottleneck in enumerate(bottlenecks, start=1):
        center = set(bottleneck["operations"])
        for radius in tuple(dict.fromkeys(map(int, radii))):
            if radius < 1:
                raise ValueError("radii must be positive")
            released = set(center)
            for operation_id in tuple(center):
                operation = operation_map[operation_id]
                suffix = [item.id for item in jobs[operation.job_id] if item.index >= max(0, operation.index - radius + 1)]
                released.update(suffix)
                for resource_id in selected_resources[operation_id]:
                    resource_assignments = sorted(
                        (
                            assignment
                            for assignment in schedule.assignments
                            if resource_id in selected_resources[assignment.operation_id]
                        ),
                        key=lambda item: (item.start, item.end, item.operation_id),
                    )
                    index = next(i for i, item in enumerate(resource_assignments) if item.operation_id == operation_id)
                    for neighbor in resource_assignments[max(0, index - radius): index + radius + 1]:
                        released.add(neighbor.operation_id)
            frozen = all_ids - released
            signature_payload = {
                "problem": problem.id,
                "kind": problem.kind,
                "incumbentMakespan": schedule.makespan,
                "bottleneck": bottleneck,
                "radius": radius,
                "released": sorted(released),
            }
            plans.append(
                {
                    "index": len(plans),
                    "signature": _hash(signature_payload),
                    "family": problem.kind,
                    "moveDepth": 1,
                    "componentBottleneckRanks": [bottleneck_rank],
                    "bottleneckRank": bottleneck_rank,
                    "bottleneck": bottleneck,
                    "radius": radius,
                    "releasedOperations": sorted(released),
                    "frozenOperations": sorted(frozen),
                    "releasedCount": len(released),
                    "frozenCount": len(frozen),
                    "repairEngine": "cp-sat-fixed-outside",
                    "requiresDomainOracle": bool(problem.metadata.get("requires_domain_validation")),
                }
            )
    if include_pairs:
        first_radius = min(tuple(dict.fromkeys(map(int, radii))))
        pair_sources = [
            item
            for item in plans
            if item["radius"] == first_radius and item["moveDepth"] == 1
        ]
        for left, right in combinations(pair_sources, 2):
            released = set(left["releasedOperations"]) | set(right["releasedOperations"])
            if len(released) / max(1, len(all_ids)) > max_pair_fraction:
                continue
            frozen = all_ids - released
            components = [left["bottleneck"], right["bottleneck"]]
            signature_payload = {
                "problem": problem.id,
                "kind": problem.kind,
                "incumbentMakespan": schedule.makespan,
                "moveDepth": 2,
                "components": components,
                "released": sorted(released),
            }
            plans.append(
                {
                    "index": len(plans),
                    "signature": _hash(signature_payload),
                    "family": problem.kind,
                    "moveDepth": 2,
                    "componentBottleneckRanks": [left["bottleneckRank"], right["bottleneckRank"]],
                    "bottleneckRank": min(left["bottleneckRank"], right["bottleneckRank"]),
                    "bottleneck": {
                        "kind": "paired_bottlenecks",
                        "components": components,
                        "recommendedOperator": "two_component_cp_sat_repair",
                    },
                    "radius": first_radius,
                    "releasedOperations": sorted(released),
                    "frozenOperations": sorted(frozen),
                    "releasedCount": len(released),
                    "frozenCount": len(frozen),
                    "repairEngine": "cp-sat-fixed-outside",
                    "requiresDomainOracle": bool(problem.metadata.get("requires_domain_validation")),
                }
            )
    return {
        "problemId": problem.id,
        "family": problem.kind,
        "incumbentMakespan": schedule.makespan,
        "objectiveOrder": ["feasibility", "makespan", "total_tardiness", "total_flow_time", "change_cost"],
        "bottlenecks": bottlenecks,
        "neighborhoods": plans,
    }


def repair_generic_neighborhood(
    problem: Problem,
    incumbent: Schedule,
    neighborhood: dict[str, Any],
    *,
    seed: int = 0,
    deterministic_time: float = 1.0,
    stability_weight: int = 1,
) -> dict[str, Any]:
    frozen = set(map(str, neighborhood["frozenOperations"]))
    result = solve_cp_sat(
        problem,
        seed=seed,
        workers=1,
        warm_start=incumbent,
        stability_weight=stability_weight,
        frozen_operation_ids=frozen,
        deterministic_time=deterministic_time,
    )
    if result.schedule is None:
        return {"status": result.status, "accepted": False, "reason": "no feasible local repair"}
    validation = validate_schedule(problem, result.schedule)
    incumbent_metrics = schedule_metrics(problem, incumbent, incumbent)
    candidate_metrics = schedule_metrics(problem, result.schedule, incumbent)
    incumbent_vector = (
        incumbent_metrics["makespan"],
        incumbent_metrics["total_tardiness"],
        incumbent_metrics["total_flow_time"],
    )
    candidate_vector = (
        candidate_metrics["makespan"],
        candidate_metrics["total_tardiness"],
        candidate_metrics["total_flow_time"],
    )
    domain_required = bool(problem.metadata.get("requires_domain_validation"))
    accepted = validation.feasible and candidate_vector < incumbent_vector and not domain_required
    return {
        "status": result.status,
        "accepted": accepted,
        "provisional": domain_required,
        "reason": (
            "requires domain Oracle"
            if validation.feasible and candidate_vector < incumbent_vector and domain_required
            else "strict generic improvement"
            if accepted
            else "not a strict validated improvement"
        ),
        "validation": validation.model_dump(mode="json"),
        "incumbentMetrics": incumbent_metrics,
        "candidateMetrics": candidate_metrics,
        "schedule": result.schedule.model_dump(mode="json"),
    }
