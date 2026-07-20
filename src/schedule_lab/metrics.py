from __future__ import annotations

from collections import defaultdict
from typing import Any

from .model import Problem, Schedule


def schedule_metrics(problem: Problem, schedule: Schedule, baseline: Schedule | None = None) -> dict[str, Any]:
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    makespan = schedule.makespan
    resource_busy: dict[str, int] = defaultdict(int)
    resource_operations: dict[str, int] = defaultdict(int)
    job_completion: dict[str, int] = defaultdict(int)
    job_release: dict[str, int] = {}
    tardiness = 0
    for assignment in schedule.assignments:
        operation = operation_map[assignment.operation_id]
        _, mode = mode_map[assignment.mode_id]
        job_completion[operation.job_id] = max(job_completion[operation.job_id], assignment.end)
        job_release[operation.job_id] = min(job_release.get(operation.job_id, operation.release), operation.release)
        if operation.due is not None:
            tardiness += max(0, assignment.end - operation.due)
        for resource_id in mode.resources:
            resource_busy[resource_id] += assignment.end - assignment.start
            resource_operations[resource_id] += 1
    resources = {}
    for resource in problem.resources:
        available = makespan * resource.capacity
        busy = resource_busy[resource.id]
        resources[resource.id] = {
            "name": resource.name,
            "capacity": resource.capacity,
            "tags": list(resource.tags),
            "operations": resource_operations[resource.id],
            "busy": busy,
            "idle": max(0, available - busy),
            "utilization": 0.0 if available == 0 else busy / available,
        }
    change_cost = None
    if baseline is not None:
        base = baseline.assignment_map()
        start_shift = 0
        mode_changes = 0
        for assignment in schedule.assignments:
            previous = base.get(assignment.operation_id)
            if previous is not None:
                start_shift += abs(assignment.start - previous.start)
                mode_changes += int(assignment.mode_id != previous.mode_id)
        change_cost = {"start_shift": start_shift, "mode_changes": mode_changes}
    return {
        "problem_id": problem.id,
        "kind": problem.kind,
        "time_scale": problem.time_scale,
        "makespan": makespan,
        "makespan_display": makespan / problem.time_scale,
        "total_flow_time": sum(job_completion[job] - job_release[job] for job in job_completion),
        "total_tardiness": tardiness,
        "resource_metrics": resources,
        "change_cost": change_cost,
    }


def bottleneck_advice(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    resources = metrics["resource_metrics"]
    ordered = sorted(resources.items(), key=lambda item: item[1]["utilization"], reverse=True)
    advice: list[dict[str, Any]] = []
    if ordered:
        resource_id, values = ordered[0]
        advice.append({"kind": "critical_resource", "resource": resource_id, "utilization": values["utilization"], "operator": "critical_block_swap"})
    for resource_id, values in ordered:
        if "global_launch" in values["tags"] and values["operations"] and values["utilization"] < 0.7:
            advice.append({"kind": "upstream_starvation", "resource": resource_id, "utilization": values["utilization"], "idle": values["idle"], "operator": "launch_gap_fill"})
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for resource_id, values in ordered:
        for tag in values["tags"]:
            grouped[tag].append((resource_id, values))
    for tag, active in grouped.items():
        if len(active) < 2:
            continue
        utilizations = [values["utilization"] for _, values in active if values["operations"]]
        if utilizations and max(utilizations) - min(utilizations) > 0.15:
            advice.append({"kind": "load_imbalance", "resource_family": tag, "operator": "alternative_resource_reassignment", "spread": max(utilizations) - min(utilizations)})
    if metrics["change_cost"] is not None and metrics["change_cost"]["start_shift"]:
        advice.append({"kind": "schedule_disruption", "operator": "rolling_horizon_freeze", "start_shift": metrics["change_cost"]["start_shift"]})
    return advice
