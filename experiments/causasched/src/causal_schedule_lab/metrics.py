from __future__ import annotations

from collections import defaultdict
from typing import Any

from .ir import Problem, Schedule


def schedule_metrics(
    problem: Problem,
    schedule: Schedule,
    baseline: Schedule | None = None,
) -> dict[str, Any]:
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    jobs = problem.job_map()
    completion: dict[str, int] = defaultdict(int)
    busy: dict[str, int] = defaultdict(int)
    energy = 0.0
    cost = 0.0
    for assignment in schedule.assignments:
        operation = operation_map[assignment.operation_id]
        _, mode = mode_map[assignment.mode_id]
        completion[operation.job_id] = max(
            completion[operation.job_id],
            assignment.end,
        )
        for resource in mode.resources:
            busy[resource] += assignment.end - assignment.start
        energy += mode.energy
        cost += mode.cost
    total_tardiness = sum(
        max(0, completion[job.id] - job.due) * job.weight
        for job in problem.jobs
        if job.due is not None
    )
    maximum_tardiness = max(
        (
            max(0, completion[job.id] - job.due)
            for job in problem.jobs
            if job.due is not None
        ),
        default=0,
    )
    total_flow = sum(
        completion[job.id] - job.release
        for job in problem.jobs
    )
    resource_metrics = {}
    for resource in problem.resources:
        available = max(1, schedule.makespan * resource.capacity)
        resource_metrics[resource.id] = {
            "busy": busy[resource.id],
            "idle": max(0, available - busy[resource.id]),
            "utilization": busy[resource.id] / available,
            "capacity": resource.capacity,
            "family": resource.family,
            "tags": list(resource.tags),
        }
    change_cost = {"start_shift": 0, "mode_changes": 0, "sequence_changes": 0}
    if baseline is not None:
        base = baseline.assignment_map()
        for assignment in schedule.assignments:
            previous = base.get(assignment.operation_id)
            if previous is None:
                continue
            change_cost["start_shift"] += abs(assignment.start - previous.start)
            change_cost["mode_changes"] += int(
                assignment.mode_id != previous.mode_id
            )
        base_order = sorted(
            baseline.assignments,
            key=lambda item: (item.start, item.end, item.operation_id),
        )
        trial_order = sorted(
            schedule.assignments,
            key=lambda item: (item.start, item.end, item.operation_id),
        )
        base_rank = {item.operation_id: index for index, item in enumerate(base_order)}
        trial_rank = {item.operation_id: index for index, item in enumerate(trial_order)}
        change_cost["sequence_changes"] = sum(
            base_rank[key] != trial_rank.get(key)
            for key in base_rank
        )
    return {
        "makespan": schedule.makespan / problem.time_scale,
        "total_tardiness": total_tardiness / problem.time_scale,
        "maximum_tardiness": maximum_tardiness / problem.time_scale,
        "total_flow_time": total_flow / problem.time_scale,
        "energy": energy,
        "cost": cost,
        "change_cost": change_cost,
        "change_cost_scalar": (
            change_cost["mode_changes"] * 1_000_000
            + change_cost["sequence_changes"] * 10_000
            + change_cost["start_shift"]
        ),
        "bottleneck_idle": sum(
            item["idle"] for item in resource_metrics.values()
        )
        / problem.time_scale,
        "resource_metrics": resource_metrics,
    }
