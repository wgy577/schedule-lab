from __future__ import annotations

from collections import defaultdict
from typing import Literal

from ..ir import Assignment, Problem, Schedule


DispatchRule = Literal[
    "earliest_finish",
    "spt",
    "lpt",
    "most_successors",
    "balanced",
]


def solve_dispatching(
    problem: Problem,
    *,
    rule: DispatchRule = "earliest_finish",
) -> Schedule:
    operation_map = problem.operation_map()
    jobs = problem.job_map()
    successors: dict[str, int] = defaultdict(int)
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successors[predecessor] += 1
    resource_lanes = {
        resource.id: [0] * resource.capacity
        for resource in problem.resources
    }
    assignments: dict[str, Assignment] = {}
    remaining_predecessors = {
        operation.id: len(operation.predecessors) for operation in problem.operations
    }
    successor_ids: dict[str, list[str]] = defaultdict(list)
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successor_ids[predecessor].append(operation.id)
    ready_ids = {
        operation_id for operation_id, count in remaining_predecessors.items() if count == 0
    }
    while len(assignments) < len(operation_map):
        ready = [operation_map[item] for item in ready_ids]
        if not ready:
            raise ValueError("precedence graph is cyclic")
        candidates = []
        for operation in ready:
            precedence_ready = max(
                [
                    operation.release,
                    jobs[operation.job_id].release,
                    *[
                        assignments[item].end
                        for item in operation.predecessors
                    ],
                ]
            )
            for mode in operation.modes:
                lane_choices = []
                for resource in mode.resources:
                    lane = min(
                        enumerate(resource_lanes[resource]),
                        key=lambda item: (item[1], item[0]),
                    )
                    lane_choices.append((resource, lane[0], lane[1]))
                start = max(
                    [precedence_ready, *[item[2] for item in lane_choices]]
                )
                end = start + mode.duration
                if rule == "spt":
                    priority = (mode.duration, end)
                elif rule == "lpt":
                    priority = (-mode.duration, end)
                elif rule == "most_successors":
                    priority = (-successors[operation.id], end)
                elif rule == "balanced":
                    priority = (
                        max(item[2] for item in lane_choices),
                        mode.duration,
                        end,
                    )
                else:
                    priority = (end, mode.duration)
                candidates.append(
                    (
                        priority,
                        operation.id,
                        mode.id,
                        start,
                        end,
                        lane_choices,
                    )
                )
        _, operation_id, mode_id, start, end, lane_choices = min(candidates)
        assignments[operation_id] = Assignment(
            operation_id=operation_id,
            mode_id=mode_id,
            start=start,
            end=end,
            provenance=f"dispatching:{rule}",
        )
        for resource, lane, _ in lane_choices:
            resource_lanes[resource][lane] = end
        ready_ids.remove(operation_id)
        for successor_id in successor_ids.get(operation_id, ()):
            remaining_predecessors[successor_id] -= 1
            if remaining_predecessors[successor_id] == 0:
                ready_ids.add(successor_id)
    return Schedule(
        problem_id=problem.id,
        assignments=tuple(
            assignments[key] for key in sorted(assignments)
        ),
        metadata={"solver": "deterministic-dispatching", "rule": rule},
    )
