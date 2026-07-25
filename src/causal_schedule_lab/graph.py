from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import EdgeType, GraphEdge, GraphNode, NodeType, SchedulingGraph


def _selected_resources(problem: Any, schedule: Any) -> dict[str, tuple[str, ...]]:
    mode_map = problem.mode_map()
    return {
        assignment.operation_id: tuple(mode_map[assignment.mode_id][1].resources)
        for assignment in schedule.assignments
    }


def build_scheduling_graph(
    problem: Any,
    schedule: Any,
    *,
    project_id: str,
) -> SchedulingGraph:
    """Build the shared A→L→W→T→D→J graph from the standalone IR."""

    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    resource_map = problem.resource_map()
    selected_resources = _selected_resources(problem, schedule)
    successors: dict[str, list[str]] = defaultdict(list)
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successors[predecessor].append(operation.id)

    by_job: dict[str, list[Any]] = defaultdict(list)
    by_resource: dict[str, list[Any]] = defaultdict(list)
    for operation in problem.operations:
        by_job[operation.job_id].append(operation)
    for assignment in schedule.assignments:
        for resource in selected_resources[assignment.operation_id]:
            by_resource[resource].append(assignment)
    for items in by_job.values():
        items.sort(key=lambda operation: (operation.index, operation.id))
    for items in by_resource.values():
        items.sort(key=lambda assignment: (assignment.start, assignment.end, assignment.operation_id))
    resource_utilization = {
        resource_id: sum(item.end - item.start for item in assignments)
        / max(1.0, schedule.makespan * resource_map[resource_id].capacity)
        for resource_id, assignments in by_resource.items()
    }

    predecessor_end: dict[str, int] = {}
    resource_previous: dict[tuple[str, str], Any] = {}
    resource_next: dict[tuple[str, str], Any] = {}
    resource_idle_before: dict[str, int] = defaultdict(int)
    resource_idle_after: dict[str, int] = defaultdict(int)
    for resource_id, assignments in by_resource.items():
        for index, assignment in enumerate(assignments):
            if index:
                previous = assignments[index - 1]
                resource_previous[(resource_id, assignment.operation_id)] = previous
                resource_idle_before[assignment.operation_id] += max(0, assignment.start - previous.end)
            if index + 1 < len(assignments):
                following = assignments[index + 1]
                resource_next[(resource_id, assignment.operation_id)] = following
                resource_idle_after[assignment.operation_id] += max(0, following.start - assignment.end)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    makespan = float(schedule.makespan)
    for operation in problem.operations:
        assignment = assignment_map[operation.id]
        predecessor_end[operation.id] = max(
            (assignment_map[item].end for item in operation.predecessors),
            default=operation.release,
        )
        wait = max(0, assignment.start - predecessor_end[operation.id])
        remaining = sum(
            assignment_map[item.id].end - assignment_map[item.id].start
            for item in by_job[operation.job_id]
            if item.index >= operation.index
        )
        available_modes = len(operation.modes)
        _, selected_mode = problem.mode_map()[assignment.mode_id]
        utilization = max(
            (
                resource_utilization.get(resource, 0.0)
                for resource in selected_resources[operation.id]
            ),
            default=0.0,
        )
        slack = max(0.0, makespan - assignment.end)
        nodes.append(
            GraphNode(
                id=operation.id,
                type=NodeType.OPERATION,
                features={
                    "job_id": operation.job_id,
                    "stage": operation.index,
                    "start": assignment.start,
                    "end": assignment.end,
                    "duration": assignment.end - assignment.start,
                    "arrival": predecessor_end[operation.id],
                    "wait": wait,
                    "remaining_operations": sum(
                        item.index >= operation.index for item in by_job[operation.job_id]
                    ),
                    "remaining_processing": remaining,
                    "release": operation.release,
                    "slack": slack,
                    "slack_to_makespan": slack,
                    "criticality": 1.0 - slack / max(1.0, makespan),
                    "is_terminal": not successors[operation.id],
                    "available_modes": available_modes,
                    "resource_idle_before": resource_idle_before[operation.id],
                    "resource_idle_after": resource_idle_after[operation.id],
                    "modifiable": available_modes > 1 or bool(resource_idle_before[operation.id]),
                    "modifiability": min(
                        1.0,
                        0.25 * available_modes
                        + resource_idle_before[operation.id] / max(1.0, makespan),
                    ),
                    "resource_utilization": utilization,
                    "risk": float(bool(problem.choice_links)),
                    "cost": selected_mode.cost,
                },
            )
        )
        for predecessor in operation.predecessors:
            delay = max(0, assignment.start - assignment_map[predecessor].end)
            edges.append(
                GraphEdge(
                    source=predecessor,
                    target=operation.id,
                    type=EdgeType.PRECEDENCE,
                    features={"delay": delay},
                )
            )
            edges.append(
                GraphEdge(
                    source=predecessor,
                    target=operation.id,
                    type=EdgeType.TEMPORAL_CAUSAL,
                    features={"arrival_delay": delay},
                )
            )
            if delay > 0:
                edges.append(
                    GraphEdge(
                        source=predecessor,
                        target=operation.id,
                        type=EdgeType.WAIT_PROPAGATION,
                        features={"wait": delay},
                    )
                )

    for job_id, operations in sorted(by_job.items()):
        completion = max(assignment_map[item.id].end for item in operations)
        nodes.append(
            GraphNode(
                id=job_id,
                type=NodeType.JOB,
                features={
                    "operation_count": len(operations),
                    "completion": completion,
                    "flow_time": completion - min(item.release for item in operations),
                },
            )
        )
        for operation in operations:
            edges.append(
                GraphEdge(
                    source=job_id,
                    target=operation.id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "contains"},
                )
            )
            edges.append(
                GraphEdge(
                    source=operation.id,
                    target=job_id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "member_of"},
                )
            )

    by_stage: dict[int, list[Any]] = defaultdict(list)
    for operation in problem.operations:
        by_stage[operation.index].append(operation)
    for stage, operations in sorted(by_stage.items()):
        stage_id = f"stage:{stage}"
        nodes.append(
            GraphNode(
                id=stage_id,
                type=NodeType.STAGE,
                features={
                    "stage": stage,
                    "operation_count": len(operations),
                    "mean_wait": sum(
                        float(
                            next(
                                node
                                for node in nodes
                                if node.id == operation.id
                            ).features.get("wait", 0.0)
                        )
                        for operation in operations
                    )
                    / max(1, len(operations)),
                },
            )
        )
        for operation in operations:
            edges.append(
                GraphEdge(
                    source=stage_id,
                    target=operation.id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "stage_contains"},
                )
            )

    for resource_id, resource in sorted(resource_map.items()):
        assignments = by_resource.get(resource_id, [])
        busy = sum(item.end - item.start for item in assignments)
        available = max(1.0, makespan * resource.capacity)
        nodes.append(
            GraphNode(
                id=resource_id,
                type=NodeType.RESOURCE,
                features={
                    "capacity": resource.capacity,
                    "load": busy,
                    "queue_length": len(assignments),
                    "utilization": busy / available,
                    "idle": max(0.0, available - busy),
                    "tags": ",".join(resource.tags),
                },
            )
        )
        eligible_operations = [
            operation
            for operation in problem.operations
            if any(
                resource_id in mode.resources for mode in operation.modes
            )
        ]
        selected_ids = {item.operation_id for item in assignments}
        for operation in eligible_operations:
            edges.append(
                GraphEdge(
                    source=operation.id,
                    target=resource_id,
                    type=EdgeType.ELIGIBILITY,
                    features={"selected": operation.id in selected_ids},
                )
            )
            edges.append(
                GraphEdge(
                    source=resource_id,
                    target=operation.id,
                    type=EdgeType.ELIGIBILITY,
                    features={
                        "selected": operation.id in selected_ids,
                        "direction": "resource_to_operation",
                    },
                )
            )
        for index, left in enumerate(eligible_operations):
            for right in eligible_operations[index + 1 :]:
                edges.append(
                    GraphEdge(
                        source=left.id,
                        target=right.id,
                        type=EdgeType.RESOURCE_COMPETITION,
                        features={"resource": resource_id},
                    )
                )
        if resource.capacity == 1:
            for left, right in zip(assignments, assignments[1:]):
                edges.append(
                    GraphEdge(
                        source=left.operation_id,
                        target=right.operation_id,
                        type=EdgeType.RESOURCE_SEQUENCE,
                        features={"gap": max(0, right.start - left.end), "resource": resource_id},
                    )
                )
                if left.end == right.start and right.end == schedule.makespan:
                    edges.append(
                        GraphEdge(
                            source=left.operation_id,
                            target=right.operation_id,
                            type=EdgeType.CRITICAL_PATH,
                            features={"resource": resource_id},
                        )
                    )

    for link in problem.choice_links:
        for left, right in zip(link.operation_ids, link.operation_ids[1:]):
            edges.append(
                GraphEdge(
                    source=left,
                    target=right,
                    type=EdgeType.PROJECT_BINDING,
                    features={"binding": link.id},
                )
            )

    return SchedulingGraph(
        project_id=project_id,
        problem_id=problem.id,
        problem_family=problem.kind,
        nodes=tuple(nodes),
        edges=tuple(edges),
        objective=makespan / getattr(problem, "time_scale", 1),
        metadata={
            "timeScale": getattr(problem, "time_scale", 1),
            "operationCount": len(problem.operations),
            "resourceCount": len(problem.resources),
            "sharedCausalSkeleton": ["A", "L", "W", "T", "D", "J"],
        },
    )
