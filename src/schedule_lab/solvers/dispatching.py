from __future__ import annotations

from collections import defaultdict
from typing import Literal

from ..model import Assignment, Mode, Operation, Problem, Schedule


DispatchRule = Literal["earliest_finish", "spt", "lpt", "most_successors", "balanced"]


def _earliest_resource_slot(
    calendars: dict[str, list[tuple[int, int]]],
    capacities: dict[str, int],
    resources: tuple[str, ...],
    earliest: int,
    duration: int,
) -> int:
    candidate = earliest
    while True:
        move_to: int | None = None
        for resource_id in resources:
            intervals = calendars[resource_id]
            events = [(candidate, 0), (candidate + duration, 0)]
            for start, end in intervals:
                if start < candidate + duration and end > candidate:
                    events.extend(((max(start, candidate), 1), (min(end, candidate + duration), -1)))
            active = 0
            for time, delta in sorted(events, key=lambda item: (item[0], item[1])):
                active += delta
                if active >= capacities[resource_id]:
                    blocking_ends = [end for start, end in intervals if start < candidate + duration and end > candidate]
                    if blocking_ends:
                        target = min(blocking_ends)
                        move_to = target if move_to is None else max(move_to, target)
                    break
        if move_to is None:
            return candidate
        candidate = max(candidate + 1, move_to)


def solve_dispatching(problem: Problem, rule: DispatchRule = "earliest_finish") -> Schedule:
    operations = problem.operation_map()
    capacities = {resource.id: resource.capacity for resource in problem.resources}
    successors: dict[str, list[str]] = defaultdict(list)
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successors[predecessor].append(operation.id)
    remaining = set(operations)
    assignments: dict[str, Assignment] = {}
    calendars: dict[str, list[tuple[int, int]]] = defaultdict(list)
    linked_keys: dict[str, str] = {}
    operation_links = {
        operation_id: link
        for link in problem.choice_links
        for operation_id in link.operation_ids
    }

    while remaining:
        ready = sorted(
            (
                operations[operation_id]
                for operation_id in remaining
                if all(predecessor in assignments for predecessor in operations[operation_id].predecessors)
            ),
            key=lambda operation: operation.id,
        )
        if not ready:
            raise ValueError("problem contains a precedence cycle")
        candidates: list[tuple[Operation, Mode, int, int]] = []
        for operation in ready:
            predecessor_end = max((assignments[item].end for item in operation.predecessors), default=0)
            earliest = max(operation.release, predecessor_end)
            link = operation_links.get(operation.id)
            required_key = linked_keys.get(link.id) if link is not None else None
            for mode in operation.modes:
                if link is not None:
                    mode_key = link.mode_keys.get(mode.id)
                    if mode_key is None or (required_key is not None and mode_key != required_key):
                        continue
                start = _earliest_resource_slot(calendars, capacities, mode.resources, earliest, mode.duration)
                candidates.append((operation, mode, start, start + mode.duration))
        if not candidates:
            raise ValueError("choice-link restrictions left no legal dispatch candidate")

        def key(candidate: tuple[Operation, Mode, int, int]) -> tuple:
            operation, mode, start, end = candidate
            workload = sum(sum(interval_end - interval_start for interval_start, interval_end in calendars[resource]) for resource in mode.resources)
            if rule == "spt":
                score = (mode.duration, end, operation.index)
            elif rule == "lpt":
                score = (-mode.duration, end, operation.index)
            elif rule == "most_successors":
                score = (-len(successors[operation.id]), end, mode.duration)
            elif rule == "balanced":
                score = (end + workload / max(len(mode.resources), 1), end, mode.duration)
            else:
                score = (end, mode.duration, operation.index)
            return (*score, operation.id, mode.id)

        operation, mode, start, end = min(candidates, key=key)
        assignment = Assignment(operation_id=operation.id, mode_id=mode.id, start=start, end=end)
        assignments[operation.id] = assignment
        remaining.remove(operation.id)
        for resource_id in mode.resources:
            calendars[resource_id].append((start, end))
        link = operation_links.get(operation.id)
        if link is not None:
            linked_keys.setdefault(link.id, link.mode_keys[mode.id])
    return Schedule(
        problem_id=problem.id,
        assignments=tuple(assignments[operation.id] for operation in problem.operations),
        metadata={"solver": "dispatching", "rule": rule, "domain_validated": not problem.metadata.get("requires_domain_validation", False)},
    )
