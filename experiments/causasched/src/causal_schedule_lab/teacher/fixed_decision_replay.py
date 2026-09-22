"""Fixed-Decision Schedule Replay (Phase2改.md §7/§16).

When the discrete decisions of a counterfactual are fully known -- every
operation's mode and every machine's operation order -- the resulting schedule
is the unique longest-path realisation of the *realised disjunctive graph*:

    E = E_J  ∪  E_M^{cf}

where ``E_J`` are job-precedence edges and ``E_M^{cf}`` are the counterfactual
machine-order edges.  Processing each operation in topological order,

    S_v = max(release_v,  C_{job-pred},  C_{machine-pred})
    C_v = S_v + p_v

yields start/end times with **no optimisation solver involved**: no mode is
re-chosen, no machine order is re-ranked.  If the counterfactual machine orders
together with job precedence form a cycle, the intervention is infeasible and
``None`` is returned (fail-closed: no global repair, §16).

This is the atomic counterfactual realiser.  It is deliberately separate from
``DeterministicOperatorExecutor`` / CP-SAT, which remain the M3 / repair path.
"""

from __future__ import annotations

from typing import Any

from ..ir import Assignment, Problem, Schedule

REPLAY_VERSION = "fixed-decision-replay-1.0"


def observed_sequences(problem: Problem, schedule: Schedule) -> dict[str, list[str]]:
    """Per-resource operation order in the incumbent, sorted by (start, end, id)."""
    mode_map = problem.mode_map()
    assignment_map = schedule.assignment_map()
    seq: dict[str, list[str]] = {}
    for assignment in schedule.assignments:
        mode = mode_map[assignment.mode_id][1]
        for resource_id in mode.resources:
            seq.setdefault(resource_id, []).append(assignment.operation_id)
    for resource_id, ops in seq.items():
        ops.sort(key=lambda o: (assignment_map[o].start, assignment_map[o].end, o))
    return seq


def observed_mode_ids(schedule: Schedule) -> dict[str, str]:
    """``operation_id -> mode_id`` for every assigned operation."""
    return {item.operation_id: item.mode_id for item in schedule.assignments}


def replay_fixed_decisions(
    problem: Problem,
    incumbent: Schedule,
    machine_sequences_cf: dict[str, list[str]],
    mode_ids_cf: dict[str, str],
) -> Schedule | None:
    """Realise a fixed-decision counterfactual by longest-path replay.

    ``machine_sequences_cf`` is the *complete* per-resource operation order
    (observed on non-target machines, edited on the target machine).  ``mode_ids_cf``
    is the *complete* operation->mode map (observed everywhere except a routing
    target).  Returns the realised ``Schedule`` or ``None`` if the orders induce a
    cycle with job precedence (infeasible intervention, fail-closed).
    """
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()

    # Validate the cf decisions are well-formed and collect the op universe.
    ops: set[str] = set()
    for op_id, mode_id in mode_ids_cf.items():
        if op_id not in operation_map:
            return None
        if mode_id not in mode_map:
            return None
        ops.add(op_id)
    # Each op in a cf machine sequence must be assigned a cf mode whose resources
    # include that machine; otherwise the order is inconsistent.
    for resource_id, order in machine_sequences_cf.items():
        for op_id in order:
            if op_id not in mode_ids_cf:
                return None
            mode = mode_map[mode_ids_cf[op_id]][1]
            if resource_id not in mode.resources:
                return None

    # Build machine-predecessor map: op -> list of (op immediately before it on a
    # resource it uses).  An op using several resources has one pred per resource.
    machine_pred: dict[str, list[str]] = {op: [] for op in ops}
    for resource_id, order in machine_sequences_cf.items():
        for prev, cur in zip(order, order[1:]):
            machine_pred.setdefault(cur, []).append(prev)

    # Job-predecessor map (only predecessors that are themselves scheduled).
    job_pred: dict[str, list[str]] = {}
    for op_id in ops:
        job_pred[op_id] = [p for p in operation_map[op_id].predecessors if p in ops]

    # In-degree (number of distinct predecessors) for Kahn topological sort.
    preds: dict[str, set[str]] = {
        op: set(job_pred[op]) | set(machine_pred[op]) for op in ops
    }
    in_degree = {op: len(preds[op]) for op in ops}
    successors: dict[str, list[str]] = {op: [] for op in ops}
    for op in ops:
        for p in preds[op]:
            successors[p].append(op)

    # Kahn's algorithm with longest-path completion times.
    from collections import deque

    queue: deque[str] = deque(op for op in ops if in_degree[op] == 0)
    start: dict[str, int] = {}
    completion: dict[str, int] = {}
    processed = 0
    while queue:
        op = queue.popleft()
        operation = operation_map[op]
        duration = mode_map[mode_ids_cf[op]][1].duration
        earliest = int(operation.release)
        for p in preds[op]:
            if p in completion:
                earliest = max(earliest, completion[p])
        start[op] = earliest
        completion[op] = earliest + duration
        processed += 1
        for s in successors[op]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    if processed != len(ops):
        # A cycle exists between the cf machine orders and job precedence.  The
        # intervention is infeasible; do not attempt a global repair (§16).
        return None

    assignments = tuple(
        Assignment(
            operation_id=op,
            mode_id=mode_ids_cf[op],
            start=start[op],
            end=completion[op],
            provenance="atomic-replay",
        )
        for op in sorted(ops)
    )
    return Schedule(problem_id=problem.id, assignments=assignments,
                    metadata={"replay": REPLAY_VERSION, "source": "fixed-decision-replay"})


def sequence_adjacent_pairs(sequence: list[str]) -> set[tuple[str, str]]:
    """Adjacent (left, right) op pairs on one machine sequence."""
    return set(zip(sequence, sequence[1:]))


def machine_of(problem: Problem, schedule: Schedule, op: str) -> tuple[str, ...] | None:
    """The resource tuple of the op's assigned mode, or ``None`` if unassigned."""
    mode_map = problem.mode_map()
    assignment = schedule.assignment_map().get(op)
    if assignment is None or assignment.mode_id not in mode_map:
        return None
    return tuple(mode_map[assignment.mode_id][1].resources)


__all__ = [
    "REPLAY_VERSION",
    "observed_sequences",
    "observed_mode_ids",
    "replay_fixed_decisions",
    "sequence_adjacent_pairs",
    "machine_of",
]
