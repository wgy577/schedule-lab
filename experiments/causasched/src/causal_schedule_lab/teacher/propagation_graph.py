"""Realized propagation graph ``G_prop`` (教师模型.md §6).

``G_prop`` is the *current realized schedule* viewed as a propagation DAG:

* nodes = assigned operations;
* edges = job-direct precedence (``job_pred -> op``) and machine-direct realized
  sequence (``prev_on_machine -> next_on_machine``);
* each directed edge ``u -> v`` carries the current time gap
  ``g_uv = S_v - C_u`` and the standard-perturbation propagation coefficient

      T_uv = max(0, delta - g_uv) / delta

  with ``delta = p_scale = Median{p_o : p_o > 0}`` over the assigned processing
  times.  ``T_uv`` is 1 when the upstream delay transmits almost fully, partial
  in ``(0, 1)`` when the buffer absorbs some of it, and 0 once the gap reaches
  the standard perturbation.

Edges with ``T_uv < prune_threshold`` (default 0.05) are dropped from the
teacher propagation walk (教师模型.md §23).

Because every edge ``u -> v`` satisfies ``S_v >= C_u > S_u``, the graph is
guaranteed acyclic, so reverse-topological propagation is well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Problem, Schedule
from ..validation import schedule_hash

EDGE_PRUNE_THRESHOLD = 0.05


@dataclass(frozen=True)
class RealizedPropagationGraph:
    """The realized propagation DAG with per-edge gaps and coefficients."""

    problem_id: str
    schedule_hash: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    starts: dict[str, int]
    completions: dict[str, int]
    gap: dict[tuple[str, str], float]
    coefficient: dict[tuple[str, str], float]
    delta: float
    edge_kind: dict[tuple[str, str], str]
    prune_threshold: float = EDGE_PRUNE_THRESHOLD

    def predecessors(self, node: str) -> tuple[str, ...]:
        """Direct propagation predecessors ``u`` with ``u -> node``."""
        return tuple(u for (u, v) in self.edges if v == node)

    def successors(self, node: str) -> tuple[str, ...]:
        """Direct propagation successors ``v`` with ``node -> v``."""
        return tuple(v for (u, v) in self.edges if u == node)

    def as_record(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "schedule_hash": self.schedule_hash,
            "delta": self.delta,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "kind": self.edge_kind[(u, v)],
                    "gap": self.gap[(u, v)],
                    "T": self.coefficient[(u, v)],
                }
                for (u, v) in self.edges
            ],
        }


def processing_time_scale(problem: Problem, schedule: Schedule) -> float:
    """``p_scale = Median{p_o : p_o > 0}`` over assigned modes (§6/§23)."""
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    durations: list[float] = []
    for assignment in schedule.assignments:
        mode = mode_map[assignment.mode_id][1]
        processing = mode.duration
        if processing is not None and processing > 0:
            durations.append(float(processing))
    if not durations:
        # Fall back to the first mode's duration of any assigned op.
        for assignment in schedule.assignments:
            operation = operation_map.get(assignment.operation_id)
            if operation is None or not operation.modes:
                continue
            for mode in operation.modes:
                if mode.duration and mode.duration > 0:
                    durations.append(float(mode.duration))
                    break
    if not durations:
        return 1.0
    ordered = sorted(durations)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def build_realized_propagation_graph(
    problem: Problem,
    schedule: Schedule,
    *,
    delta: float | None = None,
    prune_threshold: float = EDGE_PRUNE_THRESHOLD,
) -> RealizedPropagationGraph:
    """Build ``G_prop`` from the realized job-direct + machine-direct schedule.

    Only operations that actually have an assignment participate.  Machine
    sequences are grouped by the resource(s) each assigned mode occupies and
    sorted by ``(start, end, operation_id)`` for determinism.
    """
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    assignment_map = schedule.assignment_map()

    starts: dict[str, int] = {}
    completions: dict[str, int] = {}
    for operation_id, assignment in assignment_map.items():
        starts[operation_id] = int(assignment.start)
        completions[operation_id] = int(assignment.end)

    if delta is None:
        delta = processing_time_scale(problem, schedule)

    edges: list[tuple[tuple[str, str], str]] = []

    # Job-direct precedence: op -> its next job operation (realized).
    assigned = set(assignment_map)
    for operation in problem.operations:
        if operation.id not in assigned:
            continue
        for predecessor in operation.predecessors:
            if predecessor in assigned:
                edges.append(((predecessor, operation.id), "job_direct"))

    # Machine-direct realized sequence: consecutive ops on a shared resource.
    by_resource: dict[str, list[Any]] = {}
    for operation_id, assignment in assignment_map.items():
        mode = mode_map[assignment.mode_id][1]
        for resource_id in mode.resources:
            by_resource.setdefault(resource_id, []).append((operation_id, assignment))
    for resource_id, items in sorted(by_resource.items()):
        ordered = sorted(items, key=lambda item: (item[1].start, item[1].end, item[0]))
        for left, right in zip(ordered, ordered[1:]):
            edges.append(((left[0], right[0]), "machine_direct"))

    # Deduplicate (identical job + machine edge between the same pair).
    seen: dict[tuple[str, str], str] = {}
    for (u, v), kind in edges:
        seen.setdefault((u, v), kind)

    gap: dict[tuple[str, str], float] = {}
    coefficient: dict[tuple[str, str], float] = {}
    edge_kind: dict[tuple[str, str], str] = {}
    kept: list[tuple[str, str]] = []
    for (u, v), kind in sorted(seen.items()):
        g = float(starts[v] - completions[u])
        t = max(0.0, delta - g) / delta
        gap[(u, v)] = g
        coefficient[(u, v)] = t
        edge_kind[(u, v)] = kind
        if t >= prune_threshold:
            kept.append((u, v))

    return RealizedPropagationGraph(
        problem_id=problem.id,
        schedule_hash=schedule_hash(schedule),
        nodes=tuple(sorted(assignment_map)),
        edges=tuple(kept),
        starts=starts,
        completions=completions,
        gap=gap,
        coefficient=coefficient,
        delta=delta,
        edge_kind=edge_kind,
        prune_threshold=prune_threshold,
    )