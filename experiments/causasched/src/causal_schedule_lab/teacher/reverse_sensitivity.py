"""Max-Plus Reverse Sensitivity (教师模型.md §7/§8/§9).

Given the realized propagation graph and an appearance-specific seed, compute

    R_A(u) = max_{u -> v} [ T_uv * R_A(v) ]

via a reverse-topological walk.  ``R_A(u)`` answers *"if a standard time
perturbation occurred near u, how strongly would it propagate along the current
realized schedule to the target appearance"*.  It is a propagation prior, not a
root-cause truth (教师模型.md §7).

Hard propagation constraints (§9): we never propagate along representation
edges, indirect edges, eligible/assigned edges, or appearance-special edges.
``G_prop`` contains only job-direct + machine-direct realized edges, so this is
enforced by construction.

Appearance-specific seeds (§8):
* A1: the late-starting right end ``O_b`` of the block -> 1.0.
* A4: the waiting successor ``O_j`` -> 1.0 (weak 0.1 on the predecessor ``O_i``).
* A6: the assignment-symptom operation ``O_i`` -> 1.0.
* A23: block operations weighted by ``q_o`` (normalized share-style weight).
Multiple seeds are combined with a max aggregate (多 seed max 聚合).
"""

from __future__ import annotations

from typing import Iterable

from .propagation_graph import RealizedPropagationGraph

# Weak seed on the partner operation of an appearance (A1/A4).
ETA_A1 = 0.2
ETA_A4 = 0.1


def _topological_order(graph: RealizedPropagationGraph) -> list[str]:
    """Deterministic topological order of the realized DAG (Kahn's algorithm)."""
    indegree = {node: 0 for node in graph.nodes}
    for (u, v) in graph.edges:
        indegree[v] += 1
    ready = sorted([node for node, degree in indegree.items() if degree == 0])
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in graph.successors(node):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(graph.nodes):
        raise ValueError("realized propagation graph is unexpectedly cyclic")
    return order


def reverse_sensitivity(
    graph: RealizedPropagationGraph,
    seeds: dict[str, float],
) -> dict[str, float]:
    """Max-Plus reverse propagation over ``G_prop`` seeded by ``seeds``."""
    R: dict[str, float] = {node: 0.0 for node in graph.nodes}
    for node, value in seeds.items():
        if node not in R:
            continue
        R[node] = max(R[node], float(value))

    order = _topological_order(graph)
    for node in reversed(order):
        current = R[node]
        for predecessor in graph.predecessors(node):
            candidate = graph.coefficient[(predecessor, node)] * current
            if candidate > R[predecessor]:
                R[predecessor] = candidate
    return R


def appearance_specific_seed(
    block_operations: Iterable[str],
    appearance_type: str,
    *,
    block_weights: dict[str, float] | None = None,
    eta_a1: float = ETA_A1,
    eta_a4: float = ETA_A4,
    allow_legacy_a6: bool = False,
) -> dict[str, float]:
    """Appearance-specific seed values over the block's operations (§8).

    ``appearance_type`` is the primary rule (A1/A4/A6/A23).  ``block_weights``
    optionally carries the ``q_o`` weights for A23; if absent, A23 seeds every
    block operation at 1.0 (a uniform fallback).

    Phase 2.6 (改A6.md §6): A6 is auxiliary-only in the main ACCT pipeline and
    must NOT act as a standalone reverse-sensitivity seed.  Calling it raises
    unless ``allow_legacy_a6=True`` (legacy diagnostic only).
    """
    ops = list(block_operations)
    if not ops:
        return {}
    rule = (appearance_type or "").upper()
    if rule == "A1":
        # Right end O_b is the late starter; give the rest a weak seed.
        seeds = {op: eta_a1 for op in ops}
        seeds[ops[-1]] = 1.0
        return seeds
    if rule == "A4":
        # Waiting successor O_j seeds at 1.0; predecessor O_i weak.
        seeds = {op: eta_a4 for op in ops}
        seeds[ops[-1]] = 1.0
        return seeds
    if rule == "A6":
        if allow_legacy_a6:
            return {op: 1.0 for op in ops}
        raise ValueError(
            "A6 is auxiliary-only in the main ACCT pipeline; "
            "standalone A6 seeding is disabled (改A6.md §6)."
        )
    if rule == "A23":
        if block_weights:
            return {op: float(block_weights.get(op, 0.0)) for op in ops}
        return {op: 1.0 for op in ops}
    # Unknown rule: uniform seed over the block.
    return {op: 1.0 for op in ops}