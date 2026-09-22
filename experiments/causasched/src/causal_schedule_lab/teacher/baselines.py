"""Candidate-source baselines for Phase-2 ablation (教师模型.md §30).

Compares the proposed Max-Plus Reverse Sensitivity against cheaper priors that
produce the same shape of output (a relevance value per operation, which the
shared atom generator and ranker consume):

* ``random_relevance`` — uniform/random relevance (ablation A).
* ``dense_page_rank_relevance`` — standard PageRank over the *dense*
  representation adjacency (job + machine, symmetric), the "dense graph
  PageRank" baseline (ablation B).  The doc predicts this underperforms the
  scheduling-specific propagation because scheduling start times are Max-Plus,
  not a probabilistic random walk.

LLM sources (E / F) are out of Phase-2's scope (no LLM until Phase 3).
Exploration (F) is evaluated by mixing a small exploration slice into the
teacher top-K, exactly as in Phase 1.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..ir import Problem, Schedule
from .propagation_graph import RealizedPropagationGraph


def random_relevance(
    graph: RealizedPropagationGraph,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Uniform random relevance in [0, 1] per operation (ablation A)."""
    rng = random.Random(seed)
    return {node: rng.random() for node in graph.nodes}


def dense_page_rank_relevance(
    problem: Problem,
    schedule: Schedule,
    graph: RealizedPropagationGraph,
    *,
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """PageRank over the dense job+machine adjacency (ablation B).

    Builds a symmetric adjacency: two operations are adjacent when they share a
    job (job-precedence-reachable) or a machine (realized sequence).  This is
    the over-dense representation the doc argues Max-Plus should beat.
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency: set[tuple[int, int]] = set()

    # Machine adjacency: realized sequence neighbours.
    machine_seq: dict[str, list[str]] = defaultdict(list)
    for (u, v) in graph.edges:
        # machine_direct edges already encode realized machine sequence.
        pass
    # Rebuild machine sequences from the schedule directly.
    mode_map = problem.mode_map()
    by_resource: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for assignment in schedule.assignments:
        mode = mode_map[assignment.mode_id][1]
        for resource_id in mode.resources:
            by_resource[resource_id].append((assignment.operation_id, assignment.start))
    for resource_id, items in by_resource.items():
        ordered = sorted(items, key=lambda it: (it[1], it[0]))
        for left, right in zip(ordered, ordered[1:]):
            if left[0] in index and right[0] in index:
                adjacency.add((index[left[0]], index[right[0]]))
                adjacency.add((index[right[0]], index[left[0]]))

    # Job adjacency: same job, any pair.
    job_of: dict[str, str] = {}
    for operation in problem.operations:
        job_of[operation.id] = operation.job_id
    job_members: dict[str, list[int]] = defaultdict(list)
    for node in nodes:
        job_members[job_of.get(node, "")].append(index[node])
    for members in job_members.values():
        for i in members:
            for j in members:
                if i != j:
                    adjacency.add((i, j))

    out_degree = [0] * n
    for (i, j) in adjacency:
        out_degree[i] += 1

    rank = [1.0 / n] * n
    for _ in range(iterations):
        new_rank = [(1.0 - damping) / n] * n
        for (i, j) in adjacency:
            if out_degree[i] > 0:
                new_rank[j] += damping * rank[i] / out_degree[i]
        # Teleport to uniform for dangling nodes.
        dangling = sum(rank[i] for i in range(n) if out_degree[i] == 0)
        add = damping * dangling / n
        new_rank = [v + add for v in new_rank]
        diff = sum(abs(a - b) for a, b in zip(rank, new_rank))
        rank = new_rank
        if diff < tolerance:
            break

    # Normalize to [0, 1] as a relevance proxy.
    m = max(rank) if rank else 1.0
    return {node: (rank[i] / m if m > 0 else 0.0) for i, node in enumerate(nodes)}