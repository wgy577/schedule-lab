"""Makespan-critical-path analysis for a settled schedule (T2-C critical gate).

Pure analysis helper: reconstruct the schedule's solution graph
(job-order arcs u->v for every u in v.predecessors, plus machine-sequence arcs
between consecutive operations on each capacity-1 resource, ordered by start
time), then run a release-aware forward/backward longest-path pass.  An
operation lies on at least one makespan-critical path iff its total float
(ls - es) is zero (integer times).

Used by ``run_t2c_residual_gpu.py --critical-gate`` to retain only pool
candidates that touch the makespan-critical path.  This module is
analysis-only -- it changes NO architecture and NO parameter count.  The arc
set is built for JSP/FJSP single-resource modes (capacity-1 machines); modes
with multiple resources are each represented by their first resource and
capacity>1 machines do not impose a total-order machine arc (their cumulative
constraint cannot be reduced to a pair-ordering).
"""

from __future__ import annotations

from typing import Any


def _x(x: Any) -> Any:  # noqa: ANN401 -- pydantic export convenience
    return x


def critical_path_info(problem: Any, schedule: Any) -> dict[str, Any]:
    """Return per-operation slack + the makespan-critical operation set.

    Returns::

        {"makespan", "n_ops", "n_critical", "critical_ops" (frozenset),
         "op_meta": {op_id: {"machine", "start", "end", "dur",
                             "es","ef","ls","lf","slack","on_critical"}}}

    Feasibility is *relied upon*, not checked: the ops are topologically
    ordered by their scheduled start time (a feasible schedule guarantees that a
    job/machine predecessor starts strictly earlier than its successor).
    """
    asm_map = schedule.assignment_map()
    ops = [op for op in problem.operations if op.id in asm_map]
    mode_map = problem.mode_map()
    op_map = problem.operation_map()
    if not ops:
        return {"makespan": int(schedule.makespan), "n_ops": 0, "n_critical": 0,
                "critical_ops": frozenset(), "op_meta": {}}

    dur = {op.id: asm_map[op.id].end - asm_map[op.id].start for op in ops}
    start = {op.id: asm_map[op.id].start for op in ops}
    machine: dict[str, str] = {}
    for op in ops:
        _, mode = mode_map[asm_map[op.id].mode_id]
        machine[op.id] = mode.resources[0]

    # ---- solution-graph arcs -------------------------------------------------
    # job-order succ
    succ: dict[str, set[str]] = {op.id: set() for op in ops}
    for op in ops:
        for p in op.predecessors:
            if p in asm_map:
                succ[p].add(op.id)
    # machine-sequence arcs (capacity-1 resources only)
    res_cap = {r.id: r.capacity for r in problem.resources}
    by_res: dict[str, list[str]] = {}
    for op in ops:
        _, mode = mode_map[asm_map[op.id].mode_id]
        for r in mode.resources:
            by_res.setdefault(r, []).append(op.id)
    for rid, ids in by_res.items():
        if res_cap.get(rid, 1) > 1:
            continue
        ids.sort(key=lambda oid: (start[oid], dur[oid], oid))
        for a, b in zip(ids, ids[1:]):
            succ[a].add(b)

    pred: dict[str, list[str]] = {op.id: [] for op in ops}
    order_ids = sorted(ops, key=lambda op: (start[op.id], dur[op.id], op.id))
    order = [op.id for op in order_ids]
    for u in order:
        for v in succ[u]:
            pred[v].append(u)

    # ---- forward: earliest start/finish (release-aware) ----------------------
    es: dict[str, int] = {}
    ef: dict[str, int] = {}
    for oid in order:
        m = op_map[oid].release
        for u in pred[oid]:
            if ef[u] > m:
                m = ef[u]
        es[oid] = m
        ef[oid] = m + dur[oid]

    # ---- backward: latest finish/start from the schedule makespan ------------
    Cmax = int(schedule.makespan)
    lf: dict[str, int] = {}
    ls: dict[str, int] = {}
    for oid in reversed(order):
        L = Cmax
        for v in succ[oid]:
            if ls[v] < L:
                L = ls[v]
        lf[oid] = L
        ls[oid] = L - dur[oid]

    critical_ops = {oid for oid in order if ls[oid] == es[oid]}
    op_meta = {
        oid: {
            "machine": machine[oid],
            "start": start[oid],
            "end": asm_map[oid].end,
            "dur": dur[oid],
            "es": es[oid], "ef": ef[oid],
            "ls": ls[oid], "lf": lf[oid],
            "slack": ls[oid] - es[oid],
            "on_critical": oid in critical_ops,
        }
        for oid in order
    }
    return {"makespan": Cmax, "n_ops": len(order), "n_critical": len(critical_ops),
            "critical_ops": frozenset(critical_ops), "op_meta": op_meta}