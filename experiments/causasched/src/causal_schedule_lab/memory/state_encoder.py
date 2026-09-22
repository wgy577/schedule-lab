"""Patch §5 -- scheduling-state encoder.

Produces :class:`StateFeatures` (the ``S`` of :math:`(S,P,S')`) from a schedule:

* ``cmax`` -- makespan, and ``critical_machine`` -- the machine whose busy
  window reaches the makespan (the makespan-bounding resource).
* ``load_features`` -- per-machine total processing / busy-window fraction.
* ``gap_features`` -- per-machine idle gaps between consecutive operations.
* optional ``graph_embedding`` / ``appearance_embedding`` -- filled by a caller
  when an M1 encoder is available (Patch Phase-D wires the learned encoder); not
  required for retrieval, which uses the deterministic numeric
  :func:`state_vector`.

All numeric state is computed deterministically from the schedule alone (no
solver, no model) -- it is the cheap, invariant state summary the memory keys
on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev

from ..ir import Problem, Schedule


@dataclass
class StateFeatures:
    """Deterministic scheduling-state summary (S of (S,P,S')).

    ``cmax`` / ``critical_machine`` / ``machine_load`` are always computed by
    :func:`encode_state`; the Patch V2 spec (§5.2) adds a ``critical_path_length``
    (defaults to ``cmax`` -- the makespan-bounding path) and an *appearance
    block* summary (``appearance_type`` / ``overloaded_machine`` /
    ``underloaded_machine`` / ``load_difference`` / ``imbalance_score``) that a
    caller (the appearance layer) fills when available.  All are optional so the
    deterministic 5-dim :func:`state_vector` contract is unchanged.
    """

    cmax: float = 0.0
    # Frozen global gap G = m*Cmax - sum_i p_i for the selected assignments.
    gap: float = 0.0
    critical_machine: str | None = None
    # machine -> (processing_time, busy_window, gap_total, idle_spans count)
    machine_load: dict[str, tuple[float, float, float, int]] = field(default_factory=dict)
    critical_path_length: float = 0.0
    appearance_type: str = ""
    overloaded_machine: str | None = None
    underloaded_machine: str | None = None
    load_difference: float = 0.0
    imbalance_score: float = 0.0
    machine_load_vector: tuple[float, ...] = ()
    load_variance: float = 0.0
    appearance_score: float = 0.0
    local_features: tuple[float, ...] = ()
    local_operation_nodes: tuple[str, ...] = ()
    local_machine_nodes: tuple[str, ...] = ()
    local_mode_nodes: tuple[str, ...] = ()
    local_precedence_edges: tuple[tuple[str, str], ...] = ()
    local_resource_sequence_edges: tuple[tuple[str, str], ...] = ()
    # optional encoder-produced embeddings (filled by a caller, not required)
    graph_embedding: tuple[float, ...] = ()
    appearance_embedding: tuple[float, ...] = ()


def encode_state(
    problem: Problem,
    schedule: Schedule,
    *,
    graph_embedding: tuple[float, ...] = (),
    appearance_embedding: tuple[float, ...] = (),
    appearance_type: str = "",
    appearance_score: float = 0.0,
    local_features: tuple[float, ...] = (),
    local_operation_nodes: tuple[str, ...] = (),
    local_machine_nodes: tuple[str, ...] = (),
    local_mode_nodes: tuple[str, ...] = (),
    local_precedence_edges: tuple[tuple[str, str], ...] = (),
    local_resource_sequence_edges: tuple[tuple[str, str], ...] = (),
) -> StateFeatures:
    """Deterministic schedule-state summary (no solver / model)."""
    asg = schedule.assignment_map()
    mode_map = problem.mode_map()
    by_machine: dict[str, list[tuple]] = {}
    cmax = 0.0
    for oid, a in asg.items():
        mode = mode_map[a.mode_id][1]  # (operation, mode)
        m = mode.resources[0]
        by_machine.setdefault(m, []).append((a.start, a.end, oid))
        cmax = max(cmax, float(a.end))

    machine_load: dict[str, tuple[float, float, float, int]] = {}
    critical: str | None = None
    max_end = -1.0
    for m, rows in by_machine.items():
        rows.sort(key=lambda t: (t[0], t[1]))
        busy = sum((e - s) for s, e, _ in rows)
        window = (rows[-1][1] - rows[0][0]) if rows else 0.0
        gaps = 0.0
        idle_spans = 0
        for (s0, e0, _), (s1, e1, _) in zip(rows, rows[1:]):
            g = s1 - e0
            if g > 0:
                gaps += g
                idle_spans += 1
        machine_load[m] = (float(busy), float(window), float(gaps), idle_spans)
        # Makespan is determined by the machine carrying an operation ending at
        # Cmax; a late-starting critical machine need not have window == Cmax.
        row_end = float(rows[-1][1]) if rows else 0.0
        if row_end >= cmax - 1e-9 and row_end >= max_end:
            max_end = row_end
            critical = m
    if cmax <= 0:
        critical = None
    total_processing = sum(row[0] for row in machine_load.values())
    gap = len(problem.resources) * cmax - total_processing
    load_vector = tuple(machine_load.get(r.id, (0.0, 0.0, 0.0, 0))[0]
                        for r in problem.resources)
    load_variance = pstdev(load_vector) ** 2 if len(load_vector) > 1 else 0.0
    return StateFeatures(
        cmax=float(cmax),
        gap=float(gap),
        critical_machine=critical,
        machine_load=machine_load,
        critical_path_length=float(cmax),  # makespan-bounding path (Patch V2 §5.2)
        appearance_type=appearance_type,
        appearance_score=float(appearance_score),
        local_features=tuple(local_features),
        local_operation_nodes=tuple(local_operation_nodes),
        local_machine_nodes=tuple(local_machine_nodes),
        local_mode_nodes=tuple(local_mode_nodes),
        local_precedence_edges=tuple(local_precedence_edges),
        local_resource_sequence_edges=tuple(local_resource_sequence_edges),
        machine_load_vector=tuple(float(x) for x in load_vector),
        load_variance=float(load_variance),
        graph_embedding=tuple(graph_embedding),
        appearance_embedding=tuple(appearance_embedding),
    )


def state_vector(features: StateFeatures) -> tuple[float, ...]:
    """Deterministic fixed-length numeric vector for similarity retrieval.

    Components (each normalised to be ~[0,1] at typical scales):

    * ``cmax_scale`` -- log-scaled makespan (unlike the old constant 1.0).
    * ``busyfrac_critical`` -- critical machine's busy fraction of cmax
      (~how packed the bounding resource is).
    * ``mean_gap_frac`` -- mean idle gap / cmax (spread of idle).
    * ``gap_spread`` -- idleness imbalance across machines (std of gap fracs).
    * ``load_imbalance`` -- pstdev of machine busy fractions (tightness spread).
    """
    mach = features.machine_load
    n = len(mach)
    cmax = max(features.cmax, 1.0)
    busy_fracs = [v[0] / cmax for v in mach.values()]
    gap_fracs = [v[2] / cmax for v in mach.values()]
    import math
    if not mach:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    crit = mach.get(features.critical_machine, (0.0, 0.0, 0.0, 0))
    crit_busy = crit[0] / cmax
    mean_gap = sum(gap_fracs) / n
    return (
        float(math.log1p(features.cmax) / 10.0),
        float(crit_busy),
        float(features.gap / max(len(mach) * cmax, 1.0)),
        float(pstdev(gap_fracs) if n > 1 else 0.0),
        float(pstdev(busy_fracs) if n > 1 else 0.0),
    )


__all__ = ["StateFeatures", "encode_state", "state_vector"]
