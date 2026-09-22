"""Deterministic structural legality for executable route/sequence edits.

This is the pre-emit filter that stops the Transition Reasoner from composing
enabling chains that are deterministically cyclic.  It reuses the executor's
cycle semantics -- job precedence + machine order, Kahn topological sort --
with **no schedule optimisation**.  Final schedulability remains the authority
of the Frozen-Local executor; this checker only answers the narrower
question "is the composed structure deterministically illegal?".

Why: ``InterventionTransitionReasoner.check_feasibility`` historically set
``acyclic = eligibility`` (a placeholder).  The Reasoner composed
relocate-blocker chains purely on target-window occupancy, never checking
whether the composed machine orders form a directed cycle with job precedence,
so 229/229 emitted 2-edit chains reached the executor and were rejected.

Semantics (matches ``replay_fixed_decisions`` for unary-resource modes, which
``ScheduleGraphView.from_problem_schedule`` guarantees):
  * job precedence edges:  op -> its scheduled job predecessors
  * machine-order edges:  op -> the op immediately after it on its machine
  * a directed cycle in the union => CYCLE_WITH_JOB_PRECEDENCE
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    LegalEdit,
)
from .transition_reasoner_v1 import ScheduleGraphView

OK = "OK"
CYCLE_WITH_JOB_PRECEDENCE = "CYCLE_WITH_JOB_PRECEDENCE"
INVALID_RESOURCE = "INVALID_RESOURCE"
MODE_CONFLICT = "MODE_CONFLICT"
DUPLICATE_OPERATION = "DUPLICATE_OPERATION"
OTHER_DETERMINISTIC_CONFLICT = "OTHER_DETERMINISTIC_CONFLICT"


@dataclass(frozen=True)
class CompositeLegality:
    legal: bool
    reason: str


def _observed_machine_sequences(
    schedule_graph: ScheduleGraphView,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Rebuild per-machine op order + op->start from the graph view.

    ``ScheduleGraphView.intervals`` is already sorted by (machine, start, op),
    which is exactly the observed machine order.
    """
    seq: dict[str, list[str]] = {}
    start: dict[str, float] = {}
    for row in schedule_graph.intervals:
        seq.setdefault(row.machine_id, []).append(row.operation_id)
        start[row.operation_id] = row.start
    return seq, start


def check_composite_structural_legality(
    schedule_graph: ScheduleGraphView,
    edits: Sequence[LegalEdit],
) -> CompositeLegality:
    """Return whether ``edits`` composed onto the baseline are structurally legal.

    ROUTE keeps its historical insertion-by-observed-start semantics. SEQ_SWAP
    and SEQ_INSERT mirror the AtomicCounterfactualExecutor transformations. The
    final Kahn pass rejects global cycles that the local enumerator cannot see.
    """
    seq, start = _observed_machine_sequences(schedule_graph)
    seq = {k: list(v) for k, v in seq.items()}

    seen_ops: set[str] = set()
    for edit in edits:
        op = edit.operation_id
        if edit.edit_type == EDIT_ROUTE:
            if op in seen_ops:
                return CompositeLegality(False, DUPLICATE_OPERATION)
            seen_ops.add(op)
            if edit.target_machine is None or (op, edit.target_machine) not in schedule_graph.target_durations:
                return CompositeLegality(False, INVALID_RESOURCE)
            src = edit.source_machine
            tgt = edit.target_machine
            if src in seq and op in seq[src]:
                seq[src] = [o for o in seq[src] if o != op]
            tgt_seq = seq.setdefault(tgt, [])
            if op not in tgt_seq:
                op_start = start.get(op, float("inf"))
                insert_at = len(tgt_seq)
                for idx, other in enumerate(tgt_seq):
                    if start.get(other, float("inf")) >= op_start:
                        insert_at = idx
                        break
                tgt_seq.insert(insert_at, op)
            seq[tgt] = tgt_seq
        elif edit.edit_type == EDIT_SEQ_SWAP:
            order = seq.get(edit.resource_id)
            if order is None or edit.left_id not in order or edit.right_id not in order:
                return CompositeLegality(False, INVALID_RESOURCE)
            left = order.index(edit.left_id)
            if left + 1 >= len(order) or order[left + 1] != edit.right_id:
                return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)
            order[left], order[left + 1] = order[left + 1], order[left]
        elif edit.edit_type == EDIT_SEQ_INSERT:
            order = seq.get(edit.resource_id)
            if order is None or op not in order:
                return CompositeLegality(False, INVALID_RESOURCE)
            order.remove(op)
            if edit.successor_id is not None:
                if edit.successor_id not in order:
                    return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)
                order.insert(order.index(edit.successor_id), op)
            elif edit.predecessor_id is not None:
                if edit.predecessor_id not in order:
                    return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)
                order.insert(order.index(edit.predecessor_id) + 1, op)
            else:
                order.insert(0, op)
            idx = order.index(op)
            if (edit.predecessor_id is not None and
                    (idx == 0 or order[idx - 1] != edit.predecessor_id)):
                return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)
            if (edit.successor_id is not None and
                    (idx + 1 >= len(order) or order[idx + 1] != edit.successor_id)):
                return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)
        else:
            return CompositeLegality(False, OTHER_DETERMINISTIC_CONFLICT)

    # Job precedence + machine order -> Kahn topological sort.
    ops: set[str] = set(start.keys())
    for edit in edits:
        ops.add(edit.operation_id)

    job_pred: dict[str, set[str]] = {
        op: set(schedule_graph.predecessors.get(op, ())) & ops for op in ops
    }
    machine_pred: dict[str, set[str]] = {op: set() for op in ops}
    for order in seq.values():
        for prev, cur in zip(order, order[1:]):
            machine_pred.setdefault(cur, set()).add(prev)

    preds: dict[str, set[str]] = {
        op: job_pred[op] | machine_pred.get(op, set()) for op in ops
    }
    in_degree = {op: len(preds[op]) for op in ops}
    successors: dict[str, list[str]] = {op: [] for op in ops}
    for op in ops:
        for p in preds[op]:
            successors[p].append(op)

    queue = deque(op for op in ops if in_degree[op] == 0)
    processed = 0
    while queue:
        op = queue.popleft()
        processed += 1
        for s in successors[op]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    if processed != len(ops):
        return CompositeLegality(False, CYCLE_WITH_JOB_PRECEDENCE)
    return CompositeLegality(True, OK)


__all__ = [
    "CompositeLegality",
    "check_composite_structural_legality",
    "OK",
    "CYCLE_WITH_JOB_PRECEDENCE",
    "INVALID_RESOURCE",
    "MODE_CONFLICT",
    "DUPLICATE_OPERATION",
    "OTHER_DETERMINISTIC_CONFLICT",
]
