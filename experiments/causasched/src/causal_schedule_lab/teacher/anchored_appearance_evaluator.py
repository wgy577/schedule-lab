"""Anchored appearance evaluation -- B5 label contract round 1 (2026-08-26).

Primary label identity is the ORIGINAL block (member set + appearance rule +
detection config).  After an atomic intervention we do NOT re-diagnose and
rematch: we re-evaluate the original rule's quantity against the original
member set on the new schedule ("anchored re-evaluation").  If the anchor
dissolves (e.g. an A1 pair's ops moved off the machine), the anchored value is
0 -- the original appearance is measurably eliminated, which is a valid label,
not a lost one.

Contract (frozen in docs/T1_MODEL_B5_LABEL_CONTRACT_FREEZE.md):
- primary severity    = anchored main-rule value delta (before - after,
                        positive = original appearance weakened/improved)
- secondary audit     = dynamic re-diagnose + Jaccard rematch (priority delta,
                        rematch jaccard, dynamic disappearance) -- never gates
                        primary label validity
- scheduling objective= makespan delta (base - after, positive = improved)

This module is deterministic and stateless; it never falls back to "find the
most similar new block".  Each rule has its own anchored evaluator; A3 shares
the A2 pathway (the detector emits no standalone A3 blocks), and A6 primary
labels exist only for legacy standalone A6 blocks (extreme-A6 standalone is
out of contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from ..ir import Problem, Schedule
from ..symptom_pruning import (
    _availability,
    _flexibility_weight,
    _makespan_relevance,
    _percentile_rank,
    _resource_pools,
    _resource_sequences,
    _typical_duration_on_machine,
)

# Frozen status enum (label contract): only VALID_ANCHORED yields a primary
# appearance label; everything else is an explicit, non-silent failure reason.
LABEL_STATUSES: tuple[str, ...] = (
    "VALID_ANCHORED",
    "NO_LEGAL_ATOM",
    "HARD_INFEASIBLE",
    "ANCHOR_UNEVALUABLE",
    "EXECUTION_FAILED",
)


@dataclass(frozen=True)
class AnchoredEvaluation:
    """Anchored re-evaluation of one original appearance block on a schedule."""

    rule: str
    value: float
    evaluable: bool
    reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "value": self.value,
                "evaluable": self.evaluable, "reason": self.reason}


def anchored_appearance_value(
    problem: Problem, schedule: Schedule, block: Any
) -> AnchoredEvaluation:
    """Re-evaluate the original block's rule quantity on ``schedule``.

    ``block`` is a PrunedAppearanceBlock from the ORIGINAL diagnosis.  The
    member set / rule / detection config are taken from it verbatim; only the
    schedule changes.  On the original schedule this must reproduce the
    detection-time value exactly (the probe asserts this per rule).
    """
    rules = tuple(block.block.appearance_rules)
    if len(rules) != 1:
        return AnchoredEvaluation(",".join(rules), 0.0, False, "multi_rule_block")
    rule = rules[0]
    if rule == "A1":
        return _anchored_a1(problem, schedule, block)
    if rule in ("A2", "A3"):
        return _anchored_a2(problem, schedule, block)
    if rule == "A4":
        return _anchored_a4(problem, schedule, block)
    if rule == "A6":
        return _anchored_a6(problem, schedule, block)
    return AnchoredEvaluation(rule, 0.0, False, "unknown_rule")


# --------------------------------------------------------------------------
# A1 -- machine adjacent gap: original value = idle window / typical duration
# on the machine.  Anchor = the two ops + the machine.
# --------------------------------------------------------------------------
def _anchored_a1(problem: Problem, schedule: Schedule, block: Any) -> AnchoredEvaluation:
    ops = list(block.block.operations)
    machines = list(block.block.machines)
    if len(ops) != 2 or not machines:
        return AnchoredEvaluation("A1", 0.0, False, "anchor_shape")
    machine = machines[0]
    amap = schedule.assignment_map()
    if ops[0] not in amap or ops[1] not in amap:
        return AnchoredEvaluation("A1", 0.0, False, "anchor_op_missing")
    sequence = _resource_sequences(problem, schedule).get(machine, [])
    on_m = {op for _, _, op in sequence}
    if ops[0] not in on_m or ops[1] not in on_m:
        # The pair no longer shares the machine: the anchored gap is dissolved.
        return AnchoredEvaluation("A1", 0.0, True, "anchor_left_machine")

    a, b = ops
    if amap[a].end <= amap[b].start:
        lo_end, hi_start = amap[a].end, amap[b].start
    elif amap[b].end <= amap[a].start:
        lo_end, hi_start = amap[b].end, amap[a].start
    else:
        return AnchoredEvaluation("A1", 0.0, True, "anchor_overlap")
    if hi_start <= lo_end:
        return AnchoredEvaluation("A1", 0.0, True, "anchor_no_window")
    # Busy time of OTHER ops inserted between the anchored pair on the machine.
    busy_between = sum(
        min(end, hi_start) - max(start, lo_end)
        for start, end, op in sequence
        if op not in (a, b) and start < hi_start and end > lo_end
    )
    idle = max(0.0, float(hi_start - lo_end - busy_between))
    typical = _typical_duration_on_machine(sequence)
    return AnchoredEvaluation("A1", idle / max(typical, 1.0), True, "ok")


# --------------------------------------------------------------------------
# A2/A3 -- flexible load imbalance: original value = z_load * z_flex * u_m on
# the overloaded machine's contiguous segment.  Anchor = the segment's op set
# + the machine + the resource pool.
# --------------------------------------------------------------------------
def _pool_from_evidence(problem: Problem, block: Any, machine: str) -> tuple[str, ...]:
    for item in block.block.evidence_ids:
        if item.startswith("resource_pool:"):
            pool = tuple(item.split(":", 1)[1].split(","))
            if machine in pool and len(pool) >= 2:
                return pool
    for pool in _resource_pools(problem):
        if machine in pool:
            return pool
    return ()


def _anchored_a2(problem: Problem, schedule: Schedule, block: Any) -> AnchoredEvaluation:
    rule = block.block.appearance_rules[0]
    machines = list(block.block.machines)
    if not machines:
        return AnchoredEvaluation(rule, 0.0, False, "anchor_shape")
    machine = machines[0]
    pool = _pool_from_evidence(problem, block, machine)
    if len(pool) < 2:
        return AnchoredEvaluation(rule, 0.0, False, "anchor_pool_lost")

    amap = schedule.assignment_map()
    opmap = problem.operation_map()
    mode_map = problem.mode_map()
    sequences = _resource_sequences(problem, schedule)
    horizon = schedule.makespan
    utilization = {
        m: (sum(end - start for start, end, _ in sequences.get(m, ()))
            / max(_availability(problem, m, horizon), 1))
        for m in problem.resource_map()
    }
    pool_util = [utilization.get(m, 0.0) for m in pool]
    z_load = _percentile_rank(utilization.get(machine, 0.0), pool_util)

    # Anchored segment: original members that still run on the machine.
    on_m = {op for _, _, op in sequences.get(machine, [])}
    segment = {op for op in block.block.operations if op in on_m and op in amap}
    if not segment:
        return AnchoredEvaluation(rule, 0.0, True, "anchor_dissolved")
    busy = float(sum(amap[op].end - amap[op].start for op in segment))
    if busy <= 0:
        return AnchoredEvaluation(rule, 0.0, True, "anchor_dissolved")

    flex_load = 0.0
    for op in segment:
        current_mode = mode_map[amap[op].mode_id][1]
        flex_load += (amap[op].end - amap[op].start) * _flexibility_weight(
            opmap[op], current_mode)
    z_flex = flex_load / max(busy, 1)

    share_terms: list[tuple[float, float]] = []
    for peer in pool:
        if peer == machine:
            continue
        z_under = 1.0 - _percentile_rank(utilization.get(peer, 0.0), pool_util)
        shared_busy = 0.0
        for op in segment:
            operation = opmap[op]
            if not any(peer in mode.resources for mode in operation.modes):
                continue
            current_mode = mode_map[amap[op].mode_id][1]
            shared_busy += (amap[op].end - amap[op].start) * min(1.0, current_mode.duration / max(
                next((m.duration for m in operation.modes if peer in m.resources),
                     current_mode.duration), 1))
        share_terms.append((shared_busy / max(busy, 1), z_under))
    denom = sum(s for s, _ in share_terms)
    u_m = (sum(s * u for s, u in share_terms) / denom) if denom > 0 else 0.0
    value = z_load * z_flex * u_m
    return AnchoredEvaluation(rule, value, True, "ok")


# --------------------------------------------------------------------------
# A4 -- job wait with makespan propagation relevance:
# original value = W_e * M_j for the (pred, succ) pair.
# --------------------------------------------------------------------------
def _anchored_a4(problem: Problem, schedule: Schedule, block: Any) -> AnchoredEvaluation:
    ops = list(block.block.operations)
    if len(ops) != 2:
        return AnchoredEvaluation("A4", 0.0, False, "anchor_shape")
    opmap = problem.operation_map()
    amap = schedule.assignment_map()
    x, y = ops
    if x not in opmap or y not in opmap or x not in amap or y not in amap:
        return AnchoredEvaluation("A4", 0.0, False, "anchor_op_missing")
    if x in opmap[y].predecessors:
        left, succ = x, y
    elif y in opmap[x].predecessors:
        left, succ = y, x
    else:
        return AnchoredEvaluation("A4", 0.0, False, "anchor_edge_lost")

    wait = max(0, amap[succ].start - amap[left].end)
    durations = [a.end - a.start for a in schedule.assignments]
    p_scale = float(median(durations)) if durations else 1.0
    w_e = wait / (wait + max(p_scale, 1.0))
    m_j = _makespan_relevance(problem, schedule, p_scale).get(succ, 0.0)
    return AnchoredEvaluation("A4", w_e * m_j, True, "ok")


# --------------------------------------------------------------------------
# A6 -- slow processing mode (legacy standalone only; default emission is OFF
# and A6 rides along as auxiliary evidence on other rules' blocks).
# --------------------------------------------------------------------------
def _anchored_a6(problem: Problem, schedule: Schedule, block: Any) -> AnchoredEvaluation:
    if any(item == "extreme_a6_standalone" for item in block.block.evidence_ids):
        return AnchoredEvaluation("A6", 0.0, False, "a6_extreme_not_in_contract")
    ops = list(block.block.operations)
    if len(ops) != 1:
        return AnchoredEvaluation("A6", 0.0, False, "anchor_shape")
    op_id = ops[0]
    opmap = problem.operation_map()
    amap = schedule.assignment_map()
    mode_map = problem.mode_map()
    if op_id not in opmap or op_id not in amap:
        return AnchoredEvaluation("A6", 0.0, False, "anchor_op_missing")
    operation = opmap[op_id]
    if len(operation.modes) <= 1:
        return AnchoredEvaluation("A6", 0.0, False, "anchor_single_mode")
    selected = mode_map[amap[op_id].mode_id][1]
    p_cur = selected.duration or 0.0
    p_best = min(
        (mode.duration for mode in operation.modes if mode.duration and mode.duration > 0),
        default=0.0,
    )
    if not p_best:
        return AnchoredEvaluation("A6", 0.0, False, "anchor_no_positive_mode")
    z_time = 1.0 - p_best / max(p_cur, 1.0)
    return AnchoredEvaluation("A6", z_time, True, "ok")
