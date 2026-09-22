"""V5 M2 -- deterministic Legal Edit Enumerator (Phase 2).

The enumerator only does **hard feasibility enumeration** (spec §17): for a set
of subject operations (candidate root region around an Appearance) it returns
every legal Routing / Sequencing edit, with decision-time-visible features.
It never judges "which is best" -- M2's edit head later assigns intervention
relevance :math:`z_{edit}` on top of this (and a network must never hallucinate
a machine id, because it can only ever score edits already enumerated here).

Edit kinds (spec §18-19):

* ``ROUTE``   -- ``m_t in Eligible(o), m_t != m_s``  (spec §18)
* ``SEQ_SWAP``-- adjacent pair on a resource, precedence-safe (spec §19)
* ``SEQ_INSERT``-- bounded insert into a gap window, acyclicity-checked (spec §19)

Features are computed from decision-time-visible quantities only (no counterfactual
"after" information, spec §45-11).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Sequence

from .ir import Problem, Schedule
from .m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    EDIT_TIMING_SHIFT,
    LegalEdit,
    route_edit_id,
    seq_insert_edit_id,
    seq_swap_edit_id,
    timing_shift_edit_id,
)


def _job_successors(problem: Problem) -> dict[str, set[str]]:
    """Transitive job-precedence successors (u -> every op that waits on u)."""
    operation_map = problem.operation_map()
    succ: dict[str, set[str]] = {oid: set() for oid in operation_map}
    for oid, op in operation_map.items():
        for pred in op.predecessors:
            succ[pred].add(oid)
    # transitive closure over job precedence
    changed = True
    while changed:
        changed = False
        for oid in list(succ):
            added: set[str] = set()
            for s in succ[oid]:
                added |= succ[s]
            before = len(succ[oid])
            succ[oid] |= added
            if len(succ[oid]) != before:
                changed = True
    for node in succ:
        succ[node].discard(node)
    return succ


def _eligible_machine(mode) -> str:
    """For a unary-resource mode, the single machine it occupies."""
    if len(mode.resources) != 1:
        raise ValueError(f"V5 LegalEditEnumerator requires unary-resource modes; got {mode.id}")
    return mode.resources[0]


def _operation_eligible(problem: Problem) -> dict[str, dict[str, list]]:
    """op_id -> {machine_id -> [Mode,...]} (eligible modes per machine)."""
    out: dict[str, dict[str, list]] = {}
    for op in problem.operations:
        by_machine: dict[str, list] = defaultdict(list)
        for mode in op.modes:
            by_machine[_eligible_machine(mode)].append(mode)
        out[op.id] = dict(by_machine)
    return out


def _resource_loads(problem: Problem, schedule: Schedule, eligible) -> tuple[dict[str, float], float]:
    """busy-duration per machine and makespan (decision-time-visible)."""
    assignment_map = schedule.assignment_map()
    mode_map = problem.mode_map()
    load: dict[str, float] = defaultdict(float)
    makespan = 0
    for oid, asg in assignment_map.items():
        mode = mode_map[asg.mode_id][1]
        machine = _eligible_machine(mode)
        load[machine] += mode.duration
        makespan = max(makespan, float(asg.end))
    return dict(load), makespan


def _resource_sequences(problem: Problem, schedule: Schedule) -> dict[str, list[str]]:
    assignment_map = schedule.assignment_map()
    mode_map = problem.mode_map()
    out: dict[str, list[str]] = defaultdict(list)
    for oid, asg in assignment_map.items():
        machine = _eligible_machine(mode_map[asg.mode_id][1])
        out[machine].append(oid)
    for seq in out.values():
        seq.sort(key=lambda oid: (
            assignment_map[oid].start, assignment_map[oid].end, oid
        ))
    return dict(out)


def _receiver_context(sequence: list[str], eligible) -> float:
    """Normalized receiver-window summary for a target machine: the fraction of
    flexible operations on it (flexibility>1 => they could give up machine
    capacity). Decision-time-visible and deterministic (spec §18)."""
    if not sequence:
        return 0.0
    flexible = sum(1 for oid in sequence if len(eligible.get(oid, {})) > 1)
    return flexible / len(sequence)


class LegalEditEnumerator:
    """Deterministic hard-feasibility legal edit enumeration (spec §17-19)."""

    def __init__(self, problem: Problem, schedule: Schedule):
        self.problem = problem
        self.schedule = schedule
        self.eligible = _operation_eligible(problem)
        self.successors = _job_successors(problem)
        self.sequences = _resource_sequences(problem, schedule)
        self.loads, self.makespan = _resource_loads(problem, schedule, self.eligible)
        self.assignment_map = schedule.assignment_map()
        self.mode_map = problem.mode_map()

    # -- public entry ---------------------------------------------------------

    def enumerate(
        self,
        subject_operations: Iterable[str],
        *,
        request_route: bool = True,
        request_seq_swap: bool = True,
        request_seq_insert: bool = True,
        request_timing_shift: bool = False,
        max_insert_window: int = 3,
    ) -> tuple[LegalEdit, ...]:
        """Enumerate all legal edits for the subject operations.

        ``max_insert_window`` bounds the number of slots an operation may jump
        (spec §19 W_seq, default 3).
        """
        subjects = set(subject_operations)
        unknown = subjects - set(self.eligible)
        if unknown:
            raise ValueError(f"unknown subject operations: {sorted(unknown)}")
        out: list[LegalEdit] = []
        if request_route:
            out.extend(self._route_edits(subjects))
        if request_seq_swap:
            out.extend(self._seq_swap_edits(subjects))
        if request_seq_insert:
            out.extend(self._seq_insert_edits(subjects, max_insert_window))
        if request_timing_shift:
            out.extend(self._timing_shift_edits(subjects))
        return tuple(sorted(out, key=lambda e: e.edit_id))

    # -- routing ---------------------------------------------------------------

    def _current_mode(self, oid: str):
        asg = self.assignment_map.get(oid)
        if asg is None:
            return None
        return self.mode_map[asg.mode_id][1]

    def _route_edits(self, subjects: set[str]) -> list[LegalEdit]:
        edits: list[LegalEdit] = []
        for oid in sorted(subjects):
            curr = self._current_mode(oid)
            if curr is None:
                continue
            src_machine = _eligible_machine(curr)
            p_cur = float(curr.duration)
            for machine, modes in sorted(self.eligible[oid].items()):
                if machine == src_machine:
                    continue
                target_mode = min(modes, key=lambda m: m.duration)
                p_tgt = float(target_mode.duration)
                src_load = self.loads.get(src_machine, 0.0)
                tgt_load = self.loads.get(machine, 0.0)
                ms = self.makespan or 1.0
                receiver = _receiver_context(self.sequences.get(machine, []), self.eligible)
                features = (
                    ("p_current", p_cur),
                    ("p_target", p_tgt),
                    ("delta_p", p_tgt - p_cur),
                    ("source_load", src_load),
                    ("target_load", tgt_load),
                    ("source_relative_load", src_load / ms),
                    ("target_relative_load", tgt_load / ms),
                    ("flexibility", float(len(self.eligible[oid]))),
                    ("receiver_context", receiver),
                )
                edits.append(LegalEdit(
                    edit_id=route_edit_id(oid, src_machine, machine),
                    edit_type=EDIT_ROUTE,
                    operation_id=oid,
                    source_machine=src_machine,
                    target_machine=machine,
                    target_mode_id=target_mode.id,
                    features=features,
                ))
        return edits

    # -- sequencing ------------------------------------------------------------

    def _seq_swap_edits(self, subjects: set[str]) -> list[LegalEdit]:
        edits: list[LegalEdit] = []
        for resource_id, seq in sorted(self.sequences.items()):
            for left, right in zip(seq, seq[1:]):
                if left not in subjects and right not in subjects:
                    continue
                # adjacent (left,right); swapping -> order (right,left). Illegal
                # iff left must precede right via job precedence (right in
                # successors(left)).
                if right in self.successors.get(left, set()):
                    continue  # would break precedence / form precedence-cycle
                if left in self.successors.get(right, set()):
                    continue  # current order already violates precedence; skip
                edits.append(LegalEdit(
                    edit_id=seq_swap_edit_id(left, right, resource_id),
                    edit_type=EDIT_SEQ_SWAP,
                    operation_id=left,
                    resource_id=resource_id,
                    left_id=left,
                    right_id=right,
                ))
        return edits

    def _seq_insert_edits(
        self, subjects: set[str], max_insert_window: int
    ) -> list[LegalEdit]:
        """Bounded insert of a subject op into another slot on its resource.

        Positions ``0..len(seq)``: 0 = before first, ``k`` (1..len-1) = between
        ``seq[k-1]`` and ``seq[k]``, ``len(seq)`` = after last.  A move is only
        legal if the jump distance ``|target - current|`` is within the window
        and the new adjacency is acyclic (spec §19 W_seq, default 3).
        """
        edits: list[LegalEdit] = []
        for resource_id, seq in sorted(self.sequences.items()):
            index = {oid: i for i, oid in enumerate(seq)}
            n = len(seq)
            for oid in sorted(subjects):
                if oid not in index:
                    continue
                cur = index[oid]
                for pos in range(0, n + 1):
                    if abs(pos - cur) > max_insert_window:
                        continue
                    p = seq[pos - 1] if pos - 1 >= 0 else None
                    q = seq[pos] if pos < n else None
                    if p == oid or q == oid:
                        continue  # moving onto its own adjacency
                    if not self._insertion_legal(oid, p, q):
                        continue
                    edits.append(LegalEdit(
                        edit_id=seq_insert_edit_id(oid, resource_id, pos),
                        edit_type=EDIT_SEQ_INSERT,
                        operation_id=oid,
                        resource_id=resource_id,
                        insert_position=pos,
                        predecessor_id=p,
                        successor_id=q,
                    ))
        return edits

    def _insertion_legal(self, x: str, p: str | None, q: str | None) -> bool:
        """Proposed move: place x with predecessor ``p`` (or start) and successor
        ``q`` (or end) on the resource.  New machine edges ``p->x`` and ``x->q``.
        No new cycle iff:
          * x is not a job-ancestor of p   (else x->p precedence + p->x machine)
          * q is not a job-ancestor of x   (else q->x precedence + x->q machine)
        Removing x from its old slot only deletes edges, so it cannot introduce a
        cycle."""
        succ = self.successors
        if p is not None and p in succ.get(x, set()):
            return False
        if q is not None and x in succ.get(q, set()):
            return False
        return True

    def _timing_shift_edits(self, subjects: set[str]) -> list[LegalEdit]:
        """Enumerate exact starts at boundaries of real idle windows."""
        edits: list[LegalEdit] = []
        for resource_id, seq in sorted(self.sequences.items()):
            rows = [self.assignment_map[oid] for oid in seq]
            for oid in sorted(subjects & set(seq)):
                assignment = self.assignment_map[oid]
                duration = float(assignment.end - assignment.start)
                boundaries = {0.0}
                boundaries.update(float(row.end) for row in rows if row.operation_id != oid)
                for start in sorted(boundaries):
                    if abs(start - float(assignment.start)) <= 1e-9:
                        continue
                    end = start + duration
                    if any(
                        row.operation_id != oid
                        and float(row.start) < end and start < float(row.end)
                        for row in rows
                    ):
                        continue
                    predecessors = self.problem.operation_map()[oid].predecessors
                    if any(float(self.assignment_map[pred].end) > start for pred in predecessors):
                        continue
                    edits.append(LegalEdit(
                        edit_id=timing_shift_edit_id(oid, resource_id, start),
                        edit_type=EDIT_TIMING_SHIFT,
                        operation_id=oid,
                        resource_id=resource_id,
                        target_start=start,
                        features=(("timing_delta", start - float(assignment.start)),),
                    ))
        return edits


def enumerate_legal_edits(
    problem: Problem,
    schedule: Schedule,
    subject_operations: Iterable[str],
    *,
    request_route: bool = True,
    request_seq_swap: bool = True,
    request_seq_insert: bool = True,
    request_timing_shift: bool = False,
    max_insert_window: int = 3,
) -> tuple[LegalEdit, ...]:
    """Convenience wrapper (deterministic)."""
    return LegalEditEnumerator(problem, schedule).enumerate(
        subject_operations,
        request_route=request_route,
        request_seq_swap=request_seq_swap,
        request_seq_insert=request_seq_insert,
        request_timing_shift=request_timing_shift,
        max_insert_window=max_insert_window,
    )
