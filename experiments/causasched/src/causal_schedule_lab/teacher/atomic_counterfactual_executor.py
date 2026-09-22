"""Atomic Counterfactual Executor (Phase2改.md §1/§5/§6/§15/§16).

The Phase 2.6 audit showed the CE-label path reused the production *repair*
executor (``DeterministicOperatorExecutor`` -> CP-SAT), which releases job tails
and lets the solver re-optimise mode + machine order globally.  That makes
``CE_A(r)`` actually ``CE_A(r + global reoptimisation)`` -- not an atomic causal
label.

This module replaces that path for CE generation.  An atomic intervention
compiles to **exact discrete decisions** (fixed modes, exact machine
permutations), realises them by :func:`fixed_decision_replay.replay_fixed_decisions`
(longest-path DAG replay, no solver), and validates the atomic contract (§16):

    mode_o^{cf} = mode_o^{obs}              for all o  (except a routing target)
    Seq_m^{cf}   = Seq_m^{obs}              for all m != m*
    Seq_{m*}^{cf}= ApplyExactEdit(Seq_{m*}^{obs}, a)
    only timing / makespan / appearance may change

The old executor is untouched and remains the M3 / optimisation / repair path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..ir import Problem, Schedule
from .atom_generator import DecisionAtom
from .fixed_decision_replay import (
    observed_sequences,
    observed_mode_ids,
    replay_fixed_decisions,
    sequence_adjacent_pairs,
    machine_of,
)

ATOMIC_EXECUTOR_VERSION = "atomic-cf-replay-1.0"

SEQUENCING_OPERATORS = {
    "adjacent_resource_swap",
    "resource_sequence_insertion",
    "critical_block_resequence",
}
ROUTING_OPERATORS = {"machine_reassignment", "stage_machine_reassignment"}

# Phase 2.9 §10: joint-intervention conflict taxonomy.  A joint ``do(B_R)`` may
# be rejected (fail-closed, no global repair) for one of these reasons.  ``compatible``
# means the atoms' exact edits compose into one valid plan.
JOINT_CONFLICT_COMPATIBLE = "compatible"
JOINT_CONFLICT_CONTRADICTORY_ORDER = "contradictory_order"
JOINT_CONFLICT_ROUTING_SEQUENCE = "routing_sequence_conflict"
JOINT_CONFLICT_DUPLICATE = "duplicate_equivalent"
JOINT_CONFLICT_CYCLE = "cycle"
JOINT_CONFLICT_UNSUPPORTED = "unsupported_combination"



@dataclass(frozen=True)
class AtomicDecisionPlan:
    """The exact discrete decisions an atom compiles to."""

    mode_ids_cf: dict[str, str]
    machine_sequences_cf: dict[str, list[str]]
    target_machines: tuple[str, ...]
    expected_target_sequences: dict[str, list[str]]
    atom_type: str
    routing_target_op: str | None = None


@dataclass(frozen=True)
class AtomicityReport:
    feasible: bool
    classification: str  # atomic | partial | target_not_enforced | infeasible
    target_enforced: bool
    target_machines: tuple[str, ...]
    expected_target_sequences: dict[str, list[str]]
    actual_target_sequences: dict[str, list[str]]
    non_target_reassignments: int
    non_target_machine_sequence_changes: int
    expected_target_pair_changes: int
    actual_pair_changes: int
    secondary_sequence_changes: int
    reassigned_ops: tuple[str, ...]
    reason: str | None = None
    # Phase 2.9 §10: joint conflict classification (None for single-atom probes).
    conflict_type: str | None = None


@dataclass(frozen=True)
class AtomicExecutionResult:
    schedule: Schedule | None
    feasible: bool
    report: AtomicityReport
    executor_version: str = ATOMIC_EXECUTOR_VERSION


def _expected_insertion_sequence(
    before_seq: list[str],
    operation_id: str,
    predecessor_id: str | None,
    successor_id: str | None,
) -> list[str] | None:
    """Move ``operation_id`` to sit between predecessor and successor.

    The op is removed from its current slot and re-inserted so that
    ``predecessor < op < successor`` holds *and* the three are adjacent when both
    neighbours exist.  Returns ``None`` if the requested neighbours are not
    consistent with the observed sequence (stale atom).
    """
    if operation_id not in before_seq:
        return None
    seq = list(before_seq)
    seq.remove(operation_id)
    if successor_id is not None:
        if successor_id not in seq or successor_id == operation_id:
            return None
        idx = seq.index(successor_id)
        seq.insert(idx, operation_id)
    elif predecessor_id is not None:
        if predecessor_id not in seq or predecessor_id == operation_id:
            return None
        idx = seq.index(predecessor_id)
        seq.insert(idx + 1, operation_id)
    else:
        # No neighbours: move to front (position 0).
        seq.insert(0, operation_id)
    # Validate adjacency when both neighbours are named.
    if predecessor_id is not None and successor_id is not None:
        i = seq.index(operation_id)
        if i == 0 or seq[i - 1] != predecessor_id:
            return None
        if i == len(seq) - 1 or seq[i + 1] != successor_id:
            return None
    return seq


def _expected_swap_sequence(
    before_seq: list[str], left: str, right: str
) -> list[str] | None:
    """Swap two adjacent operations ``left`` then ``right`` -> ``right`` then ``left``."""
    if left not in before_seq or right not in before_seq:
        return None
    seq = list(before_seq)
    i = seq.index(left)
    if i + 1 >= len(seq) or seq[i + 1] != right:
        return None  # not adjacent in the observed sequence
    seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return seq


def _expected_routing_sequences(
    problem: Problem,
    schedule: Schedule,
    operation_id: str,
    mode_id: str,
    target_resource_ids,
) -> tuple[dict[str, list[str]], str, str] | None:
    """Remove the op from its old machine, insert into the target machine by start order."""
    mode_map = problem.mode_map()
    if mode_id not in mode_map:
        return None
    new_resources = tuple(mode_map[mode_id][1].resources)
    if not new_resources:
        return None
    target_machine = new_resources[0]
    assignment_map = schedule.assignment_map()
    assignment = assignment_map.get(operation_id)
    if assignment is None:
        return None
    old_mode = mode_map[assignment.mode_id][1]
    if not old_mode.resources:
        return None
    old_machine = old_mode.resources[0]

    before = observed_sequences(problem, schedule)
    new_sequences: dict[str, list[str]] = {}
    # Remove op from old machine.
    old_seq = [o for o in before.get(old_machine, []) if o != operation_id]
    new_sequences[old_machine] = old_seq
    # Insert op into target machine preserving observed start order.
    target_seq = list(before.get(target_machine, []))
    op_start = assignment.start
    insert_at = len(target_seq)
    for idx, other in enumerate(target_seq):
        other_assign = assignment_map.get(other)
        if other_assign is not None and other_assign.start >= op_start:
            insert_at = idx
            break
    target_seq.insert(insert_at, operation_id)
    new_sequences[target_machine] = target_seq
    return new_sequences, old_machine, target_machine


def _apply_edit_to_sequence(
    before_seq: list[str], operator_id: str, params: dict[str, Any]
) -> list[str] | None:
    """Apply one sequencing atom's edit to a running machine sequence.

    Returns the new sequence, or ``None`` if the edit is inconsistent with the
    current (possibly already-edited) sequence -- e.g. the named neighbours are
    no longer adjacent after a prior edit.  This is the ``contradictory_order``
    signal for joint probes (§10).
    """
    if operator_id == "resource_sequence_insertion":
        operation_id = str(params.get("operation_id"))
        predecessor_id = params.get("predecessor_id")
        successor_id = params.get("successor_id")
        if isinstance(predecessor_id, str) is False:
            predecessor_id = None
        if isinstance(successor_id, str) is False:
            successor_id = None
        return _expected_insertion_sequence(
            before_seq, operation_id, predecessor_id, successor_id
        )
    if operator_id == "adjacent_resource_swap":
        return _expected_swap_sequence(
            before_seq, str(params.get("left_operation_id")),
            str(params.get("right_operation_id")),
        )
    if operator_id == "critical_block_resequence":
        operation_ids = [str(x) for x in params.get("operation_ids", [])]
        if len(operation_ids) < 2:
            return None
        return _expected_swap_sequence(before_seq, operation_ids[0], operation_ids[1])
    return None


def _compose_same_machine_edits(
    observed_seq: list[str],
    edits: list[tuple[str, dict[str, Any]]],
) -> list[str] | None:
    """Compose multiple sequencing edits on one machine into one final sequence.

    ``edits`` is a list of ``(probe_operator_id, probe_parameters)`` in a stable
    order (sorted by atom_id by the caller).  Each edit is applied to the
    running sequence; the first edit that is inconsistent with the current
    sequence returns ``None`` (``contradictory_order``).
    """
    seq = list(observed_seq)
    for operator_id, params in edits:
        nxt = _apply_edit_to_sequence(seq, operator_id, params)
        if nxt is None:
            return None
        seq = nxt
    return seq


def compile_atomic_decisions(
    problem: Problem, schedule: Schedule, atom: DecisionAtom
) -> AtomicDecisionPlan | None:
    """Compile an atom's probe into exact fixed decisions (the intervention compiler)."""
    before = observed_sequences(problem, schedule)
    mode_ids_cf = dict(observed_mode_ids(schedule))
    params = dict(atom.probe_parameters or {})
    op_id = atom.probe_operator_id

    if op_id == "resource_sequence_insertion":
        resource_id = str(params.get("resource_id"))
        operation_id = str(params.get("operation_id"))
        predecessor_id = params.get("predecessor_id")
        successor_id = params.get("successor_id")
        if isinstance(predecessor_id, str) is False and predecessor_id is not None:
            predecessor_id = None
        if isinstance(successor_id, str) is False and successor_id is not None:
            successor_id = None
        before_seq = before.get(resource_id)
        if before_seq is None or operation_id not in before_seq:
            return None
        new_seq = _expected_insertion_sequence(
            before_seq, operation_id,
            predecessor_id if isinstance(predecessor_id, str) else None,
            successor_id if isinstance(successor_id, str) else None,
        )
        if new_seq is None:
            return None
        seqs = {k: list(v) for k, v in before.items()}
        seqs[resource_id] = new_seq
        return AtomicDecisionPlan(
            mode_ids_cf=mode_ids_cf, machine_sequences_cf=seqs,
            target_machines=(resource_id,),
            expected_target_sequences={resource_id: new_seq},
            atom_type=atom.atom_type,
        )

    if op_id == "adjacent_resource_swap":
        resource_id = str(params.get("resource_id"))
        left = str(params.get("left_operation_id"))
        right = str(params.get("right_operation_id"))
        before_seq = before.get(resource_id)
        if before_seq is None:
            return None
        new_seq = _expected_swap_sequence(before_seq, left, right)
        if new_seq is None:
            return None
        seqs = {k: list(v) for k, v in before.items()}
        seqs[resource_id] = new_seq
        return AtomicDecisionPlan(
            mode_ids_cf=mode_ids_cf, machine_sequences_cf=seqs,
            target_machines=(resource_id,),
            expected_target_sequences={resource_id: new_seq},
            atom_type=atom.atom_type,
        )

    if op_id == "critical_block_resequence":
        resource_id = str(params.get("resource_id"))
        operation_ids = [str(x) for x in params.get("operation_ids", [])]
        if len(operation_ids) < 2:
            return None
        before_seq = before.get(resource_id)
        if before_seq is None:
            return None
        # Reverse each adjacent pair in the named block; for a 2-op block this is
        # a single swap of the adjacent pair.
        a, b = operation_ids[0], operation_ids[1]
        new_seq = _expected_swap_sequence(before_seq, a, b)
        if new_seq is None:
            return None
        seqs = {k: list(v) for k, v in before.items()}
        seqs[resource_id] = new_seq
        return AtomicDecisionPlan(
            mode_ids_cf=mode_ids_cf, machine_sequences_cf=seqs,
            target_machines=(resource_id,),
            expected_target_sequences={resource_id: new_seq},
            atom_type=atom.atom_type,
        )

    if op_id in ROUTING_OPERATORS:
        operation_id = str(params.get("operation_id"))
        mode_id = str(params.get("mode_id"))
        target_resource_ids = params.get("target_resource_ids") or ()
        routing = _expected_routing_sequences(
            problem, schedule, operation_id, mode_id, target_resource_ids
        )
        if routing is None:
            return None
        new_seqs, old_machine, target_machine = routing
        mode_ids_cf = dict(mode_ids_cf)
        mode_ids_cf[operation_id] = mode_id
        seqs = {k: list(v) for k, v in before.items()}
        seqs[old_machine] = new_seqs[old_machine]
        seqs[target_machine] = new_seqs[target_machine]
        target_machines = tuple(sorted({old_machine, target_machine}))
        expected = {old_machine: new_seqs[old_machine], target_machine: new_seqs[target_machine]}
        return AtomicDecisionPlan(
            mode_ids_cf=mode_ids_cf, machine_sequences_cf=seqs,
            target_machines=target_machines,
            expected_target_sequences=expected,
            atom_type=atom.atom_type,
            routing_target_op=operation_id,
        )

    return None


def _classify(
    report_data: dict[str, Any],
) -> str:
    if not report_data["feasible"]:
        return "infeasible"
    if not report_data["target_enforced"]:
        return "target_not_enforced"
    if (
        report_data["non_target_reassignments"] == 0
        and report_data["non_target_machine_sequence_changes"] == 0
        and report_data["secondary_sequence_changes"] == 0
    ):
        return "atomic"
    if (
        report_data["non_target_reassignments"] <= 2
        and report_data["non_target_machine_sequence_changes"] <= 1
    ):
        return "partial"
    return "contains_global_reopt"


class AtomicCounterfactualExecutor:
    """Execute an atomic counterfactual by fixed-decision replay (no solver)."""

    def execute_atom(
        self,
        problem: Problem,
        schedule: Schedule,
        atom: DecisionAtom,
    ) -> AtomicExecutionResult:
        plan = compile_atomic_decisions(problem, schedule, atom)
        if plan is None:
            return AtomicExecutionResult(
                schedule=None, feasible=False,
                report=AtomicityReport(
                    feasible=False, classification="infeasible",
                    target_enforced=False, target_machines=(),
                    expected_target_sequences={}, actual_target_sequences={},
                    non_target_reassignments=0,
                    non_target_machine_sequence_changes=0,
                    expected_target_pair_changes=0, actual_pair_changes=0,
                    secondary_sequence_changes=0, reassigned_ops=(),
                    reason="compile_failed",
                ),
            )

        after = replay_fixed_decisions(
            problem, schedule, plan.machine_sequences_cf, plan.mode_ids_cf
        )
        if after is None:
            return AtomicExecutionResult(
                schedule=None, feasible=False,
                report=AtomicityReport(
                    feasible=False, classification="infeasible",
                    target_enforced=False, target_machines=plan.target_machines,
                    expected_target_sequences=plan.expected_target_sequences,
                    actual_target_sequences={},
                    non_target_reassignments=0,
                    non_target_machine_sequence_changes=0,
                    expected_target_pair_changes=0, actual_pair_changes=0,
                    secondary_sequence_changes=0, reassigned_ops=(),
                    reason="cycle_with_job_precedence",
                ),
            )

        report = self._validate(problem, schedule, after, plan)
        return AtomicExecutionResult(
            schedule=after, feasible=report.feasible, report=report
        )

    def execute_atoms(
        self,
        problem: Problem,
        schedule: Schedule,
        atoms: tuple[DecisionAtom, ...],
    ) -> AtomicExecutionResult:
        """Joint atomic counterfactual ``do(B_R)``: union of exact edits, one replay.

        Each atom is compiled against the *observed* schedule; their target-machine
        edits are composed (§9).  Genuine interaction conflicts (forcing one op to
        two modes, sequencing an op that a routing atom moved away, two
        contradictory orderings) are classified and returned fail-closed (§10)
        rather than silently re-optimised.
        """
        if not atoms:
            return _infeasible_result(
                (), reason="empty_atom_set", conflict_type=JOINT_CONFLICT_UNSUPPORTED
            )

        # §10 duplicate_equivalent: identical atoms (same atom_id) collapse to one.
        seen: dict[str, DecisionAtom] = {}
        deduped: list[DecisionAtom] = []
        for atom in atoms:
            if atom.atom_id in seen:
                if seen[atom.atom_id] != atom:
                    return _infeasible_result(
                        (), reason=f"duplicate_mismatch:{atom.atom_id}",
                        conflict_type=JOINT_CONFLICT_DUPLICATE,
                    )
                continue
            seen[atom.atom_id] = atom
            deduped.append(atom)
        if len(deduped) < len(atoms):
            return _infeasible_result(
                (), reason="duplicate_equivalent",
                conflict_type=JOINT_CONFLICT_DUPLICATE,
            )

        plans: list[AtomicDecisionPlan] = []
        for atom in deduped:
            plan = compile_atomic_decisions(problem, schedule, atom)
            if plan is None:
                return _infeasible_result(
                    (), reason=f"compile_failed:{atom.atom_id}",
                    conflict_type=JOINT_CONFLICT_UNSUPPORTED,
                )
            plans.append(plan)

        # §10 unsupported_combination: two routing atoms force one op to
        # different modes.  Only a *changed* mode (!= observed) is a claim;
        # sequencing atoms carry the full observed mode map unchanged, so they
        # claim nothing and never spuriously conflict with a routing atom.
        observed_modes = dict(observed_mode_ids(schedule))
        mode_ids_cf = dict(observed_modes)
        for plan in plans:
            for op, mode in plan.mode_ids_cf.items():
                if mode == observed_modes.get(op):
                    continue  # unchanged -> not a claim
                if op in mode_ids_cf and mode_ids_cf[op] != observed_modes.get(op) \
                        and mode_ids_cf[op] != mode:
                    return _infeasible_result(
                        (), reason=f"mode_conflict:{op}",
                        conflict_type=JOINT_CONFLICT_UNSUPPORTED,
                    )
                mode_ids_cf[op] = mode

        routing_target_ops = {
            plan.routing_target_op for plan in plans if plan.routing_target_op
        }
        observed_seq = observed_sequences(problem, schedule)
        seqs = {k: list(v) for k, v in observed_seq.items()}
        expected: dict[str, list[str]] = {}
        target_machines: list[str] = []

        # For each routing atom, record which machines it removes the op from /
        # inserts the op into, and which ops leave each machine.
        routing_removed: dict[str, set[str]] = {}   # machine -> ops removed from it
        routing_inserted: dict[str, set[str]] = {}  # machine -> ops inserted into it
        routing_plans = [p for p in plans if p.routing_target_op is not None]
        for plan in routing_plans:
            r_op = plan.routing_target_op
            for machine in plan.target_machines:
                exp_seq = plan.expected_target_sequences.get(machine, [])
                if r_op not in exp_seq:
                    # r_op was removed from this machine.
                    routing_removed.setdefault(machine, set()).add(r_op)
                else:
                    routing_inserted.setdefault(machine, set()).add(r_op)

        # Group sequencing edits by target machine (§9 composition).
        machine_edits: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for atom, plan in zip(deduped, plans):
            if atom.probe_operator_id in SEQUENCING_OPERATORS:
                for machine in plan.target_machines:
                    machine_edits.setdefault(machine, []).append(
                        (atom.probe_operator_id, dict(atom.probe_parameters))
                    )

        # §10 routing_sequence_conflict: a sequencing atom edits a machine whose
        # membership was changed by a routing atom (op removed/inserted).  The
        # sequencing edit was compiled against the observed ordering and is stale
        # once routing moves an op on/off that machine.
        for machine in machine_edits:
            if machine in routing_removed or machine in routing_inserted:
                moved_ops = routing_removed.get(machine, set()) | routing_inserted.get(machine, set())
                return _infeasible_result(
                    (machine,), reason=f"routing_sequence_conflict:{sorted(moved_ops)}@{machine}",
                    conflict_type=JOINT_CONFLICT_ROUTING_SEQUENCE,
                )

        # Apply routing: sequential composition on the *running* sequences.
        # Each routing atom removes its op from its old machine and inserts it
        # into its target machine preserving observed start order.  This is
        # composed atom-by-atom rather than overwriting ``seqs[machine]`` with
        # each plan's observed-derived ``expected_target_sequences``.  The old
        # per-plan overwrite reset a shared machine to its observed sequence and
        # silently dropped a relocation, so an enabling chain (dependent target
        # == relocation source) left the relocated op stale on its old machine
        # and ``replay_fixed_decisions`` returned a spurious
        # ``cycle_with_job_precedence`` (§16).
        assignment_map = schedule.assignment_map()
        for plan in routing_plans:
            r_op = plan.routing_target_op
            old_machine: str | None = None
            target_machine: str | None = None
            for machine in plan.target_machines:
                if r_op in plan.expected_target_sequences.get(machine, ()):
                    target_machine = machine
                else:
                    old_machine = machine
            if old_machine is not None and r_op in seqs.get(old_machine, ()):
                seqs[old_machine] = [o for o in seqs[old_machine] if o != r_op]
            if target_machine is not None:
                tgt_seq = seqs.setdefault(target_machine, [])
                if r_op not in tgt_seq:
                    assignment = assignment_map.get(r_op)
                    op_start = assignment.start if assignment is not None else float("inf")
                    insert_at = len(tgt_seq)
                    for idx, other in enumerate(tgt_seq):
                        other_assign = assignment_map.get(other)
                        if other_assign is not None and other_assign.start >= op_start:
                            insert_at = idx
                            break
                    tgt_seq.insert(insert_at, r_op)
                seqs[target_machine] = tgt_seq
        for machine in {m for plan in routing_plans for m in plan.target_machines}:
            expected[machine] = list(seqs.get(machine, []))
            if machine not in target_machines:
                target_machines.append(machine)

        # Compose sequencing edits per machine, sorted by atom_id for determinism.
        for machine, edits in machine_edits.items():
            edits.sort(key=lambda e: e[0])  # stable by operator id (caller may sort by atom_id too)
            composed = _compose_same_machine_edits(observed_seq.get(machine, []), edits)
            if composed is None:
                return _infeasible_result(
                    (machine,), reason=f"contradictory_order:{machine}",
                    conflict_type=JOINT_CONFLICT_CONTRADICTORY_ORDER,
                )
            seqs[machine] = composed
            expected[machine] = composed
            if machine not in target_machines:
                target_machines.append(machine)

        joint_plan = AtomicDecisionPlan(
            mode_ids_cf=mode_ids_cf, machine_sequences_cf=seqs,
            target_machines=tuple(target_machines),
            expected_target_sequences=expected,
            atom_type="joint",
            routing_target_op=sorted(routing_target_ops)[0] if routing_target_ops else None,
        )
        after = replay_fixed_decisions(problem, schedule, seqs, mode_ids_cf)
        if after is None:
            return _infeasible_result(
                tuple(target_machines), reason="cycle_with_job_precedence",
                expected=expected, conflict_type=JOINT_CONFLICT_CYCLE,
            )
        report = self._validate(problem, schedule, after, joint_plan)
        # §10 compatible: the joint plan validated as atomic/partial.
        report = replace(report, conflict_type=JOINT_CONFLICT_COMPATIBLE)
        return AtomicExecutionResult(
            schedule=after, feasible=report.feasible, report=report
        )

    def _validate(
        self,
        problem: Problem,
        before: Schedule,
        after: Schedule,
        plan: AtomicDecisionPlan,
    ) -> AtomicityReport:
        before_seq = observed_sequences(problem, before)
        after_seq = observed_sequences(problem, after)

        # Target enforcement: each target machine's actual order == expected.
        target_enforced = True
        actual_target: dict[str, list[str]] = {}
        for machine in plan.target_machines:
            actual = after_seq.get(machine, [])
            actual_target[machine] = actual
            if actual != plan.expected_target_sequences.get(machine):
                target_enforced = False

        # Non-target machine sequence changes.
        non_target_machine_changes = 0
        for machine, before_order in before_seq.items():
            if machine in plan.target_machines:
                continue
            if after_seq.get(machine, []) != before_order:
                non_target_machine_changes += 1

        # Routing reassignments excluding the routing target op.
        # Build the immutable lookup tables once.  Calling ``machine_of`` per
        # operation rebuilds both maps each time and becomes O(n²) on the large
        # static Dataset-v5 roots, while this is exactly equivalent O(n).
        mode_map = problem.mode_map()
        before_assignments = before.assignment_map()
        after_assignments = after.assignment_map()

        def assigned_resources(assignments, operation_id: str) -> tuple[str, ...] | None:
            assignment = assignments.get(operation_id)
            if assignment is None or assignment.mode_id not in mode_map:
                return None
            return tuple(mode_map[assignment.mode_id][1].resources)

        before_machine = {
            op: assigned_resources(before_assignments, op) for op in before_assignments
        }
        after_machine = {
            op: assigned_resources(after_assignments, op) for op in after_assignments
        }
        reassigned = [
            op for op in after_machine
            if before_machine.get(op) is not None and before_machine[op] != after_machine[op]
        ]
        non_target_reassignments = [
            op for op in reassigned if op != plan.routing_target_op
        ]

        # Pair-change bookkeeping (§10): secondary = actual - expected_target.
        expected_target_pair_changes = 0
        for machine in plan.target_machines:
            b_pairs = sequence_adjacent_pairs(before_seq.get(machine, []))
            e_pairs = sequence_adjacent_pairs(plan.expected_target_sequences.get(machine, []))
            expected_target_pair_changes += len(e_pairs - b_pairs) + len(b_pairs - e_pairs)
        actual_pair_changes = 0
        for machine in set(before_seq) | set(after_seq):
            b_pairs = sequence_adjacent_pairs(before_seq.get(machine, []))
            a_pairs = sequence_adjacent_pairs(after_seq.get(machine, []))
            actual_pair_changes += len(a_pairs - b_pairs) + len(b_pairs - a_pairs)
        secondary = max(0, actual_pair_changes - expected_target_pair_changes)

        data = {
            "feasible": True,
            "target_enforced": target_enforced,
            "non_target_reassignments": len(non_target_reassignments),
            "non_target_machine_sequence_changes": non_target_machine_changes,
            "secondary_sequence_changes": secondary,
        }
        classification = _classify(data)
        return AtomicityReport(
            feasible=True,
            classification=classification,
            target_enforced=target_enforced,
            target_machines=plan.target_machines,
            expected_target_sequences=plan.expected_target_sequences,
            actual_target_sequences=actual_target,
            non_target_reassignments=len(non_target_reassignments),
            non_target_machine_sequence_changes=non_target_machine_changes,
            expected_target_pair_changes=expected_target_pair_changes,
            actual_pair_changes=actual_pair_changes,
            secondary_sequence_changes=secondary,
            reassigned_ops=tuple(sorted(reassigned)),
            reason=None,
        )


__all__ = [
    "ATOMIC_EXECUTOR_VERSION",
    "AtomicDecisionPlan",
    "AtomicityReport",
    "AtomicExecutionResult",
    "AtomicCounterfactualExecutor",
    "compile_atomic_decisions",
    "JOINT_CONFLICT_COMPATIBLE",
    "JOINT_CONFLICT_CONTRADICTORY_ORDER",
    "JOINT_CONFLICT_ROUTING_SEQUENCE",
    "JOINT_CONFLICT_DUPLICATE",
    "JOINT_CONFLICT_CYCLE",
    "JOINT_CONFLICT_UNSUPPORTED",
]


def _infeasible_result(
    target_machines: tuple[str, ...] = (),
    *,
    reason: str,
    expected: dict[str, list[str]] | None = None,
    conflict_type: str | None = None,
) -> AtomicExecutionResult:
    return AtomicExecutionResult(
        schedule=None, feasible=False,
        report=AtomicityReport(
            feasible=False, classification="infeasible",
            target_enforced=False, target_machines=target_machines,
            expected_target_sequences=expected or {},
            actual_target_sequences={},
            non_target_reassignments=0,
            non_target_machine_sequence_changes=0,
            expected_target_pair_changes=0, actual_pair_changes=0,
            secondary_sequence_changes=0, reassigned_ops=(),
            reason=reason,
            conflict_type=conflict_type,
        ),
    )
