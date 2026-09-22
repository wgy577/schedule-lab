"""CE-label atomicity audit (Phase2改.md §9/§10/§16/§28).

One ``sequencing`` probe must change *only* the target decision, otherwise
``CE_A(r)`` is really ``CE_A(r + global reoptimization)`` and pollutes M2 root
supervision.  This module executes a probe through the **AtomicCounterfactualExecutor**
(fixed-decision replay, no solver) and reports the executor's atomicity verdict
plus a structured before/after diff for the code-audit bundle:

  * target ordering enforced?      (actual target-machine sequence == expected, §9)
  * secondary routing reassignments (non-target ops that moved machine)
  * secondary sequence changes       (actual pair changes - expected target pair changes, §10)
  * makespan delta

Classification (``identified=false``): ``atomic`` (only the target changed),
``partial`` (bounded secondary change), ``contains_global_reopt`` (many atoms
rearranged), ``target_not_enforced``, or ``infeasible`` (cycle).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..ir import Problem, Schedule
from .atomic_counterfactual_executor import (
    AtomicCounterfactualExecutor,
    AtomicExecutionResult,
)
from .atom_generator import DecisionAtom


@dataclass
class AtomicityCase:
    atom: DecisionAtom
    ce: float
    feasible: bool
    target_enforced: bool
    secondary_reassignments: int
    secondary_sequence_changes: int
    makespan_before: int
    makespan_after: int | None
    classification: str
    reassigned_ops: list[str] = field(default_factory=list)
    # Full before/after schedules + structured diff for the code-audit bundle
    # (改A6.md §28: the package must ship before/after schedule + diff per case).
    schedule_before: Schedule | None = None
    schedule_after: Schedule | None = None
    diff: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom.atom_id,
            "atom_type": self.atom.atom_type,
            "probe_operator": self.atom.probe_operator_id,
            "probe_parameters": self.atom.probe_parameters,
            "CE": self.ce,
            "feasible": self.feasible,
            "target_enforced": self.target_enforced,
            "secondary_reassignments": self.secondary_reassignments,
            "secondary_sequence_changes": self.secondary_sequence_changes,
            "makespan_before": self.makespan_before,
            "makespan_after": self.makespan_after,
            "classification": self.classification,
            "reassigned_ops": self.reassigned_ops,
        }

    def as_full_bundle(self, problem: Problem) -> dict[str, Any]:
        """Record + before/after schedule JSON + structured diff.

        This is what the code-audit package ships per case so a reviewer can see
        exactly which ops the solver rerouted/resequenced beyond the target.
        """
        bundle = self.as_record()
        bundle["diff"] = self.diff
        bundle["schedule_before"] = (
            self.schedule_before.model_dump(mode="json") if self.schedule_before is not None else None
        )
        bundle["schedule_after"] = (
            self.schedule_after.model_dump(mode="json") if self.schedule_after is not None else None
        )
        return bundle


def _schedule_diff(problem: Problem, before: Schedule, after: Schedule) -> dict[str, Any]:
    """Structured diff between two schedules for the audit log.

    Captures the three channels a global reoptimization can change beyond the
    target decision: routing reassignment, per-machine sequence, and op timing.
    """
    before_machine = {op: _machine_of(problem, before, op) for op in before.assignment_map()}
    after_machine = {op: _machine_of(problem, after, op) for op in after.assignment_map()}
    reassignments: list[dict[str, Any]] = []
    for op in sorted(set(before_machine) & set(after_machine)):
        if before_machine[op] != after_machine[op]:
            reassignments.append({
                "op": op,
                "before": list(before_machine[op] or []),
                "after": list(after_machine[op] or []),
            })

    before_seq = _sequences(problem, before)
    after_seq = _sequences(problem, after)
    sequence_changes: dict[str, Any] = {}
    for machine in sorted(set(before_seq) | set(after_seq)):
        b = before_seq.get(machine, [])
        a = after_seq.get(machine, [])
        b_pairs = set(zip(b, b[1:]))
        a_pairs = set(zip(a, a[1:]))
        added = [list(p) for p in sorted(a_pairs - b_pairs)]
        removed = [list(p) for p in sorted(b_pairs - a_pairs)]
        if added or removed:
            sequence_changes[machine] = {
                "before": b, "after": a,
                "added_pairs": added, "removed_pairs": removed,
            }

    before_assign = before.assignment_map()
    after_assign = after.assignment_map()
    timing_changes: list[dict[str, Any]] = []
    for op in sorted(set(before_assign) & set(after_assign)):
        b = before_assign[op]
        a = after_assign[op]
        if b.start != a.start or b.end != a.end:
            timing_changes.append({
                "op": op,
                "before_start": b.start, "after_start": a.start,
                "before_end": b.end, "after_end": a.end,
                "delta_start": a.start - b.start,
            })

    return {
        "n_reassignments": len(reassignments),
        "reassignments": reassignments,
        "n_sequence_machines_changed": len(sequence_changes),
        "n_sequence_pair_changes": sum(
            len(s["added_pairs"]) + len(s["removed_pairs"]) for s in sequence_changes.values()
        ),
        "sequence_changes": sequence_changes,
        "n_timing_changes": len(timing_changes),
        "timing_changes": timing_changes,
    }


def diff_markdown(problem: Problem, case: "AtomicityCase") -> str:
    """Human-readable diff log for one audit case (shipped as ``<case>_diff.md``)."""
    lines: list[str] = []
    r = case.as_record()
    lines.append(f"# Atomicity diff — {r['atom_id']}")
    lines.append("")
    lines.append(f"- probe: `{r['probe_operator']}` {r['probe_parameters']}")
    lines.append(f"- CE = {r['CE']:.4f}  |  classification = **{r['classification']}**")
    lines.append(f"- target_enforced = {r['target_enforced']}  |  feasible = {r['feasible']}")
    lines.append(f"- makespan: {r['makespan_before']} -> {r['makespan_after']} "
                 f"(Δ={r['makespan_after'] - r['makespan_before'] if r['makespan_after'] is not None else 'n/a'})")
    lines.append(f"- secondary_reassignments = {r['secondary_reassignments']}  |  "
                 f"secondary_sequence_changes = {r['secondary_sequence_changes']}")
    lines.append("")
    diff = case.diff or {}
    reassignments = diff.get("reassignments", [])
    lines.append(f"## Routing reassignments ({len(reassignments)})")
    if reassignments:
        lines.append("| op | before machine | after machine |")
        lines.append("|---|---|---|")
        for item in reassignments:
            lines.append(f"| {item['op']} | {item['before']} | {item['after']} |")
    else:
        lines.append("_none_")
    lines.append("")
    seq_changes = diff.get("sequence_changes", {})
    lines.append(f"## Per-machine sequence changes ({len(seq_changes)} machines)")
    for machine, ch in seq_changes.items():
        lines.append(f"### {machine}")
        lines.append(f"- before: {ch['before']}")
        lines.append(f"- after:  {ch['after']}")
        if ch["removed_pairs"]:
            lines.append(f"- removed adjacent pairs: {ch['removed_pairs']}")
        if ch["added_pairs"]:
            lines.append(f"- added adjacent pairs: {ch['added_pairs']}")
    lines.append("")
    timing = diff.get("timing_changes", [])
    lines.append(f"## Timing changes ({len(timing)} ops moved)")
    if timing:
        lines.append("| op | start before | start after | Δstart |")
        lines.append("|---|---|---|---|")
        for item in timing:
            lines.append(f"| {item['op']} | {item['before_start']} | {item['after_start']} | {item['delta_start']:+d} |")
    else:
        lines.append("_none_")
    lines.append("")
    return "\n".join(lines)


def _machine_of(problem: Problem, schedule: Schedule, op: str) -> tuple[str, ...] | None:
    mode_map = problem.mode_map()
    assignment = schedule.assignment_map().get(op)
    if assignment is None or assignment.mode_id not in mode_map:
        return None
    return tuple(mode_map[assignment.mode_id][1].resources)


def _sequences(problem: Problem, schedule: Schedule) -> dict[str, list[str]]:
    mode_map = problem.mode_map()
    assignment_map = schedule.assignment_map()
    seq: dict[str, list[str]] = defaultdict(list)
    for assignment in schedule.assignments:
        mode = mode_map[assignment.mode_id][1]
        for resource_id in mode.resources:
            seq[resource_id].append(assignment.operation_id)
    for resource_id, ops in seq.items():
        ops.sort(key=lambda o: (assignment_map[o].start, assignment_map[o].end, o))
    return dict(seq)


def _sequence_adjacent_pairs(seq: dict[str, list[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for ops in seq.values():
        for left, right in zip(ops, ops[1:]):
            pairs.add((left, right))
    return pairs


def audit_atom_atomicity(
    problem: Problem,
    schedule: Schedule,
    atom: DecisionAtom,
    *,
    executor: AtomicCounterfactualExecutor,
    ce: float = 0.0,
) -> AtomicityCase:
    """Execute ``atom`` through the atomic executor and report its verdict.

    The executor realises the exact edit by fixed-decision replay (no solver) and
    already validates the §16 contract: ``target_enforced`` compares the *full*
    target-machine sequence to the expected one (§9), and
    ``secondary_sequence_changes = actual - expected_target`` (§10).  This
    function wraps that verdict and attaches a structured before/after diff for
    the code-audit bundle.
    """
    makespan_before = int(schedule.makespan)
    result: AtomicExecutionResult = executor.execute_atom(problem, schedule, atom)
    report = result.report

    if not result.feasible or result.schedule is None:
        return AtomicityCase(
            atom, ce, feasible=False, target_enforced=False,
            secondary_reassignments=0, secondary_sequence_changes=0,
            makespan_before=makespan_before, makespan_after=None,
            classification=report.classification,
            schedule_before=schedule, schedule_after=None,
            diff={"reason": report.reason or "infeasible"},
        )

    after = result.schedule
    makespan_after = int(after.makespan)
    diff = _schedule_diff(problem, schedule, after)

    # Secondary reassignments here = non-target ops that changed machine (the
    # executor's report excludes a routing target; for sequencing this is all
    # reassignments, which must be 0 for an atomic sequencing probe).
    return AtomicityCase(
        atom, ce, feasible=True, target_enforced=report.target_enforced,
        secondary_reassignments=report.non_target_reassignments,
        secondary_sequence_changes=report.secondary_sequence_changes,
        makespan_before=makespan_before, makespan_after=makespan_after,
        classification=report.classification,
        reassigned_ops=list(report.reassigned_ops),
        schedule_before=schedule, schedule_after=after, diff=diff,
    )


def select_high_ce_sequencing_atoms(
    problem: Problem,
    schedule: Schedule,
    snapshot,
    executor: AtomicCounterfactualExecutor,
    *,
    max_cases: int = 5,
    tau_ce: float = 0.30,
    top_atoms: int = 16,
    seed: int = 0,
) -> list[tuple[DecisionAtom, float]]:
    """Collect the top high-CE sequencing atoms across retained blocks."""
    from .eval.probe import probe_block_shared

    ranked: list[tuple[DecisionAtom, float]] = []
    for block in snapshot.retained:
        try:
            probe = probe_block_shared(
                problem, schedule, block,
                executor=executor, top_atoms=top_atoms, seed=seed,
            )
        except Exception:
            continue
        for record in probe.records.values():
            # Only valid CE labels count toward the high-CE selection.
            if record.atom.atom_type == "sequencing" and record.label_valid and record.ce >= tau_ce:
                ranked.append((record.atom, record.ce))
    ranked.sort(key=lambda item: (-item[1], item[0].atom_id))
    return ranked[:max_cases]


def run_atomicity_audit(
    problem: Problem,
    schedule: Schedule,
    snapshot,
    executor: AtomicCounterfactualExecutor,
    *,
    max_cases: int = 5,
    tau_ce: float = 0.30,
    top_atoms: int = 16,
    seed: int = 0,
) -> list[AtomicityCase]:
    """Full audit: select high-CE sequencing cases and measure each."""
    cases = select_high_ce_sequencing_atoms(
        problem, schedule, snapshot, executor,
        max_cases=max_cases, tau_ce=tau_ce, top_atoms=top_atoms, seed=seed,
    )
    return [
        audit_atom_atomicity(problem, schedule, atom, executor=executor, ce=ce)
        for atom, ce in cases
    ]