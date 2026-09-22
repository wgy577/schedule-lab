"""Solver Teacher Trace V1 -- deterministic schedule diff extraction.

Compares two schedules (typically the frozen-local result ``S_local`` and the
free-global result ``S_global`` of the *same* proposal) and extracts, per
operation, the structured :class:`SolverEdit` changes the global solver made on
top of the local repair:

* ``routing``      -- the operation moved to another machine
* ``mode_change``  -- same machine, different mode (different duration)
* ``sequencing``   -- adjacent order swap with a machine neighbour
* ``insertion``    -- non-adjacent re-positioning on the same machine
* ``timing``       -- same machine/mode/order, only start/end moved
* ``compound``     -- several of the above at once for one operation

Everything here is deterministic and torch-free: it only reads assignments and
the problem's mode table.  It never judges improvement, causality or labels --
it is pure geometry of the schedule change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..ir import Problem, Schedule
from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    EDIT_TIMING_SHIFT,
    LegalEdit,
)

SOLVER_EDIT_ROUTING = "routing"
SOLVER_EDIT_SEQUENCING = "sequencing"
SOLVER_EDIT_INSERTION = "insertion"
SOLVER_EDIT_TIMING = "timing"
SOLVER_EDIT_MODE_CHANGE = "mode_change"
SOLVER_EDIT_COMPOUND = "compound"
SOLVER_EDIT_TYPES = (
    SOLVER_EDIT_ROUTING,
    SOLVER_EDIT_SEQUENCING,
    SOLVER_EDIT_INSERTION,
    SOLVER_EDIT_TIMING,
    SOLVER_EDIT_MODE_CHANGE,
    SOLVER_EDIT_COMPOUND,
)

__all__ = [
    "SolverEdit",
    "SOLVER_EDIT_TYPES",
    "SOLVER_EDIT_ROUTING",
    "SOLVER_EDIT_SEQUENCING",
    "SOLVER_EDIT_INSERTION",
    "SOLVER_EDIT_TIMING",
    "SOLVER_EDIT_MODE_CHANGE",
    "SOLVER_EDIT_COMPOUND",
    "SeedAbsorbedComponent",
    "TeacherEditSet",
    "diff_schedules",
    "decompose_diff_edits",
    "extract_teacher_components",
    "changed_operation_ids",
]


def _machine_of(problem: Problem, mode_id: str) -> str:
    mode = problem.mode_map()[mode_id][1]
    if len(mode.resources) != 1:
        raise ValueError(f"solver diff requires unary-resource modes; got {mode.id}")
    return mode.resources[0]


def _duration_of(problem: Problem, mode_id: str) -> float:
    return float(problem.mode_map()[mode_id][1].duration)


def _resource_sequences(
    problem: Problem, schedule: Schedule
) -> dict[str, list[str]]:
    """machine -> operation ids in (start, end, id) order."""
    by_machine: dict[str, list[tuple[float, float, str]]] = {}
    mode_map = problem.mode_map()
    for assignment in schedule.assignments:
        machine = mode_map[assignment.mode_id][1].resources[0]
        by_machine.setdefault(machine, []).append(
            (float(assignment.start), float(assignment.end), assignment.operation_id)
        )
    return {
        machine: [oid for _, _, oid in sorted(rows)]
        for machine, rows in by_machine.items()
    }


def _order_index(sequence: Sequence[str], operation_id: str) -> int:
    for index, oid in enumerate(sequence):
        if oid == operation_id:
            return index
    return -1


@dataclass(frozen=True)
class SolverEdit:
    """One operation's structured change between two schedules."""

    edit_id: str
    operation_id: str
    edit_type: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    machine_changed: bool
    mode_changed: bool
    timing_changed: bool
    sequence_changed: bool
    affected_resource: str        # machine the change is about (target machine)
    source_resource: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "operation_id": self.operation_id,
            "edit_type": self.edit_type,
            "before": dict(self.before),
            "after": dict(self.after),
            "machine_changed": self.machine_changed,
            "mode_changed": self.mode_changed,
            "timing_changed": self.timing_changed,
            "sequence_changed": self.sequence_changed,
            "affected_resource": self.affected_resource,
            "source_resource": self.source_resource,
        }


def changed_operation_ids(before: Schedule, after: Schedule) -> tuple[str, ...]:
    """Operations whose (mode, start, end) differ between the two schedules."""
    left, right = before.assignment_map(), after.assignment_map()
    changed = []
    for operation_id in sorted(set(left) | set(right)):
        a, b = left.get(operation_id), right.get(operation_id)
        if a is None or b is None or (
            a.mode_id != b.mode_id
            or abs(float(a.start) - float(b.start)) > 1e-9
            or abs(float(a.end) - float(b.end)) > 1e-9
        ):
            changed.append(operation_id)
    return tuple(changed)


def diff_schedules(
    problem: Problem, before: Schedule, after: Schedule
) -> tuple[SolverEdit, ...]:
    """Deterministically extract every per-operation change ``before -> after``.

    The sequence detector only considers operations that stay on the same
    machine in both schedules: an operation that changed machine is a
    routing/mode change, and its new position on the target machine is implied
    by the move, not an independent sequencing decision.
    """
    before_map, after_map = before.assignment_map(), after.assignment_map()
    before_seq = _resource_sequences(problem, before)
    after_seq = _resource_sequences(problem, after)
    edits: list[SolverEdit] = []

    for operation_id in changed_operation_ids(before, after):
        a, b = before_map.get(operation_id), after_map.get(operation_id)
        if a is None or b is None:
            continue  # appearance/disappearance is not a solver edit
        machine_before = _machine_of(problem, a.mode_id)
        machine_after = _machine_of(problem, b.mode_id)
        mode_changed = a.mode_id != b.mode_id
        machine_changed = machine_before != machine_after
        timing_changed = abs(float(a.start) - float(b.start)) > 1e-9 or abs(
            float(a.end) - float(b.end)
        ) > 1e-9

        sequence_changed = False
        position_delta = 0
        if not machine_changed:
            seq_b = before_seq.get(machine_before, ())
            seq_a = after_seq.get(machine_after, ())
            # relative order w.r.t. the ops that stayed on this machine
            stay = [oid for oid in seq_b if oid in seq_a]
            idx_b = _order_index(stay, operation_id)
            idx_a = _order_index(
                [oid for oid in seq_a if oid in stay], operation_id
            )
            if idx_b >= 0 and idx_a >= 0:
                position_delta = idx_a - idx_b
                sequence_changed = position_delta != 0

        if machine_changed or mode_changed:
            if timing_changed or sequence_changed:
                edit_type = SOLVER_EDIT_COMPOUND
            elif machine_changed:
                edit_type = SOLVER_EDIT_ROUTING
            else:
                edit_type = SOLVER_EDIT_MODE_CHANGE
        elif sequence_changed:
            edit_type = (
                SOLVER_EDIT_SEQUENCING if abs(position_delta) == 1
                else SOLVER_EDIT_INSERTION
            )
        else:
            edit_type = SOLVER_EDIT_TIMING

        edit = SolverEdit(
            edit_id=(
                f"solver:{operation_id}:{machine_before}->{machine_after}"
                if machine_changed
                else f"solver:{operation_id}:{machine_after}"
            ),
            operation_id=operation_id,
            edit_type=edit_type,
            before={
                "machine": machine_before,
                "mode": a.mode_id,
                "start": float(a.start),
                "end": float(a.end),
                "duration": _duration_of(problem, a.mode_id),
                "sequence_position": _order_index(
                    before_seq.get(machine_before, ()), operation_id
                ),
            },
            after={
                "machine": machine_after,
                "mode": b.mode_id,
                "start": float(b.start),
                "end": float(b.end),
                "duration": _duration_of(problem, b.mode_id),
                "sequence_position": _order_index(
                    after_seq.get(machine_after, ()), operation_id
                ),
            },
            machine_changed=machine_changed,
            mode_changed=mode_changed,
            timing_changed=timing_changed,
            sequence_changed=sequence_changed,
            affected_resource=machine_after,
            source_resource=machine_before if machine_changed else None,
        )
        edits.append(edit)
    return tuple(sorted(edits, key=lambda e: e.edit_id))


def decompose_diff_edits(edits: Sequence[SolverEdit]) -> tuple[SolverEdit, ...]:
    """Split each full per-operation diff into single-component edits.

    A compound diff (e.g. machine move + re-timing) becomes one *decision*
    component (routing / mode_change) plus one *consequence* component
    (timing).  The component's flags describe only that component; the
    ``before``/``after`` dicts still carry the full per-operation geometry the
    converters need.
    """
    components: list[SolverEdit] = []
    for edit in edits:
        if edit.machine_changed or edit.mode_changed:
            components.append(replace(
                edit,
                edit_id=(
                    f"{edit.edit_id}:route" if edit.machine_changed
                    else f"{edit.edit_id}:mode"
                ),
                edit_type=(
                    SOLVER_EDIT_ROUTING if edit.machine_changed
                    else SOLVER_EDIT_MODE_CHANGE
                ),
                timing_changed=False,
                sequence_changed=False,
            ))
        if edit.sequence_changed:
            position_delta = (
                int(edit.after["sequence_position"])
                - int(edit.before["sequence_position"])
            )
            components.append(replace(
                edit,
                edit_id=f"{edit.edit_id}:seq",
                edit_type=(
                    SOLVER_EDIT_SEQUENCING if abs(position_delta) == 1
                    else SOLVER_EDIT_INSERTION
                ),
                machine_changed=False,
                mode_changed=False,
                timing_changed=False,
                sequence_changed=True,
            ))
        if edit.timing_changed:
            components.append(replace(
                edit,
                edit_id=f"{edit.edit_id}:timing",
                edit_type=SOLVER_EDIT_TIMING,
                machine_changed=False,
                mode_changed=False,
                timing_changed=True,
                sequence_changed=False,
            ))
    return tuple(components)


@dataclass(frozen=True)
class SeedAbsorbedComponent:
    """A diff component that merely realizes the seed's own request.

    Happens when the frozen-local repair could NOT apply the seed component
    (S_local still shows the old value) while the global solver did apply
    exactly what the seed asked for.  Such a component is the seed's action,
    not teacher behaviour, and is subtracted (recorded, never silently
    dropped).
    """

    operation_id: str
    component: str          # routing | mode_change | sequencing | insertion | timing
    seed_edit_id: str
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "component": self.component,
            "seed_edit_id": self.seed_edit_id,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TeacherEditSet:
    """Component-level teacher diff: decisions vs induced consequences.

    * ``decisions`` -- machine/mode/sequence components: structural choices the
      solver made on top of the local repair.  These are the only candidates
      for "solver action".
    * ``consequences`` -- timing-only components: start/end shifts that may be
      pure propagation of the decisions (precedence release, freed machine
      window).  Whether they really are propagation is verified later by
      replay entailment, never assumed here.
    * ``seed_absorbed`` -- components that merely echo the seed's own request.

    The ``*_count`` / ``truncated_*_ids`` fields make the ``max_teacher_edits``
    cap **explicit**: ``decisions`` / ``consequences`` hold only the retained
    prefix, but the full totals and the ids that were dropped are recorded so a
    truncated diff never reads like a complete one.
    """

    decisions: tuple[SolverEdit, ...] = ()
    consequences: tuple[SolverEdit, ...] = ()
    seed_absorbed: tuple[SeedAbsorbedComponent, ...] = ()
    total_decision_count: int = 0
    retained_decision_count: int = 0
    truncated_decision_count: int = 0
    truncated_decision_ids: tuple[str, ...] = ()
    total_consequence_count: int = 0
    retained_consequence_count: int = 0
    truncated_consequence_count: int = 0
    truncated_consequence_ids: tuple[str, ...] = ()

    @property
    def decision_operations(self) -> tuple[str, ...]:
        return tuple(sorted({e.operation_id for e in self.decisions}))

    @property
    def consequence_operations(self) -> tuple[str, ...]:
        return tuple(sorted({e.operation_id for e in self.consequences}))

    @property
    def complete(self) -> bool:
        """True when nothing was dropped by ``max_teacher_edits``."""
        return (
            self.truncated_decision_count == 0
            and self.truncated_consequence_count == 0
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "teacher_decisions": [e.to_json() for e in self.decisions],
            "induced_consequences": [e.to_json() for e in self.consequences],
            "seed_absorbed_components": [a.to_json() for a in self.seed_absorbed],
            "teacher_edit_semantics": (
                "component-level subtraction of seed-requested dimensions; "
                "decisions vs induced consequences"
            ),
            "truncation": {
                "total_decision_count": self.total_decision_count,
                "retained_decision_count": self.retained_decision_count,
                "truncated_decision_count": self.truncated_decision_count,
                "truncated_decision_ids": list(self.truncated_decision_ids),
                "total_consequence_count": self.total_consequence_count,
                "retained_consequence_count": self.retained_consequence_count,
                "truncated_consequence_count": self.truncated_consequence_count,
                "truncated_consequence_ids": list(self.truncated_consequence_ids),
                "teacher_diff_complete": self.complete,
            },
        }


def _matches_seed_request(
    component: SolverEdit,
    seed_edits: Sequence[LegalEdit],
    *,
    global_sequences: Mapping[str, Sequence[str]],
) -> str | None:
    """Seed edit id if ``component`` realizes exactly what the seed requested.

    Direction matters: a machine change on a seed-edited op counts as the
    seed's own echo only when the global result IS the seed's target; if the
    solver moved the op somewhere else, that residual is teacher behaviour.
    """
    for seed in seed_edits:
        if (
            component.edit_type in {SOLVER_EDIT_ROUTING, SOLVER_EDIT_MODE_CHANGE}
            and seed.edit_type == EDIT_ROUTE
        ):
            if seed.target_mode_id and component.after["mode"] == seed.target_mode_id:
                return seed.edit_id
            if (
                seed.target_machine
                and not seed.target_mode_id
                and component.after["machine"] == seed.target_machine
            ):
                return seed.edit_id
        if component.edit_type == SOLVER_EDIT_TIMING and seed.edit_type == EDIT_TIMING_SHIFT:
            if seed.target_start is not None and abs(
                float(seed.target_start) - float(component.after["start"])
            ) <= 1e-9:
                return seed.edit_id
        if component.edit_type in {SOLVER_EDIT_SEQUENCING, SOLVER_EDIT_INSERTION}:
            sequence = global_sequences.get(
                str(component.after["machine"]), ()
            )
            if (
                seed.edit_type == EDIT_SEQ_INSERT
                and seed.insert_position is not None
                and component.operation_id in sequence
                and sequence.index(component.operation_id) == seed.insert_position
            ):
                return seed.edit_id
            if (
                seed.edit_type == EDIT_SEQ_SWAP
                and seed.left_id in sequence
                and seed.right_id in sequence
                and sequence.index(seed.right_id) < sequence.index(seed.left_id)
            ):
                return seed.edit_id
    return None


def extract_teacher_components(
    problem: Problem,
    local: Schedule,
    global_schedule: Schedule,
    *,
    seed_edits: Sequence[LegalEdit],
    max_teacher_edits: int = 12,
) -> TeacherEditSet:
    """``Diff(S_local, S_global)`` minus the seed's requested components.

    Component-level subtraction (NOT operation-level exclusion): a diff
    component on a seed-edited operation is dropped only when it is exactly
    what the seed requested; every other component -- including timing shifts
    of seed-edited operations -- is kept, as a decision or a consequence.
    """
    full = diff_schedules(problem, local, global_schedule)
    components = decompose_diff_edits(full)
    seeds_by_op: dict[str, list[LegalEdit]] = {}
    for seed in seed_edits:
        seeds_by_op.setdefault(seed.operation_id, []).append(seed)
    global_sequences = _resource_sequences(problem, global_schedule)

    decisions: list[SolverEdit] = []
    consequences: list[SolverEdit] = []
    absorbed: list[SeedAbsorbedComponent] = []
    for component in components:
        seed_match = _matches_seed_request(
            component, seeds_by_op.get(component.operation_id, ()),
            global_sequences=global_sequences,
        )
        if seed_match is not None:
            absorbed.append(SeedAbsorbedComponent(
                operation_id=component.operation_id,
                component=component.edit_type,
                seed_edit_id=seed_match,
                detail=(
                    f"{component.before['machine']}->{component.after['machine']} "
                    f"realizes seed {seed_match}"
                    if component.edit_type in {SOLVER_EDIT_ROUTING, SOLVER_EDIT_MODE_CHANGE}
                    else f"start {component.before['start']}->{component.after['start']} "
                    f"realizes seed {seed_match}"
                    if component.edit_type == SOLVER_EDIT_TIMING
                    else f"reposition realizes seed {seed_match}"
                ),
            ))
            continue
        if component.edit_type == SOLVER_EDIT_TIMING:
            consequences.append(component)
        else:
            decisions.append(component)
    retained_decisions = decisions[:max_teacher_edits]
    retained_consequences = consequences[:max_teacher_edits]
    dropped_decisions = decisions[max_teacher_edits:]
    dropped_consequences = consequences[max_teacher_edits:]
    return TeacherEditSet(
        decisions=tuple(retained_decisions),
        consequences=tuple(retained_consequences),
        seed_absorbed=tuple(absorbed),
        total_decision_count=len(decisions),
        retained_decision_count=len(retained_decisions),
        truncated_decision_count=len(dropped_decisions),
        truncated_decision_ids=tuple(e.edit_id for e in dropped_decisions),
        total_consequence_count=len(consequences),
        retained_consequence_count=len(retained_consequences),
        truncated_consequence_count=len(dropped_consequences),
        truncated_consequence_ids=tuple(e.edit_id for e in dropped_consequences),
    )
