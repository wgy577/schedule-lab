"""Solver Teacher Trace V1 -- deterministic replay and ablation of teacher edits.

Given a seed proposal plus its teacher diff (``Diff(S_local, S_global)``
decomposed into decision components and induced consequences), this module
answers -- deterministically, no LLM in the loop:

1. **Replay chain**: ``P0`` (seed only) -> ``P0+e1`` -> ``P0+e1+e2`` -> ... every
   step re-solved under *frozen-local* semantics.  **Only decision edits**
   (machine/mode/sequence components) are ever forced as interventions.  The
   order of the ``e_i`` is a **reconstructed edit dependency order**; it is
   explicitly NOT claimed to be CP-SAT's internal edit order, which is not
   observable.
2. **Consequence entailment**: timing-only consequences are never replayed as
   independent actions.  Each chain step's schedule is compared against
   ``S_global`` per consequence operation: reproduced timing means the shift is
   *propagation* of the decisions; never reproduced stays ``entailed=false``.
3. **Group ablation**: each LLM-suggested replay group (plus automatic
   singletons / the full set) replayed the same way, giving group-level
   contributions that can falsify "enables" hypotheses.
4. **Constraint evidence**: pure geometry checks (vacated machine window
   overlapping the seed's target window, precedence-bound release) that can
   support a blocker-release explanation without any replay.
5. **LLM hypothesis verification**: every DeepSeek hypothesis is re-checked
   against 1-4; anything the deterministic evidence does not support stays
   ``verified=false``.
6. **Classification** of each teacher decision into the frozen taxonomy
   A. direct improvement / B. enabling-preparation / C. feasibility repair /
   D. secondary optimization / E. unattributed.

Marginal contributions are reported under the exact name
``replay_based_marginal_improvement`` (they are cumulative-chain ablation
numbers, NOT Shapley values).  Nothing here writes to Trajectory Memory; the
caller passes a throwaway :class:`ExperienceStore`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..counterfactual import CounterfactualEvaluator, FROZEN_LOCAL
from ..ir import Problem, Schedule
from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    LegalEdit,
)
from ..memory import ExperienceStore
from ..validation import schedule_hash
from .solver_schedule_diff_v1 import SolverEdit

REPLAY_SCHEMA = "solver_teacher_replay_v1"

TEACHER_CLASS_DIRECT = "A_direct_improvement"
TEACHER_CLASS_ENABLING = "B_enabling_preparation"
TEACHER_CLASS_FEASIBILITY = "C_feasibility_repair"
TEACHER_CLASS_SECONDARY = "D_secondary_optimization"
TEACHER_CLASS_UNATTRIBUTED = "E_unattributed"
TEACHER_CLASSES = (
    TEACHER_CLASS_DIRECT,
    TEACHER_CLASS_ENABLING,
    TEACHER_CLASS_FEASIBILITY,
    TEACHER_CLASS_SECONDARY,
    TEACHER_CLASS_UNATTRIBUTED,
)

EVIDENCE_CONSTRAINT = "constraint"
EVIDENCE_REPLAY = "replay_verified"
EVIDENCE_LLM = "llm_hypothesis"


def _stable_group_suffix(edit_id: str) -> str:
    """Deterministic short suffix keyed on the edit id (L5).

    Keys the auto-single ablation group on the edit id so multiple decision
    components on one operation cannot collide onto a single ``g_<op>`` id.
    Keeps a readable ``<edit_id>`` when it is filesystem-safe, else falls back
    to a stable hash of it.
    """
    safe = all(c.isalnum() or c in {"_", "-", ".", ":"} for c in edit_id)
    if safe and edit_id:
        return edit_id.replace(":", "_")
    return hashlib.sha256(edit_id.encode("utf-8")).hexdigest()[:12]

__all__ = [
    "ReplayStep",
    "ConsequenceVerification",
    "TeacherReplayGroup",
    "TeacherEditDependency",
    "ReplayAnalysis",
    "teacher_edit_to_legal_edits",
    "replay_teacher_edits",
    "constraint_evidence",
    "verify_llm_hypotheses",
    "classify_teacher_edits",
    "REPLAY_SCHEMA",
    "TEACHER_CLASSES",
]


# ---------------------------------------------------------------------------
# teacher edit -> LegalEdit conversion
# ---------------------------------------------------------------------------


def _machine_sequences(problem: Problem, schedule: Schedule) -> dict[str, list[str]]:
    mode_map = problem.mode_map()
    rows: dict[str, list[tuple[float, float, str]]] = {}
    for assignment in schedule.assignments:
        resources = mode_map[assignment.mode_id][1].resources
        if len(resources) != 1:
            raise ValueError("teacher replay requires unary-resource modes")
        rows.setdefault(resources[0], []).append(
            (float(assignment.start), float(assignment.end), assignment.operation_id)
        )
    return {
        machine: [oid for _, _, oid in sorted(items)]
        for machine, items in rows.items()
    }


def _neighbours(sequence: Sequence[str], operation_id: str) -> tuple[str | None, str | None]:
    index = sequence.index(operation_id)
    return (
        sequence[index - 1] if index > 0 else None,
        sequence[index + 1] if index + 1 < len(sequence) else None,
    )


def teacher_edit_to_legal_edits(
    edit: SolverEdit,
    *,
    problem: Problem,
    local: Schedule,
    global_schedule: Schedule,
) -> tuple[LegalEdit, ...]:
    """Translate one teacher *decision* edit into executable LegalEdit steps.

    ``local`` is the edit's "before" world (the frozen-local result) and
    ``global_schedule`` the "after" world the solver reached.  The translation
    is geometric: routing -> forced mode, sequencing -> adjacency swap,
    insertion -> predecessor/successor pinning.  Timing components are never
    converted here -- they are propagation consequences, not interventions.
    """
    # M4 hard guard: a timing component is an *induced consequence* of the
    # decisions, never an intervention.  It must never fall through into a
    # sequencing/insertion LegalEdit.  Reject it explicitly before any other
    # branch so a mis-classified timing edit fails loud instead of silently
    # becoming a SEQ_INSERT.
    if edit.edit_type == "timing":
        raise ValueError(
            "timing components are induced consequences and cannot be "
            "converted to LegalEdit"
        )

    operation_id = edit.operation_id
    global_map = global_schedule.assignment_map()
    after = global_map[operation_id]
    target_mode = after.mode_id

    if edit.edit_type in {"routing", "mode_change"}:
        return (
            LegalEdit(
                edit_id=f"{edit.edit_id}:route",
                edit_type=EDIT_ROUTE,
                operation_id=operation_id,
                source_machine=(edit.source_resource if edit.source_resource else None),
                target_machine=(
                    edit.affected_resource if edit.machine_changed else None
                ),
                target_mode_id=target_mode,
            ),
        )
    if edit.edit_type == "sequencing":
        machine = edit.affected_resource
        sequence = _machine_sequences(problem, local)[machine]
        index = sequence.index(operation_id)
        delta = int(edit.after["sequence_position"]) - int(edit.before["sequence_position"])
        partner = sequence[index + 1] if delta > 0 else sequence[index - 1]
        left, right = (
            (operation_id, partner) if delta > 0 else (partner, operation_id)
        )
        return (
            LegalEdit(
                edit_id=f"{edit.edit_id}:seq",
                edit_type=EDIT_SEQ_SWAP,
                operation_id=operation_id,
                resource_id=machine,
                left_id=left,
                right_id=right,
            ),
        )
    if edit.edit_type == "insertion":
        # pin the operation between its global neighbours
        machine = edit.affected_resource
        global_sequence = _machine_sequences(problem, global_schedule)[machine]
        predecessor, successor = _neighbours(global_sequence, operation_id)
        return (
            LegalEdit(
                edit_id=f"{edit.edit_id}:seq",
                edit_type=EDIT_SEQ_INSERT,
                operation_id=operation_id,
                resource_id=machine,
                insert_position=global_sequence.index(operation_id),
                predecessor_id=predecessor,
                successor_id=successor,
            ),
        )
    # Exhaustive: no semantic fallthrough.  Any unknown / compound edit type is
    # a programming error and must fail loud rather than be coerced.
    raise ValueError(
        f"cannot convert teacher edit of type {edit.edit_type!r} to a LegalEdit; "
        "only routing / mode_change / sequencing / insertion are interventions"
    )


# ---------------------------------------------------------------------------
# replay engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayStep:
    step: int
    label: str                       # "P0" / "P0+e1" / ...
    cumulative_teacher_edits: tuple[str, ...]
    newly_added_edit: str | None
    delta_cmax: float
    replay_based_marginal_improvement: float
    feasible: bool
    solver_status: str
    cmax: float | None
    schedule_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "label": self.label,
            "cumulative_teacher_edits": list(self.cumulative_teacher_edits),
            "newly_added_edit": self.newly_added_edit,
            "delta_cmax": float(self.delta_cmax),
            "replay_based_marginal_improvement": float(
                self.replay_based_marginal_improvement
            ),
            "feasible": bool(self.feasible),
            "solver_status": self.solver_status,
            "cmax": self.cmax,
            "schedule_sha256": self.schedule_sha256,
        }


# entailment status vocabulary (M1)
ENTAILMENT_NEVER = "never_entailed"
ENTAILMENT_PRESERVED = "entailed_and_preserved"
ENTAILMENT_DISPLACED = "entailed_then_displaced"


@dataclass(frozen=True)
class ConsequenceVerification:
    """Was a timing-only consequence *entailed* by seed + the decision set?

    Entailment is an attribute of the **cumulative** decision set
    ``{seed} ∪ {e1..ek}`` (M2), never credited to the single last-added edit.
    ``entailed_by_cumulative_edits`` records the decision prefix that first
    reproduced the consequence's global assignment (empty == seed alone).

    A consequence that reproduces once is NOT permanently verified (M1):
    ``first_entailed_at_step`` records when it first matched, but the
    authoritative flag is ``entailed_at_final`` -- re-checked against the
    FINAL cumulative schedule after the whole chain replays.  ``entailed`` is
    kept as an alias and now means ``entailed_at_final`` ("still holds at the
    end"), not "appeared at some step".  ``entailment_status`` distinguishes
    ``never_entailed`` / ``entailed_and_preserved`` / ``entailed_then_displaced``.
    Only ``entailed_and_preserved`` consequences produce the strong
    ``replay_verified`` entailment edge; displaced ones stay diagnostic.
    """

    edit_id: str
    operation_id: str
    before_start: float
    after_start: float
    entailed: bool                          # == entailed_at_final (alias)
    first_entailed_at_step: int | None       # None when never entailed
    entailed_at_final: bool
    entailment_status: str                   # ENTAILMENT_* vocabulary
    entailed_by_cumulative_edits: tuple[str, ...]  # decision prefix; () == seed only
    entailed_by_edit: str                    # last-added edit at first entailment (diagnostic)
    evidence_source: str                     # replay_verified | unverified

    def to_json(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "operation_id": self.operation_id,
            "before_start": float(self.before_start),
            "after_start": float(self.after_start),
            "entailed": bool(self.entailed_at_final),
            "first_entailed_at_step": self.first_entailed_at_step,
            "entailed_at_final": bool(self.entailed_at_final),
            "entailment_status": self.entailment_status,
            "entailed_by_cumulative_edits": list(self.entailed_by_cumulative_edits),
            "entailing_set": ["seed", *self.entailed_by_cumulative_edits],
            "entailed_by_edit": self.entailed_by_edit,
            "evidence_source": self.evidence_source,
            "semantics": (
                "propagation consequence reproduced by seed+decision-set replay "
                "and still holding at the final step"
                if self.entailed_at_final else
                "not reproduced at the final step by seed+decision-set replay"
            ),
        }


@dataclass(frozen=True)
class TeacherReplayGroup:
    group_id: str
    label: str
    edit_ids: tuple[str, ...]
    source: str                      # llm_suggested | auto_single | auto_full
    delta_cmax: float
    feasible: bool
    solver_status: str
    replay_based_marginal_improvement: float

    def to_json(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "edit_ids": list(self.edit_ids),
            "source": self.source,
            "delta_cmax": float(self.delta_cmax),
            "feasible": bool(self.feasible),
            "solver_status": self.solver_status,
            "replay_based_marginal_improvement": float(
                self.replay_based_marginal_improvement
            ),
        }


@dataclass(frozen=True)
class TeacherEditDependency:
    relation_id: str
    relation_type: str               # enables | contributes_to | precedes
    source_edit_id: str              # teacher edit id or "seed"
    target_edit_id: str              # teacher edit id or "seed"
    evidence_source: str             # constraint | replay_verified | llm_hypothesis
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "source_edit_id": self.source_edit_id,
            "target_edit_id": self.target_edit_id,
            "evidence_source": self.evidence_source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReplayAnalysis:
    trace_id: str
    baseline_delta_cmax: float       # P0: seed only, frozen local
    final_delta_cmax: float
    global_delta_cmax: float
    replay_closes_gap: float         # local_delta - final replay delta
    steps: tuple[ReplayStep, ...]
    groups: tuple[TeacherReplayGroup, ...]
    dependencies: tuple[TeacherEditDependency, ...]
    classifications: Mapping[str, str]          # decision edit_id -> A-E class
    constraint_findings: tuple[Mapping[str, Any], ...]
    consequence_verifications: tuple[ConsequenceVerification, ...]
    llm_hypothesis_verification: tuple[Mapping[str, Any], ...] = ()
    edit_order_semantics: str = (
        "reconstructed edit dependency order; NOT CP-SAT's internal edit order"
    )
    # M3: replay-step truncation transparency under ``max_replay_steps``
    total_teacher_decisions: int = 0
    replayed_decision_count: int = 0
    truncated_replay_count: int = 0
    truncated_replay_edit_ids: tuple[str, ...] = ()
    replay_complete: bool = True
    training_eligible: bool = False
    identified: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "trace_id": self.trace_id,
            "baseline_delta_cmax": float(self.baseline_delta_cmax),
            "final_delta_cmax": float(self.final_delta_cmax),
            "global_delta_cmax": float(self.global_delta_cmax),
            "replay_closes_gap": float(self.replay_closes_gap),
            "steps": [step.to_json() for step in self.steps],
            "groups": [group.to_json() for group in self.groups],
            "dependencies": [dep.to_json() for dep in self.dependencies],
            "classifications": dict(self.classifications),
            "constraint_findings": [dict(f) for f in self.constraint_findings],
            "consequence_verifications": [
                v.to_json() for v in self.consequence_verifications
            ],
            "llm_hypothesis_verification": [
                dict(v) for v in self.llm_hypothesis_verification
            ],
            "edit_order_semantics": self.edit_order_semantics,
            "replay_truncation": {
                "total_teacher_decisions": self.total_teacher_decisions,
                "replayed_decision_count": self.replayed_decision_count,
                "truncated_replay_count": self.truncated_replay_count,
                "truncated_replay_edit_ids": list(self.truncated_replay_edit_ids),
                "replay_complete": self.replay_complete,
                "note": (
                    "full teacher chain verified"
                    if self.replay_complete else
                    "partial replay under configured max_replay_steps; "
                    "truncated decisions are NOT replayed and NOT attributed"
                ),
            },
            "training_eligible": False,
            "identified": False,
        }


def _reconstructed_order(edits: Sequence[SolverEdit]) -> tuple[SolverEdit, ...]:
    """Reconstructed dependency order over *decision* edits.

    This ordering only sequences the *replay*; it is never claimed to be the
    solver's own edit order (unobservable).  Routing/mode edits change which
    resource holds the operation, sequencing edits change the order --
    replaying structure first makes each step's frozen-local solve well-posed.
    """
    def rank(edit: SolverEdit) -> tuple[int, float, str]:
        if edit.edit_type in {"routing", "mode_change", "sequencing", "insertion"}:
            structural = 0
        else:
            structural = 1
        return (structural, float(edit.after["start"]), edit.edit_id)

    return tuple(sorted(edits, key=rank))


def _run_frozen_local(
    problem: Problem,
    before: Schedule,
    proposal,
    edits: Sequence[LegalEdit],
    *,
    causal_chain,
    root,
    transition_trace,
    solver_time: float,
    seed: int,
):
    store = ExperienceStore()
    return CounterfactualEvaluator(store, solver_time=solver_time, seed=seed).evaluate(
        problem, before, proposal, edits,
        store_outcome=False,
        counterfactual_mode=FROZEN_LOCAL,
        causal_chain=causal_chain,
        root=root,
        transition_trace=transition_trace,
    )


def replay_teacher_edits(
    *,
    problem: Problem,
    before: Schedule,                  # S0 (the proposal's incumbent)
    proposal,
    seed_edits: Sequence[LegalEdit],
    decisions: Sequence[SolverEdit],
    consequences: Sequence[SolverEdit],
    local: Schedule,
    global_schedule: Schedule,
    causal_chain,
    root,
    transition_trace,
    trace_id: str,
    global_delta_cmax: float,
    local_delta_cmax: float,
    solver_time: float = 5.0,
    seed: int = 0,
    max_replay_steps: int = 8,
    llm_suggested_groups: Sequence[Sequence[str]] = (),
) -> ReplayAnalysis:
    """Replay seed + teacher *decisions* stepwise under frozen-local semantics.

    Only decision edits (machine/mode/sequence components) are ever forced as
    interventions.  Timing-only consequences are NOT replayed as independent
    actions; instead each chain step's resulting schedule is compared against
    ``S_global`` per consequence operation -- a consequence whose global
    assignment the decisions replay reproduces is *entailed* (propagation),
    and one that is never reproduced stays explicitly unexplained.
    """
    full_order = _reconstructed_order(decisions)
    ordered = full_order[:max_replay_steps]
    truncated_decisions = full_order[max_replay_steps:]
    replay_complete = len(truncated_decisions) == 0
    legal_by_edit: dict[str, tuple[LegalEdit, ...]] = {}
    for edit in ordered:
        legal_by_edit[edit.edit_id] = teacher_edit_to_legal_edits(
            edit, problem=problem, local=local, global_schedule=global_schedule
        )

    global_assignments = global_schedule.assignment_map()
    steps: list[ReplayStep] = []
    cumulative: list[LegalEdit] = list(seed_edits)
    previous_delta: float | None = None
    # per-consequence first-entailment bookkeeping (M1/M2): first hit step, the
    # decision prefix that produced it, and the last edit added at that point.
    first_step: dict[str, int | None] = {e.edit_id: None for e in consequences}
    first_prefix: dict[str, tuple[str, ...]] = {e.edit_id: () for e in consequences}
    first_edit: dict[str, str] = {e.edit_id: "" for e in consequences}

    def _matches_global(schedule, operation_id: str) -> bool:
        if schedule is None:
            return False
        reached = schedule.assignment_map().get(operation_id)
        target = global_assignments.get(operation_id)
        return (
            target is not None and reached is not None
            and reached.mode_id == target.mode_id
            and abs(float(reached.start) - float(target.start)) <= 1e-9
            and abs(float(reached.end) - float(target.end)) <= 1e-9
        )

    def evaluate_cumulative():
        result = _run_frozen_local(
            problem, before, proposal, tuple(cumulative),
            causal_chain=causal_chain, root=root,
            transition_trace=transition_trace,
            solver_time=solver_time, seed=seed,
        )
        schedule = result.new_schedule
        cmax = (
            max((float(a.end) for a in schedule.assignments), default=None)
            if schedule is not None else None
        )
        sha = schedule_hash(schedule) if schedule is not None else ""
        return (
            float(result.delta_cmax),
            schedule is not None,
            str(result.solver_status),
            cmax,
            sha,
            schedule,
        )

    def check_entailments(
        schedule, step_index: int, new_edit: str | None, prefix: tuple[str, ...]
    ) -> None:
        """Record the FIRST step at which each consequence matches global.

        This is bookkeeping only (M1): first-hit never freezes the final
        verdict.  The authoritative ``entailed_at_final`` is decided by a
        separate re-check against the FINAL cumulative schedule after the
        whole chain has replayed.
        """
        if schedule is None:
            return
        for consequence in consequences:
            edit_id = consequence.edit_id
            if first_step[edit_id] is not None:
                continue
            if _matches_global(schedule, consequence.operation_id):
                first_step[edit_id] = step_index
                first_prefix[edit_id] = prefix
                first_edit[edit_id] = new_edit or ""

    # P0 -- the seed alone under frozen-local semantics
    delta, feasible, status, cmax, sha, schedule = evaluate_cumulative()
    check_entailments(schedule, 0, None, ())
    final_schedule = schedule           # tracks the last chain schedule (M1)
    steps.append(
        ReplayStep(
            step=0, label="P0", cumulative_teacher_edits=(), newly_added_edit=None,
            delta_cmax=delta, replay_based_marginal_improvement=0.0,
            feasible=feasible, solver_status=status, cmax=cmax, schedule_sha256=sha,
        )
    )
    previous_delta = delta
    baseline_delta = delta

    for index, edit in enumerate(ordered, start=1):
        cumulative.extend(legal_by_edit[edit.edit_id])
        delta, feasible, status, cmax, sha, schedule = evaluate_cumulative()
        prefix = tuple(e.edit_id for e in ordered[:index])
        check_entailments(schedule, index, edit.edit_id, prefix)
        final_schedule = schedule
        marginal = (
            float(previous_delta) - float(delta)
            if previous_delta is not None else 0.0
        )
        steps.append(
            ReplayStep(
                step=index, label="P0+" + "+".join(
                    f"e{n}" for n in range(1, index + 1)
                ),
                cumulative_teacher_edits=tuple(e.edit_id for e in ordered[:index]),
                newly_added_edit=edit.edit_id,
                delta_cmax=delta,
                replay_based_marginal_improvement=marginal,
                feasible=feasible, solver_status=status, cmax=cmax,
                schedule_sha256=sha,
            )
        )
        previous_delta = delta

    # M1 final re-verification: the authoritative verdict is whether each
    # consequence STILL matches global on the FINAL cumulative schedule, not
    # whether it appeared at some intermediate step.  Build the verifications
    # here so a consequence that reproduced once and was later displaced is
    # recorded as ``entailed_then_displaced`` and never emits a verified edge.
    entailments: dict[str, ConsequenceVerification] = {}
    for consequence in consequences:
        edit_id = consequence.edit_id
        first_at = first_step[edit_id]
        held_final = _matches_global(final_schedule, consequence.operation_id)
        if first_at is None and not held_final:
            status_str = ENTAILMENT_NEVER
        elif held_final:
            status_str = ENTAILMENT_PRESERVED
        else:
            status_str = ENTAILMENT_DISPLACED
        entailments[edit_id] = ConsequenceVerification(
            edit_id=edit_id,
            operation_id=consequence.operation_id,
            before_start=float(consequence.before["start"]),
            after_start=float(consequence.after["start"]),
            entailed=held_final,
            first_entailed_at_step=first_at,
            entailed_at_final=held_final,
            entailment_status=status_str,
            entailed_by_cumulative_edits=first_prefix[edit_id],
            entailed_by_edit=first_edit[edit_id],
            evidence_source=EVIDENCE_REPLAY if held_final else "unverified",
        )

    # group ablation (decision edits only) -------------------------------------
    groups: list[TeacherReplayGroup] = []
    edit_ids = [edit.edit_id for edit in ordered]
    group_specs: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "g_full", "all teacher decision edits",
            tuple(edit_ids), "auto_full",
        )
    ]
    for edit in ordered:
        # L5: one operation may contribute several decision components, so the
        # group id must key on the edit id (not just the operation) to avoid a
        # collision that would silently overwrite an ablation group.
        group_specs.append(
            (f"g_{_stable_group_suffix(edit.edit_id)}", f"single {edit.edit_id}",
             (edit.edit_id,), "auto_single")
        )
    seen_group_sets: set[frozenset[str]] = set()
    for index, ids in enumerate(llm_suggested_groups):
        wanted = tuple(dict.fromkeys(i for i in ids if i in set(edit_ids)))
        if not wanted:
            continue
        frozen = frozenset(wanted)
        if frozen in seen_group_sets:
            continue
        seen_group_sets.add(frozen)
        group_specs.append(
            (f"g_llm{index}", "llm suggested group", wanted, "llm_suggested")
        )

    for group_id, label, wanted, source in group_specs:
        if source == "auto_full" and frozenset(edit_ids) in seen_group_sets:
            continue
        if source != "auto_full":
            seen_group_sets.add(frozenset(wanted))
        cumulative = list(seed_edits)
        for edit_id in wanted:
            cumulative.extend(legal_by_edit[edit_id])
        delta, feasible, status, _, _, _ = evaluate_cumulative()
        groups.append(
            TeacherReplayGroup(
                group_id=group_id, label=label, edit_ids=wanted, source=source,
                delta_cmax=delta, feasible=feasible, solver_status=status,
                replay_based_marginal_improvement=(
                    float(baseline_delta) - float(delta)
                ),
            )
        )

    constraint_findings = constraint_evidence(
        problem=problem, local=local, global_schedule=global_schedule,
        seed_edits=seed_edits, teacher_edits=ordered,
    )
    dependencies = _build_dependencies(
        ordered, steps, groups, constraint_findings,
    ) + _consequence_dependencies(entailments, consequences)
    classifications = classify_teacher_edits(
        steps=steps, groups=groups, teacher_edits=ordered,
        constraint_findings=constraint_findings,
    )
    final_delta = steps[-1].delta_cmax if steps else baseline_delta
    return ReplayAnalysis(
        trace_id=trace_id,
        baseline_delta_cmax=baseline_delta,
        final_delta_cmax=final_delta,
        global_delta_cmax=float(global_delta_cmax),
        replay_closes_gap=float(local_delta_cmax) - float(final_delta),
        steps=tuple(steps),
        groups=tuple(groups),
        dependencies=tuple(dependencies),
        classifications=classifications,
        constraint_findings=tuple(constraint_findings),
        consequence_verifications=tuple(entailments.values()),
        total_teacher_decisions=len(full_order),
        replayed_decision_count=len(ordered),
        truncated_replay_count=len(truncated_decisions),
        truncated_replay_edit_ids=tuple(e.edit_id for e in truncated_decisions),
        replay_complete=replay_complete,
        training_eligible=False,
        identified=False,
    )


def _consequence_dependencies(
    entailments: Mapping[str, ConsequenceVerification],
    consequences: Sequence[SolverEdit],
) -> tuple[TeacherEditDependency, ...]:
    """Cumulative-decision-set -> consequence ``entails`` edges (M2).

    An entailment edge means: **seed + a cumulative decision set entails the
    consequence** -- NOT that a single edit uniquely caused it.  ``source`` is
    the seed when the consequence already held at P0 (L2: ``"seed"``, never an
    empty ``source_edit_id``), otherwise the last decision that completed the
    cumulative set that first reproduced it.  Only consequences that STILL hold
    at the final step (``entailed_and_preserved``) emit a verified edge (M1);
    ``entailed_then_displaced`` ones are diagnostic and produce no edge here.
    """
    relations: list[TeacherEditDependency] = []
    for edit in consequences:
        verification = entailments.get(edit.edit_id)
        if verification is None or not verification.entailed_at_final:
            continue
        if verification.entailment_status != ENTAILMENT_PRESERVED:
            continue
        cumulative_set = verification.entailed_by_cumulative_edits
        # L2: step-0 (seed-only) entailment is sourced to the seed, not "".
        source = verification.entailed_by_edit if cumulative_set else "seed"
        entailing_set = ["seed", *cumulative_set]
        relations.append(TeacherEditDependency(
            relation_id=f"dep:{source}->{edit.edit_id}:entails",
            relation_type="entails",
            source_edit_id=source,
            target_edit_id=edit.edit_id,
            evidence_source=EVIDENCE_REPLAY,
            detail=(
                f"cumulative decision set {entailing_set} first reproduces "
                f"{edit.operation_id}'s global timing at step "
                f"{verification.first_entailed_at_step} and it still holds at "
                f"the final step ({verification.before_start}->"
                f"{verification.after_start}); this is replay entailment of a "
                "cumulative decision set, NOT unique causal identification"
            ),
        ))
    return tuple(relations)


# ---------------------------------------------------------------------------
# constraint evidence (deterministic geometry, no solver, no LLM)
# ---------------------------------------------------------------------------


def constraint_evidence(
    *,
    problem: Problem,
    local: Schedule,
    global_schedule: Schedule,
    seed_edits: Sequence[LegalEdit],
    teacher_edits: Sequence[SolverEdit],
) -> tuple[dict[str, Any], ...]:
    """Deterministic blocker-release / precedence-release findings.

    A teacher routing edit counts as *vacating* evidence when it moves an
    operation OFF a machine that a seed edit wants to move an operation ONTO,
    and the vacated window overlaps the window the moved operation occupies in
    ``S_global``.  A teacher edit counts as *precedence release* when it shortens
    the end time of a same-job predecessor of a seed-edited operation.
    """
    findings: list[dict[str, Any]] = []
    global_map = global_schedule.assignments and global_schedule.assignment_map()
    predecessors: dict[str, set[str]] = {op.id: set() for op in problem.operations}
    successors: dict[str, set[str]] = {op.id: set() for op in problem.operations}
    for operation in problem.operations:
        for parent in operation.predecessors:
            predecessors[operation.id].add(parent)
            successors.setdefault(parent, set()).add(operation.id)

    for seed_edit in seed_edits:
        if seed_edit.edit_type != EDIT_ROUTE or not seed_edit.target_machine:
            continue
        moved = seed_edit.operation_id
        target_machine = seed_edit.target_machine
        target = global_map.get(moved)
        if target is None:
            continue
        window = (float(target.start), float(target.end))
        for teacher in teacher_edits:
            if not teacher.machine_changed:
                continue
            if teacher.source_resource != target_machine:
                continue
            if teacher.operation_id == moved:
                continue
            vacated = (
                float(teacher.before["start"]), float(teacher.before["end"])
            )
            overlaps = (
                vacated[0] < window[1] - 1e-9 and window[0] < vacated[1] - 1e-9
            )
            if overlaps:
                findings.append({
                    "finding": "vacated_window_overlap",
                    "seed_edit_id": seed_edit.edit_id,
                    "seed_operation": moved,
                    "seed_target_machine": target_machine,
                    "seed_global_window": list(window),
                    "teacher_edit_id": teacher.edit_id,
                    "teacher_operation": teacher.operation_id,
                    "vacated_window": list(vacated),
                    "evidence_source": EVIDENCE_CONSTRAINT,
                })
    # precedence release: teacher edit shortens a predecessor's end time
    seed_operations = {edit.operation_id for edit in seed_edits}
    for teacher in teacher_edits:
        before_end = float(teacher.before["end"])
        after_end = float(teacher.after["end"])
        if after_end >= before_end - 1e-9:
            continue
        for dependent in sorted(
            seed_operations & successors.get(teacher.operation_id, set())
        ):
            findings.append({
                "finding": "precedence_bound_release",
                "teacher_edit_id": teacher.edit_id,
                "teacher_operation": teacher.operation_id,
                "before_end": before_end,
                "after_end": after_end,
                "dependent_seed_operation": dependent,
                "evidence_source": EVIDENCE_CONSTRAINT,
            })
    return tuple(findings)


# ---------------------------------------------------------------------------
# dependency graph + classification
# ---------------------------------------------------------------------------


def _build_dependencies(
    teacher_edits: Sequence[SolverEdit],
    steps: Sequence[ReplayStep],
    groups: Sequence[TeacherReplayGroup],
    constraint_findings: Sequence[Mapping[str, Any]],
) -> tuple[TeacherEditDependency, ...]:
    """Assemble the teacher-edit dependency graph with evidence levels.

    Only ``constraint`` and ``replay_verified`` edges are strong evidence;
    ``llm_hypothesis`` edges (added by the caller after verification) are weak.
    """
    relations: list[TeacherEditDependency] = []    # constraint-level: teacher vacates a window the seed needs -> enables seed
    for finding in constraint_findings:
        if finding["finding"] == "vacated_window_overlap":
            relations.append(TeacherEditDependency(
                relation_id=(
                    f"dep:{finding['teacher_edit_id']}->seed:"
                    f"{finding['seed_edit_id']}"
                ),
                relation_type="enables",
                source_edit_id=str(finding["teacher_edit_id"]),
                target_edit_id="seed",
                evidence_source=EVIDENCE_CONSTRAINT,
                detail=(
                    f"{finding['teacher_operation']} vacates "
                    f"{finding['seed_target_machine']} window "
                    f"{finding['vacated_window']} which overlaps seed "
                    f"{finding['seed_operation']}'s global window "
                    f"{finding['seed_global_window']}"
                ),
            ))

    # replay-level: single-group no improvement but a containing group does ->
    # enabling between teacher edits (replay_verified).
    singles = {
        g.edit_ids[0]: g for g in groups
        if g.source == "auto_single" and len(g.edit_ids) == 1
    }
    for group in groups:
        if len(group.edit_ids) < 2:
            continue
        combined_singles = sum(
            singles[e].replay_based_marginal_improvement
            for e in group.edit_ids if e in singles
        )
        if group.feasible and (
            group.replay_based_marginal_improvement
            > combined_singles + 1e-6
        ):
            for edit_id in group.edit_ids:
                for other in group.edit_ids:
                    if other == edit_id:
                        continue
                    relations.append(TeacherEditDependency(
                        relation_id=f"dep:{other}->{edit_id}:replay",
                        relation_type="enables",
                        source_edit_id=other,
                        target_edit_id=edit_id,
                        evidence_source=EVIDENCE_REPLAY,
                        detail=(
                            f"group {group.label} improves "
                            f"{group.replay_based_marginal_improvement:.3f} but its "
                            f"singles sum to {combined_singles:.3f}"
                        ),
                    ))
    return tuple(relations)


def classify_teacher_edits(
    *,
    steps: Sequence[ReplayStep],
    groups: Sequence[TeacherReplayGroup],
    teacher_edits: Sequence[SolverEdit],
    constraint_findings: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-6,
) -> dict[str, str]:
    """Classify each teacher *decision* edit into the frozen A-E taxonomy.

    Classification is driven by the replay numbers only:

    * C feasibility repair -- some replay containing the edit is feasible while
      the same prefix without it is not.
    * A direct improvement -- the edit's cumulative-chain marginal improvement
      is positive.
    * B enabling preparation -- the edit alone shows no improvement but a
      containing group improves more than the sum of its singles (its value is
      released only together with others), or constraint evidence marks it as
      vacating a window the seed needs.
    * D secondary optimization -- a decision whose marginal improvement is ~0
      and which carries no enabling evidence (non-critical polish).
    * E unattributed -- everything else (including infeasible/unknown replays).

    Timing-only consequences never enter here: they are propagation effects,
    verified separately by replay entailment.
    """
    edit_ids = [edit.edit_id for edit in teacher_edits]
    singles = {
        g.edit_ids[0]: g for g in groups
        if g.source == "auto_single" and len(g.edit_ids) == 1
    }
    marginal_by_edit: dict[str, float] = {
        step.newly_added_edit: step.replay_based_marginal_improvement
        for step in steps
        if step.newly_added_edit is not None
    }
    enables_seed = {
        str(f["teacher_edit_id"]) for f in constraint_findings
        if f["finding"] == "vacated_window_overlap"
    }
    cooperative: set[str] = set()
    for group in groups:
        if len(group.edit_ids) < 2 or not group.feasible:
            continue
        combined = sum(
            singles[e].replay_based_marginal_improvement
            for e in group.edit_ids if e in singles
        )
        if group.replay_based_marginal_improvement > combined + 1e-6:
            cooperative.update(group.edit_ids)

    classifications: dict[str, str] = {}
    for edit_id in edit_ids:
        single = singles.get(edit_id)
        marginal = marginal_by_edit.get(edit_id)
        edit = next(e for e in teacher_edits if e.edit_id == edit_id)
        # feasibility repair: prefix without the edit infeasible, with feasible
        repaired = False
        for step in steps:
            if step.newly_added_edit != edit_id:
                continue
            previous = next(
                (s for s in steps if s.step == step.step - 1), None
            )
            if previous is not None and not previous.feasible and step.feasible:
                repaired = True
        if repaired:
            classifications[edit_id] = TEACHER_CLASS_FEASIBILITY
        elif marginal is not None and marginal > tolerance:
            classifications[edit_id] = TEACHER_CLASS_DIRECT
        elif (
            edit_id in cooperative
            or edit_id in enables_seed
            or (single is not None and not single.feasible)
        ):
            classifications[edit_id] = TEACHER_CLASS_ENABLING
        elif marginal is not None and abs(marginal) <= tolerance:
            classifications[edit_id] = TEACHER_CLASS_SECONDARY
        else:
            classifications[edit_id] = TEACHER_CLASS_UNATTRIBUTED
    return classifications


# ---------------------------------------------------------------------------
# LLM hypothesis verification
# ---------------------------------------------------------------------------


def verify_llm_hypotheses(
    *,
    llm_analysis: Mapping[str, Any] | None,
    replay: ReplayAnalysis,
    tolerance: float = 1e-6,
) -> tuple[dict[str, Any], ...]:
    """Re-check every DeepSeek hypothesis against deterministic evidence.

    Each hypothesis is scored against (a) constraint findings and (b) the
    replay chain/groups.  Anything the deterministic evidence does not support
    is reported with ``verified=false`` -- the LLM never upgrades itself.
    """
    if not llm_analysis:
        return ()
    opinions = (
        llm_analysis.get("edit_opinions")
        or llm_analysis.get("edits")
        or ()
    )
    findings = replay.constraint_findings
    enables_seed = {
        str(f["teacher_edit_id"]) for f in findings
        if f["finding"] == "vacated_window_overlap"
    }
    marginal_by_edit = {
        step.newly_added_edit: step.replay_based_marginal_improvement
        for step in replay.steps
        if step.newly_added_edit is not None
    }
    consequence_by_op: dict[str, ConsequenceVerification] = {
        verification.operation_id: verification
        for verification in replay.consequence_verifications
    }
    consequence_by_edit: dict[str, ConsequenceVerification] = {
        verification.edit_id: verification
        for verification in replay.consequence_verifications
    }
    decision_edit_ids: set[str] = {
        step.newly_added_edit for step in replay.steps
        if step.newly_added_edit is not None
    }
    single_results = {
        group.edit_ids[0]: group
        for group in replay.groups
        if group.source == "auto_single" and len(group.edit_ids) == 1
    }
    group_results = {
        frozenset(group.edit_ids): group for group in replay.groups
    }
    results: list[dict[str, Any]] = []

    for edit_opinion in opinions:
        edit_id = str(edit_opinion.get("edit_id") or "")
        operation_id = str(edit_opinion.get("operation_id") or "")
        role = str(edit_opinion.get("likely_role") or "")
        checks: list[str] = []
        verified_role = False

        # consequence hypotheses: verified only by replay entailment.
        # The operation-level fallback is gated on (a) the hypothesized role
        # being a propagation role and (b) the edit id not being a replayed
        # decision -- one operation can carry both a route decision and a
        # timing consequence (e.g. J1.O2), and the decision hypothesis must
        # never be misrouted into the consequence branch.
        consequence = consequence_by_edit.get(edit_id)
        if (
            consequence is None
            and role in {"induced_propagation", "propagation"}
            and edit_id not in decision_edit_ids
        ):
            consequence = consequence_by_op.get(operation_id)
        if consequence is not None:
            if role in {"induced_propagation", "propagation"}:
                # M1: only consequences still entailed at the FINAL step verify
                # a propagation hypothesis; a displaced one does not.
                verified_role = consequence.entailed_at_final
                if verified_role:
                    checks.append(
                        "replay: consequence timing reproduced by "
                        "seed+decisions replay"
                    )
            results.append({
                "hypothesis_edit_id": edit_id,
                "operation_id": operation_id,
                "hypothesized_role": role,
                "verified": bool(verified_role),
                "evidence": checks,
                "note": (
                    "consequence hypothesis checked against replay entailment"
                ),
            })
            continue

        # decision hypotheses: constraint + replay marginal evidence
        candidates = [
            step_edit for step_edit in marginal_by_edit
            if step_edit == edit_id or (operation_id and operation_id in step_edit)
        ]
        if candidates:
            target = candidates[0]
            if role in {"blocker_release", "preparation"}:
                verified_role = target in enables_seed or any(
                    f.get("teacher_edit_id") == target for f in findings
                )
                if verified_role:
                    checks.append("constraint: vacated window / precedence release")
            if role == "direct_improvement":
                single_gain = (
                    single_results[target].replay_based_marginal_improvement
                    if target in single_results else 0.0
                )
                verified_role = (
                    marginal_by_edit.get(target, 0.0) > tolerance
                    or single_gain > tolerance
                )
                if verified_role:
                    checks.append("replay: positive marginal improvement")
        results.append({
            "hypothesis_edit_id": edit_id,
            "operation_id": operation_id,
            "hypothesized_role": role,
            "verified": bool(verified_role),
            "evidence": checks,
            "note": (
                "deterministic evidence does not support this hypothesis"
                if not verified_role else
                "supported by deterministic constraint/replay evidence"
            ),
        })

    for group in llm_analysis.get("suggested_replay_groups", []) or []:
        wanted = [str(item) for item in group] if isinstance(group, list) else []
        if not wanted:
            continue
        matched = group_results.get(frozenset(wanted))
        results.append({
            "hypothesis": "suggested_replay_group",
            "group": wanted,
            "verified": matched is not None and matched.feasible,
            "evidence": (
                [f"replay group delta_cmax={matched.delta_cmax}"] if matched else []
            ),
            "note": (
                "group was replayed"
                if matched is not None else
                "group did not match a replayed set"
            ),
        })
    return tuple(results)
