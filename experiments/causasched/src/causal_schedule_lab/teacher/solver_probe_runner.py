"""Atomic counterfactual probe runner (Phase2改.md §1/§11/§13/§15).

Both the atomic probe ``do(r, a)`` and the joint probe ``do(B)`` realise the
counterfactual through the **AtomicCounterfactualExecutor** (fixed-decision DAG
replay), never through the production CP-SAT repair executor.  This is the fix
for the Phase 2.6 finding that ``CE_A(r)`` was actually ``CE_A(r + global
reoptimisation)``.

Appearance recompute is **fail-closed** (§11): a matcher exception or a missing
target block yields ``valid=False`` (no CE label), never ``appearance_after=0``
(which would fabricate ``CE≈1``).  Only ``valid==True`` produces a CE.

The old ``DeterministicOperatorExecutor`` remains the M3 / optimisation / repair
path and is no longer used for CE label generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ir import Problem, Schedule
from ..objective import evaluate_objective
from ..symptom_pruning import diagnose_and_prune
from ..validation import schedule_hash
from .atom_generator import DecisionAtom
from .atomic_counterfactual_executor import AtomicCounterfactualExecutor
from .counterfactual_label import causal_effect, makespan_gain

JOINT_PROBE_VERSION = "atomic-counterfactual-ce-probe-2.0"


@dataclass(frozen=True)
class AppearanceRecomputeResult:
    """Fail-closed appearance recompute (§11).

    ``valid=False`` means no CE label may be produced (matcher exception or
    target block not located).  ``valid=True`` with ``score=0.0`` is a genuine
    elimination (the matched block's magnitude is zero).
    """

    valid: bool
    score: float | None
    reason: str
    # Detection score D_A of the matched post-intervention block (Phase 2.10C §16
    # D/M consistency audit).  None when valid=False (no matched block).
    detection_score: float | None = None

    @property
    def value_or_none(self) -> float | None:
        return self.score if self.valid else None


@dataclass(frozen=True)
class AtomicProbeResult:
    """One ``do(r, a)`` atomic counterfactual with its appearance effect."""

    atom_id: str
    atom_type: str
    probe_operator_id: str
    probe_parameters: dict[str, Any]
    appearance_before: float
    appearance_after: float
    ce: float
    label_valid: bool
    makespan_before: int
    makespan_after: int | None
    makespan_gain: float
    feasible: bool
    classification: str
    solver_status: str
    # Phase 2.10C §8/§16: detection score D_A before/after (None when the
    # caller did not supply detection_before or the recompute was invalid).
    detection_before: float | None = None
    detection_after: float | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "probe": {
                "type": self.probe_operator_id,
                "parameters": self.probe_parameters,
            },
            "appearance_after": self.appearance_after,
            "ce": self.ce,
            "label_valid": self.label_valid,
            "cmax_before": self.makespan_before,
            "cmax_after": self.makespan_after,
            "makespan_gain": self.makespan_gain,
            "feasible": self.feasible,
            "classification": self.classification,
            "solver_status": self.solver_status,
            "detection_before": self.detection_before,
            "detection_after": self.detection_after,
        }


@dataclass(frozen=True)
class JointProbeResult:
    """One ``do(B)`` joint atomic counterfactual over a root set ``B``."""

    atom_ids: tuple[str, ...]
    appearance_before: float
    appearance_after: float
    ce: float
    label_valid: bool
    makespan_before: int
    makespan_after: int | None
    makespan_gain: float
    feasible: bool
    classification: str
    solver_status: str
    conflict: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "atoms": list(self.atom_ids),
            "ce_joint": self.ce,
            "label_valid": self.label_valid,
            "appearance_after": self.appearance_after,
            "cmax_before": self.makespan_before,
            "cmax_after": self.makespan_after,
            "makespan_gain": self.makespan_gain,
            "feasible": self.feasible,
            "classification": self.classification,
            "solver_status": self.solver_status,
            "conflict": self.conflict,
        }


def recompute_appearance_atomic(
    problem: Problem,
    schedule: Schedule,
    appearance_type: str,
    target_operations: tuple[str, ...],
) -> AppearanceRecomputeResult:
    """Recompute the target appearance on a post-probe schedule, fail-closed.

    * matcher / ``diagnose_and_prune`` exception -> ``valid=False`` (§11 情况 C)
    * no same-rule block with sufficient overlap -> ``valid=False`` (§11 情况 B;
      the weak Jaccard matcher cannot distinguish "eliminated" from "not found",
      so we refuse to fabricate a label)
    * matched block -> ``valid=True`` with the block's magnitude (may be 0.0 =
      genuine elimination, §11 情况 A)
    """
    try:
        snapshot = diagnose_and_prune(problem, schedule)
        from ..sg_sct_causal_probe import (
            find_matching_block, appearance_magnitude, appearance_detection_score,
        )

        matched = find_matching_block(snapshot, appearance_type, target_operations)
        if matched is None:
            return AppearanceRecomputeResult(
                valid=False, score=None, reason="target_block_not_found"
            )
        return AppearanceRecomputeResult(
            valid=True,
            score=appearance_magnitude(matched.block),
            reason="matched",
            detection_score=appearance_detection_score(matched.block),
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed on any recompute error
        return AppearanceRecomputeResult(
            valid=False, score=None, reason=f"recompute_exception:{type(exc).__name__}"
        )


def run_atomic_probe(
    problem: Problem,
    schedule: Schedule,
    atom: DecisionAtom,
    *,
    appearance_type: str,
    appearance_before: float,
    target_operations: tuple[str, ...],
    executor: AtomicCounterfactualExecutor,
    detection_before: float | None = None,
) -> AtomicProbeResult:
    """Run the atomic probe for one atom through the fixed-decision replay."""
    before = evaluate_objective(problem, schedule, baseline=schedule)
    makespan_before = int(before.values[0])

    result = executor.execute_atom(problem, schedule, atom)
    feasible = result.feasible
    classification = result.report.classification
    makespan_after: int | None = None
    appearance_after = 0.0
    ce = 0.0
    label_valid = False
    detection_after: float | None = None

    if feasible and result.schedule is not None:
        makespan_after = int(result.schedule.makespan)
        recompute = recompute_appearance_atomic(
            problem, result.schedule, appearance_type, target_operations
        )
        if recompute.valid and recompute.score is not None:
            appearance_after = recompute.score
            ce = causal_effect(appearance_before, appearance_after)
            label_valid = True
            detection_after = recompute.detection_score
        # label_valid stays False when the matcher could not confirm the block;
        # ce stays 0.0 (no fabricated high-CE label, §11).

    return AtomicProbeResult(
        atom_id=atom.atom_id,
        atom_type=atom.atom_type,
        probe_operator_id=atom.probe_operator_id,
        probe_parameters=dict(atom.probe_parameters),
        appearance_before=appearance_before,
        appearance_after=appearance_after,
        ce=ce,
        label_valid=label_valid,
        makespan_before=makespan_before,
        makespan_after=makespan_after,
        makespan_gain=makespan_gain(makespan_before, makespan_after),
        feasible=feasible,
        classification=classification,
        solver_status="ATOMIC_REPLAY" if feasible else "INFEASIBLE",
        detection_before=detection_before,
        detection_after=detection_after,
    )


def run_joint_probe(
    problem: Problem,
    schedule: Schedule,
    atoms: tuple[DecisionAtom, ...],
    *,
    appearance_type: str,
    appearance_before: float,
    target_operations: tuple[str, ...],
    executor: AtomicCounterfactualExecutor,
) -> JointProbeResult:
    """Joint atomic counterfactual ``do(B)``: union of exact edits, one replay."""
    before = evaluate_objective(problem, schedule, baseline=schedule)
    makespan_before = int(before.values[0])

    result = executor.execute_atoms(problem, schedule, atoms)
    atom_ids = tuple(a.atom_id for a in atoms)

    if not result.feasible:
        return JointProbeResult(
            atom_ids=atom_ids,
            appearance_before=appearance_before, appearance_after=appearance_before,
            ce=0.0, label_valid=False,
            makespan_before=makespan_before, makespan_after=None,
            makespan_gain=0.0, feasible=False,
            classification=result.report.classification,
            solver_status="INFEASIBLE",
            conflict=result.report.conflict_type or result.report.reason,
        )

    makespan_after = int(result.schedule.makespan)
    recompute = recompute_appearance_atomic(
        problem, result.schedule, appearance_type, target_operations
    )
    appearance_after = 0.0
    ce = 0.0
    label_valid = False
    if recompute.valid and recompute.score is not None:
        appearance_after = recompute.score
        ce = causal_effect(appearance_before, appearance_after)
        label_valid = True

    return JointProbeResult(
        atom_ids=atom_ids,
        appearance_before=appearance_before,
        appearance_after=appearance_after,
        ce=ce,
        label_valid=label_valid,
        makespan_before=makespan_before,
        makespan_after=makespan_after,
        makespan_gain=makespan_gain(makespan_before, makespan_after),
        feasible=True,
        classification=result.report.classification,
        solver_status="ATOMIC_REPLAY",
        conflict=result.report.conflict_type,
    )


# Backward-compat: callers that built a probe state for the old executor.  The
# atomic executor compiles directly from the atom, so this is no longer needed
# for CE generation but is kept import-stable for M3 paths that still reference it.
def _probe_state(*args, **kwargs):  # pragma: no cover - retained for compat
    from ..operator_registry_v1 import REGISTRY_VERSION, operator_masks_from_candidates
    from ..training_v1 import OperatorPolicyState, candidate_set_hash

    problem, schedule, candidates, root_cause_node_ids = args[:4]
    masks = operator_masks_from_candidates(candidates)
    all_true = {op: True for op in masks.get("final", {})}
    return OperatorPolicyState(
        instance_id=problem.id,
        base_instance_id=str(problem.metadata.get("base_instance_id", problem.id)),
        family=problem.kind,
        environment=problem.environment,
        state_id=schedule_hash(schedule),
        schedule_hash=schedule_hash(schedule),
        graph_hash="",
        surface_block_ids=(),
        root_cause_id=f"probe:{root_cause_node_ids}",
        root_cause_node_ids=root_cause_node_ids,
        causal_path_edge_ids=(),
        semantic_operator_mask=all_true,
        local_operator_mask=all_true,
        cause_operator_mask=all_true,
        final_operator_mask=all_true,
        parameter_candidates=candidates,
        registry_version=REGISTRY_VERSION,
        candidate_set_hash=candidate_set_hash(candidates),
        current_objective=evaluate_objective(problem, schedule, baseline=schedule).values,
        budget_features={},
    )
