"""Patch §6 -- Counterfactual evaluator :math:`S'=T(S,P)`.

The evaluator materialises the post-intervention state by *executing* a
proposal's intervention graph (its legal editing steps) on the incumbent with
the CP-SAT solver -- the deterministic transition :math:`T`.  It then computes
the outcome deltas (:math:`\\Delta C=C'_{max}-C_{max}`, gap, load imbalance)
and classifies the intervention:

* **Direct success** -- :math:`\\Delta C<0`; keep.
* **Delayed success** -- :math:`\\Delta C=0`; use trajectory memory
  :math:`FIV=P(\\text{future success}|S,P)\\cdot E(\\text{final gain}|S,P)` and keep iff FIV
  exceeds its threshold and observed risk is bounded.
* **Failure** -- otherwise reject.

The outcome (:math:`S'`) is attached to the Phase-B :class:`ExperienceStore` as
the third element of the :math:`(S,P,S')` triple, so a kept or rejected
intervention both leave evidence for future M3 selection (Patch §8-Stage2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import pstdev
from typing import Any, Mapping, Sequence

from ..ir import Problem, Schedule
from ..core_validation import validate_schedule
from ..validation import schedule_hash
from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    EDIT_TIMING_SHIFT,
    LegalEdit,
)
from ..memory import (
    DELAYED_SUCCESS,
    DIRECT_SUCCESS,
    FAILURE,
    PENDING,
    Outcome,
    ProposalRecord,
    StateFeatures,
    encode_state,
    experience_store,
    memory_prior_composite,
)
from causal_schedule_lab.operator_execution_v1 import (
    _job_successor_closure,
    _target_resource_neighborhood,
)
from ..solvers.cp_sat import solve_cp_sat
from ..symptom_pruning import diagnose_and_prune
from ..intervention.intervention_closure_v1 import (
    ClosureState,
    FrozenLocalCounterfactualConfig,
    InterventionClosure,
    build_intervention_closure,
    load_frozen_local_counterfactual_config,
)

FROZEN_LOCAL = "frozen_local"
FREE_GLOBAL = "free_global"
LOCAL_COUNTERFACTUAL_INFEASIBLE = "LOCAL_COUNTERFACTUAL_INFEASIBLE"

__all__ = [
    "CounterfactualResult",
    "CounterfactualEvaluator",
    "FrozenEvaluationResult",
    "GlobalEvaluationResult",
    "ProposalEvaluationComparison",
    "FROZEN_LOCAL",
    "FREE_GLOBAL",
    "LOCAL_COUNTERFACTUAL_INFEASIBLE",
    "plan_from_legal_edits",
    "proposal_from_edits",
    "proposal_from_causal",
    "mean_gap",
    "load_imbalance",
    "classify_intervention",
    "intervention_risk",
    "verify_executed_edits",
]


@dataclass(frozen=True)
class FrozenEvaluationResult:
    before_cmax: float
    after_cmax: float | None
    delta_cmax_local: float
    closure_operations: tuple[str, ...]
    changed_operations: tuple[str, ...]
    outside_closure_changes: tuple[str, ...]
    outside_closure_change_ratio: float
    feasible: bool
    solver_status: str
    freeze_level: str
    attribution_clean: bool
    training_eligible: bool


@dataclass(frozen=True)
class GlobalEvaluationResult:
    before_cmax: float
    after_cmax: float | None
    delta_cmax_global: float
    changed_operations: tuple[str, ...]
    hidden_change_ratio: float
    solver_status: str
    training_eligible: bool = False


@dataclass(frozen=True)
class ProposalEvaluationComparison:
    proposal_id: str
    comparison_id: str
    local_delta_cmax: float
    global_delta_cmax: float
    global_extra_gain: float
    local_changed_operations: int
    global_changed_operations: int
    local_attribution_clean: bool
    global_reoptimization_dependency: float
    local: FrozenEvaluationResult
    global_result: GlobalEvaluationResult
    local_counterfactual: CounterfactualResult | None = None
    global_counterfactual: CounterfactualResult | None = None


@dataclass(frozen=True)
class CounterfactualResult:
    """The outcome classification + deltas of one executed :math:`T(S,P)`."""

    key: str
    classification: str  # DIRECT_SUCCESS | DELAYED_SUCCESS | FAILURE | PENDING
    keep: bool
    delta_cmax: float
    delta_gap: float
    delta_load_imbalance: float
    delta_processing_excess: float = 0.0
    collateral_damage: float = 0.0
    ready_tightness: float = 0.0
    critical_block_change: float = 0.0
    critical_block_worsening: float = 0.0
    feasibility_degradation: float = 0.0
    new_anomaly_rate: float = 0.0
    risk: float = 0.0
    structural_changes: tuple[str, ...] = ()
    ev: float = 0.0          # FIV / P(success)*Gain, only used for delayed success
    memory_nv: int = 0
    solver_status: str = ""
    new_schedule: Schedule | None = None
    counterfactual_mode: str = FROZEN_LOCAL
    training_eligible: bool = False
    comparison_id: str = ""
    frozen_evaluation: FrozenEvaluationResult | None = None
    global_evaluation: GlobalEvaluationResult | None = None


# -- structural quantities reused for deltas ---------------------------------

def mean_gap(features: StateFeatures) -> float:
    """Compatibility alias for the frozen global gap ``G``."""
    return float(features.gap)


def load_imbalance(features: StateFeatures) -> float:
    """Population std of machine busy fractions (tightness spread)."""
    cmax = max(features.cmax, 1.0)
    rows = list(features.machine_load.values())
    fracs = [r[0] / cmax for r in rows]
    return pstdev(fracs) if len(fracs) > 1 else 0.0


def processing_excess(problem: Problem, schedule: Schedule) -> float:
    """Selected processing time above each operation's fastest legal mode."""
    mode_map = problem.mode_map()
    op_map = problem.operation_map()
    total = 0.0
    for oid, assignment in schedule.assignment_map().items():
        selected = float(mode_map[assignment.mode_id][1].duration)
        fastest = min(float(mode.duration) for mode in op_map[oid].modes)
        total += selected - fastest
    return total


def ready_tightness(problem: Problem, schedule: Schedule) -> float:
    """Mean delay from job-precedence readiness to actual start."""
    assignments = schedule.assignment_map()
    delays: list[float] = []
    for op in problem.operations:
        assignment = assignments.get(op.id)
        if assignment is None:
            continue
        ready = max(
            (float(assignments[p].end) for p in op.predecessors if p in assignments),
            default=0.0,
        )
        delays.append(max(0.0, float(assignment.start) - ready))
    return sum(delays) / len(delays) if delays else 0.0


def collateral_damage(
    problem: Problem,
    before: Schedule,
    after: Schedule,
    edited_operations: set[str],
) -> float:
    """Fraction of unedited assignments moved or re-routed by the repair."""
    bmap, amap = before.assignment_map(), after.assignment_map()
    untouched = sorted(set(bmap) - edited_operations)
    if not untouched:
        return 0.0
    changed = sum(
        bmap[oid].mode_id != amap[oid].mode_id
        or abs(float(bmap[oid].start) - float(amap[oid].start)) > 1e-9
        for oid in untouched if oid in amap
    )
    return changed / len(untouched)


def _assignment_change_ids(before: Schedule, after: Schedule) -> tuple[str, ...]:
    before_map, after_map = before.assignment_map(), after.assignment_map()
    changed = []
    for operation_id in sorted(set(before_map) | set(after_map)):
        left, right = before_map.get(operation_id), after_map.get(operation_id)
        if left is None or right is None or (
            left.mode_id != right.mode_id
            or abs(float(left.start) - float(right.start)) > 1e-9
            or abs(float(left.end) - float(right.end)) > 1e-9
            or left.route_id != right.route_id
        ):
            changed.append(operation_id)
    return tuple(changed)


def _outside_resource_orderings(
    problem: Problem, incumbent: Schedule, outside: set[str]
) -> tuple[tuple[str, str], ...]:
    """Preserve incumbent relative order among closure-external operations."""
    mode_map = problem.mode_map()
    by_resource: dict[str, list[tuple[float, str]]] = {}
    for assignment in incumbent.assignments:
        if assignment.operation_id not in outside:
            continue
        for resource in mode_map[assignment.mode_id][1].resources:
            by_resource.setdefault(resource, []).append(
                (float(assignment.start), assignment.operation_id)
            )
    pairs = {
        (left, right)
        for rows in by_resource.values()
        for left, right in zip(
            [operation_id for _, operation_id in sorted(rows)],
            [operation_id for _, operation_id in sorted(rows)][1:],
        )
    }
    return tuple(sorted(pairs))


def intervention_risk(
    *,
    collateral: float,
    critical_block_worsening: float,
    feasibility_degradation: float,
    new_anomaly_rate: float,
) -> float:
    """Bounded mean of the four frozen side-effect risk components."""
    components = (
        collateral,
        critical_block_worsening,
        feasibility_degradation,
        new_anomaly_rate,
    )
    clipped = [max(0.0, min(1.0, float(value))) for value in components]
    return sum(clipped) / len(clipped)


def classify_intervention(
    delta_cmax: float,
    fiv: float,
    risk: float,
    *,
    theta: float,
    risk_threshold: float,
) -> tuple[str, bool]:
    """Cmax-first acceptance rule; Gap is intentionally absent."""
    if delta_cmax < -1e-9:
        return DIRECT_SUCCESS, True
    if abs(delta_cmax) <= 1e-9 and fiv > theta and risk <= risk_threshold:
        return DELAYED_SUCCESS, True
    return FAILURE, False


def _anomaly_signatures(problem: Problem, schedule: Schedule) -> set[tuple]:
    """Deterministic appearance signatures used only to quantify new risk."""
    snapshot = diagnose_and_prune(problem, schedule)
    return {
        (
            tuple(block.block.appearance_rules),
            tuple(block.block.operations),
            tuple(block.block.machines),
        )
        for block in snapshot.blocks
    }


# -- proposal <-> edit helpers -----------------------------------------------

def proposal_from_edits(
    edits: Sequence[LegalEdit],
    *,
    appearance_type: str = "",
    root_nodes: Sequence[str] = (),
    proposal_id: str = "",
    dependency_edges: Sequence[tuple[str, str]] = (),
    affected_region: Sequence[str] = (),
    causal_chain: Sequence[str] = (),
    root_decision_id: str = "",
    operator_type: str = "",
    causal_search_trace: Sequence[str] = (),
) -> ProposalRecord:
    """Build the memory ``P`` (Patch §5) from executable edits.

    ``intervention_actions`` := edit ids; ``dependency_edges`` is left empty
    (the deterministic dependency head Phase-A consumes is orthogonal to this
    record's stored shape, which is the patch's edge tuple form).
    """
    actions = tuple(sorted({e.edit_id for e in edits}))
    region = tuple(affected_region) or tuple(dict.fromkeys(
        item
        for edit in edits
        for item in (
            edit.operation_id,
            *(f"machine:{machine}" for machine in
              (edit.source_machine, edit.target_machine) if machine),
        )
    ))
    return ProposalRecord(
        appearance_type=appearance_type,
        root_nodes=tuple(root_nodes),
        intervention_actions=actions,
        dependency_edges=tuple(dependency_edges),
        affected_region=region,
        proposal_id=proposal_id,
        causal_chain=tuple(causal_chain),
        root_decision_id=root_decision_id,
        operator_type=operator_type,
        causal_search_trace=tuple(causal_search_trace),
    )


def proposal_from_causal(
    proposal, *, causal_search_trace: Sequence[str] = (), causal_chain=None
) -> ProposalRecord:
    """Lossless memory projection of an executable M2 macro proposal."""
    chain_nodes = tuple(getattr(causal_chain, "nodes", proposal.causal_chain))
    root_candidate = str(getattr(causal_chain, "root_candidate_id", ""))
    root_position = (
        chain_nodes.index(root_candidate) / max(len(chain_nodes) - 1, 1)
        if root_candidate in chain_nodes else 0.0
    )
    explanation_gain = max(
        0.0,
        float(getattr(causal_chain, "causal_score", 0.0))
        - float(getattr(causal_chain, "m2_root_score", 0.0)),
    )
    return ProposalRecord(
        appearance_type=str(proposal.appearance_id),
        root_nodes=tuple(site.operation_id for site in proposal.root_decisions),
        intervention_actions=tuple(proposal.action_order or tuple(e.edit_id for e in proposal.edits)),
        dependency_edges=tuple(
            (dep.editor_id, dep.dependent_id) for dep in proposal.dependencies
        ),
        affected_region=tuple(proposal.nodes),
        proposal_id=proposal.proposal_id,
        confidence=float(proposal.confidence),
        causal_chain=tuple(proposal.causal_chain),
        root_decision_id=str(proposal.root_decision_id),
        operator_type=str(proposal.operator_type),
        causal_search_trace=tuple(causal_search_trace),
        causal_relations=tuple(getattr(causal_chain, "relations", ())),
        causal_chain_depth=int(getattr(causal_chain, "depth", 0)),
        causal_explanation_gain=float(explanation_gain),
        causal_root_position=float(root_position),
        estimated_action_complexity=float(
            len(proposal.edits) + len(proposal.dependencies)
            + 0.5 * len(proposal.nodes)
        ),
    )


def _experience_state(
    problem: Problem,
    schedule: Schedule,
    proposal: ProposalRecord,
    edits: Sequence[LegalEdit],
) -> StateFeatures:
    """Encode global, Appearance and root-local graph features for memory."""
    roots = set(proposal.root_nodes) | {edit.operation_id for edit in edits}
    op_map = problem.operation_map()
    assignments = schedule.assignment_map()
    mode_map = problem.mode_map()
    local_ops = set(roots)
    precedence: set[tuple[str, str]] = set()
    for oid in tuple(roots):
        op = op_map.get(oid)
        if op is None:
            continue
        for pred in op.predecessors:
            local_ops.add(pred)
            precedence.add((pred, oid))
    machines: set[str] = set()
    modes: set[str] = set()
    by_machine: dict[str, list[tuple[float, str]]] = {}
    for oid, assignment in assignments.items():
        mode = mode_map[assignment.mode_id][1]
        machine = mode.resources[0]
        by_machine.setdefault(machine, []).append((float(assignment.start), oid))
        if oid in local_ops:
            machines.add(machine)
            modes.add(assignment.mode_id)
    for edit in edits:
        if edit.source_machine:
            machines.add(edit.source_machine)
        if edit.target_machine:
            machines.add(edit.target_machine)
        if edit.target_mode_id:
            modes.add(edit.target_mode_id)
    resource_edges: set[tuple[str, str]] = set()
    for machine in machines:
        seq = [oid for _, oid in sorted(by_machine.get(machine, ())) ]
        for left, right in zip(seq, seq[1:]):
            if left in local_ops or right in local_ops:
                local_ops.update((left, right))
                resource_edges.add((left, right))
    base = encode_state(
        problem,
        schedule,
        appearance_type=proposal.appearance_type,
        appearance_score=proposal.confidence,
        local_features=(float(len(roots)), float(len(edits)), float(len(proposal.dependency_edges))),
        local_operation_nodes=tuple(sorted(local_ops)),
        local_machine_nodes=tuple(sorted(machines)),
        local_mode_nodes=tuple(sorted(modes)),
        local_precedence_edges=tuple(sorted(precedence)),
        local_resource_sequence_edges=tuple(sorted(resource_edges)),
    )
    graph_embedding = (
        base.cmax,
        base.gap,
        base.load_variance,
        float(len(local_ops)),
        float(len(machines)),
        float(len(precedence)),
        float(len(resource_edges)),
    )
    return replace(base, graph_embedding=graph_embedding)


def verify_executed_edits(
    problem: Problem, schedule: Schedule, edits: Sequence[LegalEdit]
) -> dict[str, bool]:
    """Verify that each declared LegalEdit is visible in the returned schedule."""
    assignments = schedule.assignment_map()
    mode_map = problem.mode_map()
    by_resource: dict[str, list[str]] = {}
    for assignment in sorted(
        schedule.assignments, key=lambda row: (row.start, row.end, row.operation_id)
    ):
        selected = mode_map.get(assignment.mode_id)
        if selected is None:
            continue
        for resource in selected[1].resources:
            by_resource.setdefault(resource, []).append(assignment.operation_id)
    checks: dict[str, bool] = {}
    for edit in edits:
        assignment = assignments.get(edit.operation_id)
        passed = assignment is not None
        if passed and edit.edit_type == EDIT_ROUTE:
            passed = assignment.mode_id == edit.target_mode_id
        elif passed and edit.edit_type == EDIT_TIMING_SHIFT:
            passed = abs(float(assignment.start) - float(edit.target_start)) <= 1e-9
        elif passed and edit.edit_type == EDIT_SEQ_SWAP:
            sequence = by_resource.get(str(edit.resource_id), [])
            passed = (
                edit.left_id in sequence and edit.right_id in sequence
                and sequence.index(str(edit.right_id)) < sequence.index(str(edit.left_id))
            )
        elif passed and edit.edit_type == EDIT_SEQ_INSERT:
            sequence = by_resource.get(str(edit.resource_id), [])
            if edit.operation_id not in sequence:
                passed = False
            elif edit.predecessor_id is not None:
                passed = (
                    edit.predecessor_id in sequence
                    and sequence.index(edit.predecessor_id) < sequence.index(edit.operation_id)
                )
            elif edit.successor_id is not None:
                passed = (
                    edit.successor_id in sequence
                    and sequence.index(edit.operation_id) < sequence.index(edit.successor_id)
                )
            else:
                passed = sequence.index(edit.operation_id) == int(edit.insert_position)
        checks[edit.edit_id] = bool(passed)
    return checks


def _counterfactual_evidence(
    problem: Problem,
    before_schedule: Schedule,
    proposal: ProposalRecord,
    edits: Sequence[LegalEdit],
    *,
    after_schedule: Schedule | None,
    solver_status: str,
    proposal_legal: bool,
    actions_executed: bool,
    delta_cmax_verified: bool,
    validator_passed: bool,
    action_checks: dict[str, bool] | None = None,
    counterfactual_mode: str = FROZEN_LOCAL,
    closure: InterventionClosure | None = None,
    changed_operations: Sequence[str] = (),
    outside_closure_changes: Sequence[str] = (),
    freeze_level: str = "",
    training_eligible: bool = False,
    comparison_id: str = "",
    parent_proposal_id: str = "",
    before_feasible: bool = False,
    after_feasible: bool = False,
) -> dict[str, object]:
    """Portable provenance required by Training Readiness Audit V1."""
    base_instance_id = str(problem.metadata.get("base_instance_id", problem.id))
    return {
        "evidence_schema": "counterfactual_trajectory_evidence_v2",
        "base_instance_id": base_instance_id,
        "schedule_instance_id": str(problem.id),
        "family": str(problem.kind),
        "source_split": str(problem.metadata.get("split", "unassigned")),
        "formal_test_source": bool(problem.metadata.get("formal_test", False)),
        "validator": "causal_schedule_lab.solvers.cp_sat.solve_cp_sat",
        "validator_kind": "real_cp_sat_counterfactual",
        "solver_status": str(solver_status),
        "proposal_legal": bool(proposal_legal),
        "actions_executed": bool(actions_executed),
        "action_checks": action_checks or {},
        "delta_cmax_verified": bool(delta_cmax_verified),
        "validator_passed": bool(validator_passed),
        "before_feasible": bool(before_feasible),
        "after_feasible": bool(after_feasible),
        "counterfactual_mode": counterfactual_mode,
        "closure": asdict(closure) if closure is not None else None,
        "closure_operations": list(closure.operation_ids) if closure else [],
        "changed_operations": list(changed_operations),
        "outside_closure_changes": list(outside_closure_changes),
        "outside_closure_change_ratio": (
            len(outside_closure_changes) / len(changed_operations)
            if changed_operations else 0.0
        ),
        "freeze_level": freeze_level,
        "training_eligible": bool(training_eligible),
        "comparison_id": comparison_id,
        "parent_proposal_id": parent_proposal_id or proposal.proposal_id,
        "requested_action_ids": [edit.edit_id for edit in edits],
        "requested_edits": [asdict(edit) for edit in edits],
        "problem_sha256": __import__("hashlib").sha256(
            problem.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "before_schedule_sha256": schedule_hash(before_schedule),
        "after_schedule_sha256": schedule_hash(after_schedule) if after_schedule else None,
        "problem_snapshot": problem.model_dump(mode="json"),
        "before_schedule_snapshot": before_schedule.model_dump(mode="json"),
        "after_schedule_snapshot": (
            after_schedule.model_dump(mode="json") if after_schedule else None
        ),
        "proposal_id": proposal.proposal_id,
    }


# -- plan construction -------------------------------------------------------

def plan_from_legal_edits(
    problem: Problem,
    schedule: Schedule,
    edits: Sequence[LegalEdit],
    *,
    reassignment_release_radius: int = 2,
) -> tuple[set[str], dict[str, str], tuple[tuple[str, str], ...], set[str]]:
    """Compile legal editing steps into CP-SAT-enforced decisions.

    Returns ``(frozen_operations, forced_mode_ids, enforced_orderings,
    released_operations)`` -- same shape :func:`solve_cp_sat` expects.  Each edit
    maps to a constraint: ROUTE => a forced mode, SEQ_SWAP => reversed adjacency,
    SEQ_INSERT => new predecessor/successor orderings.  Conflicting edits (two
    ROUTE edits on the same operation, or an ordering that would create a
    precedence cycle) raise :class:`ValueError`; the evaluator treats that as a
    fail-closed ``FAILURE`` rather than a silently wrong execution.
    """
    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    forced_modes: dict[str, str] = {}
    orderings: list[tuple[str, str]] = []
    target_resources: set[str] = set()
    seeds: set[str] = set()

    for edit in edits:
        if edit.edit_type == EDIT_ROUTE:
            if edit.operation_id in forced_modes:
                raise ValueError(f"conflicting routing edits for {edit.operation_id}")
            if edit.target_mode_id is None:
                raise ValueError(f"routing edit {edit.edit_id} lacks a target mode")
            forced_modes[edit.operation_id] = edit.target_mode_id
            if edit.target_machine is not None:
                target_resources.add(edit.target_machine)
            seeds.add(edit.operation_id)
        elif edit.edit_type == EDIT_SEQ_SWAP:
            if edit.left_id is None or edit.right_id is None:
                raise ValueError(f"swap edit {edit.edit_id} lacks left/right")
            orderings.append((edit.right_id, edit.left_id))
            seeds.update((edit.left_id, edit.right_id))
        elif edit.edit_type == EDIT_SEQ_INSERT:
            oid = edit.operation_id
            if edit.predecessor_id is not None:
                orderings.append((edit.predecessor_id, oid))
                seeds.add(edit.predecessor_id)
            if edit.successor_id is not None:
                orderings.append((oid, edit.successor_id))
                seeds.add(edit.successor_id)
            seeds.add(oid)
        elif edit.edit_type == EDIT_TIMING_SHIFT:
            if edit.target_start is None:
                raise ValueError(f"timing edit {edit.edit_id} lacks target_start")
            if abs(float(edit.target_start) - round(float(edit.target_start))) > 1e-9:
                raise ValueError("CP-SAT timing validation requires integral target_start")
            seeds.add(edit.operation_id)
        else:
            raise ValueError(f"unsupported edit type: {edit.edit_type!r}")

    for oid, mode_id in sorted(forced_modes.items()):
        op = operation_map.get(oid)
        if op is None:
            raise ValueError(f"routing edit references unknown operation {oid}")
        if mode_id not in {m.id for m in op.modes}:
            raise ValueError(f"routing edit {oid} forced to ineligible mode {mode_id}")
        if assignment_map.get(oid) is None:
            raise ValueError(f"routing edit {oid} has no incumbent assignment")

    # release the moved slot's job suffix + the target machine's neighborhood so
    # the target machine can reshuffle to make room for a moved operation.
    if target_resources:
        for oid in forced_modes:
            a = assignment_map.get(oid)
            if a is not None:
                seeds.update(
                    _target_resource_neighborhood(
                        problem, schedule, target_resources, a.start, a.end,
                        reassignment_release_radius,
                    )
                )

    released = _job_successor_closure(problem, seeds)
    all_operations = {item.id for item in problem.operations}
    frozen = all_operations - released
    return frozen, forced_modes, tuple(orderings), released


# -- evaluator -----------------------------------------------------------------

class CounterfactualEvaluator:
    """Executes :math:`S'=T(S,P)` and records the outcome into memory.

    ``store`` is the Phase-B :class:`ExperienceStore` the outcome is appended
    to (the third element of :math:`(S,P,S')`).  ``theta`` is the delayed-success
    FIV threshold (Patch §6).  All execution is deterministic (single-thread
    CP-SAT, fixed seed) so a given (state, proposal) re-runs to the same outcome.
    """

    def __init__(
        self,
        store: experience_store.ExperienceStore,
        *,
        solver_time: float = 1.0,
        seed: int = 0,
        theta: float = 0.1,
        reassignment_release_radius: int = 2,
        stability_weight: int = 1,
        prior_k: int = 5,
        risk_threshold: float = 0.5,
        closure_config: FrozenLocalCounterfactualConfig | None = None,
    ) -> None:
        self.store = store
        self.solver_time = solver_time
        self.seed = seed
        self.theta = theta
        self.reassignment_release_radius = reassignment_release_radius
        self.stability_weight = stability_weight
        self.prior_k = prior_k
        self.risk_threshold = risk_threshold
        self.closure_config = closure_config or load_frozen_local_counterfactual_config()
        self.closure_config.validate()

    def evaluate(
        self,
        problem: Problem,
        incumbent: Schedule,
        proposal: ProposalRecord,
        edits: Sequence[LegalEdit],
        *,
        store_outcome: bool = True,
        continue_trajectory_key: str | None = None,
        predicted_fiv: float | None = None,
        predicted_memory_nv: int = 0,
        counterfactual_mode: str = FROZEN_LOCAL,
        causal_chain: Any = None,
        root: Any = None,
        transition_trace: Any = None,
        comparison_id: str = "",
        parent_proposal_id: str = "",
    ) -> CounterfactualResult:
        """Execute the proposal's editing steps on ``incumbent``; return outcome."""
        if counterfactual_mode not in {FROZEN_LOCAL, FREE_GLOBAL}:
            raise ValueError(f"unsupported counterfactual mode: {counterfactual_mode!r}")
        if continue_trajectory_key is not None and counterfactual_mode != FROZEN_LOCAL:
            raise ValueError("trajectory continuation requires frozen_local counterfactuals")
        before = _experience_state(problem, incumbent, proposal, edits)
        before_feasible = validate_schedule(problem, incumbent).feasible
        closure = (
            build_intervention_closure(
                ClosureState(problem, incumbent), causal_chain, root or proposal.root_decision_id,
                proposal, transition_trace, edits=edits, config=self.closure_config,
            )
            if counterfactual_mode == FROZEN_LOCAL else None
        )
        if not comparison_id:
            comparison_id = (
                f"dual:{schedule_hash(incumbent)[:12]}:"
                f"{proposal.proposal_id or proposal.root_decision_id or 'proposal'}"
            )
        base_metadata = _counterfactual_evidence(
            problem, incumbent, proposal, edits,
            after_schedule=None, solver_status="PENDING",
            proposal_legal=False, actions_executed=False,
            delta_cmax_verified=False, validator_passed=False,
            counterfactual_mode=counterfactual_mode, closure=closure,
            comparison_id=comparison_id, parent_proposal_id=parent_proposal_id,
            before_feasible=before_feasible, after_feasible=False,
        )
        key = (
            continue_trajectory_key
            if store_outcome and continue_trajectory_key is not None
            else self.store.append(
                before, proposal, outcome=None, metadata=base_metadata
            ) if store_outcome else ""
        )

        try:
            global_frozen, forced_modes, orderings, _ = plan_from_legal_edits(
                problem, incumbent, edits,
                reassignment_release_radius=self.reassignment_release_radius,
            )
        except (ValueError, NotImplementedError) as error:
            if store_outcome and continue_trajectory_key is None:
                self.store.record_outcome(
                    key,
                    Outcome(delta_cmax=0.0, classification=FAILURE,
                            feasibility_degradation=1.0, risk=1.0,
                            structural_changes=()),
                    metadata_update={**base_metadata, "solver_status": f"REJECTED:{type(error).__name__}"},
                )
            return CounterfactualResult(
                key=key, classification=FAILURE, keep=False,
                delta_cmax=0.0, delta_gap=0.0, delta_load_imbalance=0.0,
                feasibility_degradation=1.0, risk=1.0,
                solver_status=f"REJECTED:{type(error).__name__}",
                new_schedule=None, counterfactual_mode=counterfactual_mode,
                training_eligible=False, comparison_id=comparison_id,
            )

        forced_starts = {
            edit.operation_id: int(round(float(edit.target_start)))
            for edit in edits
            if edit.edit_type == EDIT_TIMING_SHIFT and edit.target_start is not None
        }
        freeze_level = "free_global"
        if counterfactual_mode == FROZEN_LOCAL:
            assert closure is not None
            all_operations = {operation.id for operation in problem.operations}
            outside = all_operations - set(closure.operation_ids)
            freeze_level = "level_1_machine_start_sequence"
            result = solve_cp_sat(
                problem, incumbent=incumbent,
                frozen_operation_ids=outside,
                forced_mode_ids=forced_modes,
                enforced_orderings=orderings,
                forced_start_times=forced_starts,
                seed=self.seed, workers=1,
                max_deterministic_time=self.solver_time,
                stability_weight=self.stability_weight,
            )
            if result.schedule is None:
                freeze_level = "level_2_machine_sequence_timing_relaxed"
                result = solve_cp_sat(
                    problem, incumbent=incumbent,
                    frozen_mode_operation_ids=outside,
                    forced_mode_ids=forced_modes,
                    enforced_orderings=tuple(dict.fromkeys(
                        (*orderings, *_outside_resource_orderings(problem, incumbent, outside))
                    )),
                    forced_start_times=forced_starts,
                    seed=self.seed, workers=1,
                    max_deterministic_time=self.solver_time,
                    stability_weight=self.stability_weight,
                )
        else:
            result = solve_cp_sat(
                problem, incumbent=incumbent,
                frozen_operation_ids=global_frozen,
                forced_mode_ids=forced_modes,
                enforced_orderings=orderings,
                forced_start_times=forced_starts,
                seed=self.seed, workers=1,
                max_deterministic_time=self.solver_time,
                stability_weight=self.stability_weight,
            )
        if result.schedule is None:
            status = (
                LOCAL_COUNTERFACTUAL_INFEASIBLE
                if counterfactual_mode == FROZEN_LOCAL else result.status
            )
            if store_outcome and continue_trajectory_key is None:
                self.store.record_outcome(
                    key,
                    Outcome(delta_cmax=0.0, classification=FAILURE,
                            feasibility_degradation=1.0, risk=1.0,
                            structural_changes=()),
                    metadata_update={**base_metadata, "proposal_legal": True,
                                     "solver_status": status,
                                     "freeze_level": freeze_level},
                )
            frozen_detail = None
            if counterfactual_mode == FROZEN_LOCAL:
                assert closure is not None
                frozen_detail = FrozenEvaluationResult(
                    before_cmax=float(before.cmax), after_cmax=None,
                    delta_cmax_local=0.0,
                    closure_operations=closure.operation_ids,
                    changed_operations=(), outside_closure_changes=(),
                    outside_closure_change_ratio=0.0, feasible=False,
                    solver_status=status, freeze_level=freeze_level,
                    attribution_clean=False, training_eligible=False,
                )
            return CounterfactualResult(
                key=key, classification=FAILURE, keep=False,
                delta_cmax=0.0, delta_gap=0.0, delta_load_imbalance=0.0,
                feasibility_degradation=1.0, risk=1.0,
                solver_status=status, new_schedule=None,
                counterfactual_mode=counterfactual_mode,
                training_eligible=False, comparison_id=comparison_id,
                frozen_evaluation=frozen_detail,
            )

        after = _experience_state(problem, result.schedule, proposal, edits)
        delta_c = after.cmax - before.cmax
        # Frozen formula: G=m*Cmax-sum_i p_i, so delta_gap is exactly G'-G.
        delta_gap = after.gap - before.gap
        delta_imb = load_imbalance(after) - load_imbalance(before)
        delta_excess = processing_excess(problem, result.schedule) - processing_excess(problem, incumbent)
        damage = collateral_damage(
            problem,
            incumbent,
            result.schedule,
            {edit.operation_id for edit in edits},
        )
        tightness = ready_tightness(problem, result.schedule)
        critical_change = float(before.critical_machine != after.critical_machine)
        critical_worsening = max(
            0.0,
            (after.critical_path_length - before.critical_path_length)
            / max(before.critical_path_length, 1.0),
        )
        before_anomalies = _anomaly_signatures(problem, incumbent)
        after_anomalies = _anomaly_signatures(problem, result.schedule)
        new_anomaly_rate = len(after_anomalies - before_anomalies) / max(
            len(after_anomalies), 1
        )
        risk = intervention_risk(
            collateral=damage,
            critical_block_worsening=critical_worsening,
            feasibility_degradation=0.0,
            new_anomaly_rate=new_anomaly_rate,
        )
        changes: list[str] = []
        if delta_c < 0:
            changes.append("critical_block_reduced")
        if delta_imb < 0:
            changes.append("rebalanced")
        ev, nv = 0.0, 0
        if abs(delta_c) <= 1e-9:
            if predicted_fiv is None:
                prior = memory_prior_composite(
                    self.store, before, proposal, k=self.prior_k,
                    exclude_key=key or None,
                )
                ev, nv = prior["fiv"], int(prior["nv"])
            else:
                ev = max(0.0, float(predicted_fiv))
                nv = max(0, int(predicted_memory_nv))
        classification, keep = classify_intervention(
            delta_c, ev, risk, theta=self.theta,
            risk_threshold=self.risk_threshold,
        )

        changed_operations = _assignment_change_ids(incumbent, result.schedule)
        explicit_operations = {edit.operation_id for edit in edits}
        if closure is not None:
            outside_changes = tuple(sorted(set(changed_operations) - set(closure.operation_ids)))
        else:
            outside_changes = tuple(sorted(set(changed_operations) - explicit_operations))
        outside_ratio = len(outside_changes) / len(changed_operations) if changed_operations else 0.0

        action_checks = verify_executed_edits(problem, result.schedule, edits)
        after_feasible = validate_schedule(problem, result.schedule).feasible
        actions_executed = bool(action_checks) and all(action_checks.values())
        delta_verified = abs(delta_c - (result.schedule.makespan - incumbent.makespan)) <= 1e-9
        training_eligible = bool(
            counterfactual_mode == FROZEN_LOCAL
            and closure is not None and not closure.truncated
            and before_feasible and after_feasible
            and actions_executed and delta_verified
            and not outside_changes
        )
        if store_outcome:
            evidence = _counterfactual_evidence(
                problem, incumbent, proposal, edits,
                after_schedule=result.schedule, solver_status=result.status,
                proposal_legal=True,
                actions_executed=actions_executed,
                delta_cmax_verified=delta_verified,
                validator_passed=after_feasible,
                action_checks=action_checks,
                counterfactual_mode=counterfactual_mode, closure=closure,
                changed_operations=changed_operations,
                outside_closure_changes=outside_changes,
                freeze_level=freeze_level, training_eligible=training_eligible,
                comparison_id=comparison_id, parent_proposal_id=parent_proposal_id,
                before_feasible=before_feasible, after_feasible=after_feasible,
            )
            outcome = Outcome(
                delta_cmax=float(delta_c), delta_gap=float(delta_gap),
                delta_load_imbalance=float(delta_imb),
                delta_processing_excess=float(delta_excess),
                collateral_damage=float(damage),
                ready_tightness=float(tightness),
                critical_block_change=critical_change,
                critical_block_worsening=float(critical_worsening),
                feasibility_degradation=0.0,
                new_anomaly_rate=float(new_anomaly_rate),
                risk=float(risk),
                structural_changes=tuple(changes),
                classification=classification,
            )
            if continue_trajectory_key is None:
                self.store.record_outcome(
                    key, outcome, after_state=after, metadata_update=evidence
                )
            else:
                self.store.append_trajectory_step(
                    key, proposal, outcome, after_state=after,
                    validation_metadata=evidence,
                )
        frozen_detail = None
        global_detail = None
        if counterfactual_mode == FROZEN_LOCAL:
            assert closure is not None
            frozen_detail = FrozenEvaluationResult(
                before_cmax=float(before.cmax), after_cmax=float(after.cmax),
                delta_cmax_local=float(delta_c),
                closure_operations=closure.operation_ids,
                changed_operations=changed_operations,
                outside_closure_changes=outside_changes,
                outside_closure_change_ratio=float(outside_ratio),
                feasible=after_feasible, solver_status=result.status,
                freeze_level=freeze_level,
                attribution_clean=not outside_changes,
                training_eligible=training_eligible,
            )
        else:
            global_detail = GlobalEvaluationResult(
                before_cmax=float(before.cmax), after_cmax=float(after.cmax),
                delta_cmax_global=float(delta_c),
                changed_operations=changed_operations,
                hidden_change_ratio=float(outside_ratio),
                solver_status=result.status, training_eligible=False,
            )
        return CounterfactualResult(
            key=key, classification=classification, keep=keep,
            delta_cmax=float(delta_c), delta_gap=float(delta_gap),
            delta_load_imbalance=float(delta_imb),
            delta_processing_excess=float(delta_excess),
            collateral_damage=float(damage),
            ready_tightness=float(tightness),
            critical_block_change=critical_change,
            critical_block_worsening=float(critical_worsening),
            feasibility_degradation=0.0,
            new_anomaly_rate=float(new_anomaly_rate), risk=float(risk),
            structural_changes=tuple(changes), ev=ev, memory_nv=nv,
            solver_status=result.status, new_schedule=result.schedule,
            counterfactual_mode=counterfactual_mode,
            training_eligible=training_eligible, comparison_id=comparison_id,
            frozen_evaluation=frozen_detail, global_evaluation=global_detail,
        )

    def evaluate_dual(
        self,
        problem: Problem,
        incumbent: Schedule,
        proposal: ProposalRecord,
        edits: Sequence[LegalEdit],
        *,
        store_outcome: bool = True,
        causal_chain: Any = None,
        root: Any = None,
        transition_trace: Any = None,
    ) -> ProposalEvaluationComparison:
        """Run local attribution and global-potential evaluations without mixing labels."""
        comparison_id = (
            f"dual:{schedule_hash(incumbent)[:12]}:"
            f"{proposal.proposal_id or proposal.root_decision_id or 'proposal'}"
        )
        local = self.evaluate(
            problem, incumbent, proposal, edits,
            store_outcome=store_outcome, counterfactual_mode=FROZEN_LOCAL,
            causal_chain=causal_chain, root=root, transition_trace=transition_trace,
            comparison_id=comparison_id,
        )
        global_result = self.evaluate(
            problem, incumbent, proposal, edits,
            store_outcome=store_outcome, counterfactual_mode=FREE_GLOBAL,
            comparison_id=comparison_id,
        )
        local_detail = local.frozen_evaluation or FrozenEvaluationResult(
            float(incumbent.makespan), None, 0.0, (), (), (), 0.0, False,
            local.solver_status, "", False, False,
        )
        global_detail = global_result.global_evaluation or GlobalEvaluationResult(
            float(incumbent.makespan), None, 0.0, (), 0.0,
            global_result.solver_status, False,
        )
        return ProposalEvaluationComparison(
            proposal_id=proposal.proposal_id,
            comparison_id=comparison_id,
            local_delta_cmax=float(local.delta_cmax),
            global_delta_cmax=float(global_result.delta_cmax),
            global_extra_gain=float(local.delta_cmax - global_result.delta_cmax),
            local_changed_operations=len(local_detail.changed_operations),
            global_changed_operations=len(global_detail.changed_operations),
            local_attribution_clean=local_detail.attribution_clean,
            global_reoptimization_dependency=float(global_detail.hidden_change_ratio),
            local=local_detail,
            global_result=global_detail,
            local_counterfactual=local,
            global_counterfactual=global_result,
        )
