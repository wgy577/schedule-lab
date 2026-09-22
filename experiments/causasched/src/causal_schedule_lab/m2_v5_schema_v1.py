"""V5 M2 structural / output schema (Phase 1 -- Schema Compatibility).

These dataclasses are the stable machine-readable contract for
:math:`M2(S,A)\\rightarrow \\mathcal R_A=\\{R_1,\\ldots,R_K\\}` where each

.. math:: R_i=(V_i, E_i, D_i, E_i^{edit}, E_i^{dep})

* ``V_i`` -- intervention-relevant operation **and** machine nodes
* ``E_i`` -- true causal/structural relations among those nodes
* ``D_i`` -- root decision sites (routing / sequence)
* ``E_i^{edit}`` -- legal editable opportunities (hard-feasibility enumerated)
* ``E_i^{dep}`` -- enabling/dependency relations between editable edges

They deliberately live **torch-free at the record level** so the proposal schema
is testable and serializable independently of any model build.  The tensor-carrying
model output (:class:`M2V5Output`) is defined in
:mod:`causal_schedule_lab.sg_sct_model_v5`.

Design notes (spec §5, §16, §17, §22):
* A "root" is a **decision site** ``d^r=(o\\to m)`` (routing) or ``d^s=(o_i\\prec_m o_j)``
  (sequence) -- never a bare operation id.
* Legal edits are hard-feasibility enumerations (never hallucinated-by-network).
* ``per_block_root_logits[B,N]`` from V4 remains a read-compatible output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

# Root / decision kinds.
ROUTING_DECISION = "routing"
SEQUENCE_DECISION = "sequence"

# Legal edit kinds (spec §18-19).
EDIT_ROUTE = "ROUTE"
EDIT_SEQ_SWAP = "SEQ_SWAP"
EDIT_SEQ_INSERT = "SEQ_INSERT"
EDIT_TIMING_SHIFT = "TIMING_SHIFT"
LEGAL_EDIT_TYPES = (EDIT_ROUTE, EDIT_SEQ_SWAP, EDIT_SEQ_INSERT, EDIT_TIMING_SHIFT)

# Edit dependency kinds (spec §22, structural only this round).
DEP_ENABLES = "ENABLES"
DEP_RELEASES_MACHINE = "releases_machine"
DEP_CREATES_RECEIVER_WINDOW = "creates_receiver_window"
DEP_COMMON_ROOT = "shares_root_decision"

# Root tracing states (spec §9).
TRACE_EXPLAINED = "EXPLAINED_PROPAGATION"
TRACE_ROOT = "ROOT_DECISION_CANDIDATE"
TRACE_UNRESOLVED = "UNRESOLVED_ROOT_CANDIDATE"

# NOTE: the former ``APPEARANCE_ALL = ("A1","A2","A3","A4")`` constant was deleted
# in the pre-D6 appearance cleanup.  It was dead (zero consumers) and wrong
# (missing A6).  The single source of truth for the actionable set is
# ``causal_schedule_lab.appearance_taxonomy.ACTIVE_APPEARANCE_IDS``.


# ---------------------------------------------------------------------------
# Decision Site
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionSite:
    """A root decision at a specific site (routing or sequence)."""

    site_id: str
    operation_id: str
    decision_type: str  # ROUTING_DECISION | SEQUENCE_DECISION
    # routing: actual machine choice
    source_machine: str | None = None
    target_machine: str | None = None
    mode_id: str | None = None
    # sequence: (o_i prec_m o_j)
    resource_id: str | None = None
    predecessor_id: str | None = None
    successor_id: str | None = None
    # timing shift
    target_start: float | None = None
    # decision context (feature_key, value) pairs -- decision-time-visible info
    context: tuple[tuple[str, float], ...] = ()
    # residual bookkeeping (filled by Phase 5; defaults neutral)
    residual: float = 0.0
    residual_mean: float = 0.0
    residual_std: float = 1.0
    z_deviation: float = 0.0
    # appearance relevance / edit support (filled by Phase 4/6)
    appearance_relevance: float = 0.0
    edit_support: int = 0
    trace_state: str | None = None  # TRACE_* from spec §9

    def validate(self) -> None:
        if self.decision_type not in (ROUTING_DECISION, SEQUENCE_DECISION):
            raise ValueError(f"unknown decision_type: {self.decision_type!r}")
        if self.decision_type == ROUTING_DECISION:
            if self.target_machine is None:
                raise ValueError("routing DecisionSite requires target_machine")
        else:
            if self.predecessor_id is None and self.successor_id is None:
                raise ValueError("sequence DecisionSite requires predecessor_id or successor_id")
            if self.resource_id is None:
                raise ValueError("sequence DecisionSite requires resource_id")


# ---------------------------------------------------------------------------
# Legal Edit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalEdit:
    """A hard-feasibility-legal editable opportunity (spec §17-19)."""

    edit_id: str
    edit_type: str  # LEGAL_EDIT_TYPES
    operation_id: str  # subject operation (the decision owner)
    # routing
    source_machine: str | None = None
    target_machine: str | None = None
    target_mode_id: str | None = None
    # sequencing
    resource_id: str | None = None
    left_id: str | None = None  # adjacent swap: (left wrt right on resource)
    right_id: str | None = None
    insert_position: int | None = None
    predecessor_id: str | None = None
    successor_id: str | None = None
    # timing shift
    target_start: float | None = None
    # deterministic features (spec §18), decision-time-visible
    features: tuple[tuple[str, float], ...] = ()
    # relevance score filled by Phase 4 (not a final probability)
    relevance: float = 0.0

    def validate(self) -> None:
        if self.edit_type not in LEGAL_EDIT_TYPES:
            raise ValueError(f"unknown edit_type: {self.edit_type!r}")
        if self.edit_type == EDIT_ROUTE:
            if self.source_machine is None or self.target_machine is None:
                raise ValueError("ROUTE edit requires source_machine and target_machine")
            if self.source_machine == self.target_machine:
                raise ValueError("ROUTE edit source and target must differ")
        elif self.edit_type == EDIT_SEQ_SWAP:
            if self.left_id is None or self.right_id is None or self.resource_id is None:
                raise ValueError("SEQ_SWAP edit requires left_id, right_id, resource_id")
        elif self.edit_type == EDIT_SEQ_INSERT:
            if self.operation_id is None:
                raise ValueError("SEQ_INSERT edit requires operation_id")
            if self.insert_position is None or self.resource_id is None:
                raise ValueError("SEQ_INSERT edit requires insert_position and resource_id")
        else:  # TIMING_SHIFT
            if self.resource_id is None or self.target_start is None:
                raise ValueError("TIMING_SHIFT edit requires resource_id and target_start")


# ---------------------------------------------------------------------------
# Edit Dependency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditDependency:
    """Structural enabling/dependency between two legal edits (spec §22)."""

    dependency_id: str
    dependency_type: str  # DEP_ENABLES
    editor_id: str  # edit that (potentially) enables
    dependent_id: str  # edit that becomes more relevant
    kind: str = DEP_ENABLES
    description: str = ""

    def validate(self) -> None:
        if self.dependency_type == DEP_ENABLES and self.kind == DEP_ENABLES:
            pass  # default structural ENABLES
        if not all((self.editor_id, self.dependent_id)):
            raise ValueError("EditDependency requires editor_id and dependent_id")


# ---------------------------------------------------------------------------
# Causal Intervention Proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalInterventionProposal:
    """A coherent root-intervention structure (V_i, E_i, D_i, E_edit, E_dep)."""

    proposal_id: str
    appearance_id: str
    # V_i: operation and machine node ids (both can carry relevance, spec §15/§45-15)
    nodes: tuple[str, ...] = ()
    # E_i: structural/causal relations as (source, target) pairs
    edges: tuple[tuple[str, str], ...] = ()
    # D_i: root decision sites (subset of decision_sites flagged as root)
    decision_sites: tuple[DecisionSite, ...] = ()
    # D_i root = the flagged roots (single roots via trace state = TRACE_ROOT)
    root_decisions: tuple[DecisionSite, ...] = ()
    # E_i^edit: legal editable opportunities
    edits: tuple[LegalEdit, ...] = ()
    # E_i^dep: enabling/dependency between edits
    dependencies: tuple[EditDependency, ...] = ()
    # causal path from solver root to the appearance (node ids), spec §23-24
    root_path: tuple[str, ...] = ()
    proposal_score: float = 0.0
    confidence: float = 0.0
    # A proposal is executable only when every dependency endpoint is carried by
    # the same action graph.  ``action_order`` stores the topological execution
    # order (edit ids); it is empty for a primitive proposal.
    action_order: tuple[str, ...] = ()
    # Deterministic ITR provenance.  Explanation is diagnostic and never a
    # learned root label or an execution constraint.
    transition_explanation: tuple[str, ...] = ()
    transition_depth: int = 1
    transition_complete: bool = True
    causal_chain: tuple[str, ...] = ()
    root_decision_id: str = ""
    operator_type: str = ""
    # for diversity bookkeeping (spec §21): overlap uses V_i
    source_block_id: str = ""

    def validate(self) -> None:
        if not self.appearance_id:
            raise ValueError("proposal requires appearance_id")
        seen = set(self.nodes)
        if len(seen) != len(self.nodes):
            raise ValueError("proposal nodes must be unique")
        for d in self.decision_sites:
            d.validate()
        for e in self.edits:
            e.validate()
        for dep in self.dependencies:
            dep.validate()
        edit_ids = {edit.edit_id for edit in self.edits}
        for dep in self.dependencies:
            if dep.editor_id not in edit_ids or dep.dependent_id not in edit_ids:
                raise ValueError(
                    "proposal dependency endpoints must both belong to proposal.edits"
                )
        if self.action_order:
            if len(set(self.action_order)) != len(self.action_order):
                raise ValueError("proposal action_order must not contain duplicates")
            if set(self.action_order) != edit_ids:
                raise ValueError("proposal action_order must cover exactly proposal.edits")
        if self.transition_depth < 1:
            raise ValueError("proposal transition_depth must be >= 1")
        if not self.transition_complete:
            raise ValueError("executable proposal must carry a complete transition chain")

    def node_overlap(self, other: CausalInterventionProposal) -> float:
        """Jaccard overlap on V_i (spec §21)."""
        a, b = set(self.nodes), set(other.nodes)
        denom = len(a | b)
        if denom == 0:
            return 0.0
        return len(a & b) / denom


# ---------------------------------------------------------------------------
# Id helpers (id round-trip is part of the schema contract)
# ---------------------------------------------------------------------------


def route_site_id(operation_id: str, machine_id: str) -> str:
    return f"route:{operation_id}->{machine_id}"


def sequence_site_id(
    operation_a: str, operation_b: str, resource_id: str, sep: str = "<"
) -> str:
    return f"seq:{operation_a}{sep}{operation_b}|{resource_id}"


def route_edit_id(operation_id: str, src_machine: str, dst_machine: str) -> str:
    return f"{EDIT_ROUTE}:{operation_id}:{src_machine}->{dst_machine}"


def seq_swap_edit_id(left: str, right: str, resource_id: str) -> str:
    return f"{EDIT_SEQ_SWAP}:{left}<->{right}|{resource_id}"


def seq_insert_edit_id(operation_id: str, resource_id: str, position: int) -> str:
    return f"{EDIT_SEQ_INSERT}:{operation_id}@{resource_id}:pos{position}"


def timing_shift_edit_id(operation_id: str, resource_id: str, target_start: float) -> str:
    return f"{EDIT_TIMING_SHIFT}:{operation_id}@{resource_id}:start{target_start:g}"


def proposal_id(appearance_id: str, rank: int) -> str:
    return f"proposal:{appearance_id}:{rank}"


def parse_proposal_id(pid: str) -> tuple[str, int] | None:
    """Round-trip of :func:`proposal_id` -> (appearance_id, rank)."""
    if not pid.startswith("proposal:"):
        return None
    rest = pid[len("proposal:"):]
    idx = rest.rfind(":")
    if idx < 0:
        return None
    appearance_id, rank_s = rest[:idx], rest[idx + 1:]
    try:
        return appearance_id, int(rank_s)
    except ValueError:
        return None


def _sequence_features(
    source_machine: str, target_machine: str,
    source_load: float, target_load: float,
    source_rel_load: float, target_rel_load: float,
    flexibility: float, receiver_context: float,
) -> tuple[tuple[str, float], ...]:
    """The §18 routing feature bundle, decision-time-visible."""
    return (
        ("source_machine", float(source_machine)),
        ("target_machine", float(target_machine)),
        ("source_load", float(source_load)),
        ("target_load", float(target_load)),
        ("source_relative_load", float(source_rel_load)),
        ("target_relative_load", float(target_rel_load)),
        ("flexibility", float(flexibility)),
        ("receiver_context", float(receiver_context)),
    )
