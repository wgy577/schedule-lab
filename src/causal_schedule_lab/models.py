from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ClaimStatus(StrEnum):
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class DecisionType(StrEnum):
    SELECT = "select"
    RESOURCE = "resource"
    SEQUENCE = "sequence"
    RELEASE = "release"


class NodeType(StrEnum):
    OPERATION = "operation"
    JOB = "job"
    RESOURCE = "resource"
    STAGE = "stage"
    DIAGNOSTIC = "diagnostic"
    VEHICLE = "vehicle"
    ROUTE = "route"
    BUFFER = "buffer"


class EdgeType(StrEnum):
    PRECEDENCE = "precedence"
    RESOURCE_SEQUENCE = "resource_sequence"
    RESOURCE_COMPETITION = "resource_competition"
    ELIGIBILITY = "eligibility"
    WAIT_PROPAGATION = "wait_propagation"
    TEMPORAL_CAUSAL = "temporal_causal"
    CRITICAL_PATH = "critical_path"
    PROJECT_BINDING = "project_binding"


class Fidelity(StrEnum):
    CODE = "code"
    STATIC = "static"
    LIGHT = "light"
    FULL = "full"


class ControlAction(StrEnum):
    RETRY = "Retry"
    EXPAND = "Expand"
    NEXT_CIP = "NextCIP"
    BACKTRACK = "Backtrack"
    FULL_ORACLE = "FullOracle"
    STOP_LOCAL = "StopLocal"
    STOP_GLOBAL = "StopGlobal"


class FailureLabel(StrEnum):
    PRECEDENCE_VIOLATION = "PRECEDENCE_VIOLATION"
    RESOURCE_OVERLAP = "RESOURCE_OVERLAP"
    MACHINE_INELIGIBILITY = "MACHINE_INELIGIBILITY"
    OUTSIDE_MACHINE_CHANGE = "OUTSIDE_MACHINE_CHANGE"
    ROUTE_BINDING_CHANGE = "ROUTE_BINDING_CHANGE"
    VEHICLE_DISCONTINUITY = "VEHICLE_DISCONTINUITY"
    COLLISION = "COLLISION"
    CLOSURE_TOO_SMALL = "CLOSURE_TOO_SMALL"
    BOTTLENECK_TRANSFER = "BOTTLENECK_TRANSFER"
    NO_TRUE_IMPROVEMENT = "NO_TRUE_IMPROVEMENT"
    GENERATOR_FAILURE = "GENERATOR_FAILURE"
    NON_DETERMINISTIC_REPLAY = "NON_DETERMINISTIC_REPLAY"
    SEMANTIC_EVIDENCE_MISSING = "SEMANTIC_EVIDENCE_MISSING"


class EvidenceRef(FrozenModel):
    kind: Literal["code", "test", "document", "artifact", "witness"]
    file: str
    symbol: str | None = None
    detail: str | None = None


class SemanticClaim(FrozenModel):
    id: str
    claim: str
    status: ClaimStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def verified_requires_evidence(self) -> "SemanticClaim":
        if self.status == ClaimStatus.VERIFIED and not self.evidence:
            raise ValueError(f"verified claim {self.id} requires evidence")
        return self


class ResourceSemantic(FrozenModel):
    id: str
    capacity: int = Field(ge=1)
    tags: tuple[str, ...] = ()


class BindingSemantic(FrozenModel):
    id: str
    source: str
    target: str
    relation: str


class ObjectiveSemantic(FrozenModel):
    sense: Literal["minimize", "maximize"]
    expression: str
    tie_breakers: tuple[str, ...] = ()


class ProjectSemantics(FrozenModel):
    schema_version: str = "1.0"
    project_id: str
    project_type: str
    problem_families: tuple[str, ...]
    objective: ObjectiveSemantic
    decision_types: tuple[DecisionType, ...]
    resources: tuple[ResourceSemantic, ...] = ()
    bindings: tuple[BindingSemantic, ...] = ()
    hard_constraints: tuple[str, ...]
    allowed_interventions: tuple[str, ...]
    oracle_gates: tuple[str, ...]
    claims: tuple[SemanticClaim, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def verified_claims(self) -> tuple[SemanticClaim, ...]:
        return tuple(claim for claim in self.claims if claim.status == ClaimStatus.VERIFIED)


class GraphNode(FrozenModel):
    id: str
    type: NodeType
    features: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class GraphEdge(FrozenModel):
    source: str
    target: str
    type: EdgeType
    features: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class SchedulingGraph(FrozenModel):
    project_id: str
    problem_id: str
    problem_family: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    objective: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


class DiagnosticPoint(FrozenModel):
    id: str
    type: str
    location: tuple[str, ...]
    resource_id: str | None = None
    window: tuple[float, float] | None = None
    magnitude: float = Field(ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ResponsiblePoint(FrozenModel):
    operation_id: str
    decision_type: DecisionType
    reason: str
    modifiability: float = Field(ge=0.0)


class CausalPath(FrozenModel):
    nodes: tuple[str, ...]
    edge_types: tuple[EdgeType, ...]
    explanation: str

    @model_validator(mode="after")
    def path_shape(self) -> "CausalPath":
        if len(self.edge_types) != max(0, len(self.nodes) - 1):
            raise ValueError("causal path needs exactly one edge type between adjacent nodes")
        return self


class CausalClosure(FrozenModel):
    operation_ids: tuple[str, ...]
    resource_ids: tuple[str, ...] = ()
    level: int = Field(default=1, ge=1, le=3)
    predicted_outside_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str


class CausalInterventionPoint(FrozenModel):
    id: str
    diagnostic: DiagnosticPoint
    responsible: ResponsiblePoint
    causal_path: CausalPath
    closure: CausalClosure
    predicted_improvement: float = 0.0
    predicted_validity: float = Field(default=0.5, ge=0.0, le=1.0)
    predicted_cost: float = Field(default=1.0, gt=0.0)
    uncertainty: float = Field(default=0.0, ge=0.0)
    score: float = 0.0
    recommended_operators: tuple[str, ...] = ()


class AgentAction(FrozenModel):
    operator: str
    closure_level: int = Field(ge=1, le=3)
    parameters: dict[str, Any] = Field(default_factory=dict)
    control: ControlAction = ControlAction.FULL_ORACLE


class InterventionProposal(FrozenModel):
    cip_id: str
    action: AgentAction
    released_operations: tuple[str, ...]
    frozen_operations: tuple[str, ...]
    priority_overrides: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    mode_overrides: dict[str, str] = Field(default_factory=dict)
    signature: str


class VerificationResult(FrozenModel):
    fidelity: Fidelity
    passed: bool
    proxy_delta: float | None = None
    true_delta: float | None = None
    objective: float | None = None
    failures: tuple[FailureLabel, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float = Field(default=0.0, ge=0.0)


class ExperimentRecord(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    instance_id: str
    iteration: int = Field(ge=0)
    incumbent_hash: str
    incumbent_objective: float
    cip: CausalInterventionPoint
    action: AgentAction
    proposal_signature: str
    verifications: tuple[VerificationResult, ...]
    accepted: bool
    new_objective: float | None = None
    delta_objective: float = 0.0
    best_updated: bool = False
    actual_closure: tuple[str, ...] = ()
    outside_changes: tuple[str, ...] = ()
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
