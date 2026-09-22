"""Read-only DeepSeek-assisted quality audit for persisted trajectories.

CP-SAT remains the sole source of outcome truth.  The external model may only
score whether an already verified experience is causally coherent and useful
for learning.  Results are emitted as a sidecar report; input Memory records
and their labels are never mutated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..memory import InterventionExperience
from ..providers import ModelRequest
from ..training.readiness_audit_v1 import audit_counterfactual
from .llm_provider_adapter import LLMProviderAdapter


AUDIT_SCHEMA = "deepseek_trajectory_quality_audit_v1"
PROMPT_VERSION = "trajectory-quality-audit-v1"
HIGH_QUALITY = "high_quality"
MEDIUM_QUALITY = "medium_quality"
LOW_QUALITY = "low_quality"
ACCEPTED_FOR_TRAINING = "accepted_for_training"
REVIEW_REQUIRED = "review_required"
REJECTED_FOR_TRAINING = "rejected_for_training"


@dataclass(frozen=True)
class TrajectoryQualityAuditConfig:
    provider: str = "deepseek"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_attempts: int = 2
    thinking_mode: str = "disabled"
    high_quality_threshold: float = 0.75
    medium_quality_threshold: float = 0.50
    score_weights: Mapping[str, float] = field(default_factory=lambda: {
        "evidence_complete": 0.25,
        "causal_alignment": 0.20,
        "intervention_quality": 0.15,
        "counterfactual_confidence": 0.20,
        "training_value": 0.20,
    })
    hidden_change_penalty: float = 0.15
    duplicate_penalty: float = 0.10

    def validate(self) -> None:
        if self.provider.lower() != "deepseek":
            raise ValueError("trajectory quality auditor requires provider=deepseek")
        if self.max_tokens < 128 or self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("invalid trajectory quality audit provider budget")
        if self.thinking_mode not in {"enabled", "disabled", "adaptive"}:
            raise ValueError("invalid trajectory quality audit thinking_mode")
        expected = {
            "evidence_complete", "causal_alignment", "intervention_quality",
            "counterfactual_confidence", "training_value",
        }
        if set(self.score_weights) != expected:
            raise ValueError("trajectory quality score_weights have wrong keys")
        if abs(sum(float(v) for v in self.score_weights.values()) - 1.0) > 1e-9:
            raise ValueError("trajectory quality score_weights must sum to 1")
        if any(float(v) < 0 for v in self.score_weights.values()):
            raise ValueError("trajectory quality score weights cannot be negative")
        if not 0 <= self.medium_quality_threshold <= self.high_quality_threshold <= 1:
            raise ValueError("invalid trajectory quality thresholds")
        if self.hidden_change_penalty < 0 or self.duplicate_penalty < 0:
            raise ValueError("trajectory quality penalties cannot be negative")


def load_trajectory_quality_audit_config(
    path: str | Path,
) -> TrajectoryQualityAuditConfig:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    section = payload.get("trajectory_quality_audit")
    if not isinstance(section, Mapping):
        raise ValueError("missing trajectory_quality_audit hyperparameters")
    config = TrajectoryQualityAuditConfig(
        provider=str(section.get("provider", "deepseek")),
        model=str(section.get("model") or ""),
        temperature=float(section.get("temperature", 0.2)),
        max_tokens=int(section.get("max_tokens", 2048)),
        timeout_seconds=float(section.get("timeout_seconds", 120.0)),
        max_attempts=int(section.get("max_attempts", 2)),
        thinking_mode=str(section.get("thinking_mode", "disabled")),
        high_quality_threshold=float(section.get("high_quality_threshold", 0.75)),
        medium_quality_threshold=float(section.get("medium_quality_threshold", 0.50)),
        score_weights={
            str(key): float(value)
            for key, value in section.get("score_weights", {}).items()
        },
        hidden_change_penalty=float(section.get("hidden_change_penalty", 0.15)),
        duplicate_penalty=float(section.get("duplicate_penalty", 0.10)),
    )
    config.validate()
    return config


class _DeepSeekQualityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: str = Field(min_length=1)
    causal_alignment: float = Field(ge=0.0, le=1.0)
    intervention_quality: float = Field(ge=0.0, le=1.0)
    counterfactual_confidence: float = Field(ge=0.0, le=1.0)
    training_value: float = Field(ge=0.0, le=1.0)
    closure_alignment: float | None = Field(default=None, ge=0.0, le=1.0)
    proposal_to_closure_dependency: float | None = Field(default=None, ge=0.0, le=1.0)
    issues: tuple[str, ...] = Field(default=(), max_length=20)


@dataclass(frozen=True)
class QualityAuditMetadata:
    auditor: str
    model: str
    timestamp: str
    causal_alignment_score: float | None
    intervention_quality_score: float | None
    counterfactual_confidence: float | None
    training_value_score: float | None
    issues: tuple[str, ...]
    closure_alignment_score: float | None = None
    proposal_to_closure_dependency_score: float | None = None
    used_for_training_label: bool = False


@dataclass(frozen=True)
class TrajectoryQualityAuditResult:
    trajectory_id: str
    quality_class: str
    audit_status: str
    quality_score: float
    evidence_complete: bool
    evidence_failures: tuple[str, ...]
    duplicate_count: int
    changed_operation_count: int
    hidden_change_count: int
    hidden_change_ratio: float
    truth_sha256: str
    quality_audit: QualityAuditMetadata
    counterfactual_mode: str = "unknown"
    closure_size: int = 0
    outside_closure_change_count: int = 0
    failure_reason: str = ""
    api_called: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 0
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class QualityAuditReport:
    schema: str
    prompt_version: str
    timestamp: str
    provider: str
    model: str
    total_records: int
    high_quality: int
    medium_quality: int
    low_quality: int
    operator_distribution: Mapping[str, int]
    appearance_distribution: Mapping[str, int]
    common_failure_reason: Mapping[str, int]
    api_call_count: int
    api_attempt_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    formal_training: bool
    optimizer_steps: int
    test_access: int
    identified: bool
    records: tuple[TrajectoryQualityAuditResult, ...]


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _truth_payload(record: InterventionExperience) -> dict[str, Any]:
    outcome = record.outcome
    return {
        "trajectory_id": record.key,
        "delta_cmax": float(outcome.delta_cmax) if outcome else None,
        "classification": outcome.classification if outcome else "pending",
        "future_gain": float(record.trajectory_final_gain),
        "future_success": record.trajectory_future_success,
        "risk": float(outcome.risk) if outcome else None,
    }


def _state_summary(state: Any | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "cmax": float(state.cmax),
        "gap": float(state.gap),
        "critical_machine": state.critical_machine,
        "critical_path_length": float(state.critical_path_length),
        "appearance": state.appearance_type,
        "overloaded_machine": state.overloaded_machine,
        "underloaded_machine": state.underloaded_machine,
        "load_difference": float(state.load_difference),
        "load_variance": float(state.load_variance),
        "local_operation_nodes": list(state.local_operation_nodes),
        "local_machine_nodes": list(state.local_machine_nodes),
        "local_precedence_edges": [list(edge) for edge in state.local_precedence_edges],
        "local_resource_sequence_edges": [
            list(edge) for edge in state.local_resource_sequence_edges
        ],
    }


def _test_forbidden(record: InterventionExperience) -> bool:
    metadata = record.metadata
    return bool(
        metadata.get("formal_test_source") is True
        or int(metadata.get("formal_test_access", 0) or 0) != 0
        or str(metadata.get("source_split", "")).lower() == "test"
    )


def _assignment_changes(record: InterventionExperience) -> tuple[int, int, float]:
    metadata = record.metadata
    before = metadata.get("before_schedule_snapshot", {})
    after = metadata.get("after_schedule_snapshot", {})
    before_rows = before.get("assignments", ()) if isinstance(before, Mapping) else ()
    after_rows = after.get("assignments", ()) if isinstance(after, Mapping) else ()

    def index(rows: Any) -> dict[str, tuple[Any, ...]]:
        result = {}
        for row in rows if isinstance(rows, (list, tuple)) else ():
            if not isinstance(row, Mapping) or not row.get("operation_id"):
                continue
            result[str(row["operation_id"])] = (
                row.get("mode_id"), row.get("start"), row.get("end"), row.get("route_id")
            )
        return result

    before_index, after_index = index(before_rows), index(after_rows)
    changed = {
        operation for operation in set(before_index) | set(after_index)
        if before_index.get(operation) != after_index.get(operation)
    }
    requested = {
        str(edit.get("operation_id"))
        for edit in metadata.get("requested_edits", ())
        if isinstance(edit, Mapping) and edit.get("operation_id")
    }
    allowed = (
        set(str(item) for item in metadata.get("closure_operations", ()))
        if metadata.get("counterfactual_mode") == "frozen_local"
        else requested
    )
    hidden = changed - allowed
    ratio = len(hidden) / len(changed) if changed else 0.0
    return len(changed), len(hidden), float(ratio)


def _duplicate_fingerprint(record: InterventionExperience) -> str:
    proposal = asdict(record.proposal)
    proposal.pop("proposal_id", None)
    return _canonical_sha({"state": asdict(record.state), "proposal": proposal})


class TrajectoryQualityAuditorV1:
    """Audit resolved TRAIN trajectories without changing outcome truth."""

    def __init__(
        self,
        adapter: LLMProviderAdapter,
        *,
        config: TrajectoryQualityAuditConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or TrajectoryQualityAuditConfig()
        self.config.validate()

    def audit(
        self,
        trajectory_records: Sequence[InterventionExperience],
    ) -> QualityAuditReport:
        records = tuple(sorted(
            trajectory_records,
            key=lambda item: (
                0 if item.metadata.get("counterfactual_mode") == "frozen_local" else 1,
                item.key,
            ),
        ))
        if any(_test_forbidden(record) for record in records):
            raise ValueError("formal_test_trajectory_forbidden")
        duplicate_counts = Counter(_duplicate_fingerprint(record) for record in records)
        results = tuple(
            self._audit_one(
                record,
                duplicate_count=duplicate_counts[_duplicate_fingerprint(record)],
            )
            for record in records
        )
        quality_counts = Counter(item.quality_class for item in results)
        operators = Counter(record.proposal.operator_type or "unknown" for record in records)
        appearances = Counter(record.proposal.appearance_type or "unknown" for record in records)
        failure_reasons = Counter()
        for item in results:
            if item.failure_reason:
                failure_reasons[item.failure_reason] += 1
            if item.quality_class == LOW_QUALITY:
                for issue in item.quality_audit.issues:
                    failure_reasons[issue] += 1
        return QualityAuditReport(
            schema=AUDIT_SCHEMA,
            prompt_version=PROMPT_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=self.adapter.provider.name,
            model=self.adapter.provider.model,
            total_records=len(results),
            high_quality=quality_counts[HIGH_QUALITY],
            medium_quality=quality_counts[MEDIUM_QUALITY],
            low_quality=quality_counts[LOW_QUALITY],
            operator_distribution=dict(sorted(operators.items())),
            appearance_distribution=dict(sorted(appearances.items())),
            common_failure_reason=dict(failure_reasons.most_common(20)),
            api_call_count=sum(item.api_called for item in results),
            api_attempt_count=sum(item.attempts for item in results),
            input_tokens=sum(item.input_tokens for item in results),
            output_tokens=sum(item.output_tokens for item in results),
            total_tokens=sum(item.total_tokens for item in results),
            formal_training=False,
            optimizer_steps=0,
            test_access=0,
            identified=False,
            records=results,
        )

    def _audit_one(
        self,
        record: InterventionExperience,
        *,
        duplicate_count: int,
    ) -> TrajectoryQualityAuditResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        evidence = audit_counterfactual(record)
        changed_count, hidden_count, hidden_ratio = _assignment_changes(record)
        mode = str(record.metadata.get("counterfactual_mode", "unknown"))
        closure_size = len(record.metadata.get("closure_operations", ()))
        truth_sha = _canonical_sha(_truth_payload(record))
        declared_outside = tuple(record.metadata.get("outside_closure_changes", ()))
        if mode == "frozen_local" and (hidden_count > 0 or declared_outside):
            return self._failed_result(
                record, timestamp=timestamp, truth_sha=truth_sha,
                evidence_failures=("outside_closure_changes",),
                duplicate_count=duplicate_count, changed_count=changed_count,
                hidden_count=max(hidden_count, len(declared_outside)),
                hidden_ratio=hidden_ratio,
                failure_reason="deterministic_outside_closure_change",
            )
        if not evidence.valid:
            return self._failed_result(
                record, timestamp=timestamp, truth_sha=truth_sha,
                evidence_failures=evidence.failures,
                duplicate_count=duplicate_count, changed_count=changed_count,
                hidden_count=hidden_count, hidden_ratio=hidden_ratio,
                failure_reason="deterministic_evidence_incomplete",
            )
        prompt = self._build_prompt(
            record, evidence.checks,
            duplicate_count=duplicate_count,
            changed_count=changed_count,
            hidden_count=hidden_count,
            hidden_ratio=hidden_ratio,
        )
        response = None
        try:
            response = self.adapter.provider.complete(ModelRequest(
                system=(
                    "You are a scheduling-trajectory quality reviewer. CP-SAT has "
                    "already produced the immutable outcome truth. Assess learning "
                    "quality only. Never modify or restate a label, reward, delta, "
                    "future gain, risk, or success decision. Return strict JSON."
                ),
                user=prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                require_json=True,
                thinking_mode=self.config.thinking_mode,
                metadata={
                    "task": "deepseek_trajectory_quality_audit_v1",
                    "trajectory_id": record.key,
                    "prompt_version": PROMPT_VERSION,
                },
            ))
            parsed = _DeepSeekQualityResponse.model_validate_json(response.content)
            if parsed.trajectory_id != record.key:
                raise ValueError("trajectory_id_mismatch")
        except (Exception, ValidationError) as error:
            # Provider, JSON, schema, forbidden truth fields and id mismatch all
            # fail closed into a sidecar rejection; Memory remains untouched.
            return self._failed_result(
                record, timestamp=timestamp, truth_sha=truth_sha,
                evidence_failures=evidence.failures,
                duplicate_count=duplicate_count, changed_count=changed_count,
                hidden_count=hidden_count, hidden_ratio=hidden_ratio,
                failure_reason=f"audit_response_rejected:{type(error).__name__}",
                api_called=True,
                response=response,
            )
        quality_score = self._quality_score(
            evidence_complete=True,
            causal_alignment=parsed.causal_alignment,
            intervention_quality=parsed.intervention_quality,
            counterfactual_confidence=parsed.counterfactual_confidence,
            training_value=parsed.training_value,
            hidden_change_ratio=hidden_ratio,
            duplicate_count=duplicate_count,
        )
        quality_class, audit_status = self._classify(quality_score)
        if mode == "free_global":
            audit_status = REJECTED_FOR_TRAINING
        issues = tuple(dict.fromkeys(str(issue)[:500] for issue in parsed.issues if issue.strip()))
        if hidden_ratio > 0:
            issues += (f"solver_hidden_change_ratio:{hidden_ratio:.6f}",)
        if duplicate_count > 1:
            issues += (f"duplicate_experience_count:{duplicate_count}",)
        if mode == "free_global":
            issues += ("free_global_never_training_eligible",)
        return TrajectoryQualityAuditResult(
            trajectory_id=record.key,
            quality_class=quality_class,
            audit_status=audit_status,
            quality_score=quality_score,
            evidence_complete=True,
            evidence_failures=(),
            duplicate_count=duplicate_count,
            changed_operation_count=changed_count,
            hidden_change_count=hidden_count,
            hidden_change_ratio=hidden_ratio,
            truth_sha256=truth_sha,
            quality_audit=QualityAuditMetadata(
                auditor=self.adapter.provider.name,
                model=response.response_model or response.requested_model,
                timestamp=timestamp,
                causal_alignment_score=float(parsed.causal_alignment),
                intervention_quality_score=float(parsed.intervention_quality),
                counterfactual_confidence=float(parsed.counterfactual_confidence),
                training_value_score=float(parsed.training_value),
                issues=issues,
                closure_alignment_score=(
                    float(parsed.closure_alignment)
                    if parsed.closure_alignment is not None else None
                ),
                proposal_to_closure_dependency_score=(
                    float(parsed.proposal_to_closure_dependency)
                    if parsed.proposal_to_closure_dependency is not None else None
                ),
            ),
            counterfactual_mode=mode,
            closure_size=closure_size,
            outside_closure_change_count=hidden_count if mode == "frozen_local" else 0,
            api_called=True,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            attempts=response.attempts,
            latency_seconds=response.latency_seconds,
        )

    def _quality_score(
        self, *, evidence_complete: bool, causal_alignment: float,
        intervention_quality: float, counterfactual_confidence: float,
        training_value: float, hidden_change_ratio: float, duplicate_count: int,
    ) -> float:
        values = {
            "evidence_complete": 1.0 if evidence_complete else 0.0,
            "causal_alignment": causal_alignment,
            "intervention_quality": intervention_quality,
            "counterfactual_confidence": counterfactual_confidence,
            "training_value": training_value,
        }
        positive = sum(
            float(self.config.score_weights[name]) * float(value)
            for name, value in values.items()
        )
        penalty = (
            self.config.hidden_change_penalty * max(0.0, min(1.0, hidden_change_ratio))
            + self.config.duplicate_penalty * (1.0 if duplicate_count > 1 else 0.0)
        )
        return round(max(0.0, min(1.0, positive - penalty)), 8)

    def _classify(self, score: float) -> tuple[str, str]:
        if score >= self.config.high_quality_threshold:
            return HIGH_QUALITY, ACCEPTED_FOR_TRAINING
        if score >= self.config.medium_quality_threshold:
            return MEDIUM_QUALITY, REVIEW_REQUIRED
        return LOW_QUALITY, REJECTED_FOR_TRAINING

    def _failed_result(
        self, record: InterventionExperience, *, timestamp: str, truth_sha: str,
        evidence_failures: Sequence[str], duplicate_count: int,
        changed_count: int, hidden_count: int, hidden_ratio: float,
        failure_reason: str,
        api_called: bool = False,
        response: Any | None = None,
    ) -> TrajectoryQualityAuditResult:
        issues = tuple(dict.fromkeys((*evidence_failures, failure_reason)))
        return TrajectoryQualityAuditResult(
            trajectory_id=record.key,
            quality_class=LOW_QUALITY,
            audit_status=REJECTED_FOR_TRAINING,
            quality_score=0.0,
            evidence_complete=False if evidence_failures else True,
            evidence_failures=tuple(evidence_failures),
            duplicate_count=duplicate_count,
            changed_operation_count=changed_count,
            hidden_change_count=hidden_count,
            hidden_change_ratio=hidden_ratio,
            truth_sha256=truth_sha,
            quality_audit=QualityAuditMetadata(
                auditor=self.adapter.provider.name,
                model=self.adapter.provider.model,
                timestamp=timestamp,
                causal_alignment_score=None,
                intervention_quality_score=None,
                counterfactual_confidence=None,
                training_value_score=None,
                issues=issues,
            ),
            counterfactual_mode=str(record.metadata.get("counterfactual_mode", "unknown")),
            closure_size=len(record.metadata.get("closure_operations", ())),
            outside_closure_change_count=(
                hidden_count
                if record.metadata.get("counterfactual_mode") == "frozen_local" else 0
            ),
            failure_reason=failure_reason,
            api_called=api_called,
            input_tokens=(response.usage.input_tokens if response is not None else 0),
            output_tokens=(response.usage.output_tokens if response is not None else 0),
            total_tokens=(response.usage.total_tokens if response is not None else 0),
            attempts=(response.attempts if response is not None else 0),
            latency_seconds=(response.latency_seconds if response is not None else 0.0),
        )

    def _build_prompt(
        self, record: InterventionExperience, evidence_checks: Mapping[str, bool], *,
        duplicate_count: int, changed_count: int, hidden_count: int,
        hidden_ratio: float,
    ) -> str:
        outcome = record.outcome
        payload = {
            "task": "assess_verified_trajectory_learning_quality_only",
            "trajectory_id": record.key,
            "immutable_cp_sat_truth": _truth_payload(record),
            "before_state_summary": _state_summary(record.state),
            "after_state_summary": _state_summary(record.after_state),
            "appearance": record.proposal.appearance_type,
            "causal_chain": list(record.proposal.causal_chain),
            "causal_relations": list(record.proposal.causal_relations),
            "root_decision": record.proposal.root_decision_id,
            "operator": record.proposal.operator_type,
            "proposal": {
                "actions": list(record.proposal.intervention_actions),
                "dependencies": [list(edge) for edge in record.proposal.dependency_edges],
                "affected_region": list(record.proposal.affected_region),
            },
            "solver_evidence": {
                "checks": dict(evidence_checks),
                "solver_status": record.metadata.get("solver_status"),
                "validator_kind": record.metadata.get("validator_kind"),
                "requested_action_ids": record.metadata.get("requested_action_ids", ()),
                "action_checks": record.metadata.get("action_checks", {}),
                "changed_operation_count": changed_count,
                "hidden_change_count": hidden_count,
                "hidden_change_ratio": hidden_ratio,
                "duplicate_count": duplicate_count,
                "collateral_damage": (
                    float(outcome.collateral_damage) if outcome else None
                ),
                "counterfactual_mode": record.metadata.get("counterfactual_mode"),
                "closure_operations": record.metadata.get("closure_operations", ()),
                "closure_size": len(record.metadata.get("closure_operations", ())),
                "outside_closure_changes": record.metadata.get("outside_closure_changes", ()),
                "proposal_to_closure_dependency": (
                    "Each closure member must have explicit release provenance."
                ),
            },
            "review_questions": {
                "causal_alignment": (
                    "Does the proposal address this chain/root, or is it unrelated?"
                ),
                "intervention_quality": (
                    "Is this a purposeful scheduling intervention rather than a random edit?"
                ),
                "counterfactual_confidence": (
                    "Could the observed outcome mainly come from solver freedom or hidden rearrangement?"
                ),
                "training_value": (
                    "Would this teach root-to-intervention reasoning without shortcut or false causality?"
                ),
                "closure_alignment": (
                    "Does the bounded closure contain the proposal/dependency operations without unrelated releases?"
                ),
            },
            "forbidden": [
                "Do not change, correct, predict, or add delta_cmax, future_gain, risk, reward, success, or failure.",
                "Do not decide the final training gate.",
            ],
            "output_schema": {
                "trajectory_id": record.key,
                "causal_alignment": "number 0..1",
                "intervention_quality": "number 0..1",
                "counterfactual_confidence": "number 0..1",
                "training_value": "number 0..1",
                "closure_alignment": "optional number 0..1",
                "proposal_to_closure_dependency": "optional number 0..1",
                "issues": ["short issue strings"],
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ACCEPTED_FOR_TRAINING", "AUDIT_SCHEMA", "HIGH_QUALITY", "LOW_QUALITY",
    "MEDIUM_QUALITY", "PROMPT_VERSION", "QualityAuditMetadata",
    "QualityAuditReport", "REJECTED_FOR_TRAINING", "REVIEW_REQUIRED",
    "TrajectoryQualityAuditConfig", "TrajectoryQualityAuditResult",
    "TrajectoryQualityAuditorV1", "load_trajectory_quality_audit_config",
]
