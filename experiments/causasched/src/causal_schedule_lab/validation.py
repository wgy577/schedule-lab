from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from .models import (
    FailureLabel,
    Fidelity,
    InterventionProposal,
    ProjectSemantics,
    VerificationResult,
)
from .repair import GeneratedCandidate
from .semantics import audit_semantic_evidence
from .objective import compare_objectives, evaluate_objective


def schedule_hash(schedule: Any) -> str:
    normalized = [
        {
            "operation": assignment.operation_id,
            "mode": assignment.mode_id,
            "start": assignment.start,
            "end": assignment.end,
        }
        for assignment in sorted(
            schedule.assignments,
            key=lambda item: item.operation_id,
        )
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MultiFidelityValidator:
    def __init__(
        self,
        *,
        semantics: ProjectSemantics,
        project_root: str,
        domain_oracle: Callable[[Any, Any, GeneratedCandidate], dict[str, Any]] | None = None,
    ) -> None:
        self.semantics = semantics
        self.project_root = project_root
        self.domain_oracle = domain_oracle

    def code_semantics(self) -> VerificationResult:
        started = time.perf_counter()
        audit = audit_semantic_evidence(
            self.semantics,
            project_root=self.project_root,
        )
        failures = (
            ()
            if audit["passed"]
            else (FailureLabel.SEMANTIC_EVIDENCE_MISSING,)
        )
        return VerificationResult(
            fidelity=Fidelity.CODE,
            passed=audit["passed"],
            failures=failures,
            details=audit,
            runtime_seconds=time.perf_counter() - started,
        )

    def static(
        self,
        *,
        problem: Any,
        incumbent: Any,
        proposal: InterventionProposal,
        candidate: GeneratedCandidate,
    ) -> VerificationResult:
        started = time.perf_counter()
        failures: list[FailureLabel] = []
        details: dict[str, Any] = {
            "releasedCount": len(proposal.released_operations),
            "frozenCount": len(proposal.frozen_operations),
        }
        all_operations = {operation.id for operation in problem.operations}
        if (
            set(proposal.released_operations)
            | set(proposal.frozen_operations)
        ) != all_operations:
            failures.append(FailureLabel.CLOSURE_TOO_SMALL)
        if (
            set(proposal.released_operations)
            & set(proposal.frozen_operations)
        ):
            failures.append(FailureLabel.CLOSURE_TOO_SMALL)
        if candidate.schedule is None:
            failures.append(FailureLabel.GENERATOR_FAILURE)
        else:
            from .core_validation import validate_schedule

            result = validate_schedule(problem, candidate.schedule)
            details["genericValidation"] = result.as_dict()
            incumbent_map = incumbent.assignment_map()
            candidate_map = candidate.schedule.assignment_map()
            frozen_changes = [
                operation_id
                for operation_id in proposal.frozen_operations
                if (
                    incumbent_map.get(operation_id) is None
                    or candidate_map.get(operation_id) is None
                    or (
                        incumbent_map[operation_id].mode_id,
                        incumbent_map[operation_id].start,
                        incumbent_map[operation_id].end,
                        incumbent_map[operation_id].route_id,
                    )
                    != (
                        candidate_map[operation_id].mode_id,
                        candidate_map[operation_id].start,
                        candidate_map[operation_id].end,
                        candidate_map[operation_id].route_id,
                    )
                )
            ]
            details["frozenChanges"] = frozen_changes
            if frozen_changes:
                failures.append(FailureLabel.OUTSIDE_MACHINE_CHANGE)
            codes = {issue.code for issue in result.errors}
            if "precedence" in codes:
                failures.append(FailureLabel.PRECEDENCE_VIOLATION)
            if "resource_capacity" in codes:
                failures.append(FailureLabel.RESOURCE_OVERLAP)
            if "machine_eligibility" in codes:
                failures.append(FailureLabel.MACHINE_INELIGIBILITY)
            if not result.feasible and not failures:
                failures.append(FailureLabel.GENERATOR_FAILURE)
        return VerificationResult(
            fidelity=Fidelity.STATIC,
            passed=not failures,
            failures=tuple(dict.fromkeys(failures)),
            details=details,
            runtime_seconds=time.perf_counter() - started,
        )

    def light(
        self,
        *,
        problem: Any,
        incumbent: Any,
        candidate: GeneratedCandidate,
        predicted_closure_size: int,
    ) -> VerificationResult:
        started = time.perf_counter()
        if candidate.schedule is None:
            return VerificationResult(
                fidelity=Fidelity.LIGHT,
                passed=False,
                failures=(FailureLabel.GENERATOR_FAILURE,),
                runtime_seconds=time.perf_counter() - started,
            )
        base_objective = evaluate_objective(problem, incumbent, baseline=incumbent)
        trial_objective = evaluate_objective(
            problem,
            candidate.schedule,
            baseline=incumbent,
        )
        comparison, proxy_delta, decisive_component = compare_objectives(
            trial_objective,
            base_objective,
        )
        base = dict(base_objective.metrics)
        trial = dict(trial_objective.metrics)
        failures = []
        if comparison >= 0:
            failures.append(FailureLabel.NO_TRUE_IMPROVEMENT)
        change_cost = trial.get("change_cost") or {}
        mode_changes = int(change_cost.get("mode_changes", 0))
        if mode_changes > max(1, predicted_closure_size):
            failures.append(FailureLabel.BOTTLENECK_TRANSFER)
        return VerificationResult(
            fidelity=Fidelity.LIGHT,
            passed=not failures,
            proxy_delta=proxy_delta,
            objective=trial_objective.values[0],
            failures=tuple(failures),
            details={
                "incumbentMetrics": base,
                "candidateMetrics": trial,
                "modeChanges": mode_changes,
                "incumbentObjectiveVector": base_objective.as_dict(),
                "candidateObjectiveVector": trial_objective.as_dict(),
                "decisiveComponent": decisive_component,
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def full(
        self,
        *,
        problem: Any,
        incumbent: Any,
        candidate: GeneratedCandidate,
    ) -> VerificationResult:
        started = time.perf_counter()
        if candidate.schedule is None:
            return VerificationResult(
                fidelity=Fidelity.FULL,
                passed=False,
                failures=(FailureLabel.GENERATOR_FAILURE,),
                runtime_seconds=time.perf_counter() - started,
            )
        from .core_validation import validate_schedule

        generic = validate_schedule(problem, candidate.schedule)
        failures: list[FailureLabel] = []
        details: dict[str, Any] = {
            "genericValidation": generic.as_dict(),
            "candidateHash": schedule_hash(candidate.schedule),
        }
        if not generic.feasible:
            failures.append(FailureLabel.GENERATOR_FAILURE)
        oracle_constraints = [
            item.id
            for item in getattr(problem, "constraints", ())
            if item.encoded_by == "oracle"
        ]
        if (
            problem.metadata.get("requires_domain_validation")
            or oracle_constraints
            or self.domain_oracle is not None
        ):
            if self.domain_oracle is not None:
                domain = self.domain_oracle(problem, incumbent, candidate)
                details["domainOracle"] = domain
                if not domain.get("passed", False):
                    failures.extend(
                        FailureLabel(item)
                        for item in domain.get("failureLabels", [])
                    )
            elif not candidate.domain_validated:
                failures.append(FailureLabel.GENERATOR_FAILURE)
                details["domainOracle"] = {
                    "passed": False,
                    "reason": "domain validation not certified",
                    "requiredConstraints": oracle_constraints,
                }
            else:
                details["domainOracle"] = {
                    "passed": True,
                    "source": "conditional generator performed original domain replay",
                }
        base_objective = evaluate_objective(problem, incumbent, baseline=incumbent)
        trial_objective = evaluate_objective(
            problem,
            candidate.schedule,
            baseline=incumbent,
        )
        comparison, delta, decisive_component = compare_objectives(
            trial_objective,
            base_objective,
        )
        details["incumbentObjectiveVector"] = base_objective.as_dict()
        details["candidateObjectiveVector"] = trial_objective.as_dict()
        details["decisiveComponent"] = decisive_component
        if comparison >= 0:
            failures.append(FailureLabel.NO_TRUE_IMPROVEMENT)
        return VerificationResult(
            fidelity=Fidelity.FULL,
            passed=not failures,
            true_delta=delta,
            objective=trial_objective.values[0],
            failures=tuple(dict.fromkeys(failures)),
            details=details,
            runtime_seconds=time.perf_counter() - started,
        )
