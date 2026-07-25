from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import MaskedPPOAgent
from .audit import append_record
from .cip import CausalCoreDiscoverer
from .graph import build_scheduling_graph
from .models import AgentAction, ControlAction, ExperimentRecord
from .operators import ActionIndex, build_intervention
from .objective import evaluate_objective
from .repair import ConditionalRepairGenerator
from .validation import MultiFidelityValidator, schedule_hash
from .posterior import MultiFidelityPosterior


@dataclass(frozen=True)
class ImprovementResult:
    best_schedule: Any
    records: tuple[ExperimentRecord, ...]
    stop_reason: str


class AgenticImprovementController:
    """CIP → masked action → conditional repair → multi-fidelity gates."""

    def __init__(
        self,
        *,
        project_id: str,
        discoverer: CausalCoreDiscoverer,
        action_index: ActionIndex,
        policy: MaskedPPOAgent,
        generator: ConditionalRepairGenerator,
        validator: MultiFidelityValidator,
        posterior: MultiFidelityPosterior | None = None,
        audit_log: str | Path | None = None,
    ) -> None:
        self.project_id = project_id
        self.discoverer = discoverer
        self.action_index = action_index
        self.policy = policy
        self.generator = generator
        self.validator = validator
        self.posterior = posterior or MultiFidelityPosterior()
        self.audit_log = None if audit_log is None else Path(audit_log)

    def run(
        self,
        *,
        problem: Any,
        incumbent: Any,
        max_iterations: int = 3,
        top_k: int = 4,
        candidate_budget: int = 8,
        full_oracle_budget: int | None = None,
        deterministic_policy: bool = True,
    ) -> ImprovementResult:
        code_gate = self.validator.code_semantics()
        if not code_gate.passed:
            raise RuntimeError("project semantic evidence gate failed")
        best = incumbent
        records: list[ExperimentRecord] = []
        evaluated = 0
        full_calls = 0
        full_oracle_budget = (
            candidate_budget
            if full_oracle_budget is None
            else full_oracle_budget
        )
        stop_reason = "iteration_budget"
        for iteration in range(max_iterations):
            graph = build_scheduling_graph(
                problem,
                best,
                project_id=self.project_id,
            )
            candidates = self.discoverer.discover(
                problem=problem,
                schedule=best,
                graph=graph,
                top_k=top_k,
            )
            if not candidates:
                stop_reason = "no_causal_intervention_points"
                break
            accepted_this_iteration = False
            for cip in candidates:
                if evaluated >= candidate_budget:
                    stop_reason = "candidate_budget"
                    break
                started = time.perf_counter()
                action, policy_info = self.policy.choose(
                    cip,
                    remaining_budget=1.0 - evaluated / max(1, candidate_budget),
                    deterministic=deterministic_policy,
                )
                # Untrained policies are still constrained to CIP legal actions.
                if action.operator not in cip.recommended_operators:
                    action = AgentAction(
                        operator=cip.recommended_operators[0],
                        closure_level=max(cip.closure.level, action.closure_level),
                        control=ControlAction.FULL_ORACLE,
                    )
                proposal = build_intervention(
                    problem=problem,
                    schedule=best,
                    cip=cip,
                    action=action,
                )
                generated = self.generator.generate(
                    problem=problem,
                    incumbent=best,
                    cip=cip,
                    proposal=proposal,
                )
                static = self.validator.static(
                    problem=problem,
                    incumbent=best,
                    proposal=proposal,
                    candidate=generated,
                )
                verifications = [code_gate, static]
                full = None
                if static.passed:
                    light = self.validator.light(
                        problem=problem,
                        incumbent=best,
                        candidate=generated,
                        predicted_closure_size=len(proposal.released_operations),
                    )
                    verifications.append(light)
                    if light.passed and full_calls < full_oracle_budget:
                        full = self.validator.full(
                            problem=problem,
                            incumbent=best,
                            candidate=generated,
                        )
                        verifications.append(full)
                        full_calls += 1
                accepted = bool(full and full.passed)
                candidate_schedule = generated.schedule if accepted else None
                delta = 0.0 if full is None or full.true_delta is None else full.true_delta
                incumbent_objective = evaluate_objective(
                    problem,
                    best,
                    baseline=best,
                )
                candidate_objective = (
                    None
                    if candidate_schedule is None
                    else evaluate_objective(
                        problem,
                        candidate_schedule,
                        baseline=best,
                    )
                )
                record = ExperimentRecord(
                    project_id=self.project_id,
                    instance_id=problem.id,
                    iteration=iteration,
                    incumbent_hash=schedule_hash(best),
                    incumbent_objective=incumbent_objective.values[0],
                    cip=cip,
                    action=action,
                    proposal_signature=proposal.signature,
                    verifications=tuple(verifications),
                    accepted=accepted,
                    new_objective=(
                        None
                        if candidate_schedule is None
                        else candidate_objective.values[0]
                    ),
                    delta_objective=delta,
                    best_updated=accepted,
                    actual_closure=proposal.released_operations,
                    outside_changes=tuple(
                        generated.details.get("requiredExpansionJobs", [])
                    ),
                    runtime_seconds=time.perf_counter() - started,
                    metadata={
                        "policy": policy_info,
                        "generatorStatus": generated.status,
                        "generatorFidelity": generated.fidelity,
                        "incumbentObjectiveVector": incumbent_objective.as_dict(),
                        "candidateObjectiveVector": (
                            None
                            if candidate_objective is None
                            else candidate_objective.as_dict()
                        ),
                    },
                )
                records.append(record)
                self.posterior.update(record)
                if self.audit_log is not None:
                    append_record(record, self.audit_log)
                evaluated += 1
                if accepted and candidate_schedule is not None:
                    best = candidate_schedule
                    accepted_this_iteration = True
                    break
            if evaluated >= candidate_budget:
                break
            if not accepted_this_iteration:
                stop_reason = "no_validated_improvement"
                break
        return ImprovementResult(
            best_schedule=best,
            records=tuple(records),
            stop_reason=stop_reason,
        )
