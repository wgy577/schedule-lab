from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import CausalInterventionPoint, InterventionProposal


@dataclass(frozen=True)
class GeneratedCandidate:
    schedule: Any | None
    status: str
    fidelity: str
    domain_validated: bool
    details: dict[str, Any] = field(default_factory=dict)


class ConditionalRepairGenerator(Protocol):
    def generate(
        self,
        *,
        problem: Any,
        incumbent: Any,
        cip: CausalInterventionPoint,
        proposal: InterventionProposal,
    ) -> GeneratedCandidate: ...


class GenericCPSATRepairGenerator:
    """Partial-schedule completion with the incumbent as fixed context."""

    def __init__(
        self,
        *,
        seed: int = 0,
        deterministic_time: float = 1.0,
        stability_weight: int = 1,
    ) -> None:
        self.seed = seed
        self.deterministic_time = deterministic_time
        self.stability_weight = stability_weight

    def generate(
        self,
        *,
        problem: Any,
        incumbent: Any,
        cip: CausalInterventionPoint,
        proposal: InterventionProposal,
    ) -> GeneratedCandidate:
        from .solvers.cp_sat import solve_cp_sat

        result = solve_cp_sat(
            problem,
            incumbent=incumbent,
            frozen_operation_ids=set(proposal.frozen_operations),
            seed=self.seed,
            max_deterministic_time=self.deterministic_time,
            stability_weight=self.stability_weight,
        )
        if result.schedule is None:
            return GeneratedCandidate(
                schedule=None,
                status=result.status,
                fidelity="light",
                domain_validated=False,
                details={
                    "status": result.status,
                    "objectiveBound": result.objective_bound,
                    "conflicts": result.conflicts,
                    "branches": result.branches,
                    "solverTime": result.deterministic_time,
                },
            )
        return GeneratedCandidate(
            schedule=result.schedule,
            status=result.status,
            fidelity="light",
            domain_validated=False,
            details={
                "status": result.status,
                "objectiveBound": result.objective_bound,
                "conflicts": result.conflicts,
                "branches": result.branches,
                "solverTime": result.deterministic_time,
                "conditionalContext": {
                    "released": list(proposal.released_operations),
                    "frozen": list(proposal.frozen_operations),
                    "operator": proposal.action.operator,
                    "priorityOverrides": {
                        key: list(value)
                        for key, value in proposal.priority_overrides.items()
                    },
                    "modeOverrides": proposal.mode_overrides,
                },
            },
        )
