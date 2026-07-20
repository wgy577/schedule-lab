from __future__ import annotations

from dataclasses import dataclass

from .metrics import schedule_metrics
from .model import Problem, Schedule
from .solvers import solve_cp_sat, solve_dispatching, solve_pyjobshop
from .validation import ValidationResult, validate_schedule


@dataclass(frozen=True)
class CandidateResult:
    schedule: Schedule
    validation: ValidationResult
    metrics: dict


@dataclass(frozen=True)
class PortfolioResult:
    best: CandidateResult
    candidates: tuple[CandidateResult, ...]


def solve_portfolio(
    problem: Problem,
    *,
    time_limit: float = 10.0,
    seed: int = 0,
    baseline: Schedule | None = None,
    stability_weight: int = 0,
) -> PortfolioResult:
    candidates: list[CandidateResult] = []
    for rule in ("earliest_finish", "spt", "lpt", "most_successors", "balanced"):
        schedule = solve_dispatching(problem, rule=rule)
        validation = validate_schedule(problem, schedule)
        if validation.feasible:
            candidates.append(CandidateResult(schedule=schedule, validation=validation, metrics=schedule_metrics(problem, schedule, baseline)))
    warm_start = min(candidates, key=lambda item: (item.metrics["makespan"], item.metrics["total_flow_time"])).schedule if candidates else baseline
    try:
        pyjobshop_schedule = solve_pyjobshop(problem, time_limit=max(0.5, time_limit / 2))
    except (NotImplementedError, RuntimeError, ValueError):
        pyjobshop_schedule = None
    if pyjobshop_schedule is not None:
        validation = validate_schedule(problem, pyjobshop_schedule)
        if validation.feasible:
            candidates.append(CandidateResult(schedule=pyjobshop_schedule, validation=validation, metrics=schedule_metrics(problem, pyjobshop_schedule, baseline)))
    cp_result = solve_cp_sat(problem, time_limit=time_limit, seed=seed, warm_start=warm_start, stability_weight=stability_weight)
    if cp_result.schedule is not None:
        validation = validate_schedule(problem, cp_result.schedule)
        if validation.feasible:
            candidates.append(CandidateResult(schedule=cp_result.schedule, validation=validation, metrics=schedule_metrics(problem, cp_result.schedule, baseline)))
    if not candidates:
        raise RuntimeError("solver portfolio did not produce a feasible schedule")
    best = min(candidates, key=lambda item: (item.metrics["makespan"], item.metrics["total_tardiness"], item.metrics["total_flow_time"]))
    return PortfolioResult(best=best, candidates=tuple(candidates))
