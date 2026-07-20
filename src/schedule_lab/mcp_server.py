from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from .carrier import DEFAULT_CARRIER_SCHEDULE, load_carrier_baseline
from .carrier_oracle import search_carrier_policy
from .fast_controller import adaptive_improve_incumbent, fast_improve_incumbent
from .improvement_workflow import build_improvement_workflow
from .joint_schedule_trajectory import JointOptimizationSettings, build_joint_schedule_trajectory_plan
from .metrics import bottleneck_advice, schedule_metrics
from .model import Problem, Schedule
from .portfolio import solve_portfolio
from .validation import validate_schedule


mcp = FastMCP("Schedule Lab")


@mcp.tool()
def scheduling_capabilities() -> dict:
    """Lists supported problem families, solvers and safety rules."""
    return {
        "problem_types": ["JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"],
        "solvers": ["dispatching portfolio", "PyJobShop", "OR-Tools CP-SAT"],
        "improvement_methods": [
            "sequence-preserving compaction",
            "critical-block VNS",
            "causal-closure fix-and-optimize",
            "local branching",
            "shifting bottleneck",
            "assignment/sequence alternation",
            "dual-price-guided release",
            "rolling horizon",
            "deterministic ALNS",
            "elite path relinking",
            "logic-based Benders with Oracle cuts",
            "critical-path decision diagrams",
            "robust scenario polishing",
            "Bayesian experiment allocation",
            "cost-aware fast strategy shortlist",
        ],
        "hard_rule": "a schedule is never accepted unless validate_schedule reports feasible",
        "carrier_rule": "new carrier schedules additionally require legacy trajectory/collision replay",
    }


@mcp.tool()
def analyze_schedule(problem_json: str, schedule_json: str) -> dict:
    """Validates and analyzes a schedule encoded with Schedule Lab JSON schemas."""
    problem = Problem.model_validate_json(problem_json)
    schedule = Schedule.model_validate_json(schedule_json)
    validation = validate_schedule(problem, schedule)
    metrics = schedule_metrics(problem, schedule)
    return {"validation": validation.model_dump(mode="json"), "metrics": metrics, "advice": bottleneck_advice(metrics)}


@mcp.tool()
def solve_problem(problem_json: str, time_limit: float = 10.0, seed: int = 0) -> dict:
    """Runs the heuristic and CP-SAT portfolio, returning the best feasible schedule."""
    problem = Problem.model_validate_json(problem_json)
    result = solve_portfolio(problem, time_limit=time_limit, seed=seed)
    return {
        "schedule": result.best.schedule.model_dump(mode="json"),
        "validation": result.best.validation.model_dump(mode="json"),
        "metrics": result.best.metrics,
        "candidate_count": len(result.candidates),
    }


@mcp.tool()
def compare_schedules(problem_json: str, baseline_json: str, candidate_json: str) -> dict:
    """Compares objective quality, feasibility and disruption against a baseline."""
    problem = Problem.model_validate_json(problem_json)
    baseline = Schedule.model_validate_json(baseline_json)
    candidate = Schedule.model_validate_json(candidate_json)
    validation = validate_schedule(problem, candidate)
    return {
        "candidate_validation": validation.model_dump(mode="json"),
        "baseline": schedule_metrics(problem, baseline),
        "candidate": schedule_metrics(problem, candidate, baseline),
    }


@mcp.tool()
def plan_schedule_improvement(
    problem_json: str,
    incumbent_json: str,
    evidence_count: int = 0,
    validated_elite_count: int = 0,
    no_improvement_count: int = 0,
    oracle_failure_count: int = 0,
    speed_profile: str = "fast",
    max_structural_methods: int | None = None,
) -> dict:
    """Builds a deterministic multi-method plan around an existing feasible incumbent."""
    problem = Problem.model_validate_json(problem_json)
    incumbent = Schedule.model_validate_json(incumbent_json)
    return build_improvement_workflow(
        problem,
        incumbent,
        evidence_count=evidence_count,
        validated_elite_count=validated_elite_count,
        no_improvement_count=no_improvement_count,
        oracle_failure_count=oracle_failure_count,
        speed_profile=speed_profile,
        max_structural_methods=max_structural_methods,
    )


@mcp.tool()
def fast_improve_schedule(
    problem_json: str,
    incumbent_json: str,
    speed_profile: str = "fast",
    max_structural_methods: int = 3,
    seed: int = 0,
) -> dict:
    """Diagnoses one incumbent and runs only a small cost-aware repair shortlist."""
    problem = Problem.model_validate_json(problem_json)
    incumbent = Schedule.model_validate_json(incumbent_json)
    return fast_improve_incumbent(
        problem,
        incumbent,
        speed_profile=speed_profile,
        max_structural_methods=max_structural_methods,
        seed=seed,
    )


@mcp.tool()
def adaptive_improve_schedule(
    problem_json: str,
    incumbent_json: str,
    seed: int = 0,
) -> dict:
    """Runs fast first, then one nonduplicate balanced tier only if fast fails."""
    problem = Problem.model_validate_json(problem_json)
    incumbent = Schedule.model_validate_json(incumbent_json)
    return adaptive_improve_incumbent(problem, incumbent, seed=seed)


@mcp.tool()
def plan_joint_schedule_and_trajectories(
    problem_json: str,
    incumbent_json: str,
    time_resolution: float = 0.1,
    validated_experiment_count: int = 0,
    trajectory_observation_count: int = 0,
) -> dict:
    """Plans an exact/anytime scheduling plus conflict-free trajectory decomposition."""
    problem = Problem.model_validate_json(problem_json)
    incumbent = Schedule.model_validate_json(incumbent_json)
    return build_joint_schedule_trajectory_plan(
        problem,
        incumbent,
        settings=JointOptimizationSettings(
            time_resolution=time_resolution,
            validated_experiment_count=validated_experiment_count,
            trajectory_observation_count=trajectory_observation_count,
        ),
    )


@mcp.tool()
def audit_current_carrier(schedule_path: str = str(DEFAULT_CARRIER_SCHEDULE)) -> dict:
    """Audits the existing deck_update 20-aircraft schedule without changing it."""
    problem, schedule = load_carrier_baseline(schedule_path)
    validation = validate_schedule(problem, schedule)
    metrics = schedule_metrics(problem, schedule)
    return {
        "validation": validation.model_dump(mode="json"),
        "metrics": metrics,
        "reported_policy_makespan": problem.metadata["reported_policy_makespan"],
        "actual_makespan": metrics["makespan_display"],
        "advice": bottleneck_advice(metrics),
    }


@mcp.tool()
def search_current_carrier(rollouts: int = 8, seed: int = 0) -> dict:
    """Generates policy candidates and validates each inside the legacy collision-aware environment."""
    if rollouts > 64:
        raise ValueError("MCP calls are capped at 64 rollouts; use CLI for longer experiments")
    return search_carrier_policy(rollouts=rollouts, seed=seed)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
