from __future__ import annotations

import hashlib
import json
from typing import Any

from .generic_neighborhood import build_generic_neighborhood_plan, repair_generic_neighborhood
from .improvement_workflow import build_improvement_workflow
from .metrics import schedule_metrics
from .model import Problem, Schedule
from .validation import validate_schedule


BOTTLENECK_METHOD_ORDER: dict[str, tuple[str, ...]] = {
    "critical_resource_block": (
        "critical-block-vns",
        "shifting-bottleneck-decomposition",
        "local-branching-fix-and-optimize",
    ),
    "resource_idle_gap": (
        "critical-block-vns",
        "causal-closure-fix-and-optimize",
    ),
    "sink_stage_gap": (
        "shifting-bottleneck-decomposition",
        "constructive-heuristic-proposals",
        "critical-block-vns",
    ),
    "flexible_resource_imbalance": (
        "assignment-sequence-alternation",
        "dual-price-guided-release",
        "local-branching-fix-and-optimize",
    ),
}


def _schedule_hash(schedule: Schedule) -> str:
    payload = schedule.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method_for_bottleneck(kind: str, shortlist: list[str]) -> str:
    for method in BOTTLENECK_METHOD_ORDER.get(kind, ()):
        if method in shortlist:
            return method
    return shortlist[0]


def fast_improve_incumbent(
    problem: Problem,
    incumbent: Schedule,
    *,
    speed_profile: str = "fast",
    seed: int = 0,
    evidence_count: int = 0,
    validated_elite_count: int = 0,
    no_improvement_count: int = 0,
    oracle_failure_count: int = 0,
    max_structural_methods: int | None = None,
    exclude_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a small cost-aware shortlist and monotonically retain the best schedule."""

    incumbent_validation = validate_schedule(problem, incumbent)
    if not incumbent_validation.feasible:
        raise ValueError(f"incumbent is infeasible: {incumbent_validation.errors}")
    workflow = build_improvement_workflow(
        problem,
        incumbent,
        evidence_count=evidence_count,
        validated_elite_count=validated_elite_count,
        no_improvement_count=no_improvement_count,
        oracle_failure_count=oracle_failure_count,
        speed_profile=speed_profile,
        max_structural_methods=max_structural_methods,
    )
    routing = workflow["strategyRouting"]
    budget = routing["executionBudget"]
    shortlist = list(routing["selectedStructuralMethods"])
    plan = build_generic_neighborhood_plan(
        problem,
        incumbent,
        max_bottlenecks=int(budget["maxBottlenecks"]),
        radii=tuple(map(int, budget["radii"])),
        include_pairs=False,
    )
    operation_count = max(1, len(problem.operations))
    released_fraction_cap = (
        1.0 if operation_count <= 40 else float(budget["maxReleasedFraction"])
    )
    experiments: list[dict[str, Any]] = []
    best = incumbent
    oracle_handoff: list[dict[str, Any]] = []
    exclude_signatures = exclude_signatures or set()
    for neighborhood in plan["neighborhoods"]:
        if len(experiments) >= int(budget["maxCandidates"]):
            break
        released_fraction = int(neighborhood["releasedCount"]) / operation_count
        if neighborhood["signature"] in exclude_signatures:
            continue
        if released_fraction > released_fraction_cap:
            continue
        bottleneck_kind = str(neighborhood["bottleneck"]["kind"])
        method = _method_for_bottleneck(bottleneck_kind, shortlist)
        repair = repair_generic_neighborhood(
            problem,
            incumbent,
            neighborhood,
            seed=seed,
            deterministic_time=float(budget["deterministicRepairTime"]),
            stability_weight=1,
        )
        candidate = (
            Schedule.model_validate(repair["schedule"])
            if "schedule" in repair
            else None
        )
        experiment = {
            "index": len(experiments),
            "strategyId": method,
            "signature": neighborhood["signature"],
            "bottleneckKind": bottleneck_kind,
            "releasedCount": int(neighborhood["releasedCount"]),
            "releasedFraction": released_fraction,
            "status": repair["status"],
            "provisional": bool(repair.get("provisional", False)),
            "accepted": bool(repair.get("accepted", False)),
            "reason": repair.get("reason"),
            "candidateMakespan": None if candidate is None else candidate.makespan,
            "candidateHash": None if candidate is None else _schedule_hash(candidate),
        }
        experiments.append(experiment)
        if candidate is not None and repair.get("provisional", False):
            if candidate.makespan < incumbent.makespan:
                oracle_handoff.append(
                    {
                        "experimentIndex": experiment["index"],
                        "candidate": candidate.model_dump(mode="json"),
                    }
                )
            continue
        if candidate is not None and repair.get("accepted", False):
            best = candidate
            break

    accepted = _schedule_hash(best) != _schedule_hash(incumbent)
    incumbent_metrics = schedule_metrics(problem, incumbent, incumbent)
    result_metrics = schedule_metrics(problem, best, incumbent)
    return {
        "controller": "cost-aware-fast-incumbent-improvement-v1",
        "problemId": problem.id,
        "family": problem.kind,
        "speedProfile": speed_profile,
        "shortlist": shortlist,
        "budget": {
            **budget,
            "effectiveReleasedFractionCap": released_fraction_cap,
            "solverWorkers": 1,
            "seed": seed,
        },
        "incumbent": {
            "hash": _schedule_hash(incumbent),
            "makespan": incumbent.makespan,
            "metrics": incumbent_metrics,
        },
        "result": {
            "hash": _schedule_hash(best),
            "makespan": best.makespan,
            "metrics": result_metrics,
            "accepted": accepted,
            "improvement": incumbent.makespan - best.makespan,
            "fallbackUsed": not accepted,
            "schedule": best.model_dump(mode="json"),
        },
        "experiments": experiments,
        "oracleHandoff": oracle_handoff[: int(budget["maxOracleCandidates"])],
        "acceptance": (
            "strict generic improvement"
            if accepted
            else "retain exact incumbent; no validated shortlist improvement"
        ),
    }


def adaptive_improve_incumbent(
    problem: Problem,
    incumbent: Schedule,
    *,
    seed: int = 0,
    evidence_count: int = 0,
    validated_elite_count: int = 0,
    no_improvement_count: int = 0,
    oracle_failure_count: int = 0,
) -> dict[str, Any]:
    """Run fast first and escalate once to balanced only when evidence warrants it."""

    stages: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    fast = fast_improve_incumbent(
        problem,
        incumbent,
        speed_profile="fast",
        seed=seed,
        evidence_count=evidence_count,
        validated_elite_count=validated_elite_count,
        no_improvement_count=no_improvement_count,
        oracle_failure_count=oracle_failure_count,
        max_structural_methods=3,
    )
    stages.append(fast)
    seen_signatures.update(item["signature"] for item in fast["experiments"])
    if fast["result"]["accepted"] or fast["oracleHandoff"]:
        selected = fast
        stop_reason = (
            "fast strict improvement"
            if fast["result"]["accepted"]
            else "fast produced a provisional candidate requiring domain Oracle"
        )
    else:
        balanced = fast_improve_incumbent(
            problem,
            incumbent,
            speed_profile="balanced",
            seed=seed,
            evidence_count=evidence_count,
            validated_elite_count=validated_elite_count,
            no_improvement_count=max(1, no_improvement_count),
            oracle_failure_count=oracle_failure_count,
            max_structural_methods=4,
            exclude_signatures=seen_signatures,
        )
        stages.append(balanced)
        selected = balanced if balanced["result"]["accepted"] or balanced["oracleHandoff"] else fast
        stop_reason = (
            "balanced strict improvement"
            if balanced["result"]["accepted"]
            else "balanced produced a provisional candidate requiring domain Oracle"
            if balanced["oracleHandoff"]
            else "no validated improvement; retain exact incumbent"
        )
    return {
        "controller": "adaptive-fast-then-balanced-v1",
        "problemId": problem.id,
        "family": problem.kind,
        "stageCount": len(stages),
        "stopReason": stop_reason,
        "evaluatedCandidateCount": sum(len(stage["experiments"]) for stage in stages),
        "stages": stages,
        "result": selected["result"],
        "oracleHandoff": selected["oracleHandoff"],
    }
