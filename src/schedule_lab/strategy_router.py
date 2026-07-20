from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .model import Problem, Schedule


@dataclass(frozen=True)
class FamilyStrategyPack:
    family: str
    diagnostic_focus: tuple[str, ...]
    primary: tuple[str, ...]
    exact_repair: tuple[str, ...]
    plateau: tuple[str, ...]
    decomposition: tuple[str, ...]


FAMILY_PACKS: dict[str, FamilyStrategyPack] = {
    "JSP": FamilyStrategyPack(
        family="JSP",
        diagnostic_focus=("critical_resource_block", "resource_idle_gap", "critical_path_tail"),
        primary=("shifting-bottleneck-decomposition", "critical-block-vns"),
        exact_repair=("local-branching-fix-and-optimize", "critical-path-decision-diagram"),
        plateau=("causal-closure-fix-and-optimize", "deterministic-alns"),
        decomposition=("rolling-horizon-relax-and-fix",),
    ),
    "FSP": FamilyStrategyPack(
        family="FSP",
        diagnostic_focus=("sink_stage_gap", "resource_idle_gap", "blocking"),
        primary=("shifting-bottleneck-decomposition", "constructive-heuristic-proposals"),
        exact_repair=("critical-block-vns", "local-branching-fix-and-optimize"),
        plateau=("causal-closure-fix-and-optimize", "deterministic-alns"),
        decomposition=("critical-path-decision-diagram", "rolling-horizon-relax-and-fix"),
    ),
    "FJSP": FamilyStrategyPack(
        family="FJSP",
        diagnostic_focus=("flexible_resource_imbalance", "critical_resource_block", "route_congestion"),
        primary=("assignment-sequence-alternation", "dual-price-guided-release"),
        exact_repair=("local-branching-fix-and-optimize", "causal-closure-fix-and-optimize"),
        plateau=("deterministic-alns", "logic-based-benders-oracle-cuts"),
        decomposition=("rolling-horizon-relax-and-fix",),
    ),
    "HFSP": FamilyStrategyPack(
        family="HFSP",
        diagnostic_focus=("sink_stage_gap", "flexible_resource_imbalance", "blocking", "starvation"),
        primary=("shifting-bottleneck-decomposition", "assignment-sequence-alternation"),
        exact_repair=("critical-block-vns", "causal-closure-fix-and-optimize"),
        plateau=("local-branching-fix-and-optimize", "deterministic-alns"),
        decomposition=("logic-based-benders-oracle-cuts", "rolling-horizon-relax-and-fix"),
    ),
    "CARRIER": FamilyStrategyPack(
        family="CARRIER",
        diagnostic_focus=("sink_stage_gap", "route_congestion", "vehicle_continuity", "collision_delay"),
        primary=("shifting-bottleneck-decomposition", "assignment-sequence-alternation"),
        exact_repair=("causal-closure-fix-and-optimize", "local-branching-fix-and-optimize"),
        plateau=("logic-based-benders-oracle-cuts", "deterministic-alns"),
        decomposition=("rolling-horizon-relax-and-fix",),
    ),
    "GENERIC": FamilyStrategyPack(
        family="GENERIC",
        diagnostic_focus=("critical_resource_block", "sink_stage_gap", "resource_idle_gap"),
        primary=("critical-block-vns", "local-branching-fix-and-optimize"),
        exact_repair=("causal-closure-fix-and-optimize",),
        plateau=("deterministic-alns", "logic-based-benders-oracle-cuts"),
        decomposition=("rolling-horizon-relax-and-fix",),
    ),
}


DIAGNOSTIC_STRATEGIES: dict[str, tuple[str, ...]] = {
    "critical_resource_block": (
        "critical-block-vns",
        "shifting-bottleneck-decomposition",
        "local-branching-fix-and-optimize",
    ),
    "resource_idle_gap": (
        "sequence-preserving-compaction",
        "critical-block-vns",
        "causal-closure-fix-and-optimize",
    ),
    "sink_stage_gap": (
        "shifting-bottleneck-decomposition",
        "critical-block-vns",
        "causal-closure-fix-and-optimize",
    ),
    "flexible_resource_imbalance": (
        "assignment-sequence-alternation",
        "dual-price-guided-release",
        "local-branching-fix-and-optimize",
    ),
    "blocking": ("causal-closure-fix-and-optimize", "rolling-horizon-relax-and-fix"),
    "starvation": ("causal-closure-fix-and-optimize", "shifting-bottleneck-decomposition"),
    "route_congestion": ("dual-price-guided-release", "logic-based-benders-oracle-cuts"),
    "vehicle_continuity": ("causal-closure-fix-and-optimize", "logic-based-benders-oracle-cuts"),
    "collision_delay": ("logic-based-benders-oracle-cuts", "causal-closure-fix-and-optimize"),
}


STRATEGY_COST_UNITS: dict[str, float] = {
    "sequence-preserving-compaction": 1.0,
    "constructive-heuristic-proposals": 1.0,
    "tabu-signature-memory": 1.0,
    "bayesian-budget-allocation": 1.0,
    "critical-block-vns": 2.0,
    "shifting-bottleneck-decomposition": 2.0,
    "dual-price-guided-release": 2.0,
    "assignment-sequence-alternation": 3.0,
    "local-branching-fix-and-optimize": 3.0,
    "critical-path-decision-diagram": 3.0,
    "causal-closure-fix-and-optimize": 4.0,
    "rolling-horizon-relax-and-fix": 5.0,
    "deterministic-alns": 6.0,
    "elite-path-relinking": 6.0,
    "robust-scenario-polishing": 6.0,
    "logic-based-benders-oracle-cuts": 8.0,
    "domain-oracle": 10.0,
}


SPEED_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "maxStructuralMethods": 3,
        "maxBottlenecks": 2,
        "radii": [2],
        "maxCandidates": 4,
        "deterministicRepairTime": 0.25,
        "maxReleasedFraction": 0.3,
        "maxOracleCandidates": 1,
        "earlyStop": "first strict validated improvement",
    },
    "balanced": {
        "maxStructuralMethods": 4,
        "maxBottlenecks": 3,
        "radii": [2, 3],
        "maxCandidates": 8,
        "deterministicRepairTime": 0.75,
        "maxReleasedFraction": 0.4,
        "maxOracleCandidates": 2,
        "earlyStop": "first material strict validated improvement",
    },
    "thorough": {
        "maxStructuralMethods": 6,
        "maxBottlenecks": 4,
        "radii": [2, 3, 4],
        "maxCandidates": 16,
        "deterministicRepairTime": 2.0,
        "maxReleasedFraction": 0.5,
        "maxOracleCandidates": 4,
        "earlyStop": "declared candidate budget",
    },
}


def family_strategy_pack(family: str) -> dict[str, Any]:
    pack = FAMILY_PACKS[family]
    return {
        "family": pack.family,
        "diagnosticFocus": list(pack.diagnostic_focus),
        "lanes": {
            "primary": list(pack.primary),
            "exactRepair": list(pack.exact_repair),
            "plateau": list(pack.plateau),
            "decomposition": list(pack.decomposition),
        },
    }


def build_agent_diagnosis(
    problem: Problem,
    incumbent: Schedule,
    diagnostics: Iterable[dict[str, Any]],
    *,
    evidence_count: int = 0,
    validated_elite_count: int = 0,
    no_improvement_count: int = 0,
    oracle_failure_count: int = 0,
) -> dict[str, Any]:
    records = list(diagnostics)
    kinds = Counter(str(item["kind"]) for item in records)
    positive_scores = [max(0.0, float(item.get("score", 0.0))) for item in records]
    score_total = sum(positive_scores)
    top = records[0] if records else None
    flexible_operations = sum(len(operation.modes) > 1 for operation in problem.operations)
    job_count = len({operation.job_id for operation in problem.operations})
    operation_count = len(problem.operations)
    if operation_count <= 40:
        scale = "small"
    elif operation_count <= 200:
        scale = "medium"
    else:
        scale = "large"
    return {
        "family": problem.kind,
        "problemScale": scale,
        "jobCount": job_count,
        "operationCount": operation_count,
        "resourceCount": len(problem.resources),
        "flexibleOperationFraction": round(flexible_operations / max(1, operation_count), 6),
        "dominantBottleneck": None if top is None else top["kind"],
        "dominantEvidence": top,
        "bottleneckCounts": dict(sorted(kinds.items())),
        "topBottleneckConcentration": (
            0.0 if not positive_scores or score_total <= 0 else round(positive_scores[0] / score_total, 6)
        ),
        "domainOracleRequired": bool(problem.metadata.get("requires_domain_validation")),
        "plateauDetected": no_improvement_count >= 4,
        "noImprovementCount": int(no_improvement_count),
        "oracleFailureCount": int(oracle_failure_count),
        "evidenceReady": evidence_count >= 12,
        "validatedEliteReady": validated_elite_count >= 2,
    }


def route_improvement_strategies(
    diagnosis: dict[str, Any],
    *,
    compatible_strategy_ids: set[str] | None = None,
    speed_profile: str = "fast",
    max_structural_methods: int | None = None,
) -> dict[str, Any]:
    """Rank compatible strategy lanes from an auditable Agent diagnosis."""

    family = str(diagnosis["family"])
    if speed_profile not in SPEED_PROFILES:
        raise ValueError(f"unknown speed profile: {speed_profile}")
    execution_budget = dict(SPEED_PROFILES[speed_profile])
    if max_structural_methods is not None:
        if max_structural_methods < 1:
            raise ValueError("max_structural_methods must be positive")
        execution_budget["maxStructuralMethods"] = int(max_structural_methods)
    pack = FAMILY_PACKS[family]
    dominant = diagnosis.get("dominantBottleneck")
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(strategy_id: str, score: float, reason: str) -> None:
        if compatible_strategy_ids is not None and strategy_id not in compatible_strategy_ids:
            return
        scores[strategy_id] = scores.get(strategy_id, 0.0) + score
        reasons.setdefault(strategy_id, []).append(reason)

    add("sequence-preserving-compaction", 100.0, "safe timing normalization precedes structural changes")
    for rank, strategy_id in enumerate(pack.primary):
        add(strategy_id, 85.0 - 5.0 * rank, f"{family} primary strategy pack")
    for rank, strategy_id in enumerate(pack.exact_repair):
        add(strategy_id, 65.0 - 4.0 * rank, f"{family} bounded exact-repair lane")
    for rank, strategy_id in enumerate(DIAGNOSTIC_STRATEGIES.get(str(dominant), ())):
        add(strategy_id, 30.0 - 3.0 * rank, f"matches diagnosed {dominant}")

    if diagnosis.get("problemScale") == "large":
        add("rolling-horizon-relax-and-fix", 35.0, "large operation count favors time decomposition")
    if diagnosis.get("plateauDetected"):
        for rank, strategy_id in enumerate(pack.plateau):
            add(strategy_id, 45.0 - 3.0 * rank, "declared deterministic no-improvement plateau")
    if diagnosis.get("oracleFailureCount", 0) >= 2 and family in {"FJSP", "HFSP", "CARRIER", "GENERIC"}:
        add("logic-based-benders-oracle-cuts", 50.0, "repeated Oracle failures can produce reusable cuts")
    if diagnosis.get("validatedEliteReady"):
        add("elite-path-relinking", 25.0, "two validated elite schedules are available")
    if diagnosis.get("evidenceReady"):
        add("bayesian-budget-allocation", 20.0, "enough validated evidence exists to allocate experiments")
    add("tabu-signature-memory", 15.0, "avoid deterministic duplicate candidates and cycles")
    if diagnosis.get("domainOracleRequired"):
        add("domain-oracle", 1000.0, "mandatory final authority for unmodeled domain constraints")

    ranked = [
        {
            "rank": rank,
            "strategyId": strategy_id,
            "score": round(score, 6),
            "estimatedCostUnits": STRATEGY_COST_UNITS.get(strategy_id, 5.0),
            "efficiencyScore": round(
                score / STRATEGY_COST_UNITS.get(strategy_id, 5.0), 6
            ),
            "reasons": reasons[strategy_id],
            "role": (
                "mandatory-gate"
                if strategy_id == "domain-oracle"
                else "controller-memory"
                if strategy_id == "tabu-signature-memory"
                else "experiment-selector"
                if strategy_id == "bayesian-budget-allocation"
                else "candidate-method"
            ),
        }
        for rank, (strategy_id, score) in enumerate(
            sorted(
                scores.items(),
                key=lambda item: (
                    -item[1] / STRATEGY_COST_UNITS.get(item[0], 5.0),
                    -item[1],
                    item[0],
                ),
            ),
            start=1,
        )
    ]
    candidate_methods = [
        item["strategyId"]
        for item in ranked
        if item["role"] == "candidate-method"
        and item["strategyId"] != "sequence-preserving-compaction"
    ]
    return {
        "router": "deterministic-cost-aware-agent-router-v2",
        "speedProfile": speed_profile,
        "executionBudget": execution_budget,
        "familyPack": family_strategy_pack(family),
        "rankedStrategies": ranked,
        "selectedStructuralMethods": candidate_methods[
            : int(execution_budget["maxStructuralMethods"])
        ],
        "shortlistPolicy": (
            "rank compatible methods by diagnostic suitability divided by estimated cost; "
            "run cheap deterministic gates before any domain Oracle"
        ),
        "fallback": {
            "schedule": "retain incumbent",
            "guarantees": [
                "no infeasible candidate replaces the incumbent",
                "no lexicographically worse candidate replaces the incumbent",
                "the incumbent remains exactly recoverable",
            ],
            "notGuaranteed": "a strict improvement exists for every instance",
        },
    }
