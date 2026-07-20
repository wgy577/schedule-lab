from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .generic_neighborhood import diagnose_bottlenecks
from .metrics import schedule_metrics
from .model import Problem, Schedule
from .strategy_router import build_agent_diagnosis, route_improvement_strategies
from .validation import validate_schedule


@dataclass(frozen=True)
class ImprovementStrategy:
    id: str
    role: str
    families: tuple[str, ...]
    deterministic: bool
    incumbent_preserving: bool
    search_basis: str
    repair_engine: str
    trigger: str
    combines_with: tuple[str, ...] = ()


STRATEGIES: tuple[ImprovementStrategy, ...] = (
    ImprovementStrategy(
        id="sequence-preserving-compaction",
        role="timing normalization",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Fix modes and resource orders; left-shift avoidable idle without changing dispatch decisions.",
        repair_engine="difference constraints or CP-SAT",
        trigger="always run first to separate timing slack from sequencing/routing defects",
    ),
    ImprovementStrategy(
        id="critical-block-vns",
        role="local sequencing",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Rank critical blocks or sink gaps and expand radii in a fixed order.",
        repair_engine="CP-SAT or exact enumeration",
        trigger="critical block, bottleneck idle gap, starvation, or blocking",
        combines_with=("causal-closure-fix-and-optimize", "tabu-signature-memory"),
    ),
    ImprovementStrategy(
        id="causal-closure-fix-and-optimize",
        role="graph-guided repair",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis=(
            "Trace backward from an objective bottleneck through precedence, resource, route, and "
            "blocking arcs; release the smallest propagation-closed causal set instead of a radius."
        ),
        repair_engine="CP-SAT local branching with frozen complement",
        trigger="radius neighborhoods either release too much or repeatedly require propagation expansion",
        combines_with=("dual-price-guided-release", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="local-branching-fix-and-optimize",
        role="exact stable repair",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Bound the Hamming distance in mode and sequence arcs rather than selecting jobs by geometry.",
        repair_engine="CP-SAT or MILP",
        trigger="a good incumbent exists but the responsible operation set is uncertain",
        combines_with=("causal-closure-fix-and-optimize", "tabu-signature-memory"),
    ),
    ImprovementStrategy(
        id="shifting-bottleneck-decomposition",
        role="machine/stage decomposition",
        families=("JSP", "FSP", "HFSP", "CARRIER"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Optimize the currently most constraining machine or stage, then reinsert it and recompute criticality.",
        repair_engine="single-machine exact sequencing plus global propagation",
        trigger="one machine or final stage dominates critical-path length",
        combines_with=("critical-block-vns", "sequence-preserving-compaction"),
    ),
    ImprovementStrategy(
        id="assignment-sequence-alternation",
        role="routing and sequencing coordination",
        families=("FJSP", "HFSP", "CARRIER"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Alternate a bounded eligible-machine reassignment step with an exact sequence repair step.",
        repair_engine="CP-SAT fix-and-optimize",
        trigger="parallel-resource load imbalance or expensive route/machine binding",
        combines_with=("dual-price-guided-release", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="dual-price-guided-release",
        role="relaxation-guided diagnosis",
        families=("FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Use LP/Lagrangian congestion prices to rank resource-time arcs whose release has the largest estimated value.",
        repair_engine="LP/Lagrangian relaxation followed by CP-SAT repair",
        trigger="many eligible resources or several competing bottlenecks make gap ranking ambiguous",
        combines_with=("causal-closure-fix-and-optimize", "assignment-sequence-alternation"),
    ),
    ImprovementStrategy(
        id="rolling-horizon-relax-and-fix",
        role="time decomposition",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Freeze the completed prefix, optimize an overlapping time window, then advance with overlap consistency.",
        repair_engine="CP-SAT rolling horizon",
        trigger="large instances or long suffixes exceed a useful exact-repair budget",
        combines_with=("sequence-preserving-compaction", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="deterministic-alns",
        role="plateau escape",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Destroy ranked related blocks, gaps, late suffixes, or overloaded routes in a fixed order.",
        repair_engine="CP-SAT, exact insertion, or fixed-width beam repair",
        trigger="one-component VNS and exact local branching reach a declared plateau",
        combines_with=("tabu-signature-memory", "bayesian-budget-allocation"),
    ),
    ImprovementStrategy(
        id="elite-path-relinking",
        role="structured diversification",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Move from the incumbent toward a second validated elite by adopting ranked mode or sequence arcs one at a time.",
        repair_engine="deterministic repair at each checkpoint",
        trigger="at least two structurally different validated elite schedules exist",
        combines_with=("local-branching-fix-and-optimize", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="logic-based-benders-oracle-cuts",
        role="master/subproblem decomposition",
        families=("FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis=(
            "Let a master choose assignments and sequence arcs; let an exact timing or domain "
            "subproblem return reusable no-good, precedence, separation, or path cuts."
        ),
        repair_engine="CP-SAT/MILP master plus CP or domain-Oracle subproblem",
        trigger="domain replay is expensive and repeated failures share structural causes",
        combines_with=("causal-closure-fix-and-optimize", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="critical-path-decision-diagram",
        role="bounded exact/beam sequencing",
        families=("JSP", "FSP", "HFSP", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Enumerate partial critical-block orders with stable dominance and a fixed state width.",
        repair_engine="decision diagram or deterministic beam search",
        trigger="small critical blocks recur and CP-SAT repeatedly explores equivalent partial orders",
        combines_with=("shifting-bottleneck-decomposition", "tabu-signature-memory"),
    ),
    ImprovementStrategy(
        id="robust-scenario-polishing",
        role="uncertainty and stability",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Replay a fixed versioned scenario set and optimize tail degradation after hard feasibility.",
        repair_engine="scenario CP-SAT or deterministic simulation optimization",
        trigger="processing, travel, setup, or collision delays are uncertain",
        combines_with=("rolling-horizon-relax-and-fix", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="constructive-heuristic-proposals",
        role="cheap proposal generation",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=False,
        search_basis="Use stable NEH, shifting-bottleneck, insertion, ATC/SPT, or load-balancing rules only to propose orders/modes.",
        repair_engine="deterministic projection and validation",
        trigger="exact methods need diverse starting arcs or a cheap lower-quality reference",
        combines_with=("elite-path-relinking", "local-branching-fix-and-optimize"),
    ),
    ImprovementStrategy(
        id="bayesian-budget-allocation",
        role="experiment selection",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Rank operator and neighborhood configurations from validated gain, feasibility, and Oracle-cost posteriors.",
        repair_engine="posterior ranking only; never constructs or validates schedules",
        trigger="at least 12 deduplicated validated experiments are available",
        combines_with=("critical-block-vns", "deterministic-alns", "domain-oracle"),
    ),
    ImprovementStrategy(
        id="tabu-signature-memory",
        role="duplicate and cycle control",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Hash changed modes, sequence arcs, release boundary, and solver settings.",
        repair_engine="controller memory",
        trigger="always enable once more than one candidate is evaluated",
    ),
    ImprovementStrategy(
        id="domain-oracle",
        role="final feasibility authority",
        families=("JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"),
        deterministic=True,
        incumbent_preserving=True,
        search_basis="Replay real routes, time windows, collisions, continuity, and state-dependent transitions.",
        repair_engine="project simulator/oracle",
        trigger="canonical model omits any domain hard constraint",
    ),
)


def strategy_catalog() -> dict[str, ImprovementStrategy]:
    return {strategy.id: strategy for strategy in STRATEGIES}


def _phase(index: int, purpose: str, strategy_ids: list[str], *, acceptance_gate: str) -> dict[str, Any]:
    catalog = strategy_catalog()
    return {
        "phase": index,
        "purpose": purpose,
        "strategies": [asdict(catalog[strategy_id]) for strategy_id in strategy_ids],
        "acceptanceGate": acceptance_gate,
    }


def build_improvement_workflow(
    problem: Problem,
    incumbent: Schedule,
    *,
    evidence_count: int = 0,
    validated_elite_count: int = 0,
    no_improvement_count: int = 0,
    oracle_failure_count: int = 0,
    speed_profile: str = "fast",
    max_structural_methods: int | None = None,
) -> dict[str, Any]:
    """Build a stable multi-method improvement workflow around an incumbent.

    This function selects methods and their order. It does not claim that a
    method produced a feasible candidate; the validator and optional domain
    Oracle remain separate gates.
    """

    validation = validate_schedule(problem, incumbent)
    if not validation.feasible:
        raise ValueError(f"incumbent is infeasible: {validation.errors}")
    family = problem.kind
    metrics = schedule_metrics(problem, incumbent, incumbent)
    bottlenecks = diagnose_bottlenecks(problem, incumbent)[:8]
    agent_diagnosis = build_agent_diagnosis(
        problem,
        incumbent,
        bottlenecks,
        evidence_count=evidence_count,
        validated_elite_count=validated_elite_count,
        no_improvement_count=no_improvement_count,
        oracle_failure_count=oracle_failure_count,
    )
    catalog = strategy_catalog()
    compatible_strategy_ids = {
        strategy.id for strategy in STRATEGIES if family in strategy.families
    }
    strategy_routing = route_improvement_strategies(
        agent_diagnosis,
        compatible_strategy_ids=compatible_strategy_ids,
        speed_profile=speed_profile,
        max_structural_methods=max_structural_methods,
    )
    oracle_required = bool(problem.metadata.get("requires_domain_validation"))
    full_gate = "generic validation + frozen-region equality + lexicographic improvement"
    if oracle_required:
        full_gate += " + domain Oracle replay"

    family_methods = strategy_routing["selectedStructuralMethods"]

    phases = [
        _phase(
            0,
            "Audit and normalize the incumbent; diagnose causes rather than optimizing makespan blindly.",
            ["sequence-preserving-compaction"],
            acceptance_gate=full_gate,
        ),
        _phase(
            1,
            "Run only the cost-aware family shortlist under a bounded candidate budget.",
            family_methods + ["tabu-signature-memory"],
            acceptance_gate=full_gate,
        ),
    ]
    if speed_profile in {"balanced", "thorough"}:
        exact_methods = [
            strategy_id
            for strategy_id in (
                "causal-closure-fix-and-optimize",
                "local-branching-fix-and-optimize",
            )
            if strategy_id in compatible_strategy_ids and strategy_id not in family_methods
        ]
        if exact_methods:
            phases.append(
                _phase(
                    len(phases),
                    "Use a propagation-closed exact repair only after the fast shortlist fails.",
                    exact_methods,
                    acceptance_gate=full_gate,
                )
            )
    if speed_profile == "thorough" and no_improvement_count >= 4:
        phases.append(
            _phase(
                len(phases),
                "Use bounded decomposition or ALNS only after a declared plateau.",
                ["rolling-horizon-relax-and-fix", "deterministic-alns"],
                acceptance_gate=full_gate,
            )
        )
    if speed_profile != "fast" and validated_elite_count >= 2:
        phases.append(
            _phase(
                len(phases),
                "Relink two validated elite schedules through repaired intermediate schedules.",
                ["elite-path-relinking"],
                acceptance_gate=full_gate,
            )
        )

    contingent_ids = ["robust-scenario-polishing"]
    if family in {"FJSP", "HFSP", "CARRIER", "GENERIC"}:
        contingent_ids.insert(0, "logic-based-benders-oracle-cuts")
    if family in {"JSP", "FSP", "HFSP", "GENERIC"}:
        contingent_ids.append("critical-path-decision-diagram")
    if speed_profile != "fast" and evidence_count >= 12:
        phases.append(
            _phase(
                len(phases),
                "Allocate the next expensive deterministic experiment from accumulated evidence.",
                ["bayesian-budget-allocation"],
                acceptance_gate="selected experiment must still pass " + full_gate,
            )
        )
    if oracle_required:
        phases.append(
            _phase(
                len(phases),
                "Certify every surviving candidate in the original domain environment.",
                ["domain-oracle"],
                acceptance_gate=full_gate,
            )
        )

    return {
        "problemId": problem.id,
        "family": family,
        "incumbent": {
            "makespan": incumbent.makespan,
            "makespanDisplay": metrics["makespan_display"],
            "assignmentCount": len(incumbent.assignments),
            "genericValid": validation.feasible,
            "domainOracleRequired": oracle_required,
        },
        "objectiveOrder": [
            "hard-constraint violations",
            "declared tardiness/service objective",
            "true makespan",
            "bottleneck idle/starvation/blocking",
            "flow time/WIP",
            "route/setup/travel/energy cost",
            "change cost",
            "robustness",
        ],
        "diagnostics": bottlenecks,
        "agentDiagnosis": agent_diagnosis,
        "strategyRouting": strategy_routing,
        "controller": {
            "policy": "strictly improving incumbent loop",
            "speedProfile": speed_profile,
            "executionBudget": strategy_routing["executionBudget"],
            "restartRule": "after acceptance, recompute diagnostics and restart at phase 0",
            "plateauRule": "advance only after a declared deterministic no-improvement budget",
            "candidateRule": "heuristics propose; exact repair constructs; validators and Oracle decide",
        },
        "phases": phases,
        "contingentStrategies": [asdict(catalog[strategy_id]) for strategy_id in contingent_ids],
    }
