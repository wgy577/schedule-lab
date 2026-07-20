from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .generic_neighborhood import diagnose_bottlenecks
from .model import Problem, Schedule
from .validation import validate_schedule


@dataclass(frozen=True)
class JointOptimizationSettings:
    time_resolution: float = 0.1
    route_library_version: str = "unversioned"
    geometry_version: str = "unversioned"
    oracle_version: str = "legacy-fixed-trajectory-delay"
    exact_target: bool = True
    validated_experiment_count: int = 0
    trajectory_observation_count: int = 0


def version_file_set(paths: list[str | Path], *, root: str | Path | None = None) -> str:
    """Return a stable SHA-256 for named files and their contents."""

    resolved_root = Path(root).resolve() if root is not None else None
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: str(item)):
        try:
            name = path.relative_to(resolved_root) if resolved_root is not None else path
        except ValueError:
            name = path
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}:{len(set(files))}-files"


def build_agentic_rl_contract(
    problem: Problem,
    incumbent: Schedule,
    *,
    validated_experiment_count: int,
    trajectory_observation_count: int,
) -> dict[str, Any]:
    """Describe a safe learning controller that selects solver experiments."""

    operation_count = len(problem.operations)
    enough_imitation_data = validated_experiment_count >= 500
    enough_joint_data = trajectory_observation_count >= 500
    encoder_recommended = enough_imitation_data and (
        operation_count > 40 or enough_joint_data
    )
    return {
        "role": (
            "select bottleneck, neighborhood, operator, route column and solver budget; "
            "never certify feasibility or directly emit unconstrained start times"
        ),
        "learningStages": [
            {
                "stage": 0,
                "method": "deterministic contextual bandit/posterior routing",
                "trigger": "default while validated evidence is sparse",
            },
            {
                "stage": 1,
                "method": "offline imitation of deterministic and exact-solver decisions",
                "trigger": "at least 500 deduplicated validated experiments",
            },
            {
                "stage": 2,
                "method": "offline/conservative actor-critic for neighborhood control",
                "trigger": "imitation policy passes held-out feasibility and regret gates",
            },
            {
                "stage": 3,
                "method": "shielded online fine-tuning in simulator only",
                "trigger": "never before deterministic rollback and Oracle shielding exist",
            },
        ],
        "encoder": {
            "recommendedNow": encoder_recommended,
            "decision": (
                "heterogeneous graph encoder plus trajectory encoder"
                if encoder_recommended
                else "use explicit diagnostic features first; collect validated data before training an encoder"
            ),
            "scheduleGraph": {
                "nodeTypes": ["operation", "job", "resource", "vehicle", "stage"],
                "edgeTypes": [
                    "precedence",
                    "eligible-mode",
                    "current-assignment",
                    "resource-order",
                    "blocking/starvation",
                    "choice-link",
                ],
                "features": [
                    "duration and remaining work",
                    "earliest/latest time and slack",
                    "criticality and bottleneck score",
                    "resource utilization and queue length",
                    "incumbent mode/order and change cost",
                ],
                "candidate": "heterogeneous message-passing GNN or graph transformer",
            },
            "trajectoryGraph": {
                "nodeTypes": ["waypoint", "route-segment", "vehicle-state", "conflict-event"],
                "edgeTypes": ["spatial-adjacency", "route-membership", "time-overlap", "collision-conflict"],
                "features": [
                    "position, length, curvature and direction",
                    "time-window occupancy and residual capacity",
                    "vehicle radius, speed and acceleration bounds",
                    "pairwise minimum distance and inserted delay",
                ],
                "candidate": "route-graph GNN or polyline transformer",
            },
            "fusion": "cross-attention between operation/vehicle assignments and route-segment occupancy",
        },
        "actionSpace": [
            "select one diagnosed bottleneck",
            "select one compatible method from the cost-aware shortlist",
            "select bounded neighborhood radius/released fraction",
            "select eligible machine, vehicle or route-column candidates",
            "select deterministic solver conflict/node budget",
            "request one evidence-backed Benders cut or causal expansion",
        ],
        "reward": {
            "primary": "validated lexicographic objective improvement",
            "efficiency": "improvement divided by deterministic solve plus Oracle cost",
            "stability": "penalize changed modes, sequence arcs, routes and start-time disruption",
            "hardPenalty": "reject infeasible, collision, frozen-region or route-continuity violations",
            "trainingRule": "score post-Oracle schedules, never pre-Oracle estimates",
        },
        "safetyShield": [
            "mask ineligible machines, vehicles, paths and actions",
            "freeze every decision outside the declared neighborhood",
            "repair every learned proposal with CP-SAT/MILP/CBS",
            "run generic validation and trajectory/collision Oracle",
            "retain incumbent unless the complete objective is strictly better",
        ],
    }


def build_joint_schedule_trajectory_plan(
    problem: Problem,
    incumbent: Schedule,
    *,
    settings: JointOptimizationSettings | None = None,
) -> dict[str, Any]:
    """Build an auditable master/subproblem plan for joint scheduling and routing."""

    settings = settings or JointOptimizationSettings()
    validation = validate_schedule(problem, incumbent)
    if not validation.feasible:
        raise ValueError(f"incumbent is infeasible: {validation.errors}")
    diagnostics = diagnose_bottlenecks(problem, incumbent)[:8]
    return {
        "architecture": "joint-schedule-trajectory-lbbd-v1",
        "problemId": problem.id,
        "family": problem.kind,
        "settings": asdict(settings),
        "incumbent": {
            "makespan": incumbent.makespan,
            "genericValid": validation.feasible,
            "preservation": "warm start and exact fallback",
        },
        "currentCarrierGap": {
            "observedImplementation": (
                "MAT trajectory is selected indirectly by job plus machine/spot binding; pairwise "
                "collision checking resolves conflicts mainly by 0.1-second start delays"
            ),
            "missingJointDecisions": [
                "explicit route-column choice and compatibility independent of opaque environment state",
                "same-binding spatial route alternatives",
                "vehicle-path-time co-assignment",
                "conflict precedence selection",
                "speed/acceleration-aware travel time",
            ],
        },
        "masterProblem": {
            "engine": "CP-SAT or MILP with incumbent hints and one deterministic worker",
            "variables": [
                "operation-machine assignment",
                "resource and launch sequence arcs",
                "tractor/vehicle assignment and continuity",
                "preparation spot, catapult and lane binding",
                "transport release/deadline windows",
                "candidate route-column choice",
            ],
            "constraints": [
                "job precedence and eligible modes",
                "machine, vehicle, preparation and catapult capacities",
                "choice-link and route-family consistency",
                "known travel-time lower bounds",
                "accumulated Benders conflict cuts",
                "bounded Hamming distance from the incumbent during local improvement",
            ],
            "objective": [
                "hard feasibility",
                "true launch/service objective",
                "joint makespan",
                "transport and collision delay",
                "schedule/route disruption",
            ],
            "boundRole": "provides a valid lower bound when every relaxed transport bound is valid",
        },
        "trajectorySubproblem": {
            "enginePortfolio": [
                "Conflict-Based Search for small discrete optimal MAPF subproblems",
                "time-expanded network ILP/CP-SAT for exact conflict-free routing",
                "safe-interval or shortest-path pricing for individual route columns",
                "continuous trajectory optimization only after a discrete conflict-free plan",
            ],
            "inputs": [
                "vehicle assignments",
                "origins/destinations and route candidates",
                "transport release/deadline windows",
                "vehicle geometry and kinematic limits",
                "deck graph and occupied space-time intervals",
            ],
            "outputs": [
                "conflict-free timed trajectories",
                "real transport durations and inserted waits",
                "feasibility certificate or minimal conflict explanation",
                "feasible upper bound for the joint objective",
            ],
        },
        "cutFamilies": [
            {
                "kind": "route-assignment-no-good",
                "meaning": "this machine/vehicle/route combination cannot meet its windows",
            },
            {
                "kind": "conflict-precedence-disjunction",
                "meaning": "one of two transports must clear a shared segment before the other enters",
            },
            {
                "kind": "minimum-separation",
                "meaning": "enforce an evidence-backed temporal offset for a conflicting pair",
            },
            {
                "kind": "travel-time-lower-bound",
                "meaning": "feed the shortest feasible route duration back into scheduling",
            },
            {
                "kind": "vehicle-continuity",
                "meaning": "exclude assignments that make a vehicle unreachable for its next task",
            },
            {
                "kind": "causal-closure-expansion",
                "meaning": "release only outside jobs proven to propagate a route/mode change",
            },
        ],
        "outerLoop": [
            "solve a bounded master candidate from the incumbent",
            "discard duplicate or lower-bound-dominated candidates",
            "solve the exact/controlled trajectory subproblem",
            "accept a feasible joint upper bound or extract a conflict cut",
            "update lower bound, upper bound and Cut cache",
            "stop at zero certified gap or the deterministic budget",
        ],
        "optimality": {
            "certifiableWhen": [
                "the scheduling master is solved exactly",
                "the routing subproblem is solved exactly",
                "space, time, vehicle dynamics and objectives use the declared finite model",
                "the global lower and feasible upper bounds meet",
            ],
            "claim": (
                "global optimum of the declared discretized joint model at zero gap; "
                "otherwise return best feasible solution, lower bound and certified gap"
            ),
            "realWorldLimit": (
                "continuous hydrodynamics, uncertain motion and omitted physics prevent an unconditional "
                "claim of globally optimal real-world deck operations"
            ),
        },
        "diagnostics": diagnostics,
        "agenticRL": build_agentic_rl_contract(
            problem,
            incumbent,
            validated_experiment_count=settings.validated_experiment_count,
            trajectory_observation_count=settings.trajectory_observation_count,
        ),
        "implementationRoadmap": [
            "version and convert MAT trajectories into a route/space-time graph",
            "add multiple legal route columns per transport instead of one fixed path",
            "build exact pairwise and multi-vehicle conflict explanations",
            "connect CP-SAT master to trajectory subproblem through versioned cuts",
            "prove the loop on small instances before training any neural encoder",
            "collect validated decisions and train imitation/bandit controller",
            "only then evaluate shielded Agentic RL against deterministic routing",
        ],
    }
