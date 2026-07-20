from __future__ import annotations

import argparse
import json
from pathlib import Path

from .carrier_alns import alns_priorities, build_alns_plan
from .carrier_alns_oracle import search_carrier_alns
from .carrier import DEFAULT_CARRIER_SCHEDULE, build_carrier_schedule, load_carrier_baseline
from .carrier_oracle import DEFAULT_LEGACY_ROOT, search_carrier_policy
from .carrier_oracle_cuts import extract_oracle_cuts
from .carrier_route_catalog import build_carrier_route_catalog
from .carrier_joint_cuts import build_reachability_causal_closure, extract_spacetime_replay_cut
from .carrier_spacetime import solve_fixed_route_spacetime
from .carrier_route_binding_master import search_route_binding_master
from .carrier_statistics import load_evidence, rank_operator_evidence
from .carrier_vns import build_vns_plan, load_tabu_signatures
from .carrier_vns_oracle import search_carrier_vns
from .examples import example_problems
from .fast_controller import adaptive_improve_incumbent, fast_improve_incumbent
from .generic_neighborhood import build_generic_neighborhood_plan, repair_generic_neighborhood
from .generic_statistics import load_generic_evidence, rank_generic_evidence
from .io import load_problem, load_schedule, save_schedule
from .improvement_workflow import build_improvement_workflow
from .joint_schedule_trajectory import (
    JointOptimizationSettings,
    build_joint_schedule_trajectory_plan,
    version_file_set,
)
from .metrics import bottleneck_advice, schedule_metrics
from .model import Schedule
from .portfolio import solve_portfolio
from .solvers import solve_dispatching
from .validation import validate_schedule
from .visualization import render_gantt


def _json_dump(payload: object, path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path is None:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path)


def command_benchmark(args: argparse.Namespace) -> None:
    output = Path(args.output)
    gantt_dir = Path(args.gantt_dir)
    results = {}
    for kind, problem in example_problems().items():
        portfolio = solve_portfolio(problem, time_limit=args.time_limit, seed=args.seed)
        render_gantt(problem, portfolio.best.schedule, gantt_dir / f"{kind}.png")
        results[kind] = {
            "best": portfolio.best.metrics,
            "solver": portfolio.best.schedule.metadata,
            "candidate_count": len(portfolio.candidates),
        }
    _json_dump(results, output)


def command_generic_neighborhood_benchmark(args: argparse.Namespace) -> None:
    results = {}
    gantt_dir = Path(args.gantt_dir)
    for kind, problem in example_problems().items():
        incumbent = solve_dispatching(problem, rule=args.baseline_rule)
        plan = build_generic_neighborhood_plan(
            problem,
            incumbent,
            max_bottlenecks=args.max_bottlenecks,
            radii=tuple(args.radii),
            include_pairs=args.include_pairs,
            max_pair_fraction=args.max_pair_fraction,
        )
        experiments = []
        best = incumbent
        for neighborhood in plan["neighborhoods"]:
            repair = repair_generic_neighborhood(
                problem,
                incumbent,
                neighborhood,
                seed=args.seed,
                deterministic_time=args.deterministic_time,
            )
            candidate = Schedule.model_validate(repair["schedule"]) if "schedule" in repair else None
            if candidate is not None and repair["accepted"] and candidate.makespan < best.makespan:
                best = candidate
            experiments.append(
                {
                    "neighborhoodIndex": neighborhood["index"],
                    "signature": neighborhood["signature"],
                    "bottleneckKind": neighborhood["bottleneck"]["kind"],
                    "moveDepth": neighborhood["moveDepth"],
                    "radius": neighborhood["radius"],
                    "releasedCount": neighborhood["releasedCount"],
                    "frozenCount": neighborhood["frozenCount"],
                    "status": repair["status"],
                    "accepted": repair["accepted"],
                    "candidateMakespan": None if candidate is None else candidate.makespan,
                }
            )
        render_gantt(problem, best, gantt_dir / f"{kind}.png", title=f"{kind.upper()} generic local repair")
        results[kind] = {
            "family": problem.kind,
            "baselineRule": args.baseline_rule,
            "baselineMakespan": incumbent.makespan,
            "bestMakespan": best.makespan,
            "improvement": incumbent.makespan - best.makespan,
            "firstBottleneck": plan["bottlenecks"][0],
            "evaluatedNeighborhoods": len(experiments),
            "acceptedNeighborhoods": sum(int(item["accepted"]) for item in experiments),
            "experiments": experiments,
        }
    _json_dump(results, Path(args.output))


def command_generic_neighborhood_plan(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    plan = build_generic_neighborhood_plan(
        problem,
        incumbent,
        max_bottlenecks=args.max_bottlenecks,
        radii=tuple(args.radii),
        include_pairs=args.include_pairs,
        max_pair_fraction=args.max_pair_fraction,
    )
    plan["source"] = {"problem": str(Path(args.problem).resolve()), "schedule": str(Path(args.schedule).resolve())}
    _json_dump(plan, Path(args.output))


def command_generic_neighborhood_repair(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    neighborhoods = plan.get("neighborhoods", [])
    if args.index < 0 or args.index >= len(neighborhoods):
        raise IndexError(f"neighborhood index {args.index} is outside 0..{len(neighborhoods) - 1}")
    neighborhood = neighborhoods[args.index]
    result = repair_generic_neighborhood(
        problem,
        incumbent,
        neighborhood,
        seed=args.seed,
        deterministic_time=args.deterministic_time,
        stability_weight=args.stability_weight,
    )
    result["neighborhood"] = neighborhood
    if "schedule" in result:
        candidate = Schedule.model_validate(result["schedule"])
        save_schedule(candidate, args.candidate_output)
        if args.gantt:
            render_gantt(problem, candidate, args.gantt, title="Generic bounded-neighborhood repair")
    _json_dump(result, Path(args.output))


def command_generic_evidence_rank(args: argparse.Namespace) -> None:
    observations = load_generic_evidence(args.history)
    payload = rank_generic_evidence(
        observations,
        pool_strength=args.pool_strength,
        exploration_weight=args.exploration_weight,
        release_penalty=args.release_penalty,
        retire_after_zero_improvements=args.retire_after_zero_improvements,
    )
    _json_dump(payload, Path(args.output))


def command_improvement_workflow(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    payload = build_improvement_workflow(
        problem,
        incumbent,
        evidence_count=args.evidence_count,
        validated_elite_count=args.validated_elite_count,
        no_improvement_count=args.no_improvement_count,
        oracle_failure_count=args.oracle_failure_count,
        speed_profile=args.speed_profile,
        max_structural_methods=args.max_structural_methods,
    )
    payload["source"] = {
        "problem": str(Path(args.problem).resolve()),
        "schedule": str(Path(args.schedule).resolve()),
    }
    _json_dump(payload, Path(args.output))


def command_improvement_workflow_benchmark(args: argparse.Namespace) -> None:
    payload = {}
    for kind, problem in example_problems().items():
        incumbent = solve_dispatching(problem, rule=args.baseline_rule)
        payload[kind] = build_improvement_workflow(
            problem,
            incumbent,
            evidence_count=args.evidence_count,
            validated_elite_count=args.validated_elite_count,
            no_improvement_count=args.no_improvement_count,
            oracle_failure_count=args.oracle_failure_count,
            speed_profile=args.speed_profile,
            max_structural_methods=args.max_structural_methods,
        )
    _json_dump(payload, Path(args.output))


def command_fast_improve(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    payload = fast_improve_incumbent(
        problem,
        incumbent,
        speed_profile=args.speed_profile,
        seed=args.seed,
        evidence_count=args.evidence_count,
        validated_elite_count=args.validated_elite_count,
        no_improvement_count=args.no_improvement_count,
        oracle_failure_count=args.oracle_failure_count,
        max_structural_methods=args.max_structural_methods,
    )
    _json_dump(payload, Path(args.output))


def command_fast_improvement_benchmark(args: argparse.Namespace) -> None:
    payload = {}
    for kind, problem in example_problems().items():
        incumbent = solve_dispatching(problem, rule=args.baseline_rule)
        payload[kind] = fast_improve_incumbent(
            problem,
            incumbent,
            speed_profile=args.speed_profile,
            seed=args.seed,
            evidence_count=args.evidence_count,
            no_improvement_count=args.no_improvement_count,
            max_structural_methods=args.max_structural_methods,
        )
    _json_dump(payload, Path(args.output))


def command_adaptive_improve(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    payload = adaptive_improve_incumbent(
        problem,
        incumbent,
        seed=args.seed,
        evidence_count=args.evidence_count,
        validated_elite_count=args.validated_elite_count,
        no_improvement_count=args.no_improvement_count,
        oracle_failure_count=args.oracle_failure_count,
    )
    _json_dump(payload, Path(args.output))


def command_adaptive_improvement_benchmark(args: argparse.Namespace) -> None:
    payload = {}
    for kind, problem in example_problems().items():
        incumbent = solve_dispatching(problem, rule=args.baseline_rule)
        payload[kind] = adaptive_improve_incumbent(
            problem,
            incumbent,
            seed=args.seed,
            evidence_count=args.evidence_count,
            no_improvement_count=args.no_improvement_count,
        )
    _json_dump(payload, Path(args.output))


def command_joint_schedule_trajectory_plan(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    incumbent = load_schedule(args.schedule)
    settings = JointOptimizationSettings(
        time_resolution=args.time_resolution,
        route_library_version=args.route_library_version,
        geometry_version=args.geometry_version,
        oracle_version=args.oracle_version,
        exact_target=args.exact_target,
        validated_experiment_count=args.validated_experiment_count,
        trajectory_observation_count=args.trajectory_observation_count,
    )
    payload = build_joint_schedule_trajectory_plan(problem, incumbent, settings=settings)
    _json_dump(payload, Path(args.output))


def command_carrier_joint_schedule_trajectory_plan(args: argparse.Namespace) -> None:
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    problem, incumbent = load_carrier_baseline(args.schedule)
    route_paths = [
        path
        for directory in (
            legacy_root / "initialtraject",
            legacy_root / "systemtraject",
            legacy_root / "trajectory",
        )
        for path in directory.glob("*.mat")
    ]
    geometry_paths = [legacy_root / "deck_video" / "video_viz.py"]
    oracle_paths = [
        legacy_root / "collision" / "Check.py",
        legacy_root / "collision" / "cal_delay.py",
        legacy_root / "Forcalcul.py",
        legacy_root / "FJSP_Env.py",
    ]
    settings = JointOptimizationSettings(
        time_resolution=args.time_resolution,
        route_library_version=version_file_set(route_paths, root=legacy_root),
        geometry_version=version_file_set(geometry_paths, root=legacy_root),
        oracle_version=version_file_set(oracle_paths, root=legacy_root),
        exact_target=args.exact_target,
        validated_experiment_count=args.validated_experiment_count,
        trajectory_observation_count=args.trajectory_observation_count,
    )
    payload = build_joint_schedule_trajectory_plan(problem, incumbent, settings=settings)
    payload["source"] = {
        "schedule": str(Path(args.schedule).expanduser().resolve()),
        "legacyRoot": str(legacy_root),
        "routeFiles": sum(
            1 for path in route_paths if path.is_file()
        ),
    }
    _json_dump(payload, Path(args.output))


def command_carrier_route_catalog(args: argparse.Namespace) -> None:
    payload = build_carrier_route_catalog(args.legacy_root, max_points=args.max_points)
    _json_dump(payload, Path(args.output))


def command_carrier_fixed_route_spacetime(args: argparse.Namespace) -> None:
    source = Path(args.schedule).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = solve_fixed_route_spacetime(
        payload,
        legacy_root=args.legacy_root,
        time_resolution=args.time_resolution,
        max_conflicts=args.max_conflicts,
        seed=args.seed,
    )
    result["source"] = {"schedule": str(source), "legacyRoot": str(Path(args.legacy_root).resolve())}
    _json_dump(result, Path(args.output))
    if result.get("candidate") is not None and args.candidate_output:
        _json_dump(
            {
                "meta": {
                    "controller": result["solver"],
                    "status": result["status"],
                    "provisional": True,
                    "trueMakespan": result["candidate"]["makespan"],
                },
                "schedule": result["candidate"]["schedule"],
                "decisionTrace": result["candidate"].get("sourceDecisionTrace"),
            },
            Path(args.candidate_output),
        )


def command_carrier_joint_replay_cut(args: argparse.Namespace) -> None:
    spacetime = json.loads(Path(args.spacetime).read_text(encoding="utf-8"))
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.joint_plan).read_text(encoding="utf-8"))
    cut = extract_spacetime_replay_cut(
        spacetime,
        replay,
        route_library_version=plan["settings"]["route_library_version"],
        oracle_version=plan["settings"]["oracle_version"],
    )
    _json_dump(cut, Path(args.output))


def command_carrier_joint_causal_closure(args: argparse.Namespace) -> None:
    schedule_payload = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    cut = json.loads(Path(args.cut).read_text(encoding="utf-8"))
    payload = build_reachability_causal_closure(
        schedule_payload["schedule"],
        cut,
        neighbor_radius=args.neighbor_radius,
        max_released_jobs=args.max_released_jobs,
    )
    _json_dump(payload, Path(args.output))


def command_carrier_route_binding_master(args: argparse.Namespace) -> None:
    schedule_payload = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    closure = json.loads(Path(args.closure).read_text(encoding="utf-8"))
    result = search_route_binding_master(
        schedule_payload,
        closure,
        legacy_root=args.legacy_root,
        time_resolution=args.time_resolution,
        max_jobs=args.max_jobs,
        max_candidates=args.max_candidates,
        max_conflicts=args.max_conflicts,
        seed=args.seed,
    )
    _json_dump(result, Path(args.output))
    if result.get("best") is not None and args.best_output:
        _json_dump(
            {
                "meta": {
                    "controller": result["controller"],
                    "proposalIndex": result["best"]["proposalIndex"],
                    "provisional": True,
                    "trueMakespan": result["best"]["candidateMakespan"],
                    "bindingChange": result["best"]["bindingChange"],
                },
                "schedule": result["best"]["candidateSchedule"],
                "decisionTrace": result["best"].get("sourceDecisionTrace"),
            },
            Path(args.best_output),
        )


def command_carrier_audit(args: argparse.Namespace) -> None:
    problem, schedule = load_carrier_baseline(args.schedule)
    validation = validate_schedule(problem, schedule)
    metrics = schedule_metrics(problem, schedule)
    payload = {
        "validation": validation.model_dump(mode="json"),
        "metrics": metrics,
        "reported_policy_makespan": problem.metadata["reported_policy_makespan"],
        "actual_makespan": metrics["makespan_display"],
        "difference": metrics["makespan_display"] - float(problem.metadata["reported_policy_makespan"]),
        "advice": bottleneck_advice(metrics),
    }
    if args.gantt:
        render_gantt(problem, schedule, args.gantt, title="Carrier baseline · true makespan audit")
    _json_dump(payload, Path(args.output) if args.output else None)


def command_solve(args: argparse.Namespace) -> None:
    problem = load_problem(args.problem)
    baseline = load_schedule(args.baseline) if args.baseline else None
    portfolio = solve_portfolio(problem, time_limit=args.time_limit, seed=args.seed, baseline=baseline, stability_weight=args.stability_weight)
    save_schedule(portfolio.best.schedule, args.output)
    if args.gantt:
        render_gantt(problem, portfolio.best.schedule, args.gantt)
    print(json.dumps(portfolio.best.metrics, ensure_ascii=False, indent=2))


def command_carrier_search(args: argparse.Namespace) -> None:
    payload = search_carrier_policy(
        rollouts=args.rollouts,
        seed=args.seed,
        device=args.device,
        legacy_root=args.legacy_root,
    )
    best = payload["best"]
    problem, candidate = build_carrier_schedule(
        best["schedule"],
        source=f"carrier oracle rollout {best['rollout']}",
        source_metadata={
            "network": payload["meta"]["network"],
            "checkpoint": payload["meta"]["checkpoint"],
            "reported_policy_makespan": best["reported_policy_makespan"],
        },
        schedule_metadata={
            "solver": best["strategy"],
            "rollout": best["rollout"],
            "seed": best["seed"],
        },
        domain_validated=best["domain_validated"],
    )
    generic_validation = validate_schedule(problem, candidate)
    candidate_metrics = schedule_metrics(problem, candidate)
    _, baseline = load_carrier_baseline(args.baseline)
    baseline_makespan = baseline.makespan / problem.time_scale
    payload["comparison"] = {
        "generic_validation": generic_validation.model_dump(mode="json"),
        "candidate_metrics": candidate_metrics,
        "baseline_true_makespan": baseline_makespan,
        "absolute_improvement": baseline_makespan - best["true_makespan"],
        "relative_improvement": (baseline_makespan - best["true_makespan"]) / baseline_makespan,
    }
    render_gantt(problem, candidate, args.gantt, title="Carrier candidate · domain-oracle validated")
    best_export = {
        "meta": {
            **payload["meta"],
            "rollout": best["rollout"],
            "rollout_seed": best["seed"],
            "strategy": best["strategy"],
            "policyMakespan": best["reported_policy_makespan"],
            "trueMakespan": best["true_makespan"],
            "domainValidated": best["domain_validated"],
            "genericValidated": generic_validation.feasible,
        },
        "schedule": best["schedule"],
    }
    _json_dump(best_export, Path(args.best_output))
    _json_dump(payload, Path(args.output))
    print(
        json.dumps(
            {
                "best_rollout": best["rollout"],
                "strategy": best["strategy"],
                "true_makespan": best["true_makespan"],
                "reported_policy_makespan": best["reported_policy_makespan"],
                "domain_validated": best["domain_validated"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_carrier_vns_plan(args: argparse.Namespace) -> None:
    source = Path(args.schedule).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_schedule = payload["schedule"] if isinstance(payload, dict) else payload
    plan = build_vns_plan(
        raw_schedule,
        radii=tuple(args.radii),
        max_gaps=args.max_gaps,
        tabu_signatures=load_tabu_signatures(args.tabu),
    )
    plan["source"] = str(source)
    _json_dump(plan, Path(args.output) if args.output else None)


def command_carrier_vns_search(args: argparse.Namespace) -> None:
    payload = search_carrier_vns(
        args.schedule,
        radii=tuple(args.radii),
        max_gaps=args.max_gaps,
        operators=tuple(args.operators),
        seed=args.seed,
        device=args.device,
        legacy_root=args.legacy_root,
    )
    best = payload.get("best")
    if best is not None:
        problem, candidate = build_carrier_schedule(
            best["schedule"],
            source=f"deterministic carrier VNS candidate {best['candidateIndex']}",
            source_metadata={
                "network": payload["meta"]["network"],
                "checkpoint": payload["meta"]["checkpoint"],
                "reported_policy_makespan": best["reportedPolicyMakespan"],
            },
            schedule_metadata={
                "solver": "deterministic-vns-domain-replay",
                "candidate_index": best["candidateIndex"],
                "neighborhood_signature": best["neighborhoodSignature"],
            },
            domain_validated=best["domainConstructed"],
        )
        validation = validate_schedule(problem, candidate)
        payload["genericValidation"] = validation.model_dump(mode="json")
        payload["candidateMetrics"] = schedule_metrics(problem, candidate)
        export = {
            "meta": {
                **payload["meta"],
                "candidateIndex": best["candidateIndex"],
                "neighborhoodSignature": best["neighborhoodSignature"],
                "trueMakespan": best["trueMakespan"],
                "domainConstructed": best["domainConstructed"],
                "genericValidated": validation.feasible,
            },
            "schedule": best["schedule"],
            "decisionTrace": best["decisionTrace"],
        }
        _json_dump(export, Path(args.best_output))
        if args.gantt:
            render_gantt(problem, candidate, args.gantt, title="Carrier deterministic VNS candidate")
    _json_dump(payload, Path(args.output))


def command_carrier_alns_plan(args: argparse.Namespace) -> None:
    source = Path(args.schedule).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_schedule = payload["schedule"] if isinstance(payload, dict) else payload
    plan = build_alns_plan(
        raw_schedule,
        destroy_sizes=tuple(args.destroy_sizes),
        radius=args.destroy_radius,
        max_gaps=args.max_gaps,
        max_jobs=args.max_jobs,
        gap_ranks=tuple(args.gap_ranks),
        expansion_jobs=tuple(args.expansion_jobs),
        tabu_signatures=load_tabu_signatures(args.tabu),
    )
    plan["source"] = str(source)
    _json_dump(plan, Path(args.output) if args.output else None)


def command_carrier_alns_search(args: argparse.Namespace) -> None:
    payload = search_carrier_alns(
        args.schedule,
        destroy_sizes=tuple(args.destroy_sizes),
        destroy_radius=args.destroy_radius,
        max_gaps=args.max_gaps,
        max_jobs=args.max_jobs,
        gap_ranks=tuple(args.gap_ranks),
        expansion_jobs=tuple(args.expansion_jobs),
        operators=tuple(args.operators),
        seed=args.seed,
        device=args.device,
        legacy_root=args.legacy_root,
        oracle_cuts=args.oracle_cuts,
    )
    best = payload.get("best")
    if best is not None:
        problem, candidate = build_carrier_schedule(
            best["schedule"],
            source=f"deterministic carrier ALNS candidate {best['candidateIndex']}",
            source_metadata={
                "network": payload["meta"]["network"],
                "checkpoint": payload["meta"]["checkpoint"],
                "reported_policy_makespan": best["reportedPolicyMakespan"],
            },
            schedule_metadata={
                "solver": "deterministic-alns-domain-replay",
                "candidate_index": best["candidateIndex"],
                "neighborhood_signature": best["neighborhoodSignature"],
            },
            domain_validated=best["domainConstructed"],
        )
        validation = validate_schedule(problem, candidate)
        payload["genericValidation"] = validation.model_dump(mode="json")
        payload["candidateMetrics"] = schedule_metrics(problem, candidate)
        export = {
            "meta": {
                **payload["meta"],
                "candidateIndex": best["candidateIndex"],
                "neighborhoodSignature": best["neighborhoodSignature"],
                "trueMakespan": best["trueMakespan"],
                "domainConstructed": best["domainConstructed"],
                "genericValidated": validation.feasible,
            },
            "schedule": best["schedule"],
            "decisionTrace": best["decisionTrace"],
        }
        _json_dump(export, Path(args.best_output))
        if args.gantt:
            render_gantt(problem, candidate, args.gantt, title="Carrier deterministic ALNS candidate")
    _json_dump(payload, Path(args.output))


def command_carrier_oracle_cuts(args: argparse.Namespace) -> None:
    payload = extract_oracle_cuts(args.history)
    _json_dump(payload, Path(args.output))


def command_carrier_evidence_rank(args: argparse.Namespace) -> None:
    observations = load_evidence(args.history)
    payload = rank_operator_evidence(
        observations,
        expansion_penalty=args.expansion_penalty,
        exploration_weight=args.exploration_weight,
        candidate_operators=tuple(args.candidate_operators),
        retire_after_zero_improvements=args.retire_after_zero_improvements,
    )
    _json_dump(payload, Path(args.output) if args.output else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constraint-safe scheduling laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="solve JSP/FSP/FJSP/HFSP regression examples")
    benchmark.add_argument("--time-limit", type=float, default=5.0)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output", default="outputs/benchmark.json")
    benchmark.add_argument("--gantt-dir", default="outputs/gantt")
    benchmark.set_defaults(func=command_benchmark)
    generic_benchmark = subparsers.add_parser(
        "generic-neighborhood-benchmark",
        help="validate incumbent-preserving local repair on JSP/FSP/FJSP/HFSP examples",
    )
    generic_benchmark.add_argument("--baseline-rule", default="lpt")
    generic_benchmark.add_argument("--max-bottlenecks", type=int, default=4)
    generic_benchmark.add_argument("--radii", type=int, nargs="+", default=[1, 2])
    generic_benchmark.add_argument("--include-pairs", action=argparse.BooleanOptionalAction, default=True)
    generic_benchmark.add_argument("--max-pair-fraction", type=float, default=0.8)
    generic_benchmark.add_argument("--deterministic-time", type=float, default=0.2)
    generic_benchmark.add_argument("--seed", type=int, default=0)
    generic_benchmark.add_argument("--output", default="outputs/generic_neighborhood_benchmark.json")
    generic_benchmark.add_argument("--gantt-dir", default="outputs/generic_neighborhood_gantt")
    generic_benchmark.set_defaults(func=command_generic_neighborhood_benchmark)
    generic_plan = subparsers.add_parser(
        "generic-neighborhood-plan",
        help="diagnose and serialize bounded neighborhoods for a canonical problem and incumbent",
    )
    generic_plan.add_argument("problem")
    generic_plan.add_argument("schedule")
    generic_plan.add_argument("--max-bottlenecks", type=int, default=4)
    generic_plan.add_argument("--radii", type=int, nargs="+", default=[1, 2])
    generic_plan.add_argument("--include-pairs", action=argparse.BooleanOptionalAction, default=True)
    generic_plan.add_argument("--max-pair-fraction", type=float, default=0.4)
    generic_plan.add_argument("--output", default="outputs/generic_neighborhood_plan.json")
    generic_plan.set_defaults(func=command_generic_neighborhood_plan)
    generic_repair = subparsers.add_parser(
        "generic-neighborhood-repair",
        help="repair one serialized neighborhood while freezing every outside assignment",
    )
    generic_repair.add_argument("problem")
    generic_repair.add_argument("schedule")
    generic_repair.add_argument("plan")
    generic_repair.add_argument("--index", type=int, default=0)
    generic_repair.add_argument("--seed", type=int, default=0)
    generic_repair.add_argument("--deterministic-time", type=float, default=1.0)
    generic_repair.add_argument("--stability-weight", type=int, default=1)
    generic_repair.add_argument("--output", default="outputs/generic_neighborhood_repair.json")
    generic_repair.add_argument("--candidate-output", default="outputs/generic_neighborhood_candidate.json")
    generic_repair.add_argument("--gantt", default="outputs/generic_neighborhood_candidate.png")
    generic_repair.set_defaults(func=command_generic_neighborhood_repair)
    generic_evidence = subparsers.add_parser(
        "generic-evidence-rank",
        help="rank neighborhood configurations with family posteriors and cross-family priors",
    )
    generic_evidence.add_argument("--history", nargs="+", required=True)
    generic_evidence.add_argument("--pool-strength", type=float, default=2.0)
    generic_evidence.add_argument("--exploration-weight", type=float, default=0.5)
    generic_evidence.add_argument("--release-penalty", type=float, default=0.1)
    generic_evidence.add_argument("--retire-after-zero-improvements", type=int, default=6)
    generic_evidence.add_argument("--output", default="outputs/generic_operator_posterior.json")
    generic_evidence.set_defaults(func=command_generic_evidence_rank)
    workflow = subparsers.add_parser(
        "improvement-workflow",
        help="build a deterministic multi-method incumbent-improvement workflow",
    )
    workflow.add_argument("problem")
    workflow.add_argument("schedule")
    workflow.add_argument("--evidence-count", type=int, default=0)
    workflow.add_argument("--validated-elite-count", type=int, default=0)
    workflow.add_argument("--no-improvement-count", type=int, default=0)
    workflow.add_argument("--oracle-failure-count", type=int, default=0)
    workflow.add_argument("--speed-profile", choices=["fast", "balanced", "thorough"], default="fast")
    workflow.add_argument("--max-structural-methods", type=int)
    workflow.add_argument("--output", default="outputs/improvement_workflow.json")
    workflow.set_defaults(func=command_improvement_workflow)
    workflow_benchmark = subparsers.add_parser(
        "improvement-workflow-benchmark",
        help="build deterministic method plans for JSP/FSP/FJSP/HFSP regression instances",
    )
    workflow_benchmark.add_argument("--baseline-rule", default="lpt")
    workflow_benchmark.add_argument("--evidence-count", type=int, default=12)
    workflow_benchmark.add_argument("--validated-elite-count", type=int, default=2)
    workflow_benchmark.add_argument("--no-improvement-count", type=int, default=4)
    workflow_benchmark.add_argument("--oracle-failure-count", type=int, default=2)
    workflow_benchmark.add_argument("--speed-profile", choices=["fast", "balanced", "thorough"], default="fast")
    workflow_benchmark.add_argument("--max-structural-methods", type=int)
    workflow_benchmark.add_argument("--output", default="outputs/multifamily_improvement_workflows.json")
    workflow_benchmark.set_defaults(func=command_improvement_workflow_benchmark)
    fast_improve = subparsers.add_parser(
        "fast-improve",
        help="run a cost-aware shortlist of bounded repairs around one incumbent",
    )
    fast_improve.add_argument("problem")
    fast_improve.add_argument("schedule")
    fast_improve.add_argument("--speed-profile", choices=["fast", "balanced", "thorough"], default="fast")
    fast_improve.add_argument("--max-structural-methods", type=int)
    fast_improve.add_argument("--seed", type=int, default=0)
    fast_improve.add_argument("--evidence-count", type=int, default=0)
    fast_improve.add_argument("--validated-elite-count", type=int, default=0)
    fast_improve.add_argument("--no-improvement-count", type=int, default=0)
    fast_improve.add_argument("--oracle-failure-count", type=int, default=0)
    fast_improve.add_argument("--output", default="outputs/fast_improvement.json")
    fast_improve.set_defaults(func=command_fast_improve)
    fast_benchmark = subparsers.add_parser(
        "fast-improvement-benchmark",
        help="run cost-aware fast improvement across JSP/FSP/FJSP/HFSP regressions",
    )
    fast_benchmark.add_argument("--baseline-rule", default="lpt")
    fast_benchmark.add_argument("--speed-profile", choices=["fast", "balanced", "thorough"], default="fast")
    fast_benchmark.add_argument("--max-structural-methods", type=int)
    fast_benchmark.add_argument("--seed", type=int, default=0)
    fast_benchmark.add_argument("--evidence-count", type=int, default=0)
    fast_benchmark.add_argument("--no-improvement-count", type=int, default=0)
    fast_benchmark.add_argument("--output", default="outputs/fast_improvement_benchmark.json")
    fast_benchmark.set_defaults(func=command_fast_improvement_benchmark)
    adaptive = subparsers.add_parser(
        "adaptive-improve",
        help="run fast shortlist first and escalate once to balanced only on failure",
    )
    adaptive.add_argument("problem")
    adaptive.add_argument("schedule")
    adaptive.add_argument("--seed", type=int, default=0)
    adaptive.add_argument("--evidence-count", type=int, default=0)
    adaptive.add_argument("--validated-elite-count", type=int, default=0)
    adaptive.add_argument("--no-improvement-count", type=int, default=0)
    adaptive.add_argument("--oracle-failure-count", type=int, default=0)
    adaptive.add_argument("--output", default="outputs/adaptive_improvement.json")
    adaptive.set_defaults(func=command_adaptive_improve)
    adaptive_benchmark = subparsers.add_parser(
        "adaptive-improvement-benchmark",
        help="benchmark fast-then-balanced escalation on JSP/FSP/FJSP/HFSP",
    )
    adaptive_benchmark.add_argument("--baseline-rule", default="lpt")
    adaptive_benchmark.add_argument("--seed", type=int, default=0)
    adaptive_benchmark.add_argument("--evidence-count", type=int, default=0)
    adaptive_benchmark.add_argument("--no-improvement-count", type=int, default=0)
    adaptive_benchmark.add_argument("--output", default="outputs/adaptive_improvement_benchmark.json")
    adaptive_benchmark.set_defaults(func=command_adaptive_improvement_benchmark)
    joint_plan = subparsers.add_parser(
        "joint-schedule-trajectory-plan",
        help="design a CP-SAT/LBBD scheduling and conflict-free trajectory optimization loop",
    )
    joint_plan.add_argument("problem")
    joint_plan.add_argument("schedule")
    joint_plan.add_argument("--time-resolution", type=float, default=0.1)
    joint_plan.add_argument("--route-library-version", default="unversioned")
    joint_plan.add_argument("--geometry-version", default="unversioned")
    joint_plan.add_argument("--oracle-version", default="legacy-fixed-trajectory-delay")
    joint_plan.add_argument("--exact-target", action=argparse.BooleanOptionalAction, default=True)
    joint_plan.add_argument("--validated-experiment-count", type=int, default=0)
    joint_plan.add_argument("--trajectory-observation-count", type=int, default=0)
    joint_plan.add_argument("--output", default="outputs/joint_schedule_trajectory_plan.json")
    joint_plan.set_defaults(func=command_joint_schedule_trajectory_plan)
    carrier_joint_plan = subparsers.add_parser(
        "carrier-joint-schedule-trajectory-plan",
        help="version the carrier MAT/geometry/Oracle files and design the exact joint loop",
    )
    carrier_joint_plan.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    carrier_joint_plan.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    carrier_joint_plan.add_argument("--time-resolution", type=float, default=0.1)
    carrier_joint_plan.add_argument("--exact-target", action=argparse.BooleanOptionalAction, default=True)
    carrier_joint_plan.add_argument("--validated-experiment-count", type=int, default=0)
    carrier_joint_plan.add_argument("--trajectory-observation-count", type=int, default=0)
    carrier_joint_plan.add_argument(
        "--output", default="outputs/carrier_joint_schedule_trajectory_plan.json"
    )
    carrier_joint_plan.set_defaults(func=command_carrier_joint_schedule_trajectory_plan)
    route_catalog = subparsers.add_parser(
        "carrier-route-catalog",
        help="normalize fixed MAT trajectories into versioned route-column metadata",
    )
    route_catalog.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    route_catalog.add_argument("--max-points", type=int, default=32)
    route_catalog.add_argument("--output", default="outputs/carrier_route_catalog.json")
    route_catalog.set_defaults(func=command_carrier_route_catalog)
    fixed_spacetime = subparsers.add_parser(
        "carrier-fixed-route-spacetime",
        help="exactly compact fixed MAT routes under finite-grid spatial conflicts",
    )
    fixed_spacetime.add_argument("--schedule", required=True)
    fixed_spacetime.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    fixed_spacetime.add_argument("--time-resolution", type=float, default=0.1)
    fixed_spacetime.add_argument("--max-conflicts", type=int, default=200_000)
    fixed_spacetime.add_argument("--seed", type=int, default=0)
    fixed_spacetime.add_argument("--output", default="outputs/carrier_fixed_route_spacetime.json")
    fixed_spacetime.add_argument(
        "--candidate-output", default="outputs/carrier_fixed_route_spacetime_candidate.json"
    )
    fixed_spacetime.set_defaults(func=command_carrier_fixed_route_spacetime)
    replay_cut = subparsers.add_parser(
        "carrier-joint-replay-cut",
        help="turn a failed exact-grid candidate domain replay into a context-exact LBBD cut",
    )
    replay_cut.add_argument("--spacetime", required=True)
    replay_cut.add_argument("--replay", required=True)
    replay_cut.add_argument("--joint-plan", required=True)
    replay_cut.add_argument("--output", default="outputs/carrier_joint_replay_cut.json")
    replay_cut.set_defaults(func=command_carrier_joint_replay_cut)
    joint_closure = subparsers.add_parser(
        "carrier-joint-causal-closure",
        help="expand a reachability Cut through directly coupled carrier resource orders",
    )
    joint_closure.add_argument("--schedule", required=True)
    joint_closure.add_argument("--cut", required=True)
    joint_closure.add_argument("--neighbor-radius", type=int, default=1)
    joint_closure.add_argument("--max-released-jobs", type=int, default=14)
    joint_closure.add_argument("--output", default="outputs/carrier_joint_causal_closure.json")
    joint_closure.set_defaults(func=command_carrier_joint_causal_closure)
    binding_master = subparsers.add_parser(
        "carrier-route-binding-master",
        help="search a tiny explicit preparation/catapult/MAT-route binding neighborhood",
    )
    binding_master.add_argument("--schedule", required=True)
    binding_master.add_argument("--closure", required=True)
    binding_master.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    binding_master.add_argument("--time-resolution", type=float, default=0.1)
    binding_master.add_argument("--max-jobs", type=int, default=2)
    binding_master.add_argument("--max-candidates", type=int, default=4)
    binding_master.add_argument("--max-conflicts", type=int, default=500_000)
    binding_master.add_argument("--seed", type=int, default=0)
    binding_master.add_argument("--output", default="outputs/carrier_route_binding_master.json")
    binding_master.add_argument("--best-output", default="outputs/carrier_route_binding_best.json")
    binding_master.set_defaults(func=command_carrier_route_binding_master)
    carrier = subparsers.add_parser("carrier-audit", help="audit the current 20-aircraft baseline")
    carrier.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    carrier.add_argument("--output", default="outputs/carrier_audit.json")
    carrier.add_argument("--gantt", default="outputs/carrier_baseline.png")
    carrier.set_defaults(func=command_carrier_audit)
    search = subparsers.add_parser(
        "carrier-search",
        help="sample the legacy PPO policy through the trajectory/collision domain oracle",
    )
    search.add_argument("--rollouts", type=int, default=8)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--device", default="cpu")
    search.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    search.add_argument("--baseline", default=str(DEFAULT_CARRIER_SCHEDULE))
    search.add_argument("--output", default="outputs/carrier_search.json")
    search.add_argument("--best-output", default="outputs/carrier_best_schedule.json")
    search.add_argument("--gantt", default="outputs/carrier_best_schedule.png")
    search.set_defaults(func=command_carrier_search)
    vns = subparsers.add_parser(
        "carrier-vns-plan",
        help="build deterministic O7 gap neighborhoods around an existing carrier schedule",
    )
    vns.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    vns.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4])
    vns.add_argument("--max-gaps", type=int, default=5)
    vns.add_argument("--tabu")
    vns.add_argument("--output", default="outputs/carrier_vns_plan.json")
    vns.set_defaults(func=command_carrier_vns_plan)
    vns_search = subparsers.add_parser(
        "carrier-vns-search",
        help="replay deterministic O7 neighborhoods through the legacy trajectory/collision environment",
    )
    vns_search.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    vns_search.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4])
    vns_search.add_argument("--max-gaps", type=int, default=1)
    vns_search.add_argument(
        "--operators",
        nargs="+",
        default=["adjacent-boundary-swap", "forward-insertion", "backward-insertion"],
    )
    vns_search.add_argument("--seed", type=int, default=0)
    vns_search.add_argument("--device", default="cpu")
    vns_search.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    vns_search.add_argument("--output", default="outputs/carrier_vns_search.json")
    vns_search.add_argument("--best-output", default="outputs/carrier_vns_best_schedule.json")
    vns_search.add_argument("--gantt", default="outputs/carrier_vns_best_schedule.png")
    vns_search.set_defaults(func=command_carrier_vns_search)
    alns_plan = subparsers.add_parser(
        "carrier-alns-plan",
        help="build deterministic multi-gap ALNS destroy neighborhoods",
    )
    alns_plan.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    alns_plan.add_argument("--destroy-sizes", type=int, nargs="+", default=[2, 3])
    alns_plan.add_argument("--destroy-radius", type=int, default=2)
    alns_plan.add_argument("--max-gaps", type=int, default=4)
    alns_plan.add_argument("--max-jobs", type=int, default=14)
    alns_plan.add_argument("--gap-ranks", type=int, nargs="*", default=[])
    alns_plan.add_argument("--expansion-jobs", type=int, nargs="*", default=[])
    alns_plan.add_argument("--tabu")
    alns_plan.add_argument("--output", default="outputs/carrier_alns_plan.json")
    alns_plan.set_defaults(func=command_carrier_alns_plan)
    alns_search = subparsers.add_parser(
        "carrier-alns-search",
        help="repair deterministic multi-gap neighborhoods through the carrier domain oracle",
    )
    alns_search.add_argument("--schedule", default=str(DEFAULT_CARRIER_SCHEDULE))
    alns_search.add_argument("--destroy-sizes", type=int, nargs="+", default=[2, 3])
    alns_search.add_argument("--destroy-radius", type=int, default=2)
    alns_search.add_argument("--max-gaps", type=int, default=4)
    alns_search.add_argument("--max-jobs", type=int, default=14)
    alns_search.add_argument("--gap-ranks", type=int, nargs="*", default=[])
    alns_search.add_argument("--expansion-jobs", type=int, nargs="*", default=[])
    alns_search.add_argument(
        "--operators",
        nargs="+",
        default=[
            "alns-incumbent-replay",
            "alns-adjacent-o6",
            "alns-adjacent-o5",
            "alns-adjacent-o4",
            "cp-sat-fixed-mode",
        ],
    )
    alns_search.add_argument("--seed", type=int, default=0)
    alns_search.add_argument("--device", default="cpu")
    alns_search.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT))
    alns_search.add_argument("--oracle-cuts")
    alns_search.add_argument("--output", default="outputs/carrier_alns_search.json")
    alns_search.add_argument("--best-output", default="outputs/carrier_alns_best_schedule.json")
    alns_search.add_argument("--gantt", default="outputs/carrier_alns_best_schedule.png")
    alns_search.set_defaults(func=command_carrier_alns_search)
    oracle_cuts = subparsers.add_parser(
        "carrier-oracle-cuts",
        help="extract exact deterministic no-good cuts from carrier Oracle histories",
    )
    oracle_cuts.add_argument("--history", nargs="+", required=True)
    oracle_cuts.add_argument("--output", default="outputs/carrier_oracle_cuts.json")
    oracle_cuts.set_defaults(func=command_carrier_oracle_cuts)
    evidence = subparsers.add_parser(
        "carrier-evidence-rank",
        help="rank deterministic ALNS operators from deduplicated domain-Oracle evidence",
    )
    evidence.add_argument("--history", nargs="+", required=True)
    evidence.add_argument("--expansion-penalty", type=float, default=0.15)
    evidence.add_argument("--exploration-weight", type=float, default=1.0)
    evidence.add_argument("--candidate-operators", nargs="*", default=["alns-adjacent-o3"])
    evidence.add_argument("--retire-after-zero-improvements", type=int, default=4)
    evidence.add_argument("--output", default="outputs/carrier_operator_posterior.json")
    evidence.set_defaults(func=command_carrier_evidence_rank)
    solve = subparsers.add_parser("solve", help="solve a serialized generic problem")
    solve.add_argument("problem")
    solve.add_argument("--baseline")
    solve.add_argument("--time-limit", type=float, default=10.0)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--stability-weight", type=int, default=0)
    solve.add_argument("--output", default="outputs/best_schedule.json")
    solve.add_argument("--gantt")
    solve.set_defaults(func=command_solve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
