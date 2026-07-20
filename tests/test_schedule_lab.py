from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedule_lab.carrier import DEFAULT_CARRIER_SCHEDULE, load_carrier_baseline
from schedule_lab.carrier_alns import alns_dispatch_order, alns_priorities, build_alns_plan
from schedule_lab.carrier_local_cp import solve_fixed_mode_neighborhood
from schedule_lab.carrier_oracle_cuts import (
    extract_oracle_cuts,
    matching_exact_cuts,
    proposal_key,
)
from schedule_lab.carrier_route_catalog import read_route_column
from schedule_lab.carrier_joint_cuts import (
    build_reachability_causal_closure,
    extract_spacetime_replay_cut,
)
from schedule_lab.carrier_spacetime import solve_fixed_route_spacetime
from schedule_lab.carrier_statistics import rank_operator_evidence
from schedule_lab.carrier_vns import build_vns_plan, schedule_hash
from schedule_lab.carrier_vns_worker import priority_for_operator
from schedule_lab.examples import example_problems
from schedule_lab.fast_controller import adaptive_improve_incumbent, fast_improve_incumbent
from schedule_lab.generic_neighborhood import build_generic_neighborhood_plan, repair_generic_neighborhood
from schedule_lab.generic_statistics import rank_generic_evidence
from schedule_lab.improvement_workflow import build_improvement_workflow, strategy_catalog
from schedule_lab.joint_schedule_trajectory import (
    JointOptimizationSettings,
    build_joint_schedule_trajectory_plan,
    version_file_set,
)
from schedule_lab.metrics import schedule_metrics
from schedule_lab.model import Assignment, Schedule
from schedule_lab.portfolio import solve_portfolio
from schedule_lab.strategy_router import FAMILY_PACKS, SPEED_PROFILES
from schedule_lab.validation import validate_schedule


class ScheduleLabTests(unittest.TestCase):
    def test_fixed_route_spacetime_candidate_is_exact_but_remains_provisional(self) -> None:
        import json

        source = ROOT / "outputs" / "carrier_alns_best_iter3_gap6_closed_630_5.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        result = solve_fixed_route_spacetime(
            payload,
            legacy_root=ROOT.parent,
            time_resolution=0.1,
            max_conflicts=500_000,
            seed=0,
        )
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["incumbent"]["hash"], "e8e67033c5dd257c830471b279fa384fb65f6562db791a147aff66e9714c9cf5")
        self.assertTrue(result["candidate"]["genericValid"])
        self.assertTrue(result["candidate"]["exactGridValid"])
        self.assertTrue(result["candidate"]["provisional"])
        self.assertAlmostEqual(result["candidate"]["makespan"], 619.5, places=6)

    def test_failed_domain_replay_becomes_context_exact_reachability_cut(self) -> None:
        spacetime = {
            "settings": {"timeResolution": 0.1},
            "candidate": {"hash": "candidate-a"},
        }
        replay = {
            "targetHash": "target-a",
            "deterministic": True,
            "best": {
                "domainConstructed": True,
                "domainErrors": [],
                "trueMakespan": 700.0,
                "unavailableTargetMachines": [{"job": 3, "op": 2, "machine": 6}],
                "modeChanges": [{"job": 3, "op": 2, "expected": 6, "actual": 7}],
            },
        }
        cut = extract_spacetime_replay_cut(
            spacetime,
            replay,
            route_library_version="routes-a",
            oracle_version="oracle-a",
        )
        self.assertFalse(cut["feasible"])
        self.assertEqual(cut["affectedJobs"], [3])
        self.assertTrue(cut["validOnlyForExactContext"])
        self.assertIn("forbid this exact", cut["masterAction"])

        raw = [
            {
                "job": job,
                "op": operation,
                "machine": operation if operation in (0, 2, 4, 6) else operation - 1,
                "start": job * 100 + operation * 10,
                "dur": 10,
                "end": job * 100 + operation * 10 + 10,
            }
            for job in range(5)
            for operation in range(8)
        ]
        closure = build_reachability_causal_closure(raw, cut, neighbor_radius=1)
        self.assertIn(3, closure["affectedJobs"])
        self.assertGreaterEqual(len(closure["releasedJobs"]), 1)
        self.assertEqual(
            closure["frozenOperationCount"], len(closure["frozenJobs"]) * 8
        )

    def test_legacy_routes_normalize_single_and_coupled_motion(self) -> None:
        single_path = ROOT.parent / "initialtraject" / "M7J5.mat"
        coupled_path = ROOT.parent / "systemtraject" / "J4M5.mat"
        single = read_route_column(single_path, phase="initial", max_points=12)
        coupled = read_route_column(coupled_path, phase="towing", max_points=12)
        self.assertEqual((single["job"], single["machine"]), (5, 7))
        self.assertEqual(len(single["actors"]), 1)
        self.assertEqual({actor["actor"] for actor in coupled["actors"]}, {"tractor", "aircraft"})
        self.assertTrue(all(len(actor["points"]) <= 12 for actor in coupled["actors"]))
        self.assertGreater(single["duration"], 0)
        first = version_file_set([single_path, coupled_path], root=ROOT.parent)
        second = version_file_set([coupled_path, single_path], root=ROOT.parent)
        self.assertEqual(first, second)

    def test_adaptive_controller_escalates_without_repeating_signatures(self) -> None:
        from schedule_lab.solvers import solve_dispatching

        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                incumbent = solve_dispatching(problem, rule="lpt")
                result = adaptive_improve_incumbent(problem, incumbent, seed=0)
                signatures = [
                    experiment["signature"]
                    for stage in result["stages"]
                    for experiment in stage["experiments"]
                ]
                self.assertEqual(len(signatures), len(set(signatures)))
                self.assertLessEqual(result["result"]["makespan"], incumbent.makespan)
                self.assertIn(result["stageCount"], {1, 2})
                candidate = Schedule.model_validate(result["result"]["schedule"])
                self.assertTrue(validate_schedule(problem, candidate).feasible)

    def test_joint_plan_keeps_rl_outside_feasibility_and_uses_encoder_only_with_data(self) -> None:
        problem = example_problems()["fjsp"]
        incumbent = solve_portfolio(problem, time_limit=1.0, seed=0).best.schedule
        sparse = build_joint_schedule_trajectory_plan(problem, incumbent)
        learned = build_joint_schedule_trajectory_plan(
            problem,
            incumbent,
            settings=JointOptimizationSettings(
                validated_experiment_count=500,
                trajectory_observation_count=500,
            ),
        )
        self.assertFalse(sparse["agenticRL"]["encoder"]["recommendedNow"])
        self.assertTrue(learned["agenticRL"]["encoder"]["recommendedNow"])
        self.assertIn("never certify feasibility", sparse["agenticRL"]["role"])
        self.assertIn("Conflict-Based Search", learned["trajectorySubproblem"]["enginePortfolio"][0])
        self.assertGreaterEqual(len(learned["cutFamilies"]), 5)
        self.assertIn("zero gap", learned["optimality"]["claim"])

    def test_fast_controller_uses_small_shortlist_and_never_degrades_incumbent(self) -> None:
        from schedule_lab.solvers import solve_dispatching

        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                incumbent = solve_dispatching(problem, rule="lpt")
                first = fast_improve_incumbent(problem, incumbent, speed_profile="fast", seed=0)
                second = fast_improve_incumbent(problem, incumbent, speed_profile="fast", seed=0)
                self.assertEqual(first, second)
                self.assertLessEqual(len(first["shortlist"]), 3)
                self.assertLessEqual(len(first["experiments"]), 4)
                self.assertLessEqual(first["result"]["makespan"], incumbent.makespan)
                result_schedule = Schedule.model_validate(first["result"]["schedule"])
                self.assertTrue(validate_schedule(problem, result_schedule).feasible)
                if not first["result"]["accepted"]:
                    self.assertEqual(first["result"]["hash"], first["incumbent"]["hash"])

    def test_speed_profiles_bound_methods_candidates_and_oracle_calls(self) -> None:
        self.assertLess(
            SPEED_PROFILES["fast"]["maxCandidates"],
            SPEED_PROFILES["balanced"]["maxCandidates"],
        )
        self.assertLess(
            SPEED_PROFILES["balanced"]["maxCandidates"],
            SPEED_PROFILES["thorough"]["maxCandidates"],
        )
        self.assertEqual(SPEED_PROFILES["fast"]["maxOracleCandidates"], 1)

    def test_agent_router_is_deterministic_family_compatible_and_monotone(self) -> None:
        catalog = strategy_catalog()
        expected_primary = {
            "jsp": {"shifting-bottleneck-decomposition", "critical-block-vns"},
            "fsp": {"shifting-bottleneck-decomposition", "constructive-heuristic-proposals"},
            "fjsp": {"assignment-sequence-alternation", "dual-price-guided-release"},
            "hfsp": {"shifting-bottleneck-decomposition", "assignment-sequence-alternation"},
        }
        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                incumbent = solve_portfolio(problem, time_limit=1.0, seed=0).best.schedule
                first = build_improvement_workflow(
                    problem,
                    incumbent,
                    evidence_count=12,
                    validated_elite_count=2,
                    no_improvement_count=4,
                    oracle_failure_count=2,
                )
                second = build_improvement_workflow(
                    problem,
                    incumbent,
                    evidence_count=12,
                    validated_elite_count=2,
                    no_improvement_count=4,
                    oracle_failure_count=2,
                )
                self.assertEqual(first["agentDiagnosis"], second["agentDiagnosis"])
                self.assertEqual(first["strategyRouting"], second["strategyRouting"])
                routing = first["strategyRouting"]
                self.assertEqual(routing["familyPack"]["family"], problem.kind)
                self.assertTrue(
                    expected_primary[kind].issubset(set(routing["selectedStructuralMethods"]))
                )
                self.assertIn(
                    "deterministic-alns",
                    {item["strategyId"] for item in routing["rankedStrategies"]},
                )
                for item in routing["rankedStrategies"]:
                    self.assertIn(problem.kind, catalog[item["strategyId"]].families)
                    self.assertTrue(item["reasons"])
                self.assertEqual(routing["fallback"]["schedule"], "retain incumbent")
                self.assertIn(
                    "no lexicographically worse candidate replaces the incumbent",
                    routing["fallback"]["guarantees"],
                )

    def test_every_supported_family_has_multiple_strategy_lanes(self) -> None:
        self.assertEqual(set(FAMILY_PACKS), {"JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"})
        for family, pack in FAMILY_PACKS.items():
            with self.subTest(family=family):
                self.assertGreaterEqual(len(pack.diagnostic_focus), 3)
                self.assertGreaterEqual(len(pack.primary), 2)
                self.assertGreaterEqual(len(pack.exact_repair), 1)
                self.assertGreaterEqual(len(pack.plateau), 2)

    def test_oracle_cuts_are_exact_deterministic_and_do_not_block_acceptance(self) -> None:
        import json
        import tempfile

        incumbent_hash = "incumbent-a"
        rejected = {
            "candidateIndex": 0,
            "neighborhoodSignature": "neighborhood-a",
            "operator": "alns-adjacent-o4",
            "dispatchSwaps": [{"operation": 3, "leftJob": 1, "rightJob": 2}],
            "scheduleHash": "candidate-a",
            "domainConstructed": True,
            "domainErrors": [],
            "requiredExpansionJobs": [8, 5, 8],
            "propagationClosed": False,
            "trueMakespan": 630.0,
            "accepted": False,
        }
        accepted = {
            **rejected,
            "candidateIndex": 1,
            "neighborhoodSignature": "neighborhood-b",
            "scheduleHash": "candidate-b",
            "requiredExpansionJobs": [],
            "propagationClosed": True,
            "trueMakespan": 620.0,
            "accepted": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.json"
            history.write_text(
                json.dumps(
                    {
                        "meta": {"seed": 0},
                        "incumbent": {"scheduleHash": incumbent_hash},
                        "candidates": [rejected, accepted],
                    }
                ),
                encoding="utf-8",
            )
            store = extract_oracle_cuts([history])

        self.assertEqual(store["statistics"]["observations"], 2)
        self.assertEqual(store["statistics"]["uniqueCuts"], 1)
        self.assertEqual(store["cuts"][0]["requiredExpansionJobs"], [5, 8])
        rejected_key = proposal_key(
            incumbent_hash=incumbent_hash,
            neighborhood_signature="neighborhood-a",
            operator="alns-adjacent-o4",
            dispatch_swaps=rejected["dispatchSwaps"],
            seed=0,
        )
        self.assertEqual(len(matching_exact_cuts(store, rejected_key)), 1)
        changed_seed_key = proposal_key(
            incumbent_hash=incumbent_hash,
            neighborhood_signature="neighborhood-a",
            operator="alns-adjacent-o4",
            dispatch_swaps=rejected["dispatchSwaps"],
            seed=1,
        )
        self.assertFalse(matching_exact_cuts(store, changed_seed_key))

    def test_improvement_workflow_is_stable_and_family_aware(self) -> None:
        expected = {
            "jsp": "shifting-bottleneck-decomposition",
            "fsp": "constructive-heuristic-proposals",
            "fjsp": "dual-price-guided-release",
            "hfsp": "assignment-sequence-alternation",
        }
        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                incumbent = solve_portfolio(problem, time_limit=1.0, seed=0).best.schedule
                first = build_improvement_workflow(problem, incumbent)
                second = build_improvement_workflow(problem, incumbent)
                self.assertEqual(first, second)
                strategy_ids = {
                    strategy["id"]
                    for phase in first["phases"]
                    for strategy in phase["strategies"]
                }
                self.assertIn("sequence-preserving-compaction", strategy_ids)
                self.assertIn(expected[kind], strategy_ids)
                self.assertNotIn("bayesian-budget-allocation", strategy_ids)
                self.assertEqual(first["controller"]["speedProfile"], "fast")
                self.assertLessEqual(
                    len(first["strategyRouting"]["selectedStructuralMethods"]), 3
                )
                self.assertIn(
                    "causal-closure-fix-and-optimize",
                    strategy_catalog(),
                )
                contingent_ids = {
                    strategy["id"] for strategy in first["contingentStrategies"]
                }
                self.assertIn("robust-scenario-polishing", contingent_ids)
                if kind in {"fjsp", "hfsp"}:
                    self.assertIn("logic-based-benders-oracle-cuts", contingent_ids)
                if kind in {"jsp", "fsp", "hfsp"}:
                    self.assertIn("critical-path-decision-diagram", contingent_ids)

    def test_improvement_workflow_enables_evidence_and_elite_layers_only_when_ready(self) -> None:
        problem = example_problems()["fjsp"]
        incumbent = solve_portfolio(problem, time_limit=1.0, seed=0).best.schedule
        workflow = build_improvement_workflow(
            problem,
            incumbent,
            evidence_count=12,
            validated_elite_count=2,
            speed_profile="balanced",
        )
        strategy_ids = {
            strategy["id"]
            for phase in workflow["phases"]
            for strategy in phase["strategies"]
        }
        self.assertIn("elite-path-relinking", strategy_ids)
        self.assertIn("bayesian-budget-allocation", strategy_ids)

    def test_all_problem_families_solve_feasibly(self) -> None:
        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                result = solve_portfolio(problem, time_limit=2.0, seed=7)
                self.assertTrue(result.best.validation.feasible)
                self.assertGreater(result.best.schedule.makespan, 0)
                self.assertGreaterEqual(len(result.candidates), 2)

    def test_classic_jsp_reaches_known_optimum(self) -> None:
        problem = example_problems()["jsp"]
        result = solve_portfolio(problem, time_limit=3.0, seed=3)
        self.assertEqual(result.best.schedule.makespan, 11)

    def test_dispatching_tie_break_is_stable(self) -> None:
        from schedule_lab.solvers import solve_dispatching

        for problem in example_problems().values():
            first = solve_dispatching(problem, rule="balanced")
            second = solve_dispatching(problem, rule="balanced")
            self.assertEqual(first, second)

    def test_domain_problem_local_improvement_remains_provisional(self) -> None:
        from schedule_lab.solvers import solve_dispatching

        base_problem = example_problems()["jsp"]
        problem = base_problem.model_copy(
            update={"metadata": {"requires_domain_validation": True}}
        )
        incumbent = solve_dispatching(problem, rule="lpt")
        plan = build_generic_neighborhood_plan(
            problem,
            incumbent,
            max_bottlenecks=4,
            radii=(1, 2),
            include_pairs=True,
            max_pair_fraction=1.0,
        )
        improving = []
        for neighborhood in plan["neighborhoods"]:
            repair = repair_generic_neighborhood(problem, incumbent, neighborhood, deterministic_time=0.2)
            if "candidateMetrics" in repair and repair["candidateMetrics"]["makespan"] < incumbent.makespan:
                improving.append(repair)
        self.assertTrue(improving)
        self.assertTrue(all(item["provisional"] for item in improving))
        self.assertTrue(all(not item["accepted"] for item in improving))
        self.assertTrue(all(item["reason"] == "requires domain Oracle" for item in improving))

    def test_generic_neighborhood_repair_preserves_frozen_region_for_all_families(self) -> None:
        for kind, problem in example_problems().items():
            with self.subTest(kind=kind):
                incumbent = solve_portfolio(problem, time_limit=1.0, seed=0).best.schedule
                plan = build_generic_neighborhood_plan(
                    problem,
                    incumbent,
                    max_bottlenecks=2,
                    radii=(1, 2),
                    max_pair_fraction=1.0,
                )
                self.assertEqual(plan["family"].lower(), kind)
                self.assertTrue(plan["neighborhoods"])
                self.assertIn(1, {item["moveDepth"] for item in plan["neighborhoods"]})
                self.assertIn(2, {item["moveDepth"] for item in plan["neighborhoods"]})
                neighborhood = plan["neighborhoods"][0]
                repair = repair_generic_neighborhood(problem, incumbent, neighborhood, deterministic_time=0.2)
                self.assertIn(repair["status"], {"OPTIMAL", "FEASIBLE"})
                candidate = Schedule.model_validate(repair["schedule"])
                self.assertTrue(validate_schedule(problem, candidate).feasible)
                incumbent_map = incumbent.assignment_map()
                candidate_map = candidate.assignment_map()
                for operation_id in neighborhood["frozenOperations"]:
                    self.assertEqual(candidate_map[operation_id], incumbent_map[operation_id])

    def test_validator_rejects_resource_overlap(self) -> None:
        problem = example_problems()["jsp"]
        valid = solve_portfolio(problem, time_limit=1.0).best.schedule
        assignments = list(valid.assignments)
        first = assignments[0]
        conflicting_index = next(
            index
            for index, assignment in enumerate(assignments[1:], start=1)
            if problem.mode_map()[assignment.mode_id][1].resources == problem.mode_map()[first.mode_id][1].resources
        )
        conflicting = assignments[conflicting_index]
        duration = conflicting.end - conflicting.start
        assignments[conflicting_index] = Assignment(operation_id=conflicting.operation_id, mode_id=conflicting.mode_id, start=first.start, end=first.start + duration)
        broken = Schedule(problem_id=problem.id, assignments=tuple(assignments))
        result = validate_schedule(problem, broken)
        self.assertFalse(result.feasible)
        self.assertTrue(any(issue.code == "resource_capacity" for issue in result.errors))

    @unittest.skipUnless(DEFAULT_CARRIER_SCHEDULE.exists(), "local carrier schedule is unavailable")
    def test_carrier_audit_uses_true_completion_makespan(self) -> None:
        problem, schedule = load_carrier_baseline()
        validation = validate_schedule(problem, schedule)
        metrics = schedule_metrics(problem, schedule)
        self.assertTrue(validation.feasible, validation.issues)
        self.assertEqual(len(schedule.assignments), 160)
        self.assertAlmostEqual(problem.metadata["reported_policy_makespan"], 655.500003)
        self.assertAlmostEqual(metrics["makespan_display"], 675.5)
        self.assertAlmostEqual(metrics["makespan_display"] - problem.metadata["reported_policy_makespan"], 19.999997)

    @unittest.skipUnless(DEFAULT_CARRIER_SCHEDULE.exists(), "local carrier schedule is unavailable")
    def test_carrier_vns_plan_is_deterministic_and_gap_driven(self) -> None:
        import json

        payload = json.loads(DEFAULT_CARRIER_SCHEDULE.read_text(encoding="utf-8"))
        raw_schedule = payload["schedule"]
        first = build_vns_plan(raw_schedule, radii=(2, 3, 4), max_gaps=2)
        second = build_vns_plan(raw_schedule, radii=(2, 3, 4), max_gaps=2)
        self.assertEqual(first, second)
        self.assertEqual(first["incumbentHash"], schedule_hash(raw_schedule))
        self.assertAlmostEqual(first["incumbentMakespan"], 675.5)
        self.assertAlmostEqual(first["focus"]["largestGap"]["gap"], 50.0, places=3)
        self.assertEqual([item["radius"] for item in first["neighborhoods"][:3]], [2, 3, 4])
        self.assertEqual(first["neighborhoods"][0]["releasedOperationCount"], 32)
        signatures = [item["signature"] for item in first["neighborhoods"]]
        self.assertEqual(len(signatures), len(set(signatures)))

    @unittest.skipUnless(DEFAULT_CARRIER_SCHEDULE.exists(), "local carrier schedule is unavailable")
    def test_carrier_vns_operator_priorities_are_stable(self) -> None:
        import json

        payload = json.loads(DEFAULT_CARRIER_SCHEDULE.read_text(encoding="utf-8"))
        neighborhood = build_vns_plan(payload["schedule"], radii=(2,), max_gaps=1)["neighborhoods"][0]
        original = neighborhood["neighborhoodJobs"]
        swapped = priority_for_operator(neighborhood, "adjacent-boundary-swap")
        forward = priority_for_operator(neighborhood, "forward-insertion")
        backward = priority_for_operator(neighborhood, "backward-insertion")
        self.assertEqual(set(swapped), set(original))
        self.assertEqual(set(forward), set(original))
        self.assertEqual(set(backward), set(original))
        self.assertNotEqual(swapped, tuple(original))

    @unittest.skipUnless(DEFAULT_CARRIER_SCHEDULE.exists(), "local carrier schedule is unavailable")
    def test_carrier_local_cp_is_deterministic_and_fixed_mode(self) -> None:
        import json

        payload = json.loads(DEFAULT_CARRIER_SCHEDULE.read_text(encoding="utf-8"))
        neighborhood = build_vns_plan(payload["schedule"], radii=(2,), max_gaps=1)["neighborhoods"][0]
        first = solve_fixed_mode_neighborhood(payload["schedule"], neighborhood["neighborhoodJobs"])
        second = solve_fixed_mode_neighborhood(payload["schedule"], neighborhood["neighborhoodJobs"])
        self.assertIn(first["status"], {"OPTIMAL", "FEASIBLE"})
        self.assertEqual(first, second)
        self.assertTrue(first["fixedModes"])
        self.assertEqual(set(first["prioritiesByOperation"]), {str(index) for index in range(8)})

    @unittest.skipUnless(DEFAULT_CARRIER_SCHEDULE.exists(), "local carrier schedule is unavailable")
    def test_carrier_alns_plan_and_repairs_are_deterministic(self) -> None:
        import json

        payload = json.loads(DEFAULT_CARRIER_SCHEDULE.read_text(encoding="utf-8"))
        first = build_alns_plan(payload["schedule"], destroy_sizes=(2, 3), radius=2, max_gaps=4)
        second = build_alns_plan(payload["schedule"], destroy_sizes=(2, 3), radius=2, max_gaps=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first["neighborhoods"]), 2)
        self.assertEqual([item["overrideBudget"] for item in first["neighborhoods"]], [2, 3])
        self.assertGreater(first["neighborhoods"][1]["releasedOperationCount"], 0)
        neighborhood = first["neighborhoods"][0]
        for operator in ("alns-boundary-pull", "alns-sink-order", "alns-wait-first"):
            priorities = alns_priorities(payload["schedule"], neighborhood, operator)
            self.assertEqual(set(priorities), set(range(8)))
            self.assertEqual(set(priorities[6]), set(neighborhood["neighborhoodJobs"]))
        replay = alns_dispatch_order(payload["schedule"], neighborhood, "alns-incumbent-replay")
        repaired = alns_dispatch_order(payload["schedule"], neighborhood, "alns-boundary-pull")
        adjacent = alns_dispatch_order(payload["schedule"], neighborhood, "alns-adjacent-o6")
        self.assertEqual(len(replay["actions"]), 160)
        self.assertEqual(replay["overrideCount"], 0)
        self.assertLessEqual(repaired["overrideCount"], neighborhood["overrideBudget"])
        self.assertEqual(set(replay["actions"]), set(repaired["actions"]))
        self.assertLessEqual(adjacent["overrideCount"], neighborhood["overrideBudget"])
        self.assertTrue(all(item["operation"] == 5 for item in adjacent["swaps"]))

    def test_bayesian_evidence_ranking_prefers_observed_local_gain(self) -> None:
        observations = [
            {"operator": "alns-adjacent-o4", "stage": 4, "gain": 7.0, "rawImprovement": True, "propagationClosed": True, "expansionCount": 0, "accepted": True},
            {"operator": "alns-adjacent-o4", "stage": 4, "gain": -1.0, "rawImprovement": False, "propagationClosed": False, "expansionCount": 3, "accepted": False},
            {"operator": "alns-adjacent-o5", "stage": 5, "gain": -20.0, "rawImprovement": False, "propagationClosed": False, "expansionCount": 8, "accepted": False},
            {"operator": "alns-adjacent-o5", "stage": 5, "gain": -30.0, "rawImprovement": False, "propagationClosed": False, "expansionCount": 7, "accepted": False},
        ]
        ranked = rank_operator_evidence(observations, candidate_operators=())
        self.assertEqual(ranked["recommendedOperator"], "alns-adjacent-o4")

    def test_generic_evidence_uses_family_specific_posterior(self) -> None:
        observations = [
            {"family": "JSP", "bottleneckKind": "critical_resource_block", "moveDepth": 1, "gain": 3, "improved": True, "releasedFraction": 0.3},
            {"family": "JSP", "bottleneckKind": "critical_resource_block", "moveDepth": 1, "gain": 0, "improved": False, "releasedFraction": 0.2},
            {"family": "FJSP", "bottleneckKind": "flexible_resource_imbalance", "moveDepth": 1, "gain": 4, "improved": True, "releasedFraction": 0.4},
        ]
        ranked = rank_generic_evidence(observations)
        self.assertEqual(set(ranked["recommendations"]), {"JSP", "FJSP"})
        self.assertEqual(ranked["recommendations"]["FJSP"]["bottleneckKind"], "flexible_resource_imbalance")


if __name__ == "__main__":
    unittest.main()
