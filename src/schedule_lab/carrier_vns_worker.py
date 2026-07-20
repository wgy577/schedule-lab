from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from .carrier_alns import alns_dispatch_order, alns_priorities, build_alns_plan
from .carrier_local_cp import solve_fixed_mode_neighborhood
from .carrier_oracle_cuts import load_oracle_cut_store, matching_exact_cuts, proposal_key
from .carrier_vns import build_vns_plan, focus_sequence, rank_focus_gaps, schedule_hash
from .carrier_worker import DEFAULT_LEGACY_ROOT, _basic_domain_checks, _install_legacy_import_shims


VNS_OPERATORS = ("cp-sat-fixed-mode", "adjacent-boundary-swap", "forward-insertion", "backward-insertion")
ALNS_OPERATORS = (
    "alns-incumbent-replay",
    "alns-adjacent-o6",
    "alns-adjacent-o5",
    "alns-adjacent-o4",
    "alns-adjacent-o3",
    "alns-boundary-pull",
    "alns-sink-order",
    "alns-wait-first",
    "cp-sat-fixed-mode",
)
OPERATORS = tuple(dict.fromkeys((*VNS_OPERATORS, *ALNS_OPERATORS)))


def priority_for_operator(neighborhood: dict[str, Any], operator: str) -> tuple[int, ...]:
    jobs = list(map(int, neighborhood["neighborhoodJobs"]))
    left = int(neighborhood["gap"]["leftJob"])
    right = int(neighborhood["gap"]["rightJob"])
    if operator == "cp-sat-fixed-mode":
        return tuple(jobs)
    if operator == "adjacent-boundary-swap":
        left_index, right_index = jobs.index(left), jobs.index(right)
        jobs[left_index], jobs[right_index] = jobs[right_index], jobs[left_index]
    elif operator == "forward-insertion":
        jobs.remove(right)
        jobs.insert(0, right)
    elif operator == "backward-insertion":
        jobs.remove(left)
        jobs.append(left)
    else:
        raise ValueError(f"unsupported operator: {operator}")
    return tuple(jobs)


def _gap_metrics(schedule: list[dict[str, Any]]) -> dict[str, float]:
    gaps = rank_focus_gaps(focus_sequence(schedule))
    return {
        "positiveGapTotal": round(sum(float(item["gap"]) for item in gaps), 6),
        "largestGap": round(float(gaps[0]["gap"]), 6) if gaps else 0.0,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    incumbent_payload = json.loads(args.incumbent.read_text(encoding="utf-8"))
    incumbent = incumbent_payload["schedule"] if isinstance(incumbent_payload, dict) else incumbent_payload
    incumbent_trace = (
        incumbent_payload.get("decisionTrace") or incumbent_payload.get("decision_trace")
        if isinstance(incumbent_payload, dict)
        else None
    )
    if args.plan_kind == "alns":
        plan = build_alns_plan(
            incumbent,
            destroy_sizes=tuple(args.destroy_sizes),
            radius=args.destroy_radius,
            max_gaps=args.max_gaps,
            max_jobs=args.max_jobs,
            gap_ranks=tuple(args.gap_ranks),
            expansion_jobs=tuple(args.expansion_jobs),
        )
    else:
        plan = build_vns_plan(incumbent, radii=tuple(args.radii), max_gaps=args.max_gaps)
    legacy_root = args.legacy_root.expanduser().resolve()
    os.environ["ALL20"] = "1"
    os.environ["LEARN_EPS"] = "0"
    os.environ["HIDDEN_DIM"] = "128"
    sys.path.insert(0, str(legacy_root))
    sys.path.insert(0, str(legacy_root / "deck_video"))
    sys.argv = [sys.argv[0], "--device", args.device]
    _install_legacy_import_shims()

    import numpy as np
    import torch
    import video_viz as vv
    import FJSP_Env as legacy_env
    import models.PPO_Actor1 as actor_module
    from Params import configs
    from collision.TimeMat import generate_time_matrix1
    from train_data.random_plane import Select_for_train as original_select_for_train

    configs.device = args.device
    configs.hidden_dim = 128
    vv.DEV = args.device
    checkpoint = legacy_root / "saved_network" / "FJSP_J20M12h" / "100_502"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"legacy checkpoint not found: {checkpoint}")

    def select_all20(batch, plane, partial, seed=None):
        return original_select_for_train(batch, plane, partial, seed=None)

    legacy_env.Select_for_train = select_all20
    os.environ["ALL20"] = "0"
    data = generate_time_matrix1(1, 20, 8, configs.n_m, 20, 60)
    incumbent_modes = {
        (int(item["job"]), int(item["op"])): int(item["machine"])
        for item in incumbent
    }
    incumbent_makespan = max(float(item["end"]) for item in incumbent)
    incumbent_hash = schedule_hash(incumbent)
    oracle_cut_store = load_oracle_cut_store(args.oracle_cuts)
    candidates: list[dict[str, Any]] = []
    skipped_candidates: list[dict[str, Any]] = []
    previous_cwd = Path.cwd()
    original_greedy = actor_module.greedy_select_action
    try:
        os.chdir(legacy_root)
        for neighborhood in plan["neighborhoods"]:
            if neighborhood["tabu"]:
                continue
            for operator in args.operators:
                cp_sat = None
                dispatch_plan = None
                if operator == "cp-sat-fixed-mode":
                    cp_sat = solve_fixed_mode_neighborhood(
                        incumbent,
                        neighborhood["neighborhoodJobs"],
                        seed=args.seed,
                    )
                    if not cp_sat["prioritiesByOperation"]:
                        continue
                    priorities_by_operation = {
                        int(operation): tuple(map(int, jobs))
                        for operation, jobs in cp_sat["prioritiesByOperation"].items()
                    }
                    priority = priorities_by_operation.get(6, tuple(map(int, neighborhood["neighborhoodJobs"])))
                elif operator.startswith("alns-"):
                    if operator == "alns-incumbent-replay" or operator.startswith("alns-adjacent-o"):
                        priorities_by_operation = {
                            operation: tuple(map(int, neighborhood["neighborhoodJobs"]))
                            for operation in range(8)
                        }
                    else:
                        priorities_by_operation = alns_priorities(incumbent, neighborhood, operator)
                    priority = priorities_by_operation[6]
                    dispatch_plan = alns_dispatch_order(
                        incumbent,
                        neighborhood,
                        operator,
                        decision_trace=incumbent_trace,
                    )
                else:
                    priority = priority_for_operator(neighborhood, operator)
                    priorities_by_operation = {operation: priority for operation in range(8)}
                current_proposal_key = proposal_key(
                    incumbent_hash=incumbent_hash,
                    neighborhood_signature=neighborhood["signature"],
                    operator=operator,
                    dispatch_swaps=() if dispatch_plan is None else dispatch_plan["swaps"],
                    seed=args.seed,
                )
                matching_cuts = matching_exact_cuts(oracle_cut_store, current_proposal_key)
                if matching_cuts:
                    skipped_candidates.append(
                        {
                            "neighborhoodSignature": neighborhood["signature"],
                            "operator": operator,
                            "proposalKey": current_proposal_key,
                            "cutIds": [cut["id"] for cut in matching_cuts],
                            "reasons": sorted({cut["reason"] for cut in matching_cuts}),
                            "requiredExpansionJobs": sorted(
                                {
                                    int(job)
                                    for cut in matching_cuts
                                    for job in cut.get("requiredExpansionJobs", [])
                                }
                            ),
                        }
                    )
                    continue
                neighborhood_job_set = set(map(int, neighborhood["neighborhoodJobs"]))
                override_budget = neighborhood.get("overrideBudget")
                selected_action_state: dict[str, Any] = {"actions": None, "overrideCount": 0}
                dispatch_rank = (
                    {action: index for index, action in enumerate(dispatch_plan["actions"])}
                    if dispatch_plan is not None
                    else None
                )

                def controlled_greedy(probabilities, candidate):
                    probabilities_2d = probabilities.squeeze(-1)
                    distribution = torch.distributions.Categorical(probabilities_2d)
                    _, indices = probabilities_2d.max(1)
                    selected_indices = indices.clone()
                    for batch_index in range(indices.size(0)):
                        base_index = int(indices[batch_index].item())
                        if dispatch_rank is not None:
                            available = []
                            for index in range(candidate.size(1)):
                                if float(probabilities_2d[batch_index][index].item()) <= 0.0:
                                    continue
                                action = int(candidate[batch_index][index].item())
                                available.append((dispatch_rank.get(action, len(dispatch_rank)), index))
                            if available:
                                selected_indices[batch_index] = min(available)[1]
                            continue
                        base_action = int(candidate[batch_index][base_index].item())
                        base_job, base_op = divmod(base_action, configs.n_op)
                        priority_for_operation = priorities_by_operation.get(base_op, priority)
                        rank = {job: index for index, job in enumerate(priority_for_operation)}
                        if base_job not in rank:
                            continue
                        alternatives: list[tuple[int, int]] = []
                        for index in range(candidate.size(1)):
                            if float(probabilities_2d[batch_index][index].item()) <= 0.0:
                                continue
                            action = int(candidate[batch_index][index].item())
                            job, operation = divmod(action, configs.n_op)
                            if operation == base_op and job in rank:
                                alternatives.append((rank[job], index))
                        if alternatives:
                            preferred_index = min(alternatives)[1]
                            if preferred_index != base_index:
                                if (
                                    override_budget is not None
                                    and selected_action_state["overrideCount"] >= int(override_budget)
                                ):
                                    continue
                                selected_action_state["overrideCount"] += 1
                            selected_indices[batch_index] = preferred_index
                    actions = torch.stack(
                        [candidate[index][selected_indices[index]] for index in range(selected_indices.size(0))]
                    )
                    selected_action_state["actions"] = actions.detach().clone()
                    return actions, selected_indices, distribution.log_prob(selected_indices)

                class FrozenOutsideMachinePolicy(torch.nn.Module):
                    def __init__(self, inner):
                        super().__init__()
                        self.inner = inner

                    def forward(self, *forward_args, **forward_kwargs):
                        probabilities, pool = self.inner(*forward_args, **forward_kwargs)
                        actions = selected_action_state["actions"]
                        if actions is None:
                            return probabilities, pool
                        controlled = probabilities.clone()
                        for batch_index in range(actions.size(0)):
                            action = int(actions[batch_index].item())
                            job, operation = divmod(action, configs.n_op)
                            if (
                                args.plan_kind == "vns"
                                and job in neighborhood_job_set
                                and operator != "cp-sat-fixed-mode"
                            ):
                                continue
                            machine = incumbent_modes[(job, operation)]
                            if float(controlled[batch_index][machine].item()) <= 0.0:
                                continue
                            controlled[batch_index].zero_()
                            controlled[batch_index][machine] = 1.0
                        return controlled, pool

                actor_module.greedy_select_action = controlled_greedy
                seed = args.seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                policy_job, policy_mch = vv.make_policy_actors(learn_eps=False)
                policy_job.load_state_dict(
                    torch.load(checkpoint / "policy_job.pth", map_location=args.device, weights_only=True)
                )
                policy_mch.load_state_dict(
                    torch.load(checkpoint / "policy_mch.pth", map_location=args.device, weights_only=True)
                )
                policy_job.train()
                policy_mch.train()
                frozen_machine_policy = FrozenOutsideMachinePolicy(policy_mch)
                schedule, reported_makespan, steps, _ = vv.run_instance_greedy(
                    policy_job,
                    frozen_machine_policy,
                    data,
                    device=args.device,
                    collect_steps=True,
                    case_idx=0,
                    greedy=True,
                )
                schedule.sort(key=lambda item: (int(item["job"]), int(item["op"])))
                errors = _basic_domain_checks(schedule)
                neighborhood_jobs = neighborhood_job_set
                outside_mode_changes = [
                    {"job": int(item["job"]), "op": int(item["op"]), "machine": int(item["machine"])}
                    for item in schedule
                    if int(item["job"]) not in neighborhood_jobs
                    and incumbent_modes.get((int(item["job"]), int(item["op"]))) != int(item["machine"])
                ]
                true_makespan = max(float(item["end"]) for item in schedule)
                required_expansion_jobs = sorted({int(item["job"]) for item in outside_mode_changes})
                accepted = not errors and not outside_mode_changes and true_makespan < incumbent_makespan - 1e-6
                candidate_signature = schedule_hash(schedule)
                candidates.append(
                    {
                        "candidateIndex": len(candidates),
                        "neighborhoodSignature": neighborhood["signature"],
                        "scheduleHash": candidate_signature,
                        "gapRank": neighborhood["gapRank"],
                        "radius": neighborhood["radius"],
                        "operator": operator,
                        "priorityJobs": list(priority),
                        "overrideBudget": override_budget,
                        "overrideCount": (
                            dispatch_plan["overrideCount"]
                            if dispatch_plan is not None
                            else selected_action_state["overrideCount"]
                        ),
                        "dispatchSwaps": [] if dispatch_plan is None else dispatch_plan["swaps"],
                        "prioritiesByOperation": {
                            str(operation): list(jobs)
                            for operation, jobs in sorted(priorities_by_operation.items())
                        },
                        "cpSat": None if cp_sat is None else {key: value for key, value in cp_sat.items() if key != "assignments"},
                        "neighborhoodJobs": sorted(neighborhood_jobs),
                        "domainConstructed": not errors,
                        "domainErrors": errors,
                        "outsideModeChanges": outside_mode_changes,
                        "outsideModeChangeCount": len(outside_mode_changes),
                        "requiredExpansionJobs": required_expansion_jobs,
                        "propagationClosed": not required_expansion_jobs,
                        "reportedPolicyMakespan": float(reported_makespan),
                        "trueMakespan": true_makespan,
                        "gapMetrics": _gap_metrics(schedule),
                        "accepted": accepted,
                        "operationCount": len(schedule),
                        "decisionCount": len(steps),
                        "decisionTrace": steps,
                        "schedule": schedule,
                    }
                )
    finally:
        actor_module.greedy_select_action = original_greedy
        os.chdir(previous_cwd)

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    best = min(
        accepted_candidates,
        key=lambda item: (item["trueMakespan"], item["gapMetrics"]["positiveGapTotal"], item["candidateIndex"]),
        default=None,
    )
    return {
        "meta": {
            "controller": f"deterministic-{args.plan_kind}-domain-replay",
            "network": "FJSP_J20M12h",
            "checkpoint": "100_502",
            "seed": args.seed,
            "radii": args.radii,
            "destroySizes": args.destroy_sizes if args.plan_kind == "alns" else None,
            "destroyRadius": args.destroy_radius if args.plan_kind == "alns" else None,
            "maxJobs": args.max_jobs if args.plan_kind == "alns" else None,
            "maxGaps": args.max_gaps,
            "operators": args.operators,
            "device": args.device,
            "domainEnvironment": "FJSP_Env + connector_two",
            "outsideModePolicy": "reject changes outside neighborhood",
            "oracleCutPolicy": "exact deterministic no-good only",
            "oracleCutSource": None if args.oracle_cuts is None else str(args.oracle_cuts.resolve()),
        },
        "incumbent": {
            "source": str(args.incumbent.resolve()),
            "scheduleHash": incumbent_hash,
            "trueMakespan": incumbent_makespan,
            "gapMetrics": _gap_metrics(incumbent),
        },
        "plan": plan,
        "best": best,
        "skippedCandidates": skipped_candidates,
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"schedule", "decisionTrace"}
            }
            for candidate in candidates
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic carrier VNS domain replay worker")
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--plan-kind", choices=["vns", "alns"], default="vns")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max-gaps", type=int, default=1)
    parser.add_argument("--destroy-sizes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--destroy-radius", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=14)
    parser.add_argument("--gap-ranks", type=int, nargs="*", default=[])
    parser.add_argument("--expansion-jobs", type=int, nargs="*", default=[])
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=list(OPERATORS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-cuts", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = payload["best"]
    if best is None:
        print(f"carrier VNS: evaluated {len(payload['candidates'])} candidates; no strict accepted improvement")
    else:
        print(
            f"carrier VNS: evaluated {len(payload['candidates'])} candidates; "
            f"best true makespan={best['trueMakespan']:.3f}"
        )


if __name__ == "__main__":
    main()
