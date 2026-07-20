from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from .carrier_vns import schedule_hash
from .carrier_worker import DEFAULT_LEGACY_ROOT, _basic_domain_checks, _install_legacy_import_shims


def _run(args: argparse.Namespace) -> dict[str, Any]:
    target_payload = json.loads(args.target.read_text(encoding="utf-8"))
    target_schedule = target_payload["schedule"]
    source_trace = target_payload.get("decisionTrace") or target_payload.get("decision_trace") or []
    target_modes = {
        (int(item["job"]), int(item["op"])): int(item["machine"])
        for item in target_schedule
    }
    trace_rank: dict[int, int] = {}
    for index, step in enumerate(source_trace):
        if "job" in step and "op" in step:
            action = int(step["job"]) * 8 + int(step["op"])
        else:
            action = int(step.get("action", step.get("task", -1)))
        if action >= 0:
            trace_rank[action] = index
    target_rank = {
        int(item["job"]) * 8 + int(item["op"]): rank
        for rank, item in enumerate(
            sorted(
                target_schedule,
                key=lambda item: (
                    float(item["start"]),
                    trace_rank.get(int(item["job"]) * 8 + int(item["op"]), 10_000),
                    int(item["op"]),
                    int(item["job"]),
                ),
            )
        )
    }
    legacy_root = args.legacy_root.expanduser().resolve()
    os.environ.update({"ALL20": "1", "LEARN_EPS": "0", "HIDDEN_DIM": "128"})
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
    previous_cwd = Path.cwd()
    original_greedy = actor_module.greedy_select_action
    replays: list[dict[str, Any]] = []
    try:
        os.chdir(legacy_root)
        for replay_index in range(args.replays):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            selected_state: dict[str, Any] = {"actions": None, "unavailableMachines": []}

            def target_select(probabilities, candidate):
                probabilities_2d = probabilities.squeeze(-1)
                distribution = torch.distributions.Categorical(probabilities_2d)
                selected_indices = []
                for batch_index in range(candidate.size(0)):
                    eligible = [
                        (target_rank.get(int(candidate[batch_index][index].item()), 100_000), index)
                        for index in range(candidate.size(1))
                        if float(probabilities_2d[batch_index][index].item()) > 0.0
                    ]
                    if not eligible:
                        selected_indices.append(int(probabilities_2d[batch_index].argmax().item()))
                    else:
                        selected_indices.append(min(eligible)[1])
                index_tensor = torch.tensor(selected_indices, device=candidate.device)
                actions = torch.stack(
                    [candidate[index][index_tensor[index]] for index in range(candidate.size(0))]
                )
                selected_state["actions"] = actions.detach().clone()
                return actions, index_tensor, distribution.log_prob(index_tensor)

            class TargetMachinePolicy(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, *forward_args, **forward_kwargs):
                    probabilities, pool = self.inner(*forward_args, **forward_kwargs)
                    actions = selected_state["actions"]
                    if actions is None:
                        return probabilities, pool
                    controlled = probabilities.clone()
                    for batch_index in range(actions.size(0)):
                        action = int(actions[batch_index].item())
                        job, operation = divmod(action, configs.n_op)
                        machine = target_modes[(job, operation)]
                        if float(controlled[batch_index][machine].item()) <= 0.0:
                            selected_state["unavailableMachines"].append(
                                {"job": job, "op": operation, "machine": machine}
                            )
                            continue
                        controlled[batch_index].zero_()
                        controlled[batch_index][machine] = 1.0
                    return controlled, pool

            actor_module.greedy_select_action = target_select
            policy_job, policy_mch = vv.make_policy_actors(learn_eps=False)
            policy_job.load_state_dict(
                torch.load(checkpoint / "policy_job.pth", map_location=args.device, weights_only=True)
            )
            policy_mch.load_state_dict(
                torch.load(checkpoint / "policy_mch.pth", map_location=args.device, weights_only=True)
            )
            policy_job.train()
            policy_mch.train()
            schedule, reported_makespan, steps, _ = vv.run_instance_greedy(
                policy_job,
                TargetMachinePolicy(policy_mch),
                data,
                device=args.device,
                collect_steps=True,
                case_idx=0,
                greedy=True,
            )
            schedule.sort(key=lambda item: (int(item["job"]), int(item["op"])))
            errors = _basic_domain_checks(schedule)
            mode_changes = [
                {
                    "job": int(item["job"]),
                    "op": int(item["op"]),
                    "expected": target_modes[(int(item["job"]), int(item["op"]))],
                    "actual": int(item["machine"]),
                }
                for item in schedule
                if int(item["machine"])
                != target_modes[(int(item["job"]), int(item["op"]))]
            ]
            replays.append(
                {
                    "replayIndex": replay_index,
                    "scheduleHash": schedule_hash(schedule),
                    "trueMakespan": max(float(item["end"]) for item in schedule),
                    "reportedMakespan": float(reported_makespan),
                    "domainConstructed": not errors,
                    "domainErrors": errors,
                    "modeChanges": mode_changes,
                    "unavailableTargetMachines": selected_state["unavailableMachines"],
                    "decisionTrace": steps,
                    "schedule": schedule,
                }
            )
    finally:
        actor_module.greedy_select_action = original_greedy
        os.chdir(previous_cwd)

    deterministic = len({item["scheduleHash"] for item in replays}) == 1
    best = min(replays, key=lambda item: (item["trueMakespan"], item["scheduleHash"]))
    return {
        "controller": "target-order-fixed-machine-domain-replay-v1",
        "source": str(args.target.resolve()),
        "targetHash": schedule_hash(target_schedule),
        "settings": {"seed": args.seed, "workers": 1, "replays": args.replays},
        "deterministic": deterministic,
        "best": best,
        "replays": [
            {key: value for key, value in item.items() if key not in {"schedule", "decisionTrace"}}
            for item in replays
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a target schedule order in the legacy domain Oracle")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"target replay: deterministic={payload['deterministic']}, "
        f"best makespan={payload['best']['trueMakespan']:.3f}"
    )


if __name__ == "__main__":
    main()
