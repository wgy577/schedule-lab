from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any


DEFAULT_LEGACY_ROOT = Path(
    "/Users/guangyuwu/Desktop/sortie code/"
    "comparision_pass_video/comparision_pass -2025-9-7"
)


def _install_legacy_import_shims() -> None:
    """Provide the unused TensorFlow/Keras names imported by the old tree."""

    names = (
        "tensorflow",
        "tensorflow.python",
        "tensorflow.python.util",
        "tensorflow.python.util.deprecation",
        "keras",
        "keras.src",
        "keras.src.ops",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["tensorflow.python.util.deprecation"].rewrite_argument_docstring = lambda *args, **kwargs: None
    modules["keras.src.ops"].dtype = None
    sys.modules.update(modules)


def _basic_domain_checks(schedule: list[dict[str, Any]]) -> list[str]:
    """Checks invariants in addition to construction by the collision-aware env."""

    errors: list[str] = []
    if len(schedule) != 160:
        errors.append(f"expected 160 operations, received {len(schedule)}")
    keys = {(int(item["job"]), int(item["op"])) for item in schedule}
    expected = {(job, op) for job in range(20) for op in range(8)}
    if keys != expected:
        errors.append("the rollout does not contain exactly operations J1..J20/O1..O8")
    for job in range(20):
        operations = sorted(
            (item for item in schedule if int(item["job"]) == job),
            key=lambda item: int(item["op"]),
        )
        for previous, current in zip(operations, operations[1:]):
            if float(current["start"]) + 1e-4 < float(previous["end"]):
                errors.append(
                    f"precedence violation J{job + 1}: O{int(previous['op']) + 1} -> "
                    f"O{int(current['op']) + 1}"
                )
    return errors


def _sampling_profile(rollout: int) -> tuple[float, float]:
    """Returns a small epsilon neighborhood and probability temperature.

    The legacy actor already divides its logits by two. Sampling that broad
    distribution at every decision destroys good schedule structure. These
    profiles keep most greedy decisions and perturb only a controlled subset.
    """

    epsilons = (0.02, 0.04, 0.06, 0.08, 0.12)
    temperatures = (0.55, 0.75, 1.0)
    index = max(0, rollout - 1)
    return epsilons[index % len(epsilons)], temperatures[(index // len(epsilons)) % len(temperatures)]


def _run(args: argparse.Namespace) -> dict[str, Any]:
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

    # Apply ALL20 exactly once. video_viz applies the same monkey patch on every
    # call, which recursively wraps itself during a multi-rollout process.
    def select_all20(batch, plane, partial, seed=None):
        return original_select_for_train(batch, plane, partial, seed=None)

    legacy_env.Select_for_train = select_all20
    os.environ["ALL20"] = "0"
    data = generate_time_matrix1(1, 20, 8, configs.n_m, 20, 60)
    candidates: list[dict[str, Any]] = []
    previous_cwd = Path.cwd()
    try:
        os.chdir(legacy_root)
        for rollout in range(args.rollouts):
            rollout_seed = args.seed + rollout
            random.seed(rollout_seed)
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)

            # Reconstruct and reload on every rollout. This preserves the
            # validated train-mode behavior while preventing BatchNorm buffers
            # from leaking information between candidates.
            policy_job, policy_mch = vv.make_policy_actors(learn_eps=False)
            policy_job.load_state_dict(
                torch.load(checkpoint / "policy_job.pth", map_location=args.device, weights_only=True)
            )
            policy_mch.load_state_dict(
                torch.load(checkpoint / "policy_mch.pth", map_location=args.device, weights_only=True)
            )
            policy_job.train()
            policy_mch.train()
            forced_profile = args.forced_epsilon is not None
            greedy = rollout == 0 and args.include_greedy and not forced_profile
            if forced_profile:
                epsilon, temperature = args.forced_epsilon, args.forced_temperature
            else:
                epsilon, temperature = (0.0, 0.0) if greedy else _sampling_profile(rollout)
            if not greedy:
                def controlled_select_action(probabilities, candidate):
                    probabilities = probabilities.squeeze(-1)
                    sharpened = probabilities.clamp_min(1e-12).pow(1.0 / temperature)
                    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
                    greedy_index = probabilities.argmax(dim=1)
                    greedy_mass = torch.nn.functional.one_hot(
                        greedy_index, num_classes=probabilities.size(1)
                    ).to(probabilities.dtype)
                    neighborhood = (1.0 - epsilon) * greedy_mass + epsilon * sharpened
                    distribution = torch.distributions.Categorical(neighborhood)
                    selected = distribution.sample()
                    action = torch.gather(candidate, 1, selected.unsqueeze(1)).squeeze(1)
                    return action, selected, distribution.log_prob(selected)

                actor_module.select_action1 = controlled_select_action
            schedule, reported_makespan, steps, _ = vv.run_instance_greedy(
                policy_job,
                policy_mch,
                data,
                device=args.device,
                collect_steps=True,
                case_idx=0,
                greedy=greedy,
            )
            schedule.sort(key=lambda item: (int(item["job"]), int(item["op"])))
            errors = _basic_domain_checks(schedule)
            true_makespan = max((float(item["end"]) for item in schedule), default=float("inf"))
            candidates.append(
                {
                    "rollout": rollout,
                    "seed": rollout_seed,
                    "strategy": "greedy" if greedy else "controlled-policy-neighborhood",
                    "sampling": None if greedy else {"epsilon": epsilon, "temperature": temperature},
                    "domain_validated": not errors,
                    "validation_errors": errors,
                    "reported_policy_makespan": float(reported_makespan),
                    "true_makespan": true_makespan,
                    "operation_count": len(schedule),
                    "decision_count": len(steps),
                    "decision_trace": steps if args.include_steps else None,
                    "schedule": schedule,
                }
            )
    finally:
        os.chdir(previous_cwd)

    feasible = [candidate for candidate in candidates if candidate["domain_validated"]]
    if not feasible:
        raise RuntimeError("the carrier oracle produced no domain-valid candidate")
    best = min(feasible, key=lambda candidate: (candidate["true_makespan"], candidate["rollout"]))
    return {
        "meta": {
            "network": "FJSP_J20M12h",
            "checkpoint": "100_502",
            "legacy_root": str(legacy_root),
            "rollouts": args.rollouts,
            "seed": args.seed,
            "device": args.device,
            "all20": True,
            "eval_mode": False,
            "batchnorm_reset_per_rollout": True,
            "selection_objective": "minimum true max(end), not legacy env.nowtime",
            "domain_oracle": "FJSP_Env + connector_two trajectory/collision scheduling",
        },
        "best": best,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "schedule"}
            for candidate in candidates
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated legacy carrier domain-oracle worker")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--forced-epsilon", type=float)
    parser.add_argument("--forced-temperature", type=float, default=1.0)
    parser.add_argument("--include-steps", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rollouts < 1:
        raise ValueError("rollouts must be at least 1")
    payload = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.best_output:
        args.best_output.parent.mkdir(parents=True, exist_ok=True)
        args.best_output.write_text(
            json.dumps(
                {
                    "meta": {
                        **payload["meta"],
                        "rollout": payload["best"]["rollout"],
                        "rolloutSeed": payload["best"]["seed"],
                        "sampling": payload["best"]["sampling"],
                        "trueMakespan": payload["best"]["true_makespan"],
                        "domainValidated": payload["best"]["domain_validated"],
                    },
                    "schedule": payload["best"]["schedule"],
                    "decisionTrace": payload["best"]["decision_trace"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        f"carrier oracle: {args.rollouts} candidates, best true makespan="
        f"{payload['best']['true_makespan']:.3f}, rollout={payload['best']['rollout']}"
    )


if __name__ == "__main__":
    main()
