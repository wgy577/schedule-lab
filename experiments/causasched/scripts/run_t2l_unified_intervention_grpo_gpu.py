#!/usr/bin/env python3
"""AutoDL entry point for T2-L unified net-intervention GRPO."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from causal_schedule_lab.benchmarks import (  # noqa: E402
    build_fsp, build_hfsp, build_jsp, random_fjsp,
)
from causal_schedule_lab.ir import Problem, Schedule  # noqa: E402
from causal_schedule_lab.core_validation import validate_schedule  # noqa: E402
from causal_schedule_lab.m3 import hierarchical_residual as HR  # noqa: E402
from causal_schedule_lab.m3 import joint_grpo as JG  # noqa: E402
from causal_schedule_lab.m3 import persistent_frontier_grpo as PF  # noqa: E402
from causal_schedule_lab.m3 import rolling_grpo as RGRPO  # noqa: E402
from causal_schedule_lab.m3 import t2d_gpu_trainer as GT  # noqa: E402
from causal_schedule_lab.m3 import config as C  # noqa: E402
import run_m3_canonical_training as RUN  # noqa: E402


def _graph_size(graph):
    problem = graph.problem
    return {
        "jobs": len(problem.jobs),
        "resources": len(problem.resources),
        "operations": len(problem.operations),
    }


def _filter_oversized_graphs(graphs, *, max_resources, max_operations):
    """Keep ordinary 16x10 cases; reject only true resource/operation outliers."""
    kept, dropped = [], []
    for graph in graphs:
        size = _graph_size(graph)
        if (size["resources"] > int(max_resources) or
                size["operations"] > int(max_operations)):
            dropped.append({"instance_id": graph.iid, "source": graph.src, **size})
        else:
            kept.append(graph)
    return kept, dropped


def _drl_initial_graphs(bank_path, replay, *, first_episode=500_000_000):
    """Load immutable DRL-generated schedules as TRAIN starting states.

    The bank is an initial-state source only: schedules are validated and copied
    into ordinary RGRPO.Graph objects.  SFT weights remain only the policy warm
    start/base prior; all subsequent trainable updates are GRPO/RL.
    """
    bank_path = Path(bank_path)
    manifest_path = bank_path / "protocol.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DRL initial bank protocol missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"DRL initial bank is incomplete: {manifest_path}")
    graphs = []
    for idx, row in enumerate(manifest.get("entries", ())):
        file = bank_path / row["file"]
        payload = json.loads(file.read_text(encoding="utf-8"))
        problem = Problem.model_validate(payload["problem"])
        schedule = Schedule.model_validate(payload["schedule"])
        report = validate_schedule(problem, schedule)
        if not report.feasible or int(schedule.makespan) != int(payload["makespan"]):
            raise ValueError(
                f"Invalid DRL initial schedule: {row.get('instance_id', file)}")
        iid = str(row.get("instance_id", payload.get("instance_id", file.stem)))
        family = str(row.get("family", payload.get("family", "fjsp"))).lower()
        graphs.append(RGRPO.Graph(
            iid=iid, episode_id=int(first_episode + idx), problem=problem,
            schedule=schedule, progmem=copy.deepcopy(replay["progmem"]),
            src=f"drl_{family}", ms0=int(schedule.makespan)))
    if not graphs:
        raise ValueError(f"DRL initial bank has no entries: {bank_path}")
    return graphs


def _extra_graphs(args, env, re, first_episode, *, count=None, seed=None,
                  id_prefix="T2F", manifest_name="manifest.json", difficulty_offset=0):
    """Add balanced JSP/FSP/FJSP/HFSP roots without offline proposal labels.

    GRPO requires a legal S0/problem and the frozen Memory base, not proposal
    labels.  Every family uses the same canonical IR, S0 constructor, appearance
    compiler, frozen M2 pipeline and fixed-decision executor as the original roots.
    """
    n_instances = int(args.extra_synthetic if count is None else count)
    generation_seed = int(args.extra_seed if seed is None else seed)
    if n_instances <= 0:
        return []
    out_dir = ROOT / args.output_root / "generated_instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(generation_seed)
    families = ("JSP", "FSP", "FJSP", "HFSP")
    levels = (
        ("simple", 5, 4, (1, 30)),
        ("medium", 8, 6, (1, 60)),
        ("hard", 12, 8, (1, 100)),
        ("very_hard", 16, 10, (1, 160)),
    )
    rows = []
    problems = []
    for idx in range(n_instances):
        family = families[idx % len(families)]
        level, jobs, machines_or_stages, duration = levels[
            (idx // len(families) + difficulty_offset) % len(levels)]
        iid = f"{id_prefix}_{family}_{idx:04d}"
        if family == "JSP":
            machines = [f"M{i + 1}" for i in range(machines_or_stages)]
            routes = []
            for _ in range(jobs):
                order = rng.sample(machines, len(machines))
                routes.append([(m, rng.randint(*duration)) for m in order])
            problem = build_jsp(routes, problem_id=iid)
        elif family == "FSP":
            processing = [[rng.randint(*duration) for _ in range(machines_or_stages)]
                          for _ in range(jobs)]
            problem = build_fsp(processing, problem_id=iid)
        elif family == "HFSP":
            stages = max(3, machines_or_stages // 2)
            processing = [[rng.randint(*duration) for _ in range(stages)]
                          for _ in range(jobs)]
            machines_per_stage = [rng.randint(2, 4) for _ in range(stages)]
            problem = build_hfsp(processing, machines_per_stage, problem_id=iid)
        else:
            operations = max(3, machines_or_stages // 2)
            problem = random_fjsp(
                jobs=jobs, operations=operations, machines=machines_or_stages,
                flexibility=min(3, machines_or_stages), duration_range=duration,
                seed=generation_seed + idx, problem_id=iid)
        problems.append(problem)
        rows.append({"instance_id": iid, "family": family, "difficulty": level,
                     "jobs": jobs, "scale": machines_or_stages,
                     "duration_range": list(duration), "seed": generation_seed + idx,
                     "n_operations": len(problem.operations)})

    def build_state(problem, iid):
        pilot = env["pilot"]
        schedule = pilot.solve_dispatching(problem, rule="earliest_finish")
        appearance = pilot.diagnose_and_prune(problem, schedule).model_dump(mode="json")
        bundle = pilot.compile_sg_sct_input_v1_3(
            problem, schedule, appearance, case_id=f"S::{iid}::b5")
        block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
        # Force the exact frozen representation path now, so unsupported family
        # drift fails at startup rather than after hours of training.
        pilot.build_m2_runtime_context(problem, schedule, appearance,
                                       block_ids=block_ids)
        pilot.to_sg_sct_batch_v1_3(bundle, device="cpu")
        return schedule

    graphs = []
    for idx, (row, problem) in enumerate(zip(rows, problems)):
        iid = row["instance_id"]
        schedule = build_state(problem, iid)
        graphs.append(RGRPO.Graph(
            iid=iid, episode_id=int(first_episode + idx), problem=problem,
            schedule=schedule, progmem=copy.deepcopy(re["progmem"]),
            src=f"online_{row['family'].lower()}", ms0=int(schedule.makespan)))
    (out_dir / manifest_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return graphs


def _fixed_validation_evaluator(env, scorer, specs, *, horizon=12, samples=0,
                                bestn_interval=5, bestn_samples=4):
    """Return a fixed-S0, no-gradient, no-write validation callback.

    Every invocation uses identical instances, budgets and branch seeds.  Each
    rollout receives a fresh Memory copy, so validation never feeds training or
    a later validation point.
    """
    frozen_specs = list(specs)

    @torch.no_grad()
    def evaluate(jpol, cycle):
        started = time.time()
        sample_count = (int(bestn_samples)
                        if int(bestn_interval) > 0 and cycle % int(bestn_interval) == 0
                        else int(samples))
        greedy_best_gains, greedy_terminal_gains = [], []
        bestn_gains, sample_best_means, sample_terminal_means = [], [], []
        roots, families = [], []
        progress_stride = max(1, len(frozen_specs) // 8)
        for spec_index, spec in enumerate(frozen_specs, start=1):
            root_ms = int(spec["schedule"].makespan)
            roots.append(root_ms)
            families.append(str(spec.get("family", "unknown")).lower())
            def rollout(sample):
                rf = {"problem": spec["problem"], "schedule": spec["schedule"],
                      "iid": spec["iid"], "episode_id": spec["episode_id"],
                      "progmem": copy.deepcopy(spec["progmem"]),
                      "root_ms": root_ms, "horizon": int(horizon)}
                terminal_gain, _usage, steps = JG.agentic_parity_rollout(
                    env, scorer, jpol, rf, use_mem=True, horizon=int(horizon),
                    gate_mem=False, gate_variant="r14", action_space="policy_sampled",
                    sample=sample, allow_policy_stop=False,
                    stop_on_nonpositive=False, feasible_fallback=True)
                running_gain = 0.0
                best_gain = 0.0  # S0 is always an admissible incumbent.
                for step in steps:
                    improvement = step.get("improvement")
                    if improvement is None:
                        continue
                    running_gain += float(improvement)
                    best_gain = max(best_gain, running_gain)
                return float(terminal_gain), float(best_gain)
            greedy_terminal, greedy_best = rollout(None)
            sampled = [rollout({
                "T": float(C.TO1_R13_TEMP), "eps": float(C.TO1_R13_MIX_EPS),
                "rng": random.Random(JG.t2a_branch_seed(
                    spec["iid"], spec["episode_id"], branch))})
                       for branch in range(sample_count)]
            sampled_terminal = [value[0] for value in sampled]
            sampled_best = [value[1] for value in sampled]
            greedy_terminal_gains.append(greedy_terminal)
            greedy_best_gains.append(greedy_best)
            bestn_gains.append(max([greedy_best] + sampled_best))
            sample_best_means.append(
                float(np.mean(sampled_best)) if sampled_best else greedy_best)
            sample_terminal_means.append(
                float(np.mean(sampled_terminal))
                if sampled_terminal else greedy_terminal)
            if (spec_index == 1 or spec_index == len(frozen_specs) or
                    spec_index % progress_stride == 0):
                print(f"[t2m] [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"validation rollout PROGRESS {spec_index}/{len(frozen_specs)} "
                      f"cycle={cycle}", flush=True)
        denom = np.maximum(np.asarray(roots, dtype=np.float64), 1.0)
        greedy_arr = np.asarray(greedy_best_gains, dtype=np.float64)
        terminal_arr = np.asarray(greedy_terminal_gains, dtype=np.float64)
        best_arr = np.asarray(bestn_gains, dtype=np.float64)
        sample_arr = np.asarray(sample_best_means, dtype=np.float64)
        sample_terminal_arr = np.asarray(sample_terminal_means, dtype=np.float64)
        greedy_norm = 100.0 * greedy_arr / denom
        terminal_norm = 100.0 * terminal_arr / denom
        best_norm = 100.0 * best_arr / denom
        metrics = {
            "cycle": int(cycle), "n": len(frozen_specs),
            "horizon": int(horizon), "samples_per_instance": sample_count,
            "greedy_normalized_mean_pct": float(greedy_norm.mean()),
            "greedy_normalized_median_pct": float(np.median(greedy_norm)),
            "greedy_raw_gain_mean": float(greedy_arr.mean()),
            "greedy_improved_rate": float(np.mean(greedy_arr > 0)),
            "greedy_best_within_horizon_normalized_mean_pct": float(
                greedy_norm.mean()),
            "greedy_best_within_horizon_normalized_median_pct": float(
                np.median(greedy_norm)),
            "greedy_best_within_horizon_raw_gain_mean": float(greedy_arr.mean()),
            "greedy_best_within_horizon_improved_rate": float(
                np.mean(greedy_arr > 0)),
            "greedy_terminal_normalized_mean_pct": float(terminal_norm.mean()),
            "greedy_terminal_normalized_median_pct": float(
                np.median(terminal_norm)),
            "greedy_terminal_raw_gain_mean": float(terminal_arr.mean()),
            "greedy_terminal_improved_rate": float(np.mean(terminal_arr > 0)),
            "greedy_regressed_after_best_rate": float(
                np.mean(greedy_arr > terminal_arr)),
            "greedy_best_to_terminal_loss_mean": float(
                np.mean(greedy_arr - terminal_arr)),
            "seconds": time.time() - started,
        }
        if sample_count > 0:
            metrics["sample_normalized_mean_pct"] = float(
                (100.0 * sample_arr / denom).mean())
            metrics["sample_terminal_normalized_mean_pct"] = float(
                (100.0 * sample_terminal_arr / denom).mean())
            metrics["bestn_normalized_mean_pct"] = float(best_norm.mean())
            metrics["bestn_improved_rate"] = float(np.mean(best_arr > 0))
        for family in sorted(set(families)):
            mask = np.asarray([value == family for value in families], dtype=bool)
            f_denom = denom[mask]
            f_greedy = greedy_arr[mask]
            f_terminal = terminal_arr[mask]
            f_best = best_arr[mask]
            metrics[f"{family}_greedy_normalized_mean_pct"] = float(
                (100.0 * f_greedy / f_denom).mean())
            metrics[f"{family}_greedy_improved_rate"] = float(np.mean(f_greedy > 0))
            metrics[f"{family}_greedy_terminal_normalized_mean_pct"] = float(
                (100.0 * f_terminal / f_denom).mean())
            metrics[f"{family}_greedy_terminal_improved_rate"] = float(
                np.mean(f_terminal > 0))
            if sample_count > 0:
                metrics[f"{family}_bestn_normalized_mean_pct"] = float(
                    (100.0 * f_best / f_denom).mean())
        return metrics
    return evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(description="T2-L unified net-intervention GRPO GPU runner")
    ap.add_argument("--architecture", choices=["P0", "P1", "P2"], default="P2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--workers", type=int, default=8,
                    help="safe default; graph-local rollout scales workers without forced sibling sharding")
    ap.add_argument("--branches", type=int, default=24)
    ap.add_argument("--trajectory-steps", type=int, default=10,
                    help="advance selected states for 10 steps under one frozen pi_old")
    ap.add_argument("--long-trajectory-steps", type=int, default=20,
                    help="occasional long rollout used to credit valley-crossing paths")
    ap.add_argument("--long-horizon-every", type=int, default=5,
                    help="use the long horizon every N optimizer cycles; 0 disables")
    ap.add_argument("--recollects", type=int, default=1,
                    help="one on-policy sibling collection per current root")
    ap.add_argument("--grpo-epochs", type=int, default=3,
                    help="GPU reuse epochs for each freshly collected batch")
    ap.add_argument("--retained", type=int, default=4)
    ap.add_argument("--target-search-steps", type=int, default=312)
    ap.add_argument("--groups-per-update", type=int, default=48,
                    help="48 root groups per GRPO update")
    ap.add_argument("--m3-microbatch-rows", type=int, default=1024)
    ap.add_argument("--m2-microbatch-rows", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="in-training rollout validation interval; 0 disables (recommended)")
    ap.add_argument("--eval-samples", type=int, default=0,
                    help="stochastic branches on ordinary validation cycles; 0=greedy only")
    ap.add_argument("--bestn-eval-every", type=int, default=50)
    ap.add_argument("--bestn-eval-samples", type=int, default=0,
                    help="0 disables expensive best-of-N during the direct-policy pilot")
    ap.add_argument("--eval-horizon", type=int, default=10)
    ap.add_argument("--max-generations", type=int, default=512)
    ap.add_argument("--max-optimizer-cycles", type=int, default=500,
                    help="hard update cap; rounded down to complete generations")
    ap.add_argument("--milestone-checkpoint-every", type=int, default=50,
                    help="save compact policy snapshot near every N updates; 0 disables")
    ap.add_argument("--target-train-instances", type=int, default=200,
                    help="final TRAIN arena size after oversized roots are replaced")
    ap.add_argument("--max-resources", type=int, default=20,
                    help="drop inherited roots above this resource/machine count")
    ap.add_argument("--max-operations", type=int, default=200,
                    help="drop inherited roots above this total operation count")
    # Kept as an internal/default source for _extra_graphs and for old direct CLI
    # compatibility. Formal training computes the refill count dynamically.
    ap.add_argument("--extra-instances", "--extra-synthetic", dest="extra_synthetic",
                    type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--extra-seed", type=int, default=22026)
    ap.add_argument("--grpo-seed", type=int, default=0)
    ap.add_argument("--probability-trace", choices=("on", "off"), default="off")
    ap.add_argument("--training-sampling", choices=("frontier", "diverse"), default="diverse")
    ap.add_argument("--state-pool-capacity", type=int, default=12)
    ap.add_argument("--state-s0-fraction", type=float, default=0.2)
    ap.add_argument("--state-elite-fraction", type=float, default=0.3,
                    help="legacy flag name: probability of latest strict incumbent best")
    ap.add_argument("--refresh-every", type=int, default=5)
    ap.add_argument("--refresh-fraction", type=float, default=0.25)
    ap.add_argument("--output-root", default=str(
        ROOT / "outputs" / "t2l_unified_intervention"),
                    help="separate smoke/experiment outputs without mixing curves")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "b5_1_shared.pt"))
    ap.add_argument(
        "--initial-bank", default=None,
        help="complete DANIEL/DRL schedule bank used as TRAIN initial states; "
             "SFT policy initialization is retained",
    )
    ap.add_argument(
        "--initial-bank-only", action="store_true",
        help="train only on retained DRL-bank roots; skip auxiliary data and refill",
    )
    ap.add_argument(
        "--expected-initial-bank-graphs", type=int, default=0,
        help="when >0, fail unless bank-only filtering leaves exactly this many roots",
    )
    ap.add_argument("--resume", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.probability_trace != "on":
        ap.error("T2-L requires the unified probability propagation path; use K for off ablation")
    if (args.state_pool_capacity < 1 or args.refresh_every < 0 or
            not 0 <= args.refresh_fraction <= 1 or
            min(args.state_s0_fraction, args.state_elite_fraction) < 0 or
            args.state_s0_fraction + args.state_elite_fraction > 1):
        ap.error("invalid state pool fractions/capacity or refresh interval")
    if min(args.workers, args.branches, args.trajectory_steps,
           args.long_trajectory_steps, args.recollects,
           args.grpo_epochs,
           args.retained, args.target_search_steps, args.groups_per_update,
           args.m3_microbatch_rows, args.m2_microbatch_rows,
           args.eval_horizon,
           args.max_optimizer_cycles,
           args.target_train_instances, args.max_resources,
           args.max_operations) < 1:
        ap.error("all T2-L budgets must be positive")
    if min(args.long_horizon_every, args.eval_every, args.bestn_eval_every,
           args.milestone_checkpoint_every) < 0:
        ap.error("intervals must be non-negative")
    if args.eval_samples < 0 or args.bestn_eval_samples < 0:
        ap.error("validation sample counts must be non-negative")
    if args.expected_initial_bank_graphs < 0:
        ap.error("expected-initial-bank-graphs must be non-negative")
    if args.initial_bank_only and not args.initial_bank:
        ap.error("--initial-bank-only requires --initial-bank")
    if args.branches < 2:
        ap.error("GRPO needs at least two sibling branches")

    GT.set_runtime(args.device)
    GT.set_microbatch_rows(args.m3_microbatch_rows, args.m2_microbatch_rows)
    old_update = JG.grpo_update_joint
    JG.grpo_update_joint = GT.structured_gpu_update_joint
    out = Path(args.output_root) / args.architecture.lower()
    out.mkdir(parents=True, exist_ok=True)
    diagnostics_root = out / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    diagnostics_dir = diagnostics_root / run_id
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_root / "latest_run.txt").write_text(run_id + "\n", encoding="utf-8")
    timing_log_path = diagnostics_root / "latest_phase_timing.jsonl"
    timing_log_path.write_text("", encoding="utf-8")

    def startup_event(stage, status, started=None, **details):
        now = time.time()
        row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "stage": stage, "status": status, "run_id": run_id, **details}
        if started is not None:
            row["seconds"] = now - started
        with timing_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        duration = (f" duration={row['seconds']:.1f}s" if "seconds" in row else "")
        print(f"[t2m] [{row['timestamp']}] {stage} {status}{duration}", flush=True)
    (diagnostics_dir / "run_config.json").write_text(json.dumps({
        "schema": "t2l-joint-macro-run-config-v2",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out),
        "diagnostics_dir": str(diagnostics_dir),
        "run_id": run_id,
        "arguments": vars(args),
    }, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        # A full resume checkpoint is written only after a completed frontier
        # generation.  The old event file can therefore contain a few cycle
        # points newer than latest.pt when a run is stopped mid-generation.
        # purge_step hides that uncommitted tail before the resumed writer
        # appends replacement points, keeping one continuous TensorBoard curve.
        purge_step = None
        if args.resume:
            resume_meta = torch.load(
                Path(args.resume), map_location="cpu", weights_only=False
            ).get("meta", {})
            purge_step = int(resume_meta.get("optimizer_update", 0)) + 1
            print(
                f"[t2m] TensorBoard resume purge_step={purge_step} "
                "(discarding only post-checkpoint event tail)",
                flush=True,
            )
        tb = SummaryWriter(str(out / "tensorboard"), purge_step=purge_step)
    except Exception as exc:  # noqa: BLE001
            print(f"[t2m] TensorBoard unavailable: {exc}", flush=True)

    try:
        startup_event("startup_environment_replay", "START")
        t0 = time.time()
        env_args = argparse.Namespace(quick=args.quick, ckpt=args.ckpt)
        env = RUN.build_env(env_args)
        re = RUN.build_replay_env(env)
        startup_event("startup_environment_replay", "END", t0)

        sft0 = time.time()
        startup_event("startup_sft_policy", "START")
        if not C.SFT_CKPT.exists():
            raise FileNotFoundError(f"Phase-1 checkpoint missing: {C.SFT_CKPT}")
        ck = torch.load(C.SFT_CKPT, map_location="cpu", weights_only=False)
        p1 = ck.get("state", {}).get("p1")
        if p1 is None:
            raise ValueError("Phase-1 checkpoint has no p1 payload")
        scorer, reranker = p1["scorer_mem"], p1["reranker"]
        parent_args = argparse.Namespace(ckpt=args.ckpt)
        r6_sel = RUN._load_r6_parent(parent_args)
        caps = RUN._t2d_train_only_caps(env, re, scorer, reranker, r6_sel, out)
        jpol = RUN._t2d_policy(args.architecture, r6_sel, caps, env)
        if args.probability_trace == "on":
            jpol.m2.enable_probability_trace()
        print(f"[t2m] probability_trace={args.probability_trace}", flush=True)
        startup_event("startup_sft_policy", "END", sft0)

        resume_frontier = None
        if args.resume:
            resume_path = Path(args.resume)
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            prior_sampling = saved.get("meta", {}).get("training_sampling", "frontier")
            if prior_sampling != args.training_sampling:
                raise ValueError("resume sampling mode differs; start a separate experiment")
            meta = saved.get("meta", {})
            if meta.get("phase") != "t2l_joint_macro_grpo_v2":
                raise ValueError(
                    "joint-macro T2-L resumes only v2 checkpoints; the action set "
                    "differs from original T2-L/K/J, so start a new run")
            if meta.get("training_config", {}).get("probability_trace", "off") != args.probability_trace:
                raise ValueError("resume probability-trace mode mismatch")
            if str(meta.get("architecture", "")).upper() != args.architecture:
                raise ValueError("resume architecture mismatch")
            jpol.load_snapshot(saved["state"]["policy_snapshot"])
            if saved["state"].get("optimizer_state") is not None:
                jpol._t2d_resume_optimizer_state = saved["state"]["optimizer_state"]
            resume_frontier = saved["state"].get("frontier_state")
            if resume_frontier is None:
                raise ValueError("resume checkpoint has no persistent frontier_state")
            print(f"[t2m] resumed policy, optimizer and frontier from {resume_path}",
                  flush=True)

        # T2-L has no GH continuation actors: M2 and M3 score once, then only
        # the sampled proposal is executed.
        jpol.params_for_stage("C")

        pool0 = time.time()
        startup_event("startup_train_pool", "START")
        bench_graphs = []
        if args.initial_bank:
            bench_graphs = _drl_initial_graphs(
                args.initial_bank, re, first_episode=500_000_000)
            print(
                f"[t2m] initial-bank={args.initial_bank} "
                f"loaded {len(bench_graphs)} DRL-generated TRAIN starts; "
                "SFT policy initialization retained, RL updates enabled",
                flush=True,
            )
        else:
            for inst in env["train_insts"]:
                iid, st = inst["instance_id"], env["states"][inst["instance_id"]]
                bench_graphs.append(RGRPO.Graph(
                    iid=iid, episode_id=env["ep_id_of"][iid], problem=st["problem"],
                    schedule=st["schedule"], progmem=copy.deepcopy(re["progmem"]),
                    src="bench", ms0=int(st["schedule"].makespan)))
        rb = sb = None
        if not args.initial_bank_only and not args.quick:
            if C.TO1_REAL_DATA.exists():
                rb = RUN._r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env,
                                         include_held=False)
            if C.TO1_AUX_DATA.exists():
                sb = RUN._r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env,
                                         include_held=False)

        inherited_graphs = (bench_graphs + (rb["graphs"] if rb else []) +
                            (sb["graphs"] if sb else []))
        # DRL banks can reuse identifiers that are also present in old replay
        # bundles.  Keep the first occurrence so bank starts take precedence.
        unique_graphs = []
        seen_iids = set()
        duplicate_iids = []
        for graph in inherited_graphs:
            if graph.iid in seen_iids:
                duplicate_iids.append(graph.iid)
                continue
            seen_iids.add(graph.iid)
            unique_graphs.append(graph)
        if duplicate_iids:
            print(
                f"[t2m] initial-pool deduplicated {len(duplicate_iids)} "
                "duplicate instance IDs (DRL bank takes precedence)",
                flush=True,
            )
        inherited_graphs = unique_graphs
        graphs, dropped_graphs = _filter_oversized_graphs(
            inherited_graphs, max_resources=args.max_resources,
            max_operations=args.max_operations)
        if args.initial_bank_only:
            if (args.expected_initial_bank_graphs > 0 and
                    len(graphs) != args.expected_initial_bank_graphs):
                raise ValueError(
                    "DRL bank-only cohort mismatch after size filtering: "
                    f"expected={args.expected_initial_bank_graphs}, actual={len(graphs)}, "
                    f"raw={len(inherited_graphs)}, dropped={len(dropped_graphs)}")
            # In bank-only mode the retained DRL cohort is the complete arena.
            # No legacy AUX roots and no synthetic refill are allowed.
            args.target_train_instances = len(graphs)
        if len(graphs) > args.target_train_instances:
            raise ValueError(
                f"target-train-instances={args.target_train_instances} is below the "
                f"{len(graphs)} eligible inherited roots")
        first_episode = max((g.episode_id for g in inherited_graphs), default=0) + 1
        refill_count = args.target_train_instances - len(graphs)
        extra = [] if args.initial_bank_only else _extra_graphs(
            args, env, re, first_episode, count=refill_count,
            id_prefix="T2L_REFILL", manifest_name="refill_manifest.json")
        graphs.extend(extra)
        if len(graphs) != args.target_train_instances:
            raise RuntimeError("TRAIN arena refill did not reach requested size")
        # Generated 16x10 JSP/FSP roots have 160 operations and remain valid.
        unexpected = [g.iid for g in graphs
                      if _graph_size(g)["resources"] > args.max_resources or
                      _graph_size(g)["operations"] > args.max_operations]
        if unexpected:
            raise RuntimeError(f"oversized roots remained after filtering: {unexpected[:5]}")
        pool_manifest = {
            "target_train_instances": args.target_train_instances,
            "initial_bank": args.initial_bank,
            "initial_bank_only": bool(args.initial_bank_only),
            "expected_initial_bank_graphs": int(args.expected_initial_bank_graphs),
            "limits": {"max_resources": args.max_resources,
                       "max_operations": args.max_operations},
            "inherited_total": len(inherited_graphs),
            "inherited_kept": len(graphs) - len(extra),
            "inherited_dropped": dropped_graphs,
            "balanced_four_family_refill": len(extra),
            "final_total": len(graphs),
        }
        manifest_dir = ROOT / args.output_root / "generated_instances"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "training_pool_manifest.json").write_text(
            json.dumps(pool_manifest, indent=2), encoding="utf-8")
        print(f"[t2m] size filter resources<={args.max_resources} "
              f"operations<={args.max_operations}: inherited={len(inherited_graphs)} "
              f"kept={len(graphs) - len(extra)} dropped={len(dropped_graphs)} "
              f"balanced_refill={len(extra)} final={len(graphs)}", flush=True)
        if dropped_graphs:
            print(f"[t2m] dropped oversized roots: "
                  f"{json.dumps(dropped_graphs, ensure_ascii=False)}", flush=True)
        startup_event("startup_train_pool", "END", pool0,
                      final_instances=len(graphs), dropped=len(dropped_graphs),
                      refill=len(extra))
        def refresh_training_arena(arena, generation):
            rng = random.Random(PF._seed(args.grpo_seed, generation, "refresh-slots"))
            families = ("jsp", "fsp", "fjsp", "hfsp")
            slots = {f: [i for i, ag in enumerate(arena) if ag.src == f"online_{f}"]
                     for f in families}
            n = min((len(s) for s in slots.values()), default=0)
            per_family = min(n, max(1, int(round(n * args.refresh_fraction)))) if args.refresh_fraction > 0 else 0
            if not per_family:
                return []
            fresh = _extra_graphs(args, env, re, 10_000_000 + generation * 100_000,
                count=4 * per_family, seed=PF._seed(args.extra_seed, args.grpo_seed, generation, "fresh-train"),
                id_prefix=f"T2L_TRAIN_REFRESH_G{generation}",
                difficulty_offset=generation % 4,
                manifest_name=f"refresh_g{generation:04d}.json")
            if any(_graph_size(g)["resources"] > args.max_resources or
                   _graph_size(g)["operations"] > args.max_operations for g in fresh):
                raise ValueError("refresh generator exceeds configured size caps")
            replacements = []
            for f in families:
                rng.shuffle(slots[f])
                replacements.extend(zip(slots[f][:per_family],
                                        [g for g in fresh if g.src == f"online_{f}"]))
            (manifest_dir / f"refresh_slots_g{generation:04d}.json").write_text(
                json.dumps([{"slot": i, "old": arena[i].iid, "new": g.iid}
                            for i, g in replacements], indent=2), encoding="utf-8")
            return replacements
        print(f"[t2m] training_sampling={args.training_sampling} state_pool={args.state_pool_capacity} "
              f"S0/best/pool={args.state_s0_fraction}/{args.state_elite_fraction}/"
              f"{1-args.state_s0_fraction-args.state_elite_fraction:.2f} "
              f"refresh_every={args.refresh_every} generations fraction={args.refresh_fraction} "
              "(online generated TRAIN slots only; active-pool gains are not a fixed-set learning curve)", flush=True)
        # A deterministic, balanced development ruler: 8 instances per family,
        # two at each difficulty.  It is disjoint from both TRAIN and formal TEST.
        valbuild0 = time.time()
        startup_event("startup_validation_pool", "START")
        validation_graphs = _extra_graphs(
            args, env, re, 1_000_000, count=32, seed=args.extra_seed + 900_000,
            id_prefix="T2L_VAL", manifest_name="validation_manifest.json")
        validation_specs = [{
            "iid": graph.iid, "episode_id": graph.episode_id,
            "problem": graph.problem, "schedule": graph.schedule,
            "progmem": graph.progmem, "src": "four_family_validation",
            "family": graph.src.removeprefix("online_")}
            for graph in validation_graphs]
        validation_cb = None
        if args.eval_every > 0:
            validation_cb = _fixed_validation_evaluator(
                env, scorer, validation_specs, horizon=args.eval_horizon,
                samples=args.eval_samples, bestn_interval=args.bestn_eval_every,
                bestn_samples=args.bestn_eval_samples)
        startup_event("startup_validation_pool", "END", valbuild0,
                      validation_instances=len(validation_specs))
        counts = {}
        for g in graphs:
            counts[g.src] = counts.get(g.src, 0) + 1
        print(f"[t2m] TRAIN arena={len(graphs)} {counts}; held/test remain gradient-sealed",
              flush=True)
        bestn_desc = (f"{args.bestn_eval_samples}+greedy every "
                      f"{args.bestn_eval_every}" if args.bestn_eval_samples > 0
                      else "disabled")
        if args.eval_every > 0:
            print(f"[t2m] in-training validation={len(validation_specs)} instances "
                  f"every {args.eval_every} optimizer steps, H={args.eval_horizon}, "
                  f"greedy every time, bestN={bestn_desc}; formal TEST untouched",
                  flush=True)
        else:
            print(f"[t2m] in-training rollout validation=DISABLED; fixed "
                  f"four-family validation manifest={len(validation_specs)} instances; "
                  "evaluate milestone checkpoints offline with equal long-horizon budgets; "
                  "formal TEST untouched", flush=True)
        mixed_roots = (len(graphs) + int(round(len(graphs) * 2.0 / 7.0)) +
                       int(round(len(graphs) * 1.0 / 7.0)))
        cycles_per_generation = ((mixed_roots + args.groups_per_update - 1) //
                                 args.groups_per_update)
        effective_cycles = (args.max_optimizer_cycles // cycles_per_generation *
                            cycles_per_generation)
        anchor_count = min(
            max(0, int(C.T2L_ANCHOR_TRAJECTORIES)), max(args.branches - 2, 0))
        pure_count = args.branches - anchor_count
        per_root = max(1, int(C.T2L_M2_SIBLINGS_PER_ROOT))
        root_strata = max(1, (pure_count + per_root - 1) // per_root)
        print(f"[t2m] START unified-net-intervention-GRPO K={args.branches} "
              f"anchors={anchor_count} pure_GRPO={pure_count} "
              f"H={args.trajectory_steps} longH={args.long_trajectory_steps}"
              f"/every{args.long_horizon_every} "
              f"recollects/root/generation={args.recollects} retain={args.retained} "
              f"GRPO_epochs={args.grpo_epochs} STOP=masked-when-feasible "
              f"M2=multi-root-topK{C.T2L_M2_ROOT_TOP_K}+multi-appearance+learned-B5-trust "
              f"credit=M2/M3-unified(bestH-{C.T2L_NET_REGRESSION_WEIGHT:g}*"
              "best_to_terminal_regression);anchor_imitation=0.05 "
              f"root_groups=pure:{root_strata}x<={per_root}+anchors:{anchor_count} "
              f"M3_first_stratified={int(C.T2L_M3_FIRST_STEP_STRATIFIED)} "
              "context_extra_replay=0 action_space=hierarchical[single|pair]->operator "
              "operators=family-masked[JSP/FSP/DJSP:SEQ_SWAP+SEQ_INSERT;"
              "FJSP/HFSP/DFJSP:eligible_ROUTE+SEQ_SWAP+SEQ_INSERT] "
              "pairs=ROUTE+ROUTE|ROUTE+SEQ_SWAP|ROUTE+SEQ_INSERT|"
              "SEQ_SWAP+SEQ_SWAP|SEQ_SWAP+SEQ_INSERT "
              f"candidate_pool=RL-top{C.T2L_M3_POOL_TOP_FRACTION:.0%}+uniform-tail "
              f"action_caps=single:{C.T2L_SINGLE_ACTIONS};pair_base:{C.T2L_BASE_PAIR_ACTIONS}/"
              f"plateau:{C.T2L_PLATEAU_PAIR_ACTIONS}@streak{C.T2L_PLATEAU_STREAK} "
              f"T_M2={C.TO1_R13_TEMP_M2} T_M3={C.TO1_R13_TEMP} "
              f"eps_M2/M3={C.TO1_R13_MIX_EPS} full_G1=off GH=off R20=off "
              f"anchor_probe=keep{C.T2L_ANCHOR_KEEP}/scan{C.T2L_ANCHOR_SCAN} "
              f"size_caps=resources<={args.max_resources},operations<={args.max_operations} "
              f"target={args.target_search_steps}+ steps groups/update={args.groups_per_update} "
              f"root_sampling={args.training_sampling} "
              f"workers={args.workers} graph_locality=on bounded_inflight=x{C.T2L_MAX_INFLIGHT_MULTIPLIER} "
              f"gpu_microbatch=M3:{args.m3_microbatch_rows}/"
              f"M2:{args.m2_microbatch_rows} max_optimizer_cycles="
              f"{args.max_optimizer_cycles} ({effective_cycles} effective at "
              f"{cycles_per_generation} cycles/generation; "
              "target-search-steps is a milestone, not a stop)", flush=True)

        # A fresh pool per collection cycle is intentional.  Persistent workers
        # saved only their small startup cost, but one native lock/extreme graph
        # analysis could poison the same pool forever.  joint_grpo now owns the
        # pool, applies a no-progress watchdog, and rebuilds it with identical
        # policy snapshots/seeds when needed.
        pool = None
        heartbeat_dir = diagnostics_dir / "worker_heartbeats"
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        os.environ["T2M_WORKER_HEARTBEAT_DIR"] = str(heartbeat_dir)
        print(
            "[t2m] worker_pool=per-cycle-watchdog "
            f"stall_timeout={os.environ.get('T2M_ROLLOUT_STALL_TIMEOUT_S', '900')}s "
            f"retries={os.environ.get('T2M_ROLLOUT_STALL_RETRIES', '2')} "
            f"heartbeats={heartbeat_dir}",
            flush=True,
        )

        milestone_dir = out / "milestone_policies"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        saved_milestone_updates = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in milestone_dir.glob("policy_update_*.pt")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        }

        def checkpoint(frontier_state, generation):
            cycle_rows = frontier_state.get("cycle_history", ())
            optimizer_update = max(
                (int(row.get("cycle", 0)) for row in cycle_rows), default=0)
            optimizer = GT.optimizer_state_cpu(jpol)
            payload = {
                "state": {"policy_snapshot": jpol.snapshot(),
                          "optimizer_state": optimizer,
                          "r6_anchor": r6_sel.state_dict(),
                          "frontier_state": frontier_state},
                "meta": {"phase": "t2l_joint_macro_grpo_v2",
                         "training_sampling": args.training_sampling,
                         "training_config": vars(args),
                         "architecture": args.architecture,
                         "formal_test_access": 0, "generation": generation,
                         "optimizer_update": optimizer_update,
                         "K": args.branches, "H": args.trajectory_steps,
                         "long_H": args.long_trajectory_steps,
                         "long_horizon_every": args.long_horizon_every,
                         "recollects": args.recollects,
                         "grpo_epochs": args.grpo_epochs,
                         "retained": args.retained,
                         "target_search_steps": args.target_search_steps,
                         "groups_per_update": args.groups_per_update,
                         "m3_microbatch_rows": args.m3_microbatch_rows,
                         "m2_microbatch_rows": args.m2_microbatch_rows,
                         "evaluation_interval": args.eval_every,
                         "evaluation_samples": args.eval_samples,
                         "bestn_evaluation_interval": args.bestn_eval_every,
                         "bestn_evaluation_samples": args.bestn_eval_samples,
                         "max_optimizer_cycles": args.max_optimizer_cycles,
                         "target_train_instances": args.target_train_instances,
                         "max_resources": args.max_resources,
                         "max_operations": args.max_operations,
                         "dropped_oversized_instances": dropped_graphs,
                         "balanced_refill_instances": len(extra),
                         "evaluation_horizon": args.eval_horizon,
                         "validation_instances": len(validation_specs),
                         "train_graphs": len(graphs),
                         "optimizer_persistent": True}}
            tmp = out / "latest.pt.tmp"
            torch.save(payload, tmp)
            tmp.replace(out / "latest.pt")
            interval = int(args.milestone_checkpoint_every)
            crossed = (optimizer_update > 0 and interval > 0 and
                       optimizer_update // interval >
                       max((value // interval for value in saved_milestone_updates),
                           default=0))
            if crossed:
                compact = {
                    "state": {"policy_snapshot": jpol.snapshot()},
                    "meta": {key: value for key, value in payload["meta"].items()
                             if key not in ("dropped_oversized_instances",)},
                }
                target = milestone_dir / f"policy_update_{optimizer_update:04d}.pt"
                compact_tmp = target.with_suffix(".pt.tmp")
                torch.save(compact, compact_tmp)
                compact_tmp.replace(target)
                saved_milestone_updates.add(optimizer_update)
                print(f"[t2m] saved offline-validation policy milestone: {target}",
                      flush=True)

        try:
            try:
                training0 = time.time()
                startup_event("training", "START")
                result = PF.run(
                    jpol, scorer, env, graphs, workers=args.workers,
                    branches=args.branches, horizon=args.trajectory_steps,
                    recollects=args.recollects, elites=args.retained,
                    grpo_epochs=args.grpo_epochs,
                    long_horizon=args.long_trajectory_steps,
                    long_horizon_every=args.long_horizon_every,
                    target_steps=args.target_search_steps,
                    groups_per_update=args.groups_per_update,
                    max_generations=args.max_generations,
                    max_optimizer_cycles=args.max_optimizer_cycles,
                    seed=args.grpo_seed,
                    tensorboard_writer=tb, rollout_pool=pool,
                    optimizer_persistent=True, checkpoint_cb=checkpoint,
                    resume_state=resume_frontier, evaluation_cb=validation_cb,
                    evaluation_interval=args.eval_every, log_prefix="[t2m]",
                    diagnostics_dir=diagnostics_dir,
                    timing_log_path=timing_log_path,
                    diverse_training=args.training_sampling == "diverse",
                    state_pool_capacity=args.state_pool_capacity,
                    s0_fraction=args.state_s0_fraction,
                    elite_fraction=args.state_elite_fraction,
                    refresh_every=args.refresh_every,
                    refresh_cb=refresh_training_arena)
                startup_event("training", "END", training0)
            except Exception:  # noqa: BLE001
                if "training0" in locals():
                    startup_event("training", "FAILED", training0)
                crash_path = diagnostics_dir / "crash_latest.txt"
                crash_text = traceback.format_exc()
                crash_path.write_text(crash_text, encoding="utf-8")
                (diagnostics_root / "crash_latest.txt").write_text(
                    crash_text, encoding="utf-8")
                print(f"[t2m] crash traceback saved to {crash_path}", flush=True)
                raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        summary = PF.write_summary(result, out / "result.json")
        print(f"[t2m] FINISHED complete={result['complete']} "
              f"generation={result['generation']} graphs={len(summary['graphs'])} "
              f"result={out / 'result.json'}", flush=True)
        return 0 if result["complete"] else 2
    finally:
        if tb is not None:
            tb.flush()
            tb.close()
        JG.grpo_update_joint = old_update


if __name__ == "__main__":
    raise SystemExit(main())
