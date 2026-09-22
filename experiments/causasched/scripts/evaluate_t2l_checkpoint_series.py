#!/usr/bin/env python3
"""Offline, equal-budget long-horizon validation for T2-L policy checkpoints.

This deliberately does not touch the sealed formal TEST split.  It evaluates
the SFT-informed initial policy, compact milestone policies, and latest policy
on the same generated validation instances with the same branch seeds.  The
reported objective is best makespan visited within H, never terminal makespan.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from causal_schedule_lab.m3 import config as C  # noqa: E402
from causal_schedule_lab.m3 import joint_grpo as JG  # noqa: E402
from causal_schedule_lab.m3 import t2d_gpu_trainer as GT  # noqa: E402
import run_m3_canonical_training as RUN  # noqa: E402
from run_t2l_unified_intervention_grpo_gpu import _extra_graphs  # noqa: E402


def _checkpoint_series(checkpoint_root: Path):
    rows = [(0, "sft", None)]
    seen = {0}
    for path in sorted((checkpoint_root / "milestone_policies").glob(
            "policy_update_*.pt")):
        match = re.search(r"(\d+)$", path.stem)
        if not match:
            continue
        update = int(match.group(1))
        if update not in seen:
            rows.append((update, "milestone", path))
            seen.add(update)
    latest = checkpoint_root / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        update = int(payload.get("meta", {}).get("optimizer_update", 0))
        if update <= 0:
            update = max((int(row.get("cycle", 0)) for row in
                          payload.get("state", {}).get("frontier_state", {}).get(
                              "cycle_history", ())), default=0)
        rows = [row for row in rows if row[0] != update]
        rows.append((update, "final", latest))
    return sorted(rows, key=lambda row: (row[0], row[1]))


def _summary(groups, roots, families):
    best = np.asarray([
        max(float(tr.get("best_reward", 0.0)) for tr in group["trajs"])
        for group in groups], dtype=np.float64)
    mean_traj_best = np.asarray([
        np.mean([float(tr.get("best_reward", 0.0)) for tr in group["trajs"]])
        for group in groups], dtype=np.float64)
    root_ms = np.asarray([float(root.ms) for root in roots], dtype=np.float64)
    normalized = 100.0 * best / np.maximum(root_ms, 1.0)
    all_traj = [tr for group in groups for tr in group["trajs"]]
    row = {
        "n_instances": int(len(roots)),
        "bestn_normalized_mean_pct": float(normalized.mean()),
        "bestn_normalized_median_pct": float(np.median(normalized)),
        "bestn_raw_gain_mean": float(best.mean()),
        "bestn_mean_makespan": float(np.mean(root_ms - best)),
        "bestn_improved_graph_rate": float(np.mean(best > 0)),
        "trajectory_positive_rate": float(np.mean([
            float(tr.get("best_reward", 0.0)) > 0 for tr in all_traj])),
        "mean_trajectory_best_gain": float(mean_traj_best.mean()),
    }
    fam_arr = np.asarray(families)
    for family in sorted(set(families)):
        mask = fam_arr == family
        key = family.lower()
        row[f"{key}_normalized_mean_pct"] = float(normalized[mask].mean())
        row[f"{key}_improved_graph_rate"] = float(np.mean(best[mask] > 0))
        row[f"{key}_mean_makespan"] = float(np.mean(root_ms[mask] - best[mask]))
    return row


def _plot(rows, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["optimizer_update"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(x, [row["bestn_normalized_mean_pct"] for row in rows],
                 marker="o", linewidth=2.2, label="overall mean")
    axes[0].plot(x, [row["bestn_normalized_median_pct"] for row in rows],
                 marker="s", linewidth=1.6, label="overall median")
    for family in ("jsp", "fsp", "fjsp", "hfsp"):
        key = f"{family}_normalized_mean_pct"
        if all(key in row for row in rows):
            axes[0].plot(x, [row[key] for row in rows], linewidth=1.2,
                         alpha=0.75, label=family.upper())
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Optimizer updates",
                ylabel="Best-within-H makespan reduction (%)",
                title="Fixed validation: equal-budget long-horizon search")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [row["bestn_improved_graph_rate"] for row in rows],
                 marker="o", linewidth=2.2, label="improved graph rate")
    axes[1].plot(x, [row["trajectory_positive_rate"] for row in rows],
                 marker="s", linewidth=1.6, label="positive trajectory rate")
    axes[1].set(xlabel="Optimizer updates", ylabel="Rate", ylim=(-0.02, 1.02),
                title="Search success on the same validation instances")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--instances", type=int, default=32)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=922026)
    ap.add_argument("--architecture", choices=("P0", "P1", "P2"), default="P2")
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/b5_1_shared.pt"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    if min(args.instances, args.branches, args.horizon, args.workers) < 1:
        ap.error("instances, branches, horizon and workers must be positive")
    checkpoint_root = Path(args.checkpoint_root)
    output = Path(args.output) if args.output else checkpoint_root / "paper_validation"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "instances": args.instances, "branches": args.branches,
        "horizon": args.horizon, "workers": args.workers, "seed": args.seed,
        "objective": "best_makespan_visited_within_horizon",
        "policy_mode": "pure_policy_sampled_no_anchor",
        "formal_test_access": 0,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    GT.set_runtime(args.device)
    print("[paper-val] loading frozen environment/replay once ...", flush=True)
    env = RUN.build_env(argparse.Namespace(quick=False, ckpt=args.ckpt))
    replay_env = RUN.build_replay_env(env)
    phase1 = torch.load(C.SFT_CKPT, map_location="cpu", weights_only=False)
    p1 = phase1.get("state", {}).get("p1")
    if p1 is None:
        raise ValueError("Phase-1 checkpoint has no p1 payload")
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    r6_sel = RUN._load_r6_parent(argparse.Namespace(ckpt=args.ckpt))
    caps = RUN._t2d_train_only_caps(
        env, replay_env, scorer, reranker, r6_sel, checkpoint_root)
    jpol = RUN._t2d_policy(args.architecture, r6_sel, caps, env)
    jpol.m2.enable_probability_trace()
    jpol.params_for_stage("C")
    initial_snapshot = jpol.snapshot()

    val_args = argparse.Namespace(
        extra_synthetic=args.instances, extra_seed=args.seed,
        output_root=str(output.relative_to(ROOT) if output.is_relative_to(ROOT)
                        else output))
    graphs = _extra_graphs(
        val_args, env, replay_env, 1_000_000, count=args.instances,
        seed=args.seed, id_prefix="T2L_VAL",
        manifest_name="paper_validation_manifest.json")
    roots = [SimpleNamespace(
        iid=graph.iid, episode_id=graph.episode_id, problem=graph.problem,
        schedule=graph.schedule, ms=int(graph.schedule.makespan), gstep=0,
        progmem=copy.deepcopy(graph.progmem), capture_visual_plans=False)
        for graph in graphs]
    families = [graph.src.removeprefix("online_") for graph in graphs]

    pool = JG.make_graph_rollout_pool(
        jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
        env["direct_head"], workers=args.workers, mp_ctx=None, val_cache={},
        critical_gate=False, torch_threads=1)
    rows = []
    try:
        for update, label, path in _checkpoint_series(checkpoint_root):
            if path is None:
                jpol.load_snapshot(initial_snapshot)
            else:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                jpol.load_snapshot(saved["state"]["policy_snapshot"])
            started = time.time()
            print(f"[paper-val] START update={update} label={label} "
                  f"instances={args.instances} K={args.branches} H={args.horizon}",
                  flush=True)

            def progress(done, total, elapsed, **kwargs):
                if done == total or done == 1 or done % max(1, total // 8) == 0:
                    print(f"[paper-val] update={update} jobs={done}/{total} "
                          f"elapsed={elapsed:.1f}s", flush=True)

            groups = JG.collect_depth_groups_rollouts_r14(
                jpol, scorer, env["executor"], env["model_b5"],
                env["single_head"], env["direct_head"], roots,
                k=args.branches, horizon=args.horizon, seed=args.seed,
                workers=args.workers, step0_cache=env.get("cache"),
                action_space="policy_sampled", val_cache={}, critical_gate=False,
                torch_threads=1, rollout_pool=pool, stop_on_negative=False,
                allow_policy_stop=False, feasible_fallback=True,
                progress_callback=progress, anchor_trajectories=0)
            row = _summary(groups, roots, families)
            row.update({"optimizer_update": int(update), "checkpoint_kind": label,
                        "checkpoint": "SFT" if path is None else str(path),
                        "branches": args.branches, "horizon": args.horizon,
                        "seconds": time.time() - started})
            rows.append(row)
            result_path = output / f"update_{update:04d}_{label}.json"
            result_path.write_text(json.dumps(row, indent=2, sort_keys=True),
                                   encoding="utf-8")
            print(f"[paper-val] END update={update} "
                  f"bestH={row['bestn_normalized_mean_pct']:+.3f}% "
                  f"improved={row['bestn_improved_graph_rate']:.3f} "
                  f"sec={row['seconds']:.1f}", flush=True)
    finally:
        pool.shutdown(wait=True)

    keys = sorted({key for row in rows for key in row})
    with (output / "checkpoint_validation_curve.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (output / "checkpoint_validation_curve.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"))
        for row in rows:
            step = int(row["optimizer_update"])
            for key, value in row.items():
                if key in ("optimizer_update", "checkpoint_kind", "checkpoint"):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    writer.add_scalar(f"paper_validation/{key}", float(value), step)
        writer.flush()
        writer.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[paper-val] TensorBoard export unavailable: {exc}", flush=True)
    _plot(rows, output / "checkpoint_validation_curve.png")
    print(f"[paper-val] curve: {output / 'checkpoint_validation_curve.png'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
