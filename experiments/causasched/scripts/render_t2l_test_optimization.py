#!/usr/bin/env python3
"""Render best real T2-L score-sampled trajectories on held validation graphs."""
from __future__ import annotations

import argparse
import copy
import html
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from causal_schedule_lab.m3 import config as C  # noqa: E402
from causal_schedule_lab.m3 import joint_grpo as JG  # noqa: E402
from causal_schedule_lab.m3 import t2d_gpu_trainer as GT  # noqa: E402
from causal_schedule_lab.m3.optimization_trace import write_optimization_trace  # noqa: E402
import run_m3_canonical_training as RUN  # noqa: E402
from run_t2l_unified_intervention_grpo_gpu import _extra_graphs  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize the best T2-L score-sampled path on held graphs")
    parser.add_argument("--checkpoint", default=str(
        ROOT / "outputs/t2l_unified_intervention/p2/latest.pt"))
    parser.add_argument("--architecture", choices=("P0", "P1", "P2"), default="P2")
    parser.add_argument("--instances", type=int, default=4,
                        help="first four are JSP/FSP/FJSP/HFSP (recommended: 4)")
    parser.add_argument("--branches", type=int, default=24)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=922026)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints/b5_1_shared.pt"))
    parser.add_argument("--output", default=str(
        ROOT / "outputs/t2l_unified_intervention/test_optimization_traces"))
    args = parser.parse_args(argv)
    if min(args.instances, args.branches, args.steps, args.workers) < 1:
        parser.error("instances, branches, steps and workers must be positive")
    if args.branches < 2:
        parser.error("best-of-N/GRPO comparison needs at least two branches")
    use_sft_parent = str(args.checkpoint).strip().lower() == "sft"
    checkpoint = None if use_sft_parent else Path(args.checkpoint)
    if checkpoint is not None and not checkpoint.exists():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    GT.set_runtime(args.device)
    print("[trace] loading frozen environment and trained policy ...", flush=True)
    env_args = argparse.Namespace(quick=False, ckpt=args.ckpt)
    env = RUN.build_env(env_args)
    replay_env = RUN.build_replay_env(env)
    phase1 = torch.load(C.SFT_CKPT, map_location="cpu", weights_only=False)
    p1 = phase1.get("state", {}).get("p1")
    if p1 is None:
        raise ValueError("Phase-1 checkpoint has no p1 payload")
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    r6_sel = RUN._load_r6_parent(argparse.Namespace(ckpt=args.ckpt))
    out_root = Path(args.output)
    caps = RUN._t2d_train_only_caps(env, replay_env, scorer, reranker, r6_sel,
                                    ROOT / "outputs/t2l_unified_intervention/p2")
    jpol = RUN._t2d_policy(args.architecture, r6_sel, caps, env)
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        jpol.load_snapshot(saved["state"]["policy_snapshot"])
    else:
        saved = {"meta": {"generation": -1}}
        print("[trace] using the frozen SFT-informed parent (no GRPO snapshot)",
              flush=True)

    cont_m2 = JG.M2RootPolicyAdapter(C.TO1_R13_M2_FEAT_DIM,
                                     float(C.TO1_R13_ALPHA_M2))
    for parameter in cont_m2.parameters():
        parameter.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(
        r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP, alpha_stop=C.TO1_R13_ALPHA_STOP)
    for parameter in cont_m3.parameters():
        parameter.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    jpol.params_for_stage("C")

    # This is the same fixed, disjoint four-family validation generator used by
    # training evaluation, not a TRAIN root and not the sealed formal TEST set.
    validation = _extra_graphs(
        argparse.Namespace(extra_synthetic=args.instances, extra_seed=args.seed),
        env, replay_env, 2_000_000, count=args.instances, seed=args.seed,
        id_prefix="T2F_TRACE", manifest_name="trace_manifest.json")
    roots = [SimpleNamespace(
        iid=graph.iid, episode_id=graph.episode_id, problem=graph.problem,
        schedule=graph.schedule, ms=int(graph.schedule.makespan), gstep=0,
        progmem=copy.deepcopy(graph.progmem), capture_visual_plans=True)
        for graph in validation]

    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = None
    try:
        if args.workers > 1:
            pool = JG.make_graph_rollout_pool(
                jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
                env["direct_head"], workers=args.workers, mp_ctx=None,
                val_cache={}, critical_gate=False, torch_threads=1)
        groups = JG.collect_depth_groups_rollouts_r14(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], roots, k=args.branches, horizon=args.steps,
            seed=args.seed, workers=args.workers, step0_cache=env.get("cache"),
            action_space="policy_sampled", val_cache={}, critical_gate=False,
            torch_threads=1, rollout_pool=pool, stop_on_negative=False,
            allow_policy_stop=False, feasible_fallback=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    cards = []
    for root, group in zip(roots, groups):
        # Exactly what the user asked for: the trajectory with the largest
        # makespan reduction; deterministic ties prefer more executed steps.
        best = min(group["trajs"], key=lambda tr: (
            min([root.ms] + [int(p["after_ms"]) for p in tr.get("_visual_plan") or ()]),
            int(tr["final_ms"]), int(tr["traj_id"])))
        graph_dir = run_dir / root.iid
        payload = write_optimization_trace(
            root.problem, env["executor"], root.schedule, best, graph_dir,
            generation=int(saved.get("meta", {}).get("generation", -1)),
            cycle=-1, root_kind="held_validation")
        print(f"[trace] {root.iid}: best {args.branches} branches, "
              f"Cmax {payload['initial_makespan']} -> {payload['final_makespan']} "
              f"gain={payload['total_improvement']:+d} actions={payload['n_actions']} "
              f"view={graph_dir / 'index.html'}", flush=True)
        cards.append(
            f'<li><a href="{html.escape(root.iid)}/index.html">{html.escape(root.iid)}</a>'
            f'：Cmax {payload["initial_makespan"]} → {payload["final_makespan"]}，'
            f'累计优化 {payload["total_improvement"]:+d}</li>')

    gallery = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>T2L 测试优化轨迹</title><style>body{{max-width:980px;margin:40px auto;padding:0 20px;font:17px system-ui;color:#0f172a}}li{{margin:15px 0}}a{{color:#2563eb}}</style>
<h1>T2L 验证实例：最优优化轨迹</h1><p>每个实例从 {args.branches} 条轨迹中按历史最低 makespan（包含S0）选择最好的一条；每条最多 {args.steps} 步。此工具包含训练式锚点，不能作为纯策略 best-of-N 验证指标。</p><ul>{''.join(cards)}</ul></html>"""
    (run_dir / "index.html").write_text(gallery, encoding="utf-8")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_trace.txt").write_text(stamp + "\n", encoding="utf-8")
    print(f"[trace] gallery: {run_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
