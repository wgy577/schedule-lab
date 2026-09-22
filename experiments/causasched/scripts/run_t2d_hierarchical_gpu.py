#!/usr/bin/env python3
"""AutoDL entry point for T2-D P0/P1/P2 same-budget runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from causal_schedule_lab.m3 import joint_grpo as JG  # noqa: E402
from causal_schedule_lab.m3 import t2d_gpu_trainer as GT  # noqa: E402
from causal_schedule_lab.m3 import config as C  # noqa: E402
import run_m3_canonical_training as RUN  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="T2-D hierarchical residual GPU runner")
    ap.add_argument("--architecture", choices=["P0", "P1", "P2"], default="P2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--workers", type=int, default=16,
                    help="safe upper bound; a 1/2/4/8/16 profile selects the fastest")
    ap.add_argument("--grpo-seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=C.T2D_TRAINING_CYCLES)
    ap.add_argument("--graphs-per-cycle", type=int, default=C.T2D_GRAPHS_PER_CYCLE)
    ap.add_argument("--branches-per-graph", type=int,
                    default=C.T2D_BRANCHES_PER_GRAPH)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "b5_1_shared.pt"))
    ap.add_argument("--resume", default=None,
                    help="resume from outputs/t2d/<arch>/latest.pt")
    ap.add_argument("--optimizer-reset", action="store_true",
                    help="ablation only; default keeps AdamW moments across updates")
    ap.add_argument("--full-diagnostics", action="store_true",
                    help="run legacy Phase-1/E0/held/M5 diagnostics before/after training")
    ap.add_argument("--profile-workers", action="store_true",
                    help="benchmark 1/2/4/8/16 before training; default starts immediately")
    args = ap.parse_args(argv)
    if args.cycles < 1 or args.graphs_per_cycle < 1 or args.branches_per_graph < 2:
        ap.error("cycles/graphs must be positive and branches-per-graph must be >=2")
    GT.set_runtime(args.device)
    original = JG.grpo_update_joint
    JG.grpo_update_joint = GT.structured_gpu_update_joint
    forwarded = ["--stage", "t2d", "--architecture", args.architecture,
                 "--workers", str(args.workers), "--grpo-seed", str(args.grpo_seed),
                 "--load-p1", "--t2d-cycles", str(args.cycles),
                 "--t2d-graphs-per-cycle", str(args.graphs_per_cycle),
                 "--t2d-branches-per-graph", str(args.branches_per_graph)]
    if not args.optimizer_reset:
        forwarded.append("--optimizer-persistent")
    if args.quick:
        forwarded.append("--quick")
    if args.full_diagnostics:
        forwarded.append("--full-diagnostics")
    if args.profile_workers:
        forwarded.append("--profile-workers")
    forwarded.extend(["--ckpt", args.ckpt])
    if args.resume:
        forwarded.extend(["--resume", args.resume])
    try:
        return RUN.main(forwarded)
    finally:
        JG.grpo_update_joint = original


if __name__ == "__main__":
    raise SystemExit(main())
