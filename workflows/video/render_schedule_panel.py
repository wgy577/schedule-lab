#!/usr/bin/env python3
"""Render one schedule panel for the low-load comparison-video pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REPOSITORY_ROOT = HERE.parents[2]
DECK_UPDATE_ROOT = REPOSITORY_ROOT.parent / "deck_update"
for path in (HERE, REPOSITORY_ROOT, DECK_UPDATE_ROOT):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

_renderer_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from Params import configs
from render_schedule_json import load_schedule
from video_viz import (
    attach_real_dispatch_trajectories,
    draw_gantt,
    draw_real_dispatch_deck,
    job_colors,
    save_mp4,
    setup_style,
)
sys.argv = _renderer_argv


DEFAULT_SCHEDULE = PROJECT_ROOT / "outputs/carrier_alns_best_iter3_gap6_closed_630_5.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/videos/.carrier_schedule_panel.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one schedule comparison panel")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=6, help="Internal render fps; final assembly is 30 fps")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument(
        "--timeline-end",
        type=float,
        help="Shared simulated end time used by both comparison panels",
    )
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--bitrate", type=int, default=1500)
    parser.add_argument("--header", default="")
    parser.add_argument("--footer", default="")
    parser.add_argument("--gantt-label", default="Schedule")
    parser.add_argument("--deck-label", default="Carrier deck dispatch")
    args = parser.parse_args()

    setup_style()
    schedule, metadata = load_schedule(args.schedule.expanduser().resolve())
    schedule = attach_real_dispatch_trajectories(schedule, launch_len=120.0)
    makespan = max(float(item["end"]) for item in schedule)
    expected = float(metadata.get("trueMakespan", makespan))
    if abs(makespan - expected) > 1e-3:
        raise ValueError(f"schedule makespan mismatch: {makespan} != {expected}")

    fps = max(1, args.fps)
    timeline_end = makespan if args.timeline_end is None else float(args.timeline_end)
    if timeline_end + 1e-6 < makespan:
        raise ValueError(f"shared timeline {timeline_end} cannot end before schedule {makespan}")
    frames = max(2, int(round(fps * max(1.0, args.seconds))))
    colors = job_colors(configs.n_j)

    # Exactly half of the reference video's 1920x816 frame.
    figure = plt.figure(figsize=(9.6, 8.16), dpi=args.dpi, facecolor="white")
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.55, 0.95],
        # Match the established 675.5-vs-637.5 comparison: enough room for
        # complete machine labels, with identical plot widths on both sides.
        left=0.125,
        right=0.94,
        top=0.895,
        bottom=0.12,
        hspace=0.27,
    )
    gantt_ax = figure.add_subplot(grid[0, 0])
    deck_ax = figure.add_subplot(grid[1, 0])
    figure.add_artist(
        Rectangle(
            (0, 0.925),
            1,
            0.075,
            transform=figure.transFigure,
            facecolor="#123f2f",
            edgecolor="none",
            zorder=-1,
        )
    )
    figure.text(
        0.5,
        0.963,
        args.header or f"SCHEDULE · TRUE CMAX {makespan:.1f}",
        ha="center",
        va="center",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        args.footer or f"True makespan {makespan:.1f} s · WGY",
        ha="center",
        va="center",
        color="#20302a",
        fontsize=10.5,
        fontweight="bold",
    )

    def update(frame: int):
        shared_time = timeline_end * min(frame, frames - 1) / max(1, frames - 1)
        cursor = min(shared_time, makespan)
        draw_gantt(
            gantt_ax,
            schedule,
            colors,
            cursor=cursor,
            makespan=timeline_end,
            title=f"{args.gantt_label} · shared t={shared_time:.1f}/{timeline_end:.1f} s",
        )
        draw_real_dispatch_deck(
            deck_ax,
            schedule,
            cursor,
            colors,
            title=(
                f"{args.deck_label} · completed at {makespan:.1f} s"
                if shared_time > makespan
                else f"{args.deck_label} · t={cursor:.1f} s"
            ),
            show_trail=False,
        )
        return []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"rendering panel: makespan={makespan:.3f}, timeline_end={timeline_end:.3f}, frames={frames}, "
        f"internal_fps={fps}, resolution=960x816"
    )
    save_mp4(
        figure,
        update,
        frames=frames,
        out_path=str(args.output),
        fps=fps,
        dpi=args.dpi,
        bitrate=args.bitrate,
    )
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
