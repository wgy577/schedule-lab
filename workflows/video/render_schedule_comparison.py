#!/usr/bin/env python3
"""Reusable equal-panel carrier schedule comparison template.

Final delivery contract (kept intentionally stable):
  * 1920x816
  * 30 fps
  * 60 seconds
  * H.264 / yuv420p

Each side is rendered by the exact same 960x816 panel template.  A small cache
keeps unchanged schedules from being rendered again on later comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TEMPLATE_VERSION = "equal-panel-v3-shared-clock"
DEFAULT_LEFT = PROJECT_ROOT / "outputs/carrier_best_schedule_traced.json"
DEFAULT_RIGHT = PROJECT_ROOT / "outputs/carrier_alns_best_iter3_gap6_closed_630_5.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/videos/carrier_schedule_comparison_637_5_vs_627_8.mp4"


def schedule_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schedule = data.get("schedule", data.get("operations", []))
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"no schedule assignments found in {path}")
    return schedule


def schedule_hash(path: Path) -> str:
    normalized = [
        {key: item[key] for key in ("job", "op", "machine", "start", "dur", "end")}
        for item in sorted(schedule_items(path), key=lambda item: (int(item["job"]), int(item["op"])))
    ]
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def true_makespan(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", data)
    if "trueMakespan" in metadata:
        return float(metadata["trueMakespan"])
    schedule = schedule_items(path)
    return max(float(item["end"]) for item in schedule)


def comparison_manifest(left: Path, right: Path) -> dict:
    left_schedule = schedule_items(left)
    right_schedule = schedule_items(right)
    left_by_operation = {(int(item["job"]), int(item["op"])): item for item in left_schedule}
    right_by_operation = {(int(item["job"]), int(item["op"])): item for item in right_schedule}
    if set(left_by_operation) != set(right_by_operation):
        raise ValueError("left and right schedules do not contain the same operation identities")
    left_hash = schedule_hash(left)
    right_hash = schedule_hash(right)
    if left_hash == right_hash:
        raise ValueError("comparison refused: left and right schedules are identical")
    machine_changes = []
    timing_changes = 0
    for operation_id in sorted(left_by_operation):
        before = left_by_operation[operation_id]
        after = right_by_operation[operation_id]
        if int(before["machine"]) != int(after["machine"]):
            machine_changes.append(
                {
                    "job": operation_id[0],
                    "op": operation_id[1],
                    "leftMachine": int(before["machine"]),
                    "rightMachine": int(after["machine"]),
                }
            )
        if abs(float(before["start"]) - float(after["start"])) > 1e-6 or abs(
            float(before["end"]) - float(after["end"])
        ) > 1e-6:
            timing_changes += 1
    return {
        "templateVersion": TEMPLATE_VERSION,
        "left": {
            "path": str(left),
            "scheduleHash": left_hash,
            "trueMakespan": true_makespan(left),
        },
        "right": {
            "path": str(right),
            "scheduleHash": right_hash,
            "trueMakespan": true_makespan(right),
        },
        "difference": {
            "machineBindingChangeCount": len(machine_changes),
            "timingChangeCount": timing_changes,
            "machineBindingChanges": machine_changes,
        },
    }


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def panel_cache_path(
    cache_dir: Path,
    schedule: Path,
    fps: int,
    seconds: float,
    timeline_end: float,
) -> Path:
    stamp = schedule.stat().st_mtime_ns
    return cache_dir / (
        f"{TEMPLATE_VERSION}_{schedule.stem}_{stamp}_{fps}fps_"
        f"{seconds:g}s_t{timeline_end:.3f}.mp4"
    )


def render_panel(
    schedule: Path,
    output: Path,
    fps: int,
    seconds: float,
    timeline_end: float,
    header: str,
    footer: str,
    gantt_label: str,
    deck_label: str,
) -> None:
    run(
        [
            "python3",
            str(HERE / "render_schedule_panel.py"),
            "--schedule",
            str(schedule),
            "--output",
            str(output),
            "--fps",
            str(fps),
            "--seconds",
            str(seconds),
            "--timeline-end",
            str(timeline_end),
            "--header",
            header,
            "--footer",
            footer,
            "--gantt-label",
            gantt_label,
            "--deck-label",
            deck_label,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an equal-size two-schedule comparison")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--final-fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="Ignore cached panel videos")
    args = parser.parse_args()

    left = args.left.expanduser().resolve()
    right = args.right.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = comparison_manifest(left, right)
    left_ms = manifest["left"]["trueMakespan"]
    right_ms = manifest["right"]["trueMakespan"]
    delta = left_ms - right_ms
    pct = 100.0 * delta / left_ms
    timeline_end = max(left_ms, right_ms)

    cache_dir = PROJECT_ROOT / "outputs/videos/.comparison_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    left_panel = panel_cache_path(cache_dir, left, args.render_fps, args.seconds, timeline_end)
    right_panel = panel_cache_path(cache_dir, right, args.render_fps, args.seconds, timeline_end)

    if args.force or not left_panel.exists():
        render_panel(
            left,
            left_panel,
            args.render_fps,
            args.seconds,
            timeline_end,
            f"PREVIOUS VALIDATED · TRUE CMAX {left_ms:.1f}",
            f"Previous incumbent · {left_ms:.1f} s · WGY",
            "Previous validated Gantt chart",
            "Previous validated deck dispatch",
        )
    else:
        print(f"reuse cached panel: {left_panel}")

    if args.force or not right_panel.exists():
        render_panel(
            right,
            right_panel,
            args.render_fps,
            args.seconds,
            timeline_end,
            f"OPTIMIZED · TRUE CMAX {right_ms:.1f}",
            f"{right_ms:.1f} s · −{delta:.1f} s / −{pct:.2f}% vs {left_ms:.1f} s · WGY",
            "Optimized Gantt chart",
            "Optimized deck dispatch",
        )
    else:
        print(f"reuse cached panel: {right_panel}")

    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(left_panel),
            "-i",
            str(right_panel),
            "-filter_complex",
            (
                f"[0:v]fps={args.final_fps},scale=960:816:flags=lanczos[l];"
                f"[1:v]fps={args.final_fps},scale=960:816:flags=lanczos[r];"
                "[l][r]hstack=inputs=2,"
                "drawbox=x=958:y=0:w=4:h=816:color=white@0.85:t=fill[v]"
            ),
            "-map",
            "[v]",
            "-t",
            str(args.seconds),
            "-an",
            "-r",
            str(args.final_fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-b:v",
            "2800k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    manifest.update(
        {
            "output": str(output),
            "video": {
                "width": 1920,
                "height": 816,
                "finalFps": args.final_fps,
                "internalRenderFps": args.render_fps,
                "seconds": args.seconds,
                "sharedTimelineEnd": timeline_end,
                "rightCompletedWaitScheduleSeconds": max(0.0, left_ms - right_ms),
                "simulatedSecondsPerVideoSecond": timeline_end / args.seconds,
                "codec": "H.264",
                "pixelFormat": "yuv420p",
            },
        }
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(output)


if __name__ == "__main__":
    main()
