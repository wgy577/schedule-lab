from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .model import Problem, Schedule


def render_gantt(problem: Problem, schedule: Schedule, path: str | Path, *, title: str | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode_map = problem.mode_map()
    resource_index = {resource.id: index for index, resource in enumerate(problem.resources)}
    jobs = sorted({operation.job_id for operation in problem.operations})
    colors = {job: plt.cm.tab20(index % 20) for index, job in enumerate(jobs)}
    operation_map = problem.operation_map()
    figure_height = max(4.2, len(problem.resources) * 0.48)
    fig, ax = plt.subplots(figsize=(13, figure_height), dpi=150)
    for assignment in schedule.assignments:
        operation = operation_map[assignment.operation_id]
        mode = mode_map[assignment.mode_id][1]
        for resource_id in mode.resources:
            y = resource_index[resource_id]
            left = assignment.start / problem.time_scale
            width = (assignment.end - assignment.start) / problem.time_scale
            ax.barh(y, width, left=left, height=0.68, color=colors[operation.job_id], edgecolor="white", linewidth=0.45)
            if width >= 4:
                ax.text(left + width / 2, y, operation.job_id, ha="center", va="center", fontsize=6, color="white", fontweight="bold")
    ax.set_yticks(range(len(problem.resources)), [resource.name for resource in problem.resources])
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_title(title or f"{problem.kind} schedule · makespan={schedule.makespan / problem.time_scale:.1f}")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
