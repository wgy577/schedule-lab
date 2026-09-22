"""Render one real T2-G sibling trajectory as an auditable Gantt sequence."""
from __future__ import annotations

import copy
import hashlib
import html
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from causal_schedule_lab.validation import schedule_hash
from .proposal_features import _execute_step


def _snapshot(problem, schedule):
    op_map = problem.operation_map()
    mode_map = problem.mode_map()
    rows = []
    for assignment in schedule.assignments:
        operation = op_map[assignment.operation_id]
        _operation, mode = mode_map[assignment.mode_id]
        rows.append({
            "operation_id": assignment.operation_id,
            "job_id": operation.job_id,
            "machine_id": mode.resources[0] if mode.resources else "UNASSIGNED",
            "resources": list(mode.resources),
            "mode_id": assignment.mode_id,
            "start": int(assignment.start),
            "end": int(assignment.end),
        })
    rows.sort(key=lambda row: (row["machine_id"], row["start"], row["end"],
                               row["operation_id"]))
    return {"makespan": int(schedule.makespan), "assignments": rows,
            "state_hash": schedule_hash(schedule)}


def _direct_operation_ids(edits):
    ids = set()
    for edit in edits:
        for name in ("operation_id", "left_id", "right_id", "predecessor_id",
                     "successor_id"):
            value = getattr(edit, name, None)
            if value:
                ids.add(str(value))
    return sorted(ids)


def _changed_operation_ids(before, after):
    fields = ("machine_id", "mode_id", "start", "end")
    bmap = {row["operation_id"]: row for row in before["assignments"]}
    amap = {row["operation_id"]: row for row in after["assignments"]}
    return sorted(op_id for op_id in set(bmap) | set(amap)
                  if op_id not in bmap or op_id not in amap or
                  any(bmap[op_id][field] != amap[op_id][field] for field in fields))


def _edit_json(edit):
    if is_dataclass(edit):
        return asdict(edit)
    return {name: getattr(edit, name) for name in dir(edit)
            if not name.startswith("_") and not callable(getattr(edit, name))}


def _job_color(job_id):
    import matplotlib.pyplot as plt
    digest = hashlib.sha256(str(job_id).encode("utf-8")).digest()
    return plt.get_cmap("tab20")(digest[0] % 20)


def _draw_state(ax, snapshot, title, direct=(), propagated=()):
    direct, propagated = set(direct), set(propagated) - set(direct)
    machines = sorted({row["machine_id"] for row in snapshot["assignments"]})
    ypos = {machine: i for i, machine in enumerate(machines)}
    show_labels = len(snapshot["assignments"]) <= 120
    for row in snapshot["assignments"]:
        op_id = row["operation_id"]
        edge, width = ("#dc2626", 3.2) if op_id in direct else (
            ("#f59e0b", 2.2) if op_id in propagated else ("#334155", 0.65))
        ax.barh(ypos[row["machine_id"]], row["end"] - row["start"],
                left=row["start"], height=0.72, color=_job_color(row["job_id"]),
                edgecolor=edge, linewidth=width)
        if show_labels:
            ax.text((row["start"] + row["end"]) / 2, ypos[row["machine_id"]],
                    op_id, ha="center", va="center", fontsize=6, clip_on=True)
    ax.set_yticks(range(len(machines)), machines, fontsize=7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.axvline(snapshot["makespan"], color="#111827", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.18)
    ax.set_title(title, fontsize=10)


def write_optimization_trace(problem, executor, root_schedule, trajectory, out_dir,
                             *, generation, cycle, root_kind="current"):
    """Replay and render the best already-collected sibling; no policy calls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = copy.deepcopy(root_schedule)
    states = [_snapshot(problem, current)]
    actions = []
    for local_step, item in enumerate(trajectory.get("_visual_plan") or (), start=1):
        before = states[-1]
        edits = tuple(item["edits"])
        result = _execute_step(executor, problem, current, edits,
                               int(current.makespan), schedule_hash(current))
        if result is None:
            raise RuntimeError(f"visual trace replay failed at step {local_step}")
        current = result["schedule"]
        after = _snapshot(problem, current)
        expected = int(item["after_ms"])
        if after["makespan"] != expected:
            raise AssertionError(
                f"visual trace drift at step {local_step}: {after['makespan']} != {expected}")
        direct = _direct_operation_ids(edits)
        changed = _changed_operation_ids(before, after)
        actions.append({
            "causal_paths": item.get("causal_paths", []),
            "m2_selected_root_hop": item.get("m2_selected_root_hop", -1),
            "m2_selected_root_prior_trust": item.get("m2_selected_root_prior_trust"),
            "m2_selected_root_rl_value": item.get("m2_selected_root_rl_value"),
            "step": local_step, "global_step": int(item["step"]),
            "signature": item["signature"], "kind": item["kind"],
            "sampled_class": item.get("sampled_class"),
            "active_layer": item.get("active_layer"),
            "before_ms": before["makespan"], "after_ms": after["makespan"],
            "step_improvement": before["makespan"] - after["makespan"],
            "cumulative_improvement": states[0]["makespan"] - after["makespan"],
            "direct_operation_ids": direct,
            "propagated_operation_ids": sorted(set(changed) - set(direct)),
            "all_changed_operation_ids": changed,
            "edits": [_edit_json(edit) for edit in edits],
            "m2_selected_roots": list(item.get("m2_selected_roots", ())),
            "proposal_root_ops": list(item.get("proposal_root_ops", ())),
            "m2_selected_root_scores": dict(item.get("m2_selected_root_scores", {})),
            "reverse_message_passing_layers": int(
                item.get("reverse_message_passing_layers", 0)),
            "retained_appearance_blocks": list(
                item.get("retained_appearance_blocks", ())),
        })
        states.append(after)

    iid = str(trajectory.get("iid", problem.id if hasattr(problem, "id") else "graph"))
    initial_path = out_dir / "step_000_initial.png"
    fig, ax = plt.subplots(figsize=(15, max(4.5, 0.34 * len({
        row['machine_id'] for row in states[0]['assignments']}))))
    _draw_state(ax, states[0], f"{iid} | initial | Cmax={states[0]['makespan']}")
    fig.tight_layout()
    fig.savefig(initial_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    image_files = [initial_path.name]
    legend = [Patch(facecolor="white", edgecolor="#dc2626", linewidth=3,
                    label="directly edited operation"),
              Patch(facecolor="white", edgecolor="#f59e0b", linewidth=2,
                    label="operation shifted by replay")]
    for idx, action in enumerate(actions, start=1):
        machines = {row["machine_id"] for state in states[idx - 1:idx + 1]
                    for row in state["assignments"]}
        fig, axes = plt.subplots(2, 1, figsize=(15, max(8, 0.58 * len(machines))))
        suffix = (f"Cmax {action['before_ms']} -> {action['after_ms']} | "
                  f"step gain {action['step_improvement']:+d} | "
                  f"total {action['cumulative_improvement']:+d}")
        _draw_state(axes[0], states[idx - 1], f"Before operation {idx} | {suffix}",
                    action["direct_operation_ids"], action["propagated_operation_ids"])
        _draw_state(axes[1], states[idx], f"After operation {idx} | {suffix}",
                    action["direct_operation_ids"], action["propagated_operation_ids"])
        fig.legend(handles=legend, loc="upper right", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        image_path = out_dir / f"change_{idx:03d}_before_after.png"
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_files.append(image_path.name)

    payload = {
        "best_makespan": min(s["makespan"] for s in states),
        "best_step": min(range(len(states)), key=lambda i: states[i]["makespan"]),
        "best_improvement": states[0]["makespan"] - min(s["makespan"] for s in states),
        "schema": "t2g-optimization-trace-v1", "instance_id": iid,
        "generation": int(generation), "cycle": int(cycle), "root_kind": root_kind,
        "trajectory_id": int(trajectory.get("traj_id", -1)),
        "trajectory_reward": float(trajectory.get("reward", 0.0)),
        "terminal": trajectory.get("terminal"),
        "n_actions": len(actions), "initial_makespan": states[0]["makespan"],
        "final_makespan": states[-1]["makespan"],
        "total_improvement": states[0]["makespan"] - states[-1]["makespan"],
        "actions": actions, "states": states, "images": image_files,
    }
    (out_dir / "trace.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cards = []
    for idx, image_name in enumerate(image_files):
        heading = "初始调度" if idx == 0 else f"第 {idx} 次操作：前后对比"
        detail = "" if idx == 0 else (
            f"单步工期变化 {actions[idx-1]['step_improvement']:+d}；"
            f"累计优化 {actions[idx-1]['cumulative_improvement']:+d}")
        if idx:
            action = actions[idx-1]
            detail += "；算子：" + ", ".join(str(e.get("edit_type", "")) for e in action["edits"])
            for path in action["causal_paths"]:
                if path["reachable_within_limit"]:
                    detail += f"；表象 {path['appearance_id']}，{path['hops']}跳：" + " ← ".join(path["effect_to_cause_path"])
        cards.append(
            f'<section><h2>{html.escape(heading)}</h2><p>{html.escape(detail)}</p>'
            f'<img src="{html.escape(image_name)}" alt="{html.escape(heading)}"></section>')
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>T2G 优化轨迹 {html.escape(iid)}</title>
<style>body{{margin:0;background:#f8fafc;color:#0f172a;font:15px system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}header,section{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px;margin-bottom:20px;box-shadow:0 3px 16px #0f172a0d}}
h1,h2{{margin:0 0 10px}}img{{display:block;width:100%;height:auto;border-radius:8px}}
.red{{color:#dc2626}}.amber{{color:#d97706}}</style><main>
<header><h1>真实策略优化轨迹：{html.escape(iid)}</h1>
<p>历史最好 Cmax {payload['best_makespan']}，第 {payload['best_step']} 步，改善 {payload['best_improvement']:+d}。路径为结构可达证据，实际收益来自执行后的工期变化。</p>
<p>generation {generation} · trajectory {payload['trajectory_id']} · 初始 Cmax {payload['initial_makespan']} → 最终 Cmax {payload['final_makespan']} · 总优化 {payload['total_improvement']:+d}</p>
<p><span class="red">红框</span>＝本步直接操作的工序；<span class="amber">橙框</span>＝FixedDecisionReplay 后连带移动的工序。原始数据见 trace.json。</p></header>
{''.join(cards)}</main></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    return payload
