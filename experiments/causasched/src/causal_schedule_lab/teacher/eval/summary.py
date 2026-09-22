"""Markdown summary generator (教师模型.md Phase2加强 §31 + §25 questions)."""

from __future__ import annotations

from typing import Any

from .reporting import SOURCE_ACTUAL

KS = (1, 3, 5, 10)


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x is not None else "N/A"


def _rand_str(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


def _overall_table(overall: dict[str, Any]) -> str:
    rows = ["| Method | R@1 | R@3 | R@5 | R@10 | P@3 | P@5 | BestCE@3 | BestCE@5 | MeanCE@3 | Success@3 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for method in (*SOURCE_ACTUAL, "random"):
        data = overall[method]
        r1, r3, r5, r10 = [data["recall_at_k"][str(k)] for k in (1, 3, 5, 10)]
        p3 = data.get("precision_at_k", {}).get("3", 0.0)
        p5 = data.get("precision_at_k", {}).get("5", 0.0)
        bc3 = data.get("best_ce_at_k", {}).get("3", 0.0)
        bc5 = data.get("best_ce_at_k", {}).get("5", 0.0)
        mc3 = data.get("mean_ce_at_k", {}).get("3", 0.0)
        s3 = data.get("success_at_k", {}).get("3", 0.0)
        if method == "random":
            rows.append(f"| Random | {_fmt(r1)} | {_fmt(r3)} | {_fmt(r5)} | {_fmt(r10)} | {_fmt(p3)} | {_fmt(p5)} | {_fmt(bc3)} | {_fmt(bc5)} | {_fmt(mc3)} | {_fmt(s3)} |")
        else:
            lift = overall[method].get("lift_at_k", {}).get("3", 0.0)
            rows.append(f"| {method} | {_fmt(r1)} | {_fmt(r3)} | {_fmt(r5)} | {_fmt(r10)} | {_fmt(p3)} | {_fmt(p5)} | {_fmt(bc3)} | {_fmt(bc5)} | {_fmt(mc3)} | {_fmt(s3)} |")
    return "\n".join(["### Overall", *rows, ""])


def _appearance_table(by_appearance: dict[str, dict[str, Any]]) -> str:
    lines = ["### By Appearance", "| Appearance | Method | blocks | R@3 | R@5 | BestCE@3 | Success@3 |",
             "|---|---|---|---|---|---|---|"]
    for app_type, data in sorted(by_appearance.items()):
        for method in SOURCE_ACTUAL:
            m = data[method]
            lines.append(
                f"| {app_type} | {method} | {data['block_count']} | "
                f"{_fmt(m['recall_at_k']['3'])} | {_fmt(m['recall_at_k']['5'])} | "
                f"{_fmt(m['best_ce_at_k']['3'])} | {_fmt(m['success_at_k']['3'])} |"
            )
    return "\n".join([*lines, ""])


def _atom_table(by_atom: dict[str, dict[str, Any]]) -> str:
    lines = ["### By Root Atom", "| Atom Type | positive roots | R@3 | R@5 |",
             "|---|---|---|---|"]
    for atom_type, data in sorted(by_atom.items()):
        lines.append(
            f"| {atom_type} | {data['positive_count']} | "
            f"{_fmt(data['recall_at_k']['3'])} | {_fmt(data['recall_at_k']['5'])} |"
        )
    return "\n".join([*lines, ""])


def _failure_modes(missed: list[dict[str, Any]], false_top: list[dict[str, Any]]) -> str:
    lines = ["### Main Failure Modes"]
    if missed:
        lines.append(f"1. Missed high-CE roots (teacher_rank > 10): {len(missed)}; "
                     f"top CE {missed[0]['CE']:.2f} at {missed[0]['atom_id']}.")
    if false_top:
        lines.append(f"2. Teacher false positives (TeacherScore high, CE < 0.05): {len(false_top)}; "
                     f"top score {false_top[0].get('teacher_score', 0.0):.2f} at {false_top[0]['atom_id']}.")
    if not missed and not false_top:
        lines.append("1. No strong failure mode detected on this sample.")
    return "\n".join([*lines, ""])


def _answer_questions(overall: dict[str, Any], missed: list[dict[str, Any]],
                      false_top: list[dict[str, Any]], by_atom: dict[str, dict[str, Any]]) -> str:
    t3 = overall["teacher"]["recall_at_k"]["3"]
    r3 = overall["random"]["recall_at_k"]["3"]
    m3 = overall["max_plus"]["recall_at_k"]["3"]
    pr3 = overall["page_rank"]["recall_at_k"]["3"]
    lines = [
        "### Phase-3 前置问题（§25）",
        f"1. Teacher@3/@5 是否稳定优于 Random？ → Recall@3 {t3:.3f} vs Random {r3:.3f} "
        f"({overall['teacher'].get('lift_at_k', {}).get('3', 0.0):.2f}×).",
        f"2. Max-Plus 是否稳定优于 Dense PageRank？ → MaxPlus@3 {m3:.3f} vs PageRank@3 {pr3:.3f}.",
        f"3. Teacher 提升主要来自哪类 Appearance/Root Atom？ → 见 By Appearance / By Root Atom 表；"
        f"missed 根因类型分布：{_type_counts(missed)}.",
        f"4. 最强 missed 根因类型？ → {_top_missed_types(missed)}.",
        f"5. 高分但 CE≈0 的类型？ → {_type_counts(false_top)}.",
        "",
    ]
    return "\n".join(lines)


def _type_counts(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("atom_type", "?")] = counts.get(r.get("atom_type", "?"), 0) + 1
    return str(counts) if counts else "none"


def _top_missed_types(missed: list[dict[str, Any]]) -> str:
    if not missed:
        return "none"
    return f"{missed[0].get('atom_type', '?')} (CE {missed[0]['CE']:.2f})"


def render_summary_md(
    *,
    instance: str,
    makespan: int,
    n_retained: int,
    n_evaluated: int,
    tau_ce: float,
    overall: dict[str, Any],
    by_appearance: dict[str, dict[str, Any]],
    by_atom: dict[str, dict[str, Any]],
    missed: list[dict[str, Any]],
    false_top: list[dict[str, Any]],
    recommendation: str = "",
    recommendation_reason: str = "",
) -> str:
    parts = [
        f"# Phase 2 教师候选质量评估 — {instance}",
        "",
        f"- Makespan: {makespan}, retained blocks: {n_retained}, evaluated: {n_evaluated}",
        f"- τ_CE = {tau_ce}; shared atom set 公平比较；`identified=false`",
        "",
    ]
    parts.append(_overall_table(overall))
    parts.append(_appearance_table(by_appearance))
    parts.append(_atom_table(by_atom))
    parts.append(_answer_questions(overall, missed, false_top, by_atom))
    parts.append(_failure_modes(missed, false_top))
    rec = recommendation or "未自动判定"
    reason = recommendation_reason or "依据上表与失败模式人工填写。"
    parts.append(
        f"## Recommendation\n\n"
        f"- [{'x' if rec == 'Enter Phase 3' else ' '}] Enter Phase 3\n"
        f"- [{'x' if rec.startswith('Modify propagation') else ' '}] Modify propagation mechanism first\n"
        f"- [{'x' if rec.startswith('Modify intervention') else ' '}] Modify intervention prior first\n\n"
        f"**结论：{rec}**\n\nReason: {reason}"
    )
    return "\n".join(parts) + "\n"