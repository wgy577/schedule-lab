"""Full Phase-2 Mk9 run (教师模型.md Phase2加强 §4/§30).

Runs the shared-atom-set probe over *all* retained appearance blocks, computes
every metric, and writes the §30 artifact tree under ``outputs/phase2_family/``:

    summary.md / overall_metrics.csv / by_appearance_type.csv / by_atom_type.csv
    block_level_metrics.csv / atom_level_results.jsonl
    missed_high_ce_roots.jsonl / teacher_false_top.jsonl / raw/{block}.json

Per-block failures (solver error, no legal probe, empty positives) are recorded,
never silently dropped (§4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...ir import Problem, Schedule
from ..atomic_counterfactual_executor import AtomicCounterfactualExecutor
from ...symptom_pruning import SymptomPruningSnapshot
from .probe import probe_block_shared
from .random_baseline import DEFAULT_SEEDS
from .reporting import (
    aggregate_overall,
    breakdown_by_appearance,
    breakdown_by_atom_type,
    compute_block_metrics,
    missed_high_ce_roots,
    pairwise_comparison,
    teacher_false_top,
)
from .summary import render_summary_md
from .writer import (
    block_level_csv_rows,
    block_level_fieldnames,
    by_appearance_csv_rows,
    by_atom_csv_rows,
    overall_metrics_csv_rows,
    write_csv,
    write_jsonl,
)

TAU_CE = 0.30
DEFAULT_OUT = "outputs/phase2_family"


@dataclass
class BlockOutcome:
    ok: bool
    block_id: str
    reason: str = ""


def run_phase2(
    problem: Problem,
    schedule: Schedule,
    snapshot: SymptomPruningSnapshot,
    *,
    executor: AtomicCounterfactualExecutor,
    out_dir: str | Path = DEFAULT_OUT,
    max_blocks: int | None = None,
    top_atoms: int = 16,
    tau_ce: float = TAU_CE,
    random_seeds: int = DEFAULT_SEEDS,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    kept = snapshot.retained
    if max_blocks is not None:
        kept = kept[:max_blocks]

    blocks_metrics = []
    atom_rows: list[dict[str, Any]] = []
    failures: list[BlockOutcome] = []
    for block in kept:
        try:
            probe = probe_block_shared(
                problem, schedule, block,
                executor=executor, top_atoms=top_atoms, seed=seed,
            )
            if not probe.atoms:
                failures.append(BlockOutcome(False, block.block.block_id, "NO_LEGAL_PROBE"))
                continue
            metrics = compute_block_metrics(probe, tau_ce=tau_ce, random_seeds=random_seeds)
            blocks_metrics.append(metrics)
            atom_rows.extend(probe.as_atom_record())
            (raw_dir / f"{block.block.block_id}.json").write_text(
                json.dumps(metrics.as_record(), sort_keys=True, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:  # solver / target / probe error -> record, not drop
            failures.append(BlockOutcome(False, block.block.block_id, f"{type(error).__name__}:{error}"))

    overall = aggregate_overall(blocks_metrics)
    by_appearance = breakdown_by_appearance(blocks_metrics)
    by_atom = breakdown_by_atom_type(atom_rows, tau_ce=tau_ce)
    missed = missed_high_ce_roots(atom_rows, tau_ce=tau_ce)[:20]
    false_top = teacher_false_top(atom_rows, top_k=10, ce_threshold=0.05)[:20]
    pairwise = pairwise_comparison(blocks_metrics)

    # ---- write artifacts ----
    write_csv(out_dir / "overall_metrics.csv", overall_metrics_csv_rows(overall),
              ["method", "R@1", "R@3", "R@5", "R@10", "P@3", "P@5", "BestCE@3", "BestCE@5", "MeanCE@3", "Success@3"])
    write_csv(out_dir / "by_appearance_type.csv", by_appearance_csv_rows(by_appearance),
              ["appearance_type", "method", "block_count", "R@3", "R@5", "BestCE@3", "Success@3"])
    write_csv(out_dir / "by_atom_type.csv", by_atom_csv_rows(by_atom),
              ["atom_type", "positive_count", "R@3", "R@5"])
    write_csv(out_dir / "block_level_metrics.csv", block_level_csv_rows(blocks_metrics),
              block_level_fieldnames())
    write_jsonl(out_dir / "atom_level_results.jsonl", atom_rows)
    write_jsonl(out_dir / "missed_high_ce_roots.jsonl", missed)
    write_jsonl(out_dir / "teacher_false_top.jsonl", false_top)

    recommendation, recommendation_reason = _recommend(overall)

    summary_md = render_summary_md(
        instance=problem.id,
        makespan=schedule.makespan,
        n_retained=len(snapshot.retained),
        n_evaluated=len(blocks_metrics),
        tau_ce=tau_ce,
        overall=overall,
        by_appearance=by_appearance,
        by_atom=by_atom,
        missed=missed,
        false_top=false_top,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    result = {
        "instance": problem.id,
        "makespan": schedule.makespan,
        "n_retained": len(snapshot.retained),
        "n_evaluated": len(blocks_metrics),
        "n_failed": len(failures),
        "failures": [{"block_id": f.block_id, "reason": f.reason} for f in failures],
        "tau_ce": tau_ce,
        "overall": overall,
        "by_appearance": by_appearance,
        "by_atom": by_atom,
        "pairwise": pairwise,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "out_dir": str(out_dir),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _recommend(overall: dict[str, Any]) -> tuple[str, str]:
    """Auto-recommendation per Phase2加强 §33: only enter Phase 3 if the teacher
    prior is not actively hurting vs its own propagation (max_plus)."""
    t3 = overall["teacher"]["recall_at_k"]["3"]
    m3 = overall["max_plus"]["recall_at_k"]["3"]
    r3 = overall["random"]["recall_at_k"]["3"]
    if t3 > m3 and t3 > r3:
        return "Enter Phase 3", (
            f"Teacher R@3={t3:.3f} > MaxPlus R@3={m3:.3f} and > Random R@3={r3:.3f}; "
            f"the TeacherScore prior adds value over pure propagation."
        )
    if m3 > r3:
        return "Modify intervention prior first", (
            f"Pure Max-Plus R@3={m3:.3f} beats Teacher R@3={t3:.3f} and Random R@3={r3:.3f}; "
            f"the routing/sequencing prior is HURTING recall. Fix the prior before Phase 3."
        )
    return "Modify propagation mechanism first", (
        f"No method beats Random R@3={r3:.3f} (best is MaxPlus R@3={m3:.3f}); "
        f"the propagation signal itself is weak. Fix G_prop/Max-Plus before Phase 3."
    )