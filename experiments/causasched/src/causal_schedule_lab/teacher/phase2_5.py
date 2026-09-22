"""Phase 2.5: causal-ranking / intervention-utility decoupling offline re-rank
(教师模型.md Phase2加强_2 §8/§9/§10/§11/§13).

Runs entirely on the already-measured ``atom_level_results.jsonl`` (no Solver
re-run).  For each atom we recover, offline:

    rel            = max_plus_score            (Rel_A, the causal signal)
    utility        = recovered intervention utility D(r)
    causal_root    = rel                        (official Root Ranking)
    legacy_teacher = teacher_score              (baseline B)

and compare:

* A. Pure Max-Plus / causal_root_score  — official Root Ranking
* B. Old TeacherScore (legacy)          — baseline
* C. Additive diagnostic  rel + beta*D   (beta sweep, diagnostic only)
* D. Two-stage: causal RootRank -> Top-L -> probe_priority ordering

plus Root-Ranking metrics (§9.1), Probe-Budget metrics (§9.2), the beta sweep
(§10), and the miss classification Prior-caused / Propagation / Mechanism-
mismatch (§11/§12).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .eval.metrics import (
    best_ce_at_k,
    mean_ce_at_k,
    precision_at_k,
    recall_at_k,
    spearman_score_ce,
)

LAMBDA_T = 0.30
TAU_CE = 0.30
KS = (1, 3, 5, 10)
BUDGETS = (1, 3, 5, 10)
BETA_SWEEP = (0.0, 0.02, 0.05, 0.1, 0.2)
TOP_L = 20


@dataclass
class AtomRow:
    appearance_id: str
    appearance_type: str
    atom_id: str
    atom_type: str
    ce: float
    rel: float          # = max_plus_score (Rel_A)
    utility: float      # recovered D(r)
    causal_root: float  # = rel
    legacy_teacher: float  # = teacher_score


def load_atom_records(path: str | Path) -> list[AtomRow]:
    """Load ``atom_level_results.jsonl`` and recover the decoupled fields."""
    rows: list[AtomRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rel = float(r.get("max_plus_score", 0.0))
        teacher = float(r.get("teacher_score", 0.0))
        # Recover D from teacher_score = rel*(lambda + (1-lambda)*D).
        if rel > 1e-12:
            utility = (teacher / rel - LAMBDA_T) / (1.0 - LAMBDA_T)
        else:
            utility = 0.0
        ce = float(r.get("CE", r.get("solver_ce", 0.0)))
        rows.append(AtomRow(
            appearance_id=r["appearance_id"],
            appearance_type=r.get("appearance_type", "?"),
            atom_id=r["atom_id"],
            atom_type=r.get("atom_type", "?"),
            ce=ce,
            rel=rel,
            utility=max(0.0, utility),
            causal_root=rel,
            legacy_teacher=teacher,
        ))
    return rows


def _group_by_appearance(rows: list[AtomRow]) -> dict[str, list[AtomRow]]:
    groups: dict[str, list[AtomRow]] = defaultdict(list)
    for row in rows:
        groups[row.appearance_id].append(row)
    return dict(groups)


def _order(rows: list[AtomRow], score_field: str) -> list[AtomRow]:
    return sorted(rows, key=lambda r: (-getattr(r, score_field), r.atom_id))


def _positive(rows: list[AtomRow], tau_ce: float = TAU_CE) -> frozenset[str]:
    return frozenset(r.atom_id for r in rows if r.ce >= tau_ce)


def root_ranking_metrics(
    rows: list[AtomRow],
    score_field: str,
    *,
    ks: tuple[int, ...] = KS,
    tau_ce: float = TAU_CE,
) -> dict[str, Any]:
    """§9.1 Root-Ranking quality for a given score's ordering (no prior mixing)."""
    ordered = _order(rows, score_field)
    ids = [r.atom_id for r in ordered]
    ce = {r.atom_id: r.ce for r in rows}
    positive = _positive(rows, tau_ce)
    recall = {str(k): recall_at_k(ids, positive, k) for k in ks}
    precision = {str(k): precision_at_k(ids, positive, k) for k in ks}
    best = {str(k): best_ce_at_k(ids, ce, k) for k in ks}
    mean_top = {str(k): mean_ce_at_k(ids, ce, k) for k in ks}
    success = {str(k): 1.0 if recall_at_k(ids, positive, k) > 0 else 0.0 for k in ks}
    # Spearman between the score and measured CE (only defined on the block).
    scores = [getattr(r, score_field) for r in ordered]
    ces = [r.ce for r in ordered]
    rho = spearman_score_ce(scores, ces)
    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "best_ce_at_k": best,
        "mean_ce_at_k": mean_top,
        "success_at_k": success,
        "spearman": rho,
    }


def probe_budget_metrics(
    rows: list[AtomRow],
    *,
    beta: float,
    top_l: int = TOP_L,
    budgets: tuple[int, ...] = BUDGETS,
    tau_ce: float = TAU_CE,
) -> dict[str, Any]:
    """§9.2 Probe-Budget efficiency for two-stage ``probe_priority`` ordering.

    Causal Top-L (by ``causal_root``) -> reorder by ``rel + beta*D`` -> the first
    ``B`` probes are those actually run.  Measures whether intervention utility
    finds high-CE roots faster within the budget.
    """
    causal_top = _order(rows, "causal_root")[:top_l]
    ordered = sorted(
        causal_top,
        key=lambda r: (-(r.rel + beta * r.utility), r.atom_id),
    )
    ids = [r.atom_id for r in ordered]
    ce = {r.atom_id: r.ce for r in rows}
    positive = _positive(rows, tau_ce)
    out: dict[str, Any] = {}
    for b in budgets:
        first = ids[:b]
        best = best_ce_at_k(first, ce, b)
        hit = 1.0 if (set(first) & positive) else 0.0
        eff = len(set(first) & positive) / b if b else 0.0
        found = len(set(first) & positive)
        out[str(b)] = {
            "best_ce_budget": best,
            "hit_budget": hit,
            "efficiency": eff,
            "positive_found": found,
        }
    return out


def beta_sweep(rows: list[AtomRow], *, betas: tuple[float, ...] = BETA_SWEEP) -> list[dict[str, Any]]:
    """§10 beta sweep over probe-priority; all metrics averaged over appearances."""
    groups = _group_by_appearance(rows)
    sweep: list[dict[str, Any]] = []
    for beta in betas:
        best3 = [probe_budget_metrics(g, beta=beta)["3"]["best_ce_budget"] for g in groups.values()]
        hit3 = [probe_budget_metrics(g, beta=beta)["3"]["hit_budget"] for g in groups.values()]
        eff3 = [probe_budget_metrics(g, beta=beta)["3"]["efficiency"] for g in groups.values()]
        best1 = [probe_budget_metrics(g, beta=beta)["1"]["best_ce_budget"] for g in groups.values()]
        best5 = [probe_budget_metrics(g, beta=beta)["5"]["best_ce_budget"] for g in groups.values()]
        sweep.append({
            "beta": beta,
            "best_ce_budget_1": mean(best1) if best1 else 0.0,
            "best_ce_budget_3": mean(best3) if best3 else 0.0,
            "best_ce_budget_5": mean(best5) if best5 else 0.0,
            "hit_budget_3": mean(hit3) if hit3 else 0.0,
            "efficiency_3": mean(eff3) if eff3 else 0.0,
        })
    return sweep


def _agg_metrics(blocks: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {str(k): mean(b[field][str(k)] for b in blocks) for k in KS}


_MISS_K = 10  # cutoff for miss classification (改A6.md §12)


def miss_classification(rows: list[AtomRow], *, tau_ce: float = TAU_CE,
                        miss_k: int = _MISS_K) -> dict[str, Any]:
    """§11/§12 (改A6.md §12) classify high-CE roots the OLD teacher missed.

    High-CE roots (``CE >= tau_ce``) that fall outside the causal RootRanking
    top-K are split into:
      * prior_caused:      inside causal top-K but pushed out by legacy Teacher.
      * tie_saturation:    ``Rel(r) == Rel_cutoff@K`` (ties with the cutoff; the
                           deterministic atom_id tie-break pushed it out).  NOT a
                           propagation miss (改A6.md §12.2).
      * true_low_propagation: ``Rel(r) < Rel_cutoff@K`` — Max-Plus truly gave a
                           low score to a real high-CE root (改A6.md §12.3).
    """
    groups = _group_by_appearance(rows)
    prior_caused: list[AtomRow] = []
    tie_saturation: list[AtomRow] = []
    true_low_propagation: list[AtomRow] = []
    for gid, g in groups.items():
        mp_order = _order(g, "causal_root")
        mp_rank = {r.atom_id: i + 1 for i, r in enumerate(mp_order)}
        tc_order = _order(g, "legacy_teacher")
        tc_rank = {r.atom_id: i + 1 for i, r in enumerate(tc_order)}
        cutoff = getattr(mp_order[min(miss_k, len(mp_order)) - 1], "causal_root", 0.0) \
            if len(mp_order) >= 1 else 0.0
        for r in g:
            if r.ce < tau_ce:
                continue
            causal_rank = mp_rank.get(r.atom_id, 10 ** 9)
            teacher_rank = tc_rank.get(r.atom_id, 10 ** 9)
            if causal_rank <= miss_k and teacher_rank > miss_k:
                prior_caused.append(r)
            elif causal_rank > miss_k:
                if abs(r.causal_root - cutoff) <= 1e-12:
                    tie_saturation.append(r)
                else:
                    true_low_propagation.append(r)

    total_high_ce = len(prior_caused) + len(tie_saturation) + len(true_low_propagation)

    def _cat(rows_):
        n = len(rows_)
        return {
            "count": n,
            "ratio": n / max(1, total_high_ce),
            "mean_ce": mean([r.ce for r in rows_]) if rows_ else 0.0,
            "routing": sum(1 for r in rows_ if r.atom_type == "routing"),
            "sequencing": sum(1 for r in rows_ if r.atom_type == "sequencing"),
        }

    return {
        "prior_caused": {**_cat(prior_caused), "rows": prior_caused},
        "tie_saturation": {**_cat(tie_saturation), "rows": tie_saturation},
        "true_low_propagation": {**_cat(true_low_propagation), "rows": true_low_propagation},
    }


def mechanism_mismatch_cases(miss_rows: list[AtomRow]) -> list[dict[str, Any]]:
    """§11 Type III (改A6.md §14): mechanism-mismatch candidates.

    Mechanism mismatch is NOT inferred from ``atom_type``.  It stays ``None``
    ("unknown") until explicit mechanism-level evidence exists.
    """
    cases = []
    for r in miss_rows:
        cases.append({
            "appearance_id": r.appearance_id,
            "appearance_type": r.appearance_type,
            "atom_id": r.atom_id,
            "atom_type": r.atom_type,
            "CE": r.ce,
            "mechanism_mismatch": None,
        })
    return sorted(cases, key=lambda c: -c["CE"])


def breakdown_by_atom_type_causal(rows: list[AtomRow], *, ks: tuple[int, ...] = KS,
                                  tau_ce: float = TAU_CE) -> dict[str, dict[str, Any]]:
    """§13 Routing/Sequencing separate Recall@K under the official causal ranking."""
    groups = _group_by_appearance(rows)
    out: dict[str, dict[str, Any]] = {}
    for atom_type in {r.atom_type for r in rows}:
        positives = [r for r in rows if r.ce >= tau_ce and r.atom_type == atom_type]
        if not positives:
            out[atom_type] = {"positive_count": 0, "recall_at_k": {str(k): 0.0 for k in ks}}
            continue
        by_block: dict[str, list[AtomRow]] = defaultdict(list)
        for r in positives:
            by_block[r.appearance_id].append(r)
        recall = {}
        for k in ks:
            per_block = []
            for gid, g in groups.items():
                top = set(a.atom_id for a in _order(g, "causal_root")[:k])
                block_pos = [r for r in by_block.get(gid, [])]
                per_block.append(len([r for r in block_pos if r.atom_id in top]) / len(block_pos) if block_pos else 0.0)
            recall[str(k)] = mean(per_block) if per_block else 0.0
        out[atom_type] = {"positive_count": len(positives), "recall_at_k": recall}
    return out


def run_phase2_5(
    atom_level_path: str | Path,
    out_dir: str | Path,
    *,
    tau_ce: float = TAU_CE,
    beta_sweep_values: tuple[float, ...] = BETA_SWEEP,
) -> dict[str, Any]:
    """Full offline Phase 2.5 re-rank; writes the §17 artifact tree."""
    import csv

    from .eval.writer import write_csv, write_jsonl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_atom_records(atom_level_path)
    groups = _group_by_appearance(rows)

    # A. Pure Max-Plus / causal_root (official); B. legacy TeacherScore (baseline).
    block_metrics = {
        "causal_root": [root_ranking_metrics(g, "causal_root", tau_ce=tau_ce) for g in groups.values()],
        "legacy_teacher": [root_ranking_metrics(g, "legacy_teacher", tau_ce=tau_ce) for g in groups.values()],
    }
    overall = {
        "causal_root": {
            "recall_at_k": _agg_metrics(block_metrics["causal_root"], "recall_at_k"),
            "precision_at_k": _agg_metrics(block_metrics["causal_root"], "precision_at_k"),
            "best_ce_at_k": _agg_metrics(block_metrics["causal_root"], "best_ce_at_k"),
            "mean_ce_at_k": _agg_metrics(block_metrics["causal_root"], "mean_ce_at_k"),
            "success_at_k": _agg_metrics(block_metrics["causal_root"], "success_at_k"),
            "spearman": _agg_optional([b["spearman"] for b in block_metrics["causal_root"]]),
        },
        "legacy_teacher": {
            "recall_at_k": _agg_metrics(block_metrics["legacy_teacher"], "recall_at_k"),
            "precision_at_k": _agg_metrics(block_metrics["legacy_teacher"], "precision_at_k"),
            "best_ce_at_k": _agg_metrics(block_metrics["legacy_teacher"], "best_ce_at_k"),
            "mean_ce_at_k": _agg_metrics(block_metrics["legacy_teacher"], "mean_ce_at_k"),
            "success_at_k": _agg_metrics(block_metrics["legacy_teacher"], "success_at_k"),
            "spearman": _agg_optional([b["spearman"] for b in block_metrics["legacy_teacher"]]),
        },
    }

    # Beta sweep (probe budget efficiency).
    sweep = beta_sweep(rows, betas=beta_sweep_values)

    # Miss classification (v2: prior_caused / tie_saturation / true_low_propagation).
    miss = miss_classification(rows, tau_ce=tau_ce)
    mechanism_rows = miss["tie_saturation"]["rows"] + miss["true_low_propagation"]["rows"]
    mechanism = mechanism_mismatch_cases(mechanism_rows)

    # Atom-type breakdown under causal ranking.
    by_atom = breakdown_by_atom_type_causal(rows, tau_ce=tau_ce)

    # Tie-aware ranking audit (改A6.md §11/§13).
    from .tie_audit import tie_audit_by_block, tie_aware_recall

    tie_aware = tie_aware_recall(rows, tau_ce=tau_ce, score_field="causal_root")
    tie_audit_rows = tie_audit_by_block(rows, score_field="causal_root")

    # ---- write ----
    write_csv(out_dir / "causal_ranking_metrics.csv", _causal_ranking_csv(overall),
              ["score", "R@1", "R@3", "R@5", "R@10", "P@3", "BestCE@3", "MeanCE@3", "Success@3"])
    write_csv(out_dir / "beta_sweep.csv", sweep,
              ["beta", "best_ce_budget_1", "best_ce_budget_3", "best_ce_budget_5", "hit_budget_3", "efficiency_3"])
    write_csv(out_dir / "by_atom_type.csv",
              [{"atom_type": t, "positive_count": v["positive_count"], **{f"R@{k}": v["recall_at_k"][str(k)] for k in KS}}
               for t, v in by_atom.items()],
              ["atom_type", "positive_count", "R@1", "R@3", "R@5", "R@10"])
    write_csv(out_dir / "miss_category_summary.csv",
              [{"miss_type": t, **_cat_flat(v)} for t, v in miss.items()],
              ["miss_type", "count", "ratio", "mean_ce", "routing", "sequencing"])
    write_csv(out_dir / "miss_type_v2.csv",
              [{"miss_type": t, **_cat_flat(v)} for t, v in miss.items()],
              ["miss_type", "count", "ratio", "mean_ce", "routing", "sequencing"])
    write_csv(out_dir / "tie_aware_metrics.csv",
              [{"K": k, **v} for k, v in tie_aware.items()],
              ["K", "strict", "optimistic", "expected"])
    tie_fieldnames = ["appearance_id", "candidate_count", "unique_score_count",
                      "max_score_count", "max_score_ratio",
                      "top3_cutoff_score", "top3_tie_group_size",
                      "top5_cutoff_score", "top5_tie_group_size",
                      "top10_cutoff_score", "top10_tie_group_size"]
    write_csv(out_dir / "tie_audit_by_block.csv", tie_audit_rows, tie_fieldnames)
    write_jsonl(out_dir / "prior_caused_misses.jsonl", [r_atom_to_record(r) for r in miss["prior_caused"]["rows"]])
    write_jsonl(out_dir / "tie_saturation_misses.jsonl", [r_atom_to_record(r) for r in miss["tie_saturation"]["rows"]])
    write_jsonl(out_dir / "true_low_propagation_misses.jsonl", [r_atom_to_record(r) for r in miss["true_low_propagation"]["rows"]])
    write_jsonl(out_dir / "mechanism_mismatch_cases.jsonl", mechanism)
    write_jsonl(out_dir / "atom_level_decoupled.jsonl", [r_atom_to_record(r) for r in rows])

    result = {
        "n_atoms": len(rows),
        "n_appearances": len(groups),
        "overall": overall,
        "beta_sweep": sweep,
        "miss": {k: _cat_flat(v) for k, v in miss.items()},
        "tie_aware_recall": tie_aware,
        "by_atom": {t: {"positive_count": v["positive_count"], "recall_at_k": v["recall_at_k"]} for t, v in by_atom.items()},
        "out_dir": str(out_dir),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def _cat_flat(v: dict[str, Any]) -> dict[str, Any]:
    return {"count": v["count"], "ratio": v["ratio"], "mean_ce": v["mean_ce"],
            "routing": v["routing"], "sequencing": v["sequencing"]}


def _agg_optional(values: list[float | None]) -> dict[str, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "median": None, "std": None}
    return {"mean": mean(present), "median": median(present),
            "std": pstdev(present) if len(present) > 1 else 0.0}


def _causal_ranking_csv(overall: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for score, d in overall.items():
        rows.append({
            "score": score,
            "R@1": d["recall_at_k"]["1"], "R@3": d["recall_at_k"]["3"],
            "R@5": d["recall_at_k"]["5"], "R@10": d["recall_at_k"]["10"],
            "P@3": d["precision_at_k"]["3"], "BestCE@3": d["best_ce_at_k"]["3"],
            "MeanCE@3": d["mean_ce_at_k"]["3"], "Success@3": d["success_at_k"]["3"],
        })
    return rows


def r_atom_to_record(r: AtomRow) -> dict[str, Any]:
    return {
        "appearance_id": r.appearance_id,
        "appearance_type": r.appearance_type,
        "atom_id": r.atom_id,
        "atom_type": r.atom_type,
        "propagation_relevance": r.rel,
        "causal_root_score": r.causal_root,
        "intervention_utility": r.utility,
        "legacy_teacher_score": r.legacy_teacher,
        "solver_ce": r.ce,
    }