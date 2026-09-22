"""Phase-2 reporting (教师模型.md Phase2加强 §11-§23).

Computes per-block metrics, appearance-type / atom-type breakdowns, the missed
high-CE roots and teacher false-top reports, pairwise method comparisons, and
renders the markdown summary (§31) plus CSV/JSONL artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any

from .metrics import (
    best_ce_at_k,
    hit_at_k,
    lift_at_k,
    mean_ce_at_k,
    precision_at_k,
    recall_at_k,
)
from .random_baseline import K_VALUES, evaluate_random_ranking

SOURCE_ACTUAL = ("teacher", "max_plus", "page_rank")
KS = K_VALUES


@dataclass
class BlockMetrics:
    appearance_id: str
    appearance_type: str
    candidate_count: int
    positive_count: int
    positive_ratio: float
    ce_max: float
    ce_mean: float
    ce_median: float
    ce_std: float
    negative_count: int
    k_ratio: dict[int, float]
    recall: dict[str, dict[int, float]]        # source -> k -> recall
    precision: dict[str, dict[int, float]]
    best_ce: dict[str, dict[int, float]]
    mean_ce: dict[str, dict[int, float]]
    hit: dict[str, dict[int, bool]]
    spearman: dict[str, float | None]
    random_recall: dict[int, "RandSummary"]
    random_best_ce: dict[int, "RandSummary"]

    def as_record(self) -> dict[str, Any]:
        def kdict(d: dict[int, float]) -> dict[str, float]:
            return {str(k): round(float(v), 6) for k, v in d.items()}

        rec = {
            "appearance_id": self.appearance_id,
            "appearance_type": self.appearance_type,
            "candidate_count": self.candidate_count,
            "positive_count": self.positive_count,
            "positive_ratio": round(self.positive_ratio, 6),
            "ce_max": round(self.ce_max, 6),
            "ce_mean": round(self.ce_mean, 6),
            "ce_median": round(self.ce_median, 6),
            "ce_std": round(self.ce_std, 6),
            "negative_count": self.negative_count,
            "k_ratio": kdict(self.k_ratio),
            "spearman": {s: (None if v is None else round(v, 6)) for s, v in self.spearman.items()},
        }
        for src in SOURCE_ACTUAL:
            rec[f"{src}_recall"] = kdict(self.recall[src])
            rec[f"{src}_best_ce"] = kdict(self.best_ce[src])
        rec["random_recall"] = {str(k): v.as_record() for k, v in self.random_recall.items()}
        rec["random_best_ce"] = {str(k): v.as_record() for k, v in self.random_best_ce.items()}
        return rec


def compute_block_metrics(probe, *, tau_ce: float = 0.30, random_seeds: int = 50) -> BlockMetrics:
    positive = probe.positive_set(tau_ce)
    atom_ids = [a.atom_id for a in probe.atoms]
    ce_values = [probe.records[a].ce for a in atom_ids]
    positives = [c for c in ce_values if c >= tau_ce]
    negatives = len(atom_ids) - len(positives)

    recall: dict[str, dict[int, float]] = {}
    precision: dict[str, dict[int, float]] = {}
    best_ce: dict[str, dict[int, float]] = {}
    mean_ce: dict[str, dict[int, float]] = {}
    hit: dict[str, dict[int, bool]] = {}
    for src in SOURCE_ACTUAL:
        ranking = list(probe.rankings[src])
        recall[src] = {k: recall_at_k(ranking, positive, k) for k in KS}
        precision[src] = {k: precision_at_k(ranking, positive, k) for k in KS}
        best_ce[src] = {k: best_ce_at_k(ranking, probe.ce_by_atom, k) for k in KS}
        mean_ce[src] = {k: mean_ce_at_k(ranking, probe.ce_by_atom, k) for k in KS}
        hit[src] = {k: hit_at_k(ranking, positive, k) for k in KS}

    rand = evaluate_random_ranking(
        atom_ids, probe.ce_by_atom, positive, k_values=KS, seeds=random_seeds
    )
    return BlockMetrics(
        appearance_id=probe.block_id,
        appearance_type=probe.appearance_type,
        candidate_count=len(atom_ids),
        positive_count=len(positives),
        positive_ratio=len(positives) / len(atom_ids) if atom_ids else 0.0,
        ce_max=max(ce_values) if ce_values else 0.0,
        ce_mean=mean(ce_values) if ce_values else 0.0,
        ce_median=median(ce_values) if ce_values else 0.0,
        ce_std=pstdev(ce_values) if len(ce_values) > 1 else 0.0,
        negative_count=negatives,
        k_ratio={k: k / len(atom_ids) if atom_ids else 0.0 for k in KS},
        recall=recall,
        precision=precision,
        best_ce=best_ce,
        mean_ce=mean_ce,
        hit=hit,
        spearman=dict(probe.spearman),
        random_recall=rand["recall"],
        random_best_ce=rand["best_ce"],
    )


def _mean_over(blocks: list[BlockMetrics], field: str, metric: str, k: int) -> float:
    vals = []
    for b in blocks:
        d = getattr(b, field)[metric]
        vals.append(d[k])
    return mean(vals) if vals else 0.0


def aggregate_overall(blocks: list[BlockMetrics]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_blocks": len(blocks)}
    for src in SOURCE_ACTUAL:
        out[src] = {
            "recall_at_k": {str(k): _mean_over(blocks, "recall", src, k) for k in KS},
            "best_ce_at_k": {str(k): _mean_over(blocks, "best_ce", src, k) for k in KS},
            "precision_at_k": {str(k): _mean_over(blocks, "precision", src, k) for k in KS},
            "mean_ce_at_k": {str(k): _mean_over(blocks, "mean_ce", src, k) for k in KS},
            "success_at_k": {
                str(k): sum(1 for b in blocks if b.hit[src][k]) / len(blocks) if blocks else 0.0
                for k in KS
            },
        }
    # Random averaged across blocks (mean of per-block means).
    out["random"] = {
        "recall_at_k": {
            str(k): mean(b.random_recall[str(k)].mean for b in blocks) if blocks else 0.0
            for k in KS
        },
        "best_ce_at_k": {
            str(k): mean(b.random_best_ce[str(k)].mean for b in blocks) if blocks else 0.0
            for k in KS
        },
    }
    # Lift over random (overall recall ratio).
    for src in SOURCE_ACTUAL:
        out[src]["lift_at_k"] = {
            str(k): lift_at_k(out[src]["recall_at_k"][str(k)], out["random"]["recall_at_k"][str(k)])
            for k in KS
        }
    # Spearman aggregate.
    out["spearman"] = {
        src: _summarize_optional([b.spearman[src] for b in blocks]) for src in SOURCE_ACTUAL
    }
    return out


def _summarize_optional(values: list[float | None]) -> dict[str, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": mean(present),
        "median": median(present),
        "std": pstdev(present) if len(present) > 1 else 0.0,
    }


def breakdown_by_appearance(blocks: list[BlockMetrics]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, list[BlockMetrics]] = {}
    for b in blocks:
        by_type.setdefault(b.appearance_type, []).append(b)
    out: dict[str, dict[str, Any]] = {}
    for app_type, sub in by_type.items():
        out[app_type] = {
            "block_count": len(sub),
            **{src: {"recall_at_k": {str(k): _mean_over(sub, "recall", src, k) for k in KS},
                     "best_ce_at_k": {str(k): _mean_over(sub, "best_ce", src, k) for k in KS},
                     "success_at_k": {str(k): sum(1 for b in sub if b.hit[src][k]) / len(sub) for k in KS}}
                for src in SOURCE_ACTUAL},
        }
    return out


def breakdown_by_atom_type(atom_rows: list[dict[str, Any]], *, tau_ce: float = 0.30) -> dict[str, dict[str, Any]]:
    """Recall@K of positive (high-CE) roots grouped by atom type.

    ``atom_rows`` are the atom-level records (appearance_id, atom_type, CE,
    teacher_rank, ...).  For each atom type, count how many high-CE roots the
    teacher's top-K recovers.
    """
    out: dict[str, dict[str, Any]] = {}
    for atom_type in {r["atom_type"] for r in atom_rows}:
        positives = [r for r in atom_rows if r["CE"] >= tau_ce and r["atom_type"] == atom_type]
        if not positives:
            out[atom_type] = {"positive_count": 0, "recall_at_k": {str(k): 0.0 for k in KS}}
            continue
        # Recover by teacher rank within each block independently.
        by_block: dict[str, list[dict]] = {}
        for r in positives:
            by_block.setdefault(r["appearance_id"], []).append(r)
        mean_recall = {}
        for k in KS:
            per_block = []
            for block_id, rows in by_block.items():
                top = [r for r in rows if r.get("teacher_rank", 10 ** 9) <= k]
                per_block.append(len(top) / len(rows))
            mean_recall[str(k)] = mean(per_block)
        out[atom_type] = {"positive_count": len(positives), "recall_at_k": mean_recall}
    return out


def missed_high_ce_roots(atom_rows: list[dict[str, Any]], *, tau_ce: float = 0.30,
                         max_rank: int = 10) -> list[dict[str, Any]]:
    """High-CE roots the teacher ranked beyond ``max_rank``, strongest CE first."""
    missed = [
        r for r in atom_rows
        if r["CE"] >= tau_ce and r.get("teacher_rank", 10 ** 9) > max_rank
    ]
    return sorted(missed, key=lambda r: -r["CE"])


def teacher_false_top(atom_rows: list[dict[str, Any]], *, top_k: int = 10,
                      ce_threshold: float = 0.05) -> list[dict[str, Any]]:
    """Teacher top-K candidates whose measured CE is low/negative (false positives),
    ranked by teacher_score descending."""
    false_ = [
        r for r in atom_rows
        if r.get("teacher_rank", 10 ** 9) <= top_k and r["CE"] < ce_threshold
    ]
    return sorted(false_, key=lambda r: -r.get("teacher_score", 0.0))


def pairwise_comparison(blocks: list[BlockMetrics]) -> dict[str, dict[str, Any]]:
    """Per-block Teacher vs MaxPlus / PageRank / Random: wins / ties / loses.

    Compared at Recall@3 and Recall@5; Random uses the mean of its baseline.
    """
    out: dict[str, dict[str, Any]] = {}
    for other in ("max_plus", "page_rank", "random"):
        cells = {"Recall@3": {"wins": 0, "ties": 0, "loses": 0},
                 "Recall@5": {"wins": 0, "ties": 0, "loses": 0}}
        for b in blocks:
            for k in (3, 5):
                t = b.recall["teacher"][k]
                if other == "random":
                    o = b.random_recall[str(k)].mean
                else:
                    o = b.recall[other][k]
                if t > o + 1e-9:
                    cells[f"Recall@{k}"]["wins"] += 1
                elif abs(t - o) <= 1e-9:
                    cells[f"Recall@{k}"]["ties"] += 1
                else:
                    cells[f"Recall@{k}"]["loses"] += 1
        out[other] = cells
    return out