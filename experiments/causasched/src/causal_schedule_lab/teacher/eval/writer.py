"""Phase-2 artifact writers (教师模型.md Phase2加强 §23/§30)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .reporting import BlockMetrics, SOURCE_ACTUAL


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})
    return len(rows)


def block_level_csv_rows(blocks: list[BlockMetrics]) -> list[dict[str, Any]]:
    """Flatten per-block metrics into the §23 block-level CSV schema."""
    rows: list[dict[str, Any]] = []
    for b in blocks:
        row: dict[str, Any] = {
            "appearance_id": b.appearance_id,
            "appearance_type": b.appearance_type,
            "candidate_count": b.candidate_count,
            "positive_count": b.positive_count,
            "positive_ratio": round(b.positive_ratio, 6),
        }
        for src in SOURCE_ACTUAL:
            row[f"{src}_R1"] = round(b.recall[src][1], 6)
            row[f"{src}_R3"] = round(b.recall[src][3], 6)
            row[f"{src}_R5"] = round(b.recall[src][5], 6)
            row[f"{src}_R10"] = round(b.recall[src][10], 6)
        for src in SOURCE_ACTUAL:
            row[f"{src}_BestCE3"] = round(b.best_ce[src][3], 6)
        row["random_R3_mean"] = round(b.random_recall["3"].mean, 6)
        row["random_R3_std"] = round(b.random_recall["3"].std, 6)
        row["random_BestCE3_mean"] = round(b.random_best_ce["3"].mean, 6)
        row["random_BestCE3_std"] = round(b.random_best_ce["3"].std, 6)
        for src in SOURCE_ACTUAL:
            row[f"{src}_spearman"] = "" if b.spearman[src] is None else round(b.spearman[src], 6)
        rows.append(row)
    return rows


def block_level_fieldnames() -> list[str]:
    return [
        "appearance_id", "appearance_type", "candidate_count", "positive_count", "positive_ratio",
        "teacher_R1", "teacher_R3", "teacher_R5", "teacher_R10",
        "max_plus_R1", "max_plus_R3", "max_plus_R5", "max_plus_R10",
        "page_rank_R1", "page_rank_R3", "page_rank_R5", "page_rank_R10",
        "teacher_BestCE3", "max_plus_BestCE3", "page_rank_BestCE3",
        "random_R3_mean", "random_R3_std", "random_BestCE3_mean", "random_BestCE3_std",
        "teacher_spearman", "max_plus_spearman", "page_rank_spearman",
    ]


def overall_metrics_csv_rows(overall: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per method with the §20 final table columns."""
    rows: list[dict[str, Any]] = []
    for method in (*SOURCE_ACTUAL, "random"):
        data = overall[method]
        row = {"method": method}
        for k in (1, 3, 5, 10):  # Rec@
            row[f"R@{k}"] = round(data["recall_at_k"][str(k)], 6)
        for k in (3, 5):
            row[f"P@{k}"] = round(data.get("precision_at_k", {}).get(str(k), 0.0), 6)
            row[f"BestCE@{k}"] = round(data.get("best_ce_at_k", {}).get(str(k), 0.0), 6)
        row["MeanCE@3"] = round(data.get("mean_ce_at_k", {}).get("3", 0.0), 6)
        row["Success@3"] = round(data.get("success_at_k", {}).get("3", 0.0), 6)
        rows.append(row)
    return rows


def by_appearance_csv_rows(by_appearance: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for app_type, data in sorted(by_appearance.items()):
        for method in SOURCE_ACTUAL:
            m = data[method]
            rows.append({
                "appearance_type": app_type,
                "method": method,
                "block_count": data["block_count"],
                "R@3": round(m["recall_at_k"]["3"], 6),
                "R@5": round(m["recall_at_k"]["5"], 6),
                "BestCE@3": round(m["best_ce_at_k"]["3"], 6),
                "Success@3": round(m["success_at_k"]["3"], 6),
            })
    return rows


def by_atom_csv_rows(by_atom: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom_type, data in sorted(by_atom.items()):
        rows.append({
            "atom_type": atom_type,
            "positive_count": data["positive_count"],
            "R@3": round(data["recall_at_k"]["3"], 6),
            "R@5": round(data["recall_at_k"]["5"], 6),
        })
    return rows