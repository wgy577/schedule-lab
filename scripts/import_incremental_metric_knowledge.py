#!/usr/bin/env python3
"""Vendor audited incremental metric packages into the project knowledge tree.

The source packages are additions only.  This script deliberately preserves their
original files so provenance can be reviewed after import; runtime merging is done
by stable candidate/view IDs in ``SecondaryMetricKnowledgeBase``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_FILES = (
    "README.md",
    "candidate_metrics_additions.jsonl",
    "candidate_diagnostics_additions.jsonl",
    "candidate_view_memberships_additions.jsonl",
    "view_registry_additions.json",
    "source_manifest_additions.json",
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def import_package(source: Path, target: Path) -> dict[str, int]:
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"{source} is missing required files: {missing}")
    metrics = _jsonl(source / "candidate_metrics_additions.jsonl")
    diagnostics = _jsonl(source / "candidate_diagnostics_additions.jsonl")
    memberships = _jsonl(source / "candidate_view_memberships_additions.jsonl")
    metric_ids = [str(item["metric_id"]) for item in metrics]
    diagnostic_ids = [str(item["diagnostic_id"]) for item in diagnostics]
    membership_keys = [
        (str(item["candidate_id"]), str(item["view_id"]))
        for item in memberships
    ]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"duplicate metric IDs in {source}")
    if len(diagnostic_ids) != len(set(diagnostic_ids)):
        raise ValueError(f"duplicate diagnostic IDs in {source}")
    duplicate_membership_rows = len(membership_keys) - len(set(membership_keys))
    external_membership_ids = {
        candidate_id for candidate_id, _ in membership_keys
    } - set(metric_ids) - set(diagnostic_ids)
    if any(item.get("promotion_status") != "proposed" for item in metrics):
        raise ValueError("incremental metrics must remain proposed")
    if any(item.get("promotion_status") != "proposed" for item in diagnostics):
        raise ValueError("incremental diagnostics must remain proposed")

    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return {
        "metrics": len(metrics),
        "diagnostics": len(diagnostics),
        "memberships": len(memberships),
        "duplicate_membership_rows_preserved_for_runtime_merge": (
            duplicate_membership_rows
        ),
        "memberships_referencing_prior_round_ids": len(external_membership_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round5", type=Path, required=True)
    parser.add_argument("--round6", type=Path, required=True)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("src/causal_schedule_lab/knowledge/secondary_metrics"),
    )
    args = parser.parse_args()
    report = {
        "round5": import_package(args.round5.resolve(), args.target_root / "round5"),
        "round6": import_package(args.round6.resolve(), args.target_root / "round6"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
