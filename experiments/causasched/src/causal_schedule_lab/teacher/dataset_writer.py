"""Gold Set writer (教师模型.md §21/§24).

Writes one JSONL line per appearance following the §21 Gold format: instance /
state identity, the target appearance identity, the teacher candidate scores
(prior only — never the label), the atomic probe records, and the joint root
set.  Failed experiments (``CE ~ 0``, ``CE < 0``, infeasible) are preserved as
hard-negative / failure data, never silently dropped (§24).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterable

GOLD_SET_VERSION = "acct-gold-v1"


def _sorted_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: (r.get("atom_id", ""), str(r.get("probe", {}))))

def write_gold_samples(
    samples: Iterable[dict[str, Any]],
    path: str | Path,
) -> int:
    """Write Gold samples to a JSONL file.  Returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {"gold_set_version": GOLD_SET_VERSION, **sample}
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_gold_sample(
    *,
    instance_id: str,
    state_id: str,
    appearance_type: str,
    appearance_target: dict[str, Any],
    appearance_before: float,
    delta: float,
    teacher: list[dict[str, Any]],
    atomic_probes: Iterable[dict[str, Any]],
    root_set: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Assemble one §21 Gold sample dict."""
    return {
        "instance_id": instance_id,
        "state_id": state_id,
        "appearance": {
            "type": appearance_type,
            "target_id": appearance_target,
            "score_before": appearance_before,
            "features": {},
        },
        "teacher": {
            "delta": delta,
            "candidate_scores": _sorted_records(teacher),
        },
        "atomic_probes": _sorted_records(atomic_probes),
        "root_set": root_set,
        "solver_seed": seed,
    }