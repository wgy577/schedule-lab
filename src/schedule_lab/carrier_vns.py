from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FOCUS_OPERATION = 6
FOCUS_MACHINE = 11
DEFAULT_RADII = (2, 3, 4)


def _normalized_schedule(raw_schedule: Iterable[dict[str, Any]]) -> list[dict[str, int]]:
    return [
        {
            "job": int(item["job"]),
            "op": int(item["op"]),
            "machine": int(item["machine"]),
            "start": int(round(float(item["start"]) * 1000)),
            "end": int(round(float(item["end"]) * 1000)),
        }
        for item in sorted(raw_schedule, key=lambda entry: (int(entry["job"]), int(entry["op"])))
    ]


def schedule_hash(raw_schedule: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(_normalized_schedule(raw_schedule), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def focus_sequence(
    raw_schedule: Iterable[dict[str, Any]],
    *,
    operation: int = FOCUS_OPERATION,
    machine: int = FOCUS_MACHINE,
) -> list[dict[str, Any]]:
    selected = [
        {
            "job": int(item["job"]),
            "op": int(item["op"]),
            "machine": int(item["machine"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in raw_schedule
        if int(item["op"]) == operation and int(item["machine"]) == machine
    ]
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["job"]))


def rank_focus_gaps(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(sequence, sequence[1:])):
        gap = max(0.0, float(right["start"]) - float(left["end"]))
        if gap <= 1e-6:
            continue
        gaps.append(
            {
                "sequenceIndex": index,
                "gap": round(gap, 6),
                "leftJob": int(left["job"]),
                "rightJob": int(right["job"]),
                "leftEnd": float(left["end"]),
                "rightStart": float(right["start"]),
            }
        )
    return sorted(
        gaps,
        key=lambda item: (
            -item["gap"],
            item["leftEnd"],
            item["leftJob"],
            item["rightJob"],
        ),
    )


def _signature(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_tabu_signatures(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    source = Path(path)
    if not source.exists():
        return set()
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    return {
        str(record["signature"] if isinstance(record, dict) else record)
        for record in records
    }


def build_vns_plan(
    raw_schedule: list[dict[str, Any]],
    *,
    radii: tuple[int, ...] = DEFAULT_RADII,
    max_gaps: int = 5,
    tabu_signatures: set[str] | None = None,
) -> dict[str, Any]:
    if not raw_schedule:
        raise ValueError("schedule is empty")
    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("radii must contain positive integers")
    if max_gaps < 1:
        raise ValueError("max_gaps must be positive")
    tabu_signatures = tabu_signatures or set()
    sequence = focus_sequence(raw_schedule)
    if len(sequence) < 2:
        raise ValueError("focus sequence must contain at least two operations")
    ranked_gaps = rank_focus_gaps(sequence)
    plans: list[dict[str, Any]] = []
    for gap_rank, gap in enumerate(ranked_gaps[:max_gaps], start=1):
        boundary = int(gap["sequenceIndex"])
        for radius in radii:
            start_index = max(0, boundary - radius + 1)
            end_index = min(len(sequence), boundary + radius + 1)
            neighborhood_jobs = tuple(int(item["job"]) for item in sequence[start_index:end_index])
            released = tuple(
                {"job": job, "op": operation}
                for job in neighborhood_jobs
                for operation in range(8)
            )
            signature_payload = {
                "incumbent": schedule_hash(raw_schedule),
                "focusOperation": FOCUS_OPERATION,
                "focusMachine": FOCUS_MACHINE,
                "gapRank": gap_rank,
                "boundaryJobs": [gap["leftJob"], gap["rightJob"]],
                "radius": radius,
                "neighborhoodJobs": neighborhood_jobs,
            }
            signature = _signature(signature_payload)
            plans.append(
                {
                    "candidateIndex": len(plans),
                    "signature": signature,
                    "tabu": signature in tabu_signatures,
                    "gapRank": gap_rank,
                    "gap": gap,
                    "radius": radius,
                    "sequenceSlice": [start_index, end_index],
                    "neighborhoodJobs": list(neighborhood_jobs),
                    "releasedOperations": list(released),
                    "releasedOperationCount": len(released),
                    "frozenOperationCount": len(raw_schedule) - len(released),
                    "operatorOrder": [
                        "cp-sat-local-repair",
                        "adjacent-boundary-swap",
                        "forward-insertion",
                        "backward-insertion",
                    ],
                    "propagation": {
                        "freezePrefixBefore": min(
                            float(item["start"])
                            for item in raw_schedule
                            if int(item["job"]) in neighborhood_jobs
                        ),
                        "allowTimingShiftOutsideNeighborhood": True,
                        "allowModeChangeOutsideNeighborhood": False,
                    },
                }
            )
    makespan = max(float(item["end"]) for item in raw_schedule)
    total_gap = sum(item["gap"] for item in ranked_gaps)
    return {
        "incumbentHash": schedule_hash(raw_schedule),
        "focus": {
            "operation": FOCUS_OPERATION,
            "machine": FOCUS_MACHINE,
            "sequenceJobs": [int(item["job"]) for item in sequence],
            "positiveGapTotal": round(total_gap, 6),
            "largestGap": ranked_gaps[0] if ranked_gaps else None,
        },
        "incumbentMakespan": makespan,
        "searchProtocol": {
            "controller": "deterministic-vns",
            "gapOrder": "descending-gap-then-time-then-job",
            "radii": list(radii),
            "maxGaps": max_gaps,
            "acceptance": "strict-validated-improvement",
            "restartAfterAcceptance": True,
        },
        "neighborhoods": plans,
    }
