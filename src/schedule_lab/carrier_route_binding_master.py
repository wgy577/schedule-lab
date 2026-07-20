from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .carrier_spacetime import solve_fixed_route_spacetime


def _resampled_duration(path: Path, time_resolution: float) -> float:
    tf = float(loadmat(path, variable_names=["tf"])["tf"].reshape(-1)[0])
    return float(np.arange(0.0, tf + time_resolution, time_resolution)[-1])


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tractor_successors(raw_schedule: list[dict[str, Any]]) -> dict[int, int]:
    by_tractor: dict[int, list[dict[str, Any]]] = {}
    for row in raw_schedule:
        if int(row["op"]) == 0:
            by_tractor.setdefault(int(row["machine"]), []).append(row)
    successors: dict[int, int] = {}
    for rows in by_tractor.values():
        ordered = sorted(rows, key=lambda row: (float(row["start"]), int(row["job"])))
        for left, right in zip(ordered, ordered[1:]):
            successors[int(left["job"])] = int(right["job"])
    return successors


def _apply_preparation_binding(
    raw_schedule: list[dict[str, Any]],
    *,
    job: int,
    preparation_index: int,
    legacy_root: Path,
    time_resolution: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = deepcopy(raw_schedule)
    by_key = {(int(row["job"]), int(row["op"])): row for row in candidate}
    old_machine = int(by_key[(job, 2)]["machine"])
    new_preparation_machine = 5 + preparation_index
    new_catapult_machine = 8 + preparation_index
    system_path = legacy_root / "systemtraject" / f"J{job + 1}M{4 + preparation_index}.mat"
    towing_duration = _resampled_duration(system_path, time_resolution)
    for operation in (2, 3):
        by_key[(job, operation)]["machine"] = new_preparation_machine
    for operation in (4, 5, 7):
        by_key[(job, operation)]["machine"] = new_catapult_machine
    by_key[(job, 2)]["dur"] = towing_duration
    by_key[(job, 2)]["end"] = float(by_key[(job, 2)]["start"]) + towing_duration

    successor = _tractor_successors(raw_schedule).get(job)
    successor_change = None
    if successor is not None:
        return_path = legacy_root / "trajectory" / f"M{4 + preparation_index}J{successor + 1}.mat"
        return_duration = _resampled_duration(return_path, time_resolution)
        successor_row = by_key[(successor, 0)]
        successor_row["dur"] = return_duration
        successor_row["end"] = float(successor_row["start"]) + return_duration
        successor_change = {
            "job": successor,
            "operation": 0,
            "route": str(return_path),
            "duration": return_duration,
        }
    return candidate, {
        "job": job,
        "oldPreparationMachine": old_machine,
        "newPreparationMachine": new_preparation_machine,
        "newCatapultMachine": new_catapult_machine,
        "towingRoute": str(system_path),
        "towingDuration": towing_duration,
        "tractorSuccessorRouteChange": successor_change,
    }


def search_route_binding_master(
    schedule_payload: dict[str, Any],
    closure: dict[str, Any],
    *,
    legacy_root: str | Path,
    time_resolution: float = 0.1,
    max_jobs: int = 2,
    max_candidates: int = 4,
    max_conflicts: int = 500_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Enumerate a tiny deterministic binding neighborhood and solve timing exactly."""

    root = Path(legacy_root).expanduser().resolve()
    incumbent = list(schedule_payload["schedule"])
    by_key = {(int(row["job"]), int(row["op"])): row for row in incumbent}
    route_duration_options = {
        job: [
            _resampled_duration(
                root / "systemtraject" / f"J{job + 1}M{4 + preparation_index}.mat",
                time_resolution,
            )
            for preparation_index in range(3)
        ]
        for job in map(int, closure["releasedJobs"])
    }
    ranked_jobs = sorted(
        route_duration_options,
        key=lambda job: (
            -(
                float(by_key[(job, 2)]["dur"])
                - min(route_duration_options[job])
            ),
            -float(by_key[(job, 6)]["start"]),
            job,
        ),
    )[:max_jobs]
    proposals: list[dict[str, Any]] = []
    for job in ranked_jobs:
        current_preparation = int(by_key[(job, 2)]["machine"]) - 5
        for preparation_index in range(3):
            if preparation_index == current_preparation:
                continue
            if len(proposals) >= max_candidates:
                break
            raw_candidate, binding_change = _apply_preparation_binding(
                incumbent,
                job=job,
                preparation_index=preparation_index,
                legacy_root=root,
                time_resolution=time_resolution,
            )
            result = solve_fixed_route_spacetime(
                {"schedule": raw_candidate, "decisionTrace": schedule_payload.get("decisionTrace")},
                legacy_root=root,
                time_resolution=time_resolution,
                max_conflicts=max_conflicts,
                seed=seed,
                released_jobs=set(map(int, closure["releasedJobs"])),
                horizon_slack_ticks=int(round(300.0 / time_resolution)),
            )
            candidate = result.get("candidate")
            proposals.append(
                {
                    "proposalIndex": len(proposals),
                    "proposalHash": _hash({"binding": binding_change, "seed": seed}),
                    "bindingChange": binding_change,
                    "status": result["status"],
                    "incumbentExactGridConflicts": result["incumbent"].get(
                        "exactGridConflictCount",
                        result.get("conflicts", {}).get("incumbentConflictCount"),
                    ),
                    "candidateMakespan": None if candidate is None else candidate["makespan"],
                    "candidateHash": None if candidate is None else candidate["hash"],
                    "genericValid": None if candidate is None else candidate["genericValid"],
                    "exactGridValid": None if candidate is None else candidate["exactGridValid"],
                    "proof": result.get("proof"),
                    "candidateSchedule": None if candidate is None else candidate["schedule"],
                    "sourceDecisionTrace": None
                    if candidate is None
                    else candidate.get("sourceDecisionTrace"),
                }
            )
        if len(proposals) >= max_candidates:
            break
    feasible = [
        proposal
        for proposal in proposals
        if proposal["genericValid"] and proposal["exactGridValid"]
    ]
    best = min(
        feasible,
        key=lambda proposal: (proposal["candidateMakespan"], proposal["proposalIndex"]),
        default=None,
    )
    return {
        "controller": "bounded-explicit-route-binding-master-v1",
        "sourceScheduleHash": _hash(incumbent),
        "sourceCutId": closure["sourceCutId"],
        "releasedJobs": closure["releasedJobs"],
        "rankedJobs": ranked_jobs,
        "routeDurationOptions": {
            str(job): values for job, values in sorted(route_duration_options.items())
        },
        "settings": {
            "timeResolution": time_resolution,
            "maxJobs": max_jobs,
            "maxCandidates": max_candidates,
            "maxConflicts": max_conflicts,
            "seed": seed,
            "workers": 1,
        },
        "candidateCount": len(proposals),
        "best": best,
        "proposals": [
            {key: value for key, value in proposal.items() if key not in {"candidateSchedule", "sourceDecisionTrace"}}
            for proposal in proposals
        ],
    }
