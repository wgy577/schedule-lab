from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ortools.sat.python import cp_model
from scipy.io import loadmat
from scipy.spatial import cKDTree

from .carrier import build_carrier_schedule
from .validation import validate_schedule


@dataclass(frozen=True)
class FixedRouteMotion:
    operation_id: str
    job: int
    operation: int
    machine: int
    start_tick: int
    duration_ticks: int
    radius: float
    source: str
    source_sha256: str
    points: np.ndarray


def _ticks(value: float, time_resolution: float) -> int:
    return int(round(float(value) / time_resolution))


def _resampled_duration(path: Path, time_resolution: float) -> float:
    tf = float(loadmat(path, variable_names=["tf"])["tf"].reshape(-1)[0])
    return float(np.arange(0.0, tf + time_resolution, time_resolution)[-1])


def _load_points(
    path: Path,
    *,
    x_key: str,
    y_key: str,
    duration_ticks: int,
    time_resolution: float,
) -> np.ndarray:
    data = loadmat(path, variable_names=[x_key, y_key, "tf"])
    tf = float(data["tf"].reshape(-1)[0])
    raw_x = np.asarray(data[x_key], dtype=float).reshape(-1)
    raw_y = np.asarray(data[y_key], dtype=float).reshape(-1)
    raw_time = np.linspace(0.0, tf, len(raw_x))
    # Half-open occupancy: the endpoint at operation end belongs to neither
    # movement when the next movement starts at exactly the same instant.
    sample_time = np.arange(duration_ticks, dtype=float) * time_resolution
    return np.column_stack(
        (
            np.interp(sample_time, raw_time, raw_x),
            np.interp(sample_time, raw_time, raw_y),
        )
    )


def _resolve_op0_source(
    row: dict[str, Any],
    *,
    legacy_root: Path,
    earliest_op0_by_machine: dict[int, float],
    time_resolution: float,
) -> Path:
    job = int(row["job"]) + 1
    machine = int(row["machine"])
    duration = float(row["dur"])
    candidates: list[Path] = []
    for route_machine in range(4, 12):
        directory = "trajectory" if route_machine <= 6 else "initialtraject"
        path = legacy_root / directory / f"M{route_machine}J{job}.mat"
        if abs(_resampled_duration(path, time_resolution) - duration) <= time_resolution / 2 + 1e-9:
            candidates.append(path)
    initial = legacy_root / "initialtraject" / f"M{machine + 7}J{job}.mat"
    is_first_on_tractor = math.isclose(
        float(row["start"]), earliest_op0_by_machine[machine], abs_tol=time_resolution / 2
    )
    if initial in candidates and is_first_on_tractor:
        return initial
    transfer = [path for path in candidates if path.parent.name == "trajectory"]
    if len(transfer) == 1:
        return transfer[0]
    raise ValueError(
        f"cannot uniquely resolve fixed route for J{job}.O1: duration={duration}, "
        f"machine={machine}, candidates={[str(path) for path in candidates]}"
    )


def load_fixed_route_motions(
    raw_schedule: list[dict[str, Any]],
    *,
    legacy_root: str | Path,
    time_resolution: float = 0.1,
) -> list[FixedRouteMotion]:
    """Resolve the exact incumbent MAT path for every collision-relevant movement."""

    root = Path(legacy_root).expanduser().resolve()
    earliest_op0 = {
        machine: min(
            float(row["start"])
            for row in raw_schedule
            if int(row["op"]) == 0 and int(row["machine"]) == machine
        )
        for machine in sorted(
            {int(row["machine"]) for row in raw_schedule if int(row["op"]) == 0}
        )
    }
    motions: list[FixedRouteMotion] = []
    for row in sorted(raw_schedule, key=lambda item: (int(item["job"]), int(item["op"]))):
        operation = int(row["op"])
        if operation not in (0, 2):
            continue
        job = int(row["job"])
        machine = int(row["machine"])
        duration_ticks = _ticks(float(row["dur"]), time_resolution)
        if operation == 0:
            source = _resolve_op0_source(
                row,
                legacy_root=root,
                earliest_op0_by_machine=earliest_op0,
                time_resolution=time_resolution,
            )
            x_key, y_key, radius = "x", "y", 3.0
        else:
            route_machine = machine - 1  # deck machines 5..7 map to MAT M4..M6
            source = root / "systemtraject" / f"J{job + 1}M{route_machine}.mat"
            x_key, y_key, radius = "xU", "yU", 10.0
        motions.append(
            FixedRouteMotion(
                operation_id=f"J{job + 1}.O{operation + 1}",
                job=job,
                operation=operation,
                machine=machine,
                start_tick=_ticks(float(row["start"]), time_resolution),
                duration_ticks=duration_ticks,
                radius=radius,
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                points=_load_points(
                    source,
                    x_key=x_key,
                    y_key=y_key,
                    duration_ticks=duration_ticks,
                    time_resolution=time_resolution,
                ),
            )
        )
    return motions


def _compress_offsets(offsets: set[int]) -> list[tuple[int, int]]:
    if not offsets:
        return []
    ordered = sorted(offsets)
    intervals: list[tuple[int, int]] = []
    lower = upper = ordered[0]
    for value in ordered[1:]:
        if value == upper + 1:
            upper = value
        else:
            intervals.append((lower, upper))
            lower = upper = value
    intervals.append((lower, upper))
    return intervals


def build_fixed_route_conflicts(motions: list[FixedRouteMotion]) -> dict[str, Any]:
    """Build exact forbidden relative-start intervals on the declared time grid."""

    pairs: list[dict[str, Any]] = []
    incumbent_conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(motions):
        left_tree = cKDTree(left.points)
        for right in motions[left_index + 1 :]:
            neighborhoods = left_tree.query_ball_tree(
                cKDTree(right.points), left.radius + right.radius
            )
            offsets: set[int] = set()
            for left_tick, right_ticks in enumerate(neighborhoods):
                offsets.update(left_tick - right_tick for right_tick in right_ticks)
            if not offsets:
                continue
            intervals = _compress_offsets(offsets)
            relative_start = right.start_tick - left.start_tick
            record = {
                "left": left.operation_id,
                "right": right.operation_id,
                "safeDistance": left.radius + right.radius,
                "forbiddenRelativeStartIntervals": [
                    {"lowerTick": lower, "upperTick": upper}
                    for lower, upper in intervals
                ],
                "incumbentRelativeStartTick": relative_start,
                "incumbentConflict": relative_start in offsets,
            }
            pairs.append(record)
            if record["incumbentConflict"]:
                incumbent_conflicts.append(record)
    return {
        "motionCount": len(motions),
        "spatiallyInteractingPairCount": len(pairs),
        "forbiddenIntervalCount": sum(
            len(pair["forbiddenRelativeStartIntervals"]) for pair in pairs
        ),
        "incumbentConflictCount": len(incumbent_conflicts),
        "incumbentConflicts": incumbent_conflicts,
        "pairs": pairs,
    }


def _schedule_hash(raw_schedule: list[dict[str, Any]]) -> str:
    normalized = json.dumps(raw_schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def solve_fixed_route_spacetime(
    payload: dict[str, Any],
    *,
    legacy_root: str | Path,
    time_resolution: float = 0.1,
    max_conflicts: int = 200_000,
    seed: int = 0,
    released_jobs: set[int] | None = None,
    horizon_slack_ticks: int = 0,
) -> dict[str, Any]:
    """Exactly repair timing for fixed MAT routes on a finite 0.1-second grid."""

    raw_schedule = list(payload["schedule"])
    motions = load_fixed_route_motions(
        raw_schedule, legacy_root=legacy_root, time_resolution=time_resolution
    )
    conflicts = build_fixed_route_conflicts(motions)
    row_by_key = {(int(row["job"]), int(row["op"])): row for row in raw_schedule}
    incumbent_start = {
        key: _ticks(float(row["start"]), time_resolution) for key, row in row_by_key.items()
    }
    duration = {key: _ticks(float(row["dur"]), time_resolution) for key, row in row_by_key.items()}
    incumbent_makespan = max(
        incumbent_start[key] + duration[key] for key in row_by_key
    )
    horizon = incumbent_makespan + max(0, int(horizon_slack_ticks))
    model = cp_model.CpModel()
    starts = {
        key: model.new_int_var(0, horizon - duration[key], f"start_{key[0]}_{key[1]}")
        for key in row_by_key
    }
    ends = {key: starts[key] + duration[key] for key in row_by_key}
    intervals = {
        key: model.new_interval_var(starts[key], duration[key], ends[key], f"op_{key[0]}_{key[1]}")
        for key in row_by_key
    }
    for job, operation in sorted(row_by_key):
        if operation > 0:
            model.add(starts[(job, operation)] >= ends[(job, operation - 1)])

    by_machine: dict[int, list[tuple[int, int]]] = {}
    for key, row in row_by_key.items():
        by_machine.setdefault(int(row["machine"]), []).append(key)
    released_jobs = set() if released_jobs is None else set(map(int, released_jobs))
    for machine, keys in sorted(by_machine.items()):
        ordered = sorted(keys, key=lambda key: (incumbent_start[key], key))
        capacity = 2 if machine in (5, 6, 7) else 1
        if capacity == 1:
            model.add_no_overlap([intervals[key] for key in keys])
            separation = _ticks(10.0, time_resolution) if machine == 11 else 0
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    flexible = left[0] in released_jobs or right[0] in released_jobs
                    if not flexible:
                        model.add(starts[right] >= ends[left] + separation)
                    elif machine == 11:
                        left_first = model.new_bool_var(
                            f"launch_order_{left[0]}_{right[0]}"
                        )
                        model.add(starts[right] >= ends[left] + separation).only_enforce_if(
                            left_first
                        )
                        model.add(starts[left] >= ends[right] + separation).only_enforce_if(
                            left_first.Not()
                        )
        else:
            model.add_cumulative([intervals[key] for key in keys], [1] * len(keys), capacity)
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    if incumbent_start[left] + duration[left] <= incumbent_start[right]:
                        model.add(starts[right] >= ends[left])

    # Domain locks that are not representable by individual operation intervals.
    # A tractor remains committed from initial movement through the end of the
    # coupled towing sequence (O1..O4).
    tractor_jobs: dict[int, list[int]] = {}
    for job in sorted({key[0] for key in row_by_key}):
        tractor_jobs.setdefault(int(row_by_key[(job, 0)]["machine"]), []).append(job)
    for jobs in tractor_jobs.values():
        ordered_jobs = sorted(jobs, key=lambda job: (incumbent_start[(job, 0)], job))
        tractor_intervals = []
        for job in ordered_jobs:
            span = model.new_int_var(1, horizon, f"tractor_span_{job}")
            model.add(span == ends[(job, 3)] - starts[(job, 0)])
            tractor_intervals.append(
                model.new_interval_var(
                    starts[(job, 0)], span, ends[(job, 3)], f"tractor_lock_{job}"
                )
            )
        model.add_no_overlap(tractor_intervals)
        for left_job, right_job in zip(ordered_jobs, ordered_jobs[1:]):
            model.add(starts[(right_job, 0)] >= ends[(left_job, 3)])

    # A selected catapult/lane remains occupied from entry (O5) until the
    # aircraft has completed the launch stage (O8).
    catapult_jobs: dict[int, list[int]] = {}
    for job in sorted({key[0] for key in row_by_key}):
        catapult_jobs.setdefault(int(row_by_key[(job, 4)]["machine"]), []).append(job)
    for jobs in catapult_jobs.values():
        ordered_jobs = sorted(jobs, key=lambda job: (incumbent_start[(job, 4)], job))
        catapult_intervals = []
        for job in ordered_jobs:
            span = model.new_int_var(1, horizon, f"catapult_span_{job}")
            model.add(span == ends[(job, 7)] - starts[(job, 4)])
            catapult_intervals.append(
                model.new_interval_var(
                    starts[(job, 4)], span, ends[(job, 7)], f"catapult_lock_{job}"
                )
            )
        model.add_no_overlap(catapult_intervals)
        for left_job, right_job in zip(ordered_jobs, ordered_jobs[1:]):
            if left_job not in released_jobs and right_job not in released_jobs:
                model.add(starts[(right_job, 4)] >= ends[(left_job, 7)])

    motion_key = {motion.operation_id: (motion.job, motion.operation) for motion in motions}
    for pair_index, pair in enumerate(conflicts["pairs"]):
        left = motion_key[pair["left"]]
        right = motion_key[pair["right"]]
        delta = starts[right] - starts[left]
        for interval_index, forbidden in enumerate(pair["forbiddenRelativeStartIntervals"]):
            choose_left = model.new_bool_var(f"avoid_{pair_index}_{interval_index}")
            model.add(delta <= int(forbidden["lowerTick"]) - 1).only_enforce_if(choose_left)
            model.add(delta >= int(forbidden["upperTick"]) + 1).only_enforce_if(
                choose_left.Not()
            )

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(ends.values()))
    for key, value in starts.items():
        model.add_hint(value, incumbent_start[key])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_number_of_conflicts = max_conflicts
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "solver": "fixed-route-spacetime-cp-sat-v1",
            "status": status_name,
            "incumbent": {
                "hash": _schedule_hash(raw_schedule),
                "makespan": incumbent_makespan * time_resolution,
            },
            "conflicts": conflicts,
            "candidate": None,
        }

    best_makespan = int(solver.value(makespan))
    candidate = []
    for row in raw_schedule:
        key = (int(row["job"]), int(row["op"]))
        start = int(solver.value(starts[key])) * time_resolution
        dur = duration[key] * time_resolution
        candidate.append({**row, "start": start, "dur": dur, "end": start + dur})
    candidate_motions = load_fixed_route_motions(
        candidate, legacy_root=legacy_root, time_resolution=time_resolution
    )
    candidate_conflicts = build_fixed_route_conflicts(candidate_motions)
    problem, canonical = build_carrier_schedule(
        candidate,
        source="fixed-route-spacetime-cp-sat-v1",
        schedule_metadata={"solver": "fixed-route-spacetime-cp-sat-v1"},
        domain_validated=False,
    )
    generic_validation = validate_schedule(problem, canonical)
    exact_grid_valid = candidate_conflicts["incumbentConflictCount"] == 0
    return {
        "solver": "fixed-route-spacetime-cp-sat-v1",
        "status": status_name,
        "settings": {
            "timeResolution": time_resolution,
            "maxConflicts": max_conflicts,
            "workers": 1,
            "seed": seed,
            "routePolicy": "fixed incumbent MAT geometry; half-open occupancy",
            "releasedJobs": sorted(released_jobs),
            "horizonSlackTicks": int(horizon_slack_ticks),
        },
        "incumbent": {
            "hash": _schedule_hash(raw_schedule),
            "makespan": incumbent_makespan * time_resolution,
            "exactGridConflictCount": conflicts["incumbentConflictCount"],
        },
        "conflicts": conflicts,
        "candidate": {
            "hash": _schedule_hash(candidate),
            "makespan": best_makespan * time_resolution,
            "improvement": (incumbent_makespan - best_makespan) * time_resolution,
            "genericValid": generic_validation.feasible,
            "genericErrors": generic_validation.errors,
            "exactGridValid": exact_grid_valid,
            "remainingConflictCount": candidate_conflicts["incumbentConflictCount"],
            "provisional": True,
            "acceptance": (
                "requires legacy domain replay and deterministic reproduction"
                if generic_validation.feasible and exact_grid_valid
                else "reject"
            ),
            "schedule": candidate,
            "sourceDecisionTrace": payload.get("decisionTrace") or payload.get("decision_trace"),
        },
        "proof": {
            "bestObjectiveBound": solver.best_objective_bound * time_resolution,
            "objectiveValue": solver.objective_value * time_resolution,
            "optimalForDeclaredFixedRouteGrid": status == cp_model.OPTIMAL,
        },
    }
