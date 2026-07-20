from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .carrier_vns import FOCUS_MACHINE, FOCUS_OPERATION, focus_sequence, rank_focus_gaps, schedule_hash


DEFAULT_DESTROY_SIZES = (2, 3)


def _signature(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_alns_plan(
    raw_schedule: list[dict[str, Any]],
    *,
    destroy_sizes: Iterable[int] = DEFAULT_DESTROY_SIZES,
    radius: int = 2,
    max_gaps: int = 4,
    max_jobs: int = 14,
    gap_ranks: Iterable[int] | None = None,
    expansion_jobs: Iterable[int] = (),
    tabu_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic multi-gap destroy neighborhoods after VNS plateaus."""

    if not raw_schedule:
        raise ValueError("schedule is empty")
    destroy_sizes = tuple(dict.fromkeys(map(int, destroy_sizes)))
    if not destroy_sizes or any(size < 1 for size in destroy_sizes):
        raise ValueError("destroy_sizes must contain positive integers")
    if radius < 1 or max_gaps < 1 or max_jobs < 4:
        raise ValueError("invalid ALNS neighborhood bounds")
    tabu_signatures = tabu_signatures or set()
    sequence = focus_sequence(raw_schedule)
    all_ranked_gaps = rank_focus_gaps(sequence)[:max_gaps]
    selected_rank_set = set(map(int, gap_ranks or ()))
    ranked_gaps = [
        gap
        for rank, gap in enumerate(all_ranked_gaps, start=1)
        if not selected_rank_set or rank in selected_rank_set
    ]
    if not ranked_gaps:
        raise ValueError("at least one positive focus-stage gap is required")

    sequence_jobs = [int(item["job"]) for item in sequence]
    expansion_job_set = set(map(int, expansion_jobs))
    unknown_expansion_jobs = expansion_job_set - set(sequence_jobs)
    if unknown_expansion_jobs:
        raise ValueError(f"unknown expansion jobs: {sorted(unknown_expansion_jobs)}")
    plans: list[dict[str, Any]] = []
    selected_gap_sets = []
    for destroy_size in destroy_sizes:
        if destroy_size == 1:
            selected_gap_sets.extend(([gap] for gap in ranked_gaps))
        else:
            selected_gap_sets.append(ranked_gaps[: min(destroy_size, len(ranked_gaps))])
    for selected_gaps in selected_gap_sets:
        job_scores: dict[int, tuple[int, int, int]] = {}
        for gap_rank, gap in enumerate(selected_gaps, start=1):
            boundary = int(gap["sequenceIndex"])
            start = max(0, boundary - radius + 1)
            end = min(len(sequence), boundary + radius + 1)
            for sequence_index in range(start, end):
                job = sequence_jobs[sequence_index]
                distance = min(abs(sequence_index - boundary), abs(sequence_index - (boundary + 1)))
                score = (gap_rank, distance, sequence_index)
                job_scores[job] = min(job_scores.get(job, score), score)
        ranked_neighborhood_jobs = [
            job
            for job, _ in sorted(job_scores.items(), key=lambda item: (item[1], item[0]))[:max_jobs]
        ]
        neighborhood_job_set = set(ranked_neighborhood_jobs) | expansion_job_set
        if len(neighborhood_job_set) > max_jobs:
            retained = list(expansion_job_set)
            retained.extend(job for job in ranked_neighborhood_jobs if job not in expansion_job_set)
            neighborhood_job_set = set(retained[:max_jobs])
        ordered_neighborhood_jobs = tuple(job for job in sequence_jobs if job in neighborhood_job_set)
        released = tuple(
            {"job": job, "op": operation}
            for job in ordered_neighborhood_jobs
            for operation in range(8)
        )
        signature_payload = {
            "incumbent": schedule_hash(raw_schedule),
            "controller": "deterministic-alns",
            "destroyOperator": "top-gap-union",
            "destroySize": len(selected_gaps),
            "radius": radius,
            "selectedGapBoundaries": [
                [int(gap["leftJob"]), int(gap["rightJob"])] for gap in selected_gaps
            ],
            "neighborhoodJobs": ordered_neighborhood_jobs,
            "expansionJobs": sorted(expansion_job_set),
        }
        signature = _signature(signature_payload)
        plans.append(
            {
                "candidateIndex": len(plans),
                "signature": signature,
                "tabu": signature in tabu_signatures,
                "gapRank": 0,
                "gap": selected_gaps[0],
                "selectedGaps": selected_gaps,
                "radius": radius,
                "destroyOperator": "top-gap-union",
                "destroySize": len(selected_gaps),
                "overrideBudget": len(selected_gaps),
                "neighborhoodJobs": list(ordered_neighborhood_jobs),
                "expansionJobs": sorted(expansion_job_set),
                "releasedOperations": list(released),
                "releasedOperationCount": len(released),
                "frozenOperationCount": len(raw_schedule) - len(released),
                "operatorOrder": [
                    "alns-incumbent-replay",
                    "alns-adjacent-o6",
                    "alns-adjacent-o5",
                    "alns-adjacent-o4",
                    "alns-adjacent-o3",
                    "alns-boundary-pull",
                    "alns-sink-order",
                    "alns-wait-first",
                    "cp-sat-fixed-mode",
                ],
                "propagation": {
                    "allowTimingShiftOutsideNeighborhood": True,
                    "allowModeChangeOutsideNeighborhood": False,
                },
            }
        )

    return {
        "incumbentHash": schedule_hash(raw_schedule),
        "incumbentMakespan": max(float(item["end"]) for item in raw_schedule),
        "focus": {
            "operation": FOCUS_OPERATION,
            "machine": FOCUS_MACHINE,
            "sequenceJobs": sequence_jobs,
            "positiveGapTotal": round(sum(float(item["gap"]) for item in rank_focus_gaps(sequence)), 6),
            "selectedGaps": ranked_gaps,
        },
        "searchProtocol": {
            "controller": "deterministic-alns",
            "destroyOperator": "top-gap-union",
            "destroySizes": list(destroy_sizes),
            "radius": radius,
            "maxGaps": max_gaps,
            "maxJobs": max_jobs,
            "gapRanks": sorted(selected_rank_set),
            "expansionJobs": sorted(expansion_job_set),
            "overrideBudgetRule": "one sparse priority override per selected gap",
            "repairOrder": [
                "alns-incumbent-replay",
                "alns-adjacent-o6",
                "alns-adjacent-o5",
                "alns-adjacent-o4",
                "alns-adjacent-o3",
                "alns-boundary-pull",
                "alns-sink-order",
                "alns-wait-first",
                "cp-sat-fixed-mode",
            ],
            "acceptance": "strict-validated-improvement",
        },
        "neighborhoods": plans,
    }


def alns_priorities(
    raw_schedule: list[dict[str, Any]],
    neighborhood: dict[str, Any],
    operator: str,
) -> dict[int, tuple[int, ...]]:
    """Return stable per-operation repair priorities for an ALNS neighborhood."""

    jobs = tuple(map(int, neighborhood["neighborhoodJobs"]))
    job_set = set(jobs)
    by_key = {(int(item["job"]), int(item["op"])): item for item in raw_schedule}
    incumbent_by_operation = {
        operation: tuple(
            int(item["job"])
            for item in sorted(
                (
                    item
                    for item in raw_schedule
                    if int(item["op"]) == operation and int(item["job"]) in job_set
                ),
                key=lambda item: (float(item["start"]), float(item["end"]), int(item["job"])),
            )
        )
        for operation in range(8)
    }
    sink_order = incumbent_by_operation[FOCUS_OPERATION]
    if operator == "alns-boundary-pull":
        boundary_jobs = tuple(
            dict.fromkeys(int(gap["rightJob"]) for gap in neighborhood.get("selectedGaps", []))
        )
        priorities = {}
        for operation in range(6):
            base = incumbent_by_operation[operation]
            priorities[operation] = boundary_jobs + tuple(job for job in base if job not in boundary_jobs)
        priorities[6] = sink_order
        priorities[7] = sink_order
        return priorities
    if operator == "alns-sink-order":
        return {operation: sink_order for operation in range(8)}
    if operator == "alns-wait-first":
        wait_order = tuple(
            sorted(
                jobs,
                key=lambda job: (
                    -(
                        float(by_key[(job, 6)]["start"])
                        - float(by_key[(job, 5)]["end"])
                    ),
                    sink_order.index(job),
                    job,
                ),
            )
        )
        priorities = {operation: wait_order for operation in range(6)}
        priorities[6] = sink_order
        priorities[7] = sink_order
        return priorities
    raise ValueError(f"unsupported ALNS repair operator: {operator}")


def alns_dispatch_order(
    raw_schedule: list[dict[str, Any]],
    neighborhood: dict[str, Any],
    operator: str,
    decision_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Anchor replay to incumbent dispatch order and apply bounded local swaps."""

    baseline_items = decision_trace or sorted(
        raw_schedule,
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
            int(item["job"]),
            int(item["op"]),
        ),
    )
    actions = [int(item["job"]) * 8 + int(item["op"]) for item in baseline_items]
    if operator == "alns-incumbent-replay":
        return {"actions": actions, "swaps": [], "overrideCount": 0}

    if operator.startswith("alns-adjacent-o"):
        one_based_operation = int(operator.rsplit("o", 1)[1])
        operation = one_based_operation - 1
        if operation not in (2, 3, 4, 5):
            raise ValueError(f"unsupported adjacent repair stage: O{one_based_operation}")
        neighborhood_jobs = set(map(int, neighborhood["neighborhoodJobs"]))
        budget = int(neighborhood.get("overrideBudget", 0))
        swaps: list[dict[str, int]] = []
        right_jobs = list(
            dict.fromkeys(int(gap["rightJob"]) for gap in neighborhood.get("selectedGaps", []))
        )
        for right_job in right_jobs:
            if len(swaps) >= budget:
                break
            right_action = right_job * 8 + operation
            right_position = actions.index(right_action)
            previous_positions = [
                index
                for index, action in enumerate(actions[:right_position])
                if action % 8 == operation and action // 8 in neighborhood_jobs
            ]
            if not previous_positions:
                continue
            left_position = previous_positions[-1]
            left_action = actions[left_position]
            actions[left_position], actions[right_position] = actions[right_position], actions[left_position]
            swaps.append(
                {
                    "operation": operation,
                    "leftJob": left_action // 8,
                    "rightJob": right_job,
                    "leftPosition": left_position,
                    "rightPosition": right_position,
                }
            )
        return {"actions": actions, "swaps": swaps, "overrideCount": len(swaps)}

    priorities = alns_priorities(raw_schedule, neighborhood, operator)
    neighborhood_jobs = set(map(int, neighborhood["neighborhoodJobs"]))
    budget = int(neighborhood.get("overrideBudget", 0))
    swaps: list[dict[str, int]] = []
    # Repair the suffix nearest to O7 first. Only move farther upstream when
    # the configured budget is not consumed by O6/O5/O4 evidence.
    for operation in (5, 4, 3, 2, 1, 0):
        if len(swaps) >= budget:
            break
        positions = [
            index
            for index, action in enumerate(actions)
            if action % 8 == operation and action // 8 in neighborhood_jobs
        ]
        desired = [job * 8 + operation for job in priorities[operation]]
        for local_index, position in enumerate(positions):
            if len(swaps) >= budget:
                break
            desired_action = desired[local_index]
            if actions[position] == desired_action:
                continue
            other_position = actions.index(desired_action)
            displaced_action = actions[position]
            actions[position], actions[other_position] = actions[other_position], actions[position]
            swaps.append(
                {
                    "operation": operation,
                    "leftJob": displaced_action // 8,
                    "rightJob": desired_action // 8,
                    "leftPosition": position,
                    "rightPosition": other_position,
                }
            )
    return {"actions": actions, "swaps": swaps, "overrideCount": len(swaps)}
