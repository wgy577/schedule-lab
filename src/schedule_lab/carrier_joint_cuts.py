from __future__ import annotations

import hashlib
import json
from typing import Any


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_spacetime_replay_cut(
    spacetime_result: dict[str, Any],
    replay_result: dict[str, Any],
    *,
    route_library_version: str,
    oracle_version: str,
) -> dict[str, Any]:
    """Convert one failed exact-grid candidate replay into a context-exact LBBD cut."""

    candidate = spacetime_result.get("candidate")
    if not candidate:
        raise ValueError("spacetime result contains no candidate")
    best = replay_result["best"]
    unavailable = best.get("unavailableTargetMachines", [])
    mode_changes = best.get("modeChanges", [])
    affected_jobs = sorted(
        {
            int(item["job"])
            for item in [*unavailable, *mode_changes]
            if "job" in item
        }
    )
    context = {
        "candidateHash": candidate["hash"],
        "targetOrderHash": replay_result.get("targetHash"),
        "routeLibraryVersion": route_library_version,
        "oracleVersion": oracle_version,
        "timeResolution": spacetime_result["settings"]["timeResolution"],
    }
    feasible = (
        bool(best.get("domainConstructed"))
        and not best.get("domainErrors")
        and not unavailable
        and not mode_changes
    )
    reason = (
        "domain replay accepted target bindings"
        if feasible
        else "target dispatch order makes one or more incumbent machine/route bindings unreachable"
    )
    return {
        "schemaVersion": 1,
        "cutId": f"joint-cut-{_digest({**context, 'reason': reason})[:20]}",
        "kind": "target-order-binding-reachability-no-good",
        "context": context,
        "validOnlyForExactContext": True,
        "reason": reason,
        "affectedJobs": affected_jobs,
        "unavailableTargetMachines": unavailable,
        "modeChanges": mode_changes,
        "replayMakespan": best.get("trueMakespan"),
        "deterministicReplay": bool(replay_result.get("deterministic")),
        "feasible": feasible,
        "masterAction": (
            "retain candidate for objective comparison"
            if feasible
            else "forbid this exact target order/binding tuple and release affected jobs in the next causal closure"
        ),
    }


def build_reachability_causal_closure(
    raw_schedule: list[dict[str, Any]],
    cut: dict[str, Any],
    *,
    neighbor_radius: int = 1,
    max_released_jobs: int = 14,
) -> dict[str, Any]:
    """Expand an exact replay failure only across directly coupled resource orders."""

    affected = set(map(int, cut.get("affectedJobs", [])))
    if not affected:
        raise ValueError("cut has no affected jobs")
    by_key = {(int(row["job"]), int(row["op"])): row for row in raw_schedule}
    jobs = sorted({key[0] for key in by_key})
    closure = set(affected)
    evidence_edges: list[dict[str, Any]] = []
    surfaces = (
        (0, "tractor-order"),
        (2, "preparation-route-order"),
        (4, "catapult-lock-order"),
        (6, "global-launch-order"),
    )
    ranked_neighbors: list[tuple[int, int, str, int]] = []
    for operation, surface in surfaces:
        machine_groups: dict[int, list[int]] = {}
        for job in jobs:
            row = by_key[(job, operation)]
            machine_groups.setdefault(int(row["machine"]), []).append(job)
        for machine, resource_jobs in sorted(machine_groups.items()):
            ordered = sorted(
                resource_jobs,
                key=lambda job: (float(by_key[(job, operation)]["start"]), job),
            )
            positions = {job: index for index, job in enumerate(ordered)}
            for source_job in sorted(affected.intersection(ordered)):
                source_index = positions[source_job]
                for distance in range(1, neighbor_radius + 1):
                    for neighbor_index in (source_index - distance, source_index + distance):
                        if 0 <= neighbor_index < len(ordered):
                            neighbor = ordered[neighbor_index]
                            ranked_neighbors.append((distance, neighbor, surface, source_job))
                            evidence_edges.append(
                                {
                                    "fromJob": source_job,
                                    "toJob": neighbor,
                                    "surface": surface,
                                    "machine": machine,
                                    "distance": distance,
                                }
                            )
    for _, neighbor, _, _ in sorted(ranked_neighbors):
        if len(closure) >= max_released_jobs:
            break
        closure.add(neighbor)
    released_jobs = sorted(closure)
    frozen_jobs = sorted(set(jobs) - closure)
    return {
        "controller": "cut-driven-joint-causal-closure-v1",
        "sourceCutId": cut["cutId"],
        "affectedJobs": sorted(affected),
        "neighborRadius": neighbor_radius,
        "maxReleasedJobs": max_released_jobs,
        "releasedJobs": released_jobs,
        "releasedOperations": [
            f"J{job + 1}.O{operation + 1}"
            for job in released_jobs
            for operation in range(8)
        ],
        "frozenJobs": frozen_jobs,
        "frozenOperationCount": len(frozen_jobs) * 8,
        "evidenceEdges": evidence_edges,
        "nextMaster": {
            "variables": [
                "dispatch-order arcs inside the closure",
                "tractor assignment and depot/previous-spot route compatibility",
                "preparation/catapult binding and existing MAT route column",
                "start times on the 0.1-second finite grid",
            ],
            "fixed": [
                "all modes and order arcs outside the closure",
                "the incumbent prefix before each released boundary",
                "all original MAT geometry",
            ],
            "limits": {
                "machineBindingChanges": 2,
                "dispatchArcChanges": 4,
                "routeColumnChanges": 2,
                "workers": 1,
            },
            "requiredSubproblems": [
                "finite-grid pairwise trajectory conflict check",
                "legacy state-dependent machine/route reachability replay",
            ],
        },
    }
