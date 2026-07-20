from __future__ import annotations

import json
from pathlib import Path

from .model import Assignment, ChoiceLink, Mode, Operation, Problem, Resource, Schedule


DEFAULT_CARRIER_SCHEDULE = Path(
    "/Users/guangyuwu/Desktop/Sortie/carrier_3d_demo/lib/deck-update-policy-schedule.json"
)


def _scaled(value: float, scale: int) -> int:
    return int(round(value * scale))


def build_carrier_schedule(
    raw_schedule: list[dict],
    *,
    time_scale: int = 10,
    source: str = "in-memory carrier schedule",
    source_metadata: dict | None = None,
    schedule_metadata: dict | None = None,
    domain_validated: bool = False,
) -> tuple[Problem, Schedule]:
    """Builds the common IR from a carrier schedule produced by the domain env."""

    source_metadata = source_metadata or {}
    schedule_metadata = schedule_metadata or {}
    by_job: dict[int, list[dict]] = {}
    for item in raw_schedule:
        by_job.setdefault(int(item["job"]), []).append(item)
    resources = tuple(
        [Resource(id=f"T{index + 1}", name=f"Tractor {index + 1}", tags=("tractor",)) for index in range(5)]
        + [Resource(id=f"P{index + 1}", name=f"Preparation Spot {index + 1}", capacity=2, tags=("preparation",)) for index in range(3)]
        + [Resource(id=f"C{index + 1}", name=f"Catapult {index + 1}", tags=("catapult",)) for index in range(3)]
        + [Resource(id="L1", name="Global Launch Channel", tags=("global_launch",))]
    )
    machine_resource = {
        **{index: f"T{index + 1}" for index in range(5)},
        **{index + 5: f"P{index + 1}" for index in range(3)},
        **{index + 8: f"C{index + 1}" for index in range(3)},
        11: "L1",
    }
    operations: list[Operation] = []
    assignments: list[Assignment] = []
    choice_links: list[ChoiceLink] = []
    for job, items in sorted(by_job.items()):
        ordered = sorted(items, key=lambda item: int(item["op"]))
        tractor_modes: dict[str, str] = {}
        lane_modes: dict[str, str] = {}
        operation_ids: list[str] = []
        for item in ordered:
            operation_index = int(item["op"])
            operation_id = f"J{job + 1}.O{operation_index + 1}"
            operation_ids.append(operation_id)
            resource_id = machine_resource[int(item["machine"])]
            mode_id = f"{operation_id}@{resource_id}"
            duration = _scaled(float(item["dur"]), time_scale)
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=f"J{job + 1}",
                    index=operation_index,
                    predecessors=() if operation_index == 0 else (f"J{job + 1}.O{operation_index}",),
                    modes=(Mode(id=mode_id, duration=duration, resources=(resource_id,)),),
                )
            )
            assignments.append(
                Assignment(
                    operation_id=operation_id,
                    mode_id=mode_id,
                    start=_scaled(float(item["start"]), time_scale),
                    end=_scaled(float(item["end"]), time_scale),
                )
            )
            if operation_index in (0, 1):
                tractor_modes[mode_id] = resource_id
            if operation_index in (2, 3):
                lane_modes[mode_id] = resource_id.replace("P", "lane-")
            if operation_index in (4, 5, 7):
                lane_modes[mode_id] = resource_id.replace("C", "lane-")
        choice_links.append(
            ChoiceLink(
                id=f"J{job + 1}.tractor",
                operation_ids=tuple(operation_ids[index] for index in (0, 1)),
                mode_keys=tractor_modes,
            )
        )
        choice_links.append(
            ChoiceLink(
                id=f"J{job + 1}.deck-lane",
                operation_ids=tuple(operation_ids[index] for index in (2, 3, 4, 5, 7)),
                mode_keys=lane_modes,
            )
        )
    problem = Problem(
        id="carrier-20-deck-update",
        kind="CARRIER",
        resources=resources,
        operations=tuple(operations),
        choice_links=tuple(choice_links),
        time_scale=time_scale,
        metadata={
            "source": source,
            "network": source_metadata.get("network"),
            "checkpoint": source_metadata.get("checkpoint"),
            "reported_policy_makespan": source_metadata.get(
                "policyMakespan", source_metadata.get("reported_policy_makespan")
            ),
            "requires_domain_validation": True,
            "domain_validator": "legacy deck_update collision-aware replay",
            "optimization_scope": "domain-oracle assignment; further changes require another replay",
        },
    )
    schedule = Schedule(
        problem_id=problem.id,
        assignments=tuple(assignments),
        metadata={
            "solver": schedule_metadata.get("solver", "carrier domain oracle"),
            "domain_validated": domain_validated,
            "domain_validation_source": source,
            **schedule_metadata,
        },
    )
    return problem, schedule


def load_carrier_baseline(
    path: str | Path = DEFAULT_CARRIER_SCHEDULE,
    *,
    time_scale: int = 10,
) -> tuple[Problem, Schedule]:
    """Imports the validated deck_update baseline without inventing new routes."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return build_carrier_schedule(
        payload["schedule"],
        time_scale=time_scale,
        source=str(source),
        source_metadata=payload.get("meta", {}),
        schedule_metadata={"solver": "deck_update PPO greedy baseline"},
        domain_validated=True,
    )
