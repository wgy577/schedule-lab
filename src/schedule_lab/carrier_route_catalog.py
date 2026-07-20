from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from scipy.io import loadmat


_NAME_PATTERNS = (
    re.compile(r"^M(?P<machine>\d+)J(?P<job>\d+)\.mat$", re.IGNORECASE),
    re.compile(r"^J(?P<job>\d+)M(?P<machine>\d+)\.mat$", re.IGNORECASE),
)


def _identity(path: Path) -> tuple[int, int]:
    for pattern in _NAME_PATTERNS:
        match = pattern.match(path.name)
        if match:
            return int(match.group("job")), int(match.group("machine"))
    raise ValueError(f"unsupported route filename: {path.name}")


def _flat(values: Any) -> list[float]:
    return [float(value) for value in values.reshape(-1)]


def _sample_actor(
    data: dict[str, Any],
    *,
    name: str,
    x_key: str,
    y_key: str,
    theta_key: str,
    time_key: str = "timeList",
    max_points: int = 32,
) -> dict[str, Any]:
    x_values = _flat(data[x_key])
    y_values = _flat(data[y_key])
    theta_values = _flat(data[theta_key])
    time_values = _flat(data[time_key])
    point_count = min(len(x_values), len(y_values), len(theta_values), len(time_values))
    if point_count < 2:
        raise ValueError(f"route actor {name} has fewer than two samples")
    stride = max(1, math.ceil((point_count - 1) / max(1, max_points - 1)))
    indices = list(range(0, point_count, stride))
    if indices[-1] != point_count - 1:
        indices.append(point_count - 1)
    length = sum(
        math.hypot(x_values[index] - x_values[index - 1], y_values[index] - y_values[index - 1])
        for index in range(1, point_count)
    )
    return {
        "actor": name,
        "originalPointCount": point_count,
        "polylineLength": length,
        "start": {"x": x_values[0], "y": y_values[0], "theta": theta_values[0]},
        "end": {
            "x": x_values[point_count - 1],
            "y": y_values[point_count - 1],
            "theta": theta_values[point_count - 1],
        },
        "points": [
            {
                "t": time_values[index],
                "x": x_values[index],
                "y": y_values[index],
                "theta": theta_values[index],
            }
            for index in indices
        ],
    }


def read_route_column(path: str | Path, *, phase: str, max_points: int = 32) -> dict[str, Any]:
    """Normalize one legacy MAT route into an auditable route column."""

    source = Path(path).expanduser().resolve()
    job, machine = _identity(source)
    if phase == "towing":
        variables = ["xT", "yT", "thetaT", "xU", "yU", "thetaU", "timeList", "tf"]
        actor_specs = [
            ("tractor", "xT", "yT", "thetaT"),
            ("aircraft", "xU", "yU", "thetaU"),
        ]
    else:
        variables = ["x", "y", "theta", "timeList", "tf"]
        actor_specs = [("vehicle", "x", "y", "theta")]
    data = loadmat(source, variable_names=variables)
    actors = [
        _sample_actor(
            data,
            name=name,
            x_key=x_key,
            y_key=y_key,
            theta_key=theta_key,
            max_points=max_points,
        )
        for name, x_key, y_key, theta_key in actor_specs
    ]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    duration = float(data["tf"].reshape(-1)[0])
    return {
        "routeColumnId": f"{phase}:J{job}:M{machine}:{digest[:12]}",
        "phase": phase,
        "job": job,
        "machine": machine,
        "duration": duration,
        "source": str(source),
        "sourceSha256": digest,
        "actors": actors,
    }


def build_carrier_route_catalog(
    root: str | Path,
    *,
    max_points: int = 32,
) -> dict[str, Any]:
    """Convert the fixed legacy path library into versioned route-column metadata."""

    legacy_root = Path(root).expanduser().resolve()
    phase_directories = {
        "initial": legacy_root / "initialtraject",
        "towing": legacy_root / "systemtraject",
        "transfer": legacy_root / "trajectory",
    }
    columns = [
        read_route_column(path, phase=phase, max_points=max_points)
        for phase, directory in phase_directories.items()
        for path in sorted(directory.glob("*.mat"))
    ]
    binding_groups: dict[tuple[str, int, int], int] = {}
    routing_groups: dict[tuple[str, int], int] = {}
    for column in columns:
        binding_key = (column["phase"], column["job"], column["machine"])
        routing_key = (column["phase"], column["job"])
        binding_groups[binding_key] = binding_groups.get(binding_key, 0) + 1
        routing_groups[routing_key] = routing_groups.get(routing_key, 0) + 1
    binding_alternatives = [count for count in binding_groups.values() if count > 1]
    routing_alternatives = [count for count in routing_groups.values() if count > 1]
    return {
        "schemaVersion": 1,
        "root": str(legacy_root),
        "columnCount": len(columns),
        "phaseCounts": {
            phase: sum(column["phase"] == phase for column in columns)
            for phase in phase_directories
        },
        "bindingGroups": len(binding_groups),
        "bindingGroupsWithAlternatives": len(binding_alternatives),
        "routingChoiceGroups": len(routing_groups),
        "routingGroupsWithAlternatives": len(routing_alternatives),
        "minimumRoutesPerRoutingGroup": min(routing_groups.values(), default=0),
        "maximumRoutesPerRoutingGroup": max(routing_groups.values(), default=0),
        "jointOptimizationReadiness": (
            "existing alternatives are tied to machine/spot bindings; expose those compatibility "
            "choices in the master before generating additional same-binding spatial alternatives"
        ),
        "columns": columns,
    }
