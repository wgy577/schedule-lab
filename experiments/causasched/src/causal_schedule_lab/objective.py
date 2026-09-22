from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .project import load_object


@dataclass(frozen=True)
class ObjectiveValue:
    names: tuple[str, ...]
    values: tuple[float, ...]
    senses: tuple[str, ...]
    tolerances: tuple[float, ...]
    metrics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.names),
            "values": list(self.values),
            "senses": list(self.senses),
            "tolerances": list(self.tolerances),
        }


def _change_cost(metrics: Mapping[str, Any]) -> float:
    change = metrics.get("change_cost") or {}
    return float(change.get("mode_changes", 0)) * 1_000_000.0 + float(
        change.get("start_shift", 0)
    )


def evaluate_objective(
    problem: Any,
    schedule: Any,
    *,
    baseline: Any | None = None,
) -> ObjectiveValue:
    """Evaluate a project-declared lexicographic objective.

    A domain can provide ``metadata.objective_evaluator = module:function``.
    The function receives ``(problem, schedule, baseline)`` and returns a
    metric mapping.  Otherwise Schedule Lab's generic metrics are used.
    """

    evaluator = problem.metadata.get("objective_evaluator")
    if evaluator:
        metrics = dict(load_object(str(evaluator))(problem, schedule, baseline))
    else:
        from .metrics import schedule_metrics

        metrics = schedule_metrics(problem, schedule, baseline)
    metrics.setdefault("makespan", schedule.makespan)
    metrics.setdefault("total_tardiness", 0)
    metrics.setdefault("total_flow_time", 0)
    metrics.setdefault("change_cost_scalar", _change_cost(metrics))
    resource_metrics = metrics.get("resource_metrics") or {}
    metrics.setdefault(
        "bottleneck_idle",
        sum(float(item.get("idle", 0.0)) for item in resource_metrics.values()),
    )
    metrics.setdefault("robustness", float(problem.metadata.get("robustness", 0.0)))

    declared = tuple(getattr(problem, "objective", ()))
    if declared:
        aliases = {"change_cost": "change_cost_scalar"}
        order = tuple(aliases.get(item.name, item.name) for item in declared)
        sense_map = {
            aliases.get(item.name, item.name): item.sense for item in declared
        }
        tolerance_map = {
            aliases.get(item.name, item.name): item.tolerance for item in declared
        }
    else:
        order = tuple(
            problem.metadata.get(
                "objective_order",
                ("makespan", "total_tardiness", "total_flow_time", "change_cost_scalar"),
            )
        )
        sense_map = {
            **{name: "minimize" for name in order},
            **dict(problem.metadata.get("objective_senses", {})),
        }
        tolerance_map = dict(problem.metadata.get("objective_tolerances", {}))
    missing = [name for name in order if name not in metrics]
    if missing:
        raise ValueError(f"objective evaluator omitted components: {missing}")
    return ObjectiveValue(
        names=order,
        values=tuple(float(metrics[name]) for name in order),
        senses=tuple(str(sense_map[name]) for name in order),
        tolerances=tuple(float(tolerance_map.get(name, 0.0)) for name in order),
        metrics=metrics,
    )


def compare_objectives(
    candidate: ObjectiveValue,
    incumbent: ObjectiveValue,
) -> tuple[int, float, str | None]:
    """Return (-1, gain, component) when candidate is lexicographically better."""

    if candidate.names != incumbent.names or candidate.senses != incumbent.senses:
        raise ValueError("objective schemas differ")
    for name, trial, base, sense, tolerance in zip(
        candidate.names,
        candidate.values,
        incumbent.values,
        candidate.senses,
        candidate.tolerances,
    ):
        difference = trial - base
        if abs(difference) <= tolerance:
            continue
        if sense == "minimize":
            return (-1, base - trial, name) if trial < base else (1, base - trial, name)
        if sense == "maximize":
            return (-1, trial - base, name) if trial > base else (1, trial - base, name)
        raise ValueError(f"unsupported objective sense: {sense}")
    return 0, 0.0, None
