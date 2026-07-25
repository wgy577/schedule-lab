from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from .models import ExperimentRecord


def record_to_training_row(record: ExperimentRecord) -> dict[str, Any]:
    full = next(
        (
            verification
            for verification in reversed(record.verifications)
            if verification.fidelity.value == "full"
        ),
        None,
    )
    return {
        "project_id": record.project_id,
        "instance_id": record.instance_id,
        "incumbent_hash": record.incumbent_hash,
        "incumbent_objective": record.incumbent_objective,
        "diagnostic_type": record.cip.diagnostic.type,
        "diagnostic_location": list(record.cip.diagnostic.location),
        "diagnostic_magnitude": record.cip.diagnostic.magnitude,
        "responsible_point": record.cip.responsible.operation_id,
        "point_type": record.cip.responsible.decision_type.value,
        "causal_path": list(record.cip.causal_path.nodes),
        "predicted_closure": list(record.cip.closure.operation_ids),
        "predicted_improvement": record.cip.predicted_improvement,
        "predicted_validity": record.cip.predicted_validity,
        "predicted_cost": record.cip.predicted_cost,
        "operator": record.action.operator,
        "closure_level": record.action.closure_level,
        "control_action": record.action.control.value,
        "oracle_valid": bool(full and full.passed),
        "failure_labels": (
            [] if full is None else [item.value for item in full.failures]
        ),
        "actual_closure": list(record.actual_closure),
        "outside_changes": list(record.outside_changes),
        "runtime": record.runtime_seconds,
        "accepted": record.accepted,
        "new_objective": record.new_objective,
        "delta_objective": record.delta_objective,
        "best_updated": record.best_updated,
    }


def pairwise_ranking_labels(
    records: Iterable[ExperimentRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.instance_id, record.incumbent_hash),
            [],
        ).append(record)
    labels = []
    for (instance_id, incumbent_hash), group in grouped.items():
        for left, right in combinations(group, 2):
            left_value = left.delta_objective / max(left.runtime_seconds, 1e-9)
            right_value = right.delta_objective / max(right.runtime_seconds, 1e-9)
            if abs(left_value - right_value) <= 1e-12:
                continue
            winner, loser = (
                (left, right) if left_value > right_value else (right, left)
            )
            labels.append(
                {
                    "instance_id": instance_id,
                    "incumbent_hash": incumbent_hash,
                    "preferred_cip": winner.cip.id,
                    "other_cip": loser.cip.id,
                    "preferred_value_per_cost": max(left_value, right_value),
                    "other_value_per_cost": min(left_value, right_value),
                }
            )
    return labels


def closure_membership_labels(record: ExperimentRecord) -> list[dict[str, Any]]:
    predicted = set(record.cip.closure.operation_ids)
    actual = set(record.actual_closure)
    universe = predicted | actual | set(record.outside_changes)
    return [
        {
            "operation_id": operation_id,
            "predicted": operation_id in predicted,
            "actual": operation_id in actual,
            "outside_change": operation_id in record.outside_changes,
        }
        for operation_id in sorted(universe)
    ]
