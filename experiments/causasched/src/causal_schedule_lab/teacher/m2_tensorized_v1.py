"""Phase-2.15 tensor contracts for appearance-conditioned atomic signed CE.

The authoritative label remains the frozen Phase-2.11/2.12 ``ce_atomic``.
This module only aligns that label with reconstructed SG-SCT graph/Gantt
tensors, one appearance block, and one canonical executable atom.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .eval.optimization_utility import AtomOperatorMapping


SCHEMA_VERSION = "tensorized_causal_m2_input_v1"
CANDIDATE_FEATURE_FIELDS = (
    "max_plus_relevance",
    "legal_mask",
    "source_machine_scaled",
    "target_machine_scaled",
    "target_mode_scaled",
    "insertion_position_scaled",
    "predecessor_operation_scaled",
    "successor_operation_scaled",
)
FROZEN_APPEARANCE_FEATURE_FIELDS = (
    "causal_magnitude_before",
    "detection_before",
)


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_or_missing(values: Sequence[str], value: Any) -> int:
    if value is None:
        return -1
    try:
        return values.index(str(value))
    except ValueError:
        return -1


def _scaled(index: int, count: int) -> float:
    return -1.0 if index < 0 else float(index) / max(1, count - 1)


@dataclass(frozen=True)
class CandidateTensorRecord:
    atom_type: int
    primary_node_index: int
    secondary_node_index: int
    numeric: tuple[float, ...]
    operator_id: str
    canonical_atom_id: str
    parameter_hash: str


@dataclass(frozen=True)
class CandidateContextIndices:
    """Graph/Gantt gather indices reconstructed from canonical frozen identity."""

    source_resource_node_index: int
    target_resource_node_index: int
    target_mode_node_index: int
    target_machine_index: int


def candidate_context_indices(
    row: Mapping[str, Any], graph_manifest: Mapping[str, Any]
) -> CandidateContextIndices:
    """Resolve routing context without deriving identity from ordinal features."""

    node_ids = [str(item) for item in graph_manifest["id_spaces"]["node_ids"]]
    resource_ids = [str(item) for item in graph_manifest["id_spaces"]["resource_ids"]]
    mode_ids = [str(item) for item in graph_manifest["id_spaces"]["mode_ids"]]
    source = row.get("source_machine")
    targets = tuple(str(item) for item in (row.get("target_resource_ids") or ()))
    if row.get("atom_type") != "routing":
        source_node = node_ids.index(f"resource:{source}") if source is not None else -1
        return CandidateContextIndices(source_node, -1, -1, -1)
    if source is None or len(targets) != 1:
        raise ValueError(f"routing row has incomplete source/target identity: {row.get('sample_id')}")
    operation = str(row.get("operation_id") or row.get("primary_operation") or "")
    forced_modes = row.get("forced_mode_ids") or {}
    parameters = row.get("operator_parameters") or {}
    mode_id = forced_modes.get(operation) or parameters.get("mode_id")
    if not operation or not mode_id:
        raise ValueError(f"routing row has no canonical target mode: {row.get('sample_id')}")
    target = targets[0]
    if str(source) not in resource_ids or target not in resource_ids or str(mode_id) not in mode_ids:
        raise ValueError(f"routing context identity is absent from SG-SCT id space: {row.get('sample_id')}")
    try:
        source_node = node_ids.index(f"resource:{source}")
        target_node = node_ids.index(f"resource:{target}")
        mode_node = node_ids.index(f"mode:{mode_id}")
    except ValueError as exc:
        raise ValueError(f"routing context node is absent from SG-SCT graph: {row.get('sample_id')}") from exc
    return CandidateContextIndices(
        source_resource_node_index=source_node,
        target_resource_node_index=target_node,
        target_mode_node_index=mode_node,
        target_machine_index=resource_ids.index(target),
    )


def candidate_tensor_record(
    *,
    atom: Any,
    mapping: AtomOperatorMapping,
    max_plus_relevance: float,
    operation_ids: Sequence[str],
    resource_ids: Sequence[str],
    mode_ids: Sequence[str],
) -> CandidateTensorRecord:
    """Turn one complete canonical mapping into numeric indices, fail closed."""

    if mapping.mapping_status != "complete" or not mapping.canonical_atom_id or not mapping.operator_id:
        raise ValueError("candidate mapping is not complete")
    operation_ids = list(operation_ids)
    resource_ids = list(resource_ids)
    mode_ids = list(mode_ids)
    primary = _index_or_missing(operation_ids, atom.operation)
    secondary = _index_or_missing(operation_ids, atom.partner_operation)
    if primary < 0:
        raise ValueError("primary operation is absent from graph tensor id space")
    if atom.atom_type == "sequencing" and secondary < 0:
        raise ValueError("sequencing partner is absent from graph tensor id space")
    params: Mapping[str, Any] = mapping.operator_parameters
    source = _index_or_missing(resource_ids, mapping.source_machine)
    targets = tuple(mapping.target_resource_ids)
    target = _index_or_missing(resource_ids, targets[0] if targets else None)
    target_mode = _index_or_missing(mode_ids, params.get("mode_id"))
    position = int(params.get("position", -1))
    predecessor = _index_or_missing(operation_ids, params.get("predecessor_id"))
    successor = _index_or_missing(operation_ids, params.get("successor_id"))
    numeric = (
        float(max_plus_relevance),
        1.0,
        _scaled(source, len(resource_ids)),
        _scaled(target, len(resource_ids)),
        _scaled(target_mode, len(mode_ids)),
        _scaled(position, len(operation_ids)),
        _scaled(predecessor, len(operation_ids)),
        _scaled(successor, len(operation_ids)),
    )
    if not np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))):
        raise ValueError("candidate numeric tensor contains NaN/Inf")
    return CandidateTensorRecord(
        atom_type=1 if atom.atom_type == "routing" else 2,
        primary_node_index=primary,
        secondary_node_index=secondary,
        numeric=numeric,
        operator_id=mapping.operator_id,
        canonical_atom_id=mapping.canonical_atom_id,
        parameter_hash=canonical_json_hash(dict(params)),
    )


def validate_candidate_arrays(arrays: Mapping[str, np.ndarray], *, block_count: int, node_count: int) -> None:
    required = {
        "sample_block_index", "sample_atom_type", "sample_primary_node_index",
        "sample_secondary_node_index", "sample_candidate_numeric", "sample_signed_ce",
        "sample_max_plus_relevance", "sample_legal_mask", "appearance_frozen_features",
        "appearance_frozen_observed_mask",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"candidate arrays miss {sorted(missing)}")
    count = arrays["sample_signed_ce"].shape[0]
    for name, value in arrays.items():
        if value.dtype.kind in {"O", "U", "S", "V"}:
            raise ValueError(f"unsafe dtype for {name}: {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite values in {name}")
    for name in (
        "sample_block_index", "sample_atom_type", "sample_primary_node_index",
        "sample_secondary_node_index", "sample_max_plus_relevance", "sample_legal_mask",
    ):
        if arrays[name].shape != (count,):
            raise ValueError(f"{name} must have shape [sample_count]")
    if arrays["sample_candidate_numeric"].shape != (count, len(CANDIDATE_FEATURE_FIELDS)):
        raise ValueError("sample_candidate_numeric has wrong shape")
    if arrays["appearance_frozen_features"].shape != (
        block_count, len(FROZEN_APPEARANCE_FEATURE_FIELDS)
    ):
        raise ValueError("appearance_frozen_features has wrong shape")
    if arrays["appearance_frozen_observed_mask"].shape != (block_count,):
        raise ValueError("appearance_frozen_observed_mask has wrong shape")
    if count:
        if int(arrays["sample_block_index"].min()) < 0 or int(arrays["sample_block_index"].max()) >= block_count:
            raise ValueError("sample block index is out of range")
        if int(arrays["sample_primary_node_index"].min()) < 0 or int(arrays["sample_primary_node_index"].max()) >= node_count:
            raise ValueError("primary node index is out of range")
        valid_secondary = arrays["sample_secondary_node_index"] >= 0
        if valid_secondary.any() and int(arrays["sample_secondary_node_index"][valid_secondary].max()) >= node_count:
            raise ValueError("secondary node index is out of range")
        if not np.all(arrays["sample_legal_mask"] == 1):
            raise ValueError("accepted tensorized samples must all pass the legal mask")
        if not np.all(arrays["appearance_frozen_observed_mask"][arrays["sample_block_index"]] == 1):
            raise ValueError("a sample references an unobserved frozen appearance feature row")


__all__ = [
    "CANDIDATE_FEATURE_FIELDS", "FROZEN_APPEARANCE_FEATURE_FIELDS", "SCHEMA_VERSION",
    "CandidateContextIndices", "CandidateTensorRecord", "candidate_context_indices",
    "candidate_tensor_record", "canonical_json_hash",
    "validate_candidate_arrays",
]
