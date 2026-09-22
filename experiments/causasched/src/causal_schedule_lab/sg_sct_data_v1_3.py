"""SG-SCT v1.3.0 input compiler -- post-2.29R architecture refactor.

This module establishes the **new** graph schema version ``1.3.0`` alongside the
frozen ``1.2.1`` builder in :mod:`causal_schedule_lab.sg_sct_data_v1`.  It does
NOT modify the 1.2.1 path; it wraps it and applies three deterministic,
information-preserving transforms that the Phase-2.29R Failure Decomposition /
Graph Audit identified as structural defects:

1. **``reverse_causal_mask`` (TRUE_LOCAL_G_C only).**  A new boolean edge tensor
   that is ``True`` *only* for ``gc:precedence`` and ``gc:resource_sequence``
   edges.  These are the only edges that represent direct execution propagation
   (a job-order or machine-order predecessor -> operation).  ``gc:machine`` (hub),
   ``gc:rule:A*`` (rule typing) and ``l4:*`` (appearance membership / swap) are
   NOT local causal execution edges and are excluded from reverse message
   passing.  In 1.2.1 the reverse mask was ``role==2 | role==5`` which admitted
   the machine hub, the all-pairs rule typing and the all-pairs block clique,
   flooding reverse inference and oversmoothing block embeddings.

2. **Remove the ``l4:block`` all-pairs clique.**  In 1.2.1 every retained
   symptom block emitted K(K-1) pairwise member edges (37.45% of all edges on
   Mk9).  The clique is redundant with the block-membership index
   (``model_symptom_block_node_index``) plus the model's per-block attention
   pooling, which already yields an O(K) block representation.  Removing it
   eliminates the dominant oversmoothing source.  ``l4:swap`` (cross-block,
   swap-feasible, already sparsified) is retained as a forward-only proposal
   relation.

3. **Version stamp.**  ``schema_version`` -> ``1.3.0``; a ``v1_3_transform``
   block records what was removed so the census is auditable.

The frozen 1.2.1 dataset, evaluator and historical artifacts are untouched.
This module only adds a new versioned builder; loaders dispatch on version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .ir import Problem, Schedule
from .sg_sct_data_v1 import (
    GC_EDGE_TYPES,
    GF_EDGE_TYPES,
    GS_EDGE_TYPES,
    L4_EDGE_TYPES,
    MACHINE_HUB_EDGE_TYPE,
    MODEL_EDGE_TYPES,
    SCHEMA_NAME,
    SGSCTDataV1,
    compile_sg_sct_input_v1,
    to_sg_sct_batch_v1,
)


# New schema version.  Old 1.2.1 constant in sg_sct_data_v1.py is untouched.
SCHEMA_VERSION_1_3 = "1.3.0"

# Edge-type ids inside the unified MODEL_EDGE_TYPES vocabulary.  Computing them
# from the vocabulary (rather than hard-coding) keeps this module robust to
# vocabulary extensions that append at the tail (the standard contract).
_BLOCK_TYPE_INDEX = MODEL_EDGE_TYPES.index("l4:block")
_SWAP_TYPE_INDEX = MODEL_EDGE_TYPES.index("l4:swap")
_PRECEDENCE_TYPE_INDEX = MODEL_EDGE_TYPES.index("gc:precedence")
_RESOURCE_SEQUENCE_TYPE_INDEX = MODEL_EDGE_TYPES.index("gc:resource_sequence")

# The TRUE local causal edge types for G_C in v1.3.0.
TRUE_LOCAL_CAUSAL_TYPES = (
    "gc:precedence",
    "gc:resource_sequence",
)

# Edge-type ids that are *not* TRUE_LOCAL_G_C but were stamped with
# ``edge_role == CAUSAL_HARD (2)`` by the frozen 1.2.1 builder.  In 1.2.1 the M1
# causal encoder's mask is ``role==2 | role==3``; on the unified tensor that
# admits gc:machine + every gc:rule:* into M1 *causal propagation*, which the
# post-2.29R contract forbids (only precedence + resource_sequence may enter the
# causal stack).  v1.3.0 demotes these to ``CONTEXT (0)`` so V3's causal_mask
# naturally reduces to TRUE_LOCAL_G_C while the edges stay available to the
# context layers as schedule/resource context (gc:machine) and rule evidence
# (gc:rule:*).  This is the standard-allowed home; V3 code is untouched.
_TRUE_LOCAL_CAUSAL_TYPE_INDICES = frozenset(
    MODEL_EDGE_TYPES.index(name) for name in TRUE_LOCAL_CAUSAL_TYPES
)
_CAUSAL_HARD_ROLE = 2
_CONTEXT_ROLE = 0


def _demote_nonlocal_causal_roles(
    edge_type: np.ndarray, edge_role: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Demote role==CAUSAL_HARD edges that are not TRUE_LOCAL_G_C to CONTEXT.

    Returns ``(new_edge_role, demoted_mask)``.  Only edges whose role is
    currently CAUSAL_HARD *and* whose type is outside TRUE_LOCAL_CAUSAL_TYPES
    are moved; precedence / resource_sequence keep role==2.  Roles other than
    CAUSAL_HARD (e.g. l4:swap role==5, feasibility role==1) are untouched.
    """
    new_role = np.array(edge_role, copy=True)
    is_true_local = np.isin(
        edge_type, np.fromiter(_TRUE_LOCAL_CAUSAL_TYPE_INDICES, dtype=edge_type.dtype)
    )
    demote = (edge_role == _CAUSAL_HARD_ROLE) & (~is_true_local)
    new_role[demote] = _CONTEXT_ROLE
    return new_role, demote


def _keep_mask_excluding_block(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Boolean keep-mask over model edges that drops ``l4:block`` only."""
    edge_type = arrays["model_edge_type"]
    return edge_type != _BLOCK_TYPE_INDEX


def _reverse_causal_mask(edge_type: np.ndarray) -> np.ndarray:
    """True only for gc:precedence / gc:resource_sequence (TRUE_LOCAL_G_C)."""
    return (edge_type == _PRECEDENCE_TYPE_INDEX) | (
        edge_type == _RESOURCE_SEQUENCE_TYPE_INDEX
    )


def compile_sg_sct_input_v1_3(
    problem: Problem,
    schedule: Schedule,
    appearance_payload: Mapping[str, Any],
    *,
    case_id: str,
    split_id: str | None = None,
    near_critical_threshold: float | None = None,
    near_critical_fraction: float = 0.05,
) -> SGSCTDataV1:
    """Compile a v1.3.0 SG-SCT bundle (TRUE_LOCAL_G_C reverse mask, no block clique).

    Wraps the frozen 1.2.1 builder and applies the two deterministic transforms.
    The returned bundle carries ``schema_version == "1.3.0"`` plus a
    ``model_reverse_causal_mask`` array; the 1.2.1 arrays are otherwise reused
    verbatim (only the ``l4:block`` clique edges are stripped from the unified
    model-edge tensors).
    """
    # 1. Build the frozen 1.2.1 bundle (untouched, authoritative base).
    base = compile_sg_sct_input_v1(
        problem,
        schedule,
        appearance_payload,
        case_id=case_id,
        split_id=split_id,
        near_critical_threshold=near_critical_threshold,
        near_critical_fraction=near_critical_fraction,
    )

    arrays = dict(base.arrays)  # shallow copy; we replace specific keys below
    manifest = dict(base.manifest)

    # 2. Strip l4:block clique edges from the unified model-edge tensors.
    keep = _keep_mask_excluding_block(arrays)
    block_removed = int((~keep).sum())
    for name in (
        "model_edge_index",
        "model_edge_type",
        "model_edge_role",
        "model_edge_features",
    ):
        arr = arrays[name]
        if name == "model_edge_index":
            arrays[name] = arr[:, keep]
        else:
            arrays[name] = arr[keep]

    # 3. Reverse-causal mask over the *kept* edges (precedence + resource_sequence).
    reverse_mask = _reverse_causal_mask(arrays["model_edge_type"]).astype(np.bool_)
    arrays["model_reverse_causal_mask"] = reverse_mask

    # 3b. Demote non-TRUE_LOCAL causal-hard edges (gc:machine, gc:rule:*) to
    #     CONTEXT so V3's M1 causal_mask (role==2|3) reduces to TRUE_LOCAL_G_C.
    #     The edges stay in the graph and feed the context layers unchanged.
    new_role, demoted = _demote_nonlocal_causal_roles(
        arrays["model_edge_type"], arrays["model_edge_role"]
    )
    arrays["model_edge_role"] = new_role
    demoted_causal_edges = int(demoted.sum())

    # 4. Also strip l4:block from the standalone l4 arrays (swap retained) so the
    #    census and any downstream l4 consumer see only swap edges.
    l4_type = arrays.get("l4_edge_type")
    l4_index = arrays.get("l4_edge_index")
    if l4_type is not None and l4_index is not None and l4_type.size:
        l4_keep = l4_type != L4_EDGE_TYPES.index("block")
        arrays["l4_edge_index"] = l4_index[:, l4_keep]
        arrays["l4_edge_type"] = l4_type[l4_keep]
        # l4_edge_continuous aligns row-wise with l4_edge_index.
        if "l4_edge_continuous" in arrays:
            arrays["l4_edge_continuous"] = arrays["l4_edge_continuous"][l4_keep]
    arrays["model_l4_edge_index"] = arrays["l4_edge_index"]

    # 5. Version stamp + auditable transform record.
    manifest = dict(manifest)
    manifest["schema_version"] = SCHEMA_VERSION_1_3
    counts = dict(manifest.get("counts", {}))
    counts["l4_block_edges_removed"] = block_removed
    counts["l4_swap_edges_kept"] = int(
        (arrays["model_edge_type"] == _SWAP_TYPE_INDEX).sum()
    )
    counts["reverse_causal_edges"] = int(reverse_mask.sum())
    counts["reverse_causal_edge_types"] = list(TRUE_LOCAL_CAUSAL_TYPES)
    counts["m1_causal_hard_demoted_to_context"] = demoted_causal_edges
    counts["m1_causal_hard_edges"] = int(
        (arrays["model_edge_role"] == _CAUSAL_HARD_ROLE).sum()
    )
    manifest["counts"] = counts
    manifest["v1_3_transform"] = {
        "base_schema_version": "1.2.1",
        "removed_l4_block_clique": True,
        "l4_block_edges_removed": block_removed,
        "reverse_causal_mask_true_local_only": True,
        "reverse_causal_edge_types": list(TRUE_LOCAL_CAUSAL_TYPES),
        "m1_causal_encoder_true_local_only": True,
        "m1_causal_hard_demoted_to_context": demoted_causal_edges,
        "rationale": (
            "Phase-2.29R Failure Decomposition: l4:block all-pairs clique "
            "(37% of edges) + gc:rule typing entered reverse_mask=role2|role5, "
            "flooding reverse MP and oversmoothing block embeddings -> bimodal "
            "readout -> AUC ceiling ~0.6. v1.3.0 restricts reverse to TRUE_LOCAL_G_C. "
            "Final contract audit: V3 M1 causal_mask (role==2|3) also admitted "
            "gc:machine + gc:rule:* (role==2 on the unified tensor); v1.3.0 demotes "
            "those to CONTEXT (0) so the M1 causal encoder obeys TRUE_LOCAL_G_C too, "
            "keeping them as context/rule-evidence inputs. V3/1.2.1 untouched."
        ),
    }

    return SGSCTDataV1(arrays=arrays, manifest=manifest)


def to_sg_sct_batch_v1_3(bundle: SGSCTDataV1, *, device: str | None = None):
    """Project a v1.3.0 bundle into :class:`sg_sct_model_v4.SGSCTBatchV4`.

    Reuses the 1.2.1 projection (which reads the filtered model-edge arrays)
    and attaches the ``reverse_causal_mask`` tensor that V4 consumes.
    """
    import torch

    from .sg_sct_model_v4 import SGSCTBatchV4

    base_batch = to_sg_sct_batch_v1(bundle, device=device)
    reverse_mask = torch.from_numpy(bundle.arrays["model_reverse_causal_mask"]).bool()
    if device is not None:
        reverse_mask = reverse_mask.to(device=device)
    # SGSCTBatchV4 subclasses SGSCTBatch; reconstruct with the same fields plus
    # the new reverse_causal_mask.
    from dataclasses import fields

    base_fields = {f.name: getattr(base_batch, f.name) for f in fields(base_batch)}
    batch = SGSCTBatchV4(reverse_causal_mask=reverse_mask, **base_fields)
    batch.validate()
    return batch


__all__ = [
    "SCHEMA_VERSION_1_3",
    "TRUE_LOCAL_CAUSAL_TYPES",
    "compile_sg_sct_input_v1_3",
    "to_sg_sct_batch_v1_3",
]
