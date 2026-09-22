"""Standardized minimal serialization that feeds SG-SCT M2 reverse inference.

The six-family-1.0 template (`unified_representation.py`) is the lossless
contract for *every* learning-side label.  But the causal model's reverse
inference (RCBT / M2) does not need the fully-expanded graph: the converter
`compile_sg_sct_input_v1(problem, schedule, appearance_payload)` rebuilds every
edge, edge role (gs/gf/gc), topological order, binding/release, criticality and
ancestor subgraph *deterministically* from three inputs:

    problem        -> G_S ownership/eligibility context + G_F feasibility
    schedule       -> G_C (precedence + resource_sequence), time order, binding
    appearance      -> the symptom blocks (表象块) that anchor M2's reverse query

so the serialized document is minimal = ``problem + schedule + appearance``.
Nothing else needs to be carried: the disjunctive pairwise expansion, per-op
candidate-machine enumeration and constraint_scope bulk are all re-derivable.

This is the **standard** serialization the pipeline emits for every instance
across all six families (JSP/FSP/FJSP/HFSP/DJSP/DFJSP).  Schema version
``reverse-inference-1.0``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from .ir import Problem, Schedule
from .symptom_pruning import SymptomPruningSnapshot
from .unified_representation import FamilyProfile, derive_family_profile


class ReverseInferenceSerialization(BaseModel):
    """Canonical minimal document fed to SG-SCT reverse inference.

    Carries exactly what M2 needs and nothing re-derivable:
    ``problem + schedule + appearance (表象块)``.  The converter rebuilds the
    causal graph (``G_C``), feasibility graph (``G_F``) and context graph
    (``G_S``) from these alone.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "reverse-inference-1.0"
    problem_id: str
    family: FamilyProfile
    problem: Problem
    schedule: Schedule
    appearance: SymptomPruningSnapshot
    provenance: dict[str, str]


def _canonical_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_reverse_inference_serialization(
    problem: Problem,
    schedule: Schedule,
    appearance: SymptomPruningSnapshot,
) -> ReverseInferenceSerialization:
    """Package the three M2 inputs into the standard serialized document.

    Parameters mirror the SG-SCT converter's input contract exactly
    (``compile_sg_sct_input_v1(problem, schedule, appearance_payload)``) so the
    same document can be serialized to disk and later re-hydrated into tensors.
    """

    if problem.id != schedule.problem_id:
        raise ValueError("problem and schedule IDs differ")
    if problem.id != appearance.problem_id:
        raise ValueError("problem and appearance block IDs differ")

    family = derive_family_profile(problem)
    provenance = {
        "problem_hash": _canonical_hash(problem),
        "schedule_hash": _canonical_hash(schedule),
        "appearance_hash": _canonical_hash(appearance),
        "schema_version": "reverse-inference-1.0",
        "appearance_schema_version": appearance.schema_version,
        "family": family.inferred_family,
    }

    return ReverseInferenceSerialization(
        schema_version="reverse-inference-1.0",
        problem_id=problem.id,
        family=family,
        problem=problem,
        schedule=schedule,
        appearance=appearance,
        provenance=provenance,
    )