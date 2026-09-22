"""Training Readiness Audit + instance-level Dataset Builder V1.

This module never trains a model and never calls a solver.  It audits persisted
counterfactual evidence, computes retrieval diagnostics, freezes an
instance-disjoint 70/15/15 split, and emits content-addressed artifacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..core_validation import validate_schedule
from ..counterfactual.evaluator import plan_from_legal_edits, verify_executed_edits
from ..ir import Problem, Schedule
from ..legal_edit_enumerator import LegalEditEnumerator
from ..m2_v5_schema_v1 import LegalEdit
from ..memory import (
    ExperienceStore,
    InterventionExperience,
    experience_training_eligible,
)
from ..memory.similarity import composite_similarity
from ..validation import schedule_hash


SCHEMA = "sgsct_effect_trajectory_dataset_v1"
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
# Coverage must target the canonical ACTIVE appearance set, not A1..A10.  The
# deprecated ids (A5/A7/A8/A9/A10) are never emitted by the v3 detector, so a
# full-A1..A10 coverage requirement can never pass -- it was a latent bug that
# would fail every otherwise-valid dataset.  The single source of truth is
# ``appearance_taxonomy.ACTIVE_APPEARANCE_IDS``.
from ..appearance_taxonomy import ACTIVE_APPEARANCE_IDS as APPEARANCES
OPERATORS = ("routing", "sequencing", "insertion", "timing")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _root_family(experience: InterventionExperience) -> str:
    root = experience.proposal.root_decision_id
    if root:
        return root.split(":", 1)[0]
    actions = experience.proposal.intervention_actions
    return actions[0].split(":", 1)[0].lower() if actions else "unknown"


def _trajectory_class(experience: InterventionExperience) -> str:
    immediate = float(experience.outcome.delta_cmax) if experience.outcome else 0.0
    if immediate < -1e-9:
        return "success"
    if bool(experience.trajectory_future_success):
        return "partial_success"
    return "failure"


def _instance_id(experience: InterventionExperience) -> str:
    return str(experience.metadata.get("base_instance_id", "")).strip()


@dataclass(frozen=True)
class QualityResult:
    key: str
    valid: bool
    failures: tuple[str, ...]
    checks: Mapping[str, bool]


def audit_counterfactual(experience: InterventionExperience) -> QualityResult:
    """Independently verify one persisted ``S0,P,S1`` evidence packet."""
    failures: list[str] = []
    checks: dict[str, bool] = {}
    metadata = dict(experience.metadata)
    outcome = experience.outcome
    checks["resolved"] = outcome is not None and experience.after_state is not None
    checks["instance_identity"] = bool(_instance_id(experience))
    checks["real_validator"] = metadata.get("validator_kind") == "real_cp_sat_counterfactual"
    checks["proposal_legal_recorded"] = metadata.get("proposal_legal") is True
    checks["actions_executed_recorded"] = metadata.get("actions_executed") is True
    checks["delta_verified_recorded"] = metadata.get("delta_cmax_verified") is True
    checks["validator_passed_recorded"] = metadata.get("validator_passed") is True
    checks["counterfactual_mode_recorded"] = metadata.get("counterfactual_mode") in {
        "frozen_local", "free_global"
    }
    checks["before_feasible_recorded"] = metadata.get("before_feasible") is True
    checks["after_feasible_recorded"] = metadata.get("after_feasible") is True
    checks["no_formal_test_source"] = metadata.get("formal_test_source") is not True

    try:
        problem = Problem.model_validate(metadata["problem_snapshot"])
        before = Schedule.model_validate(metadata["before_schedule_snapshot"])
        after = Schedule.model_validate(metadata["after_schedule_snapshot"])
        edits = tuple(LegalEdit(**payload) for payload in metadata["requested_edits"])
        checks["problem_hash"] = (
            hashlib.sha256(problem.model_dump_json().encode("utf-8")).hexdigest()
            == metadata.get("problem_sha256")
        )
        checks["before_schedule_hash"] = schedule_hash(before) == metadata.get("before_schedule_sha256")
        checks["after_schedule_hash"] = schedule_hash(after) == metadata.get("after_schedule_sha256")
        checks["before_feasible"] = validate_schedule(problem, before).feasible
        checks["after_feasible"] = validate_schedule(problem, after).feasible
        try:
            plan_from_legal_edits(problem, before, edits)
            subjects = tuple(sorted({edit.operation_id for edit in edits}))
            requested_types = {edit.edit_type for edit in edits}
            enumerated = LegalEditEnumerator(problem, before).enumerate(
                subjects,
                request_route="ROUTE" in requested_types,
                request_seq_swap="SEQ_SWAP" in requested_types,
                request_seq_insert="SEQ_INSERT" in requested_types,
                request_timing_shift="TIMING_SHIFT" in requested_types,
            )
            legal_ids = {edit.edit_id for edit in enumerated}
            checks["proposal_legal_replayed"] = bool(edits) and all(
                edit.edit_id in legal_ids for edit in edits
            )
        except (ValueError, NotImplementedError):
            checks["proposal_legal_replayed"] = False
        replay_checks = verify_executed_edits(problem, after, edits)
        checks["actions_executed_replayed"] = bool(replay_checks) and all(replay_checks.values())
        recomputed_delta = float(after.makespan - before.makespan)
        checks["delta_cmax_recomputed"] = (
            outcome is not None and abs(recomputed_delta - float(outcome.delta_cmax)) <= 1e-9
        )
        checks["state_delta_consistent"] = (
            experience.after_state is not None
            and abs(
                float(experience.after_state.cmax - experience.state.cmax)
                - recomputed_delta
            ) <= 1e-9
        )
        if metadata.get("counterfactual_mode") == "frozen_local":
            closure = set(metadata.get("closure_operations", ()))
            before_map, after_map = before.assignment_map(), after.assignment_map()
            changed = set()
            for operation_id in set(before_map) | set(after_map):
                left, right = before_map.get(operation_id), after_map.get(operation_id)
                if left is None or right is None or (
                    left.mode_id != right.mode_id
                    or abs(float(left.start) - float(right.start)) > 1e-9
                    or abs(float(left.end) - float(right.end)) > 1e-9
                    or left.route_id != right.route_id
                ):
                    changed.add(operation_id)
            outside = changed - closure
            checks["closure_recorded"] = bool(closure)
            checks["outside_closure_changes_replayed"] = (
                sorted(outside) == sorted(metadata.get("outside_closure_changes", ()))
            )
            if metadata.get("training_eligible") is True:
                checks["attribution_clean"] = not outside
    except (KeyError, TypeError, ValueError) as error:
        checks["portable_evidence_parse"] = False
        failures.append(f"portable_evidence:{type(error).__name__}")
    else:
        checks["portable_evidence_parse"] = True
    failures.extend(name for name, passed in checks.items() if not passed)
    return QualityResult(
        key=experience.key, valid=not failures,
        failures=tuple(dict.fromkeys(failures)), checks=checks,
    )


def freeze_instance_split(
    instance_ids: Sequence[str], *, seed: int = 20260821
) -> dict[str, str]:
    """Deterministic largest-remainder 70/15/15 split by whole instance."""
    unique = sorted(set(instance_ids), key=lambda value: (
        hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(), value
    ))
    n = len(unique)
    raw = {split: SPLIT_FRACTIONS[split] * n for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    remainder = n - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - counts[split]), SPLITS.index(split)))
    for split in order[:remainder]:
        counts[split] += 1
    assignments: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        for instance_id in unique[offset:offset + counts[split]]:
            assignments[instance_id] = split
        offset += counts[split]
    return assignments


def retrieval_diagnostics(
    experiences: Sequence[InterventionExperience], *, k: int = 5
) -> dict[str, object]:
    rows: list[dict[str, float | int | str]] = []
    for query in experiences:
        scored = sorted(
            (
                (composite_similarity(query.state, query.proposal, candidate.state, candidate.proposal), candidate)
                for candidate in experiences if candidate.key != query.key
            ),
            key=lambda item: (-item[0], item[1].key),
        )[:k]
        denominator = len(scored)
        rows.append({
            "key": query.key,
            "neighbors": denominator,
            "same_appearance_rate": (
                sum(c.proposal.appearance_type == query.proposal.appearance_type for _, c in scored)
                / denominator if denominator else 0.0
            ),
            "same_operator_rate": (
                sum(c.proposal.operator_type == query.proposal.operator_type for _, c in scored)
                / denominator if denominator else 0.0
            ),
            "same_root_family_rate": (
                sum(_root_family(c) == _root_family(query) for _, c in scored)
                / denominator if denominator else 0.0
            ),
            "mean_similarity": (
                sum(score for score, _ in scored) / denominator if denominator else 0.0
            ),
        })
    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0
    return {
        "top_k": k,
        "evaluated": bool(rows),
        "query_count": len(rows),
        "mean_same_appearance_rate": mean("same_appearance_rate"),
        "mean_same_operator_rate": mean("same_operator_rate"),
        "mean_same_root_family_rate": mean("same_root_family_rate"),
        "mean_similarity": mean("mean_similarity"),
        "per_query": rows,
    }


def _dataset_row(experience: InterventionExperience, split: str) -> dict[str, object]:
    outcome = experience.outcome
    return {
        "trajectory_key": experience.key,
        "base_instance_id": _instance_id(experience),
        "schedule_instance_id": experience.metadata.get("schedule_instance_id"),
        "family": experience.metadata.get("family", "unknown"),
        "split": split,
        "appearance": experience.proposal.appearance_type or "unknown",
        "operator": experience.proposal.operator_type or "unknown",
        "root_family": _root_family(experience),
        "class": _trajectory_class(experience),
        "before_state": asdict(experience.state),
        "causal_chain": list(experience.proposal.causal_chain),
        "causal_relations": list(experience.proposal.causal_relations),
        "root_decision": experience.proposal.root_decision_id,
        "proposal": asdict(experience.proposal),
        "after_state": asdict(experience.after_state) if experience.after_state else None,
        "delta_cmax": float(outcome.delta_cmax) if outcome else None,
        "future_success": bool(experience.trajectory_future_success),
        "future_gain": float(experience.trajectory_final_gain),
        "future_steps": int(experience.trajectory_future_steps),
        "risk": float(outcome.risk) if outcome else None,
        "counterfactual_evidence": dict(experience.metadata),
    }


def load_experience_stores(paths: Iterable[str | Path]) -> tuple[list[InterventionExperience], list[dict[str, object]]]:
    experiences: list[InterventionExperience] = []
    sources: list[dict[str, object]] = []
    for item in sorted({str(Path(path).expanduser().resolve()) for path in paths}):
        path = Path(item)
        source = {"path": item, "exists": path.is_file(), "sha256": None, "raw_records": 0, "loaded_records": 0}
        if not path.is_file():
            sources.append(source)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"memory source is not an object: {path}")
        source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        source["raw_records"] = len(payload)
        store = ExperienceStore(path=path)
        source["loaded_records"] = len(store)
        experiences.extend(store.experiences())
        sources.append(source)
    return experiences, sources


def build_readiness_artifacts(
    experiences: Sequence[InterventionExperience],
    output_dir: str | Path,
    *,
    sources: Sequence[Mapping[str, object]] = (),
    split_seed: int = 20260821,
    retrieval_k: int = 5,
) -> dict[str, object]:
    """Audit, freeze valid rows, and write JSONL/manifest/audit artifacts."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved = [row for row in experiences if row.outcome is not None]
    quality = [audit_counterfactual(row) for row in resolved]
    quality_by_key = {row.key: row for row in quality}
    global_rows = [
        row for row in resolved
        if row.metadata.get("counterfactual_mode") == "free_global"
    ]
    frozen_rows = [
        row for row in resolved
        if row.metadata.get("counterfactual_mode") == "frozen_local"
    ]
    eligible_rows = [row for row in frozen_rows if experience_training_eligible(row)]
    valid = [row for row in eligible_rows if quality_by_key[row.key].valid]
    assignments = freeze_instance_split([_instance_id(row) for row in valid], seed=split_seed)
    rows = [_dataset_row(row, assignments[_instance_id(row)]) for row in valid]

    instance_sets = {
        split: {row["base_instance_id"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    state_sets = {
        split: {_sha(row["before_state"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }
    schedule_sets = {
        split: {
            row["counterfactual_evidence"].get("before_schedule_sha256")
            for row in rows if row["split"] == split
        }
        for split in SPLITS
    }
    overlap: dict[str, dict[str, int]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap[f"{left}__{right}"] = {
                "instance": len(instance_sets[left] & instance_sets[right]),
                "state": len(state_sets[left] & state_sets[right]),
                "schedule": len(schedule_sets[left] & schedule_sets[right]),
            }

    class_counts = Counter(_trajectory_class(row) for row in valid)
    appearance_counts = Counter(row.proposal.appearance_type or "unknown" for row in valid)
    operator_counts = Counter(row.proposal.operator_type or "unknown" for row in valid)
    valid_instance_count = len(set(assignments))
    missing_splits = [split for split in SPLITS if not instance_sets[split]]
    missing_appearances = [name for name in APPEARANCES if appearance_counts[name] == 0]
    missing_operators = [name for name in OPERATORS if operator_counts[name] == 0]
    leakage = any(value for pair in overlap.values() for value in pair.values())
    reasons: list[str] = []
    if not resolved:
        reasons.append("NO_PERSISTED_TRAJECTORIES")
    if len(valid) != len(eligible_rows):
        reasons.append("COUNTERFACTUAL_EVIDENCE_INVALID_OR_INCOMPLETE")
    if frozen_rows and not eligible_rows:
        reasons.append("NO_ATTRIBUTION_CLEAN_FROZEN_LOCAL_TRAJECTORIES")
    if not valid:
        reasons.append("NO_AUDIT_VALID_TRAJECTORIES")
    if missing_splits:
        reasons.append("INSTANCE_SPLIT_INCOMPLETE")
    if leakage:
        reasons.append("SPLIT_LEAKAGE_DETECTED")
    if missing_appearances:
        reasons.append("APPEARANCE_COVERAGE_INCOMPLETE")
    if missing_operators:
        reasons.append("OPERATOR_COVERAGE_INCOMPLETE")

    dataset_path = output / "trajectory_dataset_v1.jsonl"
    dataset_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    dataset_path.write_bytes(dataset_bytes)
    split_payload = {
        "schema": "sgsct_effect_instance_split_v1",
        "seed": split_seed,
        "fractions": SPLIT_FRACTIONS,
        "assignments": assignments,
    }
    split_path = output / "instance_split_v1.json"
    split_path.write_bytes(_canonical_bytes(split_payload))
    diagnostics = retrieval_diagnostics(valid, k=retrieval_k)
    audit_payload = {
        "schema": "sgsct_effect_training_readiness_audit_v1",
        "training_ready": not reasons,
        "verdict": "READY_TO_TRAIN" if not reasons else "NOT_READY_TO_TRAIN",
        "reasons": reasons,
        "statistics": {
            "total_trajectories": len(valid),
            "valid_trajectories": len(valid),
            "invalid_trajectories": len(eligible_rows) - len(valid),
            "source_resolved_count": len(resolved),
            "frozen_local_count": len(frozen_rows),
            "frozen_local_training_eligible_count": len(eligible_rows),
            "frozen_local_ineligible_count": len(frozen_rows) - len(eligible_rows),
            "global_evaluation_count": len(global_rows),
            "legacy_or_unknown_mode_count": len(resolved) - len(frozen_rows) - len(global_rows),
            "unique_instances": valid_instance_count,
            "by_class": {name: class_counts[name] for name in ("success", "partial_success", "failure")},
            "by_appearance": {name: appearance_counts[name] for name in (*APPEARANCES, "unknown")},
            "by_operator": {name: operator_counts[name] for name in (*OPERATORS, "unknown")},
            "by_split_trajectory": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
            "by_split_instance": {split: len(instance_sets[split]) for split in SPLITS},
        },
        "coverage_gaps": {"appearances": missing_appearances, "operators": missing_operators},
        "counterfactual_quality": [asdict(row) for row in quality],
        "retrieval": diagnostics,
        "leakage": {
            "evaluated": bool(rows), "overlap": overlap,
            "passed": (not leakage) if rows else None,
        },
        "sources": list(sources),
    }
    audit_path = output / "training_readiness_audit_v1.json"
    audit_path.write_bytes(_canonical_bytes(audit_payload))
    manifest = {
        "schema": SCHEMA,
        "version": "1",
        "status": "frozen" if rows else "empty_not_trainable",
        "training_ready": audit_payload["training_ready"],
        "formal_training": False,
        "optimizer_steps": 0,
        "formal_test_access": 0,
        "split_unit": "base_instance_id",
        "split_seed": split_seed,
        "split_fractions": SPLIT_FRACTIONS,
        "row_count": len(rows),
        "instance_count": valid_instance_count,
        "artifacts": {
            "dataset": dataset_path.name,
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "split": split_path.name,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "audit": audit_path.name,
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        },
        "source_hashes": [source.get("sha256") for source in sources if source.get("sha256")],
    }
    manifest_path = output / "dataset_manifest_v1.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return {"manifest": manifest, "audit": audit_payload, "rows": rows}


__all__ = [
    "APPEARANCES", "OPERATORS", "QualityResult", "SCHEMA", "SPLIT_FRACTIONS",
    "SPLITS", "audit_counterfactual", "build_readiness_artifacts",
    "freeze_instance_split", "load_experience_stores", "retrieval_diagnostics",
]
