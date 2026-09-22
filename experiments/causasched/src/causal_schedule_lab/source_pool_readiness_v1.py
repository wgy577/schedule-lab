"""Frozen Dataset-v5 static source-pool readiness evaluation.

This module only evaluates label-blind source-pool artifacts.  It does not load
TEST, call the counterfactual Executor, generate labels, or train a model.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


FAMILIES = ("FJSP", "JSP", "HFSP", "FSP")
FROZEN_GATE_NAMES = (
    "independent_total",
    "benchmark_families",
    "problem_families",
    "external_coverage_per_family",
    "benchmark_concentration",
    "train_external_coverage",
    "val_external_coverage",
    "exact_duplicates",
    "topology_near_duplicates",
    "provenance_completeness",
    "adapter_compatibility",
    "structural_coverage",
    "difficulty_coverage",
    "split_integrity",
)

_PROVENANCE_FIELDS = (
    "instance_uid",
    "problem_family",
    "source_family",
    "benchmark_family",
    "source_instance_id",
    "original_instance_name",
    "original_source_identifier",
    "generator_name",
    "generator_version",
    "random_seed",
    "generation_parameters",
    "adapter_version",
    "source_pool_round",
)


def _effective_provenance_value(row: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    """Resolve old frozen local rows without rewriting their provenance schema."""

    if field in row and row[field] is not None:
        return True, row[field]
    nested = row.get("provenance") or {}
    aliases = {"generator_name": "generator_id"}
    nested_field = aliases.get(field, field)
    if nested_field in nested:
        return True, nested[nested_field]
    # Real benchmark generator fields must be present and explicitly null.
    if row.get("is_external") and field in {
        "generator_name",
        "generator_version",
        "random_seed",
        "generation_parameters",
    } and field in row:
        return True, row[field]
    return field in row, row.get(field)


def provenance_row_complete(row: Mapping[str, Any]) -> bool:
    """Return whether one row preserves every preregistered provenance field."""

    return all(_effective_provenance_value(row, field)[0] for field in _PROVENANCE_FIELDS)


def _adapter_row_passes(row: Mapping[str, Any]) -> bool:
    if not row.get("is_external"):
        return True
    if "adapter_hard_contract" in row:
        return bool(row["adapter_hard_contract"].get("valid"))
    audit = row.get("hard_contract_audit") or row.get("latest_hard_contract_audit") or {}
    identity = audit.get("identity", {})
    family = audit.get("family_semantics", {})
    round_trip = audit.get("round_trip", {})
    return (
        bool(family.get("passed"))
        and all(identity.values())
        and all(round_trip.values())
    )


def evaluate_source_pool_readiness(
    preregistration: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    duplicate_audit: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the 14 table rows in SOURCE_DIVERSITY_PREREGISTRATION.md.

    Gate 2 and Gate 5 are deliberately composite.  Splitting their subclauses
    into extra numbered gates would silently omit the preregistered near-
    duplicate, structural, and difficulty gates.
    """

    hard = preregistration["hard_gates"]
    independent = [row for row in instances if row.get("is_independent", True)]
    external = [row for row in independent if row.get("is_external", False)]
    train_uids = set(split["train_instance_uids"])
    val_uids = set(split["val_instance_uids"])
    train = [row for row in independent if row["instance_uid"] in train_uids]
    val = [row for row in independent if row["instance_uid"] in val_uids]
    ext_train = [row for row in train if row.get("is_external", False)]
    ext_val = [row for row in val if row.get("is_external", False)]

    benchmark_counts = Counter(row["benchmark_family"] for row in independent)
    external_benchmark_counts = Counter(row["benchmark_family"] for row in external)
    family_counts = Counter(row["problem_family"] for row in independent)
    external_family_counts = Counter(row["problem_family"] for row in external)
    train_external_family_counts = Counter(row["problem_family"] for row in ext_train)
    val_external_family_counts = Counter(row["problem_family"] for row in ext_val)

    largest_share = max(benchmark_counts.values(), default=0) / max(1, len(independent))
    largest_external_share = max(external_benchmark_counts.values(), default=0) / max(1, len(external))
    provenance_pass_count = sum(provenance_row_complete(row) for row in independent)
    applicable_adapter_rows = [row for row in independent if row.get("is_external", False)]
    adapter_pass_count = sum(_adapter_row_passes(row) for row in applicable_adapter_rows)

    family_difficulty = census["difficulty_by_problem_family"]
    difficulty_ok = all(
        family_counts[family] < 10
        or sum(count > 0 for count in family_difficulty.get(family, {}).values())
        >= hard["difficulty_classes_per_family_min"]
        for family in FAMILIES
    )
    train_share = len(ext_train) / max(1, len(train))
    val_share = len(ext_val) / max(1, len(val))
    all_uids = {row["instance_uid"] for row in independent}
    split_ok = (
        not (train_uids & val_uids)
        and train_uids | val_uids == all_uids
        and all(any(row["problem_family"] == family for row in train) for family in FAMILIES)
        and all(any(row["problem_family"] == family for row in val) for family in FAMILIES)
        and not split.get("test_instance_uids_loaded", [])
        and split.get("test_access") == hard["test_access_required"]
    )

    raw_gates = (
        (
            "independent_total",
            len(independent) >= hard["independent_total_min"],
            f"{len(independent)} >= {hard['independent_total_min']}",
        ),
        (
            "benchmark_families",
            len(benchmark_counts) >= hard["benchmark_family_count_min"]
            and len(external_benchmark_counts) >= hard["external_benchmark_family_count_min"],
            f"total={len(benchmark_counts)} >= {hard['benchmark_family_count_min']}; "
            f"external={len(external_benchmark_counts)} >= {hard['external_benchmark_family_count_min']}",
        ),
        (
            "problem_families",
            all(family_counts[family] > 0 for family in FAMILIES),
            str(dict(family_counts)),
        ),
        (
            "external_coverage_per_family",
            all(
                external_family_counts[family] >= hard["external_instances_per_problem_family_min"]
                for family in FAMILIES
            ),
            str(dict(external_family_counts)),
        ),
        (
            "benchmark_concentration",
            largest_share <= hard["largest_benchmark_share_max"]
            and largest_external_share <= hard["largest_external_benchmark_share_max"],
            f"all={largest_share:.6f} <= {hard['largest_benchmark_share_max']}; "
            f"external={largest_external_share:.6f} <= {hard['largest_external_benchmark_share_max']}",
        ),
        (
            "train_external_coverage",
            all(
                train_external_family_counts[family] >= hard["train_external_per_problem_family_min"]
                for family in FAMILIES
            )
            and train_share >= hard["overall_train_external_share_min"],
            f"per_family={dict(train_external_family_counts)}; share={train_share:.6f}",
        ),
        (
            "val_external_coverage",
            all(
                val_external_family_counts[family] >= hard["val_external_per_problem_family_min"]
                for family in FAMILIES
            )
            and val_share >= hard["overall_val_external_share_min"],
            f"per_family={dict(val_external_family_counts)}; share={val_share:.6f}",
        ),
        (
            "exact_duplicates",
            duplicate_audit["exact_duplicate_count"] <= hard["exact_duplicate_count_max"],
            f"{duplicate_audit['exact_duplicate_count']} <= {hard['exact_duplicate_count_max']}",
        ),
        (
            "topology_near_duplicates",
            duplicate_audit["overall_near_duplicate_share"] <= hard["near_duplicate_excess_share_max"]
            and all(
                share <= hard["near_duplicate_family_share_max"]
                for share in duplicate_audit["family_near_duplicate_shares"].values()
            ),
            f"overall={duplicate_audit['overall_near_duplicate_share']:.6f}; "
            f"per_family={duplicate_audit['family_near_duplicate_shares']}",
        ),
        (
            "provenance_completeness",
            provenance_pass_count / max(1, len(independent)) >= hard["provenance_completeness_min"],
            f"{provenance_pass_count}/{len(independent)} rows complete",
        ),
        (
            "adapter_compatibility",
            adapter_pass_count / max(1, len(applicable_adapter_rows))
            >= hard["adapter_hard_contract_pass_rate_min"]
            and bool(census["static_superfamily_compatibility_ready"]),
            f"applicable hard contracts={adapter_pass_count}/{len(applicable_adapter_rows)}; "
            f"runtime compatibility={census['static_superfamily_compatibility_ready']}",
        ),
        (
            "structural_coverage",
            bool(census["structural_coverage_acceptance"]),
            f"tuple_counts={census['structural_tuple_count_by_problem_family']}; "
            f"semantic_checks={census.get('structural_semantic_coverage_by_family', {})}",
        ),
        (
            "difficulty_coverage",
            difficulty_ok,
            str(family_difficulty),
        ),
        (
            "split_integrity",
            split_ok,
            f"TRAIN={len(train_uids)}, VAL={len(val_uids)}, overlap={len(train_uids & val_uids)}, "
            f"TEST loaded={len(split.get('test_instance_uids_loaded', []))}",
        ),
    )
    assert tuple(name for name, _, _ in raw_gates) == FROZEN_GATE_NAMES
    gates = [
        {"gate_number": index, "name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
        for index, (name, passed, evidence) in enumerate(raw_gates, 1)
    ]
    passed = sum(gate["status"] == "PASS" for gate in gates)
    return {
        "schema": "dataset_v5_source_pool_readiness_evaluation_v1",
        "gate_definition_source": "SOURCE_DIVERSITY_PREREGISTRATION.md table rows",
        "gate_count": len(gates),
        "passed_gate_count": passed,
        "failed_gate_count": len(gates) - passed,
        "failed_gates": [gate["name"] for gate in gates if gate["status"] == "FAIL"],
        "gates": gates,
        "ready": passed == len(gates),
    }
