"""Versioned access and deterministic recall for proposed metric knowledge.

The Round 3–6 catalog is a knowledge prior, not runtime project evidence.  Loading
or recalling it does not promote a candidate, prove computability, or enable it in
the optimizer.
"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .llm_semantics import SemanticAnalysis


class KnowledgeRecord(BaseModel):
    """A validated identifier with the remaining audited payload preserved."""

    model_config = ConfigDict(frozen=True, extra="allow")


class CanonicalMetricRecord(KnowledgeRecord):
    metric_id: str
    candidate_role: str
    canonical_name: str
    promotion_status: str


class DiagnosticRecord(KnowledgeRecord):
    diagnostic_id: str
    candidate_role: str
    canonical_name: str
    promotion_status: str


class MembershipRecommendation(KnowledgeRecord):
    candidate_id: str
    view_id: str
    condition_profile_id: str = "incremental_direct"
    contextual_role: str
    contextual_direction: str
    review_status: str = "proposed"


class ComputabilityRequirement(KnowledgeRecord):
    candidate_id: str
    canonical_name: str
    canonical_role: str
    promotion_status: str


class RetrievalReferenceCase(KnowledgeRecord):
    scenario_id: str
    label: str
    reference_label_status: str


class MetricRecallContext(KnowledgeRecord):
    problem_families: tuple[str, ...]
    variant_heads: tuple[str, ...] = ()
    mechanisms: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    available_ir_fields: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ("secondary_metric", "experimental_feature")
    max_options: int = 20
    batch_size: int = 10


class MetricCandidateOption(KnowledgeRecord):
    candidate_id: str
    canonical_name: str
    canonical_role: str
    formula: str | None = None
    unit: str | None = None
    desired_direction: str
    score: float
    matched_views: tuple[str, ...]
    required_ir_fields: tuple[str, ...]
    missing_ir_fields: tuple[str, ...]
    availability: Literal["computable", "needs_adapter_or_evidence"]
    evidence_grade: str
    reason: str


class MetricRecallResult(KnowledgeRecord):
    catalog_version: str
    catalog_metric_count: int
    eligible_before_limit: int
    options: tuple[MetricCandidateOption, ...]
    batches: tuple[tuple[str, ...], ...]
    prompt_character_estimate: int
    context_fingerprint: str


def _default_root() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "secondary_metrics"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extra_tuple(record: KnowledgeRecord, key: str) -> tuple[str, ...]:
    value = (record.model_extra or {}).get(key, ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _canonical_family(value: str) -> str:
    normalized = value.upper()
    return "JSP" if normalized == "JSSP" else normalized


def _deduplicate_memberships(
    rows: list[MembershipRecommendation],
) -> tuple[MembershipRecommendation, ...]:
    """Merge duplicate candidate/view rows without silently losing provenance."""

    merged: dict[tuple[str, str], MembershipRecommendation] = {}
    alternate_reasons: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row.candidate_id, row.view_id)
        if key not in merged:
            merged[key] = row
            continue
        prior = merged[key]
        if (
            prior.contextual_role != row.contextual_role
            or prior.contextual_direction != row.contextual_direction
        ):
            raise ValueError(f"conflicting duplicate membership relation: {key}")
        reason = str((row.model_extra or {}).get("reason", "")).strip()
        if reason:
            alternate_reasons.setdefault(key, []).append(reason)
    for key, reasons in alternate_reasons.items():
        prior = merged[key]
        payload = prior.model_dump(mode="python")
        payload["alternate_reasons"] = tuple(dict.fromkeys(reasons))
        merged[key] = MembershipRecommendation.model_validate(payload)
    return tuple(merged[key] for key in sorted(merged))


class SecondaryMetricKnowledgeBase:
    """Load, audit and deterministically recall the Round 3–6 catalog.

    The optional impact Critic consumes a bounded recall result.  The optimizer
    still does not consume proposed metrics without later validation and approval.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        round3 = root / "round3"
        round4 = root / "round4"

        metric_rows = _read_jsonl(round3 / "candidate_metrics_round2.jsonl")
        diagnostic_rows = _read_jsonl(round3 / "candidate_diagnostics_round2.jsonl")
        membership_rows = _read_jsonl(
            round4 / "membership_condition_recommendations.jsonl"
        )
        for incremental in (root / "round5", root / "round6"):
            metric_rows.extend(
                _read_jsonl(incremental / "candidate_metrics_additions.jsonl")
            )
            diagnostic_rows.extend(
                _read_jsonl(incremental / "candidate_diagnostics_additions.jsonl")
            )
            membership_rows.extend(
                _read_jsonl(
                    incremental / "candidate_view_memberships_additions.jsonl"
                )
            )
        compatibility = root / "compatibility"
        metric_rows.extend(_read_jsonl(compatibility / "legacy_metric_additions.jsonl"))
        membership_rows.extend(
            _read_jsonl(compatibility / "legacy_memberships.jsonl")
        )
        self.metrics = tuple(
            CanonicalMetricRecord.model_validate(item) for item in metric_rows
        )
        self.diagnostics = tuple(
            DiagnosticRecord.model_validate(item) for item in diagnostic_rows
        )
        self.memberships = _deduplicate_memberships(
            [MembershipRecommendation.model_validate(item) for item in membership_rows]
        )
        base_computability = [
            ComputabilityRequirement.model_validate(item)
            for item in _read_jsonl(round4 / "computability_requirements.jsonl")
        ]
        covered = {item.candidate_id for item in base_computability}
        for metric in self.metrics:
            if metric.metric_id in covered:
                continue
            extra = metric.model_extra or {}
            base_computability.append(
                ComputabilityRequirement.model_validate(
                    {
                        "candidate_id": metric.metric_id,
                        "canonical_name": metric.canonical_name,
                        "canonical_role": metric.candidate_role,
                        "promotion_status": metric.promotion_status,
                        "computable_candidate_requirements": {
                            "required_ir_fields": list(
                                _extra_tuple(metric, "required_ir_fields")
                            ),
                            "calculator_inputs": list(
                                _extra_tuple(metric, "calculator_inputs")
                            ),
                            "currently_computable": (
                                "unknown_until_real_project_IR_is_checked"
                            ),
                        },
                    }
                )
            )
        self.computability_requirements = tuple(base_computability)
        self.reference_cases = tuple(
            RetrievalReferenceCase.model_validate(item)
            for item in _read_jsonl(round4 / "retrieval_reference_cases.jsonl")
        )
        self.gap_capability_groups = json.loads(
            (round4 / "gap_capability_groups.json").read_text(encoding="utf-8")
        )
        self.materialization_audit = json.loads(
            (round4 / "round4_materialization_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.legacy_bindings = json.loads(
            (compatibility / "legacy_active_metric_bindings.json").read_text(
                encoding="utf-8"
            )
        )["bindings"]
        self.view_registry = self._load_view_registry()

        self._metric_by_id = {item.metric_id: item for item in self.metrics}
        self._diagnostic_by_id = {
            item.diagnostic_id: item for item in self.diagnostics
        }
        self._computability_by_id = {
            item.candidate_id: item for item in self.computability_requirements
        }
        self._reference_case_by_id = {
            item.scenario_id: item for item in self.reference_cases
        }
        self._capability_group_by_id = {
            item["capability_group_id"]: item
            for item in self.gap_capability_groups["capability_groups"]
        }
        self._validate_integrity()

    @classmethod
    def load(
        cls, root: str | Path | None = None
    ) -> "SecondaryMetricKnowledgeBase":
        resolved = Path(root).expanduser().resolve() if root else _default_root()
        return cls(resolved)

    def _validate_integrity(self) -> None:
        if not self.materialization_audit.get("all_checks_passed", False):
            raise ValueError("Round 4 materialization audit did not pass")
        if len(self.metrics) != len(self._metric_by_id):
            raise ValueError("Duplicate canonical metric candidates")
        if len(self.diagnostics) != len(self._diagnostic_by_id):
            raise ValueError("Duplicate diagnostic candidates")
        membership_pairs = {
            (item.candidate_id, item.view_id) for item in self.memberships
        }
        if len(membership_pairs) != len(self.memberships):
            raise ValueError("Duplicate candidate/view membership relation")
        unknown_membership_ids = {
            item.candidate_id for item in self.memberships
        } - set(self._metric_by_id) - set(self._diagnostic_by_id)
        if unknown_membership_ids:
            raise ValueError(
                f"Unknown membership candidate IDs: {sorted(unknown_membership_ids)}"
            )
        if set(self._computability_by_id) != set(self._metric_by_id):
            raise ValueError("Computability rows do not cover exactly the 55 candidates")
        if any(item.promotion_status != "proposed" for item in self.metrics):
            raise ValueError("Knowledge import cannot promote canonical candidates")
        if any(
            item.reference_label_status != "design_reference_not_independent_truth"
            for item in self.reference_cases
        ):
            raise ValueError("Reference cases must remain non-independent design labels")
        binding_kinds = {item["legacy_kind"] for item in self.legacy_bindings}
        if len(binding_kinds) != 11:
            raise ValueError("Legacy active catalog must map exactly 11 metric kinds")
        missing_bindings = {
            item["candidate_id"] for item in self.legacy_bindings
        } - set(self._metric_by_id)
        if missing_bindings:
            raise ValueError(f"Legacy bindings reference missing metrics: {missing_bindings}")

    def _load_view_registry(self) -> dict[str, dict[str, Any]]:
        base = json.loads(
            (self.root / "round3" / "view_registry.json").read_text(
                encoding="utf-8"
            )
        )
        views = {
            value["view_id"]: value
            for axis in base["axes"]
            for value in axis["values"]
        }
        for round_name in ("round5", "round6"):
            addition = json.loads(
                (self.root / round_name / "view_registry_additions.json").read_text(
                    encoding="utf-8"
                )
            )
            for value in addition["views"]:
                views[value["view_id"]] = value
        return views

    def metric(self, candidate_id: str) -> CanonicalMetricRecord | None:
        return self._metric_by_id.get(candidate_id)

    def diagnostic(self, diagnostic_id: str) -> DiagnosticRecord | None:
        return self._diagnostic_by_id.get(diagnostic_id)

    def memberships_for(
        self, *, candidate_id: str | None = None, view_id: str | None = None
    ) -> tuple[MembershipRecommendation, ...]:
        return tuple(
            item
            for item in self.memberships
            if (candidate_id is None or item.candidate_id == candidate_id)
            and (view_id is None or item.view_id == view_id)
        )

    def computability(
        self, candidate_id: str
    ) -> ComputabilityRequirement | None:
        return self._computability_by_id.get(candidate_id)

    def capability_group(self, group_id: str) -> dict[str, Any] | None:
        return self._capability_group_by_id.get(group_id)

    def reference_case(self, scenario_id: str) -> RetrievalReferenceCase | None:
        return self._reference_case_by_id.get(scenario_id)

    def legacy_candidate_id(self, legacy_kind: str) -> str | None:
        for item in self.legacy_bindings:
            if item["legacy_kind"] == legacy_kind:
                return str(item["candidate_id"])
        return None

    def context_from_analysis(
        self, analysis: "SemanticAnalysis"
    ) -> MetricRecallContext:
        corpus_parts = [analysis.summary]
        corpus_parts.extend(item.statement for item in analysis.environments)
        corpus_parts.extend(item.statement for item in analysis.objectives)
        corpus_parts.extend(item.statement for item in analysis.constraints)
        corpus_parts.extend(item.statement for item in analysis.decisions)
        corpus = " ".join(corpus_parts).casefold()
        families = tuple(
            dict.fromkeys(
                _canonical_family(getattr(item, "value", str(item)))
                for item in analysis.problem_families
                if _canonical_family(getattr(item, "value", str(item)))
                in {"JSP", "FSP", "FJSP", "HFSP"}
            )
        )
        # Variant activation must come from structured, evidence-linked project
        # semantics.  Free-text summaries frequently contain negated inventory
        # statements such as "no setup or blocking"; lexical matching those
        # statements would activate exactly the variants the code ruled out.
        variants: list[str] = []
        constraint_variant_map = {
            "resource_eligibility": ("machine_eligibility",),
            "setup_time": ("setup",),
            "transport": ("transport",),
            "route_continuity": ("transport",),
            "collision_avoidance": ("transport",),
            "buffer": ("limited_buffer",),
            "blocking": ("blocking",),
            "no_wait": ("no_wait",),
            "batching": ("batch_processing",),
        }
        for item in analysis.constraints:
            variants.extend(
                constraint_variant_map.get(
                    getattr(item.kind, "value", str(item.kind)), ()
                )
            )
        decision_map = {
            "assign_resource": ("machine_assignment",),
            "sequence": ("machine_sequence", "job_sequence"),
            "start_time": ("start_time",),
            "route": ("vehicle_route",),
            "batch": ("batch_formation",),
            "release": ("job_release", "buffer_release"),
            "select": ("machine_assignment", "job_sequence"),
        }
        decisions = tuple(
            dict.fromkeys(
                mapped
                for item in analysis.decisions
                if item.modifiable
                for mapped in decision_map.get(
                    getattr(item.kind, "value", str(item.kind)), ()
                )
            )
        )
        mechanism_map = {
            "precedence": ("critical_structure", "waiting"),
            "resource_capacity": ("bottleneck_workload", "idle"),
            "resource_eligibility": ("flexibility", "bottleneck_workload"),
            "setup_time": ("setup_batch",),
            "transport": ("transport", "waiting"),
            "route_continuity": ("transport",),
            "collision_avoidance": ("transport",),
            "buffer": ("blocking_buffer",),
            "blocking": ("blocking_buffer",),
            "no_wait": ("waiting",),
            "batching": ("setup_batch",),
            "synchronization": ("waiting", "auxiliary_resource"),
        }
        mechanisms = tuple(
            dict.fromkeys(
                mechanism
                for item in analysis.constraints
                for mechanism in mechanism_map.get(
                    getattr(item.kind, "value", str(item.kind)), ()
                )
            )
        )
        all_required_fields = {
            field
            for metric in self.metrics
            for field in _extra_tuple(metric, "required_ir_fields")
        }
        available_fields = tuple(
            sorted(field for field in all_required_fields if field.casefold() in corpus)
        )
        return MetricRecallContext(
            problem_families=families,
            variant_heads=tuple(sorted(set(variants))),
            mechanisms=mechanisms,
            decisions=decisions,
            available_ir_fields=available_fields,
        )

    def recall(
        self, context: MetricRecallContext
    ) -> MetricRecallResult:
        families = {_canonical_family(item) for item in context.problem_families}
        variants = set(context.variant_heads)
        mechanisms = {f"mechanism:{item}" for item in context.mechanisms}
        decisions = {f"decision:{item}" for item in context.decisions}
        available_fields = set(context.available_ir_fields)
        allowed_roles = set(context.allowed_roles)
        memberships_by_candidate: dict[str, list[MembershipRecommendation]] = {}
        for relation in self.memberships:
            memberships_by_candidate.setdefault(relation.candidate_id, []).append(
                relation
            )
        legacy_ids = {item["candidate_id"] for item in self.legacy_bindings}
        ranked: list[MetricCandidateOption] = []
        for metric in self.metrics:
            if metric.candidate_role not in allowed_roles:
                continue
            extra = metric.model_extra or {}
            metric_families = {
                _canonical_family(item)
                for item in _extra_tuple(metric, "problem_family")
            }
            relations = memberships_by_candidate.get(metric.metric_id, [])
            metric_families.update(
                item.view_id.split(":", 1)[1]
                for item in relations
                if item.view_id.startswith("problem_family:")
            )
            if families and not (families & metric_families):
                continue
            required_variants = set(_extra_tuple(metric, "required_variant_heads"))
            if required_variants and not (required_variants & variants):
                continue
            matched_views = {
                item.view_id
                for item in relations
                if (
                    item.view_id in mechanisms
                    or item.view_id in decisions
                    or item.view_id
                    in {f"problem_family:{family}" for family in families}
                    or item.view_id
                    in {f"variant_head:{variant}" for variant in variants}
                )
            }
            required_decisions = set(_extra_tuple(metric, "required_decisions"))
            decision_overlap = required_decisions & set(context.decisions)
            required_fields = _extra_tuple(metric, "required_ir_fields")
            missing_fields = tuple(
                field for field in required_fields if field not in available_fields
            )
            role_bonus = 3.0 if metric.candidate_role == "secondary_metric" else 1.0
            evidence_grade = str(extra.get("evidence_grade", "unknown"))
            evidence_bonus = {
                "intervention_supported": 4.0,
                "variation_verified": 3.5,
                "code_backed": 3.0,
                "project_formula_defined": 2.0,
                "source_definition_verified": 1.5,
                "survey_map": 1.0,
            }.get(evidence_grade, 0.0)
            score = (
                10.0 * bool(families & metric_families)
                + 7.0 * bool(required_variants & variants)
                + 3.0 * len(mechanisms & matched_views)
                + 2.0 * len(decisions & matched_views)
                + 1.0 * len(decision_overlap)
                + role_bonus
                + evidence_bonus
                + (0.5 if metric.metric_id in legacy_ids else 0.0)
            )
            formula = extra.get("formula") or extra.get("formula_or_rule")
            direction = str(extra.get("desired_direction", "context_dependent"))
            reason = (
                f"family={sorted(families & metric_families)}; "
                f"variant={sorted(required_variants & variants)}; "
                f"views={sorted(matched_views)}; missing_ir={list(missing_fields)}"
            )
            ranked.append(
                MetricCandidateOption(
                    candidate_id=metric.metric_id,
                    canonical_name=metric.canonical_name,
                    canonical_role=metric.candidate_role,
                    formula=str(formula) if formula else None,
                    unit=str(extra.get("unit")) if extra.get("unit") else None,
                    desired_direction=direction,
                    score=score,
                    matched_views=tuple(sorted(matched_views)),
                    required_ir_fields=required_fields,
                    missing_ir_fields=missing_fields,
                    availability=(
                        "computable" if not missing_fields else "needs_adapter_or_evidence"
                    ),
                    evidence_grade=evidence_grade,
                    reason=reason,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.candidate_id))
        options = tuple(ranked[: context.max_options])
        batch_size = max(1, min(context.batch_size, 12))
        batches = tuple(
            tuple(item.candidate_id for item in options[index : index + batch_size])
            for index in range(0, len(options), batch_size)
        )
        compact = [
            {
                "candidate_id": item.candidate_id,
                "name": item.canonical_name,
                "role": item.canonical_role,
                "formula": item.formula,
                "unit": item.unit,
                "direction": item.desired_direction,
                "availability": item.availability,
                "matched_views": item.matched_views,
            }
            for item in options
        ]
        serialized = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            context.model_dump_json().encode("utf-8")
        ).hexdigest()
        return MetricRecallResult(
            catalog_version="round3+round4+round5+round6+legacy11-v1",
            catalog_metric_count=len(self.metrics),
            eligible_before_limit=len(ranked),
            options=options,
            batches=batches,
            prompt_character_estimate=len(serialized),
            context_fingerprint=fingerprint,
        )

    def recall_for_analysis(
        self, analysis: "SemanticAnalysis"
    ) -> MetricRecallResult:
        return self.recall(self.context_from_analysis(analysis))
