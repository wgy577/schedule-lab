"""Curated scheduling-family retrieval for semantic compilation.

The knowledge base provides priors, not project evidence.  A retrieved profile may
reduce repeated reasoning about classical scheduling definitions, but every
project-specific claim still needs repository or document evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeSource(FrozenModel):
    id: str
    title: str
    url: str
    kind: Literal[
        "official_documentation",
        "peer_reviewed_survey",
        "peer_reviewed_paper",
        "standard",
        "other",
    ]
    supports: tuple[str, ...]


class CoreConstraintPrior(FrozenModel):
    kind: str
    role: Literal[
        "feasibility_guard",
        "decision_lever",
        "fixed_instance_structure",
    ]
    typical_optimization_leverage: Literal[
        "critical",
        "high",
        "medium",
        "low",
        "none",
    ]
    note: str


class VariantPrior(FrozenModel):
    id: str
    aliases: tuple[str, ...]
    added_features: tuple[str, ...]
    decision_changes: tuple[str, ...]
    impact_note: str


class SchedulingFamilyProfile(FrozenModel):
    family: Literal["JSP", "FSP", "HFSP", "FJSP"]
    aliases: tuple[str, ...]
    definition: str
    defining_features: tuple[str, ...]
    core_constraints: tuple[CoreConstraintPrior, ...]
    primary_decisions: tuple[str, ...]
    default_assumptions: tuple[str, ...]
    common_variants: tuple[VariantPrior, ...]
    source_ids: tuple[str, ...]


class EngineeringPatternProfile(FrozenModel):
    id: str
    aliases: tuple[str, ...]
    applies_to: tuple[Literal["JSP", "FSP", "HFSP", "FJSP"], ...]
    introduced_entities: tuple[str, ...]
    added_decisions: tuple[str, ...]
    constraint_choices: tuple[str, ...]
    objective_choices: tuple[str, ...]
    oracle_needs: tuple[str, ...]
    screening_questions: tuple[str, ...]
    source_ids: tuple[str, ...]


class SchedulingKnowledgeDocument(FrozenModel):
    schema_version: Literal["1.0"]
    review_status: Literal["curated_seed"]
    updated_at: str
    scope: str
    sources: tuple[KnowledgeSource, ...]
    families: tuple[SchedulingFamilyProfile, ...]
    engineering_patterns: tuple[EngineeringPatternProfile, ...] = ()


class VariantHit(FrozenModel):
    id: str
    matched_terms: tuple[str, ...]


class KnowledgeHit(FrozenModel):
    family: Literal["JSP", "FSP", "HFSP", "FJSP"]
    score: float = Field(ge=0.0)
    matched_terms: tuple[str, ...]
    matched_variants: tuple[VariantHit, ...]
    profile: SchedulingFamilyProfile
    source_urls: tuple[str, ...]


class EngineeringPatternHit(FrozenModel):
    pattern_id: str
    score: float = Field(gt=0.0)
    matched_terms: tuple[str, ...]
    profile: EngineeringPatternProfile
    source_urls: tuple[str, ...]


def _default_path() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "scheduling_families.json"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _term_present(text: str, term: str) -> bool:
    normalized_term = _normalized(term)
    if not normalized_term:
        return False
    # Short Latin aliases such as JSP must match token boundaries; otherwise
    # JSP would spuriously match FJSP.
    if re.fullmatch(r"[a-z0-9-]{1,6}", normalized_term):
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
                text,
            )
            is not None
        )
    return normalized_term in text


class SchedulingKnowledgeBase:
    def __init__(self, document: SchedulingKnowledgeDocument) -> None:
        self.document = document
        self._profiles = {item.family: item for item in document.families}
        self._sources = {item.id: item for item in document.sources}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SchedulingKnowledgeBase":
        source = Path(path).expanduser().resolve() if path else _default_path()
        return cls(
            SchedulingKnowledgeDocument.model_validate_json(
                source.read_text(encoding="utf-8")
            )
        )

    def exact(self, family: str) -> SchedulingFamilyProfile | None:
        canonical = family.upper()
        if canonical == "JSSP":
            canonical = "JSP"
        return self._profiles.get(canonical)

    def retrieve(
        self,
        text: str,
        *,
        family_hints: tuple[str, ...] = (),
        top_k: int = 2,
    ) -> tuple[KnowledgeHit, ...]:
        corpus = _normalized(text)
        canonical_hints = {
            "JSP" if item.upper() == "JSSP" else item.upper()
            for item in family_hints
        }
        hits: list[KnowledgeHit] = []
        for profile in self.document.families:
            matched = tuple(
                alias for alias in profile.aliases if _term_present(corpus, alias)
            )
            variants = []
            for variant in profile.common_variants:
                variant_terms = tuple(
                    alias
                    for alias in variant.aliases
                    if _term_present(corpus, alias)
                )
                if variant_terms:
                    variants.append(
                        VariantHit(
                            id=variant.id,
                            matched_terms=variant_terms,
                        )
                    )
            hint_bonus = 100.0 if profile.family in canonical_hints else 0.0
            score = hint_bonus + 10.0 * len(matched) + 2.0 * len(variants)
            # A generic word such as ``batch_size`` must not invent an HFSP
            # classification. Variant terms may rank an already identified
            # family, but cannot establish the family by themselves.
            if not matched and hint_bonus == 0:
                continue
            source_urls = tuple(
                self._sources[source_id].url
                for source_id in profile.source_ids
                if source_id in self._sources
            )
            hits.append(
                KnowledgeHit(
                    family=profile.family,
                    score=score,
                    matched_terms=matched,
                    matched_variants=tuple(variants),
                    profile=profile,
                    source_urls=source_urls,
                )
            )
        hits.sort(key=lambda item: (-item.score, item.family))
        return tuple(hits[:top_k])

    def retrieve_engineering_patterns(
        self,
        text: str,
        *,
        family_hints: tuple[str, ...] = (),
        top_k: int = 8,
    ) -> tuple[EngineeringPatternHit, ...]:
        corpus = _normalized(text)
        canonical_hints = {
            "JSP" if item.upper() == "JSSP" else item.upper()
            for item in family_hints
        }
        hits = []
        for profile in self.document.engineering_patterns:
            if canonical_hints and not (
                canonical_hints & set(profile.applies_to)
            ):
                continue
            matched = tuple(
                alias
                for alias in profile.aliases
                if _term_present(corpus, alias)
            )
            if not matched:
                continue
            hits.append(
                EngineeringPatternHit(
                    pattern_id=profile.id,
                    score=float(len(matched)),
                    matched_terms=matched,
                    profile=profile,
                    source_urls=tuple(
                        self._sources[source_id].url
                        for source_id in profile.source_ids
                        if source_id in self._sources
                    ),
                )
            )
        hits.sort(key=lambda item: (-item.score, item.pattern_id))
        return tuple(hits[:top_k])


def compact_knowledge_context(
    hits: tuple[KnowledgeHit, ...],
    engineering_hits: tuple[EngineeringPatternHit, ...] = (),
) -> str:
    """Serialize only the fields the LLM needs for semantic comparison."""

    payload = []
    for hit in hits:
        payload.append(
            {
                "family": hit.family,
                "retrieval_score": hit.score,
                "matched_terms": hit.matched_terms,
                "matched_variants": [
                    item.model_dump(mode="json")
                    for item in hit.matched_variants
                ],
                "definition": hit.profile.definition,
                "defining_features": hit.profile.defining_features,
                "core_constraints": [
                    item.model_dump(mode="json")
                    for item in hit.profile.core_constraints
                ],
                "primary_decisions": hit.profile.primary_decisions,
                "default_assumptions": hit.profile.default_assumptions,
                "common_variants": [
                    item.model_dump(mode="json")
                    for item in hit.profile.common_variants
                ],
                "sources": hit.source_urls,
            }
        )
    engineering = [
        {
            "pattern_id": hit.pattern_id,
            "matched_terms": hit.matched_terms,
            "introduced_entities": hit.profile.introduced_entities,
            "added_decisions": hit.profile.added_decisions,
            "constraint_choices": hit.profile.constraint_choices,
            "objective_choices": hit.profile.objective_choices,
            "oracle_needs": hit.profile.oracle_needs,
            "screening_questions": hit.profile.screening_questions,
            "sources": hit.source_urls,
        }
        for hit in engineering_hits
    ]
    return json.dumps(
        {
            "family_profiles": payload,
            "engineering_pattern_candidates": engineering,
        },
        ensure_ascii=False,
        indent=2,
    )
