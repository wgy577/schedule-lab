"""Inheritance tree for scheduling-family knowledge.

Each node stores only its local delta. Querying resolves root -> family ->
variant (and optional deeper variants) deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .semantic_knowledge import SchedulingKnowledgeBase


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaxonomyNode(FrozenModel):
    id: str
    parent_id: str | None
    level: int = Field(ge=1)
    node_type: str
    name: str
    aliases: tuple[str, ...] = ()
    local_delta: dict[str, Any] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_summary_placeholders(self) -> "TaxonomyNode":
        normalized = f"{self.id} {self.name}".lower()
        forbidden = ("各类变体", "其他变体", "other_variants", "misc_variants")
        if any(token in normalized for token in forbidden):
            raise ValueError(
                "taxonomy branches must name one concrete type; "
                "summary placeholder branches are forbidden"
            )
        return self


class ResolvedTaxonomyProfile(FrozenModel):
    node_id: str
    lineage: tuple[str, ...]
    resolved: dict[str, Any]
    contributions: dict[str, dict[str, Any]]


class VariantHeadSelection(FrozenModel):
    """Exactly one L3 routing result for a detected scheduling family."""

    family: str
    selected_node_id: str
    matched_heads: tuple[str, ...] = ()
    score: int = Field(ge=0)
    fallback_to_classic: bool


def _merge(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in delta.items():
        if isinstance(value, (tuple, list)):
            prior = list(merged.get(key, ()))
            unique: list[Any] = []
            seen: set[str] = set()
            for item in (*prior, *value):
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if marker not in seen:
                    seen.add(marker)
                    unique.append(item)
            merged[key] = tuple(unique)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


class SchedulingTaxonomy:
    ROOT_ID = "taxonomy:shop_scheduling"

    def __init__(self, nodes: tuple[TaxonomyNode, ...]) -> None:
        self.nodes = nodes
        self._nodes = {item.id: item for item in nodes}
        self._aliases = {
            alias.lower(): item.id
            for item in nodes
            for alias in (item.name, *item.aliases)
        }

    @classmethod
    def from_knowledge_base(
        cls, knowledge: SchedulingKnowledgeBase
    ) -> "SchedulingTaxonomy":
        heads_path = Path(__file__).resolve().parent / "knowledge" / "variant_heads.json"
        heads_document = json.loads(heads_path.read_text(encoding="utf-8"))
        recognition_heads = heads_document["heads"]
        nodes: list[TaxonomyNode] = [
            TaxonomyNode(
                id=cls.ROOT_ID,
                parent_id=None,
                level=1,
                node_type="SchedulingDomain",
                name="车间调度",
                aliases=("shop scheduling", "production scheduling"),
                local_delta={
                    "entities": ("job", "operation", "resource", "schedule"),
                    "shared_features": (
                        "operations are allocated over finite resources and time",
                        "feasibility constraints separate valid from invalid schedules",
                        "objectives compare feasible candidate schedules",
                    ),
                },
            )
        ]
        for family in knowledge.document.families:
            family_id = f"family:{family.family}"
            nodes.append(
                TaxonomyNode(
                    id=family_id,
                    parent_id=cls.ROOT_ID,
                    level=2,
                    node_type="ProblemFamily",
                    name=family.family,
                    aliases=family.aliases,
                    local_delta={
                        "definition": family.definition,
                        "defining_features": family.defining_features,
                        "core_constraints": tuple(
                            item.model_dump(mode="json")
                            for item in family.core_constraints
                        ),
                        "primary_decisions": family.primary_decisions,
                        "default_assumptions": family.default_assumptions,
                    },
                    source_ids=family.source_ids,
                )
            )
            nodes.append(
                TaxonomyNode(
                    id=f"variant:{family.family.lower()}_classic",
                    parent_id=family_id,
                    level=3,
                    node_type="Variant",
                    name=f"classic {family.family}",
                    aliases=(f"classical {family.family}", f"经典{family.family}"),
                    local_delta={
                        "variant_id": f"{family.family.lower()}_classic",
                        "variant_features": (),
                        "decision_changes": (),
                        "impact_note": "No additional variant semantics beyond the family.",
                        "recognition_heads": (),
                    },
                    source_ids=family.source_ids,
                )
            )
            for variant in family.common_variants:
                nodes.append(
                    TaxonomyNode(
                        id=f"variant:{variant.id}",
                        parent_id=family_id,
                        level=3,
                        node_type="Variant",
                        name=variant.id,
                        aliases=variant.aliases,
                        local_delta={
                            "variant_id": variant.id,
                            "variant_features": variant.added_features,
                            "decision_changes": variant.decision_changes,
                            "impact_note": variant.impact_note,
                            "recognition_heads": tuple(
                                recognition_heads[variant.id]
                            ),
                        },
                        source_ids=family.source_ids,
                    )
                )
        return cls(tuple(nodes))

    def select_variant(self, family: str, text: str) -> VariantHeadSelection:
        """Route to one concrete child using sparse heads; classic is fallback."""

        family_node = self.get(family)
        if family_node is None or family_node.node_type != "ProblemFamily":
            raise KeyError(f"unknown scheduling family: {family}")
        normalized = text.casefold()
        ranked: list[tuple[int, int, str, tuple[str, ...]]] = []
        for child in self.children(family_node.id):
            heads = tuple(child.local_delta.get("recognition_heads", ()))
            matched = tuple(head for head in heads if head.casefold() in normalized)
            if matched:
                # Prefer more matched heads, then the most specific (longest) head.
                ranked.append((len(matched), max(map(len, matched)), child.id, matched))
        if ranked:
            count, _, node_id, matched = max(
                ranked, key=lambda item: (item[0], item[1], item[2])
            )
            return VariantHeadSelection(
                family=family_node.name,
                selected_node_id=node_id,
                matched_heads=matched,
                score=count,
                fallback_to_classic=False,
            )
        return VariantHeadSelection(
            family=family_node.name,
            selected_node_id=f"variant:{family_node.name.lower()}_classic",
            score=0,
            fallback_to_classic=True,
        )

    def get(self, node_or_alias: str) -> TaxonomyNode | None:
        return self._nodes.get(node_or_alias) or self._nodes.get(
            self._aliases.get(node_or_alias.lower(), "")
        )

    def children(self, node_id: str) -> tuple[TaxonomyNode, ...]:
        return tuple(
            sorted(
                (item for item in self.nodes if item.parent_id == node_id),
                key=lambda item: item.id,
            )
        )

    def resolve(self, node_or_alias: str) -> ResolvedTaxonomyProfile:
        node = self.get(node_or_alias)
        if node is None:
            raise KeyError(f"unknown taxonomy node or alias: {node_or_alias}")
        lineage: list[TaxonomyNode] = []
        visited: set[str] = set()
        current: TaxonomyNode | None = node
        while current is not None:
            if current.id in visited:
                raise ValueError(f"taxonomy inheritance cycle at {current.id}")
            visited.add(current.id)
            lineage.append(current)
            current = (
                self._nodes.get(current.parent_id)
                if current.parent_id is not None
                else None
            )
        lineage.reverse()
        resolved: dict[str, Any] = {}
        contributions: dict[str, dict[str, Any]] = {}
        for item in lineage:
            contributions[item.id] = item.local_delta
            resolved = _merge(resolved, item.local_delta)
        return ResolvedTaxonomyProfile(
            node_id=node.id,
            lineage=tuple(item.id for item in lineage),
            resolved=resolved,
            contributions=contributions,
        )
