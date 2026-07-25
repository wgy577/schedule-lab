"""Graph-gated hierarchical retrieval and curated seed ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..semantic_knowledge import SchedulingKnowledgeBase
from .graph_store import EvidenceChunk, GraphEdge, GraphNode, SQLiteGraphStore


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalEvidence(FrozenModel):
    chunk: EvidenceChunk
    lexical_rank: float
    layer_priority: int


class RetrievalResult(FrozenModel):
    query: str
    resolved_ids: tuple[str, ...]
    expanded_ids: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    evidence: tuple[RetrievalEvidence, ...]
    conflicts: tuple[str, ...] = ()


class IngestionReport(FrozenModel):
    changed_nodes: int = Field(ge=0)
    changed_edges: int = Field(ge=0)
    changed_evidence: int = Field(ge=0)
    aliases_processed: int = Field(ge=0)


def _edge_id(source: str, relation: str, target: str) -> str:
    return f"{source}::{relation}::{target}"


class HierarchicalMemory:
    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store

    @classmethod
    def open(cls, path: str | Path) -> "HierarchicalMemory":
        return cls(SQLiteGraphStore(path))

    def ingest_curated_seed(
        self,
        *,
        knowledge_path: str | Path | None = None,
        mechanism_path: str | Path | None = None,
    ) -> IngestionReport:
        knowledge = SchedulingKnowledgeBase.load(knowledge_path)
        mechanisms_source = (
            Path(mechanism_path).expanduser().resolve()
            if mechanism_path
            else Path(__file__).resolve().parents[1]
            / "knowledge"
            / "mechanism_targets.json"
        )
        mechanisms = json.loads(mechanisms_source.read_text(encoding="utf-8"))
        nodes = edges = evidence = aliases = 0

        source_map = {item.id: item for item in knowledge.document.sources}
        for source in knowledge.document.sources:
            source_id = f"source:{source.id}"
            nodes += self.store.upsert_node(
                GraphNode(
                    canonical_id=source_id,
                    node_type="EvidenceSource",
                    name=source.title,
                    summary=f"{source.kind}: {', '.join(source.supports)}",
                    review_status="source_verified",
                    source_ids=(source.id,),
                    metadata={"url": source.url, "kind": source.kind},
                )
            )
            evidence += self.store.add_evidence(
                EvidenceChunk(
                    chunk_id=f"source-summary:{source.id}",
                    layer=3,
                    title=source.title,
                    content=(
                        f"{source.title}. Supports: {', '.join(source.supports)}. "
                        f"Source kind: {source.kind}."
                    ),
                    source_kind=source.kind,
                    source_uri=source.url,
                    review_status="source_verified",
                    node_ids=(source_id,),
                )
            )

        for family in knowledge.document.families:
            family_id = f"family:{family.family}"
            nodes += self.store.upsert_node(
                GraphNode(
                    canonical_id=family_id,
                    node_type="ProblemFamily",
                    name=family.family,
                    summary=family.definition,
                    review_status="active",
                    source_ids=family.source_ids,
                    metadata={
                        "defining_features": family.defining_features,
                        "primary_decisions": family.primary_decisions,
                        "default_assumptions": family.default_assumptions,
                    },
                )
            )
            for alias in (family.family, *family.aliases):
                self.store.add_alias(alias, family_id)
                aliases += 1
            evidence += self.store.add_evidence(
                EvidenceChunk(
                    chunk_id=f"family:{family.family}:summary",
                    layer=0,
                    title=f"{family.family} family",
                    content=(
                        family.definition
                        + " "
                        + " ".join(family.defining_features)
                        + " "
                        + " ".join(family.primary_decisions)
                    ),
                    source_kind="curated_seed",
                    scope=family.family,
                    review_status="active",
                    node_ids=(family_id,),
                    metadata={"source_ids": family.source_ids},
                )
            )
            for source_id in family.source_ids:
                target = f"source:{source_id}"
                edges += self.store.upsert_edge(
                    GraphEdge(
                        edge_id=_edge_id(family_id, "evidenced_by", target),
                        source_id=family_id,
                        relation="evidenced_by",
                        target_id=target,
                        review_status="source_verified",
                        source_ids=(source_id,),
                    )
                )

            for constraint in family.core_constraints:
                constraint_id = f"constraint:{family.family}:{constraint.kind}"
                nodes += self.store.upsert_node(
                    GraphNode(
                        canonical_id=constraint_id,
                        node_type="Constraint",
                        name=constraint.kind,
                        summary=constraint.note,
                        scope=family.family,
                        review_status="active",
                        source_ids=family.source_ids,
                        metadata={
                            "role": constraint.role,
                            "typical_optimization_leverage": (
                                constraint.typical_optimization_leverage
                            ),
                        },
                    )
                )
                edges += self.store.upsert_edge(
                    GraphEdge(
                        edge_id=_edge_id(
                            family_id, "adds_constraint", constraint_id
                        ),
                        source_id=family_id,
                        relation="adds_constraint",
                        target_id=constraint_id,
                        scope=family.family,
                        review_status="active",
                        source_ids=family.source_ids,
                    )
                )
                evidence += self.store.add_evidence(
                    EvidenceChunk(
                        chunk_id=f"{constraint_id}:summary",
                        layer=2,
                        title=f"{family.family} {constraint.kind}",
                        content=constraint.note,
                        source_kind="curated_seed",
                        scope=family.family,
                        review_status="active",
                        node_ids=(family_id, constraint_id),
                    )
                )

            for variant in family.common_variants:
                variant_id = f"variant:{variant.id}"
                nodes += self.store.upsert_node(
                    GraphNode(
                        canonical_id=variant_id,
                        node_type="Variant",
                        name=variant.id,
                        summary=variant.impact_note,
                        scope=family.family,
                        review_status="active",
                        source_ids=family.source_ids,
                        metadata={
                            "added_features": variant.added_features,
                            "decision_changes": variant.decision_changes,
                        },
                    )
                )
                edges += self.store.upsert_edge(
                    GraphEdge(
                        edge_id=_edge_id(variant_id, "variant_of", family_id),
                        source_id=variant_id,
                        relation="variant_of",
                        target_id=family_id,
                        scope=family.family,
                        review_status="active",
                        source_ids=family.source_ids,
                    )
                )
                for alias in (variant.id, *variant.aliases):
                    self.store.add_alias(alias, variant_id, scope=family.family)
                    aliases += 1
                evidence += self.store.add_evidence(
                    EvidenceChunk(
                        chunk_id=f"{variant_id}:summary",
                        layer=1,
                        title=f"{family.family} variant {variant.id}",
                        content=(
                            " ".join(variant.added_features)
                            + " "
                            + " ".join(variant.decision_changes)
                            + " "
                            + variant.impact_note
                        ),
                        source_kind="curated_seed",
                        scope=family.family,
                        review_status="active",
                        node_ids=(family_id, variant_id),
                    )
                )

        for pattern in knowledge.document.engineering_patterns:
            pattern_id = f"pattern:{pattern.id}"
            nodes += self.store.upsert_node(
                GraphNode(
                    canonical_id=pattern_id,
                    node_type="EngineeringPattern",
                    name=pattern.id,
                    summary="; ".join(pattern.screening_questions),
                    review_status="active",
                    source_ids=pattern.source_ids,
                    metadata={
                        "applies_to": pattern.applies_to,
                        "introduced_entities": pattern.introduced_entities,
                        "added_decisions": pattern.added_decisions,
                        "constraint_choices": pattern.constraint_choices,
                        "objective_choices": pattern.objective_choices,
                        "oracle_needs": pattern.oracle_needs,
                    },
                )
            )
            for alias in (pattern.id, *pattern.aliases):
                self.store.add_alias(alias, pattern_id)
                aliases += 1
            for family in pattern.applies_to:
                edges += self.store.upsert_edge(
                    GraphEdge(
                        edge_id=_edge_id(
                            pattern_id, "applies_when", f"family:{family}"
                        ),
                        source_id=pattern_id,
                        relation="applies_when",
                        target_id=f"family:{family}",
                        review_status="active",
                        source_ids=pattern.source_ids,
                    )
                )
            evidence += self.store.add_evidence(
                EvidenceChunk(
                    chunk_id=f"{pattern_id}:summary",
                    layer=1,
                    title=f"Engineering pattern {pattern.id}",
                    content=" ".join(
                        (
                            *pattern.introduced_entities,
                            *pattern.added_decisions,
                            *pattern.constraint_choices,
                            *pattern.objective_choices,
                            *pattern.oracle_needs,
                            *pattern.screening_questions,
                        )
                    ),
                    source_kind="curated_seed",
                    review_status="active",
                    node_ids=(pattern_id,),
                    metadata={"source_ids": pattern.source_ids},
                )
            )

        objective_id = "objective:makespan"
        nodes += self.store.upsert_node(
            GraphNode(
                canonical_id=objective_id,
                node_type="Objective",
                name="makespan",
                summary="Maximum completion time Cmax; minimized by the current mechanism library.",
                review_status="active",
            )
        )
        self.store.add_alias("Cmax", objective_id)
        self.store.add_alias("makespan", objective_id)
        self.store.add_alias("最大完工时间", objective_id)
        aliases += 3

        for item in mechanisms["mechanisms"]:
            mechanism_id = f"mechanism:{item['id']}"
            nodes += self.store.upsert_node(
                GraphNode(
                    canonical_id=mechanism_id,
                    node_type="MechanismTarget",
                    name=item["name"],
                    summary=item["description"],
                    review_status="active",
                    source_ids=tuple(item.get("source_ids", ())),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"id", "name", "description", "source_ids"}
                    },
                )
            )
            for alias in (item["id"], item["name"], *item.get("aliases", ())):
                self.store.add_alias(alias, mechanism_id)
                aliases += 1
            edges += self.store.upsert_edge(
                GraphEdge(
                    edge_id=_edge_id(mechanism_id, "affects", objective_id),
                    source_id=mechanism_id,
                    relation="affects",
                    target_id=objective_id,
                    review_status="active",
                    metadata={"direction": item["objective_direction"]},
                )
            )
            evidence += self.store.add_evidence(
                EvidenceChunk(
                    chunk_id=f"{mechanism_id}:definition",
                    layer=2,
                    title=item["name"],
                    content=(
                        f"{item['description']} Calculator: {item['calculator']}. "
                        f"Required data: {', '.join(item['required_data'])}."
                    ),
                    source_kind="mechanism_library",
                    review_status="active",
                    node_ids=(mechanism_id, objective_id),
                    metadata=item,
                )
            )

        return IngestionReport(
            changed_nodes=nodes,
            changed_edges=edges,
            changed_evidence=evidence,
            aliases_processed=aliases,
        )

    def ingest_external_document(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        source_kind: str = "other",
        source_uri: str | None = None,
        scope: str = "global",
        review_status: str = "proposed",
        node_ids: tuple[str, ...] = (),
        chunk_characters: int = 4000,
    ) -> IngestionReport:
        """Store immutable document chunks without promoting their claims to facts."""

        source = Path(path).expanduser().resolve()
        if source.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(source)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = source.read_text(encoding="utf-8")
        text = re.sub(r"\r\n?", "\n", text).strip()
        if not text:
            raise ValueError(f"document contains no extractable text: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_id = f"source:external:{digest[:20]}"
        document_title = title or source.stem
        changed_nodes = int(
            self.store.upsert_node(
                GraphNode(
                    canonical_id=source_id,
                    node_type="EvidenceSource",
                    name=document_title,
                    summary=(
                        "Externally supplied evidence. Claims remain proposed until "
                        "source/code/experiment review."
                    ),
                    scope=scope,
                    review_status=review_status,
                    source_ids=(digest,),
                    metadata={
                        "path": str(source),
                        "source_uri": source_uri,
                        "source_kind": source_kind,
                        "sha256": digest,
                    },
                )
            )
        )
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > chunk_characters:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        changed_evidence = 0
        linked_nodes = tuple(dict.fromkeys((source_id, *node_ids)))
        for index, content in enumerate(chunks):
            changed_evidence += self.store.add_evidence(
                EvidenceChunk(
                    chunk_id=f"external:{digest[:20]}:{index:04d}",
                    layer=3,
                    title=f"{document_title} [{index + 1}/{len(chunks)}]",
                    content=content,
                    source_kind=source_kind,
                    source_uri=source_uri or str(source),
                    scope=scope,
                    review_status=review_status,
                    node_ids=linked_nodes,
                    metadata={"document_sha256": digest, "chunk_index": index},
                )
            )
        return IngestionReport(
            changed_nodes=changed_nodes,
            changed_edges=0,
            changed_evidence=changed_evidence,
            aliases_processed=0,
        )

    def retrieve(
        self,
        query: str,
        *,
        family_hint: str | None = None,
        layers: tuple[int, ...] = (0, 1, 2, 3, 4),
        graph_depth: int = 1,
        limit: int = 12,
    ) -> RetrievalResult:
        resolved = list(self.store.resolve_alias(query, scope=family_hint))
        if family_hint:
            resolved.extend(self.store.resolve_alias(family_hint))
        expanded = self.store.neighbors(
            tuple(dict.fromkeys(resolved)),
            depth=graph_depth,
        )
        rows = self.store.search_evidence(
            query,
            layers=layers,
            scope=family_hint,
            node_ids=expanded,
            limit=limit,
        )
        if not rows and expanded:
            rows = self.store.search_evidence(
                query,
                layers=layers,
                scope=family_hint,
                limit=limit,
            )
        evidence = tuple(
            RetrievalEvidence(
                chunk=chunk,
                lexical_rank=rank,
                layer_priority=chunk.layer,
            )
            for chunk, rank in rows
        )
        nodes = self.store.get_nodes(expanded)
        conflicts = tuple(
            node.canonical_id
            for node in nodes
            if node.review_status in {"rejected", "superseded"}
        )
        return RetrievalResult(
            query=query,
            resolved_ids=tuple(dict.fromkeys(resolved)),
            expanded_ids=expanded,
            nodes=nodes,
            evidence=evidence,
            conflicts=conflicts,
        )
