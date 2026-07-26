"""Serializable models for a project-level scheduling constraint graph."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SemanticGraphNode(FrozenModel):
    id: str
    node_type: str
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SemanticGraphEdge(FrozenModel):
    source: str
    relation: str
    target: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ProjectConstraintGraph(FrozenModel):
    """Evidence-linked graph richer than a classical disjunctive graph."""

    schema_version: str = "1.0"
    project_id: str
    nodes: tuple[SemanticGraphNode, ...]
    edges: tuple[SemanticGraphEdge, ...]
    views: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
