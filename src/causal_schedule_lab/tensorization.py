"""Stable conversion of Pydantic scheduling graphs into torch tensors."""

from __future__ import annotations

from typing import Iterable

import torch

from .learning import GraphBatch
from .models import CausalInterventionPoint, EdgeType, NodeType, SchedulingGraph


NUMERIC_FEATURES = (
    "start",
    "end",
    "duration",
    "wait",
    "slack",
    "resource_idle_before",
    "resource_utilization",
    "criticality",
    "modifiability",
    "risk",
    "cost",
)

NODE_TYPE_INDEX = {item: index for index, item in enumerate(NodeType)}
EDGE_TYPE_INDEX = {item: index for index, item in enumerate(EdgeType)}


def tensorize_graph(
    graph: SchedulingGraph,
    candidates: Iterable[CausalInterventionPoint],
) -> GraphBatch:
    candidates = tuple(candidates)
    node_index = {node.id: index for index, node in enumerate(graph.nodes)}
    features = torch.tensor(
        [
            [
                float(node.features.get(name, 0.0) or 0.0)
                if not isinstance(node.features.get(name), str)
                else 0.0
                for name in NUMERIC_FEATURES
            ]
            for node in graph.nodes
        ],
        dtype=torch.float32,
    )
    node_types = torch.tensor(
        [NODE_TYPE_INDEX[node.type] for node in graph.nodes],
        dtype=torch.long,
    )
    edge_pairs = [
        (node_index[edge.source], node_index[edge.target])
        for edge in graph.edges
        if edge.source in node_index and edge.target in node_index
    ]
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).T.contiguous()
        if edge_pairs
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_types = torch.tensor(
        [
            EDGE_TYPE_INDEX[edge.type]
            for edge in graph.edges
            if edge.source in node_index and edge.target in node_index
        ],
        dtype=torch.long,
    )
    candidate_nodes = []
    candidate_batch = []
    for index, candidate in enumerate(candidates):
        ids = tuple(
            dict.fromkeys(
                (
                    *candidate.diagnostic.location,
                    candidate.responsible.operation_id,
                    *candidate.causal_path.nodes,
                    *candidate.closure.operation_ids,
                )
            )
        )
        members = [node_index[item] for item in ids if item in node_index]
        if not members:
            members = [0]
        candidate_nodes.extend(members)
        candidate_batch.extend([index] * len(members))
    return GraphBatch(
        node_features=features,
        node_types=node_types,
        edge_index=edge_index,
        edge_types=edge_types,
        candidate_nodes=torch.tensor(candidate_nodes, dtype=torch.long),
        candidate_batch=torch.tensor(candidate_batch, dtype=torch.long),
        graph_batch=torch.zeros(len(graph.nodes), dtype=torch.long),
    )
