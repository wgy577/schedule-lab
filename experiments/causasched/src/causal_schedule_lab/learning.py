"""Relation-aware CIP, path, closure, risk and ranking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphBatch:
    node_features: Tensor
    node_types: Tensor
    edge_index: Tensor
    edge_types: Tensor
    candidate_nodes: Tensor
    candidate_batch: Tensor
    graph_batch: Tensor


@dataclass(frozen=True)
class MultiTaskTargets:
    improvement: Tensor
    validity: Tensor
    log_cost: Tensor
    risk: Tensor
    closure_membership: Tensor
    path_membership: Tensor
    rank_pairs: Tensor | None = None


class RelationalEncoder(nn.Module):
    def __init__(
        self,
        *,
        numeric_dim: int,
        node_type_count: int,
        edge_type_count: int,
        hidden_dim: int = 96,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.node_type_embedding = nn.Embedding(node_type_count, hidden_dim)
        self.input_projection = nn.Linear(numeric_dim, hidden_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        )
        self.relations = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim, bias=False)
                for _ in range(edge_type_count)
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(self, batch: GraphBatch) -> Tensor:
        hidden = self.input_projection(batch.node_features)
        hidden = hidden + self.node_type_embedding(batch.node_types)
        source, target = batch.edge_index
        for self_layer, relation_layers, norm in zip(
            self.self_layers,
            self.relations,
            self.norms,
        ):
            messages = torch.zeros_like(hidden)
            degree = torch.zeros(
                hidden.shape[0],
                1,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            for relation_id, transform in enumerate(relation_layers):
                mask = batch.edge_types == relation_id
                if not torch.any(mask):
                    continue
                relation_source = source[mask]
                relation_target = target[mask]
                transformed = transform(hidden[relation_source])
                messages.index_add_(0, relation_target, transformed)
                degree.index_add_(
                    0,
                    relation_target,
                    torch.ones(
                        relation_target.shape[0],
                        1,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                )
            hidden = norm(
                hidden + F.gelu(self_layer(hidden) + messages / degree.clamp_min(1.0))
            )
        return hidden


class CausalCoreMultiTaskModel(nn.Module):
    """Document heads: gain, validity, cost, risk, rank, closure and path."""

    def __init__(
        self,
        *,
        numeric_dim: int,
        node_type_count: int,
        edge_type_count: int,
        risk_classes: int,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.encoder = RelationalEncoder(
            numeric_dim=numeric_dim,
            node_type_count=node_type_count,
            edge_type_count=edge_type_count,
            hidden_dim=hidden_dim,
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.improvement_head = nn.Linear(hidden_dim, 1)
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, risk_classes)
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.closure_head = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.path_head = nn.Bilinear(hidden_dim, hidden_dim, 1)

    @staticmethod
    def _mean_pool(values: Tensor, index: Tensor, count: int) -> Tensor:
        result = torch.zeros(count, values.shape[-1], device=values.device)
        cardinality = torch.zeros(count, 1, device=values.device)
        result.index_add_(0, index, values)
        cardinality.index_add_(
            0,
            index,
            torch.ones(index.shape[0], 1, device=values.device),
        )
        return result / cardinality.clamp_min(1.0)

    def forward(self, batch: GraphBatch) -> dict[str, Tensor]:
        nodes = self.encoder(batch)
        candidate_count = int(batch.candidate_batch.max().item()) + 1
        graph_count = int(batch.graph_batch.max().item()) + 1
        graph_embedding = self._mean_pool(nodes, batch.graph_batch, graph_count)
        candidate_embedding = self._mean_pool(
            nodes[batch.candidate_nodes],
            batch.candidate_batch,
            candidate_count,
        )
        candidate_graph_ids = torch.zeros(
            candidate_count,
            dtype=torch.long,
            device=nodes.device,
        )
        for candidate_id in range(candidate_count):
            member = batch.candidate_nodes[batch.candidate_batch == candidate_id][0]
            candidate_graph_ids[candidate_id] = batch.graph_batch[member]
        candidate_graph = graph_embedding[candidate_graph_ids]
        candidate = self.candidate_projection(
            torch.cat([candidate_embedding, candidate_graph], dim=-1)
        )
        expanded_nodes = nodes.unsqueeze(0).expand(candidate_count, -1, -1)
        expanded_candidates = candidate.unsqueeze(1).expand(-1, nodes.shape[0], -1)
        return {
            "improvement": self.improvement_head(candidate).squeeze(-1),
            "validity_logit": self.validity_head(candidate).squeeze(-1),
            "log_cost": self.cost_head(candidate).squeeze(-1),
            "risk_logits": self.risk_head(candidate),
            "rank_score": self.rank_head(candidate).squeeze(-1),
            "closure_logits": self.closure_head(
                expanded_nodes.reshape(-1, nodes.shape[-1]),
                expanded_candidates.reshape(-1, candidate.shape[-1]),
            ).reshape(candidate_count, nodes.shape[0]),
            "path_logits": self.path_head(
                expanded_nodes.reshape(-1, nodes.shape[-1]),
                expanded_candidates.reshape(-1, candidate.shape[-1]),
            ).reshape(candidate_count, nodes.shape[0]),
        }


class CIPFeatureMLP(nn.Module):
    """Graph-free ablation/baseline with the same scalar prediction heads."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 96,
        risk_classes: int,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.improvement_head = nn.Linear(hidden_dim, 1)
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, risk_classes)
        self.rank_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        hidden = self.body(features)
        return {
            "improvement": self.improvement_head(hidden).squeeze(-1),
            "validity_logit": self.validity_head(hidden).squeeze(-1),
            "log_cost": self.cost_head(hidden).squeeze(-1),
            "risk_logits": self.risk_head(hidden),
            "rank_score": self.rank_head(hidden).squeeze(-1),
        }


def multitask_loss(
    prediction: dict[str, Tensor],
    target: MultiTaskTargets,
    *,
    sparse_closure_weight: float = 0.02,
) -> tuple[Tensor, dict[str, float]]:
    losses = {
        "improvement": F.smooth_l1_loss(
            prediction["improvement"], target.improvement
        ),
        "validity": F.binary_cross_entropy_with_logits(
            prediction["validity_logit"], target.validity
        ),
        "cost": F.smooth_l1_loss(prediction["log_cost"], target.log_cost),
        "risk": F.binary_cross_entropy_with_logits(
            prediction["risk_logits"],
            target.risk.float(),
        ),
        "closure": F.binary_cross_entropy_with_logits(
            prediction["closure_logits"], target.closure_membership
        ),
        "path": F.binary_cross_entropy_with_logits(
            prediction["path_logits"], target.path_membership
        ),
        "closure_sparse": torch.sigmoid(prediction["closure_logits"]).mean(),
    }
    if target.rank_pairs is not None and target.rank_pairs.numel():
        left = prediction["rank_score"][target.rank_pairs[:, 0]]
        right = prediction["rank_score"][target.rank_pairs[:, 1]]
        losses["ranking"] = -F.logsigmoid(left - right).mean()
    else:
        # Multiply by 0 before summing so masked (finfo.min) logits cannot
        # overflow the sum to +/-inf (inf * 0.0 == NaN under IEEE 754).
        losses["ranking"] = (prediction["rank_score"] * 0.0).sum()
    total = (
        losses["improvement"]
        + losses["validity"]
        + losses["cost"]
        + losses["risk"]
        + losses["closure"]
        + losses["path"]
        + losses["ranking"]
        + sparse_closure_weight * losses["closure_sparse"]
    )
    return total, {key: float(value.detach()) for key, value in losses.items()}


def pairwise_rank_pairs(
    gains: Iterable[float],
    *,
    margin: float = 1e-9,
) -> Tensor:
    values = list(gains)
    pairs = [
        (left, right)
        for left, left_gain in enumerate(values)
        for right, right_gain in enumerate(values)
        if left_gain > right_gain + margin
    ]
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
