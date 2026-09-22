"""SG-SCT v1 state encoder and reverse causal block tracer.

This module implements the trainable M1/STSE and M2/RCBT core described by
the project design notes.  It deliberately does not execute interventions,
call an Oracle, use ESWA/PageRank as inference, or claim trained/identified
causal effects.  Graph attention is edge-sparse; no dense N x N graph
attention matrix is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EdgeRole(IntEnum):
    CONTEXT = 0
    FEASIBILITY = 1
    CAUSAL_HARD = 2
    CAUSAL_SOFT = 3
    CAUSAL_FORBIDDEN = 4
    CAUSAL_CROSS_SYMPTOM = 5


@dataclass(frozen=True)
class SGSCTBatch:
    """Flat variable-size batch for SG-SCT.

    Node and token graph IDs must be contiguous from zero.  Gantt operation
    indices are global node indices; use -1 for an event token without a
    one-to-one operation node.  Trajectory features are already structured
    numeric summaries of state/action/reward and are limited to the latest
    eight items per graph inside the model.
    """

    node_numeric: Tensor
    node_type: Tensor
    node_graph: Tensor
    edge_index: Tensor
    edge_type: Tensor
    edge_role: Tensor
    edge_features: Tensor
    symptom_node_mask: Tensor
    candidate_node_mask: Tensor
    gantt_numeric: Tensor
    gantt_graph: Tensor
    gantt_operation_node: Tensor
    gantt_machine_index: Tensor
    gantt_job_index: Tensor
    gantt_time_index: Tensor
    trajectory_numeric: Tensor
    trajectory_graph: Tensor
    trajectory_step: Tensor
    mechanism_values: Tensor | None = None
    mechanism_value_mask: Tensor | None = None
    action_features: Tensor | None = None
    action_graph: Tensor | None = None
    action_observed_mask: Tensor | None = None
    # Optional teacher-forced root→symptom path [C, steps] (node indices, STOP
    # = node_count). When provided, the path decoder follows it instead of
    # greedy argmax so the step logits are well-defined for every ESWA hop.
    path_target: Tensor | None = None
    # Per-symptom-block structure (L4 shared-symptom connectivity). When present
    # (symptom_block_count > 0), reverse inference is run per block: each block
    # gets its own state query, root neighbourhood, anchors and root blocks.
    symptom_block_count: int = 0
    symptom_block_node_index: Tensor | None = None  # [2, M] (block, node) COO
    node_symptom_block: Tensor | None = None  # [N] node->block, -1 if none
    symptom_block_neighborhood_mask: Tensor | None = None  # [B, N]
    # v2 §4 Appearance Query: per-block primary-rule type id [B] (int index into
    # APPEARANCE_RULES) and a fixed-dim continuous feature vector [B, F_a].
    appearance_type: Tensor | None = None  # [B]
    appearance_features: Tensor | None = None  # [B, F_a]

    def validate(self) -> None:
        node_count = self.node_numeric.shape[0]
        edge_count = self.edge_index.shape[1]
        token_count = self.gantt_numeric.shape[0]
        history_count = self.trajectory_numeric.shape[0]
        if self.node_numeric.ndim != 2:
            raise ValueError("node_numeric must have shape [N,F]")
        if self.edge_index.shape != (2, edge_count):
            raise ValueError("edge_index must have shape [2,E]")
        for name, value in (
            ("node_type", self.node_type),
            ("node_graph", self.node_graph),
            ("symptom_node_mask", self.symptom_node_mask),
            ("candidate_node_mask", self.candidate_node_mask),
        ):
            if value.shape != (node_count,):
                raise ValueError(f"{name} must have shape [N]")
        for name, value in (("edge_type", self.edge_type), ("edge_role", self.edge_role)):
            if value.shape != (edge_count,):
                raise ValueError(f"{name} must have shape [E]")
        if self.edge_features.shape[0] != edge_count:
            raise ValueError("edge_features must have shape [E,F_e]")
        for name, value in (
            ("gantt_graph", self.gantt_graph),
            ("gantt_operation_node", self.gantt_operation_node),
            ("gantt_machine_index", self.gantt_machine_index),
            ("gantt_job_index", self.gantt_job_index),
            ("gantt_time_index", self.gantt_time_index),
        ):
            if value.shape != (token_count,):
                raise ValueError(f"{name} must have shape [T]")
        if self.trajectory_graph.shape != (history_count,) or self.trajectory_step.shape != (history_count,):
            raise ValueError("trajectory graph/step tensors must have shape [H]")
        if edge_count and (int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= node_count):
            raise ValueError("edge_index references a missing node")
        valid_token_nodes = self.gantt_operation_node >= 0
        if torch.any(valid_token_nodes) and int(self.gantt_operation_node[valid_token_nodes].max()) >= node_count:
            raise ValueError("gantt_operation_node references a missing node")
        if self.mechanism_values is not None and self.mechanism_values.shape[0] != node_count:
            raise ValueError("mechanism_values must have shape [N,M]")
        if self.mechanism_value_mask is not None and (
            self.mechanism_values is None
            or self.mechanism_value_mask.shape != self.mechanism_values.shape
        ):
            raise ValueError("mechanism_value_mask must match mechanism_values")
        if self.action_features is not None:
            if self.action_features.ndim != 2:
                raise ValueError("action_features must have shape [A,F_a]")
            if self.action_graph is None or self.action_graph.shape != (self.action_features.shape[0],):
                raise ValueError("action_graph must have shape [A]")
            if self.action_observed_mask is not None and self.action_observed_mask.shape != (
                self.action_features.shape[0],
            ):
                raise ValueError("action_observed_mask must have shape [A]")
        if self.symptom_block_count > 0:
            if self.symptom_block_node_index is None or self.symptom_block_node_index.ndim != 2:
                raise ValueError("symptom_block_node_index must have shape [2,M]")
            elif self.symptom_block_node_index.shape[0] != 2:
                raise ValueError("symptom_block_node_index must have shape [2,M]")
            if self.node_symptom_block is None or self.node_symptom_block.shape != (node_count,):
                raise ValueError("node_symptom_block must have shape [N]")
            if self.symptom_block_neighborhood_mask is None or self.symptom_block_neighborhood_mask.shape != (
                self.symptom_block_count,
                node_count,
            ):
                raise ValueError("symptom_block_neighborhood_mask must have shape [B,N]")
            if self.appearance_type is not None or self.appearance_features is not None:
                # v2 appearance query is validated only when present; pre-v2
                # frozen bundles (no appearance fields) run the legacy query.
                if self.appearance_type is None or self.appearance_type.shape != (self.symptom_block_count,):
                    raise ValueError("appearance_type must have shape [B]")
                if self.appearance_features is None or self.appearance_features.shape[0] != self.symptom_block_count:
                    raise ValueError("appearance_features must have [B, F_a]")


@dataclass(frozen=True)
class SGSCTOutput:
    node_embeddings: Tensor
    state_embeddings: Tensor
    history_embeddings: Tensor
    gantt_embeddings: Tensor
    m1_mask_reconstruction: Tensor
    m1_transition: Tensor | None
    m1_metric: Tensor | None
    m1_action_observed_mask: Tensor
    causal_gate_logits: Tensor
    causal_gate_values: Tensor
    mechanism_mu: Tensor
    mechanism_scale: Tensor
    mechanism_df: Tensor
    standardized_residual: Tensor
    mechanism_nll: Tensor
    epistemic_uncertainty: Tensor
    anomaly_embeddings: Tensor
    reverse_embeddings: Tensor
    anchor_logits: Tensor
    anchor_node_index: Tensor
    anchor_scores: Tensor
    root_block_logits: Tensor
    root_type_logits: Tensor
    candidate_confidence: Tensor
    block_pointer_logits: Tensor
    block_node_index: Tensor
    block_stopped: Tensor
    path_pointer_logits: Tensor
    path_node_index: Tensor
    path_stopped: Tensor
    top_root_node_index: Tensor
    top_root_scores: Tensor
    # Per-symptom-block reverse-inference outputs (present when block_count>0).
    symptom_block_state: Tensor | None = None  # [B, H]
    per_block_root_logits: Tensor | None = None  # [B, N]
    block_row_symptom_index: Tensor | None = None  # [C] root row -> symptom block
    # v2 §9/§10 CE heads: predicted appearance causal effect of each node/each
    # edge on the target appearance, in [0, 1] after sigmoid.  Regressed against
    # the solver-counterfactual CE_A labels (§11) produced by sg_sct_causal_probe.
    node_ce_scores: Tensor | None = None  # [N]
    edge_ce_scores: Tensor | None = None  # [E]
    symptom_block_count: int = 0
    identified: bool = False
    training_status: str = "code_implemented_not_trained"


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    if index.numel():
        output.index_add_(0, index, values)
    return output


def _scatter_mean(values: Tensor, index: Tensor, size: int) -> Tensor:
    total = _scatter_sum(values, index, size)
    count = values.new_zeros(size)
    if index.numel():
        count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return total / count.clamp_min(1.0).view(size, *([1] * (values.ndim - 1)))


def _attention_pool(
    values: Tensor, index: Tensor, size: int, query: nn.Module
) -> Tensor:
    """v2 §4.1 h_A: attention-pool member node embeddings per appearance block.

    ``query`` is a shared linear projection producing a scalar gate per member;
    ``_segment_softmax`` normalises within each block so the pooled vector is a
    convex combination of the block's member node embeddings.
    """
    scores = query(values).squeeze(-1)
    if index.numel():
        maxima = scores.new_full((size,), -torch.inf)
        maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        stabilized = torch.exp(scores - maxima[index])
        denominator = scores.new_zeros(size)
        denominator.index_add_(0, index, stabilized)
        weights = stabilized / denominator[index].clamp_min(1e-12)
    else:
        weights = scores.new_zeros(0)
    return _scatter_sum(values * weights[index].unsqueeze(-1), index, size)


def _segment_softmax(scores: Tensor, index: Tensor, size: int) -> Tensor:

    if not index.numel():
        return scores
    expanded = index.view(-1, 1).expand_as(scores)
    maxima = scores.new_full((size, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    stabilized = torch.exp(scores - maxima[index])
    denominator = scores.new_zeros((size, scores.shape[1]))
    denominator.index_add_(0, index, stabilized)
    return stabilized / denominator[index].clamp_min(1e-12)


def _graph_mean(values: Tensor, graph_index: Tensor, graph_count: int) -> Tensor:
    return _scatter_mean(values, graph_index, graph_count)


class TypeSpecificProjection(nn.Module):
    def __init__(self, numeric_dim: int, node_type_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_type_count = node_type_count
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(numeric_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(node_type_count)
        )
        self.type_embedding = nn.Embedding(node_type_count, hidden_dim)

    def forward(self, numeric: Tensor, node_type: Tensor) -> Tensor:
        if torch.any((node_type < 0) | (node_type >= self.node_type_count)):
            raise ValueError("node_type is outside the configured vocabulary")
        output = numeric.new_zeros((numeric.shape[0], self.type_embedding.embedding_dim))
        for type_id, projection in enumerate(self.projections):
            selected = node_type == type_id
            if torch.any(selected):
                output[selected] = projection(numeric[selected])
        return output + self.type_embedding(node_type)


class SparseRelationalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_type_count: int,
        edge_feature_dim: int,
        *,
        heads: int,
        dropout: float,
        use_soft_gate: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.use_soft_gate = use_soft_gate
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_type_key = nn.Embedding(edge_type_count, hidden_dim)
        self.edge_type_value = nn.Embedding(edge_type_count, hidden_dim)
        self.edge_bias = nn.Linear(edge_feature_dim, heads, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if use_soft_gate
            else None
        )

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_features: Tensor,
        edge_mask: Tensor,
        *,
        hard_edge_mask: Tensor | None = None,
        soft_edge_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_count = nodes.shape[0]
        source_all, target_all = edge_index
        chosen = torch.nonzero(edge_mask, as_tuple=False).flatten()
        full_gate_logits = nodes.new_full((edge_index.shape[1],), -20.0)
        full_gate_values = nodes.new_zeros(edge_index.shape[1])
        if not chosen.numel():
            updated = self.norm2(nodes + self.dropout(self.ffn(self.norm1(nodes))))
            return updated, full_gate_logits, full_gate_values
        source = source_all[chosen]
        target = target_all[chosen]
        relation = edge_type[chosen]
        features = edge_features[chosen]
        query = self.query(nodes[target]).view(-1, self.heads, self.head_dim)
        key = (
            self.key(nodes[source]) + self.edge_type_key(relation)
        ).view(-1, self.heads, self.head_dim)
        value = (
            self.value(nodes[source]) + self.edge_type_value(relation)
        ).view(-1, self.heads, self.head_dim)
        scores = (query * key).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(features)

        selected_gate_logits = nodes.new_full((chosen.shape[0],), 20.0)
        selected_gate_values = nodes.new_ones(chosen.shape[0])
        if self.use_soft_gate and self.gate is not None and soft_edge_mask is not None:
            selected_soft = soft_edge_mask[chosen]
            if torch.any(selected_soft):
                logits = self.gate(
                    torch.cat([nodes[source[selected_soft]], nodes[target[selected_soft]], features[selected_soft]], dim=-1)
                ).squeeze(-1)
                selected_gate_logits[selected_soft] = logits
                selected_gate_values[selected_soft] = torch.sigmoid(logits)
            if hard_edge_mask is not None:
                selected_hard = hard_edge_mask[chosen]
                selected_gate_logits[selected_hard] = 20.0
                selected_gate_values[selected_hard] = 1.0
            scores = scores + torch.log(selected_gate_values.clamp_min(1e-8)).unsqueeze(-1)

        attention = _segment_softmax(scores, target, node_count)
        messages = value * attention.unsqueeze(-1)
        aggregated = _scatter_sum(messages, target, node_count).reshape(node_count, self.hidden_dim)
        hidden = self.norm1(nodes + self.dropout(self.output(aggregated)))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        full_gate_logits[chosen] = selected_gate_logits
        full_gate_values[chosen] = selected_gate_values
        return hidden, full_gate_logits, full_gate_values


class LocalGanttEncoder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        hidden_dim: int,
        *,
        heads: int,
        layers: int,
        machine_vocab: int,
        job_vocab: int,
        time_vocab: int,
        dropout: float,
        use_machine_identity_embedding: bool = True,
        use_identity_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.numeric = nn.Linear(numeric_dim, hidden_dim)
        self.use_machine_identity_embedding = use_machine_identity_embedding
        self.use_identity_embeddings = use_identity_embeddings
        if use_identity_embeddings:
            self.machine = nn.Embedding(machine_vocab, hidden_dim)
            self.job = nn.Embedding(job_vocab, hidden_dim)
            self.time = nn.Embedding(time_vocab, hidden_dim)
        else:
            # Shared (cross-instance) mode: absolute machine/job/time IDs carry no
            # cross-instance semantics (an ID is an arbitrary enumeration index), so
            # their vocab-sized embeddings would (a) break parameter sharing (vocab
            # varies per instance) and (b) let the model memorize instance identity.
            # Structural + numeric identity already lives in gantt_numeric
            # (machine_position, operation.index, normalized times).
            self.machine = None
            self.job = None
            self.time = None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _checked_embedding(embedding: nn.Embedding, index: Tensor, name: str) -> Tensor:
        if index.numel() and (
            int(index.min()) < 0 or int(index.max()) >= embedding.num_embeddings
        ):
            raise ValueError(
                f"{name} index is outside [0,{embedding.num_embeddings}); "
                "increase the corresponding SG-SCT vocabulary"
            )
        return embedding(index)

    def forward(self, batch: SGSCTBatch, graph_count: int) -> Tensor:
        token_count = batch.gantt_numeric.shape[0]
        if not token_count:
            return batch.node_numeric.new_zeros((0, self.numeric.out_features))
        hidden = self.numeric(batch.gantt_numeric)
        if self.use_identity_embeddings:
            if self.use_machine_identity_embedding:
                hidden = hidden + self._checked_embedding(
                    self.machine, batch.gantt_machine_index, "machine"
                )
            hidden = (
                hidden
                + self._checked_embedding(self.job, batch.gantt_job_index, "job")
                + self._checked_embedding(self.time, batch.gantt_time_index, "time")
            )
        output = torch.zeros_like(hidden)
        for graph_id in range(graph_count):
            selected = torch.nonzero(batch.gantt_graph == graph_id, as_tuple=False).flatten()
            if selected.numel():
                output[selected] = self.encoder(hidden[selected].unsqueeze(0)).squeeze(0)
        return self.norm(output)


class CausalTrajectoryEncoder(nn.Module):
    def __init__(self, numeric_dim: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.numeric = nn.Linear(numeric_dim, hidden_dim)
        self.round_embedding = nn.Embedding(8, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: SGSCTBatch, state: Tensor, graph_count: int) -> Tensor:
        result = state.new_zeros((graph_count, state.shape[-1]))
        for graph_id in range(graph_count):
            selected = torch.nonzero(batch.trajectory_graph == graph_id, as_tuple=False).flatten()
            if not selected.numel():
                continue
            order = torch.argsort(batch.trajectory_step[selected], stable=True)
            selected = selected[order][-8:]
            length = selected.shape[0]
            token = self.numeric(batch.trajectory_numeric[selected])
            token = token + self.round_embedding(torch.arange(length, device=token.device))
            token = token + state[graph_id].unsqueeze(0)
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=token.device), diagonal=1
            )
            encoded = self.encoder(token.unsqueeze(0), mask=mask).squeeze(0)
            result[graph_id] = encoded[-1]
        return self.norm(result)


class StudentTMechanismEnsemble(nn.Module):
    def __init__(self, hidden_dim: int, mechanism_dim: int, heads: int = 3) -> None:
        super().__init__()
        self.mechanism_dim = mechanism_dim
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, mechanism_dim * 3),
            )
            for _ in range(heads)
        )

    def forward(self, nodes: Tensor, parent_context: Tensor, observed: Tensor) -> tuple[Tensor, ...]:
        raw = torch.stack([head(torch.cat([nodes, parent_context], dim=-1)) for head in self.heads])
        mu, raw_scale, raw_df = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + 1e-4
        degrees = F.softplus(raw_df) + 2.01
        standardized = (observed.unsqueeze(0) - mu) / scale
        distribution = torch.distributions.StudentT(degrees, loc=mu, scale=scale)
        nll = -distribution.log_prob(observed.unsqueeze(0))
        epistemic = mu.var(dim=0, unbiased=False)
        return mu, scale, degrees, standardized, nll, epistemic


class SGSCTRootCauseModel(nn.Module):
    """Variable-size SG-SCT M1+M2 core with sparse graph propagation."""

    model_schema_version = "sg-sct-m1-m2-v3"

    def __init__(
        self,
        *,
        node_numeric_dim: int,
        node_type_count: int,
        edge_type_count: int,
        edge_feature_dim: int,
        gantt_numeric_dim: int,
        trajectory_numeric_dim: int,
        mechanism_dim: int,
        root_type_count: int,
        action_numeric_dim: int = 0,
        appearance_type_count: int = 10,
        appearance_feature_dim: int = 17,
        hidden_dim: int = 256,
        heads: int = 8,
        machine_vocab: int = 4096,
        job_vocab: int = 16384,
        time_vocab: int = 4096,
        dropout: float = 0.1,
        max_anchors: int = 8,
        max_block_size: int = 6,
        max_path_length: int = 12,
        top_root_candidates: int = 5,
        use_machine_identity_embedding: bool = True,
        use_identity_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if max_block_size < 1 or max_path_length < 1:
            raise ValueError("block and path limits must be positive")
        self.hidden_dim = hidden_dim
        self.mechanism_dim = mechanism_dim
        self.max_anchors = max_anchors
        self.max_block_size = max_block_size
        self.max_path_length = max_path_length
        self.top_root_candidates = top_root_candidates
        self.node_projection = TypeSpecificProjection(node_numeric_dim, node_type_count, hidden_dim)
        self.mask_reconstruction_head = nn.Linear(hidden_dim, node_numeric_dim)
        self.action_numeric_dim = action_numeric_dim
        self.action_projection = (
            nn.Sequential(nn.Linear(action_numeric_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            if action_numeric_dim > 0
            else None
        )
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        self.context_layers = nn.ModuleList(
            SparseRelationalAttention(
                hidden_dim, edge_type_count, edge_feature_dim,
                heads=heads, dropout=dropout,
            )
            for _ in range(3)
        )
        self.causal_layers = nn.ModuleList(
            SparseRelationalAttention(
                hidden_dim, edge_type_count, edge_feature_dim,
                heads=heads, dropout=dropout, use_soft_gate=True,
            )
            for _ in range(3)
        )
        self.gantt_encoder = LocalGanttEncoder(
            gantt_numeric_dim, hidden_dim, heads=heads, layers=4,
            machine_vocab=machine_vocab, job_vocab=job_vocab,
            time_vocab=time_vocab, dropout=dropout,
            use_machine_identity_embedding=use_machine_identity_embedding,
            use_identity_embeddings=use_identity_embeddings,
        )
        self.token_from_graph = nn.Linear(hidden_dim, hidden_dim)
        self.node_from_token = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.state_query = nn.Linear(hidden_dim, hidden_dim)
        # v2 §4 Appearance Query: q_A = MLP_A[ h_A, e_A, x_A ].
        #   h_A = attention-pool of the block's member nodes (graph side),
        #   e_A = AppearanceTypeEmbedding(primary rule),
        #   x_A = AppearanceFeatureEncoder(continuous feature vector).
        self.appearance_type_count = appearance_type_count
        self.appearance_feature_dim = appearance_feature_dim
        self.appearance_type_embedding = nn.Embedding(appearance_type_count, hidden_dim)
        self.appearance_feature_encoder = nn.Sequential(
            nn.Linear(appearance_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.appearance_query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.block_attention = nn.Linear(hidden_dim, 1, bias=False)
        # v2 §9/§10 CausalEffectHeads.  Node CE reads the per-node reverse
        # embedding; edge CE reads the concatenation of its two endpoint reverse
        # embeddings.  Sigmoid keeps the output in the supervised CE_A ∈ [0,1].
        self.node_ce_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.edge_ce_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.trajectory_encoder = CausalTrajectoryEncoder(
            trajectory_numeric_dim, hidden_dim, heads, dropout
        )
        self.mechanism = StudentTMechanismEnsemble(hidden_dim, mechanism_dim, heads=3)
        self.anomaly_projection = nn.Sequential(
            nn.Linear(hidden_dim + mechanism_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.reverse_layers = nn.ModuleList(
            SparseRelationalAttention(
                hidden_dim, edge_type_count, edge_feature_dim,
                heads=heads, dropout=dropout,
            )
            for _ in range(4)
        )
        self.reverse_query = nn.Linear(hidden_dim, hidden_dim)
        # v2 §13: root-anchor scoring is CE-conditioned.  The anchor head takes
        # the predicted appearance causal effect of each node (projected to the
        # node dimension) as an extra input, so the model learns to prefer
        # high-CE nodes as root causes.
        self.ce_projection = nn.Linear(1, hidden_dim)
        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.root_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, root_type_count)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.block_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.block_context = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.block_stop = nn.Linear(hidden_dim * 2, 1)
        # v2 §14-15: a learnable scalar gate folds each node's CE into the
        # block/path pointer scores, so the reconstructed root block and causal
        # path are CE-grounded without changing the decoder input dims.
        self.ce_scale = nn.Parameter(torch.zeros(1))
        self.path_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.path_context = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.path_stop = nn.Linear(hidden_dim * 3, 1)

    @staticmethod
    def _graph_count(batch: SGSCTBatch) -> int:
        candidates = [
            int(batch.node_graph.max()) + 1 if batch.node_graph.numel() else 0,
            int(batch.gantt_graph.max()) + 1 if batch.gantt_graph.numel() else 0,
            int(batch.trajectory_graph.max()) + 1 if batch.trajectory_graph.numel() else 0,
        ]
        return max(candidates, default=0)

    @staticmethod
    def _masked_graph_mean(values: Tensor, graph: Tensor, mask: Tensor, graph_count: int) -> Tensor:
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        fallback = _graph_mean(values, graph, graph_count)
        if not selected.numel():
            return fallback
        selected_mean = _scatter_mean(values[selected], graph[selected], graph_count)
        counts = values.new_zeros(graph_count)
        counts.index_add_(0, graph[selected], torch.ones_like(selected, dtype=values.dtype))
        return torch.where(counts.unsqueeze(-1) > 0, selected_mean, fallback)

    def _align_graph_and_gantt(
        self, nodes: Tensor, tokens: Tensor, batch: SGSCTBatch, graph_count: int
    ) -> tuple[Tensor, Tensor]:
        node_count = nodes.shape[0]
        valid = batch.gantt_operation_node >= 0
        token_to_node = nodes.new_zeros((node_count, self.hidden_dim))
        token_count = nodes.new_zeros(node_count)
        if torch.any(valid):
            mapping = batch.gantt_operation_node[valid]
            token_to_node.index_add_(0, mapping, tokens[valid])
            token_count.index_add_(0, mapping, torch.ones_like(mapping, dtype=nodes.dtype))
        token_to_node = token_to_node / token_count.clamp_min(1.0).unsqueeze(-1)
        graph_summary = _graph_mean(nodes, batch.node_graph, graph_count)
        aligned_nodes = graph_summary[batch.gantt_graph] if tokens.shape[0] else tokens
        if torch.any(valid):
            aligned_nodes = aligned_nodes.clone()
            aligned_nodes[valid] = nodes[batch.gantt_operation_node[valid]]
        fused_tokens = tokens + self.token_from_graph(aligned_nodes)
        causal = nodes
        context = nodes + self.node_from_token(token_to_node)
        gate = self.fusion_gate(torch.cat([causal, context, token_to_node], dim=-1))
        fused_nodes = self.fusion_norm(gate * causal + (1.0 - gate) * context)
        return fused_nodes, fused_tokens

    def _parent_context(self, nodes: Tensor, batch: SGSCTBatch, causal_mask: Tensor) -> Tensor:
        source, target = batch.edge_index
        chosen = torch.nonzero(causal_mask, as_tuple=False).flatten()
        if not chosen.numel():
            return torch.zeros_like(nodes)
        return _scatter_mean(nodes[source[chosen]], target[chosen], nodes.shape[0])

    def _top_anchors(
        self, logits: Tensor, batch: SGSCTBatch, graph_count: int, eligible_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        indices = torch.full(
            (graph_count, self.max_anchors), -1, dtype=torch.long, device=logits.device
        )
        scores = logits.new_full((graph_count, self.max_anchors), -torch.inf)
        for graph_id in range(graph_count):
            eligible = torch.nonzero(
                (batch.node_graph == graph_id) & eligible_mask,
                as_tuple=False,
            ).flatten()
            if not eligible.numel():
                continue
            count = min(self.max_anchors, eligible.shape[0])
            values, local = torch.topk(logits[eligible], count)
            indices[graph_id, :count] = eligible[local]
            scores[graph_id, :count] = values
        return indices, scores

    def _top_anchors_per_block(
        self, logits: Tensor, batch: SGSCTBatch, block_count: int, eligible_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Per-symptom-block greedy top-K anchors.

        ``eligible_mask`` has shape [B, N]; each block selects its own top
        anchors within its neighbourhood mask.  Returns ``[B, max_anchors]``.
        """

        indices = torch.full(
            (block_count, self.max_anchors), -1, dtype=torch.long, device=logits.device
        )
        scores = logits.new_full((block_count, self.max_anchors), -torch.inf)
        for block_id in range(block_count):
            eligible = torch.nonzero(eligible_mask[block_id], as_tuple=False).flatten()
            if not eligible.numel():
                continue
            count = min(self.max_anchors, eligible.shape[0])
            values, local = torch.topk(logits[eligible], count)
            indices[block_id, :count] = eligible[local]
            scores[block_id, :count] = values
        return indices, scores

    def _block_decode(
        self,
        nodes: Tensor,
        query_by_node: Tensor,
        anchors: Tensor,
        batch: SGSCTBatch,
        legal_mask: Tensor,
        node_ce: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flat_anchor = anchors.flatten()
        active_anchor = torch.nonzero(flat_anchor >= 0, as_tuple=False).flatten()
        candidate_count = flat_anchor.shape[0]
        per_block = legal_mask.ndim == 2
        node_count = nodes.shape[0]
        # Per-step logits are built with torch.cat so every step keeps its
        # gradient through node_score/stop; a pre-allocated buffer would detach.
        step_logits_buffer: list[Tensor | None] = [
            None
        ] * (candidate_count * self.max_block_size)
        selected_nodes = torch.full(
            (candidate_count, self.max_block_size), -1, dtype=torch.long, device=nodes.device
        )
        stopped = torch.ones(candidate_count, dtype=torch.bool, device=nodes.device)
        source, target = batch.edge_index
        usable_edge = batch.edge_role != int(EdgeRole.CAUSAL_FORBIDDEN)
        for candidate_id in active_anchor.tolist():
            anchor = int(flat_anchor[candidate_id])
            graph_id = int(batch.node_graph[anchor])
            block_id = candidate_id // self.max_anchors if per_block else None
            base_legal = legal_mask[block_id] if block_id is not None else legal_mask
            selected = [anchor]
            selected_nodes[candidate_id, 0] = anchor
            stopped[candidate_id] = False
            for step in range(self.max_block_size - 1):
                pool = nodes[selected].mean(0)
                query = query_by_node[anchor]
                context = torch.cat([pool, query])
                node_score = (self.block_node(nodes) * self.block_context(context)).sum(-1) / math.sqrt(self.hidden_dim)
                if node_ce is not None:
                    node_score = node_score + self.ce_scale * node_ce
                connected = torch.zeros(node_count, dtype=torch.bool, device=nodes.device)
                for current in selected:
                    # Successors (edges leaving current) and predecessors (edges
                    # entering current); the block expands along the graph.
                    connected[target[(source == current) & usable_edge]] = True
                    connected[source[(target == current) & usable_edge]] = True
                legal = (
                    connected
                    & base_legal
                    & (batch.node_graph == graph_id)
                )
                legal[selected] = False
                step_logits = torch.cat(
                    [
                        node_score.masked_fill(~legal, -torch.inf),
                        self.block_stop(context).squeeze(-1).unsqueeze(0),
                    ]
                )
                step_logits_buffer[candidate_id * self.max_block_size + step] = step_logits
                choice = int(torch.argmax(step_logits.detach()))
                if choice == node_count:
                    stopped[candidate_id] = True
                    break
                selected.append(choice)
                selected_nodes[candidate_id, len(selected) - 1] = choice
        # Stack non-None steps into [C, max_block_size, N+1]; inactive slots
        # are represented by a constant -inf vector (no gradient, no loss terms).
        filled = [
            item if item is not None else nodes.new_full((node_count + 1,), -torch.inf)
            for item in step_logits_buffer
        ]
        logits = torch.stack(filled).view(candidate_count, self.max_block_size, node_count + 1)
        return logits, selected_nodes, stopped

    @staticmethod
    def _causal_ancestor_mask(batch: SGSCTBatch, causal_mask: Tensor) -> Tensor:
        """Nodes with a directed G_C path to at least one symptom node."""

        ancestors = batch.symptom_node_mask.bool().clone()
        source, target = batch.edge_index
        causal_edges = torch.nonzero(causal_mask, as_tuple=False).flatten()
        if not causal_edges.numel():
            return ancestors
        causal_source = source[causal_edges]
        causal_target = target[causal_edges]
        while True:
            reaches = ancestors[causal_target]
            if not torch.any(reaches):
                break
            updated = ancestors.clone()
            updated[causal_source[reaches]] = True
            if torch.equal(updated, ancestors):
                break
            ancestors = updated
        return ancestors

    @staticmethod
    def _localized_ancestor_mask(
        batch: SGSCTBatch, causal_mask: Tensor, local_masks: Tensor
    ) -> Tensor:
        """Per-block TIME-WINDOWED ancestor closure: nodes with a G_C path to the
        block's symptom nodes that stays entirely within ``local_masks`` (the
        block's time-windowed neighbourhood). Returns [B, N].

        This is the user's ''time-windowed ancestor closure'': reverse inference
        cannot flood the whole (fully-connected) schedule -- it only walks causal
        edges whose source lies inside the block's temporal window, so root blocks
        are time-adjacent. Falls back to the plain neighbourhood (which already
        bounds roots) when no causal edge qualifies.
        """
        B, N = local_masks.shape
        out = torch.zeros_like(local_masks, dtype=torch.bool)
        source, target = batch.edge_index
        causal_edges = torch.nonzero(causal_mask, as_tuple=False).flatten()
        if not causal_edges.numel():
            return out
        src = source[causal_edges]
        tgt = target[causal_edges]
        symptom = batch.symptom_node_mask.bool()
        for b in range(B):
            local = local_masks[b]
            valid = local[src]
            b_src = src[valid]
            b_tgt = tgt[valid]
            reach = symptom.clone() & local
            changed = True
            while changed:
                changed = False
                reached_target = reach[b_tgt]
                new_src = b_src[reached_target]
                for s in new_src.tolist():
                    if not reach[s]:
                        reach[s] = True
                        changed = True
            out[b] = reach
        return out

    def _path_decode(
        self,
        nodes: Tensor,
        query_by_node: Tensor,
        anchors: Tensor,
        blocks: Tensor,
        batch: SGSCTBatch,
        causal_mask: Tensor,
        ancestor_mask: Tensor,
        forced_paths: Tensor | None = None,
        node_ce: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flat_anchor = anchors.flatten()
        candidate_count = flat_anchor.shape[0]
        node_count = nodes.shape[0]
        # Differentiable per-step logits (see _block_decode); buffer keeps
        # gradient through path_node/path_context/path_stop.
        step_logits_buffer: list[Tensor | None] = [
            None
        ] * (candidate_count * self.max_path_length)
        paths = torch.full(
            (candidate_count, self.max_path_length), -1, dtype=torch.long, device=nodes.device
        )
        stopped = torch.ones(candidate_count, dtype=torch.bool, device=nodes.device)
        source, target = batch.edge_index
        # Align ESWA path rows to anchors by the row's first node (the anchor),
        # not by positional slot: the model scores its own Top-8 anchors so their
        # order is not guaranteed to match the supervision slot order.
        forced_rows: dict[int, Tensor] = {}
        if forced_paths is not None:
            for row in range(forced_paths.shape[0]):
                start = int(forced_paths[row, 0])
                if start >= 0:
                    forced_rows[start] = forced_paths[row]
        for candidate_id, anchor_tensor in enumerate(flat_anchor):
            anchor = int(anchor_tensor)
            if anchor < 0:
                continue
            graph_id = int(batch.node_graph[anchor])
            block_members = blocks[candidate_id][blocks[candidate_id] >= 0]
            block_pool = nodes[block_members].mean(0)
            current = anchor
            visited = {current}
            paths[candidate_id, 0] = current
            stopped[candidate_id] = False
            for step in range(self.max_path_length - 1):
                query = query_by_node[current]
                context = torch.cat([nodes[current], block_pool, query])
                node_score = (self.path_node(nodes) * self.path_context(context)).sum(-1) / math.sqrt(self.hidden_dim)
                if node_ce is not None:
                    node_score = node_score + self.ce_scale * node_ce
                legal_edges = causal_mask & (source == current)
                legal_nodes = target[legal_edges]
                legal = torch.zeros(node_count, dtype=torch.bool, device=nodes.device)
                legal[legal_nodes] = True
                legal &= ancestor_mask
                if visited:
                    legal[list(visited)] = False
                at_symptom = bool(batch.symptom_node_mask[current])
                # A decoded root path is required to travel forward through
                # G_C to the symptom. STOP is legal at the symptom, or as an
                # explicit failure marker when no valid successor remains.
                stop_logit = (
                    self.path_stop(context).squeeze(-1)
                    if at_symptom or not torch.any(legal)
                    else nodes.new_tensor(-torch.inf)
                )
                if at_symptom:
                    stop_logit = stop_logit + 20.0
                step_logits = torch.cat(
                    [
                        node_score.masked_fill(~legal, -torch.inf),
                        stop_logit.unsqueeze(0),
                    ]
                )
                step_logits_buffer[candidate_id * self.max_path_length + step] = step_logits
                # Teacher forcing: follow the ESWA path anchored at this
                # candidate's anchor so every target hop is a legal G_C
                # successor and the step logits stay finite.
                forced_row = forced_rows.get(anchor) if forced_paths is not None else None
                if forced_row is not None:
                    next_node = int(forced_row[step + 1])
                    if next_node == node_count:
                        stopped[candidate_id] = True
                        break
                    if next_node < 0:
                        break
                    current = next_node
                else:
                    choice = int(torch.argmax(step_logits.detach()))
                    if choice == node_count:
                        stopped[candidate_id] = True
                        break
                    current = choice
                visited.add(current)
                paths[candidate_id, step + 1] = current
        filled = [
            item if item is not None else nodes.new_full((node_count + 1,), -torch.inf)
            for item in step_logits_buffer
        ]
        logits = torch.stack(filled).view(candidate_count, self.max_path_length, node_count + 1)
        return logits, paths, stopped

    def forward(self, batch: SGSCTBatch) -> SGSCTOutput:
        batch.validate()
        graph_count = self._graph_count(batch)
        if not graph_count or not batch.node_numeric.shape[0]:
            raise ValueError("SG-SCT requires at least one graph and one node")
        nodes0 = self.node_projection(batch.node_numeric, batch.node_type)
        context_mask = (batch.edge_role == int(EdgeRole.CONTEXT)) | (
            batch.edge_role == int(EdgeRole.FEASIBILITY)
        )
        context = nodes0
        for layer in self.context_layers:
            context, _, _ = layer(
                context, batch.edge_index, batch.edge_type, batch.edge_features, context_mask
            )

        hard_mask = batch.edge_role == int(EdgeRole.CAUSAL_HARD)
        soft_mask = batch.edge_role == int(EdgeRole.CAUSAL_SOFT)
        causal_mask = hard_mask | soft_mask
        causal = nodes0
        gate_logits = nodes0.new_full((batch.edge_index.shape[1],), -20.0)
        gate_values = nodes0.new_zeros(batch.edge_index.shape[1])
        for layer in self.causal_layers:
            causal, gate_logits, gate_values = layer(
                causal,
                batch.edge_index,
                batch.edge_type,
                batch.edge_features,
                causal_mask,
                hard_edge_mask=hard_mask,
                soft_edge_mask=soft_mask,
            )

        gantt = self.gantt_encoder(batch, graph_count)
        # Local operation-ID alignment is the memory-safe bidirectional
        # graph/sequence cross-attention equivalent for v1.
        fused_nodes, fused_gantt = self._align_graph_and_gantt(
            0.5 * (context + causal), gantt, batch, graph_count
        )
        state = self._masked_graph_mean(
            fused_nodes, batch.node_graph, batch.symptom_node_mask.bool(), graph_count
        )
        state = self.state_query(state)
        history = self.trajectory_encoder(batch, state, graph_count)
        reconstruction = self.mask_reconstruction_head(fused_nodes)
        action_mask = torch.zeros(graph_count, dtype=torch.bool, device=state.device)
        transition: Tensor | None = None
        metric: Tensor | None = None
        if batch.action_features is not None:
            if self.action_projection is None:
                raise ValueError("action_features were supplied but action_numeric_dim is zero")
            if batch.action_features.shape[1] != self.action_numeric_dim:
                raise ValueError("action_features do not match action_numeric_dim")
            observed_actions = (
                batch.action_observed_mask.bool()
                if batch.action_observed_mask is not None
                else torch.ones(batch.action_features.shape[0], dtype=torch.bool, device=state.device)
            )
            action_hidden = self.action_projection(batch.action_features)
            selected = torch.nonzero(observed_actions, as_tuple=False).flatten()
            action_summary = state.new_zeros((graph_count, self.hidden_dim))
            if selected.numel():
                action_summary = _scatter_mean(
                    action_hidden[selected], batch.action_graph[selected], graph_count
                )
                action_mask[batch.action_graph[selected]] = True
            transition_input = torch.cat([state, history, action_summary], dim=-1)
            transition = self.transition_head(transition_input)
            metric = self.metric_head(transition_input)

        parent_context = self._parent_context(fused_nodes, batch, causal_mask)
        if batch.mechanism_values is None:
            if batch.node_numeric.shape[1] < self.mechanism_dim:
                raise ValueError("node_numeric is smaller than mechanism_dim")
            observed = batch.node_numeric[:, : self.mechanism_dim]
            observed_mask = torch.ones_like(observed, dtype=torch.bool)
        else:
            observed = batch.mechanism_values
            observed_mask = (
                batch.mechanism_value_mask.bool()
                if batch.mechanism_value_mask is not None
                else torch.ones_like(observed, dtype=torch.bool)
            )
        mu, scale, degrees, standardized, nll, epistemic = self.mechanism(
            fused_nodes, parent_context, observed
        )
        standardized = standardized.masked_fill(~observed_mask.unsqueeze(0), 0.0)
        nll = nll.masked_fill(~observed_mask.unsqueeze(0), 0.0)
        anomaly = self.anomaly_projection(
            torch.cat(
                [
                    fused_nodes,
                    standardized.abs().mean(0),
                    nll.mean(0),
                    epistemic,
                ],
                dim=-1,
            )
        )

        # Reverse inference walks the combined L1/L2/L4 connectivity (causal
        # execution edges plus the L4 shared-symptom cross links), so the roots
        # of two scheduling blocks become reachable through the shared symptom.
        reverse_mask = causal_mask | (batch.edge_role == int(EdgeRole.CAUSAL_CROSS_SYMPTOM))
        reverse = fused_nodes + anomaly
        reversed_edges = torch.stack([batch.edge_index[1], batch.edge_index[0]])
        # Per-symptom-block state query: each block gets its own root query, so
        # reverse inference is anchored per block rather than pooled globally.
        block_count = batch.symptom_block_count
        symptom_block_state: Tensor | None = None
        node_query = state[batch.node_graph]
        if block_count > 0 and batch.symptom_block_node_index is not None:
            b_idx = batch.symptom_block_node_index[0]
            n_idx = batch.symptom_block_node_index[1]
            # v2 §4 Appearance-conditioned per-block query:
            #   h_A = AttentionPool(block member nodes),
            #   e_A = AppearanceTypeEmbedding(primary rule),
            #   x_A = AppearanceFeatureEncoder(continuous features),
            #   q_A = MLP_A[h_A, e_A, x_A].
            # When appearance fields are absent (pre-v2 frozen bundles), fall
            # back to the legacy state projection so old checkpoints still run.
            if batch.appearance_type is not None and batch.appearance_features is not None:
                h_A = _attention_pool(fused_nodes[n_idx], b_idx, block_count, self.block_attention)
                e_A = self.appearance_type_embedding(batch.appearance_type)
                x_A = self.appearance_feature_encoder(batch.appearance_features)
                block_query = self.appearance_query_mlp(
                    torch.cat([h_A, e_A, x_A], dim=-1)
                )
            else:
                block_repr = _scatter_mean(fused_nodes[n_idx], b_idx, block_count)
                block_query = self.state_query(block_repr)
            symptom_block_state = block_query
            if batch.node_symptom_block is not None:
                nb = batch.node_symptom_block
                per_block = block_query[nb.clamp_min(0)]
                node_query = torch.where(
                    (nb >= 0).unsqueeze(-1), per_block, node_query
                )
        for layer in self.reverse_layers:
            reverse, _, _ = layer(
                reverse,
                reversed_edges,
                batch.edge_type,
                batch.edge_features,
                reverse_mask,
            )
            reverse = reverse + self.reverse_query(node_query)

        # v2 §9/§10 CE heads: predict the appearance causal effect per node and
        # per edge from the reverse embeddings.  Sigmoid keeps outputs in [0,1]
        # to match the solver-counterfactual CE_A labels they regress against.
        node_ce = self.node_ce_head(reverse).squeeze(-1).sigmoid()  # [N]
        edge_ce = self.edge_ce_head(
            torch.cat([reverse[batch.edge_index[0]], reverse[batch.edge_index[1]]], dim=-1)
        ).squeeze(-1).sigmoid()  # [E]

        node_history = history[batch.node_graph]
        anchor_logits = self.anchor_head(
            torch.cat(
                [reverse, anomaly, node_query, node_history, self.ce_projection(node_ce.unsqueeze(-1))],
                dim=-1,
            )
        ).squeeze(-1)
        ancestor_mask = self._causal_ancestor_mask(batch, reverse_mask)
        # Symptom nodes are legitimate root candidates (a symptom may be its
        # own root cause), so we no longer exclude them here.
        root_legal_mask = batch.candidate_node_mask.bool() & ancestor_mask
        if block_count > 0 and batch.symptom_block_neighborhood_mask is not None:
            # Time-windowed ancestor closure: reverse inference stays within each
            # block's temporal window (root blocks are time-adjacent), instead of
            # flooding the fully-connected causal graph then trimming.
            local_ancestor = self._localized_ancestor_mask(
                batch, reverse_mask, batch.symptom_block_neighborhood_mask.bool()
            )
            per_block_legal = local_ancestor
            anchors, anchor_scores = self._top_anchors_per_block(
                anchor_logits, batch, block_count, per_block_legal
            )
        else:
            anchor_logits = anchor_logits.masked_fill(~root_legal_mask, -torch.inf)
            anchors, anchor_scores = self._top_anchors(
                anchor_logits, batch, graph_count, root_legal_mask
            )
        flat_anchors = anchors.flatten()
        safe_anchors = flat_anchors.clamp_min(0)
        anchor_graph = batch.node_graph[safe_anchors]
        candidate_repr = torch.cat([reverse[safe_anchors], node_query[safe_anchors]], dim=-1)
        root_types = self.root_type_head(candidate_repr)
        confidence = torch.sigmoid(self.confidence_head(candidate_repr).squeeze(-1))
        invalid_anchor = flat_anchors < 0
        root_types = root_types.masked_fill(invalid_anchor.unsqueeze(-1), -torch.inf)
        confidence = confidence.masked_fill(invalid_anchor, 0.0)

        # Block members are ancestors of a symptom and may themselves be
        # symptom nodes (the root block spans the root mechanism up to where it
        # manifests), so the block legality mask is looser than the anchor mask.
        # When per-block inference is active, each block is bounded to its own
        # neighbourhood mask (roots must not be too far apart).
        if block_count > 0 and batch.symptom_block_neighborhood_mask is not None:
            block_legal_mask = local_ancestor
        else:
            block_legal_mask = batch.candidate_node_mask.bool() & ancestor_mask
        block_logits, block_nodes, block_stopped = self._block_decode(
            reverse, node_query + node_history, anchors, batch, block_legal_mask,
            node_ce=node_ce,
        )
        root_block_logits = torch.logsumexp(block_logits[..., :-1], dim=1)
        root_block_logits = torch.nan_to_num(
            root_block_logits, nan=-20.0, neginf=-20.0, posinf=20.0
        )
        # Remap each slot's anchor score into its anchor column via a
        # non-inplace scatter so the block logits stay differentiable.
        # (Inplace writes to the logsumexp output corrupt its autograd view.)
        anchor_val = anchor_scores.flatten()
        valid_anchor = flat_anchors >= 0
        anchor_placed = torch.zeros_like(root_block_logits)
        anchor_mask = torch.zeros_like(root_block_logits, dtype=torch.bool)
        if torch.any(valid_anchor):
            anchor_idx = flat_anchors.clamp_min(0).unsqueeze(1)
            anchor_placed = anchor_placed.scatter(
                1, anchor_idx, (anchor_val * valid_anchor.to(anchor_val.dtype)).unsqueeze(1)
            )
            anchor_mask = anchor_mask.scatter(1, anchor_idx, valid_anchor.unsqueeze(1))
        # Only cap the anchor column with its score; a plain torch.maximum against
        # an all-zero baseline would sieve gradient away from every position whose
        # logit is negative, which at init is all of them (masked slots are -20).
        root_block_logits = torch.where(
            anchor_mask,
            torch.maximum(root_block_logits, anchor_placed),
            root_block_logits,
        )
        path_logits, path_nodes, path_stopped = self._path_decode(
            reverse,
            node_query + node_history,
            anchors,
            block_nodes,
            batch,
            reverse_mask,
            ancestor_mask,
            forced_paths=batch.path_target,
            node_ce=node_ce,
        )
        # Per-symptom-block root logits: collapse each block's anchor rows into
        # one [B, N] score (logsumexp over the block's max_anchors rows).
        per_block_root_logits: Tensor | None = None
        block_row_symptom_index: Tensor | None = None
        if block_count > 0:
            node_c = root_block_logits.shape[1]
            per_block_root_logits = torch.logsumexp(
                root_block_logits.view(block_count, self.max_anchors, node_c), dim=1
            )
            per_block_root_logits = torch.nan_to_num(
                per_block_root_logits, nan=-20.0, neginf=-20.0, posinf=20.0
            )
            block_row_symptom_index = torch.arange(
                block_count * self.max_anchors, device=root_block_logits.device
            ).div(self.max_anchors, rounding_mode="floor")
        top_count = min(self.top_root_candidates, self.max_anchors)
        top_nodes = anchors[:, :top_count]
        top_scores = anchor_scores[:, :top_count]
        return SGSCTOutput(
            node_embeddings=fused_nodes,
            state_embeddings=state,
            history_embeddings=history,
            gantt_embeddings=fused_gantt,
            m1_mask_reconstruction=reconstruction,
            m1_transition=transition,
            m1_metric=metric,
            m1_action_observed_mask=action_mask,
            causal_gate_logits=gate_logits,
            causal_gate_values=gate_values,
            mechanism_mu=mu,
            mechanism_scale=scale,
            mechanism_df=degrees,
            standardized_residual=standardized,
            mechanism_nll=nll,
            epistemic_uncertainty=epistemic,
            anomaly_embeddings=anomaly,
            reverse_embeddings=reverse,
            anchor_logits=anchor_logits,
            anchor_node_index=anchors,
            anchor_scores=anchor_scores,
            root_block_logits=root_block_logits,
            root_type_logits=root_types,
            candidate_confidence=confidence,
            block_pointer_logits=block_logits,
            block_node_index=block_nodes,
            block_stopped=block_stopped,
            path_pointer_logits=path_logits,
            path_node_index=path_nodes,
            path_stopped=path_stopped,
            top_root_node_index=top_nodes,
            top_root_scores=top_scores,
            symptom_block_state=symptom_block_state,
            per_block_root_logits=per_block_root_logits,
            block_row_symptom_index=block_row_symptom_index,
            node_ce_scores=node_ce,
            edge_ce_scores=edge_ce,
            symptom_block_count=int(block_count),
        )


__all__ = [
    "EdgeRole",
    "SGSCTBatch",
    "SGSCTOutput",
    "SGSCTRootCauseModel",
]
