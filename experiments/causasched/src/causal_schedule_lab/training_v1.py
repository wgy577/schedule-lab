"""First runnable learning contracts for causal-root and operator SFT/GRPO.

The module contains trainable mathematical components and strict data gates;
it does not claim that a production checkpoint has been trained.  LLM output
is treated only as a proposal source.  Positive operator demonstrations must
pass the exact masks, Full Oracle and deterministic replay recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .learning import GraphBatch, RelationalEncoder


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


EVIDENCE_TIER_WEIGHT = {
    "weak_observational": 0.35,
    "mechanism_consistent": 0.55,
    "solver_supported": 0.75,
    "bidirectionally_supported": 0.9,
    "ground_truth": 1.0,
}


@dataclass(frozen=True)
class RootCauseBatch:
    graph: GraphBatch
    symptom_node_mask: Tensor
    candidate_node_mask: Tensor
    causal_edge_mask: Tensor


@dataclass(frozen=True)
class RootCauseTargets:
    root_node_soft_target: Tensor
    root_block_membership: Tensor
    path_edge_soft_target: Tensor
    confidence: Tensor
    rank_pairs: Tensor | None = None


class SymptomConditionedRootModel(nn.Module):
    """Shared relational encoder plus symptom-conditioned reverse scoring."""

    def __init__(
        self,
        *,
        numeric_dim: int,
        node_type_count: int,
        edge_type_count: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = RelationalEncoder(
            numeric_dim=numeric_dim,
            node_type_count=node_type_count,
            edge_type_count=edge_type_count,
            hidden_dim=hidden_dim,
        )
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.reverse_layers = nn.ModuleList(
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(3)
        )
        self.root_head = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.actionability_head = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.block_head = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.path_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(0) / weights.sum(0).clamp_min(1.0)

    def forward(self, batch: RootCauseBatch) -> dict[str, Tensor]:
        nodes = self.encoder(batch.graph)
        symptom = self.query(self._masked_mean(nodes, batch.symptom_node_mask))
        reverse = nodes
        source, target = batch.graph.edge_index
        for layer in self.reverse_layers:
            messages = torch.zeros_like(reverse)
            eligible = batch.causal_edge_mask.bool()
            if torch.any(eligible):
                # Reverse propagation: target/effect sends to source/candidate cause.
                messages.index_add_(0, source[eligible], reverse[target[eligible]])
            query = symptom.unsqueeze(0).expand_as(reverse)
            reverse = F.gelu(layer(torch.cat([reverse + messages, query], dim=-1)))
        query = symptom.unsqueeze(0).expand_as(reverse)
        minimum = torch.finfo(reverse.dtype).min
        root_logits = self.root_head(reverse, query).squeeze(-1).masked_fill(
            ~batch.candidate_node_mask.bool(), minimum
        )
        block_logits = self.block_head(reverse, query).squeeze(-1).masked_fill(
            ~batch.candidate_node_mask.bool(), minimum
        )
        edge_query = symptom.unsqueeze(0).expand(source.shape[0], -1)
        edge_logits = self.path_head(
            torch.cat([reverse[source], reverse[target], edge_query], dim=-1)
        ).squeeze(-1).masked_fill(~batch.causal_edge_mask.bool(), minimum)
        return {
            "root_node_logits": root_logits,
            "actionability_logits": self.actionability_head(reverse, query).squeeze(-1).masked_fill(
                ~batch.candidate_node_mask.bool(), minimum
            ),
            "root_block_logits": block_logits,
            "path_edge_logits": edge_logits,
        }


def root_cause_sft_loss(
    prediction: dict[str, Tensor],
    target: RootCauseTargets,
    *,
    candidate_node_mask: Tensor,
    causal_edge_mask: Tensor,
    sparse_weight: float = 0.02,
) -> tuple[Tensor, dict[str, float]]:
    """Confidence-weighted weak/strong supervision with no truth upgrade."""

    node_mask = candidate_node_mask.bool()
    edge_mask = causal_edge_mask.bool()
    confidence = target.confidence.mean().clamp(0.0, 1.0)
    root = F.binary_cross_entropy_with_logits(
        prediction["root_node_logits"][node_mask],
        target.root_node_soft_target[node_mask],
    )
    block_bce = F.binary_cross_entropy_with_logits(
        prediction["root_block_logits"][node_mask],
        target.root_block_membership[node_mask],
    )
    block_probability = torch.sigmoid(prediction["root_block_logits"][node_mask])
    block_target = target.root_block_membership[node_mask]
    dice = 1 - (2 * (block_probability * block_target).sum() + 1.0) / (
        block_probability.sum() + block_target.sum() + 1.0
    )
    if torch.any(edge_mask):
        path = F.binary_cross_entropy_with_logits(
            prediction["path_edge_logits"][edge_mask],
            target.path_edge_soft_target[edge_mask],
        )
    else:
        # Zero-with-gradient: multiply by 0 first so masked (finfo.min) logits
        # cannot overflow the sum into +/-inf (inf * 0.0 == NaN under IEEE 754).
        path = (prediction["path_edge_logits"] * 0.0).sum()
    if target.rank_pairs is not None and target.rank_pairs.numel():
        left = prediction["root_node_logits"][target.rank_pairs[:, 0]]
        right = prediction["root_node_logits"][target.rank_pairs[:, 1]]
        ranking = -F.logsigmoid(left - right).mean()
    else:
        ranking = (prediction["root_node_logits"] * 0.0).sum()
    sparse = block_probability.mean()
    total = confidence * (
        root + block_bce + dice + path + ranking + sparse_weight * sparse
    )
    pieces = {
        "root_node": root,
        "root_block_bce": block_bce,
        "root_block_dice": dice,
        "path_edge": path,
        "ranking": ranking,
        "sparse": sparse,
        "confidence": confidence,
    }
    return total, {key: float(value.detach()) for key, value in pieces.items()}


@dataclass(frozen=True)
class OperatorPolicyBatch:
    graph: GraphBatch
    symptom_node_mask: Tensor
    root_node_mask: Tensor
    family_index: Tensor
    environment_index: Tensor
    budget_features: Tensor
    operator_mask: Tensor
    parameter_features: Tensor
    parameter_state_index: Tensor
    parameter_operator_index: Tensor
    parameter_mask: Tensor


class RootConditionedOperatorPolicy(nn.Module):
    """Critic-free hierarchical operator/finite-parameter-candidate policy."""

    def __init__(
        self,
        *,
        numeric_dim: int,
        node_type_count: int,
        edge_type_count: int,
        operator_count: int,
        parameter_dim: int,
        family_count: int = 6,
        environment_count: int = 3,
        budget_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.operator_count = operator_count
        self.encoder = RelationalEncoder(
            numeric_dim=numeric_dim,
            node_type_count=node_type_count,
            edge_type_count=edge_type_count,
            hidden_dim=hidden_dim,
        )
        self.family_embedding = nn.Embedding(family_count, hidden_dim)
        self.environment_embedding = nn.Embedding(environment_count, hidden_dim)
        self.budget_projection = nn.Linear(budget_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.operator_embedding = nn.Embedding(operator_count, hidden_dim)
        self.operator_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.parameter_projection = nn.Linear(parameter_dim, hidden_dim)
        self.parameter_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _pool(nodes: Tensor, graph_index: Tensor, mask: Tensor, count: int) -> Tensor:
        result = torch.zeros(count, nodes.shape[-1], device=nodes.device, dtype=nodes.dtype)
        cardinality = torch.zeros(count, 1, device=nodes.device, dtype=nodes.dtype)
        weights = mask.to(nodes.dtype).unsqueeze(-1)
        result.index_add_(0, graph_index, nodes * weights)
        cardinality.index_add_(0, graph_index, weights)
        return result / cardinality.clamp_min(1.0)

    def forward(self, batch: OperatorPolicyBatch) -> dict[str, Tensor]:
        nodes = self.encoder(batch.graph)
        batch_count = batch.operator_mask.shape[0]
        global_pool = self._pool(
            nodes, batch.graph.graph_batch, torch.ones_like(batch.symptom_node_mask), batch_count
        )
        symptom_pool = self._pool(
            nodes, batch.graph.graph_batch, batch.symptom_node_mask, batch_count
        )
        root_pool = self._pool(
            nodes, batch.graph.graph_batch, batch.root_node_mask, batch_count
        )
        context = self.fusion(
            torch.cat(
                [
                    global_pool,
                    symptom_pool,
                    root_pool,
                    self.family_embedding(batch.family_index),
                    self.environment_embedding(batch.environment_index),
                    self.budget_projection(batch.budget_features),
                ],
                dim=-1,
            )
        )
        operator_ids = torch.arange(self.operator_count, device=nodes.device)
        operator_embedding = self.operator_embedding(operator_ids)
        expanded_context = context.unsqueeze(1).expand(-1, self.operator_count, -1)
        expanded_operator = operator_embedding.unsqueeze(0).expand(batch_count, -1, -1)
        operator_logits = self.operator_scorer(
            torch.cat([expanded_context, expanded_operator], dim=-1)
        ).squeeze(-1)
        operator_logits = operator_logits.masked_fill(
            ~batch.operator_mask.bool(), torch.finfo(operator_logits.dtype).min
        )
        param_context = context[batch.parameter_state_index]
        param_operator = operator_embedding[batch.parameter_operator_index]
        param_hidden = self.parameter_projection(batch.parameter_features)
        parameter_logits = self.parameter_scorer(
            torch.cat([param_context, param_operator, param_hidden], dim=-1)
        ).squeeze(-1)
        parameter_logits = parameter_logits.masked_fill(
            ~batch.parameter_mask.bool(), torch.finfo(parameter_logits.dtype).min
        )
        return {
            "operator_logits": operator_logits,
            "parameter_logits": parameter_logits,
            "parameter_validity_logit": self.validity_head(param_hidden).squeeze(-1),
            "parameter_gain": self.gain_head(param_hidden).squeeze(-1),
        }


@dataclass(frozen=True)
class OperatorSFTTargets:
    operator_distribution: Tensor
    parameter_distribution: Tensor
    parameter_validity: Tensor
    parameter_gain: Tensor
    state_weight: Tensor


def _ragged_log_softmax(
    logits: Tensor,
    state_index: Tensor,
    operator_index: Tensor,
) -> Tensor:
    result = torch.empty_like(logits)
    for state in torch.unique(state_index):
        for operator in torch.unique(operator_index[state_index == state]):
            mask = (state_index == state) & (operator_index == operator)
            result[mask] = F.log_softmax(logits[mask], dim=0)
    return result


def operator_sft_loss(
    prediction: dict[str, Tensor],
    target: OperatorSFTTargets,
    batch: OperatorPolicyBatch,
    *,
    validity_weight: float = 0.25,
    gain_weight: float = 0.25,
) -> tuple[Tensor, dict[str, float]]:
    _validate_operator_sft_contract(prediction, target, batch)
    op_log_probability = F.log_softmax(prediction["operator_logits"], dim=-1)
    per_state_operator = -(target.operator_distribution * op_log_probability).sum(-1)
    operator_loss = (target.state_weight * per_state_operator).sum() / target.state_weight.sum().clamp_min(1.0)
    param_log_probability = _ragged_log_softmax(
        prediction["parameter_logits"],
        batch.parameter_state_index,
        batch.parameter_operator_index,
    )
    parameter_groups = []
    parameter_group_weights = []
    for state in torch.unique(batch.parameter_state_index):
        operators = torch.unique(
            batch.parameter_operator_index[batch.parameter_state_index == state]
        )
        for operator in operators:
            group = (
                (batch.parameter_state_index == state)
                & (batch.parameter_operator_index == operator)
                & batch.parameter_mask.bool()
            )
            if not torch.any(group):
                continue
            parameter_groups.append(
                -(target.parameter_distribution[group] * param_log_probability[group]).sum()
            )
            parameter_group_weights.append(target.state_weight[state])
    if parameter_groups:
        group_losses = torch.stack(parameter_groups)
        group_weights = torch.stack(parameter_group_weights)
        parameter_loss = (group_losses * group_weights).sum() / group_weights.sum().clamp_min(1.0)
    else:
        parameter_loss = (prediction["parameter_logits"] * 0.0).sum()
    validity = F.binary_cross_entropy_with_logits(
        prediction["parameter_validity_logit"], target.parameter_validity
    )
    gain = F.smooth_l1_loss(prediction["parameter_gain"], target.parameter_gain)
    total = operator_loss + parameter_loss + validity_weight * validity + gain_weight * gain
    return total, {
        "operator": float(operator_loss.detach()),
        "parameter": float(parameter_loss.detach()),
        "validity": float(validity.detach()),
        "gain": float(gain.detach()),
    }


def _validate_operator_sft_contract(
    prediction: dict[str, Tensor],
    target: OperatorSFTTargets,
    batch: OperatorPolicyBatch,
    *,
    tolerance: float = 1e-5,
) -> None:
    """Reject malformed distributions before they can silently train a policy."""

    operator_logits = prediction["operator_logits"]
    if target.operator_distribution.shape != operator_logits.shape:
        raise ValueError("operator target shape must match operator logits")
    if batch.operator_mask.shape != operator_logits.shape:
        raise ValueError("operator mask shape must match operator logits")
    parameter_count = prediction["parameter_logits"].numel()
    one_dimensional = (
        target.parameter_distribution,
        target.parameter_validity,
        target.parameter_gain,
        batch.parameter_state_index,
        batch.parameter_operator_index,
        batch.parameter_mask,
    )
    if any(value.ndim != 1 or value.numel() != parameter_count for value in one_dimensional):
        raise ValueError("all parameter targets and indices must match parameter logits")
    if target.state_weight.ndim != 1 or target.state_weight.numel() != operator_logits.shape[0]:
        raise ValueError("state weights must contain one value per state")
    float_targets = (
        target.operator_distribution,
        target.parameter_distribution,
        target.parameter_validity,
        target.parameter_gain,
        target.state_weight,
    )
    if any(not torch.all(torch.isfinite(value)) for value in float_targets):
        raise ValueError("operator SFT targets must be finite")
    if torch.any(target.operator_distribution < 0) or torch.any(target.parameter_distribution < 0):
        raise ValueError("operator and parameter distributions must be nonnegative")
    if torch.any(target.parameter_validity < 0) or torch.any(target.parameter_validity > 1):
        raise ValueError("parameter validity targets must be in [0, 1]")
    if torch.any(target.state_weight <= 0):
        raise ValueError("state weights must be positive")
    if torch.any(batch.operator_mask.sum(-1) == 0):
        raise ValueError("each state must have at least one legal operator")
    if torch.any(target.operator_distribution[~batch.operator_mask.bool()] > tolerance):
        raise ValueError("illegal operators must have zero target probability")
    if torch.any(target.parameter_distribution[~batch.parameter_mask.bool()] > tolerance):
        raise ValueError("illegal parameter candidates must have zero target probability")
    operator_sums = target.operator_distribution.sum(-1)
    if not torch.allclose(operator_sums, torch.ones_like(operator_sums), atol=tolerance, rtol=0):
        raise ValueError("each operator target distribution must sum to one")
    if torch.any(batch.parameter_state_index < 0) or torch.any(
        batch.parameter_state_index >= operator_logits.shape[0]
    ):
        raise ValueError("parameter state index is out of range")
    if torch.any(batch.parameter_operator_index < 0) or torch.any(
        batch.parameter_operator_index >= operator_logits.shape[1]
    ):
        raise ValueError("parameter operator index is out of range")
    for state in torch.unique(batch.parameter_state_index):
        for operator in torch.unique(
            batch.parameter_operator_index[batch.parameter_state_index == state]
        ):
            group = (batch.parameter_state_index == state) & (
                batch.parameter_operator_index == operator
            )
            legal = group & batch.parameter_mask.bool()
            probability = target.parameter_distribution[group].sum()
            expected = 1.0 if torch.any(legal) else 0.0
            if not torch.isclose(
                probability,
                probability.new_tensor(expected),
                atol=tolerance,
                rtol=0,
            ):
                raise ValueError(
                    "each legal (state, operator) parameter distribution must sum to one"
                )


class OperatorParameterCandidate(FrozenModel):
    candidate_id: str
    operator_id: str
    parameters: dict[str, Any]
    semantic_legal: bool
    local_legal: bool
    cause_legal: bool
    parameter_legal: bool

    @property
    def legal(self) -> bool:
        return self.semantic_legal and self.local_legal and self.cause_legal and self.parameter_legal


class OperatorPolicyState(FrozenModel):
    schema_version: str = "root-conditioned-operator-state-1.0"
    instance_id: str
    base_instance_id: str
    family: str
    environment: str
    state_id: str
    schedule_hash: str
    graph_hash: str
    surface_block_ids: tuple[str, ...]
    root_cause_id: str
    root_cause_node_ids: tuple[str, ...]
    causal_path_edge_ids: tuple[str, ...]
    semantic_operator_mask: dict[str, bool]
    local_operator_mask: dict[str, bool]
    cause_operator_mask: dict[str, bool]
    final_operator_mask: dict[str, bool]
    parameter_candidates: tuple[OperatorParameterCandidate, ...]
    registry_version: str
    candidate_set_hash: str
    current_objective: tuple[float, ...] = ()
    lower_bound: float | None = None
    budget_features: dict[str, float] = Field(default_factory=dict)
    recent_action_ids: tuple[str, ...] = ()


class TeacherActionProposal(FrozenModel):
    source: Literal["llm", "heuristic", "enumeration", "solver"]
    operator_id: str
    parameters: dict[str, Any]
    mechanism_code: str | None = None
    evidence_ids: tuple[str, ...] = ()


class TeacherCandidateResult(FrozenModel):
    candidate_id: str
    source: str
    action_id: str
    parse_pass: bool
    mask_pass: bool
    static_pass: bool
    light_pass: bool
    full_pass: bool
    deterministic_replay_pass: bool
    full_feasible_pass: bool | None = None
    objective_before: float
    objective_after: float | None = None
    best_lookahead_objective: float | None = None
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    oracle_cost: float = Field(default=0.0, ge=0.0)
    failure_labels: tuple[str, ...] = ()

    @property
    def positive_demonstration(self) -> bool:
        best_after = self.objective_after
        if self.best_lookahead_objective is not None:
            best_after = (
                self.best_lookahead_objective
                if best_after is None
                else min(best_after, self.best_lookahead_objective)
            )
        feasibility = self.full_pass if self.full_feasible_pass is None else self.full_feasible_pass
        return bool(
            self.parse_pass
            and self.mask_pass
            and self.static_pass
            and feasibility
            and self.deterministic_replay_pass
            and best_after is not None
            and best_after < self.objective_before
        )


class StopActionProof(FrozenModel):
    state_id: str
    candidate_set_hash: str
    legal_candidate_ids: tuple[str, ...]
    evaluated_candidate_ids: tuple[str, ...]
    full_oracle_pass_by_candidate: dict[str, bool]
    improving_candidate_ids: tuple[str, ...]
    fixed_budget_exhausted: bool

    @property
    def positive_stop_demonstration(self) -> bool:
        return bool(
            self.fixed_budget_exhausted
            and set(self.legal_candidate_ids) == set(self.evaluated_candidate_ids)
            and all(
                candidate_id in self.full_oracle_pass_by_candidate
                for candidate_id in self.legal_candidate_ids
            )
            and not self.improving_candidate_ids
        )


def canonical_action_id(
    state_hash: str,
    registry_version: str,
    operator_id: str,
    parameters: dict[str, Any],
) -> str:
    payload = {
        "state": state_hash,
        "registry": registry_version,
        "operator": operator_id,
        "parameters": parameters,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def candidate_set_hash(candidates: tuple[OperatorParameterCandidate, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in sorted(candidates, key=lambda item: item.candidate_id)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonicalize_teacher_proposal(
    proposal: TeacherActionProposal,
    candidates: tuple[OperatorParameterCandidate, ...],
) -> OperatorParameterCandidate | None:
    """Map LLM/teacher output onto the deterministic finite candidate set."""

    canonical = json.dumps(proposal.parameters, sort_keys=True, separators=(",", ":"))
    matches = [
        item for item in candidates
        if item.operator_id == proposal.operator_id
        and json.dumps(item.parameters, sort_keys=True, separators=(",", ":")) == canonical
    ]
    return min(matches, key=lambda item: item.candidate_id) if matches else None


@dataclass(frozen=True)
class GRPOGroup:
    new_log_probability: Tensor
    old_log_probability: Tensor
    branch_reward: Tensor
    valid_step_mask: Tensor
    best_prefix_index: Tensor


@dataclass(frozen=True)
class CausalGRPOGroup:
    """Root-branch group with an explicit strong-evidence update gate."""

    new_causal_log_probability: Tensor
    old_causal_log_probability: Tensor
    new_actionability_log_probability: Tensor
    old_actionability_log_probability: Tensor
    branch_reward: Tensor
    strong_causal_evidence_mask: Tensor


def group_relative_advantage(reward: Tensor, *, epsilon: float = 1e-8) -> tuple[Tensor, bool]:
    std = reward.std(unbiased=False)
    if float(std) <= epsilon:
        return torch.zeros_like(reward), False
    return (reward - reward.mean()) / (std + epsilon), True


def best_prefix_mask(
    valid_step_mask: Tensor,
    best_prefix_index: Tensor,
) -> Tensor:
    steps = torch.arange(valid_step_mask.shape[1], device=valid_step_mask.device)
    return valid_step_mask.bool() & (steps.unsqueeze(0) <= best_prefix_index.unsqueeze(1))


def macro_grpo_loss(
    group: GRPOGroup,
    *,
    clip_epsilon: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
    """Critic-free grouped PPO objective with best-prefix credit assignment."""

    advantage, informative = group_relative_advantage(group.branch_reward)
    mask = best_prefix_mask(group.valid_step_mask, group.best_prefix_index)
    if not informative or not torch.any(mask):
        zero = (group.new_log_probability * 0.0).sum()
        return zero, {"informative": 0.0, "advantage_std": float(group.branch_reward.std(unbiased=False))}
    log_ratio = (group.new_log_probability - group.old_log_probability).clamp(-30.0, 30.0)
    ratio = torch.exp(log_ratio)
    expanded_advantage = advantage.unsqueeze(1).expand_as(ratio)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * expanded_advantage
    loss = -torch.minimum(unclipped, clipped)[mask].mean()
    return loss, {
        "informative": 1.0,
        "advantage_mean": float(advantage.mean()),
        "advantage_std": float(advantage.std(unbiased=False)),
        "credited_steps": float(mask.sum()),
    }


def _clipped_group_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    reward: Tensor,
    mask: Tensor,
    clip_epsilon: float,
) -> tuple[Tensor, bool]:
    selected = mask.bool()
    if int(selected.sum()) < 2:
        return (new_log_probability * 0.0).sum(), False
    advantage, informative = group_relative_advantage(reward[selected])
    if not informative:
        return (new_log_probability * 0.0).sum(), False
    # Clamp the log-ratio before exp so a large policy/reference gap cannot
    # overflow ratio to +/-inf; in backward the clipped PPO path zeroes the
    # ratio gradient, and ``0 * inf == NaN`` under IEEE 754 would poison the
    # gradient.  +/-30 is far outside the PPO clip range [1-eps, 1+eps] so it
    # never changes in-range behaviour, only blocks pathological overflow.
    log_ratio = (new_log_probability[selected] - old_log_probability[selected]).clamp(-30.0, 30.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    return -torch.minimum(unclipped, clipped).mean(), True


def causal_root_grpo_loss(
    group: CausalGRPOGroup,
    *,
    clip_epsilon: float = 0.2,
    causal_weight: float = 1.0,
    actionability_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Update causal attribution only on solver/replay-supported branches.

    Ordinary objective improvement remains valid supervision for the separate
    actionability ranking, but cannot silently upgrade observational weak
    labels into causal evidence.
    """

    all_branches = torch.ones_like(group.branch_reward, dtype=torch.bool)
    actionability, actionability_updated = _clipped_group_loss(
        group.new_actionability_log_probability,
        group.old_actionability_log_probability,
        group.branch_reward,
        all_branches,
        clip_epsilon,
    )
    causal, causal_updated = _clipped_group_loss(
        group.new_causal_log_probability,
        group.old_causal_log_probability,
        group.branch_reward,
        group.strong_causal_evidence_mask,
        clip_epsilon,
    )
    total = causal_weight * causal + actionability_weight * actionability
    return total, {
        "causal_head_updated": float(causal_updated),
        "actionability_head_updated": float(actionability_updated),
        "strong_evidence_branches": float(group.strong_causal_evidence_mask.sum()),
        "causal_loss": float(causal.detach()),
        "actionability_loss": float(actionability.detach()),
    }


class TrainingShard(FrozenModel):
    shard_id: int = Field(ge=0)
    base_instance_ids: tuple[str, ...]


def build_instance_shards(
    base_instance_ids: list[str] | tuple[str, ...],
    *,
    shard_size: int = 128,
) -> tuple[TrainingShard, ...]:
    """Shard only after grouping by base instance to prevent derived-state leakage."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    unique = tuple(sorted(set(base_instance_ids)))
    return tuple(
        TrainingShard(shard_id=index // shard_size, base_instance_ids=unique[index:index + shard_size])
        for index in range(0, len(unique), shard_size)
    )


class VerifiedTrajectoryStep(FrozenModel):
    state_id: str
    candidate_set_hash: str
    candidate_results: tuple[TeacherCandidateResult, ...]
    selected_action_id: str
    next_state_id: str
    return_to_go: float
    trajectory_quality: float

    @model_validator(mode="after")
    def selected_action_is_verified(self) -> "VerifiedTrajectoryStep":
        selected = next(
            (item for item in self.candidate_results if item.action_id == self.selected_action_id),
            None,
        )
        if selected is None or not selected.positive_demonstration:
            raise ValueError("selected SFT action must be Full-Oracle/replay verified and improving")
        return self


__all__ = [
    "EVIDENCE_TIER_WEIGHT",
    "CausalGRPOGroup",
    "GRPOGroup",
    "OperatorParameterCandidate",
    "OperatorPolicyBatch",
    "OperatorPolicyState",
    "OperatorSFTTargets",
    "RootCauseBatch",
    "RootCauseTargets",
    "RootConditionedOperatorPolicy",
    "SymptomConditionedRootModel",
    "StopActionProof",
    "TeacherActionProposal",
    "TeacherCandidateResult",
    "TrainingShard",
    "VerifiedTrajectoryStep",
    "best_prefix_mask",
    "build_instance_shards",
    "candidate_set_hash",
    "causal_root_grpo_loss",
    "canonical_action_id",
    "canonicalize_teacher_proposal",
    "group_relative_advantage",
    "macro_grpo_loss",
    "operator_sft_loss",
    "root_cause_sft_loss",
]
