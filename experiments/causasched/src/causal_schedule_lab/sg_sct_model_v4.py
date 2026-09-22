"""SG-SCT M1+M2 v4 model -- post-2.29R architecture refactor.

Establishes the **new** model schema ``sg-sct-m1-m2-v4`` alongside the frozen
``sg-sct-m1-m2-v3`` in :mod:`causal_schedule_lab.sg_sct_model_v1`.  V3 is not
modified; V4 subclasses it, reuses every V3 component (backbone, heads, reverse
layers, decode methods) and overrides ``forward`` to apply four fixes that the
Phase-2.29R Failure Decomposition / Graph Audit identified:

1. **TRUE_LOCAL_G_C reverse mask.**  Reverse message passing uses
   ``batch.reverse_causal_mask`` (precedence + resource_sequence only) instead
   of ``role==2 | role==5``.  The machine hub, rule typing and block/swap
   relations no longer flood the reverse walk.
2. **Gated residual reverse stack.**  The reverse state starts from the M1
   fused node representation (initial residual) and each of the four reverse
   layers blends its output through a learnable per-node gate
   ``g = sigmoid(W h)``: ``h <- g * layer(h) + (1-g) * h``.  This preserves
   operation identity across depth instead of unconditionally overwriting.
3. **Branch A / Branch B split.**  Branch A is the proposal/localization
   decoder (anchor -> block -> path -> ``per_block_root_logits``) plus the
   sigmoid appearance-CE heads; it is a *proposal decoder*, not an effect
   estimator.  Branch B is the atomic intervention effect head
   (:class:`TargetConditionedGraphGanttAtomicCEAdapter`) -- an unconstrained
   signed-CE regressor conditioned on the exact intervention ``r`` and the
   appearance query, reading the **fixed** reverse embeddings.  Branch B does
   not consume Branch A outputs.
4. **Forward-only oversmoothing diagnostics.**  Per reverse layer the forward
   pass records within-block cosine, Dirichlet energy and effective rank so the
   de-cliquing / de-flooding effect is measurable without training.

No training is performed in this round.  ``identified`` stays ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sg_sct_model_v1 import (
    EdgeRole,
    SGSCTBatch,
    SGSCTOutput,
    SGSCTRootCauseModel,
    _attention_pool,
    _scatter_mean,
)

# Branch B reuses the existing target-conditioned atomic CE adapter, whose
# SignedAtomicCEHead is explicitly unbounded ("never applies sigmoid") and whose
# routing path reads contextual source/target resource/mode node indices rather
# than candidate-catalog counts -- exactly the Branch B contract.
from .teacher.m2_atomic_ce_v1 import (
    GraphGanttAtomicCandidateBatch,
    TargetConditionedGraphGanttAtomicCEAdapter,
)
from .teacher.m2_tensorized_v1 import (
    CANDIDATE_FEATURE_FIELDS,
    FROZEN_APPEARANCE_FEATURE_FIELDS,
)


MODEL_SCHEMA_V4 = "sg-sct-m1-m2-v4"

# Default Branch-B feature widths, taken from the frozen atomic-CE tensorizer.
# atom_numeric columns are per-atom intervention descriptors (max_plus_relevance,
# source/target machine/mode/position ordinals), NOT candidate-catalog counts.
_DEFAULT_ATOM_NUMERIC_DIM = len(CANDIDATE_FEATURE_FIELDS)
_DEFAULT_APPEARANCE_NUMERIC_DIM = len(FROZEN_APPEARANCE_FEATURE_FIELDS)


# ---------------------------------------------------------------------------
# Batch / output extensions (subclass -- V3 dataclasses untouched)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SGSCTBatchV4(SGSCTBatch):
    """V4 batch: V3 batch + the explicit reverse-causal mask.

    ``reverse_causal_mask`` is ``True`` only for ``gc:precedence`` /
    ``gc:resource_sequence`` edges (TRUE_LOCAL_G_C).  When ``None`` (e.g. a
    legacy 1.2.1 batch used in a diagnostic), V4 falls back to computing the
    same mask from ``edge_type`` so the mask fix can be measured on old graphs.
    """

    reverse_causal_mask: Tensor | None = None

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.reverse_causal_mask is not None:
            if self.reverse_causal_mask.shape != (self.edge_index.shape[1],):
                raise ValueError("reverse_causal_mask must have shape [E]")
            if self.reverse_causal_mask.dtype != torch.bool:
                raise ValueError("reverse_causal_mask must be boolean")


@dataclass(frozen=True)
class SGSCTOutputV4:
    """V4 output.  Carries Branch A (proposal) + Branch B (atomic effect) +
    forward-only oversmoothing diagnostics."""

    # M1 outputs (mirrored from V3 for Branch B / downstream consumers).
    node_embeddings: Tensor
    state_embeddings: Tensor
    history_embeddings: Tensor
    anomaly_embeddings: Tensor
    m1_mask_reconstruction: Tensor
    m1_transition: Tensor | None
    m1_metric: Tensor | None
    causal_gate_values: Tensor
    mechanism_mu: Tensor
    mechanism_nll: Tensor
    # Branch A -- proposal / localization (NOT an effect estimator).
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
    per_block_root_logits: Tensor | None
    block_row_symptom_index: Tensor | None
    symptom_block_state: Tensor | None
    node_ce_scores: Tensor | None
    edge_ce_scores: Tensor | None
    symptom_block_count: int
    # Branch B -- atomic intervention effect (unconstrained signed CE).
    atomic_signed_ce: Tensor | None = None
    # Forward-only oversmoothing diagnostics: one dict per reverse layer.
    oversmoothing_diagnostics: tuple[dict[str, float], ...] = ()
    # Audit: the reverse mask actually used (for the census / direction test).
    reverse_causal_mask: Tensor | None = None
    identified: bool = False
    training_status: str = "code_implemented_not_trained"
    model_schema_version: str = MODEL_SCHEMA_V4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SGSCTRootCauseModelV4(SGSCTRootCauseModel):
    """V4: TRUE_LOCAL_G_C reverse mask + gated residual + Branch A/B split.

    Subclasses V3, reuses the V3 backbone and decode methods, and overrides
    ``forward``.  V3 is not modified.
    """

    model_schema_version = MODEL_SCHEMA_V4

    def __init__(
        self,
        *args: Any,
        atom_numeric_dim: int = _DEFAULT_ATOM_NUMERIC_DIM,
        appearance_numeric_dim: int = _DEFAULT_APPEARANCE_NUMERIC_DIM,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.hidden_dim
        # Per-layer learnable message gate for the reverse stack (Part 4).
        # g = sigmoid(W_g h); h <- g * layer(h) + (1 - g) * h.
        self.reverse_gate = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(len(self.reverse_layers))
        )
        # Bias the gate toward "preserve" at init (sigmoid(0)=0.5); a small
        # negative bias favours identity early, slowing oversmoothing.
        for gate in self.reverse_gate:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -1.0)
        # Branch B: atomic intervention effect head (unconstrained signed CE).
        self.atomic_effect_head = TargetConditionedGraphGanttAtomicCEAdapter(
            hidden_dim=hidden_dim,
            atom_numeric_dim=atom_numeric_dim,
            appearance_numeric_dim=appearance_numeric_dim,
        )
        self.atom_numeric_dim = atom_numeric_dim
        self.appearance_numeric_dim = appearance_numeric_dim

    # -- reverse mask --------------------------------------------------------

    def _v4_reverse_mask(self, batch: SGSCTBatchV4) -> Tensor:
        """TRUE_LOCAL_G_C only: precedence + resource_sequence.

        Uses the explicit ``reverse_causal_mask`` when present (1.3.0 bundles);
        otherwise falls back to computing it from ``edge_type`` so V4 can be
        run on legacy 1.2.1 graphs for mask-isolation diagnostics.
        """
        rcm = getattr(batch, "reverse_causal_mask", None)
        if rcm is not None:
            return rcm.bool()
        # Fallback: derive TRUE_LOCAL_G_C from the unified edge-type vocabulary.
        from .sg_sct_data_v1_3 import (
            _PRECEDENCE_TYPE_INDEX,
            _RESOURCE_SEQUENCE_TYPE_INDEX,
        )

        edge_type = batch.edge_type
        return (edge_type == _PRECEDENCE_TYPE_INDEX) | (
            edge_type == _RESOURCE_SEQUENCE_TYPE_INDEX
        )

    # -- block-state refinement seam (identity in V4) ------------------------

    def _refine_block_state(self, block_state: Tensor, batch: SGSCTBatchV4) -> Tensor:
        """Hook between the Appearance Query and its two consumers.

        In V4 this is the **identity function**, so V4's numerics, contracts and
        parameter count are unchanged -- it exists only so a successor version
        (``sg-sct-m1-m2-v4.1``) can insert block-level reasoning between the
        Appearance Query and (a) the member-operation query injection, (b) the
        Branch B appearance state, without duplicating this forward pass.
        """
        return block_state

    # -- oversmoothing diagnostics (forward-only) ---------------------------

    @torch.no_grad()
    def _oversmoothing_diag(self, reverse: Tensor, batch: SGSCTBatch) -> dict[str, float]:
        h = reverse.detach()
        node_count, hidden_dim = h.shape
        diag: dict[str, float] = {}

        # Within-block pairwise cosine (mean over blocks): high => collapsed.
        block_count = batch.symptom_block_count
        if block_count > 0 and batch.symptom_block_node_index is not None:
            b_idx = batch.symptom_block_node_index[0]
            n_idx = batch.symptom_block_node_index[1]
            cos_sum = 0.0
            cos_n = 0
            for b in range(block_count):
                members = n_idx[b_idx == b]
                if members.numel() < 2:
                    continue
                mv = h[members]
                mv = mv / (mv.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                sim = mv @ mv.t()
                m = members.numel()
                # off-diagonal mean
                cos_sum += float(sim.sum().item() - m) / (m * (m - 1))
                cos_n += 1
            diag["within_block_cosine"] = cos_sum / cos_n if cos_n else 0.0
        else:
            diag["within_block_cosine"] = 0.0

        # Dirichlet energy over reverse-causal edges: low => oversmoothed.
        rmask = self._v4_reverse_mask(batch)
        src, dst = batch.edge_index[0][rmask], batch.edge_index[1][rmask]
        if src.numel():
            de = (h[src] - h[dst]).pow(2).sum(dim=-1).mean().item()
        else:
            de = 0.0
        diag["dirichlet_energy"] = de

        # Effective rank of the node embedding matrix (entropy of singular
        # value spectrum): low => collapsed into fewer directions.
        try:
            s = torch.linalg.svdvals(h.float())
            s = s.clamp_min(1e-12)
            p = s / s.sum()
            entropy = (-(p * p.log())).sum().item()
            diag["effective_rank"] = float(entropy)
        except Exception:
            diag["effective_rank"] = 0.0

        return diag

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        batch: SGSCTBatchV4,
        *,
        candidate_batch: GraphGanttAtomicCandidateBatch | None = None,
    ) -> SGSCTOutputV4:
        """Run M1 (via V3), then V4 reverse + Branch A decode + optional Branch B.

        ``candidate_batch`` carries the exact atomic intervention ``r`` per row;
        when supplied, Branch B emits an unconstrained signed-CE prediction per
        row.  Without it, only Branch A (proposal/localization) is produced.
        """
        # M1 (context + causal + gantt fusion + mechanism).  V3's own reverse /
        # decode run too and are discarded -- only the M1 outputs (computed
        # before V3 reverse) are kept.  This reuses M1 without duplicating it
        # and leaves V3 untouched.
        v3 = super().forward(batch)
        fused_nodes = v3.node_embeddings
        anomaly = v3.anomaly_embeddings
        state = v3.state_embeddings
        history = v3.history_embeddings

        # ---- V4 reverse: TRUE_LOCAL_G_C + gated residual ------------------
        reverse_mask = self._v4_reverse_mask(batch)
        reversed_edges = torch.stack([batch.edge_index[1], batch.edge_index[0]])

        block_count = batch.symptom_block_count
        symptom_block_state: Tensor | None = None
        node_query = state[batch.node_graph]
        if block_count > 0 and batch.symptom_block_node_index is not None:
            b_idx = batch.symptom_block_node_index[0]
            n_idx = batch.symptom_block_node_index[1]
            if batch.appearance_type is not None and batch.appearance_features is not None:
                h_A = _attention_pool(fused_nodes[n_idx], b_idx, block_count, self.block_attention)
                e_A = self.appearance_type_embedding(batch.appearance_type)
                x_A = self.appearance_feature_encoder(batch.appearance_features)
                block_query = self.appearance_query_mlp(torch.cat([h_A, e_A, x_A], dim=-1))
            else:
                block_repr = _scatter_mean(fused_nodes[n_idx], b_idx, block_count)
                block_query = self.state_query(block_repr)
            symptom_block_state = self._refine_block_state(block_query, batch)
            if batch.node_symptom_block is not None:
                nb = batch.node_symptom_block
                per_block = symptom_block_state[nb.clamp_min(0)]
                node_query = torch.where(
                    (nb >= 0).unsqueeze(-1), per_block, node_query
                )

        # Initial residual from the M1 fused representation (operation identity).
        reverse = fused_nodes
        diagnostics: list[dict[str, float]] = []
        for i, layer in enumerate(self.reverse_layers):
            delta, _, _ = layer(
                reverse, reversed_edges, batch.edge_type, batch.edge_features, reverse_mask
            )
            gate = torch.sigmoid(self.reverse_gate[i](reverse))  # [N, 1]
            reverse = gate * delta + (1.0 - gate) * reverse
            reverse = reverse + self.reverse_query(node_query)
            diagnostics.append(self._oversmoothing_diag(reverse, batch))

        # ---- Branch A: appearance-CE heads + proposal/localization decode ---
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
        root_legal_mask = batch.candidate_node_mask.bool() & ancestor_mask
        graph_count = state.shape[0]
        if block_count > 0 and batch.symptom_block_neighborhood_mask is not None:
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
        candidate_repr = torch.cat([reverse[safe_anchors], node_query[safe_anchors]], dim=-1)
        root_types = self.root_type_head(candidate_repr)
        confidence = torch.sigmoid(self.confidence_head(candidate_repr).squeeze(-1))
        invalid_anchor = flat_anchors < 0
        root_types = root_types.masked_fill(invalid_anchor.unsqueeze(-1), -torch.inf)
        confidence = confidence.masked_fill(invalid_anchor, 0.0)

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

        # ---- Branch B: atomic intervention effect (unconstrained signed CE) -
        atomic_signed_ce: Tensor | None = None
        if candidate_batch is not None:
            if symptom_block_state is None:
                raise ValueError(
                    "Branch B (atomic effect) requires per-block appearance state "
                    "(symptom_block_count > 0); the input batch has none."
                )
            # The target-conditioned adapter reads appearance block state, graph
            # state, reverse node state and Gantt token embeddings, plus the
            # per-token machine/graph selectors for the routing target readout.
            view = SimpleNamespace(
                symptom_block_state=symptom_block_state,
                state_embeddings=state,
                reverse_embeddings=reverse,
                gantt_embeddings=v3.gantt_embeddings,
            )
            atomic_signed_ce = self.atomic_effect_head(
                view,
                candidate_batch,
                gantt_machine_index=batch.gantt_machine_index,
                gantt_graph=batch.gantt_graph,
            )

        return SGSCTOutputV4(
            node_embeddings=fused_nodes,
            state_embeddings=state,
            history_embeddings=history,
            anomaly_embeddings=anomaly,
            m1_mask_reconstruction=v3.m1_mask_reconstruction,
            m1_transition=v3.m1_transition,
            m1_metric=v3.m1_metric,
            causal_gate_values=v3.causal_gate_values,
            mechanism_mu=v3.mechanism_mu,
            mechanism_nll=v3.mechanism_nll,
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
            per_block_root_logits=per_block_root_logits,
            block_row_symptom_index=block_row_symptom_index,
            symptom_block_state=symptom_block_state,
            node_ce_scores=node_ce,
            edge_ce_scores=edge_ce,
            symptom_block_count=int(block_count),
            atomic_signed_ce=atomic_signed_ce,
            oversmoothing_diagnostics=tuple(diagnostics),
            reverse_causal_mask=reverse_mask,
            identified=False,
            training_status="code_implemented_not_trained",
            model_schema_version=MODEL_SCHEMA_V4,
        )


# ---------------------------------------------------------------------------
# Branch B loss (Part 8): SmoothL1 only, no ranking / no auxiliary
# ---------------------------------------------------------------------------


def compute_branch_b_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    beta: float = 1.0,
) -> Tensor:
    """SmoothL1 (Huber) loss between predicted and target signed CE.

    ``predicted`` is the unconstrained Branch B output (no sigmoid); ``target``
    is the solver-counterfactual signed CE label for the exact intervention.
    No ranking term, no auxiliary term -- the atomic supervision contract is a
    single direct regression.  Defined and wired here; not trained this round.
    """
    if predicted.shape != target.shape:
        raise ValueError("Branch B predicted/target shapes must match")
    return F.smooth_l1_loss(predicted, target, beta=beta, reduction="mean")


# ---------------------------------------------------------------------------
# Version-aware model selection (Part 10)
# ---------------------------------------------------------------------------


def select_model_class(model_schema_version: str) -> type[nn.Module]:
    """Dispatch on model schema version.  Fail-closed on unknown versions.

    ``sg-sct-m1-m2-v3`` -> V3 (frozen), ``sg-sct-m1-m2-v4`` -> V4 (this module).
    """
    if model_schema_version == "sg-sct-m1-m2-v3":
        return SGSCTRootCauseModel
    if model_schema_version == MODEL_SCHEMA_V4:
        return SGSCTRootCauseModelV4
    raise ValueError(f"unknown model schema version: {model_schema_version!r}")


def from_manifest_v4(bundle: Any) -> SGSCTRootCauseModelV4:
    """Instantiate a V4 model from a (1.3.0 or 1.2.1) bundle.

    Reads dimension invariants from the bundle arrays + manifest vocabularies,
    mirroring the V3 ``instantiate_shared_backbone`` helper.  ``hidden_dim`` /
    ``heads`` default to the Phase-2.21 training contract (128 / 8).
    """
    arrays = bundle.arrays
    vocab = bundle.manifest["vocabularies"]
    machine_vocab = int(arrays["model_gantt_machine_index"].max()) + 1
    job_vocab = int(arrays["model_gantt_job_index"].max()) + 1
    time_vocab = int(arrays["model_gantt_time_index"].max()) + 1
    return SGSCTRootCauseModelV4(
        node_numeric_dim=arrays["model_node_numeric"].shape[1],
        node_type_count=len(vocab["node_types"]),
        edge_type_count=len(vocab["model_edge_types"]),
        edge_feature_dim=arrays["model_edge_features"].shape[1],
        gantt_numeric_dim=arrays["model_gantt_numeric"].shape[1],
        trajectory_numeric_dim=arrays["model_trajectory_numeric"].shape[1],
        mechanism_dim=arrays["model_mechanism_values"].shape[1],
        root_type_count=4,
        appearance_type_count=len(vocab.get("appearance_rules", [])) or 10,
        appearance_feature_dim=len(vocab.get("appearance_feature_fields", ())) or 17,
        hidden_dim=128,
        heads=8,
        machine_vocab=machine_vocab,
        job_vocab=job_vocab,
        time_vocab=time_vocab,
        dropout=0.05,
        max_anchors=8,
        max_block_size=6,
        max_path_length=12,
        top_root_candidates=5,
    )


__all__ = [
    "MODEL_SCHEMA_V4",
    "SGSCTBatchV4",
    "SGSCTOutputV4",
    "SGSCTRootCauseModelV4",
    "compute_branch_b_loss",
    "from_manifest_v4",
    "select_model_class",
]
