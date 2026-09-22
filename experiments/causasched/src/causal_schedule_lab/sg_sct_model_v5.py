"""SGSCT Causal-Intervention M2 V5 -- output schema (Phase 1).

This module starts with the stable :class:`M2V5Output` contract (spec §16) and
the schema version constant.  The actual :class:`SGSCTRootCauseModelV5`
(M1 + dual-stream M2) is added in Phase 3+; this file keeps the output contract
stable and testable independent of the build.

Output contract (spec §16):

.. code-block:: python

    M2V5Output(
        per_block_node_logits,            # [B, N_total]
        per_block_edge_logits,            # [B, E]
        per_block_edit_logits,            # [B, N_edit]  (legal edits)
        per_block_decision_root_logits,   # [B, N_site]
        per_block_decision_deviation,     # [B, N_site]  (Z_route/Z_seq)
        per_block_relation_gates,         # [B, R]       (appearance-conditioned gates)
        proposals,                        # tuple[CausalInterventionProposal, ...]
        # compatibility only
        per_block_root_logits,            # [B, N]  from V4
    )

``per_block_root_logits`` is kept read-compatible but is **not** the unique
training objective anymore (spec §16).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .ir import Problem, Schedule
from .legal_edit_enumerator import LegalEditEnumerator
from .m2_decision_residual_v1 import build_decision_sites
from .m2_proposal_builder_v1 import build_operator_runtime
from .intervention import ScheduleGraphView
from .m2_v5_schema_v1 import (
    CausalInterventionProposal,
    DecisionSite,
    EditDependency,
    LegalEdit,
)
from .sg_sct_model_v4 import SGSCTRootCauseModelV4, from_manifest_v4

MODEL_SCHEMA_V5 = "sg-sct-m1-m2-v5"

# Relations surfaced through the per-block relation gates (spec §12).
V5_RELATIONS = ("precedence", "resource_sequence", "assignment", "eligibility")

# Deterministic decision-time feature bundle scored by the edit relevance head
# (spec §18, §45-11: no counterfactual "after" info).  Featurisation is exact
# for ROUTE edits; sequencing edits carry no numeric bundle, so they are padded.
EDIT_RELEVANCE_FIELDS = (
    "p_current", "p_target", "delta_p",
    "source_load", "target_load",
    "source_relative_load", "target_relative_load", "flexibility",
)


@dataclass(frozen=True)
class M2RuntimeContext:
    """Executable schedule context required by the V5 proposal path.

    The tensor batch deliberately remains backward compatible with V4.  Facts
    that must never be guessed from anonymous tensors (mode identity, legal
    alternatives and executable schedule assignments) travel in this explicit
    context.  ``block_members`` is ordered by the compiled block ids.
    """

    problem: Problem
    schedule: Schedule
    block_ids: tuple[str, ...]
    block_members: Mapping[str, tuple[str, ...]]
    appearance_scores: Mapping[str, float] | None = None


def build_m2_runtime_context(
    problem: Problem,
    schedule: Schedule,
    appearance: Mapping[str, Any],
    *,
    block_ids: Sequence[str] | None = None,
) -> M2RuntimeContext:
    """Build the non-mock runtime context from the canonical appearance record."""
    rows: dict[str, tuple[str, ...]] = {}
    scores: dict[str, float] = {}
    for row in appearance.get("blocks", ()):
        block = row.get("block", row)
        bid = str(block.get("block_id", row.get("block_id", "")))
        if not bid:
            continue
        members = block.get("operations", block.get("members", ()))
        rows[bid] = tuple(str(item) for item in members)
        score = row.get("score", block.get("score", 0.0))
        scores[bid] = float(score or 0.0)
    ordered = tuple(block_ids) if block_ids is not None else tuple(rows)
    missing = [bid for bid in ordered if bid not in rows]
    if missing:
        raise ValueError(f"appearance is missing compiled blocks: {missing[:5]}")
    return M2RuntimeContext(
        problem=problem,
        schedule=schedule,
        block_ids=ordered,
        block_members={bid: rows[bid] for bid in ordered},
        appearance_scores={bid: scores.get(bid, 0.0) for bid in ordered},
    )


def featurize_legal_edit_relevance(edit: CausalInterventionProposal | object) -> list[float]:
    """Deterministic decision-time vector for one :class:`LegalEdit`.

    Reads only ``edit.features`` (decision-time-visible per spec §45-11); any
    missing key defaults to 0.0 so a SEQUENCE edit (no numeric bundle) still
    yields a fixed-length vector.  The network can only ever re-score edits
    already enumerated (hard-feasibility), so it never hallucinates a machine.
    """
    feats = dict(getattr(edit, "features", ()) or ())
    return [float(feats.get(k, 0.0)) for k in EDIT_RELEVANCE_FIELDS]


def featurize_edit_dependency_pair(
    editor: CausalInterventionProposal | object,
    dependent: CausalInterventionProposal | object,
) -> list[float]:
    """Deterministic decision-time vector for an (editor, dependent) edit pair.

    The dependency head (Patch §4) predicts :math:`P(e_{ij}=1)` -- whether
    applying edit ``editor`` *enables* edit ``dependent`` (e.g. vacating machine
    :math:`M_s` frees a receiver window for another edit onto :math:`M_s`).  The
    input is the concatenation of the two edits' decision-time feature bundles;
    it reads only ``edit.features`` (no counterfactual "after"), mirroring the
    no-leakage contract of the single-edit relevance head (spec §45-11).
    """
    return featurize_legal_edit_relevance(editor) + featurize_legal_edit_relevance(dependent)


@dataclass
class M2V5Output:
    """Stable tensor + proposal output of the V5 M2 model."""

    # per-appearance relevance / edit / root heads
    per_block_node_logits: Tensor | None = None  # [B, N_total]
    per_block_edge_logits: Tensor | None = None  # [B, E]
    per_block_edit_logits: Tensor | None = None  # [B, N_edit]
    per_block_decision_root_logits: Tensor | None = None  # [B, N_site]
    per_block_decision_deviation: Tensor | None = None  # [B, N_site]
    per_block_relation_gates: Tensor | None = None  # [B, R]
    # B4B pilot -- learned transition scorer over role-2 causal edges
    # (utility-guided, NOT a causal probability).  None unless the model was
    # built with use_transition_scorer=True.
    transition_scores: Tensor | None = None  # [B, n_role2_edges]
    # Patch §4 -- dependency head logits over (editor, dependent) edit pairs.
    edit_dependency_logits: Tensor | None = None  # [B, n_pairs]
    proposal_scores: Tensor | None = None  # [B, K], -inf only for absent slots
    decision_sites: tuple[DecisionSite, ...] = ()
    legal_edits: tuple[LegalEdit, ...] = ()
    dependency_pairs: tuple[tuple[str, str], ...] = ()
    proposals: tuple[CausalInterventionProposal, ...] = ()  # R_A = {R_1..R_K}
    # Torch-free causal/operator audit surfaces.  The root logits above remain
    # the trainable M2 distribution; these records are runtime derivations.
    causal_explanation_chains: tuple[object, ...] = ()
    actionable_root_ids: tuple[str, ...] = ()
    decision_candidate_scores: tuple[tuple[str, str, float], ...] = ()
    causal_search_traces: tuple[tuple[str, tuple[str, ...]], ...] = ()

    # auxiliary M1 outputs reused downstream (V4-shaped)
    node_embeddings: Tensor | None = None  # [N, H]
    state_embeddings: Tensor | None = None  # [G, H]
    history_embeddings: Tensor | None = None  # [G, H]
    reverse_embeddings: Tensor | None = None  # [N, H]

    # compatibility output from V4 (read-accessible, not the training target)
    per_block_root_logits: Tensor | None = None  # [B, N]

    # bookkeeping
    symptom_block_count: int = 0
    identified: bool = False
    training_status: str = "code_implemented_not_trained"
    model_schema_version: str = MODEL_SCHEMA_V5

    @property
    def has_proposals(self) -> bool:
        return len(self.proposals) > 0

    def validate_contract(self) -> None:
        """Structural shape checks (spec §16); tensors may be None when a head
        is not requested (e.g. a pure proposal-shadow run)."""
        if self.per_block_root_logits is None:
            # A V5 pass is allowed to skip the compatibility head only when at
            # least one of the real V5 heads or proposals is present.
            if not (self.per_block_node_logits is not None or self.has_proposals):
                raise ValueError(
                    "M2V5Output has neither a V5 head nor the V4 compatibility "
                    "root logits; output is empty"
                )

    def as_compat_v4(self) -> "M2V5Output":
        """Return an output guaranteed to expose ``per_block_root_logits`` if it
        was provided (V4-loaders read it the same way).  No-op copy."""
        return self


class SGSCTRootCauseModelV5(SGSCTRootCauseModelV4):
    """V5 M2 causal-intervention model -- dual-stream + appearance gates.

    Builds on V4 (which subclasses V3 and already gives M1 + Appearance Query +
    the gated TRUE_LOCAL_G_C reverse = **Stream-C**).  V5 adds:

    * **Stream-I** (intervention opportunity): the *same* shared reverse encoder
      run forward over the intervention edge set (eligibility + machine hub:
      ``gf:eligible_resource``, ``gc:machine``) => what receivers/escape/relay
      structures exist around each node.
    * **Appearance-conditioned fusion**: per-node gate
      :math:`g_v^A=\\sigma(MLP([h_v^C,h_v^I,q_A]))`, h_v^A = g*h_v^C + (1-g)*h_v^I.
      The gate is initialised toward the causal/identity stream so the upgrade
      does not randomly perturb V4 behaviour at step 0 (spec §13).
    * **Per-block relation gates** :math:`\\alpha_r^A` over
      {precedence, resource_sequence, assignment, eligibility} for
      debug/audit (spec §12, no ``if A1/elif A4`` hard-coding).

    ``per_block_root_logits`` from V4 is retained read-compatible (spec §16).
    """
    model_schema_version = MODEL_SCHEMA_V5

    def __init__(self, *args, **kwargs):
        self._edge_feature_dim = int(kwargs.get("edge_feature_dim", 4))
        # B4B pilot -- learned transition scorer (feature-flagged, non-destructive).
        # Default OFF so the parameter schema and forward are byte-identical to
        # the frozen V5 (old checkpoints still load, old forward unchanged).
        self._use_transition_scorer = bool(kwargs.pop("use_transition_scorer", False))
        # B4B-R1 ablation mode for the transition head (post-hoc input masking):
        #   "full"        -> current + relation + next + features (default)
        #   "no_relation" -> zero the relation embedding
        #   "no_current"  -> zero the current (effect) node embedding
        #   "next_only"   -> keep only the next (cause) node + features
        self._transition_ablation = "full"
        super().__init__(*args, **kwargs)
        self.relation_count = len(V5_RELATIONS)
        h = self.hidden_dim
        # Per-node dual-stream fusion gate: g=1 => causal path (V4-identity).
        self.dual_fusion = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1)
        )
        last = self.dual_fusion[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 4.0)  # sigmoid(4)=0.982 => mostly causal@init
        # Per-block appearance-conditioned relation gates alpha_r^A, R=4.
        self.relation_gate_mlp = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, self.relation_count)
        )
        # -- Phase 4: per-node/edge/edit relevance heads ----------------------
        # Node relevance spans ALL node types (operation, job, resource/machine,
        # mode) -- machines explicitly carry relevance (spec §15, §45-15).
        self.node_relevance_head = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.edge_relevance_head = nn.Linear(2 * h + self._edge_feature_dim, 1)
        # B4B pilot -- utility-guided learned transition scorer.  Reuses the
        # same node embeddings (h_a), edge features and a tiny relation
        # embedding; scores role-2 causal edges current--relation-->next.  NOT a
        # causal probability (identified=false).  Only built when flagged on.
        self.transition_relation_embedding = None
        self.transition_scorer_head = None
        if self._use_transition_scorer:
            self.transition_relation_embedding = nn.Embedding(2, h)
            self.transition_scorer_head = nn.Sequential(
                nn.Linear(2 * h + self._edge_feature_dim + h, h),
                nn.GELU(),
                nn.Linear(h, 1),
            )
        # Edit relevance scores already-enumerated legal edits from their
        # decision-time feature bundle (spec §17, §18).
        self.edit_relevance_head = nn.Linear(len(EDIT_RELEVANCE_FIELDS), 1)
        # Patch §4 -- dependency head: P(e_ij=1) roots on the concatenation of
        # two edits' decision-time bundles (editor enables dependent).
        self.edit_dependency_head = nn.Sequential(
            nn.Linear(2 * len(EDIT_RELEVANCE_FIELDS), 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        # Decision residual is represented as ||h_decision-h_reference||.  The
        # deterministic reference residual is added at runtime; both routing and
        # sequence sites share the encoder and receive a type embedding.
        self.decision_type_embedding = nn.Embedding(2, h)
        self.decision_reference_head = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h)
        )
        # Frozen, auditable RootScore coefficients: alpha, beta, gamma, lambda.
        self.register_buffer(
            "root_score_weights", torch.tensor([1.0, 1.0, 0.5, 0.25])
        )
        self._runtime_context: M2RuntimeContext | None = None
        self._runtime_node_ids: tuple[str, ...] = ()

    def bind_runtime_context(
        self,
        context: M2RuntimeContext,
        *,
        node_ids: Sequence[str],
    ) -> "SGSCTRootCauseModelV5":
        """Bind real schedule facts used by executable V5 proposals."""
        self._runtime_context = context
        self._runtime_node_ids = tuple(str(item) for item in node_ids)
        return self

    # -- intervention edge mask ----------------------------------------------

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _intervention_type_indices() -> frozenset[int]:
        from .sg_sct_data_v1 import MODEL_EDGE_TYPES

        wanted = {"gf:eligible_resource", "gc:machine"}
        return frozenset(
            idx for idx, name in enumerate(MODEL_EDGE_TYPES) if name in wanted
        )

    def _intervention_mask(self, batch) -> Tensor:
        wanted = self._intervention_type_indices()
        if not wanted:
            raise RuntimeError("intervention edge types not found in MODEL_EDGE_TYPES")
        return torch.isin(batch.edge_type, torch.tensor(sorted(wanted), device=batch.edge_type.device))

    # -- dual-stream forward --------------------------------------------------

    def _per_block_window(
        self, values: Tensor, block_count: int, batch, *, edges: bool
    ) -> Tensor:
        """Broadcast entity scores to every block and add a finite locality bias.

        Locality is a prior, never a visibility boundary: machine/job/mode and
        remote operation nodes remain finite and trainable for every Appearance.
        """
        total = values.shape[0]
        out = values.unsqueeze(0).repeat(block_count, 1)
        if block_count == 0:
            return out
        if not edges and batch.symptom_block_neighborhood_mask is not None:
            prior = batch.symptom_block_neighborhood_mask.to(values.dtype)
            if prior.shape == out.shape:
                out = out + 0.25 * prior
        elif edges and batch.symptom_block_node_index is not None and batch.edge_index.numel():
            b_idx, n_idx = batch.symptom_block_node_index
            src, dst = batch.edge_index
            for b in range(block_count):
                members = n_idx[b_idx == b]
                if members.numel():
                    incident = torch.isin(src, members) | torch.isin(dst, members)
                    out[b] = out[b] + 0.25 * incident.to(values.dtype)
        return out

    def set_transition_ablation(self, mode: str) -> None:
        """Post-hoc input-masking ablation for the transition head (B4B-R1/R2).

        Only meaningful when ``use_transition_scorer`` is on.  ``mode`` is one
        of "full" / "no_relation" / "no_current" / "next_only" / "no_appearance".
        ``no_appearance`` swaps the per-block appearance-conditioned node
        representations ``h_a`` for the global Stream-C reverse embeddings
        ``h_c`` (block-invariant), isolating the appearance-query contribution.
        """
        assert mode in ("full", "no_relation", "no_current", "next_only", "no_appearance"), mode
        self._transition_ablation = mode

    def forward(
        self,
        batch,
        *,
        candidate_batch=None,
        runtime_context: M2RuntimeContext | None = None,
    ) -> M2V5Output:
        # V4 pass: Stream-C reverse (TRUE_LOCAL_G_C) + Appearance Query +
        # compatibility per_block_root_logits + M1 state.
        v4 = super().forward(batch, candidate_batch=candidate_batch)

        h_c = v4.reverse_embeddings  # [N, H]  (Stream-C, gated TRUE_LOCAL)
        fused = v4.node_embeddings  # [N, H]  (M1 fused)
        state = v4.state_embeddings  # [G, H]
        q_A = v4.symptom_block_state  # [B, H] | None
        block_count = int(v4.symptom_block_count)
        n_nodes, h_dim = h_c.shape

        # Per-node appearance condition q_A^v (mirror V4 node_query).
        node_query = state[batch.node_graph] if batch.node_graph.numel() else h_c
        if block_count > 0 and batch.node_symptom_block is not None and q_A is not None:
            nb = batch.node_symptom_block
            per_block = q_A[nb.clamp_min(0)]
            node_query = torch.where((nb >= 0).unsqueeze(-1), per_block, node_query)

        # Stream-I: shared encoder, forward direction over intervention edges.
        h_i = fused
        inter_mask = self._intervention_mask(batch) if batch.edge_index.numel() else None
        for i, layer in enumerate(self.reverse_layers):
            if inter_mask is not None and torch.any(inter_mask):
                delta, _, _ = layer(
                    h_i, batch.edge_index, batch.edge_type,
                    batch.edge_features, inter_mask,
                )
            else:
                delta = h_i
            gate = torch.sigmoid(self.reverse_gate[i](h_i))
            h_i = gate * delta + (1.0 - gate) * h_i
            h_i = h_i + self.reverse_query(node_query)

        # Appearance-conditioned fusion is genuinely per block.  Every node is
        # visible in every block; block membership contributes only a finite bias.
        if block_count > 0 and q_A is not None:
            hc_b = h_c.unsqueeze(0).expand(block_count, -1, -1)
            hi_b = h_i.unsqueeze(0).expand(block_count, -1, -1)
            q_b = q_A.unsqueeze(1).expand(-1, n_nodes, -1)
            g = torch.sigmoid(self.dual_fusion(torch.cat([hc_b, hi_b, q_b], dim=-1)))
            h_a = g * hc_b + (1.0 - g) * hi_b  # [B,N,H]
            per_block_node_logits = self.node_relevance_head(
                torch.cat([h_a, q_b], dim=-1)
            ).squeeze(-1)
            if batch.symptom_block_neighborhood_mask is not None:
                prior = batch.symptom_block_neighborhood_mask.to(per_block_node_logits.dtype)
                if prior.shape == per_block_node_logits.shape:
                    per_block_node_logits = per_block_node_logits + 0.25 * prior
            per_block_node_logits = torch.sigmoid(per_block_node_logits)
        else:
            h_a = h_c.new_empty((0, n_nodes, h_dim))
            per_block_node_logits = h_c.new_empty((0, n_nodes))

        # B4B pilot -- learned transition scorer over role-2 causal edges
        # (utility-guided; NOT a causal probability).  Scored block-by-block:
        #   score(current=src, relation, next=dst | appearance) per role-2 edge.
        transition_scores: Tensor | None = None
        if self._use_transition_scorer and block_count > 0 and batch.edge_index.numel():
            role2_mask = batch.edge_role == 2
            if bool(role2_mask.any()):
                e_src = batch.edge_index[0, role2_mask]
                e_dst = batch.edge_index[1, role2_mask]
                e_type = batch.edge_type[role2_mask]
                e_feat = batch.edge_features[role2_mask]
                # relation index over the 2 causal relations: precedence(7)->0,
                # resource_sequence(8)->1.
                rel_idx = (e_type == 8).long()
                rel_emb = self.transition_relation_embedding(rel_idx)  # [n_role2, H]
                # B4B-R1 post-hoc ablation: mask inputs per mode.  In the search
                # orientation the role-2 edge is (cause=src -> effect=dst); the
                # backward transition is current=effect(dst) -> next=cause(src).
                abl = self._transition_ablation
                zero_rel = torch.zeros_like(rel_emb)
                rows = []
                for b in range(block_count):
                    h_src_b = h_a[b, e_src]   # next (cause)
                    h_dst_b = h_a[b, e_dst]   # current (effect)
                    if abl == "no_appearance":
                        # block-invariant Stream-C reverse embeddings (isolate
                        # the appearance-query / per-block fusion contribution).
                        h_src_b = h_c[e_src]
                        h_dst_b = h_c[e_dst]
                    r_b = rel_emb
                    if abl in ("no_relation", "next_only"):
                        r_b = zero_rel
                    if abl in ("no_current", "next_only"):
                        h_dst_b = torch.zeros_like(h_dst_b)
                    feat = torch.cat([h_src_b, h_dst_b, e_feat, r_b], dim=-1)
                    rows.append(self.transition_scorer_head(feat).squeeze(-1))
                transition_scores = torch.stack(rows, dim=0)  # [B, n_role2]
            else:
                transition_scores = h_c.new_empty((block_count, 0))

        # Edge relevance is evaluated block-by-block to avoid a dense B*E*H
        # materialisation while preserving full-graph visibility.
        per_block_edge_logits: Tensor | None = None
        if block_count > 0 and batch.edge_index.numel():
            src, dst = batch.edge_index
            rows = []
            for b in range(block_count):
                rows.append(self.edge_relevance_head(
                    torch.cat([h_a[b, src], h_a[b, dst], batch.edge_features], dim=-1)
                ).squeeze(-1))
            per_block_edge_logits = torch.stack(rows, dim=0)
            # finite incident-edge prior, never -inf.
            per_block_edge_logits = torch.sigmoid(self._add_edge_locality_prior(
                per_block_edge_logits, batch
            ))
        elif block_count > 0:
            per_block_edge_logits = h_c.new_empty((block_count, 0))

        # Per-block relation gates alpha_r^A.
        rel_gates: Tensor | None = None
        if block_count > 0 and q_A is not None and batch.symptom_block_node_index is not None:
            b_idx = batch.symptom_block_node_index[0]
            n_idx = batch.symptom_block_node_index[1]
            rows = []
            for b in range(block_count):
                members = n_idx[b_idx == b]
                if not members.numel():
                    continue
                # rel_feat[b] = mean over block members of [h_c, h_i, q_A]
                hc_m = h_c[members].mean(0)
                hi_m = h_i[members].mean(0)
                rows.append(torch.cat([hc_m, hi_m, q_A[b]], dim=-1))
            if rows:
                rel_gates = torch.sigmoid(
                    self.relation_gate_mlp(torch.stack(rows, dim=0))
                )

        context = runtime_context or self._runtime_context
        runtime = self._runtime_outputs(
            batch=batch,
            context=context,
            h_a=h_a,
            q_A=q_A,
            state=state,
            node_logits=per_block_node_logits,
        )

        return M2V5Output(
            per_block_node_logits=per_block_node_logits,
            per_block_edge_logits=per_block_edge_logits,
            per_block_edit_logits=runtime["edit_logits"],
            per_block_decision_root_logits=runtime["root_logits"],
            per_block_decision_deviation=runtime["deviation"],
            per_block_relation_gates=rel_gates,
            transition_scores=transition_scores,
            edit_dependency_logits=runtime["dependency_logits"],
            proposal_scores=runtime["proposal_scores"],
            decision_sites=runtime["decision_sites"],
            legal_edits=runtime["legal_edits"],
            dependency_pairs=runtime["dependency_pairs"],
            proposals=runtime["proposals"],
            causal_explanation_chains=runtime["causal_explanation_chains"],
            actionable_root_ids=runtime["actionable_root_ids"],
            decision_candidate_scores=runtime["decision_candidate_scores"],
            causal_search_traces=runtime["causal_search_traces"],
            node_embeddings=fused,
            state_embeddings=state,
            history_embeddings=v4.history_embeddings,
            reverse_embeddings=(h_a.mean(0) if block_count else h_c),
            per_block_root_logits=v4.per_block_root_logits,  # compatibility
            symptom_block_count=block_count,
            identified=False,
            training_status=(
                "runtime_integrated_training_not_started"
                if context is not None else "runtime_context_required_for_executable_proposals"
            ),
            model_schema_version=MODEL_SCHEMA_V5,
        )

    def _add_edge_locality_prior(self, logits: Tensor, batch) -> Tensor:
        """Add a finite per-Appearance incident-edge prior."""
        if batch.symptom_block_node_index is None or not batch.edge_index.numel():
            return logits
        b_idx, n_idx = batch.symptom_block_node_index
        src, dst = batch.edge_index
        out = logits.clone()
        for b in range(logits.shape[0]):
            members = n_idx[b_idx == b]
            if members.numel():
                incident = torch.isin(src, members) | torch.isin(dst, members)
                out[b] = out[b] + 0.25 * incident.to(out.dtype)
        return out

    @staticmethod
    def _node_index(node_ids: Sequence[str], entity_id: str) -> int | None:
        """Resolve canonical IR ids without guessing a node type."""
        candidates = (
            entity_id,
            f"operation:{entity_id}",
            f"resource:{entity_id}",
            f"machine:{entity_id}",
            f"job:{entity_id}",
        )
        lookup = {value: idx for idx, value in enumerate(node_ids)}
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        return None

    def _runtime_outputs(
        self,
        *,
        batch,
        context: M2RuntimeContext | None,
        h_a: Tensor,
        q_A: Tensor | None,
        state: Tensor,
        node_logits: Tensor,
    ) -> dict[str, Any]:
        """Execute root localization and proposal closure inside ``forward``."""
        block_count = int(h_a.shape[0])
        empty = h_a.new_empty((block_count, 0))
        base = {
            "root_logits": empty,
            "deviation": empty,
            "edit_logits": empty,
            "dependency_logits": empty,
            "proposal_scores": empty,
            "decision_sites": (),
            "legal_edits": (),
            "dependency_pairs": (),
            "proposals": (),
            "causal_explanation_chains": (),
            "actionable_root_ids": (),
            "decision_candidate_scores": (),
            "causal_search_traces": (),
        }
        if context is None or block_count == 0:
            return base
        if tuple(context.block_ids) != tuple(context.block_members):
            raise ValueError("runtime context block ordering is not canonical")
        if len(context.block_ids) != block_count:
            raise ValueError(
                f"runtime context has {len(context.block_ids)} blocks, batch has {block_count}"
            )
        if not self._runtime_node_ids:
            raise ValueError("V5 runtime context requires manifest node_ids binding")

        deterministic_sites, _, _ = build_decision_sites(
            context.problem, context.schedule
        )
        sites = tuple(deterministic_sites)
        enum = LegalEditEnumerator(context.problem, context.schedule)
        all_operations = tuple(sorted(context.schedule.assignment_map()))
        edits = enum.enumerate(
            all_operations,
            request_route=True,
            request_seq_swap=True,
            request_seq_insert=True,
            request_timing_shift=True,
        )

        if not sites:
            return {**base, "legal_edits": edits}
        site_indices: list[int] = []
        for site in sites:
            idx = self._node_index(self._runtime_node_ids, site.operation_id)
            if idx is None:
                raise ValueError(f"decision operation absent from graph: {site.operation_id}")
            site_indices.append(idx)
        site_idx = torch.tensor(site_indices, dtype=torch.long, device=h_a.device)
        type_idx = torch.tensor(
            [0 if site.decision_type == "routing" else 1 for site in sites],
            dtype=torch.long,
            device=h_a.device,
        )
        type_embedding = self.decision_type_embedding(type_idx)
        learned_rows: list[Tensor] = []
        for b in range(block_count):
            site_h = h_a[b, site_idx]
            q = q_A[b].expand_as(site_h) if q_A is not None else state[0].expand_as(site_h)
            graph_h = state[0].expand_as(site_h)
            reference = self.decision_reference_head(
                torch.cat([q, graph_h, type_embedding], dim=-1)
            )
            residual = torch.linalg.vector_norm(site_h - reference, dim=-1)
            residual = (residual - residual.mean()) / residual.std(unbiased=False).clamp_min(1e-6)
            learned_rows.append(residual)
        learned_z = torch.stack(learned_rows, dim=0)
        deterministic_z = h_a.new_tensor(
            [max(min(float(site.z_deviation), 5.0), -5.0) for site in sites]
        ).unsqueeze(0)
        deviation = 0.5 * learned_z + 0.5 * deterministic_z

        support: list[int] = []
        for site in sites:
            if site.decision_type == "routing":
                count = sum(
                    e.operation_id == site.operation_id and e.edit_type == "ROUTE"
                    for e in edits
                )
            else:
                pair = {site.predecessor_id, site.successor_id}
                count = sum(
                    e.edit_type.startswith("SEQ_")
                    and bool({e.operation_id, e.left_id, e.right_id} & pair)
                    for e in edits
                )
            support.append(int(count))
        support_t = h_a.new_tensor(support).log1p().unsqueeze(0)
        relevance = node_logits[:, site_idx]

        context_rows: list[Tensor] = []
        for site in sites:
            machine = site.source_machine or site.resource_id
            midx = self._node_index(self._runtime_node_ids, machine) if machine else None
            context_rows.append(
                node_logits[:, midx] if midx is not None else node_logits.new_zeros(block_count)
            )
        context_score = torch.stack(context_rows, dim=1)
        alpha, beta, gamma, lam = self.root_score_weights
        root_logits = (
            alpha * deviation
            + beta * relevance
            + gamma * support_t
            - lam * context_score
        )

        if edits:
            edit_features = h_a.new_tensor(
                [featurize_legal_edit_relevance(edit) for edit in edits]
            )
            raw_edit = self.edit_relevance_head(edit_features).squeeze(-1)
            edit_nodes = torch.tensor(
                [
                    self._node_index(self._runtime_node_ids, edit.operation_id)
                    for edit in edits
                ],
                dtype=torch.long,
                device=h_a.device,
            )
            edit_logits = raw_edit.unsqueeze(0) + node_logits[:, edit_nodes]
        else:
            edit_logits = empty

        dep_pairs: list[tuple[LegalEdit, LegalEdit]] = []
        for editor in edits:
            if editor.edit_type != "ROUTE":
                continue
            for dependent in edits:
                if (
                    dependent.edit_type == "ROUTE"
                    and editor.operation_id != dependent.operation_id
                    and editor.source_machine == dependent.target_machine
                ):
                    dep_pairs.append((editor, dependent))
        if dep_pairs:
            dep_features = h_a.new_tensor(
                [featurize_edit_dependency_pair(a, b) for a, b in dep_pairs]
            )
            dep_base = self.edit_dependency_head(dep_features).squeeze(-1)
            dependency_logits = dep_base.unsqueeze(0).expand(block_count, -1)
        else:
            dependency_logits = empty

        operator_runtime = build_operator_runtime(
            block_ids=context.block_ids,
            block_members=context.block_members,
            decision_sites=sites,
            legal_edits=edits,
            root_scores=root_logits.detach(),
            edit_scores=edit_logits.detach(),
            top_k=3,
            schedule_graph=ScheduleGraphView.from_problem_schedule(
                context.problem, context.schedule
            ),
        )
        proposals = operator_runtime.proposals
        grouped = [
            [p.proposal_score for p in proposals if p.source_block_id == block_id]
            for block_id in context.block_ids
        ]
        max_k = max((len(row) for row in grouped), default=0)
        proposal_scores = h_a.new_full((block_count, max_k), float("-inf"))
        for b, row in enumerate(grouped):
            if row:
                proposal_scores[b, : len(row)] = h_a.new_tensor(row)
        return {
            "root_logits": root_logits,
            "deviation": deviation,
            "edit_logits": edit_logits,
            "dependency_logits": dependency_logits,
            "proposal_scores": proposal_scores,
            "decision_sites": sites,
            "legal_edits": edits,
            "dependency_pairs": tuple((a.edit_id, b.edit_id) for a, b in dep_pairs),
            "proposals": proposals,
            "causal_explanation_chains": operator_runtime.causal_chains,
            "actionable_root_ids": operator_runtime.actionable_root_ids,
            "decision_candidate_scores": operator_runtime.decision_candidate_scores,
            "causal_search_traces": operator_runtime.proposal_search_traces,
        }

    # -- edit relevance scoring ----------------------------------------------

    @torch.no_grad()
    def score_edits(self, edits: Sequence[Any]) -> list[float]:
        """Score an already-enumerated list of :class:`LegalEdit` with the edit
        relevance head from their deterministic decision-time features.

        Never fabricates an edit: it can only re-score what was enumerated
        (spec §17).  Returns softmax-normalised relevance in ``[0, 1]``.
        """
        if not edits:
            return []
        feats = torch.tensor(
            [featurize_legal_edit_relevance(e) for e in edits],
            dtype=torch.float32,
            device=self.edit_relevance_head.weight.device,
        )
        if feats.numel() == 0 or feats.dim() == 1:
            feats = feats.reshape(len(edits), len(EDIT_RELEVANCE_FIELDS))
        logits = self.edit_relevance_head(feats).squeeze(-1)
        probs = logits.softmax(dim=0).tolist()
        return probs

    @torch.no_grad()
    def score_edit_dependencies(
        self, pairs: Sequence[tuple[Any, Any]]
    ) -> list[float]:
        """Dependency head (Patch §4): probability that ``editor`` enables
        ``dependent`` for a list of already-enumerated (editor, dependent) edit
        pairs, from their decision-time feature bundles.

        Returns sigmoid logits-as-probabilities in ``[0, 1]``.  Like
        :meth:`score_edits` this only re-scores enumerated edits -- it never
        fabricates an edit pair.
        """
        if not pairs:
            return []
        feats = torch.tensor(
            [featurize_edit_dependency_pair(a, b) for a, b in pairs],
            dtype=torch.float32,
            device=next(self.edit_dependency_head.parameters()).device,
        )
        feats = feats.reshape(len(pairs), 2 * len(EDIT_RELEVANCE_FIELDS))
        logits = self.edit_dependency_head(feats).squeeze(-1)
        return torch.sigmoid(logits).tolist()


# Future model dispatch (Phase 3+ adds SGSCTRootCauseModelV5 to this module).
# Kept fail-closed so a caller can already select the schema version.
def select_model_class_v5(model_schema_version: str):
    if model_schema_version == MODEL_SCHEMA_V5:
        return SGSCTRootCauseModelV5
    raise ValueError(f"unknown V5 model schema version: {model_schema_version!r}")


def from_manifest_v5(
    bundle: Any,
    *,
    runtime_context: M2RuntimeContext | None = None,
    use_identity_embeddings: bool = True,
    use_transition_scorer: bool = False,
) -> SGSCTRootCauseModelV5:
    """Instantiate the V5 dual-stream model from a (1.3.0/1.2.1) bundle.

    Mirrors V4's dimension derivation (identical kwargs), then builds the V5
    subclass so the twin-stream + appearance-gate heads ride on the same M1
    backbone.  Does not modify the frozen V4 module.

    ``use_identity_embeddings=False`` builds a *shared* (cross-instance) model:
    the three absolute-ID embeddings (machine/job/time) are omitted so the
    parameter schema no longer depends on per-instance vocabularies.
    """
    from .sg_sct_model_v4 import from_manifest_v4 as _mv4  # noqa: F401 (guard)

    arrays = bundle.arrays
    vocab = bundle.manifest["vocabularies"]
    machine_vocab = int(arrays["model_gantt_machine_index"].max()) + 1
    job_vocab = int(arrays["model_gantt_job_index"].max()) + 1
    time_vocab = int(arrays["model_gantt_time_index"].max()) + 1
    model = SGSCTRootCauseModelV5(
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
        use_identity_embeddings=use_identity_embeddings,
        use_transition_scorer=use_transition_scorer,
    )
    model._runtime_node_ids = tuple(bundle.manifest["id_spaces"]["node_ids"])
    if runtime_context is not None:
        model.bind_runtime_context(
            runtime_context,
            node_ids=model._runtime_node_ids,
        )
    return model


def from_manifest_v5_shared(
    bundle: Any,
    *,
    runtime_context: M2RuntimeContext | None = None,
) -> SGSCTRootCauseModelV5:
    """Build the cross-instance *shared* M2 (no absolute-ID embeddings).

    The returned model has a parameter schema that is identical for every
    instance: machine/job/time vocabularies no longer enter the model, so the
    same ``state_dict`` is loadable across all 14 TRAIN + 3 VAL states.
    """
    return from_manifest_v5(
        bundle,
        runtime_context=runtime_context,
        use_identity_embeddings=False,
    )


def from_manifest_v5_shared_transition(
    bundle: Any,
    *,
    runtime_context: M2RuntimeContext | None = None,
) -> SGSCTRootCauseModelV5:
    """B4B pilot -- shared M2 with the learned transition scorer enabled.

    Same cross-instance shared schema as ``from_manifest_v5_shared`` plus the
    feature-flagged utility-guided transition head (``transition_scores``).
    """
    return from_manifest_v5(
        bundle,
        runtime_context=runtime_context,
        use_identity_embeddings=False,
        use_transition_scorer=True,
    )


def run_m2_v5_schedule(
    problem: Problem,
    schedule: Schedule,
    appearance: Mapping[str, Any],
    *,
    case_id: str = "runtime",
    model: SGSCTRootCauseModelV5 | None = None,
    device: str = "cpu",
) -> tuple[M2V5Output, SGSCTRootCauseModelV5, Any]:
    """Canonical real-schedule M2 entrypoint.

    Compilation, full-graph model inference, root localization and executable
    macro-proposal construction happen in one call.  Callers do not invoke the
    proposal builder separately.
    """
    from .sg_sct_data_v1_3 import (
        compile_sg_sct_input_v1_3,
        to_sg_sct_batch_v1_3,
    )

    bundle = compile_sg_sct_input_v1_3(
        problem, schedule, appearance, case_id=case_id
    )
    block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
    context = build_m2_runtime_context(
        problem, schedule, appearance, block_ids=block_ids
    )
    if model is None:
        model = from_manifest_v5(bundle, runtime_context=context)
    else:
        model.bind_runtime_context(
            context,
            node_ids=bundle.manifest["id_spaces"]["node_ids"],
        )
    model = model.to(device)
    model.eval()
    batch = to_sg_sct_batch_v1_3(bundle, device=device)
    with torch.no_grad():
        output = model(batch, runtime_context=context)
    output.validate_contract()
    return output, model, bundle


__all__ = [
    "MODEL_SCHEMA_V5",
    "V5_RELATIONS",
    "EDIT_RELEVANCE_FIELDS",
    "M2RuntimeContext",
    "M2V5Output",
    "SGSCTRootCauseModelV5",
    "build_m2_runtime_context",
    "featurize_legal_edit_relevance",
    "featurize_edit_dependency_pair",
    "select_model_class_v5",
    "from_manifest_v5",
    "from_manifest_v5_shared",
    "from_manifest_v5_shared_transition",
    "run_m2_v5_schedule",
]
