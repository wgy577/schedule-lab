"""B5.1 -- Causal Influence Attribution (Phase-1 minimal closed loop).

Scope (frozen, see docs/T1_MODEL_B5_CAUSAL_INFLUENCE_ATTRIBUTION_DESIGN.md):
  * c(u|A)  = c_prior(u)  -- DETERMINISTIC physics tensor, NOT a learnable head
              (Q3, §1.1 non-identifiability: c and e must not both be free).
  * e(u,v|state) = sigmoid(edge_transmission_head(...))  -- the ONLY learned
              propagation parameter in Phase 1 (Q2).
  * r_u(A)  = differentiable reverse message passing over role-2 causal edges,
              anchored at appearance A's members (Q4).
  * I_uv    = OFF (Phase 2).  Reasoner / Explorer authority = untouched.

The encoder is the shared M2 V5 encoder, reused verbatim by subclassing
``SGSCTRootCauseModelV5``.  Only ``edge_transmission_head`` is new; every other
parameter is the shared theta.  Two feature flags drive the Gate 3 / Gate 4
ablations without editing model code at eval time:
  * use_appearance=False  -> block-invariant node reps (h_c) + zeroed appearance
                             context.  ONLY the anchor init still varies by block
                             (pure topology baseline; Gate 3 = full must beat it).
  * use_physics=False     -> zero the 4 GC edge-physics columns, keep topology +
                             relation embedding (Gate 4).

identified = false throughout.  This is attribution-model learnability, not a
causal identification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .sg_sct_model_v5 import SGSCTRootCauseModelV5, from_manifest_v5
from .sg_sct_data_v1 import GC_EDGE_CONTINUOUS_FIELDS

# role-2 causal edge types (see MODEL_EDGE_TYPES): precedence=7, resource_seq=8.
_PRECEDENCE_TYPE = 7
_RESOURCE_SEQUENCE_TYPE = 8
_N_EDGE_PHYSICS = len(GC_EDGE_CONTINUOUS_FIELDS)  # == 4


# ---------------------------------------------------------------------------
# c_prior  --  deterministic node self-effect (Q3, Phase-1 frozen)
# ---------------------------------------------------------------------------
def compute_c_prior(problem, schedule, node_ids: Sequence[str]) -> Tensor:
    """Deterministic c_prior(u) in [0,1] aligned to ``node_ids`` order.

    Node self-anomaly physics (NO learnable parameter -- see §1.1):
      * queueing wait  = start - earliest_possible_start
                         (earliest_possible = max(op.release, job.release,
                          max predecessor end)); how long u sat idle when it
                          could already have run.  A1/A4 quantity family.
      * criticality    = 1 - cp_slack/makespan  (on/near critical path =
                          more load-bearing self-anomaly).

    c_prior(u) = 0.5 * wait_norm(u) + 0.5 * criticality(u), both in [0,1],
    wait normalised by the per-state max wait.  Non-operation nodes (machine /
    job / mode) get 0.0 -- they are never legal roots (candidate_node_mask
    excludes them) and carry no self-anomaly.

    The value is a plain tensor with requires_grad=False.  It enters the
    propagation as a fixed multiplier; no gradient ever reaches it.
    """
    # operation end times from the schedule
    end_of: dict[str, float] = {}
    start_of: dict[str, float] = {}
    for a in schedule.assignments:
        end_of[a.operation_id] = float(a.end)
        start_of[a.operation_id] = float(a.start)

    job_release = {j.id: float(getattr(j, "release", 0) or 0) for j in problem.jobs}

    makespan = float(schedule.makespan) if schedule.makespan else 1.0
    makespan = max(makespan, 1.0)

    op_wait: dict[str, float] = {}
    op_crit: dict[str, float] = {}
    for op in _iter_operations(problem):
        start = start_of.get(op.id)
        if start is None:
            continue
        pred_end = 0.0
        for pred in op.predecessors:
            pe = end_of.get(pred)
            if pe is not None:
                pred_end = max(pred_end, pe)
        earliest = max(
            float(getattr(op, "release", 0) or 0),
            job_release.get(op.job_id, 0.0),
            pred_end,
        )
        op_wait[op.id] = max(0.0, start - earliest)
        # criticality via realised tail is expensive; use the schedule slack
        # proxy start-based:  operations that finish at the makespan tail are
        # more critical.  cp_slack proxy = makespan - end (no negative).
        op_crit[op.id] = _clip01(1.0 - (makespan - end_of[op.id]) / makespan)

    max_wait = max(op_wait.values()) if op_wait else 0.0
    max_wait = max(max_wait, 1e-9)

    values = torch.zeros(len(node_ids), dtype=torch.float32)
    for i, nid in enumerate(node_ids):
        if not nid.startswith("operation:"):
            continue
        op_id = nid[len("operation:"):]
        w = op_wait.get(op_id)
        if w is None:
            continue
        wait_norm = w / max_wait
        crit = op_crit.get(op_id, 0.0)
        values[i] = 0.5 * wait_norm + 0.5 * crit
    values.requires_grad_(False)
    return values


def _iter_operations(problem):
    # Operations are a flat list on the Problem; jobs carry only metadata.
    yield from problem.operations


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# B5.1 attribution model
# ---------------------------------------------------------------------------
@dataclass
class B5AttributionOutput:
    per_block_candidate_score: Tensor  # [B, N] = r_u(A), in [0,1]
    per_block_edge_transmission: Tensor  # [B, E_causal] = e(u,v|state), in [0,1]
    causal_edge_index: Tensor  # [2, E_causal] forward (cause->effect)
    used_appearance: bool
    used_physics: bool


class SGSCTAttributionModelB5(SGSCTRootCauseModelV5):
    """Shared M2 encoder + learned edge-transmission head + frozen-c propagation.

    Adds exactly one parameter group beyond the shared V5 theta:
    ``edge_transmission_head`` (+ a tiny relation embedding + appearance
    projection).  ``c_prior`` is supplied at call time as a fixed tensor.
    """

    def __init__(self, *args, **kwargs):
        # B5.1 propagation depth (radius-2 neighbourhood -> a few hops suffice).
        self._b5_hops = int(kwargs.pop("b5_hops", 4))
        self._b5_gamma = float(kwargs.pop("b5_gamma", 0.9))
        self._b5_appearance_ctx_dim = int(kwargs.pop("b5_appearance_ctx_dim", 8))
        # T1-B5 Phase 13: appearance-conditioned propagation adapter.  When ON,
        # appearance DIRECTLY modulates each causal edge's transmission via a
        # low-rank gated residual e = sigmoid(base_logit + gamma_A . edge_key) --
        # never an ignorable concat feature.  OFF reproduces the round-1 head.
        self._b5_use_adapter = bool(kwargs.pop("b5_adapter", False))
        self._b5_mod_dim = int(kwargs.pop("b5_mod_dim", 16))
        super().__init__(*args, **kwargs)
        h = self.hidden_dim
        # relation embedding for the 2 causal relations (prec / resource_seq).
        self.b5_transmission_relation = nn.Embedding(2, h)
        # project the raw appearance_features [.,17] into a small block context.
        # This lives on the e-path (never touches c_prior), so appearance
        # conditioning is genuine and ablatable.
        self.b5_appearance_proj = nn.Linear(
            self._appearance_feature_dim(), self._b5_appearance_ctx_dim
        )
        # e(u,v|state) head:
        #   cat([ h_a[b,src], h_a[b,dst], edge_physics(4), rel_emb(h), app_ctx ])
        in_dim = 2 * h + _N_EDGE_PHYSICS + h + self._b5_appearance_ctx_dim
        self.edge_transmission_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, 1)
        )
        if self._b5_use_adapter:
            # ---- appearance-conditioned propagation adapter (ONE shared theta) ----
            # base logit over topology+physics only (no app_ctx in the input):
            self.b5_base_head = nn.Sequential(
                nn.Linear(2 * h + _N_EDGE_PHYSICS + h, h), nn.GELU(), nn.Linear(h, 1)
            )
            # per-edge key (shared across blocks): physics + relation -> [mod]
            self.b5_edge_key = nn.Sequential(
                nn.Linear(_N_EDGE_PHYSICS + h, self._b5_mod_dim), nn.GELU(),
                nn.Linear(self._b5_mod_dim, self._b5_mod_dim),
            )
            # per-block modulation gamma_A from (app_ctx, q_mod); q_mod = linear of
            # the block's state query q_A so the model's OWN block identity drives
            # the propagation weights.
            self.b5_query_mod = nn.Linear(h, self._b5_mod_dim)
            self.b5_mod_head = nn.Sequential(
                nn.Linear(self._b5_appearance_ctx_dim + self._b5_mod_dim, self._b5_mod_dim),
                nn.GELU(), nn.Linear(self._b5_mod_dim, self._b5_mod_dim),
            )
        # ablation flags (Gate 3 / Gate 4); default full.
        self._b5_use_appearance = True
        self._b5_use_physics = True

    def _appearance_feature_dim(self) -> int:
        # appearance_features is [B, F_a]; F_a is fixed by the data schema (17).
        return 17

    def set_b5_ablation(self, *, use_appearance: bool = True, use_physics: bool = True) -> None:
        self._b5_use_appearance = bool(use_appearance)
        self._b5_use_physics = bool(use_physics)

    # -- encoder reuse: reproduce the shared V5 h_a fusion exactly ------------
    def _encode_h_a(self, batch):
        """Return (h_a[B,N,H], h_c[N,H], q_A[B,H]|None, block_count).

        This mirrors SGSCTRootCauseModelV5.forward lines that build h_a, using
        the SAME inherited encoder parameters (reverse_layers / reverse_gate /
        reverse_query / dual_fusion).  A unit test asserts that running the
        inherited ``node_relevance_head`` on this h_a reproduces the frozen V5
        ``per_block_node_logits`` bit-for-bit.
        """
        if getattr(self, 'e2e_fast_encoder', False):
            from .m3.e2e_encoder import encode_many
            # CPU collection supplies CPU batches. GPU replay uses encode_many
            # on a group of raw CPU batches, rather than this one-state entry.
            return encode_many(self, [batch], batch.node_numeric.device)[0]
        v4 = super(SGSCTRootCauseModelV5, self).forward(batch)
        h_c = v4.reverse_embeddings
        fused = v4.node_embeddings
        state = v4.state_embeddings
        q_A = v4.symptom_block_state
        block_count = int(v4.symptom_block_count)
        n_nodes, h_dim = h_c.shape

        node_query = state[batch.node_graph] if batch.node_graph.numel() else h_c
        if block_count > 0 and batch.node_symptom_block is not None and q_A is not None:
            nb = batch.node_symptom_block
            per_block = q_A[nb.clamp_min(0)]
            node_query = torch.where((nb >= 0).unsqueeze(-1), per_block, node_query)

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

        if block_count > 0 and q_A is not None:
            hc_b = h_c.unsqueeze(0).expand(block_count, -1, -1)
            hi_b = h_i.unsqueeze(0).expand(block_count, -1, -1)
            q_b = q_A.unsqueeze(1).expand(-1, n_nodes, -1)
            g = torch.sigmoid(self.dual_fusion(torch.cat([hc_b, hi_b, q_b], dim=-1)))
            h_a = g * hc_b + (1.0 - g) * hi_b  # [B,N,H]
        else:
            h_a = h_c.new_empty((0, n_nodes, h_dim))
        return h_a, h_c, q_A, block_count

    def attribution_forward(self, batch, c_prior: Tensor, *, encoded=None) -> B5AttributionOutput:
        """Compute r_u(A) for every candidate under every appearance block.

        Honours the ablation flags:
          * use_appearance=False -> node reps become block-invariant h_c and the
            appearance context vector is zeroed (only the anchor init still
            varies per block -> pure-topology baseline).
          * use_physics=False -> the 4 causal-edge physics columns are zeroed.
        """
        # T2-D runtime may provide the exact frozen encoding it also caches for
        # root latents.  This removes a duplicate full shared-encoder forward per
        # uncached state without changing attribution math.
        h_a, h_c, q_A, block_count = (self._encode_h_a(batch)
                                      if encoded is None else encoded)
        n_nodes = h_c.shape[0]
        device = h_c.device

        if getattr(self, 'e2e_fast_encoder', False):
            from .m3.e2e_attribution import attribution
            return attribution(self,batch,c_prior,(h_a,h_c,q_A,block_count))

        # role-2 causal edges (forward: cause -> effect).
        role2 = batch.edge_role == 2
        causal_ei = batch.edge_index[:, role2]  # [2, Ec]
        causal_type = batch.edge_type[role2]
        causal_feat = batch.edge_features[role2]  # [Ec, 4]
        e_src = causal_ei[0]  # cause (u)
        e_dst = causal_ei[1]  # effect (v)
        n_causal = int(causal_ei.shape[1])

        use_app = self._b5_use_appearance
        use_phys = self._b5_use_physics

        if not use_phys:
            causal_feat = torch.zeros_like(causal_feat)

        # relation index: precedence(7)->0, resource_sequence(8)->1.
        rel_idx = (causal_type == _RESOURCE_SEQUENCE_TYPE).long()
        rel_emb = self.b5_transmission_relation(rel_idx)  # [Ec, H]

        # appearance context per block (e-path only).
        app_feats = batch.appearance_features  # [B, F_a]
        if use_app and app_feats is not None and app_feats.numel():
            app_ctx = self.b5_appearance_proj(app_feats)  # [B, ctx]
        else:
            app_ctx = torch.zeros(
                block_count, self._b5_appearance_ctx_dim, device=device
            )

        # node representations for the e-head: h_a (appearance-conditioned) or,
        # under the no_appearance ablation, block-invariant h_c.
        if use_app:
            node_rep = h_a  # [B, N, H]
        else:
            node_rep = h_c.unsqueeze(0).expand(block_count, -1, -1)

        cand_mask = batch.candidate_node_mask  # [N] bool

        scores = h_c.new_zeros((block_count, n_nodes))
        transmissions = h_c.new_zeros((block_count, max(n_causal, 0)))

        if n_causal == 0 or block_count == 0:
            return B5AttributionOutput(
                per_block_candidate_score=scores,
                per_block_edge_transmission=transmissions,
                causal_edge_index=causal_ei,
                used_appearance=use_app,
                used_physics=use_phys,
            )

        b_idx = batch.symptom_block_node_index[0]
        n_idx = batch.symptom_block_node_index[1]
        gamma = self._b5_gamma

        for b in range(block_count):
            rep = node_rep[b]  # [N, H]
            if self._b5_use_adapter:
                # appearance-conditioned propagation: base logit + low-rank gated
                # residual from (app_ctx, q_mod) so e(u,v|state,A) genuinely
                # depends on the block's appearance -- never an ignorable concat.
                base_in = torch.cat(
                    [
                        rep[e_src],           # cause embedding
                        rep[e_dst],           # effect embedding
                        causal_feat,          # 4 physics cols (zeroed if ablated)
                        rel_emb,              # relation embedding
                    ],
                    dim=-1,
                )
                base_logit = self.b5_base_head(base_in).squeeze(-1)  # [Ec]
                if use_app:
                    edge_key = self.b5_edge_key(torch.cat([causal_feat, rel_emb], dim=-1))  # [Ec,mod]
                    q_mod = self.b5_query_mod(q_A[b])  # [mod]
                    gamma_A = self.b5_mod_head(torch.cat([app_ctx[b], q_mod], dim=-1))  # [mod]
                    logit = base_logit + (edge_key * gamma_A.unsqueeze(0)).sum(-1)
                else:
                    logit = base_logit
                e_uv = torch.sigmoid(logit)  # [Ec]
                transmissions[b] = e_uv
            else:
                # e(u,v|state) for every causal edge under block b.
                e_in = torch.cat(
                    [
                        rep[e_src],           # cause embedding
                        rep[e_dst],           # effect embedding
                        causal_feat,          # 4 physics cols (zeroed if ablated)
                        rel_emb,              # relation embedding
                        app_ctx[b].unsqueeze(0).expand(n_causal, -1),
                    ],
                    dim=-1,
                )
                e_uv = torch.sigmoid(self.edge_transmission_head(e_in).squeeze(-1))  # [Ec]
                transmissions[b] = e_uv

            # anchor init: r_v(A) = 1 for A's members, else 0.
            members = n_idx[b_idx == b]
            r = h_c.new_zeros(n_nodes)
            if members.numel():
                r[members] = 1.0

            # reverse message passing: influence flows effect -> cause.
            # r_u += c(u) * e(u,v) * r_v * gamma  (v = effect = e_dst, u = cause = e_src)
            for _hop in range(self._b5_hops):
                msg = e_uv * r[e_dst] * gamma  # per causal edge
                agg = h_c.new_zeros(n_nodes)
                agg.index_add_(0, e_src, msg)
                r_new = r + c_prior.to(device) * agg
                r = r_new
            scores[b] = r

        # only candidates carry a score (others masked to 0 for ranking).
        if cand_mask is not None:
            scores = scores * cand_mask.to(scores.dtype).unsqueeze(0)

        return B5AttributionOutput(
            per_block_candidate_score=scores,
            per_block_edge_transmission=transmissions,
            causal_edge_index=causal_ei,
            used_appearance=use_app,
            used_physics=use_phys,
        )


def from_manifest_b5(bundle, **kwargs) -> SGSCTAttributionModelB5:
    """Build the shared-schema B5.1 attribution model (no absolute-ID embeddings).

    Mirrors ``from_manifest_v5``'s dimension derivation verbatim but instantiates
    the B5 subclass; ``use_identity_embeddings=False`` so the parameter schema is
    the cross-instance shared theta (loadable across all TRAIN + VAL states).
    """
    arrays = bundle.arrays
    vocab = bundle.manifest["vocabularies"]
    machine_vocab = int(arrays["model_gantt_machine_index"].max()) + 1
    job_vocab = int(arrays["model_gantt_job_index"].max()) + 1
    time_vocab = int(arrays["model_gantt_time_index"].max()) + 1
    model = SGSCTAttributionModelB5(
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
        use_identity_embeddings=False,
        **kwargs,
    )
    model._runtime_node_ids = tuple(bundle.manifest["id_spaces"]["node_ids"])
    return model
