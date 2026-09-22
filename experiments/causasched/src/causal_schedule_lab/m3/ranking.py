"""Canonical M3 ranking: Stage-1 wide recall + Stage-2 dedicated reranker.

VERBATIM extraction from R4 (no semantic change):
  - _proposal_roles / _struct_priority_mask / wide_pool / M3ProposalReranker /
    _rerank_feats_all / _sample_rerank_pairs / _fit_reranker / train_reranker /
    rerank_order / wide_and_final_metrics / s0_slice_metrics /
    frozen_memory_channel_audit  <- r4
`r2.`/`r4.` internal refs rewritten to canonical module equivalents (identical
code / numbers).  `_scores` = scorer._scores; `_old_base_arr` = gate._old_base_arr.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .config import (
    FINAL_TOP_K,
    LAMBDA_CLS,
    MARGIN_HARD,
    MARGIN_NORMAL,
    MEM_FEAT_DIM,
    MISSED_OVERSAMPLE,
    RERANK_EPOCHS,
    RERANK_HIDDEN,
    RERANK_LR,
    RERANK_PAIRS_PER_STATE,
    ROLE_N,
    SCORER_HIDDEN,
    STATE_FEAT_DIM,
    WIDE_K_CLS,
    WIDE_K_RANK,
)
from .gate import _old_base_arr
from .scorer import _scores


# ---------------------------------------------------------------------------
# scoring helpers  [R4]
# ---------------------------------------------------------------------------
def _proposal_roles(ex):
    """role index per proposal: single CONTRIBUTOR/ENABLER -> 0/1; pair CC/EE/CE -> 2/3/4."""
    idx = []
    for rt in ex["role"]:
        idx.append({"CONTRIBUTOR": 0, "ENABLER": 1, "CC": 2, "EE": 3, "CE": 4}.get(rt, 0))
    return np.array(idx, dtype=np.int64)


def _struct_priority_mask(ex, logit_pos, rank):
    """Structural-priority candidates: contributor/enabler proposals with ANY
    positive signal from a head (frozen base, usefulness, or rank).  This is the
    only mechanism that stops a scorer slip from completely losing them."""
    N = len(ex["metas"])
    old = _old_base_arr(ex["prop_feats"].numpy())
    lp = logit_pos.numpy()
    rk = rank.numpy()
    m = np.zeros(N, dtype=bool)
    for k in range(N):
        if ex["role"][k] in ("CONTRIBUTOR", "ENABLER", "CC", "EE", "CE") and \
                (old[k] > 0 or lp[k] > 0 or rk[k] > 0):
            m[k] = True
    return m


def wide_pool(ex, logit_pos, rank, k_cls=WIDE_K_CLS, k_rank=WIDE_K_RANK):
    """Stage-1 wide recall: C_wide = top-K_cls usefulness U top-K_rank rank U
    structural-priority candidates."""
    N = len(ex["metas"])
    cls_top = set(torch.topk(logit_pos, min(k_cls, N)).indices.tolist()) if N else set()
    rank_top = set(torch.topk(rank, min(k_rank, N)).indices.tolist()) if N else set()
    S = _struct_priority_mask(ex, logit_pos, rank)
    sp = set(np.where(S)[0].tolist())
    pool = sorted(cls_top | rank_top | sp)
    return pool, {"cls": len(cls_top), "rank": len(rank_top), "struct": len(sp),
                  "union": len(pool)}


# ---------------------------------------------------------------------------
# Stage-2 Dedicated Reranker  [R4]
# ---------------------------------------------------------------------------
class M3ProposalReranker(nn.Module):
    """Scores proposals inside the WIDE pool: [scorer-hidden | rank | logit |
    old_base | role-onehot | state_feat | mem6] -> MLP -> scalar."""

    def __init__(self, repr_dim=SCORER_HIDDEN, state_dim=STATE_FEAT_DIM,
                 mem_dim=MEM_FEAT_DIM, role_n=ROLE_N, hidden=RERANK_HIDDEN):
        super().__init__()
        self.in_dim = repr_dim + 3 + role_n + state_dim + mem_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)          # [N]


def _rerank_feats_all(scorer, ex, mem):
    """[N, D] reranker feature matrix for ALL proposals in the state."""
    N = len(ex["metas"])
    if N == 0:
        return None
    with torch.no_grad():
        x = torch.cat([ex["prop_feats"], mem,
                       ex["state_feat"].unsqueeze(0).expand(N, -1)], dim=-1)
        h = scorer.shared(x)
        logit_pos, rank, _ = scorer(ex["prop_feats"], mem, ex["state_feat"])
    sf = ex["state_feat"].unsqueeze(0).expand(N, -1)
    roles = torch.from_numpy(_proposal_roles(ex)).long()
    role_oh = torch.nn.functional.one_hot(roles, num_classes=ROLE_N).float()
    old = torch.from_numpy(_old_base_arr(ex["prop_feats"].numpy())).float().unsqueeze(1)
    return torch.cat([h, rank.unsqueeze(1), logit_pos.unsqueeze(1), old,
                      role_oh, sf, mem.float()], dim=-1)


def _sample_rerank_pairs(ex, scorer, mem, rng):
    """Directive Section 26 primary pairs within a single state.

    margins: missed-positive vs hard pairs = MARGIN_HARD; all others MARGIN_NORMAL.
    Hard pairs are oversampled in train_reranker (pair-level, no identity emb)."""
    with torch.no_grad():
        logit_pos, rank = _scores(scorer, ex, mem)
    U = ex["true_U"]
    old = _old_base_arr(ex["prop_feats"].numpy())
    N = len(U)
    pos = [k for k in range(N) if U[k] > 0]
    if not pos:
        return [], {}, {"missed_pos": 0, "hardneg": 0}
    missed = [k for k in pos if old[k] <= 0]
    neutral = [k for k in range(N) if U[k] == 0]
    neg = [k for k in range(N) if U[k] <= 0]
    hardneg = [k for k in range(N) if old[k] > 0 and U[k] <= 0]
    best = max(pos, key=lambda k: float(U[k]))
    rk_sorted = sorted(neg, key=lambda k: -float(rank[k]))[:15]
    lp_sorted = sorted(neg, key=lambda k: -float(logit_pos[k]))[:15]
    pairs, margins = [], []
    for m in missed:
        for h in hardneg:
            pairs.append((m, h)); margins.append(MARGIN_HARD)
        for h in rk_sorted:
            pairs.append((m, h)); margins.append(MARGIN_HARD)
        for h in lp_sorted:
            pairs.append((m, h)); margins.append(MARGIN_HARD)
    for h in hardneg:
        pairs.append((best, h)); margins.append(MARGIN_HARD)
    for p in pos:
        for j in neutral[:2]:
            pairs.append((p, j)); margins.append(MARGIN_NORMAL)
    po = sorted(pos, key=lambda k: -float(U[k]))
    for a, b in zip(po, po[1:]):
        pairs.append((a, b)); margins.append(MARGIN_NORMAL)
    counts = {"missed_pos": len(missed), "hardneg": len(hardneg),
              "n_pairs_raw": len(pairs), "n_high_rank_neg": len(rk_sorted),
              "n_high_logit_fp": len(lp_sorted), "n_neutral": len(neutral)}
    return pairs, margins, counts


def _fit_reranker(X_all, pgi, pii, pjj, marg, gi_off):
    """Train a fresh M3ProposalReranker on prebuilt feature matrix + pair tensors.
    Used for BOTH the real-memory arm and the masked-memory arm (identical pairs,
    identical seed / epochs / lr) so the only difference is the memory channel."""
    torch.manual_seed(0)
    model = M3ProposalReranker()
    opt = torch.optim.AdamW(model.parameters(), lr=RERANK_LR, weight_decay=1e-4)
    hist = []
    for ep in range(RERANK_EPOCHS):
        opt.zero_grad()
        scores = model(X_all)
        si = scores[pgi * 0 + gi_off[pgi] + pii]
        sj = scores[pgi * 0 + gi_off[pgi] + pjj]
        loss = torch.clamp(marg - (si - sj), min=0.0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append({"ep": ep, "loss": float(loss)})
    model.eval()
    return model, hist


def train_reranker(state_examples, scorer, mem_values):
    """Train M3ProposalReranker on missed-positive-centric pairs.

    For every state: within-state pairs per Section 26, missed-vs-hard pair
    margin = MARGIN_HARD, all other pairs MARGIN_NORMAL; missed-positive pairs
    oversampled MISSED_OVERSAMPLE-fold (pair-level only, no identity embedding).

    Returns (reranker, hist, counts, rr_data) where rr_data carries the identical
    pair tensors so the masked-channel arm can retrain on the SAME pairs."""
    rng = random.Random(0)
    X_parts = []
    offsets = []
    state_gi_list = []
    pair_gi, pair_i, pair_j, pair_marg = [], [], [], []
    counts = {"n_states": 0, "n_states_with_pos": 0, "total_missed": 0,
              "total_hardneg": 0, "n_missed_pairs": 0, "n_normal_pairs": 0}
    offset = 0
    for gi, ex in enumerate(state_examples):
        f = _rerank_feats_all(scorer, ex, mem_values[gi])
        N = len(ex["metas"])
        if f is None or N == 0:
            continue
        pairs, margins, c = _sample_rerank_pairs(ex, scorer, mem_values[gi], rng)
        counts["n_states"] += 1
        counts["total_missed"] += c["missed_pos"]
        counts["total_hardneg"] += c["hardneg"]
        if not pairs:
            continue
        counts["n_states_with_pos"] += 1
        state_gi_list.append(gi)
        # oversample missed-vs-hard pairs
        hard_idx = [k for k, m in enumerate(margins) if m == MARGIN_HARD]
        if hard_idx:
            pairs = pairs + [pairs[k] for k in hard_idx] * (MISSED_OVERSAMPLE - 1)
            margins = margins + [MARGIN_HARD] * len(hard_idx) * (MISSED_OVERSAMPLE - 1)
        # cap per state (keeps training balanced)
        cap = RERANK_PAIRS_PER_STATE
        pairs = pairs[:cap]
        margins = margins[:cap]
        X_parts.append(f)
        offsets.append((offset, offset + N))
        for (i, j), m in zip(pairs, margins):
            pair_gi.append(len(X_parts) - 1)
            pair_i.append(i)
            pair_j.append(j)
            pair_marg.append(m)
            if m > MARGIN_NORMAL:
                counts["n_missed_pairs"] += 1
            else:
                counts["n_normal_pairs"] += 1
        offset += N
    if not pair_gi:
        raise RuntimeError("reranker: no training pairs")
    X_all = torch.cat(X_parts)
    pgi = torch.tensor(pair_gi, dtype=torch.long)
    pii = torch.tensor(pair_i, dtype=torch.long)
    pjj = torch.tensor(pair_j, dtype=torch.long)
    marg = torch.tensor(pair_marg, dtype=torch.float32)
    gi_off = torch.tensor([o for o, _ in offsets], dtype=torch.long)
    reranker, hist = _fit_reranker(X_all, pgi, pii, pjj, marg, gi_off)
    counts["n_pairs"] = len(pair_gi)
    counts["n_features"] = int(X_all.shape[0])
    rr_data = {"X_all": X_all, "pgi": pgi, "pii": pii, "pjj": pjj,
               "marg": marg, "gi_off": gi_off, "n_features": int(X_all.shape[0]),
               "state_gi_list": list(state_gi_list)}
    return reranker, hist, counts, rr_data


def rerank_order(scorer, reranker, ex, mem):
    """Stage-2 final order: reranker scores over ALL proposals, filtered to the
    Stage-1 wide pool.  Returns (pool, final_order_idx, pool_rerank_scores)."""
    with torch.no_grad():
        logit_pos, rank = _scores(scorer, ex, mem)
    pool, info = wide_pool(ex, logit_pos, rank)
    pool_set = set(pool)
    f = _rerank_feats_all(scorer, ex, mem)
    if f is None or len(pool) == 0:
        return [], [], info
    with torch.no_grad():
        s = reranker(f)
    s_np = s.numpy()
    order_all = list(np.argsort(-s_np))
    final_order = [k for k in order_all if k in pool_set]
    return pool, final_order, {**info, "rerank_top1_in_pool": (final_order[0] if final_order else None)}


def wide_and_final_metrics(state_examples, scorer, reranker, mem_values):
    """summary = {
      wide_pool: missed recall (in pool) + pool size,
      broad_fused: missed recall @20/30/50 over additive fusion (all proposals),
      final: missed/best/positive recall @1..20 over reranked wide pool,
    }"""
    n_states = 0
    n_missed = 0
    n_best = 0
    n_pos = 0
    wide_hit = 0
    wide_hit_s0 = 0
    n_s0 = 0
    n_missed_s0 = 0
    final_miss_hit = {k: 0 for k in (1, 3, 5, 10, 20)}
    final_best_hit = {k: 0 for k in (1, 3, 5, 10, 20)}
    final_pos_hit = {k: 0 for k in (1, 3, 5, 10, 20)}
    broad_miss_hit = {k: 0 for k in (20, 30, 50)}
    pool_sizes = []
    s0_wide_hit = 0
    s0_final_best_hit = {k: 0 for k in (10,)}
    for ex in state_examples:
        U = ex["true_U"]
        old = _old_base_arr(ex["prop_feats"].numpy())
        pos = [k for k in range(len(U)) if U[k] > 0]
        N = len(U)
        if not pos:
            continue
        is_s0 = bool(ex.get("s0", False))
        n_states += 1
        n_pos += len(pos)
        missed = [k for k in pos if old[k] <= 0]
        if is_s0:
            n_s0 += 1
            n_missed_s0 += len(missed)
        n_missed += len(missed)
        best = max(pos, key=lambda k: float(U[k]))
        n_best += 1
        with torch.no_grad():
            logit_pos, rank = _scores(scorer, ex, mem_values[ex["gi"]])
        pool, final_order, pinfo = rerank_order(scorer, reranker, ex, mem_values[ex["gi"]])
        pool_sizes.append(pinfo.get("union", len(pool)))
        pool_set = set(pool)
        wide_hit += sum(1 for k in missed if k in pool_set)
        if is_s0:
            s0_wide_hit += sum(1 for k in missed if k in pool_set)
        # broad fused (additive over all proposals) for @20/30/50
        fused = (rank + LAMBDA_CLS * logit_pos).numpy()
        order_all = list(np.argsort(-fused))
        top20 = set(order_all[:20]); top30 = set(order_all[:30]); top50 = set(order_all[:50])
        broad_miss_hit[20] += sum(1 for k in missed if k in top20)
        broad_miss_hit[30] += sum(1 for k in missed if k in top30)
        broad_miss_hit[50] += sum(1 for k in missed if k in top50)
        for K in (1, 3, 5, 10, 20):
            topk = set(final_order[:K])
            final_miss_hit[K] += sum(1 for k in missed if k in topk)
            if best in topk:
                final_best_hit[K] += 1
            if any(k in topk for k in pos):
                final_pos_hit[K] += 1
        if best in set(final_order[:10]):
            s0_final_best_hit[10] += 1
    return {
        "n_states": n_states, "n_missed": n_missed, "n_best": n_best, "n_positive": n_pos,
        "n_s0_states": n_s0, "n_missed_s0": n_missed_s0,
        "wide_pool_missed_recall": wide_hit / max(n_missed, 1),
        "wide_pool_missed_recall_s0": s0_wide_hit / max(n_missed_s0, 1),
        "wide_pool_size": {"mean": float(np.mean(pool_sizes)),
                           "min": int(min(pool_sizes or [0])),
                           "max": int(max(pool_sizes or [0]))},
        "broad_fused_missed_recall": {str(k): broad_miss_hit[k] / max(n_missed, 1)
                                      for k in (20, 30, 50)},
        "final_missed_recall": {str(k): final_miss_hit[k] / max(n_missed, 1)
                                for k in (1, 3, 5, 10, 20)},
        "final_best_recall": {str(k): final_best_hit[k] / max(n_best, 1)
                              for k in (1, 3, 5, 10, 20)},
        "final_any_positive_recall": {str(k): final_pos_hit[k] / max(n_states, 1)
                                      for k in (1, 3, 5, 10, 20)},
        "final_best_recall_s0_at10": s0_final_best_hit[10] / max(n_s0, 1),
        "s0_final_missed_recall": None,   # filled by caller for S0-only slice if needed
    }


def s0_slice_metrics(state_examples, scorer, reranker, mem_values):
    """Same metrics as wide_and_final_metrics but only over S0 states (n=92).
    Uses shallow copies so the original gi mapping is not mutated."""
    s0s = []
    mv = []
    for i, ex in enumerate(state_examples):
        if ex.get("s0", False):
            c = dict(ex)
            c["gi"] = len(s0s)
            s0s.append(c)
            mv.append(mem_values[i])
    out = wide_and_final_metrics(s0s, scorer, reranker, mv)
    out["n_s0_states_used"] = len(s0s)
    return out


def frozen_memory_channel_audit(state_examples, scorer, reranker, mem_values):
    """Directive Section 15: same scorer, inference with real vs zero mem6.
    Measures how sensitive the frozen model is to the memory channel."""
    n_states = 0
    n_order_change = 0
    delta_scores = []
    top10_overlap = 0
    for ex in state_examples:
        U = ex["true_U"]
        if not (U > 0).any():
            continue
        n_states += 1
        mem_real = mem_values[ex["gi"]]
        mem_zero = torch.zeros_like(mem_real)
        with torch.no_grad():
            logit_r, rank_r = _scores(scorer, ex, mem_real)
            logit_z, rank_z = _scores(scorer, ex, mem_zero)
        fused_r = (rank_r + LAMBDA_CLS * logit_r).numpy()
        fused_z = (rank_z + LAMBDA_CLS * logit_z).numpy()
        delta_scores.append(float(np.mean(np.abs(fused_r - fused_z))))
        o_r = np.argsort(-fused_r)[:10].tolist()
        o_z = np.argsort(-fused_z)[:10].tolist()
        if o_r != o_z:
            n_order_change += 1
        top10_overlap += len(set(o_r) & set(o_z))
    return {
        "n_states": n_states,
        "states_with_order_change": n_order_change,
        "mean_abs_score_delta_on_mem_channel": float(np.mean(delta_scores)) if delta_scores else 0.0,
        "mean_top10_overlap_real_vs_zero": top10_overlap / max(n_states * 10, 1),
    }


def coverage_audit(s0_examples, all_examples, progmem):
    """Directive Section 16 coverage audit on the corrected store."""
    # neighbours stats for all replay queries
    d = progmem.diag_summary()
    d["n_s0_states"] = len(s0_examples)
    d["n_all_states"] = len(all_examples)
    return d