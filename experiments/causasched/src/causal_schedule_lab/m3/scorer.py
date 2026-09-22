"""Canonical M3 SFT scorer: shared encoder -> usefulness (P(U>0)) + rank head.

VERBATIM extraction (no semantic change):
  - M3ProposalScorer / STOPGate / train_scorer / train_gate / _sample_rank_pairs /
    _rank_metrics / _pairwise_rank_metrics / _classification_metrics /
    _average_precision / _scores_for                <- r2
  - _collect_train_tensors / _train_scorer_fixed / train_scorer_ablation /
    _scores (R4 variant)                            <- R4 (clean 313-d masked ablation)
Internal references rewritten to canonical modules (identical numbers).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .config import (
    EPOCHS,
    GATE_HIDDEN,
    INFEASIBLE_U,
    LAMBDA_CLS,
    LR,
    MEM_FEAT_DIM,
    PROP_FEAT_DIM,
    RANK_PAIRS_PER_STATE,
    SCORER_HIDDEN,
    STATE_FEAT_DIM,
    USE_REG_HEAD,
)


# ---------------------------------------------------------------------------
# M3 Proposal-level SFT scorer + STOP/ACT gate  [r2]
# ---------------------------------------------------------------------------
class M3ProposalScorer(nn.Module):
    """shared encoder -> usefulness_head (logit P(U>0)) + utility_rank_head (score).

    Input = [prop_feats(300) | mem_feats(MEM_DIM) | state_feat(7) broadcast].
    Frozen utility heads (uhat/direct) already live inside prop_feats idx 296/298,
    so the scorer can learn when to trust vs correct the old heads.
    """

    def __init__(self, prop_dim=PROP_FEAT_DIM, mem_dim=MEM_FEAT_DIM, state_dim=STATE_FEAT_DIM,
                 hidden=SCORER_HIDDEN):
        super().__init__()
        self.mem_dim = mem_dim
        in_dim = prop_dim + mem_dim + state_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.usefulness_head = nn.Linear(hidden, 1)   # logit P(U>0)
        self.rank_head = nn.Linear(hidden, 1)         # proposal score
        self.reg_head = nn.Linear(hidden, 1)          # optional predicted U

    def forward(self, prop_feats, mem_feats, state_feat):
        N = prop_feats.shape[0]
        if mem_feats is None or self.mem_dim == 0:
            mem = torch.zeros(N, 0, device=prop_feats.device)
        else:
            mem = mem_feats
        if state_feat.dim() == 1:
            sf = state_feat.unsqueeze(0).expand(N, -1)
        else:
            sf = state_feat                          # already [N, state_dim]
        x = torch.cat([prop_feats, mem, sf], dim=-1)
        h = self.shared(x)
        logit_pos = self.usefulness_head(h).squeeze(-1)   # [N]
        rank = self.rank_head(h).squeeze(-1)              # [N]
        reg = self.reg_head(h).squeeze(-1)                # [N]
        return logit_pos, rank, reg


class STOPGate(nn.Module):
    def __init__(self, state_dim=STATE_FEAT_DIM, hidden=GATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, state_feat):
        return self.net(state_feat).squeeze(0)            # [2]


# ---------------------------------------------------------------------------
# replay tensor collection + pair sampling  [r2]
# ---------------------------------------------------------------------------
def _collect_train_tensors(state_examples, mem_values):
    """Flatten all TRAIN replay proposals.  mem_values: [N,6] or None(zeros).

    (R4 canonical variant: mem_values param; identical r2 semantics when None.)"""
    all_pf, all_mem, all_sf = [], [], []
    all_trueU, all_feasible = [], []
    group_ids = []
    for gi, ex in enumerate(state_examples):
        N = len(ex["metas"])
        all_pf.append(ex["prop_feats"])
        if mem_values is None:
            all_mem.append(torch.zeros(N, MEM_FEAT_DIM, dtype=torch.float32))
        else:
            all_mem.append(mem_values[gi])
        all_sf.append(ex["state_feat"].unsqueeze(0).expand(N, -1))
        all_trueU.append(torch.tensor(ex["true_U"], dtype=torch.float32))
        all_feasible.append(torch.tensor(ex["feasible"], dtype=torch.bool))
        group_ids.extend([gi] * N)
    return (torch.cat(all_pf), torch.cat(all_mem) if True else None,
            torch.cat(all_sf), torch.cat(all_trueU), torch.cat(all_feasible), group_ids)


def _sample_rank_pairs(group_ids, true_U, n_pairs_per_state):
    """Same-state pairs (i,j) with U[i] > U[j], prioritised: best-vs-hard,
    positive-vs-neutral, positive-vs-positive.  Never neg-vs-neg."""
    idx_by_group = {}
    for i, g in enumerate(group_ids):
        idx_by_group.setdefault(g, []).append(i)
    hard_pairs, rest_pairs = [], []
    for g, idxs in idx_by_group.items():
        U = true_U[idxs]
        n = len(idxs)
        pos_loc = [j for j in range(n) if U[j] > 0]
        if not pos_loc:
            continue
        pos_loc_sorted = sorted(pos_loc, key=lambda j: -U[j])
        neutral_loc = [j for j in range(n) if U[j] == 0]
        neg_loc = [j for j in range(n) if U[j] < 0]
        best = idxs[pos_loc_sorted[0]]
        for j in sorted(neg_loc, key=lambda j: -U[j]):
            hard_pairs.append((best, idxs[j]))
        for p_loc in pos_loc_sorted:
            for j in neutral_loc[:2]:
                rest_pairs.append((idxs[p_loc], idxs[j]))
        for a_loc, b_loc in zip(pos_loc_sorted, pos_loc_sorted[1:]):
            rest_pairs.append((idxs[a_loc], idxs[b_loc]))
    cap = n_pairs_per_state * len(idx_by_group)
    pairs = hard_pairs[:cap]
    if len(pairs) < cap:
        pairs += rest_pairs[: cap - len(pairs)]
    return pairs


def _train_scorer_fixed(scorer, pf, mem, sf, trueU, feasible, group_ids, pairs,
                        seed, epochs=EPOCHS, lr=LR):
    """Train the given scorer (313-d, mem present) with fixed pair tensors."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    y_pos = (trueU > 0).float()
    n_pos = int(y_pos.sum().item())
    n_neg = int((~y_pos.bool()).sum().item())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    pi = torch.tensor([a for a, b in pairs], dtype=torch.long) if pairs else None
    pj = torch.tensor([b for a, b in pairs], dtype=torch.long) if pairs else None
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        logit_pos, rank, reg = scorer(pf, mem, sf)
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit_pos, y_pos, pos_weight=pos_weight)
        rank_loss = torch.zeros(())
        if pi is not None:
            si = rank[pi]
            sj = rank[pj]
            rank_loss = torch.clamp(1.0 - (si - sj), min=0.0).mean()
        reg_loss = torch.zeros(())
        if USE_REG_HEAD:
            mask = feasible
            reg_loss = ((reg[mask] - trueU[mask]) ** 2).mean() if mask.any() else torch.zeros(())
        loss = cls_loss + rank_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
        hist.append({"ep": ep, "cls": float(cls_loss), "rank": float(rank_loss),
                     "reg": float(reg_loss), "loss": float(loss)})
    scorer.eval()
    return scorer, hist


def train_scorer(state_examples, use_mem, seed=0, epochs=EPOCHS, lr=LR):
    torch.manual_seed(seed)
    scorer = M3ProposalScorer(mem_dim=MEM_FEAT_DIM if use_mem else 0)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    pf, mem, sf, trueU, feasible, group_ids = _collect_train_tensors(state_examples, use_mem)
    y_pos = (trueU > 0).float()
    n_pos = int(y_pos.sum().item())
    n_neg = int((~y_pos.bool()).sum().item())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    pairs = _sample_rank_pairs(group_ids, trueU, RANK_PAIRS_PER_STATE)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        logit_pos, rank, reg = scorer(pf, mem, sf)
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit_pos, y_pos, pos_weight=pos_weight)
        rank_loss = torch.zeros(())
        if pairs:
            pi = torch.tensor([a for a, b in pairs], dtype=torch.long)
            pj = torch.tensor([b for a, b in pairs], dtype=torch.long)
            si = rank[pi]
            sj = rank[pj]
            rank_loss = torch.clamp(1.0 - (si - sj), min=0.0).mean()
        reg_loss = torch.zeros(())
        if USE_REG_HEAD:
            mask = feasible
            reg_loss = ((reg[mask] - trueU[mask]) ** 2).mean() if mask.any() else torch.zeros(())
        loss = cls_loss + rank_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
        hist.append({"ep": ep, "cls": float(cls_loss), "rank": float(rank_loss),
                     "reg": float(reg_loss), "loss": float(loss)})
    scorer.eval()          # dropout off -> deterministic scoring (was nondeterministic before)
    return scorer, hist, {"n_pos": n_pos, "n_neg": n_neg, "n_pairs": len(pairs)}


def train_scorer_ablation(state_examples):
    """Build scorer_mem (real progressive mem) and scorer_masked (zero mem) with
    IDENTICAL architecture and initialization (deepcopy), same pairs, same seed.

    (R4 canonical: the clean ablation -- masked arm trains on the SAME pair
    tensors, so Memory-vs-noMemory is the only difference.)"""
    _mem_list = [torch.zeros_like(ex["prog_mem_feats"]) for ex in state_examples]
    pf, _, sf, trueU, feasible, group_ids = _collect_train_tensors(state_examples, _mem_list)
    mem_real = torch.cat([ex["prog_mem_feats"] for ex in state_examples])
    pairs = _sample_rank_pairs(group_ids, trueU, RANK_PAIRS_PER_STATE)
    torch.manual_seed(0)
    scorer_mem = M3ProposalScorer(mem_dim=MEM_FEAT_DIM)
    scorer_masked = copy.deepcopy(scorer_mem)          # identical initial weights
    scorer_mem, _ = _train_scorer_fixed(scorer_mem, pf, mem_real, sf, trueU, feasible,
                                        group_ids, pairs, seed=0)
    zero_mem = torch.zeros_like(mem_real)
    scorer_masked, _ = _train_scorer_fixed(scorer_masked, pf, zero_mem, sf, trueU, feasible,
                                           group_ids, pairs, seed=0)
    return scorer_mem, scorer_masked, {"n_pos": int((trueU > 0).sum()),
                                       "n_pairs": len(pairs),
                                       "n_states": len(state_examples)}


def train_gate(state_examples, seed=0, epochs=EPOCHS, lr=LR):
    torch.manual_seed(seed)
    gate = STOPGate()
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    sf_all = torch.stack([ex["state_feat"] for ex in state_examples])
    y_act = torch.tensor([1.0 if (ex["true_U"] > 0).any() else 0.0
                          for ex in state_examples], dtype=torch.float32)
    n_act = int(y_act.sum().item())
    pos_weight = torch.tensor((len(y_act) - n_act) / max(n_act, 1), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        logits = gate(sf_all)                      # [Nstates, 2]
        loss = torch.nn.functional.cross_entropy(
            logits, y_act.long(), weight=torch.tensor([1.0, float(pos_weight)]))
        loss.backward()
        opt.step()
    gate.eval()
    return gate


# ---------------------------------------------------------------------------
# scoring + evaluation  [r2]
# ---------------------------------------------------------------------------
def _scores(scorer, ex, mem):
    with torch.no_grad():
        logit_pos, rank, _ = scorer(ex["prop_feats"], mem, ex["state_feat"])
    return logit_pos, rank


def _scores_for(scorer, ex, use_mem, lambda_cls=LAMBDA_CLS):
    with torch.no_grad():
        mem = ex["mem_feats"] if use_mem else None
        logit_pos, rank, _ = scorer(ex["prop_feats"], mem, ex["state_feat"])
        fused = rank + lambda_cls * logit_pos
    return fused.numpy(), rank.numpy(), logit_pos.numpy()


def _rank_metrics(state_examples, score_fn, Ks=(1, 3, 5, 10, 20)):
    pos_recall = {K: 0 for K in Ks}
    best_recall = {K: 0 for K in Ks}
    miss_hit = {K: 0 for K in Ks}
    n_states = 0
    n_best = 0
    n_miss = 0
    n_pos = 0
    for ex in state_examples:
        scores = score_fn(ex)
        U = ex["true_U"]
        pos = [i for i in range(len(U)) if U[i] > 0]
        if not pos:
            continue
        n_states += 1
        n_pos += len(pos)
        best = max(pos, key=lambda i: U[i])
        missed = [i for i in pos if ex["old_base"][i] <= 0]
        n_best += 1
        n_miss += len(missed)
        order = np.argsort(-scores)
        for K in Ks:
            topk = set(order[:K].tolist())
            if any(i in topk for i in pos):
                pos_recall[K] += 1
            if best in topk:
                best_recall[K] += 1
            miss_hit[K] += sum(1 for i in missed if i in topk)
    return {
        "n_states": n_states, "n_positive": n_pos, "n_missed": n_miss, "n_best": n_best,
        "positive_recall": {str(K): (pos_recall[K] / max(n_states, 1)) for K in Ks},
        "best_recall": {str(K): (best_recall[K] / max(n_best, 1)) for K in Ks},
        "missed_hit": {str(K): miss_hit[K] for K in Ks},
        "missed_recall": {str(K): (miss_hit[K] / max(n_miss, 1)) for K in Ks},
    }


def _pairwise_rank_metrics(state_examples, score_fn):
    """same-state ranking accuracy + hard-negative rejection (directive Sec 16/22).

    ranking_accuracy: over all within-state pairs (i,j) with U[i] > U[j],
        fraction score[i] > score[j].
    hard_negative_rejection: over all (missed_positive, hard_negative) pairs in the
        same state -- m: true_U>0 & old_base<=0; h: old_base>0 & true_U<=0 --
        fraction score[m] > score[h] (the Sec-16 "most critical training pair").
    """
    n_rank = 0
    n_rank_correct = 0
    n_hn = 0
    n_hn_correct = 0
    for ex in state_examples:
        scores = score_fn(ex)
        U = ex["true_U"]
        old = ex["old_base"]
        N = len(U)
        pos_idx = np.array([i for i in range(N) if U[i] > 0])
        if len(pos_idx) == 0:
            continue
        for i in pos_idx:
            lower = U < U[i]
            c = int(lower.sum())
            if c:
                n_rank += c
                n_rank_correct += int((scores[i] > scores[lower]).sum())
        missed = [int(i) for i in pos_idx if old[i] <= 0]
        hardneg = [j for j in range(N) if old[j] > 0 and U[j] <= 0]
        for m in missed:
            for h in hardneg:
                n_hn += 1
                if scores[m] > scores[h]:
                    n_hn_correct += 1
    return {
        "ranking_accuracy": float(n_rank_correct / max(n_rank, 1)),
        "n_rank_pairs": int(n_rank),
        "hard_negative_rejection": float(n_hn_correct / max(n_hn, 1)),
        "n_hardneg_pairs": int(n_hn),
    }


def _classification_metrics(state_examples, score_fn_logit_pos):
    y = []
    p = []
    for ex in state_examples:
        U = ex["true_U"]
        logit = score_fn_logit_pos(ex)
        y.extend([1.0 if U[i] > 0 else 0.0 for i in range(len(U))])
        p.extend([float(logit[i]) for i in range(len(U))])
    y = np.array(y)
    p = np.array(p)
    pred = (p > 0).astype(float)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": float(precision), "recall": float(recall),
        "positive_recall": float(recall), "false_negative_rate": float(fn / max(tp + fn, 1)),
        "pr_auc": float(_average_precision(y, p)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
    }


def _average_precision(y, scores):
    order = np.argsort(-scores)
    y = y[order]
    precisions = []
    hits = 0
    total = 0
    for i, yv in enumerate(y):
        if yv > 0:
            hits += 1
            precisions.append(hits / (i + 1))
        total += 1
    return float(np.mean(precisions)) if precisions else 0.0


def _old_base_arr(prop_feats):
    """frozen utility base per proposal (single->idx296, pair->idx298)."""
    if isinstance(prop_feats, np.ndarray):
        return np.where(prop_feats[:, 299] < 0.5, prop_feats[:, 296], prop_feats[:, 298])
    return torch.where(prop_feats[:, 299] < 0.5, prop_feats[:, 296], prop_feats[:, 298])