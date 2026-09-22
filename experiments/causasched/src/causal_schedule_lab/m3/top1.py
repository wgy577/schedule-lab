"""Canonical Top-1 utility selection SFT (R6) + STOP calibration / OOD (R7).

Directive R7 (T1-M3-STOP-CALIBRATION-AND-OOD-GENERALIZATION-R7): STOP relative-
scale calibration by margin supervision (frozen proposal backbone in Phase A,
small-LR last-layer Phase B only if the hard DPPaulli margin gate still fails),
TRAIN-only state augmentation (source = oracle replay / R6 policy walk / bounded
FixedDecisionReplay exploration, unique (instance_id, state_hash) + provenance),
selector mem6 channel dropout (p_drop = 0.5), masked-channel equivalence with the
SAME weights, VAL no-backward decomposition.  GRPO still forbidden this round.

Directive R8 (T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION-R8).  Blocker is
re-defined as POOL_ARGMAX_ERROR: exists positive P* with argmax_i score(P_i) !=
argmax_i true_U(P_i).  Primary objective = best-vs-hardest-competitor argmax loss
(online hard-negative mining, utility-weighted, clipped); original Top1 CE kept as
auxiliary; STOP head FROZEN (diagnostic only, no further STOP calibration).  New
cross-instance internal validation: fixed 3-fold by INSTANCE (all model selection
on TRAIN14 held-out instances only; formal VAL evaluated once, no tuning).  Memory
kept but confidence-gated (g_mem from retrieval quality).  GRPO still forbidden.

Directive R6 -- train the state-wise top-1 selector supervised, WITHOUT GRPO.

Group            : key = (instance_id, state_hash).  Each group is one state
                   (balance unit = state; each state contributes one L_top1 term).
Action set       : WIDE pool ∪ STOP, where STOP is a first-class listwise action
                   (R3 act_primary gate is NOT final authority here).
Label            : best_idx = argmax_i true_U(P_i) if max pool true_U > 0 else STOP.
Primary objective: L_top1 = CrossEntropy([score(P_1..P_M), score_STOP]) ->
                   argmax accuracy, NOT Pearson.  (§4-§8)
Score base       : R5 canonical per-proposal representation (frozen scorer hidden
                   + rank + usefulness + old-utility + role-oh + state + mem6),
                   consumed by a selector prop head warm-started from the R5
                   utility reranker; STOP head is fresh over state_feat + pool
                   statistics.  M2 is untouched.  (§2-§3, §9)
Auxiliary        : L_pair with weight w = clip(|U_i-U_j|/UA_UTILITY_SCALE,
                   UA_MIN_W, UA_MAX_W), fixed lambda_pair=0.3 (<1, no sweep).
Hard states      : DPpaulli10a + TOP1_FAILURE_STATE (has positive, and the R5
                   current top-1 has true_U<=0) oversampled by state multiplicity
                   -- NO state identity embedding.  (§14-§15)
Metrics          : state-wise top1 / positive-state / STOP-state acc, selected
                   true_U mean/median, oracle-best true_U mean, top1 regret
                   (max true_U - selected true_U), best-positive recall@1/3/5/10,
                   missed@10, Memory masked ablation (regression).
Closed loop B6   : S_t -> M2 -> Reasoner -> Wide Recall -> Top1 SFT selector ->
                   Proposal/STOP -> FixedDecisionReplay -> re-diagnose (HORIZON=5,
                   no CP-SAT).  Baselines B0-B6 (§20-§23).

identified=false, formal_test_access=0, Formal TEST SEALED.
"""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch
from torch import nn

from causal_schedule_lab.validation import schedule_hash
from causal_schedule_lab.intervention import ScheduleGraphView

from .config import (
    MEM_FEAT_DIM,
    MEM_TOP_N_STATES,
    RERANK_HIDDEN,
    ROLE_N,
    SCORER_HIDDEN,
    STATE_FEAT_DIM,
)
from .proposal_features import (
    _edits_for, _execute_step, old_base_of, proposal_identity, state_feature_vec,
)
from .ranking import _rerank_feats_all, wide_pool
from .rollout import _b5_summary
from .scorer import _scores
from . import config as C
from .memory import ProgressiveMemory   # symbol import: the bare module name
                                        # "memory" trips the no-legacy firewall

# R5 rerank feature matrix column layout (see ranking._rerank_feats_all):
#   [0:SCORER_HIDDEN) = h          (shared-encoder hidden)
#   [SCORER_HIDDEN]   = rank
#   [SCORER_HIDDEN+1] = logit_pos
#   [SCORER_HIDDEN+2] = old_base
#   then role one-hot / state_feat / mem6
COL_RANK = SCORER_HIDDEN
COL_LOGIT = SCORER_HIDDEN + 1
COL_OLD = SCORER_HIDDEN + 2

_N_POOL_SCALE = 100.0

# prop-head feature dimension (matches M3ProposalReranker.in_dim)
PROP_FEAT_DIM = SCORER_HIDDEN + 3 + ROLE_N + STATE_FEAT_DIM + MEM_FEAT_DIM  # 277


def _pool_stats_from(F_pool: torch.Tensor):
    """[1,5] = [n_pool/100, max_old, mean_old, max_rank, max_logit] over the pool."""
    if F_pool is None or len(F_pool) == 0:
        return torch.zeros(1, C.TO1_STOP_POOL_STAT_DIM, dtype=torch.float32)
    old = F_pool[:, COL_OLD]
    rank = F_pool[:, COL_RANK]
    logit = F_pool[:, COL_LOGIT]
    return torch.tensor(
        [[float(len(F_pool)) / _N_POOL_SCALE,
          float(old.max()), float(old.mean()),
          float(rank.max()), float(logit.max())]],
        dtype=torch.float32)


class M3Top1Selector(nn.Module):
    """Listwise Top-1 selector over (WIDE pool ∪ STOP).

    prop_head : same architecture as the R5 M3ProposalReranker network so its
                weights can be warm-started from p1['reranker'].net (initialization
                only).  stop_head : fresh MLP over [state_feat | pool_stats].
    forward(F_pool[M,277], state_feat[7], pool_stats[1,5]) -> (prop[M], stop[1]).
    prop_scores(F) scores an arbitrary [N,277] matrix (used for recall metrics /
    missed@10 where full-state ranking is needed).
    """

    def __init__(self):
        super().__init__()
        self.prop_head = nn.Sequential(
            nn.Linear(PROP_FEAT_DIM, RERANK_HIDDEN), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(RERANK_HIDDEN, RERANK_HIDDEN), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(RERANK_HIDDEN, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(STATE_FEAT_DIM + C.TO1_STOP_POOL_STAT_DIM, C.TO1_STOP_HIDDEN),
            nn.GELU(),
            nn.Linear(C.TO1_STOP_HIDDEN, 1),
        )

    def warm_start_from(self, reranker):
        """Copy R5 utility-reranker weights into prop_head (guarded on shape)."""
        if reranker is None:
            return False
        try:
            self.prop_head.load_state_dict(reranker.net.state_dict())
            return True
        except Exception:  # noqa: BLE001 - shape mismatch -> keep random init
            return False

    def prop_scores(self, F):
        return self.prop_head(F).squeeze(-1)          # [N]

    def forward(self, F_pool, state_feat, pool_stats=None):
        prop_logits = self.prop_scores(F_pool)        # [M]
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        stop_in = torch.cat([state_feat.reshape(1, -1),
                             pool_stats.reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1)
        return prop_logits, self.stop_head(stop_in).squeeze(0)   # ([M], [1])


# ---------------------------------------------------------------------------
# group construction
# ---------------------------------------------------------------------------
def build_top1_groups(state_examples, scorer, mem_values, reranker=None,
                      dpp_substr="DPpaulli", rng=None, gate_mem=False):
    """One action group per state.

    Precomputes R5 rerank features once per state (frozen scorer), the WIDE pool,
    the Top-1 label (index within pool, or M=STOP), pool statistics, the R5
    current pick (for TOP1_FAILURE_STATE), and the weighted |ΔU| pair cache.

    gate_mem (R8 §23-24): when True, scale the selector's mem6 channel by the
    state's memory-reliability gate g_mem = ex["mem_gate"] (default 1.0).  All
    R6/R7 reproduction paths keep gate_mem=False so their numbers stay bit-identical.
    """
    if rng is None:
        rng = random.Random(0)
    groups = []
    dist_proposals = []
    n_zero_pool = 0
    n_pos_states = 0
    n_stop_target_states = 0
    n_failure = 0
    n_dpp = 0
    r5_pick_positive = 0
    r5_pick_matched_best = 0

    for gi, ex in enumerate(state_examples):
        N = len(ex["metas"])
        if N == 0:
            continue
        mem = mem_values[gi]
        with torch.no_grad():
            logit_pos, rank = _scores(scorer, ex, mem)
        pool, _info = wide_pool(ex, logit_pos, rank)
        F_all = _rerank_feats_all(scorer, ex, mem)     # [N,277]
        if F_all is None:
            continue
        if gate_mem:
            gm = float(ex.get("mem_gate", 1.0))
            if gm < 1.0:
                F_all = F_all.clone()
                F_all[..., MEM_CHAN_SLICE] *= gm       # g_mem·mem_embedding (§24)
        F_pool = F_all[pool] if pool else None
        U = ex["true_U"]
        feas = ex["feasible"]
        pos_full = [k for k in range(N) if feas[k] and U[k] > 0]
        pos_in_pool = [k for k in pool if feas[k] and U[k] > 0]
        has_pos_full = len(pos_full) > 0
        n_pos_states += 1 if has_pos_full else 0

        M = len(pool)
        dist_proposals.append(M)
        if M == 0:
            n_zero_pool += 1
            pos_in_pool = []

        if pos_in_pool:
            best_in_pool = max(pos_in_pool, key=lambda k: float(U[k]))
            target_within_pool = pool.index(best_in_pool)
            is_stop_target = False
            max_U_pool = float(U[best_in_pool])
        else:
            target_within_pool = M
            is_stop_target = True
            max_U_pool = 0.0
            n_stop_target_states += 1

        # current R5 pick (utility reranker argmax within pool) for the failure flag
        current_pick_U = None
        if F_pool is not None and reranker is not None:
            with torch.no_grad():
                s_r = reranker(F_pool)
            cur = int(s_r.argmax().item())
            current_pick_U = float(U[pool[cur]])
        elif F_pool is not None:
            cur = int(np.argmax(ex["old_base"][pool]))
            current_pick_U = float(U[pool[cur]])
        if has_pos_full and current_pick_U is not None:
            r5_pick_positive += 1 if current_pick_U > 0 else 0
            if pos_in_pool:
                best_in_pool = max(pos_in_pool, key=lambda k: float(U[k]))
                r5_pick_matched_best += 1 if current_pick_U == float(U[best_in_pool]) else 0

        is_failure = bool(has_pos_full and (current_pick_U is None or current_pick_U <= 0))
        is_dpp = dpp_substr in ex["iid"]
        n_failure += 1 if is_failure else 0
        n_dpp += 1 if is_dpp else 0

        # auxiliary weighted |ΔU| pair cache over pool ∪ STOP(0); skip zero-diff
        pairs = []
        if M >= 1:
            items = [(k, float(U[k])) for k in pool] + [(M, 0.0)]
            candidates = []
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    d = items[a][1] - items[b][1]
                    if abs(d) < 1e-6:
                        continue
                    hi, lo = (a, b) if d > 0 else (b, a)
                    w = float(np.clip(abs(d) / C.UA_UTILITY_SCALE, C.UA_MIN_W, C.UA_MAX_W))
                    candidates.append((hi, lo, w))
            rng.shuffle(candidates)
            pairs = candidates[: C.TO1_PAIRS_PER_STATE]

        groups.append({
            "gi": gi,
            "iid": ex["iid"], "state_hash": ex["state_hash"],
            "ex": ex, "mem": mem,
            "N": N, "M": M, "pool": list(pool),
            "F_all": F_all, "F_pool": F_pool,
            "pool_stats": _pool_stats_from(F_pool),
            "target_within_pool": target_within_pool,
            "is_stop_target": is_stop_target,
            "has_pos_full": has_pos_full,
            "pos_full": list(pos_full),
            "pos_in_pool": list(pos_in_pool),
            "max_U_full": float(max((float(U[k]) for k in pos_full), default=0.0)),
            "max_U_pool": max_U_pool,
            "oracle_best_full": (int(max(pos_full, key=lambda k: float(U[k]))) if pos_full else None),
            "is_failure": is_failure,
            "is_dpp": is_dpp,
            "current_pick_U": current_pick_U,
            "pairs": pairs,
        })

    stats = {
        "n_groups": len(groups),
        "n_prop_total": int(sum(len(g["ex"]["metas"]) for g in groups)),
        "pool_size_hist": {"min": min(dist_proposals) if dist_proposals else 0,
                           "max": max(dist_proposals) if dist_proposals else 0,
                           "mean": float(np.mean(dist_proposals)) if dist_proposals else 0.0},
        "n_pos_states": n_pos_states,
        "n_stop_target_states": n_stop_target_states,
        "n_zero_pool": n_zero_pool,
        "n_failure_states": n_failure,
        "n_dpp_states": n_dpp,
        "pos_state_frac": float(n_pos_states / len(groups)) if groups else 0.0,
        "stop_target_frac": float(n_stop_target_states / len(groups)) if groups else 0.0,
        "r5_pick_positive": r5_pick_positive,
        "r5_pick_matched_best": r5_pick_matched_best,
    }
    return groups, stats


def _sel_logit(sel_out, within_pool_idx, M):
    prop_logits, stop_logit = sel_out
    if within_pool_idx == M:
        return stop_logit[0:1]
    return prop_logits[within_pool_idx:within_pool_idx + 1]


def _action_logits(sel_out, M):
    prop_logits, stop_logit = sel_out
    return torch.cat([prop_logits, stop_logit.reshape(1)], dim=-1)   # [M+1]


def _pair_loss(g, sel_out):
    """Weighted hinge on utility-ranked pairs: w * max(0, 1 - (s_hi - s_lo))."""
    if not g["pairs"]:
        return torch.zeros((), dtype=torch.float32)
    M = g["M"]
    sh, sl, w = [], [], []
    for hi, lo, ww in g["pairs"]:
        sh.append(_sel_logit(sel_out, hi, M))
        sl.append(_sel_logit(sel_out, lo, M))
        w.append(ww)
    s_hi = torch.cat(sh)
    s_lo = torch.cat(sl)
    wt = torch.tensor(w, dtype=torch.float32)
    return (wt * torch.clamp(1.0 - (s_hi - s_lo), min=0.0)).mean()


def _stop_logits_of(selector, g):
    """STOP-only logits for an (empty-pool) group."""
    stop_in = torch.cat([g["ex"]["state_feat"].reshape(1, -1),
                         g["pool_stats"].reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1)
    with torch.no_grad():
        return selector.stop_head(stop_in)[0:1]


def train_top1_sft(state_examples, scorer, mem_values, reranker=None, seed=0):
    """State-balanced Top-1 CE (+ auxiliary L_pair) over the TRI5 train states.

    Balance unit = state: each state contributes one L_top1 term per occurrence.
    Hard states (DPpaulli10a + TOP1_FAILURE_STATE) are repeated by multiplicity
    (TO1_HARD_OVERSAMPLE) -- no identity embedding.
    """
    rng = random.Random(seed)
    groups, gstats = build_top1_groups(state_examples, scorer, mem_values,
                                       reranker=reranker, rng=rng)
    print(f"[top1] groups={gstats['n_groups']} pos_states={gstats['n_pos_states']} "
          f"stop_target={gstats['n_stop_target_states']} failure={gstats['n_failure_states']} "
          f"dpp={gstats['n_dpp_states']}", flush=True)

    selector = M3Top1Selector()
    ws = selector.warm_start_from(reranker)
    print(f"[top1] prop_head warm-started from R5 utility reranker: {ws}", flush=True)

    opt = torch.optim.AdamW(selector.parameters(), lr=C.TO1_LR, weight_decay=1e-4)
    hist = []
    fail_gis = [i for i, g in enumerate(groups) if g["is_failure"]]
    n_aux = C.TO1_HARD_OVERSAMPLE - 1

    for ep in range(C.TO1_EPOCHS):
        selector.train()
        order = list(range(len(groups)))
        order += fail_gis * n_aux           # hard-state multiplicity (state-balanced)
        rng.shuffle(order)
        total_ce = 0.0
        total_pair = 0.0
        n_terms = 0
        for gi in order:
            g = groups[gi]
            if g["F_pool"] is not None:
                prop, stop = selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
            else:
                prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
            logits = _action_logits((prop, stop), g["M"])       # [M+1]
            target = torch.tensor([g["target_within_pool"]], dtype=torch.long)
            ce = nn.functional.cross_entropy(logits.reshape(1, -1), target)
            pair = _pair_loss(g, (prop, stop))
            loss = ce + C.TO1_LAMBDA_PAIR * pair
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            total_ce += float(ce.detach())
            total_pair += float(pair.detach())
            n_terms += 1
        selector.eval()
        acc = top1_accuracy(groups, selector)
        hist.append({"ep": ep, "ce": total_ce / max(n_terms, 1),
                     "pair": total_pair / max(n_terms, 1), "top1_acc": acc["all"]})
        if ep in (0, C.TO1_EPOCHS - 1) or ep % 5 == 4:
            print(f"[top1] ep {ep} ce={hist[-1]['ce']:.4f} pair={hist[-1]['pair']:.4f} "
                  f"acc={acc['all']:.3f}", flush=True)
    return selector, hist, groups, gstats


# ---------------------------------------------------------------------------
# R16: pool-relative LISTWISE ranking SFT (T1-M3-POOL-LISTWISE-RANKING-SFT)
# ---------------------------------------------------------------------------
def _u_tilde(U):
    """§7 utility transform: sign(U)·log1p(|U|) — bounded, monotonic, order-preserving.
    STOP has U=0 -> U_tilde=0 (its natural pool-utility position, §10)."""
    return torch.sign(U) * torch.log1p(U.abs())


@torch.no_grad()
def _r16_epoch_metrics(selector, groups, keys=(1, 3, 5, 10, 20, 32)):
    """§18 pool-relative ranking metrics over prebuilt R16 groups (FULL gated pool + STOP).

    Per state (equal weight §13): model prop-score order over the full gated pool,
    aggregated exactly like R15 §28 so numbers are directly comparable:
      rec{K}              = mean over positives of (positive in top-K)
      best_positive_rank  = 1 + #(score above the model's top-scored positive)
      oracle_rank         = 1 + #(score above the TRUE max-U proposal)
      mrr                 = mean 1/best_positive_rank
      false_stop_rate     = frac pos-states where stop-logit outranks every positive
      stop_gap            = mean(s_best_pos - s_stop)
    """
    rec = {f"rec{K}": [] for K in keys}
    bpos_rank, oracle_rank, mrr, fs, gap = [], [], [], [], []
    for g in groups:
        N = int(len(g["U"]))
        if N == 0:
            continue
        prop, stop = selector(g["F_all"], g["sf"], g["pstats"])
        s = prop.detach().reshape(-1)
        s_stop = float(stop)
        U = g["U"]
        pos = [k for k in range(N) if float(U[k]) > 0.0]
        if not pos:
            continue
        order = torch.argsort(-s).tolist()
        pos_set = set(pos)
        for K in keys:
            topk = set(order[: min(K, N)])
            rec[f"rec{K}"].append(float(np.mean(
                [1.0 if k in topk else 0.0 for k in pos])))
        bp = max(pos, key=lambda k: float(s[k]))            # model top-positive
        br = 1 + int(torch.sum(s > s[bp]))
        bpos_rank.append(br)
        mrr.append(1.0 / br)
        ob = max(pos, key=lambda k: float(U[k]))            # true max-U
        oracle_rank.append(1 + int(torch.sum(s > s[ob])))
        fs.append(1.0 if s_stop > float(s[bp]) else 0.0)
        gap.append(float(s[bp]) - s_stop)
    out = {}
    for k_, v in rec.items():
        out[k_] = float(np.mean(v)) if v else None
    out["best_positive_rank"] = (float(np.mean(bpos_rank)) if bpos_rank else None)
    out["oracle_rank"] = (float(np.mean(oracle_rank)) if oracle_rank else None)
    out["mrr"] = (float(np.mean(mrr)) if mrr else None)
    out["false_stop_rate"] = (float(np.mean(fs)) if fs else None)
    out["stop_gap"] = (float(np.mean(gap)) if gap else None)
    out["n_pos_states"] = len(bpos_rank)
    return out


def _r16_group_loss(selector, g, tau_u, tau_p):
    """§6-11 per-state listwise loss (state = equal-weight unit §13, pair §14 per-state).

    Returns (l_list, l_posmargin, l_pair).  Target q from U_tilde/tau_U incl STOP(0);
    model p = softmax(action_logits/tau_p).  P0 = best zero-U action incl STOP (§8).
    Positive pairs = U_i >= U_j > 0, weight |Utilde_i - Utilde_j|, hinge mean (§9)."""
    F_all, sf, pstats, U = g["F_all"], g["sf"], g["pstats"], g["U"]
    N = int(len(U))
    prop, stop = selector(F_all, sf, pstats)
    logits = _action_logits((prop, stop), N)                  # [N+1]
    s_prop, s_stop = logits[:N], logits[N]
    p = torch.softmax(logits / tau_p, dim=-1)
    U_t = _u_tilde(U).detach()
    q = torch.softmax(torch.cat([U_t, torch.zeros(1, dtype=U_t.dtype)]) / tau_u,
                      dim=-1)
    l_list = -torch.sum(q * torch.log(p + 1e-12))             # §6 KL(q||p) + H(q) const
    pos_mask = U > 0
    if bool(pos_mask.any()):
        zero_acts = torch.cat([s_prop[U == 0], s_stop.reshape(1)])
        s0 = zero_acts.max()
        l_posmargin = torch.nn.functional.softplus(
            float(C.TO1_R16_POS_MARGIN) - (s_prop[pos_mask] - s0)).mean()
    else:
        l_posmargin = torch.zeros((), dtype=torch.float32)
    n_pos = int(pos_mask.sum())
    if n_pos >= 2:
        idx = pos_mask.nonzero(as_tuple=False).flatten()
        Ui = U_t[idx]
        order = torch.argsort(-Ui)                            # highest-U first
        idx_s = idx[order]
        si = s_prop[idx_s]
        wi = Ui[order]
        ar = torch.arange(n_pos, dtype=torch.long)
        a, b = torch.combinations(ar, r=2).unbind(dim=1)
        w_ab = (wi[a] - wi[b]).abs()                          # ≥ 0 (desc order)
        l_pair = (w_ab * torch.clamp(1.0 - (si[a] - si[b]), min=0.0)).mean()
    else:
        l_pair = torch.zeros((), dtype=torch.float32)
    return l_list, l_posmargin, l_pair


def train_pool_listwise_r16(tr_groups, held_groups=None, reranker=None, seed=0,
                            epochs=None, tau_u=None, tau_p=None):
    """R16 SFT: pool-relative listwise objective (§6-14).  Balance unit = state (§13).
    tr_groups/held_groups = [{F_all[N,277] frozen, sf[7], pstats[5], U[N], split}] built
    ONCE by the runner from the FULL gated pool + STOP (§2), U from FixedDecisionReplay
    labels (§3, TRAIN-only §4).  #15 selection: epoch = argmax held mean pos-recall@K
    (tie -> earliest).  Returns best-epoch selector + per-epoch hist + metrics."""
    epochs = int(epochs if epochs is not None else C.TO1_R16_EPOCHS)
    tau_u = float(tau_u if tau_u is not None else C.TO1_R16_TAU_U)
    tau_p = float(tau_p if tau_p is not None else C.TO1_R16_TAU_P)
    torch.manual_seed(seed)              # MUST precede selector init: deterministic
    rng = random.Random(seed)            # weights AND per-epoch group shuffle
    selector = M3Top1Selector()
    ws = selector.warm_start_from(reranker)
    print(f"[r16] prop_head warm-started from R5 utility reranker: {ws}", flush=True)
    print(f"[r16] train groups {len(tr_groups)} | held groups {len(held_groups or [])} "
          f"| tau_u={tau_u} tau_p={tau_p} | lambda_pos={C.TO1_R16_LAMBDA_POS} "
          f"lambda_pair={C.TO1_R16_LAMBDA_PAIR} | select rec@"
          f"{C.TO1_R16_SELECT_RECALL_K}", flush=True)
    opt = torch.optim.AdamW(selector.parameters(), lr=C.TO1_R16_LR,
                            weight_decay=C.TO1_R16_WEIGHT_DECAY)
    sel_k = C.TO1_R16_SELECT_RECALL_K
    hist = []
    best = {"score": None, "epoch": -1, "state": None}
    for ep in range(epochs):
        selector.train()
        order = list(range(len(tr_groups)))
        rng.shuffle(order)
        acc = {"l_list": 0.0, "l_pos": 0.0, "l_pair": 0.0}
        for gi in order:
            l_list, l_pos, l_pair = _r16_group_loss(selector, tr_groups[gi],
                                                    tau_u, tau_p)
            loss = (l_list + C.TO1_R16_LAMBDA_POS * l_pos
                    + C.TO1_R16_LAMBDA_PAIR * l_pair)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            acc["l_list"] += float(l_list.detach())
            acc["l_pos"] += float(l_pos.detach())
            acc["l_pair"] += float(l_pair.detach())
        selector.eval()
        tm = _r16_epoch_metrics(selector, tr_groups)
        hm = (_r16_epoch_metrics(selector, held_groups) if held_groups else None)
        row = {"ep": ep,
               "l_list": acc["l_list"] / max(len(order), 1),
               "l_pos": acc["l_pos"] / max(len(order), 1),
               "l_pair": acc["l_pair"] / max(len(order), 1),
               "train": tm, "held": hm}
        hist.append(row)
        h_score = (None if hm is None else
                   (hm.get(f"rec{sel_k}") if hm.get(f"rec{sel_k}") is not None
                    else -1e9))
        if h_score is not None and (best["score"] is None or h_score > best["score"]):
            # strictly-greater keeps the EARLIEST epoch on ties (stability)
            best = {"score": float(h_score), "epoch": ep,
                    "state": copy.deepcopy(selector.state_dict())}
        if ep == 0 or ep == epochs - 1 or ep % 5 == 4:
            tr_r = ("-" if tm.get(f"rec{sel_k}") is None
                    else f"{tm[f'rec{sel_k}']:.3f}")
            h_r = ("-" if hm is None or hm.get(f"rec{sel_k}") is None
                   else f"{hm[f'rec{sel_k}']:.3f}")
            print(f"[r16] ep {ep} L_list={row['l_list']:.4f} "
                  f"L_posmargin={row['l_pos']:.4f} L_pair={row['l_pair']:.4f} "
                  f"rec@{sel_k} train={tr_r} held={h_r}", flush=True)
    if best["state"] is None:                      # no held groups or all scores -inf
        best = {"score": 0.0, "epoch": epochs - 1,
                "state": copy.deepcopy(selector.state_dict())}
    selector.load_state_dict(best["state"])
    selector.eval()
    tm = _r16_epoch_metrics(selector, tr_groups)
    hm = (_r16_epoch_metrics(selector, held_groups) if held_groups else None)
    return {"selector": selector, "hist": hist, "best_epoch": int(best["epoch"]),
            "best_held_score": best["score"], "train_metrics": tm,
            "held_metrics": hm, "warm_start": ws}


def pool_utility_mass(u_mass_rows):
    """R16 §31 PositiveUtilityMass@K mean over S0 diag u_mass dicts.

    Each row (PER STATE) = {"base": {K: util-frac}, "cur": {K: util-frac}},
    where util-frac@K = (sum of true_U over positives in the model-score top-K) /
    (sum of true_U over ALL positives) -- the §32 utility-weighted recall that does
    not cap at the 32 slot bound.  base = frozen M3 prop-score order; cur =
    action-logit order incl STOP-prized.  States pool together with EQUAL weight (§13).
    Returns {"base": {K: mean}, "cur": {K: mean}, "n_states": n}."""
    base, cur = {f"K{K}": [] for K in (10, 20, 32)}, {f"K{K}": [] for K in (10, 20, 32)}
    for um in u_mass_rows:
        if not um:
            continue
        for K in (10, 20, 32):
            if um["base"].get(f"K{K}") is not None:
                base[f"K{K}"].append(um["base"][f"K{K}"])
            if um["cur"].get(f"K{K}") is not None:
                cur[f"K{K}"].append(um["cur"][f"K{K}"])
    def _m(v):
        return float(np.mean(v)) if v else None
    return {"base": {f"K{K}": _m(base[f"K{K}"]) for K in (10, 20, 32)},
            "cur": {f"K{K}": _m(cur[f"K{K}"]) for K in (10, 20, 32)},
            "n_states": len(base["K10"])}


# ---------------------------------------------------------------------------
# R17: POOL-CONTEXT proposal representation SFT (T1-M3-POOL-CONTEXT-SFT)
# ---------------------------------------------------------------------------
# R16 (verdict B) proved the listwise LOSS works but head-only retrain on FROZEN
# per-proposal features caps at rec32 +0.013 / util-mass@32 0.882->0.907.  R17
# upgrades the M3 proposal REPRESENTATION with a permutation-invariant POOL-CONTEXT
# scorer (§5-8) + runtime M2/probe evidence (§9-12) + direct utility regressor (§19).
# score(P)   = base_R6(F_P) + alpha_ctx·tanh(delta_ctx(z_i))      (§25-26, bounded)
# score_STOP = base_R6_stop(sf,pstats) + alpha_ctx·tanh(delta_stop(sf,c_pool,pext)) (§13)
# z_i = cat(h_i, c_pool, h_i-mean(h), h_i/(std(h)+eps), state_feat, X_i)   (§8)
# c_pool = cat(mean(h), max(h), std(h)) over the M3 action pool (§7, permutation-inv).

R17_ATOM_DIM = 9
R17_STRUCT_DIM = 15
R17_PAIR_DIM = 5 * R17_ATOM_DIM                     # a1 + a2 + mean + |diff| + prod
R17_EVID_DIM = R17_STRUCT_DIM + R17_PAIR_DIM        # 60
R17_STOP_PSTATS_DIM = C.TO1_STOP_POOL_STAT_DIM + 5  # orig 5 + single/route/tierA/tierB/std


def _r17_atom_of(cand, op2idx, probe, retained, tier_a, tier_b, best_gain, q2map,
                 mach_of, e, op):
    """§12 per-atom descriptor (9 dims): family / enabler / root support / tier /
    probe gain (robust state-normalized §10) / memory support / machine util / start.
    All fields runtime-known BEFORE the M3 decision; NEVER true_U."""
    out = [0.0] * R17_ATOM_DIM
    out[0] = 1.0 if getattr(e, "edit_type", "ROUTE") == "ROUTE" else 0.0
    ci = op2idx.get(op)
    if ci is None:
        return out
    f = cand["feats"][ci]
    out[1] = float(f[2])                          # is_enabler
    out[2] = 1.0 if op in retained else 0.0
    out[3] = 1.0 if op in tier_a else 0.0
    out[4] = 1.0 if op in tier_b else 0.0
    pr = probe.get(op)
    if pr is not None:
        out[5] = float(min(max(pr["g_probe"] / max(best_gain, 1e-9), 0.0), 1.0))   # §10
        out[6] = float(min(max(pr["mem_support"] / 10.0, 0.0), 1.0))
    out[7] = float(f[5])                          # machine_util (op's current machine)
    out[8] = float(f[3])                          # start_norm
    return out


def r17_evidence(ast, gated_metas, gate):
    """[G, R17_EVID_DIM] runtime evidence per gated proposal (§9-12).

    Structural block (§11): single/joint, atom count, ROUTE/SEQ counts, affected ops/
    machines, same-machine, shared-root, dependency-expanded.  M2 block (§9-10):
    supporting-root count, Tier-A/B indicators, robust probe-gain (state-normalized),
    positive/dependency probe, max q2.  Pair block (§12): separate atom encodings
    e1/e2 + mean/absdiff/prod; single proposals -> second atom zero-masked.
    Permutation-invariant under pool reorder (per-proposal + state-level aggregates
    only; never reads the pool ORDER)."""
    m2 = gate["m2_rec"]
    from .joint_grpo import _root_candidate_arrays  # lazy (top1<->joint_grpo cycle)
    cand = _root_candidate_arrays(ast)              # cached: the gate already built it
    op2idx = {op: i for i, op in enumerate(cand["ops"])}
    probe = m2["probe"]
    best_gain = max([p["g_probe"] for p in probe.values() if p["g_probe"] > 0.0]
                    or [0.0])
    retained, tier_a, tier_b = (set(m2["retained"]), set(m2["tier_a"]),
                                set(m2["tier_b"]))
    q2map = {}
    for d in m2["draws"]:
        q2map.setdefault(cand["ops"][d["idx"]], float(d.get("q2", 0.0)))
    try:
        _gv = ScheduleGraphView.from_problem_schedule(ast["problem"], ast["schedule"])
        _mach = {}
        def _mach_of(op):
            if op not in _mach:
                iv = _gv.interval_for(op)
                _mach[op] = (iv.machine_id if iv else "")
            return _mach[op]
    except Exception:  # noqa: BLE001 - fallback: no machine info (same-machine=0)
        _mach_of = lambda op: ""

    rows = []
    for m in gated_metas:
        edits, kind = _edits_for(ast, m)
        ops = [e.operation_id for e in edits]
        a1 = _r17_atom_of(cand, op2idx, probe, retained, tier_a, tier_b, best_gain,
                          q2map, _mach_of, edits[0], ops[0])
        a2 = [0.0] * R17_ATOM_DIM
        if kind == "pair" and len(edits) >= 2:
            a2 = _r17_atom_of(cand, op2idx, probe, retained, tier_a, tier_b,
                              best_gain, q2map, _mach_of, edits[1], ops[1])
        is_joint = (kind == "pair")
        n_atoms = len(edits)
        route = sum(1 for e in edits
                    if getattr(e, "edit_type", "ROUTE") == "ROUTE")
        seq = n_atoms - route
        machs = {_mach_of(op) for op in ops}
        supports = [op for op in ops if op in retained]
        sup_gains = [probe[o]["g_probe"] / max(best_gain, 1e-9)
                     for o in supports if o in probe]
        q2s = [q2map.get(op, 0.0) for op in ops]
        st = [
            float(is_joint),
            n_atoms / 2.0,
            route / max(n_atoms, 1),
            seq / max(n_atoms, 1),
            len(set(ops)) / 2.0,
            len(machs) / 2.0,
            float(is_joint and n_atoms == 2 and len(machs) == 1),
            float(len(supports) > 0),
            min(len(supports), 4) / 4.0,
            float(any(o in tier_a for o in ops)),
            float(any(o in tier_b for o in ops)),
            float(min(max(sup_gains, default=0.0), 1.0)),
            float(min(sup_gains, default=0.0) if supports else 0.0),
            float(any(probe.get(o) and probe[o]["g_probe"] > 0.0 for o in ops)),
            float(min(max(q2s, default=0.0) / 3.0, 1.0)),
        ]
        pair = (a1 + a2
                + [(a1[k] + a2[k]) / 2.0 for k in range(R17_ATOM_DIM)]
                + [abs(a1[k] - a2[k]) for k in range(R17_ATOM_DIM)]
                + [a1[k] * a2[k] for k in range(R17_ATOM_DIM)])
        rows.append(st + pair)
    return torch.tensor(rows, dtype=torch.float32)   # [G, E]


def r17_pool_stats(F_all, X):
    """§14 extended pool statistics [R17_STOP_PSTATS_DIM=10]:
    [0:5) = original _pool_stats_from; [5] single_frac; [6] route_frac; [7] tierA-
    supported frac; [8] tierB-supported frac; [9] base-score dispersion (std of old)."""
    p = _pool_stats_from(F_all).reshape(-1).tolist()
    if len(F_all) == 0:
        p = [0.0] * TO1_STOP_POOL_STAT_DIM
    role_oh = F_all[:, SCORER_HIDDEN + 3: SCORER_HIDDEN + 3 + ROLE_N]
    single = (role_oh[:, :2].sum(1) > 0.5).float().mean().item() if len(F_all) else 0.0
    route_frac = float(X[:, 2].mean().item()) if X is not None and len(X) else 0.0
    tierA = float(X[:, 9].mean().item()) if X is not None and len(X) else 0.0
    tierB = float(X[:, 10].mean().item()) if X is not None and len(X) else 0.0
    disp = float(F_all[:, COL_OLD].std().item()) if len(F_all) > 1 else 0.0
    return torch.tensor([p + [single, route_frac, tierA, tierB, disp]],
                        dtype=torch.float32)        # [1,10]


class PoolContextProposalScorer(nn.Module):
    """R17 M3 SFT scorer: FROZEN R6 base + bounded pool-context residual (§5-8,25-26)
    + utility regressor auxiliary (§19).  Only `enc_prop / ctx_head / stop_ctx_head /
    util_head` train (§23); the embedded R6 base is FROZEN (§24)."""

    def __init__(self, base_selector, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.base = base_selector
        self.base.requires_grad_(False)
        self.alpha_ctx = float(C.TO1_R17_ALPHA_CTX)
        H = int(C.TO1_R17_PROP_HIDDEN)
        self.enc_prop = nn.Sequential(
            nn.Linear(PROP_FEAT_DIM, H), nn.LayerNorm(H), nn.GELU(),
            nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(),
        )
        self.z_dim = 6 * H + STATE_FEAT_DIM + R17_EVID_DIM   # 6H = h+c(3H)+(h-mean)+h/std
        self.ctx_head = nn.Sequential(
            nn.Linear(self.z_dim, 96), nn.GELU(), nn.Linear(96, 1))
        self.stop_ctx_head = nn.Sequential(
            nn.Linear(STATE_FEAT_DIM + 3 * H + R17_STOP_PSTATS_DIM, 64),
            nn.GELU(), nn.Linear(64, 1))
        self.util_head = nn.Sequential(
            nn.Linear(self.z_dim, 64), nn.GELU(), nn.Linear(64, 1))
        for head in (self.ctx_head, self.stop_ctx_head):
            with torch.no_grad():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def _base_raw(self, F_pool):
        """Frozen R6 base scores, SAME contract as M3RollingGRPOPolicy._base_raw
        (§31: R17 shortlist construction is VERBATIM the R15/R16 one -- Source1
        GLOBAL uses the frozen R6 prop head, never the context residual)."""
        if F_pool is not None and len(F_pool):
            with torch.no_grad():
                raw = self.base.prop_scores(F_pool)     # frozen, no grad
            return raw, raw, {"scale": 1.0, "selected": "raw"}
        empty = {"center": 0.0, "scale": 0.0, "mad": 0.0, "std": 0.0, "selected": "empty"}
        return torch.zeros(0, dtype=torch.float32), torch.zeros(0, dtype=torch.float32), empty

    def _h_agg(self, F):
        h = self.enc_prop(F)                       # [N,H]
        mean = h.mean(0)
        hmax = h.max(0).values
        std = h.std(0) + 1e-6
        return h, mean, hmax, std

    def _z(self, F, X, sf):
        """§8 z_i = cat(h_i, c_pool, h_i-mean(h), h_i/(std+eps), state, X_i). [N,6H+7+E]"""
        h, mean, hmax, std = self._h_agg(F)
        c = torch.cat([mean, hmax, std])           # [3H]
        N = h.shape[0]
        z = torch.cat([h, c.unsqueeze(0).expand(N, -1),
                       h - mean.unsqueeze(0), h / std.unsqueeze(0),
                       sf.reshape(1, -1).expand(N, -1), X], dim=-1)
        return z, c

    def prop_scores(self, F, X=None, sf=None):
        """[N] §25 residual: base_R6(F) + alpha·tanh(delta(z)).  X=None -> exactly the
        frozen R6 base (parity)."""
        with torch.no_grad():
            base = self.base.prop_scores(F).detach()
        if X is None or sf is None:
            return base
        z, _ = self._z(F, X, sf)
        delta = self.ctx_head(z).squeeze(-1)
        return base + self.alpha_ctx * torch.tanh(delta)

    def stop_score(self, sf, pstats, pstats_ext=None, c_pool=None):
        """[1] §13 context-dependent STOP.  Without (pstats_ext, c_pool) -> frozen base."""
        with torch.no_grad():
            base = self.base.stop_head(
                torch.cat([sf.reshape(1, -1), pstats.reshape(1, -1)], -1))[0].detach()
        if pstats_ext is None or c_pool is None:
            return base
        x = torch.cat([sf.reshape(1, -1), pstats_ext.reshape(1, -1),
                       c_pool.reshape(1, -1)], -1)
        delta = self.stop_ctx_head(x)[0]
        return base + self.alpha_ctx * torch.tanh(delta)

    def util_hat(self, F, X, sf):
        """[N] §19 learned utility predictor (Huber to U_tilde; NEVER fed back §22)."""
        z, _ = self._z(F, X, sf)
        return self.util_head(z).squeeze(-1)

    def forward(self, F, sf, pstats, X=None, pstats_ext=None):
        """([N] prop, [1] stop)."""
        if X is None or pstats_ext is None:
            prop = self.prop_scores(F, None)
            stop = self.stop_score(sf, pstats, None, None)
        else:
            _, c = self._z(F, X, sf)
            prop = self.prop_scores(F, X, sf)
            stop = self.stop_score(sf, pstats, pstats_ext, c)
        return prop, stop


def r17_action_logits(sel, F, sf, pstats, X, pstats_ext):
    """[M+1] prop-item logits + context STOP, computed ONCE (c_pool shared)."""
    _, c = sel._z(F, X, sf)
    prop = sel.prop_scores(F, X, sf)
    stop = sel.stop_score(sf, pstats, pstats_ext, c)
    return torch.cat([prop, stop.reshape(1)], dim=-1)


@torch.no_grad()
def _r17_epoch_metrics(sel, groups, keys=(1, 3, 5, 10, 20, 32)):
    """§2-3 primary + capacity-aware metrics over prebuilt R17 groups.

    Per state (equal weight §21): rec{K} / best_positive_rank / oracle_rank / mrr /
    false_stop_rate / stop_gap (as R16 §18) PLUS the §34-36 utility family:
      capacity_recall_K  = actual positives in the model top-32 / min(n_pos,32)   (§35)
      capacity_bound     = min(1, 32/n_pos) per positive state (the §34 ceiling)
      utility_mass@K     = Sum(U of positives in model top-K) / Sum(U of positives)
      selected_U         = true_U of the argmax action (STOP->0)
      utility_regret     = max_U - selected_U (mean/median/p90) + normalized (§36)
    """
    rec = {f"rec{K}": [] for K in keys}
    bpos_rank, oracle_rank, mrr, fs, gap = [], [], [], [], []
    cap_rec, cap_bnd, util = {f"rec{K}": [] for K in keys}, [], []
    um = {f"K{K}": [] for K in (10, 20, 32)}
    sU, reg, nreg = [], [], []
    for g in groups:
        N = int(len(g["U"]))
        if N == 0:
            continue
        prop, stop = sel(g["F_all"], g["sf"], g["pstats"], g["X"], g["pstats_ext"])
        s = prop.detach().reshape(-1)
        s_stop = float(stop)
        U = g["U"]
        pos = [k for k in range(N) if float(U[k]) > 0.0]
        if not pos:
            continue
        order = torch.argsort(-s).tolist()
        pos_set = set(pos)
        for K in keys:
            topk = set(order[: min(K, N)])
            rec[f"rec{K}"].append(float(np.mean(
                [1.0 if k in topk else 0.0 for k in pos])))
        bp = max(pos, key=lambda k: float(s[k]))
        br = 1 + int(torch.sum(s > s[bp]))
        bpos_rank.append(br)
        mrr.append(1.0 / br)
        ob = max(pos, key=lambda k: float(U[k]))
        oracle_rank.append(1 + int(torch.sum(s > s[ob])))
        fs.append(1.0 if s_stop > float(s[bp]) else 0.0)
        gap.append(float(s[bp]) - s_stop)
        # §34-36 utility family
        denom = sum(float(U[k]) for k in pos) or 1.0
        for K in keys:
            cap_rec[f"rec{K}"].append(
                float(sum(1 for k in pos if k in set(order[: min(K, N)])))
                / min(len(pos), K))
        cap_bnd.append(float(min(1.0, 32.0 / max(len(pos), 1))))
        for K in (10, 20, 32):
            um[f"K{K}"].append(float(
                sum(max(float(U[k]), 0.0) for k in order[: min(K, N)]
                    if float(U[k]) > 0.0) / denom))
        logits = torch.cat([s, torch.tensor([s_stop], dtype=s.dtype)])
        sel_idx = int(logits.argmax().item())
        sel_u = 0.0 if sel_idx == N else float(U[sel_idx])
        mx = float(U[ob])
        sU.append(sel_u)
        reg.append(max(0.0, mx - sel_u))
        nreg.append(max(0.0, (mx - sel_u) / max(mx, 1e-9)))
    out = {}
    for k_, v in rec.items():
        out[k_] = float(np.mean(v)) if v else None
    out["best_positive_rank"] = (float(np.mean(bpos_rank)) if bpos_rank else None)
    out["oracle_rank"] = (float(np.mean(oracle_rank)) if oracle_rank else None)
    out["mrr"] = (float(np.mean(mrr)) if mrr else None)
    out["false_stop_rate"] = (float(np.mean(fs)) if fs else None)
    out["stop_gap"] = (float(np.mean(gap)) if gap else None)
    out["capacity_recall32"] = (float(np.mean(cap_rec["rec32"])) if cap_rec["rec32"] else None)
    out["capacity_bound32"] = (float(np.mean(cap_bnd)) if cap_bnd else None)
    out["utility_mass"] = {f"K{K}": (float(np.mean(um[f"K{K}"])) if um[f"K{K}"] else None)
                           for K in (10, 20, 32)}
    out["selected_true_U"] = (float(np.mean(sU)) if sU else None)
    out["utility_regret"] = {"mean": (float(np.mean(reg)) if reg else None),
                             "median": (float(np.median(reg)) if reg else None),
                             "p90": (float(np.percentile(reg, 90)) if reg else None),
                             "normalized_mean": (float(np.mean(nreg)) if nreg else None)}
    out["n_pos_states"] = len(bpos_rank)
    return out


def _r17_group_loss(sel, g, tau_u, tau_p):
    """§16-21 per-state loss.  L = L_list + λ_pos·L_posmargin + λ_pair·L_pair
    + λ_util·L_utility (Huber to U_tilde, Hutch-aware log-scaled weighting §18)."""
    F, sf, pstats, U = g["F_all"], g["sf"], g["pstats"], g["U"]
    X, pext = g["X"], g["pstats_ext"]
    N = int(len(U))
    logits = r17_action_logits(sel, F, sf, pstats, X, pext)
    s_prop, s_stop = logits[:N], logits[N]
    p = torch.softmax(logits / tau_p, dim=-1)
    U_t = _u_tilde(U).detach()
    q = torch.softmax(torch.cat([U_t, torch.zeros(1, dtype=U_t.dtype)]) / tau_u, dim=-1)
    l_list = -torch.sum(q * torch.log(p + 1e-12))
    pos_mask = U > 0
    if bool(pos_mask.any()):
        zero_acts = torch.cat([s_prop[U == 0], s_stop.reshape(1)])
        s0 = zero_acts.max()
        l_posmargin = torch.nn.functional.softplus(
            float(C.TO1_R17_POS_MARGIN) - (s_prop[pos_mask] - s0)).mean()
    else:
        l_posmargin = torch.zeros((), dtype=torch.float32)
    n_pos = int(pos_mask.sum())
    if n_pos >= 2:
        idx = pos_mask.nonzero(as_tuple=False).flatten()
        Ui = U_t[idx]
        order = torch.argsort(-Ui)
        idx_s = idx[order]
        si = s_prop[idx_s]
        wi = Ui[order]
        ar = torch.arange(n_pos, dtype=torch.long)
        a, b = torch.combinations(ar, r=2).unbind(dim=1)
        w_ab = (wi[a] - wi[b]).abs()
        l_pair = (w_ab * torch.clamp(1.0 - (si[a] - si[b]), min=0.0)).mean()
    else:
        l_pair = torch.zeros((), dtype=torch.float32)
    Uhat = sel.util_hat(F, X, sf)
    l_util = torch.nn.functional.huber_loss(Uhat, U_t.detach(), delta=1.0).mean()
    return l_list, l_posmargin, l_pair, l_util


def _r17_held_utility_mass(sel, groups, K=32, context=True):
    """§28 model-selection composite (primary): mean PositiveUtilityMass@K over the
    held positive-state groups under the R17 prop-score order (utility outcome).
    context=False -> orders by the frozen R6 base only (prop_scores X=None) so the
    same holder measures the R17 context vs its own base on identical groups."""
    vals = []
    for g in (groups or []):
        if not bool((g["U"] > 0).any()):
            continue
        if context:
            prop = sel.prop_scores(g["F_all"], g["X"], g["sf"]).detach()
        else:
            with torch.no_grad():
                prop = sel.prop_scores(g["F_all"], None).detach()
        order = torch.argsort(-prop).tolist()
        U = g["U"]
        pos = [k for k in range(len(U)) if float(U[k]) > 0.0]
        denom = sum(float(U[k]) for k in pos) or 1.0
        vals.append(float(sum(max(float(U[k]), 0.0) for k in order[: min(K, len(U))]
                              if float(U[k]) > 0.0) / denom))
    return (float(np.mean(vals)) if vals else None)


def train_pool_context_r17(tr_groups, held_groups=None, base_selector=None, seed=0,
                           epochs=None, tau_u=None, tau_p=None):
    """R17 SFT (§21): state-balanced pool-context listwise + utility-aux regressor.
    Balance unit = state (§13 modes carry over).  #27-28 model selection on TRAIN
    internal-held + AUX-held (never VAL) via held PositiveUtilityMass@32 (not item
    count).  Returns best-epoch scorer + hist + metrics."""
    epochs = int(epochs if epochs is not None else C.TO1_R17_EPOCHS)
    tau_u = float(tau_u if tau_u is not None else C.TO1_R17_TAU_U)
    tau_p = float(tau_p if tau_p is not None else C.TO1_R17_TAU_P)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sel = PoolContextProposalScorer(base_selector, seed=seed)
    sel.train()
    opt = torch.optim.AdamW(
        [p for p in sel.parameters() if p.requires_grad],
        lr=C.TO1_R17_LR, weight_decay=C.TO1_R17_WEIGHT_DECAY)
    sel_k = int(C.TO1_R17_SELECT_K)
    hist = []
    best = {"score": None, "epoch": -1, "state": None}
    for ep in range(epochs):
        sel.train()
        order = list(range(len(tr_groups)))
        rng.shuffle(order)
        acc = {"l_list": 0.0, "l_pos": 0.0, "l_pair": 0.0, "l_util": 0.0}
        for gi in order:
            l_list, l_pos, l_pair, l_util = _r17_group_loss(sel, tr_groups[gi],
                                                            tau_u, tau_p)
            loss = (l_list + C.TO1_R17_LAMBDA_POS * l_pos
                    + C.TO1_R17_LAMBDA_PAIR * l_pair
                    + C.TO1_R17_LAMBDA_UTILITY * l_util)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in sel.parameters() if p.requires_grad], 1.0)
            opt.step()
            acc["l_list"] += float(l_list.detach())
            acc["l_pos"] += float(l_pos.detach())
            acc["l_pair"] += float(l_pair.detach())
            acc["l_util"] += float(l_util.detach())
        sel.eval()
        tm = _r17_epoch_metrics(sel, tr_groups)
        hm = (_r17_epoch_metrics(sel, held_groups) if held_groups else None)
        h_um = _r17_held_utility_mass(sel, held_groups, K=sel_k)
        row = {"ep": ep,
               "l_list": acc["l_list"] / max(len(order), 1),
               "l_pos": acc["l_pos"] / max(len(order), 1),
               "l_pair": acc["l_pair"] / max(len(order), 1),
               "l_util": acc["l_util"] / max(len(order), 1),
               "train": tm, "held": hm, "held_um": h_um}
        hist.append(row)
        if h_um is not None and (best["score"] is None or h_um > best["score"]):
            best = {"score": float(h_um), "epoch": ep,
                    "state": copy.deepcopy(sel.state_dict())}
        if ep == 0 or ep == epochs - 1 or ep % 5 == 4:
            tr_r = ("-" if tm.get("rec32") is None else f"{tm['rec32']:.3f}")
            h_r = ("-" if hm is None or hm.get("rec32") is None
                   else f"{hm['rec32']:.3f}")
            tr_um = ("-" if not tm["utility_mass"] or tm["utility_mass"]["K32"] is None
                     else f"{tm['utility_mass']['K32']:.3f}")
            print(f"[r17] ep {ep} L_list={row['l_list']:.4f} L_pos={row['l_pos']:.4f} "
                  f"L_pair={row['l_pair']:.4f} L_util={row['l_util']:.4f} "
                  f"rec32 tr={tr_r} hd={h_r} um32 tr={tr_um} "
                  f"held_um32={row['held_um']}", flush=True)
    if best["state"] is None:
        best = {"score": 0.0, "epoch": epochs - 1,
                "state": copy.deepcopy(sel.state_dict())}
    sel.load_state_dict(best["state"])
    sel.eval()
    tm = _r17_epoch_metrics(sel, tr_groups)
    hm = (_r17_epoch_metrics(sel, held_groups) if held_groups else None)
    return {"selector": sel, "hist": hist, "best_epoch": int(best["epoch"]),
            "best_held_score": best["score"], "train_metrics": tm,
            "held_metrics": hm}


def pool_utility_mass_u(u_mass_rows):
    """R16 §31 aggregate over per-state {base:{K}, cur:{K}} dicts (additive)."""
    return pool_utility_mass(u_mass_rows)


@torch.no_grad()
def top1_accuracy(groups, selector):
    """State-wise Top-1 accuracy (STOP included) over the action set."""
    g_with = [g for g in groups if g["F_pool"] is not None]
    g_empty = [g for g in groups if g["F_pool"] is None]

    def _acc_of(_g, prop, stop):
        M = _g["M"]
        logits = _action_logits((prop, stop), M)
        return int(logits.argmax().item()) == _g["target_within_pool"]

    hits_with = sum(1 for g in g_with
                    if _acc_of(g, *selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])))
    hits_empty = sum(1 for g in g_empty if _acc_of(g, torch.zeros(0, dtype=torch.float32),
                                                   _stop_logits_of(selector, g)))
    n_groups = len(groups)
    n_hit = hits_with + hits_empty

    n_pos_hit, n_pos_states = 0, 0
    n_stop_hit, n_stop_states = 0, 0
    n_fail_hit, n_fail_states = 0, 0
    for g in groups:
        if g["F_pool"] is not None:
            prop, stop = selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        else:
            prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
        logits = _action_logits((prop, stop), g["M"])
        sel = int(logits.argmax().item())
        if g["has_pos_full"]:
            n_pos_states += 1
            sel_in_full = g["pool"][sel] if sel < g["M"] else None
            n_pos_hit += 1 if (sel_in_full is not None and sel_in_full == g["oracle_best_full"]) else 0
        if g["target_within_pool"] == g["M"]:
            n_stop_states += 1
            n_stop_hit += 1 if sel == g["M"] else 0
        if g["is_failure"]:
            n_fail_states += 1
            n_fail_hit += 1 if sel == g["target_within_pool"] else 0
    return {"all": float(n_hit / n_groups) if n_groups else 0.0,
            "positive": float(n_pos_hit / n_pos_states) if n_pos_states else 0.0,
            "stop": float(n_stop_hit / n_stop_states) if n_stop_states else 0.0,
            "hard": float(n_fail_hit / n_fail_states) if n_fail_states else 0.0,
            "n_positive_states": n_pos_states, "n_stop_states": n_stop_states,
            "n_hard_states": n_fail_states}


@torch.no_grad()
def top1_metrics(state_examples, scorer, mem_values, reranker, selector):
    """Directive §16-§17 metric table over TRAIN state groups."""
    groups, gstats = build_top1_groups(state_examples, scorer, mem_values,
                                       reranker=reranker, rng=random.Random(0))
    acc = top1_accuracy(groups, selector)

    sel_U = []
    oracle_U = []
    regrets = []
    regrets_r5 = []
    n_sel_positive = 0
    n_stop_selected = 0
    best_recall = {k: 0 for k in (1, 3, 5, 10)}
    n_best_states = 0
    missed_pos = 0
    n_pos_total = 0
    missed_states = 0
    n_any_pos_states = 0
    kl_diag = 0.0
    n_kl = 0

    for g in groups:
        M = g["M"]
        U = g["ex"]["true_U"]
        if g["F_pool"] is not None:
            prop, stop = selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        else:
            prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
        logits = _action_logits((prop, stop), M)
        sel = int(logits.argmax().item())
        sel_U_val = 0.0 if sel == M else float(U[g["pool"][sel]])
        sel_U.append(sel_U_val)
        oracle_U.append(g["max_U_full"])
        if sel_U_val > 0:
            n_sel_positive += 1
        if sel == M:
            n_stop_selected += 1
        regrets.append(max(0.0, g["max_U_full"] - sel_U_val))

        # R5 utility-reranker regret on the SAME group (regret-down comparison)
        if g["F_pool"] is not None and reranker is not None:
            s_r5 = reranker(g["F_pool"]).detach().numpy()
            pick_r5 = int(np.argmax(s_r5))
            u_r5 = float(U[g["pool"][pick_r5]])
            regrets_r5.append(max(0.0, g["max_U_full"] - u_r5))
        else:
            regrets_r5.append(g["max_U_full"])

        # best-positive recall@1/3/5/10 within the pool ranking (selector scores)
        if g["pos_in_pool"]:
            order = list(np.argsort(-prop.detach().numpy()))
            best_in_pool = max(g["pos_in_pool"], key=lambda k: float(U[k]))
            within = g["pool"].index(best_in_pool)
            rank_of_best = order.index(within) + 1 if within in order else None
            if rank_of_best is not None:
                n_best_states += 1
                for k in (1, 3, 5, 10):
                    best_recall[k] += 1 if rank_of_best <= k else 0
        # missed@10: positives of the FULL state missing from selector's top-10
        if g["has_pos_full"]:
            n_any_pos_states += 1
            full_prop = selector.prop_scores(g["F_all"]).detach().numpy()
            top10 = set(np.argsort(-full_prop)[:10].tolist())
            missed = [k for k in g["pos_full"] if k not in top10]
            missed_pos += len(missed)
            n_pos_total += len(g["pos_full"])
            if missed:
                missed_states += 1
        # soft listwise KL diagnostic (report-only, no gradient)
        if M >= 1 and g["max_U_pool"] > 0:
            item_U = torch.tensor([float(U[k]) for k in g["pool"]] + [0.0], dtype=torch.float32)
            p = torch.softmax(item_U / C.UA_UTILITY_SCALE, dim=-1)
            q = torch.softmax(logits, dim=-1)
            kl_diag += float((p * (p.log() - q.log())).sum())
            n_kl += 1

    metrics = {
        "state_wise_top1_acc": acc,
        "selected_true_U": {"mean": float(np.mean(sel_U)) if sel_U else 0.0,
                            "median": float(np.median(sel_U)) if sel_U else 0.0,
                            "n_selected_positive": n_sel_positive,
                            "n_stop_selected": n_stop_selected},
        "oracle_best_true_U": {"mean": float(np.mean(oracle_U)) if oracle_U else 0.0},
        "top1_regret": {"mean": float(np.mean(regrets)) if regrets else 0.0,
                        "median": float(np.median(regrets)) if regrets else 0.0},
        "top1_regret_r5": {"mean": float(np.mean(regrets_r5)) if regrets_r5 else 0.0,
                           "median": float(np.median(regrets_r5)) if regrets_r5 else 0.0},
        "regret_down_vs_r5": float(np.mean(regrets) < np.mean(regrets_r5)) if (regrets and regrets_r5) else 0.0,
        "best_positive_recall": {str(k): float(best_recall[k] / n_best_states) if n_best_states else 0.0
                                 for k in (1, 3, 5, 10)},
        "n_best_states": n_best_states,
        "missed_at10": {"frac_positives": float(missed_pos / n_pos_total) if n_pos_total else 0.0,
                        "frac_positive_states_with_miss": float(missed_states / n_any_pos_states) if n_any_pos_states else 0.0},
        "group_stats": gstats,
        "soft_kl_diagnostic": {"mean": float(kl_diag / n_kl) if n_kl else 0.0, "n_states": n_kl},
    }
    return metrics, groups


# ---------------------------------------------------------------------------
# Closed loop B6: Wide Recall -> Top1 SFT selector -> Proposal/STOP
# ---------------------------------------------------------------------------
def _rollex_of(ast, metas, prop_feats, sf_t):
    roles, ptypes, srcs, tgts = [], [], [], []
    for k in range(len(metas)):
        _e, _kind, _sig, role, ptype, src, tgt, _oids = proposal_identity(ast, metas[k])
        roles.append(role); ptypes.append(ptype); srcs.append(src); tgts.append(tgt)
    return {"metas": metas, "prop_feats": prop_feats, "state_feat": sf_t,
            "role": roles, "type": ptypes, "src": srcs, "tgt": tgts}


def rollout_top1(env, rf, scorer, selector, use_mem=True, horizon=C.HORIZON, gate_mem=False):
    """FixedDecisionReplay closed loop: wide pool -> Top1 selector (incl STOP)
    -> top1 execute -> progressive-memory write.  No gate authority.  gate_mem=True
    (R8 canonical): the selector's mem6 channel is scaled by the policy-observable
    retrieval reliability g_mem(lab session) (§24)."""
    problem, schedule0 = rf["problem"], rf["schedule"]
    iid, progmem, episode_id = rf["iid"], rf["progmem"], rf["episode_id"]
    cache, executor = env["cache"], env["executor"]
    s0_ms = int(schedule0.makespan)
    ms_cur = s0_ms
    schedule = schedule0
    visited = {schedule_hash(schedule0)}
    act_usage = {"single": 0, "pair": 0, "stop_by_selector": 0, "stop_neg": 0}
    steps = []
    for t in range(horizon):
        prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        n_prop = len(metas)
        if n_prop == 0:
            act_usage["stop_by_selector"] += 1
            break
        ast = cache.ast(problem, schedule, iid)
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = _rollex_of(ast, metas, prop_feats, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = (torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                            dtype=torch.float32) if use_mem
               else torch.zeros(n_prop, MEM_FEAT_DIM, dtype=torch.float32))
        gmem = 1.0
        if gate_mem and use_mem:
            gmem = float(progmem.retrieval_gate(iid, episode_id, t, sf))
        logit_pos, rank = _scores(scorer, rolex, mem)
        pool, _info = wide_pool(rolex, logit_pos, rank)
        if not pool:
            act_usage["stop_by_selector"] += 1
            break
        mem_sel = mem * gmem if gate_mem else mem
        F_pool = _rerank_feats_all(scorer, rolex, mem_sel)[pool]
        pool_stats = _pool_stats_from(F_pool)
        with torch.no_grad():
            prop, stop = selector(F_pool, sf_t, pool_stats)
        logits = _action_logits((prop, stop), len(pool))
        sel = int(logits.argmax().item())
        if sel == len(pool):                     # STOP
            act_usage["stop_by_selector"] += 1
            break
        a = pool[sel]
        edits, kind = _edits_for(ast, metas[a])
        res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = proposal_identity(ast, metas[a])[2]
        steps.append({"t": t, "n_prop": n_prop, "n_wide": len(pool),
                      "top_sig": sig, "kind": kind,
                      "improvement": (None if res is None else float(res["improvement"]))})
        if use_mem:
            u = None if res is None else float(res["improvement"])
            outcome = ("success" if u is not None and u > 0 else
                       "neutral" if u is not None and u == 0 else
                       "negative" if u is not None else "infeasible")
            progmem.add_executed(iid, t, {
                "instance_id": iid, "episode_id": episode_id, "state_hash": h,
                "state_feat": sf, "proposal_signature": sig,
                "proposal_type": rolex["type"][a], "role": rolex["role"][a],
                "src": rolex["src"][a], "tgt": rolex["tgt"][a], "true_U": u,
                "outcome": outcome, "successor_state_hash": None,
                "trajectory_step": t, "written_at_step": t,
                "fine_key": ((rolex["type"][a], rolex["role"][a], rolex["src"][a],
                              rolex["tgt"][a]) if rolex["type"][a] == "single"
                             else (rolex["type"][a], rolex["role"][a])),
                "coarse_key": (rolex["type"][a], rolex["role"][a]),
            })
        if res is None or res["improvement"] <= 0:
            act_usage["stop_neg"] += 1
            break
        act_usage[kind] += 1
        schedule = res["schedule"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        if nh in visited:
            break
        visited.add(nh)
    return int(schedule0.makespan) - ms_cur, act_usage, steps


def closed_loop_top1(env, re, scorer, selector, use_mem=True, gate_mem=False):
    gains_by_iid = {}
    steps_by_iid = {}
    for idx, i in enumerate(env["order"]):
        iid = i["instance_id"]
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        rf = {"problem": env["states"][iid]["problem"], "schedule": env["states"][iid]["schedule"],
              "iid": iid, "progmem": re["progmem"], "episode_id": eid}
        gain, usage, steps = rollout_top1(env, rf, scorer, selector,
                                          use_mem=use_mem, gate_mem=gate_mem)
        gains_by_iid[iid] = gain
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    return _b5_summary(gains_by_iid), steps_by_iid


def split_summary(summary, env):
    train_iids = {i["instance_id"] for i in env["train_insts"]}
    val_iids = {i["instance_id"] for i in env["val_insts"]}
    out = {}
    for name, iids in (("train", train_iids), ("val", val_iids)):
        gains = {k: v for k, v in summary["per_instance"].items() if k in iids}
        arr = np.array(list(gains.values()), dtype=np.float64) if gains else np.array([])
        out[name] = {
            "total": int(arr.sum()) if len(arr) else 0,
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "median": float(np.median(arr)) if len(arr) else 0.0,
            "n_positive": int((arr > 0).sum()),
            "worst_instance": (min(gains, key=gains.get) if gains else None),
            "per_instance": gains,
        }
    return out


# ---------------------------------------------------------------------------
# DPpaulli10a trace (R6 §18) + normal-M5 regression (R6 §26)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dpp_top1_trace(iid, episode_id, st, env, re, scorer, selector, s0_trueU_by_sig,
                   gate_mem=False):
    """Trace the Top-1 selector at the DPpaulli10a s0 state."""
    out = {"iid": iid, "episode_id": episode_id, "found": False}
    cache, executor = env["cache"], env["executor"]
    problem, schedule = st["problem"], st["schedule"]
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    if not metas:
        return out
    out["found"] = True
    ast = cache.ast(problem, schedule, iid)
    h0 = schedule_hash(schedule)
    N = len(metas)
    sf = state_feature_vec(int(schedule.makespan), int(schedule.makespan),
                           N, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _rollex_of(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(re["progmem"].features(iid, episode_id, 0, sf, queries),
                       dtype=torch.float32)
    gmem = 1.0
    if gate_mem:
        try:
            gmem = float(re["progmem"].retrieval_gate(iid, episode_id, 0, sf))
        except Exception:  # noqa: BLE001
            gmem = 1.0
    logit_pos, rank = _scores(scorer, rolex, mem)
    pool, pinfo = wide_pool(rolex, logit_pos, rank)
    pool_set = set(pool)
    F_all = _rerank_feats_all(scorer, rolex, mem * gmem if gate_mem else mem)
    sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
    lut = s0_trueU_by_sig
    pos = [k for k in range(N) if lut.get(sigs[k], 0) > 0]
    out["n_pos_total"] = len(pos)
    out["n_pos_in_wide"] = sum(1 for k in pos if k in pool_set)
    best = None
    if pos:
        best = max(pos, key=lambda k: lut[sigs[k]])
        out["best_positive"] = {"sig": sigs[best], "true_U": float(lut[sigs[best]]),
                                "in_wide": bool(best in pool_set)}
    if F_all is not None and pool:
        F_pool = F_all[pool]
        pool_stats = _pool_stats_from(F_pool)
        prop, stop = selector(F_pool, sf_t, pool_stats)
        logits = _action_logits((prop, stop), len(pool))
        probs = torch.softmax(logits, dim=-1)
        order = list(np.argsort(-prop.detach().numpy()))
        if best is not None and best in pool_set:
            within = pool.index(best)
            out["best_predicted"] = {"pool_rank": order.index(within) + 1,
                                     "action_prob": float(probs[within]),
                                     "score": float(prop[within])}
        sel = int(logits.argmax().item())
        sel_is_stop = sel == len(pool)
        sel_u = 0.0 if sel_is_stop else float(lut.get(sigs[pool[sel]], 0))
        out["selected"] = {"is_stop": sel_is_stop,
                           "sig": (None if sel_is_stop else sigs[pool[sel]]),
                           "in_wide": True,
                           "true_U": sel_u,
                           "action_prob": float(probs[sel])}
        out["stop"] = {"score": float(stop[0]),
                       "prob": float(probs[len(pool)])}
        out["score_margin"] = (float(prop[within] - stop[0])
                               if best is not None and best in pool_set else None)
        out["top1_regret"] = float(max(0.0, (lut[sigs[best]] if best is not None else 0.0) - sel_u))
        if best is not None and best in pool_set:
            within = pool.index(best)
            out["dpp_satisfaction"] = (
                "best_selected" if (not sel_is_stop and pool[sel] == best)
                else "selected_positive" if sel_u > 0
                else "best_in_top10" if within in set(order[:10])
                else "no")
        else:
            out["dpp_satisfaction"] = "wide_recall_miss"
    else:
        out["selected"] = {"is_stop": True, "true_U": 0.0, "action_prob": None}
        out["top1_regret"] = float(lut[sigs[best]] if best is not None else 0.0)
        out["dpp_satisfaction"] = "no_pool"
    return out


@torch.no_grad()
def normal_m5_top1(iid, episode_id, st, env, re, scorer, selector, s0_trueU_by_sig,
                   gate_mem=False):
    """normal-M5 regression in Top-1 mode: M5 enabler visibility + selection."""
    out = {"iid": iid, "episode_id": episode_id, "found": False}
    cache, executor = env["cache"], env["executor"]
    problem, schedule = st["problem"], st["schedule"]
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    if not metas:
        return out
    out["found"] = True
    ast = cache.ast(problem, schedule, iid)
    h0 = schedule_hash(schedule)
    N = len(metas)
    sf = state_feature_vec(int(schedule.makespan), int(schedule.makespan),
                           N, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _rollex_of(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(re["progmem"].features(iid, episode_id, 0, sf, queries),
                       dtype=torch.float32)
    gmem = 1.0
    if gate_mem:
        try:
            gmem = float(re["progmem"].retrieval_gate(iid, episode_id, 0, sf))
        except Exception:  # noqa: BLE001
            gmem = 1.0
    logit_pos, rank = _scores(scorer, rolex, mem)
    pool, pinfo = wide_pool(rolex, logit_pos, rank)
    pool_set = set(pool)
    F_all = _rerank_feats_all(scorer, rolex, mem * gmem if gate_mem else mem)
    sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
    rank_np = rank.detach().numpy() if torch.is_tensor(rank) else np.asarray(rank)
    lut = s0_trueU_by_sig
    enabler_m5 = []
    for k in range(N):
        _e, kind, sig, role, ptype, src, tgt, _oids = proposal_identity(ast, metas[k])
        if kind == "single" and role == "ENABLER" and src == "M5":
            rank_head = int((rank_np > rank_np[k]).sum()) + 1
            in_pool = k in pool_set
            full_prop = (selector.prop_scores(F_all).detach().numpy()
                         if F_all is not None else np.zeros(N))
            sel_rank = int(np.sum(full_prop > full_prop[k])) + 1
            enabler_m5.append({
                "sig": sig, "rank_head": rank_head,
                "in_wide": bool(in_pool), "selector_rank": sel_rank,
                "in_selector_top10": sel_rank <= 10,
                "old_base": float(_old_base_single(prop_feats, k)),
                "true_U": float(lut.get(sig, 0.0)),
            })
    out["enabler_m5"] = enabler_m5
    out["m5_stats"] = {"n": len(enabler_m5),
                       "n_with_positive": sum(1 for e in enabler_m5 if e["true_U"] > 0),
                       "n_in_top10": sum(1 for e in enabler_m5 if e["in_selector_top10"])}
    selected = {}
    if F_all is not None and pool:
        F_pool = F_all[pool]
        pool_stats = _pool_stats_from(F_pool)
        prop, stop = selector(F_pool, sf_t, pool_stats)
        logits = _action_logits((prop, stop), len(pool))
        sel = int(logits.argmax().item())
        if sel < len(pool):
            a = pool[sel]
            _e, _kind, _sig, _role, _ptype, _src, _tgt, _oids = proposal_identity(ast, metas[a])
            res = _execute_step(executor, problem, schedule, _e, int(schedule.makespan), h0)
            selected = {"sig": _sig, "role": _role, "src": _src, "tgt": _tgt,
                        "true_U": float(lut.get(_sig, 0.0)),
                        "executed_improvement": (None if res is None else float(res["improvement"]))}
        else:
            selected = {"is_stop": True}
    out["selected"] = selected
    return out


# ---------------------------------------------------------------------------
# R7: STOP relative-scale calibration + TRAIN state augmentation +
#     memory channel dropout (corrected semantics, SFT only)
# ---------------------------------------------------------------------------
# The selector's DIRECT memory exposure is the mem6 tail of the R5 rerank feature
# matrix; the frozen scorer's h channel is untouched (frozen).  Memory dropout
# therefore drops the mem6 channel of the selector input.
MEM_CHAN_SLICE = slice(-MEM_FEAT_DIM, None)      # last 6 cols of [N,277]


def _stop_margin_loss(g, sel_out):
    """R7 §3-5: relative STOP-vs-best-legal-Proposal margin, fixed m_act = m_stop
    = 1.0.  positive state: score(best_idx) >= score(STOP) + m_act;  STOP state:
    score(STOP) >= max_i score(P_i) + m_stop.  Empty pool -> margin satisfied."""
    prop, stop = sel_out
    if g["is_stop_target"]:
        if prop.numel() == 0:
            return torch.zeros((), dtype=torch.float32)
        return torch.clamp(prop.max() - stop[0:1] + C.TO1_MARGIN_STOP, min=0.0)
    best = g["target_within_pool"]
    return torch.clamp(stop[0:1] - prop[best:best + 1] + C.TO1_MARGIN_ACT, min=0.0)


@torch.no_grad()
def top1_accuracy_stop_audit(groups, selector):
    """R7 §7: accuracy + false_stop / false_act / ACT-recall / STOP-recall.
    false_stop = positive-state selected STOP; false_act = STOP-target state
    selected a Proposal.  Reported over pos_full and over pos_in_pool."""
    acc = top1_accuracy(groups, selector)
    n_pos_full = n_pos_act = 0
    n_pos_pool = n_pos_pool_stop = 0
    n_stop_tgt = n_stop_tgt_act = 0
    for g in groups:
        M = g["M"]
        if g["F_pool"] is not None:
            prop, stop = selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        else:
            prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
        sel = int(_action_logits((prop, stop), M).argmax().item())
        is_stop = sel == M
        if g["has_pos_full"]:
            n_pos_full += 1
            n_pos_act += 0 if is_stop else 1
        if g["pos_in_pool"]:
            n_pos_pool += 1
            n_pos_pool_stop += 1 if is_stop else 0
        if g["is_stop_target"]:
            n_stop_tgt += 1
            n_stop_tgt_act += 0 if is_stop else 1
    out = dict(acc)
    out.update({
        "false_stop_over_pos_full": (float(n_pos_full - n_pos_act) / max(n_pos_full, 1)),
        "act_recall_over_pos_full": (float(n_pos_act) / max(n_pos_full, 1)),
        "false_stop_over_pos_pool": (float(n_pos_pool_stop) / max(n_pos_pool, 1)),
        "false_act_over_stop_target": (float(n_stop_tgt_act) / max(n_stop_tgt, 1)),
        "stop_recall_over_stop_target": (float(n_stop_tgt - n_stop_tgt_act) / max(n_stop_tgt, 1)),
        "n_pos_full": n_pos_full, "n_pos_pool": n_pos_pool, "n_stop_target": n_stop_tgt,
    })
    return out


def _queries_of(ex):
    return [{"type": ex["type"][k], "role": ex["role"][k],
             "src": ex["src"][k], "tgt": ex["tgt"][k]}
            for k in range(len(ex["metas"]))]


def _appearance_summary(agg, ex, root_ms):
    return {"n_prop": len(ex["metas"]), "best_uhat": agg["best_uhat"],
            "best_direct": agg["best_direct"], "n_contrib": agg["n_contrib"],
            "n_enab": agg["n_enab"], "ms_ratio": float(ex["root_ms"] / max(root_ms, 1))}


def _executed_rec(iid, eid, t, sf_list, ex, k, u, sig):
    return {
        "instance_id": iid, "episode_id": eid, "state_hash": ex["state_hash"],
        "state_feat": sf_list, "proposal_signature": sig,
        "proposal_type": ex["type"][k], "role": ex["role"][k],
        "src": ex["src"][k], "tgt": ex["tgt"][k], "true_U": u,
        "outcome": ("success" if u > 0 else "neutral" if u == 0 else "negative"),
        "trajectory_step": t, "written_at_step": t, "successor_state_hash": None,
        "fine_key": ((ex["type"][k], ex["role"][k], ex["src"][k], ex["tgt"][k])
                     if ex["type"][k] == "single" else (ex["type"][k], ex["role"][k])),
        "coarse_key": (ex["type"][k], ex["role"][k]),
    }


def _label_state_ex(cache, executor, problem, schedule, iid, root_ms, ep_id, tstep):
    """Label every bounded Proposal of `schedule` via Frozen-Local FixedDecisionReplay
    (real execution; model-legal only).  Returns (ex, ast, agg) with the same keys
    as build_replay examples, or (None, None, None) when no proposals exist."""
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    if not metas:
        return None, None, None
    ast = cache.ast(problem, schedule, iid)
    h = schedule_hash(schedule)
    ms_cur = int(schedule.makespan)
    sf = state_feature_vec(ms_cur, root_ms, len(metas), agg["best_uhat"],
                           agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    N = len(metas)
    true_U = np.full(N, C.INFEASIBLE_U, dtype=np.float32)
    feasible = np.zeros(N, dtype=bool)
    old_base = np.zeros(N, dtype=np.float32)
    sigs = [None] * N; roles = [None] * N; types = [None] * N
    srcs = [None] * N; tgts = [None] * N; op_ids = [None] * N
    for k in range(N):
        edits, kind, sig, role, ptype, src, tgt, oids = proposal_identity(ast, metas[k])
        sigs[k], roles[k], types[k], srcs[k], tgts[k], op_ids[k] = sig, role, ptype, src, tgt, oids
        old_base[k] = old_base_of(prop_feats, k)
        res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
        if res is None:
            continue
        feasible[k] = True
        true_U[k] = float(res["improvement"])
    ex = {
        "iid": iid, "state_hash": h, "root_ms": root_ms,
        "state_feat": torch.tensor(sf, dtype=torch.float32),
        "prop_feats": prop_feats, "metas": metas, "true_U": true_U,
        "feasible": feasible, "old_base": old_base, "sig": sigs, "role": roles,
        "type": types, "src": srcs, "tgt": tgts, "op_id": op_ids,
        "n_contrib": agg["n_contrib"], "n_enab": agg["n_enab"],
        "ep_id": ep_id, "tstep": tstep, "s0": (tstep == 0),
    }
    return ex, ast, agg


def _policy_pick(scorer, selector, ex, mem, sf_t):
    """R6-frozen policy action within the WIDE pool ∪ STOP (data generation only)."""
    logit_pos, rank = _scores(scorer, ex, mem)
    pool, _ = wide_pool(ex, logit_pos, rank)
    if not pool or selector is None:
        return None
    F_pool = _rerank_feats_all(scorer, ex, mem)[pool]
    ps = _pool_stats_from(F_pool)
    with torch.no_grad():
        prop, stop = selector(F_pool, sf_t, ps)
    sel = int(_action_logits((prop, stop), len(pool)).argmax().item())
    if sel == len(pool):
        return None
    return int(pool[sel])


def augment_train_states(env, re, scorer, selector_r6,
                         per_inst_budget=8, explore_bases=2, explore_k=2, seed=0,
                         log_prefix="[aug]"):
    """R7 §9-14: TRAIN-only legal state augmentation.  Every successor comes from
    REAL FixedDecisionReplay (Frozen-Local); no oracle/true_U runtime rule, no fake
    states.  Sources: A = existing oracle replay states (unchanged examples kept);
    B = R6 policy walk; C = bounded legal exploration from policy states.  Unique
    (instance_id, state_hash); each new example carries provenance (source, parent
    state_hash, parent proposal_signature, depth, Cmax, appearance summary).
    """
    cache, executor = env["cache"], env["executor"]
    rng = random.Random(seed)
    orig_hashes = {}
    orig_ex_by_hash = {}
    for ex0 in re["state_examples"]:
        orig_hashes.setdefault(ex0["iid"], set()).add(ex0["state_hash"])
        orig_ex_by_hash[(ex0["iid"], ex0["state_hash"])] = ex0

    aug_examples = []
    aug_prov = []
    stats = {"policy": 0, "exploration": 0, "reused_replay": 0, "skipped_dup": 0,
             "break_neg": 0, "break_stop": 0, "instances": 0}
    per_iid = {}

    for ti in env["train_insts"]:
        iid = ti["instance_id"]
        st = env["states"][iid]
        problem, schedule0 = st["problem"], st["schedule"]
        root_ms = int(schedule0.makespan)
        eid = env["ep_id_of"][iid]
        inst_sfs = [ex0["state_feat"].tolist() for ex0 in re["state_examples"]
                    if ex0["iid"] == iid]
        aug_pm = ProgressiveMemory(state_feats=inst_sfs)
        for oeid in range(eid):                        # evidence-only cross context
            aug_pm.cross_episodes[oeid] = list(re["progmem"].cross_episodes.get(oeid, ()))
        visited = {schedule_hash(schedule0)} | set(orig_hashes.get(iid, ()))
        made = 0
        per_iid[iid] = 0
        stats["instances"] += 1

        # ---- source B: R6 policy walk (label NEW states; reuse oracle states) ----
        schedule, ms_cur = schedule0, root_ms
        parent_prov = {"parent_state_hash": None, "parent_proposal_signature": None}
        policy_pts = []
        for t in range(C.HORIZON):
            if made >= per_inst_budget:
                break
            cur_hash = schedule_hash(schedule)
            orig_ex = orig_ex_by_hash.get((iid, cur_hash))
            if orig_ex is not None:                 # oracle replay state: reuse, no re-label
                ex = orig_ex
                ast = cache.ast(problem, schedule, iid)
            else:
                ex, ast, agg = _label_state_ex(cache, executor, problem, schedule, iid,
                                               root_ms, eid, t)
            if ex is None:
                break
            h = ex["state_hash"]
            sf_list = ex["state_feat"].tolist()
            mem = torch.tensor(aug_pm.features(iid, eid, t, sf_list,
                                               _queries_of(ex)), dtype=torch.float32)
            ex["prog_mem_feats"] = mem
            ex["mem_gate"] = float(aug_pm.retrieval_gate(iid, eid, t, sf_list))
            is_new = h not in orig_hashes.get(iid, ())
            if is_new:
                ex["aug_prov"] = {
                    "source": "policy", "root_ms": root_ms,
                    "parent_state_hash": parent_prov["parent_state_hash"],
                    "parent_proposal_signature": parent_prov["parent_proposal_signature"],
                    "depth": t, "Cmax": ms_cur,
                    "appearance": _appearance_summary(agg, ex, root_ms),
                }
                aug_examples.append(ex)
                aug_prov.append({"iid": iid, "state_hash": h, **ex["aug_prov"]})
                made += 1
                per_iid[iid] += 1
                stats["policy"] += 1
                inst_sfs.append(sf_list)
                aug_pm.set_state_stats(inst_sfs)
            else:
                stats["reused_replay"] += 1
            policy_pts.append((h, ex, mem, t, ast, schedule))
            # continue via the frozen R6 policy (this is not an oracle rule)
            pick = _policy_pick(scorer, selector_r6, ex, mem, ex["state_feat"])
            if pick is None:
                stats["break_stop"] += 1
                break
            _e, _k, sig = proposal_identity(ast, ex["metas"][pick])[0:3]
            res = _execute_step(executor, problem, schedule, _e, ms_cur, h)
            if res is None or res["improvement"] <= 0:
                stats["break_neg"] += 1
                break
            u = float(res["improvement"])
            aug_pm.add_executed(iid, t, _executed_rec(iid, eid, t, sf_list, ex,
                                                      pick, u, sig))
            schedule = res["schedule"]
            ms_cur = int(schedule.makespan)
            nh = schedule_hash(schedule)
            if nh in visited:
                stats["skipped_dup"] += 1
                break
            visited.add(nh)
            parent_prov = {"parent_state_hash": h,
                           "parent_proposal_signature": sig}

        # ---- source C: bounded exploration from the first policy states ----
        for (ph, p_ex, p_mem, pt, p_ast, base_sched) in policy_pts[:explore_bases]:
            if made >= per_inst_budget:
                break
            F_all = _rerank_feats_all(scorer, p_ex, p_mem)
            if F_all is None:
                continue
            # R6 prop-score priority over feasible proposals (legal-only)
            s_all = selector_r6.prop_scores(F_all).detach().numpy()
            feasible = p_ex["feasible"]
            order = [k for k in np.argsort(-s_all) if feasible[k]]
            if not order:
                continue
            base_ms = int(base_sched.makespan)
            for k in order[:explore_k]:
                if made >= per_inst_budget:
                    break
                sig_k = p_ex["sig"][k]
                res = _execute_step(executor, problem, base_sched,
                                    _edits_for(p_ast, p_ex["metas"][k])[0], base_ms, ph)
                if res is None:
                    continue
                nh = schedule_hash(res["schedule"])
                if nh in visited:
                    stats["skipped_dup"] += 1
                    continue
                visited.add(nh)
                # label the successor (fresh state, real Frozen-Local)
                ex2, ast2, agg2 = _label_state_ex(cache, executor, problem,
                                                  res["schedule"], iid, root_ms, eid, pt + 1)
                if ex2 is None:
                    continue
                # exploration executed record BEFORE successor memory (causal-time)
                aug_pm.add_executed(iid, pt, _executed_rec(iid, eid, pt,
                                                           p_ex["state_feat"].tolist(),
                                                           p_ex, k, float(res["improvement"]), sig_k))
                sf2 = ex2["state_feat"].tolist()
                mem2 = torch.tensor(aug_pm.features(iid, eid, pt + 1, sf2,
                                                    _queries_of(ex2)), dtype=torch.float32)
                ex2["prog_mem_feats"] = mem2
                ex2["mem_gate"] = float(aug_pm.retrieval_gate(iid, eid, pt + 1, sf2))
                ex2["aug_prov"] = {
                    "source": "exploration", "root_ms": root_ms,
                    "parent_state_hash": p_ex["state_hash"],
                    "parent_proposal_signature": sig_k,
                    "depth": pt + 1, "Cmax": int(res["schedule"].makespan),
                    "appearance": _appearance_summary(agg2, ex2, root_ms),
                }
                aug_examples.append(ex2)
                aug_prov.append({"iid": iid, "state_hash": ex2["state_hash"], **ex2["aug_prov"]})
                made += 1
                per_iid[iid] += 1
                stats["exploration"] += 1
                inst_sfs.append(sf2)
                aug_pm.set_state_stats(inst_sfs)
        print(f"{log_prefix} {iid}: +{made} new states "
              f"(policy {stats['policy']} / exploration {stats['exploration']} / "
              f"reused {stats['reused_replay']})", flush=True)

    aug_mem_values = [ex["prog_mem_feats"] for ex in aug_examples]
    stats["total"] = len(aug_examples)
    stats["per_instance"] = per_iid
    print(f"{log_prefix} total {len(aug_examples)} new unique TRAIN states "
          f"({json.dumps(stats, default=str)})", flush=True)
    return aug_examples, aug_mem_values, aug_prov, stats


def train_top1_sft_r7(ex_replay, mem_replay, ex_aug, mem_aug, scorer, reranker,
                      r6_selector, phase_gate=None, seed=0):
    """R7 training.  Groups = replay (source A) ∪ augmented (B/C).  Phase A freezes
    the proposal backbone/head (R6 ordering intact); only the STOP head + margin
    calibration train.  Phase B (small-LR last proposal layer) runs ONLY when the
    phase gate still fails: DPPaulli +27 score <= STOP score, or false_stop on
    positive-in-pool states >= 0.35.  Instance-balanced epoch order (sample instance
    first, then state within instance); hard-states keep TO1_HARD_OVERSAMPLE;
    L_total = L_top1 + λ_pair·L_pair + λ_stop·L_stop_margin."""
    rng = random.Random(seed)
    groups_r, _ = build_top1_groups(ex_replay, scorer, mem_replay, reranker=reranker, rng=rng)
    for g_ in groups_r:
        g_["src"] = "replay"
    groups_a, _ = (build_top1_groups(ex_aug, scorer, mem_aug, reranker=reranker, rng=rng)
                   if ex_aug else ([], {}))
    for g_ in groups_a:
        g_["src"] = "aug"
    groups = groups_r + groups_a
    gstats = {"replay": (groups_r and True), "n_replay": len(groups_r),
              "n_aug": len(groups_a), "n_total": len(groups)}
    print(f"[r7] groups replay={len(groups_r)} aug={len(groups_a)} "
          f"total={len(groups)} (pos {sum(1 for g in groups if g['has_pos_full'])})",
          flush=True)

    selector = M3Top1Selector()
    ws_r6 = False
    if r6_selector is not None:
        try:
            selector.prop_head.load_state_dict(r6_selector.prop_head.state_dict())
            ws_r6 = True
        except Exception:                       # shape mismatch -> fallback below
            ws_r6 = False
    if not ws_r6:
        print("[r7] WARN: R6 selector unavailable; prop head warm-start from R5 reranker",
              flush=True)
        selector.warm_start_from(reranker)
    print(f"[r7] prop_head warm-start from R6 v2: {ws_r6}", flush=True)

    for p in selector.prop_head.parameters():
        p.requires_grad_(False)
    fail_gis = [i for i, g in enumerate(groups) if g["is_failure"]]
    n_aux = C.TO1_HARD_OVERSAMPLE - 1

    def _instance_order(rng_):
        by = {}
        for gi_, g_ in enumerate(groups):
            by.setdefault(g_["iid"], []).append(gi_)
        iids = list(by)
        mx = max((len(v) for v in by.values()), default=0)
        order = []
        for r in range(mx):
            rng_.shuffle(iids)
            for ii in iids:
                if r < len(by[ii]):
                    order.append(by[ii][r])
        return order

    def _epoch(opt, ep, mem_drop=True):
        selector.train()
        order = _instance_order(rng)
        order += fail_gis * n_aux
        rng.shuffle(order)
        c_ce = c_pair = c_margin = 0.0
        n_ = 0
        for gi in order:
            g = groups[gi]
            F = g["F_pool"]
            if mem_drop and F is not None and rng.random() < C.TO1_MEM_DROP_P:
                F = F.clone()
                F[..., MEM_CHAN_SLICE] = 0.0
            if F is not None:
                prop, stop = selector(F, g["ex"]["state_feat"], g["pool_stats"])
            else:
                prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
            logits = _action_logits((prop, stop), g["M"])
            ce = nn.functional.cross_entropy(logits.reshape(1, -1),
                                             torch.tensor([g["target_within_pool"]], dtype=torch.long))
            pair = _pair_loss(g, (prop, stop))
            margin = _stop_margin_loss(g, (prop, stop))
            loss = ce + C.TO1_LAMBDA_PAIR * pair + C.TO1_LAMBDA_STOP * margin
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            c_ce += float(ce.detach()); c_pair += float(pair.detach())
            c_margin += float(margin.detach()); n_ += 1
        selector.eval()
        acc = top1_accuracy(groups, selector)
        return {"ep": ep, "ce": c_ce / max(n_, 1), "pair": c_pair / max(n_, 1),
                "margin": c_margin / max(n_, 1), "top1_acc": acc["all"]}

    hist = []
    # ---- Phase A: STOP head only ----
    opt_a = torch.optim.AdamW(selector.stop_head.parameters(), lr=C.TO1_LR, weight_decay=1e-4)
    for ep in range(C.TO1_EPOCHS_A):
        h = _epoch(opt_a, ep, mem_drop=True)
        hist.append(h)
        if ep in (0, C.TO1_EPOCHS_A - 1) or ep % 4 == 3:
            print(f"[r7] A ep {ep} ce={h['ce']:.4f} pair={h['pair']:.4f} "
                  f"margin={h['margin']:.4f} acc={h['top1_acc']:.3f}", flush=True)
    gate_a = (phase_gate(selector) if phase_gate is not None else {})
    _dm_a = gate_a.get("dpp_margin")
    _fsp_a = gate_a.get("false_stop_over_pos_pool")
    need_b = bool((_dm_a is not None and _dm_a <= 0)
                  or (_fsp_a is not None and _fsp_a >= 0.35))
    print(f"[r7] Phase A gate: {json.dumps(gate_a, default=str)} need_phaseB={need_b}",
          flush=True)

    # ---- Phase B (only if the hard gate still fails): last prop layer small LR ----
    applied_b = False
    gate_b = {}
    if need_b:
        for p in selector.prop_head[-1].parameters():
            p.requires_grad_(True)
        opt_b = torch.optim.AdamW(
            list(selector.stop_head.parameters()) + list(selector.prop_head[-1].parameters()),
            lr=C.TO1_LR_PHASE_B, weight_decay=1e-4)
        gate_b = {}
        for ep in range(C.TO1_EPOCHS_B):
            h = _epoch(opt_b, C.TO1_EPOCHS_A + ep, mem_drop=True)
            hist.append(h)
            print(f"[r7] B ep {h['ep']} ce={h['ce']:.4f} margin={h['margin']:.4f} "
                  f"acc={h['top1_acc']:.3f}", flush=True)
        applied_b = True
        gate_b = (phase_gate(selector) if phase_gate is not None else {})
        print(f"[r7] Phase B gate: {json.dumps(gate_b, default=str)}", flush=True)

    phases = {"phase_a_epochs": C.TO1_EPOCHS_A, "phase_b_applied": applied_b,
              "phase_b_epochs": (C.TO1_EPOCHS_B if applied_b else 0),
              "gate_a": gate_a, "gate_b": gate_b}
    return selector, hist, groups, gstats, phases


# ===========================================================================
# R8 -- T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION-R8
# Primary objective = best-vs-hardest-competitor POOL_ARGMAX (L_argmax, §4-§7);
# STOP head frozen (R6's), never trained (§2/§8); original Top1 CE retained as
# AUXILIARY loss on positive states only (no global prop-lowering, §10); fixed
# 3-fold by-INSTANCE internal validation on TRAIN14 only (§15-§18); memory gated
# by reliability g_mem (§23-§24).  true_U = TRAIN label / offline diagnostic ONLY.
# GRPO / PPO / AC / REINFORCE forbidden (§37).
# ===========================================================================


def _argmax_loss(g, sel_out):
    """R8 §4-§7: best-vs-hardest-competitor pool-argmax margin (primary loss).

    Applies to POSITIVE states only (g["is_stop_target"]/no positive in pool -> 0).
      best = argmax true_U among proposals in the WIDE pool;
      j_hard = argmax_i score(P_i), i != best   (online hard-negative mining, §6);
      L = w * relu(score[j_hard] - score[best] + margin);
      w = clip((U_best - U_j)`/`TO1_R8_UTILITY_SCALE, w_min, w_max)  (§7, clipped so a
      Fattahi-type +200 outlier cannot dominate).
    A half-weighted explicit 'second-best positive' term is added when that proposal is
    NOT the hard competitor (§4).  STOP-target states contribute 0 (STOP head frozen)."""
    prop, _stop = sel_out
    if g["is_stop_target"] or not g["pos_in_pool"] or g["F_pool"] is None:
        return torch.zeros((), dtype=torch.float32)
    U = g["ex"]["true_U"]
    pool = g["pool"]                      # FULL proposal indices (0..N-1)
    wpos = {k: i for i, k in enumerate(pool)}   # full index -> within-pool position
    pos = sorted(g["pos_in_pool"], key=lambda k: -float(U[k]))
    best = pos[0]
    s_b = prop[wpos[best]]
    u_b = float(U[best])
    others = [k for k in pool if k != best]
    if not others:
        return torch.zeros((), dtype=torch.float32)
    hard = max(others, key=lambda k: float(prop[wpos[k]].detach()))
    w = float(np.clip((u_b - float(U[hard])) / C.TO1_R8_UTILITY_SCALE,
                      C.TO1_R8_W_MIN, C.TO1_R8_W_MAX))
    L = w * torch.clamp(prop[wpos[hard]] - s_b + C.TO1_R8_ARGMAX_MARGIN, min=0.0)
    if len(pos) >= 2 and pos[1] != hard:
        w2 = float(np.clip((u_b - float(U[pos[1]])) / C.TO1_R8_UTILITY_SCALE,
                           C.TO1_R8_W_MIN, C.TO1_R8_W_MAX))
        L = L + 0.5 * w2 * torch.clamp(prop[wpos[pos[1]]] - s_b + C.TO1_R8_ARGMAX_MARGIN,
                                       min=0.0)
    return L


def _r8_top1_ce(g, sel_out):
    """R8 §8/§10 - auxiliary Top-1 CE.  Positive states: CrossEntropy over pool∪STOP
    targeted at best_idx (the STOP logit is that of the FROZEN R6 head -> never drags
    Proposal scores down).  STOP-target states contribute ZERO (their CE would push every
    Proposal below the frozen STOP and re-create the R7 stop-recall collapse; ordering
    supervision for those states is covered by the auxiliary pairwise loss)."""
    if g["is_stop_target"]:
        return torch.zeros((), dtype=torch.float32)
    prop, stop = sel_out
    logits = _action_logits((prop, stop), g["M"])
    return nn.functional.cross_entropy(logits.reshape(1, -1),
                                       torch.tensor([g["target_within_pool"]], dtype=torch.long))


def _instance_balanced_order(groups, rng_):
    """R8 §19: sample INSTANCE first, then state within the instance, round-robin, so a
    state-heavy instance (Fattahi15) cannot dominate the batch."""
    by = {}
    for gi_, g_ in enumerate(groups):
        by.setdefault(g_["iid"], []).append(gi_)
    iids = list(by)
    mx = max((len(v) for v in by.values()), default=0)
    order = []
    for r in range(mx):
        rng_.shuffle(iids)
        for ii in iids:
            if r < len(by[ii]):
                order.append(by[ii][r])
    return order


def _instance_folds(iids, n_folds, seed):
    """R8 §15-§16: FIXED instance-level split.  Instances are partitioned (never re-rolled
    by results); an instance's states can never appear on both sides of a fold."""
    n_folds = max(1, min(n_folds, len(iids)))
    rng = random.Random(seed)
    iids = sorted(iids)
    rng.shuffle(iids)
    folds = [[] for _ in range(n_folds)]
    for idx, ii in enumerate(iids):
        folds[idx % n_folds].append(ii)
    out = []
    for f in range(n_folds):
        hold = sorted(folds[f])
        train = sorted([ii for f2, fold in enumerate(folds) if f2 != f for ii in fold])
        out.append((train, hold))
    return out


@torch.no_grad()
def pool_argmax_metrics(groups, selector):
    """R8 §11/§18 - per-state pool-argmax decision + best-positive rank distribution.

    For every state with >=1 positive-in-WIDE-pool proposal:
      best  = argmax_{P in pool} true_U(P);  rank = 1-based pos of best in descending
      selector proposal scores;  argmax_correct = (rank == 1).  selected = argmax over
      (pool ∪ STOP) using the FROZEN STOP head (R8 canonical runtime; §8).  STOP-target
      states: only STOP behavior recorded.  true_U = offline diagnostic only."""
    records = {"n_positive_states": 0, "argmax_ok": 0, "recall": {1: 0, 3: 0, 5: 0, 10: 0},
               "ranks": [], "rg_pool": [], "rg_full": [], "sel_pos": 0,
               "n_stop_tgt": 0, "stop_tgt_stop": 0, "rows": []}
    for g in groups:
        M = g["M"]
        if g["F_pool"] is not None:
            prop, stop = selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        else:
            prop, stop = (torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g))
        if g["pos_in_pool"] and M >= 2:
            U = g["ex"]["true_U"]
            pool = g["pool"]
            order = list(np.argsort(-prop.numpy()))
            best = max(g["pos_in_pool"], key=lambda k: float(U[k]))
            within = pool.index(best)
            rank = order.index(within) + 1
            g0 = records
            g0["n_positive_states"] += 1
            g0["ranks"].append(rank)
            ok = (rank == 1)
            g0["argmax_ok"] += int(ok)
            for k_ in (1, 3, 5, 10):
                g0["recall"][k_] += int(rank <= k_)
            sel = int(_action_logits((prop, stop), M).argmax().item())
            sel_idx = None if sel == M else pool[sel]
            sel_u = 0.0 if sel_idx is None else float(U[sel_idx])
            g0["sel_pos"] += int(sel_u > 0)
            g0["rg_pool"].append(max(0.0, float(U[best]) - sel_u))
            g0["rg_full"].append(max(0.0, g["max_U_full"] - sel_u))
            g0["rows"].append({"gi": g.get("gi", None), "iid": g["iid"],
                               "state_hash": g["state_hash"], "src": g.get("src", "?"),
                               "rank_of_best": rank, "argmax_correct": bool(ok),
                               "best_true_U": float(U[best]), "selected_u": sel_u,
                               "is_stop": bool(sel == M)})
        else:
            records["n_stop_tgt"] += 1
            records["stop_tgt_stop"] += int(int(_action_logits((prop, stop), M).argmax().item()) == M)
    n = records["n_positive_states"]
    ranks = np.array(records["ranks"], dtype=np.float64)
    cm = {}
    for k_ in (1, 3, 5, 10, 20):
        cm[str(k_)] = float((ranks <= k_).mean()) if len(ranks) else 0.0
    return {"n_positive_states": n,
            "pool_argmax_accuracy": float(records["argmax_ok"] / max(n, 1)),
            "positive_state_argmax_accuracy": float(records["argmax_ok"] / max(n, 1)),
            "best_positive_rank_cum": cm,
            "best_positive_rank_mean": float(ranks.mean()) if len(ranks) else 0.0,
            "best_positive_rank_median": float(np.median(ranks)) if len(ranks) else 0.0,
            "recall": {str(k_): float(records["recall"][k_] / max(n, 1)) for k_ in (1, 3, 5, 10)},
            "top1_regret_pool": {"mean": float(np.mean(records["rg_pool"])) if records["rg_pool"] else 0.0,
                                 "median": float(np.median(records["rg_pool"])) if records["rg_pool"] else 0.0},
            "top1_regret_full": {"mean": float(np.mean(records["rg_full"])) if records["rg_full"] else 0.0,
                                 "median": float(np.median(records["rg_full"])) if records["rg_full"] else 0.0},
            "selected_positive_frac": float(records["sel_pos"] / max(n, 1)),
            "stop_target_states": records["n_stop_tgt"],
            "stop_target_selected_stop_frac": float(records["stop_tgt_stop"] / max(records["n_stop_tgt"], 1)),
            "rows": records["rows"]}


# ---------------------------------------------------------------- feature audit
def _feature_families():
    old_end = COL_OLD + 1
    role_end = old_end + ROLE_N
    state_end = role_end + STATE_FEAT_DIM
    return [("scorer_h", slice(0, SCORER_HIDDEN)),
            ("rank", slice(COL_RANK, COL_RANK + 1)),
            ("logit", slice(COL_LOGIT, COL_LOGIT + 1)),
            ("old_base", slice(COL_OLD, old_end)),
            ("role_oh", slice(old_end, role_end)),
            ("state_feat", slice(role_end, state_end)),
            ("mem6", slice(-MEM_FEAT_DIM, None))]


def _scale(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else 0.0, "std": 0.0,
                "min": float(x.min()) if len(x) else 0.0, "max": float(x.max()) if len(x) else 0.0,
                "absmax": float(np.abs(x).max()) if len(x) else 0.0}
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "max": float(x.max()), "absmax": float(np.abs(x).max())}


def _rank_one_misrank_vectors(groups, selector):
    """Split per-state mean F_pool vectors by correct (rank==1) vs misranked."""
    corr, mis = [], []
    for g in groups:
        if not g["pos_in_pool"] or g["F_pool"] is None or g["M"] < 2:
            continue
        prop = np.squeeze(selector.prop_scores(g["F_pool"]).detach().numpy())
        U, pool = g["ex"]["true_U"], g["pool"]
        best = max(g["pos_in_pool"], key=lambda k: float(U[k]))
        rank = int(np.argsort(-prop).tolist().index(pool.index(best))) + 1
        row = g["F_pool"].numpy().mean(axis=0)
        (corr if rank <= 1 else mis).append(row)
    return corr, mis


@torch.no_grad()
def feature_scale_audit(groups, selector):
    """R8 §21-§22 - feature-scale + generalization audit (diagnostic only).

    (1) per-instance scale of {old_base, true_U, rank, logit} over its proposals ->
        which TRAIN instances carry instance-specific magnitudes; (2) correctly-ranked
        vs misranked positive states: mean Cohen's |d| per feature family + the top
        individually-shifted dims.  Never tunes, never suppresses policy-observable
        normalization work -- it only REPORTS so a later normalization round has a target."""
    per_inst = {}
    for g in groups:
        iid = g["iid"]; F = g["F_all"]
        if iid in per_inst or F is None:
            continue
        U = g["ex"]["true_U"].astype(float)
        per_inst[iid] = {"n_prop": int(F.shape[0]),
                         "old_base": _scale(F[:, COL_OLD].numpy().astype(float)),
                         "true_U": _scale(U), "n_pos": int((U > 0).sum()),
                         "rank": _scale(F[:, COL_RANK].numpy().astype(float)),
                         "logit": _scale(F[:, COL_LOGIT].numpy().astype(float))}
    corr, mis = _rank_one_misrank_vectors(groups, selector)
    fam = []
    for name, sl in _feature_families():
        if not corr or not mis:
            fam.append({"family": name, "mean_cohens_d": 0.0, "max_cohens_d": 0.0})
            continue
        a = np.stack(corr)[:, sl]; b = np.stack(mis)[:, sl]
        d = _cohen_d(a, b)
        fam.append({"family": name, "mean_cohens_d": float(np.mean(d)),
                    "max_cohens_d": float(np.max(d))})
    top_dims = _top_shifted_dims(corr, mis)
    return {"per_instance_scale": per_inst,
            "argmax_group": {"n_correct": len(corr), "n_misranked": len(mis)},
            "family_shift": fam, "top_shifted_dims": top_dims}


def _cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = a.shape[0], b.shape[0]
    va = a.var(axis=0); vb = b.var(axis=0)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)) + 1e-9
    return (a.mean(axis=0) - b.mean(axis=0)) / sp


def _top_shifted_dims(corr, mis):
    if not corr or not mis:
        return []
    a = np.stack(corr); b = np.stack(mis)
    d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / (np.sqrt(a.var(axis=0) / a.shape[0] +
                                                         b.var(axis=0) / b.shape[0]) + 1e-9)
    idx = np.argsort(-d)[:10]
    return [{"dim": int(i), "family": _feat_family_name(i), "abs_cohens_se": float(d[i])}
            for i in idx if d[i] >= 0.8]


def _feat_family_name(i):
    total = PROP_FEAT_DIM
    for name, sl in _feature_families():
        start = sl.start if sl.start is not None else 0
        stop = sl.stop if sl.stop is not None else total
        if i >= start and i < stop:
            return name
    return f"dim{i}"


# ---------------------------------------------------------------- R8 training
def train_top1_sft_r8(ex_replay, mem_replay, ex_aug, mem_aug, scorer, reranker,
                      r6_selector, seed=0, epochs_a=None, phase_b=False, epochs_b=None,
                      held_eval_groups=None, record_every=2,
                      lr_a=C.TO1_R8_LR_PHASE_A, lr_b=C.TO1_R8_LR_PHASE_B,
                      log_prefix="[r8]"):
    """R8 pool-argmax SFT (GRPO forbidden, §37).

    Groups = replay ∪ augmented, built with gate_mem=True (R8 memory semantics).
    All selector weights INITIALIZED from the R6 v2 checkpoint (prop backbone + head +
    FROZEN stop_head).  Phase A trains ONLY the top prop-head Linear(128,1) at small LR,
    backbone frozen (§9).  L = lambda_argmax·L_argmax + L_top1_CE(positive-only) +
    lambda_pair·L_pair, with mem6 channel dropout p=TO1_MEM_DROP_P (§25).  Phase B
    (phase_b=True, caller-gated on held-out-instance evidence) additionally unfreezes the
    last shared Linear(128,128) at lr_b.  held_eval_groups: optional held-out-instance
    groups scored each epoch (no grad) to record the cross-instance trajectory (§17)."""
    epochs_a = C.TO1_R8_EPOCHS_A if epochs_a is None else epochs_a
    epochs_b = C.TO1_R8_EPOCHS_B if epochs_b is None else epochs_b
    rng = random.Random(seed)
    groups, _ = build_top1_groups(ex_replay, scorer, mem_replay, reranker=reranker,
                                  rng=rng, gate_mem=True)
    for g_ in groups:
        g_["src"] = "replay"
    n_replay = len(groups)
    n_aug = 0
    if ex_aug:
        groups_a, _ = build_top1_groups(ex_aug, scorer, mem_aug, reranker=reranker,
                                        rng=rng, gate_mem=True)
        for g_ in groups_a:
            g_["src"] = "aug"
        groups += groups_a
        n_aug = len(groups_a)
    print(f"{log_prefix} groups replay={n_replay} aug={n_aug} total={len(groups)} "
          f"pos_states={sum(1 for g in groups if g['has_pos_full'])}", flush=True)

    selector = M3Top1Selector()
    ws = False
    if r6_selector is not None:
        try:
            selector.prop_head.load_state_dict(r6_selector.prop_head.state_dict())
            selector.stop_head.load_state_dict(r6_selector.stop_head.state_dict())
            ws = True
        except Exception:  # noqa: BLE001
            ws = False
    if not ws:
        print(f"{log_prefix} WARN: R6 selector unavailable; R5-reranker init", flush=True)
        selector.warm_start_from(reranker)
    print(f"{log_prefix} selector warm-start from R6 v2 (prop + FROZEN STOP): {ws}", flush=True)

    # ---- freeze policy (R8 §2/§8/§9) --------------------------------------
    for p in selector.stop_head.parameters():
        p.requires_grad_(False)
    for m in selector.prop_head:
        for p in m.parameters():
            p.requires_grad_(False)
    for p in selector.prop_head[-1].parameters():          # Phase A: top head only
        p.requires_grad_(True)

    fail_gis = [i for i, g in enumerate(groups) if g["is_failure"]]
    n_aux = C.TO1_HARD_OVERSAMPLE - 1

    def _epoch_loss(opt, ep, mem_drop):
        selector.train()
        order = _instance_balanced_order(groups, rng)
        order += fail_gis * n_aux
        rng.shuffle(order)
        c_arg = c_ce = c_pair = 0.0
        n_ = 0
        for gi in order:
            g = groups[gi]
            F = g["F_pool"]
            if mem_drop and F is not None and rng.random() < C.TO1_MEM_DROP_P:
                F = F.clone()
                F[..., MEM_CHAN_SLICE] = 0.0
            if F is not None and F.numel():
                prop, stop = selector(F, g["ex"]["state_feat"], g["pool_stats"])
            else:
                prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(selector, g)
            arg = _argmax_loss(g, (prop, stop))
            ce = _r8_top1_ce(g, (prop, stop))
            pair = _pair_loss(g, (prop, stop))
            loss = (C.TO1_R8_LAMBDA_ARGMAX * arg + ce + C.TO1_LAMBDA_PAIR * pair)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for p_ in selector.parameters() if p_.requires_grad], 1.0)
            opt.step()
            c_arg += float(arg.detach()); c_ce += float(ce.detach())
            c_pair += float(pair.detach()); n_ += 1
        selector.eval()
        out = {"ep": ep, "argmax": c_arg / max(n_, 1), "ce": c_ce / max(n_, 1),
               "pair": c_pair / max(n_, 1),
               "pool_argmax_acc": pool_argmax_metrics(groups, selector)["pool_argmax_accuracy"]}
        if held_eval_groups is not None:
            hm = pool_argmax_metrics(held_eval_groups, selector)
            out["held_pa_acc"] = hm["pool_argmax_accuracy"]
            out["held_pos_acc"] = hm["positive_state_argmax_accuracy"]
            out["held_regret"] = hm["top1_regret_full"]["mean"]
        return out

    hist = []
    opt_a = torch.optim.AdamW([p_ for p_ in selector.prop_head[-1].parameters()],
                              lr=lr_a, weight_decay=1e-4)
    for ep in range(epochs_a):
        h = _epoch_loss(opt_a, ep, mem_drop=True)
        hist.append(h)
        if ep in (0, epochs_a - 1) or ep % record_every == record_every - 1:
            _he = (" held_pa=" + f"{h['held_pa_acc']:.3f}" + " held_reg=" + f"{h['held_regret']:.2f}"
                   if "held_pa_acc" in h else "")
            print(f"{log_prefix} A ep {ep} argmax={h['argmax']:.4f} ce={h['ce']:.4f} "
                  f"pa_acc={h['pool_argmax_acc']:.3f}{_he}", flush=True)

    applied_b = False
    if phase_b:
        # last shared Linear(RERANK_HIDDEN,RERANK_HIDDEN) at index 3 + output head
        for mm in (selector.prop_head[3], selector.prop_head[-1]):
            for p in mm.parameters():
                p.requires_grad_(True)
        opt_b = torch.optim.AdamW([p_ for p_ in selector.prop_head.parameters() if p_.requires_grad],
                                  lr=lr_b, weight_decay=1e-4)
        applied_b = True
        for ep in range(epochs_b):
            h = _epoch_loss(opt_b, epochs_a + ep, mem_drop=True)
            hist.append(h)
            if ep in (0, epochs_b - 1) or ep % record_every == record_every - 1:
                print(f"{log_prefix} B ep {h['ep']} argmax={h['argmax']:.4f} "
                      f"pa_acc={h['pool_argmax_acc']:.3f}"
                      + (f" held_pa={h['held_pa_acc']:.3f}" if "held_pa_acc" in h else ""),
                      flush=True)
    phases = {"epochs_a": epochs_a, "phase_b_applied": applied_b,
              "epochs_b": epochs_b if applied_b else 0}
    m_final = pool_argmax_metrics(groups, selector)
    return selector, hist, groups, {"pool_argmax_acc_final": m_final["pool_argmax_accuracy"]}, phases


# ---------------------------------------------------------------------------
# R9 residual selector + instance-mixed training (T1-M3-AUXILIARY-INSTANCE-
# GENERALIZATION-R9): score_R9(P) = score_R6(P) + alpha·tanh(delta(P)), delta≡0
# init, R6 frozen anchor (§13); L_ref preserves R6 on R6-correct states (§14);
# improvement-aware argmax drives delta only where R6 margin is violated (§15);
# instance-mixed batches over BENCHMARK-TRAIN14 + AUX-TRAIN (§12).  GRPO forbidden.
# ---------------------------------------------------------------------------
class M3Top1ResidualSelector(nn.Module):
    """R9 §13: bounded-supervision residual on top of the FROZEN R6 selector.

    score_R9(P) = score_R6(P) + alpha·tanh(delta_theta(P)); delta_theta init ≡ 0, so
    R9 starts bit-identical to R6 and only drifts where gradient pressure exists.
    res_head = shallow MLP over the SAME 277-d proposal features (no representation
    rebuild, §27).  stop_head = deep copy of R6's, FROZEN (same anchor policy, §17).
    Interface mirrors M3Top1Selector so every R8 eval path (pool_argmax_metrics /
    top1_accuracy_stop_audit / closed_loop_top1 / dpp_top1_trace / VAL decomposition)
    works unchanged.
    """

    def __init__(self, r6_selector, alpha: float | None = None):
        super().__init__()
        self.r6 = r6_selector                                   # frozen anchor
        for p in self.r6.parameters():
            p.requires_grad_(False)
        self.stop_head = copy.deepcopy(r6_selector.stop_head)   # R6's STOP, frozen
        for p in self.stop_head.parameters():
            p.requires_grad_(False)
        self.res_head = nn.Sequential(
            nn.Linear(PROP_FEAT_DIM, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.res_head[-1].weight)                # delta ≡ 0 at init
        nn.init.zeros_(self.res_head[-1].bias)
        self.alpha = C.TO1_R9_ALPHA if alpha is None else alpha

    def prop_scores(self, F):
        base = self.r6.prop_scores(F)                          # [N] frozen R6
        delta = self.res_head(F).squeeze(-1)                   # [N]
        return base + self.alpha * torch.tanh(delta)

    def delta_scores(self, F):
        return self.res_head(F).squeeze(-1)

    def forward(self, F_pool, state_feat, pool_stats=None):
        prop = self.prop_scores(F_pool)
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        stop_in = torch.cat([state_feat.reshape(1, -1),
                             pool_stats.reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1)
        return prop, self.stop_head(stop_in).squeeze(0)


@torch.no_grad()
def tag_r6_correct(groups, r6_selector):
    """R9 §14: mark each group ``r6_correct`` = R6 already puts best-positive at
    pool-argmax (rank 1).  Stop-target states are marked correct (protect the frozen
    R6 STOP by default; there is no positive to argue for)."""

    for g in groups:
        if g["pos_in_pool"] and g["M"] >= 2 and g["F_pool"] is not None:
            prop, _ = r6_selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
            U = g["ex"]["true_U"]
            pool = g["pool"]
            order = np.argsort(-prop.numpy()).tolist()
            best = max(g["pos_in_pool"], key=lambda k: float(U[k]))
            within = pool.index(best)
            g["r6_correct"] = bool(order.index(within) == 0)
        else:
            g["r6_correct"] = True


def _ref_loss_r9(g, residual_sel, lam: float):
    """R9 §14: pin delta≈0 on R6-correct states only (never freezes R6 errors)."""

    if not g.get("r6_correct", False) or g["F_pool"] is None:
        return torch.zeros((), dtype=torch.float32)
    d = residual_sel.delta_scores(g["F_pool"])
    return lam * torch.mean(d * d)


def _argmax_loss_r9(g, sel_out):
    """R9 §15: improvement-aware pool-argmax on the COMBINED residual score.
    Identical formula to R8 (only R6-correct margins are already satisfied -> zero
    gradient there; strong correction only where R6 ordering is wrong)."""

    return _argmax_loss(g, sel_out)


def _instance_mixed_batch(groups, rng_, batch_size, mix_ratio=C.TO1_R9_MIX_BENCH_RATIO):
    """R9 §12: sample SOURCE (benchmark 50% / AUX 50%) -> INSTANCE -> state, so every
    mini-batch mixes multiple instances and no burst of updates is single-instance."""

    into = {"bench": [], "aux": []}
    for gi_, g_ in enumerate(groups):
        src = g_.get("src", "bench")
        into.setdefault(src if src in into else "bench", []).append(gi_)
    order = []
    for _ in range(batch_size):
        src = "bench" if rng_.random() < mix_ratio else "aux"
        pool = into.get(src)
        if not pool:
            pool = into["bench"] or into["aux"]
        if not pool:
            return order
        gi_ = rng_.choice(pool)
        order.append(gi_)
    rng_.shuffle(order)
    return order


def train_top1_sft_r9(tr_groups, aux_groups, r6_selector, held_groups=None,
                      seed=0, epochs=None, lr=None, alpha=None,
                      batch_size=None, lambda_ref=None, record_every=2,
                      log_prefix="[r9]"):
    """R9 residual SFT (GRPO forbidden §39).

    tr_groups  : benchmark TRAIN14 groups (can be already a fold-train subset);
    aux_groups : AUX-TRAIN groups (synthetic; 80% split of the AUX set);
    held_groups: eval-only groups scored each epoch WITHOUT grad — caller supplies
                 AUX held-out and/or TRAIN14 held-out groups (never VAL3, §21).
    Only res_head is trained (alpha bounded; R6 prop+stop frozen, §13/§17).
    L = lambda_argmax·L_argmax + lambda_ref·L_ref(R6-correct states only, §14).
    mem6 channel dropout p=TO1_MEM_DROP_P retained (§18/§19)."""
    epochs = C.TO1_R9_EPOCHS if epochs is None else epochs
    lr = C.TO1_R9_LR if lr is None else lr
    alpha = C.TO1_R9_ALPHA if alpha is None else alpha
    batch_size = C.TO1_R9_BATCH if batch_size is None else batch_size
    lambda_ref = C.TO1_R9_LAMBDA_REF if lambda_ref is None else lambda_ref
    groups = tr_groups + aux_groups
    print(f"{log_prefix} groups bench={len(tr_groups)} aux={len(aux_groups)} "
          f"total={len(groups)} pos_states={sum(1 for g in groups if g['has_pos_full'])} "
          f"r6_correct={sum(1 for g in groups if g.get('r6_correct', False))}", flush=True)

    residual = M3Top1ResidualSelector(r6_selector, alpha=alpha)
    opt = torch.optim.AdamW(residual.res_head.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    rng = random.Random(seed)

    def _epoch(ep):
        residual.train()
        c_arg = c_ref = 0.0
        n_ = 0
        for _ in range(3):   # three passes over instance-mixed mini-batches/epoch
            order = _instance_mixed_batch(groups, rng, batch_size)
            for gi_ in order:
                g = groups[gi_]
                F = g["F_pool"]
                if F is not None and rng.random() < C.TO1_MEM_DROP_P:
                    F = F.clone()
                    F[..., MEM_CHAN_SLICE] = 0.0
                if F is not None and F.numel():
                    prop, stop = residual(F, g["ex"]["state_feat"], g["pool_stats"])
                else:
                    prop, stop = torch.zeros(0, dtype=torch.float32), _stop_logits_of(residual, g)
                arg = _argmax_loss_r9(g, (prop, stop))
                ref = _ref_loss_r9(g, residual, lambda_ref)
                loss = (C.TO1_R9_LAMBDA_ARGMAX * arg + ref)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in residual.parameters() if p_.requires_grad], 1.0)
                opt.step()
                c_arg += float(arg.detach()); c_ref += float(ref.detach()); n_ += 1
        residual.eval()
        out = {"ep": ep, "argmax": c_arg / max(n_, 1), "ref": c_ref / max(n_, 1),
               "pa_acc": pool_argmax_metrics(groups, residual)["pool_argmax_accuracy"]}
        if held_groups:
            hm = pool_argmax_metrics(held_groups, residual)
            out["held_pa"] = hm["positive_state_argmax_accuracy"]
            out["held_regret"] = hm["top1_regret_full"]["mean"]
            out["held_recall10"] = hm["recall"]["10"]
            out["held_best_rank_median"] = hm["best_positive_rank_median"]
        return out

    for ep in range(epochs):
        h = _epoch(ep)
        hist.append(h)
        if ep == 0 or ep == epochs - 1 or ep % record_every == record_every - 1:
            _he = (f" held_pa={h['held_pa']:.3f} held_regret={h['held_regret']:.2f} "
                   f"held_recall10={h['held_recall10']:.3f}" if "held_pa" in h else "")
            print(f"{log_prefix} ep {ep} argmax={h['argmax']:.4f} ref={h['ref']:.4f} "
                  f"pa_acc={h['pa_acc']:.3f}{_he}", flush=True)
    return residual, hist


@torch.no_grad()
def r9_preservation_metrics(groups, r6_selector, r9_selector):
    """R9 §25: R6-reference preservation on the R6 replay groups.

    - pooled Spearman between R6 and R9 proposal scores (all states' WIDE pools);
    - r6_correct preserved: fraction of R6-correct states whose best-positive is still
      pool-argmax under R9 (and those that regressed);
    - pool-argmax acc R6 vs R9, top1 regret both, recall@10 both.
    """

    sp_r_rho, sp_r_n = [], 0
    n_correct = 0
    preserved = 0
    regressed = 0
    for g in groups:
        if g["F_pool"] is None or g["M"] < 2:
            continue
        s6, _ = r6_selector(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        s9 = r9_selector.prop_scores(g["F_pool"])
        a6 = s6.numpy()
        a9 = s9.numpy()
        if a6.size >= 2 and a9.size >= 2:
            rho = scipy_spearman(a6, a9)
            sp_r_rho.append(rho)
            sp_r_n += 1
        if g.get("r6_correct"):
            U = g["ex"]["true_U"]
            pool = g["pool"]
            best = max(g["pos_in_pool"], key=lambda k: float(U[k])) if g["pos_in_pool"] else None
            order6 = np.argsort(-a6).tolist()
            order9 = np.argsort(-a9).tolist()
            if best is not None and g["M"] >= 2:
                within = pool.index(best)
                n_correct += 1
                preserved += int(order9.index(within) == 0)
                regressed += int(order6.index(within) == 0 and order9.index(within) != 0)
    return {
        "n_pool_spearman": sp_r_n,
        "spearman_mean": float(np.mean(sp_r_rho)) if sp_r_n else None,
        "r6_correct_states": n_correct,
        "r6_correct_preserved": float(preserved / n_correct) if n_correct else None,
        "r6_correct_regressed": float(regressed / n_correct) if n_correct else None,
        "r6_pa_acc": pool_argmax_metrics(groups, r6_selector)["positive_state_argmax_accuracy"],
        "r9_pa_acc": pool_argmax_metrics(groups, r9_selector)["positive_state_argmax_accuracy"],
        "r6_regret": pool_argmax_metrics(groups, r6_selector)["top1_regret_full"]["mean"],
        "r9_regret": pool_argmax_metrics(groups, r9_selector)["top1_regret_full"]["mean"],
    }


def scipy_spearman(a, b):
    """Pure-numpy Spearman rank correlation (same output as scipy.stats.spearmanr)."""
    ra = np.empty_like(a)
    rb = np.empty_like(b)
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def internal_cv_r9(env, re, scorer, reranker, r6_selector,
                   aux_tr_examples, aux_tr_mem, aux_held_groups,
                   folds=None, seed=None, epochs=None, quick=False,
                   log_prefix="[cv9]"):
    """R9 §23-B/§24/§37 - fixed 3-fold INSTANCE-level internal validation over
    BENCHMARK-TRAIN14.  Each fold trains the residual on (fold-train TRAIN14 ∪
    AUX-TRAIN) with instance-mixed batches, then evaluates ONLY fold-held instances
    (pool-argmax/regret/recall + FixedDecisionReplay closed-loop real + masked) with
    the frozen R6 selector on the SAME held instances as comparator.  AUX held-out is
    evaluated too (three-layer transfer, §23-A).  Model selection uses ONLY these
    held-out-instance numbers -- never VAL3 (§21)."""

    folds = C.TO1_R8_CV_FOLDS if folds is None else folds
    seed = C.TO1_R8_CV_SEED if seed is None else seed
    epochs = C.TO1_R9_CV_EPOCHS if epochs is None else epochs
    by = {}
    for gi, ex in enumerate(re["state_examples"]):
        by.setdefault(ex["iid"], []).append(gi)
    iids = sorted(by)
    if len(iids) < 2 or quick:
        folds = min(folds, max(len(iids), 1))

    # TRAIN groups (replay, gated canonical memory) split by instance on the fly
    grp_all, _ = build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                   reranker=reranker, rng=random.Random(0), gate_mem=True)
    tag_r6_correct(grp_all, r6_selector)
    grp_by_iid = {}
    for g in grp_all:
        grp_by_iid.setdefault(g["iid"], []).append(g)

    # AUX-TRAIN groups (self-contained AUX memory)
    aux_groups, _ = build_top1_groups(aux_tr_examples, scorer, aux_tr_mem,
                                      reranker=reranker, rng=random.Random(0), gate_mem=True)
    tag_r6_correct(aux_groups, r6_selector)
    for g in aux_groups:
        g["src"] = "aux"

    folds_out = []
    agg = {"mean_held_pos_acc_r9": 0.0, "mean_held_pa_acc_r6": 0.0,
           "mean_held_regret_r9": 0.0, "mean_held_regret_r6": 0.0,
           "mean_held_gain_total_r9": 0.0, "mean_held_gain_total_r6": 0.0,
           "mean_held_gain_total_masked": 0.0, "held_argmax_improves_r6": 0.0,
           "mean_best_held_epoch": 0.0, "aux_held_pa": 0.0}
    n_held = 0
    best_epochs = []
    held_pas = []
    for (f_train_iids, f_hold_iids) in _instance_folds(iids, folds, seed):
        f_train_groups = [g for iid in f_train_iids for g in grp_by_iid[iid]]
        for g in f_train_groups:
            g["src"] = "bench"
        residual, hist = train_top1_sft_r9(
            f_train_groups, aux_groups, r6_selector, held_groups=aux_held_groups,
            seed=1000 + len(folds_out), epochs=epochs, log_prefix=f"{log_prefix} f{len(folds_out)}")
        held_groups_f = [g for iid in f_hold_iids for g in grp_by_iid[iid]]
        m9 = pool_argmax_metrics(held_groups_f, residual)
        m6 = pool_argmax_metrics(held_groups_f, r6_selector)
        gain9 = closed_loop_subset(env, re, scorer, residual, f_hold_iids,
                                   use_mem=True, gate_mem=True)
        gain6 = closed_loop_subset(env, re, scorer, r6_selector, f_hold_iids,
                                   use_mem=True, gate_mem=True)
        gainm = closed_loop_subset(env, re, scorer, residual, f_hold_iids,
                                   use_mem=False, gate_mem=False)
        hm = pool_argmax_metrics(aux_held_groups, residual)
        best_ep = 0
        best_pa = -1.0
        for h in hist:
            if h.get("held_pa", -1.0) > best_pa:
                best_pa = h["held_pa"]
                best_ep = h["ep"]
        best_epochs.append(best_ep)
        held_pas.append([h.get("held_pa", None) for h in hist])
        fout = {
            "fold": len(folds_out), "train_instances": f_train_iids,
            "heldout_instances": f_hold_iids, "best_held_epoch": best_ep,
            "r9_held": {"positive_state_argmax_accuracy": float(m9["positive_state_argmax_accuracy"]),
                        "pool_argmax_accuracy": float(m9["pool_argmax_accuracy"]),
                        "top1_regret_full_mean": float(m9["top1_regret_full"]["mean"]),
                        "top1_regret_full_median": float(m9["top1_regret_full"]["median"]),
                        "recall10": float(m9["recall"]["10"]),
                        "best_positive_rank_median": float(m9["best_positive_rank_median"]),
                        "n_positive_states": int(m9["n_positive_states"]),
                        "closed_loop_gain_total": gain9["total"],
                        "closed_loop_gain_per_instance": gain9["per_instance"]},
            "r6_same_held": {"pool_argmax_accuracy": float(m6["pool_argmax_accuracy"]),
                             "top1_regret_full_mean": float(m6["top1_regret_full"]["mean"]),
                             "recall10": float(m6["recall"]["10"]),
                             "n_positive_states": int(m6["n_positive_states"]),
                             "closed_loop_gain_total": gain6["total"],
                             "closed_loop_gain_per_instance": gain6["per_instance"]},
            "masked_held": {"closed_loop_gain_total": gainm["total"]},
            "aux_held": {"positive_state_argmax_accuracy": float(hm["positive_state_argmax_accuracy"]),
                         "top1_regret_full_mean": float(hm["top1_regret_full"]["mean"]),
                         "recall10": float(hm["recall"]["10"]),
                         "n_positive_states": int(hm["n_positive_states"])},
            "trajectory_held_pa": held_pas[-1],
        }
        folds_out.append(fout)
        n_held += 1
        print(f"{log_prefix} f{len(folds_out)-1} hold={sorted(f_hold_iids)} "
              f"r9_pacc={m9['pool_argmax_accuracy']:.3f} r6_pacc={m6['pool_argmax_accuracy']:.3f} "
              f"r9_regret={m9['top1_regret_full']['mean']:.2f} "
              f"gain9={gain9['total']} gain6={gain6['total']} "
              f"aux_held_pa={hm['pool_argmax_accuracy']:.3f} best_ep={best_ep}", flush=True)
    if n_held:
        agg = {
            "mean_held_pos_acc_r9": float(np.mean([f["r9_held"]["positive_state_argmax_accuracy"] for f in folds_out])),
            "mean_held_pa_acc_r6": float(np.mean([f["r6_same_held"]["pool_argmax_accuracy"] for f in folds_out])),
            "mean_held_regret_r9": float(np.mean([f["r9_held"]["top1_regret_full_mean"] for f in folds_out])),
            "mean_held_regret_r6": float(np.mean([f["r6_same_held"]["top1_regret_full_mean"] for f in folds_out])),
            "mean_held_gain_total_r9": float(np.mean([f["r9_held"]["closed_loop_gain_total"] for f in folds_out])),
            "mean_held_gain_total_r6": float(np.mean([f["r6_same_held"]["closed_loop_gain_total"] for f in folds_out])),
            "mean_held_gain_total_masked": float(np.mean([f["masked_held"]["closed_loop_gain_total"] for f in folds_out])),
            "held_argmax_improves_r6": float(np.mean([f["r9_held"]["pool_argmax_accuracy"] for f in folds_out])
                                             - float(np.mean([f["r6_same_held"]["pool_argmax_accuracy"] for f in folds_out]))
                                             ),
            "mean_best_held_epoch": float(np.mean(best_epochs)),
            "aux_held_pa": float(np.mean([f["aux_held"]["positive_state_argmax_accuracy"] for f in folds_out])),
        }
    print(f"{log_prefix} CV agg: held_pos_argmax r9={agg['mean_held_pos_acc_r9']:.3f} vs "
          f"r6={agg['mean_held_pa_acc_r6']:.3f} (Δ={agg['held_argmax_improves_r6']:+.3f}) "
          f"regret r9={agg['mean_held_regret_r9']:.2f} vs r6={agg['mean_held_regret_r6']:.2f} "
          f"closed-loop r9={agg['mean_held_gain_total_r9']:.1f} r6={agg['mean_held_gain_total_r6']:.1f} "
          f"masked={agg['mean_held_gain_total_masked']:.1f} aux_held_pa={agg['aux_held_pa']:.3f} "
          f"best_epoch={agg['mean_best_held_epoch']:.1f}", flush=True)
    return {"folds": folds_out, "aggregate": agg, "n_held_instances": n_held}


# ------------------------------------------------------------ closed-loop subset
def closed_loop_subset(env, re, scorer, selector, iids, use_mem=True, gate_mem=True):
    """R8 §18 - closed-loop FixedDecisionReplay gains restricted to `iids` (hold-out
    instances of the internal CV), canonical g_mem for `gate_mem=True`."""
    gains = {}
    for idx, iid0 in enumerate(iids):
        eid = env["ep_id_of"][iid0] if iid0 in env["ep_id_of"] else len(env["train_insts"]) + idx
        rf = {"problem": env["states"][iid0]["problem"],
              "schedule": env["states"][iid0]["schedule"],
              "iid": iid0, "progmem": re["progmem"], "episode_id": eid}
        g_, _u, _s = rollout_top1(env, rf, scorer, selector, use_mem=use_mem, gate_mem=gate_mem)
        gains[iid0] = g_
    return _b5_summary(gains)


# ------------------------------------------------------------ internal CV (r8)
def internal_cv_r8(env, re, scorer, reranker, r6_selector, aug_examples, aug_mem,
                   folds=None, seed=None, quick=False, log_prefix="[cv]"):
    """R8 §15-§18 - fixed 3-fold INSTANCE-level internal validation (TRAIN14 only).

    TRAIN instances are partitioned BY INSTANCE (never both sides).  Each fold trains
    R8 argmax SFT on fold-train instances and evaluates ONLY fold-heldout instances:
    pool-argmax metrics (real g_mem + masked zero-mem), closed-loop gain (real + masked),
    plus the SAME held metrics for the frozen R6 selector as cross-instance comparator.
    All model selection uses these held-out-instance numbers -- the formal VAL is closed.
    Memory masked-eval asserts mem consistency (§26)."""
    folds = C.TO1_R8_CV_FOLDS if folds is None else folds
    seed = C.TO1_R8_CV_SEED if seed is None else seed
    by = {}
    for ex in re["state_examples"] + aug_examples:
        by.setdefault(ex["iid"], []).append(ex)
    iids = sorted(by)
    if len(iids) < 2 or quick:
        folds = min(folds, max(len(iids), 1))
    mem_map = {}
    for gi, ex in enumerate(re["state_examples"]):
        mem_map[(ex["iid"], ex["state_hash"])] = re["mem_values"][gi]
    for gi, ex in enumerate(aug_examples):
        mem_map[(ex["iid"], ex["state_hash"])] = aug_mem[gi]
    fold_splits = _instance_folds(iids, folds, seed)
    results, held_corr, held_mis = [], [], []
    n_held_inst = 0
    for fi, (tr_iid, hd_iid) in enumerate(fold_splits):
        if not hd_iid:
            continue
        n_held_inst += len(hd_iid)
        tr_ex = [ex for iid0 in tr_iid for ex in by[iid0]]
        tr_mem = [mem_map[(ex["iid"], ex["state_hash"])] for ex in tr_ex]
        hd_ex = [ex for iid0 in hd_iid for ex in by[iid0]]
        hd_mem = [mem_map[(ex["iid"], ex["state_hash"])] for ex in hd_ex]
        hd_gates, _ = build_top1_groups(hd_ex, scorer, hd_mem, reranker=reranker,
                                        rng=random.Random(0), gate_mem=True)
        epa = (2 if quick else C.TO1_R8_EPOCHS_A)
        sel, hist, _grp, _st, _ph = train_top1_sft_r8(
            tr_ex, tr_mem, [], [], scorer, reranker, r6_selector,
            seed=1000 + fi, epochs_a=epa, phase_b=False,
            held_eval_groups=hd_gates, log_prefix=f"{log_prefix} f{fi}")
        best_ep = max(range(len(hist)), key=lambda e_: hist[e_].get("held_pa_acc", -1.0))
        m_r8 = pool_argmax_metrics(hd_gates, sel)
        m_r6 = pool_argmax_metrics(hd_gates, r6_selector)
        hd_mem0 = [torch.zeros_like(x) for x in hd_mem]
        hd_gates0, _ = build_top1_groups(hd_ex, scorer, hd_mem0, reranker=reranker,
                                         rng=random.Random(0), gate_mem=True)
        m_r8m = pool_argmax_metrics(hd_gates0, sel)
        gain_r = closed_loop_subset(env, re, scorer, sel, hd_iid, use_mem=True, gate_mem=True)
        gain_r6 = closed_loop_subset(env, re, scorer, r6_selector, hd_iid, use_mem=True, gate_mem=True)
        gain_m = closed_loop_subset(env, re, scorer, sel, hd_iid, use_mem=False, gate_mem=False)
        c0, m0 = _rank_one_misrank_vectors(hd_gates, sel)
        held_corr += c0; held_mis += m0
        results.append({
            "fold": fi, "train_instances": tr_iid, "heldout_instances": hd_iid,
            "best_held_epoch": best_ep,
            "r6_same_held": {"pool_argmax_accuracy": m_r6["pool_argmax_accuracy"],
                             "positive_state_argmax_accuracy": m_r6["positive_state_argmax_accuracy"],
                             "top1_regret_full_mean": m_r6["top1_regret_full"]["mean"],
                             "recall10": m_r6["recall"]["10"],
                             "n_positive_states": m_r6["n_positive_states"],
                             "closed_loop_gain_total": gain_r6["total"],
                             "closed_loop_gain_per_instance": gain_r6["per_instance"]},
            "r8_held": {"pool_argmax_accuracy": m_r8["pool_argmax_accuracy"],
                        "positive_state_argmax_accuracy": m_r8["positive_state_argmax_accuracy"],
                        "top1_regret_full_mean": m_r8["top1_regret_full"]["mean"],
                        "top1_regret_full_median": m_r8["top1_regret_full"]["median"],
                        "recall": m_r8["recall"], "best_positive_rank_mean": m_r8["best_positive_rank_mean"],
                        "selected_positive_frac": m_r8["selected_positive_frac"],
                        "n_positive_states": m_r8["n_positive_states"],
                        "closed_loop_gain_total": gain_r["total"],
                        "closed_loop_gain_per_instance": gain_r["per_instance"]},
            "r8_masked_held": {"pool_argmax_accuracy": m_r8m["pool_argmax_accuracy"],
                               "positive_state_argmax_accuracy": m_r8m["positive_state_argmax_accuracy"],
                               "top1_regret_full_mean": m_r8m["top1_regret_full"]["mean"],
                               "closed_loop_gain_total": gain_m["total"],
                               "closed_loop_gain_per_instance": gain_m["per_instance"]},
        })
    if not results:
        return {"folds": [], "aggregate": None, "per_heldout_instance": {}, "held_shift": {}}
    def _mean(key, path):
        vals = [r[path][key] for r in results]
        return float(np.mean([v for v in vals if v is not None]))
    agg = {
        "folds": len(results),
        "n_held_instances": n_held_inst,
        "mean_held_pa_acc_r8": _mean("pool_argmax_accuracy", "r8_held"),
        "mean_held_pa_acc_r6": _mean("pool_argmax_accuracy", "r6_same_held"),
        "mean_held_pos_acc_r8": _mean("positive_state_argmax_accuracy", "r8_held"),
        "mean_held_regret_r8": _mean("top1_regret_full_mean", "r8_held"),
        "mean_held_regret_r6": _mean("top1_regret_full_mean", "r6_same_held"),
        "mean_held_gain_total_r8": _mean("closed_loop_gain_total", "r8_held"),
        "mean_held_gain_total_r6": _mean("closed_loop_gain_total", "r6_same_held"),
        "mean_held_gain_total_r8_masked": _mean("closed_loop_gain_total", "r8_masked_held"),
        "held_argmax_improves_r6": float(_mean("pool_argmax_accuracy", "r8_held")
                                         - _mean("pool_argmax_accuracy", "r6_same_held")),
        "mem_masked_ge_real_held": bool(_mean("closed_loop_gain_total", "r8_masked_held")
                                        >= _mean("closed_loop_gain_total", "r8_held")),
        "mean_best_held_epoch": float(np.mean([r["best_held_epoch"] for r in results])),
    }
    shifts = {}
    if held_corr and held_mis:
        fam = []
        for name, sl in _feature_families():
            a = np.stack(held_corr)[:, sl]; b = np.stack(held_mis)[:, sl]
            d = _cohen_d(a, b)
            fam.append({"family": name, "mean_cohens_d": float(np.mean(d)),
                        "max_cohens_d": float(np.max(d))})
        shifts = {"family_shift": fam, "top_shifted_dims": _top_shifted_dims(held_corr, held_mis)}
    per_held = {}
    for r in results:
        pi = r["r8_held"]["closed_loop_gain_per_instance"] or {}
        for iid0, g_ in pi.items():
            per_held.setdefault(iid0, []).append(g_)
    per_held = {k2: float(np.mean(v2)) for k2, v2 in per_held.items()}
    return {"folds": results, "aggregate": agg, "per_heldout_instance": per_held,
            "held_shift": shifts}


# -------------------------------------------------- runtime gate wiring helpers
def _runtime_gate_map(examples, mem_vals, progmem):
    """Attach per-example g_mem when not already present (rollout/dpp/val wrappers)."""
    out = {}
    for gi, ex in enumerate(examples):
        if "mem_gate" in ex and ex["mem_gate"] is not None:
            val = float(ex["mem_gate"])
        else:
            iid = ex["iid"]
            sf = None
            if "state_feat" in ex:
                sf = ex["state_feat"]
            try:
                val = float(progmem.retrieval_gate(iid, ex.get("episode_id", 0), ex.get("step", 0), sf))
            except Exception:  # noqa: BLE001
                val = 1.0
        out[(ex["iid"], ex["state_hash"])] = val
    return out


@torch.no_grad()
def val_failure_decomposition(env, re, scorer, selector, gate_mem=False):
    """R7 §20: VAL no-backward eval.  For every VAL instance at S0 report which
    failure class blocks improvement: candidate_miss / wide_recall_miss /
    proposal_ranking_error / false_STOP / memory_interference / trajectory_compounding.
    true_U used for DIAGNOSTIC ONLY -- never to tune training."""
    cache, executor = env["cache"], env["executor"]
    out = {}
    for idx, vi in enumerate(env["val_insts"]):
        iid = vi["instance_id"]
        st = env["states"][iid]
        problem, schedule = st["problem"], st["schedule"]
        root_ms = int(schedule.makespan)
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        row = {"found": bool(metas)}
        if not metas:
            out[iid] = row
            continue
        ast = cache.ast(problem, schedule, iid)
        h0 = schedule_hash(schedule)
        N = len(metas)
        sf = state_feature_vec(root_ms, root_ms, N, agg["best_uhat"], agg["best_direct"],
                               agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = _rollex_of(ast, metas, prop_feats, sf_t)
        queries = _queries_of(rolex)
        mem_real = torch.tensor(re["progmem"].features(iid, eid, 0, sf, queries),
                                dtype=torch.float32)
        sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
        U = np.full(N, C.INFEASIBLE_U, dtype=np.float32)
        feas = np.zeros(N, dtype=bool)
        for k in range(N):
            res = _execute_step(executor, problem, schedule, _edits_for(ast, metas[k])[0],
                                root_ms, h0)
            if res is None:
                continue
            feas[k] = True
            U[k] = float(res["improvement"])
        pos = [k for k in range(N) if feas[k] and U[k] > 0]
        best = max(pos, key=lambda k: float(U[k])) if pos else None
        row["n_pos"] = len(pos)
        row["oracle_best_true_U"] = float(U[best]) if best is not None else 0.0
        row["candidate_miss"] = best is None
        flags = {}
        logit_pos, rank = _scores(scorer, rolex, mem_real)
        pool, pinfo = wide_pool(rolex, logit_pos, rank)
        pool_set = set(pool)
        mem_sel = mem_real
        if gate_mem:
            try:
                mem_sel = mem_real * float(re["progmem"].retrieval_gate(iid, eid, 0, sf))
            except Exception:  # noqa: BLE001
                mem_sel = mem_real
        F_all = _rerank_feats_all(scorer, rolex, mem_sel)
        row["wide_recall_miss"] = best is not None and best not in pool_set
        if F_all is not None and pool:
            F_pool = F_all[pool]
            ps = _pool_stats_from(F_pool)
            prop, stop = selector(F_pool, sf_t, ps)
            logits = _action_logits((prop, stop), len(pool))
            probs = torch.softmax(logits, dim=-1)
            order = list(np.argsort(-prop.detach().numpy()))
            if best is not None and best in pool_set:
                within = pool.index(best)
                rk_best = order.index(within) + 1
                flags["best_in_pool_top10"] = rk_best <= 10
                flags["proposal_ranking_error"] = rk_best > 1
                flags["best_pool_rank"] = rk_best
                flags["best_score"] = float(prop[within])
                flags["stop_score"] = float(stop[0])
                flags["score_margin"] = float(prop[within] - stop[0])
            sel = int(logits.argmax().item())
            sel_stop = sel == len(pool)
            sel_u = 0.0 if sel_stop else float(U[pool[sel]])
            flags["selected"] = {"is_stop": bool(sel_stop), "true_U": sel_u,
                                 "stop_prob": float(probs[len(pool)])}
            flags["false_stop"] = bool(sel_stop and best is not None and best in pool_set
                                       and flags.get("best_in_pool_top10", False))
            # memory interference: same weights, zeroed mem channel (§18)
            mem0 = torch.zeros_like(mem_real)
            lp0, rk0 = _scores(scorer, rolex, mem0)
            pool0, _ = wide_pool(rolex, lp0, rk0)
            F0 = _rerank_feats_all(scorer, rolex, mem0)[pool0]
            ps0 = _pool_stats_from(F0)
            prop0, stop0 = selector(F0, sf_t, ps0)
            lg0 = _action_logits((prop0, stop0), len(pool0))
            sel0 = int(lg0.argmax().item())
            sel0_stop = sel0 == len(pool0)
            sel0_u = 0.0 if sel0_stop else float(U[pool0[sel0]])
            flags["masked_selected"] = {"is_stop": bool(sel0_stop), "true_U": sel0_u}
            flags["memory_interference"] = bool(
                flags["selected"]["true_U"] < flags["masked_selected"]["true_U"]
                and flags["selected"]["is_stop"])
        row["flags"] = flags
        # trajectory compounding: oracle single-step vs full closed loop
        rf = {"problem": problem, "schedule": schedule, "iid": iid,
              "progmem": re["progmem"], "episode_id": eid}
        gain, usage, steps = rollout_top1(env, rf, scorer, selector, use_mem=True,
                                          gate_mem=gate_mem)
        row["closed_loop_gain"] = gain
        row["trajectory_compounding"] = bool(row["oracle_best_true_U"] > 0 and gain <= 0)
        row["closed_loop_steps"] = steps
        out[iid] = row
        print(f"[r7] VAL {iid}: n_pos={row['n_pos']} oracle_best={row['oracle_best_true_U']} "
              f"closed_loop_gain={gain} {json.dumps({k: v for k, v in flags.items() if k != 'selected'})}",
              flush=True)
    return out


def _old_base_single(prop_feats, k):
    p = prop_feats[k].detach().cpu().numpy() if torch.is_tensor(prop_feats) else prop_feats[k]
    return float(p[296] if p[299] < 0.5 else p[298])


# ============================================================================
# R10 -- T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION
# ============================================================================
# R6 proposal rank is FROZEN.  Every state's pool is robust within-state
# normalized (median/MAD z, §7); a small pool-conditioned STOP calibrator emits
# stop_z from policy-observable pool statistics (§8-§9).  Decision =
# argmax(z_1..z_N, stop_z); STOP only owns ACT-vs-STOP.  All of this is
# deterministic in the proposal ranking: rank_R10 == rank_R6 per state (assert §17).
# ----------------------------------------------------------------------------


def robust_within_state_z(scores):
    """R10 §7: z_i = (s_i - center) / (scale + eps); center=median,
    scale=1.4826*MAD, fallback std, then scale=1 if degenerate.
    Monotonic positive-affine per state -> rank-preserving (ties preserved).
    Returns (z [M], stats dict)."""

    if torch.is_tensor(scores):
        s = scores.detach().cpu().numpy().reshape(-1).astype(np.float64)
    else:
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
    eps = C.TO1_R10_Z_EPS
    if s.size == 0:
        return torch.zeros(0, dtype=torch.float32), {
            "center": 0.0, "scale": 0.0, "mad": 0.0, "std": 0.0, "selected": "empty"}
    center = float(np.median(s))
    mad = float(np.median(np.abs(s - center)))
    scale = float(C.TO1_R10_MAD_CONST * mad)
    selected = "mad"
    if scale < eps:
        scale = float(np.std(s)) if s.size > 1 else 0.0
        selected = "std"
        if scale < eps:
            scale = 1.0
            selected = "one"
    z = (s - center) / (scale + eps)
    return (torch.as_tensor(z, dtype=torch.float32),
            {"center": center, "scale": scale, "mad": mad,
             "std": float(np.std(s)) if s.size > 1 else 0.0, "selected": selected})


class PoolConditionedStopCalibrator(nn.Module):
    """R10 §8-§9: pool-observables + state + old-STOP-raw + g_mem -> stop_z.

    Small arch (Linear->GELU->Linear(1)).  Only authority: ACT-vs-STOP.  It never
    re-ranks Proposals (the ranking belongs to the frozen R6 head)."""

    def __init__(self, in_dim=C.TO1_R10_CALIB_IN_DIM, hidden=C.TO1_R10_CALIB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)     # [*,1] -> [*]


def calibrator_extra(F_pool, state_feat, pool_stats, r6_selector, mem_gate=1.0):
    """R10 policy-observable pool extras [1,CALIB_EXTRA_DIM] (state_feat appended
    by the caller -> CALIB_IN_DIM).

    No true_U, no oracle, no future.  old STOP raw = frozen R6 stop_head (a
    covariate the calibrator may discount -- never the decision itself)."""

    if F_pool is not None and len(F_pool):
        with torch.no_grad():
            raw = r6_selector.prop_scores(F_pool).detach().cpu().numpy().reshape(-1)
    else:
        raw = np.zeros(0, dtype=np.float64)
    n = raw.size
    if n == 0:
        r_max = r_med = r_mad = r_std = 0.0
        gap12 = gap_med = z_max = z_top3 = 0.0
    else:
        r = np.sort(raw)
        r_max = float(r[-1])
        r_med = float(np.median(raw))
        r_mad = float(np.median(np.abs(raw - r_med)))
        r_std = float(np.std(raw)) if n > 1 else 0.0
        gap12 = float(r[-1] - r[-2]) if n >= 2 else 0.0
        gap_med = r_max - r_med
        scale = max(C.TO1_R10_MAD_CONST * r_mad, C.TO1_R10_Z_EPS)
        if scale < C.TO1_R10_Z_EPS or r_mad == 0.0:
            scale = max(r_std, C.TO1_R10_Z_EPS)
        z_vec = (raw - r_med) / (scale + C.TO1_R10_Z_EPS)
        z_max = float(np.max(z_vec))
        top3 = float(np.mean(np.sort(z_vec)[-3:])) if n >= 1 else 0.0
        z_top3 = top3
    stop_in = torch.cat([state_feat.reshape(1, -1),
                         pool_stats.reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1)
    with torch.no_grad():
        old_stop = float(r6_selector.stop_head(stop_in).detach().reshape(-1)[0])
    feats = np.array([[float(n) / _N_POOL_SCALE, r_max, r_med, r_mad, r_std,
                       gap12, gap_med, z_max, z_top3, old_stop, float(mem_gate)]],
                     dtype=np.float32)
    return torch.as_tensor(feats, dtype=torch.float32)


class M3ScoreCalibratedSelector(nn.Module):
    """R10: frozen R6 prop-head (rank anchor) + robust within-state z + calibrator.

    forward(F_pool, state_feat, pool_stats=None) -> (z[M], stop_z[1]).
    prop_scores(F) -> robust z (rank-identical to R6 raw).
    stop_head(stop_in[state|pool_stats]) -> stop_z under a DEgenerate (empty) pool
    context, so _stop_logits_of / top1_accuracy_stop_audit stay interface-identical.
    """

    def __init__(self, r6_selector):
        super().__init__()
        self.r6 = r6_selector
        self.r6.eval()
        for p in self.r6.parameters():
            p.requires_grad_(False)
        self.calibrator = PoolConditionedStopCalibrator()

    def _frozen_raw(self, F):
        return self.r6.prop_scores(F)

    def prop_scores(self, F):
        z, _ = robust_within_state_z(self._frozen_raw(F))
        return z

    def z_raw_stats(self, F_pool):
        """Diagnostics: (z, raw, robust stats)."""
        raw = self._frozen_raw(F_pool)
        z, stats = robust_within_state_z(raw)
        return z, raw, stats

    def forward(self, F_pool, state_feat, pool_stats=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        if F_pool is not None and len(F_pool):
            z, _ = robust_within_state_z(self._frozen_raw(F_pool))
        else:
            z = torch.zeros(0, dtype=torch.float32)
        self._last_mem_gate = 1.0
        extra = calibrator_extra(
            F_pool, state_feat, pool_stats, self.r6, float(self._last_mem_gate))
        x = torch.cat([state_feat.detach().reshape(1, -1), extra], dim=-1)
        stop_z = self.calibrator(x)
        return z, stop_z     # (z[M], [1])

    def stop_head(self, stop_in):
        """Empty-pool path (matches _stop_logits_of's interface)."""
        sf = stop_in[0, :STATE_FEAT_DIM].reshape(1, -1)
        ps = stop_in[0, STATE_FEAT_DIM:].reshape(1, C.TO1_STOP_POOL_STAT_DIM)
        with torch.no_grad():
            old_stop = float(self.r6.stop_head(stop_in).detach().reshape(-1)[0])
        extra = torch.as_tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, old_stop, 1.0]],
            dtype=torch.float32)
        x = torch.cat([sf, extra], dim=-1)
        return self.calibrator(x)       # [1]


def stop_calib_metrics(groups, selector):
    """R10 ACT/STOP decision metrics (the calibrator's scope only)."""
    aud = top1_accuracy_stop_audit(groups, selector)
    fs_pool = float(aud["false_stop_over_pos_pool"])
    stop_recall = float(aud["stop_recall_over_stop_target"])
    act_pool = 1.0 - fs_pool
    fa = float(aud["false_act_over_stop_target"])
    balanced = 0.5 * (act_pool + stop_recall)
    return {"balanced_acc": balanced, "act_recall_pos_pool": act_pool,
            "stop_recall": stop_recall, "false_stop_pos_pool": fs_pool,
            "false_act_stop_target": fa, "n_pos_pool": int(aud["n_pos_pool"]),
            "n_stop_target": int(aud["n_stop_target"]),
            "top1_acc_all": float(aud.get("all", 0.0))}


def assert_rank_preservation(groups, selector10):
    """R10 §17 DETERMINISTIC guarantee-run: per state, rank(z) == rank(R6 raw)
    exactly (Spearman = 1.0, inversions = 0, max |rank move| = 0)."""

    n_states = n_nonempty = 0
    inversions = 0
    worst_spear = 1.0
    worst_move = 0
    for g in groups:
        if g is None or g["F_pool"] is None or g["M"] < 1:
            continue
        n_states += 1
        n_nonempty += 1
        with torch.no_grad():
            raw = selector10._frozen_raw(g["F_pool"]).detach().cpu().numpy()
            z = selector10.prop_scores(g["F_pool"]).detach().cpu().numpy()
        if raw.size < 2:
            continue
        # exact inversion count over all comparable pairs
        d = 0
        for i in range(raw.size):
            for j in range(i + 1, raw.size):
                if (raw[i] - raw[j]) * (z[i] - z[j]) < 0:
                    d += 1
        inversions += d
        order_raw = list(np.argsort(raw, kind="stable"))
        order_z = list(np.argsort(z, kind="stable"))
        move = sum(1 for a, b in zip(order_raw, order_z) if a != b)
        worst_move = max(worst_move, int(move))
        if order_raw != order_z:
            worst_spear = 0.0
    ok = bool(inversions == 0 and worst_move == 0)
    return {"n_states_checked": n_states, "n_nonempty_pools": n_nonempty,
            "rank_inversions": int(inversions), "max_rank_move": int(worst_move),
            "spearman_min": float(worst_spear if n_states else 1.0),
            "rank_preserved": ok}


def _calibration_loss(g, selector10):
    """BCE on (max_z_prop - stop_z) vs ACT/STOP label (max true_U(P_pool) > 0)."""
    M = g["M"]
    if g["F_pool"] is not None and M >= 1:
        prop, stop = selector10(g["F_pool"], g["ex"]["state_feat"], g["pool_stats"])
        max_z = prop.max()                    # no grad (frozen R6 -> deterministic z)
    else:
        stop = selector10.stop_head(torch.cat(
            [g["ex"]["state_feat"].reshape(1, -1),
             g["pool_stats"].reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1))
        max_z = torch.zeros(1)
    logit = (max_z - stop[0]).reshape(1)      # grad flows through stop_z only
    label = torch.tensor(1.0 if g["max_U_pool"] > 0 else 0.0, dtype=torch.float32)
    return nn.functional.binary_cross_entropy_with_logits(logit, label.reshape(1))


def _calibration_indices(groups, rng_):
    """Class-balanced ACT/STOP + instance-balanced ordering (returns group idx list)."""
    act = [i for i, g in enumerate(groups) if g["max_U_pool"] > 0]
    stp = [i for i, g in enumerate(groups) if g["max_U_pool"] <= 0]
    rng_.shuffle(act)
    rng_.shuffle(stp)
    k = max(1, min(len(act), len(stp)))
    order = []
    for a, s in zip(act[:k], stp[:k]):
        order.append(a)
        order.append(s)
    # instance balance: stable shuffle of instances
    by_iid = {}
    for i in order:
        by_iid.setdefault(groups[i]["iid"], []).append(i)
    iids = list(by_iid)
    rng_.shuffle(iids)
    out = [i for iid in iids for i in by_iid[iid]]
    return out


def train_top1_sft_r10(source_groups, r6_selector, seed=0, epochs=C.TO1_R10_EPOCHS,
                       lr=C.TO1_R10_LR, batch_size=C.TO1_R10_BATCH,
                       held_groups=None, log_prefix="[r10]"):
    """Train ONLY the PoolConditionedStopCalibrator; R6 prop/backbone fully frozen.

    source_groups: list of (name, groups) with class-balanced ACT/STOP batches,
    mixed across sources by weight bench:real:syn (TO1_R10_MIX_BENCH_REAL_SYN).
    Returns (selector10, hist)."""

    rng_ = random.Random(seed)
    # deterministic calibrator init: the global torch RNG is advanced by upstream
    # forward passes, so pin it to the fold seed here or CV is run-to-run unstable
    torch.manual_seed(seed)
    weights = list(C.TO1_R10_MIX_BENCH_REAL_SYN[: len(source_groups)])
    wsum = sum(weights)
    selector10 = M3ScoreCalibratedSelector(r6_selector)
    # pre-freeze safety: nothing but the calibrator is trainable
    trainable = [n for n, p in selector10.named_parameters() if p.requires_grad]
    for name in trainable:
        assert "calibrator" in name, f"unexpected trainable param {name}"
    opt = torch.optim.AdamW(selector10.calibrator.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    for ep in range(epochs):
        selector10.train()
        # class-balanced balanced-order per source
        source_pairs = []
        total_pairs = 0
        for (name, groups), w in zip(source_groups, weights):
            gi = _calibration_indices(groups, rng_)
            pairs = [(name, groups[i]) for i in gi]
            source_pairs.append(pairs)
            total_pairs += len(pairs) // 2 * 2          # complete (act, stp) pairs
        # source-proportional pair budget, then round-robin interleave
        order = []
        ptrs = [0] * len(source_pairs)
        budgets = [int(w / wsum * total_pairs * 2) for w in weights]
        any_left = True
        while any_left:
            any_left = False
            for j in range(len(source_pairs)):
                if ptrs[j] >= len(source_pairs[j]) or budgets[j] <= 0:
                    continue
                take = min(batch_size // max(1, len(source_pairs)),
                           len(source_pairs[j]) - ptrs[j], budgets[j])
                if take <= 0:
                    continue
                order.extend(source_pairs[j][ptrs[j]: ptrs[j] + take])
                ptrs[j] += take
                budgets[j] -= take
                any_left = True
        # instance-balanced final shuffle
        by_iid = {}
        for item in order:
            by_iid.setdefault(item[1]["iid"], []).append(item)
        iids = list(by_iid)
        rng_.shuffle(iids)
        order = [it for iid in iids for it in by_iid[iid]]
        batches = [order[i0:i0 + batch_size] for i0 in range(0, len(order), batch_size)]
        rng_.shuffle(batches)
        total = 0.0
        n = 0
        for batch in batches:
            if not batch:
                continue
            loss = torch.stack([_calibration_loss(g, selector10) for _n, g in batch]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector10.calibrator.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            n += 1
        selector10.eval()
        held_m = {}
        if held_groups is not None:
            held_m = stop_calib_metrics(held_groups, selector10)
        hist.append({"ep": ep, "loss": total / max(n, 1), **held_m})
        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
            held_s = (f" held_bal={held_m.get('balanced_acc'):.3f} "
                      f"stop_rec={held_m.get('stop_recall'):.3f} "
                      f"fs_pool={held_m.get('false_stop_pos_pool'):.3f}"
                      if held_m else "")
            print(f"{log_prefix} ep {ep} loss={hist[-1]['loss']:.4f}{held_s}", flush=True)
    return selector10, hist


def internal_cv_r10(env, re, scorer, reranker, r6_selector,
                    aux_real_tr_examples, aux_real_tr_mem, aux_real_held_groups,
                    aux_syn_tr_groups, folds=None, seed=None, epochs=None,
                    quick=False, log_prefix="[cv10]"):
    """R10 §26: fixed 3-fold INSTANCE-level internal CV on BENCHMARK-TRAIN14.

    Per fold the calibrator trains on (fold-train TRAIN14 ∪ AUX-REAL-train ∪
    AUX-synth-train) and validates ONLY on fold-held TRAIN14 instances (balanced
    ACT/STOP acc + STOP recall + false_stop) with the frozen R6 rank.  Model
    selection by held balanced acc -- never VAL3."""

    folds = C.TO1_R8_CV_FOLDS if folds is None else folds
    seedv = C.TO1_R8_CV_SEED if seed is None else seed
    epochs = C.TO1_R10_CV_EPOCHS if epochs is None else epochs

    by = {}
    for gi, ex in enumerate(re["state_examples"]):
        by.setdefault(ex["iid"], []).append(gi)
    iids = sorted(by)
    if len(iids) < 2 or quick:
        folds = min(folds, max(len(iids), 1))

    grp_all, _ = build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                   reranker=reranker, rng=random.Random(0), gate_mem=True)
    grp_by_iid = {}
    for g in grp_all:
        grp_by_iid.setdefault(g["iid"], []).append(g)

    aux_real_groups, _ = build_top1_groups(
        aux_real_tr_examples, scorer, aux_real_tr_mem,
        reranker=reranker, rng=random.Random(0), gate_mem=True)
    for g in aux_real_groups:
        g["src"] = "real"

    folds_out = []
    n_held = 0
    best_epochs = []
    for (f_train_iids, f_hold_iids) in _instance_folds(iids, folds, seedv):
        f_train_groups = [g for iid in f_train_iids for g in grp_by_iid[iid]]
        for g in f_train_groups:
            g["src"] = "bench"
        held_groups_f = [g for iid in f_hold_iids for g in grp_by_iid[iid]]
        sel, hist = train_top1_sft_r10(
            [("bench", f_train_groups), ("real", aux_real_groups),
             ("syn", aux_syn_tr_groups)], r6_selector,
            seed=2000 + len(folds_out), epochs=epochs,
            held_groups=held_groups_f, log_prefix=f"{log_prefix} f{len(folds_out)}")
        hm = stop_calib_metrics(held_groups_f, sel)
        real_hd = stop_calib_metrics(aux_real_held_groups, sel) if aux_real_held_groups else {}
        best_ep, best_bal = 0, -1.0
        for h in hist:
            if h.get("balanced_acc", -1.0) > best_bal:
                best_bal = h["balanced_acc"]
                best_ep = h["ep"]
        best_epochs.append(best_ep)
        folds_out.append({
            "fold": len(folds_out), "train_instances": f_train_iids,
            "heldout_instances": f_hold_iids, "best_held_epoch": best_ep,
            "held_metrics": {k: float(v) for k, v in hm.items()},
            "aux_real_held_metrics": {k: float(v) for k, v in real_hd.items()},
        })
        n_held += 1
        print(f"{log_prefix} f{len(folds_out)-1} hold={sorted(f_hold_iids)} "
              f"bal={hm['balanced_acc']:.3f} stop_rec={hm['stop_recall']:.3f} "
              f"fs_pool={hm['false_stop_pos_pool']:.3f} "
              f"auxreal_held_bal={real_hd.get('balanced_acc', -1):.3f} "
              f"best_ep={best_ep}", flush=True)

    agg = None
    if n_held:
        agg = {
            "mean_held_balanced_acc": float(np.mean([f["held_metrics"]["balanced_acc"]
                                                    for f in folds_out])),
            "mean_held_stop_recall": float(np.mean([f["held_metrics"]["stop_recall"]
                                                    for f in folds_out])),
            "mean_held_false_stop": float(np.mean([f["held_metrics"]["false_stop_pos_pool"]
                                                   for f in folds_out])),
            "mean_held_act_recall": float(np.mean([f["held_metrics"]["act_recall_pos_pool"]
                                                   for f in folds_out])),
            "mean_best_held_epoch": float(np.mean(best_epochs)),
            "aux_real_held_balanced": float(np.mean(
                [f["aux_real_held_metrics"]["balanced_acc"] for f in folds_out
                 if f["aux_real_held_metrics"].get("balanced_acc") is not None]))
            if any(f["aux_real_held_metrics"].get("balanced_acc") is not None
                   for f in folds_out) else None,
        }
    print(f"{log_prefix} CV agg: held_bal={agg['mean_held_balanced_acc']:.3f} "
          f"(stop_rec {agg['mean_held_stop_recall']:.3f}, fs_pool {agg['mean_held_false_stop']:.3f}) "
          f"best_epoch={agg['mean_best_held_epoch']:.1f}", flush=True)
    return {"folds": folds_out, "aggregate": agg, "n_held_instances": n_held}


def val_decomposition_r10(env, re, scorer, selector, gate_mem=False):
    """R10 §30: VAL no-backward decomposition with the calibrated selector.  Per VAL
    instance at S0 report error class A/B/C/D (diagnostic only; true_U never used for
    training): A=WIDE_RECALL_MISS, B=PROPOSAL_RANKING_MISS (best in pool, rank>1),
    C=FALSE_STOP_SCALE (best in pool, rank<=10, selected STOP), D=TRAJECTORY_COMPOUNDING."""

    cache, executor = env["cache"], env["executor"]
    out = {}
    for idx, vi in enumerate(env["val_insts"]):
        iid = vi["instance_id"]
        st = env["states"][iid]
        problem, schedule = st["problem"], st["schedule"]
        root_ms = int(schedule.makespan)
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        row = {"found": bool(metas), "error_class": []}
        if not metas:
            out[iid] = row
            continue
        ast = cache.ast(problem, schedule, iid)
        h0 = schedule_hash(schedule)
        N = len(metas)
        sf = state_feature_vec(root_ms, root_ms, N, agg["best_uhat"], agg["best_direct"],
                               agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = _rollex_of(ast, metas, prop_feats, sf_t)
        queries = _queries_of(rolex)
        mem_real = torch.tensor(re["progmem"].features(iid, eid, 0, sf, queries),
                                dtype=torch.float32)
        mem_sel = mem_real
        if gate_mem:
            try:
                mem_sel = mem_real * float(re["progmem"].retrieval_gate(iid, eid, 0, sf))
            except Exception:  # noqa: BLE001
                mem_sel = mem_real
        sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
        U = np.full(N, C.INFEASIBLE_U, dtype=np.float32)
        feas = np.zeros(N, dtype=bool)
        for k in range(N):
            res = _execute_step(executor, problem, schedule, _edits_for(ast, metas[k])[0],
                                root_ms, h0)
            if res is None:
                continue
            feas[k] = True
            U[k] = float(res["improvement"])
        pos = [k for k in range(N) if feas[k] and U[k] > 0]
        best = max(pos, key=lambda k: float(U[k])) if pos else None
        row["n_pos"] = len(pos)
        row["oracle_best_true_U"] = float(U[best]) if best is not None else 0.0
        if best is None:
            row["error_class"].append("CANDIDATE_MISS_NO_POSITIVE")
            out[iid] = row
            continue
        logit_pos, rank = _scores(scorer, rolex, mem_sel)
        pool, _ = wide_pool(rolex, logit_pos, rank)
        pool_set = set(pool)
        F_all = _rerank_feats_all(scorer, rolex, mem_sel)
        if F_all is None or not pool:
            row["error_class"].append("NO_POOL")
            out[iid] = row
            continue
        F_pool = F_all[pool]
        ps = _pool_stats_from(F_pool)
        z, raw, nstats = selector.z_raw_stats(F_pool)
        prop, stop = selector(F_pool, sf_t, ps)
        logits = _action_logits((prop, stop), len(pool))
        order = list(np.argsort(-prop.detach().numpy()))
        best_in_pool = best in pool_set
        flags = {}
        if best_in_pool:
            within = pool.index(best)
            rk = order.index(within) + 1
            flags["best_pool_rank"] = rk
            flags["raw_best_score"] = float(raw[within])
            flags["z_best"] = float(z[within])
            flags["raw_median"] = nstats["center"]
            flags["raw_MAD"] = nstats["mad"]
            flags["raw_scale"] = nstats["scale"]
            flags["norm_selected"] = nstats["selected"]
        else:
            row["error_class"].append("A_WIDE_RECALL_MISS")
        sel = int(logits.argmax().item())
        sel_stop = sel == len(pool)
        sel_u = 0.0 if sel_stop else float(U[pool[sel]])
        flags["selected"] = {"is_stop": bool(sel_stop), "true_U": sel_u}
        flags["stop_z"] = float(stop[0])
        flags["max_z_prop"] = float(prop.max())
        if best_in_pool and rk <= 10:
            if sel_stop:
                row["error_class"].append("C_FALSE_STOP_SCALE")
                flags["false_stop_fixed"] = False
            else:
                flags["false_stop_fixed"] = True
        elif best_in_pool and rk > 1:
            row["error_class"].append("B_PROPOSAL_RANKING_MISS")
        else:
            row["error_class"].append("OK_BEST_SELECTED" if not sel_stop
                                      else "OK_STOP_NO_BEST_IN_POOL")
        row["flags"] = flags
        gain, usage, steps = rollout_top1(env, {"problem": problem, "schedule": schedule,
                                                "iid": iid, "progmem": re["progmem"],
                                                "episode_id": eid},
                                          scorer, selector, use_mem=True, gate_mem=gate_mem)
        row["closed_loop_gain"] = gain
        row["trajectory_compounding"] = bool(row["oracle_best_true_U"] > 0 and gain <= 0)
        if row["trajectory_compounding"] and "C_FALSE_STOP_SCALE" not in row["error_class"] \
           and not sel_stop:
            row["error_class"].append("D_TRAJECTORY_COMPOUNDING")
        row["closed_loop_steps"] = steps
        out[iid] = row
        def _f(v):
            return "NA" if v is None else f"{v:.2f}"
        print(f"[r10] VAL {iid}: n_pos={row['n_pos']} oracle={row['oracle_best_true_U']} "
              f"classes={row['error_class']} sel={flags.get('selected')} "
              f"rank={flags.get('best_pool_rank')} raw={_f(flags.get('raw_best_score'))}/"
              f"med={_f(flags.get('raw_median'))}/z={_f(flags.get('z_best'))} "
              f"stop_z={_f(flags.get('stop_z'))} gain={gain}", flush=True)
    return out


def r10_dpp_trace(iid, episode_id, st, env, re, scorer, selector, s0_trueU_by_sig,
                  gate_mem=True):
    """R10 §28: DPpaulli-style s0 trace with the calibrated selector, printing the RAW
    best score / raw STOP / median/MAD / normalized best z / calibrated stop_z /
    selected proposal / true_U diagnostic."""

    base = dpp_top1_trace(iid, episode_id, st, env, re, scorer, selector,
                          s0_trueU_by_sig, gate_mem=gate_mem)
    if not base.get("found"):
        return base
    out = dict(base)
    cache, executor = env["cache"], env["executor"]
    problem, schedule = st["problem"], st["schedule"]
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    ast = cache.ast(problem, schedule, iid)
    h0 = schedule_hash(schedule)
    N = len(metas)
    sf = state_feature_vec(int(schedule.makespan), int(schedule.makespan),
                           N, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _rollex_of(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(re["progmem"].features(iid, episode_id, 0, sf, queries),
                       dtype=torch.float32)
    logit_pos, rank = _scores(scorer, rolex, mem)
    pool, _ = wide_pool(rolex, logit_pos, rank)
    F_all = _rerank_feats_all(scorer, rolex, mem)
    if F_all is not None and pool:
        F_pool = F_all[pool]
        ps = _pool_stats_from(F_pool)
        z, raw, nstats = selector.z_raw_stats(F_pool)
        prop, stop = selector(F_pool, sf_t, ps)
        order = list(np.argsort(-prop.detach().numpy()))
        best = None
        sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
        pos = [k for k in range(N) if s0_trueU_by_sig.get(sigs[k], 0) > 0]
        if pos:
            best = max(pos, key=lambda k: s0_trueU_by_sig[sigs[k]])
        out["diagnostic"] = {
            "raw_best_score": (float(raw[pool.index(best)]) if best in set(pool) else None),
            "raw_stop_old": out.get("stop", {}).get("score"),
            "raw_median": nstats["center"], "raw_MAD": nstats["mad"],
            "raw_scale": nstats["scale"], "norm_selected": nstats["selected"],
            "z_best": (float(z[pool.index(best)]) if best in set(pool) else None),
            "max_z_prop": float(prop.max().detach()),
            "stop_z": float(stop[0].detach()),
            "rank_of_best_norm": (order.index(pool.index(best)) + 1
                                  if best in set(pool) else None),
        }
        out["selected_norm"] = out["selected"]
    return out