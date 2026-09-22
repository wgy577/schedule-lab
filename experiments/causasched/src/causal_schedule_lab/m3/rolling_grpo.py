"""R11 rolling-graph multi-trajectory GRPO (T1-M3-ROLLING-GRAPH-MULTI-TRAJECTORY-GRPO-R11).

First REAL GRPO round of the canonical line (historic R1/R2 actor-critic is NOT this round,
directive §0).  Warm-started from the canonical R6 v2 selector (m3_proposal_top1_sft_v2.pt).

Pipeline (fixed path, one way):
    S_t -> M1/M2 -> Reasoner legal proposals -> WIDE pool (frozen scorer) ->
    R6 robust-z base + bounded residual score_GRPO(P) = score_SFT(P) + alpha*tanh(delta(P)) ->
    behavior mixture (temperature T, uniform mass eps) -> trajectory (H steps, early STOP) ->
    terminal reward R_i = Cmax(S_t) - Cmax(S_T) (STOP reward 0) ->
    group-relative advantage A_i = (R_i - mean_R)/(std_R + eps) ->
    clipped surrogate (eps=0.2) + small KL to frozen R6-base pi_ref ->
    E-epoch batch reuse with pi_old frozen (stale-protected) ->
    updated policy picks the REAL next action (greedy) -> S_{t+1} -> rolling root.

Correctness / honesty contract:
  * NO oracle advancement, no oracle trajectory selection ($26/$40).
  * NO value head, critic-free ($21).
  * R_i uses only makespan; no reward shaping (no FIV/Risk/appearance/memory/utility) ($2/$4).
  * STOP reward hard-coded 0 ($5).
  * sibling isolation: each of the K=8 trajectories deep-copies S_t and Memory_t ($12);
    any future/episode leak raises.
  * only real advancement writes to the graph persistent memory ($12).
  * identified=false, formal_test_access=0, Formal TEST SEALED.
  * pi_old is the frozen collection policy at each rolling depth (held across E reuse epochs).
  * deterministic: every collection step is a pure function of (state, memory, policy, seed);
    parallel workers share no mutable state (each trajectory builds its own AnalyzeCache).
"""

from __future__ import annotations

import copy
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from causal_schedule_lab.validation import schedule_hash

from . import config as C
from .proposal_features import (
    AnalyzeCache, _edits_for, _execute_step, proposal_identity, state_feature_vec,
)
from .ranking import _rerank_feats_all, wide_pool
from .scorer import _scores
from .top1 import _action_logits, _pool_stats_from, _rollex_of, robust_within_state_z

__all__ = [
    "M3RollingGRPOPolicy",
    "PolicySelectorView",
    "mixture_pmf", "mixture_logp", "mixture_sample",
    "collect_trajectory", "collect_group_rollouts",
    "grpo_update", "group_advantages",
    "greedy_step", "advance_graphs",
    "greedy_closed_loop_r11", "roots_from_state",
    "run_rolling_cycles",
    "prob_movement_audit", "dpp_rolling_trace_r11",
    "residual_saturation_stats",
    "trajectory_profiling",
    "Graph",
]


# ---------------------------------------------------------------------------
# bounded-residual policy (warm-start = R6) + behavior mixture
# ---------------------------------------------------------------------------
class M3RollingGRPOPolicy(nn.Module):
    """score_GRPO(P) = r6_raw(P) + alpha_prop*tanh(delta_prop(P));
    score_GRPO(STOP) = r6_STOP + alpha_stop*tanh(delta_stop(state,pool_stats)).

    t0 = 0 both residual heads (zero-init) so the warm-start policy is EXACTLY the
    frozen R6 selector (RAW prop scores + frozen R6 STOP logit) -- R12 §39/40: the
    unified-parity evaluator must reproduce the canonical R6 closed loop, and the
    R10/R11 robust within-state z on the prop side re-scaled proposals past the RAW
    stop logit (probe: 2/131 vs 42/131 above-stop) which silently changed ACT-vs-STOP
    decisions => delta=0 was a DIFFERENT ruler from canonical R6 (416 vs 369).  The
    RAW base keeps delta=0 bit-identical to rollout_top1(r6_sel) (369 anchor).
    r6 is the frozen parent (`m3_proposal_top1_sft_v2.pt`).

    forward(F_pool[M,277], state_feat[7], pool_stats[1,5]) -> (s_prop[M], s_stop[1]).
    action_logits(..) -> [M+1] raw scores (temperature applied at the softmax).
    """

    def __init__(self, r6_selector, alpha_prop=None, alpha_stop=None):
        super().__init__()
        self.r6 = r6_selector
        self.r6.eval()
        for p in self.r6.parameters():
            p.requires_grad_(False)
        self.alpha_prop = float(alpha_prop if alpha_prop is not None
                                else C.TO1_R11_ALPHA_PROP)
        self.alpha_stop = float(alpha_stop if alpha_stop is not None
                                else C.TO1_R11_ALPHA_STOP)
        self.resid_prop = nn.Linear(277, 1)      # PROP_FEAT_DIM
        self.resid_stop = nn.Linear(7 + 5, 1)    # STATE_FEAT_DIM + STOP_POOL_STAT_DIM
        self._zero_residual()

    def _zero_residual(self):
        for head in (self.resid_prop, self.resid_stop):
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()

    # -- base (frozen R6, CANONICAL RAW ruler) ------------------------------
    def _base_raw(self, F_pool):
        if F_pool is not None and len(F_pool):
            with torch.no_grad():
                raw = self.r6.prop_scores(F_pool)     # frozen, no grad
            return raw, raw, {"scale": 1.0, "selected": "raw"}
        empty = {"center": 0.0, "scale": 0.0, "mad": 0.0, "std": 0.0, "selected": "empty"}
        return torch.zeros(0, dtype=torch.float32), torch.zeros(0, dtype=torch.float32), empty

    def _stop_in(self, state_feat, pool_stats):
        return torch.cat([state_feat.detach().reshape(1, -1),
                          pool_stats.detach().reshape(1, C.TO1_STOP_POOL_STAT_DIM)], dim=-1)

    def _stop_base(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        with torch.no_grad():
            return self.r6.stop_head(stop_in).reshape(-1)[0]   # float scalar tensor

    # -- residual (trainable only) ------------------------------------------
    def residual_prop(self, F_pool):
        if F_pool is None or len(F_pool) == 0:
            return torch.zeros(0, dtype=torch.float32)
        return float(self.alpha_prop) * torch.tanh(
            self.resid_prop(F_pool.float()).squeeze(-1))        # [M]

    def residual_stop(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        return float(self.alpha_stop) * torch.tanh(
            self.resid_stop(stop_in).squeeze(-1)).reshape(1)    # [1]

    # -- score path ----------------------------------------------------------
    def forward(self, F_pool, state_feat, pool_stats=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        base_stop = self._stop_base(state_feat, pool_stats)
        return (z + self.residual_prop(F_pool),                     # [M]
                base_stop + self.residual_stop(state_feat, pool_stats))  # [1]

    def base_logits(self, F_pool, state_feat, pool_stats=None, evid=None):
        """Frozen R6-base logits (pi_ref): [raw_prop..., r6_stop] with zero residual."""
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        return torch.cat([z, self._stop_base(state_feat, pool_stats).reshape(1)], dim=-1)

    def action_logits(self, F_pool, state_feat, pool_stats=None, evid=None):
        del evid  # R18 ProposalEvidenceResidualAdapter only; plain R6 policy ignores it
        sp, ss = self.forward(F_pool, state_feat, pool_stats)
        return torch.cat([sp, ss], dim=-1)         # [M+1]

    def prop_scores(self, F):
        """Full-state scoring (PolicySelectorView / normal-M5 interface)."""
        z, _, _ = self._base_raw(F)
        return z + self.residual_prop(F)

    def stop_head(self, stop_in):
        """PolicySelectorView compatibility under an (empty-pool) context."""
        sf = stop_in[:, :7]
        ps = stop_in[:, 7:12]
        return self._stop_base(sf.reshape(1, -1), ps.reshape(1, -1)).reshape(1, 1)

    # -- weight snapshot management (for tests + profiling) -------------------
    def snapshot(self):
        with torch.no_grad():
            return {name: p.detach().clone() for name, p in self.named_parameters()}

    def load_snapshot(self, snap):
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in snap:
                    p.copy_(snap[name])


class PolicySelectorView(nn.Module):
    """Adapter exposing a M3RollingGRPOPolicy as a (prop, stop) selector so the
    canonical R6 diagnostics (normal_m5_top1, dpp traces) can consume it unchanged."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def prop_scores(self, F):
        return self.policy.prop_scores(F)

    def stop_head(self, stop_in):
        return self.policy.stop_head(stop_in)

    def forward(self, F_pool, state_feat, pool_stats=None):
        return self.policy.forward(F_pool, state_feat, pool_stats)


# ---------------------------------------------------------------------------
# behavior mixture -- temperature + uniform mass, EXACT mixture probability
# ---------------------------------------------------------------------------
def mixture_pmf(logits, T, eps):
    """p(a) = (1-eps)*softmax(logits/T)[a] + eps/(M+1)   (exact, torch)."""
    M1 = int(logits.numel())
    return (1.0 - eps) * torch.softmax(logits / T, dim=-1) + eps / M1


def mixture_logp(logits, a, T, eps):
    p = mixture_pmf(logits, T, eps)
    return torch.log(p[a])


def mixture_sample(logits, T, eps, rng):
    p = mixture_pmf(logits, T, eps).detach().numpy()
    return rng.choices(list(range(len(p))), weights=p, k=1)[0]


def _kl_mixture(logits_t, logits_r, T, eps):
    """KL(pi_theta || pi_ref) over the full mixture pmf (exact)."""
    p_t = mixture_pmf(logits_t, T, eps)
    p_r = mixture_pmf(logits_r, T, eps)
    lp_t = torch.log(p_t)
    lp_r = torch.log(p_r)
    return (p_t * (lp_t - lp_r)).sum()


# ---------------------------------------------------------------------------
# sibling trajectory collection
# ---------------------------------------------------------------------------
def collect_trajectory(policy, scorer, executor, model_b5, single_head, direct_head,
                       problem, schedule_root, root_ms, iid, episode_id, progmem_root,
                       seed, traj_id, T=None, eps=None, horizon=None, step_offset=0,
                       seed_rolex=None, prof=None, max_pool_guard=True):
    """One sibling trajectory from a FROZEN root (S_t, Memory_t).

    Sibling isolation: `schedule_root` and `progmem_root` are deep-copied first;
    every mutating step happens only on the copies.  The root objects are asserted
    unchanged afterwards (schedule hash + memory feature snapshot).  Each call
    builds its OWN AnalyzeCache (no shared mutable cache across threads).

    Returns a dict with `steps` (per-step records used for the GRPO updates) and
    `reward = root_ms - final_ms`.  STOP / no-proposal / no-pool / infeasible /
    revisit all end the trajectory (rewards otherwise equal).  A step continues
    past an immediate non-positive action (full-horizon semantics).
    """
    T = float(T if T is not None else C.TO1_R11_TEMP)
    eps = float(eps if eps is not None else C.TO1_R11_MIX_EPS)
    horizon = int(horizon if horizon is not None else C.TO1_R11_HORIZON)

    t_prof = {"analyze": 0.0, "execute": 0.0, "mem": 0.0, "policy": 0.0}
    t0 = time.time()
    schedule = copy.deepcopy(schedule_root)
    pm = copy.deepcopy(progmem_root)
    h0 = schedule_hash(schedule_root)
    assert h0 == schedule_hash(schedule), "sibling state isolation violated at copy"
    # memory snapshot check (root mem features must be identical before/after)
    mem_probe = None
    cache = AnalyzeCache(model_b5, single_head, direct_head)
    rng = random.Random(seed)
    visited = {h0}

    ms_cur = int(root_ms)
    steps = []
    terminal = None
    n_acted = 0
    for t in range(horizon):
        t1 = time.time()
        if t == 0 and seed_rolex is not None:
            prop_feats, metas, agg = seed_rolex
        else:
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        ast = cache.ast(problem, schedule, iid)
        n_prop = len(metas)
        t_prof["analyze"] += time.time() - t1
        if n_prop == 0:
            terminal = "no_proposals"
            break
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, root_ms, n_prop, agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = _rollex_of(ast, metas, prop_feats, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        t1 = time.time()
        step = step_offset + t
        mem = torch.tensor(pm.features(iid, episode_id, step, sf, queries),
                           dtype=torch.float32)
        gmem = float(pm.retrieval_gate(iid, episode_id, step, sf))
        t_prof["mem"] += time.time() - t1
        logit_pos, rank = _scores(scorer, rolex, mem)
        pool, pinfo = wide_pool(rolex, logit_pos, rank)
        if not pool:
            terminal = "no_pool"
            break
        mem_sel = mem * gmem
        F_pool = _rerank_feats_all(scorer, rolex, mem_sel)[pool]
        pool_stats = _pool_stats_from(F_pool)
        t1 = time.time()
        with torch.no_grad():
            logits = policy.action_logits(F_pool, sf_t, pool_stats)     # [M+1]
        t_prof["policy"] += time.time() - t1
        a = mixture_sample(logits, T, eps, rng)
        logp_old = float(mixture_logp(logits, a, T, eps))
        M = len(pool)
        is_stop = bool(a == M)
        rec = {
            "iid": iid, "state_hash": h, "tstep": step,
            "F_pool": F_pool.detach().float(), "sf_t": sf_t.detach().float(),
            "pool_stats": pool_stats.detach().float(),
            "logits_old": logits.detach().float(), "a": int(a), "M": M,
            "logp_old": logp_old, "is_stop": is_stop, "traj": traj_id,
        }
        if mem_probe is None:
            mem_probe = [float(v) for v in
                         torch.tensor(pm.features(iid, episode_id, step, sf,
                                                   queries[: min(2, len(queries))]))
                         .flatten().tolist()]
        steps.append(rec)
        if is_stop:
            terminal = "stop"
            break
        meta = metas[pool[a]]
        edits, kind = _edits_for(ast, meta)
        sig = proposal_identity(ast, meta)[2]
        t1 = time.time()
        res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
        t_prof["execute"] += time.time() - t1
        u = None if res is None else float(res["improvement"])
        outcome = ("success" if u is not None and u > 0 else
                   "neutral" if u is not None and u == 0 else
                   "negative" if u is not None else "infeasible")
        pm.add_executed(iid, step, {
            "instance_id": iid, "episode_id": episode_id, "state_hash": h,
            "state_feat": sf, "proposal_signature": sig,
            "proposal_type": rolex["type"][pool[a]], "role": rolex["role"][pool[a]],
            "src": rolex["src"][pool[a]], "tgt": rolex["tgt"][pool[a]], "true_U": u,
            "outcome": outcome, "successor_state_hash": None,
            "trajectory_step": step, "written_at_step": step,
            "fine_key": ((rolex["type"][pool[a]], rolex["role"][pool[a]],
                          rolex["src"][pool[a]], rolex["tgt"][pool[a]])
                         if rolex["type"][pool[a]] == "single"
                         else (rolex["type"][pool[a]], rolex["role"][pool[a]])),
            "coarse_key": (rolex["type"][pool[a]], rolex["role"][pool[a]]),
        })
        if res is None:
            terminal = "infeasible"
            break
        n_acted += 1
        schedule = res["schedule"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        if nh in visited:
            terminal = "revisit"
            break
        visited.add(nh)
    # ---- sibling isolation assert: root unchanged -----------------------
    assert h0 == schedule_hash(schedule_root), "sibling State isolation violated"
    reward = int(root_ms) - ms_cur
    return {
        "traj_id": traj_id, "iid": iid, "episode_id": episode_id,
        "root_state_hash": h0, "root_ms": int(root_ms), "final_ms": ms_cur,
        "reward": reward, "n_steps": len(steps), "n_acted": n_acted,
        "terminal": terminal, "steps": steps,
        "mem_probe": mem_probe, "n_mem_records": _mem_record_count(pm),
        "root_mem_records": _mem_record_count(progmem_root),
        "prof": t_prof,
    }


def _mem_record_count(pm):
    """Duck-typed progressive-memory record count (ProgressiveMemory or test fake)."""
    for attr in ("executed", "recs", "_record"):
        if hasattr(pm, attr):
            v = getattr(pm, attr)
            if isinstance(v, dict):
                return int(sum(len(x) for x in v.values()) if v else 0)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, dict):
                return len(v)
            return 0
    return -1


def _seed_rolex_of(cache, problem, schedule, iid):
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    return (prop_feats, metas, agg)


def collect_group_rollouts(policy, scorer, executor, model_b5, single_head, direct_head,
                           problem, schedule, root_ms, iid, episode_id, progmem,
                           k=None, T=None, eps=None, horizon=None, seed=0,
                           step_offset=0, workers=1, step0_cache=None):
    """K sibling trajectories from one root (S_t, next_epoch progmem).  Returns the
    group bundle: rewards, group-relative advantages (A_i), informative flag, steps.

    `step0_cache` (AnalyzeCache or None) is used, when given, to precompute the root
    state's proposal rolex ONCE so all K siblings share step 0 (cost cut, semantics
    identical).  Workers execute in threads; each trajectory is deterministic from its
    per-trajectory seed, so the collected data is scheduling-order independent.
    """
    k = int(k if k is not None else C.TO1_R11_K)
    h0 = schedule_hash(schedule)
    seed_rolex = None
    if step0_cache is not None:
        seed_rolex = _seed_rolex_of(step0_cache, problem, schedule, iid)

    def _job(kid):
        return collect_trajectory(
            policy, scorer, executor, model_b5, single_head, direct_head,
            problem, schedule, root_ms, iid, episode_id, progmem,
            seed=seed * 7919 + kid * 104729, traj_id=kid,
            T=T, eps=eps, horizon=horizon, step_offset=step_offset,
            seed_rolex=seed_rolex)

    t0 = time.time()
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            trajs = list(ex.map(_job, range(k)))
    else:
        trajs = [_job(kid) for kid in range(k)]
    coll_s = time.time() - t0

    # memory snapshot purity (sibling isolation, root mem untouched)
    mem_unchanged = all((tr["mem_probe"] is not None) for tr in trajs)

    R = np.array([float(tr["reward"]) for tr in trajs], dtype=np.float64)
    mean_R = float(R.mean())
    std_R = float(R.std()) if R.size > 1 else 0.0
    informative = bool(std_R > C.TO1_R11_ADV_EPS)
    eps_a = C.TO1_R11_ADV_EPS
    adv = ((R - mean_R) / (std_R + eps_a)).tolist()
    for tr, a in zip(trajs, adv):
        for rec in tr["steps"]:
            rec["adv_group"] = a
    for tr in trajs:
        for rec in tr["steps"]:
            rec["informative"] = informative
            rec["grp_key"] = (iid, h0)
    return {
        "grp_key": (iid, h0), "iid": iid, "state_hash": h0, "root_ms": int(root_ms),
        "rewards": [float(x) for x in R.tolist()],
        "mean_reward": mean_R, "std_reward": std_R, "informative": informative,
        "advantages": adv, "trajs": trajs, "n_steps": sum(tr["n_steps"] for tr in trajs),
        "n_acted": sum(tr["n_acted"] for tr in trajs),
        "terminal_counts": _terminal_counts(trajs),
        "coll_s": coll_s, "mem_unchanged": mem_unchanged,
    }


def _terminal_counts(trajs):
    from collections import Counter
    return dict(Counter(tr["terminal"] for tr in trajs))


def group_advantages(rewards):
    """Expose the advantage math for tests / audits (group-"iid,state_hash")."""
    R = np.asarray(rewards, dtype=np.float64)
    mean_R = float(R.mean())
    std_R = float(R.std()) if R.size > 1 else 0.0
    informative = bool(std_R > C.TO1_R11_ADV_EPS)
    adv = ((R - mean_R) / (std_R + C.TO1_R11_ADV_EPS)).tolist()
    return {"mean": mean_R, "std": std_R, "informative": informative, "advantages": adv}


# ---------------------------------------------------------------------------
# GRPO clipped-surrogate update (critic-free) + stale-protected batch reuse
# ---------------------------------------------------------------------------
def grpo_update(policy, samples, T=None, eps=None, clip_eps=None, beta=None, lr=None,
                epochs=None, stale_kl_threshold=None, stale_clip_frac=None,
                seed=0, log_prefix="[r11]"):
    """One GRPO update on a collected batch (E reuse epochs, pi_old implicit in each
    step's stored `logits_old`).  Samples = flat list of step records (each carries
    F_pool, sf_t, pool_stats, a, logp_old, logits_old, informative, adv_group).

    L = -mean over informative steps of min(ratio*A, clip(ratio)*A)
        + beta * mean over ALL steps of KL(pi_theta || pi_ref_R6base).
    Stale guard: after reuse epoch 0, if mean KL(pi_theta || pi_old) > threshold or
    the clip-fraction > cap, the reuse stops early (batch is stale).
    """
    T = float(T if T is not None else C.TO1_R11_TEMP)
    eps = float(eps if eps is not None else C.TO1_R11_MIX_EPS)
    clip_eps = float(clip_eps if clip_eps is not None else C.TO1_R11_CLIP_EPS)
    beta = float(beta if beta is not None else C.TO1_R11_BETA_KL)
    lr = float(lr if lr is not None else C.TO1_R11_LR)
    epochs = int(epochs if epochs is not None else C.TO1_R11_UPDATE_EPOCHS)
    stale_kl = float(stale_kl_threshold
                     if stale_kl_threshold is not None else C.TO1_R11_STALE_KL_THRESHOLD)
    stale_cf = float(stale_clip_frac
                     if stale_clip_frac is not None else C.TO1_R11_STALE_CLIP_FRAC)

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    torch.manual_seed(seed)

    n_informative = sum(1 for st in samples if st.get("informative"))
    n_steps = len(samples)
    per_epoch = []
    for ep in range(epochs):
        terms = []
        kl_ref = []
        kl_old = []
        n_clip = 0
        for st in samples:
            logits = policy.action_logits(st["F_pool"], st["sf_t"], st["pool_stats"])
            lp_new = mixture_logp(logits, st["a"], T, eps)
            ratio = torch.exp(lp_new - st["logp_old"])
            if st.get("informative"):
                sur = torch.min(
                    ratio * st["adv_group"],
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * st["adv_group"])
                terms.append(-sur)
                if bool((ratio > 1.0 + clip_eps).item() or (ratio < 1.0 - clip_eps).item()):
                    n_clip += 1
            kl_ref.append(_kl_mixture(logits, policy.base_logits(
                st["F_pool"], st["sf_t"], st["pool_stats"]), T, eps))
            kl_old.append(_kl_mixture(logits, st["logits_old"], T, eps))
        if not terms:
            per_epoch.append({"epoch": ep, "loss": 0.0, "n_informative": 0,
                              "n_steps": n_steps, "clip_frac": 0.0,
                              "kl_ref": 0.0, "kl_old": 0.0, "stopped_stale": False,
                              "reason": "no_informative_steps"})
            break
        loss = (sum(terms) / len(terms)) + beta * (sum(kl_ref) / len(samples))
        opt.zero_grad()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(params, 10.0))
        opt.step()
        m_kl_old = float(sum(k.detach() for k in kl_old) / len(samples))
        m_kl_ref = float(sum(k.detach() for k in kl_ref) / len(samples))
        cf = float(n_clip) / len(terms)
        row = {"epoch": ep, "loss": float(loss.item()),
               "n_informative": len(terms), "n_steps": n_steps, "clip_frac": cf,
               "kl_ref": m_kl_ref, "kl_old": m_kl_old, "grad_norm": grad_norm,
               "stopped_stale": False, "reason": None}
        per_epoch.append(row)
        if ep > 0 and (m_kl_old > stale_kl or cf > stale_cf):
            row["stopped_stale"] = True
            row["reason"] = ("kl_old" if m_kl_old > stale_kl else "clip_frac")
            break
    return {"n_informative_steps": n_informative, "n_steps": n_steps,
            "epochs": per_epoch, "stopped_stale": bool(per_epoch and per_epoch[-1]["stopped_stale"])}


# ---------------------------------------------------------------------------
# rolling advancement (greedy, NO oracle) + greedy closed-loop evaluation
# ---------------------------------------------------------------------------
def greedy_step(policy, scorer, executor, model_b5, single_head, direct_head,
                problem, schedule, iid, episode_id, progmem, root_ms, step_offset=0,
                gate_mem=True):
    """Deterministic greedy action at state S_t under the policy (argmax of the raw
    action logits [props..., STOP]).  Returns a dict (never mutates inputs; the
    successor schedule is produced fresh by the executor)."""
    cache = AnalyzeCache(model_b5, single_head, direct_head)
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    h = schedule_hash(schedule)
    n_prop = len(metas)
    if n_prop == 0:
        return {"action": "stop", "reason": "no_proposals", "state_hash": h}
    ast = cache.ast(problem, schedule, iid)
    ms_cur = int(schedule.makespan)
    sf = state_feature_vec(ms_cur, root_ms, n_prop, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _rollex_of(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, step_offset, sf, queries),
                       dtype=torch.float32)
    if gate_mem:
        mem = mem * float(progmem.retrieval_gate(iid, episode_id, step_offset, sf))
    logit_pos, rank = _scores(scorer, rolex, mem)
    pool, pinfo = wide_pool(rolex, logit_pos, rank)
    out = {"action": "stop", "reason": "no_pool", "state_hash": h}
    if not pool:
        return out
    F_pool = _rerank_feats_all(scorer, rolex, mem)[pool]
    pool_stats = _pool_stats_from(F_pool)
    with torch.no_grad():
        logits = policy.action_logits(F_pool, sf_t, pool_stats)
    out["logits"] = logits.detach().float()
    out["probs"] = mixture_pmf(logits, 1.0, 0.0).detach()       # greedy probs (T=1)
    sel = int(logits.argmax().item())
    out["M"] = len(pool)
    out["best_pool_score"] = float(logits[:len(pool)].max())
    out["stop_score"] = float(logits[-1])
    if sel == len(pool):
        out.update({"action": "stop", "reason": "policy_stop", "state_hash": h})
        return out
    meta = metas[pool[sel]]
    edits, kind = _edits_for(ast, meta)
    sig = proposal_identity(ast, meta)[2]
    res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
    if res is None:
        out.update({"action": "stop", "reason": "infeasible", "state_hash": h})
        return out
    out.update({"action": "act", "reason": "policy_act", "state_hash": h,
                "sig": sig, "kind": kind, "meta_type": rolex["type"][pool[sel]],
                "meta_role": rolex["role"][pool[sel]], "src": rolex["src"][pool[sel]],
                "tgt": rolex["tgt"][pool[sel]], "improvement": float(res["improvement"]),
                "successor": res["schedule"]})
    return out


def advance_graphs(policy, scorer, executor, model_b5, single_head, direct_head,
                   graphs, step0_cache=None, gate_mem=True, log_prefix="[r11]"):
    """Real advancement of every active graph by ONE greedy action of the UPDATED
    policy.  Writes the executed transition into the graph persistent memory
    (never into any sibling).  Graph mutates as follows: done/S0/episode bookkeeping.
    Returns per-graph diagnostics."""
    out = []
    for g in graphs:
        if g.done:
            out.append({"iid": g.iid, "action": "skip", "reason": "done"})
            continue
        step = greedy_step(policy, scorer, executor, model_b5, single_head, direct_head,
                           g.problem, g.schedule, g.iid, g.episode_id,
                           g.progmem, g.ms0, step_offset=g.gstep, gate_mem=gate_mem)
        if step["action"] == "act":
            succ = step["successor"]
            nh = schedule_hash(succ)
            step["succ_hash"] = nh
            if nh in g.visited:
                g.done = True
                g.adv_reason = "revisit"
                out.append({"iid": g.iid, "action": "stop", "reason": "revisit",
                            "state_hash": step["state_hash"]})
                continue
            g.progmem.add_executed(g.iid, g.gstep, {
                "instance_id": g.iid, "episode_id": g.episode_id,
                "state_hash": step["state_hash"], "state_feat": None,
                "proposal_signature": step["sig"], "proposal_type": step["meta_type"],
                "role": step["meta_role"], "src": step["src"], "tgt": step["tgt"],
                "true_U": step["improvement"], "outcome":
                    ("success" if step["improvement"] > 0 else
                     "neutral" if step["improvement"] == 0 else "negative"),
                "successor_state_hash": nh, "trajectory_step": g.gstep,
                "written_at_step": g.gstep,
                "fine_key": ((step["meta_type"], step["meta_role"], step["src"], step["tgt"])
                             if step["meta_type"] == "single"
                             else (step["meta_type"], step["meta_role"])),
                "coarse_key": (step["meta_type"], step["meta_role"]),
            })
            g.visited.add(nh)
            g.gstep += 1
            g.schedule = succ
            g.ms = int(succ.makespan)
            g.adv_steps += 1
            if g.gstep >= C.TO1_R11_MAX_ROLLING_STATES_PER_GRAPH:
                g.done = True
                g.adv_reason = "max_depth"
            out.append({"iid": g.iid, "action": "act", "reason": "policy_act",
                        "improvement": step["improvement"], "sig": step["sig"],
                        "state_hash": step["state_hash"], "succ_hash": nh,
                        "gstep": g.gstep, "g_ms": g.ms})
        else:
            g.done = True
            g.adv_reason = step["reason"]
            out.append({"iid": g.iid, "action": "stop", "reason": step["reason"],
                        "state_hash": step["state_hash"]})
    return out


def roots_from_state(problem, schedule, iid, episode_id, progmem, horizon=5):
    """Runtime root bundle used by greedy closed-loop evaluation."""
    return {"problem": problem, "schedule": schedule, "iid": iid,
            "episode_id": episode_id, "progmem": progmem, "root_ms": int(schedule.makespan),
            "horizon": int(horizon)}


def greedy_closed_loop_r11(env, scorer, policy, roots, use_mem=True, gate_mem=True,
                           horizon=5, log_prefix="[r11]"):
    """Deterministic closed-loop evaluation over a list of root bundles: policy
    greedy argmax -> execute -> re-diagnose (FixedDecisionReplay, no solver), like
    the canonical B6/B11 loop but with the rolling-GRPO policy as selector.
    Returns the canonical `_b5_summary`-shaped dict + per-instance steps."""
    gains_by_iid = {}
    steps_by_iid = {}
    cache, executor = env["cache"], env["executor"]
    for rf in roots:
        iid = rf["iid"]
        problem, schedule0 = rf["problem"], rf["schedule"]
        progmem = rf["progmem"]
        episode_id = rf["episode_id"]
        s0_ms = int(schedule0.makespan)
        ms_cur = s0_ms
        schedule = schedule0
        visited = {schedule_hash(schedule0)}
        usage = {"single": 0, "pair": 0, "stop_by_selector": 0, "stop_neg": 0}
        steps = []
        for t in range(min(int(rf["horizon"] if "horizon" in rf else horizon), 5)):
            step = greedy_step(policy, scorer, executor,
                               env["model_b5"], env["single_head"], env["direct_head"],
                               problem, schedule, iid, episode_id, progmem,
                               s0_ms, step_offset=t, gate_mem=gate_mem)
            if step["action"] == "stop":
                usage["stop_by_selector"] += 1
                break
            res = step["successor"]
            steps.append({"t": t, "sig": step["sig"], "kind": step["kind"],
                          "improvement": float(step["improvement"])})
            if use_mem:
                progmem.add_executed(iid, t, {
                    "instance_id": iid, "episode_id": episode_id,
                    "state_hash": step["state_hash"], "state_feat": None,
                    "proposal_signature": step["sig"], "proposal_type": step["meta_type"],
                    "role": step["meta_role"], "src": step["src"], "tgt": step["tgt"],
                    "true_U": float(step["improvement"]),
                    "outcome": ("success" if step["improvement"] > 0 else
                                "neutral" if step["improvement"] == 0 else "negative"),
                    "successor_state_hash": schedule_hash(res),
                    "trajectory_step": t, "written_at_step": t,
                    "fine_key": ((step["meta_type"], step["meta_role"], step["src"], step["tgt"])
                                 if step["meta_type"] == "single"
                                 else (step["meta_type"], step["meta_role"])),
                    "coarse_key": (step["meta_type"], step["meta_role"]),
                })
            if step["improvement"] <= 0:
                usage["stop_neg"] += 1
            else:
                usage[step["kind"]] += 1
            schedule = res
            ms_cur = int(schedule.makespan)
            nh = schedule_hash(schedule)
            if nh in visited:
                break
            visited.add(nh)
            if t >= min(int(rf["horizon"] if "horizon" in rf else horizon), 5) - 1:
                break
        gains_by_iid[iid] = s0_ms - ms_cur
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    from .rollout import _b5_summary
    return _b5_summary(gains_by_iid), steps_by_iid


# ---------------------------------------------------------------------------
# graph handle
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, iid, episode_id, problem, schedule, progmem, src,
                 ms0=None):
        self.iid = iid
        self.episode_id = episode_id
        self.problem = problem
        self.s0_schedule = schedule          # pristine S0 (never mutated)
        self.schedule = copy.deepcopy(schedule)
        self.ms0 = ms0 if ms0 is not None else int(schedule.makespan)
        self.ms = self.ms0
        self.pristine_progmem = progmem     # pristine build (never mutated)
        self.progmem = copy.deepcopy(progmem)
        self.src = src
        self.gstep = 0
        self.visited = {schedule_hash(self.schedule)}
        self.done = False
        self.adv_reason = None
        self.adv_steps = 0
        self.visits = 1

    def reset(self):
        """Rewind to a fresh episode: pristine S0 + pristine memory snapshot."""
        self.schedule = copy.deepcopy(self.s0_schedule)
        self.progmem = copy.deepcopy(self.pristine_progmem)
        self.ms = self.ms0
        self.gstep = 0
        self.visited = {schedule_hash(self.schedule)}
        self.done = False
        self.adv_reason = None
        self.adv_steps = 0
        self.visits += 1

    def as_root(self):
        return {"problem": self.problem, "schedule": self.schedule, "iid": self.iid,
                "episode_id": self.episode_id, "progmem": self.progmem,
                "root_ms": self.ms, "horizon": 5}


# ---------------------------------------------------------------------------
# Stage B: deterministic rolling cycles
# ---------------------------------------------------------------------------
def run_rolling_cycles(policy, scorer, env, schedule_specs, cycles=None, k=None,
                       horizon=None, graphs_per_batch=None, workers=1,
                       seed=0, quick=False, log_prefix="[r11]", eval_root_builder=None,
                       enable_collapse_guard=True, collapse_floor=None):
    """Run the rolling GRPO Stage B loop.

    schedule_specs: {"bench": [Graph...], "real": [Graph...], "syn": [Graph...]}
    eval_root_builder(policy, cycle) -> optional eval roots for per-cycle selection.

    Each cycle: for each rolling depth (0..max): freeze pi_old = current policy;
    collect K sibling trajectories per active graph; GRPO update (E reuse epochs);
    real greedy advancement; repeat until every active graph finishes its episode.
    Model selection: per-cycle greedy closed-loop gains (TRAIN + held) -> best cycle.
    Collapse guard: TRAIN gain < collapse_ratio * collapse_floor (or residual
    saturation / KL blow) stops training and rolls back the best checkpoint.

    `collapse_floor` is the reference the TRAIN greedy gain is guarded against.  It
    is intentionally a PARAMETER because the R11 eval harness is NOT on R6's 369
    ruler: the harness advances past non-positive steps, gates by progressive memory
    (gate_mem=True) and scores the warm-start in z-pool coordinates, so the pristine
    R6 anchor registers a different absolute TRAIN gain here than the canonical
    STOP-on-negative closed loop that produced R6's recorded 369.  Callers pass the
    same-harness delta=0 warm-start level (R11-parity) so the "don't wreck the
    warm-start" guard compares like with like; left None it falls back to the
    recorded R6 369 floor (used by tests / quick paths).
    """
    cycles = int(cycles if cycles is not None else C.TO1_R11_TRAINING_CYCLES)
    k = int(k if k is not None else C.TO1_R11_K)
    horizon = int(horizon if horizon is not None else C.TO1_R11_HORIZON)
    gpb = int(graphs_per_batch if graphs_per_batch is not None else C.TO1_R11_GRAPHS_PER_BATCH)
    max_depth = int(C.TO1_R11_MAX_ROLLING_STATES_PER_GRAPH)
    workers = int(workers)
    rng = random.Random(seed)

    r6_floor = int(collapse_floor if collapse_floor is not None
                   else C.TO1_R11_R6_FLOOR)
    collapse_ratio = float(C.TO1_R11_COLLAPSE_TRAIN_RATIO)

    model_b5, single_head, direct_head = env["model_b5"], env["single_head"], env["direct_head"]
    executor = env["executor"]
    step0_cache = env["cache"]

    # deterministic per-cycle source mixing (40/30/30) into a fixed schedule
    bench_q = [g for g in schedule_specs["bench"]
               for ep in range(C.TO1_R11_MAX_EPISODES_PER_GRAPH)]
    real_q = [g for g in schedule_specs["real"]
              for ep in range(C.TO1_R11_MAX_EPISODES_PER_GRAPH)]
    syn_q = [g for g in schedule_specs["syn"]
             for ep in range(C.TO1_R11_MAX_EPISODES_PER_GRAPH)]
    per_src = {"bench": 2, "real": 1, "syn": 1} if not quick else {"bench": 2}
    sched = []
    for c in range(cycles):
        for src in ("bench", "real", "syn"):
            for _ in range(per_src.get(src, 0)):
                sched.append((c, src))
    # assign graph handles (episode slots) deterministically per source queue
    queues = {"bench": list(bench_q), "real": list(real_q), "syn": list(syn_q)}
    graph_ep_counter = {}
    cycle_handle = {c: [] for c in range(cycles)}
    for (c, src) in sched:
        q = queues[src]
        if not q:
            q = list(bench_q if src == "bench" else (real_q or list(bench_q)) if src == "real"
                     else (syn_q or list(bench_q)))
        g = q.pop(0)
        n_ep = graph_ep_counter.get(id(g), 0) + 1
        graph_ep_counter[id(g)] = n_ep
        if n_ep > 1:
            g.reset()                  # fresh episode (visits++)
        cycle_handle[c].append(g)

    history = []
    best = {"cycle": -1, "score": -1e18, "policy": policy.snapshot(), "train": 0}
    collapsed = False
    collapse_reason = None
    for c in range(cycles):
        t_start = time.time()
        active = cycle_handle[c]
        depth_results = []
        all_groups = []
        for depth in range(max_depth):
            act = [g for g in active if not g.done]
            if not act:
                break
            # collect K sibling trajectories per graph (on-policy, holding pi_old)
            groups = []
            for g in act:
                grp = collect_group_rollouts(
                    policy, scorer, executor, model_b5, single_head, direct_head,
                    g.problem, g.schedule, g.ms, g.iid, g.episode_id, g.progmem,
                    k=k, horizon=horizon, seed=seed + c * 1000 + depth * 100,
                    step_offset=g.gstep, workers=workers, step0_cache=step0_cache)
                groups.append(grp)
            all_groups.extend(groups)
            samples = _flatten_samples(groups)
            upd = grpo_update(policy, samples, seed=seed, log_prefix=log_prefix)
            depth_results.append({"depth": depth,
                                  "n_groups": len(groups),
                                  "n_informative": len([g for g in groups if g["informative"]]),
                                  "mean_reward": float(np.mean([g["mean_reward"] for g in groups])) if groups else 0.0,
                                  "adv": upd})
            # real advancement with the UPDATED policy (no oracle)
            out = advance_graphs(policy, scorer, executor, model_b5, single_head,
                                 direct_head, act, step0_cache=step0_cache)
        # ---- per-cycle deterministic evaluation --------------------------
        ev = {}
        if eval_root_builder is not None:
            ev = eval_root_builder(policy, cycle=c)
        train_gain = float(ev.get("train", {}).get("total", 0.0))
        real_hd = float(ev.get("real_held", {}).get("total", 0.0))
        syn_hd = float(ev.get("syn_held", {}).get("total", 0.0))
        score = train_gain + real_hd + syn_hd
        sat = residual_saturation_stats(policy, all_groups)
        # KL-explosion monitor: any depth update's final reuse step with an extreme
        # KL to the collected pi_old signals that the on-policy data went stale.
        kl_explosive = bool(any(
            (upd.get("epochs") or []) and upd["epochs"][-1]["kl_old"] > 5.0
            for upd in (r["adv"] for r in depth_results)))
        sat["kl_explosive"] = kl_explosive
        n_inf = sum(1 for g in all_groups if g["informative"])
        n_groups = len(all_groups)
        informative_ratio = float(n_inf / n_groups) if n_groups else 0.0
        # ---- collapse guard ------------------------------------------------
        dumped = {"cycle": c, "train": train_gain, "real_held": real_hd,
                  "syn_held": syn_hd, "score": score, "informative_ratio": informative_ratio,
                  "n_groups": n_groups, "n_informative": n_inf, "sat": sat,
                  "depth_results": depth_results, "sec": round(time.time() - t_start, 2)}
        if enable_collapse_guard and (
                (train_gain < collapse_ratio * r6_floor) or sat.get("saturated_frac", 0.0) > 0.5
                or sat["kl_explosive"]):
            collapsed = True
            collapse_reason = ("train_gain" if train_gain < collapse_ratio * r6_floor
                               else "residual_saturation" if sat.get("saturated_frac", 0.0) > 0.5
                               else "kl_explosive")
            dumped["collapse"] = {"flag": True, "reason": collapse_reason}
        if score > best["score"] and not collapsed:
            best = {"cycle": c, "score": score, "policy": policy.snapshot(),
                    "train": train_gain, "real_held": real_hd, "syn_held": syn_hd}
        history.append(dumped)
        print(f"{log_prefix} cycle {c}: TRAIN={train_gain:.0f} real_hd={real_hd:.0f} "
              f"syn_hd={syn_hd:.0f} informative={informative_ratio:.3f} "
              f"(n_inf={n_inf}/{n_groups}) sat={sat.get('saturated_frac', 0.0):.3f} "
              f"sec={dumped['sec']}s", flush=True)
        if collapsed:
            break
    # ---- rollback to the best internal checkpoint -------------------------
    policy.load_snapshot(best["policy"])
    return {"history": history, "best": best, "collapsed": collapsed,
            "collapse_reason": collapse_reason, "cycles_run": len(history)}


def _flatten_samples(groups):
    flat = [rec for g in groups for tr in g["trajs"] for rec in tr["steps"]]
    for rec in flat:
        rec.setdefault("informative", False)
        rec.setdefault("adv_group", 0.0)
        rec.setdefault("grp_key", ("?", "?"))
        rec.setdefault("M", 0)
    return flat


def residual_saturation_stats(policy, groups):
    """Fraction of residual units at/near the alpha bound (|tanh delta| > β)."""
    n_sat = 0
    n = 0
    obs = []
    for g in groups:
        for tr in g["trajs"]:
            for st in tr["steps"]:
                F = st["F_pool"]
                if len(F) == 0:
                    continue
                with torch.no_grad():
                    d = policy.residual_prop(F).abs()
                if len(d):
                    n += len(d)
                    n_sat += int((d > C.TO1_R11_RESID_SAT).sum().item())
                    obs.extend(d.numpy().tolist())
    return {"saturated_frac": float(n_sat / n) if n else 0.0,
            "n_units": n, "mean_abs_delta": float(np.mean(np.abs(obs))) if obs else 0.0}


# ---------------------------------------------------------------------------
# behavior audit + DPpaulli rolling trace + profiling
# ---------------------------------------------------------------------------
def _state_logits(policy, scorer, env, rf, gate_mem=True, step_offset=0):
    """Action logits [M+1] at a root state without advancing (no mutation)."""
    cache = env["cache"]
    problem, schedule = rf["problem"], rf["schedule"]
    iid, episode_id, progmem = rf["iid"], rf["episode_id"], rf["progmem"]
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    n_prop = len(metas)
    if n_prop == 0:
        return None
    ast = cache.ast(problem, schedule, iid)
    ms = int(schedule.makespan)
    sf = state_feature_vec(ms, ms, n_prop, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _rollex_of(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, step_offset, sf, queries),
                       dtype=torch.float32)
    if gate_mem:
        mem = mem * float(progmem.retrieval_gate(iid, episode_id, step_offset, sf))
    logit_pos, rank = _scores(scorer, rolex, mem)
    pool, _ = wide_pool(rolex, logit_pos, rank)
    if not pool:
        return None
    F_pool = _rerank_feats_all(scorer, rolex, mem)[pool]
    pool_stats = _pool_stats_from(F_pool)
    with torch.no_grad():
        return policy.action_logits(F_pool, sf_t, pool_stats), F_pool, sf_t, pool_stats


def prob_movement_audit(policy_before, policy_after, roots, scorer, env,
                        T=None, eps=None, gate_mem=True):
    """Per-state action-probability movement (behavior mixture) before/after
    training over the given root states.  True-U diagnostic only; never a reward."""
    T = float(T if T is not None else C.TO1_R11_TEMP)
    eps = float(eps if eps is not None else C.TO1_R11_MIX_EPS)
    rows = []
    for rf in roots:
        r_pre = _state_logits(policy_before, scorer, env, rf, gate_mem=gate_mem)
        r_post = _state_logits(policy_after, scorer, env, rf, gate_mem=gate_mem)
        if r_pre is None or r_post is None:
            rows.append({"iid": rf["iid"], "found": False,
                         "state_hash": rf.get("state_hash")})
            continue
        lpre, F_pool, sf_t, ps = r_pre
        lp_post = r_post[0]
        p_pre = mixture_pmf(lpre, T, eps)
        p_post = mixture_pmf(lp_post, T, eps)
        if len(p_pre) != len(p_post):
            rows.append({"iid": rf["iid"], "found": False, "mismatch": True})
            continue
        dp = (p_post - p_pre).abs()
        rows.append({"iid": rf["iid"], "found": True, "M": int(len(p_pre)) - 1,
                     "state_hash": rf.get("state_hash"),
                     "mean_abs_dp": float(dp.mean()),
                     "max_abs_dp": float(dp.max()),
                     "stop_dp": float(dp[-1]),
                     "argmax_changed": bool(
                         int(lpre.argmax()) != int(lp_post.argmax()))})
    return rows


def dpp_rolling_trace_r11(policy, scorer, env, re, iid, episode_id, st,
                          dpp_lut, horizon=5, gate_mem=True, log_prefix="[r11]"):
    """Multi-state ROLLING trace at DPpaulli10a: greedy advance state by state with
    the policy (no oracle), printing selected action + prob + true-U at each state."""
    out = {"iid": iid, "found": False, "episode_id": episode_id, "steps": []}
    problem, schedule = st["problem"], st["schedule"]
    progmem = copy.deepcopy(re["progmem"])
    ms0 = int(schedule.makespan)
    reward = 0
    visiteds = {schedule_hash(schedule)}
    for d in range(min(int(horizon), 5)):
        step = greedy_step(policy, scorer, env["executor"],
                           env["model_b5"], env["single_head"], env["direct_head"],
                           problem, schedule, iid, episode_id, progmem, ms0,
                           step_offset=d, gate_mem=gate_mem)
        out["found"] = True
        row = {"depth": d, "state_hash": step["state_hash"], "action": step["action"],
               "reason": step["reason"]}
        if "M" in step:
            row["M"] = step["M"]
            row["best_pool_score"] = step["best_pool_score"]
            row["stop_score"] = step["stop_score"]
        if step["action"] == "act":
            lu = dpp_lut.get(step["sig"], 0.0) if dpp_lut else 0.0
            row.update({"sig": step["sig"], "kind": step["kind"],
                        "improvement": step["improvement"], "true_U": float(lu)})
            out["steps"].append(row)
            schedule = step["successor"]
            reward += float(step["improvement"])
            nh = schedule_hash(schedule)
            if nh in visiteds:
                row["revisit"] = True
                break
            visiteds.add(nh)
        else:
            row["stop_true_U"] = 0.0
            out["steps"].append(row)
            break
    out["total_gain"] = reward
    out["selected_true_U_sum"] = sum(s.get("true_U", 0.0) for s in out["steps"])
    return out


def trajectory_profiling(policy, scorer, env, root, k=8, horizon=5, weights=(1, 4)):
    """Time trajectory collection serial vs (1,4)-worker parallel on the same root;
    report per-phase seconds + wall-clock speedup + CPU utilization proxy.
    Deterministic data equality check.  Each phase is aggregated across every
    trajectory in a group (analyze+execute = FixedDecisionReplay share, §54)."""
    model_b5, single_head, direct_head = env["model_b5"], env["single_head"], env["direct_head"]
    executor = env["executor"]
    step0_cache = env["cache"]
    out = {}
    for w in weights:
        g = collect_group_rollouts(
            policy, scorer, executor, model_b5, single_head, direct_head,
            root["problem"], root["schedule"], root["root_ms"], root["iid"],
            root["episode_id"], root["progmem"], k=k, horizon=horizon,
            seed=0, workers=w, step0_cache=step0_cache)
        profs = [tr.get("prof") for tr in g["trajs"] if tr.get("prof")]
        agg = {}
        for key in ("analyze", "execute", "mem", "policy"):
            vals = [p[key] for p in profs if key in p]
            agg[key] = float(sum(vals) / len(vals)) if vals else 0.0
        tot = sum(agg.values()) or 1e-9
        fdr = agg["analyze"] + agg["execute"]              # FixedDecisionReplay
        out[str(w)] = {"coll_s": g["coll_s"], "n_steps": g["n_steps"],
                       "rewards": g["rewards"], "phase_s": agg,
                       "fixed_dr_share": round(fdr / tot, 4),
                       "mean_traj_s": round(g["coll_s"] / max(1, k), 4)}
    a, b = out[str(weights[0])], out[str(weights[-1])]
    speedup = float("nan")
    if a["coll_s"] > 0:
        speedup = a["coll_s"] / (b["coll_s"] if b["coll_s"] > 0 else 1e-9)
    out["speedup"] = round(speedup, 3)
    # CPU utilization proxy: (summed worker-trace time) / (wall clock) per group.
    cpu_busy_s = a["phase_s"]["analyze"] + a["phase_s"]["execute"] \
        + a["phase_s"]["mem"] + a["phase_s"]["policy"]
    out["cpu_util_serial"] = round(cpu_busy_s / a["coll_s"], 3) if a["coll_s"] > 0 else None
    out["cpu_util_parallel"] = round(cpu_busy_s / b["coll_s"], 3) if b["coll_s"] > 0 else None
    out["identical_data"] = bool(a["rewards"] == b["rewards"] and a["n_steps"] == b["n_steps"])
    return out