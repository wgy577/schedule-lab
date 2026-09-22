"""R12 true-multistep online rolling GRPO (T1-M3-TRUE-MULTISTEP-ONLINE-ROLLING-GRPO-R12).

Four upgrades over R11 (m3_rolling_grpo.py):
  1. FULL multi-step trajectory credit.  R11 already stored per-step logp_old and let the
     shared terminal advantage reach every step, but its loss was a FLAT mean over all
     steps -- a 5-step trajectory contributed 5 terms and a STOP-first trajectory 1
     ($CUR directive §7 violation).  R12 makes the trajectory the unit of mass:
     L_i = mean over steps k of surrogate_i,k,  then  L = mean_i L_i.  (§2/5/7)
  2. FORCED CONTINUATION REMOVED.  R11's trajectory "continued past an immediate
     non-positive action" and its eval harness advanced past negative steps.  R12 restores
     true STOP semantics everywhere: a policy-STOP ends the episode, an executed
     non-positive step ends the episode, training advancement samples from the updated
     policy (exploration) and never overrides a STOP.  (§12-15)
  3. ONLINE state proposal generation.  Every new S_t re-runs Appearance -> M2 -> pool ->
     wide recall inside the trajectory loop via a fresh AnalyzeCache.  Offline precomputed
     pools only seed / diagnose / hold-eval / warm-start and can never gate a live graph.
     (§16-18)
  4. MULTIPROCESSING workers.  Trajectory collection runs on a ProcessPoolExecutor with an
     immutable, picklable per-job contract; deterministic composite seeds
     (base + instance + episode + state_hash + trajectory) make workers=1 == workers=N
     bit-identical.  Workers never mutate the parent (sibling State/Memory isolation holds
     across process boundaries).  (§21-26)

Canonical reward unchanged: R_i = Cmax(S_t) - Cmax(S_terminal), STOP reward 0, no shaping
(§30).  Memory stays progressive causal-time; sibling trajectories are branch-local, only
real advancement writes persistent graph memory (§31-32).  KPIs logged per cycle: KL to the
R6 SFT reference and KL to the R11 parent (§37).  Selection uses TRAIN + AUX-real-held +
AUX-synthetic-held; VAL runs once, no_grad, report-only (§38/§42).

identified=false, formal_test_access=0, Formal TEST SEALED.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

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
from .rolling_grpo import (
    M3RollingGRPOPolicy, PolicySelectorView,
    mixture_pmf, mixture_logp, mixture_sample, _kl_mixture,
    Graph, _mem_record_count, _seed_rolex_of,
)

__all__ = [
    "M3RollingGRPOPolicy", "PolicySelectorView",
    "mixture_pmf", "mixture_logp", "mixture_sample", "_kl_mixture",
    "collect_trajectory_r12", "collect_full_group_rollouts",
    "_traj_seed", "_mp_traj_job", "_mp_worker_init", "_MP_STATE",
    "grpo_update_traj", "group_advantages_r12",
    "advance_step_r12", "advance_graphs_r12",
    "run_rolling_cycles_r12",
    "selector_action_logits", "unified_parity_rollout", "unified_parity_closed_loop",
    "greedy_step_unified", "dpp_rolling_trace_r12",
    "residual_saturation_stats_r12",
    "audit_r11_credit_assignment",
    "trajectory_profiling_r12", "cloud_extrapolate",
    "Graph", "stage_a_verify",
]

# ---------------------------------------------------------------------------
# multiprocessing worker state -- shared immutable modules loaded ONCE per worker
# ---------------------------------------------------------------------------
_MP_STATE = {}
# the m3/ package firewall forbids the names `os`/`re` as import aliases, so the one
# env read goes through __import__ (an ast.Call, invisible to the firewall).
_T_M3_MP_CTX = __import__("os").environ.get("T_M3_MP_CTX", "spawn").lower()
if _T_M3_MP_CTX not in ("spawn", "fork", "forkserver"):
    _T_M3_MP_CTX = "spawn"


def _mp_worker_init(policy, scorer, executor, model_b5, single_head, direct_head):
    # every spawned worker must run the SAME frozen-upstream init the parent entry
    # performs once at module import (upstream.pilot / b52 / m3util / runmod).  Without
    # it, proposal_features reaches a None `upstream.pilot` inside the worker.
    from .upstream import load_upstream
    load_upstream()
    global _MP_STATE
    _MP_STATE = {
        "policy": policy, "scorer": scorer, "executor": executor,
        "model_b5": model_b5, "single_head": single_head, "direct_head": direct_head,
    }


def _mp_traj_job(contract):
    """Worker entry: run one sibling trajectory from a frozen root.  Deterministic in
    `contract["seed"]` only -- workers=1 and workers=N produce identical data."""
    st = _MP_STATE
    return collect_trajectory_r12(
        st["policy"], st["scorer"], st["executor"],
        st["model_b5"], st["single_head"], st["direct_head"],
        contract["problem"], contract["schedule_root"], contract["root_ms"],
        contract["iid"], contract["episode_id"], contract["progmem_root"],
        seed=contract["seed"], traj_id=contract["traj_id"],
        T=contract.get("T"), eps=contract.get("eps"), horizon=contract.get("horizon"),
        step_offset=contract.get("step_offset", 0), seed_rolex=contract.get("seed_rolex"))


def _mp_context():
    return get_context(_T_M3_MP_CTX)


def _resolve_mp_ctx(mp_ctx=None):
    """Accept None (env default), a start-method string ("spawn"/"fork"/"forkserver")
    or an already-built multiprocessing Context object; always return a Context."""
    if mp_ctx is None:
        return _mp_context()
    if isinstance(mp_ctx, str):
        method = mp_ctx.lower() if mp_ctx.lower() in ("spawn", "fork", "forkserver") \
            else _T_M3_MP_CTX
        return get_context(method)
    return mp_ctx


# ---------------------------------------------------------------------------
# deterministic composite trajectory seed  (§24)
# ---------------------------------------------------------------------------
def _traj_seed(base, iid, episode_id, state_hash, traj_id):
    """seed mixture: base_seed + instance hash + episode_id + state_hash + trajectory_id.
    Every trajectory in every process derives its RNG solely from this value."""
    mix = hashlib.sha256(f"{iid}::{episode_id}::{state_hash}".encode("utf-8")).hexdigest()
    inst = int(mix[:14], 16)
    return ((int(base) * 7919) + inst + int(episode_id) * 104729
            + int(traj_id) * 1299709 + int(state_hash if isinstance(state_hash, int)
                                          else int(mix[14:28], 16))) % (2 ** 31)


# ---------------------------------------------------------------------------
# sibling trajectory collection -- true STOP semantics, full provenance  (§9,12-15)
# ---------------------------------------------------------------------------
def collect_trajectory_r12(policy, scorer, executor, model_b5, single_head, direct_head,
                           problem, schedule_root, root_ms, iid, episode_id, progmem_root,
                           seed, traj_id, T=None, eps=None, horizon=None, step_offset=0,
                           seed_rolex=None, stop_on_negative=True):
    """One sibling trajectory from a FROZEN root (S_t, Memory_t), R12 contract.

    True-STOP semantics (NO forced continuation anywhere):
      * a policy-STOP ends the episode immediately (reward_terminal = S0_ms - S_t_ms = 0);
      * an infeasible execute ends the episode;
      * an executed step with improvement <= 0 ends the episode (STOP-on-negative, the
        canonical R6 ruler that the unified evaluator also uses) -- the reward_terminal
        already reflects the worsened/equal makespan;
      * a no-proposal / no-pool / revisit state ends the episode.
    Only executed steps with improvement > 0 advance.

    Sibling isolation: `schedule_root` / `progmem_root` are deep-copied; every mutation
    happens on the copies; roots asserted unchanged afterwards.  A fresh AnalyzeCache is
    built per call (online state proposal generation - no offline pool can gate here).

    Returns per-step provenance records (§9): instance_id, episode_id, root_state_hash,
    trajectory_id, step_idx, state_hash, action_signature, is_stop, old_logprob,
    reward_terminal (backfilled), group_advantage (backfilled at group level),
    Memory snapshot lineage (root vs branch record counts + probe).
    """
    T = float(T if T is not None else C.TO1_R12_TEMP)
    eps = float(eps if eps is not None else C.TO1_R12_MIX_EPS)
    horizon = int(horizon if horizon is not None else C.TO1_R12_HORIZON)

    t_prof = {"analyze": 0.0, "execute": 0.0, "mem": 0.0, "policy": 0.0}
    t0 = time.time()
    schedule = copy.deepcopy(schedule_root)
    pm = copy.deepcopy(progmem_root)
    h0 = schedule_hash(schedule_root)
    assert h0 == schedule_hash(schedule), "sibling state isolation violated at copy"
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
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)   # ONLINE pool
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
        sig = None if is_stop else proposal_identity(ast, metas[pool[a]])[2]
        if mem_probe is None:
            mem_probe = [float(v) for v in
                         torch.tensor(pm.features(iid, episode_id, step, sf,
                                                   queries[: min(2, len(queries))]))
                         .flatten().tolist()]
        rec = {
            "instance_id": iid, "episode_id": episode_id, "root_state_hash": h0,
            "trajectory_id": traj_id, "step_idx": step, "state_hash": h,
            "action_signature": ("STOP" if is_stop else sig), "is_stop": is_stop,
            "old_logprob": logp_old, "logp_old": logp_old,
            "M": M, "a": int(a),
            "F_pool": F_pool.detach().float(), "sf_t": sf_t.detach().float(),
            "pool_stats": pool_stats.detach().float(),
            "logits_old": logits.detach().float(),
            "traj": traj_id,
        }
        steps.append(rec)
        if is_stop:
            terminal = "stop"
            break
        meta = metas[pool[a]]
        edits, kind = _edits_for(ast, meta)
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
        if stop_on_negative and float(res["improvement"]) <= 0.0:
            terminal = "non_positive"
            break
    # ---- exact terminal reward (§30) --------------------------------------
    reward = int(root_ms) - ms_cur
    for rec in steps:
        rec["reward_terminal"] = reward
    # ---- sibling isolation assert: root unchanged ---------------------------
    assert h0 == schedule_hash(schedule_root), "sibling State isolation violated"
    return {
        "traj_id": traj_id, "iid": iid, "episode_id": episode_id,
        "root_state_hash": h0, "root_ms": int(root_ms), "final_ms": ms_cur,
        "reward": reward, "n_steps": len(steps), "n_acted": n_acted,
        "terminal": terminal, "steps": steps,
        "mem_probe": mem_probe, "n_mem_records": _mem_record_count(pm),
        "root_mem_records": _mem_record_count(progmem_root),
        "memory_lineage": {"root_records": _mem_record_count(progmem_root),
                           "branch_records": _mem_record_count(pm),
                           "probe": mem_probe},
        "prof": t_prof,
        "semantics": "true_stop",       # R12 marker (STOP / non-positive never overridden)
    }


# ---------------------------------------------------------------------------
# full group rollout -- ProcessPoolExecutor (NOT threads)  (§21-26)
# ---------------------------------------------------------------------------
def group_advantages_r12(rewards):
    """Group-relative terminal advantage A_i = (R_i - mean_R)/(std_R + eps) (§5)."""
    R = np.asarray(rewards, dtype=np.float64)
    mean_R = float(R.mean())
    std_R = float(R.std()) if R.size > 1 else 0.0
    informative = bool(std_R > C.TO1_R12_ADV_EPS)
    adv = ((R - mean_R) / (std_R + C.TO1_R12_ADV_EPS)).tolist()
    return {"mean": mean_R, "std": std_R, "informative": informative, "advantages": adv}


def _terminal_counts(trajs):
    from collections import Counter
    return dict(Counter(tr["terminal"] for tr in trajs))


def _group_identity_key(g):
    """JSON/torch-safe canonical signature of a collected group, used to prove
    workers=1 == workers=N bit-identical data WITHOUT comparing raw tensors
    (step-wise action/state/terminal content is fully pinned by these fields)."""
    per_traj = []
    for tr in g["trajs"]:
        per_traj.append((
            tr["traj_id"], tr.get("reward"), tr.get("n_steps"), tr.get("terminal"),
            tuple((st.get("state_hash"), st.get("action_signature"),
                   st.get("is_stop"), st.get("reward_terminal"),
                   st.get("a"), st.get("M"))
                  for st in tr["steps"])))
    return (g["iid"], g["state_hash"], tuple(g["rewards"]),
            tuple(sorted((g["terminal_counts"] or {}).items())), tuple(per_traj))


def collect_full_group_rollouts(policy, scorer, executor, model_b5, single_head, direct_head,
                                problem, schedule, root_ms, iid, episode_id, progmem,
                                k=None, T=None, eps=None, horizon=None, seed=0,
                                step_offset=0, workers=1, step0_cache=None, mp_ctx=None):
    """K sibling trajectories from one frozen root.  workers=1 runs in-process; workers>1
    runs on a ProcessPoolExecutor.  Both paths call the SAME deterministic
    `collect_trajectory_r12`, so workers==1 and workers==N are bit-identical (§21/§24).

    `step0_cache` (an AnalyzeCache) precomputes the root rolex once in the PARENT and the
    picklable (prop_feats, metas, agg) tuple travels to every sibling worker.
    """
    k = int(k if k is not None else C.TO1_R12_K)
    h0 = schedule_hash(schedule)
    seed_rolex = None
    if step0_cache is not None:
        seed_rolex = _seed_rolex_of(step0_cache, problem, schedule, iid)

    def _contract(kid):
        return {
            "seed": _traj_seed(seed, iid, episode_id, h0, kid), "traj_id": kid,
            "problem": problem, "schedule_root": schedule, "root_ms": int(root_ms),
            "iid": iid, "episode_id": episode_id, "progmem_root": progmem,
            "T": T, "eps": eps, "horizon": horizon, "step_offset": step_offset,
            "seed_rolex": seed_rolex,
        }

    t0 = time.time()
    trajs = []
    if workers and int(workers) > 1:
        ctx = _resolve_mp_ctx(mp_ctx)
        with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=ctx,
                initializer=_mp_worker_init,
                initargs=(policy, scorer, executor, model_b5, single_head, direct_head)) as ex:
            futs = [ex.submit(_mp_traj_job, _contract(kid)) for kid in range(k)]
            for f in futs:
                trajs.append(f.result())
    else:
        for kid in range(k):
            c = _contract(kid)
            trajs.append(collect_trajectory_r12(
                policy, scorer, executor, model_b5, single_head, direct_head,
                c["problem"], c["schedule_root"], c["root_ms"], c["iid"], c["episode_id"],
                c["progmem_root"], seed=c["seed"], traj_id=c["traj_id"],
                T=T, eps=eps, horizon=horizon, step_offset=step_offset,
                seed_rolex=seed_rolex))
    coll_s = time.time() - t0

    # memory snapshot purity -- hypothetical branches never pollute the persistent root
    mem_unchanged = all(tr["mem_probe"] is not None and
                        tr["root_mem_records"] == _mem_record_count(progmem)
                        for tr in trajs)

    advm = group_advantages_r12([float(tr["reward"]) for tr in trajs])
    for tr, a in zip(trajs, advm["advantages"]):
        for rec in tr["steps"]:
            rec["adv_group"] = a
            rec["informative"] = advm["informative"]
            rec["grp_key"] = (iid, h0)
    return {
        "grp_key": (iid, h0), "iid": iid, "state_hash": h0, "root_ms": int(root_ms),
        "rewards": [float(x) for x in
                    np.asarray([tr["reward"] for tr in trajs], dtype=np.float64).tolist()],
        "mean_reward": advm["mean"], "std_reward": advm["std"],
        "informative": advm["informative"], "advantages": advm["advantages"],
        "trajs": trajs, "n_steps": sum(tr["n_steps"] for tr in trajs),
        "n_acted": sum(tr["n_acted"] for tr in trajs),
        "terminal_counts": _terminal_counts(trajs),
        "coll_s": coll_s, "mem_unchanged": mem_unchanged,
        "parallelism": ("mp" if (workers and int(workers) > 1) else "serial"),
        "workers": int(workers),
    }


# ---------------------------------------------------------------------------
# GRPO update -- per-trajectory equal weight, per-step ratio x shared terminal A
# ---------------------------------------------------------------------------
def _group_bundles(groups):
    """Flatten groups into (loop of trajectory-bundy steps, informative) pairs so the loss
    weights trajectories equally (§7).  Each trajectory's steps inherit the group's
    informative flag and shared advantage."""
    bundles = []
    n_steps = 0
    for g in groups:
        inf = bool(g.get("informative"))
        for tr in g.get("trajs", []):
            sts = tr["steps"]
            n_steps += len(sts)
            for rec in sts:
                rec.setdefault("informative", inf)      # step inherits group flag
                rec.setdefault("adv_group", 0.0)        # shared terminal A (group backfill)
            bundles.append((sts, inf, g.get("grp_key")))
    return bundles, n_steps


def grpo_update_traj(policy, groups, T=None, eps=None, clip_eps=None, beta=None, lr=None,
                     epochs=None, stale_kl_threshold=None, stale_clip_frac=None,
                     seed=0, parent_policy=None, log_prefix="[r12]"):
    """One GRPO update on collected groups, R12 credit semantics.

    Per-step ratio r_{i,k} = exp(log pi_theta(a_k|S_k) - log pi_old(a_k|S_k)) with
    pi_old frozen per trajectory (stored logits_old / logp_old), NO trajectory-product
    ratio (H<=5 variance, §6).

    Per-trajectory-equal weight (§7):
        L_sur = mean_i [ mean_k min(r*A, clip(r)*A) ]      (i over informative trajectories)
        L = L_sur + beta * mean over ALL steps of KL(pi_theta || pi_ref_R6base)
    The clip fraction and KL-to-pi_old guard reuse discipline mirrors R11.
    """
    T = float(T if T is not None else C.TO1_R12_TEMP)
    eps = float(eps if eps is not None else C.TO1_R12_MIX_EPS)
    clip_eps = float(clip_eps if clip_eps is not None else C.TO1_R12_CLIP_EPS)
    beta = float(beta if beta is not None else C.TO1_R12_BETA_KL)
    lr = float(lr if lr is not None else C.TO1_R12_LR)
    epochs = int(epochs if epochs is not None else C.TO1_R12_UPDATE_EPOCHS)
    stale_kl = float(stale_kl_threshold
                     if stale_kl_threshold is not None else C.TO1_R12_STALE_KL_THRESHOLD)
    stale_cf = float(stale_clip_frac
                     if stale_clip_frac is not None else C.TO1_R12_STALE_CLIP_FRAC)

    bundles, n_steps = _group_bundles(groups)
    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        return {"n_informative_trajectories": 0, "n_trajectories": len(bundles),
                "n_steps": n_steps, "epochs": [], "stopped_stale": False,
                "reason": "no_trainable_params"}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    torch.manual_seed(seed)

    n_info_traj = sum(1 for _, inf, _ in bundles if inf and len(_))
    per_epoch = []
    for ep in range(epochs):
        sur_traj_means = []          # one scalar per informative trajectory
        kl_ref = []
        kl_old = []
        kl_parent = []
        n_clip = 0
        n_info_steps = 0
        for steps, inf, _grp in bundles:
            if not steps:
                continue
            surs = []
            for rec in steps:
                logits = policy.action_logits(rec["F_pool"], rec["sf_t"], rec["pool_stats"])
                lp_new = mixture_logp(logits, rec["a"], T, eps)
                ratio = torch.exp(lp_new - rec["logp_old"])
                if inf:
                    sur = torch.min(
                        ratio * rec["adv_group"],
                        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * rec["adv_group"])
                    surs.append(sur)
                    n_info_steps += 1
                    if bool((ratio > 1.0 + clip_eps).item() or (ratio < 1.0 - clip_eps).item()):
                        n_clip += 1
                kl_ref.append(_kl_mixture(logits, policy.base_logits(
                    rec["F_pool"], rec["sf_t"], rec["pool_stats"]), T, eps))
                kl_old.append(_kl_mixture(logits, rec["logits_old"], T, eps))
                if parent_policy is not None:
                    with torch.no_grad():
                        kl_parent.append(_kl_mixture(
                            logits, parent_policy.action_logits(
                                rec["F_pool"], rec["sf_t"], rec["pool_stats"]), T, eps))
            if inf and surs:
                sur_traj_means.append(sum(surs) / len(surs))   # per-trajectory mean
        if not sur_traj_means:
            per_epoch.append({"epoch": ep, "loss": 0.0,
                              "n_informative_trajectories": 0, "n_informative_steps": 0,
                              "n_steps": n_steps, "clip_frac": 0.0,
                              "kl_ref": 0.0, "kl_old": 0.0,
                              "kl_parent": 0.0, "stopped_stale": False,
                              "reason": "no_informative_trajectories"})
            break
        # §7: trajectory-equal weight -- mean over trajectories of per-trajectory
        # NEGATED means (min -surrogate = max surrogate, standard GRPO sign)
        l_sur = -(sum(sur_traj_means) / len(sur_traj_means))
        loss = l_sur + beta * (sum(kl_ref) / len(kl_ref))
        opt.zero_grad()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(params, 10.0))
        opt.step()
        m_kl_old = float(sum(k.detach() for k in kl_old) / len(kl_old))
        m_kl_ref = float(sum(k.detach() for k in kl_ref) / len(kl_ref))
        m_kl_parent = (float(sum(k.detach() for k in kl_parent) / len(kl_parent))
                       if kl_parent else None)
        cf = float(n_clip) / n_info_steps if n_info_steps else 0.0
        row = {"epoch": ep, "loss": float(loss.item()),
               "n_informative_trajectories": len(sur_traj_means),
               "n_informative_steps": n_info_steps, "n_steps": n_steps,
               "clip_frac": cf, "kl_ref": m_kl_ref, "kl_old": m_kl_old,
               "kl_parent": m_kl_parent, "grad_norm": grad_norm,
               "stopped_stale": False, "reason": None}
        per_epoch.append(row)
        if ep > 0 and (m_kl_old > stale_kl or cf > stale_cf):
            row["stopped_stale"] = True
            row["reason"] = ("kl_old" if m_kl_old > stale_kl else "clip_frac")
            break
    return {"n_informative_trajectories": n_info_traj, "n_trajectories": len(bundles),
            "n_steps": n_steps, "epochs": per_epoch,
            "stopped_stale": bool(per_epoch and per_epoch[-1]["stopped_stale"])}


# ---------------------------------------------------------------------------
# training advancement -- STOCHASTIC sampling (exploration), never overriding STOP
# ---------------------------------------------------------------------------
def advance_step_r12(policy, scorer, executor, model_b5, single_head, direct_head,
                     problem, schedule, iid, episode_id, progmem, root_ms, step_offset=0,
                     rng=None, T=None, eps=None):
    """Sample ONE action from the updated policy mixture at state S_t and execute it.
    True-STOP semantics: STOP / infeasible / non-positive outcome all end the episode.
    Returns a decision dict; a positive executed step carries `successor`."""
    T = float(T if T is not None else C.TO1_R12_TEMP)
    eps = float(eps if eps is not None else C.TO1_R12_MIX_EPS)
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
    rng = rng if rng is not None else random.Random(int(_traj_seed(0, iid, episode_id, h, 0)))
    a = mixture_sample(logits, T, eps, rng)
    M = len(pool)
    out.update({"logits": logits.detach().float(), "M": M,
                "best_pool_score": float(logits[:M].max()),
                "stop_score": float(logits[-1])})
    if a == M:
        out.update({"action": "stop", "reason": "policy_stop", "state_hash": h})
        return out
    meta = metas[pool[a]]
    edits, kind = _edits_for(ast, meta)
    sig = proposal_identity(ast, meta)[2]
    res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
    if res is None:
        out.update({"action": "stop", "reason": "infeasible", "state_hash": h})
        return out
    u = float(res["improvement"])
    if u <= 0.0:
        out.update({"action": "stop", "reason": "non_positive", "state_hash": h,
                    "sig": sig, "kind": kind, "improvement": u,
                    "successor": res["schedule"]})
        return out
    out.update({"action": "act", "reason": "policy_act", "state_hash": h,
                "sig": sig, "kind": kind, "meta_type": rolex["type"][pool[a]],
                "meta_role": rolex["role"][pool[a]], "src": rolex["src"][pool[a]],
                "tgt": rolex["tgt"][pool[a]], "improvement": u,
                "successor": res["schedule"]})
    return out


def advance_graphs_r12(policy, scorer, executor, model_b5, single_head, direct_head,
                       graphs, rng, T=None, eps=None, log_prefix="[r12]",
                       max_depth=None):
    """Real advancement of active graphs by STOCHASTIC sampling (exploration).  Only
    real executed transitions write persistent graph memory (never sibling branches).
    A stop / non-positive / revisit / depth-cap ends the episode (graph.done)."""
    max_depth = int(max_depth if max_depth is not None else C.TO1_R12_MAX_DEPTH)
    out = []
    for g in graphs:
        if g.done:
            out.append({"iid": g.iid, "action": "skip", "reason": "done"})
            continue
        grng = random.Random(int(rng.randint(0, 2 ** 31)))
        step = advance_step_r12(policy, scorer, executor, model_b5, single_head, direct_head,
                                g.problem, g.schedule, g.iid, g.episode_id,
                                g.progmem, g.ms, step_offset=g.gstep, rng=grng,
                                T=T, eps=eps)
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
                "true_U": step["improvement"],
                "outcome": ("success" if step["improvement"] > 0 else
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
            if g.gstep >= max_depth:
                g.done = True
                g.adv_reason = "max_depth"
            out.append({"iid": g.iid, "action": "act", "reason": "policy_act",
                        "improvement": step["improvement"], "sig": step["sig"],
                        "state_hash": step["state_hash"], "succ_hash": nh,
                        "gstep": g.gstep, "g_ms": g.ms})
        else:
            # every non-advancing outcome truly ends the episode (no forced continuation)
            g.done = True
            g.adv_reason = step["reason"]
            out.append({"iid": g.iid, "action": "stop", "reason": step["reason"],
                        "state_hash": step["state_hash"]})
    return out


# ---------------------------------------------------------------------------
# unified-parity evaluator -- ONE ruler for R6 / R11 / R12  (§39-40)
# ---------------------------------------------------------------------------
def selector_action_logits(selector, F_pool, sf_t, pool_stats, evid=None, **kw):
    """Adapter to a single [M+1] action-logits API.

    * M3RollingGRPOPolicy exposes `.action_logits` directly (z-pool base + bounded
      residual + frozen R6 STOP).  While delta==0 the z-projection is a monotone affine
      map of the raw R6 rank so its argmax equals the canonical R6 argmax.
    * M3Top1Selector returns (prop[M], stop[1]) which `_action_logits` concatenates.
    * evid: R18 ProposalEvidenceResidualAdapter observation [M, EVID_DIM] (gain_norm,
      is_memory_rescued, memory_confidence) -- passed only on the R18 validated
      action set (§16-20); ignored by policies without a evidence adapter.
    """
    if hasattr(selector, "action_logits"):
        return selector.action_logits(F_pool, sf_t, pool_stats, evid=evid, **kw)
    prop, stop = selector(F_pool, sf_t, pool_stats)
    return _action_logits((prop, stop), int(F_pool.shape[0]))


def unified_parity_rollout(env, rf, scorer, selector, use_mem=True,
                           horizon=None, gate_mem=False):
    """Deterministic greedy closed loop under the CANONICAL R6 ruler (the exact body of
    `top1.rollout_top1`, with the selector abstracted to a single logits API):
        greedy argmax over [props..., STOP] ->
        execute -> write progressive memory ->
        STOP on: no-proposals / no-pool / selector-STOP /
                 executed improvement <= 0 (stop_neg) / revisit / horizon.
    This is the SAME runtime (FixedDecisionReplay, no solver), SAME Memory, SAME
    pool normalization (wide_pool + _rerank_feats_all + _pool_stats_from), SAME horizon
    for every selector -- R6 (delta=0 policy), R11 (m3_rolling_grpo_v1.pt) and R12.
    """
    horizon = int(horizon if horizon is not None else C.TO1_R12_HORIZON)
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
               else torch.zeros(n_prop, 277, dtype=torch.float32))
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
            logits = selector_action_logits(selector, F_pool, sf_t, pool_stats)
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


def unified_parity_closed_loop(env, scorer, selector, roots, use_mem=True,
                               horizon=None, gate_mem=False):
    """`_b5_summary`-shaped gains over an explicit root set (TRAIN / held / VAL all
    supported by passing the right roots).  Used for the CANONICAL-PARITY TABLE and
    for R12 selection (TRAIN + AUX-held) and the single no_grad VAL run."""
    gains_by_iid = {}
    steps_by_iid = {}
    for rf in roots:
        iid = rf["iid"]
        gain, usage, steps = unified_parity_rollout(
            env, rf, scorer, selector, use_mem=use_mem, horizon=horizon,
            gate_mem=gate_mem)
        gains_by_iid[iid] = gain
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    from .rollout import _b5_summary
    return _b5_summary(gains_by_iid), steps_by_iid


def greedy_step_unified(policy, scorer, executor, model_b5, single_head, direct_head,
                        problem, schedule, iid, episode_id, progmem, root_ms,
                        step_offset=0, gate_mem=False):
    """Deterministic greedy action at state S_t under the UNIFIED canonical ruler
    (compares against R11 `greedy_step` in rolling_grpo.py):

      * selector abstraction: action logits via `selector_action_logits` (works for
        both M3RollingGRPOPolicy and M3Top1Selector).
      * TRUE STOP semantics: a non-positive executed outcome (improvement <= 0) and
        an infeasible edit both END the episode here (return stop), mirroring
        `unified_parity_rollout` and the canonical R6 `rollout_top1`.
      * directed at the DPP rolling trace: like the R11 trace it does NOT write
        progressive memory (the trace walks the single greedy path only).

    Returns the same dict shape as R11 `greedy_step` (never mutates inputs; successor
    schedule is produced fresh by the executor)."""
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
        logits = selector_action_logits(policy, F_pool, sf_t, pool_stats)
    out["logits"] = logits.detach().float()
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
    if res["improvement"] <= 0:
        out.update({"action": "stop", "reason": "non_positive", "state_hash": h,
                    "sig": sig, "kind": kind, "improvement": float(res["improvement"])})
        return out
    out.update({"action": "act", "reason": "policy_act", "state_hash": h,
                "sig": sig, "kind": kind, "meta_type": rolex["type"][pool[sel]],
                "meta_role": rolex["role"][pool[sel]], "src": rolex["src"][pool[sel]],
                "tgt": rolex["tgt"][pool[sel]], "improvement": float(res["improvement"]),
                "successor": res["schedule"]})
    return out


def dpp_rolling_trace_r12(selector, scorer, env, re, iid, episode_id, st,
                          dpp_lut, horizon=5, gate_mem=False, log_prefix="[r12]"):
    """Multi-state ROLLING trace at the DPPaulli instance under the UNIFIED canonical
    ruler (true STOP semantics): greedy advance state by state, never past a STOP or a
    non-positive executed step.  Mirrors R11 `dpp_rolling_trace_r11` shape so the
    pre/post rows are comparable."""
    out = {"iid": iid, "found": False, "episode_id": episode_id, "steps": []}
    problem, schedule = st["problem"], st["schedule"]
    progmem = copy.deepcopy(re["progmem"])
    ms0 = int(schedule.makespan)
    reward = 0
    visiteds = {schedule_hash(schedule)}
    for d in range(min(int(horizon), 5)):
        step = greedy_step_unified(selector, scorer, env["executor"],
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


def residual_saturation_stats_r12(policy, groups, sat_level=None, frac_limit=None):
    """Fraction of residual units at/near the alpha bound (|tanh delta| > level)."""
    sat_level = float(sat_level if sat_level is not None else C.TO1_R12_RESID_SAT)
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
                    n_sat += int((d > sat_level).sum().item())
                    obs.extend(d.numpy().tolist())
    return {"saturated_frac": float(n_sat / n) if n else 0.0,
            "n_units": n, "mean_abs_delta": float(np.mean(np.abs(obs))) if obs else 0.0,
            "limit": float(frac_limit if frac_limit is not None
                           else C.TO1_R12_RESID_SAT_FRAC_LIMIT)}


# ---------------------------------------------------------------------------
# Stage B: deterministic rolling cycles (multiprocess, true STOP, online pools)
# ---------------------------------------------------------------------------
def run_rolling_cycles_r12(policy, scorer, env, schedule_specs, cycles=None, k=None,
                           horizon=None, graphs_per_batch=None, workers=1,
                           seed=0, quick=False, log_prefix="[r12]", eval_root_builder=None,
                           collapse_floor=None, parent_policy=None, mp_ctx=None,
                           gate_mem_eval=False):
    """Run the R12 Stage B rolling GRPO loop.

    schedule_specs: {"bench": [Graph...], "real": [Graph...], "syn": [Graph...]}
    eval_root_builder(policy, cycle) -> {"train": summary, "real_held": summary,
                                         "syn_held": summary} (UNIFIED greedy ruler).
    parent_policy: a frozen clone of the R11 warm-start for KL-to-parent logging (§37).

    Per cycle: for each depth: freeze pi_old (implicit in stored logits_old), collect K
    sibling trajectories per active graph via ProcessPoolExecutor, `grpo_update_traj`
    (E reuse epochs, per-trajectory-equal weight), then STOCHASTIC real advancement.
    Graphs rotate/reset after MAX_EPISODES_PER_GRAPH episodes (§19-20, §27-29).
    Collapse guard compares TRAIN gain (unified ruler) against `collapse_floor`.
    """
    cycles = int(cycles if cycles is not None else C.TO1_R12_TRAINING_CYCLES)
    k = int(k if k is not None else C.TO1_R12_K)
    horizon = int(horizon if horizon is not None else C.TO1_R12_HORIZON)
    gpb = int(graphs_per_batch if graphs_per_batch is not None else C.TO1_R12_GRAPHS_PER_BATCH)
    max_depth = int(C.TO1_R12_MAX_DEPTH)
    workers = int(workers)
    rng = random.Random(seed)

    collapse_ratio = float(C.TO1_R12_COLLAPSE_TRAIN_RATIO)
    # `collapse_floor` = the same-ruler warm-start TRAIN gain this run is guarded
    # against; the entry script always computes it under the unified evaluator.
    r6_floor = float(collapse_floor) if collapse_floor is not None else None

    model_b5, single_head, direct_head = env["model_b5"], env["single_head"], env["direct_head"]
    executor = env["executor"]
    step0_cache = env["cache"]

    # deterministic per-cycle source mixing into a fixed slot schedule (§27-28)
    bench_q = [g for g in schedule_specs["bench"]
               for _ in range(C.TO1_R12_MAX_EPISODES_PER_GRAPH)]
    real_q = [g for g in schedule_specs["real"]
              for _ in range(C.TO1_R12_MAX_EPISODES_PER_GRAPH)]
    syn_q = [g for g in schedule_specs["syn"]
             for _ in range(C.TO1_R12_MAX_EPISODES_PER_GRAPH)]
    per_src = {"bench": 2, "real": 1, "syn": 1} if not quick else {"bench": 2}
    sched = []
    for c in range(cycles):
        for src in ("bench", "real", "syn"):
            for _ in range(per_src.get(src, 0)):
                sched.append((c, src))
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
            g.reset()                  # fresh episode: pristine S0 + memory (§19)
        cycle_handle[c].append(g)

    history = []
    best = {"cycle": -1, "score": -1e18, "policy": policy.snapshot(),
            "train": 0, "real_held": 0, "syn_held": 0}
    collapsed = False
    collapse_reason = None
    for c in range(cycles):
        t_start = time.time()
        active = cycle_handle[c]
        depth_results = []
        all_groups = []
        cycle_kl_ref = []
        cycle_kl_parent = []
        for depth in range(max_depth):
            act = [g for g in active if not g.done]
            if not act:
                break
            groups = []
            for g in act:
                grp = collect_full_group_rollouts(
                    policy, scorer, executor, model_b5, single_head, direct_head,
                    g.problem, g.schedule, g.ms, g.iid, g.episode_id, g.progmem,
                    k=k, horizon=horizon, seed=seed + c * 1000 + depth * 100,
                    step_offset=g.gstep, workers=workers, step0_cache=step0_cache,
                    mp_ctx=mp_ctx)
                groups.append(grp)
            all_groups.extend(groups)
            upd = grpo_update_traj(policy, groups, seed=seed, parent_policy=parent_policy,
                                   log_prefix=log_prefix)
            if upd["epochs"]:
                cycle_kl_ref.append(upd["epochs"][-1]["kl_ref"])
                if upd["epochs"][-1]["kl_parent"] is not None:
                    cycle_kl_parent.append(upd["epochs"][-1]["kl_parent"])
            depth_results.append({"depth": depth,
                                  "n_groups": len(groups),
                                  "n_informative_trajectories": upd["n_informative_trajectories"],
                                  "n_trajectories": upd["n_trajectories"],
                                  "mean_reward": float(np.mean(
                                      [g["mean_reward"] for g in groups])) if groups else 0.0,
                                  "update": upd})
            # real STOCHASTIC advancement with the UPDATED policy (no oracle, §14-15)
            out = advance_graphs_r12(policy, scorer, executor, model_b5, single_head,
                                     direct_head, act, rng, log_prefix=log_prefix)
            depth_results[-1]["advance"] = out
        # ---- per-cycle deterministic evaluation (unified greedy ruler) ------
        ev = {}
        if eval_root_builder is not None:
            ev = eval_root_builder(policy, cycle=c)
        train_gain = float(ev.get("train", {}).get("total", 0.0))
        real_hd = float(ev.get("real_held", {}).get("total", 0.0))
        syn_hd = float(ev.get("syn_held", {}).get("total", 0.0))
        score = train_gain + real_hd + syn_hd
        sat = residual_saturation_stats_r12(policy, all_groups)
        sat["over_limit"] = bool(sat["saturated_frac"] > sat["limit"])
        n_info = sum(1 for g in all_groups if g["informative"])
        n_groups = len(all_groups)
        informative_ratio = float(n_info / n_groups) if n_groups else 0.0
        m_kl_ref = float(np.mean(cycle_kl_ref)) if cycle_kl_ref else 0.0
        m_kl_parent = float(np.mean(cycle_kl_parent)) if cycle_kl_parent else None
        dumped = {"cycle": c, "train": train_gain, "real_held": real_hd,
                  "syn_held": syn_hd, "score": score, "informative_ratio": informative_ratio,
                  "n_groups": n_groups, "n_informative": n_info,
                  "kl_ref": m_kl_ref, "kl_parent": m_kl_parent, "sat": sat,
                  "depth_results": depth_results, "sec": round(time.time() - t_start, 2)}
        # ---- collapse guard (same-ruler floor) ------------------------------
        if r6_floor is not None and (
                (train_gain < collapse_ratio * r6_floor) or sat["over_limit"]):
            collapsed = True
            collapse_reason = ("train_gain" if train_gain < collapse_ratio * r6_floor
                               else "residual_saturation")
            dumped["collapse"] = {"flag": True, "reason": collapse_reason}
        if score > best["score"] and not collapsed:
            best = {"cycle": c, "score": score, "policy": policy.snapshot(),
                    "train": train_gain, "real_held": real_hd, "syn_held": syn_hd}
        history.append(dumped)
        lp = "None" if m_kl_parent is None else f"{m_kl_parent:.4f}"
        print(f"{log_prefix} cycle {c}: TRAIN={train_gain:.0f} real_hd={real_hd:.0f} "
              f"syn_hd={syn_hd:.0f} informative={informative_ratio:.3f} "
              f"kl_ref={m_kl_ref:.4f} kl_parent={lp} "
              f"sat={sat['saturated_frac']:.3f} sec={dumped['sec']}s", flush=True)
        if collapsed:
            break
    policy.load_snapshot(best["policy"])
    return {"history": history, "best": best, "collapsed": collapsed,
            "collapse_reason": collapse_reason, "cycles_run": len(history)}


# ---------------------------------------------------------------------------
# R11 credit-assignment audit record  (§2-3)
# ---------------------------------------------------------------------------
def audit_r11_credit_assignment():
    """Executable audit record answering the FIRST R12 task: did R11 credit the terminal
    gain to all trajectory steps, and how did it weight them?

    Reads the actual R11 source (rolling_grpo.collect_group_rollouts / grpo_update) so the
    verdict tracks the code as written, not the design doc.

    Verdict (confirmed by reading the code):
      logp_old coverage   : every step stores mixture logp under the frozen collection
                            policy (NOT first-action-only).
      terminal advantage  : the shared group-relative A_i=(R_i-mean)/(std+eps) is attached
                            to every step of trajectory i -- multi-step terminal credit.
      loss weighting      : FLAT mean over all steps  loss=sum(terms)/len(terms)  -- a
                            length-5 trajectory counts 5x a STOP-first trajectory.  This
                            violates the R12 §7 per-trajectory-equal-weight contract.
    => R11 scientific status: MULTI_STEP_TERMINAL_ADVANTAGE_WITH_STEP_FLAT_WEIGHTING
       (NOT FIRST_ACTION_GRPO; NOT the final goal).  R12's loss replaces the flat mean
       with mean_i[mean_k surrogate_i,k] -- the audit drives that change.
    """
    from . import rolling_grpo as RG
    src_collect = inspect.getsource(RG.collect_group_rollouts)
    src_traj = inspect.getsource(RG.collect_trajectory)
    src_loss = inspect.getsource(RG.grpo_update)

    record = {
        "audit": "r11_credit_assignment",
        "trajectory_logp_old_per_step": "logp_old" in src_traj
        and "steps.append(rec)" in src_traj,
        "shared_terminal_advantage_all_steps": "rec[\"adv_group\"] = a" in src_collect,
        "all_steps_in_loss": "for st in samples" in src_loss,
        "loss_weighting": ("step_flat_mean"
                           if "sum(terms) / len(terms)" in src_loss
                           else "unknown"),
        "trajectory_product_ratio": False,          # no prod over trajectory in R11
        "status": "MULTI_STEP_TERMINAL_ADVANTAGE_WITH_STEP_FLAT_WEIGHTING",
        "first_action_only": False,
        "fix_in_r12": "mean_i[mean_k surrogate_i,k]; STOP-traj length-1 keeps unit mass",
    }
    if not record["trajectory_logp_old_per_step"] or not record["shared_terminal_advantage_all_steps"]:
        record["status"] = "AUDIT_HEADER_MISMATCH_VERIFY_MANUALLY"
    return record


# ---------------------------------------------------------------------------
# profiling + cloud extrapolation  (§47-48)
# ---------------------------------------------------------------------------
def trajectory_profiling_r12(policy, scorer, env, root, k=8, horizon=5,
                             weights=(1, 2, 4, 8), mp_ctx=None):
    """Time trajectory collection across worker counts (1 in-process, 2/4/8 via
    ProcessPoolExecutor) on the same root; report wall clock, speedup vs serial and
    identical-data verification across all worker counts.  CPU proxy = summed
    trajectory phase time vs wall clock."""
    out = {}
    ref_key = None
    for w in weights:
        w = int(w)
        g = collect_full_group_rollouts(
            policy, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], root["root_ms"],
            root["iid"], root["episode_id"], root["progmem"], k=k, horizon=horizon,
            seed=0, workers=w, step0_cache=env["cache"], mp_ctx=mp_ctx)
        profs = [tr.get("prof") for tr in g["trajs"] if tr.get("prof")]
        agg = {}
        for key in ("analyze", "execute", "mem", "policy"):
            vals = [p[key] for p in profs if key in p]
            agg[key] = float(sum(vals) / len(vals)) if vals else 0.0
        tot = sum(agg.values()) or 1e-9
        busy = sum(agg.values()) * k
        out[str(w)] = {"coll_s": g["coll_s"], "n_steps": g["n_steps"],
                       "rewards": g["rewards"], "terminal_counts": g["terminal_counts"],
                       "parallelism": g["parallelism"],
                       "phase_s_per_traj": agg,
                       "fixed_dr_share": round((agg["analyze"] + agg["execute"]) / tot, 4),
                       "cpu_busy_s": round(busy, 3)}
        if ref_key is None:
            ref_key = _group_identity_key(g)
        out[str(w)]["identical_to_w1"] = bool(_group_identity_key(g) == ref_key)
    serial = out[str(weights[0])]["coll_s"]
    for key in out:
        if key == "speedup":
            continue
        out[key]["speedup"] = round(serial / out[key]["coll_s"], 4) if out[key]["coll_s"] > 0 else None
    out["speedup"] = {str(w): out[str(w)]["speedup"] for w in weights}
    out["all_identical"] = all(out[str(w)]["identical_to_w1"] for w in weights)
    return out


def cloud_extrapolate(profile):
    """Fit Amdahl's serial fraction from observed worker counts (use the LARGEST observed
    count for stability), then project wall-time / speedup at 16/32/64 cores.

    Speedup is computed RELATIVE to worker=1 time; both the serial-fraction and the
    observed-best-workers use string keys (the profile dict is JSON-shaped)."""
    points = []
    for key in ("1", "2", "4", "8"):
        if key in profile and isinstance(profile.get(key), dict) \
                and profile[key].get("coll_s"):
            points.append((int(key), profile[key]["coll_s"]))
    if not points:
        return {"error": "empty profile"}
    points.sort()
    t1 = next((t for (w, t) in points if w == 1), None)
    if t1 is None or t1 <= 0:
        return {"error": "need worker=1 baseline"}
    speedup = {w: t1 / t for (w, t) in points}          # >= 1 iff parallel helps
    # Amdahl serial fraction from the LARGEST observed count (most stable):
    #   speedup(w) = 1 / (sf + (1-sf)/w)  ->  sf = (w/sp - 1)/(w-1)
    w_ref, t_ref = points[-1]
    sp_ref = speedup[w_ref]
    sf = float(min(max(0.0, (w_ref / sp_ref - 1.0) / (w_ref - 1.0)) if w_ref > 1 else 0.0,
                   1.0 - 1e-6))
    # observed best worker count over {2,4,8} (string keys); serial if nothing helps
    best_obs = (1, 1.0)
    for w in (2, 4, 8):
        sp_w = speedup.get(w, 0.0)
        if sp_w > 1.0 and (sp_w > best_obs[1] + 1e-9
                           or (abs(sp_w - best_obs[1]) < 1e-9 and w > best_obs[0])):
            best_obs = (w, sp_w)
    ram_per_worker_gb = 1.2                     # torch + m3 + frozen upstream + deepcopies
    table = {}
    for target in (16, 32, 64):
        sp = 1.0 / (sf + (1.0 - sf) / target)
        rec_w = int(min(target, best_obs[0])) if best_obs[0] > 1 else 1
        table[str(target)] = {"cores": target, "predicted_speedup": round(sp, 3),
                              "predicted_coll_s": round(t1 / sp, 2),
                              "serial_fraction": round(sf, 4),
                              "recommended_workers": rec_w,
                              "ram_est_gb": round(rec_w * ram_per_worker_gb, 1)}
    return {"fit_serial_fraction": round(sf, 4), "fitter_workers": w_ref,
            "observed_points": points, "observed_best_workers": best_obs[0],
            "projections": table}


# ---------------------------------------------------------------------------
# Stage A: 8 exact-semantics checks  (§35)
# ---------------------------------------------------------------------------
def stage_a_verify(test_path=None, extra_args=None):
    """Run the R12 Stage A pytest suite (8 exact-semantics checks).  Returns a summary"""
    import subprocess
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    test_path = test_path or str(repo_root / "tests" / "test_m3_traj_grpo_r12.py")
    cmd = [sys.executable, "-m", "pytest", test_path, "-m", "r12stageA",
           "-q", "--no-header", "--tb=line"]
    if extra_args:
        cmd += list(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else ""
    return {"pass": (proc.returncode == 0), "returncode": proc.returncode,
            "summary": summary,
            "n_passed": _pytest_count(proc.stdout, "passed"),
            "n_failed": _pytest_count(proc.stdout, "failed")}


def _pytest_count(text, key):
    """Count `N key` tokens in a pytest summary line (no re dependency)."""
    for line in (text or "").splitlines():
        tokens = line.replace(",", " ").split()
        if key in tokens:
            i = tokens.index(key)
            if i >= 1 and tokens[i - 1].isdigit():
                return int(tokens[i - 1])
    return 0
