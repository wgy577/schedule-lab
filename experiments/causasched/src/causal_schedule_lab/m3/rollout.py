"""Canonical M3 replay + closed-loop rollout + regression traces.

VERBATIM extraction (no semantic change):
  - build_replay                                <- r2
  - _roll_ex_from / _scores_raw / rollout_wide_rerank / _b5_summary /
    dppaulli_r4_trace / normal_m5_r4_path       <- R4
Internal `r1.`/`r2.`/`r3.` refs rewritten to canonical module equivalents.
"""

from __future__ import annotations

import numpy as np
import torch

from causal_schedule_lab.validation import schedule_hash

from .config import HORIZON, INFEASIBLE_U, K_C, K_R, MEM_FEAT_DIM
from .gate import _old_base_arr, pooled_gate_input
from .memory import MemoryStore, compute_mem_features
from .proposal_features import (
    _edits_for,
    _execute_step,
    old_base_of,
    proposal_identity,
    state_feature_vec,
)
from .ranking import _rerank_feats_all, rerank_order


# ---------------------------------------------------------------------------
# persistent TRAIN replay  [r2]
# ---------------------------------------------------------------------------
def build_replay(cache, executor, train_insts, states):
    """Walk oracle trajectory per TRAIN instance; label every bounded Proposal.

    Returns (state_examples, memory_store).
    state_example = {iid, state_hash, root_ms, state_feat, prop_feats, metas,
                     true_U, feasible, old_base, sig, role, type, src, tgt, op_id}
    """
    store = MemoryStore()
    state_examples = []
    n_infeasible = 0
    for inst in train_insts:
        iid = inst["instance_id"]
        st = states[iid]
        problem, schedule = st["problem"], st["schedule"]
        root_ms = int(schedule.makespan)
        visited = {schedule_hash(schedule)}
        step = 0
        while step < HORIZON:
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
            if not metas:
                break
            ast = cache.ast(problem, schedule, iid)
            h = schedule_hash(schedule)
            ms_cur = int(schedule.makespan)
            sf = state_feature_vec(ms_cur, root_ms, len(metas), agg["best_uhat"],
                                   agg["best_direct"], agg["n_contrib"], agg["n_enab"])
            sf_t = torch.tensor(sf, dtype=torch.float32)

            N = len(metas)
            true_U = np.full(N, INFEASIBLE_U, dtype=np.float32)
            feasible = np.zeros(N, dtype=bool)
            old_base = np.zeros(N, dtype=np.float32)
            sigs, roles, types, srcs, tgts, op_ids = [], [], [], [], [], []
            for k in range(N):
                edits, kind, sig, role, ptype, src, tgt, oids = proposal_identity(ast, metas[k])
                sigs.append(sig); roles.append(role); types.append(ptype)
                srcs.append(src); tgts.append(tgt); op_ids.append(oids)
                old_base[k] = old_base_of(prop_feats, k)
                res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
                if res is None:
                    feasible[k] = False
                    n_infeasible += 1
                    outcome = "infeasible"
                else:
                    u = float(res["improvement"])
                    feasible[k] = True
                    true_U[k] = u
                    outcome = "success" if u > 0 else ("neutral" if u == 0 else "negative")
                store.add({
                    "instance_id": iid, "state_hash": h, "proposal_signature": sig,
                    "proposal_type": ptype, "role": role, "src": src, "tgt": tgt,
                    "op_ids": oids, "true_U": (float(true_U[k]) if feasible[k] else None),
                    "outcome": outcome, "successor_state_hash": None, "trajectory_step": step,
                    "fine_key": (ptype, role, src, tgt) if ptype == "single" else (ptype, role),
                    "coarse_key": (ptype, role),
                })

            state_examples.append({
                "iid": iid, "state_hash": h, "root_ms": root_ms, "state_feat": sf_t,
                "prop_feats": prop_feats, "metas": metas, "true_U": true_U,
                "feasible": feasible, "old_base": old_base, "sig": sigs, "role": roles,
                "type": types, "src": srcs, "tgt": tgts, "op_id": op_ids,
                "n_contrib": agg["n_contrib"], "n_enab": agg["n_enab"],
            })

            # oracle best-positive (over feasible only)
            best = None
            for k in range(N):
                if feasible[k] and true_U[k] > 0 and (best is None or true_U[k] > true_U[best]):
                    best = k
            if best is None:
                break
            edits, kind = _edits_for(ast, metas[best])
            res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
            schedule = res["schedule"]
            nh = schedule_hash(schedule)
            if nh in visited:
                break
            visited.add(nh)
            step += 1
    print(f"[replay] {len(state_examples)} states, {len(store.records)} proposal labels, "
          f"{n_infeasible} infeasible", flush=True)
    return state_examples, store


# ---------------------------------------------------------------------------
# closed-loop B5 (wide + rerank + gate + progressive memory)  [R4]
# ---------------------------------------------------------------------------
def _scores_raw(scorer, prop_feats, mem, sf_t):
    with torch.no_grad():
        logit_pos, rank, _ = scorer(prop_feats, mem, sf_t)
    return logit_pos, rank


def _roll_ex_from(ast, metas, prop_feats, sf_t):
    """Lightweight ex-like dict for pool/rerank helpers at rollout time."""
    N = len(metas)
    roles, ptypes, srcs, tgts = [], [], [], []
    for k in range(N):
        _e, _kind, _sig, role, ptype, src, tgt, _oids = proposal_identity(ast, metas[k])
        roles.append(role); ptypes.append(ptype); srcs.append(src); tgts.append(tgt)
    return {"metas": metas, "prop_feats": prop_feats, "state_feat": sf_t,
            "role": roles, "type": ptypes, "src": srcs, "tgt": tgts}


def rollout_wide_rerank(problem, schedule0, cache, scorer, reranker, gate, executor, iid,
                        progmem, episode_id, use_mem, horizon=HORIZON):
    """FixedDecisionReplay roll: gate (ACT-biased) -> wide pool -> rerank -> top1
    execute -> write executed-transition to progressive memory (real arm)."""
    s0_ms = int(schedule0.makespan)
    ms_cur = s0_ms
    schedule = schedule0
    visited = {schedule_hash(schedule0)}
    act_usage = {"single": 0, "pair": 0, "stop_by_gate": 0, "stop_neg": 0}
    steps = []
    for t in range(horizon):
        prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        n_prop = len(metas)
        if n_prop == 0:
            act_usage["stop_by_gate"] += 1
            break
        ast = cache.ast(problem, schedule, iid)
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = _roll_ex_from(ast, metas, prop_feats, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        if use_mem:
            mem = torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                               dtype=torch.float32)
        else:
            mem = torch.zeros(n_prop, MEM_FEAT_DIM, dtype=torch.float32)
        logit_pos, rank = _scores_raw(scorer, prop_feats, mem, sf_t)
        oldbase = _old_base_arr(prop_feats)
        x, *_ = pooled_gate_input(sf_t, logit_pos, rank, oldbase, K_R, K_C)
        with torch.no_grad():
            pred_cls = int(torch.argmax(gate(x)).item())
        if pred_cls == 1:                     # STOP
            act_usage["stop_by_gate"] += 1
            break
        _, final_order, pinfo = rerank_order(scorer, reranker, rolex, mem)
        if not final_order:
            act_usage["stop_by_gate"] += 1
            break
        a = final_order[0]
        edits, kind = _edits_for(ast, metas[a])
        res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = proposal_identity(ast, metas[a])[2]
        steps.append({"t": t, "gate": "ACT", "n_prop": n_prop,
                      "n_wide": int(pinfo.get("union", 0)),
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


def _b5_summary(gains_by_iid):
    """total / mean / median / n_positive / leave-max-out, with per-instance gains."""
    iids = list(gains_by_iid.keys())
    gains = [gains_by_iid[i] for i in iids]
    total = int(sum(gains))
    arr = np.array(gains, dtype=np.float64)
    pos = sum(1 for g in gains if g > 0)
    others = [g for g in gains if g != max(gains)] if len(gains) > 1 else gains
    return {"total": total,
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "n_positive_instances": pos,
            "leave_max_out_total": int(sum(others)),
            "per_instance": gains_by_iid}


# ---------------------------------------------------------------------------
# DPpaulli10a trace and normal-M5 via wide+rerank  [R4]
# ---------------------------------------------------------------------------
def dppaulli_r4_trace(iid, episode_id, st, cache, scorer, reranker, gate, executor,
                      progmem, s0_trueU_by_sig):
    out = {"iid": iid, "episode_id": episode_id, "found": False}
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return out
    out["found"] = True
    ast = cache.ast(st["problem"], st["schedule"], iid)
    s0_hash = schedule_hash(st["schedule"])
    N = len(metas)
    sf = state_feature_vec(int(st["schedule"].makespan), int(st["schedule"].makespan),
                           N, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _roll_ex_from(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, 0, sf, queries),
                       dtype=torch.float32)
    logit_pos, rank = _scores_raw(scorer, prop_feats, mem, sf_t)
    oldbase = _old_base_arr(prop_feats)
    x, *_ = pooled_gate_input(sf_t, logit_pos, rank, oldbase, K_R, K_C)
    with torch.no_grad():
        pred_cls = int(torch.argmax(gate(x)).item())
    out["gate_pred"] = "ACT" if pred_cls == 0 else "STOP"
    pool, final_order, pinfo = rerank_order(scorer, reranker, rolex, mem)
    pool_set = set(pool)
    lut = s0_trueU_by_sig
    sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
    pos = [k for k in range(N) if lut.get(sigs[k], 0) > 0]
    out["n_pos_total"] = len(pos)
    out["n_pos_in_wide"] = sum(1 for k in pos if k in pool_set)
    out["n_pos_in_final_top10"] = sum(1 for k in pos if k in set(final_order[:10]))
    best = None
    if pos:
        best = max(pos, key=lambda k: lut[sigs[k]])
        topk = set(final_order[:10])
        out["best_in_final_top10"] = best in topk
        old_base_arr = _old_base_arr(prop_feats)
        old_rank = int(np.where(np.argsort(np.argsort(-old_base_arr)) == best)[0][0]) + 1
        f = _rerank_feats_all(scorer, rolex, mem)
        if f is not None:
            with torch.no_grad():
                s_all = reranker(f).numpy()
            rr = int(np.where(np.argsort(np.argsort(-s_all)) == best)[0][0]) + 1
            ranked_pool = sorted(pool, key=lambda k: -s_all[k])
            best_rank_in_pool_reranked = (ranked_pool.index(best) + 1) if best in ranked_pool else None
        else:
            rr = None
            best_rank_in_pool_reranked = None
        wide_rank = (list(final_order).index(best) + 1) if best in set(final_order) else None
        out["best_positive"] = {"old_rank": old_rank, "wide_rank": wide_rank,
                                "rerank_rank_all": rr,
                                "best_rank_in_pool_reranked": best_rank_in_pool_reranked,
                                "true_U": float(lut[sigs[best]])}
    sel_k = final_order[0] if final_order else None
    if sel_k is not None:
        sel_edits, sel_kind = _edits_for(ast, metas[sel_k])
        res = _execute_step(executor, st["problem"], st["schedule"], sel_edits,
                            int(st["schedule"].makespan), s0_hash)
        out["selected"] = {"sig": sigs[sel_k], "in_wide": sel_k in pool_set,
                           "in_final_top10": sel_k in set(final_order[:10]),
                           "kind": sel_kind,
                           "true_U": (float(res["improvement"]) if res else None),
                           "matches_best_positive": bool(best is not None and sel_k == best)}
        out["selected_true_U"] = (float(res["improvement"]) if res else None)
    if pred_cls == 0:
        if not pos:
            out["diagnosis"] = "gate_act_no_positive"
        elif out.get("best_in_final_top10"):
            out["diagnosis"] = "selected_correct_positive"
        elif out.get("n_pos_in_wide", 0) > 0:
            out["diagnosis"] = "rerank_error_positive_in_wide"
        else:
            out["diagnosis"] = "wide_recall_error_no_positive_in_wide"
    else:
        out["diagnosis"] = "false_stop_gate_error" if pos else "correct_stop"
    return out


def normal_m5_r4_path(iid, episode_id, st, cache, scorer, reranker, gate, executor,
                      progmem, s0_trueU_by_sig):
    out = {"iid": iid, "episode_id": episode_id, "found": False}
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return out
    ast = cache.ast(st["problem"], st["schedule"], iid)
    s0_hash = schedule_hash(st["schedule"])
    N = len(metas)
    sf = state_feature_vec(int(st["schedule"].makespan), int(st["schedule"].makespan),
                           N, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rolex = _roll_ex_from(ast, metas, prop_feats, sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, 0, sf, queries),
                       dtype=torch.float32)
    logit_pos, rank = _scores_raw(scorer, prop_feats, mem, sf_t)
    oldbase = _old_base_arr(prop_feats)
    x, *_ = pooled_gate_input(sf_t, logit_pos, rank, oldbase, K_R, K_C)
    with torch.no_grad():
        pred_cls = int(torch.argmax(gate(x)).item())
    out["gate_pred"] = "ACT" if pred_cls == 0 else "STOP"
    pool, final_order, pinfo = rerank_order(scorer, reranker, rolex, mem)
    pool_set = set(pool)
    final10 = set(final_order[:10])
    lut = s0_trueU_by_sig
    sigs = [proposal_identity(ast, metas[k])[2] for k in range(N)]
    rank_np = rank.numpy()
    enabler_m5 = []
    for k in range(N):
        _e, kind, sig, role, ptype, src, tgt, _oids = proposal_identity(ast, metas[k])
        if kind == "single" and role == "ENABLER" and src == "M5":
            res = _execute_step(executor, st["problem"], st["schedule"], _e,
                                int(st["schedule"].makespan), s0_hash)
            enabler_m5.append({
                "sig": sig, "rank_head": int((rank_np > rank_np[k]).sum()) + 1,
                "in_wide": k in pool_set, "in_final_top10": k in final10,
                "old_base": float(oldbase[k]),
                "executed_improvement": (None if res is None else float(res["improvement"])),
            })
    out["n_enabler_m5"] = len(enabler_m5)
    sel_k = final_order[0] if final_order else None
    if sel_k is not None:
        sel_edits, sel_kind = _edits_for(ast, metas[sel_k])
        res = _execute_step(executor, st["problem"], st["schedule"], sel_edits,
                            int(st["schedule"].makespan), s0_hash)
        out["selected"] = {"sig": sigs[sel_k], "in_wide": sel_k in pool_set,
                           "in_final_top10": sel_k in final10, "kind": sel_kind,
                           "true_U": (float(res["improvement"]) if res else None)}
    out["enabler_m5"] = enabler_m5[:8]
    out["found"] = True
    return out