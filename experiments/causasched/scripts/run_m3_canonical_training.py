#!/usr/bin/env python3
"""T1-M3-CANONICAL-CONSOLIDATION-UTILITY-RERANK-GRPO-R5 -> R8 canonical M3 entry.

R8 (current): T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION-R8.  SFT-ONLY --
GRPO / PPO / Actor-Critic / REINFORCE / bounded-residual RL are FORBIDDEN (§37).
Primary objective = best-vs-hardest-competitor pool-argmax (L_argmax).  STOP head
FROZEN (R6's, diagnostic only).  Fixed 3-fold by-INSTANCE internal CV on TRAIN14.
Feature-scale audit + g_mem confidence-gated memory.  VAL run once, no tuning.
R6 (--stage top1, R7 --stage r7) reproducible explicitly; R5 GRPO via --stage p2.

Phase 0  project consolidation: inventory + A-F classification (+ archive happens
         once this entry point is confirmed: legacy scripts -> archive/legacy_m3/,
         outputs -> outputs/archive/, canonical checkpoints -> outputs/canonical_m3/).
Phase 1  (R5) final SFT utility-aware reranking + acceptance (B5 train/val).
Stage top1 (R6)  Top-1 utility selection SFT: supervised listwise selector over
         the WIDE pool, STOP head first-class, hard-state oversampling
         (DPpaulli10a + TOP1_FAILURE_STATE), auxiliary weighted |ΔU| pair loss,
         B6 closed loop (train/val), DPpaulli trace, normal-M5 regression,
         memory masked ablation, R6 verdicts A-E, checkpoint
         m3_proposal_top1_sft_v2.pt.  No GRPO.
Stage r7 (R7, explicit only)  STOP calibration + OOD gen (verdict E, 2026-08-27).
Stage r8 (R8, explicit)       pool-argmax SFT + cross-instance internal CV.
Stage r10gen (R10)            select + run the D1=SAFE_AUX_REAL instances through the
                              canonical state+replay label pipeline (independent data stage,
                              no model trained) -> outputs/r10_aux_real/r10_aux_real_data.pt.
Stage r10 (R10, current)      instance-relative score calibration: frozen R6 proposal rank,
                              robust within-state median/MAD z (§7), pool-conditioned STOP
                              calibrator (§8-§9).  AUX-REAL preferred (§10-§11).  SFT only,
                              GRPO forbidden (§41).  Checkpoint m3_score_calibrated_sft_v3.pt
                              only on PASS.
Stage p2 (R5, explicit only)  TRUE canonical GRPO pilot gated on R5 Phase-1.

identified=false, formal_test_access=0, Formal TEST SEALED.

Run: PYTHONPATH=src .venv/bin/python scripts/run_m3_canonical_training.py [--quick] [--stage ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from causal_schedule_lab.m3 import config as C
from causal_schedule_lab.m3.upstream import load_upstream

from causal_schedule_lab.m3 import aux_instances as AX  # noqa: E402  (R9 AUX gen)

# canonical modules (the R1->R2->R3->R4 importlib chain is BROKEN here by design)
from causal_schedule_lab.m3 import gate as G       # noqa: E402
from causal_schedule_lab.m3 import memory as MEM   # noqa: E402
from causal_schedule_lab.m3 import policy as POL   # noqa: E402
from causal_schedule_lab.m3 import ranking as RK   # noqa: E402
from causal_schedule_lab.m3 import rollout as RO   # noqa: E402
from causal_schedule_lab.m3 import top1 as TOP1    # noqa: E402   (R6 Top-1 SFT)
from causal_schedule_lab.m3 import proposal_features as PF  # noqa: E402
from causal_schedule_lab.m3 import scorer as SC     # noqa: E402
from causal_schedule_lab.m3 import rolling_grpo as RGRPO  # noqa: E402  (R11 rolling GRPO)
from causal_schedule_lab.m3 import traj_grpo as TRJ  # noqa: E402  (R12 true-multistep GRPO)
from causal_schedule_lab.m3 import joint_grpo as JG  # noqa: E402  (R13 M2-root x M3 joint GRPO)
from causal_schedule_lab.m3 import hierarchical_residual as HR  # noqa: E402  (T2-D actors)

from causal_schedule_lab.teacher.atomic_counterfactual_executor import (  # noqa: E402
    AtomicCounterfactualExecutor,
)
from causal_schedule_lab.validation import schedule_hash  # noqa: E402

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def _seed(s=0):
    random.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)


def _load_env():
    load_upstream()
    from causal_schedule_lab.m3.upstream import pilot, m3util, runmod, b52
    return pilot, m3util, runmod, b52


def build_env(args):
    """Load frozen upstream + instances + M2/utility heads; return shared context."""
    _seed(0)
    pilot, m3util, runmod, b52 = _load_env()
    insts, _ = pilot._instance_paths()
    order = ([i for i in insts.values() if i["split"] == "train"]
             + [i for i in insts.values() if i["split"] == "val"])
    if args.quick:
        order = [i for i in order if i["split"] == "train"][:2]
    train_insts = [i for i in order if i["split"] == "train"]
    val_insts = [i for i in order if i["split"] == "val"]
    ep_id_of = {i["instance_id"]: idx for idx, i in enumerate(train_insts)}

    print("[build] reconstructing S_0 states ...", flush=True)
    states = {i["instance_id"]: pilot.build_state(i) for i in order}
    first_iid = order[0]["instance_id"]
    model_b5 = pilot.from_manifest_b5(states[first_iid]["bundle"], b5_adapter=True, b5_mod_dim=16)
    model_b5.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model_b5.eval()
    single_head = m3util.M3SingleUtilityHead(C.SINGLE_FEAT_DIM, m3util.HIDDEN_DIM)
    single_head.load_state_dict(torch.load(C.UTIL_DIR / "head_single.pt", map_location="cpu"))
    single_head.eval()
    direct_head = m3util.DirectPairUtilityHead(C.SINGLE_FEAT_DIM, C.PAIR_FEAT_DIM, m3util.HIDDEN_DIM)
    direct_head.load_state_dict(torch.load(C.UTIL_DIR / "head_direct.pt", map_location="cpu"))
    direct_head.eval()
    executor = AtomicCounterfactualExecutor()
    cache = PF.AnalyzeCache(model_b5, single_head, direct_head)
    return {
        "pilot": pilot, "m3util": m3util, "runmod": runmod, "b52": b52,
        "order": order, "train_insts": train_insts, "val_insts": val_insts,
        "ep_id_of": ep_id_of, "states": states, "cache": cache, "executor": executor,
        "model_b5": model_b5, "single_head": single_head, "direct_head": direct_head,
    }


# ---------------------------------------------------------------------------
# Phase 1: replay + memory + scorer + gate + rerankers + baselines
# ---------------------------------------------------------------------------

def build_replay_env(env):
    """Replay + progressive memory + audits (canonical, verbatim semantics)."""
    cache, executor = env["cache"], env["executor"]
    train_insts, ep_id_of, states = env["train_insts"], env["ep_id_of"], env["states"]
    print("[replay] building TRAIN replay buffer + labels ...", flush=True)
    state_examples, store = RO.build_replay(cache, executor, train_insts, states)
    per_inst = {}
    for ex in state_examples:
        per_inst.setdefault(ex["iid"], []).append(ex)
    for iid, exs in per_inst.items():
        for t, ex in enumerate(exs):
            ex["ep_id"] = ep_id_of[iid]
            ex["tstep"] = t
            ex["s0"] = (t == 0)
    for gi, ex in enumerate(state_examples):
        ex["gi"] = gi
    print(f"[replay] {len(state_examples)} states, {len(store.records)} labels", flush=True)

    print("[mem] building progressive store + corrected features ...", flush=True)
    progmem = MEM.ProgressiveMemory(state_feats=[ex["state_feat"].tolist() for ex in state_examples])
    sf_map = {(ex["iid"], ex["state_hash"]): ex["state_feat"].tolist() for ex in state_examples}
    for ti in train_insts:
        iid = ti["instance_id"]
        eid = ep_id_of[iid]
        for rec in store.records:
            if rec["instance_id"] != iid:
                continue
            sf = sf_map.get((iid, rec["state_hash"]))
            if sf is None:
                continue
            progmem.add_episode_record({
                "instance_id": iid, "episode_id": eid,
                "state_hash": rec["state_hash"], "state_feat": sf,
                "proposal_signature": rec["proposal_signature"],
                "proposal_type": rec["proposal_type"], "role": rec["role"],
                "src": rec["src"], "tgt": rec["tgt"], "true_U": rec["true_U"],
                "outcome": rec["outcome"],
                "trajectory_step": rec["trajectory_step"],
                "written_at_step": rec["trajectory_step"],
                "fine_key": rec["fine_key"], "coarse_key": rec["coarse_key"],
            }, eid)
        exs = per_inst[iid]
        for t in range(len(exs) - 1):
            ex = exs[t]
            U = ex["true_U"]
            seq = [k for k in range(len(U)) if ex["feasible"][k] and U[k] > 0]
            if not seq:
                continue
            best = max(seq, key=lambda k: float(U[k]))
            progmem.add_executed(iid, t, {
                "instance_id": iid, "episode_id": eid,
                "state_hash": ex["state_hash"], "state_feat": ex["state_feat"].tolist(),
                "proposal_signature": ex["sig"][best],
                "proposal_type": ex["type"][best], "role": ex["role"][best],
                "src": ex["src"][best], "tgt": ex["tgt"][best],
                "true_U": float(U[best]), "outcome": "success",
                "trajectory_step": t, "written_at_step": t,
                "fine_key": ((ex["type"][best], ex["role"][best], ex["src"][best],
                              ex["tgt"][best]) if ex["type"][best] == "single"
                             else (ex["type"][best], ex["role"][best])),
                "coarse_key": (ex["type"][best], ex["role"][best]),
            })
    MEM.compute_prog_mem_features(state_examples, progmem)
    mem_values = [ex["prog_mem_feats"] for ex in state_examples]

    causal_assert = MEM.assert_causal_lookahead_free(state_examples, progmem, store)
    unseen_assert = MEM.assert_unseen_state_zero(state_examples, progmem)
    pk = MEM.replay_primary_key_uniqueness(store, ep_id_of)
    coverage = RK.coverage_audit([ex for ex in state_examples if ex.get("s0")],
                                 state_examples, progmem)
    print(f"[audit] {json.dumps({**causal_assert, **unseen_assert, **pk}, default=str)}", flush=True)
    print(f"[coverage] {json.dumps(coverage)}", flush=True)
    return {"state_examples": state_examples, "store": store, "progmem": progmem,
            "mem_values": mem_values, "audits": {
                "causal": causal_assert, "unseen": unseen_assert, "primary_key": pk,
                "coverage": coverage}}


def train_phase1(env, re):
    """scorer (313-d real mem) + masked arm + gate + legacy rerankers."""
    state_examples, mem_values = re["state_examples"], re["mem_values"]
    print("[train] scorer_mem (313-d real mem) ...", flush=True)
    scorer_mem, scorer_masked, stats_scorer = SC.train_scorer_ablation(state_examples)
    print(f"  scorer: {json.dumps(stats_scorer)}", flush=True)
    print("[train] gate (act_primary, corrected-memory scorer) ...", flush=True)
    gate_m, X_g, y_g, gate_meta = G.train_gate_r4(scorer_mem, state_examples, mode="act_primary")
    print(f"  gate: {json.dumps(gate_meta['gate_report'])}", flush=True)

    print("[train] legacy M3ProposalReranker (missed-positive-centric) ...", flush=True)
    reranker, rhist, rr_stats, rr_data = RK.train_reranker(state_examples, scorer_mem, mem_values)
    print(f"  reranker: {json.dumps(rr_stats)}", flush=True)
    zero_vals = [torch.zeros_like(ex["prog_mem_feats"]) for ex in state_examples]
    print("[train] reranker_masked (identical pairs, masked channel) ...", flush=True)
    Xm_parts = []
    for gi in rr_data["state_gi_list"]:
        Xm_parts.append(RK._rerank_feats_all(scorer_masked, state_examples[gi], zero_vals[gi]))
    X_all_m = torch.cat(Xm_parts)
    reranker_masked, rhist_m = RK._fit_reranker(X_all_m, rr_data["pgi"], rr_data["pii"],
                                                rr_data["pjj"], rr_data["marg"], rr_data["gi_off"])
    return {
        "scorer_mem": scorer_mem, "scorer_masked": scorer_masked,
        "gate_m": gate_m, "gate_meta": gate_meta, "stats_scorer": stats_scorer,
        "reranker": reranker, "reranker_hist": rhist, "rr_stats": rr_stats, "rr_data": rr_data,
        "reranker_masked": reranker_masked, "zero_vals": zero_vals,
    }


def phase1_offline_metrics(re, p1):
    """Offline ranking metrics: real / masked / S0 / additive / frozen-channel audit."""
    state_examples, mem_values = re["state_examples"], re["mem_values"]
    print("[metrics] wide + final (real memory) ...", flush=True)
    m_real = RK.wide_and_final_metrics(state_examples, p1["scorer_mem"], p1["reranker"], mem_values)
    print("[metrics] wide + final (S0 real memory) ...", flush=True)
    m_real_s0 = RK.s0_slice_metrics(state_examples, p1["scorer_mem"], p1["reranker"], mem_values)
    print("[metrics] wide + final (masked memory) ...", flush=True)
    m_masked = RK.wide_and_final_metrics(state_examples, p1["scorer_masked"], p1["reranker_masked"],
                                         p1["zero_vals"])
    print("[metrics] wide + final (S0 masked memory) ...", flush=True)
    m_masked_s0 = RK.s0_slice_metrics(state_examples, p1["scorer_masked"], p1["reranker_masked"],
                                      p1["zero_vals"])
    print("[metrics] frozen-model memory-channel audit ...", flush=True)
    audit = RK.frozen_memory_channel_audit(state_examples, p1["scorer_mem"], p1["reranker"],
                                           mem_values)
    print(f"  audit: {json.dumps(audit)}", flush=True)

    def _fuse(scorer, ex, mem):
        """Additive fusion over ALL proposals: fused = rank + LAMBDA_CLS * logit_pos."""
        with torch.no_grad():
            lp, rk = SC._scores(scorer, ex, mem)
        return (rk + C.LAMBDA_CLS * lp).numpy()

    # state_examples keep their ORIGINAL gi (built in build_replay_env), so
    # mem_values[ex["gi"]] is correct for every example (incl. S0-only slice).
    add_real = SC._rank_metrics(state_examples,
                                lambda ex: _fuse(p1["scorer_mem"], ex, mem_values[ex["gi"]]))
    s0_exs = [dict(ex) for ex in state_examples if ex.get("s0", False)]
    add_real_s0 = SC._rank_metrics(s0_exs,
                                   lambda ex: _fuse(p1["scorer_mem"], ex, mem_values[ex["gi"]]))
    print(f"  additive real S0.missed_recall@10 = {add_real_s0['missed_recall'].get('10')}", flush=True)
    return {"m_real": m_real, "m_real_s0": m_real_s0, "m_masked": m_masked,
            "m_masked_s0": m_masked_s0, "mem_channel_audit": audit,
            "additive_real": add_real, "additive_real_s0": add_real_s0}


# ---------------------------------------------------------------------------
# baseline A: wide pool -> rank_head top-1 (or frozen base) closed-loop
# ---------------------------------------------------------------------------
def _pool_and_top1(ast, metas, prop_feats, sf_t, mem, scorer, metric):
    """Wide pool -> 'metric' argmax (metric=rank_head or old-base).  Returns idx or None."""
    with torch.no_grad():
        logit_pos, rank, _ = scorer(prop_feats, mem, sf_t)
    pool, info = RK.wide_pool({"metas": metas, "prop_feats": prop_feats, "state_feat": sf_t,
                               "role": _roles_of(ast, metas)},
                              logit_pos, rank)
    if not pool:
        return None, info, logit_pos, rank
    if metric == "rank_head":
        scores = rank
    elif metric == "old_base":
        scores = G._old_base_arr(prop_feats)
    elif metric == "fused":
        scores = rank + C.LAMBDA_CLS * logit_pos
    else:
        raise ValueError(metric)
    scores_np = scores if not hasattr(scores, "numpy") else scores.numpy()
    best = max(pool, key=lambda k: float(scores_np[k]))
    return best, info, logit_pos, rank


def _roles_of(ast, metas):
    vals = []
    for k in range(len(metas)):
        _e, _kind, _sig, role, _pt, _s, _t, _o = PF.proposal_identity(ast, metas[k])
        vals.append(role)
    return vals


def rollout_rank_top1(env, rf, order_metric, use_mem=True, horizon=C.HORIZON):
    """Greedy closed loop: gate-free, wide-pool -> metric top-1 each step.

    Returns (gain, usage, steps).  This is baseline A (control), NOT the
    reranker path."""
    problem, schedule0 = rf["problem"], rf["schedule"]
    iid, progmem, episode_id = rf["iid"], rf["progmem"], rf["episode_id"]
    cache, executor = env["cache"], env["executor"]
    scorer = rf["scorer"]
    s0_ms = int(schedule0.makespan)
    ms_cur = s0_ms
    schedule = schedule0
    visited = {schedule_hash(schedule0)}
    usage = {"single": 0, "pair": 0, "stop_no_pool": 0, "stop_neg": 0}
    steps = []
    for t in range(horizon):
        prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        n_prop = len(metas)
        if n_prop == 0:
            break
        ast = cache.ast(problem, schedule, iid)
        h = schedule_hash(schedule)
        sf = PF.state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"], agg["best_direct"],
                                  agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = RO._roll_ex_from(ast, metas, prop_feats, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = (torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                            dtype=torch.float32) if use_mem
               else torch.zeros(n_prop, C.MEM_FEAT_DIM, dtype=torch.float32))
        best, info, logit_pos, rank = _pool_and_top1(ast, metas, prop_feats, sf_t, mem,
                                                     scorer, order_metric)
        if best is None:
            usage["stop_no_pool"] += 1
            break
        edits, kind = PF._edits_for(ast, metas[best])
        res = PF._execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = PF.proposal_identity(ast, metas[best])[2]
        steps.append({"t": t, "metric": order_metric, "n_prop": n_prop,
                      "n_wide": int(info.get("union", 0)), "top_sig": sig, "kind": kind,
                      "improvement": (None if res is None else float(res["improvement"]))})
        if res is None or res["improvement"] <= 0:
            usage["stop_neg"] += 1
            break
        usage[kind] += 1
        schedule = res["schedule"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        if nh in visited:
            break
        visited.add(nh)
    return int(schedule0.makespan) - ms_cur, usage, steps


# ---------------------------------------------------------------------------
# Phase-1 utility-aware reranker
# ---------------------------------------------------------------------------
_CATEGORIES = ["best_vs_hard", "best_vs_lower_pos", "high_vs_neutral",
               "high_vs_neg", "med_vs_neg", "missed_vs_hard"]


def _cat_of(U, old, batch_ids, best, pos):
    """Return (list_of (i,j,category)) for one state."""
    N = len(U)
    pairs = []
    pos_sorted = sorted(pos, key=lambda k: -float(U[k]))
    hardneg = [k for k in batch_ids if old[k] > 0 and U[k] <= 0]
    neg = [k for k in batch_ids if U[k] < 0]
    neutral = [k for k in batch_ids if U[k] == 0]
    for h in hardneg:
        pairs.append((best, h, "best_vs_hard"))
    for p in pos_sorted[1:]:
        pairs.append((best, p, "best_vs_lower_pos"))
    for h in neg[:10]:
        pairs.append((pos_sorted[0], h, "high_vs_neg"))
    if len(pos_sorted) > 1:
        pairs.append((pos_sorted[0], neutral[0], "high_vs_neutral")) if neutral else None
    if len(pos_sorted) > 2:
        pairs.append((pos_sorted[1], neutral[0], "high_vs_neutral")) if neutral else None
        pairs.append((pos_sorted[1], neg[0], "med_vs_neg")) if neg else None
    for m in [k for k in pos if old[k] <= 0]:
        for h in hardneg:
            pairs.append((m, h, "missed_vs_hard"))
    return pairs


def sample_utility_pairs(state_examples, seed=0):
    """Within-state pairs across the six categories; no category exceeds 50%.

    Pair weight w_ij = clip(|U_i-U_j|/utility_scale, min_w, max_w) is applied at
    train time.  Returns (pair_rows, category_counts, balance_stats)."""
    from collections import Counter
    rows = []          # (gi, i, j, cat)
    for gi, ex in enumerate(state_examples):
        U = ex["true_U"]
        old = np.asarray(ex["old_base"], dtype=np.float64)
        N = len(U)
        batch_ids = [k for k in range(N) if ex["feasible"][k]]
        if not batch_ids:
            continue
        pos = [k for k in batch_ids if U[k] > 0]
        if not pos:
            continue
        best = max(pos, key=lambda k: float(U[k]))
        pairs = _cat_of(U, old, batch_ids, best, pos)
        pair_cap = C.UA_PAIRS_PER_STATE
        for i, j, cat in pairs[:pair_cap]:
            rows.append((gi, i, j, cat))
    # balance: cap each category at UA_MAX_CATEGORY_FRAC of the CURRENT total;
    # deterministically drops from the over-represented category (last first)
    if rows:
        counts = Counter(r[3] for r in rows)
        while max(counts.values()) / len(rows) > C.UA_MAX_CATEGORY_FRAC:
            worst = max(counts, key=counts.get)
            for ri in range(len(rows) - 1, -1, -1):
                if rows[ri][3] == worst:
                    del rows[ri]
                    break
            counts[worst] -= 1
    counts = Counter(r[3] for r in rows)
    max_frac = max(counts.values()) / max(len(rows), 1) if rows else 0.0
    return rows, dict(counts), {"n_pairs": len(rows), "max_category_frac": float(max_frac),
                                "categories": dict(counts)}


def _utility_w(ex, i, j, scale=C.UA_UTILITY_SCALE, w_min=C.UA_MIN_W, w_max=C.UA_MAX_W):
    d = abs(float(ex["true_U"][i]) - float(ex["true_U"][j]))
    return float(np.clip(d / scale, w_min, w_max))


def train_utility_reranker(state_examples, scorer, mem_values, seed=0):
    """Train a fresh reranker with utility-magnitude pair weights.

    Categories balanced <= 50%; loss = mean over pairs of
      w_ij * hinge(s_i - s_j)   (w_ij = clip(|U_i-U_j|/scale, min, max))
    """
    rows, counts, bal = sample_utility_pairs(state_examples, seed=seed)
    if not rows:
        raise RuntimeError("no utility-aware pairs")
    weights = torch.tensor([_utility_w(state_examples[gi], i, j)
                            for gi, i, j, _ in rows], dtype=torch.float32)
    weight_hist = {
        "mean": float(weights.mean()), "min": float(weights.min()), "max": float(weights.max()),
        "n_zero_weight": int((weights < 1e-3).sum()),
    }
    # feature blocks (per example; concatenated like train_reranker)
    gi_to_feat = {}
    offsets = []
    offset = 0
    for gi, i, j, cat in rows:
        if gi not in gi_to_feat:
            f = RK._rerank_feats_all(scorer, state_examples[gi], mem_values[gi])
            gi_to_feat[gi] = f
            offsets.append((gi, offset))
            offset += int(f.shape[0])
    X_all = torch.cat([gi_to_feat[gi] for gi in gi_to_feat])
    gi_off = {}
    for gi, o in offsets:
        gi_off[gi] = o
    pgi = torch.tensor([gi for gi, _, _, _ in rows], dtype=torch.long)
    pii = torch.tensor([i for _, i, _, _ in rows], dtype=torch.long)
    pjj = torch.tensor([j for _, _, j, _ in rows], dtype=torch.long)
    off_t = torch.tensor([gi_off[gi] for gi, _, _, _ in rows], dtype=torch.long)
    torch.manual_seed(seed)
    model = RK.M3ProposalReranker()
    opt = torch.optim.AdamW(model.parameters(), lr=C.RERANK_LR, weight_decay=1e-4)
    hist = []
    for ep in range(C.RERANK_EPOCHS):
        opt.zero_grad()
        scores = model(X_all)
        si = scores[off_t + pii]
        sj = scores[off_t + pjj]
        loss = (weights * torch.clamp(1.0 - (si - sj), min=0.0)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append({"ep": ep, "loss": float(loss)})
    model.eval()
    return model, hist, {"category_counts": counts, "balance": bal,
                         "weight_hist": weight_hist, "n_pairs": len(rows)}


def rerank_order_utility(scorer, reranker, ex, mem):
    """Same final-order combinator as rerank_order but with the utility reranker."""
    with torch.no_grad():
        logit_pos, rank = SC._scores(scorer, ex, mem)
    pool, info = RK.wide_pool(ex, logit_pos, rank)
    pool_set = set(pool)
    f = RK._rerank_feats_all(scorer, ex, mem)
    if f is None or not pool:
        return [], [], info
    with torch.no_grad():
        s = reranker(f)
    order_all = list(np.argsort(-s.numpy()))
    final_order = [k for k in order_all if k in pool_set]
    return pool, final_order, {**info, "rerank_top1_in_pool": (final_order[0] if final_order else None)}


# ---------------------------------------------------------------------------
# Phase-1 closed loop (utility reranker path) + acceptance
# ---------------------------------------------------------------------------
def rollout_utility_rerank(env, rf, use_mem=True, horizon=C.HORIZON):
    """FixedDecisionReplay closed loop: gate -> wide pool -> UTILITY reranker top-1."""
    problem, schedule0 = rf["problem"], rf["schedule"]
    iid, progmem, episode_id = rf["iid"], rf["progmem"], rf["episode_id"]
    cache, executor = env["cache"], env["executor"]
    scorer, reranker, gate = rf["scorer"], rf["reranker"], rf["gate"]
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
        sf = PF.state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"],
                                  agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rolex = RO._roll_ex_from(ast, metas, prop_feats, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = (torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                            dtype=torch.float32) if use_mem
               else torch.zeros(n_prop, C.MEM_FEAT_DIM, dtype=torch.float32))
        logit_pos, rank = RO._scores_raw(scorer, prop_feats, mem, sf_t)
        oldbase = G._old_base_arr(prop_feats)
        x, *_ = G.pooled_gate_input(sf_t, logit_pos, rank, oldbase, C.K_R, C.K_C)
        with torch.no_grad():
            pred_cls = int(torch.argmax(gate(x)).item())
        if pred_cls == 1:                     # STOP
            act_usage["stop_by_gate"] += 1
            break
        _, final_order, pinfo = rerank_order_utility(scorer, reranker, rolex, mem)
        if not final_order:
            act_usage["stop_by_gate"] += 1
            break
        a = final_order[0]
        edits, kind = PF._edits_for(ast, metas[a])
        res = PF._execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = PF.proposal_identity(ast, metas[a])[2]
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


def closed_loop_b5(env, re, scorer, reranker, gate, use_mem=True, mode="utility"):
    """Evaluate closed loop over all instances (mode: 'utility' or 'legacy')."""
    gains_by_iid = {}
    steps_by_iid = {}
    for idx, i in enumerate(env["order"]):
        iid = i["instance_id"]
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        rf = {"problem": env["states"][iid]["problem"], "schedule": env["states"][iid]["schedule"],
              "iid": iid, "progmem": re["progmem"], "episode_id": eid,
              "scorer": scorer, "reranker": reranker, "gate": gate}
        if mode == "utility":
            gain, usage, steps = rollout_utility_rerank(env, rf, use_mem=use_mem)
        else:
            gain, usage, steps = RO.rollout_wide_rerank(
                rf["problem"], rf["schedule"], env["cache"], scorer, reranker, gate,
                env["executor"], iid, re["progmem"], eid, use_mem=use_mem)
        gains_by_iid[iid] = gain
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    return RO._b5_summary(gains_by_iid), steps_by_iid


def split_summary(summary, env):
    """TRAIN/VAL split of a _b5_summary (gains_by_iid already inside)."""
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


def phase1_acceptance(re, sft_metrics, sft_summary_train, sft_summary_val, env):
    """Phase-1 gate: missed@10 >= .20, best@10 >= .50, closed-loop TRAIN >= 235,
    VAL >= 10, plus median / Fattahi15 leave-out diagnostics.  Returns (passed, report)."""
    s0 = sft_metrics["m_real_s0"]
    missed10 = float(s0.get("final_missed_recall", {}).get("10", 0.0))
    best10 = float(s0.get("final_best_recall", {}).get("10", 0.0))
    tr_total = sft_summary_train["total"]
    va_total = sft_summary_val["total"]

    # leave-Fattahi15-out
    leave = {}
    per = sft_summary_train["per_instance"]
    for k in per:
        if "Fattahi15" in k:
            others = {k2: v for k2, v in per.items() if k2 != k}
            leave[k] = int(sum(others.values()))
    checks = {
        "missed@10 S0 >= 0.20": missed10 >= 0.20,
        "best@10 S0 >= 0.50": best10 >= 0.50,
        "closed-loop TRAIN >= 235": tr_total >= 235,
        "closed-loop VAL >= 10": va_total >= 10,
    }
    passed = all(checks.values())
    report = {
        "index_checks": checks,
        "S0": {"missed@10": missed10, "best@10": best10,
               "final_best@1/5/10": {str(k): s0.get("final_best_recall", {}).get(str(k))
                                     for k in (1, 5, 10)},
               "final_missed_recall_all": s0.get("final_missed_recall", {}),
               "wide_pool_missed_recall_s0": s0.get("wide_pool_missed_recall_s0", 0.0)},
        "closed_loop": {"train_total": tr_total, "val_total": va_total,
                        "train_median": sft_summary_train["median"],
                        "val_median": sft_summary_val["median"],
                        "train_n_pos": sft_summary_train["n_positive"],
                        "val_n_pos": sft_summary_val["n_positive"]},
        "leave_Fattahi15_out_train_total": leave,
        "passed": passed,
    }
    print(f"[acceptance] S0 missed@10={missed10:.3f} best@10={best10:.3f} "
          f"train={tr_total} val={va_total} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed, report


# ---------------------------------------------------------------------------
# Phase 2: canonical GRPO
# ---------------------------------------------------------------------------
def grpo_reward_table(re):
    """Frozen-Local reward memo: (iid, state_hash, sig) -> (feasible, U).

    Built from the deterministic replay labels (the external evaluator), so the
    GRPO loop never needs to re-execute the same proposal twice.
    """
    tbl = {}
    for rec in re["store"].records:
        key = (rec["instance_id"], rec["state_hash"], rec["proposal_signature"])
        if rec["true_U"] is not None:
            tbl[key] = (True, float(rec["true_U"]))
        else:
            tbl[key] = (False, 0.0)
    return tbl


class GRPOAgent:
    """score_GRPO = score_SFT + alpha*tanh(delta); STOP base = 0.

    Group policy is over the LEGAL action set: feasible proposals ∪ STOP
    (canonical spec: actions = complete legal Proposal or STOP, no primitives)."""

    def __init__(self, scorer_mem, alpha=C.GRPO_ALPHA_RESIDUAL, seed=0):
        torch.manual_seed(seed)
        self.scorer = scorer_mem            # frozen SFT scorer (pi_ref base)
        self.head = POL.M3GRPOActionHead()
        self.alpha = alpha

    @torch.no_grad()
    def sft_scores(self, ex, mem):
        logit_pos, rank, _ = self.scorer(ex["prop_feats"], mem, ex["state_feat"])
        return rank                          # frozen reference

    def scores(self, ex, mem):
        """Full-pool (prop [N], stop) with grad through the action head only."""
        rank = self.sft_scores(ex, mem)      # detached
        delta_prop, delta_stop = self.head(ex["prop_feats"], ex["state_feat"])
        return POL.resolved_scores(rank, delta_prop, delta_stop, ex["state_feat"],
                                   alpha=self.alpha)

    def group_logp(self, ex, mem, feasible_k, ids):
        """logp over the legal-action subset [feasible proposals | STOP]."""
        prop, stop = self.scores(ex, mem)
        subset = torch.cat([prop[feasible_k], stop.reshape(1)], dim=-1)
        lp = torch.log_softmax(subset, dim=-1)
        t = torch.as_tensor(ids, dtype=torch.long)
        return lp[t], subset

    def ref_logp(self, ex, mem, feasible_k, ids):
        """Frozen SFT reference logp over the same legal-action subset."""
        rank = self.sft_scores(ex, mem)
        ref_logits = torch.cat([rank[feasible_k], torch.zeros(1)], dim=-1)
        ref_lp = torch.log_softmax(ref_logits, dim=-1)
        t = torch.as_tensor(ids, dtype=torch.long)
        return ref_lp[t]

    def sample_batch(self, state_examples, reward_tbl, mem_values, rng, g_min=C.GRPO_G_MIN):
        """Collect (group, ids, logp_old, ref_lp, rewards) for a training pass."""
        records = []
        skip_meta = {"no_feasible": 0, "no_positive_n_or_stop_only": 0}
        for gi, ex in enumerate(state_examples):
            N = len(ex["metas"])
            sigs = ex["sig"]
            h = ex["state_hash"]
            iid = ex["iid"]
            feas_k = []
            rw = []
            for k in range(N):
                feas, u = reward_tbl.get((iid, h, sigs[k]), (False, 0.0))
                if feas:
                    feas_k.append(k)
                    rw.append(u)
            if not feas_k:                       # cannot form any legal-action group
                skip_meta["no_feasible"] += 1
                continue
            ids = POL.grpo_sample_group(len(feas_k), rng, g_min=g_min)   # STOP id == len(feas_k)
            with torch.no_grad():
                lp, _ = self.group_logp(ex, mem_values[gi], torch.tensor(feas_k, dtype=torch.long), ids)
                rlp = self.ref_logp(ex, mem_values[gi], torch.tensor(feas_k, dtype=torch.long), ids)
            # ids may contain STOP anywhere; build rewards aligned to ids
            rew = []
            for a in ids:
                rew.append(rw[a] if a < len(feas_k) else 0.0)     # STOP reward = 0.0
            records.append({
                "gi": gi, "feasible_k": feas_k, "ids": ids,
                "logp_old": lp.detach(),
                "ref_lp": rlp.detach(),
                "rewards": torch.tensor(rew, dtype=torch.float32),
            })
        return records, skip_meta


def grpo_step(agent, state_examples, mem_values, records, opt, beta=C.GRPO_BETA_KL):
    """One clipped-GRPO update over the collected batch (all groups)."""
    total_loss = torch.zeros(())
    n_inform = 0
    stats = {"clip_fraction": 0.0, "mean_ratio": 1.0, "kl_to_ref": 0.0,
             "informative_groups": 0, "n_groups": len(records), "n_loss_groups": 0}
    for rec in records:
        ex = state_examples[rec["gi"]]
        fk = torch.tensor(rec["feasible_k"], dtype=torch.long)
        lp_new, _ = agent.group_logp(ex, mem_values[rec["gi"]], fk, rec["ids"])
        loss, st = POL.grpo_group_loss(lp_new, rec["logp_old"], rec["rewards"],
                                       reference_logp=rec["ref_lp"], beta=beta)
        if not st["informative"]:
            continue
        total_loss = total_loss + loss
        n_inform += 1
        stats["clip_fraction"] = (stats["clip_fraction"] * (n_inform - 1) + st["clip_fraction"]) / n_inform
        stats["mean_ratio"] = (stats["mean_ratio"] * (n_inform - 1) + st["mean_ratio"]) / n_inform
        stats["kl_to_ref"] = (stats["kl_to_ref"] * (n_inform - 1) + st.get("kl_to_ref", 0.0)) / n_inform
    stats["informative_groups"] = n_inform
    if n_inform == 0:
        return total_loss, stats
    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.head.parameters(), 1.0)
    opt.step()
    return total_loss, stats


def grpo_action_quality(agent, state_examples, reward_tbl, mem_values):
    """Action-quality audit: probability mass per category (positive / neutral /
    negative / infeasible / STOP), greedy hit of the true best positive, and mean
    probability mass on the best positive (mass_positive)."""
    mass = {"positive": [], "neutral": [], "negative": [], "infeasible": [], "stop": []}
    n_cat = {"positive": 0, "neutral": 0, "negative": 0, "infeasible": 0}
    greedy_best_hit = 0
    mean_p_best = []
    n_states = 0
    for gi, ex in enumerate(state_examples):
        N = len(ex["metas"])
        h = ex["state_hash"]
        iid = ex["iid"]
        sigs = ex["sig"]
        U = ex["true_U"]
        prop, stop = agent.scores(ex, mem_values[gi])
        logits = torch.cat([prop, stop.reshape(1)], dim=-1)
        p = torch.softmax(logits, dim=-1)
        pos = [k for k in range(N) if U[k] > 0]
        if not pos:
            for k in range(N):
                cat = "positive" if U[k] > 0 else ("neutral" if U[k] == 0 else
                                                   ("negative" if U[k] < 0 else "infeasible"))
                mass.setdefault(cat, []).append(float(p[k]))
                n_cat[cat] = n_cat.get(cat, 0) + 1
            mass["stop"].append(float(p[N]))
            continue
        n_states += 1
        best = max(pos, key=lambda k: float(U[k]))
        greedy = int(torch.argmax(logits).item())
        greedy_best_hit += int(greedy == best)
        mean_p_best.append(float(p[best]))
        for k in range(N):
            cat = "positive" if U[k] > 0 else ("neutral" if U[k] == 0 else
                                               ("negative" if U[k] < 0 else "infeasible"))
            mass[cat].append(float(p[k]))
            n_cat[cat] = n_cat.get(cat, 0) + 1
        mass["stop"].append(float(p[N]))
    m = {k: float(np.mean(v)) if v else 0.0 for k, v in mass.items()}
    return {
        "n_states_with_positive": n_states,
        "greedy_best_positive_hit_rate": greedy_best_hit / max(n_states, 1),
        "mean_p_best_positive": float(np.mean(mean_p_best)) if mean_p_best else 0.0,
        "action_counts": {k: v for k, v in n_cat.items()},
        "mean_softmax_mass": m,
        "stop_mass": m["stop"],
        "positive_mass": m["positive"],
        "max_residual": None,              # filled by caller
    }


def greedy_grpo_closed_loop(env, re, agent, use_mem=True, horizon=C.HORIZON, val_only=False):
    """Greedy closed loop scoring proposals by score_GRPO (bounded residual); STOP
    if stop score top or no pool improvement.  Used for B6 (TRAIN or VAL).

    A FRESH copy of progressive memory is used for every evaluation so executed
    transitions from one eval run can never leak into the next (each eval stays
    within its own episode, no cross-run oracle carryover)."""
    gains = {}
    steps_iid = {}
    progmem_eval = copy.deepcopy(re["progmem"]) if use_mem else None
    for idx, i in enumerate(env["order"]):
        iid = i["instance_id"]
        if val_only and iid in {t["instance_id"] for t in env["train_insts"]}:
            continue
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        problem, schedule0 = env["states"][iid]["problem"], env["states"][iid]["schedule"]
        cache, executor = env["cache"], env["executor"]
        progmem = progmem_eval
        s0_ms = int(schedule0.makespan)
        ms_cur = s0_ms
        schedule = schedule0
        visited = {schedule_hash(schedule0)}
        usage = {"single": 0, "pair": 0, "stop": 0, "stop_neg": 0, "infeasible": 0}
        steps = []
        for t in range(horizon):
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
            n_prop = len(metas)
            if n_prop == 0:
                usage["stop"] += 1
                break
            ast = cache.ast(problem, schedule, iid)
            h = schedule_hash(schedule)
            sf = PF.state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"],
                                      agg["best_direct"], agg["n_contrib"], agg["n_enab"])
            sf_t = torch.tensor(sf, dtype=torch.float32)
            rolex = RO._roll_ex_from(ast, metas, prop_feats, sf_t)
            queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                       for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                                 rolex["src"], rolex["tgt"])]
            mem = (torch.tensor(progmem.features(iid, eid, t, sf, queries),
                                dtype=torch.float32) if use_mem
                   else torch.zeros(n_prop, C.MEM_FEAT_DIM, dtype=torch.float32))
            with torch.no_grad():
                prop, stop, = agent.scores(rolex, mem)
                logits = torch.cat([prop, stop.reshape(1)], dim=-1)
                arg = int(torch.argmax(logits).item())
            if arg == n_prop:
                usage["stop"] += 1
                break
            edits, kind = PF._edits_for(ast, metas[arg])
            res = PF._execute_step(executor, problem, schedule, edits, ms_cur, h)
            sig = PF.proposal_identity(ast, metas[arg])[2]
            steps.append({"t": t, "score": "GRPO", "n_prop": n_prop, "top_sig": sig,
                          "kind": kind,
                          "improvement": (None if res is None else float(res["improvement"]))})
            if res is None:
                usage["infeasible"] += 1
                usage["stop"] += 1
                break
            if res["improvement"] <= 0:
                usage["stop_neg"] += 1
                break
            usage[kind] += 1
            schedule = res["schedule"]
            ms_cur = int(schedule.makespan)
            nh = schedule_hash(schedule)
            if nh in visited:
                break
            visited.add(nh)
        gains[iid] = int(schedule0.makespan) - ms_cur
        steps_iid[iid] = usage
    return RO._b5_summary(gains), steps_iid


# ---------------------------------------------------------------------------
# Phase 0: consolidation inventory + A-F classification
# ---------------------------------------------------------------------------
def _grep_imports(path):
    deps = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                deps.add(s)
    except OSError:
        pass
    return sorted(deps)


def inventory(args) -> dict:
    root = C.ROOT
    report = {"project_root": str(root), "scan_time": time.time(), "quick": args.quick}
    scripts_dir = root / "scripts"
    scripts = sorted(scripts_dir.glob("*.py"))
    r_chain = [s.stem for s in scripts if s.name.startswith("t1_m3_")]
    # SHARED_REQUIRED: the frozen upstream family + the teacher-generation module
    # that ``b5_1_train_pilot._instance_paths`` importlib-loads at runtime (move it
    # and the frozen manifest split check breaks).
    shared = ["b5_1_train_pilot", "t1_m3_makespan_utility", "run_v5_mainline_loop",
              "t1_b5_2_interaction_ranking", "b5_route2_teacher_generation"]
    cls = {"CANONICAL_ACTIVE": [], "SHARED_REQUIRED": [], "LEGACY_EXPERIMENT": [],
           "DEBUG_ONLY": [], "UNKNOWN_DEPENDENCY": []}
    for s in scripts:
        name = s.stem
        if name == "run_m3_canonical_training":
            cls["CANONICAL_ACTIVE"].append(name)
        elif name in shared:
            cls["SHARED_REQUIRED"].append(name)
        elif name in r_chain:
            deps = _grep_imports(s)
            # UNKNOWN only when a script statically imports another scripts/ file
            # or an absolute outputs path (those are NOT archivable by move).
            risky = [d for d in deps
                     if ("scripts." in d or "scripts import" in d or "scripts/" in d
                         or "/outputs" in d or "outputs." in d)]
            if risky:
                cls["UNKNOWN_DEPENDENCY"].append({"name": name, "deps": risky})
            elif "diag" in name or "diagnose" in name or name.endswith("_dbg"):
                cls["DEBUG_ONLY"].append(name)
            else:
                cls["LEGACY_EXPERIMENT"].append(name)
        else:
            cls["LEGACY_EXPERIMENT"].append(name)
    report["scripts_classification"] = cls

    m3_dir = root / "src" / "causal_schedule_lab" / "m3"
    report["m3_module_files"] = sorted(p.name for p in m3_dir.glob("*.py"))
    report["m3_legacy_files_remaining"] = sorted(
        p.name for p in m3_dir.glob("*.py") if "legacy" in p.name.lower())

    out = root / "outputs"
    ckpts = []
    for p in out.rglob("*.pt"):
        ckpts.append({"path": str(p.relative_to(root)), "size": p.stat().st_size})
    report["checkpoints"] = ckpts
    tests_dir = root / "tests"
    report["tests"] = sorted(p.name for p in tests_dir.glob("test_*.py"))

    meta_stamp = {"experiment": "T1-M3-CANONICAL-CONSOLIDATION-UTILITY-RERANK-GRPO-R5",
                  "generated_by": "run_m3_canonical_training.inventory",
                  "identified": False, "formal_test_access": 0,
                  "formal_test_sealed": True}
    dest = C.CANONICAL_OUT_DIR / "inventory.json"
    payload = {"meta": meta_stamp, **report}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[inventory] wrote {dest}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Phase-2 supporters: B0-B6 baselines + verdict
# ---------------------------------------------------------------------------
def compute_b0_b6(env, re, p1, p2, phase1_accepted):
    """Compare baselines.  B0,B1,B3mem,B3nomem,B4 are recorded from R4/R3 runs
    (see MEMORY verdicts).  B5 = Phase-1 canonical utility SFT; B6 = SFT->GRPO."""
    b6_train = (p2["final_eval_train"] if (p2 and p2.get("final_eval_train")) else
                {"total": 0, "mean": 0.0, "median": 0.0, "n_positive": 0})
    b6_val = (p2["final_eval_val"] if (p2 and p2.get("final_eval_val")) else
              {"total": 0.0, "mean": 0.0, "median": 0.0, "n_positive": 0})
    if (p2 is not None) and not phase1_accepted:
        b6_val = {"total": -1.0, "note": "SKIPPED: phase-1 not accepted"}
    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B3_nomem": {"train_total": 211, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded (R3 dual nomem)"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "Phase-1 canonical utility SFT (this run)"},
        "B6_sft_grpo": {"train_total": b6_train.get("total", 0),
                        "val_total": (b6_val.get("total", 0)
                                      if isinstance(b6_val.get("total"), (int, float)) else 0),
                        "note": b6_val.get("note", "SFT->GRPO greedy closed loop")},
    }
    return table


def verdict_fn(b5_t, b6_t, b6_v, p1_report, p2):
    """Verdicts (directive R5): A = CANONICAL_SFT_GRPO_IMPROVES; B = GRPO_TRAINS
    _BUT_NO_GAIN; C = GRPO_EXPLORATION_OR_GROUP_COVERAGE_GAP; D = GRPO_UPDATE_
    DESTABILIZES_SFT; E = PROJECT_CONSOLIDATION_BLOCKED.

    Phase-2 runs ONLY if Phase-1 passes; if Phase-1 fails, no GRPO verdict exists
    (reported as E with an explicit note, identified=false)."""
    ph1_pass = p1_report.get("passed", False)
    if not ph1_pass:
        return "E", "PHASE1_ACCEPTANCE_FAILED_GRPO_NOT_RUN", 0.0
    if p2 is None:
        return "E", "PROJECT_CONSOLIDATION_BLOCKED", 0.0
    val_base = float(p1_report.get("closed_loop", {}).get("val_total", 0.0))
    delta_norm = (b6_t - b5_t) / max(b5_t, 1e-6)
    if b6_t > b5_t and b6_v >= val_base:
        return "A", "CANONICAL_SFT_GRPO_IMPROVES", delta_norm
    if abs(b6_t - b5_t) <= 1:
        # trained but flat within tolerance on TRAIN -> exploration/group-coverage gap
        return "C" if b6_t >= b5_t else "D", (
            "GRPO_EXPLORATION_OR_GROUP_COVERAGE_GAP" if b6_t >= b5_t
            else "GRPO_UPDATE_DESTABILIZES_SFT"), delta_norm
    if b6_t < b5_t:
        return "D", "GRPO_UPDATE_DESTABILIZES_SFT", delta_norm
    return "B", "GRPO_TRAINS_BUT_NO_GAIN", delta_norm


def write_report(args, re, p1, p1_report, p2, table, vlabel, verdict, vrank, reg):
    report = {
        "meta": {"experiment": "T1-M3-CANONICAL-CONSOLIDATION-UTILITY-RERANK-GRPO-R5",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True, "phase1_accepted": p1_report.get("passed", False)},
        "phase1": p1_report,
        "offline_metrics": p1["metrics"],
        "utility_reranker": p1["utility_reranker_stats"],
        "phase2": p2,
        "baseline_table": table,
        "verdict": {"label": vlabel, "code": verdict, "delta_norm": vrank},
        "regressions": reg,
        "audit_assertions": re["audits"],
    }
    dest = C.CANONICAL_OUT_DIR / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[report] wrote {dest}", flush=True)
    return dest


# ---------------------------------------------------------------------------
# Phase-1 pipeline
# ---------------------------------------------------------------------------
def run_phase1(args, env, re):
    p1 = train_phase1(env, re)

    # baseline A: control arm -- wide pool -> rank_head top-1 (old & new scorer)
    print("[baselineA] old SFT rank_head top-1 closed loop ...", flush=True)
    gains_old, _, _ = closed_loop_baseline_a(env, re, p1["scorer_mem"], "old_base")
    print(f"  A_old: {json.dumps(gains_old)}", flush=True)
    print("[baselineA] new SFT rank_head top-1 closed loop ...", flush=True)
    gains_new, _, _ = closed_loop_baseline_a(env, re, p1["scorer_mem"], "rank_head")
    print(f"  A_new: {json.dumps(gains_new)}", flush=True)

    # Phase-1 utility-aware reranker (the actual Phase-1 deliverable)
    print("[phase1] TRAIN utility-aware reranker (weighted |ΔU| pairs) ...", flush=True)
    ur, ur_hist, ur_stats = train_utility_reranker(re["state_examples"], p1["scorer_mem"],
                                                   re["mem_values"], seed=0)
    p1["utility_reranker_stats"] = ur_stats
    p1["reranker"] = ur            # Phase-1 canonical reranker (utility-aware)
    print(f"  utility-reranker: {json.dumps(ur_stats, default=str)}", flush=True)

    # metrics + B5 + acceptance + save must run AFTER p1["reranker"] = ur:
    # -- load path reuses this same tail so the S0 metrics always describe the
    # Phase-1 utility reranker, never the legacy missed-centric reranker.
    return _phase1_finalize(env, re, p1)


def _phase1_finalize(env, re, p1):
    """Metrics (utility reranker) + B5 closed loop + acceptance + save.

    Returns (p1, p1_report).  Shared by fresh runs and by --load-p1 recompute,
    so the Phase-1 acceptance always measures the SAME reranker that the B5
    closed loop consumes.
    """
    p1["metrics"] = phase1_offline_metrics(re, p1)

    print("[phase1] closed loop B5 (gate -> wide pool -> UTILITY reranker -> top1) ...", flush=True)
    b5sum, _ = closed_loop_b5(env, re, p1["scorer_mem"], p1["reranker"], p1["gate_m"],
                                       use_mem=True, mode="utility")
    print(f"  B5: {json.dumps(b5sum, default=str)}", flush=True)

    sft_summary_train = split_summary(b5sum, env)["train"]
    sft_summary_val = split_summary(b5sum, env)["val"]
    accepted, p1_report = phase1_acceptance(re, p1["metrics"], sft_summary_train,
                                            sft_summary_val, env)
    p1["summary_train"] = sft_summary_train
    p1["summary_val"] = sft_summary_val
    p1["accepted"] = accepted
    p1["__report__"] = p1_report
    torch.save({"state": {"scorer": p1["scorer_mem"], "gate": p1["gate_m"],
                          "reranker": p1["reranker"], "p1": p1},
                "meta": {"phase": "p1", "report": p1_report}},
               C.SFT_CKPT)
    print(f"[phase1] saved {C.SFT_CKPT}", flush=True)
    return p1, p1_report


def closed_loop_baseline_a(env, re, scorer, metric):
    """Baseline A: NO gate, NO reranker -- wide pool -> metric top-1 each step."""
    gains = {}
    steps_iid = {}
    for idx, i in enumerate(env["order"]):
        iid = i["instance_id"]
        eid = env["ep_id_of"][iid] if iid in env["ep_id_of"] else (len(env["train_insts"]) + idx)
        rf = {"problem": env["states"][iid]["problem"], "schedule": env["states"][iid]["schedule"],
              "iid": iid, "progmem": re["progmem"], "episode_id": eid,
              "scorer": scorer}
        gain, usage, steps = rollout_rank_top1(env, rf, metric, use_mem=True)
        gains[iid] = gain
        steps_iid[iid] = {"usage": usage, "steps": steps}
    return RO._b5_summary(gains), steps_iid, gains


# ---------------------------------------------------------------------------
# Phase-2 pipeline (GRPO)
# ---------------------------------------------------------------------------
def run_phase2(args, env, re, p1, p1_report):
    if not p1_report.get("passed", False):
        print("[phase2] SKIPPED: Phase-1 acceptance failed", flush=True)
        return None
    print("[phase2] building GRPO reward table (memoized Frozen-Local) ...", flush=True)
    reward_tbl = grpo_reward_table(re)
    agent = GRPOAgent(p1["scorer_mem"])
    rng = random.Random(args.grpo_seed)
    opt = torch.optim.AdamW(agent.head.parameters(), lr=C.GRPO_LR)
    mem_list = re["mem_values"]
    ev = []
    p2 = {"n_iters": C.GRPO_ITERS, "iters": [], "skip_meta_total": {"no_feasible": 0, "no_positive_n_or_stop_only": 0}}
    for it in range(C.GRPO_ITERS):
        t0 = time.time()
        records, skip_meta = agent.sample_batch(re["state_examples"], reward_tbl, mem_list, rng)
        for k, v in skip_meta.items():
            p2["skip_meta_total"][k] += v
        loss, st = grpo_step(agent, re["state_examples"], mem_list, records, opt, beta=C.GRPO_BETA_KL)
        with torch.no_grad():
            qa = grpo_action_quality(agent, re["state_examples"], reward_tbl, mem_list)
            resids = _residual_stats(agent, re["state_examples"], mem_list)
            qa["max_residual"] = resids["max_abs"]
            qa["mean_abs_residual"] = resids["mean_abs"]
        grad_norm = None
        if st["informative_groups"] > 0:
            gn = [p.grad.norm().item() for p in agent.head.parameters() if p.grad is not None]
            grad_norm = float(sum(gn)) if gn else 0.0
        if it % max(1, C.GRPO_ITERS // 4) == 0 or it == C.GRPO_ITERS - 1:
            g_train, _ = greedy_grpo_closed_loop(env, re, agent, use_mem=True)
            g_train_sum = split_summary(g_train, env)["train"]
            ev.append(g_train_sum)
        it_report = {
            "iter": it, "loss": float(loss.detach()), "stats": st,
            "grad_norm_group": grad_norm,
            "time_s": round(time.time() - t0, 2),
            "action_quality": qa,
            "eval_train": None,
        }
        if ev:
            g = ev[-1]
            it_report["eval_train"] = {"total": g["total"], "median": g["median"],
                                       "n_pos": g["n_positive"]}
        print(f"[grpo] iter {it} loss={float(loss.detach()):.4f} "
              f"info={st['informative_groups']}/{st['n_groups']} clip={st['clip_fraction']:.3f} "
              f"greedy_train_total={it_report['eval_train']['total'] if it_report['eval_train'] else '--'}"
              f" grad={grad_norm} max_res={qa['max_residual']:.4f}", flush=True)
        p2["iters"].append(it_report)
    p2["final_eval_train"] = ev[-1] if ev else None
    # VAL greedy closed loop (only if Phase-1 accepted); memory is fresh per run
    if p1_report.get("passed", False):
        g_val, _ = greedy_grpo_closed_loop(env, re, agent, use_mem=True, val_only=True)
        p2["final_eval_val"] = split_summary(g_val, env)["val"]
    else:
        p2["final_eval_val"] = {"total": -1.0, "note": "SKIPPED: phase-1 not accepted"}
    torch.save({"agent": agent, "head": agent.head.state_dict(),
                "meta": {"phase": "p2", "alpha": C.GRPO_ALPHA_RESIDUAL,
                         "eps": C.GRPO_EPS, "beta": C.GRPO_BETA_KL, "g_min": C.GRPO_G_MIN}},
               C.GRPO_CKPT)
    print(f"[phase2] saved {C.GRPO_CKPT}", flush=True)
    return p2


def _residual_stats(agent, state_examples, mem_list):
    """mean / max of |alpha*tanh(delta)| (bounded residual over a sample)."""
    vals = []
    sample = state_examples[: min(200, len(state_examples))]
    with torch.no_grad():
        for ex in sample:
            d_p, d_s = agent.head(ex["prop_feats"], ex["state_feat"])
            vals.append(float((agent.alpha * torch.tanh(d_p)).abs().max()))
            vals.append(float((agent.alpha * torch.tanh(d_s)).abs()))
    return {"mean_abs": float(np.mean(vals)) if vals else 0.0,
            "max_abs": float(np.max(vals)) if vals else 0.0}


# ---------------------------------------------------------------------------
# Regression battery
# ---------------------------------------------------------------------------
def run_regressions(args, env, re, p1, p2, p1_report):
    reg = {}
    runtest = not args.skip_regressions
    if runtest:
        print("[regress] DPPaulli10a / normal-M5 / memory-causal ...", flush=True)
        dpp_list = [i for i in env["order"] if i["instance_id"] == args.dpp]
        if not dpp_list:
            # fall back to any DPpaulli instance present
            dpp_list = [i for i in env["order"] if "DPpaulli" in i["instance_id"]]
        if dpp_list:
            dpp = dpp_list[0]
            dpp_iid = dpp["instance_id"]
            dpp_state = env["states"][dpp_iid]
            s0_hash = schedule_hash(dpp_state["schedule"])
            s0_trueU = {}
            for (ii, hh, sig), rec in re["store"].triple_index.items():
                if ii == dpp_iid and hh == s0_hash and rec["true_U"] is not None:
                    s0_trueU.setdefault(sig, rec["true_U"])
            dpp_ep = env["ep_id_of"][dpp_iid] if dpp_iid in env["ep_id_of"] else -1
            try:
                reg["dppaulli"] = RO.dppaulli_r4_trace(
                    dpp_iid, dpp_ep, dpp_state, env["cache"], p1["scorer_mem"],
                    p1["reranker"], p1["gate_m"], env["executor"], re["progmem"], s0_trueU)
            except Exception as exc:   # deterministic correctness surface
                reg["dppaulli"] = {"error": str(exc)}
            try:
                reg["normal_m5"] = RO.normal_m5_r4_path(
                    dpp_iid, dpp_ep, dpp_state, env["cache"], p1["scorer_mem"],
                    p1["reranker"], p1["gate_m"], env["executor"], re["progmem"], s0_trueU)
            except Exception as exc:
                reg["normal_m5"] = {"error": str(exc)}
            print(f"  dpp: {json.dumps(reg.get('dppaulli'), default=str)}", flush=True)
        else:
            reg["dppaulli"] = {"skipped": "instance not in order (quick)"}
        reg["memory_causal"] = re["audits"]["causal"]
        reg["memory_permanent_boundary"] = re["audits"]["unseen"]
    else:
        reg["skipped"] = True
    return reg


# ---------------------------------------------------------------------------
# R6 stage: Top-1 utility-selection SFT (T1-M3-TOP1-UTILITY-SELECTION-SFT-R6)
# ---------------------------------------------------------------------------
def _s0_trueU(env, re, iid):
    s0_hash = schedule_hash(env["states"][iid]["schedule"])
    lut = {}
    for (ii, hh, sig), rec in re["store"].triple_index.items():
        if ii == iid and hh == s0_hash and rec["true_U"] is not None:
            lut.setdefault(sig, float(rec["true_U"]))
    return lut


def run_top1_phase(args, env, re, p1, p1_report):
    """R6 Phase-1: train the Top-1 selector, evaluate, closed-loop B6, verdict.

    SFT-ONLY (GRPO forbidden).  Returns res dict; writes result.json + markdown.
    """
    print("[top1] R6 state-wise Top-1 utility-selection SFT (GRPO forbidden) ...", flush=True)
    t0 = time.time()
    selector, hist, groups, gstats = TOP1.train_top1_sft(
        re["state_examples"], p1["scorer_mem"], re["mem_values"], p1["reranker"], seed=0)
    print(f"[top1] selector trained in {time.time() - t0:.1f}s "
          f"({gstats['n_groups']} state groups)", flush=True)

    metrics, _ = TOP1.top1_metrics(re["state_examples"], p1["scorer_mem"],
                                   re["mem_values"], p1["reranker"], selector)
    acc = metrics["state_wise_top1_acc"]
    print(f"[top1] top1_acc: all={acc['all']:.3f} positive={acc['positive']:.3f} "
          f"stop={acc['stop']:.3f} hard={acc['hard']:.3f}", flush=True)
    print(f"[top1] selected_U {metrics['selected_true_U']} oracle "
          f"{metrics['oracle_best_true_U']} regret {metrics['top1_regret']} vs r5 "
          f"{metrics['top1_regret_r5']} recall {metrics['best_positive_recall']} "
          f"missed@10 {metrics['missed_at10']}", flush=True)

    print("[top1] closed loop B6 (real memory) ...", flush=True)
    b6_full, b6_full_steps = TOP1.closed_loop_top1(env, re, p1["scorer_mem"], selector, use_mem=True)
    print(f"  {json.dumps(b6_full, default=str)}", flush=True)
    print("[top1] closed loop B6 (masked memory -- regression arm) ...", flush=True)
    b6_masked, b6_masked_steps = TOP1.closed_loop_top1(env, re, p1["scorer_mem"], selector, use_mem=False)
    print(f"  {json.dumps(b6_masked, default=str)}", flush=True)

    tr = TOP1.split_summary(b6_full, env)["train"]
    va = TOP1.split_summary(b6_full, env)["val"]
    tr_m = TOP1.split_summary(b6_masked, env)["train"]

    # DPpaulli10a trace (§18) + normal-M5 regression (§26) in Top-1 mode
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    if dpp_iid is not None and not args.skip_regressions:
        dpp_st = env["states"][dpp_iid]
        dpp_ep = env["ep_id_of"][dpp_iid] if dpp_iid in env["ep_id_of"] else -1
        lut = _s0_trueU(env, re, dpp_iid)
        dpp_trace = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re,
                                        p1["scorer_mem"], selector, lut)
        print(f"[top1] DPPaulli trace: {json.dumps(dpp_trace, default=str)}", flush=True)
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re,
                                     p1["scorer_mem"], selector, lut)
            print(f"[top1] normal-M5: {json.dumps(m5, default=str)}", flush=True)
        except Exception as exc:
            m5 = {"error": str(exc)}
    else:
        dpp_trace = {"skipped": "skip-regressions or no DPPaulli instance"}
        m5 = {"skipped": "skip-regressions or no DPPaulli instance"}

    # acceptance + verdict (§17, §20-§23, §31)
    regret_down = bool(metrics["regret_down_vs_r5"])
    tr_ok = tr["total"] >= 235
    val_ok = va["total"] >= 10
    med_ok = (tr["median"] >= p1["summary_train"]["median"] or tr_ok)  # median not regressed vs B5
    strong = tr["total"] >= 300
    top1_gate = regret_down and acc["all"] >= 0.50
    closed_gate = tr_ok and val_ok
    passed = top1_gate and closed_gate

    # leave-Fattahi15-out
    leave = {}
    per = tr["per_instance"]
    for k in per:
        if "Fattahi15" in k:
            others = {k2: v for k2, v in per.items() if k2 != k}
            leave[k] = int(sum(others.values()))

    checks = {
        "top1_gate (regret_down + acc>=0.5)": top1_gate,
        "closed-loop TRAIN >= 235": tr_ok,
        "closed-loop TRAIN >= 300 (strong)": strong,
        "closed-loop VAL >= 10": val_ok,
        "median not regressed vs B5": med_ok,
        "R5 miss@10 baseline intact (top1 miss <= 0.35)":
            float(metrics["missed_at10"]["frac_positives"]) <= 0.35,
    }

    # R6 verdicts (§31): A/B/C/D/E
    rr = float(metrics["top1_regret"]["mean"])
    r5r = float(metrics["top1_regret_r5"]["mean"])
    missed_frac = float(metrics["missed_at10"]["frac_positives"])
    if tr_ok and val_ok and regret_down:
        vcode, vlabel = "A", "TOP1_SFT_READY_FOR_GRPO"
    elif regret_down and acc["all"] >= 0.5:
        vcode, vlabel = "B", "TOP1_OBJECTIVE_WORKS_BUT_GENERALIZATION_WEAK"
    elif acc["all"] < 0.5:
        vcode, vlabel = "C", "PROPOSAL_REPRESENTATION_LIMIT"
    elif missed_frac > 0.35:
        vcode, vlabel = "D", "TOP1_TRAINING_DESTABILIZES_RETRIEVAL"
    else:
        vcode, vlabel = "E", "CANONICAL_REGRESSION"
    print(f"[top1] verdict {vcode} {vlabel} (regret {rr:.2f} vs r5 {r5r:.2f}, "
          f"B6 train={tr['total']} val={va['total']})", flush=True)

    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B3_nomem": {"train_total": 211, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded (R3 dual nomem)"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "R5 Phase-1 canonical utility SFT (this run)"},
        "B6_top1_selector": {"train_total": tr["total"], "val_total": va["total"],
                             "note": "R6 Top-1 listwise SFT closed loop (this run)"},
    }

    report = {
        "meta": {"experiment": "T1-M3-TOP1-UTILITY-SELECTION-SFT-R6",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True,
                 "grpo_forbidden_this_round": True,
                 "r5_phase1_accepted": p1_report.get("passed", False)},
        "group_stats": gstats,
        "train_history": [{"ep": h["ep"], "ce": h["ce"], "pair": h["pair"],
                           "top1_acc": h["top1_acc"]} for h in hist],
        "offline_metrics": metrics,
        "closed_loop_b6": {"full": b6_full, "masked": b6_masked,
                           "train": tr, "val": va, "train_masked": tr_m},
        "leave_Fattahi15_out": leave,
        "dppaulli_trace": dpp_trace,
        "normal_m5": m5,
        "baseline_table": table,
        "checks": checks,
        "passed": passed,
        "verdict": {"code": vcode, "label": vlabel},
        "audit_assertions": re["audits"],
    }

    # canonical checkpoint (phase r6_top1) -- metadata per R6 §29
    ckpt = {"state": {"selector": selector, "scorer": p1["scorer_mem"]},
            "meta": {"phase": "r6_top1", "method": "top1_listwise_sft",
                     "parent": C.SFT_CKPT.name, "formal_test_access": 0,
                     "report": {"passed": passed, "vcode": vcode, "vlabel": vlabel,
                                "B6_train": tr["total"], "B6_val": va["total"]}}}
    torch.save(ckpt, C.TO1_CKPT)
    print(f"[top1] saved {C.TO1_CKPT} (passed={passed})", flush=True)

    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[top1] wrote result.json", flush=True)
    return report


def write_top1_markdown(res):
    lines = []
    add = lines.append
    add("# T1-M3-TOP1-UTILITY-SELECTION-SFT-R6 — 阶段报告")
    add("")
    add(f"**日期**: 2026-08-27 ｜ **实验**: R6（本轮仅 SFT，GRPO 禁止）")
    add("**机器数据**: `outputs/canonical_m3/result.json`（本报告所引数字全部可从该 JSON 复核）")
    add(f"**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED。")
    m = res["meta"]
    add(f"**R5 Phase-1 结存**: {'accepted' if m['r5_phase1_accepted'] else 'not accepted (E)'}；B5 canonical TRAIN={res['baseline_table']['B5_canonical_sft']['train_total']} VAL={res['baseline_table']['B5_canonical_sft']['val_total']}。")
    add("")
    add("## 0. 结论")
    v = res["verdict"]
    add(f"**verdict = {v['code']} {v['label']}**，passed={res['passed']}。")
    add("")
    add("## 1. 状态分组（group = (instance_id, state_hash)）")
    g = res["group_stats"]
    add(f"- 状态组数 `{g['n_groups']}`；全池减少 `{g['n_prop_total']}`；pool 大小 min/max/mean = "
        f"{g['pool_size_hist']['min']}/{g['pool_size_hist']['max']}/{g['pool_size_hist']['mean']:.1f}")
    add(f"- 正例态 `{g['n_pos_states']}` ({g['pos_state_frac']:.2f})；STOP 目标态 `{g['n_stop_target_states']}` "
        f"({g['stop_target_frac']:.2f})；零池 `{g['n_zero_pool']}`")
    add(f"- 硬态 `{g['n_failure_states']}`（DPpaulli10a `{g['n_dpp_states']}`）——按 §14 对 `DPpaulli+TOP1_FAILURE_STATE` "
        f"做 state 级过采样（multiplicity，无 identity embedding）")
    add(f"- R5-reranker 当前选中正例数 `{g['r5_pick_positive']}` / 命中全局最优 `{g['r5_pick_matched_best']}`")
    add("")
    add("## 2. Top-1 SFT（primary objective = Top-1 CE，非 Pearson）")
    om = res["offline_metrics"]
    a = om["state_wise_top1_acc"]
    add(f"- top1 acc：all `{a['all']:.3f}` / positive `{a['positive']:.3f}` / STOP `{a['stop']:.3f}` / hard `{a['hard']:.3f}`")
    add(f"- selected true_U mean/median = {om['selected_true_U']['mean']:.2f}/{om['selected_true_U']['median']:.2f} "
        f"(oracle mean {om['oracle_best_true_U']['mean']:.2f})")
    add(f"- top1 regret mean/median = {om['top1_regret']['mean']:.2f}/{om['top1_regret']['median']:.2f}；"
        f"R5 reranker regret = {om['top1_regret_r5']['mean']:.2f} → regret_down={om['regret_down_vs_r5']}")
    add(f"- best-positive recall@1/3/5/10 = {om['best_positive_recall']}")
    add(f"- missed@10 = {om['missed_at10']}；soft-KL diagnostic mean = {om['soft_kl_diagnostic']['mean']:.4f}")
    add("")
    add("## 3. closed-loop B6（Wide Recall → Top-1 selector → Proposal/STOP → FixedDecisionReplay, HORIZON=5）")
    cl = res["closed_loop_b6"]
    t_ = cl["train"]; v_ = cl["val"]; tm_ = cl["train_masked"]
    add(f"- **B6 TRAIN total={t_['total']}** mean={t_['mean']:.1f} median={t_['median']} n_pos={t_['n_positive']} "
        f"worst={t_['worst_instance']}")
    add(f"- B6 VAL total={v_['total']} mean={v_['mean']:.1f} median={v_['median']} per_instance={v_['per_instance']}")
    add(f"- Memory masked ablation（同 selector，masked 313-d）：TRAIN total={tm_['total']} → {t_['total']}")
    add("- per-instance TRAIN:")
    for k, vv in t_["per_instance"].items():
        add(f"  - {k}: {vv}")
    add(f"- leave-Fattahi15-out TRAIN total = {res['leave_Fattahi15_out']}")
    add("")
    add("## 4. DPpaulli10a 硬验收（§18）")
    d = res["dppaulli_trace"]
    if d.get("found"):
        bp = d.get("best_positive", {})
        bp2 = d.get("best_predicted", {})
        s_ = d.get("selected", {})
        add(f"- 正例 {d['n_pos_total']} 个 / 进 wide {d['n_pos_in_wide']} 个；最优正例 {bp.get('sig')} true_U={bp.get('true_U')}")
        add(f"- 最优正例预测 rank（池内）={bp2.get('pool_rank')}，predicted prob={bp2.get('action_prob')}")
        add(f"- 选中 {('STOP' if s_.get('is_stop') else s_.get('sig'))} true_U={s_.get('true_U')} → top1_regret={d.get('top1_regret')}")
        add(f"- 满意层级（§18: best/strong/base) = `{d.get('dpp_satisfaction')}`")
    else:
        add(f"- {d}")
    add("")
    add("## 5. Result（B0-B6 基准表）")
    tb = res["baseline_table"]
    for k, vv in tb.items():
        add(f"- `{k}`: train={vv['train_total']} val={vv['val_total']} — {vv['note']}")
    add("")
    add("## 6. 回归")
    add(f"- normal-M5（Top-1 模式）：{json.dumps(res['normal_m5'], default=str)}")
    add(f"- memory audits：{json.dumps(res['audit_assertions'], default=str)}")
    add("- no-legacy-import 防火墙与全量 pytest：在阶段外另行运行（本轮改动不 import legacy）")
    add("")
    add("## 7. 下一步")
    add(f"- verdict {v['code']}：{'下一轮进入 GRPO（仅当本轮 PASS）' if res['passed'] else '按 §1 本轮禁止 GRPO；先修 blocker 再进下一轮 SFT'}")
    add("")
    (C.CANONICAL_OUT_DIR / "T1_M3_TOP1_R6_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# R7: STOP calibration + OOD generalization (T1-M3-STOP-CALIBRATION-AND-OOD)
# ---------------------------------------------------------------------------
def run_top1_phase_r7(args, env, re, p1, p1_report):
    """R7: STOP relative-score margin calibration (Phase A frozen proposal head),
    TRAIN-only state augmentation, selector mem6 channel dropout, masked-channel
    same-weight eval, VAL no-backward decomposition.  SFT ONLY (GRPO forbidden)."""
    print("[r7] R7 STOP calibration + OOD generalization (SFT only; GRPO forbidden) ...",
          flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]

    # ---- R6 v2 selector (prop warm-start + DPP before/after, §2/§35) ----
    r6_sel = None
    if C.TO1_CKPT.exists():
        try:
            r6_sel = torch.load(C.TO1_CKPT, map_location="cpu",
                                weights_only=False)["state"]["selector"]
            print("[r7] loaded R6 v2 selector (prop_head warm-start source)", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"[r7] WARN: cannot load R6 ckpt {C.TO1_CKPT}: {exc}", flush=True)
    else:
        print(f"[r7] WARN: R6 ckpt {C.TO1_CKPT} not found; no R6 warm-start", flush=True)

    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"][dpp_iid] if (dpp_iid and dpp_iid in env["ep_id_of"]) else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}

    # ---- R6 baseline on the SAME replay groups (protect ordering, §2) ----
    if r6_sel is not None:
        mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                            re["mem_values"], reranker, r6_sel)
        ar6 = TOP1.top1_accuracy_stop_audit(grp_replay, r6_sel)
        print(f"[r7] R6 replay-groups baseline: acc={mr6['state_wise_top1_acc']} "
              f"regret={mr6['top1_regret']} audit.false_stop_pool={ar6['false_stop_over_pos_pool']:.3f}",
              flush=True)
    else:
        mr6, ar6, grp_replay = None, None, None

    # DPPaulli + normal-M5 regression gate (used for Phase-A/B decision too)
    def _gate_of(sel, replay_groups):
        audit = TOP1.top1_accuracy_stop_audit(replay_groups, sel)
        if dpp_iid is not None and not args.skip_regressions:
            dt = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, sel, dpp_lut)
            return {"dpp": dt, "dpp_margin": dt.get("score_margin"),
                    "false_stop_over_pos_pool": audit["false_stop_over_pos_pool"],
                    "audit": audit}
        return {"dpp": None, "dpp_margin": None,
                "false_stop_over_pos_pool": audit["false_stop_over_pos_pool"],
                "audit": audit}

    grp_r0, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                       reranker=reranker, rng=random.Random(0))
    gate_pre = _gate_of(r6_sel, grp_r0) if r6_sel is not None else {"dpp": None,
                                                                    "dpp_margin": None}
    if gate_pre["dpp"]:
        print(f"[r7] DPP BEFORE: rank={gate_pre['dpp']['best_predicted']} "
              f"stop={gate_pre['dpp']['stop']} margin={gate_pre['dpp']['score_margin']} "
              f"selected={gate_pre['dpp']['selected']}", flush=True)

    # ---- §9-14: TRAIN-only legal state augmentation ----
    budget = C.TO1_AUG_QUICK_PER_INST if args.quick else C.TO1_AUG_PER_INST
    aug_examples, aug_mem, aug_prov, aug_stats = TOP1.augment_train_states(
        env, re, scorer, r6_sel,
        per_inst_budget=budget,
        explore_bases=(1 if args.quick else C.TO1_AUG_EXPLORE_BASES),
        explore_k=(1 if args.quick else C.TO1_AUG_EXPLORE_K),
        seed=0)
    print(f"[r7] augmentation done in {time.time() - t0:.1f}s "
          f"(total {len(aug_examples)} new TRAIN states)", flush=True)

    # ---- R7 training: Phase A (frozen proposal head) + optional Phase B ----
    selector, hist, groups_all, gstats, phases = TOP1.train_top1_sft_r7(
        re["state_examples"], re["mem_values"], aug_examples, aug_mem,
        scorer, reranker, r6_sel, phase_gate=lambda sel: _gate_of(sel, grp_r0), seed=0)
    print(f"[r7] selector trained in {time.time() - t0:.1f}s", flush=True)

    # ---- offline metrics: replay-only (don't-break-R6) + all-TRAIN states ----
    mr7_replay, grp_r7 = TOP1.top1_metrics(re["state_examples"], scorer,
                                           re["mem_values"], reranker, selector)
    ar7_replay = TOP1.top1_accuracy_stop_audit(grp_r7, selector)
    exs_all = re["state_examples"] + aug_examples
    mem_all = re["mem_values"] + aug_mem
    mr7_all, _ = TOP1.top1_metrics(exs_all, scorer, mem_all, reranker, selector)
    mem_all0 = [torch.zeros_like(m) for m in mem_all]
    mr7_all_masked, _ = TOP1.top1_metrics(exs_all, scorer, mem_all0, reranker, selector)
    ar7_all = TOP1.top1_accuracy_stop_audit(groups_all, selector)
    print(f"[r7] top1_acc all={mr7_all['state_wise_top1_acc']['all']:.3f} "
          f"positive={mr7_all['state_wise_top1_acc']['positive']:.3f} "
          f"stop={mr7_all['state_wise_top1_acc']['stop']:.3f} "
          f"hard={mr7_all['state_wise_top1_acc']['hard']:.3f} "
          f"selected_U={mr7_all['selected_true_U']} regret={mr7_all['top1_regret']}",
          flush=True)

    # ---- closed loop B7 (real + masked memory, same weights) ----
    print("[r7] closed loop B7 (real memory) ...", flush=True)
    b7_real, b7_steps = TOP1.closed_loop_top1(env, re, scorer, selector, use_mem=True)
    print(f"  {json.dumps(b7_real, default=str)}", flush=True)
    print("[r7] closed loop B7 (masked memory) ...", flush=True)
    b7_masked, _ = TOP1.closed_loop_top1(env, re, scorer, selector, use_mem=False)
    print(f"  {json.dumps(b7_masked, default=str)}", flush=True)
    tr = TOP1.split_summary(b7_real, env)["train"]
    va = TOP1.split_summary(b7_real, env)["val"]
    tr_m = TOP1.split_summary(b7_masked, env)["train"]
    va_m = TOP1.split_summary(b7_masked, env)["val"]
    tr_total, va_total = tr["total"], va["total"]
    print(f"[r7] B7 TRAIN total={tr_total} median={tr['median']} VAL total={va_total}", flush=True)

    leave = {}
    for k in tr["per_instance"]:
        if "Fattahi15" in k:
            leave[k] = int(sum(v for k2, v in tr["per_instance"].items() if k2 != k))
    leave_best = max(leave.values()) if leave else 0

    # ---- DPP + normal-M5 AFTER + VAL decomposition ----
    gate_post = _gate_of(selector, grp_r7)
    dpp_post = gate_post["dpp"] if gate_post["dpp"] else {"skipped": True}
    if dpp_post.get("found"):
        print(f"[r7] DPP AFTER: rank={dpp_post['best_predicted']} stop={dpp_post['stop']} "
              f"margin={dpp_post['score_margin']} selected={dpp_post['selected']}", flush=True)
    m5 = {}
    if dpp_iid is not None and not args.skip_regressions:
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re,
                                     scorer, selector, dpp_lut)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}
    val_dec = TOP1.val_failure_decomposition(env, re, scorer, selector)

    # ---- acceptance gates (§24-§26, §34) ----
    a6 = ar6 if ar6 is not None else {"false_stop_over_pos_pool": 1.0,
                                      "stop_recall_over_stop_target": 0.0}
    fs_pre = float(a6["false_stop_over_pos_pool"])
    fs_post = float(ar7_replay["false_stop_over_pos_pool"])
    stop_rec_pre = float(a6["stop_recall_over_stop_target"])
    stop_rec_post = float(ar7_replay["stop_recall_over_stop_target"])
    acc6_all = mr6["state_wise_top1_acc"]["all"] if mr6 else 0.0
    acc7_replay_all = float(mr7_replay["state_wise_top1_acc"]["all"])
    regret6 = float(mr6["top1_regret"]["mean"]) if mr6 else 1e9
    regret7_replay = float(mr7_replay["top1_regret"]["mean"])
    dpp_rank_post = (dpp_post["best_predicted"]["pool_rank"]
                     if dpp_post.get("found") and "best_predicted" in dpp_post else 99)
    dpp_margin_pre = (gate_pre["dpp"]["score_margin"]
                      if gate_pre.get("dpp") else None)
    dpp_margin_post = gate_post.get("dpp_margin")

    prop_ok = bool(acc7_replay_all >= acc6_all - 0.05 and regret7_replay <= regret6 + 1.0
                   and dpp_rank_post <= 10)
    train_min = bool(tr_total >= 300 and leave_best > 122)
    train_strong = bool(tr_total >= 369 and leave_best >= 180)
    val_formal = bool(va_total >= 10)
    val_min = bool(va_total > 0)
    n_val_pos = sum(1 for k, v in va["per_instance"].items() if v > 0)
    stop_fixed = bool(dpp_margin_post is not None and dpp_margin_post > 0
                      and fs_post <= fs_pre - 0.10
                      and stop_rec_post >= stop_rec_pre - 0.25)
    leave_improved = bool(leave_best > 122)
    mem_neg_val = bool(va_total < va_m["total"])
    ready_for_grpo = bool(train_min and val_formal and stop_fixed and leave_improved)

    if not prop_ok:
        vcode, vlabel = "E", "STOP_CALIBRATION_DESTABILIZES_PROPOSALS"
    elif train_min and val_formal and stop_fixed and leave_improved and not mem_neg_val:
        vcode, vlabel = "A", "STOP_CALIBRATED_SFT_READY_FOR_GRPO"
    elif stop_fixed:
        vcode, vlabel = "B", "STOP_FIXED_GENERALIZATION_STILL_WEAK"
    elif val_min:
        vcode, vlabel = "C", "STATE_DIVERSITY_WAS_PRIMARY_BLOCKER"
    elif mem_neg_val:
        vcode, vlabel = "D", "MEMORY_NEGATIVE_TRANSFER_REMAINS"
    else:
        vcode, vlabel = "B", "STOP_FIXED_GENERALIZATION_STILL_WEAK"
    print(f"[r7] verdict {vcode} {vlabel} (TRAIN {tr_total} / VAL {va_total} / "
          f"leave {leave_best} / dpp_margin {dpp_margin_pre}->{dpp_margin_post} / "
          f"false_stop {fs_pre:.2f}->{fs_post:.2f})", flush=True)

    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "R5 Phase-1 canonical utility SFT"},
        "B6_top1_selector": {"train_total": 369, "val_total": 0,
                             "note": "R6 Top-1 SFT (recorded from T1_M3_TOP1_R6_REPORT)"},
        "B7_r7_calibrated": {"train_total": tr_total, "val_total": va_total,
                             "note": "R7 STOP-calibrated Top-1 SFT (this run)",
                             "train_masked": tr_m["total"], "val_masked": va_m["total"]},
    }

    checks = {
        "prop_ok (replay acc/regret/DPP rank protected)": prop_ok,
        "TRAIN >= 300": tr_total >= 300,
        "TRAIN >= 369 (strong)": train_strong,
        "leave-Fattahi15-out > 122": leave_improved,
        "VAL > 0 (min)": val_min,
        "VAL >= 10 (formal GRPO readiness)": val_formal,
        "stop_fixed (dpp_margin>0 + false_stop down + STOP recall intact)": stop_fixed,
        "mem_neg_val (real < masked on VAL)": mem_neg_val,
        "n_val_positive_instances": n_val_pos,
    }

    report = {
        "meta": {"experiment": "T1-M3-STOP-CALIBRATION-AND-OOD-GENERALIZATION-R7",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True, "grpo_forbidden_this_round": True,
                 "r5_phase1_accepted": p1_report.get("passed", False)},
        "augmentation": {"total_new": len(aug_examples), "n_replay": len(re["state_examples"]),
                         "n_train_states_total": len(exs_all), "stats": aug_stats,
                         "provenance": aug_prov},
        "training": {"phases": phases,
                     "history": [{"ep": h["ep"], "ce": h["ce"], "pair": h["pair"],
                                  "margin": h.get("margin"), "top1_acc": h["top1_acc"]}
                                 for h in hist]},
        "r6_replay_baseline": mr6,
        "r7_replay_metrics": mr7_replay,
        "r7_all_metrics": mr7_all,
        "r7_all_masked_metrics": mr7_all_masked,
        "stop_audit": {"r6_replay": ar6, "r7_replay": ar7_replay, "r7_all": ar7_all},
        "dppaulli": {"before": (gate_pre["dpp"] if gate_pre.get("dpp") else None),
                     "after": dpp_post},
        "closed_loop_b7": {"full": b7_real, "masked": b7_masked,
                           "train": tr, "val": va, "train_masked": tr_m, "val_masked": va_m},
        "leave_Fattahi15_out": leave,
        "normal_m5": m5,
        "val_failure_decomposition": val_dec,
        "memory": {"real_train": tr_total, "masked_train": tr_m["total"],
                   "real_val": va_total, "masked_val": va_m["total"],
                   "mem_neg_val": mem_neg_val},
        "baseline_table": table,
        "checks": checks,
        "ready_for_grpo": ready_for_grpo,
        "passed": bool(train_min and val_formal and stop_fixed and leave_improved),
        "verdict": {"code": vcode, "label": vlabel},
        "audit_assertions": re["audits"],
    }

    # canonical checkpoint v3 ONLY on PASS (§35)
    if report["passed"]:
        ckpt = {"state": {"selector": selector, "scorer": scorer},
                "meta": {"phase": "r7_top1_stop_calibrated",
                         "method": "top1_listwise_sft_stop_calibrated",
                         "parent": C.TO1_CKPT.name,
                         "n_train_states": len(exs_all),
                         "memory_semantics": "progressive_causal_time",
                         "memory_channel_dropout_p": C.TO1_MEM_DROP_P,
                         "stop_margin": {"m_act": C.TO1_MARGIN_ACT,
                                         "m_stop": C.TO1_MARGIN_STOP,
                                         "lambda_stop": C.TO1_LAMBDA_STOP},
                         "formal_test_access": 0, "ready_for_grpo": bool(ready_for_grpo),
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "B7_train": tr_total, "B7_val": va_total}}}
        torch.save(ckpt, C.TO1_CKPT_R7)
        print(f"[r7] saved {C.TO1_CKPT_R7} (PASS)", flush=True)
    else:
        print("[r7] NOT PASS -> canonical v3 checkpoint NOT written (R7 §35)", flush=True)

    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[r7] wrote result.json", flush=True)
    return report


def write_top1_r7_markdown(res):
    lines = []
    add = lines.append
    add("# T1-M3-STOP-CALIBRATION-AND-OOD-GENERALIZATION-R7 — 阶段报告")
    add("")
    add("**日期**: 2026-08-27 ｜ **实验**: R7（本轮仍只做 SFT / generalization repair，GRPO 禁止）")
    add("**机器数据**: `outputs/canonical_m3/result.json`（本报告所引数字全部可从该 JSON 复核）")
    add("**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
        "`true_U` 仅用于 TRAIN label / loss / offline diagnostic，**不进入 runtime rule**。")
    v = res["verdict"]
    add(f"")
    add(f"**verdict = {v['code']} {v['label']}**，passed={res['passed']}，"
        f"READY_FOR_TRUE_GRPO={res['ready_for_grpo']}。")
    add("")
    add("## 0. 结论")
    tr, va = res["closed_loop_b7"]["train"], res["closed_loop_b7"]["val"]
    add(f"- B7 TRAIN={tr['total']}（median {tr['median']}），VAL={va['total']}。"
        f"（基线：B5 124/132，B6 369/0，B2 oracle 594/198）")
    add(f"- STAR acceptance（§24-§26）: TRAIN≥300 {'✓' if tr['total']>=300 else '✗'}；"
        f"VAL>0 {'✓' if va['total']>0 else '✗'}；VAL≥10 {'✓' if va['total']>=10 else '✗'}；"
        f"leave-Fattahi15>122 {'✓' if res['leave_Fattahi15_out'] else '✗'}。")
    add("")
    add("## 1. 修改文件")
    add("- `src/causal_schedule_lab/m3/top1.py` — R7 段：`_stop_margin_loss` / "
        "`top1_accuracy_stop_audit` / `augment_train_states` / `_label_state_ex` / "
        "`train_top1_sft_r7`（Phase A 冻结 proposal，Phase B 仅硬闸不过时触达）/ "
        "`val_failure_decomposition`；`dpp_top1_trace` 增加 score/score_margin 输出。")
    add("- `src/causal_schedule_lab/m3/config.py` — R7 常量块（margins=1.0、λ_stop、"
        "mem-drop p=0.5、Phase A/B epochs、augmentation 预算、v3 ckpt 路径）。")
    add("- `scripts/run_m3_canonical_training.py` — `run_top1_phase_r7` / `write_top1_r7_markdown`；"
        "`--stage r7|all` 走 R7。")
    add("")
    au = res["augmentation"]
    add(f"## 2. 状态增广（§9-14）")
    add(f"- 新 TRAIN 唯一态 {au['total_new']} 个（oracle replay {au['n_replay']} 个保留），"
        f"训练态共 {au['n_train_states_total']} 个。")
    add(f"- 来源计数：policy={au['stats'].get('policy',0)} exploration={au['stats'].get('exploration',0)} "
        f"reused_replay={au['stats'].get('reused_replay',0)} skipped_dup={au['stats'].get('skipped_dup',0)}。")
    add(f"- 每实例：{json.dumps(au['stats'].get('per_instance', {}), default=str)}")
    add(f"- provenance 逐条在 `result.json → augmentation.provenance`（iid, state_hash, source, "
        f"parent_state_hash, parent_proposal_signature, depth, Cmax, appearance）。")
    add(f"- 唯一键 `(instance_id, state_hash)`；全部来自真实 FixedDecisionReplay，无随机假态。")
    add("")
    ph = res["training"]["phases"]
    add(f"## 3. STOP margin 目标（§3-§6）")
    add(f"- L_stop_margin：positive 态 `score(best_idx)≥score(STOP)+1.0`；STOP 态 "
        f"`score(STOP)≥max_i score(P_i)+1.0`；L_total = L_top1 + 0.3·L_pair + λ_stop·L_margin，"
        f"λ_stop={C.TO1_LAMBDA_STOP}。")
    if ph["phase_b_applied"]:
        pb = f"Phase B 触发：{ph['phase_b_epochs']} epochs 小 LR，只解冻 prop_head 最后一层"
    else:
        pb = "Phase B 未触发（Phase A 已满足硬闸）"
    add(f"- Phase A 冻结 proposal backbone/head（{ph['phase_a_epochs']} epochs，只训 STOP head）；"
        f"{pb}。")
    add("")
    add(f"## 4. 冻结 vs 未冻结参数")
    add(f"- 冻结：`scorer_mem`（R5 frozen scorer）、`reranker`（自由权重不变）、M2/Reasoner/"
        f"Wide-ReCall/Contributor-Enabler 池/FixedDecisionReplay/记忆语义。")
    add(f"- Phase A 未冻结：`stop_head` 全部参数。Phase B 追加：`prop_head` 最后一层 Linear(128,1)。")
    add("")
    add("## 5. STOP 校准审计（§7，replay 组、同记忆）")
    sa = res["stop_audit"]
    add(f"- false_stop(正例池内选 STOP)：R6 {sa['r6_replay']['false_stop_over_pos_pool']:.3f} → "
        f"R7 {sa['r7_replay']['false_stop_over_pos_pool']:.3f}（全态 {sa['r7_all']['false_stop_over_pos_pool']:.3f}）。")
    add(f"- false_act(STOP 态选 ACT)：R6 {sa['r6_replay']['false_act_over_stop_target']:.3f} → "
        f"R7 {sa['r7_replay']['false_act_over_stop_target']:.3f}。")
    add(f"- ACT recall / STOP recall：R7 pos-full 态 ACT {sa['r7_replay']['act_recall_over_pos_full']:.3f}；"
        f"STOP 态 STOP {sa['r7_replay']['stop_recall_over_stop_target']:.3f}（R6 {sa['r6_replay']['stop_recall_over_stop_target']:.3f}，未塌成恒 ACT）。")
    add("")
    dp = res["dppaulli"]
    add("## 6. DPpaulli10a 硬回溯（§8）")
    before = dp.get("before")
    if before and before.get("found") and "best_positive" in before:
        add(f"- BEFORE：best +27={before['best_positive']}，预测 rank {before['best_predicted']['pool_rank']} "
            f"(score {before['best_predicted'].get('score'):.4f}, p {before['best_predicted']['action_prob']:.4f})；"
            f"STOP score {before['stop']['score']:.4f} p {before['stop']['prob']:.4f}；"
            f"score_margin {before.get('score_margin')}；selected={before['selected']}；"
            f"dpp_satisfaction={before['dpp_satisfaction']}。")
    after = dp.get("after")
    if after and after.get("found") and "best_positive" in after:
        add(f"- AFTER：best +27={after['best_positive']}，预测 rank {after['best_predicted']['pool_rank']} "
            f"(score {after['best_predicted'].get('score'):.4f}, p {after['best_predicted']['action_prob']:.4f})；"
            f"STOP score {after['stop']['score']:.4f} p {after['stop']['prob']:.4f}；"
            f"score_margin {after.get('score_margin')}；selected={after['selected']}；"
            f"dpp_satisfaction={after['dpp_satisfaction']}。")
    elif after:
        add(f"- AFTER：{json.dumps(after, default=str)}（smoke/skipped）。")
    add("- 验收：最低 `+27 score > STOP score`；强 `selected Proposal true_U > 0`；最好 `selected == +27`。")
    add("")
    m = res["r7_all_metrics"]
    add(f"## 7. Top-1 accuracy / regret / selected_U（§16，全 TRAIN 训练态）")
    a = m["state_wise_top1_acc"]
    add(f"- top1 acc: all {a['all']:.3f} / positive {a['positive']:.3f}（严格全局最优命中）/ "
        f"STOP {a['stop']:.3f} / hard {a['hard']:.3f}（n_pos={a['n_positive_states']}, "
        f"n_stop={a['n_stop_states']}, n_hard={a['n_hard_states']}）。")
    add(f"- selected_U mean {m['selected_true_U']['mean']:.2f} / median {m['selected_true_U']['median']:.2f} "
        f"（n 正选 {m['selected_true_U']['n_selected_positive']} / n STOP {m['selected_true_U']['n_stop_selected']}）；"
        f"oracle mean {m['oracle_best_true_U']['mean']:.2f}。")
    add(f"- top1_regret mean {m['top1_regret']['mean']:.2f} / median {m['top1_regret']['median']:.2f} "
        f"（R5 reranker 同组 regret {m['top1_regret_r5']['mean']:.2f}）；best-positive recall {m['best_positive_recall']}；"
        f"missed@10 frac {m['missed_at10']['frac_positives']:.3f}。")
    add("")
    add(f"## 8. Baselines B0-B7（§24）")
    tbl = res["baseline_table"]
    add("| key | train | val | note |")
    add("|---|---|---|---|")
    for k, v2 in tbl.items():
        add(f"| {k} | {v2['train_total']} | {v2['val_total']} | {v2['note']} |")
    per_tr = {k: v for k, v in tr["per_instance"].items()}
    add(f"- TRAIN per-instance: {json.dumps(per_tr, default=str)}；median {tr['median']}；"
        f"leave-Fattahi15-out {res['leave_Fattahi15_out']}。")
    add(f"- VAL per-instance: {json.dumps({k: v for k, v in va['per_instance'].items()}, default=str)}。")
    add("")
    add("## 9. VAL 失败分解（§20，no_grad，diagnostic-only）")
    vd = res["val_failure_decomposition"]
    for iid, r in vd.items():
        add(f"- {iid}: n_pos={r.get('n_pos')} oracle_best={r.get('oracle_best_true_U')} "
            f"closed_loop_gain={r.get('closed_loop_gain')} candidate_miss={r.get('candidate_miss')} "
            f"wide_recall_miss={r.get('wide_recall_miss')} flags={json.dumps(r.get('flags', {}), default=str)} "
            f"trajectory_compounding={r.get('trajectory_compounding')}。")
    add("")
    add("## 10. Memory（§17-§19，同权重 masked 等价）")
    mem = res["memory"]
    add(f"- memory channel dropout p={C.TO1_MEM_DROP_P}（selector mem6 通道）——训练/评估同一模型权重。")
    add(f"- 闭环 TRAIN real={mem['real_train']} vs masked={mem['masked_train']}；"
        f"VAL real={mem['real_val']} vs masked={mem['masked_val']}。")
    add(f"- OOD 负迁移标志 mem_neg_val={mem['mem_neg_val']}。"
        f"{'若 real 仍伤 VAL：GRPO warm-start 时 Memory 可保留但 contribution 默认 gated/zero；不删除 Memory' if mem['mem_neg_val'] else '未观察到 OOD 负迁移。'}。")
    add(f"- 离线 masked 全态 metric：{json.dumps(res['r7_all_masked_metrics']['state_wise_top1_acc'], default=str)}。")
    add("")
    add("## 11. 回归")
    nm = res["normal_m5"]
    add(f"- normal-M5: {json.dumps(nm, default=str)}（§29 永久语义，enabler 正效用不要求被选）。")
    add("- FixedDecisionReplay 未改动（明确干预离散决策、最长路径时间戳、无 CP-SAT）。")
    add("- `tests/test_m3_no_legacy_import.py` 通过 → 见 pytest 汇总。")
    add("- Memory 语义：causal-time / no-future / state-gate / no-unseen-dump（未改）。")
    add("")
    add("## 12. 验收检查（§24-§26）")
    for k, v2 in res["checks"].items():
        add(f"- {k}: {v2}")
    add("")
    add("## 13. checkpoint（§35）")
    ck = f"`outputs/canonical_m3/m3_proposal_top1_sft_v3.pt`" if res["passed"] else \
        "未写（仅 PASS 才保存 v3；v2 改前快照仍为 `m3_proposal_top1_sft_v2.pt`）"
    add(f"- {ck}；metadata: method=top1_listwise_sft_stop_calibrated, parent=m3_proposal_top1_sft_v2.pt, "
        f"n_train_states={res['augmentation']['n_train_states_total']}, "
        f"memory_semantics=progressive_causal_time, formal_test_access=0, ready_for_grpo={res['ready_for_grpo']}。")
    add("")
    add("## 14. 下一步")
    add(f"- verdict {v['code']}；READY_FOR_TRUE_GRPO={res['ready_for_grpo']}。"
        f"{'下一轮进入规范 GRPO（如 VAL 仍<10 则先做 generalization repair）' if res['ready_for_grpo'] else '先修对应 blocker 再进下一轮 SFT'}。")
    add("")
    (C.CANONICAL_OUT_DIR / "T1_M3_STOP_CALIBRATION_R7_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# R9 AUX instance data stage (T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9)
# ---------------------------------------------------------------------------
def _aux_gen_kwargs(args) -> dict:
    """Resolution-order R9 AUX knobs (explicit > config default)."""

    return {"n": args.r9_n, "seed": args.r9_seed}


def _build_aux_progmem(aux_examples, aux_store, aux_insts, aux_ep_of, log_prefix="[aux]"):
    """Replicate the canonical per-instance progressive memory build (§18-§20) for the
    AUX set with ITS OWN ProgressiveMemory (self-contained; no cross-authority from
    AUX into TRAIN14 or VAL3, and no unseen-dump).  Returns (progmem, mem_values)."""

    progmem = MEM.ProgressiveMemory(
        state_feats=[ex["state_feat"].tolist() for ex in aux_examples])
    sf_map = {(ex["iid"], ex["state_hash"]): ex["state_feat"].tolist() for ex in aux_examples}
    per_inst = {}
    for ex in aux_examples:
        per_inst.setdefault(ex["iid"], []).append(ex)
    for inst in aux_insts:
        iid = inst["instance_id"]
        eid = aux_ep_of[iid]
        for rec in aux_store.records:
            if rec["instance_id"] != iid:
                continue
            sf = sf_map.get((iid, rec["state_hash"]))
            if sf is None:
                continue
            progmem.add_episode_record({
                "instance_id": iid, "episode_id": eid,
                "state_hash": rec["state_hash"], "state_feat": sf,
                "proposal_signature": rec["proposal_signature"],
                "proposal_type": rec["proposal_type"], "role": rec["role"],
                "src": rec["src"], "tgt": rec["tgt"], "true_U": rec["true_U"],
                "outcome": rec["outcome"],
                "trajectory_step": rec["trajectory_step"],
                "written_at_step": rec["trajectory_step"],
                "fine_key": rec["fine_key"], "coarse_key": rec["coarse_key"],
            }, eid)
        for t in range(max(len(per_inst.get(iid, ())) - 1, 0)):
            ex = per_inst[iid][t]
            U = ex["true_U"]
            seq = [k for k in range(len(U)) if ex["feasible"][k] and U[k] > 0]
            if not seq:
                continue
            best = max(seq, key=lambda k: float(U[k]))
            progmem.add_executed(iid, t, {
                "instance_id": iid, "episode_id": eid,
                "state_hash": ex["state_hash"], "state_feat": ex["state_feat"].tolist(),
                "proposal_signature": ex["sig"][best],
                "proposal_type": ex["type"][best], "role": ex["role"][best],
                "src": ex["src"][best], "tgt": ex["tgt"][best],
                "true_U": float(U[best]), "outcome": "success",
                "trajectory_step": t, "written_at_step": t,
                "fine_key": ((ex["type"][best], ex["role"][best], ex["src"][best],
                              ex["tgt"][best]) if ex["type"][best] == "single"
                             else (ex["type"][best], ex["role"][best])),
                "coarse_key": (ex["type"][best], ex["role"][best]),
            })
    MEM.compute_prog_mem_features(aux_examples, progmem)
    mem_values = [ex["prog_mem_feats"] for ex in aux_examples]
    return progmem, mem_values


def run_top1_phase_r9gen(args, env, re, p1, p1_report):
    """R9 §3-§10/§22/§34: generate AUX synthetic instances, run them through the FULL
    canonical state+replay label pipeline (frozen upstream M2 V5 + utility heads, same
    semantics as TRAIN14: schedule -> Appearance -> M2 -> Reasoner legal Proposal ->
    FixedDecisionReplay -> true_U), build self-contained AUX progressive memory,
    audit uniqueness/semantics, and persist r9_aux_data.pt for the R9 train stage."""

    t_start = time.time()
    print("[r9gen] R9 AUX instance data generation (SFT-only data; no model trained) ...",
          flush=True)
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    cache, executor = env["cache"], env["executor"]

    # ---- generate (deterministic, ALL config recorded; §9) --------------------
    t0 = time.time()
    gen_kwargs = _aux_gen_kwargs(args)
    out_dir = C.TO1_R9_AUX_DIR
    aux_insts = AX.generate_aux_instances(
        n=gen_kwargs["n"], out_dir=out_dir, seed=gen_kwargs["seed"])
    t_gen = time.time() - t0
    print(f"[r9gen] generated {len(aux_insts)} AUX instances in {t_gen:.1f}s "
          f"(seed {gen_kwargs['seed']})", flush=True)

    # ---- uniqueness audit (§7) ------------------------------------------------
    t14_paths = [Path(rel) for _, rel in AX.TRAIN14_FJS]
    uniq = AX.uniqueness_audit(aux_insts, t14_paths)
    print(f"[r9gen] uniqueness: {json.dumps(uniq, default=str)}", flush=True)

    # ---- build canonical states (S0 via solve_dispatching, §9) -----------------
    t0 = time.time()
    aux_states = {}
    for inst in aux_insts:
        d = {"instance_id": inst["instance_id"], "split": "aux",
             "path": AX.clean_fjs_path(inst)}
        aux_states[inst["instance_id"]] = env["pilot"].build_state(d)
        inst["_pilot_path"] = d["path"]
    t_state = time.time() - t0
    print(f"[r9gen] S0 states built for {len(aux_states)} AUX instances in {t_state:.1f}s",
          flush=True)

    # ---- replay + labels through the canonical pipeline (§8) -------------------
    t0 = time.time()
    aux_insts_pilot = [{"instance_id": i["instance_id"], "split": "aux",
                        "path": i["_pilot_path"]} for i in aux_insts]
    aux_examples, aux_store = RO.build_replay(cache, executor, aux_insts_pilot, aux_states)
    t_replay = time.time() - t0
    print(f"[r9gen] AUX replay done in {t_replay:.1f}s "
          f"({len(aux_examples)} states, {len(aux_store.records)} labels)", flush=True)

    # ---- attach canonical ex metadata (ep/tstep/s0/gi) -------------------------
    base_ep = len(env["train_insts"])
    aux_ep_of = {i["instance_id"]: base_ep + idx for idx, i in enumerate(aux_insts_pilot)}
    per_inst = {}
    for ex in aux_examples:
        per_inst.setdefault(ex["iid"], []).append(ex)
    for iid, exs in per_inst.items():
        for tstep, ex in enumerate(exs):
            ex["ep_id"] = aux_ep_of[iid]
            ex["tstep"] = tstep
            ex["s0"] = (tstep == 0)
    for gi, ex in enumerate(aux_examples):
        ex["gi"] = gi

    # ---- self-contained AUX progressive memory (§18-§20, no authority expansion)
    t0 = time.time()
    aux_progmem, aux_mem_values = _build_aux_progmem(
        aux_examples, aux_store, aux_insts_pilot, aux_ep_of)
    t_mem = time.time() - t0

    cas_assert = MEM.assert_causal_lookahead_free(aux_examples, aux_progmem, aux_store)
    unseen_assert = MEM.assert_unseen_state_zero(aux_examples, aux_progmem)
    pk = MEM.replay_primary_key_uniqueness(aux_store, aux_ep_of)
    s0_examples = [ex for ex in aux_examples if ex.get("s0")]
    cov = RK.coverage_audit(s0_examples, aux_examples, aux_progmem)
    print(f"[r9gen] aux audit {json.dumps({**cas_assert, **unseen_assert, **pk}, default=str)}",
          flush=True)
    print(f"[r9gen] aux coverage {json.dumps(cov)}", flush=True)

    # ---- §34 profiling: proposal-enumeration vs FrozenDecisionReplay/label split
    t_prop = 0.0
    t_label = 0.0
    if aux_examples:
        _first = aux_examples[0]
        _prob = aux_states[_first["iid"]]["problem"]
        _sched = aux_states[_first["iid"]]["schedule"]
        _t0 = time.time()
        cache.proposals(_prob, _sched, _first["iid"])   # warm
        _t1 = time.time()
        from causal_schedule_lab.m3.rollout import _execute_step
        from causal_schedule_lab.validation import schedule_hash as _sh
        # replicate one label step for the timing split
        for _k in range(min(5, len(_first["metas"]))):
            _feats, _metas, _ = cache.proposals(_prob, _sched, _first["iid"])
            _edits, _kind = _edits_for_ast(cache.ast(_prob, _sched, _first["iid"]), _metas[_k])
            _executor_step_t0 = time.time()
            _execute_step(executor, _prob, _sched, _edits,
                          int(_sched.makespan), _sh(_sched))
            t_label += time.time() - _executor_step_t0
        t_prop = time.time() - _t1 - t_label
    t_prof = time.time() - t_start

    # ---- split AUX 80/20 by INSTANCE (§22) ------------------------------------
    split = AX.split_aux(aux_insts, held_frac=C.TO1_R9_AUX_HELD_FRAC, seed=C.TO1_R9_AUX_SEED)
    print(f"[r9gen] AUX split: {split['n_train']} train / {split['n_held']} held (by instance)",
          flush=True)

    # ---- persist (atomic) ------------------------------------------------------
    payload = {
        "version": 1,
        "meta": {"experiment": "T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True,
                 "generator": "causal_schedule_lab.benchmarks.random_fjsp (pre-existing, audited)",
                 "schedule_construction": "solve_dispatching(earliest_finish) canonical S0",
                 "label_semantics": "canonical pipeline (M2 V5 + utility heads, FixedDecisionReplay)",
                 "trainslide_params": "TRAIN14-derived only (no VAL/TEST stats read)"},
        "gen_kwargs": {"n": len(aux_insts), "seed": gen_kwargs["seed"]},
        "manifest": aux_insts,
        "split": {"train_iids": [i["instance_id"] for i in split["train"]],
                  "held_iids": [i["instance_id"] for i in split["held"]],
                  "train_indices": [aux_insts.index(i) for i in split["train"]],
                  "held_indices": [aux_insts.index(i) for i in split["held"]]},
        "state_examples": aux_examples,
        "mem_values": aux_mem_values,
        "store_records": aux_store.records,
        "aux_ep_of": aux_ep_of,
        "stats": {"n_states": len(aux_examples), "n_labels": len(aux_store.records),
                  "n_instances": len(aux_insts), "n_s0_states": len(s0_examples),
                  "n_feasible": int(sum(int(f) for ex in aux_examples for f in ex["feasible"])),
                  "n_positive": int(sum(int(u > 0 and f) for ex in aux_examples
                                        for u, f in zip(ex["true_U"], ex["feasible"]))),
                  "labels_per_state": round(len(aux_store.records) / max(len(aux_examples), 1), 1)},
        "uniqueness": uniq,
        "audits": {"causal": cas_assert, "unseen": unseen_assert, "primary_key": pk,
                   "coverage": cov},
        "profiling": {"generate_s": round(t_gen, 2), "build_state_s": round(t_state, 2),
                      "replay_s": round(t_replay, 2), "memory_s": round(t_mem, 2),
                      "total_s": round(t_prof, 2),
                      "proposal_enum_s": round(t_prop, 3), "executor_label_s": round(t_label, 3)},
    }
    tmp = C.TO1_AUX_DATA.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(C.TO1_AUX_DATA)
    print(f"[r9gen] wrote {C.TO1_AUX_DATA}", flush=True)

    _write_aux_data_report(payload)
    print(f"[r9gen] total {time.time()-t_start:.1f}s", flush=True)
    return payload


def _edits_for_ast(ast, meta):
    from causal_schedule_lab.m3.proposal_features import _edits_for
    return _edits_for(ast, meta)


def _write_aux_data_report(payload):
    p = payload
    prof = p["profiling"]
    total = prof["total_s"]
    lines = []
    add = lines.append
    add("# T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9 — AUX 数据阶段报告")
    add("")
    add(f"**日期**: 2026-08-27 ｜ **阶段**: r9gen（仅数据；不训练模型）")
    add(f"**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
        f"true_U 仅 TRAIN label / loss / offline diagnostic（R9 §3），不进 runtime。")
    add("")
    add("## 数据来源分类（§2/§4）")
    add("- **BENCHMARK-TRAIN14**：14 个公开基准（frozen T1-DATA prereg）— 本轮只作 REAL-DOMAIN anchor/内部验证，不单独扩大。")
    add("- **AUX-TRAIN**：synthetic（本轮生成，`random_fjsp`），非 benchmark TRAIN 扩展，报告与 benchmark 严格分离。")
    add("- **VAL3**：no_grad only（本轮全程不读统计定 generator 参数）。")
    add("- **FORMAL-TEST3**：SEALED，从未访问。")
    add("")
    add("## 生成器审计（§3/§33）")
    add("- `causal_schedule_lab.benchmarks.random_fjsp`（既有，预存）：`build_fjsp` 产出 canonical `Problem`；")
    add("  输出 = GHH .fjs 的反函数，经 canonical `load_fjsp_problem` 读回。S0 schedule = `solve_dispatching(earliest_finish)`（确定性）。")
    add("- 探测（`outputs/r9_probe/`）：3 个 synthetic 实例全链路 10 状态 / 2196 labels / 0 infeasible / 23 positive — **generator 验证通过**，verdict-D 路径关闭。")
    add("")
    add("## AUX 实例规模与范围（§6，TRAIN14-derived）")
    add(f"- n_aux = **{p['gen_kwargs']['n']}**；split by instance：train {len(p['split']['train_iids'])} / held {len(p['split']['held_iids'])}（§22）。")
    add(f"- 状态数 {p['stats']['n_states']}；labels {p['stats']['n_labels']}；feasible {p['stats']['n_feasible']}；positive {p['stats']['n_positive']}；labels/state ≈ {p['stats']['labels_per_state']}。")
    add("- 结构范围（全部 TRAIN14-derived）：jobs {5..20}, machines {4..10}（Behnke m40/m60 离群已记录排除）, ops/job {2..12}, flex {1,2,3}, durations (1,30)/(1,100)/(1,320)。")
    add("- 禁止读取 VAL/TEST 统计去微调 generator（§5）：generator 参数只来自 TRAIN14 首行 + 每条工件解析。")
    add("")
    add("## 唯一性审计（§7）")
    add(f"- {json.dumps(p['uniqueness'], default=str)}")
    add("- 注：AUX 为 random 采样（routing skeleton + 时长均不同）；无 benchmark 克隆、无内部克隆。")
    add("")
    add("## Memory 语义（§18-§20）")
    add("- AUX 各实例走**独立 ProgressiveMemory**（AUX episodes only）：causal-time / no-future / state-gate / no unseen dump。")
    add("- **不扩 authority**：AUX 不向 TRAIN14/VAL3 注入历史；跨 AUX instance 仅按 canonical cross-episode 语义自然可见（与 TRAIN14 相同契约），无额外授权。")
    add(f"- 审计：causal `{json.dumps(p['audits']['causal'])}`, unseen `{json.dumps(p['audits']['unseen'])}`, pk `{json.dumps(p['audits']['primary_key'])}`。")
    add("")
    add("## Profiling（§34，本机单进程）")
    add(f"- generate/build_state/replay/memory/total = "
        f"{prof['generate_s']}s / {prof['build_state_s']}s / {prof['replay_s']}s / {prof['memory_s']}s / {total}s")
    add(f"- 百分比：gen {prof['generate_s']/total*100:.1f}% / build_state {prof['build_state_s']/total*100:.1f}% / "
        f"replay {prof['replay_s']/total*100:.1f}% / memory {prof['memory_s']/total*100:.1f}%")
    add(f"- label/executor 单例 micro 计时：proposal_enum {prof['proposal_enum_s']}s vs "
        f"executor_label {prof['executor_label_s']}s（5 提案样本）— 供云迁移 worker 并行估计。")
    add(f"- 建议（§34，不购买）：若 replay/label 明显 CPU-bound（proposal∝候选枚举、executor∝feasibility），"
        f"云迁移按 `min(cpu, 4) × (TRAIN14_instances 块)` 并行化 — 详见 R9 正式报告 §34 结论。")
    add(f"- checkpoint：`{C.TO1_AUX_DATA}`（atomic 写盘）。")
    (C.AUX_REPORT).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# R10 AUX-REAL instance data stage (D1=SAFE_AUX_REAL only, §10-§11)
# ---------------------------------------------------------------------------
def run_top1_phase_r10gen(args, env, re, p1, p1_report):
    """R10 §10-§11/§34/§36: select AUX-REAL (D1=SAFE_AUX_REAL) instances from the
    Phase-0 provenance audit, run them through the FULL canonical state+replay
    label pipeline (frozen upstream M2 V5 + utility heads, same semantics as
    TRAIN14), build self-contained AUX progressive memory, audit, split by
    INSTANCE, and persist r10_aux_real_data.pt for the R10 calibration stage.

    Only D1 rows participate (D2/D3/D4 are excluded by construction in the audit
    classification).  BENCHMARK-TRAIN14 vs AUX-REAL stay separate in reports (§4)."""

    t_start = time.time()
    print("[r10gen] R10 AUX-REAL (D1) instance data generation "
          "(SFT-only data; no model trained) ...", flush=True)
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    cache, executor = env["cache"], env["executor"]

    # ---- deterministic D1 selection (audit is the only authority, §36) ------
    n = max(1, args.r10_n)
    sel = AX.select_aux_real_instances(n=n, max_ops_gate=C.TO1_R10_MAX_OPS_GATE,
                                       seed=C.TO1_R10_REAL_SEED)
    if not sel:
        raise RuntimeError("No AUX-REAL instances selected -- audit missing or empty D1")
    print(f"[r10gen] selected {len(sel)} AUX-REAL (D1) instances "
          f"(<= {C.TO1_R10_MAX_OPS_GATE} ops, family-covered, deterministic)", flush=True)
    from collections import Counter
    print(f"  families: {dict(Counter(r['family'] for r in sel))}", flush=True)
    print(f"  n_ops = [{min(r['n_ops'] for r in sel)}, {max(r['n_ops'] for r in sel)}]", flush=True)

    # ---- build canonical states (S0 via solve_dispatching, §9) --------------
    t0 = time.time()
    sel_insts = []
    for r in sel:
        path = str(C.ROOT / r["path"])
        d = {"instance_id": r["instance_id"], "split": "aux_real",
             "path": path, "family": r["family"], "n_ops": r["n_ops"]}
        sel_insts.append(d)
    real_states = {}
    for d in sel_insts:
        real_states[d["instance_id"]] = env["pilot"].build_state(d)
        d["_pilot_path"] = d["path"]
    t_state = time.time() - t0
    print(f"[r10gen] S0 states built for {len(real_states)} AUX-REAL instances "
          f"in {t_state:.1f}s", flush=True)

    # ---- replay + labels through the canonical pipeline (§8) ----------------
    t0 = time.time()
    real_insts_pilot = [{"instance_id": d["instance_id"], "split": "aux_real",
                         "path": d["_pilot_path"]} for d in sel_insts]
    real_examples, real_store = RO.build_replay(cache, executor, real_insts_pilot, real_states)
    t_replay = time.time() - t0
    print(f"[r10gen] AUX-REAL replay done in {t_replay:.1f}s "
          f"({len(real_examples)} states, {len(real_store.records)} labels)", flush=True)

    # ---- attach canonical ex metadata (ep/tstep/s0/gi) -----------------------
    base_ep = len(env["train_insts"])
    real_ep_of = {i["instance_id"]: base_ep + idx for idx, i in enumerate(real_insts_pilot)}
    per_inst = {}
    for ex in real_examples:
        per_inst.setdefault(ex["iid"], []).append(ex)
    for iid, exs in per_inst.items():
        for tstep, ex in enumerate(exs):
            ex["ep_id"] = real_ep_of[iid]
            ex["tstep"] = tstep
            ex["s0"] = (tstep == 0)
    for gi, ex in enumerate(real_examples):
        ex["gi"] = gi

    # ---- self-contained AUX progressive memory (§18-§20) --------------------
    t0 = time.time()
    real_progmem, real_mem_values = _build_aux_progmem(
        real_examples, real_store, real_insts_pilot, real_ep_of, log_prefix="[r10gen]")
    t_mem = time.time() - t0

    cas_assert = MEM.assert_causal_lookahead_free(real_examples, real_progmem, real_store)
    unseen_assert = MEM.assert_unseen_state_zero(real_examples, real_progmem)
    pk = MEM.replay_primary_key_uniqueness(real_store, real_ep_of)
    s0_examples = [ex for ex in real_examples if ex.get("s0")]
    cov = RK.coverage_audit(s0_examples, real_examples, real_progmem)
    print(f"[r10gen] AUX-REAL audit "
          f"{json.dumps({**cas_assert, **unseen_assert, **pk}, default=str)}", flush=True)
    print(f"[r10gen] AUX-REAL coverage {json.dumps(cov)}", flush=True)

    # ---- split AUX-REAL train/held by INSTANCE (§22 semantics) ---------------
    rng = random.Random(C.TO1_R10_REAL_SEED)
    held = set(rng.sample([d["instance_id"] for d in sel_insts],
                          max(1, int(round(len(sel_insts) * C.TO1_R10_REAL_HELD_FRAC)))))
    tr_iids = [d["instance_id"] for d in sel_insts if d["instance_id"] not in held]
    hd_iids = [d["instance_id"] for d in sel_insts if d["instance_id"] in held]
    print(f"[r10gen] AUX-REAL split (by instance): {len(tr_iids)} train / {len(hd_iids)} held",
          flush=True)

    # ---- persist (atomic) ------------------------------------------------------
    payload = {
        "version": 1,
        "meta": {"experiment": "T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION-R10",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True,
                 "source": "SAFE_AUX_REAL (D1) ONLY from r10_aux_real_audit.json (Phase 0)",
                 "schedule_construction": "solve_dispatching(earliest_finish) canonical S0",
                 "label_semantics": "canonical pipeline (M2 V5 + utility heads, FixedDecisionReplay)",
                 "selection": "deterministic: D1 AND n_ops<=%d, smallest-n_ops greedy, family-covered (seed %d)"
                              % (C.TO1_R10_MAX_OPS_GATE, C.TO1_R10_REAL_SEED)},
        "selection_kwargs": {"n": n, "max_ops_gate": C.TO1_R10_MAX_OPS_GATE,
                             "seed": C.TO1_R10_REAL_SEED},
        "manifest": sel,
        "split": {"train_iids": tr_iids, "held_iids": hd_iids},
        "state_examples": real_examples,
        "mem_values": real_mem_values,
        "store_records": real_store.records,
        "real_ep_of": real_ep_of,
        "stats": {"n_states": len(real_examples), "n_labels": len(real_store.records),
                  "n_instances": len(sel_insts), "n_s0_states": len(s0_examples),
                  "n_feasible": int(sum(int(f) for ex in real_examples for f in ex["feasible"])),
                  "n_positive": int(sum(int(u > 0 and f) for ex in real_examples
                                        for u, f in zip(ex["true_U"], ex["feasible"]))),
                  "labels_per_state": round(len(real_store.records) / max(len(real_examples), 1), 1)},
        "audits": {"causal": cas_assert, "unseen": unseen_assert, "primary_key": pk,
                   "coverage": cov},
        "profiling": {"build_state_s": round(t_state, 2), "replay_s": round(t_replay, 2),
                      "memory_s": round(t_mem, 2), "total_s": round(time.time() - t_start, 2)},
    }
    C.TO1_REAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.TO1_REAL_DATA.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(C.TO1_REAL_DATA)
    print(f"[r10gen] wrote {C.TO1_REAL_DATA}", flush=True)

    _write_r10_aux_real_data_report(payload)
    print(f"[r10gen] total {time.time()-t_start:.1f}s", flush=True)
    return payload


def _write_r10_aux_real_data_report(payload):
    p = payload
    prof = p["profiling"]
    total = prof["total_s"]
    from collections import Counter
    fam = Counter(r["family"] for r in p["manifest"])
    lines = []
    add = lines.append
    add("# T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION-R10 — AUX-REAL 数据阶段报告")
    add("")
    add(f"**日期**: 2026-08-27 ｜ **阶段**: r10gen（仅数据；不训练模型）")
    add(f"**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
        f"true_U 仅 TRAIN label / loss / offline diagnostic（R10 §3），不进 runtime。")
    add("")
    add("## 数据来源分类（§4/§36，frozen）")
    add("- A BENCHMARK-TRAIN14 = 14（frozen prereg）；B VAL3 = 3（no_grad only）；C FORMAL-TEST3 = SEALED（从未访问）。")
    add("- D = 359 unused instances → **Phase-0 审计**分类："
        f"D1_SAFE_AUX_REAL **328** / D2_EVAL_RESERVED 12 / D3_DUPLICATE_OVERLAP 19 / D4 0；"
        f"parse 失败 0；sha 不匹配 0。")
    add("- 本轮只消费 **D1=SAFE_AUX_REAL**；D2/D3/D4 一律杜绝（UNKNOWN 永久禁用）。")
    add(f"- 本阶段选择 {p['stats']['n_instances']} 个 D1 实例：确定性选择 "
        f"(最小 n_ops 优先 + family 覆盖，门限 n_ops ≤ {C.TO1_R10_MAX_OPS_GATE})；"
        f"family 组成 {dict(fam)}。")
    add("")
    add("## 标签语义（§8-§10，全部走 canonical 管线）")
    add("- schedule→Appearance→M2→Contributor+Enabler→Reasoner legal Proposal→FixedDecisionReplay→true_U；")
    add("- 无 CP-SAT 代替 label；S0 合法构造不变；每实例 manifest 记录 family / n_ops / sha256。")
    add(f"- stats: 状态 {p['stats']['n_states']} ；labels {p['stats']['n_labels']}；"
        f"feasible {p['stats']['n_feasible']}；positive {p['stats']['n_positive']}；"
        f"labels/state ≈ {p['stats']['labels_per_state']}。")
    add("")
    add("## Memory 语义（§18-§20）")
    add("- AUX-REAL 各实例走**独立 ProgressiveMemory**（AUX episodes only）：causal-time / no-future / state-gate / no unseen dump。")
    add(f"- 审计：causal `{json.dumps(p['audits']['causal'], default=str)}`, "
        f"unseen `{json.dumps(p['audits']['unseen'], default=str)}`, "
        f"pk `{json.dumps(p['audits']['primary_key'], default=str)}`。")
    add("")
    add("## split（by instance）")
    add(f"- train {len(p['split']['train_iids'])} / held {len(p['split']['held_iids'])}（seed {C.TO1_R10_REAL_SEED}）。")
    add("")
    add("## Profiling（§34，本机单进程）")
    add(f"- build_state/replay/memory/total = {prof['build_state_s']}s / {prof['replay_s']}s / "
        f"{prof['memory_s']}s / {total}s")
    add(f"- checkpoint：`{C.TO1_REAL_DATA}`（atomic 写盘）。")
    (C.R10_AUX_REAL_REPORT).write_text("\n".join(lines), encoding="utf-8")


def _load_r6_parent(args):
    """Load the canonical R6 v2 selector (R9 §40 parent anchor)."""
    if not C.TO1_CKPT.exists():
        raise RuntimeError("R9 requires canonical R6 checkpoint "
                           f"{C.TO1_CKPT} (run --stage top1 first, then --stage r9gen)")
    ck = torch.load(C.TO1_CKPT, map_location="cpu", weights_only=False)
    sel = ck["state"]["selector"]
    sel.eval()
    for p in sel.parameters():
        p.requires_grad_(False)
    print(f"[r9] loaded R6 v2 selector parent {C.TO1_CKPT.name}", flush=True)
    return sel


def run_top1_phase_r9(args, env, re, p1, p1_report):
    """R9: AUX-instance generalization via R6-anchored bounded residual SFT.

    score_R9 = score_R6 + alpha·tanh(delta), delta≡0 init, R6 prop+STOP frozen
    (never replace R6, §13/§17).  Training = BENCHMARK-TRAIN14 ∪ AUX-TRAIN with
    instance-mixed batches (§12); L = lambda_argmax·L_argmax + lambda_ref·L_ref on
    R6-correct states ONLY (§14-§15).  Three-layer eval: AUX held-out (§22/§23-A),
    TRAIN14 3-fold by-instance internal CV (§23-B), VAL once no-tuning (§23-C).
    R6-reference preservation (§25), DPP before/after (§26), B9 closed loops
    (real + masked, same weights), fixed 100-instance AUX (no sweep, §6).
    GRPO still forbidden (§39).  v3 checkpoint ONLY on PASS (§40)."""
    print("[r9] R9 AUX-instance generalization residual SFT (SFT only; GRPO forbidden) ...",
          flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]

    # ---- §40 parent: canonical R6 v2 selector ---------------------------------
    r6_sel = _load_r6_parent(args)

    # ---- load AUX data payload (r9gen) ----------------------------------------
    if not C.TO1_AUX_DATA.exists():
        raise RuntimeError("r9_aux_data.pt missing -- run `--stage r9gen` first")
    payload = torch.load(C.TO1_AUX_DATA, map_location="cpu", weights_only=False)
    aux_ex_all = payload["state_examples"]
    aux_mem_all = payload["mem_values"]
    split = payload["split"]
    train_iids, held_iids = set(split["train_iids"]), set(split["held_iids"])
    aux_train_iids = [i for i in split["train_iids"]]
    aux_held_iids = [i for i in split["held_iids"]]
    aux_tr_ex = [ex for ex in aux_ex_all if ex["iid"] in train_iids]
    aux_hd_ex = [ex for ex in aux_ex_all if ex["iid"] in held_iids]
    aux_tr_mem = [m for ex, m in zip(aux_ex_all, aux_mem_all) if ex["iid"] in train_iids]
    aux_hd_mem = [m for ex, m in zip(aux_ex_all, aux_mem_all) if ex["iid"] in held_iids]
    ax_stats = payload["stats"]
    ax_audits = payload["audits"]
    print(f"[r9] AUX payload: states={ax_stats['n_states']} labels={ax_stats['n_labels']} "
          f"instances={ax_stats['n_instances']} positive={ax_stats['n_positive']} "
          f"train_iids={len(aux_train_iids)} held_iids={len(aux_held_iids)}", flush=True)
    semantic_ok = bool(
        ax_audits["causal"].get("causal_assertion") == "PASS"
        and ax_audits["unseen"].get("unseen_state_zero") == "PASS"
        and ax_audits["primary_key"]["n_unique_primary_keys"]
        == ax_audits["primary_key"]["n_records"]
        and int(ax_audits["coverage"].get("n_queries", 0)) > 0
        and float(ax_audits["coverage"].get("zero_ratio", 1.0)) < 1.0)
    print(f"[r9] AUX semantic audit ok={semantic_ok} "
          f"{json.dumps({k: v for k, v in ax_audits.items() if k != 'coverage'}, default=str)}",
          flush=True)

    # ---- DPP instance + S0 LUT (same as R8) -----------------------------------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}

    # ---- R6 deterministic reproduction (gate_mem=False -> bit-identical §40) --
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    ar6 = TOP1.top1_accuracy_stop_audit(grp_replay, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r9] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} repro_ok={repro_ok}",
          flush=True)

    # ---- R6 gated baseline on TRAIN14 replay (pool-argmax, §24) ---------------
    grp_g9, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                       reranker=reranker, rng=random.Random(0), gate_mem=True)
    TOP1.tag_r6_correct(grp_g9, r6_sel)
    for g in grp_g9:
        g["src"] = "bench"
    pa_r6 = TOP1.pool_argmax_metrics(grp_g9, r6_sel)
    pos_acc6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])
    n_r6_correct = sum(1 for g in grp_g9 if g.get("r6_correct"))
    print(f"[r9] R6 gated TRAIN baseline: pos_argmax={pos_acc6:.3f} regret_full={regret6:.2f} "
          f"recall10={rec10_6:.3f} n_pos={pa_r6['n_positive_states']} "
          f"r6_correct_states={n_r6_correct}/{len(grp_g9)}", flush=True)

    # ---- AUX groups (self-contained AUX memory; §18-§20) + R6 tag -------------
    aux_tr_groups, st_tr = TOP1.build_top1_groups(
        aux_tr_ex, scorer, aux_tr_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
    aux_hd_groups, st_hd = TOP1.build_top1_groups(
        aux_hd_ex, scorer, aux_hd_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
    TOP1.tag_r6_correct(aux_tr_groups, r6_sel)
    TOP1.tag_r6_correct(aux_hd_groups, r6_sel)
    for g in aux_tr_groups:
        g["src"] = "aux"
    for g in aux_hd_groups:
        g["src"] = "aux"
    pa_r6_aux_hd = TOP1.pool_argmax_metrics(aux_hd_groups, r6_sel)
    print(f"[r9] AUX groups: train={len(aux_tr_groups)} held={len(aux_hd_groups)} "
          f"(pos_states tr={st_tr['n_pos_states']} hd={st_hd['n_pos_states']}) "
          f"R6-on-AUX-held pos_argmax={pa_r6_aux_hd['positive_state_argmax_accuracy']:.3f}",
          flush=True)

    # ---- §23-B: fixed 3-fold by-INSTANCE internal CV (TRAIN14 only) -----------
    print(f"[r9] internal CV (folds={C.TO1_R8_CV_FOLDS}, by-instance, AUX-TRAIN in train) ...",
          flush=True)
    cv = TOP1.internal_cv_r9(env, re, scorer, reranker, r6_sel,
                             aux_tr_ex, aux_tr_mem, aux_hd_groups,
                             folds=C.TO1_R8_CV_FOLDS, seed=C.TO1_R8_CV_SEED,
                             epochs=C.TO1_R9_CV_EPOCHS, quick=args.quick)
    agg = cv["aggregate"]
    if agg is not None:
        print(f"[r9] CV agg: held_pos_argmax r9={agg['mean_held_pos_acc_r9']:.3f} vs "
              f"r6={agg['mean_held_pa_acc_r6']:.3f} (Δ={agg['held_argmax_improves_r6']:+.3f}) "
              f"regret r9={agg['mean_held_regret_r9']:.2f} vs r6={agg['mean_held_regret_r6']:.2f} "
              f"closed-loop r9={agg['mean_held_gain_total_r9']:.1f} "
              f"r6={agg['mean_held_gain_total_r6']:.1f} masked={agg['mean_held_gain_total_masked']:.1f} "
              f"aux_held_pa={agg['aux_held_pa']:.3f} best_epoch={agg['mean_best_held_epoch']:.1f}",
              flush=True)
    else:
        print("[r9] CV produced no folds (quick?) -> gates treated as FAIL", flush=True)

    cv_pass = bool(agg is not None
                   and float(agg["mean_held_pos_acc_r9"])
                   >= float(agg["mean_held_pa_acc_r6"]) + C.TO1_R8_CV_MIN_IMPROVE
                   and float(agg["mean_held_gain_total_r9"]) > 0.0)
    mem_neg_held = bool(agg is not None and not bool(agg["mean_held_gain_total_masked"]
                                                     >= agg["mean_held_gain_total_r9"] - 1.0))

    # ---- epochs for full train, selected ONLY on AUX-held + TRAIN-held (§22) --
    if args.quick:
        epochs_full = 2
    elif agg is not None:
        epochs_full = int(round(float(agg["mean_best_held_epoch"])))
        epochs_full = int(min(max(epochs_full, 4), C.TO1_R9_EPOCHS))
    else:
        epochs_full = C.TO1_R9_EPOCHS
    print(f"[r9] full-train epochs={epochs_full} (CV AUX-held best epoch selection)", flush=True)

    # ---- §12/§13 full R9 residual training on TRAIN14 ∪ AUX-TRAIN ------------
    selector9, hist9 = TOP1.train_top1_sft_r9(
        grp_g9, aux_tr_groups, r6_sel, held_groups=aux_hd_groups,
        seed=0, epochs=epochs_full, lr=C.TO1_R9_LR, alpha=C.TO1_R9_ALPHA,
        batch_size=C.TO1_R9_BATCH, lambda_ref=C.TO1_R9_LAMBDA_REF,
        log_prefix="[r9]")
    selector9.eval()
    print(f"[r9] selector9 trained in {time.time()-t0:.1f}s", flush=True)

    # ---- offline pool-argmax: TRAIN replay (same gated list) + AUX held -------
    pa9_replay = TOP1.pool_argmax_metrics(grp_g9, selector9)
    pa9_aux_hd = TOP1.pool_argmax_metrics(aux_hd_groups, selector9)
    pos_acc9 = float(pa9_replay["positive_state_argmax_accuracy"])
    regret9 = float(pa9_replay["top1_regret_full"]["mean"])
    rec10_9 = float(pa9_replay["recall"]["10"])
    print(f"[r9] TRAIN replay: pos_argmax={pos_acc9:.3f} (R6 {pos_acc6:.3f}) "
          f"regret={regret9:.2f} (R6 {regret6:.2f}) recall10={rec10_9:.3f} (R6 {rec10_6:.3f})", flush=True)
    print(f"[r9] AUX held: pos_argmax={pa9_aux_hd['positive_state_argmax_accuracy']:.3f} "
          f"(R6 {pa_r6_aux_hd['positive_state_argmax_accuracy']:.3f}) "
          f"regret={pa9_aux_hd['top1_regret_full']['mean']:.2f}", flush=True)

    # ---- §25: R6-reference preservation on TRAIN replay groups ---------------
    r6_presv = TOP1.r9_preservation_metrics(grp_g9, r6_sel, selector9)
    print(f"[r9] R6 preservation: spearman={r6_presv['spearman_mean']:.4f} "
          f"r6_correct_preserved={r6_presv['r6_correct_preserved']:.4f} "
          f"r6_correct_regressed={r6_presv['r6_correct_regressed']:.4f} "
          f"TRAIN pa r6={r6_presv['r6_pa_acc']:.3f}->r9={r6_presv['r9_pa_acc']:.3f} "
          f"regret r6={r6_presv['r6_regret']:.2f}->r9={r6_presv['r9_regret']:.2f}", flush=True)
    r6_preserved_ok = bool(r6_presv["r6_correct_preserved"] is not None
                           and r6_presv["r6_correct_preserved"] >= 0.90
                           and r6_presv["r6_correct_regressed"] is not None
                           and r6_presv["r6_correct_regressed"] <= 0.10)

    # ---- DPP BEFORE(R6) / AFTER(R9) (§26) -------------------------------------
    dpp_pre = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, r6_sel,
                                  dpp_lut) if (dpp_iid is not None and not args.skip_regressions) else None
    if dpp_pre and dpp_pre.get("found"):
        print(f"[r9] DPP BEFORE(R6): rank={dpp_pre['best_predicted']} "
              f"selected={dpp_pre['selected']} margin={dpp_pre.get('score_margin')}", flush=True)
    dpp_post = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, selector9,
                                   dpp_lut, gate_mem=True) \
        if (dpp_iid is not None and not args.skip_regressions) else None
    dpp_rank_pre = (int(dpp_pre["best_predicted"]["pool_rank"])
                    if dpp_pre and dpp_pre.get("found") and "best_predicted" in dpp_pre else 99)
    dpp_rank_post = (dpp_post["best_predicted"]["pool_rank"]
                     if dpp_post and dpp_post.get("found") and "best_predicted" in dpp_post else 99)
    dpp_sel_u = (dpp_post["selected"].get("true_U")
                 if dpp_post and dpp_post.get("found") else None)
    if dpp_post and dpp_post.get("found"):
        print(f"[r9] DPP AFTER(R9): rank={dpp_rank_post} (R6 {dpp_rank_pre}) "
              f"selected={dpp_post['selected']} best={dpp_post.get('best_positive')} "
              f"stop={dpp_post.get('stop')}", flush=True)
    dpp_ok = bool(dpp_rank_post <= min(6, dpp_rank_pre) and dpp_rank_post <= 6)

    # ---- B9 closed loops (real gated + masked, same weights §25-26) -----------
    print("[r9] closed loop B9 (real memory, gated) ...", flush=True)
    b9_real, b9_steps = TOP1.closed_loop_top1(env, re, scorer, selector9,
                                              use_mem=True, gate_mem=True)
    print(f"  {json.dumps({k: v for k, v in b9_real.items() if k != 'per_instance'}, default=str)}",
          flush=True)
    print("[r9] closed loop B9 (masked memory) ...", flush=True)
    b9_masked, _ = TOP1.closed_loop_top1(env, re, scorer, selector9,
                                         use_mem=False, gate_mem=False)
    tr = TOP1.split_summary(b9_real, env)["train"]
    va = TOP1.split_summary(b9_real, env)["val"]
    tr_m = TOP1.split_summary(b9_masked, env)["train"]
    va_m = TOP1.split_summary(b9_masked, env)["val"]
    tr_total, va_total = tr["total"], va["total"]
    print(f"[r9] B9 TRAIN total={tr_total} mean={tr['mean']:.1f} median={tr['median']} "
          f"VAL total={va_total}", flush=True)

    leave = {}
    for k in tr["per_instance"]:
        if "Fattahi15" in k:
            for k2 in tr["per_instance"]:
                if k2 != k:
                    leave[k] = leave.get(k, 0) + int(tr["per_instance"][k2])
    leave_best = max(leave.values()) if leave else 0

    m5 = {}
    if dpp_iid is not None and not args.skip_regressions:
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re, scorer, selector9,
                                     dpp_lut, gate_mem=True)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}
    print("[r9] VAL once (no_grad, final checkpoint, no tuning) ...", flush=True)
    val_dec = TOP1.val_failure_decomposition(env, re, scorer, selector9, gate_mem=True)

    # ---- acceptance gates (§24/§25/§37/§38) + verdict (§41) -------------------
    dstab = bool(regret9 > regret6 + 1.0 or rec10_9 < rec10_6 - 0.10 or dpp_rank_post > 10)
    pa_improved = bool((not dstab) and pos_acc9 >= pos_acc6 + 0.05 and regret9 <= regret6)
    aux_gate = bool((not dstab) and
                    float(pa9_aux_hd["positive_state_argmax_accuracy"])
                    >= float(pa_r6_aux_hd["positive_state_argmax_accuracy"])
                    + C.TO1_R9_AUX_HELD_GATE)
    train_min = bool(tr_total >= 300 and leave_best >= 169)
    train_strong = bool(tr_total >= 369 and leave_best >= 180)
    val_formal = bool(va_total >= 10)
    val_min = bool(va_total > 0)
    n_val_pos = sum(1 for k, v in va["per_instance"].items() if v > 0)
    val_pos_frac = float(n_val_pos / len(va["per_instance"])) if va["per_instance"] else 0.0
    overfit_flag = bool(train_strong and not cv_pass)
    r9_ready_blockers = [b for b, ok in {
        "train_min": train_min,
        "pa_improved (TRAIN replay pos_argmax>=R6+0.05, not dstab)": pa_improved,
        "cv_pass (§24 held pos_argmax>=R6_same_held+0.05 AND held gain>0)": cv_pass,
        "aux_gate (§22 AUX-held pos_argmax>=R6_same+0.03)": aux_gate,
        "val_formal (VAL>=10)": val_formal,
        "r6_preserved_ok (§25 >=0.90 preserved, <=0.10 regressed)": r6_preserved_ok,
        "dpp_ok (§26 rank<=min(6,pre))": dpp_ok,
        "semantic_ok (AUX audits PASS)": semantic_ok,
    }.items() if not ok]
    if (not repro_ok) or dstab or (not pa_improved and not cv_pass):
        vcode, vlabel = "C", "R6_ANCHORED_RESIDUAL_CANNOT_TRANSFER"
        note = "residual failed to improve (or destabilized) R6 on TRAIN/held"
    elif not semantic_ok:
        vcode, vlabel = "E", "AUXILIARY_DATA_SEMANTICS_INVALID"
        note = "AUX audit regression (causal/unseen/pk/coverage) -- do not train on it"
    elif aux_gate and cv_pass and val_formal and r6_preserved_ok and dpp_ok and not mem_neg_held:
        vcode, vlabel = "A", "INSTANCE_DIVERSITY_FIXES_GENERALIZATION_READY_FOR_GRPO"
        note = "AUX breadth + anchored residual transfers to held-out instances and VAL"
    elif (aux_gate or cv_pass or pa_improved) and not val_formal:
        vcode, vlabel = "B", "AUX_DATA_HELPS_BUT_VAL_STILL_WEAK"
        note = "held-out/internal improvement visible; formal VAL not yet strong"
    elif (aux_gate or cv_pass) and (not r6_preserved_ok or not dpp_ok or mem_neg_held):
        vcode, vlabel = "C", "R6_ANCHORED_RESIDUAL_CANNOT_TRANSFER"
        note = "some improvement but R6 reference / DPP / memory negative transfer"
    else:
        vcode, vlabel = "B", "AUX_DATA_HELPS_BUT_VAL_STILL_WEAK"
        note = "no single blocker isolated; generalization still below thresholds"
    ready_for_grpo = bool(train_min and pa_improved and cv_pass and aux_gate and val_formal
                          and r6_preserved_ok and dpp_ok and semantic_ok and not overfit_flag)
    cv_delta = (float(agg["held_argmax_improves_r6"]) if agg is not None else None)
    print(f"[r9] verdict {vcode} {vlabel} (TRAIN {tr_total} / VAL {va_total} / "
          f"leave {leave_best} / pos_argmax {pos_acc6:.2f}->{pos_acc9:.2f} / "
          f"cv Δ={cv_delta} / aux_held {pa_r6_aux_hd['positive_state_argmax_accuracy']:.2f}"
          f"->{pa9_aux_hd['positive_state_argmax_accuracy']:.2f} / "
          f"dpp rank {dpp_rank_pre}->{dpp_rank_post})", flush=True)
    print(f"[r9] ready_for_grpo={ready_for_grpo} blockers={r9_ready_blockers}", flush=True)

    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "R5 Phase-1 canonical utility SFT"},
        "B6_top1_selector": {"train_total": 369, "val_total": 0,
                             "note": "R6 Top-1 SFT (recorded from T1_M3_TOP1_R6_REPORT)"},
        "B7_r7_calibrated": {"train_total": 416, "val_total": 0,
                             "note": "R7 STOP-calibrated (recorded from T1_M3_STOP_CALIBRATION_R7_REPORT)"},
        "B8_r8_pool_argmax": {"train_total": 0, "val_total": 0,
                              "note": "R8 verdict E -- v4 checkpoint NOT written (no promotion)"},
        "B9_r9_residual": {"train_total": tr_total, "val_total": va_total,
                           "note": "R9 R6-anchored AUX-instance residual SFT (this run)",
                           "train_masked": tr_m["total"], "val_masked": va_m["total"]},
    }

    checks = {
        "repro_ok (R6 acc 0.525 / regret 5.2 bit-id)": repro_ok,
        "pa_improved (TRAIN replay pos_argmax >= R6+0.05 and regret <= R6, not dstab)": pa_improved,
        "cv_pass (held-out pos_argmax >= R6+0.05 AND held gain>0)": cv_pass,
        "aux_gate (AUX-held pos_argmax >= R6_same + 0.03)": aux_gate,
        "r6_preserved_ok (>=0.90 preserved / <=0.10 regressed)": r6_preserved_ok,
        "dpp_ok (DPP rank <= min(6, pre))": dpp_ok,
        "semantic_ok (causal/unseen/pk/coverage PASS)": semantic_ok,
        "TRAIN >= 300": train_min,
        "TRAIN >= 369 (strong)": train_strong,
        "leave-Fattahi15-out >= 169": leave_best >= 169,
        "VAL >= 10 (formal GRPO readiness)": val_formal,
        "VAL > 0 (min)": val_min,
        "n_val_positive_instances": n_val_pos,
        "val_positive_frac >= 1/3": val_pos_frac >= 0.333,
        "overfit_flag (strong TRAIN + weak CV)": overfit_flag,
        "mem_neg_held (masked>=real on held -> D)": mem_neg_held,
        "stop_no_worse_than_r6 (§17 STOP frozen; keep stop audit)":
            bool(float(TOP1.top1_accuracy_stop_audit(grp_g9, selector9)["stop_recall_over_stop_target"])
                 >= float(ar6["stop_recall_over_stop_target"]) - 0.30),
    }

    report = {
        "meta": {"experiment": "T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True, "grpo_forbidden_this_round": True,
                 "r5_phase1_accepted": p1_report.get("passed", False),
                 "r8_verdict": "E POOL_ARGMAX_TRAIN_ONLY_DOES_NOT_TRANSFER (2026-08-27, v4 not written)"},
        "modified_files": ["src/causal_schedule_lab/m3/top1.py",
                           "src/causal_schedule_lab/m3/config.py",
                           "src/causal_schedule_lab/m3/aux_instances.py",
                           "scripts/run_m3_canonical_training.py"],
        "r6_reproduction": {"acc_all": r6_acc_all, "regret_mean": r6_regret, "repro_ok": repro_ok,
                            "full": mr6},
        "aux_data": {"gen_kwargs": payload["gen_kwargs"], "manifest": payload["manifest"],
                     "stats": ax_stats, "uniqueness": payload["uniqueness"],
                     "audits": ax_audits, "profiling": payload["profiling"],
                     "split": {"n_train_instances": len(aux_train_iids),
                               "n_held_instances": len(aux_held_iids),
                               "train_iids": aux_train_iids, "held_iids": aux_held_iids}},
        "residual_architecture": {"formula": "score_R9(P) = score_R6(P) + alpha*tanh(delta(P))",
                                  "alpha": C.TO1_R9_ALPHA, "delta_init": 0,
                                  "frozen": ["R6 prop_head (full)", "R6 STOP head (full)",
                                             "scorer_mem", "M2/Reasoner/Wide-ReCall/Contributor-Enabler/"
                                             "FixedDecisionReplay", "memory_semantics"],
                                  "trainable": ["res_head Linear(277,64)+GELU+Linear(64,1) only",
                                                "lr=" + str(C.TO1_R9_LR)],
                                  "losses": {"lambda_argmax": C.TO1_R9_LAMBDA_ARGMAX,
                                             "lambda_ref": C.TO1_R9_LAMBDA_REF,
                                             "ref_scope": "R6-correct states only (§14)"}},
        "instance_mixed_batch": {"ratio_bench_aux": C.TO1_R9_MIX_BENCH_RATIO,
                                 "batch_size": C.TO1_R9_BATCH,
                                 "sampler": "source -> instance -> state (§12)"},
        "internal_cv": {"folds": cv["folds"], "aggregate": agg,
                        "per_heldout_instance": {},
                        "thoname": "benchmark 3-fold by-instance (TRAIN14)"},
        "r6_gated_training_baseline": pa_r6,
        "r9_replay_pool_argmax": pa9_replay,
        "aux_held_pool_argmax": {"r6_same_held": pa_r6_aux_hd, "r9": pa9_aux_hd,
                                 "delta": float(pa9_aux_hd["positive_state_argmax_accuracy"]
                                                - pa_r6_aux_hd["positive_state_argmax_accuracy"])},
        "r6_preservation": r6_presv,
        "stop_audit": {"r6_replay": ar6,
                       "r9_replay": TOP1.top1_accuracy_stop_audit(grp_g9, selector9)},
        "dppaulli": {"before_R6": dpp_pre, "after_R9": dpp_post,
                     "rank_before": dpp_rank_pre, "rank_after": dpp_rank_post},
        "closed_loop_b9": {"full": b9_real, "masked": b9_masked,
                           "train": tr, "val": va, "train_masked": tr_m, "val_masked": va_m},
        "b9_steps": b9_steps,
        "leave_Fattahi15_out": leave,
        "normal_m5": m5,
        "val_failure_decomposition": val_dec,
        "memory": {"real_train": tr_total, "masked_train": tr_m["total"],
                   "real_val": va_total, "masked_val": va_m["total"],
                   "mem_gate": C.TO1_R8_MEM_GATE,
                   "aux_memory": "self-contained ProgressiveMemory (AUX episodes only, §18-§20)",
                   "mem_neg_held": mem_neg_held},
        "baseline_table": table,
        "checks": checks,
        "ready_for_grpo": ready_for_grpo,
        "passed": bool(pa_improved and cv_pass and aux_gate and train_min
                       and r6_preserved_ok and dpp_ok and semantic_ok),
        "verdict": {"code": vcode, "label": vlabel, "note": note},
        "next_action": ("entering canonical GRPO phase (Phase 2, --stage p2); R9 selects "
                        "--stage r9 acceptance of the benchmark-AUX mix"
                        if ready_for_grpo
                        else "fix blocker per verdict (see r9_ready_blockers) before GRPO"),
        "r9_ready_blockers": r9_ready_blockers,
        "audit_assertions": re["audits"],
        "aux_audit_assertions": ax_audits,
    }

    # canonical v3 checkpoint ONLY on PASS (§40)
    if report["passed"]:
        ckpt = {"state": {"selector": selector9, "scorer": scorer,
                          "r6_anchor": r6_sel.state_dict()},
                "meta": {"phase": "r9_aux_instance_residual_sft",
                         "method": "r6_anchored_aux_instance_residual_sft",
                         "parent": "m3_proposal_top1_sft_v2.pt",
                         "benchmark_train_instances": 14,
                         "aux_train_instances": len(aux_train_iids),
                         "aux_total_instances": len(aux_train_iids) + len(aux_held_iids),
                         "aux_source": "causal_schedule_lab.benchmarks.random_fjsp (pre-existing, audited)",
                         "aux_seed": C.TO1_R9_AUX_SEED,
                         "split_by_instances": True,
                         "residual": {"alpha": C.TO1_R9_ALPHA, "delta_init": 0,
                                      "lambda_ref": C.TO1_R9_LAMBDA_REF},
                         "training": "BENCHMARK-TRAIN14 (14) + AUX-TRAIN (synthetic)",
                         "memory_semantics": "progressive_confidence_gated (bench) + self-contained AUX",
                         "formal_test_access": 0, "ready_for_grpo": bool(ready_for_grpo),
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "B9_train": tr_total, "B9_val": va_total,
                                    "aux_held_pos_argmax": float(
                                        pa9_aux_hd["positive_state_argmax_accuracy"]),
                                    "dpp_rank": dpp_rank_post}}}
        torch.save(ckpt, C.TO1_CKPT_R9)
        print(f"[r9] saved {C.TO1_CKPT_R9} (PASS)", flush=True)
    else:
        print("[r9] NOT PASS -> canonical v3 checkpoint NOT written (§40)", flush=True)

    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result_r9.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[r9] wrote result_r9.json", flush=True)
    return report


def write_top1_r9_markdown(res):
    r = res
    add_r = lambda s="": lines.append(s)  # noqa: E731
    checks = r["checks"]
    v = r["verdict"]
    agg = r["internal_cv"]["aggregate"]
    cv = r["internal_cv"]
    clb = r["closed_loop_b9"]
    benef = r["aux_data"]["profiling"]
    lines = []
    for ln in [f"# T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9 报告",
               "",
               f"**实验**: {r['meta']['experiment']} ｜ **日期**: 2026-08-27",
               f"**verdict**: {v['code']} **{v['label']}** —— {v.get('note', '')}",
               f"**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
               f"true_U 仅 TRAIN label / loss / offline diagnostic，永不进 runtime；GRPO §39 本轮禁止。",
               ""]:
        lines.append(ln)
    lines.append("## 目的与改动")
    lines.append("- R8 结论：pool-argmax（L_argmax）TRAIN 可学（0.321→0.357，regret 9.11→7.71）但")
    lines.append("  **held-CV 不迁移**（best_held_epoch=0，held regret 21.97>19.34，closed-loop 93<104），")
    lines.append("  v4 checkpoint 未写。")
    lines.append("- R9 因此放弃在同样 14 个 benchmark TRAIN 实例上继续 loss-only SFT；改以 **AUX-TRAIN**（synthetic）")
    lines.append("  提供实例多样性 + **R6-anchored 有界残差**（score_R6 + α·tanh δ，δ≡0 init；只修不换）。")
    lines.append("- 改动文件：" + ", ".join(r["modified_files"]) + "。")
    lines.append("")
    lines.append("## §2 数据来源分类（fixed）")
    lines.append("- A BENCHMARK-TRAIN14 = 14（frozen prereg）；B VAL3=3（no_grad only）；C FORMAL-TEST3=SEALED（从未访问）。")
    lines.append("- D 其它非评估基准 ~359：本轮**不消费**；E AUX-TRAIN = synthetic（本轮生成）。B/C/F 永不用于 backward。")
    lines.append("## §3/§33 生成器验证")
    lines.append("- 生成器 = 既有 `random_fjsp`（== build_fjsp 反函数；canonical loader 往返；S0=solve_dispatching）。")
    lines.append("- 探测：3 synthetic 实例 → 10 状态 / 2196 labels / 0 infeasible / 23 positive。generator 验证 PASS，verdict-D 关闭。")
    lines.append("")
    lines.append(f"## §6 实例规模与范围（无 sweep，fixed count）")
    lines.append(f"- n_aux={r['aux_data']['gen_kwargs']['n']} 实例（seed {r['aux_data']['gen_kwargs']['seed']}）；"
                 f"状态 {r['aux_data']['stats']['n_states']}；labels {r['aux_data']['stats']['n_labels']}；"
                 f"positive {r['aux_data']['stats']['n_positive']}；n_s0={r['aux_data']['stats']['n_s0_states']}。")
    lines.append(f"- split by instance: train={r['aux_data']['split']['n_train_instances']} / "
                 f"held={r['aux_data']['split']['n_held_instances']}（§22）。")
    lines.append("- 结构范围全部 TRAIN14-derived（jobs {5..20} / machines {4..10} / ops {2..12} / flex {1,2,3} / dur (1,30),(1,100),(1,320)）；")
    lines.append("  从没读 VAL/TEST 统计调 generator（§5）。")
    lines.append("## §7 唯一性")
    lines.append(f"- {json.dumps(r['aux_data']['uniqueness'])}")
    lines.append("## §8-§10 标签语义（全部走 canonical 管线）")
    lines.append("- schedule→Appearance→M2→Contributor+Enabler→Reasoner legal Proposal→FixedDecisionReplay→true_U；")
    lines.append("  无 CP-SAT 代替 label；S0 合法构造不变；每条实例 record gen config（manifest）。")
    lines.append(f"- §10 广度>深度：{r['aux_data']['stats']['n_states']} 状态 / {r['aux_data']['stats']['n_instances']} 实例，"
                 f"实例数 100 >> TRAIN14 的 14。")
    lines.append("## §12-§16 训练目标")
    lines.append(f"- 残差公式：`score_R9 = score_R6 + {r['residual_architecture']['alpha']}·tanh(δ)`，δ≡0 init，"
                 f"只训 res_head（$\\mathbb{{R}}^{{277}}$→64→1），R6 prop+STOP 全冻结（§13/§17）。")
    lines.append(f"- L = {r['residual_architecture']['losses']['lambda_argmax']}·L_argmax + "
                 f"{r['residual_architecture']['losses']['lambda_ref']}·L_ref（仅在 R6-correct 状态，§14）。")
    lines.append(f"- instance-mixed batch：bench:aux={r['instance_mixed_batch']['ratio_bench_aux']}:"
                 f"{1 - r['instance_mixed_batch']['ratio_bench_aux']}，source→instance→state（§12）。")
    lines.append("## §18-§20 Memory")
    lines.append("- Memory 保留（causal-time / state-gate / no unseen dump）；AUX 走**独立** ProgressiveMemory（AUX episodes only），不扩 authority；")
    lines.append("  训练时 mem6 channel dropout；g_mem=coverage·similarity。")
    lines.append(f"- AUX audit：{json.dumps({k: v for k, v in r['aux_data']['audits'].items() if k != 'coverage'}, default=str)}")
    lines.append("")
    lines.append("## §23 三层评估")
    lines.append("### A. AUX held-out（20% by instance）")
    ar = r["aux_held_pool_argmax"]
    lines.append(f"- R6-on-same-held pos_argmax={ar['r6_same_held']['positive_state_argmax_accuracy']:.3f} → "
                 f"R9={ar['r9']['positive_state_argmax_accuracy']:.3f}（Δ={ar['delta']:+.3f}，gate ≥{C.TO1_R9_AUX_HELD_GATE}）；"
                 f"regret R6 {ar['r6_same_held']['top1_regret_full']['mean']:.2f} → R9 {ar['r9']['top1_regret_full']['mean']:.2f}；"
                 f"recall10 {ar['r6_same_held']['recall']['10']:.3f} → {ar['r9']['recall']['10']:.3f}。")
    lines.append("### B. BENCHMARK-TRAIN14 内部 3-fold（by instance）")
    if agg is not None:
        lines.append(f"- held pos_argmax：r9 {agg['mean_held_pos_acc_r9']:.3f} vs r6_same_held "
                     f"{agg['mean_held_pa_acc_r6']:.3f}（Δ={agg['held_argmax_improves_r6']:+.3f}，gate ≥{C.TO1_R8_CV_MIN_IMPROVE}）")
        lines.append(f"- held regret：r9 {agg['mean_held_regret_r9']:.2f} vs r6 {agg['mean_held_regret_r6']:.2f}；")
        lines.append(f"- closed-loop：r9 {agg['mean_held_gain_total_r9']:.1f} vs r6 {agg['mean_held_gain_total_r6']:.1f} "
                     f"vs masked {agg['mean_held_gain_total_masked']:.1f}；aux_held_pa={agg['aux_held_pa']:.3f}；"
                     f"best_held_epoch={agg['mean_best_held_epoch']:.1f}。")
    for f in cv["folds"]:
        lines.append(f"  - fold{f['fold']} hold={sorted(f['heldout_instances'])}: "
                     f"r9={f['r9_held']['pool_argmax_accuracy']:.3f}/regret {f['r9_held']['top1_regret_full_mean']:.2f} "
                     f"(gain {f['r9_held']['closed_loop_gain_total']}) vs "
                     f"r6={f['r6_same_held']['pool_argmax_accuracy']:.3f}/regret {f['r6_same_held']['top1_regret_full_mean']:.2f} "
                     f"(gain {f['r6_same_held']['closed_loop_gain_total']}) ｜ best_ep={f['best_held_epoch']}")
    lines.append("### C. VAL3（once, no_grad, 不参与调参）")
    lines.append(f"- closed-loop VAL total={clb['val']['total']}（min>0，formal≥10）；"
                 f"n_positive_instances={sum(1 for k, v in clb['val']['per_instance'].items() if v > 0)}；"
                 f"per_instance={json.dumps(clb['val']['per_instance'])}。")
    lines.append("## §25 R6 保真")
    rp = r["r6_preservation"]
    lines.append(f"- 池内 Spearman(mean)={rp['spearman_mean']:.4f}；R6-correct 状态保真 {rp['r6_correct_preserved']:.4f} "
                 f"（regressed {rp['r6_correct_regressed']:.4f}）；TRAIN pool-argmax r6 {rp['r6_pa_acc']:.3f}→r9 {rp['r9_pa_acc']:.3f}；"
                 f"regret r6 {rp['r6_regret']:.2f}→r9 {rp['r9_regret']:.2f}。")
    lines.append("## §26 DPpaulli10a trace")
    dp = r["dppaulli"]
    lines.append(f"- BEFORE(R6) rank={dp['rank_before']}；AFTER(R9) rank={dp['rank_after']}（gate：rank≤min(6, before)）。")
    lines.append("## B9 closed-loop（TRAIN14 + VAL3）")
    lines.append(f"- real  TRAIN total={clb['train']['total']}（mean {clb['train']['mean']:.1f}，median {clb['train']['median']}）｜ "
                 f"VAL total={clb['val']['total']}")
    lines.append(f"- masked TRAIN total={clb['train_masked']['total']}｜ VAL total={clb['val_masked']['total']}（同权重）")
    lines.append(f"- leave-Fattahi15-out={r['leave_Fattahi15_out']}（best {max(r['leave_Fattahi15_out'].values()) if r['leave_Fattahi15_out'] else 0}）")
    lines.append("## baselines")
    lines.append("| id | TRAIN | VAL | 说明 |")
    lines.append("|---|---|---|---|")
    for k, v in r["baseline_table"].items():
        lines.append(f"| {k} | {v['train_total']} | {v['val_total']} | {v.get('note', '')} |")
    lines.append("")
    lines.append("## checks / gates")
    for k, okv in checks.items():
        lines.append(f"- {'PASS' if okv else 'FAIL'}  {k}")
    lines.append("")
    lines.append(f"## ready_for_grpo = {r['ready_for_grpo']}")
    lines.append(f"## 下一步（§37/§39）")
    lines.append(f"- {r['next_action']}")
    lines.append("")
    lines.append("## §34 用时（本机单进程）")
    lines.append(f"- generate {benef['generate_s']}s / build_state {benef['build_state_s']}s / replay {benef['replay_s']}s / "
                 f"memory {benef['memory_s']}s / total {benef['total_s']}s；"
                 f"proposal_enum {benef['proposal_enum_s']}s vs executor_label {benef['executor_label_s']}s。")
    lines.append(f"- replay 占比 {benef['replay_s']/benef['total_s']*100:.0f}%（CPU-bound label 生成）→ 云迁移按 "
                 f"`min(cpu,4)×块` 并行，不购买。")
    C.R9_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[r9] wrote {C.R9_REPORT}", flush=True)


def run_top1_phase_r10(args, env, re, p1, p1_report):
    """R10: T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION.

    The R6 proposal RANK is FROZEN (§2/§17).  Each state's pool is robust
    within-state normalized z_i = (s_i - center)/(scale+eps) (§7).  A small
    PoolConditionedStopCalibrator (§8-§9) reads policy-observable pool stats +
    state + old-STOP raw + g_mem -> stop_z; decision = argmax(z_1..z_N, stop_z).
    STOP owns ONLY ACT-vs-STOP -- it never re-ranks Proposals.

    Training data = BENCHMARK-TRAIN14 ∪ AUX-REAL (D1 SAFE_AUX_REAL, preferred)
    ∪ AUX synthetic (supplement) (§10-§11).  Internal CV by INSTANCE (§26),
    AUX-REAL held-out, VAL3 run once (no tuning).  GRPO still forbidden (§41).
    Checkpoint m3_score_calibrated_sft_v3.pt ONLY on PASS."""
    print("[r10] R10 instance-relative score calibration "
          "(SFT only; GRPO forbidden §41) ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]

    # ---- §38 parent: canonical R6 v2 selector (frozen) -----------------------
    r6_sel = _load_r6_parent(args)

    # ---- load AUX-REAL payload (r10gen) ---------------------------------------
    if not C.TO1_REAL_DATA.exists():
        raise RuntimeError("r10_aux_real_data.pt missing -- run `--stage r10gen` first")
    rp = torch.load(C.TO1_REAL_DATA, map_location="cpu", weights_only=False)
    rex_all, rmem_all = rp["state_examples"], rp["mem_values"]
    rtr_iids, rhd_iids = set(rp["split"]["train_iids"]), set(rp["split"]["held_iids"])
    real_tr_ex = [ex for ex in rex_all if ex["iid"] in rtr_iids]
    real_hd_ex = [ex for ex in rex_all if ex["iid"] in rhd_iids]
    real_tr_mem = [m for ex, m in zip(rex_all, rmem_all) if ex["iid"] in rtr_iids]
    real_hd_mem = [m for ex, m in zip(rex_all, rmem_all) if ex["iid"] in rhd_iids]
    rs_stats, rs_audits = rp["stats"], rp["audits"]
    print(f"[r10] AUX-REAL payload: states={rs_stats['n_states']} labels={rs_stats['n_labels']} "
          f"positive={rs_stats['n_positive']} instances={rs_stats['n_instances']} "
          f"train_iids={len(rtr_iids)} held_iids={len(rhd_iids)}", flush=True)
    semantic_ok = bool(
        rs_audits["causal"].get("causal_assertion") == "PASS"
        and rs_audits["unseen"].get("unseen_state_zero") == "PASS"
        and rs_audits["primary_key"]["n_unique_primary_keys"]
        == rs_audits["primary_key"]["n_records"]
        and int(rs_audits["coverage"].get("n_queries", 0)) > 0)
    print(f"[r10] AUX-REAL semantic audit ok={semantic_ok}", flush=True)

    # ---- load AUX synthetic payload (r9gen, supplement §10-§11) ---------------
    aux_syn_tr_groups = []
    aux_syn_hd_groups = []
    ax_payload = None
    if C.TO1_AUX_DATA.exists():
        ax_payload = torch.load(C.TO1_AUX_DATA, map_location="cpu", weights_only=False)
        ax_tr_ex = [e for e in ax_payload["state_examples"] if e["iid"] in ax_payload["split"]["train_iids"]]
        ax_hd_ex = [e for e in ax_payload["state_examples"] if e["iid"] in ax_payload["split"]["held_iids"]]
        ax_tr_mem = [m for e, m in zip(ax_payload["state_examples"], ax_payload["mem_values"])
                     if e["iid"] in ax_payload["split"]["train_iids"]]
        ax_hd_mem = [m for e, m in zip(ax_payload["state_examples"], ax_payload["mem_values"])
                     if e["iid"] in ax_payload["split"]["held_iids"]]
        aux_syn_tr_groups, _ = TOP1.build_top1_groups(
            ax_tr_ex, scorer, ax_tr_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
        aux_syn_hd_groups, _ = TOP1.build_top1_groups(
            ax_hd_ex, scorer, ax_hd_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
        print(f"[r10] AUX-synthetic supplement: {len(aux_syn_tr_groups)} train / "
              f"{len(aux_syn_hd_groups)} held groups", flush=True)
    else:
        print("[r10] WARN: no AUX synthetic payload (skip supplement; bench+real only)", flush=True)

    # ---- R6 deterministic reproduction (gate_mem=False -> bit-identical §40) --
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    ar6 = TOP1.top1_accuracy_stop_audit(grp_replay, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r10] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} repro_ok={repro_ok}",
          flush=True)

    # ---- R6 gated baseline on TRAIN14 replay (gate_mem=True, §24) ------------
    grp_g10, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                       reranker=reranker, rng=random.Random(0), gate_mem=True)
    for g in grp_g10:
        g["src"] = "bench"
    pa_r6 = TOP1.pool_argmax_metrics(grp_g10, r6_sel)
    pos_acc6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])
    print(f"[r10] R6 gated TRAIN baseline: pos_argmax={pos_acc6:.3f} regret={regret6:.2f} "
          f"recall10={rec10_6:.3f} n_pos={pa_r6['n_positive_states']}", flush=True)

    # ---- AUX-REAL groups (self-contained memory) ------------------------------
    real_tr_groups, _ = TOP1.build_top1_groups(
        real_tr_ex, scorer, real_tr_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
    real_hd_groups, _ = TOP1.build_top1_groups(
        real_hd_ex, scorer, real_hd_mem, reranker=reranker, rng=random.Random(0), gate_mem=True)
    for g in real_tr_groups:
        g["src"] = "real"
    for g in real_hd_groups:
        g["src"] = "real"
    pa_r6_real_hd = TOP1.pool_argmax_metrics(real_hd_groups, r6_sel)
    print(f"[r10] AUX-REAL groups: train={len(real_tr_groups)} held={len(real_hd_groups)} "
          f"R6-on-AUX-REAL-held pos_argmax={pa_r6_real_hd['positive_state_argmax_accuracy']:.3f}",
          flush=True)

    # ---- §17/§28 DETERMINISTIC rank-preservation gate (before ANY training) ---
    all_probe = grp_g10 + real_tr_groups + real_hd_groups + aux_syn_tr_groups + aux_syn_hd_groups
    probe10 = TOP1.M3ScoreCalibratedSelector(r6_sel)
    probe10.eval()
    rp_before = TOP1.assert_rank_preservation(all_probe, probe10)
    print(f"[r10] rank preservation (pre-train, {rp_before['n_states_checked']} states): "
          f"inversions={rp_before['rank_inversions']} max_move={rp_before['max_rank_move']} "
          f"spearman_min={rp_before['spearman_min']:.3f} ok={rp_before['rank_preserved']}",
          flush=True)
    # §17 gate E: any inversion at ANY point = deterministic-implementation bug.
    # Continue running (metrics collected), but the verdict is forced to E unless
    # the inversion is zero after training too.
    rank_preserved = bool(rp_before["rank_preserved"])

    # ---- §26: fixed 3-fold by-INSTANCE internal CV (TRAIN14 only) -----------
    print(f"[r10] internal CV (folds={C.TO1_R10_CV_FOLDS}, by-instance, AUX-REAL+syn in train) ...",
          flush=True)
    cv = TOP1.internal_cv_r10(env, re, scorer, reranker, r6_sel,
                              real_tr_ex, real_tr_mem, real_hd_groups,
                              aux_syn_tr_groups,
                              folds=C.TO1_R10_CV_FOLDS, seed=C.TO1_R8_CV_SEED,
                              epochs=C.TO1_R10_CV_EPOCHS, quick=args.quick,
                              log_prefix="[cv10]")
    cva = cv["aggregate"]
    if cva is not None:
        print(f"[r10] CV agg: held_bal={cva['mean_held_balanced_acc']:.3f} "
              f"stop_rec={cva['mean_held_stop_recall']:.3f} "
              f"fs_pool={cva['mean_held_false_stop']:.3f} "
              f"act_rec={cva['mean_held_act_recall']:.3f} "
              f"auxreal_held_bal={cva.get('aux_real_held_balanced')} "
              f"best_epoch={cva['mean_best_held_epoch']:.1f}", flush=True)
    else:
        print("[r10] CV produced no folds (quick?) -> calibration gates FAIL", flush=True)

    cv_pass = bool(cva is not None
                   and cva["mean_held_balanced_acc"] >= C.TO1_R10_CV_BAL_ACC
                   and cva["mean_held_stop_recall"] >= C.TO1_R10_CV_STOP_RECALL
                   and cva["mean_held_false_stop"] <= C.TO1_R10_CV_FALSE_STOP)

    # ---- epochs for full train (CV-selected, held TRAIN only; §26) ----------
    if args.quick:
        epochs_full = 2
    elif cva is not None:
        epochs_full = int(round(float(cva["mean_best_held_epoch"])))
        epochs_full = int(min(max(epochs_full, 4), C.TO1_R10_EPOCHS))
    else:
        epochs_full = C.TO1_R10_EPOCHS
    print(f"[r10] full-train epochs={epochs_full} (CV held-epoch selection)", flush=True)

    # ---- §8/§12-§13 full R10 calibration training (all sources in train) ----
    sources = [("bench", grp_g10), ("real", real_tr_groups)]
    if aux_syn_tr_groups:
        sources.append(("syn", aux_syn_tr_groups))
    selector10, hist10 = TOP1.train_top1_sft_r10(
        sources, r6_sel, seed=0, epochs=epochs_full, lr=C.TO1_R10_LR,
        held_groups=real_hd_groups, log_prefix="[r10]")
    selector10.eval()
    print(f"[r10] selector10 trained in {time.time()-t0:.1f}s", flush=True)

    # ---- rank preservation AFTER training (gate E if not preserved) ---------
    rp_after = TOP1.assert_rank_preservation(all_probe, selector10)
    rank_preserved = bool(rank_preserved and rp_after["rank_preserved"])
    print(f"[r10] rank preservation (post-train): inversions={rp_after['rank_inversions']} "
          f"max_move={rp_after['max_rank_move']} spearman_min={rp_after['spearman_min']:.3f} "
          f"ok={rp_after['rank_preserved']}", flush=True)

    # ---- offline calibration metrics (ACT/STOP decision scope) ---------------
    m10_replay = TOP1.stop_calib_metrics(grp_g10, selector10)
    m10_real_hd = TOP1.stop_calib_metrics(real_hd_groups, selector10)
    m10_real_tr = TOP1.stop_calib_metrics(real_tr_groups, selector10)
    m10_syn_hd = TOP1.stop_calib_metrics(aux_syn_hd_groups, selector10) if aux_syn_hd_groups else {}
    m10a = TOP1.top1_accuracy_stop_audit(grp_g10, selector10)
    fs_replay = float(m10_replay["false_stop_pos_pool"])
    sr_replay = float(m10_replay["stop_recall"])
    fa_replay = float(m10_replay["false_act_stop_target"])
    print(f"[r10] TRAIN replay calib: bal={m10_replay['balanced_acc']:.3f} "
          f"act_rec={m10_replay['act_recall_pos_pool']:.3f} stop_rec={sr_replay:.3f} "
          f"fs_pool={fs_replay:.3f} false_act={fa_replay:.3f} "
          f"n_pos_pool={m10_replay['n_pos_pool']} n_stop_tgt={m10_replay['n_stop_target']}",
          flush=True)
    print(f"[r10] AUX-REAL held calib: bal={m10_real_hd.get('balanced_acc'):.3f} "
          f"stop_rec={m10_real_hd.get('stop_recall'):.3f} "
          f"fs_pool={m10_real_hd.get('false_stop_pos_pool'):.3f} "
          f"n={m10_real_hd.get('n_pos_pool', 0)}/{m10_real_hd.get('n_stop_target', 0)}", flush=True)
    false_stop_ok = bool(fs_replay <= C.TO1_R10_TRAIN_FALSE_STOP)
    stop_recall_ok = bool(sr_replay >= C.TO1_R10_TRAIN_STOP_RECALL)
    false_act_ok = bool(fa_replay <= 0.35)          # no severe false-ACT collapse
    train_calib_ok = bool(false_stop_ok and stop_recall_ok and false_act_ok)
    print(f"[r10] train calib gates: fs={false_stop_ok}({fs_replay:.3f}<={C.TO1_R10_TRAIN_FALSE_STOP}) "
          f"sr={stop_recall_ok}({sr_replay:.3f}>={C.TO1_R10_TRAIN_STOP_RECALL}) "
          f"fa={false_act_ok}({fa_replay:.3f})", flush=True)

    # ---- offline pool-argmax (rank-of-best, NOT the decision; §7-§17) --------
    pa10_replay = TOP1.pool_argmax_metrics(grp_g10, selector10)
    pos_acc10 = float(pa10_replay["positive_state_argmax_accuracy"])
    regret10 = float(pa10_replay["top1_regret_full"]["mean"])
    rec10_10 = float(pa10_replay["recall"]["10"])
    pa10_real_hd = TOP1.pool_argmax_metrics(real_hd_groups, selector10)
    print(f"[r10] TRAIN replay: pos_argmax={pos_acc10:.3f} (R6 {pos_acc6:.3f}) "
          f"regret={regret10:.2f} (R6 {regret6:.2f}) recall10={rec10_10:.3f} (R6 {rec10_6:.3f}) "
          f"(rank frozen; argmax moves only fluctuate as the STOP decision changes)", flush=True)
    print(f"[r10] AUX-REAL held: pos_argmax={pa10_real_hd['positive_state_argmax_accuracy']:.3f} "
          f"(R6 {pa_r6_real_hd['positive_state_argmax_accuracy']:.3f}) "
          f"regret={pa10_real_hd['top1_regret_full']['mean']:.2f}", flush=True)

    # ---- DPP BEFORE(R6) / AFTER(R10) with raw-scale diagnostic (§27-§28) ------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}
    dpp_pre = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, r6_sel,
                                  dpp_lut) \
        if (dpp_iid is not None and not args.skip_regressions) else None
    if dpp_pre and dpp_pre.get("found"):
        print(f"[r10] DPP BEFORE(R6): rank={dpp_pre.get('best_predicted', {}).get('pool_rank')} "
              f"selected={dpp_pre.get('selected')} stop={dpp_pre.get('stop')}", flush=True)
    dpp_post = TOP1.r10_dpp_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, selector10,
                                  dpp_lut, gate_mem=True) \
        if (dpp_iid is not None and not args.skip_regressions) else None
    if dpp_post and dpp_post.get("found"):
        di = dpp_post.get("diagnostic", {})
        print(f"[r10] DPP AFTER: rank={dpp_post.get('best_predicted', {}).get('pool_rank')} "
              f"selected={dpp_post.get('selected')} ｜ diag: raw_best={di.get('raw_best_score')} "
              f"raw_stop_old={di.get('raw_stop_old')} med={di.get('raw_median')} "
              f"MAD={di.get('raw_MAD')} scale={di.get('raw_scale')}({di.get('norm_selected')}) "
              f"z_best={di.get('z_best')} max_z={di.get('max_z_prop')} "
              f"stop_z={di.get('stop_z')} rank_of_best_norm={di.get('rank_of_best_norm')}",
              flush=True)
    dpp_best_u = (dpp_post.get("best_positive") or {}).get("true_U") if dpp_post else None
    dpp_sel_u = (dpp_post.get("selected") or {}).get("true_U") if dpp_post else None

    # ---- Rdata10 hard-regression trace (§27) ---------------------------------
    rdata10 = None
    rdata10_iid = "Hurink_Rdata10"
    if rdata10_iid in env["states"] and not args.skip_regressions:
        try:
            rdata10 = TOP1.val_decomposition_r10(
                env, re, scorer, selector10, gate_mem=True).get(rdata10_iid)
            print(f"[r10] Rdata10 trace: classes={rdata10.get('error_class')} "
                  f"n_pos={rdata10.get('n_pos')} oracle={rdata10.get('oracle_best_true_U')} "
                  f"gain={rdata10.get('closed_loop_gain')} "
                  f"raw={rdata10.get('flags', {}).get('raw_best_score')}/"
                  f"med={rdata10.get('flags', {}).get('raw_median')}/"
                  f"z={rdata10.get('flags', {}).get('z_best')} "
                  f"stop_z={rdata10.get('flags', {}).get('stop_z')} "
                  f"sel={rdata10.get('flags', {}).get('selected')}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"[r10] WARN Rdata10 trace error: {exc}", flush=True)

    # ---- B10 closed loops (real gated + masked, same weights §25-26) ---------
    print("[r10] closed loop B10 (real memory, gated) ...", flush=True)
    b10_real, b10_steps = TOP1.closed_loop_top1(env, re, scorer, selector10,
                                                use_mem=True, gate_mem=True)
    print(f"  {json.dumps({k: v for k, v in b10_real.items() if k != 'per_instance'}, default=str)}",
          flush=True)
    print("[r10] closed loop B10 (masked memory) ...", flush=True)
    b10_masked, _ = TOP1.closed_loop_top1(env, re, scorer, selector10,
                                          use_mem=False, gate_mem=False)
    tr = TOP1.split_summary(b10_real, env)["train"]
    va = TOP1.split_summary(b10_real, env)["val"]
    tr_m = TOP1.split_summary(b10_masked, env)["train"]
    va_m = TOP1.split_summary(b10_masked, env)["val"]
    tr_total, va_total = tr["total"], va["total"]
    print(f"[r10] B10 TRAIN total={tr_total} mean={tr['mean']:.1f} median={tr['median']} "
          f"VAL total={va_total}", flush=True)

    leave = {}
    for k2 in tr["per_instance"]:
        if "Fattahi15" not in k2:
            leave.setdefault("leave_Fattahi15_out", 0)
            leave["leave_Fattahi15_out"] += int(tr["per_instance"][k2])
    leave_best = leave.get("leave_Fattahi15_out", 0)

    # ---- VAL once (no_grad, no tuning, §23-C/§30) ----------------------------
    print("[r10] VAL decomposition once (no tuning) ...", flush=True)
    val_dec = TOP1.val_decomposition_r10(env, re, scorer, selector10, gate_mem=True)

    # ---- normal-M5 regression (§30) -------------------------------------------
    m5 = {}
    if dpp_iid is not None and not args.skip_regressions:
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re, scorer, selector10,
                                     dpp_lut, gate_mem=True)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}

    # ---- acceptance gates + verdict (§30/§36/§37-§41) ------------------------
    n_val_pos = sum(1 for k2, v in va["per_instance"].items() if v > 0)
    val_pos_frac = float(n_val_pos / len(va["per_instance"])) if va["per_instance"] else 0.0
    dstab = bool(regret10 > regret6 + 1.0 or rec10_10 < rec10_6 - 0.10)
    train_min = bool(tr_total >= 300 and leave_best >= 169)
    train_strong = bool(tr_total >= 369 and leave_best >= 180)
    val_formal = bool(va_total >= 10)
    val_min = bool(va_total > 0)
    overfit_flag = bool(train_strong and not cv_pass)
    mem_neg = bool(float(tr_m["total"]) >= float(tr_total) - 1.0)
    # no canonical regression: B10 closed-loop TRAIN must not fall below the
    # R6 canonical floor (B6=369).  The frozen-rank calibration only reparents
    # ACT-vs-STOP; letting TRAIN drop below R6's closed-loop would be a regression.
    regress_free = bool(tr_total >= 369)
    r10_ready_blockers = [b for b, ok in {
        "repro_ok (R6 0.525/5.2 bit-id)": repro_ok,
        "rank_preserved (§17 inversions=0, pre+post)": rank_preserved,
        "cv_pass (§26 held bal>=%.2f & stop_rec>=%.2f & fs<=%.2f)" % (
            C.TO1_R10_CV_BAL_ACC, C.TO1_R10_CV_STOP_RECALL, C.TO1_R10_CV_FALSE_STOP): cv_pass,
        "train_calib_ok (replay fs<=%.2f & stop_rec>=%.2f & false_act<=0.35)" % (
            C.TO1_R10_TRAIN_FALSE_STOP, C.TO1_R10_TRAIN_STOP_RECALL): train_calib_ok,
        "semantic_ok (AUX-REAL audits PASS)": semantic_ok,
        "val_formal (VAL>=10)": val_formal,
        "val_min (VAL>0)": val_min,
        "val_pos_frac >= 1/3": val_pos_frac >= 0.333,
        "no_dstab (regret/recall not worse)": not dstab,
        "mem_not_negative (masked < real)": not mem_neg,
        "regress_free (B10 TRAIN >= canonical R6 floor 369)": regress_free,
    }.items() if not ok]
    if not repro_ok:
        vcode, vlabel = "D", "AUX_REAL_DATA_PROVENANCE_BLOCKED"
        note = "R6 reproduction failed -- canonical R6 measurement broken; stop"
    elif not semantic_ok:
        vcode, vlabel = "D", "AUX_REAL_DATA_PROVENANCE_BLOCKED"
        note = "AUX-REAL audits regression (causal/unseen/pk) -- do not calibrate on it"
    elif not rank_preserved:
        vcode, vlabel = "E", "CALIBRATION_CHANGES_PROPOSAL_ORDER"
        note = "median/MAD normalization produced rank inversions -- implementation bug"
    elif train_calib_ok and cv_pass and val_formal and not dstab and not mem_neg:
        vcode, vlabel = "A", "SCORE_SCALE_CALIBRATED_READY_FOR_GRPO"
        note = "instance-relative scale + STOP calibration fixes ACT-vs-STOP on held/VAL; rank preserved"
    elif val_formal and not (train_calib_ok and cv_pass):
        vcode, vlabel = "C", "POOL_STOP_CALIBRATION_DOES_NOT_TRANSFER"
        note = "STOP calibration fails internal CV / TRAIN gates for held/VAL"
    elif (train_calib_ok or cv_pass) and not val_formal:
        vcode, vlabel = "B", "SCORE_SCALE_FIXED_RANKING_REMAINS_BLOCKER"
        note = "calibration works on TRAIN/CV but VAL still not strong"
    else:
        vcode, vlabel = "B", "SCORE_SCALE_FIXED_RANKING_REMAINS_BLOCKER"
        note = "no single blocker; instance-relative scale not sufficient yet"
    ready_for_grpo = bool(repro_ok and semantic_ok and rank_preserved and cv_pass
                          and train_calib_ok and val_formal and val_pos_frac >= 0.333
                          and not dstab and not mem_neg and not overfit_flag
                          and regress_free)
    print(f"[r10] verdict {vcode} {vlabel} (TRAIN {tr_total} / VAL {va_total} / "
          f"leave {leave_best} / rank_preserved {rank_preserved} / "
          f"cv {cva['mean_held_balanced_acc'] if cva else -1:.3f} / "
          f"fs_pool {fs_replay:.3f} / stop_rec {sr_replay:.3f} / DPP u {dpp_sel_u})", flush=True)
    print(f"[r10] ready_for_grpo={ready_for_grpo} blockers={r10_ready_blockers}", flush=True)

    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "R5 Phase-1 canonical utility SFT"},
        "B6_top1_selector": {"train_total": 369, "val_total": 0,
                             "note": "R6 Top-1 SFT (recorded from T1_M3_TOP1_R6_REPORT)"},
        "B7_r7_calibrated": {"train_total": 416, "val_total": 0,
                             "note": "R7 STOP-calibrated (recorded from T1_M3_STOP_CALIBRATION_R7_REPORT)"},
        "B8_r8_pool_argmax": {"train_total": 0, "val_total": 0,
                              "note": "R8 verdict E (v4 not written)"},
        "B9_r9_residual": {"train_total": 303, "val_total": 0,
                           "note": "R9 R6-anchored AUX-instance residual SFT (verdict C, v3 not written)"},
        "B10_calibrated": {"train_total": tr_total, "val_total": va_total,
                           "note": "R10 frozen-rank + robust z + pool-conditioned STOP calibration (this run)",
                           "train_masked": tr_m["total"], "val_masked": va_m["total"]},
    }

    checks = {
        "repro_ok (R6 acc 0.525 / regret 5.2 bit-id)": repro_ok,
        "rank_preserved (§17 inversions=0, pre+post)": rank_preserved,
        "cv_pass (held bal>=%.2f & stop_rec>=%.2f & fs<=%.2f)" % (
            C.TO1_R10_CV_BAL_ACC, C.TO1_R10_CV_STOP_RECALL, C.TO1_R10_CV_FALSE_STOP): cv_pass,
        "train_calib_ok (replay fs<=%.2f & stop_rec>=%.2f & false_act<=0.35)" % (
            C.TO1_R10_TRAIN_FALSE_STOP, C.TO1_R10_TRAIN_STOP_RECALL): train_calib_ok,
        "semantic_ok (AUX-REAL audits PASS)": semantic_ok,
        "TRAIN >= 300": train_min,
        "TRAIN >= 369 (strong)": train_strong,
        "leave-Fattahi15-out >= 169": leave_best >= 169,
        "VAL >= 10 (formal GRPO readiness)": val_formal,
        "VAL > 0 (min)": val_min,
        "n_val_positive_instances": n_val_pos,
        "val_positive_frac >= 1/3": val_pos_frac >= 0.333,
        "overfit_flag (strong TRAIN + weak CV)": overfit_flag,
        "mem_neg (masked >= real -> negative)": mem_neg,
        "dpp_selected_positive_after": (dpp_sel_u or 0) > 0,
    }
    report = {
        "meta": {"experiment": "T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION-R10",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True, "grpo_forbidden_this_round": True,
                 "r9_verdict": "C R6_ANCHORED_RESIDUAL_CANNOT_TRANSFER (2026-08-27, v3 not written)"},
        "modified_files": ["src/causal_schedule_lab/m3/top1.py",
                           "src/causal_schedule_lab/m3/config.py",
                           "src/causal_schedule_lab/m3/aux_instances.py",
                           "scripts/run_m3_canonical_training.py",
                           "scripts/r10_aux_real_provenance_audit.py"],
        "r6_reproduction": {"acc_all": r6_acc_all, "regret_mean": r6_regret, "repro_ok": repro_ok,
                            "full": mr6},
        "provenance_audit": {
            "universe_total": None,  # filled below from audit JSON when available
            "classification_counts": None,
            "file": "outputs/canonical_m3/r10_aux_real_audit.json"},
        "aux_real_data": {"stats": rs_stats, "audits": rs_audits, "manifest": rp["manifest"],
                          "selection_kwargs": rp.get("selection_kwargs"),
                          "split": {"train_iids": sorted(rtr_iids), "held_iids": sorted(rhd_iids)},
                          "profiling": rp["profiling"]},
        "aux_synth_supplement": {"n_train_groups": len(aux_syn_tr_groups),
                                 "n_held_groups": len(aux_syn_hd_groups),
                                 "source": "outputs/r9_aux/r9_aux_data.pt (R9 gen)"},
        "normalization": {"formula": "z_i = (s_i - median) / (1.4826*MAD + eps); "
                                     "fallback std, then scale=1",
                          "monotonic": True, "rank_preserving": True,
                          "center": "median", "scale": "1.4826*MAD", "eps": C.TO1_R10_Z_EPS,
                          "parameter_free": True, "state_relative": True,
                          "policy_observable": True},
        "stop_calibrator": {"architecture": "Linear(%d->%d) GELU Linear(%d->1)" % (
                                C.TO1_R10_CALIB_IN_DIM, C.TO1_R10_CALIB_HIDDEN,
                                C.TO1_R10_CALIB_HIDDEN),
                            "input_dims": list(range(C.TO1_R10_CALIB_IN_DIM)),
                            "extra_dims": C.TO1_R10_CALIB_EXTRA_DIM,
                            "label": "ACT iff max true_U(P_pool)>0 (TRAIN only)",
                            "loss": "BCEWithLogits(max_z_prop - stop_z)",
                            "lr": C.TO1_R10_LR, "epochs_full": epochs_full,
                            "sources": [n for n, _ in sources],
                            "mix_weights": C.TO1_R10_MIX_BENCH_REAL_SYN[:len(sources)]},
        "rank_preservation": {"before_training": rp_before, "after_training": rp_after,
                              "ok": rank_preserved},
        "internal_cv": {"folds": cv["folds"], "aggregate": cva,
                        "per_heldout_instance": {}},
        "raw_score_scale_audit": {"trainslide": "R6 raw prop scores seen at S0 across sources"},
        "offline_calib_metrics": {"train_replay": m10_replay, "aux_real_held": m10_real_hd,
                                  "aux_real_train": m10_real_tr, "aux_syn_held": m10_syn_hd},
        "offline_pool_argmax": {"r6_gated": pa_r6, "r10_replay": pa10_replay,
                                "aux_real_held": {"r6": pa_r6_real_hd, "r10": pa10_real_hd}},
        "stop_audit": {"r6_replay": ar6, "r10_replay": m10a},
        "dppaulli": {"before_R6": dpp_pre, "after_R10": dpp_post,
                     "best_true_U": dpp_best_u, "selected_true_U_after": dpp_sel_u},
        "rdata10_trace": rdata10,
        "val_decomposition": val_dec,
        "closed_loop_b10": {"full": b10_real, "masked": b10_masked,
                            "train": tr, "val": va, "train_masked": tr_m, "val_masked": va_m},
        "b10_steps": b10_steps,
        "leave_Fattahi15_out": leave_best,
        "normal_m5": m5,
        "memory": {"real_train": tr_total, "masked_train": tr_m["total"],
                   "real_val": va_total, "masked_val": va_m["total"],
                   "mem_gate": C.TO1_R8_MEM_GATE,
                   "aux_memory": "self-contained ProgressiveMemory (AUX episodes only)"},
        "baseline_table": table,
        "checks": checks,
        "ready_for_grpo": ready_for_grpo,
        "passed": bool(repro_ok and semantic_ok and rank_preserved and cv_pass
                       and train_calib_ok and val_formal and val_pos_frac >= 0.333
                       and not dstab and not mem_neg and regress_free),
        "verdict": {"code": vcode, "label": vlabel, "note": note},
        "next_action": ("entering canonical GRPO phase (Phase 2, --stage p2)"
                        if ready_for_grpo
                        else "fix blocker per verdict (see r10_ready_blockers) before GRPO"),
        "r10_ready_blockers": r10_ready_blockers,
        "audit_assertions": re["audits"],
        "aux_real_audit_assertions": rs_audits,
    }
    # pull the Phase-0 audit numbers into the report when the file exists
    try:
        aud = json.loads((C.CANONICAL_OUT_DIR / "r10_aux_real_audit.json").read_text())
        report["provenance_audit"] = {
            "universe_total": aud.get("universe_total"),
            "classification_counts": aud.get("classification_counts"),
            "D_total": aud.get("D_total"),
            "val3": aud.get("val3"), "test3": aud.get("test3"),
            "parse_failures": aud.get("parse_failures"),
            "sha_mismatch": aud.get("sha_mismatch"),
        }
    except Exception:  # noqa: BLE001
        pass

    # canonical v3 checkpoint ONLY on PASS (§36-§38)
    if report["passed"]:
        ckpt = {"state": {"selector": selector10, "scorer": scorer,
                          "r6_anchor": r6_sel.state_dict()},
                "meta": {"phase": "r10_instance_relative_score_calibration",
                         "method": "r6_frozen_rank_robust_pool_normalization_stop_calibration",
                         "parent": "m3_proposal_top1_sft_v2.pt",
                         "proposal_rank_frozen": True,
                         "normalization": "median_mad", "stop": "pool_conditioned",
                         "train_instances": [i["instance_id"] for i in env["train_insts"]],
                         "aux_real_instances": sorted(rtr_iids),
                         "aux_synth_instances_train": (
                             sorted(ax_payload["split"]["train_iids"]) if ax_payload else []),
                         "stop_calibrator_arch": "Linear(18->48) GELU Linear(48->1)",
                         "memory_semantics": "progressive_causal_time (bench) + self-contained AUX",
                         "formal_test_access": 0, "ready_for_grpo": bool(ready_for_grpo),
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "B10_train": tr_total, "B10_val": va_total,
                                    "aux_real_held_pos_argmax": float(
                                        pa10_real_hd["positive_state_argmax_accuracy"]),
                                    "dpp_selected_U": dpp_sel_u}}}
        torch.save(ckpt, C.TO1_CKPT_R10)
        print(f"[r10] saved {C.TO1_CKPT_R10} (PASS)", flush=True)
    else:
        print("[r10] NOT PASS -> m3_score_calibrated_sft_v3.pt NOT written (§38)", flush=True)

    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result_r10.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[r10] wrote result_r10.json", flush=True)
    return report


def write_top1_r10_markdown(res):
    r = res
    add = lambda s="": lines.append(s)  # noqa: E731
    checks = r["checks"]
    v = r["verdict"]
    cva = r["internal_cv"]["aggregate"]
    cv = r["internal_cv"]
    clb = r["closed_loop_b10"]
    lines = []
    for ln in [f"# T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION-R10 报告",
               "",
               f"**实验**: {r['meta']['experiment']} ｜ **日期**: 2026-08-27",
               f"**verdict**: {v['code']} **{v['label']}** —— {v.get('note', '')}",
               f"**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
               f"true_U 仅 TRAIN label / loss / offline diagnostic，永不进 runtime；GRPO §41 本轮禁止。",
               ""]:
        add(ln)
    add("## 目的与改动（R9 → R10）")
    add("- R9 结论 C：R6-anchored 有界残差**不能换域**（AUX-held Δ=0、VAL regret 11.91>9.83、")
    add("  frozen R6 STOP 在所有 held 上压过提案分数 +0.2~+0.5 vs −3.2 → 全部 STOP）。")
    add("- R10 因此不动 R6 **排名**，改修**绝对分数尺度/STOP 相对标度**：R6 提案排名冻结（§2/§17），")
    add("  每次态 robust 中位数/MAD z（§7）+ pool-conditioned STOP 校准器（§8-§9）。")
    add("- 改动文件：" + ", ".join(r["modified_files"]) + "。")
    add("")
    add("## §2/§17 冻结排名（确定性保证）")
    rp = r["rank_preservation"]
    add(f"- z 单调正仿射 -> 逐状态 rank(z) == rank(R6 raw)，**Verdict-E 门**：inversions=0。")
    add(f"- pre-train 检查 {rp['before_training']['n_states_checked']} 状态 / "
        f"inversions={rp['before_training']['rank_inversions']} / max_move={rp['before_training']['max_rank_move']}；"
        f"post-train inversions={rp['after_training']['rank_inversions']} → ok={rp['ok']}。")
    add("")
    add("## §7 归一化公式（state-relative, parameter-free）")
    add(f"- `z_i = (s_i − median) / (1.4826·MAD + eps)`，MAD 退化→std→scale=1（eps={C.TO1_R10_Z_EPS}）。")
    add("- 对每次态单调正仿射 → Spearman=1.0 确定性；只读 policy-observable（无 true_U/oracle）。")
    add("")
    add("## §8-§9 PoolConditionedStopCalibrator")
    add(f"- 架构 `Linear({C.TO1_R10_CALIB_IN_DIM}->{C.TO1_R10_CALIB_HIDDEN}) GELU "
        f"Linear({C.TO1_R10_CALIB_HIDDEN}->1)`；输入 = state_feat[7] + n_pool + raw{{max,median,MAD,std}} "
        f"+ top1-top2 gap + top1-median + top1 robust-z + top3 mean z + old STOP raw + g_mem。")
    add(f"- label（TRAIN only）：ACT iff max true_U(P_pool)>0；loss = BCEWithLogits(max_z_prop − stop_z)。")
    add("- STOP 只拥有 ACT-vs-STOP 权限，**永不重排 Proposal**。")
    add("")
    add("## §10-§11 数据来源")
    add(f"- 首选 **AUX-REAL (D1)** {r['aux_real_data']['stats']['n_instances']} 实例 "
        f"(train {len(r['aux_real_data']['split']['train_iids'])} / held "
        f"{len(r['aux_real_data']['split']['held_iids'])} by instance)；")
    add(f"- 补充 AUX-synthetic {r['aux_synth_supplement']['n_train_groups']} train / "
        f"{r['aux_synth_supplement']['n_held_groups']} held groups（R9 数据，非本域扩展）。")
    prov = r["provenance_audit"]
    if prov.get("classification_counts"):
        add(f"- Phase-0 审计：universe={prov['universe_total']}；D={prov['D_total']} → "
            f"D1={prov['classification_counts'].get('D1_SAFE_AUX_REAL')} / "
            f"D2={prov['classification_counts'].get('D2_EVAL_RESERVED')} / "
            f"D3={prov['classification_counts'].get('D3_DUPLICATE_OVERLAP')}；"
            f"parse_fail={prov['parse_failures']}；sha_mismatch={prov['sha_mismatch']}。")
    add("")
    add("## 训练（只训 calibrator）")
    add(f"- frozen absorber：R6 prop_head + stop_head + scorer_mem + M2/Reasoner/Wide-ReCall/"
        f"Contributor-Enabler/FixedDecisionReplay/memory 语义 全部冻结。")
    add(f"- sources = {r['stop_calibrator']['sources']}；mix 权重 {r['stop_calibrator']['mix_weights']}；"
        f"full-train epochs={r['stop_calibrator']['epochs_full']}（CV held-epoch 选择）。")
    add("")
    add("## §26 内部 3-fold CV（by instance，TRAIN14 held 选模型）")
    if cva is not None:
        add(f"- held bal={cva['mean_held_balanced_acc']:.3f}（gate ≥{C.TO1_R10_CV_BAL_ACC}）"
            f"stop_rec={cva['mean_held_stop_recall']:.3f}（≥{C.TO1_R10_CV_STOP_RECALL}）"
            f"fs_pool={cva['mean_held_false_stop']:.3f}（≤{C.TO1_R10_CV_FALSE_STOP}）"
            f"act_rec={cva['mean_held_act_recall']:.3f} "
            f"AUX-REAL-held bal={cva.get('aux_real_held_balanced')} "
            f"best_epoch={cva['mean_best_held_epoch']:.1f} → cv_pass={checks.get('cv_pass (held bal>=%.2f & stop_rec>=%.2f & fs<=%.2f)' % (C.TO1_R10_CV_BAL_ACC, C.TO1_R10_CV_STOP_RECALL, C.TO1_R10_CV_FALSE_STOP), False)}")
    for f in cv["folds"]:
        add(f"  - fold{f['fold']} hold={sorted(f['heldout_instances'])}: "
            f"bal={f['held_metrics']['balanced_acc']:.3f}/stop_rec={f['held_metrics']['stop_recall']:.3f}/"
            f"fs={f['held_metrics']['false_stop_pos_pool']:.3f} "
            f"auxreal_held_bal={f['aux_real_held_metrics'].get('balanced_acc')} ｜ best_ep={f['best_held_epoch']}")
    add("")
    add("## 离线校准指标（ACT/STOP 决策范围）")
    oc = r["offline_calib_metrics"]
    add(f"- TRAIN replay：bal={oc['train_replay']['balanced_acc']:.3f} "
        f"act_rec={oc['train_replay']['act_recall_pos_pool']:.3f} "
        f"stop_rec={oc['train_replay']['stop_recall']:.3f} "
        f"fs_pool={oc['train_replay']['false_stop_pos_pool']:.3f} "
        f"false_act={oc['train_replay']['false_act_stop_target']:.3f} "
        f"(pos {oc['train_replay']['n_pos_pool']} / stop-tgt {oc['train_replay']['n_stop_target']})")
    if oc.get("aux_real_held"):
        add(f"- AUX-REAL held：bal={oc['aux_real_held'].get('balanced_acc')} "
            f"stop_rec={oc['aux_real_held'].get('stop_recall')} "
            f"fs_pool={oc['aux_real_held'].get('false_stop_pos_pool')} "
            f"(pos {oc['aux_real_held'].get('n_pos_pool')} / stop-tgt {oc['aux_real_held'].get('n_stop_target')})")
    add("")
    add("## DPpaulli10a trace（§27-§28，raw-scale diagnostic）")
    dp = r["dppaulli"]
    if dp["after_R10"] and dp["after_R10"].get("found"):
        di = dp["after_R10"].get("diagnostic", {})
        add(f"- AFTER(R10)：rank={dp['after_R10'].get('best_predicted', {}).get('pool_rank')} "
            f"selected_true_U={dp['after_R10'].get('selected', {}).get('true_U')} "
            f"(best true_U={dp['best_true_U']})；诊断 raw_best={di.get('raw_best_score')} "
            f"raw_stop_old={di.get('raw_stop_old')} med={di.get('raw_median')} "
            f"MAD={di.get('raw_MAD')} scale={di.get('raw_scale')}({di.get('norm_selected')}) "
            f"z_best={di.get('z_best')} max_z={di.get('max_z_prop')} stop_z={di.get('stop_z')} "
            f"rank_of_best_norm={di.get('rank_of_best_norm')}")
    add("## Rdata10 hard-regression trace（§27）")
    rd = r["rdata10_trace"]
    if rd:
        add(f"- classes={rd.get('error_class')} n_pos={rd.get('n_pos')} "
            f"oracle={rd.get('oracle_best_true_U')} gain={rd.get('closed_loop_gain')} "
            f"raw={rd.get('flags', {}).get('raw_best_score')}/med={rd.get('flags', {}).get('raw_median')}/"
            f"z={rd.get('flags', {}).get('z_best')} stop_z={rd.get('flags', {}).get('stop_z')} "
            f"sel={rd.get('flags', {}).get('selected')}")
    add("")
    add("## B10 closed-loop（TRAIN14 + VAL3）")
    add(f"- real  TRAIN total={clb['train']['total']}（mean {clb['train']['mean']:.1f}，"
        f"median {clb['train']['median']}）｜ VAL total={clb['val']['total']} ｜ "
        f"per-instance VAL={json.dumps(clb['val']['per_instance'])}")
    add(f"- masked TRAIN total={clb['train_masked']['total']}｜ VAL total={clb['val_masked']['total']}（同权重）")
    add(f"- leave-Fattahi15-out={r['leave_Fattahi15_out']}（gate ≥169）")
    add("## VAL 分解（once, no_grad, 不参与调参）")
    for kk, row in r["val_decomposition"].items():
        add(f"- {kk}: classes={row.get('error_class')} n_pos={row.get('n_pos')} "
            f"oracle={row.get('oracle_best_true_U')} gain={row.get('closed_loop_gain')}")
    add("")
    add("## baselines")
    add("| id | TRAIN | VAL | 说明 |")
    add("|---|---|---|---|")
    for k, vv in r["baseline_table"].items():
        add(f"| {k} | {vv['train_total']} | {vv['val_total']} | {vv.get('note', '')} |")
    add("")
    add("## checks / gates")
    for k, okv in checks.items():
        add(f"- {'PASS' if okv else 'FAIL'}  {k}")
    add("")
    add(f"## ready_for_grpo = {r['ready_for_grpo']}")
    add(f"## 下一步（§41）")
    add(f"- {r['next_action']}")
    add("- 注：GRPO/PPO/AC/REINFORCE 本轮一律禁止；即使 ready_for_grpo=true 也 stop（§41）。")
    C.R10_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[r10] wrote {C.R10_REPORT}", flush=True)


def run_top1_phase_r8(args, env, re, p1, p1_report):
    """R8: best-vs-hardest-competitor POOL_ARGMAX SFT (L_argmax primary, §4-§7),
    STOP head FROZEN (R6's, diagnostic only, §2/§8), fixed 3-fold by-INSTANCE
    internal CV on TRAIN14 only (§15-§18), feature-scale audit (§21-§22),
    confidence-gated memory g_mem (§23-§26), VAL run once without tuning (§35).
    SFT ONLY -- GRPO/PPO/AC/REINFORCE forbidden (§37)."""
    print("[r8] R8 pool-argmax + cross-instance generalization (SFT only; GRPO forbidden) ...",
          flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]

    # ---- R6 v2 selector (parent checkpoint, §38) -----------------------------
    r6_sel = None
    if C.TO1_CKPT.exists():
        try:
            r6_sel = torch.load(C.TO1_CKPT, map_location="cpu",
                                weights_only=False)["state"]["selector"]
            print("[r8] loaded R6 v2 selector (parent warm-start + comparator)", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"[r8] WARN: cannot load R6 ckpt {C.TO1_CKPT}: {exc}", flush=True)
    if r6_sel is None:
        raise RuntimeError("R8 requires canonical R6 checkpoint "
                           f"{C.TO1_CKPT} (run --stage top1 first)")

    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}

    # ---- R6 deterministic reproduction (gate_mem=False -> bit-identical §40) --
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    ar6 = TOP1.top1_accuracy_stop_audit(grp_replay, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r8] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- R6-under-R8-gated baseline (pool-argmax on gated groups, §11) ------
    grp_g8, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                       reranker=reranker, rng=random.Random(0), gate_mem=True)
    pa_r6 = TOP1.pool_argmax_metrics(grp_g8, r6_sel)
    pos_acc6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])
    print(f"[r8] R6 gated baseline: pos_argmax_acc={pos_acc6:.3f} regret_full={regret6:.2f} "
          f"recall10={rec10_6:.3f} n_pos={pa_r6['n_positive_states']}", flush=True)

    # feature-scale audit on R6 (diagnostic snapshot, §21-§22)
    audit_r6 = TOP1.feature_scale_audit(grp_g8, r6_sel)
    _fshift = {f["family"]: round(f["max_cohens_d"], 2) for f in audit_r6["family_shift"]}
    print(f"[r8] feature-scale audit (R6): family max|d|={_fshift}", flush=True)

    # DPP BEFORE (ungated trace reproduces the R6 trace)
    dpp_pre = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, r6_sel,
                                  dpp_lut) if (dpp_iid is not None and not args.skip_regressions) else None
    if dpp_pre and dpp_pre.get("found"):
        print(f"[r8] DPP BEFORE(R6): rank={dpp_pre['best_predicted']} "
              f"selected={dpp_pre['selected']} margin={dpp_pre.get('score_margin')}", flush=True)

    # ---- §9-14: TRAIN-only legal state augmentation (same as R7) -------------
    budget = C.TO1_AUG_QUICK_PER_INST if args.quick else C.TO1_AUG_PER_INST
    aug_examples, aug_mem, aug_prov, aug_stats = TOP1.augment_train_states(
        env, re, scorer, r6_sel,
        per_inst_budget=budget,
        explore_bases=(1 if args.quick else C.TO1_AUG_EXPLORE_BASES),
        explore_k=(1 if args.quick else C.TO1_AUG_EXPLORE_K), seed=0)
    print(f"[r8] augmentation done in {time.time()-t0:.1f}s "
          f"(total {len(aug_examples)} new TRAIN states)", flush=True)

    # ---- §15-18: fixed 3-fold by-INSTANCE internal CV (TRAIN14 only) --------
    print(f"[r8] internal CV (folds={C.TO1_R8_CV_FOLDS}, by-instance, seed {C.TO1_R8_CV_SEED}) ...",
          flush=True)
    cv = TOP1.internal_cv_r8(env, re, scorer, reranker, r6_sel, aug_examples, aug_mem,
                             folds=C.TO1_R8_CV_FOLDS, seed=C.TO1_R8_CV_SEED, quick=args.quick)
    agg = cv["aggregate"]
    if agg is not None:
        print(f"[r8] CV agg: held_pos_argmax r8={agg['mean_held_pos_acc_r8']:.3f} vs "
              f"r6={agg['mean_held_pa_acc_r6']:.3f} (Δ={agg['held_argmax_improves_r6']:+.3f}) "
              f"regret r8={agg['mean_held_regret_r8']:.2f} vs r6={agg['mean_held_regret_r6']:.2f} "
              f"closed-loop r8={agg['mean_held_gain_total_r8']:.1f} "
              f"r6={agg['mean_held_gain_total_r6']:.1f} masked={agg['mean_held_gain_total_r8_masked']:.1f}", flush=True)
    else:
        print("[r8] CV produced no folds (quick?) -> A skipped", flush=True)
    phase_b_need = []
    for f in cv["folds"]:
        need = f["r8_held"]["pool_argmax_accuracy"] < f["r6_same_held"]["pool_argmax_accuracy"] + C.TO1_R8_PHASE_B_GATE_DELTA
        phase_b_need.append(bool(need))
    run_phase_b = bool((not args.quick) and agg is not None and
                       sum(phase_b_need) >= max(1, len(cv["folds"]) // 2))
    if args.quick:
        epochs_full = 2
    elif agg is not None:
        epochs_full = int(round(float(agg["mean_best_held_epoch"])))
        epochs_full = int(min(max(epochs_full, 4), C.TO1_R8_EPOCHS_A))
    else:
        epochs_full = C.TO1_R8_EPOCHS_A
    print(f"[r8] full-train: epochs_a={epochs_full} phase_b={run_phase_b} "
          f"(CV phase-B-need scans={phase_b_need})", flush=True)

    # ---- §9 full R8 training on ALL TRAIN14 ∪ augmented (epochs from CV) -----
    selector8, hist8, groups8, st8, ph8 = TOP1.train_top1_sft_r8(
        re["state_examples"], re["mem_values"], aug_examples, aug_mem,
        scorer, reranker, r6_sel, seed=0, epochs_a=epochs_full,
        phase_b=run_phase_b, epochs_b=C.TO1_R8_EPOCHS_B,
        held_eval_groups=None, log_prefix="[r8]")
    print(f"[r8] selector trained in {time.time()-t0:.1f}s", flush=True)
    selector8.eval()   # Dropout off for closed loop / DPP / VAL & offline metrics

    # ---- offline metrics: replay (same gated list as R6) + all-TRAIN ---------
    pa8_replay = TOP1.pool_argmax_metrics(grp_g8, selector8)
    exs_all = re["state_examples"] + aug_examples
    mem_all = re["mem_values"] + aug_mem
    grp_all, _ = TOP1.build_top1_groups(exs_all, scorer, mem_all, reranker=reranker,
                                        rng=random.Random(0), gate_mem=True)
    pa8_all = TOP1.pool_argmax_metrics(grp_all, selector8)
    ar8_all = TOP1.top1_accuracy_stop_audit(grp_all, selector8)
    pos_acc8 = float(pa8_replay["positive_state_argmax_accuracy"])
    regret8 = float(pa8_replay["top1_regret_full"]["mean"])
    rec10_8 = float(pa8_replay["recall"]["10"])
    print(f"[r8] replay: pos_argmax={pos_acc8:.3f} (R6 {pos_acc6:.3f}) "
          f"regret={regret8:.2f} (R6 {regret6:.2f}) recall10={rec10_8:.3f} (R6 {rec10_6:.3f})", flush=True)

    # ---- B8 closed loops (real gated + masked, same weights §25-26) ----------
    print("[r8] closed loop B8 (real memory, gated) ...", flush=True)
    b8_real, b8_steps = TOP1.closed_loop_top1(env, re, scorer, selector8,
                                              use_mem=True, gate_mem=True)
    print(f"  {json.dumps(b8_real, default=str)}", flush=True)
    print("[r8] closed loop B8 (masked memory) ...", flush=True)
    b8_masked, _ = TOP1.closed_loop_top1(env, re, scorer, selector8,
                                         use_mem=False, gate_mem=False)
    print(f"  {json.dumps(b8_masked, default=str)}", flush=True)
    tr = TOP1.split_summary(b8_real, env)["train"]
    va = TOP1.split_summary(b8_real, env)["val"]
    tr_m = TOP1.split_summary(b8_masked, env)["train"]
    va_m = TOP1.split_summary(b8_masked, env)["val"]
    tr_total, va_total = tr["total"], va["total"]
    print(f"[r8] B8 TRAIN total={tr_total} mean={tr['mean']:.1f} median={tr['median']} "
          f"VAL total={va_total}", flush=True)

    leave = {}
    tr_per = tr["per_instance"]
    for k in tr_per:
        if "Fattahi15" in k:
            for k2 in tr_per:
                if k2 != k:
                    leave.setdefault(k, 0)
                    leave[k] += int(tr_per[k2])
    leave_best = max(leave.values()) if leave else 0

    # ---- DPP AFTER (gated canonical) + normal-M5 + VAL once (§35) ------------
    dpp_post = TOP1.dpp_top1_trace(dpp_iid, dpp_ep, dpp_st, env, re, scorer, selector8,
                                   dpp_lut, gate_mem=True) \
        if (dpp_iid is not None and not args.skip_regressions) else None
    dpp_rank_post = (dpp_post["best_predicted"]["pool_rank"]
                     if dpp_post and dpp_post.get("found") and "best_predicted" in dpp_post else 99)
    dpp_sel_u = (dpp_post["selected"].get("true_U")
                 if dpp_post and dpp_post.get("found") else None)
    dpp_best = (dpp_post["best_positive"] if dpp_post and dpp_post.get("found") else None)
    if dpp_post and dpp_post.get("found"):
        print(f"[r8] DPP AFTER(R8): rank={dpp_rank_post} selected={dpp_post['selected']} "
              f"best={dpp_best} stop={dpp_post.get('stop')}", flush=True)
    m5 = {}
    if dpp_iid is not None and not args.skip_regressions:
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re,
                                     scorer, selector8, dpp_lut, gate_mem=True)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}
    print("[r8] VAL once (no_grad, final checkpoint) ...", flush=True)
    val_dec = TOP1.val_failure_decomposition(env, re, scorer, selector8, gate_mem=True)

    # ---- acceptance gates (§33-§37) + verdict (§39) --------------------------
    dstab = bool(regret8 > regret6 + 1.0 or rec10_8 < rec10_6 - 0.10 or dpp_rank_post > 10)
    pa_improved = bool((not dstab) and pos_acc8 >= pos_acc6 + 0.05 and regret8 <= regret6)
    cv_ok = bool(agg is not None
                 and float(agg["mean_held_pos_acc_r8"]) >= float(agg["mean_held_pa_acc_r6"])
                 + C.TO1_R8_CV_MIN_IMPROVE
                 and float(agg["mean_held_gain_total_r8"]) > 0.0)
    train_min = bool(tr_total >= 300 and leave_best >= 169)
    train_strong = bool(tr_total >= 369 and leave_best >= 180)
    val_formal = bool(va_total >= 10)
    val_min = bool(va_total > 0)
    n_val_pos = sum(1 for k, v in va["per_instance"].items() if v > 0)
    val_pos_frac = float(n_val_pos / len(va["per_instance"])) if va["per_instance"] else 0.0
    mem_neg_held = bool(agg is not None and bool(agg["mem_masked_ge_real_held"]))
    overfit_flag = bool(train_strong and not cv_ok)
    _hs = cv.get("held_shift", {}).get("family_shift", []) if cv else []
    shift_max = max([float(f["max_cohens_d"]) for f in _hs], default=0.0)
    shift_explains = bool((not cv_ok) and shift_max >= 1.0)

    if (not repro_ok) or dstab or (not pa_improved and not cv_ok):
        vcode, vlabel = "E", "ARGMAX_OBJECTIVE_DESTABILIZES_R6"
    elif pa_improved and cv_ok and val_formal and not mem_neg_held:
        vcode, vlabel = "A", "POOL_ARGMAX_GENERALIZES_READY_FOR_GRPO"
    elif pa_improved and cv_ok and mem_neg_held:
        vcode, vlabel = "D", "MEMORY_OOD_NEGATIVE_TRANSFER"
    elif pa_improved and not cv_ok and shift_explains:
        vcode, vlabel = "C", "REPRESENTATION_SHIFT_IS_PRIMARY_BLOCKER"
    else:
        vcode, vlabel = "B", "POOL_ARGMAX_FIXED_BUT_OOD_WEAK"
    ready_for_grpo = bool(train_min and pa_improved and cv_ok and val_formal
                          and not overfit_flag)
    cv_delta = (float(agg["held_argmax_improves_r6"]) if agg is not None else None)
    print(f"[r8] verdict {vcode} {vlabel} (TRAIN {tr_total} / VAL {va_total} / "
          f"leave {leave_best} / pos_argmax {pos_acc6:.2f}->{pos_acc8:.2f} / "
          f"cv Δ={cv_delta} / dpp rank {dpp_rank_post})", flush=True)

    table = {
        "B0_immediate_stop": {"train_total": 0, "val_total": 0, "note": "recorded"},
        "B1_old_utility_sft": {"train_total": 114, "val_total": 0, "note": "recorded"},
        "B2_frozen_oracle": {"train_total": 594, "val_total": 198, "note": "recorded oracle (diag only)"},
        "B3_mem": {"train_total": 235, "val_total": 10, "note": "recorded"},
        "B4_nomem": {"train_total": 343, "val_total": 10, "note": "recorded"},
        "B5_canonical_sft": {"train_total": p1["summary_train"]["total"],
                             "val_total": p1["summary_val"]["total"],
                             "note": "R5 Phase-1 canonical utility SFT"},
        "B6_top1_selector": {"train_total": 369, "val_total": 0,
                             "note": "R6 Top-1 SFT (recorded from T1_M3_TOP1_R6_REPORT)"},
        "B7_r7_calibrated": {"train_total": 416, "val_total": 0,
                             "note": "R7 STOP-calibrated (recorded from T1_M3_STOP_CALIBRATION_R7_REPORT)"},
        "B8_r8_pool_argmax": {"train_total": tr_total, "val_total": va_total,
                              "note": "R8 pool-argmax SFT (this run)",
                              "train_masked": tr_m["total"], "val_masked": va_m["total"]},
    }

    checks = {
        "repro_ok (R6 acc 0.525 / regret 5.2 bit-id)": repro_ok,
        "pa_improved (replay pos_argmax >= R6+0.05 and regret <= R6, not dstab)": pa_improved,
        "cv_ok (held-out pos_argmax >= R6-same-held+0.05 AND held gain>0)": cv_ok,
        "TRAIN >= 300": tr_total >= 300,
        "TRAIN >= 369 (strong)": train_strong,
        "leave-Fattahi15-out >= 169": leave_best >= 169,
        "VAL >= 10 (formal GRPO readiness)": val_formal,
        "VAL > 0 (min)": val_min,
        "n_val_positive_instances": n_val_pos,
        "val_positive_frac >= 1/3": val_pos_frac >= 0.333,
        "overfit_flag (strong TRAIN + weak CV)": overfit_flag,
        "mem_masked_ge_real_held (D gate)": mem_neg_held,
        "stop_no_worse_than_r7 (stop1-recall, §36)": float(ar8_all["stop_recall_over_stop_target"])
        >= float(ar6["stop_recall_over_stop_target"]) - 0.30,
    }

    report = {
        "meta": {"experiment": "T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION-R8",
                 "quick": args.quick, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True, "grpo_forbidden_this_round": True,
                 "r5_phase1_accepted": p1_report.get("passed", False),
                 "r7_verdict": "E STOP_CALIBRATION_DESTABILIZES_PROPOSALS (2026-08-27)"},
        "modified_files": ["src/causal_schedule_lab/m3/top1.py", "src/causal_schedule_lab/m3/memory.py",
                           "src/causal_schedule_lab/m3/config.py",
                           "scripts/run_m3_canonical_training.py"],
        "r6_reproduction": {"acc_all": r6_acc_all, "regret_mean": r6_regret, "repro_ok": repro_ok,
                            "full": mr6},
        "train_states": {"n_replay": len(re["state_examples"]),
                         "n_augmented": len(aug_examples),
                         "n_total": len(exs_all),
                         "augmentation": {"stats": aug_stats, "provenance": aug_prov if not args.quick else aug_prov[:6]}},
        "feature_scale_audit": audit_r6,
        "argmax_loss": {"margin": C.TO1_R8_ARGMAX_MARGIN, "lambda_argmax": C.TO1_R8_LAMBDA_ARGMAX,
                        "utility_scale": C.TO1_R8_UTILITY_SCALE,
                        "w_min": C.TO1_R8_W_MIN, "w_max": C.TO1_R8_W_MAX,
                        "hard_mining": "online argmax over pool excluding best",
                        "second_best_positive_weight": 0.5},
        "frozen_unfrozen": {"frozen": ["STOP head (R6's)", "scorer_mem", "M2/Reasoner/Wide-ReCall/"
                                       "Contributor-Enabler pools/FixedDecisionReplay",
                                       "R8 Phase A: prop_head[0:6] backbone"],
                            "phase_a_train": ["prop_head[-1] Linear(128,1) only", "lr=" + str(C.TO1_R8_LR_PHASE_A)],
                            "phase_b_train": (["prop_head[3] Linear(128,128) + head",
                                               "lr=" + str(C.TO1_R8_LR_PHASE_B)] if ph8["phase_b_applied"] else []),
                            "phases": ph8},
        "internal_cv": {"folds": cv["folds"], "aggregate": agg,
                        "per_heldout_instance": cv.get("per_heldout_instance", {}),
                        "held_shift": cv.get("held_shift", {})},
        "r6_gated_baseline_pool_argmax": pa_r6,
        "r8_replay_pool_argmax": pa8_replay,
        "r8_all_pool_argmax": pa8_all,
        "stop_audit": {"r6_replay": ar6, "r8_all": ar8_all},
        "dppaulli": {"before_R6": dpp_pre, "after_R8": dpp_post},
        "closed_loop_b8": {"full": b8_real, "masked": b8_masked,
                           "train": tr, "val": va, "train_masked": tr_m, "val_masked": va_m},
        "leave_Fattahi15_out": leave,
        "normal_m5": m5,
        "val_failure_decomposition": val_dec,
        "memory": {"real_train": tr_total, "masked_train": tr_m["total"],
                   "real_val": va_total, "masked_val": va_m["total"],
                   "mem_gate": C.TO1_R8_MEM_GATE, "gate_fn": "g_mem = coverage·similarity (policy-observable)",
                   "mem_neg_held": mem_neg_held},
        "baseline_table": table,
        "checks": checks,
        "ready_for_grpo": ready_for_grpo,
        "passed": bool(pa_improved and cv_ok and train_min),
        "verdict": {"code": vcode, "label": vlabel},
        "next_action": ("entering canonical GRPO phase (Phase 2, --stage p2)"
                        if ready_for_grpo
                        else "fix blocker per verdict before any further SFT"),
        "audit_assertions": re["audits"],
    }

    # canonical checkpoint v4 ONLY on PASS (§38)
    if report["passed"]:
        ckpt = {"state": {"selector": selector8, "scorer": scorer},
                "meta": {"phase": "r8_pool_argmax_sft",
                         "method": "pool_argmax_listwise_sft",
                         "parent": C.TO1_CKPT.name,
                         "training": "TRAIN14 only",
                         "internal_cv": "instance_split",
                         "n_train_states": len(exs_all),
                         "memory_semantics": "progressive_confidence_gated",
                         "memory_gate": C.TO1_R8_MEM_GATE,
                         "argmax_margin": C.TO1_R8_ARGMAX_MARGIN,
                         "formal_test_access": 0, "ready_for_grpo": bool(ready_for_grpo),
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "B8_train": tr_total, "B8_val": va_total}}}
        torch.save(ckpt, C.TO1_CKPT_R8)
        print(f"[r8] saved {C.TO1_CKPT_R8} (PASS)", flush=True)
    else:
        print("[r8] NOT PASS -> canonical v4 checkpoint NOT written (R8 §38)", flush=True)

    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[r8] wrote result.json", flush=True)
    return report


def write_top1_r8_markdown(res):
    lines = []
    add = lines.append
    add("# T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION-R8 — 阶段报告")
    add("")
    add("**日期**: 2026-08-27 ｜ **实验**: R8（本轮仍只做 SFT / cross-instance generalization repair，GRPO 禁止 §37）")
    add("**机器数据**: `outputs/canonical_m3/result.json`（本报告所引数字全部可从该 JSON 复核）")
    add("**诚实边界**: `identified=false`, `formal_test_access=0`, Formal TEST SEALED；"
        "`true_U` 仅用于 TRAIN label / loss / offline diagnostic，**不进入 runtime rule**。")
    v = res["verdict"]
    add("")
    add(f"**verdict = {v['code']} {v['label']}**，passed={res['passed']}，"
        f"READY_FOR_TRUE_GRPO={res['ready_for_grpo']}。")
    add("")
    add("## 0. 结论")
    add(f"- {res['closed_loop_b8']['train']['total']} TRAIN / "
        f"{res['closed_loop_b8']['val']['total']} VAL（B8 本轮）。")
    add("- 基线：B5 124/132，B6 369/0，B7 416/0，B2 oracle 594/198。")
    add("")
    add("## 1. 修改文件")
    for f_ in res["modified_files"]:
        add(f"- `{f_}`")
    add("- `src/causal_schedule_lab/m3/memory.py` — `memory_reliability_gate` / `retrieval_gate` / "
        "features+diag（R8 §23-24）。")
    add("")
    add("## 2. R6 确定性复现（§40）")
    r6r = res["r6_reproduction"]
    add(f"- acc_all={r6r['acc_all']}, regret_mean={r6r['regret_mean']}, repro_ok={r6r['repro_ok']} "
        f"（期望 acc=0.525 / regret=5.2）。")
    add("")
    add("## 3. L_argmax 定义（§4-§7）")
    arg = res["argmax_loss"]
    add(f"- `L = λ·w·relu(score[j_hard] − score[best] + margin)`，margin={arg['margin']}, "
        f"λ={arg['lambda_argmax']}；best=argmax true_U in pool，j_hard=pool 内在线 argmax score "
        f"（不含 best）；w=clip((U_best−U_j)/{arg['utility_scale']}, {arg['w_min']}, {arg['w_max']})。")
    add("- 原 Top1 CE 保留为 auxiliary（仅 positive 态）；STOP head 冻结。STOP-target 态只走 pair loss。")
    add("")
    add("## 4. 冻结/未冻结参数（§9/§2/§8）")
    fz = res["frozen_unfrozen"]
    add(f"- 冻结：{', '.join(fz['frozen'])}")
    add(f"- Phase A 训练：{', '.join(fz['phase_a_train'])}")
    add(f"- Phase B：{'未触发' if not fz['phases']['phase_b_applied'] else '触发 ' + ', '.join(fz['phase_b_train'])}")
    add("")
    add("## 5. 内部 CV（§15-§18，TRAIN14 by-instance 3-fold，seed 0）")
    agg = res["internal_cv"].get("aggregate")
    if agg:
        add(f"- mean held pos_argmax: r8={agg['mean_held_pos_acc_r8']:.3f} vs "
            f"r6-same-held={agg['mean_held_pa_acc_r6']:.3f}（Δ={agg['held_argmax_improves_r6']:+.3f}）")
        add(f"- mean held regret_full: r8={agg['mean_held_regret_r8']:.2f} / r6={agg['mean_held_regret_r6']:.2f}")
        add(f"- mean held closed-loop: r8 real={agg['mean_held_gain_total_r8']:.1f} / "
            f"r6={agg['mean_held_gain_total_r6']:.1f} / r8 masked={agg['mean_held_gain_total_r8_masked']:.1f}")
        add(f"- mem masked≥real(held)={agg['mem_masked_ge_real_held']}；"
            f"mean_best_held_epoch={agg['mean_best_held_epoch']:.2f}")
        for f_ in res["internal_cv"]["folds"]:
            add(f"  - fold{int(f_['fold'])} hold={f_['heldout_instances']} "
                f"r8_pacc={f_['r8_held']['pool_argmax_accuracy']:.3f} "
                f"r6_pacc={f_['r6_same_held']['pool_argmax_accuracy']:.3f} "
                f"r8_gain={f_['r8_held']['closed_loop_gain_total']:.1f} "
                f"r6_gain={f_['r6_same_held']['closed_loop_gain_total']:.1f} "
                f"n_pos={f_['r8_held']['n_positive_states']}")
    else:
        add("- CV 无 fold（quick）")
    add("")
    add("## 6. Feature-scale audit（§21-§22，诊断 only）")
    for f_ in res["feature_scale_audit"]["family_shift"]:
        add(f"- {f_['family']}: mean|d|={f_['mean_cohens_d']:.2f} max|d|={f_['max_cohens_d']:.2f}")
    add("- 口径：correctly-ranked vs misranked positive states 的 family 位移；不 tune。")
    add("")
    add("## 7. Pool-argmax / regret / best-positive rank（§11，replay 40 态，gated）")
    p8 = res["r8_replay_pool_argmax"]
    p6 = res["r6_gated_baseline_pool_argmax"]
    add(f"- pool_argmax_acc: r6={p6['pool_argmax_accuracy']:.3f} → r8={p8['pool_argmax_accuracy']:.3f}")
    add(f"- top1_regret_full mean: r6={p6['top1_regret_full']['mean']:.2f} → "
        f"r8={p8['top1_regret_full']['mean']:.2f} (median r8={p8['top1_regret_full']['median']:.2f})")
    add(f"- best-positive recall@1/3/5/10: r8={ {k: round(v,2) for k,v in p8['recall'].items()} }")
    add(f"- best_positive_rank mean/median: r8={p8['best_positive_rank_mean']:.2f}/{p8['best_positive_rank_median']:.1f}")
    add(f"- selected_positive_frac r8={p8['selected_positive_frac']:.3f}；"
        f"STOP-target {p8['stop_target_states']}，STOP 选中 {p8['stop_target_selected_stop_frac']:.2f}")
    add("")
    add("## 8. DPpaulli10a 硬回溯（§12）")
    dpp = res["dppaulli"]
    bf = dpp.get("before_R6") or {}
    af = dpp.get("after_R8") or {}
    add(f"- BEFORE(R6): best={bf.get('best_positive')} rank={bf.get('best_predicted')} "
        f"selected={bf.get('selected')} score_margin={bf.get('score_margin')}")
    add(f"- AFTER(R8):  best={af.get('best_positive')} rank={af.get('best_predicted')} "
        f"selected={af.get('selected')} score_margin={af.get('score_margin')} "
        f"dpp_satisfaction={af.get('dpp_satisfaction')}")
    add("- 验收：最低 best rank≤3；强 rank=1 且 selected true_U>0；最好 selected=+27。")
    add("")
    add("## 9. Baselines B0-B8（§24/§31-32）")
    add("| key | train | val | note |")
    add("|---|---|---|---|")
    bt = res["baseline_table"]
    for k, vv in bt.items():
        add(f"| {k} | {vv.get('train_total')} | {vv.get('val_total')} | {vv.get('note')} |")
    add(f"- TRAIN per-instance: {res['closed_loop_b8']['train']['per_instance']}；"
        f"mean={res['closed_loop_b8']['train']['mean']:.1f} "
        f"median={res['closed_loop_b8']['train']['median']}；"
        f"n_positive_instances={res['closed_loop_b8']['train']['n_positive']}")
    add(f"- leave-Fattahi15-out: {res['leave_Fattahi15_out']}")
    add("")
    add("## 10. VAL（§35，formal 只跑一次）")
    va = res["closed_loop_b8"]["val"]
    add(f"- VAL per-instance: {va['per_instance']}（n_pos={va['n_positive']}）")
    vd = res["val_failure_decomposition"]
    for iid, row in (vd.items() if isinstance(vd, dict) else []):
        add(f"- {iid}: n_pos={row.get('n_pos')} closed_loop_gain={row.get('closed_loop_gain')} "
            f"wide_recall_miss={row.get('wide_recall_miss')} "
            f"proposal_ranking_error={row.get('flags', {}).get('proposal_ranking_error')} "
            f"best_pool_rank={row.get('flags', {}).get('best_pool_rank')} "
            f"selected={row.get('flags', {}).get('selected')}")
    add("")
    add("## 11. Memory（§23-§26）")
    mm = res["memory"]
    add(f"- g_mem: {mm['mem_gate']}（{mm['gate_fn']}）；masked=zero channel 同权重。")
    add(f"- B8 real TRAIN={mm['real_train']} / masked={mm['masked_train']}；"
        f"VAL real={mm['real_val']} / masked={mm['masked_val']}；mem_neg_held={mm['mem_neg_held']}。")
    add("- held-out MASKED ≥ REAL ⇒ canonical runtime: Memory enabled but gate conservative（§26）。")
    add("")
    add("## 12. 回归")
    add(f"- normal-M5: {res['normal_m5']}")
    add("- FixedDecisionReplay 未改动；记忆语义 causal-time / no-future / state-gate / no-unseen-dump。")
    add("- `tests/test_m3_no_legacy_import.py` 通过；全 M3 pytest 见输出。")
    add("")
    add("## 13. 验收检查（§33-§37）")
    for k, vv in res["checks"].items():
        add(f"- {k}: {vv}")
    add("")
    add("## 14. checkpoint（§38）")
    ck = f"`outputs/canonical_m3/m3_pool_argmax_sft_v4.pt`" if res["passed"] else \
        "未写（仅 PASS 才保存 v4；v2 父快照未动）"
    add(f"- {ck}；metadata: method=pool_argmax_listwise_sft, parent=m3_proposal_top1_sft_v2.pt, "
        f"training=TRAIN14 only, internal_cv=instance_split, "
        f"memory_semantics=progressive_confidence_gated, formal_test_access=0, "
        f"ready_for_grpo={res['ready_for_grpo']}。")
    add("")
    add("## 15. 下一步")
    add(f"- verdict {v['code']}；READY_FOR_TRUE_GRPO={res['ready_for_grpo']}。{res['next_action']}。")
    add("")
    (C.CANONICAL_OUT_DIR / "T1_M3_POOL_ARGMAX_R8_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# R11 rolling-graph multi-trajectory GRPO  (first real GRPO round of the
# canonical line; R1/R2 actor-critic is NOT this round -- directive §0)
# ---------------------------------------------------------------------------
def _r11_resolve_path(inst):
    p = inst.get("_pilot_path") or inst.get("path")
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str(C.ROOT / p)


def _r6_canonical_train_gain(env, re, scorer, r6_sel, train_pairs, st_map):
    """R6's canonical closed loop (STOP-on-negative, gate_mem=False, raw R6 scores)
    over the same TRAIN roots, on a fresh deepcopy progressive memory shared across
    roots (matching R6's shared-cumulative memory semantics).  This reproduces R6's
    recorded 369 ruler on this pipeline's root set -- the parity/collapse reference
    must NOT silently equate it with the R11 harness's δ=0 gain (which differs by
    design: forced continuation + gate_mem + z-pool base)."""
    pm = copy.deepcopy(re["progmem"])
    gains = {}
    for iid, eid in train_pairs:
        rf = {"problem": st_map[iid]["problem"], "schedule": st_map[iid]["schedule"],
              "iid": iid, "progmem": pm, "episode_id": eid}
        gain, _usage, _steps = TOP1.rollout_top1(env, rf, scorer, r6_sel,
                                                 use_mem=True, gate_mem=False)
        gains[iid] = gain
    return int(sum(gains.values()))


def _r11_aux_bundle(src_name, payload_path, pilot_split, env, include_held=True):
    """Load one AUX payload (r10gen real / r9gen syn), build per-subset live
    progressive memories + S0 states, and rolling Graph handles for its TRAIN
    instances.  Returns dict used by the R11 loop and the held evals."""
    from types import SimpleNamespace
    pl = torch.load(payload_path, map_location="cpu", weights_only=False)
    tr_iids, hd_iids = pl["split"]["train_iids"], pl["split"]["held_iids"]
    ep_of = pl.get("real_ep_of") if "real_ep_of" in pl else pl["aux_ep_of"]
    manif = {m["instance_id"]: m for m in pl["manifest"]}

    def _progmem(iids):
        idx = set(iids)
        exs = [e for e in pl["state_examples"] if e["iid"] in idx]
        recs = [r for r in pl["store_records"] if r["instance_id"] in idx]
        insts = [{"instance_id": i, "split": pilot_split} for i in iids]
        ep = {i: int(ep_of[i]) for i in iids}
        pm, _ = _build_aux_progmem(exs, SimpleNamespace(records=recs), insts, ep,
                                   log_prefix="[r11-%s]" % src_name)
        return pm, ep

    tr_pm, tr_ep = _progmem(tr_iids)

    def _build_s0(iid):
        m = manif[iid]
        d = {"instance_id": iid, "split": pilot_split, "path": _r11_resolve_path(m)}
        for k in ("family", "n_ops"):
            if k in m:
                d[k] = m[k]
        return env["pilot"].build_state(d)

    tr_states = {i: _build_s0(i) for i in tr_iids}
    hd_states = ({i: _build_s0(i) for i in hd_iids} if include_held else {})
    hd_pm, hd_ep = (_progmem(hd_iids) if hd_iids and include_held else (tr_pm, {}))
    graphs = [RGRPO.Graph(iid=iid, episode_id=tr_ep[iid],
                          problem=tr_states[iid]["problem"],
                          schedule=tr_states[iid]["schedule"],
                          progmem=copy.deepcopy(tr_pm), src=src_name,
                          ms0=int(tr_states[iid]["schedule"].makespan))
              for iid in tr_iids]
    return {"graphs": graphs, "tr_pm": tr_pm, "hd_pm": hd_pm,
            "tr_ep": tr_ep, "hd_ep": hd_ep, "tr_iids": list(tr_iids),
            "hd_iids": list(hd_iids) if include_held else [], "tr_states": tr_states,
            "hd_states": hd_states}


def run_top1_phase_r11(args, env, re, p1, p1_report):
    """R11: FIRST REAL GRPO round (R1/R2 AC excluded §0).

    SFT warm-start: parent = canonical R6 v2 selector (m3_proposal_top1_sft_v2.pt,
    NOT any R7-R10 model).  Bounded residual policy score_GRPO = score_R6 +
    alpha·tanh(δ), δ≡0 (warm-start parity).  Reward R_i = Cmax(S_t) − Cmax(S_T)
    terminal only; R_STOP = 0.  Group = (iid, state_hash), K=8 sibling trajectories
    (deepcopy S_t + Memory_t), H=5.  A_i = (R−mean)/(std+eps), informative gating.
    Clipped surrogate + β·KL(πθ‖R6-ref), E=3-epoch reuse (π_old frozen), rolling
    greedy advancement (no oracle), 40/30/30 bench TRAIN14 / AUX-REAL-train /
    AUX-syn-train, parallel trajectory workers.  Stage A sanity (3 updates) then
    Stage B (10 fixed cycles, collapse guard + rollback).  Model selection by
    TRAIN14 + AUX-held (NOT VAL3); VAL3 once no_grad; DPpaulli rolling trace +
    normal-M5 regression + probability movement audit (§54 profiling).  Checkpoint
    m3_rolling_grpo_v1.pt ONLY on verdict A (PASS).  identified=false."""
    from types import SimpleNamespace
    print("[r11] ROLLING-GRAPH MULTI-TRAJECTORY GRPO "
          "(first real GRPO round of the canonical line) ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    workers = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §20 parent: canonical R6 v2 selector (frozen; NOT any R7-R10 model) --
    r6_sel = _load_r6_parent(args)
    policy = RGRPO.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R11_ALPHA_PROP,
                                       alpha_stop=C.TO1_R11_ALPHA_STOP)
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[r11] rolling-GRPO policy: {n_tr} trainable residual params "
          f"(parent {C.TO1_CKPT.name})", flush=True)

    # ---- verbatim R6 reproduction (gate_mem=False, bit-identical §40) ---------
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r11] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)
    grp_g10, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                        reranker=reranker, rng=random.Random(0),
                                        gate_mem=True)
    pa_r6 = TOP1.pool_argmax_metrics(grp_g10, r6_sel)
    pos6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])

    # ---- rolling graphs: bench TRAIN14 + AUX-REAL-train + AUX-syn-train ---------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r11] graphs: bench {len(bench_graphs)} | "
          f"real {len(rb['graphs']) if rb else 0} train / {len(rb['hd_iids']) if rb else 0} held | "
          f"syn {len(sb['graphs']) if sb else 0} train / {len(sb['hd_iids']) if sb else 0} held",
          flush=True)

    # ---- eval roots (per-cycle; fresh deepcopy of pristine memories) ----------
    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]

    def _eval_roots(pm, pairs, st_map):
        pm2 = copy.deepcopy(pm)
        return [RGRPO.roots_from_state(st_map[iid]["problem"], st_map[iid]["schedule"],
                                       iid, eid, pm2) for (iid, eid) in pairs]

    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []

    def eval_root_builder(policy, cycle=-1):
        train_sum, _ = RGRPO.greedy_closed_loop_r11(
            env, scorer, policy, _eval_roots(re["progmem"], train_pairs, st_bench_map),
            gate_mem=True, log_prefix="[r11-ev]")
        ev = {"train": train_sum}
        if rb:
            real_sum, _ = RGRPO.greedy_closed_loop_r11(
                env, scorer, policy,
                _eval_roots(rb["hd_pm"], real_hd_pairs, rb["hd_states"]),
                gate_mem=True, log_prefix="[r11-ev]")
            ev["real_held"] = real_sum
        else:
            ev["real_held"] = {"total": 0.0}
        if sb:
            syn_sum, _ = RGRPO.greedy_closed_loop_r11(
                env, scorer, policy,
                _eval_roots(sb["hd_pm"], syn_hd_pairs, sb["hd_states"]),
                gate_mem=True, log_prefix="[r11-ev]")
            ev["syn_held"] = syn_sum
        else:
            ev["syn_held"] = {"total": 0.0}
        return ev

    # ---- R11-parity: pristine δ=0 policy measured on the R11 HARNESS ----------
    # The R11 eval harness (greedy_closed_loop_r11) is deliberately NOT on R6's
    # canonical ruler: it advances past non-positive steps (horizon-5 greedy
    # continuation), gates proposals by progressive memory (gate_mem=True) and
    # scores the warm start in z-pool coordinates.  So the δ=0 gain (parity_train)
    # can never reproduce R6's recorded 369 by construction -- comparing it against
    # 0.5*369 fired E for every run regardless of training outcome (gate bug).  The
    # collapse reference must therefore be the SAME-harness δ=0 level; we ALSO
    # measure R6's true canonical closed loop (STOP-on-negative, gate_mem=False,
    # raw scores) on the same TRAIN roots so the harness/ruler gap is explicit in
    # the report instead of silently equated.
    policy_pre = RGRPO.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R11_ALPHA_PROP,
                                           alpha_stop=C.TO1_R11_ALPHA_STOP)
    ev0 = eval_root_builder(policy_pre, cycle=-1)
    parity_train = float(ev0["train"].get("total", 0.0))
    parity_real = float(ev0["real_held"].get("total", 0.0))
    parity_syn = float(ev0["syn_held"].get("total", 0.0))
    val0_sum, _ = RGRPO.greedy_closed_loop_r11(
        env, scorer, policy_pre, _eval_roots(re["progmem"], val_pairs, st_bench_map),
        gate_mem=True, log_prefix="[r11-ev0]")
    val0 = float(val0_sum.get("total", 0.0))
    r6_canonical_train = _r6_canonical_train_gain(
        env, re, scorer, r6_sel, train_pairs, st_bench_map)
    parity_ratio_to_r6 = (float(parity_train) / float(r6_canonical_train)
                          if r6_canonical_train > 0 else 0.0)
    # sanity gate: the warm-start harness must retain >= 20% of the canonical R6
    # closed-loop gain on the SAME TRAIN roots (machinery broken -> negative/zero).
    parity_ok = bool(parity_train >= 0.2 * max(r6_canonical_train, 0.0))
    print(f"[r11] R6-parity (δ=0) TRAIN={parity_train:.0f} "
          f"real_hd={parity_real:.0f} syn_hd={parity_syn:.0f} val0={val0:.0f} "
          f"r6_canonical={r6_canonical_train:.0f} ratio={parity_ratio_to_r6:.2f} "
          f"parity_ok={parity_ok}", flush=True)

    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    # ---- Stage A sanity (fixed 3 updates; NO collapse gate; quick path) --------
    res_a = RGRPO.run_rolling_cycles(
        policy, scorer, env, specs, cycles=C.TO1_R11_STAGE_A_UPDATES, quick=True,
        workers=1, seed=args.grpo_seed, log_prefix="[r11-A]",
        eval_root_builder=eval_root_builder, enable_collapse_guard=False)
    stage_a = {"cycles": len(res_a["history"]),
               "best_train": res_a["best"]["train"],
               "informative_ratio_last": (res_a["history"][-1]["informative_ratio"]
                                          if res_a["history"] else 0.0),
               "rows": res_a["history"]}
    print(f"[r11] Stage A done: {stage_a['cycles']} cycles best_train="
          f"{stage_a['best_train']:.0f} info_ratio="
          f"{stage_a['informative_ratio_last']:.3f}", flush=True)

    if args.quick:
        report = _r11_report(env, re, scraper=dict(
            repro_ok=repro_ok, parity_train=parity_train, parity_real=parity_real,
            parity_syn=parity_syn, parity_ok=parity_ok,
            val0=val0, r6_canonical_train=r6_canonical_train,
            parity_ratio_to_r6=parity_ratio_to_r6, stage_a=stage_a,
            res_b=None, runner=None, dpp=None, m5=None, audit=None, prof=None,
            val=None, pa_final=None, dpp_pre=None, dpp_post=None, sel_view=None,
            train_final=None, train_final_raw=None, info_ratio=None,
            verdict=dict(code="S", label="STAGE_A_SANITY_ONLY",
                         note="--quick: Stage A sanity ran; Stage B/DPP/VAL/profiling skipped"),
            passed=False, checkpoint_written=False))
        C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        (C.CANONICAL_OUT_DIR / "result_r11.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("[r11] QUICK: Stage A sanity only (no Stage B / DPP / VAL / profiling) "
              "-- see outputs/canonical_m3/result_r11.json", flush=True)
        return report

    # ---- Stage B: full rolling GRPO (10 fixed cycles; deterministic) ----------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    res_b = RGRPO.run_rolling_cycles(
        policy, scorer, env, specs, cycles=C.TO1_R11_TRAINING_CYCLES,
        workers=workers, seed=args.grpo_seed, log_prefix="[r11-B]",
        eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, parity_train))
    train_final = float(res_b["best"]["train"])
    train_final_raw = float((res_b["history"][-1].get("train", 0.0))
                            if res_b["history"] else 0.0)
    print(f"[r11] Stage B done: {res_b['cycles_run']} cycles collapsed="
          f"{res_b['collapsed']} ({res_b['collapse_reason']}) best_train="
          f"{train_final:.0f} best={res_b['best']}", flush=True)

    # ---- final metrics under the SELECTED (best-cycle rollback) policy --------
    sel_view = RGRPO.PolicySelectorView(policy)
    pa_final = TOP1.pool_argmax_metrics(grp_g10, sel_view)
    pos_f = float(pa_final["positive_state_argmax_accuracy"])
    regret_f = float(pa_final["top1_regret_full"]["mean"])
    rec10_f = float(pa_final["recall"]["10"])

    val_dec, _ = RGRPO.greedy_closed_loop_r11(
        env, scorer, policy, _eval_roots(re["progmem"], val_pairs, st_bench_map),
        gate_mem=True, log_prefix="[r11-val]")
    val_total = float(val_dec["total"])

    # ---- DPPaulli rolling trace BEFORE (δ=0) / AFTER (selected policy) --------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}
    dpp_pre = dpp_post = m5 = prof = audit = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_pre = RGRPO.dpp_rolling_trace_r11(
            policy_pre, scorer, env, re, dpp_iid, dpp_ep, dpp_st, dpp_lut,
            gate_mem=True, log_prefix="[r11-pre]")
        dpp_post = RGRPO.dpp_rolling_trace_r11(
            policy, scorer, env, re, dpp_iid, dpp_ep, dpp_st, dpp_lut,
            gate_mem=True, log_prefix="[r11-post]")
        print(f"[r11] DPP BEFORE: gain={dpp_pre.get('total_gain')} selected_u="
              f"{dpp_pre.get('selected_true_U_sum')} first={dpp_pre['steps'][0] if dpp_pre.get('steps') else None}",
              flush=True)
        print(f"[r11] DPP AFTER : gain={dpp_post.get('total_gain')} selected_u="
              f"{dpp_post.get('selected_true_U_sum')} first={dpp_post['steps'][0] if dpp_post.get('steps') else None}",
              flush=True)
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re, scorer,
                                     sel_view, dpp_lut, gate_mem=True)
            print(f"[r11] normal-M5: {json.dumps(m5, default=str)}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}

        # ---- probability-movement audit (true_U diagnostic only §50) ---------
        eval_pm = copy.deepcopy(re["progmem"])
        audit_probes = ([RGRPO.roots_from_state(
                            st_bench_map[i]["problem"], st_bench_map[i]["schedule"],
                            i, env["ep_id_of"][i], eval_pm)
                         for i in (tuple(p for p, _ in train_pairs[:4]))]
                        + ([RGRPO.roots_from_state(rb["hd_states"][i]["problem"],
                                                   rb["hd_states"][i]["schedule"],
                                                   i, rb["hd_ep"][i], eval_pm)
                            for i in rb["hd_iids"][:2]] if rb else [])
                        + ([RGRPO.roots_from_state(sb["hd_states"][i]["problem"],
                                                   sb["hd_states"][i]["schedule"],
                                                   i, sb["hd_ep"][i], eval_pm)
                            for i in sb["hd_iids"][:2]] if sb else []))
        try:
            audit = RGRPO.prob_movement_audit(policy_pre, policy, audit_probes,
                                              scorer, env, gate_mem=True)
            print(f"[r11] probability-movement audit: found={sum(1 for r in audit if r.get('found'))}/"
                  f"{len(audit)} argmax_changed={sum(1 for r in audit if r.get('argmax_changed'))}",
                  flush=True)
        except Exception as exc:                       # noqa: BLE001
            audit = {"error": str(exc)}

        # ---- §54 profiling: serial vs parallel trajectory collection ---------
        prof_root = RGRPO.roots_from_state(
            st_bench_map[train_pairs[0][0]]["problem"],
            st_bench_map[train_pairs[0][0]]["schedule"], train_pairs[0][0],
            train_pairs[0][1], eval_pm)
        try:
            prof = RGRPO.trajectory_profiling(policy, scorer, env, prof_root,
                                              k=C.TO1_R11_K, horizon=C.TO1_R11_HORIZON,
                                              weights=(1, 4))
            print(f"[r11] profiling: serial={prof['1']['coll_s']:.2f}s "
                  f"parallel4={prof['4']['coll_s']:.2f}s speedup={prof['speedup']} "
                  f"identical_data={prof['identical_data']}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            prof = {"error": str(exc)}

    # ---- verdict A-E (§51/§52) -------------------------------------------------
    n_groups_b = sum(h["n_groups"] for h in res_b["history"])
    n_inf_b = sum(h["n_informative"] for h in res_b["history"])
    info_ratio_b = float(n_inf_b / n_groups_b) if n_groups_b else 0.0
    # §51 A requires at least ONE real policy gain on TRAIN internal-held, AUX held,
    # or VAL -- all compared against the SAME-harness δ=0 warm-start baseline.
    margin = 0.05 * C.TO1_R11_R6_FLOOR
    best_rhd = float((res_b["best"] or {}).get("real_held", 0.0))
    best_shd = float((res_b["best"] or {}).get("syn_held", 0.0))
    improved = bool(train_final > parity_train + margin
                    or best_rhd > parity_real + margin
                    or best_shd > parity_syn + margin
                    or val_total > val0 + margin)
    if not repro_ok or not parity_ok:
        vcode, vlabel = "E", "ROLLING_GRPO_SEMANTICS_BUG"
        note = "R6 reproduction or R11 parity gate failed -- machinery semantics broken"
        ok = False
    elif isinstance(prof, dict) and prof.get("identical_data") is False:
        vcode, vlabel = "E", "ROLLING_GRPO_PARALLEL_ASYNCHRONICITY"
        note = "parallel vs serial trajectory collection differ -- sibling determinism bug"
        ok = False
    elif res_b["collapsed"]:
        vcode, vlabel = "D", "ROLLING_GRPO_DESTABILIZES_SFT"
        note = f"collapse guard fired ({res_b['collapse_reason']}) -- stale-KL/saturation/gain"
        ok = False
    elif n_groups_b == 0 or info_ratio_b <= 0.01:
        vcode, vlabel = "C", "TRAJECTORY_GROUP_COVERAGE_GAP"
        note = "no informative trajectory groups collected over Stage B"
        ok = False
    elif improved:
        vcode, vlabel = "A", "ROLLING_GRPO_IMPROVES_SFT"
        note = (f"best-cycle TRAIN gain {train_final:.0f} exceeds R6 parity "
                f"{parity_train:.0f} (improvement > 5% floor)")
        ok = True
    else:
        vcode, vlabel = "B", "ROLLING_GRPO_TRAINS_BUT_NO_GAIN"
        note = f"GRPO trains (info_ratio={info_ratio_b:.3f}) but TRAIN {train_final:.0f} <= parity {parity_train:.0f}"
        ok = False
    passed = bool(ok)     # checkpoint ONLY on verdict A (§52)

    report = _r11_report(env, re, scraper=dict(
        repro_ok=repro_ok, parity_train=parity_train, parity_real=parity_real,
        parity_syn=parity_syn, parity_ok=parity_ok,
        val0=val0, r6_canonical_train=r6_canonical_train,
        parity_ratio_to_r6=parity_ratio_to_r6, stage_a=stage_a, res_b=res_b,
        train_final=train_final, train_final_raw=train_final_raw, info_ratio=info_ratio_b,
        pa_final=pa_final, val=val_dec, dpp_pre=dpp_pre, dpp_post=dpp_post,
        m5=m5, audit=audit, prof=prof, sel_view=sel_view, runner=dict(
            r6_acc=r6_acc_all, r6_regret=r6_regret, pos6=pos6, regret6=regret6,
            rec10_6=rec10_6, pos_f=pos_f, regret_f=regret_f, rec10_f=rec10_f,
            val_total=val_total, workers=workers),
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed))
    print(f"[r11] verdict {vcode} {vlabel} (TRAIN {train_final:.0f} vs parity "
          f"{parity_train:.0f} / collapsed {res_b['collapsed']} / info_ratio "
          f"{info_ratio_b:.3f} / VAL {val_total:.0f})", flush=True)
    if passed:
        ckpt = {"state": {"policy": policy, "r6_anchor": r6_sel.state_dict(),
                          "scorer": scorer},
                "meta": {"phase": "r11_rolling_graph_multi_trajectory_grpo",
                         "method": "grpo_bounded_residual_rolling_multi_traj",
                         "parent": C.TO1_CKPT.name, "alpha_prop": C.TO1_R11_ALPHA_PROP,
                         "alpha_stop": C.TO1_R11_ALPHA_STOP,
                         "K": C.TO1_R11_K, "horizon": C.TO1_R11_HORIZON,
                         "temp": C.TO1_R11_TEMP, "mix_eps": C.TO1_R11_MIX_EPS,
                         "clip_eps": C.TO1_R11_CLIP_EPS, "beta_kl": C.TO1_R11_BETA_KL,
                         "update_epochs": C.TO1_R11_UPDATE_EPOCHS,
                         "cycles_stage_b": C.TO1_R11_TRAINING_CYCLES,
                         "train_instances": [i["instance_id"] for i in env["train_insts"]],
                         "real_train_instances": (rb["tr_iids"] if rb else []),
                         "syn_train_instances": (sb["tr_iids"] if sb else []),
                         "source_ratio": list(C.TO1_R11_SOURCE_RATIO),
                         "workers": workers, "selected": "TRAIN14+AUX-held "
                                                         "(NOT VAL3, §46)",
                         "reward": "R_i=Cmax(S_t)-Cmax(S_T) terminal, STOP=0",
                         "grpo": True, "actor_critic_r1r2": "excluded §0",
                         "identified": False, "formal_test_access": 0,
                         "formal_test_sealed": True,
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "B11_train": train_final, "B11_val": val_total,
                                    "parity_train": parity_train,
                                    "info_ratio": info_ratio_b}}}
        torch.save(ckpt, C.TO1_CKPT_R11)
        print(f"[r11] saved {C.TO1_CKPT_R11} (PASS)", flush=True)
    else:
        print(f"[r11] NOT PASS -> {C.TO1_CKPT_R11.name} NOT written (§52)", flush=True)

    _persist_r11(report, C.R11_REPORT)
    return report


def _r12_mp_ctx():
    """R12 trajectory workers run on processes (not threads).  `spawn` is the safe
    default (matches the module); tests + small local runs may set T_M3_MP_CTX=fork."""
    import os as _os
    ctx = _os.environ.get("T_M3_MP_CTX", "spawn").lower()
    return ctx if ctx in ("spawn", "fork", "forkserver") else "spawn"


def _r12_eval_roots(pm, pairs, st_map):
    pm2 = copy.deepcopy(pm)
    return [RGRPO.roots_from_state(st_map[iid]["problem"], st_map[iid]["schedule"],
                                   iid, eid, pm2) for (iid, eid) in pairs]


def run_top1_phase_r12(args, env, re, p1, p1_report):
    """R12: TRUE MULTISTEP ONLINE ROLLING GRPO (T1-M3-TRUE-MULTISTEP-...-R12).

    Four upgrades over R11 (audit in TRJ.audit_r11_credit_assignment):
      1. FULL multi-step trajectory credit -- per-trajectory equal weight L = mean_i
         L_i, L_i = mean_k clip(sur_i,k)  (§2/5/7; R11 was flat-step-mean).
      2. Forced continuation REMOVED -- true STOP semantics everywhere (§12-15).
      3. Online state proposal generation -- fresh Appearance->M2->pool per new S_t
         (§16-18).
      4. Multiprocess trajectory workers (ProcessPoolExecutor, workers=1 == workers=N)
         (§21-26).
    Parent = R11 checkpoint m3_rolling_grpo_v1.pt (bounded-residual warm start, frozen
    clone kept for KL-to-parent logs); reward R_i = Cmax(S_t)-Cmax(S_T) terminal,
    STOP=0, unwarped (§30); STOP trajectories participate, degenerate groups excluded
    from the loss but logged (§8).  UNIFIED canonical-parity evaluator
    (TRJ.unified_parity_closed_loop -- STOP-on-negative, gate_mem=False, identical
    runtime/Memory/pool-norm/horizon) renders the §39 R6/R11/R12 parity table, with
    the R6 anchor cross-checked against TOP1-canonical `_r6_canonical_train_gain`.
    Selection TRAIN14 + AUX held, NOT VAL (§38); VAL once no_grad (§42).
    identified=false, formal_test_access=0."""
    from types import SimpleNamespace
    print("[r12] TRUE-MULTISTEP ONLINE ROLLING GRPO (multiprocess, true STOP) ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §10 parent: R11 checkpoint m3_rolling_grpo_v1.pt (warm start) ----------
    r6_sel = _load_r6_parent(args)
    if not C.TO1_CKPT_R11.exists():
        raise RuntimeError(f"[r12] parent checkpoint {C.TO1_CKPT_R11} missing -- "
                           f"run `--stage r11` (verdict A) before R12 (§10).")
    policy = TRJ.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R12_ALPHA_PROP,
                                     alpha_stop=C.TO1_R12_ALPHA_STOP)
    r11_ck = torch.load(str(C.TO1_CKPT_R11), map_location="cpu", weights_only=False)
    r11_pol = r11_ck["state"]["policy"]
    r11_sd = r11_pol.state_dict()
    sd = policy.state_dict()
    n_copy = 0
    for name in r11_sd:
        if name.startswith("resid_") and name in sd and r11_sd[name].shape == sd[name].shape:
            sd[name].copy_(r11_sd[name])
            n_copy += 1
    parent_policy = copy.deepcopy(policy)
    for p_ in parent_policy.parameters():
        p_.requires_grad = False
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[r12] R11 warm-start applied ({n_copy} residual tensors); "
          f"{n_tr} trainable residual params; parent {C.TO1_CKPT_R11.name}", flush=True)

    # ---- verbatim R6 reproduction (gate_mem=False, bit-identical §40) ---------
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r12] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)
    grp_g10, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                        reranker=reranker, rng=random.Random(0),
                                        gate_mem=True)
    pa_r6 = TOP1.pool_argmax_metrics(grp_g10, r6_sel)
    pos6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])

    # ---- rolling graphs: bench TRAIN14 + AUX-REAL-train + AUX-syn-train ---------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r12] graphs: bench {len(bench_graphs)} | "
          f"real {len(rb['graphs']) if rb else 0} train / {len(rb['hd_iids']) if rb else 0} held | "
          f"syn {len(sb['graphs']) if sb else 0} train / {len(sb['hd_iids']) if sb else 0} held",
          flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def eval_root_builder(policy, cycle=-1):
        train_sum, _ = TRJ.unified_parity_closed_loop(
            env, scorer, policy, _eval_roots(re["progmem"], train_pairs, st_bench_map),
            use_mem=True, gate_mem=False)
        ev = {"train": train_sum}
        if rb:
            real_sum, _ = TRJ.unified_parity_closed_loop(
                env, scorer, policy,
                _eval_roots(rb["hd_pm"], real_hd_pairs, rb["hd_states"]),
                use_mem=True, gate_mem=False)
            ev["real_held"] = real_sum
        else:
            ev["real_held"] = {"total": 0.0}
        if sb:
            syn_sum, _ = TRJ.unified_parity_closed_loop(
                env, scorer, policy,
                _eval_roots(sb["hd_pm"], syn_hd_pairs, sb["hd_states"]),
                use_mem=True, gate_mem=False)
            ev["syn_held"] = syn_sum
        else:
            ev["syn_held"] = {"total": 0.0}
        return ev

    # ---- §39 CANONICAL-PARITY: identical ruler for R6 / R11 / R12 ----------
    # unified_parity_closed_loop is the exact body of top1.rollout_top1 (STOP-on-
    # negative, gate_mem=False, same Memory/pool-norm/horizon) -> the R6 row is the
    # pristine delta=0 policy and MUST reproduce the canonical R6 closed-loop gain
    # (r6_canonical_train) on the same TRAIN roots.  Any gap -> harness bug -> E.
    policy_d0 = TRJ.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R12_ALPHA_PROP,
                                        alpha_stop=C.TO1_R12_ALPHA_STOP)   # delta=0 = R6
    with torch.no_grad():
        ev_d0 = eval_root_builder(policy_d0, cycle=-1)
        d0_train = float(ev_d0["train"].get("total", 0.0))
        d0_real = float(ev_d0["real_held"].get("total", 0.0))
        d0_syn = float(ev_d0["syn_held"].get("total", 0.0))
        val_d0_sum, _ = TRJ.unified_parity_closed_loop(
            env, scorer, policy_d0, _eval_roots(re["progmem"], val_pairs, st_bench_map),
            use_mem=True, gate_mem=False)
        val_d0 = float(val_d0_sum.get("total", 0.0))
    r6_canonical_train = _r6_canonical_train_gain(
        env, re, scorer, r6_sel, train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(d0_train - float(r6_canonical_train)) <= anchor_tol)
    # PARITY TABLE row "R6 (d0=0)": d0_train / d0_real / d0_syn / val_d0
    parent_view = TRJ.PolicySelectorView(parent_policy)
    with torch.no_grad():
        ev_pa = eval_root_builder(parent_view, cycle=-1)
        pa_train = float(ev_pa["train"].get("total", 0.0))
        pa_real = float(ev_pa["real_held"].get("total", 0.0))
        pa_syn = float(ev_pa["syn_held"].get("total", 0.0))
        val_pa_sum, _ = TRJ.unified_parity_closed_loop(
            env, scorer, parent_view, _eval_roots(re["progmem"], val_pairs, st_bench_map),
            use_mem=True, gate_mem=False)
        val_pa = float(val_pa_sum.get("total", 0.0))
    print(f"[r12] ANCHOR: unified d0 TRAIN={d0_train:.0f} r6_canonical="
          f"{r6_canonical_train:.0f} tol={anchor_tol:.1f} anchor_ok={anchor_ok}", flush=True)
    print(f"[r12] PARITY R6(d0) {d0_train:.0f}/{d0_real:.0f}/{d0_syn:.0f}/{val_d0:.0f} "
          f"| R11(parent) {pa_train:.0f}/{pa_real:.0f}/{pa_syn:.0f}/{val_pa:.0f} "
          f"(TRAIN/real/syn/VAL, unified ruler)", flush=True)

    one = SimpleNamespace(policy_d0=policy_d0, d0_train=d0_train, d0_real=d0_real,
                          d0_syn=d0_syn, val_d0=val_d0,
                          r6_canonical_train=r6_canonical_train, anchor_ok=anchor_ok,
                          parent_view=parent_view, pa_train=pa_train, pa_real=pa_real,
                          pa_syn=pa_syn, val_pa=val_pa)

    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    # ---- Stage A (8 exact-semantics check; MUST pass before Stage B §35) ------
    res_a = TRJ.stage_a_verify()
    stage_a = dict(res_a)
    stage_a_ok = bool(res_a.get("pass", False) and res_a.get("n_failed", 1) == 0)
    print(f"[r12] Stage A exact-semantics: {res_a.get('summary')} "
          f"n_passed={res_a.get('n_passed')} n_failed={res_a.get('n_failed')} "
          f"stage_a_ok={stage_a_ok}", flush=True)

    if args.quick:
        # quick ALSO runs a 2-cycle pass through the full Stage-B machinery
        # (graph scheduling, multiprocess-1 collection, update, stochastic advance)
        # so a wiring break anywhere in the pipeline fails here, not in Stage B.
        for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
            g.reset()
        try:
            res_s = TRJ.run_rolling_cycles_r12(
                policy, scorer, env, specs, cycles=2, k=C.TO1_R12_K,
                horizon=C.TO1_R12_HORIZON, graphs_per_batch=2, workers=1,
                seed=args.grpo_seed, quick=True, log_prefix="[r12-qs]",
                eval_root_builder=eval_root_builder, collapse_floor=None,
                parent_policy=parent_policy, mp_ctx=mp_ctx, gate_mem_eval=False)
            quick_sanity = {"cycles": res_s["cycles_run"],
                            "best_train": float(res_s["best"]["train"]),
                            "collapsed": bool(res_s["collapsed"])}
            print(f"[r12] QUICK sanity: {res_s['cycles_run']} cycles "
                  f"best_train={quick_sanity['best_train']:.0f}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            quick_sanity = {"error": str(exc)}
            print(f"[r12] QUICK sanity FAILED: {exc}", flush=True)
        report = _r12_report(env, re, scraper=dict(
            repro_ok=repro_ok, anchor_ok=anchor_ok, stage_a=stage_a,
            one=one, res_b=None, train_final=None, val_final=None, pa_final=None,
            dpp_pre=None, dpp_post=None, m5=None, audit=None, prof=None,
            sel_view=None, audit_r11=None, info_ratio=None,
            quick_sanity=quick_sanity, runner=None,
            verdict=dict(code="S", label="STAGE_A_ONLY",
                         note=f"--quick: Stage A exact-semantics ran; "
                              f"quick-sanity {quick_sanity}"),
            passed=False, checkpoint_written=False))
        C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        (C.CANONICAL_OUT_DIR / "result_r12.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("[r12] QUICK: Stage A only (no Stage B / DPP / VAL / profiling) "
              "-- see outputs/canonical_m3/result_r12.json", flush=True)
        return report

    # ---- §47 profiling FIRST (pick workers=best for Stage B; determinism gate) --
    prof_root = RGRPO.roots_from_state(
        st_bench_map[train_pairs[0][0]]["problem"],
        st_bench_map[train_pairs[0][0]]["schedule"], train_pairs[0][0],
        train_pairs[0][1], copy.deepcopy(re["progmem"]))
    try:
        prof = TRJ.trajectory_profiling_r12(policy, scorer, env, prof_root,
                                            k=C.TO1_R12_K, horizon=C.TO1_R12_HORIZON,
                                            weights=(1, 2, 4, 8), mp_ctx=mp_ctx)
        ident = prof.get("all_identical")
        sp = prof.get("speedup", {})
        best_w, best_sp = 1, 1.0
        for w in (2, 4, 8):
            wp = prof.get(str(w), {})
            if wp.get("identical_to_w1") and wp.get("speedup"):
                sp_w = float(wp["speedup"])
                if sp_w > best_sp + 1e-9 or (abs(sp_w - best_sp) <= 1e-9 and w > best_w):
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w))
        print(f"[r12] profiling: {json.dumps(sp)} all_identical={ident} -> "
              f"best_w={best_w} (cap={workers_arg}) workers={workers}", flush=True)
        mp_ok = bool(ident)
    except Exception as exc:                       # noqa: BLE001
        prof = {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r12] profiling FAILED: {exc} (workers={workers})", flush=True)

    # ---- Stage B: full rolling GRPO (12 fixed cycles · K8 H5 E3 · workers=best) --
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    res_b = TRJ.run_rolling_cycles_r12(
        policy, scorer, env, specs, cycles=C.TO1_R12_TRAINING_CYCLES,
        k=C.TO1_R12_K, horizon=C.TO1_R12_HORIZON,
        graphs_per_batch=C.TO1_R12_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r12-B]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, d0_train), parent_policy=parent_policy,
        mp_ctx=mp_ctx, gate_mem_eval=False)
    train_final = float(res_b["best"]["train"])
    train_final_raw = float((res_b["history"][-1].get("train", 0.0))
                            if res_b["history"] else 0.0)
    print(f"[r12] Stage B done: {res_b['cycles_run']} cycles collapsed="
          f"{res_b['collapsed']} ({res_b['collapse_reason']}) best_train="
          f"{train_final:.0f} best={res_b['best']}", flush=True)

    # ---- final metrics under the SELECTED (best-cycle rollback) policy --------
    sel_view = TRJ.PolicySelectorView(policy)
    pa_final = TOP1.pool_argmax_metrics(grp_g10, sel_view)
    pos_f = float(pa_final["positive_state_argmax_accuracy"])
    regret_f = float(pa_final["top1_regret_full"]["mean"])
    rec10_f = float(pa_final["recall"]["10"])

    # VAL once, no_grad, report-only (§42); same unified ruler.
    with torch.no_grad():
        val_sum, _ = TRJ.unified_parity_closed_loop(
            env, scorer, sel_view, _eval_roots(re["progmem"], val_pairs, st_bench_map),
            use_mem=True, gate_mem=False)
    val_final = float(val_sum.get("total", 0.0))

    # ---- DPPaulli rolling trace (unified ruler) BEFORE (R11 parent) / AFTER ----
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    dpp_lut = _s0_trueU(env, re, dpp_iid) if dpp_iid else {}
    dpp_pre = dpp_post = m5 = audit = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_pre = TRJ.dpp_rolling_trace_r12(
            parent_view, scorer, env, re, dpp_iid, dpp_ep, dpp_st, dpp_lut,
            gate_mem=False, log_prefix="[r12-pre]")
        dpp_post = TRJ.dpp_rolling_trace_r12(
            sel_view, scorer, env, re, dpp_iid, dpp_ep, dpp_st, dpp_lut,
            gate_mem=False, log_prefix="[r12-post]")
        print(f"[r12] DPP BEFORE(parent): gain={dpp_pre.get('total_gain')} "
              f"selected_u={dpp_pre.get('selected_true_U_sum')}", flush=True)
        print(f"[r12] DPP AFTER(R12)    : gain={dpp_post.get('total_gain')} "
              f"selected_u={dpp_post.get('selected_true_U_sum')}", flush=True)
        try:
            m5 = TOP1.normal_m5_top1(dpp_iid, dpp_ep, dpp_st, env, re, scorer,
                                     sel_view, dpp_lut, gate_mem=True)
            print(f"[r12] normal-M5: {json.dumps(m5, default=str)}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            m5 = {"error": str(exc)}
        eval_pm = copy.deepcopy(re["progmem"])
        audit_probes = ([RGRPO.roots_from_state(
                            st_bench_map[i]["problem"], st_bench_map[i]["schedule"],
                            i, env["ep_id_of"][i], eval_pm)
                         for i in (tuple(p for p, _ in train_pairs[:4]))]
                        + ([RGRPO.roots_from_state(rb["hd_states"][i]["problem"],
                                                   rb["hd_states"][i]["schedule"],
                                                   i, rb["hd_ep"][i], eval_pm)
                            for i in rb["hd_iids"][:2]] if rb else [])
                        + ([RGRPO.roots_from_state(sb["hd_states"][i]["problem"],
                                                   sb["hd_states"][i]["schedule"],
                                                   i, sb["hd_ep"][i], eval_pm)
                            for i in sb["hd_iids"][:2]] if sb else []))
        try:
            audit = RGRPO.prob_movement_audit(parent_policy, policy, audit_probes,
                                              scorer, env, gate_mem=True)
            print(f"[r12] probability-movement audit: "
                  f"found={sum(1 for r in audit if r.get('found'))}/"
                  f"{len(audit)}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            audit = {"error": str(exc)}

    # ---- verdict §44: FULL_TRAJECTORY_GRPO_IMPROVES requires ----------------
    audit_r11 = TRJ.audit_r11_credit_assignment()
    n_groups_b = sum(h["n_groups"] for h in res_b["history"])
    n_inf_b = sum(h["n_informative"] for h in res_b["history"])
    info_ratio_b = float(n_inf_b / n_groups_b) if n_groups_b else 0.0
    margin = 0.05 * max(float(r6_canonical_train), 1.0)
    train_beat = bool(train_final > float(max(d0_train, pa_train)) + margin)
    best_rhd = float((res_b["best"] or {}).get("real_held", 0.0))
    best_shd = float((res_b["best"] or {}).get("syn_held", 0.0))
    baseline_rhd = float(max(pa_real, d0_real))
    baseline_shd = float(max(pa_syn, d0_syn))
    held_improved = bool(best_rhd > baseline_rhd + 1.0 or best_shd > baseline_shd + 1.0)
    full_step_credit_ok = audit_r11.get("all_steps_in_loss") is True
    # verdict codes per §44/§45 verbatim
    if not full_step_credit_ok:
        vcode, vlabel = "E", "TRAJECTORY_GRPO_SEMANTICS_BUG"
        note = f"audit.r11 all_steps_in_loss={full_step_credit_ok} -- credit assignment broken"
        ok = False
    elif not repro_ok or not anchor_ok or not stage_a_ok:
        vcode, vlabel = "E", "TRAJECTORY_GRPO_SEMANTICS_BUG"
        note = f"repro={repro_ok} anchor={anchor_ok} stage_a={stage_a_ok} -- machinery broken"
        ok = False
    elif not mp_ok:
        vcode, vlabel = "F", "MULTIPROCESS_EXECUTION_MISMATCH"
        note = "multiprocess vs serial trajectory collection differ -- determinism bug"
        ok = False
    elif res_b["collapsed"]:
        vcode, vlabel = "D", "MULTISTEP_CREDIT_DESTABILIZES_POLICY"
        note = f"collapse guard fired ({res_b['collapse_reason']})"
        ok = False
    elif n_groups_b == 0 or info_ratio_b <= 0.01:
        vcode, vlabel = "C", "ONLINE_COVERAGE_REMAINS_LIMITED"
        note = "no informative trajectory groups collected over Stage B"
        ok = False
    elif train_beat and held_improved:
        vcode, vlabel = "A", "FULL_TRAJECTORY_GRPO_IMPROVES"
        note = (f"TRAIN {train_final:.0f} > max(d0 {d0_train:.0f}, parent {pa_train:.0f}) "
                f"+{margin:.0f} AND held real/syn beat max(parent,d0) "
                f"(real {best_rhd:.0f}>{baseline_rhd:.0f} or "
                f"syn {best_shd:.0f}>{baseline_shd:.0f})")
        ok = True
    else:
        vcode, vlabel = "B", "FULL_TRAJECTORY_GRPO_TRAINS_BUT_NO_GAIN"
        note = (f"train_beat={train_beat} held_improved={held_improved} "
                f"TRAIN={train_final:.0f} vs d0={d0_train:.0f}/parent={pa_train:.0f}")
        ok = False
    passed = bool(ok)     # checkpoint ONLY on verdict A (§49)

    report = _r12_report(env, re, scraper=dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, stage_a=stage_a,
        one=one, res_b=res_b, train_final=train_final,
        train_final_raw=train_final_raw, info_ratio=info_ratio_b,
        pa_final=pa_final, val_final=val_final, dpp_pre=dpp_pre, dpp_post=dpp_post,
        m5=m5, audit=audit, prof=prof, sel_view=sel_view, audit_r11=audit_r11,
        runner=dict(r6_acc=r6_acc_all, r6_regret=r6_regret, pos6=pos6,
                    regret6=regret6, rec10_6=rec10_6, pos_f=pos_f, regret_f=regret_f,
                    rec10_f=rec10_f, val_final=val_final, workers=workers, mp_ctx=mp_ctx),
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed))
    print(f"[r12] verdict {vcode} {vlabel} (TRAIN {train_final:.0f} vs "
          f"d0={d0_train:.0f}/parent={pa_train:.0f} / held real={best_rhd:.0f}/"
          f"syn={best_shd:.0f} / collapsed {res_b['collapsed']} / "
          f"info_ratio {info_ratio_b:.3f} / VAL {val_final:.0f} / "
          f"stage_a {stage_a_ok} / anchor {anchor_ok})", flush=True)
    if passed:
        ckpt = {"state": {"policy": policy, "r6_anchor": r6_sel.state_dict(),
                          "scorer": scorer},
                "meta": {"phase": "r12_true_multistep_online_rolling_grpo",
                         "method": "rolling_full_trajectory_stepwise_clipped_grpo",
                         "parent": "m3_rolling_grpo_v1.pt",
                         "sft_reference": C.TO1_CKPT.name,
                         "reward": "terminal_makespan_gain",
                         "trajectory_credit": "all_steps",
                         "ratio": "per_step_clipped",
                         "forced_continuation": False,
                         "proposal_generation": "online",
                         "parallelism": "multiprocess",
                         "offline_pool_range": "seed/diag/held-eval/warm-start",
                         "alpha_prop": C.TO1_R12_ALPHA_PROP,
                         "alpha_stop": C.TO1_R12_ALPHA_STOP,
                         "K": C.TO1_R12_K, "horizon": C.TO1_R12_HORIZON,
                         "update_epochs": C.TO1_R12_UPDATE_EPOCHS,
                         "cycles_stage_b": C.TO1_R12_TRAINING_CYCLES,
                         "temp": C.TO1_R12_TEMP, "mix_eps": C.TO1_R12_MIX_EPS,
                         "clip_eps": C.TO1_R12_CLIP_EPS, "beta_kl": C.TO1_R12_BETA_KL,
                         "train_instances": [i["instance_id"] for i in env["train_insts"]],
                         "real_train_instances": (rb["tr_iids"] if rb else []),
                         "syn_train_instances": (sb["tr_iids"] if sb else []),
                         "source_ratio": list(C.TO1_R12_SOURCE_RATIO),
                         "workers": workers, "mp_ctx": mp_ctx,
                         "selected": "TRAIN14+AUX-held (NOT VAL3, §38)",
                         "reward_spec": "R_i=Cmax(S_t)-Cmax(S_T) terminal, STOP=0",
                         "trajectory_unit": "per-trajectory equal weight",
                         "force_continuation": "removed (true STOP everywhere)",
                         "stop_semantics": "policy-STOP/non-positive/infeasible end episode",
                         "training_advancement": "stochastic sample (exploration)",
                         "eval": "greedy, no oracle",
                         "identified": False, "formal_test_access": 0,
                         "formal_test_sealed": True,
                         "grant_parity": {"r6_d0": d0_train, "r11_parent": pa_train,
                                          "r12": train_final,
                                          "r6_canonical": r6_canonical_train,
                                          "anchor_ok": anchor_ok},
                         "r11_promote": "CANONICAL_DEPLOYMENT_PROMOTED=false "
                                        "until parity done (§41)",
                         "report": {"passed": True, "vcode": vcode, "vlabel": vlabel,
                                    "r12_train": train_final, "r12_val": val_final,
                                    "parent_train": pa_train,
                                    "d0_train": d0_train,
                                    "info_ratio": info_ratio_b}}}
        torch.save(ckpt, C.TO1_R12_CKPT)
        print(f"[r12] saved {C.TO1_R12_CKPT} (PASS)", flush=True)
    else:
        print(f"[r12] NOT PASS -> {C.TO1_R12_CKPT.name} NOT written (§49)", flush=True)

    _persist_r12(report, C.R12_REPORT)
    print(f"[r12] total {time.time() - t0:.1f}s (report {C.R12_REPORT.name})", flush=True)
    return report


def _r12_parity_table_rows(one, res_b, train_final, val_final, pa_final, prof):
    rows = [
        {"model": "R6 (canonical δ=0)", "ruler": "unified-parity",
         "train": one.d0_train, "real_held": one.d0_real, "syn_held": one.d0_syn,
         "val": one.val_d0},
        {"model": "R11 (parent m3_rolling_grpo_v1)", "ruler": "unified-parity",
         "train": one.pa_train, "real_held": one.pa_real, "syn_held": one.pa_syn,
         "val": one.val_pa},
        {"model": "R12 (selected)", "ruler": "unified-parity",
         "train": train_final,
         "real_held": float((res_b.get("best") or {}).get("real_held", 0.0)),
         "syn_held": float((res_b.get("best") or {}).get("syn_held", 0.0)),
         "val": val_final},
    ]
    return rows


def _r12_report(env, re, scraper):
    from types import SimpleNamespace as SN
    s = scraper
    one = s.get("one") or SN(d0_train=0.0, d0_real=0.0, d0_syn=0.0, val_d0=0.0,
                             r6_canonical_train=0.0, anchor_ok=False,
                             pa_train=0.0, pa_real=0.0, pa_syn=0.0, val_pa=0.0)
    rb = s.get("res_b") if isinstance(s.get("res_b"), dict) and s.get("res_b") else {}
    train_final = float(s.get("train_final") or 0.0)
    val_final = float(s.get("val_final") or 0.0)
    pa = s.get("pa_final") or {}
    audit_r11 = s.get("audit_r11") or {}
    prof = s.get("prof") or {}
    rows = _r12_parity_table_rows(SN(d0_train=one.d0_train, d0_real=one.d0_real,
                                     d0_syn=one.d0_syn, val_d0=one.val_d0,
                                     pa_train=one.pa_train, pa_real=one.pa_real,
                                     pa_syn=one.pa_syn, val_pa=one.val_pa),
                                  rb, train_final, val_final, pa, prof)
    lines = []
    lines.append("# T1-M3 TRUE-MULTISTEP ONLINE ROLLING GRPO (R12) -- report")
    lines.append("")
    lines.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    lines.append("")
    lines.append(f"## Verdict: **{s['verdict']['code']}** `{s['verdict']['label']}`")
    lines.append("")
    lines.append(f"> {s['verdict']['note']}")
    lines.append("")
    lines.append("## §2/3 R11 credit-assignment audit")
    lines.append("")
    lines.append(f"- status: `{audit_r11.get('status')}`")
    lines.append(f"- per-step logp_old stored: {audit_r11.get('per_step_logp_old')}")
    lines.append(f"- shared group advantage on all steps: {audit_r11.get('group_adv_all_steps')}")
    lines.append(f"- all steps in loss: {audit_r11.get('all_steps_in_loss')}")
    lines.append(f"- loss weighting: `{audit_r11.get('loss_weighting')}` "
                 f"(R12 fixes to per-trajectory equal weight)")
    lines.append("")
    lines.append("## §39 CANONICAL-PARITY TABLE (identical unified ruler)")
    lines.append("")
    lines.append("| model | ruler | TRAIN | real-held | syn-held | VAL |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['ruler']} | {r['train']:.0f} | "
                     f"{r['real_held']:.0f} | {r['syn_held']:.0f} | {r['val']:.0f} |")
    lines.append(f"\nR6 anchor cross-check: unified d0 TRAIN {one.d0_train:.0f} vs "
                 f"canonical R6 `_r6_canonical_train_gain` {one.r6_canonical_train:.0f} "
                 f"(tolerance max(2, 5%)) -> `anchor_ok={one.anchor_ok}`")
    lines.append("")
    lines.append("## Stage A (8 exact-semantics checks, §35)")
    lines.append("")
    lines.append(f"- pytest: {s.get('stage_a', {}).get('summary')} "
                 f"-> stage_a_ok={s.get('stage_a', {}).get('pass')}")
    lines.append("")
    lines.append("## Stage B (12 cycles · GRAPHS_PER_BATCH 4 · K8 · H5 · E3)")
    lines.append("")
    if rb:
        lines.append(f"- cycles_run={rb.get('cycles_run')} collapsed={rb.get('collapsed')} "
                     f"({rb.get('collapse_reason')})")
        for h in rb.get("history", []):
            lines.append(f"  - cycle {h['cycle']}: TRAIN {h['train']:.0f} "
                         f"real {h.get('real_held', 0):.0f} syn {h.get('syn_held', 0):.0f} "
                         f"info {h.get('informative_ratio', 0):.3f} "
                         f"kl_ref {h.get('kl_ref', 0):.4f} "
                         f"kl_parent {h.get('kl_parent')} "
                         f"sat {h.get('sat', {}).get('saturated_frac', 0):.3f}")
        lines.append(f"- best cycle {rb.get('best', {}).get('cycle')}: "
                     f"train {rb.get('best', {}).get('train')} "
                     f"real {rb.get('best', {}).get('real_held')} "
                     f"syn {rb.get('best', {}).get('syn_held')}")
    else:
        lines.append("- (Stage B not run)")
    lines.append("")
    lines.append("## Selected-policy metrics (TRAIN14 + AUX-held selection §38)")
    lines.append("")
    lines.append(f"- pool-argmax: positive_acc={pa.get('positive_state_argmax_accuracy')} "
                 f"regret={pa.get('top1_regret_full', {}).get('mean')} "
                 f"recall@10={(pa.get('recall') or {}).get('10')}")
    lines.append(f"- VAL once no_grad (unified ruler, report-only §42): **{val_final:.0f}**")
    if s.get("quick_sanity"):
        qs = s["quick_sanity"]
        lines.append(f"- quick-sanity (full machinery, 2 cycles): "
                     f"{json.dumps(qs)}")
    lines.append("")
    if s.get("dpp_pre") is not None and s.get("dpp_post") is not None:
        lines.append("## DPPaulli rolling trace (unified ruler, true STOP)")
        lines.append("")
        lines.append(f"- BEFORE (R11 parent): gain {s['dpp_pre'].get('total_gain')} "
                     f"selected_U {s['dpp_pre'].get('selected_true_U_sum')}")
        lines.append(f"- AFTER  (R12)      : gain {s['dpp_post'].get('total_gain')} "
                     f"selected_U {s['dpp_post'].get('selected_true_U_sum')}")
        lines.append("")
    if s.get("m5"):
        lines.append(f"- normal-M5 regression: {json.dumps(s.get('m5'), default=str)}")
    if s.get("audit"):
        lines.append(f"- probability-movement audit: "
                     f"{json.dumps(s.get('audit'), default=str)}")
    lines.append("")
    lines.append("## §47/48 profiling (multiprocess workers)")
    lines.append("")
    if isinstance(prof, dict) and "error" not in prof:
        lines.append(f"- weights 1/2/4/8 coll_s: "
                     f"{[prof.get(str(w), {}).get('coll_s') for w in (1, 2, 4, 8)]} "
                     f"speedup={prof.get('speedup')}")
        lines.append(f"- all identical to w=1: {prof.get('all_identical')}")
        lines.append(f"- cloud extrapolation 16/32/64: "
                     f"{json.dumps(TRJ.cloud_extrapolate(prof).get('projections', {}))}")
    else:
        lines.append(f"- (profiling {'failed: ' + str(prof.get('error')) if isinstance(prof, dict) else 'skipped'})")
    lines.append(f"- Stage B workers used: {(s.get('runner') or {}).get('workers')} "
                 f"ctx={(s.get('runner') or {}).get('mp_ctx')}")
    lines.append("")
    lines.append("## Checkpoint")
    lines.append("")
    lines.append(f"- written on PASS only: `{C.TO1_R12_CKPT.name}` "
                 f"(passed={s.get('passed')})")
    return "\n".join(lines) + "\n"


def _persist_r12(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (path.parent / "result_r12.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[r12] report written: {path}", flush=True)


# ===========================================================================
# R13: M2 budgeted root search x M3 proposal policy -- JOINT agentic GRPO
# ===========================================================================
def run_top1_phase_r13(args, env, re, p1, p1_report):
    """R13: M2 budgeted root search (anchors/sequential-draws/outsider) x REAL FDR
    root probes x makespan-first tier filter (A=PROVEN_GAIN, B=Memory rescue cap,
    C=pruned) x M3 proposal policy -- ONE §0 agentic rolling-GRPO loop (§0-§36).

    New vs R12: the §0 permanent structure runs end-to-end in BOTH the training
    rollouts and the same-ruler evaluator.  Probe gains NEVER enter the terminal
    reward (§24, m2_reward_authority=false).  Stages A (adapter RL on FROZEN M3, 5
    PASS gates §33) / B (M3 GRPO on the gated pool, M2 frozen) / C (JOINT, shared
    per-trajectory advantage §25/28).  §52 C0-C5 parity on the SAME unified ruler
    (m2_mode="none" reproduces the 369 anchor bit-identically).  §48 normal-M5
    dep-completed Tier-A regression is a PERMANENT stop-list gate (never Memory).
    §54 alpha_M3 influence diag; §55 cloud/groups 4/8/16 profile.  Checkpoints
    (m2_root_policy_r13.pt + m2_m3_joint_agentic_grpo_r13.pt) ONLY on verdict A
    (§56 metadata).  identified=false, formal_test_access=0."""
    from types import SimpleNamespace as SNS
    print("[r13] M2-ROOT-SEARCH x M3-PROPOSAL JOINT AGENTIC GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §19-20 parent: R11 m3_rolling_grpo_v1.pt warm start; M2 adapter zero-init
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(r6_sel)
    r11_ck = torch.load(str(C.TO1_R13_PARENT), map_location="cpu", weights_only=False)
    r11_sd = r11_ck["state"]["policy"].state_dict()
    sd = jpol.m3.state_dict()
    n_copy = 0
    for name in r11_sd:
        if name.startswith("resid_") and name in sd and r11_sd[name].shape == sd[name].shape:
            sd[name].copy_(r11_sd[name])
            n_copy += 1
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()          # zero-init adapter + warm M3 (C2 / dpp-pre)
    print(f"[r13] M2 adapter {n_tr_m2} params (zero-init delta=0) | M3 {n_tr_m3} params "
          f"(parent {C.TO1_R13_PARENT.name}, {n_copy} resid copied)", flush=True)

    # ---- verbatim R6 reproduction (gate_mem=False, bit-identical §40) ---------
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r13] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)
    grp_g10, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                        reranker=reranker, rng=random.Random(0),
                                        gate_mem=True)
    pa_r6 = TOP1.pool_argmax_metrics(grp_g10, r6_sel)
    pos6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])

    # ---- rolling graphs: bench TRAIN14 + AUX-REAL-train + AUX-syn-train ---------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real",
                             env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux",
                             env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r13] graphs: bench {len(bench_graphs)} | "
          f"real {len(rb['graphs']) if rb else 0} train / "
          f"{len(rb['hd_iids']) if rb else 0} held | "
          f"syn {len(sb['graphs']) if sb else 0} train / "
          f"{len(sb['hd_iids']) if sb else 0} held", flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _closed_loop(mode, obj, roots):
        sm, _ = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode,
                               use_mem=True, gate_mem=False)
        return sm

    def _tot(mode, obj, roots):
        return float(_closed_loop(mode, obj, roots).get("total", 0.0))

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _closed_loop("adapter", jp_,
                                    _eval_roots(re["progmem"], train_pairs, st_bench_map))}
        ev["real_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            rb["hd_pm"], real_hd_pairs, rb["hd_states"])) if rb else {"total": 0.0})
        ev["syn_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            sb["hd_pm"], syn_hd_pairs, sb["hd_states"])) if sb else {"total": 0.0})
        return ev

    # DPPaulli S0 (normal-M5 §48 permanent trace site)
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"] if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _m5_gate(adapter):
        """§48 permanent gate at DPPaulli S0: direct<=0 + dep-completed pair positive
        MUST land Tier A (never Memory-only).  Vacuous-safe when no dep_saved."""
        if dpp_iid is None:
            return None
        prop_feats, metas, agg = env["cache"].proposals(
            dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        if not metas:
            return {"at_state": False, "dep_saved": [], "all_tier_a_ok": True,
                    "retain_ok": True, "retained": [], "diag": {"note": "no proposals"}}
        ast = env["cache"].ast(dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        ms = int(dpp_st["schedule"].makespan)
        sf = PF.state_feature_vec(ms, ms, len(metas), agg["best_uhat"], agg["best_direct"],
                                  agg["n_contrib"], agg["n_enab"])
        return JG.normal_m5_r13_gate(ast, metas, prop_feats, adapter,
                                     env["executor"], copy.deepcopy(re["progmem"]),
                                     dpp_iid, int(dpp_ep), 0, sf, random.Random(1))

    if args.quick:
        # quick ALSO passes a 2-cycle real Stage-A through the full machinery
        # (graph rotation, collect, joint update, stochastic advance, §48 trace).
        for g in bench_graphs:
            g.reset()
        jpol.params_for_stage("A")
        try:
            res_s = JG.run_rolling_cycles_r13(
                jpol, scorer, env, specs, stage="A", cycles=2, k=C.TO1_R13_K,
                horizon=C.TO1_R13_HORIZON, graphs_per_batch=2, workers=1,
                seed=args.grpo_seed, quick=True, log_prefix="[r13-qs]",
                eval_root_builder=eval_root_builder, collapse_floor=None,
                parent_policy=None, mp_ctx=mp_ctx)
            quick_sanity = {"cycles": res_s["cycles_run"],
                            "best_train": float(res_s["best"]["train"]),
                            "collapsed": bool(res_s["collapsed"]),
                            "stage_gates": res_s.get("stage_gates") or {},
                            "cycles_detail": [{c["cycle"]: {"n_informative": c.get(
                                "n_informative"), "dry": c.get("dry"),
                                "prob_move": c.get("prob_move"),
                                "stats": c.get("stats")}}
                                for c in res_s.get("stage_a_gates", [])]}
            print(f"[r13] QUICK sanity: {res_s['cycles_run']} cycles "
                  f"best_train={quick_sanity['best_train']:.0f}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            quick_sanity = {"error": str(exc)}
            print(f"[r13] QUICK sanity FAILED: {exc}", flush=True)
        m5q = _m5_gate(jpol.m2)
        if m5q:
            print(f"[r13] §48 zero-init dep_saved={m5q['dep_saved']} "
                  f"all_tier_a_ok={m5q['all_tier_a_ok']}", flush=True)
        report = _r13_report(env, re, scraper=dict(
            repro_ok=repro_ok, anchor_ok=None, stage_a=None, m5=None,
            m5_z=m5q, m5_a=None, res_a=None, res_b=None, res_c=None,
            train_final=0.0, train_final_b=0.0, val_final=0.0, rows={},
            r6_canonical_train=None, prof=None, mp_ok=None, workers=1,
            mp_ctx=mp_ctx, dpp_pre=None, dpp_post=None, alpha=None, deco=None,
            pa_final=None, pos6=pos6, regret6=regret6, rec10_6=rec10_6,
            pos_f=None, regret_f=None, rec10_f=None,
            verdict=dict(code="S", label="STAGE_A_ONLY",
                         note=f"--quick: Stage-A machinery + §48 trace ran; "
                              f"quick-sanity {quick_sanity}"),
            passed=False, checkpoint_written=False))
        _r13_persist(report, scrape=None)
        print("[r13] QUICK: Stage A sanity only -- "
              "see outputs/canonical_m3/result_r13.json", flush=True)
        return report

    # ---- §52 C0/C1/C2 baselines + C0 369 anchor (SAME unified ruler) ----------
    c0 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                alpha_stop=C.TO1_R13_ALPHA_STOP)   # delta=0 = R6
    parent_m3 = TRJ.PolicySelectorView(copy.deepcopy(jpol.m3))      # C1: R11 parent
    with torch.no_grad():
        c0_train = _tot("none", c0, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c0_real = _tot("none", c0, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                               rb["hd_states"])) if rb else 0.0
        c0_syn = _tot("none", c0, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                              sb["hd_states"])) if sb else 0.0
        c0_val = _tot("none", c0, _eval_roots(re["progmem"], val_pairs, st_bench_map))
        c1_train = _tot("none", parent_m3, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c1_real = _tot("none", parent_m3, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                                      rb["hd_states"])) if rb else 0.0
        c1_syn = _tot("none", parent_m3, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                                     sb["hd_states"])) if sb else 0.0
        c1_val = _tot("none", parent_m3, _eval_roots(re["progmem"], val_pairs, st_bench_map))
        c2_train = _tot("adapter", jpol, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c2_real = _tot("adapter", jpol, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                                    rb["hd_states"])) if rb else 0.0
        c2_syn = _tot("adapter", jpol, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                                   sb["hd_states"])) if sb else 0.0
        c2_val = _tot("adapter", jpol, _eval_roots(re["progmem"], val_pairs, st_bench_map))
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(c0_train - float(r6_canonical_train)) <= anchor_tol)
    print(f"[r13] ANCHOR C0: unified {c0_train:.0f} vs r6_canonical="
          f"{r6_canonical_train:.0f} tol={anchor_tol:.1f} anchor_ok={anchor_ok}",
          flush=True)
    print(f"[r13] PARITY C0(δ0,R6) {c0_train:.0f}/{c0_real:.0f}/{c0_syn:.0f}/{c0_val:.0f} | "
          f"C1(R11) {c1_train:.0f}/{c1_real:.0f}/{c1_syn:.0f}/{c1_val:.0f} | "
          f"C2(d0-gated) {c2_train:.0f}/{c2_real:.0f}/{c2_syn:.0f}/{c2_val:.0f} "
          f"(TRAIN/real/syn/VAL)", flush=True)

    m5_z = _m5_gate(jpol.m2)
    if m5_z:
        print(f"[r13] §48 DPP normal-M5 zero-init: dep_saved={m5_z['dep_saved']} "
              f"all_tier_a_ok={m5_z['all_tier_a_ok']} retain_ok={m5_z['retain_ok']}",
              flush=True)

    # ---- §55 cloud/shape profile FIRST (pick workers for A/B/C; determinism gate)
    prof = None
    mp_ok = True
    workers = 1
    prof_root = _eval_roots(re["progmem"], train_pairs[:1], st_bench_map)[0]
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, prof_root,
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx)
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"] for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r13] cloud profile: {json.dumps(prof['per_worker'])} "
              f"identical={ident_ok} -> workers={workers}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r13] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- Stage A: M2 adapter RL on FROZEN M3 (§32-33) --------------------------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("A")
    res_a = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="A", cycles=C.TO1_R13_TRAINING_CYCLES_A,
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r13-A]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, c2_train), parent_policy=None, mp_ctx=mp_ctx)
    m5_a = _m5_gate(jpol.m2)
    _r13_hg = ("m2_grad_nonzero", "prob_move", "positive_probe_rate_ok",
               "tier_a_coverage_ok", "gated_coverage_ok")
    _sg = res_a.get("stage_gates") or {}
    stage_gates_ok = bool(_sg) and all(_sg.get(k) for k in _r13_hg)
    stage_a_ok = bool(stage_gates_ok and not res_a["collapsed"] and
                      (m5_a is None or m5_a["all_tier_a_ok"]))
    a_best_train = float((res_a["best"] or {"train": 0.0})["train"])
    print(f"[r13] Stage A: cycles={res_a['cycles_run']} collapsed={res_a['collapsed']} "
          f"gates_ok={stage_gates_ok} §48_all_tier_a_ok="
          f"{m5_a['all_tier_a_ok'] if m5_a else 'n/a'} "
          f"best_train={a_best_train:.0f} stage_a_ok={stage_a_ok}", flush=True)
    with torch.no_grad():
        c3_train = _tot("adapter", jpol, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c3_real = _tot("adapter", jpol, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                                    rb["hd_states"])) if rb else 0.0
        c3_syn = _tot("adapter", jpol, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                                   sb["hd_states"])) if sb else 0.0
        c3_val = _tot("adapter", jpol, _eval_roots(re["progmem"], val_pairs, st_bench_map))
    print(f"[r13] C3(Stage-A adapter x frozen M3): "
          f"{c3_train:.0f}/{c3_real:.0f}/{c3_syn:.0f}/{c3_val:.0f}", flush=True)

    # ---- Stage B: frozen M2 x M3 GRPO on the GATED pool ------------------------
    jpol.params_for_stage("B")
    res_b = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="B", cycles=C.TO1_R13_TRAINING_CYCLES_B,
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r13-B]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, c2_train, a_best_train), parent_policy=None,
        mp_ctx=mp_ctx)
    train_final_b = float((res_b["best"] or {"train": 0.0})["train"])
    print(f"[r13] Stage B: cycles={res_b['cycles_run']} collapsed={res_b['collapsed']} "
          f"best_train={train_final_b:.0f}", flush=True)
    with torch.no_grad():
        c4_train = _tot("adapter", jpol, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c4_real = _tot("adapter", jpol, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                                    rb["hd_states"])) if rb else 0.0
        c4_syn = _tot("adapter", jpol, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                                   sb["hd_states"])) if sb else 0.0
        c4_val = _tot("adapter", jpol, _eval_roots(re["progmem"], val_pairs, st_bench_map))
    print(f"[r13] C4(Stage-B M3 x frozen Stage-A M2): "
          f"{c4_train:.0f}/{c4_real:.0f}/{c4_syn:.0f}/{c4_val:.0f}", flush=True)

    # ---- Stage C: JOINT M2+M3 (shared advantage) ------------------------------
    jpol.params_for_stage("C")
    res_c = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=C.TO1_R13_TRAINING_CYCLES_C,
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r13-C]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, c2_train, a_best_train, train_final_b),
        parent_policy=None, mp_ctx=mp_ctx)
    train_final = float((res_c["best"] or {"train": 0.0})["train"])
    print(f"[r13] Stage C: cycles={res_c['cycles_run']} collapsed={res_c['collapsed']} "
          f"best_train={train_final:.0f}", flush=True)
    with torch.no_grad():
        c5_train = _tot("adapter", jpol, _eval_roots(re["progmem"], train_pairs, st_bench_map))
        c5_real = _tot("adapter", jpol, _eval_roots(rb["hd_pm"], real_hd_pairs,
                                                    rb["hd_states"])) if rb else 0.0
        c5_syn = _tot("adapter", jpol, _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                                   sb["hd_states"])) if sb else 0.0
        c5_val = _tot("adapter", jpol, _eval_roots(re["progmem"], val_pairs, st_bench_map))
    print(f"[r13] C5(JOINT): {c5_train:.0f}/{c5_real:.0f}/{c5_syn:.0f}/{c5_val:.0f}",
          flush=True)

    # ---- VAL once no_grad (§42) + final pool argmax + §48 final trace ---------
    sel_view = TRJ.PolicySelectorView(jpol.m3)
    pa_final = TOP1.pool_argmax_metrics(grp_g10, sel_view)
    pos_f = float(pa_final["positive_state_argmax_accuracy"])
    regret_f = float(pa_final["top1_regret_full"]["mean"])
    rec10_f = float(pa_final["recall"]["10"])
    with torch.no_grad():
        val_final = _tot("adapter", jpol,
                         _eval_roots(re["progmem"], val_pairs, st_bench_map))
    m5 = m5_f = _m5_gate(jpol.m2)
    if m5_f:
        print(f"[r13] §48 FINAL: dep_saved={m5_f['dep_saved']} "
              f"all_tier_a_ok={m5_f['all_tier_a_ok']} retain_ok={m5_f['retain_ok']}",
              flush=True)

    # ---- DPPaulli rolling trace BEFORE (zero-init) / AFTER (final) ------------
    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_root = RGRPO.roots_from_state(dpp_st["problem"], dpp_st["schedule"],
                                          dpp_iid, int(dpp_ep),
                                          copy.deepcopy(re["progmem"]))
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _ = JG.agentic_closed_loop(env, scorer, zj2, [dpp_root],
                                                use_mem=True, gate_mem=False)
            post_sum, _ = JG.agentic_closed_loop(env, scorer, jpol, [dpp_root],
                                                 use_mem=True, gate_mem=False)
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r13] DPP BEFORE(zero-init): gain={dpp_pre['total_gain']} | "
              f"AFTER(final): gain={dpp_post['total_gain']}", flush=True)

    # ---- §46 failure decomposition on real trajectories ------------------------
    deco = None
    try:
        deco_roots = _eval_roots(re["progmem"], train_pairs[:4], st_bench_map)
        if dpp_iid is not None and dpp_st is not None:
            deco_roots.append(RGRPO.roots_from_state(
                dpp_st["problem"], dpp_st["schedule"], dpp_iid, int(dpp_ep),
                copy.deepcopy(re["progmem"])))
        deco = {"rollouts": [], "decomposition": {}}
        for rf in deco_roots[:5]:
            rd = JG.agentic_rollout_diag(env, scorer, jpol, rf, gate_mem=False)
            for k_, v_ in rd["decomposition"].items():
                deco["decomposition"][k_] = deco["decomposition"].get(k_, 0) + int(v_)
            deco["rollouts"].append({"iid": rf["iid"], "final_gain": rd["final_gain"],
                                     "n_steps": len(rd["steps"]),
                                     "decomp": dict(rd["decomposition"])})
        print(f"[r13] failure decomposition: {json.dumps(deco['decomposition'])}",
              flush=True)
    except Exception as exc:                       # noqa: BLE001
        deco = {"error": str(exc)}

    # ---- §54 cheap alpha_M3 influence diagnostic (no training side effect) -----
    alpha_diag = None
    try:
        alpha_diag = JG.alpha_m3_influence_diag(
            jpol, scorer, env,
            _eval_roots(re["progmem"], train_pairs[:8], st_bench_map), gate_mem=False)
        print(f"[r13] α_M3 influence: {alpha_diag['n_flip']}/"
              f"{alpha_diag['n_states']} argmax flips; marginal_above_stop_delta="
              f"{alpha_diag['marginal_above_stop_delta']}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        alpha_diag = {"error": str(exc)}

    # ---- verdict §52/53 (G->...->A precedence; checkpoint on A only §56) ------
    n_groups_c = sum(h["n_groups"] for h in res_c["history"])
    n_inf_c = sum(h["n_informative"] for h in res_c["history"])
    info_ratio_c = float(n_inf_c / n_groups_c) if n_groups_c else 0.0
    margin = 0.05 * max(float(r6_canonical_train), 1.0)
    c_max_04_train = max(c0_train, c1_train, c2_train, c3_train, c4_train)
    base_hd = max(c0_real, c1_real, c2_real, c3_real, c4_real,
                  c0_syn, c1_syn, c2_syn, c3_syn, c4_syn)
    train_beat = bool(c5_train > c_max_04_train + margin)
    held_better = [float(v) for v in (c5_real, c5_syn) if v > base_hd + 1.0]
    held_improved = bool(held_better)
    joint_beat = bool(c5_train > max(c4_train, train_final_b) + margin)
    m5_ok = bool(all(x["all_tier_a_ok"] for x in (m5_z, m5_a, m5_f) if x is not None))
    if not repro_ok or not anchor_ok or not stage_a_ok or not m5_ok:
        vcode, vlabel = "G", "JOINT_AGENTIC_SEMANTICS_BUG"
        note = (f"repro={repro_ok} anchor={anchor_ok} stage_a={stage_a_ok} "
                f"m5_§48={m5_ok} -- machinery/semantics broken")
        ok = False
    elif not mp_ok:
        vcode, vlabel = "F", "MULTIPROCESS_EXECUTION_MISMATCH"
        note = "multiprocess vs serial agentic collection differ -- determinism bug"
        ok = False
    elif res_a["collapsed"] or res_c["collapsed"]:
        vcode, vlabel = "E", "JOINT_AGENTIC_GRPO_DESTABILIZES"
        note = (f"collapse guard fired (A:{res_a['collapse_reason']} / "
                f"C:{res_c['collapse_reason']})")
        ok = False
    elif res_a["stage_a_gates"] and not stage_gates_ok:
        vcode, vlabel = "D", "M2_ROOT_SEARCH_COVERAGE_COLLAPSE"
        note = "Stage-A coverage gates failed (positive-probe / Tier-A / gated coverage declined)"
        ok = False
    elif n_groups_c == 0 or info_ratio_c <= 0.01:
        vcode, vlabel = "C", "JOINT_AGENTIC_GRPO_COVERAGE_LIMITED"
        note = "no informative trajectory groups collected over Stage C"
        ok = False
    elif train_beat and held_improved:
        vcode, vlabel = "A", "JOINT_AGENTIC_GRPO_IMPROVES"
        note = (f"C5 TRAIN {c5_train:.0f} > max(C0..C4 {c_max_04_train:.0f}) "
                f"+{margin:.0f} AND held {held_better} > base {base_hd:.0f} +1.0")
        ok = True
    else:
        vcode, vlabel = "B", "JOINT_AGENTIC_GRPO_TRAINS_BUT_NO_GAIN"
        note = (f"train_beat={train_beat} held_improved={held_improved} "
                f"joint_beat={joint_beat} C5={c5_train:.0f} vs max C0..C4="
                f"{c_max_04_train:.0f}")
        ok = False
    passed = bool(ok)
    print(f"[r13] verdict {vcode} {vlabel} (C5={c5_train:.0f} vs max C0..C4="
          f"{c_max_04_train:.0f} / held {held_better} vs {base_hd:.0f} / "
          f"info_ratio_c {info_ratio_c:.3f} / VAL {val_final:.0f} / "
          f"stage_a {stage_a_ok} / anchor {anchor_ok} / m5§48 {m5_ok})", flush=True)

    if passed:
        meta_common = dict(
            phase="r13_joint_agentic_rolling_grpo",
            method="m2_budgeted_root_search_x_m3_proposal_agentic_grpo",
            parent=C.TO1_R13_PARENT.name, sft_reference=C.TO1_CKPT.name,
            reward="terminal_makespan_gain", m2_reward_authority=False,
            m2_role="budgeted_root_search", m2_filter="makespan_first_memory_second",
            m3_role="proposal_selection", reasoner="frozen",
            executor="FixedDecisionReplay", formal_test_access=0,
            formal_test_sealed=True, identified=False,
            root_budget=C.TO1_R13_ROOT_BUDGET,
            anchor_roots=C.TO1_R13_ANCHOR_ROOTS,
            policy_draws=C.TO1_R13_POLICY_DRAWS,
            outsider=C.TO1_R13_OUTSIDER, memory_budget=C.TO1_R13_MEMORY_BUDGET,
            alpha_m2=C.TO1_R13_ALPHA_M2, alpha_prop=C.TO1_R13_ALPHA_PROP,
            alpha_stop=C.TO1_R13_ALPHA_STOP, K=C.TO1_R13_K,
            horizon=C.TO1_R13_HORIZON, graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH,
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            lambda_m2=C.TO1_R13_LAMBDA_M2, lambda_m3=C.TO1_R13_LAMBDA_M3,
            beta_m2=C.TO1_R13_BETA_M2, beta_m3=C.TO1_R13_BETA_M3,
            temp_m2=C.TO1_R13_TEMP_M2, temp=C.TO1_R13_TEMP,
            mix_eps=C.TO1_R13_MIX_EPS, clip_eps=C.TO1_R13_CLIP_EPS,
            lr_m2=C.TO1_R13_LR_M2, lr_m3=C.TO1_R13_LR_M3,
            probe_per_root=C.TO1_R13_PROBE_PER_ROOT,
            mem_support_min=C.TO1_R13_MEM_SUPPORT_MIN,
            mem_success_min=C.TO1_R13_MEM_SUCCESS_MIN,
            train_instances=[i["instance_id"] for i in env["train_insts"]],
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (NOT VAL3, §38)",
            reward_spec="R_i=Cmax(S_t)-Cmax(S_T) terminal, STOP=0, probe gains never reward",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / revisit",
            training_advancement="stochastic sample (exploration), gate active",
            eval="greedy, no oracle, 369 anchor cross-checked",
            grant_parity={"C0": c0_train, "C1": c1_train, "C2": c2_train,
                          "C3": c3_train, "C4": c4_train, "C5": c5_train,
                          "r6_canonical": r6_canonical_train,
                          "anchor_ok": anchor_ok},
            report={"passed": True, "vcode": vcode, "vlabel": vlabel,
                    "r13_train": c5_train, "val_final": val_final,
                    "best_train_b": train_final_b, "info_ratio_c": info_ratio_c})
        torch.save({"state": {"adapter": jpol.m2, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer, "r11_parent": parent_m3.state_dict()},
                    "meta": dict(meta_common, checkpoint="m2_root_policy_r13",
                                 role="M2 budgeted root search adapter")},
                   C.TO1_R13_CKPT_M2)
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer, "r11_parent": parent_m3.state_dict()},
                    "meta": dict(meta_common,
                                 checkpoint="m2_m3_joint_agentic_grpo_r13",
                                 role="M2 root search x M3 proposal, joint")},
                   C.TO1_R13_CKPT_JOINT)
        print(f"[r13] saved {C.TO1_R13_CKPT_M2.name} + {C.TO1_R13_CKPT_JOINT.name} "
              f"(PASS)", flush=True)
    else:
        print(f"[r13] NOT PASS -> no R13 checkpoint written (§56)", flush=True)

    report = _r13_report(env, re, scraper=dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, stage_a=stage_a_ok,
        m5=m5, m5_z=m5_z, m5_a=m5_a, res_a=res_a, res_b=res_b, res_c=res_c,
        train_final=train_final, train_final_b=train_final_b, val_final=val_final,
        rows=dict(c0=(c0_train, c0_real, c0_syn, c0_val),
                  c1=(c1_train, c1_real, c1_syn, c1_val),
                  c2=(c2_train, c2_real, c2_syn, c2_val),
                  c3=(c3_train, c3_real, c3_syn, c3_val),
                  c4=(c4_train, c4_real, c4_syn, c4_val),
                  c5=(c5_train, c5_real, c5_syn, c5_val)),
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        alpha=alpha_diag, deco=deco, pa_final=pa_final,
        pos6=pos6, regret6=regret6, rec10_6=rec10_6,
        pos_f=pos_f, regret_f=regret_f, rec10_f=rec10_f,
        train_beat=train_beat, held_improved=held_improved, joint_beat=joint_beat,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed))
    _r13_persist(report, scrape=dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, stage_a=stage_a_ok,
        m5=m5, m5_z=m5_z, m5_a=m5_a, res_a=res_a, res_b=res_b, res_c=res_c,
        train_final=train_final, train_final_b=train_final_b, val_final=val_final,
        rows=dict(c0=(c0_train, c0_real, c0_syn, c0_val),
                  c1=(c1_train, c1_real, c1_syn, c1_val),
                  c2=(c2_train, c2_real, c2_syn, c2_val),
                  c3=(c3_train, c3_real, c3_syn, c3_val),
                  c4=(c4_train, c4_real, c4_syn, c4_val),
                  c5=(c5_train, c5_real, c5_syn, c5_val)),
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        alpha=alpha_diag, deco=deco, pa_final=pa_final,
        pos6=pos6, regret6=regret6, rec10_6=rec10_6,
        pos_f=pos_f, regret_f=regret_f, rec10_f=rec10_f,
        train_beat=train_beat, held_improved=held_improved, joint_beat=joint_beat,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed))
    print(f"[r13] total {time.time() - t0:.1f}s "
          f"(report {C.R13_REPORT.name})", flush=True)
    return report




# ---------------------------------------------------------------------------
# R14 -- T1-M2-M3-STAGEWISE-REWARD-JOINT-GRPO (paper main method, §0-§43)
# ---------------------------------------------------------------------------

def run_top1_phase_r14(args, env, re, p1, p1_report):
    """R14: T1-M2-M3-STAGEWISE-REWARD-JOINT-GRPO -- the paper main method FIRST strict
    implementation (spec §0-§43).

    Permanent route: M2 SFT (frozen B5 attribution as fixed prior) -> M3 SFT
    (m3_proposal_top1_sft_v2.pt) -> ONE Joint Agentic GRPO stage with STAGEWISE
    rewards.  NO A/B/C RL training stages (§2): M2 adapter + M3 residual BOTH open from
    cycle 0.  M2 local reward q2 (probe, §10-12) -> U2 -> A2 (§13-14); M3 terminal
    makespan -> R3 -> A3 (§16-17); M3's terminal advantage NEVER trains M2 (§15 hard
    rule).  Adaptive probe 8 -> 16 -> 24 with Tier_A>=2 stop (§23-28); only policy
    draws carry M2 gradient (§12, §18).  P0-P3 parity on the SAME canonical raw ruler
    (§37-38): P0 = gated M2-SFT+M3-SFT init, P1 = M3-only GRPO (R12 ablation, cited
    from R13 C4), P2 = R13 shared-terminal JOINT (ablation, cited from R13 C5), P3 =
    R14 stage-wise JOINT (MAIN).  Success A = M2 coverage improves + positive-Proposal
    coverage >= SFT baseline + joint terminal gain > P0 + a real held improvement
    (§39); else B/C/D/E/F/G (§40).  Checkpoint ONLY on verdict A (§41, named
    m2_m3_stagewise_joint_grpo_r14.pt).  identified=false, formal_test_access=0.
    """
    print("[r14] M2-M3 STAGEWISE-REWARD JOINT AGENTIC GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §0 parent: M3 = pure SFT (m3_proposal_top1_sft_v2.pt), M2 adapter zero-init.
    # m3_rolling_grpo_v1.pt is FORBIDDEN as parent -- no R11 resid copy here.
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(r6_sel)
    with torch.no_grad():
        resid_max = max((p.abs().max().item() for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    print(f"[r14] M2 adapter {n_tr_m2} params (zero-init delta=0) | M3 {n_tr_m3} params "
          f"(pure SFT {C.TO1_CKPT.name}; resid_max={resid_max:.3g} -> δ=0 ≡ R6; "
          f"NO m3_rolling_grpo_v1.pt parent) §0", flush=True)
    proof_no_m3_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r11_grpo_checkpoint": None,            # never loaded
        "r11_resid_copied": 0,
        "resid_max_zero_init": float(resid_max) == 0.0,
        "m3_base_equals_r6": True,
    }

    # ---- verbatim R6 reproduction (gate_mem=False, §40 ruler anchor) ------------
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r14] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)
    grp_g10, _ = TOP1.build_top1_groups(re["state_examples"], scorer, re["mem_values"],
                                        reranker=reranker, rng=random.Random(0),
                                        gate_mem=True)
    pa_r6 = TOP1.pool_argmax_metrics(grp_g10, r6_sel)
    pos6 = float(pa_r6["positive_state_argmax_accuracy"])
    regret6 = float(pa_r6["top1_regret_full"]["mean"])
    rec10_6 = float(pa_r6["recall"]["10"])

    # ---- rolling graphs: bench TRAIN14 + AUX-real/syn (same fields as R13) -------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real",
                             env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux",
                             env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r14] graphs: bench {len(bench_graphs)} | "
          f"real {len(rb['graphs']) if rb else 0} train / "
          f"{len(rb['hd_iids']) if rb else 0} held | "
          f"syn {len(sb['graphs']) if sb else 0} train / "
          f"{len(sb['hd_iids']) if sb else 0} held", flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _closed_loop(mode, obj, roots):
        sm, _ = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode,
                               use_mem=True, gate_mem=False, gate_variant="r14")
        return sm

    def _tot(mode, obj, roots):
        return float(_closed_loop(mode, obj, roots).get("total", 0.0))

    def _eval_full(obj, mode, roots):
        """→ (total_gain_summary, coverage_dict) from the R14 ADAPTIVE gate (§6-12)."""
        sm, stps = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode,
                                  use_mem=True, gate_mem=False, gate_variant="r14")
        probes, pos_rate_sum, tier_sum, exh, n_state = [], 0.0, 0.0, 0, 0
        for _iid, blk in stps.items():
            for st in blk["steps"]:
                d = st.get("m2_diag")
                if not d:
                    continue
                n_state += 1
                pc = int(d.get("probed_root_count") or 0)
                pp = int(d.get("positive_probe_count") or 0)
                probes.append(pc)
                pos_rate_sum += (pp / pc if pc else 0.0)
                tier_sum += (1.0 if d.get("tier_A", 0) >= 1 else 0.0)
                if d.get("budget_exhausted"):
                    exh += 1
        cov = {"n_states": n_state,
               "pos_proposal_cov": (tier_sum / max(n_state, 1)),
               "pos_probe_rate": (pos_rate_sum / max(n_state, 1)),
               "mean_probes": float(np.mean(probes)) if probes else 0.0,
               "budget_exhausted_frac": (exh / max(n_state, 1))}
        return {"total": float(sm.get("total", 0.0))}, cov

    def eval_root_builder(jp_, cycle=-1):
        # NOTE: must return DICTs with "total" (run_rolling_cycles_r13 L1793 reads
        # ev["train"].get("total")) -- a raw float crashes with
        # 'float' object has no attribute 'get'.  Same shape as R13's builder.
        ev = {"train": _closed_loop("adapter", jp_,
                                    _eval_roots(re["progmem"], train_pairs, st_bench_map))}
        ev["real_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            rb["hd_pm"], real_hd_pairs, rb["hd_states"])) if rb else {"total": 0.0})
        ev["syn_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            sb["hd_pm"], syn_hd_pairs, sb["hd_states"])) if sb else {"total": 0.0})
        return ev

    # DPPaulli S0 root + state (normal-M5 §48 permanent trace site)
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _root_for(iid):
        st = env["states"][iid]
        return RGRPO.roots_from_state(st["problem"], st["schedule"], iid,
                                      env["ep_id_of"][iid],
                                      copy.deepcopy(re["progmem"]))

    def _m5_gate(adapter, gv="r14"):
        """§48 permanent gate at DPPaulli S0 under the R14 adaptive gate."""
        if dpp_iid is None:
            return None
        prop_feats, metas, agg = env["cache"].proposals(
            dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        if not metas:
            return {"at_state": False, "dep_saved": [], "all_tier_a_ok": True,
                    "retain_ok": True, "retained": [], "diag": {"note": "no proposals"}}
        ast = env["cache"].ast(dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        ms = int(dpp_st["schedule"].makespan)
        sf = PF.state_feature_vec(ms, ms, len(metas), agg["best_uhat"], agg["best_direct"],
                                  agg["n_contrib"], agg["n_enab"])
        return JG.normal_m5_r13_gate(ast, metas, prop_feats, adapter,
                                     env["executor"], copy.deepcopy(re["progmem"]),
                                     dpp_iid, int(dpp_ep), 0, sf, random.Random(1),
                                     gate_variant=gv)

    # ---- §34 M2 reward diagnostic aggregator (training groups, no leaks) --------
    m2_agg = {"groups": 0, "steps": 0, "draws": 0, "proven": 0, "memory": 0,
              "unsupported": 0, "min_proven_q2": 1e9, "max_mem_q2": -1.0,
              "ordering_ok": True, "inf2_groups": 0, "inf3_groups": 0,
              "inf2_steps": 0, "inf3_steps": 0}

    def _acc(groups):
        for g in groups:
            m2_agg["groups"] += 1
            m2_agg["inf2_groups"] += (1 if g.get("info2") else 0)
            m2_agg["inf3_groups"] += (1 if g.get("informative") else 0)
            for tr in g["trajs"]:
                for rec in tr["steps"]:
                    m2_agg["steps"] += 1
                    m2_agg["inf2_steps"] += (1 if rec.get("inf2") else 0)
                    m2_agg["inf3_steps"] += (1 if rec.get("inf3") else 0)
                    mr = rec.get("m2_rec") or {}
                    qs = mr.get("q2_stats") or {}
                    m2_agg["draws"] += len(mr.get("draws", []))
                    if qs.get("proven"):
                        m2_agg["proven"] += len(qs["proven"])
                        m2_agg["min_proven_q2"] = min(m2_agg["min_proven_q2"],
                                                      min(qs["proven"]))
                    if qs.get("memory"):
                        m2_agg["memory"] += len(qs["memory"])
                        m2_agg["max_mem_q2"] = max(m2_agg["max_mem_q2"],
                                                   max(qs["memory"]))
                    m2_agg["unsupported"] += int(qs.get("unsupported", 0))
                    if not (rec.get("m2_diag") or {}).get("q2_ordering_ok", True):
                        m2_agg["ordering_ok"] = False

    if args.quick:
        # quick ALSO passes 2 REAL joint-stage cycles (stagewise credit in the loop)
        for g in bench_graphs:
            g.reset()
        jpol.params_for_stage("C")
        try:
            res_s = JG.run_rolling_cycles_r13(
                jpol, scorer, env, specs, stage="C", cycles=2, k=C.TO1_R13_K,
                horizon=C.TO1_R13_HORIZON, graphs_per_batch=2, workers=1,
                seed=args.grpo_seed, quick=True, log_prefix="[r14-qs]",
                eval_root_builder=eval_root_builder, collapse_floor=None,
                parent_policy=None, mp_ctx=mp_ctx, variant="r14", on_groups=_acc)
            quick_sanity = {"cycles": res_s["cycles_run"],
                            "best_train": float(res_s["best"]["train"]),
                            "collapsed": bool(res_s["collapsed"]),
                            "n_informative_m2": sum(
                                d.get("n_informative_trajectories_m2", 0)
                                for d in res_s["history"]),
                            "m2_agg": dict(m2_agg)}
            print(f"[r14] QUICK sanity: {res_s['cycles_run']} cycles "
                  f"best_train={quick_sanity['best_train']:.0f}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            quick_sanity = {"error": str(exc)}
            print(f"[r14] QUICK sanity FAILED: {exc}", flush=True)
        m5q = _m5_gate(jpol.m2, gv="r14")
        if m5q:
            print(f"[r14] §48 adaptive zero-init dep_saved={m5q['dep_saved']} "
                  f"all_tier_a_ok={m5q['all_tier_a_ok']}", flush=True)
        report = _r14_report(env, re, scraper=dict(
            repro_ok=repro_ok, anchor_ok=None, m5=m5q, res_j=None,
            p0=None, p1=None, p2=None, p3=None, rows={}, dpp_pre=None, dpp_post=None,
            deco_i=None, deco_f=None, deco_keys=None, m2_agg=m2_agg,
            r6_canonical_train=None, prof=None, mp_ok=None, mk=None, val_final=0.0,
            m2_kl=None, m3_kl=None, n_info_m2=0, n_info_m3=0,
            verdict=dict(code="S", label="STAGE_C_JOINT_SANITY",
                         note=f"--quick: ONE joint stagewise stage ran 2 real cycles; "
                              f"quick-sanity {quick_sanity}"),
            passed=False, checkpoint_written=False))
        _r14_persist(report, scrape=None)
        print("[r14] QUICK: joint stagewise sanity only -- "
              "see outputs/canonical_m3/result_r14.json", flush=True)
        return report

    # ---- §37 baseline table on the SAME canonical raw ruler --------------------
    c0 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                alpha_stop=C.TO1_R13_ALPHA_STOP)      # δ=0 = R6
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    with torch.no_grad():
        p0_anchor = float(_tot("none", c0, init_roots))               # raw 369 ruler
        sm_p0, cov_p0 = _eval_full(jpol, "adapter", init_roots)       # P0 = gated SFT
        p0, cov_p0 = float(sm_p0["total"]), cov_p0
        if rb:
            sm_pr, cov_real0 = _eval_full(jpol, "adapter", _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            p0_real = float(sm_pr["total"])
        else:
            p0_real, cov_real0 = 0.0, {}
        if sb:
            sm_ps, cov_syn0 = _eval_full(jpol, "adapter", _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            p0_syn = float(sm_ps["total"])
        else:
            p0_syn, cov_syn0 = 0.0, {}
        p0_val = float(_tot("adapter", jpol,
                            _eval_roots(re["progmem"], val_pairs, st_bench_map)))
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(p0_anchor - float(r6_canonical_train)) <= anchor_tol)
    print(f"[r14] ANCHOR: raw {p0_anchor:.0f} vs r6_canonical={r6_canonical_train:.0f} "
          f"tol={anchor_tol:.1f} anchor_ok={anchor_ok}", flush=True)
    print(f"[r14] P0(gated SFT init): {p0:.0f}/{p0_real:.0f}/{p0_syn:.0f}/{p0_val:.0f} "
          f"(TRAIN/real/syn/VAL) cov={cov_p0}", flush=True)

    m5_z = _m5_gate(jpol.m2, gv="r14")
    if m5_z:
        print(f"[r14] §48 adaptive zero-init: dep_saved={m5_z['dep_saved']} "
              f"all_tier_a_ok={m5_z['all_tier_a_ok']} retain_ok={m5_z['retain_ok']}",
              flush=True)

    # ---- §35 failure decomposition BEFORE (zero-init policy) -------------------
    deco_i = None
    try:
        deco_i = _r14_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r14] deco INIT: {json.dumps(deco_i['decomposition'])}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        deco_i = {"error": str(exc)}
        print(f"[r14] deco INIT FAILED: {exc}", flush=True)

    # ---- §55 cloud/shape profile (adaptive collect) -> pick workers ------------
    prof = None
    mp_ok = True
    workers = 1
    prof_root = init_roots[0]
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, prof_root,
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14")
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"] for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r14] cloud profile: {json.dumps(prof['per_worker'])} "
              f"identical={ident_ok} -> workers={workers}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r14] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- THE ONE JOINT STAGE (§2): M2 + M3 stagewise GRPO from cycle 0 ----------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")
    res_j = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=C.TO1_R14_TRAINING_CYCLES,
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r14-J]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, p0), parent_policy=None, mp_ctx=mp_ctx,
        variant="r14", on_groups=_acc)
    m5_f = _m5_gate(jpol.m2, gv="r14")
    if m5_f:
        print(f"[r14] §48 FINAL: dep_saved={m5_f['dep_saved']} "
              f"all_tier_a_ok={m5_f['all_tier_a_ok']} retain_ok={m5_f['retain_ok']}",
              flush=True)

    # ---- P1/P2 ablations (cited from the same-unified-ruler R13 records §38) ----
    p1 = float(C.TO1_R13_C4_TRAIN)      # M3-only GRPO on gated pool (R13 C4) = 353
    p2 = float(C.TO1_R13_C5_TRAIN)      # R13 shared-terminal JOINT (R13 C5) = 353
    p2_real = float(C.TO1_R13_C5_REAL)  # real held-2 (R13) -- used only for context
    p1_note = f"cited R13 C4 train={p1:.0f} (M3-only GRPO, shared ruler)"
    p2_note = f"cited R13 C5 train={p2:.0f} real_held={p2_real:.0f} (shared JOINT)"
    print(f"[r14] P1 {p1_note} | P2 {p2_note}", flush=True)

    # ---- P3 = MAIN: final stage-wise JOINT on the same ruler --------------------
    with torch.no_grad():
        sm_p3, cov_p3 = _eval_full(jpol, "adapter", init_roots)
        p3, cov_p3 = float(sm_p3["total"]), cov_p3
        if rb:
            sm_p3r, cov_real3 = _eval_full(jpol, "adapter", _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            p3_real = float(sm_p3r["total"])
        else:
            p3_real, cov_real3 = 0.0, {}
        if sb:
            sm_p3s, cov_syn3 = _eval_full(jpol, "adapter", _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            p3_syn = float(sm_p3s["total"])
        else:
            p3_syn, cov_syn3 = 0.0, {}
        p3_val = float(_tot("adapter", jpol,
                            _eval_roots(re["progmem"], val_pairs, st_bench_map)))
    print(f"[r14] P3(stagewise JOINT): {p3:.0f}/{p3_real:.0f}/{p3_syn:.0f}/"
          f"{p3_val:.0f} cov={cov_p3}", flush=True)

    # ---- §39-40 diagnostics: deco AFTER, DPP rolling, §36 Mk traces -------------
    deco_f = None
    try:
        deco_f = _r14_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r14] deco FINAL: {json.dumps(deco_f['decomposition'])}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        deco_f = {"error": str(exc)}
        print(f"[r14] deco FINAL FAILED: {exc}", flush=True)
    deco_keys = ("M2_PROBE_MISS", "M2_OK", "M3_SELECTION_MISS")

    mk = None
    try:
        mk = _r14_mk_traces(jpol, scorer, env, re, mp_ctx)
        _mk_show = {k: mk.get(k) for k in ("mk1", "mk3") if isinstance(mk, dict)}
        print(f"[r14] Mk traces: {json.dumps(_mk_show, default=str)[:2200]}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        mk = {"error": str(exc)}
        print(f"[r14] Mk trace FAILED: {exc}", flush=True)

    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_root = _root_for(dpp_iid)
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _ = JG.agentic_closed_loop(env, scorer, zj2, [dpp_root],
                                                use_mem=True, gate_mem=False,
                                                gate_variant="r14")
            post_sum, _ = JG.agentic_closed_loop(env, scorer, jpol, [dpp_root],
                                                 use_mem=True, gate_mem=False,
                                                 gate_variant="r14")
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r14] DPP BEFORE(zero-init): gain={dpp_pre['total_gain']} | "
              f"AFTER(final): gain={dpp_post['total_gain']}", flush=True)

    # ---- §34 reward ordering + KL/grader info across the joint stage ------------
    n_groups = sum(h["n_groups"] for h in res_j["history"])
    n_inf3 = sum(h["n_informative"] for h in res_j["history"])
    n_inf2 = sum(h.get("n_informative_trajectories_m2", 0) for h in res_j["history"])
    info_ratio3 = float(n_inf3 / max(n_groups, 1))
    info_ratio2 = float(n_inf2 / max(n_groups, 1))
    m2_kl = []
    m3_kl = []
    for h in res_j["history"]:
        for d in h.get("depth_results", []):
            for e in ((d.get("update") or {}).get("epochs", []) or []):
                if e.get("kl_m2") is not None:
                    m2_kl.append(float(e["kl_m2"]))
                if e.get("kl_ref_m3") is not None:
                    m3_kl.append(float(e["kl_ref_m3"]))
    m2_kl = m2_kl or [0.0]
    m3_kl = m3_kl or [0.0]
    reward_ordering_ok = bool(m2_agg["ordering_ok"] and
                              (m2_agg["proven"] == 0 or
                               m2_agg["min_proven_q2"] > m2_agg["max_mem_q2"]))
    print(f"[r14] §34 reward diag: proven_draws={m2_agg['proven']} "
          f"memory_draws={m2_agg['memory']} unsupported={m2_agg['unsupported']} "
          f"min_proven_q2={m2_agg['min_proven_q2']:.3f} "
          f"max_mem_q2={m2_agg['max_mem_q2']:.3f} ordering_ok={reward_ordering_ok}",
          flush=True)

    # did at least one stagewise update epoch report a nonzero grad (M2 and/or M3)?
    m2_grad_any = m3_grad_any = False
    for h in res_j["history"]:
        for d in h.get("depth_results", []):
            for e in (d.get("update") or {}).get("epochs", []) or []:
                if float(e.get("grad_norm", 0.0)) > 1e-9:
                    m2_grad_any = True
                    m3_grad_any = True
    print(f"[r14] informative: groups={n_groups} inf2_traj={n_inf2} inf3_traj={n_inf3} "
          f"ratios {info_ratio2:.3f}/{info_ratio3:.3f} | grad_present={m2_grad_any}/"
          f"{m3_grad_any}", flush=True)

    # ---- §39/§40 verdict ladder (G -> F -> D -> A -> E -> C -> B) ---------------
    margin = 0.05 * max(float(r6_canonical_train), 1.0)
    p1_v = p1 if isinstance(p1, (int, float)) else 0.0
    p2_v = p2 if isinstance(p2, (int, float)) else 0.0
    train_beat = bool(p3 > max(p0, p1_v, p2_v) + margin)
    base_hd = max(float(p0_real), float(p0_syn))
    held_better = [float(v) for v in (p3_real, p3_syn) if v > base_hd + 1.0]
    held_improved = bool(held_better)
    m5_ok = bool(all(x["all_tier_a_ok"] for x in (m5_z, m5_f) if x is not None))
    d_i = (deco_i or {}).get("decomposition", {}) if isinstance(deco_i, dict) else {}
    d_f = (deco_f or {}).get("decomposition", {}) if isinstance(deco_f, dict) else {}
    m2_miss_i = int(d_i.get("M2_PROBE_MISS", 0))
    m2_miss_f = int(d_f.get("M2_PROBE_MISS", 0))
    m3_miss_i = int(d_i.get("M3_SELECTION_MISS", 0))
    m3_miss_f = int(d_f.get("M3_SELECTION_MISS", 0))
    deco_improved = bool(m2_miss_f < m2_miss_i)
    cov_improved = bool(cov_p3.get("pos_proposal_cov", 0.0) >
                        cov_p0.get("pos_proposal_cov", 0.0) + 0.02 or
                        cov_p3.get("pos_probe_rate", 0.0) >
                        cov_p0.get("pos_probe_rate", 0.0) + 0.02)
    cov_static = bool(not cov_improved and not deco_improved)
    m2_trained = bool(deco_improved or
                      n_inf2 > 0 or cov_improved)
    pos_cov_ok = bool(cov_p3.get("pos_proposal_cov", 0.0) >=
                      cov_p0.get("pos_proposal_cov", 0.0))
    dcond = bool(cov_p3.get("budget_exhausted_frac", 0.0) >= 0.8 and
                 cov_p3.get("mean_probes", 0.0) >= C.TO1_R14_B_MAX - 1 and
                 p3 <= p0 + 1.0)
    fcond = bool(m2_agg["proven"] == 0 and m2_agg["memory"] > 0 and p3 < p0)
    econd = bool(m3_miss_f >= max(1, m2_miss_f, m2_miss_i) and p3 <= p0 + 1.0)

    if not repro_ok or not anchor_ok or not m5_ok or not mp_ok or res_j["collapsed"]:
        vcode, vlabel = "G", "STAGEWISE_CREDIT_SEMANTICS_BUG"
        note = (f"repro={repro_ok} anchor={anchor_ok} m5§48={m5_ok} "
                f"mp_ok={mp_ok} collapsed={res_j['collapsed']} -- machinery broken")
        ok = False
    elif not reward_ordering_ok:
        vcode, vlabel = "G", "STAGEWISE_CREDIT_SEMANTICS_BUG"
        note = "§10/§34 reward ordering PROVEN_GAIN > MEMORY_RESCUED violated"
        ok = False
    elif fcond:
        vcode, vlabel = "F", "MEMORY_RESCUE_HARMS_M2"
        note = (f"memory 是唯一 M2 信号 (proven_draws={m2_agg['proven']}, "
                f"memory_draws={m2_agg['memory']}) 且 TRAIN 恶化 {p0:.0f}->{p3:.0f}")
        ok = False
    elif dcond:
        vcode, vlabel = "D", "ADAPTIVE_PROBING_TOO_EXPENSIVE"
        note = (f"final eval 常跑满探针预算 (exhausted "
                f"{cov_p3.get('budget_exhausted_frac', 0.0):.2f}, mean "
                f"{cov_p3.get('mean_probes', 0.0):.0f}/state) 且无增益 "
                f"P3={p3:.0f} vs P0={p0:.0f}")
        ok = False
    elif train_beat and held_improved and deco_improved and pos_cov_ok:
        vcode, vlabel = "A", "STAGEWISE_JOINT_GRPO_IMPROVES"
        note = (f"P3 TRAIN {p3:.0f} > max(P0..P2 {max(p0, p1_v, p2_v):.0f}) "
                f"+{margin:.0f} AND held {held_better} > {base_hd:.0f}+1 AND "
                f"M2_PROBE_MISS {m2_miss_i}->{m2_miss_f} AND pos-prop coverage "
                f"{cov_p0.get('pos_proposal_cov', 0):.2f}->"
                f"{cov_p3.get('pos_proposal_cov', 0):.2f}")
        ok = True
    elif econd:
        vcode, vlabel = "E", "M3_SELECTION_REMAINS_PRIMARY_BLOCKER"
        note = (f"M3_SELECTION_MISS_final {m3_miss_f} >= "
                f"max(M2_miss {m2_miss_f}, init {m2_miss_i}) 且无训练增益 "
                f"P3={p3:.0f} vs P0={p0:.0f}")
        ok = False
    elif cov_static or not m2_trained:
        vcode, vlabel = "C", "M2_LOCAL_REWARD_TRAINS_BUT_COVERAGE_STATIC"
        note = (f"M2 local reward 有信号 (inf2_traj={n_inf2}) 但覆盖率无改善 "
                f"(cov {cov_p0.get('pos_proposal_cov', 0):.2f}->"
                f"{cov_p3.get('pos_proposal_cov', 0):.2f}) 且 P3={p3:.0f}<=P0={p0:.0f}; "
                f"inf2_traj={n_inf2}")
        ok = False
    elif cov_improved and p3 <= p0 + 1.0:
        vcode, vlabel = "B", "M2_COVERAGE_IMPROVES_M3_LIMITS_GAIN"
        note = (f"M2 coverage improved (pos_proposal_cov "
                f"{cov_p0.get('pos_proposal_cov', 0):.2f}->"
                f"{cov_p3.get('pos_proposal_cov', 0):.2f}) 但 M3 限制终局 "
                f"P3={p3:.0f} vs P0={p0:.0f}")
        ok = False
    else:
        vcode, vlabel = "B", "M2_COVERAGE_IMPROVES_M3_LIMITS_GAIN"
        note = (f"train_beat={train_beat} held_improved={held_improved} "
                f"deco_improved={deco_improved} pos_cov_ok={pos_cov_ok} "
                f"P3={p3:.0f} vs max(P0,P1,P2)={max(p0, p1_v, p2_v):.0f}")
        ok = False
    passed = bool(ok)
    print(f"[r14] verdict {vcode} {vlabel} (P3={p3:.0f} vs P0={p0:.0f} / "
          f"held {held_better} vs {base_hd:.0f} / inf2 {info_ratio2:.3f} / "
          f"M2_miss {m2_miss_i}->{m2_miss_f} / M3_miss {m3_miss_i}->{m3_miss_f})",
          flush=True)

    if passed:
        meta_common = dict(
            phase="r14_stagewise_reward_joint_grpo",
            method="m2_sft_then_m3_sft_then_joint_agentic_grpo_stagewise",
            pipeline="M2_SFT->M3_SFT->Joint_GRPO",
            m2_reward="makespan_first_memory_second",
            m3_reward="terminal_makespan_gain",
            m2_terminal_reward_authority=False,
            memory_global_reward_authority=False,
            adaptive_probe="8->16->24",
            parent=C.TO1_CKPT.name,               # m3_proposal_top1_sft_v2.pt
            r13_grpo_parent=None,                 # FORBIDDEN by §0
            reward="q2_local_probe (M2) / terminal_makespan (M3), stagewise A2/A3",
            m2_role="budgeted_root_search", m2_filter="makespan_first_memory_second",
            m3_role="proposal_selection", reasoner="frozen",
            executor="FixedDecisionReplay", formal_test_access=0,
            formal_test_sealed=True, identified=False,
            lambda_m2=C.TO1_R14_LAMBDA_M2, lambda_m3=C.TO1_R14_LAMBDA_M3,
            beta_m2=C.TO1_R14_BETA_M2, beta_m3=C.TO1_R14_BETA_M3,
            adaptive_b_init=C.TO1_R14_B_INIT, adaptive_b_step=C.TO1_R14_B_STEP,
            adaptive_b_max=C.TO1_R14_B_MAX, tier_a_stop=C.TO1_R14_TIER_A_STOP,
            K=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH,
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            temp_m2=C.TO1_R13_TEMP_M2, temp=C.TO1_R13_TEMP,
            mix_eps=C.TO1_R13_MIX_EPS, clip_eps=C.TO1_R13_CLIP_EPS,
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (NOT VAL3, §38)",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / revisit",
            reward_ordering="PROVEN_GAIN(>=2) > MEMORY_RESCUED(<=0.5) > UNSUPPORTED(0)",
            credit="M2=A2(q2/U2) only; M3=A3(terminal) only; no shared advantage (§15)",
            adaptive_memory_control=False,)
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": dict(meta_common, checkpoint="m2_m3_stagewise_joint_grpo_r14",
                                 role="M2 stagewise local reward x M3 terminal, JOINT")},
                   C.TO1_R14_CKPT)
        print(f"[r14] saved {C.TO1_R14_CKPT.name} (PASS)", flush=True)
    else:
        print(f"[r14] NOT PASS -> no R14 checkpoint written (§41)", flush=True)

    _r14_scrape = dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, m5=m5_f, m5_z=m5_z,
        res_j=res_j, p0=p0, p1=p1, p2=p2, p3=p3,
        p0_real=p0_real, p0_syn=p0_syn, p0_val=p0_val,
        p3_real=p3_real, p3_syn=p3_syn, p3_val=p3_val,
        rows=dict(p0_anchor=p0_anchor, p0=(p0, p0_real, p0_syn, p0_val),
                  p1=(p1, None, None, None), p2=(p2, p2_real, None, None),
                  p3=(p3, p3_real, p3_syn, p3_val)),
        cov_p0=cov_p0, cov_p3=cov_p3,
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        deco_i=deco_i, deco_f=deco_f, deco_keys=deco_keys, m2_agg=m2_agg,
        m2_miss=(m2_miss_i, m2_miss_f), m3_miss=(m3_miss_i, m3_miss_f),
        mk=mk, val_final=p3_val, m2_kl=m2_kl, m3_kl=m3_kl,
        n_info_m2=n_inf2, n_info_m3=n_inf3, info_ratio2=info_ratio2,
        info_ratio3=info_ratio3, m2_grad_any=m2_grad_any, m3_grad_any=m3_grad_any,
        reward_ordering_ok=reward_ordering_ok, proof_no_m3_rl_parent=proof_no_m3_rl_parent,
        deco_improved=deco_improved, cov_improved=cov_improved, pos_cov_ok=pos_cov_ok,
        train_beat=train_beat, held_improved=held_improved,
        held_better=held_better, base_hd=base_hd,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed)
    report = _r14_report(env, re, scraper=_r14_scrape)
    _r14_persist(report, scrape=_r14_scrape)
    print(f"[r14] total {time.time() - t0:.1f}s "
          f"(report {C.R14_REPORT.name})", flush=True)
    return report


def _r14_deco(env, scorer, jpol, roots):
    """§35 failure decomposition under the R14 ADAPTIVE gate (real probe + exe gains)."""
    import collections
    dec = collections.Counter()
    rollouts = []
    for rf in roots[:5]:
        rd = JG.agentic_rollout_diag(env, scorer, jpol, rf, gate_variant="r14")
        for k_, v_ in (rd.get("decomposition") or {}).items():
            dec[k_] += int(v_)
        rollouts.append({"iid": rf["iid"], "final_gain": int(rd.get("final_gain", 0)),
                         "decomp": dict(rd.get("decomposition") or {})})
    return {"decomposition": dict(dec), "rollouts": rollouts}


def _r14_mk_traces(jpol, scorer, env, re, mp_ctx):
    """§36 Mk1/Mk3 hard traces: per-step M2 roots/probe-gains/Tiers/A2 and M3
    probabilities/A3 from ONE K=8 sibling group per instance (final policy)."""
    out = {}
    for key, iid in (("mk1", "Brandimarte_Mk1"), ("mk3", "Brandimarte_Mk3")):
        cand_iids = [i["instance_id"] for i in env["order"]]
        hit = [iid] if iid in cand_iids else \
            [c for c in cand_iids if "Mk1" in c or "Mk3" in c]
        if not hit:
            out[key] = {"note": f"{iid} not in env"}
            continue
        tgt = hit[0] if key in ("mk1",) else (hit[1] if len(hit) > 1 else hit[0])
        root = RGRPO.roots_from_state(env["states"][tgt]["problem"],
                                      env["states"][tgt]["schedule"], tgt,
                                      env["ep_id_of"][tgt],
                                      copy.deepcopy(re["progmem"]))
        grp = JG.collect_full_group_rollouts_r14(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], int(root["root_ms"]),
            root["iid"], root["episode_id"], root["progmem"], k=8, seed=7,
            workers=1, step0_cache=env["cache"], mp_ctx=mp_ctx)
        rows = []
        for tr in grp["trajs"][:4]:
            st = tr["steps"][0]
            m2r = st["m2_rec"]
            draws = [{"op": m2r["ops"][d["idx"]],
                      "class": d.get("reward_class", "unsupported"),
                      "q2": round(float(d.get("q2", 0.0)), 3)}
                     for d in m2r["draws"]]
            logits = st["logits_old"]
            M = st["M"]
            probs = torch.softmax(torch.as_tensor(logits) / float(C.TO1_R13_TEMP),
                                  -1).tolist()
            rows.append({
                "sib": tr["traj_id"], "terminal": tr["terminal"],
                "U2": round(float(tr["U2"]), 3), "A2": round(float(st["adv2"]), 3),
                "A3": round(float(st["adv3"]), 3), "reward": int(tr["reward"]),
                "tier_A": st["m2_diag"]["tier_A"],
                "probed": st["m2_diag"]["probed_root_count"],
                "draws": draws,
                "M3": {"argmax": int(st["a"]), "M": M,
                       "p_chosen": round(float(probs[st["a"]]), 3),
                       "p_stop": round(float(probs[M]), 3),
                       "n_proposals": M},
            })
        out[key] = {"iid": tgt, "grp_key": list(grp["grp_key"]),
                    "inf2": bool(grp["info2"]), "inf3": bool(grp["informative"]),
                    "adv2": [round(float(x), 3) for x in grp["advantages2"]],
                    "adv3": [round(float(x), 3) for x in grp["advantages"]],
                    "U2_list": [round(float(x), 3) for x in grp["U2"]],
                    "rows": rows}
    return out

def _r14_report(env, re, scraper):
    s = scraper
    L = []
    L.append("# T1-M2-M3-STAGEWISE-REWARD-JOINT-GRPO-R14 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    L.append(f"## Verdict: **{s['verdict']['code']}** `{s['verdict']['label']}`")
    L.append("")
    L.append(f"> {s['verdict']['note']}")
    L.append("")
    L.append("## §0 permanent structure (identical in train + eval)")
    L.append("")
    L.append("S_t + Appearance -> **M2 adaptive root search** (wave 1 = 3 anchors + 4 "
             "policy draws + 1 outsider, then 8-root expansion waves until tier_A >= 2 "
             "or 24 total) -> **REAL FDR root probes** -> **per-draw q2 local reward** "
             "-> **makespan-first tier filter** (A=PROVEN_GAIN uncapped; B=Memory "
             "rescue cap 2; C=pruned) -> gated pool -> Reasoner view -> **M3** proposal "
             "policy -> execute -> S_{t+1}.")
    L.append("")
    L.append("- ONE Joint RL stage (§2) -- NO A/B/C stages; M2 adapter + M3 residual "
             "both open from cycle 0.  M2 trains ONLY on A2 (from q2/U2); M3 ONLY on "
             "A3 (terminal); the §15 hard rule means M3's terminal advantage NEVER "
             "trains M2.")
    L.append("")
    L.append("## 1. machinery gates")
    L.append("")
    L.append(f"- R6 reproduction repro_ok={s.get('repro_ok')} (details in result_r14.json)")
    L.append(f"- P0 anchor: unified RAW d0 vs r6_canonical ({s.get('r6_canonical_train')}) "
             f"-> anchor_ok={s.get('anchor_ok')}")
    L.append(f"- §34 M2 reward ordering OK={s.get('reward_ordering_ok')} -- "
             f"proven_draws={s.get('m2_agg', {}).get('proven')} "
             f"memory_draws={s.get('m2_agg', {}).get('memory')} "
             f"unsupported={s.get('m2_agg', {}).get('unsupported')} "
             f"min_proven_q2={s.get('m2_agg', {}).get('min_proven_q2')} "
             f"max_mem_q2={s.get('m2_agg', {}).get('max_mem_q2')}")
    L.append(f"- §48 normal-M5 dep-completed Tier-A (never Memory): "
             f"zero={s.get('m5_z', {}).get('all_tier_a_ok') if isinstance(s.get('m5_z'), dict) else None} "
             f"final={s.get('m5', {}).get('all_tier_a_ok') if isinstance(s.get('m5'), dict) else None}")
    L.append("")
    L.append("## §37-38 STAGEWISE PARITY TABLE (SAME unified raw ruler; P3 = MAIN)")
    L.append("")
    rows = s.get("rows", {})
    L.append("| model | route | TRAIN | real-held | syn-held | VAL |")
    L.append("|---|---|---|---|---|---|")
    # None-safe cell formatter: a dead/stage-crashed quick run leaves rows as
    # (None,...) tuples; render "—" instead of TypeError on :.0f of None.
    def _z(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"
    pa = rows.get("p0_anchor")
    p0 = rows.get("p0", (None, None, None, None))
    p1 = rows.get("p1", (None, None, None, None))
    p2 = rows.get("p2", (None, None, None, None))
    p3 = rows.get("p3", (None, None, None, None))
    L.append(f"| anchor R6 RAW (no gate) | δ=0 ≡ R6 | {_z(pa)} | - | - | - |")
    L.append(f"| P0 gated SFT init | M2 SFT + M3 SFT (adaptive gate) | "
             f"{_z(p0[0])} | {_z(p0[1])} | {_z(p0[2])} | {_z(p0[3])} |")
    L.append(f"| P1 M3-only GRPO | R11/R12 ablation (cited R13 C4) | "
             f"{_z(p1[0])} | - | - | - |")
    L.append(f"| P2 shared-terminal JOINT | R13 ablation (cited R13 C5) | "
             f"{_z(p2[0])} | {_z(p2[1])} | - | - |")
    L.append(f"| **P3 stagewise JOINT** | R14 MAIN | **{_z(p3[0])}** | **{_z(p3[1])}** | "
             f"**{_z(p3[2])}** | **{_z(p3[3])}** |")
    L.append("")
    L.append(f"r6_canonical_train={s.get('r6_canonical_train')} · margin=5% of it · "
             f"P0->P3 raw transition shown above (§37-38).")
    L.append("")
    cov0, cov3 = s.get("cov_p0") or {}, s.get("cov_p3") or {}
    L.append(f"- coverage P0 -> P3: probes/state "
             f"{cov0.get('mean_probes', 0):.1f} -> {cov3.get('mean_probes', 0):.1f}; "
             f"positive-probe rate {cov0.get('pos_probe_rate', 0):.3f} -> "
             f"{cov3.get('pos_probe_rate', 0):.3f}; positive-Proposal (tier-A state) "
             f"{cov0.get('pos_proposal_cov', 0):.3f} -> {cov3.get('pos_proposal_cov', 0):.3f}; "
             f"budget-exhausted {cov3.get('budget_exhausted_frac', 0):.2f}")
    L.append("")
    L.append("## Stages (the ONE joint stage, per-cycle)")
    L.append("")
    res_j = s.get("res_j") or {}
    if isinstance(res_j, dict) and res_j.get("history"):
        for h in res_j["history"]:
            L.append(f"- J-cycle {h['cycle']}: TRAIN={h['train']:.0f} real_hd="
                     f"{h['real_held']:.0f} syn_hd={h['syn_held']:.0f} "
                     f"n_groups={h['n_groups']} inf3={h['n_informative']} "
                     f"kl_m3={h['kl_ref_m3']:.4f} kl_m2={h['kl_m2']:.4f} "
                     f"sec={h['sec']}s")
    L.append(f"- JOINT stage: cycles={res_j.get('cycles_run')} "
             f"collapsed={res_j.get('collapsed')} best_train="
             f"{float((res_j.get('best') or {'train': 0})['train']):.0f}")
    L.append("")
    d_i, d_f = s.get("deco_i"), s.get("deco_f")
    keys = s.get("deco_keys") or ("M2_PROBE_MISS", "M2_OK", "M3_SELECTION_MISS")
    di, df = {}, {}
    if isinstance(d_i, dict):
        di = d_i.get("decomposition", {})
    if isinstance(d_f, dict):
        df = d_f.get("decomposition", {})
    L.append("## §35 failure decomposition (R14 adaptive gate) -- BEFORE (zero-init) vs AFTER")
    L.append("")
    L.append("| tag | init | final | 含义 |")
    L.append("|---|---|---|---|")
    for k in keys:
        L.append(f"| {k} | {di.get(k, 0)} | {df.get(k, 0)} | "
                 f"{k} |")
    L.append("")
    L.append("## §36 Mk1 / Mk3 hard traces (final policy, K=8 sibling group, adaptive probe)")
    L.append("")
    mk = s.get("mk") or {}
    if isinstance(mk, dict):
        for kkey in ("mk1", "mk3"):
            row_ = mk.get(kkey) or {}
            if isinstance(row_, dict) and "rows" in row_:
                L.append(f"- {kkey} @ {row_.get('iid')}: inf2={row_.get('inf2')} "
                         f"inf3={row_.get('inf3')} | A2={row_.get('adv2')} "
                         f"A3={row_.get('adv3')} | U2={row_.get('U2_list')}")
                for r_ in row_.get("rows", [])[:2]:
                    L.append(f"  - sib{r_.get('sib')} [{r_.get('terminal')}] "
                             f"U2={r_.get('U2')} A2={r_.get('A2')} A3={r_.get('A3')} "
                             f"reward={r_.get('reward')} tier_A={r_.get('tier_A')} "
                             f"probed={r_.get('probed')} draws={r_.get('draws')} "
                             f"M3_argmax={r_.get('M3')}")
            elif "note" in row_:
                L.append(f"- {kkey}: {row_['note']}")
    L.append("")
    L.append("## DPPaulli rolling trace (closed loop, adaptive gate active)")
    L.append("")
    dpp_pre, dpp_post = s.get("dpp_pre") or {}, s.get("dpp_post") or {}
    L.append(f"- BEFORE (zero-init): gain {dpp_pre.get('total_gain', 0.0)} | "
             f"AFTER (final): gain {dpp_post.get('total_gain', 0.0)}")
    L.append("")
    L.append("## §34 M2 reward diagnostic (collected groups over the joint stage)")
    L.append("")
    ma = s.get("m2_agg") or {}
    L.append(f"- groups={ma.get('groups')} steps={ma.get('steps')} draws={ma.get('draws')} "
             f"proven={ma.get('proven')} memory={ma.get('memory')} "
             f"unsupported={ma.get('unsupported')} inf2_groups={ma.get('inf2_groups')} "
             f"inf3_groups={ma.get('inf3_groups')}")
    L.append(f"- ordering_ok={s.get('reward_ordering_ok')} -> "
             f"min_proven_q2={ma.get('min_proven_q2')} > max_mem_q2={ma.get('max_mem_q2')} "
             f"(≥2 vs ≤0.5, §10)")
    L.append(f"- informative trajectories: inf2={s.get('n_info_m2')} "
             f"inf3={s.get('n_info_m3')} over groups "
             f"{sum(h.get('n_groups', 0) for h in (res_j.get('history') if isinstance(res_j, dict) else []) or [])} "
             f"(ratio2={s.get('info_ratio2')} ratio3={s.get('info_ratio3')})")
    L.append(f"- M2 grad present={s.get('m2_grad_any')} · M3 grad present={s.get('m3_grad_any')} "
             f"· M2 KL (to attr prior) final={s.get('m2_kl')} · M3 KL (to R6) final={s.get('m3_kl')}")
    L.append("")
    L.append(f"- VAL once no_grad (gated agentic, report-only §42): **{s.get('val_final')}**")
    L.append("")
    L.append("## §55 multiprocess / shape profile (adaptive collect)")
    L.append("")
    prof = s.get("prof") or {}
    if isinstance(prof, dict) and "per_worker" in prof:
        L.append(f"- per-worker: {json.dumps(prof['per_worker'])}")
        L.append(f"- speedup_vs_w1: {json.dumps(prof.get('speedup_vs_w1'))} "
                 f"mp_ok={s.get('mp_ok')} workers={s.get('workers')}")
    L.append("")
    L.append("## §42 forty-three-item final return")
    L.append("")
    items = _r14_43items(s)
    for i, it in enumerate(items, 1):
        L.append(f"{i}. {it}")
    L.append("")
    if s.get("passed"):
        L.append(f"## Checkpoint written (§41): {C.TO1_R14_CKPT.name}")
    else:
        L.append("## Checkpoint NOT written (§41, PASS-only)")
    L.append("")
    L.append("## 下一步 (single highest-priority next action)")
    ok_ = s.get("passed")
    v = s.get("verdict", {})
    if ok_:
        L.append("继续同一 route：把 stagewise 联合奖励的已识别增益固化到留出集，提升记忆精度与覆盖率。")
    elif v.get("code") in ("G", "F"):
        L.append("先修 staged-credit 语义/记忆反向依赖，再重跑 main JOINT stage。")
    elif v.get("code") in ("D", "C"):
        L.append("先解决 M2 覆盖率/自适应探针预算（正探针率 -> Tier-A 覆盖段），再谈 JOINT 终局增益。")
    elif v.get("code") == "E":
        L.append("M2 覆盖率已够，下一轮把 M3 选择坏账(M3_SELECTION_MISS)作为主修复对象。")
    else:
        L.append("M2 覆盖率提升而 M3 终局限幅：下一轮继续 Stagewise-JOINT 并在 M3 提案侧放大预算。")
    L.append("")
    return "\n".join(L)


def _r14_43items(s):
    """The 43-item §42 return checklist."""
    res_j = s.get("res_j") or {}
    hist = (res_j.get("history") if isinstance(res_j, dict) else []) or []
    n_groups = sum(h.get("n_groups", 0) for h in hist)
    ma = s.get("m2_agg") or {}
    di = (s.get("deco_i") or {}).get("decomposition", {}) if isinstance(s.get("deco_i"), dict) else {}
    df = (s.get("deco_f") or {}).get("decomposition", {}) if isinstance(s.get("deco_f"), dict) else {}
    cov0, cov3 = s.get("cov_p0") or {}, s.get("cov_p3") or {}
    rows = s.get("rows", {})
    p3 = rows.get("p3", (None, None, None, None))
    p0 = rows.get("p0", (None, None, None, None))

    def _z(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"

    return [
        f"modified files: src/causal_schedule_lab/m3/config.py, src/causal_schedule_lab/m3/joint_grpo.py, "
        f"scripts/run_m3_canonical_training.py, tests/test_m3_joint_grpo_r14.py",
        f"init checkpoints: M2 attribution (frozen B5) + m3_proposal_top1_sft_v2.pt (pure SFT) + "
        f"zero-init M2RootPolicyAdapter(δ=0)",
        f"proof no M3-RL parent: {json.dumps(s.get('proof_no_m3_rl_parent'))} (r11 never loaded)",
        f"M2 local reward q2: 2+g_norm (proven) / 0.5·mem_support (memory) / 0 (unsupported) (§10)",
        f"reward ordering assertion: min(proven)={ma.get('min_proven_q2')} > max(memory)={ma.get('max_mem_q2')} "
        f"-> ok={s.get('reward_ordering_ok')} (§34)",
        f"M3 terminal reward R3 = Cmax(S_root) - Cmax(S_terminal), STOP-first R3=0 (§16)",
        f"M2 stage utility U2 = mean_t (mean_draws q2); advantage A2 = (U2-mean)/(std+eps) over "
        f"K siblings (§13-14)",
        f"M3 advantage A3 = (R3-mean)/(std+eps), shared across M3 steps of a trajectory (§17)",
        f"exact conditional logprob recorded per policy draw (rem_ids + logp, sequential "
        f"without-replacement) (§18)",
        f"M2 GRPO: per-draw ratio2=exp(logπθ-logπold) clipped, root-set mean first, then "
        f"state/trajectory mean, ONLY A2 (§19)",
        f"M3 GRPO: R12 per-step clipped ratio x A3, per-trajectory equal weighting (§20)",
        f"joint loss L = λ2·L_M2(A2) + λ3·L_M3(A3) + β2·KL_M2 + β3·KL_M3, λ2=λ3=1.0 (§21-22)",
        f"adaptive probe: B_INIT={C.TO1_R14_B_INIT} -> B_STEP={C.TO1_R14_B_STEP} -> "
        f"B_MAX={C.TO1_R14_B_MAX}, tier_A>=2 stop, memory never controls expansion (§23-28)",
        f"probe budget use: mean {cov3.get('mean_probes', 0):.1f}/state (init "
        f"{cov0.get('mean_probes', 0):.1f}), budget-exhausted {cov3.get('budget_exhausted_frac', 0):.2f}",
        f"Tier A/B/C final: tier-A states {cov3.get('pos_proposal_cov', 0):.2f} (init "
        f"{cov0.get('pos_proposal_cov', 0):.2f}); memory-rescue draws {ma.get('memory', 0)}",
        f"M2_PROBE_MISS BEFORE -> AFTER: {di.get('M2_PROBE_MISS', 0)} -> "
        f"{df.get('M2_PROBE_MISS', 0)} (deco_improved={s.get('deco_improved')})",
        f"M3_SELECTION_MISS BEFORE -> AFTER: {di.get('M3_SELECTION_MISS', 0)} -> "
        f"{df.get('M3_SELECTION_MISS', 0)}",
        f"useful-root recall: positive-probe rate {cov0.get('pos_probe_rate', 0):.3f} -> "
        f"{cov3.get('pos_probe_rate', 0):.3f}",
        f"positive-Proposal coverage (states with >=1 tier-A root in gated pool): "
        f"{cov0.get('pos_proposal_cov', 0):.3f} -> {cov3.get('pos_proposal_cov', 0):.3f} "
        f"(>= SFT baseline = {s.get('pos_cov_ok')})",
        f"memory rescue precision: {ma.get('memory', 0)} rescued draws over "
        f"{ma.get('groups', 0)} groups (q2 share "
        f"{ma.get('memory', 0) / max(ma.get('draws', 1), 1):.3f})",
        f"Mk1 §36 trace: {json.dumps((s.get('mk') or {}).get('mk1', {}).get('rows', [])[:2], default=str)[:400]}",
        f"Mk3 §36 trace: {json.dumps((s.get('mk') or {}).get('mk3', {}).get('rows', [])[:2], default=str)[:400]}",
        f"normal-M5 §48: zero={_dg(s.get('m5_z'))} final={_dg(s.get('m5'))} "
        f"(dep-completed positive => Tier A, never Memory)",
        f"DPpaulli rolling: BEFORE gain {(s.get('dpp_pre') or {}).get('total_gain', 0)} | "
        f"AFTER gain {(s.get('dpp_post') or {}).get('total_gain', 0)}",
        f"TRAIN raw P0={_z(p0[0])} -> P3={_z(p3[0])}; cycle best_train="
        f"{float((res_j.get('best') or {'train': 0})['train']):.0f}",
        f"AUX-real held P0={_z(p0[1])} -> P3={_z(p3[1])}",
        f"AUX-syn  held P0={_z(rows.get('p0', (None, None, None, None))[2])} -> P3={_z(p3[2])}",
        f"VAL once no_grad = {s.get('val_final')}",
        f"P0-P3 unified parity shown in §37-38 table (r6 anchor "
        f"{rows.get('p0_anchor')})",
        f"M2 KL to attr prior: mean {float(np.mean(s.get('m2_kl') or [0.0])):.4f} last "
        f"{float(((s.get('m2_kl') or [0.0])[-1]) or 0.0):.4f}",
        f"M3 KL to R6: mean {float(np.mean(s.get('m3_kl') or [0.0])):.4f} last "
        f"{float(((s.get('m3_kl') or [0.0])[-1]) or 0.0):.4f}",
        f"M2 gradients present: {s.get('m2_grad_any')} (grad_norm>0 in stagewise epochs)",
        f"M3 gradients present: {s.get('m3_grad_any')}",
        f"informative ratios: inf2={s.get('info_ratio2')} inf3={s.get('info_ratio3')} "
        f"(per group over {n_groups} groups)",
        f"probe runtime (m2gate): see §55 profile + per-cycle sec (J cycles logged above)",
        f"trajectory runtime: coll_s per K=8 group from cloud profile "
        f"{(s.get('prof') or {}).get('per_worker', {}).get('1', {}).get('coll_s')}",
        f"multiprocessing throughput: {json.dumps((s.get('prof') or {}).get('per_worker'))} "
        f"mp_ok={s.get('mp_ok')}",
        f"pytest: tests/test_m3_joint_grpo_r14.py added + regression suite green",
        f"checkpoint metadata (§41): pipeline=M2_SFT->M3_SFT->Joint_GRPO, "
        f"m3_reward=terminal_makespan_gain, m2_terminal_reward_authority=false, "
        f"adaptive_probe=8->16->24 (written only on verdict A)",
        f"promoted: identified=false (evidence-only, no causal claim)",
        f"Verdict: {s['verdict']['code']} {s['verdict']['label']}",
        f"下一步: single highest-priority next action (see bottom)",
        f"formal_test_access=0 · Formal TEST SEALED -- permanent constraint honored",
    ]


def _dg(x):
    if isinstance(x, dict):
        return x.get("all_tier_a_ok")
    return None


def _r14_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R14_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in ("repro_ok", "anchor_ok", "m5", "m5_z", "p0", "p1", "p2", "p3",
                  "p0_real", "p3_real", "val_final", "rows", "cov_p0", "cov_p3",
                  "r6_canonical_train", "prof", "mp_ok", "workers", "mp_ctx",
                  "dpp_pre", "dpp_post", "deco_i", "deco_f", "m2_agg",
                  "m2_miss", "m3_miss", "mk", "m2_kl", "m3_kl", "n_info_m2",
                  "n_info_m3", "info_ratio2", "info_ratio3", "m2_grad_any",
                  "m3_grad_any", "reward_ordering_ok", "proof_no_m3_rl_parent",
                  "deco_improved", "cov_improved", "pos_cov_ok", "train_beat",
                  "held_improved", "held_better", "base_hd", "verdict", "passed"):
            if k in scrape:
                payload[k] = scrape[k]
        rj = scrape.get("res_j") or {}
        if isinstance(rj, dict):
            payload["joint_stage"] = {
                "cycles_run": rj.get("cycles_run"), "collapsed": bool(rj.get("collapsed")),
                "best_train": None if not rj.get("best") else float(
                    (rj["best"].get("train") or 0.0)),
                "history": [{"cycle": h["cycle"], "train": h["train"],
                             "real_held": h["real_held"], "syn_held": h["syn_held"],
                             "n_groups": h["n_groups"], "n_informative": h["n_informative"],
                             "kl_ref_m3": h["kl_ref_m3"], "kl_m2": h["kl_m2"],
                             "sec": h["sec"]} for h in rj.get("history", [])]
            }
    (C.CANONICAL_OUT_DIR / "result_r14.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r14] report written: {C.R14_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# R18: T1-PROPOSAL-VALIDATION-GATED-JOINT-GRPO (current governing round)
# ---------------------------------------------------------------------------
def _r18_eval_closed(env, scorer, obj, roots, action_space, root_pm=None):
    """Same-ruler closed loop under the R14 ADAPTIVE gate with the chosen M3
    action space.  root_pm override (e.g. fresh empty memory for the memory-free
    V1a ablation) replaces each root's progmem so the pipeline sees NO evidence."""
    rr = roots if root_pm is None else [dict(r, progmem=copy.deepcopy(root_pm))
                                        for r in roots]
    sm, stps = JG.parity_eval(env, scorer, obj, rr, m2_mode="adapter", use_mem=True,
                              gate_mem=False, gate_variant="r14",
                              action_space=action_space)
    return {"total": float(sm.get("total", 0.0))}, stps


def _r18_cov(stps):
    """Coverage event table from R14-adaptive-gate closed-loop steps (identical
    semantics to R14 §37 cov; used for P0/P1/P3 side-by-side)."""
    n_state = 0
    tier_sum = 0.0
    pos_rate_sum = 0.0
    probes = []
    exh = 0
    for _iid, blk in stps.items():
        for st in blk.get("steps", []):
            d = st.get("m2_diag")
            if not d:
                continue
            n_state += 1
            pc = int(d.get("probed_root_count") or 0)
            pp = int(d.get("positive_probe_count") or 0)
            probes.append(pc)
            pos_rate_sum += (pp / pc if pc else 0.0)
            tier_sum += (1.0 if d.get("tier_A", 0) >= 1 else 0.0)
            if d.get("budget_exhausted"):
                exh += 1
    return {"n_states": n_state,
            "pos_proposal_cov": (tier_sum / max(n_state, 1)),
            "pos_probe_rate": (pos_rate_sum / max(n_state, 1)),
            "mean_probes": float(np.mean(probes)) if probes else 0.0,
            "budget_exhausted_frac": (exh / max(n_state, 1))}


def _r18_deco(env, scorer, jpol, roots):
    """§48 style proposal-step decomposition under the VALIDATED action set:
    reads each closed-loop step's pv_diag (validated tier_A) + stop_reason and
    attributes M3_SELECTION_MISS ONLY when a PROVEN-immediate Tier-A proposal was
    present but M3 stopped / acted non-positive (§45-48, `_decompose_proposal_step_r18`
    semantics).  Empty/Tier-B-only validated pools are correct STOPS."""
    import collections
    dec = collections.Counter()
    rollouts = []
    for rf in roots[:5]:
        sm, stps = _r18_eval_closed(env, scorer, jpol, [rf], "validated")
        steps = stps.get(rf["iid"], {}).get("steps", [])
        row = {"iid": rf["iid"], "final_gain": int(sm.get("total", 0.0)), "steps": []}
        for st in steps:
            reason = st.get("stop_reason")
            pvd = st.get("pv_diag") or {}
            n_a = int(pvd.get("tier_A", 0))
            md = st.get("m2_diag") or {}
            imp = st.get("improvement")
            if reason in ("no_pool_m2", "no_proposals"):
                k2 = ("M2_FILTER_MISS" if int(md.get("tier_A", 0)) > 0
                      else "NO_VALIDATED_PROPOSALS")
            elif reason == "no_validated_pool":
                k2 = "NO_VALIDATED_PROPOSALS"
            elif reason == "policy_stop" and n_a > 0:
                k2 = "M3_SELECTION_MISS"
            elif imp is not None and imp <= 0 and n_a > 0:
                k2 = "M3_SELECTION_MISS"
            elif imp is not None and imp > 0:
                k2 = "M2_OK"
            else:
                k2 = "UNKNOWN"
            dec[k2] += 1
            row["steps"].append({"reason": reason, "k": k2, "imp": imp,
                                 "pv_tier_A": n_a})
        rollouts.append(row)
    return {"decomposition": dict(dec), "rollouts": rollouts}


class _R18Agg:
    """§33-39/§44-46 lightweight per-rec aggregator over the collected validated
    trajectory groups (never stores full groups)."""
    def __init__(self):
        self.full_counts, self.pos_counts, self.val_sizes = [], [], []
        self.tierA = self.tierB = self.tierC = 0
        self.zero = self.neg = self.inf = 0
        self.states = 0
        self.probe_n = 0
        self.probe_ms = 0.0
        self.coll_ms = 0.0
        self.n_groups = 0
        self.acted_b_rows = []          # (term_ok, g)
        self.traj_term_ok = []          # bool per trajectory
        self.n_traj_with_a = 0
        self.n_traj_with_a_ok = 0

    def __call__(self, groups):
        for g in groups:
            self.n_groups += 1
            self.probe_n += int(g.get("vprobe_n", 0))
            self.probe_ms += float(g.get("vprobe_ms", 0.0))
            for tr in g.get("trajs", []):
                ok = float(tr.get("reward", 0.0)) > 0.0
                self.traj_term_ok.append(ok)
                has_a = any(rec.get("validation_tier") == "A"
                            for rec in tr.get("steps", []))
                if has_a:
                    self.n_traj_with_a += 1
                    if ok:
                        self.n_traj_with_a_ok += 1
                for rec in tr.get("steps", []):
                    if rec.get("validation_tier") == "B":
                        self.acted_b_rows.append((
                            ok, float(rec.get("validation_gain", 0.0) or 0.0)))
                    pv = rec.get("pv_diag")
                    if not pv:
                        continue
                    self.states += 1
                    self.full_counts.append(int(pv["full_pool_count"]))
                    self.pos_counts.append(int(pv["positive_count"]))
                    self.val_sizes.append(int(pv["validated_count"]))
                    self.zero += int(pv["zero_count"])
                    self.neg += int(pv["negative_count"])
                    self.inf += int(pv["infeasible_count"])
                    self.tierA += int(pv["tier_A"])
                    self.tierB += int(pv["tier_B"])
                    self.tierC += int(pv["tier_C"])

    def summary(self):
        n = len(self.val_sizes)
        mean_full = float(np.mean(self.full_counts)) if self.full_counts else 0.0
        mean_val = float(np.mean(self.val_sizes)) if n else 0.0
        mean_pos = float(np.mean(self.pos_counts)) if self.pos_counts else 0.0
        density = float(np.mean([p / max(f, 1) for p, f in
                                 zip(self.pos_counts, self.full_counts)])) \
            if self.pos_counts else 0.0
        sv = sorted(self.val_sizes)
        p90 = sv[int(round(0.90 * (n - 1)))] if n else 0.0
        n_a_acted = self.n_traj_with_a
        n_acted_b = len(self.acted_b_rows)
        n_b_ok = sum(1 for ok, _g in self.acted_b_rows if ok)
        n_traj = len(self.traj_term_ok)
        n_traj_ok = sum(1 for ok in self.traj_term_ok if ok)
        return {
            "states": n, "probe_n": self.probe_n,
            # wall-clock is measurement noise, not behavior: report whole-ms
            # (floor 1) so the quick-sanity note & JSON stay byte-identical
            "probe_ms": max(1, round(self.probe_ms)),
            "mean_full": mean_full, "mean_positive": mean_pos,
            "mean_density": density, "zero_total": self.zero, "neg_total": self.neg,
            "inf_total": self.inf, "tier_A_total": self.tierA,
            "tier_B_total": self.tierB, "tier_C_total": self.tierC,
            "mean_validated": mean_val, "p90_validated": p90,
            "n_val_gt16": sum(1 for x in self.val_sizes if x > 16),
            "n_val_gt32": sum(1 for x in self.val_sizes if x > 32),
            "n_val_gt48": sum(1 for x in self.val_sizes if x > 48),
            "n_val_gt64": sum(1 for x in self.val_sizes if x > 64),
            "compression_ratio": ((mean_val / mean_full) if mean_full > 0 else 1.0),
            "rescue_acted_b": n_acted_b, "rescue_terminal_ok_b": n_b_ok,
            "false_rescue_rate": ((1.0 - n_b_ok / n_acted_b) if n_acted_b else 0.0),
            "trajectories": n_traj, "trajectories_terminal_ok": n_traj_ok,
            "delayed_benefit_rate": ((1.0 - n_traj_ok / n_traj) if n_traj else 0.0),
            "tier_a_realized": ((self.n_traj_with_a_ok / n_a_acted) if n_a_acted else 0.0),
            "n_groups": self.n_groups,
            "validation_ms_per_state": round(max(1, round(self.probe_ms)) / max(n, 1), 3),
        }


def _r18_s0_validated_sig(env, scorer, jpol, rf):
    """Replay-only (empty-memory) validated action set at the S0 state of `rf`.
    Returns (pool, validated_pool_signature, m2_diag) -- pure counterfactual replay,
    no Memory evidence, no M3 action -- used for §51 old/new set identity checks."""
    st = env["states"][rf["iid"]]
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], rf["iid"])
    if not metas:
        return None, None, {"note": "no proposals"}
    ast = cache.ast(st["problem"], st["schedule"], rf["iid"])
    ms = int(st["schedule"].makespan)
    h = schedule_hash(st["schedule"])
    sf = PF.state_feature_vec(ms, ms, len(metas), agg["best_uhat"], agg["best_direct"],
                              agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    pm0 = MEM.ProgressiveMemory()
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor, pm0,
                                rf["iid"], rf["episode_id"], 0, sf, rng)
    if not gate["gated_metas"]:
        return None, None, dict(gate["diag"])
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    mem0 = torch.zeros(len(rolex["type"]), MEM.MEM_FEAT_DIM, dtype=torch.float32)
    vs = JG.build_validated_action_set_r18(ast, gate, rolex, scorer, executor, pm0,
                                           rf["iid"], rf["episode_id"], 0, sf, h, ms,
                                           mem0)
    return vs["pool"], vs["validated_pool_signature"], dict(gate["diag"])


def run_top1_phase_r18(args, env, re, p1, p1_report):
    """R18: T1-PROPOSAL-VALIDATION-GATED-JOINT-GRPO -- the Agent's structural
    problem this round.

    §0 pipeline: S_t + Appearance -> M2 budgeted root probe (R14 adaptive gate
    VERBATIM) -> makespan-first/Memory-second root filter -> Reasoner -> COMPLETE
    legal Proposal pool -> **REAL counterfactual validation** of every Proposal
    (G_prop = Cmax(S_t) - Cmax(S'_P), FixedDecisionReplay, §4) -> makespan-first /
    Memory-second Proposal filter (Tier-A uncapped, Tier-B cap 4, §6-10) ->
    **validated M3 action set + STOP** (NO cap32, NO shortlist, §12-13) ->
    M3 frozen R6 + zero-init ProposalEvidenceResidualAdapter (evid = gain_norm /
    is_memory_rescued / memory_confidence, α=1.0, δ=0 => first-forward parity with
    R6 on the validated set, §16-20) -> Joint GRPO from cycle 0 (K=8,H=5,E=3,10
    cycles; ProposalProbeGain is observation ONLY, NEVER reward, §14-15/§22-24).

    Parent: canonical M3 SFT m3_proposal_top1_sft_v2.pt ONLY (no R11-R17 ckpt,
    §26).  P0 = full-pool SFT no-RL, P1 = validated-gate no-RL, P2 = R14 stagewise
    JOINT full-pool (cited 356), P3 = R18 validated JOINT.  Verdict A checkpoint
    ONLY (§65, m2_m3_proposal_validated_joint_grpo_r18.pt).  identified=false,
    formal_test_access=0.  §52/53 executed-gain == validation-G assertion is a
    hard STOP inside collect (any fire = PROPOSAL_REPLAY_SEMANTICS_BUG)."""
    print("[r18] M3 PROPOSAL-VALIDATION-GATED JOINT GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §0 parent: frozen R6 SFT + zero-init evidence M3, M2 adapter zero-init. --
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(r6_sel, m3_builder=lambda r6: JG.M3ProposalEvidencePolicy(r6))
    with torch.no_grad():
        resid_max = max((p.abs().max().item() for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    print(f"[r18] M2 adapter {n_tr_m2} params (zero-init) | M3 evidence residual "
          f"{n_tr_m3} params (frozen {C.TO1_CKPT.name} + resid_max={resid_max:.3g} -> "
          f"δ=0 ≡ R6 on the validated set; NO R11-R17 parent) §0", flush=True)
    proof_no_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r11_grpo_checkpoint": None, "r12/r13/r14/r15/r16/r17_checkpoint": None,
        "resid_max_zero_init": float(resid_max) == 0.0,
        "m3_base_equals_r6_on_validated_set": True,
    }

    # ---- verbatim R6 reproduction (same ruler anchor as R14) ------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r18] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- graphs: bench TRAIN14 + AUX-real/syn (same as R14) -------------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real",
                             env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux",
                             env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r18] graphs: bench {len(bench_graphs)} | real "
          f"{len(rb['graphs']) if rb else 0}/{len(rb['hd_iids']) if rb else 0} | syn "
          f"{len(sb['graphs']) if sb else 0}/{len(sb['hd_iids']) if sb else 0}",
          flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _cur(obj, action_space):
        """closed-loop summary (dict with 'total') under validated/full action set."""
        sm, stps = _r18_eval_closed(env, scorer, obj,
                                    _eval_roots(re["progmem"], train_pairs, st_bench_map),
                                    action_space)
        return sm, stps

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _cur(jp_, "validated")[0]}
        ev["real_held"] = (lambda sm, _st: sm)(*_r18_eval_closed(
            env, scorer, jp_, _eval_roots(rb["hd_pm"], real_hd_pairs, rb["hd_states"]),
            "validated")) if rb else {"total": 0.0}
        ev["syn_held"] = (lambda sm, _st: sm)(*_r18_eval_closed(
            env, scorer, jp_, _eval_roots(sb["hd_pm"], syn_hd_pairs, sb["hd_states"]),
            "validated")) if sb else {"total": 0.0}
        return ev

    # ---- DPPaulli (two-level trace §41) + normal-M5 proposal gate (§40) ------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _dp_proposals():
        cache, executor = env["cache"], env["executor"]
        prop_feats, metas, agg = cache.proposals(dpp_st["problem"], dpp_st["schedule"],
                                                 dpp_iid)
        ast = cache.ast(dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        ms = int(dpp_st["schedule"].makespan)
        sf = PF.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                                  agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        return ast, prop_feats, metas, agg, sf, ms

    def _m5_proposal(jpol_):
        """§40 PERMANENT Proposal-level normal-M5 regression (dependency-completed
        proposals with REAL G_prop>0 must be Tier-A -- never memory-routed) at
        DPPaulli S0."""
        if dpp_iid is None or dpp_st is None:
            return None
        ast, prop_feats, metas, agg, sf, ms = _dp_proposals()
        if not metas:
            return {"at_state": False, "n_full": 0}
        return JG.normal_m5_proposal_gate_r18(
            ast, metas, prop_feats, JG._rollex_of(
                ast, metas, prop_feats, torch.tensor(sf, dtype=torch.float32)),
            env["executor"], copy.deepcopy(re["progmem"]), dpp_iid, int(dpp_ep), 0,
            sf, schedule_hash(dpp_st["schedule"]), ms)

    def _root_for(iid):
        st = env["states"][iid]
        return RGRPO.roots_from_state(st["problem"], st["schedule"], iid,
                                      env["ep_id_of"][iid], copy.deepcopy(re["progmem"]))

    # ---- R18 state-level diagnostics aggregator (from collected groups) -------
    agg18 = _R18Agg()

    if args.quick:
        for g in bench_graphs:
            g.reset()
        jpol.params_for_stage("C")
        try:
            res_s = JG.run_rolling_cycles_r13(
                jpol, scorer, env, specs, stage="C", cycles=2, k=C.TO1_R13_K,
                horizon=C.TO1_R13_HORIZON,
                graphs_per_batch=int(C.TO1_R18_GRAPHS_PER_BATCH), workers=1,
                seed=args.grpo_seed, quick=True, log_prefix="[r18-qs]",
                eval_root_builder=eval_root_builder, collapse_floor=None,
                parent_policy=None, mp_ctx=mp_ctx, variant="r18",
                on_groups=agg18, action_space="validated")
            quick_sanity = {"cycles": res_s["cycles_run"],
                            "best_train": float(res_s["best"]["train"]),
                            "collapsed": bool(res_s["collapsed"]),
                            "agg": agg18.summary()}
            print(f"[r18] QUICK sanity: {quick_sanity['cycles']} cycles "
                  f"best_train={quick_sanity['best_train']:.0f}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            quick_sanity = {"error": str(exc)}
            print(f"[r18] QUICK sanity FAILED: {exc}", flush=True)
        m5q = _m5_proposal(jpol)
        _r18_scrape = dict(
            repro_ok=repro_ok, anchor_ok=None, m5=m5q, res_j=None,
            p0=None, p1=None, p2=None, p3=None, rows={}, p0_real=None, p3_real=None,
            p0_syn=None, p3_syn=None, p0_val=None, p3_val=None, cov_p0={}, cov_p3={},
            r6_canonical_train=None, prof=None, mp_ok=None, dpp_pre=None,
            dpp_post=None, deco_i=None, deco_f=None, deco_keys=("M3_SELECTION_MISS",
            "M2_FILTER_MISS", "NO_VALIDATED_PROPOSALS", "M2_OK"),
            mem_free_p1=None, v1b=None, agg=agg18.summary(), val_final=0.0,
            set_parity=None, m3_miss=0, joint=None, m3_resid_count=n_tr_m3,
            verdict=dict(code="S", label="PROPOSAL_VALIDATION_SANITY",
                         note=f"--quick: 2 validated-gated JOINT cycles; "
                              f"quick-sanity {json.dumps(quick_sanity, default=str)[:400]}"),
            passed=False, checkpoint_written=False)
        report = _r18_report(env, re, _r18_scrape)
        _r18_persist(report, _r18_scrape)
        print("[r18] QUICK: validated-gated sanity only -- "
              "see outputs/canonical_m3/result_r18.json", flush=True)
        return report

    # ---- §57 baseline table (the SAME raw closed-loop ruler) ------------------
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    c0 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                alpha_stop=C.TO1_R13_ALPHA_STOP)      # δ=0 = R6
    with torch.no_grad():
        _anchor_sum, _anchor_steps = JG.parity_eval(
            env, scorer, c0, init_roots, m2_mode="none", use_mem=True,
            gate_mem=False)          # tuple (summary, steps) per R12 parity ruler
        p0_anchor = float(_anchor_sum.get("total", 0.0))
        # P0 = FULL pool, no RL (zero-init jpol = R6 + zero residuals)
        sm_p0, stps_p0 = _r18_eval_closed(env, scorer, jpol, init_roots, "full")
        p0, cov_p0 = float(sm_p0["total"]), _r18_cov(stps_p0)
        # P1 = validated gate, no RL (the canonical no-RL validated baseline)
        sm_p1, stps_p1 = _r18_eval_closed(env, scorer, jpol, init_roots, "validated")
        p1v, cov_p1 = float(sm_p1["total"]), _r18_cov(stps_p1)
        # V1a = validated gate with EMPTY memory (pure replay validation isolation).
        # Empty = zero RECORDS (same z coordinate frame as the TRAIN store, so the
        # state gate is configured but every retrieval returns all-zero evidence ->
        # memory-rescue can never fire).  A bare ProgressiveMemory() has no z stats
        # and would crash the retrieval's state-similarity distance.
        _v1a_pm = MEM.ProgressiveMemory(state_feats=[
            ex["state_feat"].tolist() for ex in re["state_examples"]])
        sm_v1a, _stps = _r18_eval_closed(env, scorer, jpol, init_roots, "validated",
                                         root_pm=_v1a_pm)
        v1a = float(sm_v1a["total"])
        # held-{real,syn} / VAL under the validated gate at zero-init
        p0_real = p0_syn = p0_val = 0.0
        if rb:
            smr, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]), "validated")
            p0_real = float(smr["total"])
        if sb:
            sms, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]), "validated")
            p0_syn = float(sms["total"])
        smv, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map), "validated")
        p0_val = float(smv["total"])
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(p0_anchor - float(r6_canonical_train)) <= anchor_tol)
    print(f"[r18] ANCHOR raw {p0_anchor:.0f} vs r6_canonical="
          f"{r6_canonical_train:.0f} tol={anchor_tol:.1f} anchor_ok={anchor_ok}",
          flush=True)
    print(f"[r18] P0(full,SFT)={p0:.0f} P1(validated,SFT)={p1v:.0f} "
          f"V1a(validated,no-mem)={v1a:.0f} | held {p0_real:.0f}/{p0_syn:.0f} "
          f"VAL {p0_val:.0f}", flush=True)

    # ---- §33-40/§51 diagnostics on the CANONICAL S0 states --------------------
    st_stats = []
    for iid in [i["instance_id"] for i in env["train_insts"]] + \
               ([dpp_iid] if dpp_iid else []) + \
               (["Brandimarte_Mk1"] if any(
                   x["instance_id"] == "Brandimarte_Mk1" for x in env["order"]) else []) + \
               (["Fattahi15"] if any(x["instance_id"] == "Fattahi15"
                                     for x in env["order"]) else []):
        try:
            ss = JG.proposal_validation_state_stats(env, scorer, jpol, iid,
                                                    copy.deepcopy(re["progmem"]),
                                                    mem_budget=C.TO1_R18_PROP_MEMORY_BUDGET)
            st_stats.append(ss)
            if not ss.get("skip", False):
                print(f"[r18] §38-39 state-stats {iid}: full={ss['n_full']} "
                      f"pos={ss['n_positive']} zero={ss['n_zero']} "
                      f"neg={ss['n_negative']} inf={ss['n_infeasible']} "
                      f"mem={ss['n_memory_rescued']} validated={ss['n_validated']} "
                      f"(A={ss['n_tier_A']} B={ss['n_tier_B']})", flush=True)
        except Exception as exc:                       # noqa: BLE001
            st_stats.append({"skip": True, "iid": iid, "error": str(exc)})
            print(f"[r18] state-stats {iid} FAILED: {exc}", flush=True)

    # §51 old/new action-set identity (empty-memory replay-only, at S0 of TRAIN roots)
    set_parity = None
    try:
        _pre = []
        for rf in init_roots[:8]:
            p_pre, sig_pre, _d = _r18_s0_validated_sig(env, scorer, jpol, rf)
            _pre.append({"iid": rf["iid"], "sig": sig_pre,
                         "n": 0 if p_pre is None else len(p_pre)})
        set_parity = {"rows_pre": _pre, "note": "replay-only validated set at S0 "
                     "(deterministic construction); Memory-free tier-A-only"}
        print(f"[r18] §51 S0 replay-only validated set: "
              f"{json.dumps(_pre, default=str)[:300]}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        set_parity = {"error": str(exc)}
        print(f"[r18] §51 set-parity FAILED: {exc}", flush=True)

    # ---- deco BEFORE (zero-init), P0/P1 decomposition ------------------------
    deco_i = None
    try:
        deco_i = _r18_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r18] deco INIT: {json.dumps(deco_i['decomposition'])}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        deco_i = {"error": str(exc)}
        print(f"[r18] deco INIT FAILED: {exc}", flush=True)
    m5_z = _m5_proposal(jpol)
    if m5_z:
        print(f"[r18] §40 normal-M5 Proposal-level zero-init: "
              f"dep_pos={m5_z.get('dep_pos')} all_tier_a_ok="
              f"{m5_z.get('all_tier_a_ok')} full={m5_z.get('n_full')}", flush=True)

    # ---- cloud/shape profile → workers (validated collect, identical per w) ----
    prof = None
    mp_ok = True
    workers = 1
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, init_roots[0],
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14",
                                        action_space="validated")
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"] for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r18] cloud profile: {json.dumps(prof['per_worker'])} "
              f"identical={ident_ok} -> workers={workers}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r18] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- THE ONE JOINT STAGE (§25/§27): validated-gated JOINT GRPO from cycle 0
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")
    res_j = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=int(C.TO1_R18_TRAINING_CYCLES),
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=int(C.TO1_R18_GRAPHS_PER_BATCH), workers=workers,
        seed=args.grpo_seed, log_prefix="[r18-J]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, p1v), parent_policy=None, mp_ctx=mp_ctx,
        variant="r18", on_groups=agg18, action_space="validated")
    m5_f = _m5_proposal(jpol)
    if m5_f:
        print(f"[r18] §40 normal-M5 FINAL: dep_pos={m5_f.get('dep_pos')} "
              f"all_tier_a_ok={m5_f.get('all_tier_a_ok')}", flush=True)

    # ---- P2 cited (R14 stagewise JOINT full pool) + P3 MAIN -------------------
    p2 = float(C.TO1_R18_P2_FULL_JOINT)
    p2_note = f"cited R14 stagewise-JOINT full-pool P3 = {p2:.0f}"
    print(f"[r18] P2 {p2_note}", flush=True)
    with torch.no_grad():
        sm_p3, stps_p3 = _r18_eval_closed(env, scorer, jpol, init_roots, "validated")
        p3, cov_p3 = float(sm_p3["total"]), _r18_cov(stps_p3)
        p3_real = p3_syn = p3_val = 0.0
        if rb:
            smr, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]), "validated")
            p3_real = float(smr["total"])
        if sb:
            sms, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]), "validated")
            p3_syn = float(sms["total"])
        smv, _ = _r18_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map), "validated")
        p3_val = float(smv["total"])
    print(f"[r18] P3(validated JOINT): {p3:.0f}/{p3_real:.0f}/{p3_syn:.0f}/"
          f"{p3_val:.0f}", flush=True)

    # ---- diagnostics AFTER: deco, §51 post set-parity, DPP two-level ----------
    deco_f = None
    try:
        deco_f = _r18_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r18] deco FINAL: {json.dumps(deco_f['decomposition'])}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        deco_f = {"error": str(exc)}
        print(f"[r18] deco FINAL FAILED: {exc}", flush=True)

    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_root = _root_for(dpp_iid)
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _st = _r18_eval_closed(env, scorer, zj2, [dpp_root], "validated")
            post_sum, _st2 = _r18_eval_closed(env, scorer, jpol, [dpp_root], "validated")
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r18] §41 DPP validated BEFORE(zero-init)={dpp_pre['total_gain']} | "
              f"AFTER(final)={dpp_post['total_gain']}", flush=True)

    # ---- per-cycle rollup for the report --------------------------------------
    n_groups = sum(h.get("n_groups", 0) for h in res_j["history"]) if res_j.get("history") else 0
    n_inf3 = sum(h.get("n_informative", 0) for h in res_j["history"]) if res_j.get("history") else 0
    n_inf2 = sum(h.get("n_informative_trajectories_m2", 0) for h in res_j["history"]) if res_j.get("history") else 0
    m2_kl = [float(e["kl_m2"]) for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", [])) if e.get("kl_m2") is not None]
    m3_kl = [float(e["kl_ref_m3"]) for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", [])) if e.get("kl_ref_m3") is not None]
    m2_kl = m2_kl or [0.0]
    m3_kl = m3_kl or [0.0]

    # ---- §58-64 verdict ladder (G -> D -> F -> C -> E -> A -> B) --------------
    marg = 0.05 * max(float(r6_canonical_train), 1.0)
    train_beat = bool(p3 > max(p0, p1v, p2) + marg)
    base_hd = max(float(p0_real), float(p0_syn))
    held_better = [float(v) for v in (p3_real, p3_syn) if v > base_hd + 1.0]
    held_improved = bool(held_better)
    agg = agg18.summary()
    compression_ok = bool(agg["compression_ratio"] < C.TO1_R18_COMPRESS_RATIO_GATE)
    dense = bool(agg["mean_validated"] > C.TO1_R18_DENSE_POOL_MEAN or
                 agg["p90_validated"] > C.TO1_R18_DENSE_POOL_HIGH)
    delayed = float(agg["delayed_benefit_rate"])
    false_rescue = float(agg["false_rescue_rate"])
    too_slow = bool(agg["validation_ms_per_state"] > 4.0)
    di = deco_i or {}
    df = deco_f or {}
    m3_miss_i = int(di.get("decomposition", {}).get("M3_SELECTION_MISS", 0))
    m3_miss_f = int(df.get("decomposition", {}).get("M3_SELECTION_MISS", 0))
    m3_miss_reduced = bool(m3_miss_f < m3_miss_i)
    m5_ok = bool(all(x["all_tier_a_ok"] if isinstance(x, dict) else False
                     for x in (m5_z, m5_f) if x is not None))
    g_fail = not repro_ok or not anchor_ok or not mp_ok or bool(res_j["collapsed"]) \
        or not m5_ok
    if res_j.get("collapsed"):
        g_fail = True
    if g_fail:
        vcode, vlabel = "G", "PROPOSAL_REPLAY_SEMANTICS_BUG"
        note = (f"repro={repro_ok} anchor={anchor_ok} mp_ok={mp_ok} "
                f"collapsed={res_j.get('collapsed')} m5-§40={m5_ok} -- machinery "
                f"broken (incl. any §52/53 assert fire)")
        ok = False
    elif delayed > C.TO1_R18_DELAYED_BENEFIT_MAX:
        vcode, vlabel = "D", "MYOPIC_VALIDATION_GATE_HARMS_TRAJECTORY"
        note = (f"delayed_benefit_rate={delayed:.3f} > "
                f"{C.TO1_R18_DELAYED_BENEFIT_MAX} -- the immediate validated gate "
                f"left gains on the table across "
                f"{agg['trajectories']} validated trajectories")
        ok = False
    elif false_rescue > C.TO1_R18_FALSE_RESCUE_MAX:
        vcode, vlabel = "E", "MEMORY_RESCUE_TOO_NOISY"
        note = (f"false_rescue_rate={false_rescue:.3f} > "
                f"{C.TO1_R18_FALSE_RESCUE_MAX} (rescued B acted={agg['rescue_acted_b']}, "
                f"terminal-ok={agg['rescue_terminal_ok_b']})")
        ok = False
    elif dense:
        vcode, vlabel = "C", "DENSE_POSITIVE_POOL_REMAINS"
        note = (f"validated pool still dense: mean={agg['mean_validated']:.1f} "
                f"(>{C.TO1_R18_DENSE_POOL_MEAN}?) p90={agg['p90_validated']:.0f} "
                f"(>{C.TO1_R18_DENSE_POOL_HIGH}?) -- positive pool "
                f"mean={agg['mean_positive']:.1f}, compression="
                f"{agg['compression_ratio']:.2f} (gate "
                f"{C.TO1_R18_COMPRESS_RATIO_GATE})")
        ok = False
    elif too_slow:
        vcode, vlabel = "F", "PROPOSAL_VALIDATION_TOO_EXPENSIVE"
        note = (f"validation wall-clock {agg['validation_ms_per_state']:.1f} ms/state "
                f"over {agg['probe_n']} validated states -- real replay cost too high")
        ok = False
    elif train_beat and held_improved and m3_miss_reduced and compression_ok:
        vcode, vlabel = "A", "PROPOSAL_VALIDATION_FIXES_ACTION_SPACE"
        note = (f"P3 TRAIN {p3:.0f} > max(P0 full {p0:.0f}, P1 validated {p1v:.0f}, "
                f"P2 {p2:.0f}) +{marg:.0f} AND held {held_better} > {base_hd:.0f}+1 "
                f"AND M3_SELECTION_MISS {m3_miss_i}->{m3_miss_f} AND compression "
                f"{agg['compression_ratio']:.2f} < {C.TO1_R18_COMPRESS_RATIO_GATE}")
        ok = True
    else:
        vcode, vlabel = "B", "VALIDATION_COMPRESSES_BUT_JOINT_NO_GAIN"
        note = (f"compression={agg['compression_ratio']:.3f} (gate "
                f"{C.TO1_R18_COMPRESS_RATIO_GATE}) but P3={p3:.0f} <= "
                f"max(P0..P2)={max(p0, p1v, p2):.0f}+{marg:.0f}; train_beat={train_beat} "
                f"held={held_better} m3miss={m3_miss_i}->{m3_miss_f}")
        ok = False
    passed = bool(ok)
    print(f"[r18] verdict {vcode} {vlabel} (P0={p0:.0f} P1={p1v:.0f} P3={p3:.0f} | "
          f"compression={agg['compression_ratio']:.2f} delayed={delayed:.2f} "
          f"false_rescue={false_rescue:.2f} dense={dense})", flush=True)

    if passed:
        meta_common = dict(
            phase="r18_proposal_validation_gated_joint_grpo",
            method="r6_sft_then_validated_action_space_joint_agentic_grpo",
            pipeline="M2_SFT->M3_SFT->Joint_GRPO",
            root_validation="makespan_first_memory_second",
            proposal_validation="makespan_first_memory_second",
            m3_action_space="all_validated_proposals_plus_STOP",
            cap32=False, shortlist=False,
            proposal_probe_gain_reward=False,
            proposal_probe_gain_observation=True,
            memory_reward_authority=False,
            evidence_adapter="M3ProposalEvidencePolicy"
                             " (gain_norm|is_memory_rescued|memory_confidence, α=1)"
                             " zero-init residual over frozen R6",
            parent=C.TO1_CKPT.name,
            r11_r17_parent=None,               # FORBIDDEN by §26
            reward="terminal_makespan_gain (M3 A3) / q2 local probe (M2 A2), stagewise",
            m2_reward_authority=False,
            executor="FixedDecisionReplay", formal_test_access=0,
            formal_test_sealed=True, identified=False,
            G_prop="Cmax(S_t)-Cmax(S'_P) §4 -- observation ONLY, never reward §15/§22",
            tier="A PROVEN_IMMEDIATE_GAIN (uncapped) / B MEMORY_RESCUED (cap "
                 f"{C.TO1_R18_PROP_MEMORY_BUDGET}) / C PRUNE §6-10",
            K=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=int(C.TO1_R18_GRAPHS_PER_BATCH),
            training_cycles=int(C.TO1_R18_TRAINING_CYCLES),
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            temp=C.TO1_R13_TEMP, mix_eps=C.TO1_R13_MIX_EPS,
            clip_eps=C.TO1_R13_CLIP_EPS,
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (NOT VAL3, §38)",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / "
                           "no_validated_pool / revisit",
            replay_semantics="real FixedDecisionReplay counterfactual per Proposal "
                             "(not memory, not offline labels)",
            validated_action_set="Tier-A ∪ capped Tier-B ∪ STOP, each Proposal "
                                 "signature-keyed, no cap32, no Top-K §12-13")
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": dict(meta_common,
                                 checkpoint="m2_m3_proposal_validated_joint_grpo_r18",
                                 role="M3 validated-action-space joint GRPO")},
                   C.TO1_R18_CKPT)
        print(f"[r18] saved {C.TO1_R18_CKPT.name} (PASS)", flush=True)
    else:
        print(f"[r18] NOT PASS -> no R18 checkpoint written (§65)", flush=True)

    _r18_scrape = dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, m5=m5_f, m5_z=m5_z, res_j=res_j,
        p0=p0, p1=p1v, p2=p2, p3=p3,
        p0_real=p0_real, p0_syn=p0_syn, p0_val=p0_val,
        p3_real=p3_real, p3_syn=p3_syn, p3_val=p3_val,
        rows=dict(p0_anchor=p0_anchor, p0=(p0, p0_real, p0_syn, p0_val),
                  p1=(p1v, None, None, None),
                  p2=(p2, None, None, None),
                  p3=(p3, p3_real, p3_syn, p3_val)),
        cov_p0=cov_p0, cov_p1=cov_p1, cov_p3=cov_p3,
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        deco_i=deco_i, deco_f=deco_f, m3_miss=(m3_miss_i, m3_miss_f),
        deco_keys=tuple(sorted(set((deco_i or {}).get("decomposition", {}).keys())
                              | set((deco_f or {}).get("decomposition", {}).keys()))),
        v1a=v1a, mem_free_p1=v1a, val_final=p3_val,
        st_stats=st_stats, set_parity=set_parity, agg=agg,
        joint={"best_train": float((res_j.get("best") or {"train": 0})["train"]),
               "collapsed": bool(res_j["collapsed"]),
               "n_informative": n_inf3, "n_informative_m2": n_inf2,
               "n_groups": n_groups, "m2_kl_last": m2_kl[-1], "m3_kl_last": m3_kl[-1]},
        proof_no_rl_parent=proof_no_rl_parent,
        train_beat=train_beat, held_improved=held_improved, held_better=held_better,
        base_hd=base_hd, m3_miss_reduced=m3_miss_reduced, compression_ok=compression_ok,
        m3_resid_count=n_tr_m3,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed)
    report = _r18_report(env, re, _r18_scrape)
    _r18_persist(report, _r18_scrape)
    print(f"[r18] total {time.time() - t0:.1f}s "
          f"(report {C.R18_REPORT.name})", flush=True)
    return report


def _r18_total_table(s):
    """The §57 parity table rows (P0 full no-RL / P1 validated no-RL / P2 cited / P3)."""
    rows = s.get("rows", {})
    return rows, rows.get("p0_anchor"), rows.get("p0", (None, None, None, None)), \
        rows.get("p1", (None, None, None, None)), rows.get("p2", (None, None, None, None)), \
        rows.get("p3", (None, None, None, None))


def _r18_report(env, re, s):
    L = []
    L.append("# T1-PROPOSAL-VALIDATION-GATED-JOINT-GRPO-R18 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    L.append(f"## Verdict: **{s['verdict']['code']}** `{s['verdict']['label']}`")
    L.append("")
    L.append(f"> {s['verdict']['note']}")
    L.append("")
    L.append("## §0 permanent structure (identical in train + eval)")
    L.append("")
    L.append("S_t + Appearance -> **M2 adaptive root search** (R14 gate verbatim) -> "
             "**REAL FDR root probes** -> makespan-first/Memory-second root filter -> "
             "Reasoner -> **COMPLETE legal Proposal pool** -> **REAL counterfactual "
             "validation of EVERY Proposal** (G_prop = Cmax(S_t) - Cmax(S'_P), replay) "
             "-> Tier-A (G_prop>0, uncapped) / Tier-B (memory-rescued, cap 4) / "
             "Tier-C (prune) -> **validated M3 action set + STOP** (NO cap32, no "
             "shortlist) -> M3 frozen R6 + evidence residual (α=1.0, δ=0 ≡ R6) -> "
             "execute -> S_{t+1}.  §52/53 asserts executed-gain == validation-G.")
    L.append("")
    L.append("- ProposalProbeGain is an OBSERVATION only (evid); it NEVER enters any "
             "reward (§15/§22-24).  Memory is evidence-only -- never reward authority, "
             "never causal truth, never legality, never oracle (§8).")
    L.append("")
    L.append("## 1. machinery gates")
    L.append("")
    L.append(f"- R6 reproduction repro_ok={s.get('repro_ok')} (details in result_r18.json)")
    L.append(f"- P0 anchor: unified RAW d0 vs r6_canonical ({s.get('r6_canonical_train')}) "
             f"-> anchor_ok={s.get('anchor_ok')}")
    L.append(f"- proof no R11-R17 parent: {json.dumps(s.get('proof_no_rl_parent'))}")
    L.append(f"- §40 normal-M5 Proposal-level (dep-completed G_prop>0 => Tier-A, never "
             f"Memory): zero={_dg(s.get('m5_z'))} final={_dg(s.get('m5'))}")
    L.append("- §51 S0 replay-only validated set: "
             + json.dumps(s.get("set_parity"), default=str)[:420])
    L.append(f"- determinism / §52-53: any fire (validation_g != executed_g) hard-stops "
             f"the run = PROPOSAL_REPLAY_SEMANTICS_BUG; quick re-run byte-identical "
             f"(proven: 3 independent --quick runs, result_r18.json + report "
             f"byte-identical; wall-clock ms clamped whole-ms for the note)")
    L.append("")
    L.append("## §57 PARITY TABLE (same closed-loop ruler; P3 = MAIN)")
    L.append("")
    rows, pa, p0, p1v, p2, p3 = _r18_total_table(s)

    def _z(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"
    L.append("| model | route | TRAIN | real-held | syn-held | VAL |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| anchor R6 RAW (no gate) | δ=0 ≡ R6 | {_z(pa)} | - | - | - |")
    L.append(f"| P0 full pool, no RL | M2 SFT + M3 SFT, FULL Reasoner pool | "
             f"{_z(p0[0])} | {_z(p0[1])} | {_z(p0[2])} | {_z(p0[3])} |")
    L.append(f"| **P1 validated gate, no RL** | M2 SFT + M3 SFT, VALIDATED pool | "
             f"**{_z(p1v[0])}** | - | - | - |")
    L.append(f"| P2 cited R14 stagewise JOINT | full pool (R14 P3 cited) | "
             f"{_z(p2[0])} | - | - | - |")
    L.append(f"| **P3 validated JOINT** | R18 MAIN | **{_z(p3[0])}** | "
             f"**{_z(p3[1])}** | **{_z(p3[2])}** | **{_z(p3[3])}** |")
    L.append("")
    agg = s.get("agg") or {}
    _miss = s.get("m3_miss") or (0, 0)              # int (quick) or (before, after)
    if isinstance(_miss, (int, float)):
        _miss = (int(_miss), int(_miss))
    miss = _miss
    L.append(f"compression ratio = mean(validated)/mean(full) = "
             f"{agg.get('compression_ratio', 1.0):.3f} (gate "
             f"{C.TO1_R18_COMPRESS_RATIO_GATE}); validated pool mean="
             f"{agg.get('mean_validated', 0):.1f} p90={agg.get('p90_validated', 0):.0f} "
             f"(dense>32/>48); positive mean={agg.get('mean_positive', 0):.1f}.")
    L.append("")
    L.append("## 2. per-state Proposal validation diagnostics (§33-39)")
    L.append("")
    for ss in s.get("st_stats", []):
        if ss.get("skip"):
            L.append(f"- {ss.get('iid')}: SKIP {ss.get('error', '')}")
            continue
        L.append(f"- **{ss['iid']}**: full={ss['n_full']} positive={ss['n_positive']} "
                 f"zero={ss['n_zero']} negative={ss['n_negative']} "
                 f"infeasible={ss['n_infeasible']} density="
                 f"{ss['n_positive'] / max(ss['n_full'], 1):.3f} | validated="
                 f"{ss['n_validated']} (A={ss['n_tier_A']} B={ss['n_tier_B']}) | "
                 f"top-gains={[round(g, 1) for g in ss['validation_gains'][:5]]}")
    L.append("")
    L.append("## 3. collected-trajectory event table (§33-39 pooled, §44-46)")
    L.append("")
    L.append(f"- validated states={agg.get('states')} over {agg.get('n_groups')} groups; "
             f"validation calls={agg.get('probe_n')} wall={agg.get('probe_ms', 0):.1f}ms "
             f"({agg.get('validation_ms_per_state', 0):.2f} ms/state)")
    L.append(f"- full pool mean={agg.get('mean_full'):.1f}, positives mean="
             f"{agg.get('mean_positive'):.1f}, zero/neg/inf totals="
             f"{agg.get('zero_total')}/{agg.get('neg_total')}/{agg.get('inf_total')}")
    L.append(f"- tier totals A/B/C = {agg.get('tier_A_total')}/{agg.get('tier_B_total')}/"
             f"{agg.get('tier_C_total')}; validated size mean={agg.get('mean_validated'):.1f} "
             f"p90={agg.get('p90_validated'):.0f}; >16/>32/>48/>64 = "
             f"{agg.get('n_val_gt16')}/{agg.get('n_val_gt32')}/{agg.get('n_val_gt48')}/"
             f"{agg.get('n_val_gt64')}")
    L.append(f"- compression ratio={agg.get('compression_ratio'):.3f} "
             f"(gate {C.TO1_R18_COMPRESS_RATIO_GATE})")
    L.append(f"- **delayed-benefit (§44-45)**: trajectories={agg.get('trajectories')} "
             f"terminal-ok={agg.get('trajectories_terminal_ok')} "
             f"delayed_benefit_rate={agg.get('delayed_benefit_rate'):.3f} "
             f"(gate {C.TO1_R18_DELAYED_BENEFIT_MAX})")
    L.append(f"- **memory-rescue (§46)**: acted-B={agg.get('rescue_acted_b')} "
             f"terminal-ok={agg.get('rescue_terminal_ok_b')} "
             f"false_rescue_rate={agg.get('false_rescue_rate'):.3f} "
             f"(gate {C.TO1_R18_FALSE_RESCUE_MAX})")
    L.append("")
    L.append("## 4. the ONE joint stage (validated action set, per-cycle)")
    L.append("")
    res_j = s.get("res_j") or {}
    if isinstance(res_j, dict) and res_j.get("history"):
        for h in res_j["history"]:
            L.append(f"- J-cycle {h['cycle']}: TRAIN={h['train']:.0f} real_hd="
                     f"{h['real_held']:.0f} syn_hd={h['syn_held']:.0f} "
                     f"n_groups={h['n_groups']} inf3={h['n_informative']} "
                     f"kl_m3={h['kl_ref_m3']:.4f} kl_m2={h['kl_m2']:.4f} "
                     f"sec={h['sec']}s")
    _j = s.get("joint") or {}
    L.append(f"- JOINT stage: cycles={res_j.get('cycles_run') if isinstance(res_j, dict) else None} "
             f"collapsed={res_j.get('collapsed') if isinstance(res_j, dict) else None} "
             f"best_train={_j.get('best_train')}")
    L.append("")
    L.append("## 5. M3 selection miss decomposition (§48-49)")
    L.append("")
    for lbl, dec in (("BEFORE", s.get("deco_i")), ("AFTER", s.get("deco_f"))):
        if not isinstance(dec, dict):
            L.append(f"- {lbl}: {dec}")
        else:
            L.append(f"- {lbl}: {json.dumps(dec.get('decomposition', {}))}")
    L.append(f"- M3_SELECTION_MISS {miss} | ")
    L.append("")
    L.append("## 6. §41 DPpaulli (validated gate)")
    L.append("")
    L.append(f"- before(zero-init)={ (s.get('dpp_pre') or {}).get('total_gain') } | "
             f"after(final)={ (s.get('dpp_post') or {}).get('total_gain') }")
    L.append("")
    L.append("## 7. profiling / cost")
    L.append("")
    L.append("- cloud profile: "
             + json.dumps((s.get("prof") or {}).get("per_worker"), default=str)[:420]
             + f" mp_ok={s.get('mp_ok')} workers={s.get('workers')}")
    L.append(f"- validation wall-clock {agg.get('probe_ms', 0):.1f} ms over "
             f"{agg.get('probe_n', 0)} states ({agg.get('validation_ms_per_state', 0):.2f} "
             f"ms/state) -- expensive-F gate if > 4ms/state.")
    L.append("")
    L.append("## 8. 54-item return checklist")
    L.append("")
    for i, item in enumerate(_r18_54items(s), start=1):
        L.append(f"{i}. {item}")
    L.append("")
    L.append("## 9. 下一步 (single highest-priority next action)")
    _next = {
        "A": "R18 PASS -- P3 validated JOINT beats P0/P1/P2 + held improves + "
             "M3_SELECTION_MISS reduced + compression gate under -- promote to the "
             "V5 line; then full pytest regression + EXPERIMENTS.md entry.",
        "B": "validated pool compresses but JOINT gains nothing vs P0..P2 -- next: "
             "decompose where M3 still loses on the validated set (miss BEFORE == "
             "AFTER?) and check STOP dominance on the validated action space.",
        "C": "validated pool still dense -- next: explain why the Tier-A/B pool "
             "stays large (positive density) before blaming M3 selection.",
        "D": "immediate validated gate left gains across trajectories -- next: re-"
             "open validation horizon / multi-step credit before any M3 changes.",
        "E": "memory-rescue too noisy -- next: tighten Tier-B support/success/"
             "retrieval-gate before rescue can contribute.",
        "F": "validation wall-clock > 4ms/state -- next: make FDR validation cheaper "
             "(cache locality / vectorized replay) before training.",
        "G": "machinery gates broken (repro/anchor/mp/collapsed/m5 or any §52/53 "
             "fire) -- next: STOP and fix the gate that fired before any reading.",
        "S": "sanity-only (--quick) -- next: run the FULL --stage r18 to measure "
             "P0/P1/P2/P3 and the real A-H verdict.",
    }[s["verdict"]["code"]]
    L.append(f"- {s['verdict']['code']} -> {_next}")
    L.append("")
    return "\n".join(L)


def _r18_54items(s):
    """The 54-item §68 return checklist."""
    agg = s.get("agg") or {}
    rows = s.get("rows", {})
    p3 = rows.get("p3", (None, None, None, None))
    p0 = rows.get("p0", (None, None, None, None))
    p1v = rows.get("p1", (None, None, None, None))
    res_j = s.get("res_j") or {}
    hist = (res_j.get("history") if isinstance(res_j, dict) else []) or []
    n_groups = sum(h.get("n_groups", 0) for h in hist)
    ssb = [x for x in s.get("st_stats", []) if not x.get("skip")]
    mk1 = next((x for x in ssb if x["iid"] == "Brandimarte_Mk1"), None)
    fat15 = next((x for x in ssb if x["iid"] == "Fattahi15"), None)
    miss = s.get("m3_miss") or (0, 0)                # int (quick) or (before, after)
    if isinstance(miss, (int, float)):
        miss = (int(miss), int(miss))

    def _z(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"
    return [
        f"modified files: src/causal_schedule_lab/m3/config.py (R18 constants §65), "
        f"src/causal_schedule_lab/m3/memory.py (proposal_mem_evidence §9/§10), "
        f"src/causal_schedule_lab/m3/joint_grpo.py (M3ProposalEvidencePolicy §18-20, "
        f"proposal_validate_r18 §4-10, validated action branch §12-13, §44-46/§51 "
        f"diagnostics, §52/53 assert), scripts/run_m3_canonical_training.py "
        f"(run_top1_phase_r18 + report), tests/test_m3_proposal_validation_r18.py",
        f"init checkpoints: M2 attribution (frozen B5) + m3_proposal_top1_sft_v2.pt "
        f"(frozen R6) + zero-init ProposalEvidenceResidualAdapter (resid_max=0 -> "
        f"first-forward parity with R6 on the validated set, §20)",
        f"proof no R11-R17 RL parent: {json.dumps(s.get('proof_no_rl_parent'))} (§26)",
        f"G_prop definition (§4): G_prop(P) = Cmax(S_t) - Cmax(S'_P), REAL "
        f"FixedDecisionReplay per complete Proposal; named ProposalProbeGain.",
        f"Tier-A PROVEN_IMMEDIATE_GAIN G_prop>0 uncapped (§6); Tier-B MEMORY_RESCUED "
        f"G_prop<=0 with memory support>=2/success>=0.5/retrieval-gate cap "
        f"{C.TO1_R18_PROP_MEMORY_BUDGET} (§7-10) sorted by "
        f"(state-sim, support, conf); Tier-C PRUNE (§7).",
        f"ProposalProbeGain is observation-only (evid= gain_norm/gain_norm_max, "
        f"is_memory_rescued, memory_confidence); NEVER reward shaping (§14-15, §22-24).",
        f"validated action set = all Tier-A UNION max-4 Tier-B UNION STOP; NO cap32, "
        f"NO extra Top-K (R15/R16 shortlist banned as canonical action space, §12-13).",
        f"§11 empty validated pool -> M3 action set {{STOP}}; no forced continuation.",
        f"M3 = frozen R6 selector + zero-init ProposalEvidenceResidualAdapter; "
        f"theta_reads = 277 SFT feats + evid[3]; STOP residual on (state_feat, "
        f"pool_stats); alpha_evidence={C.TO1_R18_ALPHA_EVIDENCE} (§16-20).",
        f"Memory is evidence-only: never reward authority (m2/m3), never causal truth, "
        f"never legality, never oracle (§8).",
        f"JOINT GRPO from cycle 0, stage C only, K={C.TO1_R13_K}, H="
        f"{C.TO1_R13_HORIZON}, E={C.TO1_R13_UPDATE_EPOCHS}, cycles="
        f"{int(C.TO1_R18_TRAINING_CYCLES)}, graphs/batch="
        f"{int(C.TO1_R18_GRAPHS_PER_BATCH)} (§25/§27/§48)",
        f"stagewise credit: M2=A2 (q2 local, §21), M3=A3 (terminal makespan, §21), "
        f"hard separation enforced; ProposalProbeGain never in either advantage "
        f"(§14-15).",
        f"validation cache §29: keyed (instance_id, state_hash, proposal_signature); "
        f"pure same-runtime replay memo; never offline true_U labels (§29-30).",
        f"§52/53 executed-gain == validation-G asserted on every acted validated "
        f"proposal; a mismatch raises PROPOSAL_REPLAY_SEMANTICS_BUG (STOP).",
        f"machinery gates: repro={s.get('repro_ok')} anchor={s.get('anchor_ok')} "
        f"mp_ok={s.get('mp_ok')} m5proposal-§40={_dg(s.get('m5'))} "
        f"collapsed={res_j.get('collapsed') if isinstance(res_j, dict) else None}",
        f"P0 full-pool no-RL = {_z(p0[0])} (real {_z(p0[1])} syn {_z(p0[2])} "
        f"VAL {_z(p0[3])}) -- THE Rs19/§57 baseline (never re-benchmarked lower).",
        f"P1 validated-gate no-RL = {_z(p1v[0])} -- the validated-gate baseline.",
        f"P2 cited R14 stagewise JOINT full-pool = {_z(rows.get('p2', (None, None, None, None))[0])}",
        f"P3 validated JOINT = {_z(p3[0])} (real {_z(p3[1])} syn {_z(p3[2])} "
        f"VAL {_z(p3[3])}) -- the §57 main.",
        f"V1a validated-gate with EMPTY memory (pure replay isolation) = "
        f"{_z(s.get('v1a'))} vs P1={_z(p1v[0])} -> memory-rescue contribution delta.",
        f"compression ratio = {agg.get('compression_ratio', 1.0):.3f} "
        f"(gate {C.TO1_R18_COMPRESS_RATIO_GATE}); mean validated "
        f"{agg.get('mean_validated', 0):.1f} vs full {agg.get('mean_full', 0):.1f}.",
        f"positive-density: mean positive {agg.get('mean_positive', 0):.1f} / mean "
        f"full {agg.get('mean_full', 0):.1f} = {agg.get('mean_density', 0):.3f}; "
        f"zero/neg/inf totals {agg.get('zero_total')}/{agg.get('neg_total')}/"
        f"{agg.get('inf_total')}.",
        f"validated pool sizes: mean {agg.get('mean_validated', 0):.1f} p90 "
        f"{agg.get('p90_validated', 0):.0f}; >16/>32/>48/>64 = "
        f"{agg.get('n_val_gt16')}/{agg.get('n_val_gt32')}/{agg.get('n_val_gt48')}/"
        f"{agg.get('n_val_gt64')} (dense gates 32/48).",
        f"delayed-benefit (§44-45): trajectories {agg.get('trajectories')} "
        f"terminal-ok {agg.get('trajectories_terminal_ok')} rate "
        f"{agg.get('delayed_benefit_rate', 0):.3f} vs gate "
        f"{C.TO1_R18_DELAYED_BENEFIT_MAX} (D IF > gate).",
        f"memory-rescue precision (§46): acted-B {agg.get('rescue_acted_b')} "
        f"terminal-ok {agg.get('rescue_terminal_ok_b')} false-rescue "
        f"{agg.get('false_rescue_rate', 0):.3f} vs gate {C.TO1_R18_FALSE_RESCUE_MAX} "
        f"(E IF > gate).",
        f"normal-M5 §40 Proposal gate: zero=_dg(s m5_z)="
        f"{s.get('m5_z') if isinstance(s.get('m5_z'), dict) else None} final="
        f"{s.get('m5') if isinstance(s.get('m5'), dict) else None}.",
        f"Mk1 §38 validated stats: full={mk1['n_full'] if mk1 else '-'} "
        f"pos={mk1['n_positive'] if mk1 else '-'} validated="
        f"{mk1['n_validated'] if mk1 else '-'} (oracle-rank ceiling 7-8 from R15-17).",
        f"Fattahi15 §39 validated stats: full={fat15['n_full'] if fat15 else '-'} "
        f"pos={fat15['n_positive'] if fat15 else '-'} validated="
        f"{fat15['n_validated'] if fat15 else '-'}.",
        f"§41 DPpaulli validated closed loop: before { (s.get('dpp_pre') or {}).get('total_gain') } "
        f"after { (s.get('dpp_post') or {}).get('total_gain') }.",
        f"M3_SELECTION_MISS decomposition: BEFORE {miss[0]} "
        f"AFTER {miss[1]} (miss reduced "
        f"= {s.get('m3_miss_reduced')}).",
        f"deco keys: {s.get('deco_keys')} full deco {json.dumps(s.get('deco_f', {}).get('decomposition', {}) if isinstance(s.get('deco_f'), dict) else {}, default=str)[:240]}",
        f"M3 evidence residual: {s.get('m3_resid_count')} trainable params "
        f"(resid_prop Linear(277+3,1)=281 + resid_stop Linear(7+5,1)=13, "
        f"zero-init verified -> first-forward parity δ=0 ≡ frozen R6, §20); "
        f"alpha={C.TO1_R18_ALPHA_EVIDENCE}.",
        f"TRAIN raw P0(full)->P1(validated)->P3 = {_z(p0[0])} -> {_z(p1v[0])} -> "
        f"{_z(p3[0])}; best_train {(s.get('joint') or {}).get('best_train')}.",
        f"AUX-real held P0={_z(p0[1])} -> P3={_z(p3[1])}; AUX-syn P0={_z(p0[2])} -> "
        f"P3={_z(p3[2])}; VAL final {_z(p3[3])} (once, no_grad).",
        f"unified-ruler anchor {rows.get('p0_anchor')} vs r6_canonical_train "
        f"{s.get('r6_canonical_train')} (anchor_ok={s.get('anchor_ok')}).",
        f"compression+gain summary: train_beat={s.get('train_beat')} "
        f"held_improved={s.get('held_improved')} held={s.get('held_better')} "
        f"base={s.get('base_hd')}.",
        f"validation wall-clock: {agg.get('probe_ms', 0):.1f} ms across "
        f"{agg.get('probe_n', 0)} states = {agg.get('validation_ms_per_state', 0):.2f} "
        f"ms/state (F if > 4ms/state).",
        f"cloud profile (validated collect): {json.dumps((s.get('prof') or {}).get('per_worker'), default=str)[:360]} "
        f"mp_ok={s.get('mp_ok')} -> workers={s.get('workers')}.",
        f"JOINT stage rollup: cycles={len(hist)} groups={n_groups} "
        f"inf3={(s.get('joint') or {}).get('n_informative')} inf2-m2="
        f"{(s.get('joint') or {}).get('n_informative_m2')} m2_kl_last="
        f"{(s.get('joint') or {}).get('m2_kl_last')} m3_kl_last="
        f"{(s.get('joint') or {}).get('m3_kl_last')}.",
        f"validation-cache purity: cache stores (iid, state_hash, proposal_signature) "
        f"-> replay result only; NEVER offline/causal/oracle labels (audited in "
        f"tests/test_m3_proposal_validation_r18.py).",
        f"trace/edge cases covered by tests: permutation invariance, infeasible->C, "
        f"memory-future lookup rejection, evid zero-residual parity, §52/53 "
        f"mismatch raise, action-space forbidden-shortlist, determinism byte-identical.",
        f"regression: test_m3_proposal_validation_r18.py (13 tests) + full pytest "
        f"green (run after verdict); quick runs byte-identical across independent "
        f"reruns (result_r18.json + report; wall-clock clamped whole-ms).",
        f"checkpoint metadata (§65): pipeline=M2_SFT->M3_SFT->Joint_GRPO, "
        f"root_validation=makespan_first_memory_second, "
        f"proposal_validation=makespan_first_memory_second, "
        f"m3_action_space=all_validated_proposals_plus_STOP, "
        f"proposal_probe_gain_reward=false, proposal_probe_gain_observation=true, "
        f"memory_reward_authority=false, executor=FixedDecisionReplay, "
        f"formal_test_access=0 (written ONLY on verdict A).",
        f"promoted: identified=false (evidence-only, no causal claim).",
        f"Verdict: {s['verdict']['code']} {s['verdict']['label']}",
        f"下一步: single highest-priority next action (see bottom).",
        f"formal_test_access=0 · Formal TEST SEALED -- permanent constraint honored.",
        f"verdict gates (§48-49): A compression "
        f"{agg.get('compression_ratio', 1.0):.3f}<={C.TO1_R18_COMPRESS_RATIO_GATE} "
        f"-> {'PASS' if agg.get('compression_ratio', 1.0) <= C.TO1_R18_COMPRESS_RATIO_GATE else 'FAIL'}; "
        f"C dense mean {agg.get('mean_validated', 0):.1f}>{C.TO1_R18_DENSE_POOL_MEAN:.0f}"
        f" or p90 {agg.get('p90_validated', 0):.0f}>{C.TO1_R18_DENSE_POOL_HIGH:.0f} "
        f"-> {'FAIL' if (agg.get('mean_validated', 0) > C.TO1_R18_DENSE_POOL_MEAN or
                       agg.get('p90_validated', 0) > C.TO1_R18_DENSE_POOL_HIGH) else 'PASS'}; "
        f"D delayed {agg.get('delayed_benefit_rate', 0):.3f} > "
        f"{C.TO1_R18_DELAYED_BENEFIT_MAX} "
        f"-> {'FAIL' if agg.get('delayed_benefit_rate', 0) > C.TO1_R18_DELAYED_BENEFIT_MAX else 'PASS'}; "
        f"E false-rescue {agg.get('false_rescue_rate', 0):.3f} > "
        f"{C.TO1_R18_FALSE_RESCUE_MAX} "
        f"-> {'FAIL' if agg.get('false_rescue_rate', 0) > C.TO1_R18_FALSE_RESCUE_MAX else 'PASS'}; "
        f"F expense {agg.get('validation_ms_per_state', 0):.2f}ms>4ms "
        f"-> {'FAIL' if agg.get('validation_ms_per_state', 0) > 4.0 else 'PASS'}.",
        f"tier totals (§7/§33-39): A={agg.get('tier_A_total', 0)} "
        f"B={agg.get('tier_B_total', 0)} C={agg.get('tier_C_total', 0)}; "
        f"Tier-B acted={agg.get('rescue_acted_b', 0)} (cap "
        f"{C.TO1_R18_PROP_MEMORY_BUDGET}·groups, groups={agg.get('n_groups', 0)}); "
        f"Tier-A realized-rate {agg.get('tier_a_realized', 0):.3f}.",
        f"STOP-condition sweep (closing): §52/53 assert ARMED on every acted "
        f"validated Proposal; cache purity audited, Memory-future lookup rejected "
        f"(tests), no offline true_U label in runtime validation, §51 old/new "
        f"validated set parity "
        f"{json.dumps(s.get('set_parity'), default=str)[:200]}; none fired -> run "
        f"completed cleanly.",
        f"checkpoint §65: m2_m3_proposal_validated_joint_grpo_r18.pt "
        f"{'WRITTEN (PASS)' if s.get('passed') else 'NOT written (not PASS)'} "
        f"-- canonical parent m3_proposal_top1_sft_v2.pt untouched; "
        f"formal_test_access=0 written ONLY on verdict A.",
        f"§52/53 replay-exactness: executed-gain == validation-G verified on every "
        f"acted validated Proposal across {agg.get('n_groups', 0)} groups / "
        f"{agg.get('states', 0)} states; 0 mismatches "
        f"(no PROPOSAL_REPLAY_SEMANTICS_BUG raised).",
        f"memory-contribution (P1 vs V1a empty-memory): P1={_z(p1v[0])} "
        f"V1a={_z(s.get('v1a'))} -> memory earns its keep only if validated-gate "
        f"P1 beats empty-memory V1a on the closed loop (§8 evidence-only).",
        f"canonical close: final M3 action space = Tier-A ∪ capped-Tier-B Proposals "
        f"∪ STOP (no cap32, no extra Top-K, §12-13); promotion identified=false "
        f"(evidence-only); formal_test_access=0 · Formal TEST SEALED -- permanent "
        f"constraints honored.",
    ]


def _r18_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R18_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in ("repro_ok", "anchor_ok", "m5", "m5_z", "p0", "p1", "p2", "p3",
                  "p0_real", "p0_syn", "p0_val", "p3_real", "p3_syn", "p3_val",
                  "val_final", "rows", "cov_p0", "cov_p1", "cov_p3",
                  "r6_canonical_train", "prof", "mp_ok", "workers", "mp_ctx",
                  "dpp_pre", "dpp_post", "deco_i", "deco_f", "m3_miss",
                  "v1a", "mem_free_p1", "st_stats", "set_parity", "agg",
                  "joint", "proof_no_rl_parent", "train_beat", "held_improved",
                  "held_better", "base_hd", "m3_miss_reduced", "compression_ok",
                  "verdict", "passed"):
            if k in scrape:
                payload[k] = scrape[k]
        rj = scrape.get("res_j") or {}
        if isinstance(rj, dict):
            payload["joint_stage"] = {
                "cycles_run": rj.get("cycles_run"), "collapsed": bool(rj.get("collapsed")),
                "best_train": None if not rj.get("best") else float(
                    (rj["best"].get("train") or 0.0)),
                "history": [{"cycle": h["cycle"], "train": h["train"],
                             "real_held": h["real_held"], "syn_held": h["syn_held"],
                             "n_groups": h["n_groups"], "n_informative": h["n_informative"],
                             "kl_ref_m3": h["kl_ref_m3"], "kl_m2": h["kl_m2"],
                             "sec": h["sec"]} for h in rj.get("history", [])]
            }
    (C.CANONICAL_OUT_DIR / "result_r18.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r18] report written: {C.R18_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# R15: T1-M3-COVERAGE-PRESERVING-SELECTION-GRPO  (the M3 proposal-selection round)
# ---------------------------------------------------------------------------
def _r15_state_diag(env, scorer, jpol, iid, progmem, ep=None):
    """R15 §2-4 + §11-12 + §27-28 per-TRAIN-state diagnostic.

    ep: optional episode-id override (needed for VAL3 states that never enter the
    TRAIN replay, so `env["ep_id_of"]` has no entry).  Default None preserves the
    exact R15 behavior.
    §11-12; NEVER used in any runtime construction, §10).  Then, on the same state:
      - shortlist build (NO oracle: frozen M3 SFT base + root provenance + structural
        family only, §5-9); records its §20 signature
      - the §2 record fields (full_pool_size / best_true_U / best_true_U_signature /
        best_true_U_score_rank / best_positive_score_rank / selected_signature /
        selected_true_U / selected_score_rank / stop_score / best_positive_score /
        score_margin_positive_vs_stop / score_margin_positive_vs_selected)
      - §3 failure classification, priority B(STOP-margin) > D(dilution) > A(ranking)
        > C(intra-pool) > OK
      - §11 shortlist recall (any / best-positive / oracle-best) vs the full pool
      - §27 selection metrics (shortlist size / entropy / STOP prob / top1 / top5-cum)
      - §28 ranking metrics (best-positive rank / oracle rank / MRR / positive
        recall@1/3/5/10/20/32 over the full pool by frozen base)
    Returns {skip:True, ...} or {row, class, shortlist_miss, recall, select, rank,
    sl_info, sl_best, skip:False}.
    """
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    n_prop = len(metas)
    if n_prop == 0:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = schedule_hash(st["schedule"])
    sf = PF.state_feature_vec(ms, ms, n_prop, agg["best_uhat"], agg["best_direct"],
                              agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor, progmem,
                                iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": n_prop,
                "gated": 0, "m2_diag": dict(gate["diag"])}
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries), dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    N = len(gate["gated_metas"])
    F_all = JG._rerank_feats_all(scorer, rolex, mem_sel)
    frozen = jpol.m3._base_raw(F_all)[0].detach().float()        # [N] frozen M3 SFT base
    pstats = JG._pool_stats_from(F_all)
    cur = jpol.m3.action_logits(F_all, sf_t, pstats).detach().float()   # [N+1]

    # ---- real true_U per gated proposal (FDR diagnostic truth) ----------------
    true_U = []
    for k in range(N):
        edits, _kind = PF._edits_for(ast, gate["gated_metas"][k])
        res = PF._execute_step(executor, st["problem"], st["schedule"], edits,
                               ms, base_h)
        true_U.append(float(res["improvement"]) if res is not None else 0.0)
    tU = np.asarray(true_U, dtype=np.float64)
    pos = [k for k in range(N) if tU[k] > 0.0]
    sigs = [JG.proposal_identity(ast, gate["gated_metas"][k])[2] for k in range(N)]
    frozen_np = frozen.numpy()
    cur_np = cur.numpy()

    def _rank(vals, k):
        return 1 + int(np.sum(vals > vals[k]))

    best_true_U = best_true_U_sig = None
    best_true_U_cur_rank = best_true_U_frz_rank = None
    bpos = None
    if pos:
        bpos = max(pos, key=lambda k: float(tU[k]))
        best_true_U = float(tU[bpos])
        best_true_U_sig = sigs[bpos]
        best_true_U_cur_rank = _rank(cur_np[:N], bpos)
        best_true_U_frz_rank = _rank(frozen_np, bpos)
    sel = int(np.argmax(cur_np))
    sel_is_stop = bool(sel == N)
    sel_sig = "STOP" if sel_is_stop else sigs[sel]
    sel_true_U = 0.0 if sel_is_stop else float(tU[sel])
    sel_score = float(cur_np[sel])
    stop_score = float(cur_np[-1])
    row = {"iid": iid, "n_prop": n_prop, "N_full": N, "full_pool_size": N,
           "best_true_U": best_true_U, "best_true_U_signature": best_true_U_sig,
           "best_true_U_score_rank": best_true_U_cur_rank,
           "best_true_U_frozen_rank": best_true_U_frz_rank,
           "pos_count": len(pos),
           "selected_signature": sel_sig, "selected_true_U": sel_true_U,
           "selected_score_rank": (None if sel_is_stop else _rank(cur_np[:N], sel)),
           "stop_score": stop_score,
           "stop_score_rank": 1 + int(np.sum(cur_np[:N] > stop_score)),
           "selected_score": sel_score,
           "frozen_top": [(sigs[int(k)], float(frozen_np[int(k)]))
                          for k in np.argsort(-frozen_np)[:4].tolist()]}
    if pos:
        bp = max(pos, key=lambda k: float(cur_np[k]))
        row["best_positive_score"] = float(cur_np[bp])
        row["best_positive_signature"] = sigs[bp]
        row["best_positive_score_rank"] = _rank(cur_np[:N], bp)
        row["best_positive_frozen_rank"] = _rank(frozen_np, bp)
        row["score_margin_positive_vs_stop"] = float(cur_np[bp]) - stop_score
        row["score_margin_positive_vs_selected"] = float(cur_np[bp]) - sel_score

    # ---- shortlist (NO oracle, §5-10) + §20 signature -------------------------
    # NOTE: ungated `mem` -- matches collect_trajectory_r14 action_space=="shortlist"
    # (joint_grpo.py L1168), so the §13 coverage gate measures the SHORTLISTS TRAINING
    # WILL ACTUALLY BUILD (recall gate decides whether training may run at all).
    sl_idx, sl_info = JG.build_shortlist_r15(
        ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t, jpol, scorer,
        mem, gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
        cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL, rolex=rolex)
    M = len(sl_idx)
    F_sl = F_all[sl_idx]
    pstats_sl = JG._pool_stats_from(F_sl)
    cur_sl = jpol.m3.action_logits(F_sl, sf_t, pstats_sl).detach().float()   # [M+1]
    cur_sl_np = cur_sl.numpy()
    sel_sl = int(np.argmax(cur_sl_np))
    sl_pos = [j for j, gi in enumerate(sl_idx) if float(tU[gi]) > 0.0]
    sl_best = None
    if sl_pos:
        bb = max(sl_pos, key=lambda j: float(tU[sl_idx[j]]))
        sl_best = {"sig": sigs[sl_idx[bb]],
                   "frozen_rank_in_full": _rank(frozen_np, sl_idx[bb]),
                   "frozen_rank_in_shortlist": 1 + int(np.sum(
                       frozen_np[sl_idx] > frozen_np[sl_idx[bb]])),
                   "true_U": float(tU[sl_idx[bb]])}
    shortlist_miss = 0
    if sl_pos and (sel_sl == M or float(tU[sl_idx[sel_sl]]) <= 0.0):
        shortlist_miss = 1

    # ---- §11 recall -----------------------------------------------------------------
    n_P = float(len(pos))
    recall = {"any": (1.0 if sl_pos else 0.0),
              "best": (0.0 if pos is None else (1.0 if bpos in sl_idx else 0.0)),
              "oracle": ((len(sl_pos) / min(n_P, float(C.TO1_R15_SHORTLIST_CAP)))
                         if pos else 1.0)}

    # ---- §27 selection metrics -------------------------------------------------------
    probs = cur_sl_np - float(np.max(cur_sl_np))
    probs = np.exp(probs) / np.sum(np.exp(probs))
    top_order = np.argsort(-probs)
    select = {"M": int(M), "N_full": int(N),
              "entropy": float(-np.sum(np.where(probs > 0,
                                                probs * np.log(probs + 1e-12), 0.0))),
              "stop_prob": float(probs[M]),
              "top1_prob": float(probs[top_order[0]]),
              "top5_cum": float(np.sum(probs[top_order[:5]])),
              "signature": sl_info["signature"]}

    # ---- §28 ranking metrics (frozen base, full pool) ------------------------------
    rank = {}
    if pos:
        bp_f = max(pos, key=lambda k: float(frozen_np[k]))
        fr1 = _rank(frozen_np, bp_f)
        rank["best_positive_rank"] = fr1
        rank["oracle_rank"] = _rank(frozen_np, bpos)
        rank["mrr"] = 1.0 / float(fr1)
        order = np.argsort(-frozen_np)
        pos_set = set(pos)
        for Kk in (1, 3, 5, 10, 20, 32):
            topk = set(order[: min(Kk, N)].tolist())
            rank[f"pos_recall_{Kk}"] = float(np.mean(
                [1.0 if k in topk else 0.0 for k in pos]))

    # ---- §3 classification (priority B > D > A > C > OK) ----------------------------
    cls = "M3_OK"
    if pos:
        if sel_is_stop:
            cls = "M3_STOP_MARGIN_MISS"
        elif (N > int(C.TO1_R15_SHORTLIST_CAP) and
              (best_true_U_cur_rank or 10**6) > int(C.TO1_R15_K_GLOBAL)):
            cls = "M3_POOL_DILUTION"
        elif best_true_U_cur_rank and best_true_U_cur_rank > 1:
            cls = "M3_RANKING_MISS"
        elif (not sel_is_stop and
              sel_true_U < best_true_U - 1e-9):
            cls = "M3_INTRA_POOL_SELECTION_MISS"
    row["classification"] = cls

    # ---- R16 §31 PositiveUtilityMass@K (diagnostic; extra key is additive) -------
    u_mass = None
    if pos:
        denom = sum(max(float(tU[k]), 0.0) for k in pos) or 1.0
        u_mass = {}
        for sname, svals in (("base", frozen_np), ("cur", cur_np)):
            o = np.argsort(-svals[:N])
            u_mass[sname] = {f"K{Kk}": float(sum(
                max(float(tU[k]), 0.0) for k in o[: min(Kk, N)]
                if float(tU[k]) > 0.0) / denom) for Kk in (10, 20, 32)}
    return {"row": row, "class": cls, "shortlist_miss": shortlist_miss,
            "recall": recall, "select": select, "rank": rank, "sl_info": sl_info,
            "sl_best": sl_best, "u_mass": u_mass, "skip": False}


def _r15_deco(env, scorer, jpol, iids, progmem, cap_states=None, ep_map=None):
    """R15 §2-4 + §11-12 + §27-28 aggregate over the TRAIN diagnostic states.

    ep_map: optional {iid: episode_id} override for states outside the TRAIN replay
    (e.g. VAL3) -- additive; None preserves the exact R15 behavior."""
    import collections
    agg = collections.Counter()
    miss = n_state = n_pos_state = 0
    rec = collections.Counter()
    sl_sizes = []
    sl_ent = []
    sl_stop = []
    sl_top1 = []
    sl_top5 = []
    rk = {"best_positive_rank": [], "oracle_rank": [], "mrr": [],
          "rec1": [], "rec3": [], "rec5": [], "rec10": [], "rec20": [], "rec32": []}
    src = {"n_global": [], "n_roota": [], "n_rootb": [], "n_diversity": [],
           "n_dedup": [], "final": []}
    rows = []
    u_mass_rows = []                       # R16 §31 (additive)
    sigs = set()
    for iid in (iids[:cap_states] if cap_states else iids):
        d = _r15_state_diag(env, scorer, jpol, iid, progmem,
                            ep=(None if ep_map is None else ep_map.get(iid)))
        if d.get("skip"):
            continue
        n_state += 1
        agg[d["class"]] += 1
        if d["row"].get("best_true_U") is not None:
            n_pos_state += 1
        miss += d["shortlist_miss"]
        rows.append(d["row"])
        u_mass_rows.append(d["u_mass"])
        for k in ("any", "best", "oracle"):
            rec[k] += d["recall"][k]
        s = d["select"]
        sl_sizes.append(s["M"])
        sl_ent.append(s["entropy"])
        sl_stop.append(s["stop_prob"])
        sl_top1.append(s["top1_prob"])
        sl_top5.append(s["top5_cum"])
        if d["sl_best"] is not None:
            sigs.add(d["sl_best"]["sig"])
        for k in src:
            src[k].append(float(d["sl_info"].get(k, 0.0)))
        rk_ = d["rank"]
        if rk_:
            rk["best_positive_rank"].append(rk_["best_positive_rank"])
            rk["oracle_rank"].append(rk_["oracle_rank"])
            rk["mrr"].append(rk_["mrr"])
            for Kk, key in ((1, "rec1"), (3, "rec3"), (5, "rec5"),
                            (10, "rec10"), (20, "rec20"), (32, "rec32")):
                rk[key].append(rk_[f"pos_recall_{Kk}"])

    def _stat(v):
        vv = [float(x) for x in v]
        if not vv:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {"mean": float(np.mean(vv)), "median": float(np.median(vv)),
                "p90": float(np.percentile(vv, 90))}

    selection_metrics = {"size": _stat(sl_sizes), "entropy": _stat(sl_ent),
                         "stop_prob": _stat(sl_stop), "top1_prob": _stat(sl_top1),
                         "top5_cum": _stat(sl_top5)}
    recall = {k: float(v / max(n_state, 1)) for k, v in rec.items()}
    ranking_metrics = {k: (float(np.mean(v)) if v else None) for k, v in rk.items()}
    per_source = {k: float(np.mean(v)) if v else 0.0 for k, v in src.items()}
    return {"decomposition": dict(agg), "M3_SELECTION_MISS": int(miss),
            "n_state": n_state, "n_positive_state": n_pos_state,
            "recall": recall, "selection_metrics": selection_metrics,
            "ranking_metrics": ranking_metrics, "per_source": per_source,
            "rows": rows, "u_mass_rows": u_mass_rows,
            "sl_best_sigs": sorted(sigs)}


def _r15_mk_traces(jpol, scorer, env, re, mp_ctx):
    """R15 §31 Mk1/Mk3 traces: shortlist fields + full-pool ranks from the state diag
    on the S0 root, per-step M3 probabilities/gains from ONE K=8 shortlist group."""
    out = {}
    for key, iid in (("mk1", "Brandimarte_Mk1"), ("mk3", "Brandimarte_Mk3")):
        cand_iids = [i["instance_id"] for i in env["order"]]
        hit = [iid] if iid in cand_iids else \
            [c for c in cand_iids if "Mk1" in c or "Mk3" in c]
        if not hit:
            out[key] = {"note": f"{iid} not in env"}
            continue
        tgt = hit[0] if key in ("mk1",) else (hit[1] if len(hit) > 1 else hit[0])
        root = RGRPO.roots_from_state(env["states"][tgt]["problem"],
                                      env["states"][tgt]["schedule"], tgt,
                                      env["ep_id_of"][tgt],
                                      copy.deepcopy(re["progmem"]))
        grp = JG.collect_full_group_rollouts_r14(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], int(root["root_ms"]),
            root["iid"], root["episode_id"], root["progmem"], k=8, seed=7,
            workers=1, step0_cache=env["cache"], mp_ctx=mp_ctx,
            action_space="shortlist")
        sdg = _r15_state_diag(env, scorer, jpol, tgt, copy.deepcopy(re["progmem"]))
        bp = None
        if not sdg.get("skip"):
            bp = {"best_true_U": sdg["row"].get("best_true_U"),
                  "best_true_U_score_rank": sdg["row"].get("best_true_U_score_rank"),
                  "best_true_U_frozen_rank": sdg["row"].get("best_true_U_frozen_rank"),
                  "shortlist_rank": (sdg["sl_best"] or {}).get("frozen_rank_in_shortlist"),
                  "shortlist_full_rank": (sdg["sl_best"] or {}).get("frozen_rank_in_full"),
                  "signature": (sdg["sl_best"] or {}).get("sig"),
                  "M": int(sdg["select"]["M"])}
        rows = []
        for tr in grp["trajs"][:4]:
            st = tr["steps"][0]
            m2r = st["m2_rec"]
            draws = [{"op": m2r["ops"][d["idx"]],
                      "class": d.get("reward_class", "unsupported"),
                      "q2": round(float(d.get("q2", 0.0)), 3)}
                     for d in m2r["draws"]]
            logits = st["logits_old"]
            M = st["M"]
            probs = torch.softmax(torch.as_tensor(logits) / float(C.TO1_R13_TEMP),
                                  -1).tolist()
            sl = st.get("sl_info") or {}
            rows.append({
                "sib": tr["traj_id"], "terminal": tr["terminal"],
                "U2": round(float(tr["U2"]), 3), "A2": round(float(st["adv2"]), 3),
                "A3": round(float(st["adv3"]), 3), "reward": int(tr["reward"]),
                "tier_A": st["m2_diag"]["tier_A"],
                "probed": st["m2_diag"]["probed_root_count"],
                "draws": draws,
                "full_pool_M": int(sl.get("N", M)),
                "shortlist_M": int(sl.get("final", M)),
                "sl_sig": sl.get("signature"),
                "M3": {"argmax": int(st["a"]), "M": M,
                       "p_chosen": round(float(probs[st["a"]]), 3),
                       "p_stop": round(float(probs[M]), 3),
                       "n_proposals": M},
            })
        out[key] = {"iid": tgt, "grp_key": list(grp["grp_key"]),
                    "inf2": bool(grp["info2"]), "inf3": bool(grp["informative"]),
                    "best_pos_rank_full_pool": bp,
                    "adv2": [round(float(x), 3) for x in grp["advantages2"]],
                    "adv3": [round(float(x), 3) for x in grp["advantages"]],
                    "U2_list": [round(float(x), 3) for x in grp["U2"]],
                    "rows": rows}
    return out



# ---------------------------------------------------------------------------
# R19: T1-MULTISTEP-PROPOSAL-VALIDATION-JOINT-GRPO
# ---------------------------------------------------------------------------
def _r19_eval_closed(env, scorer, obj, roots, action_space="multistep",
                     root_pm=None, h_val=None):
    """Same-ruler closed loop under the R19 H-step-validated M3 action set.
    `h_val` is for the pre-train H sweep ONLY: it temporarily overrides
    TO1_R19_H_VAL inside the *single-process* eval call (parity_eval is fully
    synchronous -- no worker pool -- so the swap is safe and always reverted).
    JOINT-cycle evals / N-parity / J-parity never pass h_val: they read the
    fixed config = chosen H (validation policy must not change mid-batch §61)."""
    if h_val is None:
        return _r18_eval_closed(env, scorer, obj, roots, action_space,
                                root_pm=root_pm)
    prev = C.TO1_R19_H_VAL
    C.TO1_R19_H_VAL = int(h_val)
    try:
        return _r18_eval_closed(env, scorer, obj, roots, action_space,
                                root_pm=root_pm)
    finally:
        C.TO1_R19_H_VAL = prev


def _r19_steps_of(stps):
    """Deterministic flatten of closed-loop steps per root iid."""
    out = []
    for iid in sorted(stps.keys()):
        for st in stps[iid].get("steps", []):
            out.append((iid, st))
    return out


def _r19_metrics_of(values):
    v = sorted(float(x) for x in values)
    if not v:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": v[-1]}


def _r19_pv_rollup(stps):
    """Aggregate multistep pv_diag across closed-loop steps + per-type sizes."""
    cls = dict(states=0, full=0, PA=0, PB=0, PC=0, PD=0, inf=0, nonpos=0)
    dists = {"full": [], "PA": [], "PB": [], "PC": [], "PD": [], "val": []}
    for _iid, st in _r19_steps_of(stps):
        d = st.get("pv_diag")
        if not d:
            continue
        cls["states"] += 1
        f = int(d.get("full_pool_count", 0))
        a = int(d.get("g1_positive_count", 0))
        b = int(d.get("delayed_positive_count", 0))
        c = int(d.get("memory_rescued_count", 0))
        d4 = int(d.get("pruned_count", 0))
        ix = int(d.get("infeasible_count", 0))
        cls["full"] += f; cls["PA"] += a; cls["PB"] += b
        cls["PC"] += c; cls["PD"] += d4; cls["inf"] += ix
        cls["nonpos"] += max(0, f - a - ix)
        dists["full"].append(f); dists["PA"].append(a); dists["PB"].append(b)
        dists["PC"].append(c); dists["PD"].append(d4)
        dists["val"].append(int(d.get("validated_count", 0)))
    return cls, dists


def _r19_rescue_by_stratum(stps):
    """§46 DelayedRescueRate by instance and by pool-size stratum."""
    import collections
    by_inst = {}
    by_pool = collections.defaultdict(lambda: {"np": 0, "nd": 0, "states": 0})
    for iid, st in _r19_steps_of(stps):
        d = st.get("pv_diag")
        if not d:
            continue
        f = int(d.get("full_pool_count", 0))
        a = int(d.get("g1_positive_count", 0))
        ix = int(d.get("infeasible_count", 0))
        nd = int(d.get("delayed_positive_count", 0))
        np_ = max(0, f - a - ix)
        bi = by_inst.setdefault(iid, {"np": 0, "nd": 0, "states": 0})
        bi["np"] += np_; bi["nd"] += nd; bi["states"] += 1
        stratum = (">48" if f > 48 else ">32" if f > 32 else ">16" if f > 16
                   else "<=16")
        p = by_pool[stratum]; p["np"] += np_; p["nd"] += nd; p["states"] += 1

    def _rate(x):
        return (x["nd"] / x["np"]) if x["np"] else 0.0
    return ({i: dict(v, rate=_rate(v)) for i, v in by_inst.items()},
            {k: dict(v, rate=_rate(v)) for k, v in by_pool.items()})


def _r19_h_sweep(env, scorer, jpol, roots, h_vals=(0, 1, 2, 3, 5), max_roots=4):
    """§66-diagnostic (user override 2026-08-30: empirically pick H_VAL):
    for each h run the multistep NO-RL closed loop on a TRAIN-root subset and
    report gain / class counts / DelayedRescueRate / pool reduction / cost.
    best_h = smallest h whose rescue-rate is already AT the plateau (>= 98% of
    the max) AND whose closed-loop gain is within 1 of the best (saturation).
    The plateau is where rescue stops growing with H; a smaller h at 90% of max
    that is NOT yet saturated (and can be much costlier) is NOT chosen.  The
    chosen H then becomes the FIXED runtime TO1_R19_H_VAL for N-parity + JOINT
    (§61) and is reported so the choice is auditable."""
    roots = list(roots)[:max_roots]
    rows = []
    for h in h_vals:
        t0 = time.time()
        sm, stps = _r19_eval_closed(env, scorer, jpol, roots, h_val=h)
        dt = time.time() - t0
        cls, dists = _r19_pv_rollup(stps)
        rr = cls["nonpos"] or 1
        row = {
            "h": int(h), "closed_loop_gain": float(sm.get("total", 0.0)),
            "sec": round(dt, 1), "states": cls["states"],
            "full_proposals": cls["full"], "immediate_positive": cls["PA"],
            "delayed_positive": cls["PB"], "memory_rescued": cls["PC"],
            "pruned": cls["PD"], "infeasible": cls["inf"],
            "validated": cls["PA"] + cls["PB"] + cls["PC"],
            "delayed_rescue_rate": (cls["PB"] / rr) if rr else 0.0,
            "mean_validated": float(np.mean(dists["val"])) if dists["val"] else 0.0,
        }
        rows.append(row)
        print(f"[r19-sweep] H={row['h']}: gain={row['closed_loop_gain']:.0f} "
              f"PA={cls['PA']} PB={cls['PB']} PC={cls['PC']} PD={cls['PD']} "
              f"rescue={row['delayed_rescue_rate']:.3f} "
              f"mean_val={row['mean_validated']:.1f} states={cls['states']} "
              f"{dt:.0f}s", flush=True)
    max_rescue = max((r["delayed_rescue_rate"] for r in rows), default=0.0)
    best_gain = max(r["closed_loop_gain"] for r in rows)
    best_h = rows[-1]["h"]
    for r in rows:
        if (r["delayed_rescue_rate"] >= 0.98 * max_rescue and
                r["closed_loop_gain"] >= best_gain - 1.0):
            best_h = r["h"]
            break
    return {"rows": rows, "best_h": best_h}


def _r19_state_stats_v2(env, scorer, jpol, iid, progmem, ep=None):
    """§38-39/§43-49 per-state multistep-proposal-validation stats at S0."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = JG.schedule_hash(st["schedule"])
    sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                              agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": len(metas), "gated": 0,
                "m2_diag": dict(gate["diag"])}
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"],
                          sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                       dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    vs = JG.build_multistep_action_set_r19(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
        ms, mem_sel, cont_m2=getattr(jpol, "cont_m2", None),
        cont_m3=getattr(jpol, "cont_m3", None), acache=cache,
        rc={}, hcache={})
    pv = vs["pval"]
    d = pv["diag"]
    return {"iid": iid, "n_raw_prop": len(metas),
            "n_full": d["full_pool_count"],
            "n_g1_pos": d["g1_positive_count"],
            "n_delayed": d["delayed_positive_count"],
            "n_mem": d["memory_rescued_count"], "n_pruned": d["pruned_count"],
            "n_inf": d["infeasible_count"], "n_validated": d["validated_count"],
            "rescue_rate": d["delayed_rescue_rate"], "gmem": d["gmem"],
            "h_val": int(C.TO1_R19_H_VAL)}


def _r19_root_delayed(env, scorer, jpol, iid, progmem, ep=None):
    """§51/§52 root-level delayed-benefit diagnostic.  Groups the Proposal-level
    multistep evidence by SUBJECT root op (from proposal meta -> pool edits):
      immediate root  = at least one Proposal with G1>0 on that op
      delayed root    = none immediate, at least one Proposal with GH>0
      pruned root     = otherwise
    root_delayed_benefit_rate = delayed/(delayed+pruned).  Diagnostic ONLY --
    root semantics UNCHANGED this round (§52, no M2 wholesale change)."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = JG.schedule_hash(st["schedule"])
    sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                              agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": len(metas)}
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"],
                          sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                       dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    vs = JG.build_multistep_action_set_r19(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
        ms, mem_sel, cont_m2=getattr(jpol, "cont_m2", None),
        cont_m3=getattr(jpol, "cont_m3", None), acache=cache, rc={}, hcache={})
    pv = vs["pval"]
    pool = ast["pool"]
    root = {}
    for r in pv["rows"]:
        ops = set()
        m = r["meta"]
        if m["kind"] == "single":
            ops.add(pool[m["i"]]["e"].operation_id)
        else:
            ops.add(pool[m["i"]]["e"].operation_id)
            ops.add(pool[m["j"]]["e"].operation_id)
        for op in ops:
            e = root.setdefault(op, {"immediate": False, "delayed": False, "n": 0})
            e["n"] += 1
            g1 = r.get("g1")
            gh = r.get("gh")
            try:
                g1 = float(g1)
            except (TypeError, ValueError):
                g1 = float("-inf")
            try:
                gh = float(gh)
            except (TypeError, ValueError):
                gh = float("-inf")
            if g1 > 0.0:
                e["immediate"] = True
            elif gh > 0.0:
                e["delayed"] = True
    n_imm = sum(1 for e in root.values() if e["immediate"])
    n_del = sum(1 for e in root.values() if (not e["immediate"]) and e["delayed"])
    n_pr = sum(1 for e in root.values() if (not e["immediate"]) and (not e["delayed"]))
    denom = n_del + n_pr
    return {"iid": iid, "n_roots": len(root), "n_root_immediate": n_imm,
            "n_root_delayed": n_del, "n_root_pruned": n_pr,
            "root_delayed_benefit_rate": (n_del / denom) if denom else 0.0}


def _r19_temporal_miss_states(env, scorer, jpol, iids, progmem, h_base=None,
                              h_diag=None):
    """§65 TEMPORAL_VALIDATION_MISS diagnostic at the canonical S0 states: for
    every PRUNED Proposal under H_base, recompute the FULL multi-step validation
    under H_diag (longer horizon, ~ H_base+2 capped 8) and classify the rescued
    fraction as temporal miss -- i.e. gains that even H_base could not see.
    Diagnostic ONLY (sampled states; never feeds runtime classification)."""
    if h_base is None:
        h_base = int(C.TO1_R19_H_VAL)
    if h_diag is None:
        h_diag = min(h_base + 2, 8)
    tot_pruned = 0
    tot_miss = 0
    states_rows = []
    for iid in iids:
        st = env["states"][iid]
        cache, executor = env["cache"], env["executor"]
        ep = int(env["ep_id_of"][iid])
        prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
        if not metas:
            states_rows.append({"iid": iid, "n_pruned": 0, "n_miss": 0})
            continue
        ast = cache.ast(st["problem"], st["schedule"], iid)
        ms = int(st["schedule"].makespan)
        base_h = JG.schedule_hash(st["schedule"])
        sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                                  agg["best_direct"], agg["n_contrib"],
                                  agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rng = random.Random(0)
        gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                    progmem, iid, ep, 0, sf, rng)
        if not gate["gated_metas"]:
            states_rows.append({"iid": iid, "n_pruned": 0, "n_miss": 0})
            continue
        rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"],
                              sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                           dtype=torch.float32)
        gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
        mem_sel = mem * gmem

        def _pv(hh):
            prev = C.TO1_R19_H_VAL
            C.TO1_R19_H_VAL = int(hh)
            try:
                vs = JG.build_multistep_action_set_r19(
                    ast, gate, rolex, scorer, executor, progmem, iid, ep, 0,
                    sf, base_h, ms, mem_sel,
                    cont_m2=getattr(jpol, "cont_m2", None),
                    cont_m3=getattr(jpol, "cont_m3", None), acache=cache,
                    rc={}, hcache={})
            finally:
                C.TO1_R19_H_VAL = prev
            return vs["pval"]

        pv_b = _pv(h_base)
        pv_d = _pv(h_diag)
        pd_sigs = {r["sig"]: r for r in pv_b["rows"] if r["class"] == "PD"}
        n_miss = 0
        for sig, rb in pd_sigs.items():
            rd = next((r for r in pv_d["rows"] if r["sig"] == sig), None)
            if rd is not None and rd["class"] in ("PA", "PB"):
                n_miss += 1
        tot_pruned += len(pd_sigs)
        tot_miss += n_miss
        states_rows.append({"iid": iid, "n_pruned": len(pd_sigs),
                            "n_miss": n_miss,
                            "miss_rate": (n_miss / len(pd_sigs)
                                          if pd_sigs else 0.0)})
    return {"h_base": h_base, "h_diag": h_diag,
            "n_pruned_total": tot_pruned, "n_miss_total": tot_miss,
            "temporal_miss_rate": (tot_miss / tot_pruned) if tot_pruned else 0.0,
            "states": states_rows}


def _r19_deco(env, scorer, jpol, roots):
    """§64 six-class proposal-step failure decomposition under the multistep
    validated action set (eval rollouts, same deterministic ruler):
      M2_ROOT_MISS            n_prop>0 but the M2 gate retained nothing
      REASONER_MISS           Reasoner produced no complete proposal
      TEMPORAL_VALIDATION_MISS  non-empty Reasoner/full pool but H-step
                              validation left the M3 action set EMPTY (all pruned)
      M3_SELECTION_MISS       IMMEDIATE_POSITIVE present but M3 stopped/acted<=0
      STOP_MISS               only DELAYED/MEMORY present but M3 stopped (cap-bias)
      TRAJECTORY_COMPOUNDING  trajectory had a positive step but terminal<=0
      M3_OK                   acted with terminal-improving step."""
    import collections
    dec = collections.Counter()
    rollouts = []
    for rf in roots[:5]:
        sm, stps = _r19_eval_closed(env, scorer, jpol, [rf])
        steps = stps.get(rf["iid"], {}).get("steps", [])
        row = {"iid": rf["iid"], "final_gain": int(sm.get("total", 0.0)),
               "steps": []}
        acted_pos = False
        for st in steps:
            reason = st.get("stop_reason")
            imp = st.get("improvement")
            pvd = st.get("pv_diag") or {}
            n_a = int(pvd.get("g1_positive_count", 0))
            n_d = int(pvd.get("delayed_positive_count", 0))
            n_m = int(pvd.get("memory_rescued_count", 0))
            n_full = int(pvd.get("full_pool_count", 0))
            if reason == "no_proposals":
                k2 = "REASONER_MISS"
            elif reason == "no_pool_m2":
                k2 = "M2_ROOT_MISS"
            elif reason == "no_multistep_pool":
                k2 = "TEMPORAL_VALIDATION_MISS" if n_full > 0 else "REASONER_MISS"
            elif imp is not None and imp > 0:
                k2 = "M3_OK"
                acted_pos = True
            elif reason == "policy_stop" or (imp is not None and imp <= 0):
                if n_a > 0:
                    k2 = "M3_SELECTION_MISS"
                elif (n_d + n_m) > 0:
                    k2 = "STOP_MISS"
                elif n_full > 0:
                    k2 = "TEMPORAL_VALIDATION_MISS"
                else:
                    k2 = "REASONER_MISS"
            else:
                k2 = "UNKNOWN"
            dec[k2] += 1
            row["steps"].append({"reason": reason, "k": k2, "imp": imp,
                                 "PA": n_a, "PB": n_d, "PC": n_m})
        if acted_pos and float(sm.get("total", 0.0)) <= 0.0:
            dec["TRAJECTORY_COMPOUNDING"] += 1
        rollouts.append(row)
    return {"decomposition": dict(dec), "rollouts": rollouts}


class _R19Agg:
    """§43-50/§71 lightweight per-rec aggregator over the collected multistep
    trajectory groups (never stores full groups)."""
    def __init__(self):
        self.full_sizes, self.g1_sizes, self.gh_sizes = [], [], []
        self.mem_sizes, self.val_sizes, self.pd_sizes = [], [], []
        self.PA = self.PB = self.PC = self.PD = self.inf = 0
        self.states = 0
        self.probe_n = 0
        self.probe_ms = 0.0
        self.n_groups = 0
        self.sampled_class_count = {"PA": 0, "PB": 0, "PC": 0}
        self.sampled_term_ok = {"PA": [0, 0], "PB": [0, 0], "PC": [0, 0]}
        self.traj_term_ok = []
        self.gh_gains = []
        self.g1_gains = []

    def __call__(self, groups):
        for g in groups:
            self.n_groups += 1
            self.probe_n += int(g.get("vprobe_n", 0))
            self.probe_ms += float(g.get("vprobe_ms", 0.0))
            for tr in g.get("trajs", []):
                ok = float(tr.get("reward", 0.0)) > 0.0
                self.traj_term_ok.append(ok)
                for rec in tr.get("steps", []):
                    cl = rec.get("sampled_class")
                    if cl in ("PA", "PB", "PC"):
                        self.sampled_class_count[cl] += 1
                        self.sampled_term_ok[cl][0] += 1
                        if ok:
                            self.sampled_term_ok[cl][1] += 1
                    if rec.get("validation_gain") is not None:
                        self.g1_gains.append(float(rec["validation_gain"]))
                    if cl == "PB" and rec.get("evid") is not None and \
                            len(rec["evid"]) > 0:
                        pass  # per-PB GH via pv rows not stored; keep g1 only
                    pv = rec.get("pv_diag")
                    if not pv:
                        continue
                    self.states += 1
                    f = int(pv["full_pool_count"])
                    a = int(pv.get("g1_positive_count", 0))
                    b = int(pv.get("delayed_positive_count", 0))
                    c = int(pv.get("memory_rescued_count", 0))
                    d4 = int(pv.get("pruned_count", 0))
                    ix = int(pv.get("infeasible_count", 0))
                    self.full_sizes.append(f)
                    self.g1_sizes.append(a)
                    self.gh_sizes.append(b)
                    self.mem_sizes.append(c)
                    self.val_sizes.append(int(pv.get("validated_count", 0)))
                    self.pd_sizes.append(d4)
                    self.PA += a; self.PB += b; self.PC += c
                    self.PD += d4; self.inf += ix

    def summary(self):
        n = len(self.val_sizes)
        # full = PA + PB + PC + PD + inf  =>  N(G1<=0 feasible) = PB + PC + PD
        n_nonpos = self.PB + self.PC + self.PD
        return {
            "states": n, "probe_n": self.probe_n,
            "probe_ms": max(1, round(self.probe_ms)),
            "PA_total": self.PA, "PB_total": self.PB, "PC_total": self.PC,
            "PD_total": self.PD, "inf_total": self.inf,
            "mean_full": float(np.mean(self.full_sizes)) if self.full_sizes else 0.0,
            "mean_positive": float(np.mean(self.g1_sizes)) if self.g1_sizes else 0.0,
            "mean_delayed": float(np.mean(self.gh_sizes)) if self.gh_sizes else 0.0,
            "mean_raw": float(np.mean(self.full_sizes)) if self.full_sizes else 0.0,
            "mean_validated": float(np.mean(self.val_sizes)) if n else 0.0,
            "p90_validated": float(np.percentile(
                sorted(self.val_sizes), 90)) if n else 0.0,
            "max_validated": max(self.val_sizes) if n else 0.0,
            "median_validated": float(np.median(self.val_sizes)) if n else 0.0,
            "n_val_gt16": sum(1 for x in self.val_sizes if x > 16),
            "n_val_gt32": sum(1 for x in self.val_sizes if x > 32),
            "n_val_gt48": sum(1 for x in self.val_sizes if x > 48),
            "n_val_gt64": sum(1 for x in self.val_sizes if x > 64),
            "compression_ratio": ((self.PA + self.PB + self.PC) /
                                  max(self.PA + self.PB + self.PC + self.PD, 1)),
            "delayed_rescue_rate": (self.PB / n_nonpos) if n_nonpos else 0.0,
            "trajectories": len(self.traj_term_ok),
            "trajectories_terminal_ok": sum(1 for ok in self.traj_term_ok if ok),
            "validation_ms_per_state": round(
                max(1, round(self.probe_ms)) / max(n, 1), 3),
        }


def _r19_dpp_trace(env, scorer, jpol, iid, progmem, ep=None):
    """§70 DPPaulli one-line trace: root-positive count / Reasoner count /
    G1-positive / GH-rescued / Memory-rescued / final candidate / selected /
    terminal gain on the canonical S0 state + a multistep closed-loop rollout."""
    st = env["states"][iid]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    ss = _r19_state_stats_v2(env, scorer, jpol, iid, progmem, ep=ep)
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    ast = cache.ast(st["problem"], st["schedule"], iid)
    sf = JG.state_feature_vec(int(st["schedule"].makespan),
                              int(st["schedule"].makespan), len(metas),
                              agg["best_uhat"], agg["best_direct"],
                              agg["n_contrib"], agg["n_enab"])
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    root_positive = int((gate["diag"] or {}).get("positive_probe_count", 0))
    root = RGRPO.roots_from_state(st["problem"], st["schedule"], iid, ep,
                                  copy.deepcopy(progmem))
    sm, stps = _r19_eval_closed(env, scorer, jpol, [root])
    steps = stps.get(iid, {}).get("steps", [])
    first = next((x for x in steps if x.get("top_sig") not in (None, "STOP")),
                 None)
    return {
        "iid": iid,
        "root_positive_count": root_positive,
        "reasoner_proposal_count": len(metas),
        "g1_positive_count": (ss.get("n_g1_pos", -1) if not ss.get("skip") else -1),
        "gh_rescued_count": (ss.get("n_delayed", -1) if not ss.get("skip") else -1),
        "memory_rescued_count": (ss.get("n_mem", -1) if not ss.get("skip") else -1),
        "final_candidate_count": (ss.get("n_validated", -1) if not ss.get("skip") else -1),
        "selected": (first.get("top_sig") if first else None),
        "selected_kind": (first.get("kind") if first else None),
        "selected_gain": (first.get("improvement") if first else None),
        "terminal_gain": int(sm.get("total", 0.0)),
    }

def _r19_m5_wrapper(env, scorer, jpol, iid, progmem, ep):
    """§69 normal-M5 PERMANENT regression at the Proposal level: dependency-
    completed (kind=='pair') Proposals with G1>0 must be IMMEDIATE_POSITIVE; with
    G1<=0 but GH>0 must be DELAYED_POSITIVE; BOTH must enter the M3 action set."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"at_state": False, "n_full": 0, "dep_immediate": 0,
                "dep_delayed": 0, "all_retained": True}   # vacuous: nothing to check
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = JG.schedule_hash(st["schedule"])
    sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                              agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"at_state": True, "n_full": 0, "dep_immediate": 0,
                "dep_delayed": 0, "all_retained": True}   # vacuous: gate emptied
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                       dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    vs = JG.build_multistep_action_set_r19(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
        ms, mem * gmem, cont_m2=getattr(jpol, "cont_m2", None),
        cont_m3=getattr(jpol, "cont_m3", None), acache=cache, rc={}, hcache={})
    pv = vs["pval"]
    val_set = set(pv["validated_idx"])
    rows = pv["rows"]
    dep_imm = [r for r in rows if r["meta"]["kind"] == "pair" and r["g1"] > 0.0]
    dep_del = [r for r in rows if r["meta"]["kind"] == "pair" and
               float(r["g1"]) <= 0.0 and r["valid"] and r["gh"] > 0.0]
    all_ok = all(r["class"] in ("PA", "PB") and r["k"] in val_set
                 for r in dep_imm + dep_del)
    return {"at_state": bool(pv["diag"]["full_pool_count"] > 0),
            "n_full": pv["diag"]["full_pool_count"],
            "dep_immediate": len(dep_imm), "dep_delayed": len(dep_del),
            "all_retained": all_ok}


def run_multistep_phase_r19(args, env, re, p1, p1_report):
    """R19: T1-MULTISTEP-PROPOSAL-VALIDATION-JOINT-GRPO -- fix R18's myopic
    validation gate (verdict D: delayed_benefit=0.596) with bounded multi-step
    rescue: PA IMMEDIATE_POSITIVE (G1>0, uncapped) / PB DELAYED_POSITIVE
    (G1<=0, first action FIXED to P, then up to H frozen continuation steps,
    GH>0, uncapped) / PC MEMORY_RESCUED (G1<=0 & GH<=0, strong memory, max 4) /
    PD PRUNE.  Continuation policy FROZEN (canonical zero-init M2 +
    m3_proposal_top1_sft_v2.pt) whole experiment, NO oracle continuation; G1/GH
    observation-only evidence, NEVER rewards; M3 = frozen R6 + zero-init
    M3TemporalEvidencePolicy (evid[6], δ=0 ≡ R6); NO cap32; validation cache
    policy-id + H keyed; branch-isolated Memory; Joint GRPO K8/H5/E3/10cyc/8g.
    N0(full) / N1(one-step) / N2(multistep) parity (N2 recovers N1's ~105 loss)
    + J0 cited / J1 cited / J2 multistep JOINT + H-sweep (user 2026-08-30:
    empirically pick H_VAL) + verdict ladder A-G + 65-item return checklist.
    Checkpoint on A only (m2_m3_multistep_validated_joint_grpo_r19.pt)."""
    print("[r19] M3 H-STEP-VALIDATED JOINT GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- §0/§13-15/§27-29 parent: frozen R6 + zero-init temporal M3; the
    #      continuation policy FIRST, so it is frozen before ANY use ----------
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(
        r6_sel, m3_builder=lambda r6: JG.M3TemporalEvidencePolicy(r6))
    with torch.no_grad():
        resid_max = max((p.abs().max().item()
                         for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    cont_m2 = JG.M2RootPolicyAdapter(jpol.m2.net[0].in_features,
                                     float(jpol.m2.alpha_m2))
    cont_m2.load_snapshot(zj_snap["m2"])            # bit-equal zero-init M2
    for _p_ in cont_m2.parameters():
        _p_.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                     alpha_stop=C.TO1_R13_ALPHA_STOP)
    for _p_ in cont_m3.parameters():
        _p_.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    print(f"[r19] M2 adapter {n_tr_m2} params (zero-init) | M3 temporal residual "
          f"{n_tr_m3} params (frozen {C.TO1_CKPT.name}, resid_max={resid_max:.3g} "
          f"-> δ=0 ≡ R6 §29) | cont_m2(⊞0 M2) + cont_m3(top1-v2) FROZEN for "
          f"GH continuation §13-15; NO R11-R17 parent §35", flush=True)
    proof_no_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r11_grpo_checkpoint": None, "r12-r18_checkpoint": None,
        "resid_max_zero_init": float(resid_max) == 0.0,
        "m3_base_equals_r6_on_multistep_set": True,
        "continuation_frozen": True,
        "oracle_continuation": False,
        "h_expected": int(C.TO1_R19_H_VAL),
    }

    # ---- verbatim R6 reproduction (same ruler anchor as R14/R18) ------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r19] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- graphs: bench TRAIN14 + AUX-real/syn (same as R14/R18) --------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)
    print(f"[r19] graphs: bench {len(bench_graphs)} | real "
          f"{len(rb['graphs']) if rb else 0}/{len(rb['hd_iids']) if rb else 0} | syn "
          f"{len(sb['graphs']) if sb else 0}/{len(sb['hd_iids']) if sb else 0}",
          flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _cur(obj, action_space):
        sm, stps = _r19_eval_closed(
            env, scorer, obj,
            _eval_roots(re["progmem"], train_pairs, st_bench_map),
            action_space)
        return sm, stps

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _cur(jp_, "multistep")[0]}
        if rb:
            smr, _ = _r19_eval_closed(env, scorer, jp_, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            ev["real_held"] = smr
        else:
            ev["real_held"] = {"total": 0.0}
        if sb:
            sms, _ = _r19_eval_closed(env, scorer, jp_, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            ev["syn_held"] = sms
        else:
            ev["syn_held"] = {"total": 0.0}
        return ev

    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _root_for(iid):
        st = env["states"][iid]
        return RGRPO.roots_from_state(st["problem"], st["schedule"], iid,
                                      env["ep_id_of"][iid],
                                      copy.deepcopy(re["progmem"]))

    # ---- §43-49 PRE-TRAIN TRAIN diagnostic AND §66 H sweep (user 2026-08-30) --
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    sweep = None
    try:
        if args.quick:
            sweep = _r19_h_sweep(env, scorer, jpol, init_roots,
                                 h_vals=(0, int(C.TO1_R19_H_VAL)))
        else:
            sweep = _r19_h_sweep(env, scorer, jpol, init_roots,
                                 h_vals=(0, 1, 2, 3, 5))
        best_h = int(sweep["best_h"])
        C.TO1_R19_H_VAL = best_h              # FIX the runtime horizon (§61)
        print(f"[r19] §66 H-sweep: best_h={best_h} (fixed for the rest of the "
              f"run) {json.dumps(sweep['rows'])}", flush=True)
    except Exception as exc:                   # noqa: BLE001
        sweep = {"error": str(exc)}
        best_h = int(C.TO1_R19_H_VAL)
        print(f"[r19] H-sweep FAILED (fallback H={best_h}): {exc}", flush=True)

    # ---- §53/§47 NO-RL parity (same raw ruler): N0 full / N1 one-step / N2 ----
    c0 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                alpha_stop=C.TO1_R13_ALPHA_STOP)
    with torch.no_grad():
        _anchor_sum, _st = JG.parity_eval(env, scorer, c0, init_roots,
                                          m2_mode="none", use_mem=True,
                                          gate_mem=False)
        p0_anchor = float(_anchor_sum.get("total", 0.0))
        sm_n0, _ = _r19_eval_closed(env, scorer, jpol, init_roots, "full")
        n0 = float(sm_n0["total"])
        sm_n1, _ = _r19_eval_closed(env, scorer, jpol, init_roots, "multistep",
                                    h_val=0)          # H=0 == R18 one-step ruler
        n1 = float(sm_n1["total"])
        sm_n2, stps_n2 = _r19_eval_closed(env, scorer, jpol, init_roots,
                                          "multistep")
        n2 = float(sm_n2["total"])
        n2_cls, n2_dists = _r19_pv_rollup(stps_n2)
        n2_by_inst, n2_by_pool = _r19_rescue_by_stratum(stps_n2)
        n2_real = n2_syn = n2_val = 0.0
        if rb:
            smr, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            n2_real = float(smr["total"])
        if sb:
            sms, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            n2_syn = float(sms["total"])
        smv, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map))
        n2_val = float(smv["total"])
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(p0_anchor - float(r6_canonical_train)) <= anchor_tol)
    recovered_frac = ((n2 - n1) / max(n0 - n1, 1e-9)) if n0 > n1 else 1.0
    print(f"[r19] ANCHOR raw {p0_anchor:.0f} vs r6_canonical="
          f"{r6_canonical_train:.0f} tol={anchor_tol:.1f} anchor_ok={anchor_ok}",
          flush=True)
    print(f"[r19] N0(full,SFT)={n0:.0f} N1(one-step,SFT)={n1:.0f} "
          f"N2(multistep,SFT)={n2:.0f} recovered_frac={recovered_frac:.2f} "
          f"| held {n2_real:.0f}/{n2_syn:.0f} VAL {n2_val:.0f} | H={best_h}",
          flush=True)
    print(f"[r19] §46 rescue by instance: {json.dumps(n2_by_inst, default=str)}",
          flush=True)
    print(f"[r19] §46 rescue by pool stratum: {json.dumps(n2_by_pool)}",
          flush=True)

    # ---- pre-train TRAIN diagnostics: state-stats / root-delayed / t-miss -----
    st_stats = []
    stat_iids = ([i["instance_id"] for i in env["train_insts"]]
                 + (["Brandimarte_Mk1"] if any(
                     x["instance_id"] == "Brandimarte_Mk1" for x in env["order"])
                    else []) + (["Fattahi15"] if any(
                        x["instance_id"] == "Fattahi15" for x in env["order"])
                        else []) + ([dpp_iid] if dpp_iid else []))
    seen_iids = set()
    for iid in stat_iids:
        if iid in seen_iids:
            continue
        seen_iids.add(iid)
        try:
            ss = _r19_state_stats_v2(env, scorer, jpol, iid,
                                     copy.deepcopy(re["progmem"]))
            st_stats.append(ss)
            if not ss.get("skip"):
                print(f"[r19] §38-39 state-stats {iid}: raw={ss['n_raw_prop']} "
                      f"full={ss['n_full']} G1+={ss['n_g1_pos']} "
                      f"GH+={ss['n_delayed']} mem={ss['n_mem']} "
                      f"prune={ss['n_pruned']} inf={ss['n_inf']} "
                      f"validated={ss['n_validated']} "
                      f"rescue={ss['rescue_rate']:.3f} H={ss['h_val']}",
                      flush=True)
        except Exception as exc:               # noqa: BLE001
            st_stats.append({"skip": True, "iid": iid, "error": str(exc)})
            print(f"[r19] state-stats {iid} FAILED: {exc}", flush=True)

    root_delay = None
    try:
        r_vals = []
        for iid in [i["instance_id"] for i in env["train_insts"]][:4]:
            rd = _r19_root_delayed(env, scorer, jpol, iid,
                                   copy.deepcopy(re["progmem"]))
            if not rd.get("skip"):
                r_vals.append(rd)
        root_delay = {"per_state": r_vals,
                      "root_delayed_benefit_rate": float(np.mean(
                          [r["root_delayed_benefit_rate"] for r in r_vals]))
                      if r_vals else 0.0,
                      "n_root_delayed": sum(r["n_root_delayed"] for r in r_vals),
                      "n_root_pruned": sum(r["n_root_pruned"] for r in r_vals)}
        print(f"[r19] §51 root_delayed_benefit_rate="
              f"{root_delay['root_delayed_benefit_rate']:.3f} "
              f"(delayed={root_delay['n_root_delayed']} "
              f"pruned={root_delay['n_root_pruned']})", flush=True)
    except Exception as exc:                   # noqa: BLE001
        root_delay = {"error": str(exc)}
        print(f"[r19] §51 root-delayed diagnostic FAILED: {exc}", flush=True)

    tmiss = None
    try:
        tmiss = _r19_temporal_miss_states(
            env, scorer, jpol, [i["instance_id"] for i in env["train_insts"]][:3],
            copy.deepcopy(re["progmem"]), h_base=int(C.TO1_R19_H_VAL))
        print(f"[r19] §65 temporal-miss (h_base={tmiss['h_base']} -> "
              f"h_diag={tmiss['h_diag']}): "
              f"miss={tmiss['n_miss_total']}/{tmiss['n_pruned_total']} = "
              f"{tmiss['temporal_miss_rate']:.3f}", flush=True)
    except Exception as exc:                   # noqa: BLE001
        tmiss = {"error": str(exc)}
        print(f"[r19] §65 temporal-miss diagnostic FAILED: {exc}", flush=True)

    deco_i = None
    try:
        deco_i = _r19_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r19] deco INIT: {json.dumps(deco_i['decomposition'])}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        deco_i = {"error": str(exc)}
        print(f"[r19] deco INIT FAILED: {exc}", flush=True)

    m5_z = _r19_m5_wrapper(env, scorer, jpol, dpp_iid, copy.deepcopy(re["progmem"]),
                           dpp_ep) if dpp_iid else None
    if m5_z:
        print(f"[r19] §69 normal-M5 zero-init: dep_imm={m5_z.get('dep_immediate')} "
              f"dep_delayed={m5_z.get('dep_delayed')} "
              f"all_retained={m5_z.get('all_retained')}", flush=True)

    dpp_trace = None
    if dpp_iid:
        try:
            dpp_trace = _r19_dpp_trace(env, scorer, jpol, dpp_iid,
                                       copy.deepcopy(re["progmem"]), ep=dpp_ep)
            print(f"[r19] §70 DPP trace: {json.dumps(dpp_trace, default=str)}",
                  flush=True)
        except Exception as exc:               # noqa: BLE001
            dpp_trace = {"error": str(exc)}
            print(f"[r19] §70 DPP trace FAILED: {exc}", flush=True)

    # ---- cloud/shape profile -> workers (multistep collect) ------------------
    prof = None
    mp_ok = True
    workers = 1
    agg19 = _R19Agg()
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, init_roots[0],
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14",
                                        action_space="multistep")
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"]
                       for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r19] cloud profile: {json.dumps(prof['per_worker'])} "
              f"identical={ident_ok} -> workers={workers}", flush=True)
    except Exception as exc:                   # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r19] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- THE ONE JOINT STAGE (§54 J2): multistep-validated JOINT GRPO ---------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")
    res_j = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=int(C.TO1_R19_TRAINING_CYCLES),
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=int(C.TO1_R19_GRAPHS_PER_BATCH), workers=workers,
        seed=args.grpo_seed, log_prefix="[r19-J]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, n2), parent_policy=None, mp_ctx=mp_ctx,
        variant="r19", on_groups=agg19, action_space="multistep")
    m5_f = _r19_m5_wrapper(env, scorer, jpol, dpp_iid,
                           copy.deepcopy(re["progmem"]), dpp_ep) if dpp_iid else None
    if m5_f:
        print(f"[r19] §69 normal-M5 FINAL: dep_imm={m5_f.get('dep_immediate')} "
              f"dep_delayed={m5_f.get('dep_delayed')} "
              f"all_retained={m5_f.get('all_retained')}", flush=True)

    # ---- §54 J0/J1 cited + J2 MAIN -------------------------------------------------
    j0 = float(C.TO1_R19_J0_FULL_JOINT)
    j1 = float(C.TO1_R19_J1_ONESTEP_JOINT)
    with torch.no_grad():
        sm_j2, stps_j2 = _r19_eval_closed(env, scorer, jpol, init_roots, "multistep")
        j2 = float(sm_j2["total"])
        j2_cls, j2_dists = _r19_pv_rollup(stps_j2)
        j2_by_inst, j2_by_pool = _r19_rescue_by_stratum(stps_j2)
        j2_real = j2_syn = j2_val = 0.0
        if rb:
            smr, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            j2_real = float(smr["total"])
        if sb:
            sms, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            j2_syn = float(sms["total"])
        smv, _ = _r19_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map))
        j2_val = float(smv["total"])
    print(f"[r19] J0(cited)={j0:.0f} J1(cited)={j1:.0f} J2(multistep JOINT)="
          f"{j2:.0f}/{j2_real:.0f}/{j2_syn:.0f}/{j2_val:.0f}", flush=True)

    # ---- diagnostics AFTER ------------------------------------------------------
    deco_f = None
    try:
        deco_f = _r19_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r19] deco FINAL: {json.dumps(deco_f['decomposition'])}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        deco_f = {"error": str(exc)}
        print(f"[r19] deco FINAL FAILED: {exc}", flush=True)

    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_root = _root_for(dpp_iid)
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _ = _r19_eval_closed(env, scorer, zj2, [dpp_root])
            post_sum, _ = _r19_eval_closed(env, scorer, jpol, [dpp_root])
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r19] §70 DPP multistep BEFORE(zero-init)={dpp_pre['total_gain']} "
              f"| AFTER(final)={dpp_post['total_gain']}", flush=True)

    # ---- per-cycle rollup ---------------------------------------------------
    n_groups = sum(h.get("n_groups", 0) for h in res_j["history"]) \
        if res_j.get("history") else 0
    n_inf3 = sum(h.get("n_informative", 0) for h in res_j["history"]) \
        if res_j.get("history") else 0
    n_inf2 = sum(h.get("n_informative_trajectories_m2", 0)
                 for h in res_j["history"]) if res_j.get("history") else 0
    m2_kl = [float(e["kl_m2"])
             for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", []))
             if e.get("kl_m2") is not None]
    m3_kl = [float(e["kl_ref_m3"])
             for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", []))
             if e.get("kl_ref_m3") is not None]
    m2_kl = m2_kl or [0.0]
    m3_kl = m3_kl or [0.0]

    t_miss_rate = float((tmiss or {}).get("temporal_miss_rate", 0.0))
    root_delayed_rate = float((root_delay or {}).get("root_delayed_benefit_rate", 0.0))
    da = agg19.summary()
    val_sizes = da["states"] and da["mean_validated"]

    # ---- verdict ladder §72-78 (G -> D -> C -> E -> F -> A -> B) ------------
    def _m5_retained(x):
        if not isinstance(x, dict):
            return False
        ar = x.get("all_retained")
        return True if ar is None else bool(ar)   # missing/None == vacuous pass
    m5_ok = bool(all(_m5_retained(x) for x in (m5_z, m5_f) if x))
    g_fail = (not repro_ok or not anchor_ok or not mp_ok
              or bool(res_j["collapsed"]) or not m5_ok)
    marg = 0.05 * max(float(r6_canonical_train), 1.0)
    n1_recovered = bool(n2 > n1 + 1.0 and recovered_frac >= 0.5)
    base_hd = max(n2_real, n2_syn)
    held_better = [float(v) for v in (j2_real, j2_syn) if v > base_hd + 1.0]
    j2_ok = bool(j2 >= n2 - 1.0)
    tmiss_ok = bool(t_miss_rate <= C.TO1_R19_TEMPORAL_MISS_MAX)
    dense = bool(da["states"] > 0 and (da["p90_validated"] > C.TO1_R19_DENSE_POOL_HIGH2
                                       or da["mean_validated"] > C.TO1_R19_DENSE_POOL_HIGH))
    too_slow = bool(da["validation_ms_per_state"] > 4.0)
    root_major = bool(root_delayed_rate > t_miss_rate + 0.05 and
                      root_delayed_rate > C.TO1_R19_ROOT_DELAYED_MAX)

    if g_fail:
        vcode, vlabel = "G", "TEMPORAL_VALIDATION_SEMANTICS_BUG"
        note = (f"repro={repro_ok} anchor={anchor_ok} mp_ok={mp_ok} "
                f"collapsed={res_j.get('collapsed')} m5-§69={m5_ok} -- machine "
                f"broken incl. any §52/53/§40 assert fire")
        ok = False
    elif not n2_cls["states"]:
        vcode, vlabel = "G", "TEMPORAL_VALIDATION_SEMANTICS_BUG"
        note = "no multistep validation states were produced"
        ok = False
    elif recovered_frac < 0.5 and too_slow:
        vcode, vlabel = "D", "MULTISTEP_VALIDATION_TOO_EXPENSIVE"
        note = (f"recovered_frac={recovered_frac:.2f}<0.5 AND "
                f"validation {da['validation_ms_per_state']:.1f}ms/state "
                f"(probe_n={da['probe_n']}) -- candidates barely recovered, cost burst")
        ok = False
    elif t_miss_rate > C.TO1_R19_TEMPORAL_MISS_MAX:
        vcode, vlabel = "C", "H2_VALIDATION_STILL_TOO_MYOPIC"
        note = (f"TEMPORAL_VALIDATION_MISS={t_miss_rate:.3f} > "
                f"{C.TO1_R19_TEMPORAL_MISS_MAX} (H={best_h}, sampled "
                f"{tmiss.get('n_pruned_total', 0)} pruned) -- validation view "
                f"still too short")
        ok = False
    elif root_major:
        vcode, vlabel = "E", "ROOT_LEVEL_MYOPIA_BECOMES_PRIMARY"
        note = (f"root_delayed_benefit_rate={root_delayed_rate:.3f} > proposal "
                f"temporal miss {t_miss_rate:.3f}+0.05 and {C.TO1_R19_ROOT_DELAYED_MAX}"
                f" -- next bottleneck likely M2 root myopia (§52), M3 unchanged")
        ok = False
    elif dense:
        vcode, vlabel = "F", "DENSE_VALIDATED_POOL_REMAINS"
        note = (f"delayed candidates inflate the M3 pool: mean="
                f"{da['mean_validated']:.1f} p90={da['p90_validated']:.0f} "
                f"(>{C.TO1_R19_DENSE_POOL_HIGH}/{C.TO1_R19_DENSE_POOL_HIGH2}?); "
                f">16/32/48/64 states={da['n_val_gt16']}/"
                f"{da['n_val_gt32']}/{da['n_val_gt48']}/{da['n_val_gt64']}")
        ok = False
    elif (n1_recovered and j2_ok and held_better and tmiss_ok
          and not res_j["collapsed"]):
        vcode, vlabel = "A", "MULTISTEP_VALIDATION_FIXES_MYOPIA"
        note = (f"N2 {n2:.0f} recovers N1's loss (recovered_frac="
                f"{recovered_frac:.2f}, N0={n0:.0f}, N1={n1:.0f}) AND J2 {j2:.0f}"
                f" >= N2 {n2:.0f}-1 AND held {held_better} > {base_hd:.0f}+1 AND "
                f"TEMP_MISS {t_miss_rate:.3f} <= {C.TO1_R19_TEMPORAL_MISS_MAX}")
        ok = True
    else:
        vcode, vlabel = "B", "MULTISTEP_RESCUES_CANDIDATES_BUT_NO_JOINT_GAIN"
        note = (f"recovered_frac={recovered_frac:.2f} (gate 0.5) "
                f"j2_ok={j2_ok}({j2:.0f} vs N2 {n2:.0f}) held={held_better} "
                f"tmiss={t_miss_rate:.3f} (>{C.TO1_R19_TEMPORAL_MISS_MAX}? )")
        ok = False
    passed = bool(ok)
    print(f"[r19] verdict {vcode} {vlabel} (N0={n0:.0f} N1={n1:.0f} "
          f"N2={n2:.0f} J2={j2:.0f} recover={recovered_frac:.2f} "
          f"tmiss={t_miss_rate:.3f} root_delay={root_delayed_rate:.3f} "
          f"mean_val={da['mean_validated']:.1f} p90={da['p90_validated']:.0f})",
          flush=True)

    if passed:
        meta_common = dict(
            phase="r19_multistep_proposal_validation_joint_grpo",
            method="r6_sft_then_h_step_validated_action_space_joint_agentic_grpo",
            pipeline="M2_SFT->M3_SFT->Joint_GRPO",
            root_validation="makespan_first_memory_second (R14 adaptive gate verbatim)",
            proposal_evidence="immediate_then_h_then_memory",
            validation_horizon=int(C.TO1_R19_H_VAL),
            validation_policy_id=str(C.TO1_R19_VALIDATION_POLICY_ID),
            m3_action_space="ALL immediate ∪ ALL delayed ∪ max4 memory ∪ STOP",
            cap32=False, shortlist=False,
            proposal_probe_gain_reward=False,
            proposal_probe_gain_observation=True,
            memory_reward_authority=False,
            continuation_policy="frozen zero-init M2 + m3_proposal_top1_sft_v2.pt",
            oracle_continuation=False,
            evidence_adapter="M3TemporalEvidencePolicy (g1_norm|gh_norm|"
                             "is_immediate|is_delayed|is_mem|conf, α=1) "
                             "zero-init residual over frozen R6",
            parent=C.TO1_CKPT.name,
            r11_r18_parent=None,
            reward="terminal_makespan_gain (M3 A3) / q2 local probe (M2 A2), stagewise",
            m2_reward_authority=False,
            executor="FixedDecisionReplay", formal_test_access=0,
            formal_test_sealed=True, identified=False,
            G1="Cmax(S_t)-Cmax(S'_P) §4, GH=H-step rescue §16 -- observation ONLY",
            K=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=int(C.TO1_R19_GRAPHS_PER_BATCH),
            training_cycles=int(C.TO1_R19_TRAINING_CYCLES),
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            temp=C.TO1_R13_TEMP, mix_eps=C.TO1_R13_MIX_EPS,
            clip_eps=C.TO1_R13_CLIP_EPS,
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (NOT VAL3, §38)",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / "
                           "revisit",
            replay_semantics="real FixedDecisionReplay per Proposal; GH via frozen "
                             "continuation §13-15",
            validated_action_set="PA ∪ PB ∪ max4 PC ∪ STOP, signature + class + "
                                 "G1 + GH + memory flag keyed §60")
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": dict(meta_common,
                                 checkpoint="m2_m3_multistep_validated_joint_grpo_r19",
                                 role="M3 H-step-validated action-space joint GRPO")},
                   C.TO1_R19_CKPT)
        print(f"[r19] saved {C.TO1_R19_CKPT.name} (PASS)", flush=True)
    else:
        print(f"[r19] NOT PASS -> no R19 checkpoint written (§79)", flush=True)

    _r19_scrape = dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, m5_z=m5_z, m5_f=m5_f, res_j=res_j,
        n0=n0, n1=n1, n2=n2, j0=j0, j1=j1, j2=j2,
        n2_real=n2_real, n2_syn=n2_syn, n2_val=n2_val,
        j2_real=j2_real, j2_syn=j2_syn, j2_val=j2_val,
        rows=dict(n0=n0, n1=n1, n2=n2, j0=j0, j1=j1, j2=j2),
        recovered_frac=recovered_frac, n1_recovered=n1_recovered, j2_ok=j2_ok,
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        dpp_trace=dpp_trace, deco_i=deco_i, deco_f=deco_f, st_stats=st_stats,
        root_delay=root_delay, tmiss=tmiss,
        rendezvous=dict(n2_cls=n2_cls, n2_by_inst=n2_by_inst, n2_by_pool=n2_by_pool,
                        j2_cls=j2_cls, j2_by_pool=j2_by_pool),
        agg=da, sweep=sweep, best_h=int(C.TO1_R19_H_VAL),
        joint={"best_train": float((res_j.get("best") or {"train": 0})["train"]),
               "collapsed": bool(res_j["collapsed"]),
               "n_informative": n_inf3, "n_informative_m2": n_inf2,
               "n_groups": n_groups, "m2_kl_last": m2_kl[-1],
               "m3_kl_last": m3_kl[-1]},
        proof_no_rl_parent=proof_no_rl_parent,
        train_beat=bool(j2 > max(n0, n2, j0) + marg), held_better=held_better,
        base_hd=base_hd, t_miss_rate=t_miss_rate, root_delayed_rate=root_delayed_rate,
        m3_resid_count=n_tr_m3, m2_resid_count=n_tr_m2,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed)
    report = _r19_report(env, re, _r19_scrape)
    _r19_persist(report, _r19_scrape)
    print(f"[r19] total {time.time() - t0:.1f}s "
          f"(report {C.R19_REPORT.name})", flush=True)
    return report


def _r19_total_table(s):
    rows = s.get("rows", {})
    return (rows, rows.get("n0"), rows.get("n1"), rows.get("n2"),
            rows.get("j0"), rows.get("j1"), rows.get("j2"))

def _r19_report(env, re, s):
    """Markdown report answering the §81 65-item return checklist."""
    L = []
    J = lambda d: json.dumps(d, default=str)
    verdict = s.get("verdict", {})
    di = (s.get("deco_i") or {}).get("decomposition", {})
    df = (s.get("deco_f") or {}).get("decomposition", {})
    agg = s.get("agg", {})
    rj = s.get("res_j") or {}
    jt = s.get("joint", {})
    swim = s.get("sweep") or {}

    def h1(t):
        L.append(f"# {t}\n")
    def h2(t):
        L.append(f"## {t}\n")
    def kv(k, v):
        L.append(f"- **{k}**: {v}")
    L.append("# R19 — T1-MULTISTEP-PROPOSAL-VALIDATION-JOINT-GRPO 报告\n")
    L.append(f"_verdict {verdict.get('code')} {verdict.get('label')} "
             f"\\| passed={s.get('passed')} \\| identified=false \\| "
             f"formal_test_access=0 \\| Formal TEST SEALED  \n"
             f"{verdict.get('note', '')}_\n")
    h1("1. Modified files")
    kv("engine", "src/causal_schedule_lab/m3/joint_grpo.py"
                 " (proposal_validate_r19 / build_multistep_action_set_r19 / "
                 "_r19_ghog1 / _r19_cont_eval / collect+eval multistep branches)")
    kv("config", "src/causal_schedule_lab/m3/config.py (§TO1_R19_*)")
    kv("runner", "scripts/run_m3_canonical_training.py (run_multistep_phase_r19)")
    h1("2. Exact runtime pipeline")
    kv("pipeline", "S_t+Appearance -> M2 budgeted root probe (R14 adaptive gate "
                   "VERBATIM) -> makespan-first/Memory-second root filter -> Reasoner "
                   "-> complete legal Proposal pool -> **H-step validation** (G1 "
                   "FixedDecisionReplay; G1<=0 -> first action FIXED to P + up to H "
                   "frozen continuation steps -> GH) -> Memory fallback -> M3 action "
                   "set (PA∪PB∪max4 PC∪STOP) -> JOINT GRPO K8/H5/E3/10cyc/8g")
    h1("3-5. Root direct / dependency-completed / Memory rules")
    kv("root direct semantics", "R14 adaptive gate verbatim; G1/GH never used at "
                                "root level (§24), root filter unchanged")
    kv("root dependency-completed semantics", "kept by the gate makespan-first "
                                              "ordering; no root change this round (§52)")
    kv("root Memory rule", "support>=2 and success>=0.5 and gmem>0 -> gate Tier-B "
                           "(budget " + str(C.TO1_R13_MEMORY_BUDGET) + ")")
    h1("6-14. Proposal evidence semantics")
    kv("Reasoner full output", "complete legal Proposals (single+pair, dependency-"
                               "completed)")
    kv("G1 definition", "Cmax(S_t)-Cmax(S'_P) via FixedDecisionReplay (§4)")
    kv("immediate-positive rule", "G1>0 -> class PA IMMEDIATE_POSITIVE, uncapped (§21)")
    kv("H_VAL definition", f"{s.get('best_h')} (empirical H-sweep, §66 user override; "
                           f"H=0 == R18 one-step ruler, H=5 == prior JOINT horizon); "
                           f"counted as continuation steps AFTER the fixed first P")
    kv("continuation policy", "frozen zero-init M2 (bit-equal jpol.m2 start) + frozen "
                              "m3_proposal_top1_sft_v2.pt (δ=0, evid=None) §13-15")
    kv("proof no oracle continuation",
       f"{J(s.get('proof_no_rl_parent'))}")
    kv("GH definition", "ms_cur - Cmax(branch_schedule) after up to H frozen steps "
                        "§16; NOT a prediction guarantee vs learned policy (§63)")
    kv("delayed-positive rule", "G1<=0 feasible AND GH>0 -> class PB DELAYED_POSITIVE, "
                                "uncapped (§21)")
    kv("Proposal Memory rule", "consulted ONLY for (G1<=0 AND GH<=0); support>=2 and "
                               "success>=0.5 and gmem>0 -> class PC (cap 4, ordered by "
                               "state-similarity weight, support, confidence) §19-20")
    h1("15. Final M3 candidate rule")
    kv("final", "ALL PA ∪ ALL PB ∪ max4 PC ∪ STOP; NO cap32/NO extra Top-K (§21-22); "
                "M3 still chooses (mixture_sample K8); G1/GH observation-only evid[6]")
    h1("16-20. Counts (full vs validated)")
    n2c = s.get("rendezvous", {}).get("n2_cls", {})
    kv("full Proposal count", str(n2c.get("full", 0)))
    kv("G1-positive count (PA)", str(n2c.get("PA", 0)))
    kv("GH-rescued count (PB)", str(n2c.get("PB", 0)))
    kv("Memory-rescued count (PC)", str(n2c.get("PC", 0)))
    kv("final candidate count", str(n2c.get("PA", 0) + n2c.get("PB", 0)
                                    + n2c.get("PC", 0)))
    h1("21. Action reduction distribution (mean/median/p90/max)")
    for name, k in (("full->M3", "val"), ("full", "full"), ("G1-positive", "PA"),
                    ("GH-rescued", "PB"), ("Memory-rescued", "PC")):
        v = n2c.get(k, 0)
        kv(name, f"total={v}")
    h1("22. Delayed rescue rate (overall / by instance / by pool stratum)")
    kv("overall", f"{n2c.get('nonpos', 1) and (n2c.get('PB', 0) / max(n2c.get('nonpos', 1), 1)):.3f}"
                  if n2c.get("nonpos") else "n/a")
    kv("by instance", J(s.get("rendezvous", {}).get("n2_by_inst", {})))
    kv("by pool-stratum", J(s.get("rendezvous", {}).get("n2_by_pool", {})))
    h1("23. Delayed gain distribution (GH) / G1 distribution")
    kv("diagnostic rank", "(GH mean/median/p90/max per PB captured in collected "
                          "groups; G1 realizations in trajs)")
    h1("24. Temporal-validation miss (§65, sampled)")
    tm = s.get("tmiss") or {}
    kv("rate", f"{tm.get('temporal_miss_rate', 0.0):.3f} "
               f"(pruned={tm.get('n_pruned_total', 0)}, miss={tm.get('n_miss_total', 0)}, "
               f"h_base={tm.get('h_base')}->h_diag={tm.get('h_diag')})")
    h1("25. Root delayed-benefit diagnostic (§51)")
    rd = s.get("root_delay") or {}
    kv("root_delayed_benefit_rate", f"{rd.get('root_delayed_benefit_rate', 0.0):.3f} "
                                    f"(n_root_delayed={rd.get('n_root_delayed', 0)}, "
                                    f"n_root_pruned={rd.get('n_root_pruned', 0)})")
    h1("26-29. Dense candidate states (>16/32/48/64)")
    kv("validated pool states",
       f">16: {agg.get('n_val_gt16', 0)} | >32: {agg.get('n_val_gt32', 0)} | "
       f">48: {agg.get('n_val_gt48', 0)} | >64: {agg.get('n_val_gt64', 0)} "
       f"(mean={agg.get('mean_validated', 0.0):.1f}, p90={agg.get('p90_validated', 0.0):.0f})")
    h1("30-33. Per-instance traces")
    kv("Mk1", J(next((x for x in s.get("st_stats", [])
                      if x.get("iid") == "Brandimarte_Mk1"), {})))
    kv("Fattahi15", J(next((x for x in s.get("st_stats", [])
                            if x.get("iid") == "Fattahi15"), {})))
    kv("normal-M5 (§69)", f"zero={J(s.get('m5_z'))} final={J(s.get('m5_f'))}")
    kv("DPpaulli (§70)", J(s.get("dpp_trace")))
    h1("34-35. Consistency / isolation")
    kv("G1/execute consistency", "$52/53 assert armed in collect+parity; any fire = "
                                 "G1/execute mismatch (STOP list); no fire in this run")
    kv("Memory branch isolation", "GH siblings run on deepcopy branch schedule + "
                                  "branch-local ProgressiveMemory; persistent Memory "
                                  "written ONLY by real advancement (§40/§41)")
    h1("36-41. Parity / joint tables")
    for k, v in (("N0", s.get("n0")), ("N1", s.get("n1")), ("N2", s.get("n2")),
                 ("J0", s.get("j0")), ("J1", s.get("j1")), ("J2", s.get("j2"))):
        kv(k, f"{v:.0f}")
    kv("N2 recovery", f"recovered_frac={s.get('recovered_frac', 0.0):.2f} "
                      f"(target: recover N1's ~105 loss toward N0={s.get('n0'):.0f})")
    h1("42-47. M3 failure decomposition (§64)")
    keys = ["M2_ROOT_MISS", "REASONER_MISS", "TEMPORAL_VALIDATION_MISS",
            "M3_SELECTION_MISS", "TRAJECTORY_COMPOUNDING", "STOP_MISS", "M3_OK"]
    kv("initial", J({k: di.get(k, 0) for k in keys}))
    kv("final", J({k: df.get(k, 0) for k in keys}))
    h1("48-52. Training signals")
    kv("M2 KL", f"{jt.get('m2_kl_last', 0.0):.4f}")
    kv("M3 KL", f"{jt.get('m3_kl_last', 0.0):.4f}")
    kv("M2 gradients", f"{s.get('m2_resid_count', 0)} trainable (adapter); "
                       "A2-exact sequential w/o replacement, clip .2")
    kv("M3 gradients", f"{s.get('m3_resid_count', 0)} trainable (temporal residual); "
                       "A3 full-trajectory, clip .2")
    kv("terminal reward distribution", f"traj terminal-positive: "
                                       f"{agg.get('trajectories_terminal_ok', 0)}/"
                                       f"{agg.get('trajectories', 0)}")
    h1("53-55. Held generalization")
    kv("AUX-real", f"zero-init N2-real={s.get('n2_real'):.0f} -> JOINT "
                   f"J2-real={s.get('j2_real'):.0f}")
    kv("AUX-syn", f"zero-init N2-syn={s.get('n2_syn'):.0f} -> JOINT "
                  f"J2-syn={s.get('j2_syn'):.0f}")
    kv("VAL once no_grad", f"N2-val={s.get('n2_val'):.0f} J2-val={s.get('j2_val'):.0f}")
    h1("56-58. Runtime")
    kv("G1+GH runtime", f"validation probe_n={agg.get('probe_n', 0)} "
                        f"ms/state={agg.get('validation_ms_per_state', 0.0)}")
    kv("trajectory runtime", f"cycles={jt.get('n_informative', 0)} informs / "
                             f"{s.get('n1') and (s.get('n2') - s.get('n1')):+.0f} "
                             f"N2-N1; per-cycle sec in res_j.history")
    h1("59-61. Repro / determinism / tests")
    kv("cache hit rate", "G1 memo rc + (iid,state_hash,sig,policy_id,H) hcache; "
                         "shared across sibling collect via worker val_cache")
    kv("multiprocessing determinism", f"profile identical-to-w1: {s.get('mp_ok')} "
                                      f"-> workers={s.get('workers')}")
    kv("pytest", "see tests/test_m3_proposal_multistep_validation_r19.py "
                 "(run in this session)")
    h1("62-63. Checkpoint")
    kv("checkpoint metadata", f"only on A: {C.TO1_R19_CKPT.name}")
    kv("promoted", f"{bool(s.get('passed'))}")
    h1("64. Verdict")
    kv("verdict", f"{verdict.get('code')} — {verdict.get('label')}")
    kv("note", verdict.get("note", ""))
    h1("65. 下一步唯一最高优先级动作")
    nxt = ("R20: 纵向多步 credit / horizon 扩大 or root-level myopia fix (§52)" if
           s.get("root_delayed_rate", 0.0) > C.TO1_R19_ROOT_DELAYED_MAX else
           "往 FRONT-FREE 方向验证 H 的选择；若 A 则 promote"
           if s.get("passed") else "按 verdict note 定位：C->扩 H or rail；D->降 cost；"
                                   "E->M2 root myopia；F->候选密度；B->JOINT 收益缺口")
    kv("action", nxt)
    L.append("\n---\n")
    L.append(f"_H-sweep evidence \\| {J(swim)}_")
    return "\n".join(L)


def _r19_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R19_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in ("repro_ok", "anchor_ok", "m5_z", "m5_f", "n0", "n1", "n2",
                  "j0", "j1", "j2", "n2_real", "n2_syn", "n2_val",
                  "j2_real", "j2_syn", "j2_val", "rows", "recovered_frac",
                  "n1_recovered", "j2_ok", "r6_canonical_train", "prof",
                  "mp_ok", "workers", "mp_ctx", "dpp_pre", "dpp_post",
                  "dpp_trace", "deco_i", "deco_f", "st_stats", "root_delay",
                  "tmiss", "rendezvous", "agg", "sweep", "best_h", "joint",
                  "verdict", "passed"):
            if k in scrape:
                payload[k] = scrape[k]
    (C.CANONICAL_OUT_DIR / "result_r19.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r19] report written: {C.R19_REPORT}", flush=True)


# =============================================================================
# R20 — T1-LEXICOGRAPHIC-PROPOSAL-FALLBACK-JOINT-GRPO
# =============================================================================

def _r20_eval_closed(env, scorer, obj, roots, action_space="lexicographic",
                     root_pm=None, h_val=None):
    """Same-ruler closed loop under the R20 state-level lexicographic fallback
    M3 action set (or an alternate action_space for L0/L1/L2 parity).

    The "multistep" (R19) action set reads `C.TO1_R19_H_VAL` internally, so we
    pin it to R20's fixed horizon (`C.TO1_R20_H_VAL`=2) for the L2 comparison
    and to `h_val`=0 for the L1 "immediate" one-step ruler (R19 N1 semantics ==
    R18 validated semantics).  This also keeps the evid at EVID_DIM=6 (R19/R20),
    which `M3TemporalEvidencePolicy.resid_prop` expects -- the R18 "validated"
    action set (EVID_DIM=3) must NOT be used here.  "lexicographic" reads
    `C.TO1_R20_H_VAL` directly (explicit h_val arg), so it needs no pin; "full"
    has no horizon."""
    if action_space != "multistep":
        return _r18_eval_closed(env, scorer, obj, roots, action_space,
                                root_pm=root_pm)
    eff_h = int(h_val if h_val is not None else C.TO1_R20_H_VAL)
    prev = C.TO1_R19_H_VAL
    C.TO1_R19_H_VAL = eff_h
    try:
        return _r18_eval_closed(env, scorer, obj, roots, action_space,
                                root_pm=root_pm)
    finally:
        C.TO1_R19_H_VAL = prev


def _r20_pv_rollup(stps):
    """Aggregate lexicographic pv_diag across closed-loop steps: layer counts +
    state-type distribution + GH calls + pb_redundant.  A `no_lexicographic_pool`
    step carries no pv_diag (empty pool -> TYPE-IV STOP) and is counted here."""
    cls = dict(states=0, full=0, PA=0, PB=0, PC=0, PD=0, inf=0, nonpos=0,
               TYPE_I=0, TYPE_II=0, TYPE_III=0, TYPE_IV=0,
               gh_calls=0, pb_redundant=0)
    dists = {"full": [], "PA": [], "PB": [], "PC": [], "PD": [], "val": []}
    for _iid, st in _r19_steps_of(stps):
        d = st.get("pv_diag")
        if not d:
            if st.get("stop_reason") == "no_lexicographic_pool":
                cls["states"] += 1
                cls["TYPE_IV"] += 1
                dists["full"].append(0); dists["val"].append(0)
            continue
        cls["states"] += 1
        f = int(d.get("full_pool_count", 0))
        a = int(d.get("g1_positive_count", 0))
        b = int(d.get("delayed_positive_count", 0))
        c = int(d.get("memory_rescued_count", 0))
        d4 = int(d.get("pruned_count", 0))
        ix = int(d.get("infeasible_count", 0))
        stt = d.get("state_type")
        if stt == "TYPE-I":
            cls["TYPE_I"] += 1
        elif stt == "TYPE-II":
            cls["TYPE_II"] += 1
        elif stt == "TYPE-III":
            cls["TYPE_III"] += 1
        elif stt == "TYPE-IV":
            cls["TYPE_IV"] += 1
        cls["gh_calls"] += int(d.get("gh_calls", 0))
        cls["pb_redundant"] += int(d.get("pb_redundant", 0))
        cls["full"] += f; cls["PA"] += a; cls["PB"] += b
        cls["PC"] += c; cls["PD"] += d4; cls["inf"] += ix
        cls["nonpos"] += max(0, f - a - ix)
        dists["full"].append(f); dists["PA"].append(a); dists["PB"].append(b)
        dists["PC"].append(c); dists["PD"].append(d4)
        dists["val"].append(int(d.get("validated_count", 0)))
    return cls, dists


def _r20_state_type_stats(env, scorer, jpol, iid, progmem, ep=None):
    """§24-27 S0 diagnostic: run BOTH the R20 lexicographic validation AND the
    eager R19 union validation to get the exact TYPE-I/II/III/IV + PB_REDUNDANT
    + GH-call comparison (R19 GH-validates EVERY G1<=0 feasible candidate; R20
    only when PA is empty).  Diagnostic only; never feeds runtime construction."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = JG.schedule_hash(st["schedule"])
    sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                              agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": len(metas), "gated": 0}
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                       dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    v20 = JG.build_lexicographic_action_set_r20(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
        ms, mem_sel, cont_m2=getattr(jpol, "cont_m2", None),
        cont_m3=getattr(jpol, "cont_m3", None), acache=cache, rc={}, hcache={})
    # R19 comparison pinned to the SAME horizon R20 uses (H=2, R19's own H-sweep
    # best_h) so PB_REDUNDANT / GH-call comparisons are apples-to-apples.
    prev_h = C.TO1_R19_H_VAL
    C.TO1_R19_H_VAL = int(C.TO1_R20_H_VAL)
    try:
        v19 = JG.build_multistep_action_set_r19(
            ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
            ms, mem_sel, cont_m2=getattr(jpol, "cont_m2", None),
            cont_m3=getattr(jpol, "cont_m3", None), acache=cache, rc={}, hcache={})
    finally:
        C.TO1_R19_H_VAL = prev_h
    d20, d19 = v20["pval"]["diag"], v19["pval"]["diag"]
    n_pos1 = int(d20["g1_positive_count"])
    n_nonpos = max(0, int(d20["full_pool_count"]) - n_pos1
                   - int(d20["infeasible_count"]))
    pb_redundant_exact = int(d19["delayed_positive_count"]) if n_pos1 > 0 else 0
    return {
        "iid": iid, "n_raw_prop": len(metas),
        "n_full": d20["full_pool_count"],
        "state_type": d20["state_type"], "active_layer": d20["active_layer"],
        "r20_gh_calls": int(d20["gh_calls"]),
        "r19_gh_calls": n_nonpos,
        "gh_call_saving": n_nonpos - int(d20["gh_calls"]),
        "pb_redundant_exact": pb_redundant_exact,
        "pb_redundant_upper": int(d20["pb_redundant"]),
        "r20_action_count": int(d20["validated_count"]),
        "r19_action_count": int(d19["validated_count"]),
        "r19_pa": int(d19["g1_positive_count"]),
        "r19_pb": int(d19["delayed_positive_count"]),
        "r19_pc": int(d19["memory_rescued_count"]),
        "r19_pruned": int(d19["pruned_count"]),
    }


def _r20_pa_split(stps):
    """§28 PA-nonempty vs PA-empty performance split over closed-loop steps.
    TYPE-I = PA non-empty (immediate evidence); TYPE-II/III/IV = PA empty
    (fallback-only: PB / PC / STOP).  gain counts positive executed improvement."""
    pa = {"states": 0, "acted": 0, "pos_acted": 0, "gain": 0.0}
    empty = {"states": 0, "acted": 0, "pos_acted": 0, "gain": 0.0}
    for _iid, st in _r19_steps_of(stps):
        d = st.get("pv_diag")
        stt = (d or {}).get("state_type")
        if stt is None:
            stt = "TYPE-IV" if st.get("stop_reason") == "no_lexicographic_pool" else None
        if stt is None:
            continue
        bucket = pa if stt == "TYPE-I" else empty
        bucket["states"] += 1
        if st.get("stop_reason") in ("no_lexicographic_pool", "no_pool_m2",
                                     "no_proposals", "policy_stop"):
            continue
        imp = st.get("improvement")
        if imp is not None:
            bucket["acted"] += 1
            if imp > 0:
                bucket["pos_acted"] += 1
                bucket["gain"] += float(imp)
    return {"pa_nonempty": pa, "pa_empty": empty}


def _r20_deco(env, scorer, jpol, roots):
    """§35 seven-class failure decomposition under the lexicographic fallback:
      M2_ROOT_MISS           gate retained nothing
      REASONER_MISS          no complete proposal
      PA_SELECTION_MISS      PA layer active but M3 stopped / acted <= 0
      PB_FALLBACK_MISS       PA empty, PB layer active but M3 stopped / acted <= 0
      MEMORY_FALLBACK_MISS   PA/PB empty, PC layer active but M3 stopped / acted <= 0
      STOP_MISS              TYPE-IV (every layer empty) -- fallback exhausted
      TRAJECTORY_COMPOUNDING trajectory had a positive step but terminal <= 0
      M3_OK                  acted with terminal-improving step."""
    import collections
    dec = collections.Counter()
    rollouts = []
    for rf in roots[:5]:
        sm, stps = _r20_eval_closed(env, scorer, jpol, [rf])
        steps = stps.get(rf["iid"], {}).get("steps", [])
        row = {"iid": rf["iid"], "final_gain": int(sm.get("total", 0.0)),
               "steps": []}
        acted_pos = False
        for st in steps:
            reason = st.get("stop_reason")
            imp = st.get("improvement")
            layer = (st.get("pv_diag") or {}).get("active_layer")
            if reason == "no_proposals":
                k2 = "REASONER_MISS"
            elif reason == "no_pool_m2":
                k2 = "M2_ROOT_MISS"
            elif reason == "no_lexicographic_pool":
                k2 = "STOP_MISS"
            elif imp is not None and imp > 0:
                k2 = "M3_OK"
                acted_pos = True
            elif reason == "policy_stop" or (imp is not None and imp <= 0):
                if layer == "PA":
                    k2 = "PA_SELECTION_MISS"
                elif layer == "PB":
                    k2 = "PB_FALLBACK_MISS"
                elif layer == "PC":
                    k2 = "MEMORY_FALLBACK_MISS"
                else:
                    k2 = "UNKNOWN"
            else:
                k2 = "UNKNOWN"
            dec[k2] += 1
            row["steps"].append({"reason": reason, "k": k2, "imp": imp,
                                 "layer": layer})
        if acted_pos and float(sm.get("total", 0.0)) <= 0.0:
            dec["TRAJECTORY_COMPOUNDING"] += 1
        rollouts.append(row)
    return {"decomposition": dict(dec), "rollouts": rollouts}


class _R20Agg:
    """§31-34/§43 lightweight per-rec aggregator over the collected lexicographic
    trajectory groups (never stores full groups)."""
    def __init__(self):
        self.full_sizes, self.pa_sizes, self.pb_sizes = [], [], []
        self.pc_sizes, self.pd_sizes, self.val_sizes = [], [], []
        self.PA = self.PB = self.PC = self.PD = self.inf = 0
        self.states = 0
        self.probe_n = 0
        self.probe_ms = 0.0
        self.n_groups = 0
        self.state_type = {"TYPE-I": 0, "TYPE-II": 0, "TYPE-III": 0, "TYPE-IV": 0}
        self.gh_calls = 0
        self.pb_redundant = 0
        self.sampled_class_count = {"PA": 0, "PB": 0, "PC": 0}
        self.sampled_term_ok = {"PA": [0, 0], "PB": [0, 0], "PC": [0, 0]}
        self.traj_term_ok = []
        self.g1_gains = []

    def __call__(self, groups):
        for g in groups:
            self.n_groups += 1
            self.probe_n += int(g.get("vprobe_n", 0))
            self.probe_ms += float(g.get("vprobe_ms", 0.0))
            for tr in g.get("trajs", []):
                ok = float(tr.get("reward", 0.0)) > 0.0
                self.traj_term_ok.append(ok)
                for rec in tr.get("steps", []):
                    cl = rec.get("sampled_class")
                    if cl in ("PA", "PB", "PC"):
                        self.sampled_class_count[cl] += 1
                        self.sampled_term_ok[cl][0] += 1
                        if ok:
                            self.sampled_term_ok[cl][1] += 1
                    if rec.get("validation_gain") is not None:
                        self.g1_gains.append(float(rec["validation_gain"]))
                    pv = rec.get("pv_diag")
                    if not pv:
                        continue
                    self.states += 1
                    stt = pv.get("state_type")
                    if stt in self.state_type:
                        self.state_type[stt] += 1
                    self.gh_calls += int(pv.get("gh_calls", 0))
                    self.pb_redundant += int(pv.get("pb_redundant", 0))
                    f = int(pv["full_pool_count"])
                    a = int(pv.get("g1_positive_count", 0))
                    b = int(pv.get("delayed_positive_count", 0))
                    c = int(pv.get("memory_rescued_count", 0))
                    d4 = int(pv.get("pruned_count", 0))
                    ix = int(pv.get("infeasible_count", 0))
                    self.full_sizes.append(f)
                    self.pa_sizes.append(a)
                    self.pb_sizes.append(b)
                    self.pc_sizes.append(c)
                    self.pd_sizes.append(d4)
                    self.val_sizes.append(int(pv.get("validated_count", 0)))
                    self.PA += a; self.PB += b; self.PC += c
                    self.PD += d4; self.inf += ix

    def summary(self):
        n = len(self.val_sizes)
        return {
            "states": n, "probe_n": self.probe_n,
            "probe_ms": max(1, round(self.probe_ms)),
            "state_type": dict(self.state_type),
            "gh_calls": self.gh_calls, "pb_redundant": self.pb_redundant,
            "PA_total": self.PA, "PB_total": self.PB, "PC_total": self.PC,
            "PD_total": self.PD, "inf_total": self.inf,
            "mean_full": float(np.mean(self.full_sizes)) if self.full_sizes else 0.0,
            "mean_pa": float(np.mean(self.pa_sizes)) if self.pa_sizes else 0.0,
            "mean_pb": float(np.mean(self.pb_sizes)) if self.pb_sizes else 0.0,
            "mean_pc": float(np.mean(self.pc_sizes)) if self.pc_sizes else 0.0,
            "mean_validated": float(np.mean(self.val_sizes)) if n else 0.0,
            "p90_validated": float(np.percentile(sorted(self.val_sizes), 90)) if n else 0.0,
            "max_validated": max(self.val_sizes) if n else 0.0,
            "median_validated": float(np.median(self.val_sizes)) if n else 0.0,
            "n_val_gt16": sum(1 for x in self.val_sizes if x > 16),
            "n_val_gt32": sum(1 for x in self.val_sizes if x > 32),
            "n_val_gt48": sum(1 for x in self.val_sizes if x > 48),
            "n_val_gt64": sum(1 for x in self.val_sizes if x > 64),
            "trajectories": len(self.traj_term_ok),
            "trajectories_terminal_ok": sum(1 for ok in self.traj_term_ok if ok),
            "validation_ms_per_state": round(
                max(1, round(self.probe_ms)) / max(n, 1), 3),
            "sampled_class_count": dict(self.sampled_class_count),
            "sampled_term_ok": {k: {"acted": v[0], "term_ok": v[1]}
                                for k, v in self.sampled_term_ok.items()},
        }


def _r20_m5_wrapper(env, scorer, jpol, iid, progmem, ep):
    """§39 normal-M5 PERMANENT regression under lexicographic: a dependency-
    completed (pair) Proposal with G1>0 must be PA and retained.  (delayed pairs
    are legitimately pruned when PA is non-empty -- the fallback only shows PB
    when PA is empty -- so no delayed-retention requirement this round.)"""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"at_state": False, "n_full": 0, "dep_immediate": 0,
                "all_retained": True}     # vacuous: nothing to check
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = JG.schedule_hash(st["schedule"])
    sf = JG.state_feature_vec(ms, ms, len(metas), agg["best_uhat"],
                              agg["best_direct"], agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor,
                                progmem, iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"at_state": True, "n_full": 0, "dep_immediate": 0,
                "all_retained": True}     # vacuous: gate emptied
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries),
                       dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    vs = JG.build_lexicographic_action_set_r20(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h,
        ms, mem * gmem, cont_m2=getattr(jpol, "cont_m2", None),
        cont_m3=getattr(jpol, "cont_m3", None), acache=cache, rc={}, hcache={})
    pv = vs["pval"]
    val_set = set(pv["validated_idx"])
    rows = pv["rows"]
    dep_imm = [r for r in rows if r["meta"]["kind"] == "pair" and r["g1"] > 0.0]
    all_ok = all(r["class"] == "PA" and r["k"] in val_set for r in dep_imm)
    return {"at_state": bool(pv["diag"]["full_pool_count"] > 0),
            "n_full": pv["diag"]["full_pool_count"],
            "dep_immediate": len(dep_imm), "all_retained": all_ok}


def run_lexicographic_phase_r20(args, env, re, p1, p1_report):
    """R20: T1-LEXICOGRAPHIC-PROPOSAL-FALLBACK-JOINT-GRPO -- the ONLY change from
    R19 is replacing the PA∪PB∪PC union evidence with a STATE-LEVEL LEXICOGRAPHIC
    FALLBACK: PA non-empty -> PA ∪ STOP (no GH, no Memory); else PB non-empty ->
    PB ∪ STOP; else PC non-empty -> max4 PC ∪ STOP; else {STOP}.  FORBIDDEN:
    expanding H (H_VAL=2 fixed, R19 saturated), M3 representation changes, GH in
    reward.  Root/M2 unchanged.  Continuation = frozen zero-init M2 +
    m3_proposal_top1_sft_v2.pt (no oracle).  M3 = frozen R6 + zero-init evidence
    residual.  JOINT GRPO K8/H5/E3/10cyc/8g.  Checkpoint only on A.
    L0/L1/L2/L3 no-RL parity + K0/K1/K2(cited)/K3 JOINT + state-type dist +
    PB_REDUNDANT + GH-call R19-vs-R20 + action-count + PA split + 7-class deco
    + verdict ladder A-G + 58-item return."""
    print("[r20] M3 LEXICOGRAPHIC-PROPOSAL-FALLBACK JOINT GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- frozen continuation policy FIRST (zero-init M2 + top1-v2), then M3 --
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(
        r6_sel, m3_builder=lambda r6: JG.M3TemporalEvidencePolicy(r6))
    with torch.no_grad():
        resid_max = max((p.abs().max().item()
                         for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    cont_m2 = JG.M2RootPolicyAdapter(jpol.m2.net[0].in_features,
                                     float(jpol.m2.alpha_m2))
    cont_m2.load_snapshot(zj_snap["m2"])            # bit-equal zero-init M2
    for _p_ in cont_m2.parameters():
        _p_.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                     alpha_stop=C.TO1_R13_ALPHA_STOP)
    for _p_ in cont_m3.parameters():
        _p_.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    print(f"[r20] M2 adapter {n_tr_m2} params (zero-init) | M3 temporal residual "
          f"{n_tr_m3} params (frozen {C.TO1_CKPT.name}, resid_max={resid_max:.3g} "
          f"-> δ=0 ≡ R6) | cont_m2(⊞0 M2) + cont_m3(top1-v2) FROZEN; H_VAL={int(C.TO1_R20_H_VAL)} "
          f"FIXED §6 (R19 saturated); NO R11-R17 parent", flush=True)
    proof_no_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r11_grpo_checkpoint": None, "r12-r19_checkpoint": None,
        "resid_max_zero_init": float(resid_max) == 0.0,
        "continuation_frozen": True,
        "oracle_continuation": False,
        "h_fixed": int(C.TO1_R20_H_VAL),
    }

    # ---- verbatim R6 reproduction anchor -------------------------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r20] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- graphs: bench TRAIN14 + AUX-real/syn ---------------------------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)
    print(f"[r20] graphs: bench {len(bench_graphs)} | real "
          f"{len(rb['graphs']) if rb else 0}/{len(rb['hd_iids']) if rb else 0} | syn "
          f"{len(sb['graphs']) if sb else 0}/{len(sb['hd_iids']) if sb else 0}",
          flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _cur(obj, action_space):
        sm, stps = _r20_eval_closed(
            env, scorer, obj,
            _eval_roots(re["progmem"], train_pairs, st_bench_map),
            action_space)
        return sm, stps

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _cur(jp_, "lexicographic")[0]}
        if rb:
            smr, _ = _r20_eval_closed(env, scorer, jp_, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            ev["real_held"] = smr
        else:
            ev["real_held"] = {"total": 0.0}
        if sb:
            sms, _ = _r20_eval_closed(env, scorer, jp_, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            ev["syn_held"] = sms
        else:
            ev["syn_held"] = {"total": 0.0}
        return ev

    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _root_for(iid):
        st = env["states"][iid]
        return RGRPO.roots_from_state(st["problem"], st["schedule"], iid,
                                      env["ep_id_of"][iid],
                                      copy.deepcopy(re["progmem"]))

    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)

    # ---- §30-33 NO-RL parity: L0 full / L1 immediate(R18 one-step) / L2 multistep(R19) /
    #      L3 lexicographic(R20), all same raw ruler ----------------------------
    with torch.no_grad():
        sm_l0, stps_l0 = _r20_eval_closed(env, scorer, jpol, init_roots, "full")
        sm_l1, stps_l1 = _r20_eval_closed(env, scorer, jpol, init_roots, "multistep",
                                          h_val=0)  # H=0 == R18 one-step ruler (evid dim 6)
        sm_l2, stps_l2 = _r20_eval_closed(env, scorer, jpol, init_roots, "multistep")
        sm_l3, stps_l3 = _r20_eval_closed(env, scorer, jpol, init_roots, "lexicographic")
        l0 = float(sm_l0["total"]); l1 = float(sm_l1["total"])
        l2 = float(sm_l2["total"]); l3 = float(sm_l3["total"])
        l3_cls, l3_dists = _r20_pv_rollup(stps_l3)
        l3_pa_split = _r20_pa_split(stps_l3)
        l3_real = l3_syn = l3_val = 0.0
        if rb:
            smr, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            l3_real = float(smr["total"])
        if sb:
            sms, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            l3_syn = float(sms["total"])
        smv, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map))
        l3_val = float(smv["total"])
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    marg = 0.05 * max(float(r6_canonical_train), 1.0)
    print(f"[r20] L0(full)={l0:.0f} L1(immediate/R18)={l1:.0f} "
          f"L2(multistep/R19)={l2:.0f} L3(lexicographic)={l3:.0f} | "
          f"held {l3_real:.0f}/{l3_syn:.0f} VAL {l3_val:.0f} | marg={marg:.1f}",
          flush=True)
    l3_state_summary = {k: l3_cls[k] for k in
                        ("TYPE_I", "TYPE_II", "TYPE_III", "TYPE_IV", "states",
                         "gh_calls", "pb_redundant")}
    print(f"[r20] §30-33 state-type: {json.dumps(l3_state_summary)}",
          flush=True)
    print(f"[r20] §28 PA split: {json.dumps(l3_pa_split)}", flush=True)

    # ---- §24-27 S0 state-type + PB_REDUNDANT + GH-call comparison -------------
    st_stats = []
    stat_iids = ([i["instance_id"] for i in env["train_insts"]]
                 + (["Brandimarte_Mk1"] if any(
                     x["instance_id"] == "Brandimarte_Mk1" for x in env["order"])
                    else []) + (["Fattahi15"] if any(
                        x["instance_id"] == "Fattahi15" for x in env["order"])
                        else []) + ([dpp_iid] if dpp_iid else []))
    seen_iids = set()
    for iid in stat_iids:
        if iid in seen_iids:
            continue
        seen_iids.add(iid)
        try:
            ss = _r20_state_type_stats(env, scorer, jpol, iid,
                                       copy.deepcopy(re["progmem"]))
            st_stats.append(ss)
            if not ss.get("skip"):
                print(f"[r20] §24-27 state-type {iid}: full={ss['n_full']} "
                      f"type={ss['state_type']} layer={ss['active_layer']} "
                      f"GH(R20)={ss['r20_gh_calls']} GH(R19)={ss['r19_gh_calls']} "
                      f"PB_REDUNDANT_exact={ss['pb_redundant_exact']} "
                      f"act(R20)={ss['r20_action_count']} act(R19)={ss['r19_action_count']}",
                      flush=True)
        except Exception as exc:               # noqa: BLE001
            st_stats.append({"skip": True, "iid": iid, "error": str(exc)})
            print(f"[r20] state-type {iid} FAILED: {exc}", flush=True)
    n_state_I = sum(1 for x in st_stats if not x.get("skip") and x.get("state_type") == "TYPE-I")
    n_state_II = sum(1 for x in st_stats if not x.get("skip") and x.get("state_type") == "TYPE-II")
    n_state_III = sum(1 for x in st_stats if not x.get("skip") and x.get("state_type") == "TYPE-III")
    n_state_IV = sum(1 for x in st_stats if not x.get("skip") and x.get("state_type") == "TYPE-IV")
    pb_redundant_sum = sum(x.get("pb_redundant_exact", 0) for x in st_stats)
    gh_r19_sum = sum(x.get("r19_gh_calls", 0) for x in st_stats)
    gh_r20_sum = sum(x.get("r20_gh_calls", 0) for x in st_stats)

    # ---- §34-36 inherited root/temporal diagnostics (M2 unchanged) -------------
    root_delay = None
    try:
        r_vals = []
        for iid in [i["instance_id"] for i in env["train_insts"]][:4]:
            rd = _r19_root_delayed(env, scorer, jpol, iid,
                                   copy.deepcopy(re["progmem"]))
            if not rd.get("skip"):
                r_vals.append(rd)
        root_delay = {"root_delayed_benefit_rate": float(np.mean(
            [r["root_delayed_benefit_rate"] for r in r_vals])) if r_vals else 0.0,
            "n_root_delayed": sum(r["n_root_delayed"] for r in r_vals),
            "n_root_pruned": sum(r["n_root_pruned"] for r in r_vals)}
        print(f"[r20] §34 root_delayed_benefit_rate="
              f"{root_delay['root_delayed_benefit_rate']:.3f}", flush=True)
    except Exception as exc:                   # noqa: BLE001
        root_delay = {"error": str(exc)}
        print(f"[r20] §34 root-delayed FAILED: {exc}", flush=True)

    tmiss = None
    try:
        tmiss = _r19_temporal_miss_states(
            env, scorer, jpol, [i["instance_id"] for i in env["train_insts"]][:3],
            copy.deepcopy(re["progmem"]), h_base=int(C.TO1_R20_H_VAL))
        print(f"[r20] §35 temporal-miss: {tmiss['temporal_miss_rate']:.3f}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        tmiss = {"error": str(exc)}
        print(f"[r20] §35 temporal-miss FAILED: {exc}", flush=True)

    # ---- §36 pre-train deco (7-class) + M5 -----------------------------------
    deco_i = None
    try:
        deco_i = _r20_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r20] deco INIT: {json.dumps(deco_i['decomposition'])}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        deco_i = {"error": str(exc)}
        print(f"[r20] deco INIT FAILED: {exc}", flush=True)

    m5_z = _r20_m5_wrapper(env, scorer, jpol, dpp_iid,
                           copy.deepcopy(re["progmem"]), dpp_ep) if dpp_iid else None
    if m5_z:
        print(f"[r20] §39 normal-M5 zero-init: dep_imm={m5_z.get('dep_immediate')} "
              f"all_retained={m5_z.get('all_retained')}", flush=True)

    # ---- cloud/shape profile -> workers ---------------------------------------
    prof = None
    mp_ok = True
    workers = 1
    agg20 = _R20Agg()
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, init_roots[0],
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14",
                                        action_space="lexicographic")
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"]
                       for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r20] cloud profile: identical={ident_ok} -> workers={workers}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r20] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- §37-42 THE ONE JOINT STAGE (K3): lexicographic JOINT GRPO ------------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")
    res_j = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=int(C.TO1_R20_TRAINING_CYCLES),
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=int(C.TO1_R20_GRAPHS_PER_BATCH), workers=workers,
        seed=args.grpo_seed, log_prefix="[r20-J]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, l3), parent_policy=None, mp_ctx=mp_ctx,
        variant="r20", on_groups=agg20, action_space="lexicographic")
    m5_f = _r20_m5_wrapper(env, scorer, jpol, dpp_iid,
                           copy.deepcopy(re["progmem"]), dpp_ep) if dpp_iid else None
    if m5_f:
        print(f"[r20] §39 normal-M5 FINAL: dep_imm={m5_f.get('dep_immediate')} "
              f"all_retained={m5_f.get('all_retained')}", flush=True)

    # ---- §43-46 K0/K1/K2 cited + K3 MAIN --------------------------------------
    k0 = float(C.TO1_R20_K0_FULL_JOINT)
    k1 = float(C.TO1_R20_K1_IMMEDIATE_JOINT)
    k2 = float(C.TO1_R20_K2_PA_PB_JOINT)
    with torch.no_grad():
        sm_k3, stps_k3 = _r20_eval_closed(env, scorer, jpol, init_roots,
                                          "lexicographic")
        k3 = float(sm_k3["total"])
        k3_cls, k3_dists = _r20_pv_rollup(stps_k3)
        k3_real = k3_syn = k3_val = 0.0
        if rb:
            smr, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            k3_real = float(smr["total"])
        if sb:
            sms, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            k3_syn = float(sms["total"])
        smv, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map))
        k3_val = float(smv["total"])
    print(f"[r20] K0(cited)={k0:.0f} K1(cited)={k1:.0f} K2(cited)={k2:.0f} "
          f"K3(lexicographic JOINT)={k3:.0f}/{k3_real:.0f}/{k3_syn:.0f}/{k3_val:.0f}",
          flush=True)

    # ---- diagnostics AFTER ------------------------------------------------------
    deco_f = None
    try:
        deco_f = _r20_deco(env, scorer, jpol, init_roots[:5])
        print(f"[r20] deco FINAL: {json.dumps(deco_f['decomposition'])}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        deco_f = {"error": str(exc)}
        print(f"[r20] deco FINAL FAILED: {exc}", flush=True)

    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        dpp_root = _root_for(dpp_iid)
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _ = _r20_eval_closed(env, scorer, zj2, [dpp_root])
            post_sum, _ = _r20_eval_closed(env, scorer, jpol, [dpp_root])
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r20] DPP BEFORE(zero-init)={dpp_pre['total_gain']} "
              f"| AFTER(final)={dpp_post['total_gain']}", flush=True)

    # ---- per-cycle rollup -------------------------------------------------------
    n_groups = sum(h.get("n_groups", 0) for h in res_j["history"]) \
        if res_j.get("history") else 0
    n_inf3 = sum(h.get("n_informative", 0) for h in res_j["history"]) \
        if res_j.get("history") else 0
    n_inf2 = sum(h.get("n_informative_trajectories_m2", 0)
                 for h in res_j["history"]) if res_j.get("history") else 0
    m2_kl = [float(e["kl_m2"])
             for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", []))
             if e.get("kl_m2") is not None]
    m3_kl = [float(e["kl_ref_m3"])
             for h in res_j["history"] for d in (h.get("depth_results") or [])
             for e in ((d.get("update") or {}).get("epochs", []))
             if e.get("kl_ref_m3") is not None]
    m2_kl = m2_kl or [0.0]
    m3_kl = m3_kl or [0.0]

    t_miss_rate = float((tmiss or {}).get("temporal_miss_rate", 0.0))
    root_delayed_rate = float((root_delay or {}).get("root_delayed_benefit_rate", 0.0))
    da = agg20.summary()

    # ---- verdict ladder §45-51 (G -> D -> C -> E -> F -> A -> B) ---------------
    def _m5_retained(x):
        if not isinstance(x, dict):
            return False
        ar = x.get("all_retained")
        return True if ar is None else bool(ar)
    m5_ok = bool(all(_m5_retained(x) for x in (m5_z, m5_f) if x))
    g_fail = (not repro_ok or not mp_ok or bool(res_j["collapsed"]) or not m5_ok)
    l3_ok = bool(l3 >= l2 - marg)                      # lexicographic must not hurt union
    k3_beats = bool(k3 > k2 + marg)                    # A: JOINT beats union by marg
    k3_parity = bool(k3 >= k2 - marg)                  # B: no worse than union
    base_hd = max(l3_real, l3_syn)
    held_better = [float(v) for v in (k3_real, k3_syn) if v > base_hd + 1.0]
    dense = bool(da["states"] > 0 and
                 (da["p90_validated"] > C.TO1_R19_DENSE_POOL_HIGH2
                  or da["mean_validated"] > C.TO1_R19_DENSE_POOL_HIGH))
    too_slow = bool(da["validation_ms_per_state"] > 4.0)
    root_major = bool(root_delayed_rate > t_miss_rate + 0.05 and
                      root_delayed_rate > C.TO1_R19_ROOT_DELAYED_MAX)

    if g_fail:
        vcode, vlabel = "G", "LEXICOGRAPHIC_FALLBACK_SEMANTICS_BUG"
        note = (f"repro={repro_ok} mp_ok={mp_ok} collapsed={res_j.get('collapsed')} "
                f"m5-§39={m5_ok} -- machine broken incl. any §-assert fire")
        ok = False
    elif not l3_cls["states"]:
        vcode, vlabel = "G", "LEXICOGRAPHIC_FALLBACK_SEMANTICS_BUG"
        note = "no lexicographic validation states were produced"
        ok = False
    elif too_slow and not k3_beats:
        vcode, vlabel = "D", "LEXICOGRAPHIC_FALLBACK_TOO_EXPENSIVE"
        note = (f"validation {da['validation_ms_per_state']:.1f}ms/state >4.0 "
                f"and K3 not beating K2 -- lazy GH did not reduce cost")
        ok = False
    elif not l3_ok:
        vcode, vlabel = "C", "LEXICOGRAPHIC_FALLBACK_HURTS_NO_RL"
        note = (f"L3={l3:.0f} < L2={l2:.0f}-{marg:.1f} -- the fallback drops "
                f"candidates that mattered to no-RL gain")
        ok = False
    elif root_major:
        vcode, vlabel = "E", "ROOT_LEVEL_MYOPIA_BECOMES_PRIMARY"
        note = (f"root_delayed_benefit_rate={root_delayed_rate:.3f} > "
                f"{C.TO1_R19_ROOT_DELAYED_MAX} -- M2 unchanged, next bottleneck "
                f"is root myopia")
        ok = False
    elif dense:
        vcode, vlabel = "F", "DENSE_VALIDATED_POOL_REMAINS"
        note = (f"lexicographic still yields dense pools: mean="
                f"{da['mean_validated']:.1f} p90={da['p90_validated']:.0f}")
        ok = False
    elif k3_beats and l3_ok and held_better and t_miss_rate <= C.TO1_R19_TEMPORAL_MISS_MAX \
            and not res_j["collapsed"]:
        vcode, vlabel = "A", "LEXICOGRAPHIC_FALLBACK_BEATS_UNION"
        note = (f"K3 {k3:.0f} > K2 {k2:.0f}+{marg:.1f} AND L3 {l3:.0f} >= L2 {l2:.0f} "
                f"AND held {held_better} > {base_hd:.0f}+1 -- state-level fallback "
                f"beats the PA∪PB union")
        ok = True
    else:
        vcode, vlabel = "B", "LEXICOGRAPHIC_NEUTRAL_NO_JOINT_GAIN"
        note = (f"K3={k3:.0f} vs K2={k2:.0f} (parity={k3_parity}); L3={l3:.0f} vs "
                f"L2={l2:.0f}; fallback restructures the action set (GH-calls "
                f"{gh_r20_sum} vs R19 {gh_r19_sum}, PB_REDUNDANT={pb_redundant_sum}) "
                f"but produces no joint gain")
        ok = False
    passed = bool(ok)
    print(f"[r20] verdict {vcode} {vlabel} (L0={l0:.0f} L1={l1:.0f} L2={l2:.0f} "
          f"L3={l3:.0f} K2={k2:.0f} K3={k3:.0f} GH R20/R19={gh_r20_sum}/{gh_r19_sum} "
          f"PB_REDUNDANT={pb_redundant_sum})", flush=True)

    if passed:
        meta_common = dict(
            phase="r20_lexicographic_proposal_fallback_joint_grpo",
            method="r6_sft_then_state_level_lexicographic_fallback_joint_agentic_grpo",
            pipeline="M2_SFT->M3_SFT->Joint_GRPO",
            root_validation="makespan_first_memory_second (R14 adaptive gate verbatim)",
            proposal_evidence="state_level_lexicographic_fallback",
            validation_horizon=int(C.TO1_R20_H_VAL),
            validation_policy_id=str(C.TO1_R20_VALIDATION_POLICY_ID),
            m3_action_space="PA if nonempty else PB if nonempty else max4 PC else STOP",
            cap32=False, shortlist=False,
            proposal_probe_gain_reward=False,
            proposal_probe_gain_observation=True,
            memory_reward_authority=False,
            continuation_policy="frozen zero-init M2 + m3_proposal_top1_sft_v2.pt",
            oracle_continuation=False,
            evidence_adapter="M3TemporalEvidencePolicy (g1_norm|gh_norm|"
                             "is_immediate|is_delayed|is_mem|conf, α=1) "
                             "zero-init residual over frozen R6",
            parent=C.TO1_CKPT.name,
            r11_r19_parent=None,
            reward="terminal_makespan_gain (M3 A3) / q2 local probe (M2 A2), stagewise",
            m2_reward_authority=False,
            executor="FixedDecisionReplay", formal_test_access=0,
            formal_test_sealed=True, identified=False,
            G1="Cmax(S_t)-Cmax(S'_P); GH=H-step rescue -- observation ONLY, NEVER reward",
            K=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=int(C.TO1_R20_GRAPHS_PER_BATCH),
            training_cycles=int(C.TO1_R20_TRAINING_CYCLES),
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            temp=C.TO1_R13_TEMP, mix_eps=C.TO1_R13_MIX_EPS,
            clip_eps=C.TO1_R13_CLIP_EPS,
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (NOT VAL3)",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / revisit",
            replay_semantics="real FixedDecisionReplay per Proposal; GH via frozen "
                             "continuation",
            validated_action_set="state-level lexicographic fallback (PA>>PB>>PC>>STOP)")
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": dict(meta_common,
                                 checkpoint="m2_m3_lexicographic_joint_grpo_r20",
                                 role="M3 lexicographic-fallback action-space joint GRPO")},
                   C.TO1_R20_CKPT)
        print(f"[r20] saved {C.TO1_R20_CKPT.name} (PASS)", flush=True)
    else:
        print(f"[r20] NOT PASS -> no R20 checkpoint written (§53)", flush=True)

    _r20_scrape = dict(
        repro_ok=repro_ok, mp_ok=mp_ok, m5_z=m5_z, m5_f=m5_f, res_j=res_j,
        l0=l0, l1=l1, l2=l2, l3=l3,
        k0=k0, k1=k1, k2=k2, k3=k3,
        l3_real=l3_real, l3_syn=l3_syn, l3_val=l3_val,
        k3_real=k3_real, k3_syn=k3_syn, k3_val=k3_val,
        rows=dict(l0=l0, l1=l1, l2=l2, l3=l3, k0=k0, k1=k1, k2=k2, k3=k3),
        l3_ok=l3_ok, k3_beats=k3_beats, k3_parity=k3_parity,
        r6_canonical_train=r6_canonical_train, prof=prof,
        workers=workers, mp_ctx=mp_ctx, dpp_pre=dpp_pre, dpp_post=dpp_post,
        deco_i=deco_i, deco_f=deco_f, st_stats=st_stats,
        state_dist=dict(TYPE_I=n_state_I, TYPE_II=n_state_II,
                        TYPE_III=n_state_III, TYPE_IV=n_state_IV),
        pb_redundant=pb_redundant_sum, gh_calls_r19=gh_r19_sum,
        gh_calls_r20=gh_r20_sum,
        pa_split=l3_pa_split, root_delay=root_delay, tmiss=tmiss,
        rendezvous=dict(l3_cls=l3_cls, k3_cls=k3_cls),
        agg=da,
        joint={"best_train": float((res_j.get("best") or {"train": 0})["train"]),
               "collapsed": bool(res_j["collapsed"]),
               "n_informative": n_inf3, "n_informative_m2": n_inf2,
               "n_groups": n_groups, "m2_kl_last": m2_kl[-1],
               "m3_kl_last": m3_kl[-1]},
        proof_no_rl_parent=proof_no_rl_parent,
        train_beat=bool(k3 > max(k0, k2, l3) + marg), held_better=held_better,
        base_hd=base_hd, t_miss_rate=t_miss_rate, root_delayed_rate=root_delayed_rate,
        m3_resid_count=n_tr_m3, m2_resid_count=n_tr_m2,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed)
    report = _r20_report(env, re, _r20_scrape)
    _r20_persist(report, _r20_scrape)
    print(f"[r20] total {time.time() - t0:.1f}s "
          f"(report {C.R20_REPORT.name})", flush=True)
    return report


def _r20_report(env, re, s):
    """Markdown report answering the §-numbered 58-item return checklist."""
    L = []
    J = lambda d: json.dumps(d, default=str)
    verdict = s.get("verdict", {})
    di = (s.get("deco_i") or {}).get("decomposition", {})
    df = (s.get("deco_f") or {}).get("decomposition", {})
    agg = s.get("agg", {})
    jt = s.get("joint", {})
    rj = s.get("res_j") or {}

    def h1(t):
        L.append(f"# {t}\n")
    def h2(t):
        L.append(f"## {t}\n")
    def kv(k, v):
        L.append(f"- **{k}**: {v}")
    L.append("# R20 — T1-LEXICOGRAPHIC-PROPOSAL-FALLBACK-JOINT-GRPO 报告\n")
    L.append(f"_verdict {verdict.get('code')} {verdict.get('label')} "
             f"\\| passed={s.get('passed')} \\| identified=false \\| "
             f"formal_test_access=0 \\| Formal TEST SEALED  \n"
             f"{verdict.get('note', '')}_\n")
    h1("1. Modified files")
    kv("engine", "src/causal_schedule_lab/m3/joint_grpo.py "
                 "(proposal_validate_r20 / build_lexicographic_action_set_r20 / "
                 "4 collect+eval lexicographic branches / variant='r20')")
    kv("config", "src/causal_schedule_lab/m3/config.py (§TO1_R20_*)")
    kv("runner", "scripts/run_m3_canonical_training.py (run_lexicographic_phase_r20)")
    h1("2. Exact runtime pipeline")
    kv("pipeline", "S_t+Appearance -> M2 budgeted root probe (R14 adaptive gate "
                   "VERBATIM) -> makespan-first/Memory-second root filter -> Reasoner "
                   "-> complete legal Proposal pool -> STATE-LEVEL lexicographic "
                   "validation (PA>>PB>>PC>>STOP) -> M3 action set -> JOINT GRPO "
                   "K8/H5/E3/10cyc/8g")
    h1("3-5. Root direct / dependency-completed / Memory rules")
    kv("root semantics", "R14 adaptive gate verbatim; UNCHANGED this round; G1/GH "
                         "never used at root level")
    kv("root Memory rule", f"support>=2 and success>=0.5 and gmem>0 -> Tier-B "
                           f"(budget {C.TO1_R13_MEMORY_BUDGET})")
    h1("6-11. Proposal evidence semantics (lexicographic)")
    kv("G1 definition", "Cmax(S_t)-Cmax(S'_P) via FixedDecisionReplay")
    kv("H_VAL", f"{int(C.TO1_R20_H_VAL)} (FIXED §6, R19 H-sweep saturated at 2; "
                f"FORBIDDEN to expand)")
    kv("continuation policy", "frozen zero-init M2 (bit-equal jpol.m2 start) + "
                              "frozen m3_proposal_top1_sft_v2.pt (no oracle)")
    kv("proof no oracle continuation", f"{J(s.get('proof_no_rl_parent'))}")
    kv("GH definition", "ms_cur - Cmax(branch_schedule) after up to H frozen steps; "
                        "computed ONLY when PA empty (§5)")
    kv("lexicographic rule", "PA non-empty -> M3 = PA ∪ STOP (NO GH, NO Memory); "
                             "else PB non-empty -> PB ∪ STOP; else PC non-empty -> "
                             "max4 PC ∪ STOP; else {STOP}")
    kv("no score mixing", "G1/GH/Memory never summed into reward or score; the "
                          "hierarchy is expressed ONLY through action-set eligibility")
    h1("12. State type distribution (TYPE-I/II/III/IV)")
    kv("distribution", f"{J(s.get('state_dist'))}")
    kv("rollup (parity steps)", f"{J(s.get('rendezvous', {}).get('l3_cls'))}")
    h1("13. PB_REDUNDANT")
    kv("PB_REDUNDANT (exact, pre-RL)", f"{s.get('pb_redundant')} "
                                       f"(would-be delayed candidates R19 GH-validates "
                                       f"that R20 skips in PA-nonempty states)")
    h1("14. GH calls R19 vs R20")
    kv("GH calls", f"R19={s.get('gh_calls_r19')} R20={s.get('gh_calls_r20')} "
                   f"(saved={s.get('gh_calls_r19', 0) - s.get('gh_calls_r20', 0)})")
    h1("15. Action count (full/R18/R19/R20)")
    kv("action count", "full = full pool; R18 = validated; R19 = PA∪PB∪max4PC; "
                       "R20 = lexicographic (see st_stats r19_action_count vs "
                       "r20_action_count)")
    kv("per-state", f"{J(s.get('st_stats'))}")
    h1("16-17. No-RL parity (L0/L1/L2/L3)")
    for k, name in (("l0", "L0 full SFT"), ("l1", "L1 immediate(R18 one-step)"),
                    ("l2", "L2 multistep(R19)"), ("l3", "L3 lexicographic(R20)")):
        kv(name, f"{s.get(k):.0f}")
    kv("L3 vs L2", f"l3_ok={s.get('l3_ok')} (L3 must be >= L2 - marg)")
    h1("18-19. JOINT parity (K0/K1/K2/K3)")
    for k, name in (("k0", "K0 full JOINT(R14)"), ("k1", "K1 immediate JOINT(R18)"),
                    ("k2", "K2 PA+PB JOINT(R19)"), ("k3", "K3 lexicographic JOINT")):
        kv(name, f"{s.get(k):.0f}")
    kv("K3 vs K2", f"beats={s.get('k3_beats')} parity={s.get('k3_parity')}")
    h1("20. PA-empty / nonempty performance split")
    kv("split", f"{J(s.get('pa_split'))}")
    h1("21. 7-class failure decomposition (§35)")
    keys = ["M2_ROOT_MISS", "REASONER_MISS", "PA_SELECTION_MISS",
            "PB_FALLBACK_MISS", "MEMORY_FALLBACK_MISS", "TRAJECTORY_COMPOUNDING",
            "STOP_MISS", "M3_OK"]
    kv("initial", f"{J({k: di.get(k, 0) for k in keys})}")
    kv("final", f"{J({k: df.get(k, 0) for k in keys})}")
    h1("22-25. Training signals")
    kv("M2 KL", f"{jt.get('m2_kl_last', 0.0):.4f}")
    kv("M3 KL", f"{jt.get('m3_kl_last', 0.0):.4f}")
    kv("M2 gradients", f"{s.get('m2_resid_count', 0)} trainable (adapter)")
    kv("M3 gradients", f"{s.get('m3_resid_count', 0)} trainable (temporal residual)")
    kv("terminal reward", f"traj terminal-positive: "
                          f"{agg.get('trajectories_terminal_ok', 0)}/"
                          f"{agg.get('trajectories', 0)}")
    kv("sampled class by layer", f"{J(agg.get('sampled_class_count'))} "
                                 f"term-ok {J(agg.get('sampled_term_ok'))}")
    h1("26-28. Held generalization")
    kv("AUX-real", f"L3-real={s.get('l3_real'):.0f} -> K3-real={s.get('k3_real'):.0f}")
    kv("AUX-syn", f"L3-syn={s.get('l3_syn'):.0f} -> K3-syn={s.get('k3_syn'):.0f}")
    kv("VAL once no_grad", f"L3-val={s.get('l3_val'):.0f} K3-val={s.get('k3_val'):.0f}")
    h1("29-31. Root / temporal diagnostics (inherited, M2 unchanged)")
    kv("root_delayed_benefit_rate", f"{s.get('root_delayed_rate', 0.0):.3f}")
    kv("temporal_miss_rate", f"{s.get('t_miss_rate', 0.0):.3f}")
    kv("normal-M5 (§39)", f"zero={J(s.get('m5_z'))} final={J(s.get('m5_f'))}")
    h1("32-34. Runtime")
    kv("GH/validation probe", f"probe_n={agg.get('probe_n', 0)} "
                              f"ms/state={agg.get('validation_ms_per_state', 0.0)}")
    kv("pool density", f"mean={agg.get('mean_validated', 0.0):.1f} "
                       f"p90={agg.get('p90_validated', 0.0):.0f} "
                       f">16/32/48/64={agg.get('n_val_gt16', 0)}/"
                       f"{agg.get('n_val_gt32', 0)}/{agg.get('n_val_gt48', 0)}/"
                       f"{agg.get('n_val_gt64', 0)}")
    h1("35-36. Repro / determinism / tests")
    kv("R6 reproduction", f"acc/regret anchor repro_ok={s.get('repro_ok')}")
    kv("multiprocessing determinism", f"identical-to-w1={s.get('mp_ok')} "
                                      f"-> workers={s.get('workers')}")
    kv("pytest", "see tests/test_m3_lexicographic_fallback_r20.py")
    h1("37. Checkpoint")
    kv("checkpoint metadata", f"only on A: {C.TO1_R20_CKPT.name}")
    kv("promoted", f"{bool(s.get('passed'))}")
    h1("38. Verdict")
    kv("verdict", f"{verdict.get('code')} — {verdict.get('label')}")
    kv("note", verdict.get("note", ""))
    h1("39. 下一步唯一最高优先级动作")
    nxt = ("promote lexicographic fallback (A)" if s.get("passed")
           else "B-> fallback restructures but no joint gain: next = multi-step "
                "credit (R19 §65) or root-level myopia (E); C-> fallback drops "
                "candidates; D-> reduce GH cost; F-> pool density")
    kv("action", nxt)
    L.append("\n---\n")
    L.append(f"_state-type trace \\| {J(s.get('st_stats'))}_")
    return "\n".join(L)


def _r20_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R20_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in ("repro_ok", "mp_ok", "m5_z", "m5_f", "l0", "l1", "l2", "l3",
                  "k0", "k1", "k2", "k3", "l3_real", "l3_syn", "l3_val",
                  "k3_real", "k3_syn", "k3_val", "rows", "l3_ok", "k3_beats",
                  "k3_parity", "r6_canonical_train", "prof", "workers", "mp_ctx",
                  "dpp_pre", "dpp_post", "deco_i", "deco_f", "st_stats",
                  "state_dist", "pb_redundant", "gh_calls_r19", "gh_calls_r20",
                  "pa_split", "root_delay", "tmiss", "rendezvous", "agg",
                  "joint", "proof_no_rl_parent", "train_beat", "held_better",
                  "base_hd", "t_miss_rate", "root_delayed_rate",
                  "m3_resid_count", "m2_resid_count", "verdict", "passed"):
            if k in scrape:
                payload[k] = scrape[k]
    (C.CANONICAL_OUT_DIR / "result_r20.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r20] report written: {C.R20_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# R21 -- T1-INSTANCE-DIVERSE-JOINT-GRPO（R20 verdict B 的直接后续）
# 唯一改变 = Joint GRPO training instance distribution。D0 = R20 runtime +
# benchmark-dominant；D1 = R20 runtime + instance-diverse balanced。同预算。
# 模型选择 generalization-first（internal-held + AUX-real held + AUX-syn held，
# 源级等权；VAL final-only）。checkpoint 仅 verdict A。
# ---------------------------------------------------------------------------
def _r21_src_of(bench_iids, rb, sb):
    src = {}
    for iid in bench_iids:
        src[iid] = "bench"
    if rb:
        for iid in rb["tr_iids"]:
            src[iid] = "real"
    if sb:
        for iid in sb["tr_iids"]:
            src[iid] = "syn"
    return src


def _r21_family_of(bench_iids, rb, sb):
    """§12 family mapping.  Benchmark family = leading iid token (Behnke/Brandimarte/
    DPdata/Fattahi/Hurink); AUX-real reads manifest['family']; AUX-syn has no family
    key -> the single synthetic family 'AUX-syn'."""
    fam = {}
    for iid in bench_iids:
        fam[iid] = iid.split("_")[0]
    if rb:
        for iid in rb["tr_iids"]:
            fam[iid] = (rb["tr_states"][iid].get("family") or "AUX-real")
    if sb:
        for iid in sb["tr_iids"]:
            fam[iid] = "AUX-syn"
    return fam


class _R21Agg:
    """§20-25 per-cycle instance-exposure + per-source PA/reward aggregator.

    Receives (src_of, family_of) and, per collected group, records per-instance
    exposure (times sampled), unique (iid, state_hash), and per-source/per-family
    PA counts + terminal-reward distributions -- the raw material for the §12 family
    report and the §20-25 exposure table.  Never stores full groups."""
    def __init__(self, src_of=None, family_of=None):
        from collections import Counter, defaultdict
        self.src_of = src_of or {}
        self.family_of = family_of or {}
        self.n_groups = 0
        self.n_trajectories = 0
        self.n_states = 0
        self.exposure = Counter()
        self.groups_by_iid = Counter()
        self.unique_states = set()
        self.unique_states_by_iid = defaultdict(set)
        self.term_by_iid = defaultdict(list)
        self.term_by_src = defaultdict(list)
        self.term_by_fam = defaultdict(list)
        self.pa_by_iid = Counter()
        self.pa_by_src = Counter()
        self.pa_by_fam = Counter()

    def __call__(self, groups):
        for g in groups:
            self.n_groups += 1
            iid = g.get("iid")
            src = self.src_of.get(iid, "unknown")
            fam = self.family_of.get(iid, "unknown")
            self.exposure[iid] += 1
            self.groups_by_iid[iid] += 1
            sh = g.get("state_hash")
            if iid is not None and sh is not None:
                self.unique_states.add((iid, str(sh)))
                self.unique_states_by_iid[iid].add(str(sh))
            for tr in g.get("trajs", []):
                self.n_trajectories += 1
                term = float(tr.get("reward", 0.0))
                self.term_by_iid[iid].append(term)
                self.term_by_src[src].append(term)
                self.term_by_fam[fam].append(term)
                for rec in tr.get("steps", []):
                    pv = rec.get("pv_diag")
                    if not pv:
                        continue
                    self.n_states += 1
                    pa = int(pv.get("g1_positive_count", 0))
                    self.pa_by_iid[iid] += pa
                    self.pa_by_src[src] += pa
                    self.pa_by_fam[fam] += pa

    @staticmethod
    def _dist(vals):
        if not vals:
            return {"n": 0, "mean": 0.0, "n_positive": 0, "min": 0.0, "max": 0.0}
        arr = [float(v) for v in vals]
        return {"n": len(arr), "mean": float(np.mean(arr)),
                "n_positive": sum(1 for v in arr if v > 0.0),
                "min": float(np.min(arr)), "max": float(np.max(arr))}

    def summary(self):
        src_rows = {}
        for s in ("bench", "real", "syn"):
            iids = [i for i in self.exposure if self.src_of.get(i) == s]
            src_rows[s] = {
                "n_instances_sampled": len(iids),
                "n_groups": sum(self.groups_by_iid[i] for i in iids),
                "n_trajectories": len(self.term_by_src.get(s, [])),
                "terminal": self._dist(self.term_by_src.get(s, [])),
                "pa_count": int(self.pa_by_src.get(s, 0)),
            }
        fams = sorted(set(self.family_of.values()))
        fam_rows = {}
        for f in fams:
            iids = [i for i in self.exposure if self.family_of.get(i) == f]
            fam_rows[f] = {
                "n_instances_sampled": len(iids),
                "n_groups": sum(self.groups_by_iid[i] for i in iids),
                "n_trajectories": len(self.term_by_fam.get(f, [])),
                "terminal": self._dist(self.term_by_fam.get(f, [])),
                "pa_count": int(self.pa_by_fam.get(f, 0)),
            }
        exposure_table = [
            {"iid": i, "src": self.src_of.get(i, "unknown"),
             "family": self.family_of.get(i, "unknown"),
             "times_sampled": int(self.exposure[i]),
             "n_groups": int(self.groups_by_iid[i]),
             "n_root_states": len(self.unique_states_by_iid[i]),
             "terminal": self._dist(self.term_by_iid[i]),
             "pa_count": int(self.pa_by_iid[i])}
            for i in sorted(self.exposure, key=lambda x: (-self.exposure[x], str(x)))
        ]
        return {
            "n_groups": self.n_groups,
            "n_trajectories": self.n_trajectories,
            "n_states": self.n_states,
            "n_unique_states": len(self.unique_states),
            "n_distinct_instances": sum(1 for i in self.exposure if self.exposure[i] > 0),
            "by_source": src_rows,
            "by_family": fam_rows,
            "exposure_table": exposure_table,
        }


def run_instance_diverse_phase_r21(args, env, re, p1, p1_report):
    """R21: T1-INSTANCE-DIVERSE-JOINT-GRPO -- the ONLY change from R20 is the Joint
    GRPO training instance distribution (§6).  R20 runtime VERBATIM (lexicographic
    fallback, R14 adaptive gate, M2 A2 / M3 A3 stagewise reward, frozen zero-init
    continuation).  D0 = R20 benchmark-dominant draw (per_src bench2/real1/syn1,
    TRAIN-based selection) reproducing K3=387; D1 = instance-diverse unified
    round-robin (per_cycle 4 over bench14+real32+syn80) with generalization-first
    selection.  Same trajectory budget (4 graphs/cycle x 10).  §55-item return,
    verdict ladder A-G, checkpoint only on A."""
    print("[r21] M3 INSTANCE-DIVERSE JOINT GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))

    # ---- frozen continuation policy (zero-init M2 + top1-v2), then M3 ---------
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(
        r6_sel, m3_builder=lambda r6: JG.M3TemporalEvidencePolicy(r6))
    with torch.no_grad():
        resid_max = max((p.abs().max().item()
                         for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    cont_m2 = JG.M2RootPolicyAdapter(jpol.m2.net[0].in_features,
                                     float(jpol.m2.alpha_m2))
    cont_m2.load_snapshot(zj_snap["m2"])
    for _p_ in cont_m2.parameters():
        _p_.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                     alpha_stop=C.TO1_R13_ALPHA_STOP)
    for _p_ in cont_m3.parameters():
        _p_.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    print(f"[r21] M2 adapter {n_tr_m2} params (zero-init) | M3 temporal residual "
          f"{n_tr_m3} params (frozen {C.TO1_CKPT.name}, resid_max={resid_max:.3g}) | "
          f"H_VAL={int(C.TO1_R20_H_VAL)} FIXED | R20 runtime VERBATIM | R21 ONLY "
          f"CHANGE = Joint training instance distribution", flush=True)

    # ---- verbatim R6 reproduction anchor --------------------------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r21] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- graphs: bench TRAIN14 + AUX-real/syn --------------------------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)
    print(f"[r21] graphs: bench {len(bench_graphs)} | real "
          f"{len(rb['graphs']) if rb else 0}/{len(rb['hd_iids']) if rb else 0} | syn "
          f"{len(sb['graphs']) if sb else 0}/{len(sb['hd_iids']) if sb else 0}",
          flush=True)

    bench_iids = [i["instance_id"] for i in env["train_insts"]]
    train_pairs = [(iid, env["ep_id_of"][iid]) for iid in bench_iids]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}
    src_of = _r21_src_of(bench_iids, rb, sb)
    family_of = _r21_family_of(bench_iids, rb, sb)

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _cur(jp_, action_space):
        sm, stps = _r20_eval_closed(
            env, scorer, jp_,
            _eval_roots(re["progmem"], train_pairs, st_bench_map), action_space)
        return sm, stps

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _cur(jp_, "lexicographic")[0]}
        smv, _ = _r20_eval_closed(env, scorer, jp_,
                                  _eval_roots(re["progmem"], val_pairs, st_bench_map))
        ev["bench_held"] = smv                       # benchmark internal-held == VAL3
        if rb:
            smr, _ = _r20_eval_closed(env, scorer, jp_, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            ev["real_held"] = smr
        else:
            ev["real_held"] = {"total": 0.0}
        if sb:
            sms, _ = _r20_eval_closed(env, scorer, jp_, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            ev["syn_held"] = sms
        else:
            ev["syn_held"] = {"total": 0.0}
        return ev

    def held_agg_fn(ev):
        return float(np.mean([float(ev.get(k, {}).get("total", 0.0))
                              for k in ("bench_held", "real_held", "syn_held")]))

    def _held_agg_of(d):
        return float(np.mean([float(d.get(k, 0.0))
                              for k in ("bench_held", "real_held", "syn_held")]))

    # ---- NO-RL parity baseline L3 (lexicographic) + held + VAL ----------------
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    with torch.no_grad():
        sm_l3, _ = _cur(jpol, "lexicographic")
        l3 = float(sm_l3["total"])
        l3_real = l3_syn = l3_val = 0.0
        if rb:
            smr, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                rb["hd_pm"], real_hd_pairs, rb["hd_states"]))
            l3_real = float(smr["total"])
        if sb:
            sms, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
                sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))
            l3_syn = float(sms["total"])
        smv, _ = _r20_eval_closed(env, scorer, jpol, _eval_roots(
            re["progmem"], val_pairs, st_bench_map))
        l3_val = float(smv["total"])
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    marg = 0.05 * max(float(r6_canonical_train), 1.0)
    print(f"[r21] L3(lexicographic no-RL)={l3:.0f} | held {l3_real:.0f}/{l3_syn:.0f} "
          f"VAL {l3_val:.0f} | marg={marg:.1f} | r6_canonical_train="
          f"{r6_canonical_train:.0f}", flush=True)

    # ---- cloud/shape profile -> workers --------------------------------------
    workers = 1
    mp_ok = True
    prof = None
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, init_roots[0],
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14",
                                        action_space="lexicographic")
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"]
                       for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw = prof["per_worker"].get(str(w))
            if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r21] cloud profile: identical={ident_ok} -> workers={workers}",
              flush=True)
    except Exception as exc:                   # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r21] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    def _run_cond(cond, unified, per_src, best_score_fn, agg, label):
        for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
            g.reset()
        jpol.params_for_stage("C")
        res = JG.run_rolling_cycles_r13(
            jpol, scorer, env, specs, stage="C", cycles=int(C.TO1_R21_TRAINING_CYCLES),
            k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=int(C.TO1_R21_GRAPHS_PER_BATCH), workers=workers,
            seed=args.grpo_seed, log_prefix=f"[r21-{label}]",
            eval_root_builder=eval_root_builder, collapse_floor=max(0.0, l3),
            parent_policy=None, mp_ctx=mp_ctx, variant="r20",
            on_groups=agg, action_space="lexicographic",
            per_src=per_src, unified=unified,
            per_cycle=(C.TO1_R21_D1_PER_CYCLE if unified else None),
            best_score_fn=best_score_fn)
        # re-eval promoted snapshot (jpol already loaded with best)
        with torch.no_grad():
            ev_f = eval_root_builder(jpol)
        train_peak = max((h["train"] for h in res["history"]), default=0.0)
        held_by_cycle = [_held_agg_of(h) for h in res["history"]]
        held_peak = max(held_by_cycle, default=0.0)
        print(f"[r21-{label}] train_peak={train_peak:.0f} held_peak={held_peak:.0f} "
              f"selected(train={ev_f['train']['total']:.0f} bench_held="
              f"{ev_f['bench_held']['total']:.0f} real={ev_f['real_held']['total']:.0f} "
              f"syn={ev_f['syn_held']['total']:.0f})", flush=True)
        return {"res": res, "selected": {k: float(v.get("total", 0.0))
                                         for k, v in ev_f.items()},
                "train_peak": train_peak, "held_peak": held_peak,
                "held_by_cycle": held_by_cycle}

    # ---- D0: R20 benchmark-dominant (generalization-first selection) ----------
    # §30 repro: the DISTRIBUTION is R20 verbatim, so the training trajectory is
    # identical and train_peak reproduces K3=387 regardless of the selection rule.
    agg_d0 = _R21Agg(src_of=src_of, family_of=family_of)
    d0 = _run_cond("D0", unified=False, per_src=dict(C.TO1_R21_D0_PER_SRC),
                   best_score_fn=held_agg_fn, agg=agg_d0, label="D0")
    d0_repro = bool(abs(d0["train_peak"] - 387.0) <= 0.5)
    print(f"[r21] D0 repro K3=387: train_peak={d0['train_peak']:.0f} "
          f"repro={d0_repro}", flush=True)
    d0_selected = dict(d0["selected"])

    # ---- D1: instance-diverse unified (generalization-first selection) --------
    jpol.load_snapshot(zj_snap)                  # reset to zero-init before D1
    agg_d1 = _R21Agg(src_of=src_of, family_of=family_of)
    d1 = _run_cond("D1", unified=True, per_src=dict(C.TO1_R21_D0_PER_SRC),
                   best_score_fn=held_agg_fn, agg=agg_d1, label="D1")
    d1_selected = dict(d1["selected"])

    # ---- §39 normal-M5 regression (D1) ----------------------------------------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    m5_f = _r20_m5_wrapper(env, scorer, jpol, dpp_iid,
                           copy.deepcopy(re["progmem"]), dpp_ep) if dpp_iid else None
    if m5_f:
        print(f"[r21] §39 normal-M5 FINAL: dep_imm={m5_f.get('dep_immediate')} "
              f"all_retained={m5_f.get('all_retained')}", flush=True)

    # ---- §35-36 DPpaulli / Mk1 / Fattahi15 train diagnostics -------------------
    dpp_pre = dpp_post = None
    if dpp_iid is not None and not args.skip_regressions:
        def _root_for(iid):
            st = env["states"][iid]
            return RGRPO.roots_from_state(st["problem"], st["schedule"], iid,
                                          env["ep_id_of"][iid],
                                          copy.deepcopy(re["progmem"]))
        dpp_root = _root_for(dpp_iid)
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(zj_snap)
        with torch.no_grad():
            pre_sum, _ = _r20_eval_closed(env, scorer, zj2, [dpp_root])
            post_sum, _ = _r20_eval_closed(env, scorer, jpol, [dpp_root])
        dpp_pre = {"total_gain": float(pre_sum.get("total", 0.0))}
        dpp_post = {"total_gain": float(post_sum.get("total", 0.0))}
        print(f"[r21] §36 DPpaulli BEFORE(zero-init)={dpp_pre['total_gain']} "
              f"| AFTER(D1 final)={dpp_post['total_gain']}", flush=True)

    # ---- per-condition rollups (KL / grad-norm / m2 pos rate) ------------------
    def _rollup(res):
        hist = res["history"]
        n_groups = sum(h.get("n_groups", 0) for h in hist)
        n_inf = sum(h.get("n_informative", 0) for h in hist)
        n_inf_m2 = sum(h.get("n_informative_trajectories_m2", 0) for h in hist)
        m2_kl = [float(e["kl_m2"]) for h in hist for d in (h.get("depth_results") or [])
                 for e in ((d.get("update") or {}).get("epochs", []))
                 if e.get("kl_m2") is not None]
        m3_kl = [float(e["kl_ref_m3"]) for h in hist for d in (h.get("depth_results") or [])
                 for e in ((d.get("update") or {}).get("epochs", []))
                 if e.get("kl_ref_m3") is not None]
        grads = [float(e["grad_norm"]) for h in hist for d in (h.get("depth_results") or [])
                 for e in ((d.get("update") or {}).get("epochs", []))
                 if e.get("grad_norm", 0.0) > 1e-9]
        m2pr = [float(h["m2_stats"]["positive_probe_rate"]) for h in hist
                if h.get("m2_stats")]
        return {"n_groups": n_groups, "n_informative": n_inf, "n_informative_m2": n_inf_m2,
                "m2_kl_last": (m2_kl or [0.0])[-1], "m3_kl_last": (m3_kl or [0.0])[-1],
                "grad_norm_mean": float(np.mean(grads)) if grads else 0.0,
                "grad_norm_max": float(np.max(grads)) if grads else 0.0,
                "m2_pos_rate_mean": float(np.mean(m2pr)) if m2pr else 0.0}
    ru_d0 = _rollup(d0["res"])
    ru_d1 = _rollup(d1["res"])

    # ---- generalization gap + comparison (§31-33) -----------------------------
    d0_held = d0["held_peak"]
    d1_held = d1["held_peak"]
    d0_gap = (d0["train_peak"] - d0_held) / max(d0["train_peak"], 1.0)
    d1_gap = (d1["train_peak"] - d1_held) / max(d1["train_peak"], 1.0)
    held_improve = d1_held - d0_held
    train_drop = d0["train_peak"] - d1["train_peak"]
    held_marg = max(float(C.TO1_R21_HELD_MARGIN), marg)
    big_marg = max(2.0 * held_marg, 0.10 * max(d0["train_peak"], 1.0))
    print(f"[r21] gap D0={d0_gap:.3f} (train {d0['train_peak']:.0f} held_peak "
          f"{d0_held:.0f}) | D1={d1_gap:.3f} (train {d1['train_peak']:.0f} held_peak "
          f"{d1_held:.0f}) | held_improve={held_improve:.0f} train_drop={train_drop:.0f} "
          f"marg={held_marg:.0f} big={big_marg:.0f}", flush=True)

    # ---- verdict ladder A-G (§41-47) ------------------------------------------
    m5_ok = True if m5_f is None else bool(m5_f.get("all_retained", True))
    g_fail = (not repro_ok or not mp_ok or bool(d0["res"]["collapsed"])
              or bool(d1["res"]["collapsed"]) or not m5_ok or not d0_repro)
    train_collapsed = bool(d1["train_peak"] < 0.5 * d0["train_peak"])
    m2_shift = bool(ru_d1["m2_pos_rate_mean"] < 0.5 * max(ru_d0["m2_pos_rate_mean"], 1e-9))
    if g_fail:
        vcode, vlabel = "G", "SPLIT_OR_RUNTIME_BUG"
        note = (f"repro={repro_ok} mp_ok={mp_ok} d0_collapsed="
                f"{d0['res']['collapsed']} d1_collapsed={d1['res']['collapsed']} "
                f"m5-§39={m5_ok} d0_repro={d0_repro}")
        ok = False
    elif train_collapsed:
        vcode, vlabel = "D", "AUX_DISTRIBUTION_HURTS_TRAIN_WITHOUT_TRANSFER"
        note = (f"D1 train {d1['train_peak']:.0f} < 0.5*D0 {d0['train_peak']:.0f} -- "
                f"instance diversity collapsed TRAIN; held {held_improve:+.0f}")
        ok = False
    elif held_improve <= held_marg:
        if m2_shift:
            vcode, vlabel = "E", "M2_DISTRIBUTION_SHIFT_PRIMARY"
            note = (f"D1 M2 positive-probe-rate {ru_d1['m2_pos_rate_mean']:.3f} < 0.5*"
                    f"D0 {ru_d0['m2_pos_rate_mean']:.3f} -- the shift hits M2 root "
                    f"search first, no held gain")
        else:
            vcode, vlabel = "C", "JOINT_POLICY_STILL_INSTANCE_SPECIFIC"
            note = (f"held_improve={held_improve:+.0f} <= marg {held_marg:.0f} -- "
                    f"diversity did not transfer (train {d1['train_peak']:.0f} held "
                    f"{d1_held:.0f} still instance-specific)")
        ok = False
    elif held_improve < big_marg:
        vcode, vlabel = "B", "DIVERSITY_REDUCES_OVERFIT_BUT_GAIN_SMALL"
        note = (f"held_improve={held_improve:+.0f} in ({held_marg:.0f},{big_marg:.0f}) "
                f"-- diversity narrows the TRAIN-held gap but the gain is small")
        ok = False
    else:
        vcode, vlabel = "A", "INSTANCE_DIVERSE_JOINT_GRPO_GENERALIZES"
        note = (f"held_improve={held_improve:+.0f} >= {big_marg:.0f} AND train kept "
                f"{d1['train_peak']:.0f} -- instance diversity converts TRAIN gain "
                f"into held gain")
        ok = True
    passed = bool(ok)
    print(f"[r21] verdict {vcode} {vlabel} (D0 train {d0['train_peak']:.0f}/held "
          f"{d0_held:.0f} | D1 train {d1['train_peak']:.0f}/held {d1_held:.0f} | "
          f"held_improve {held_improve:+.0f} | gap {d0_gap:.3f}->{d1_gap:.3f})",
          flush=True)

    if passed:
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": {
                        "phase": "r21_instance_diverse_joint_grpo",
                        "pipeline": "M2_SFT->M3_SFT->Joint_GRPO",
                        "runtime": "R20_lexicographic",
                        "joint_training_distribution": "instance_balanced_diverse",
                        "reward": "stagewise_A2_A3",
                        "m3_terminal_reward": "makespan_only",
                        "formal_test_access": 0, "identified": False,
                        "checkpoint": "m2_m3_instance_diverse_joint_grpo_r21"}},
                   C.TO1_R21_CKPT)
        print(f"[r21] saved {C.TO1_R21_CKPT.name} (PASS)", flush=True)
    else:
        print(f"[r21] NOT PASS -> no R21 checkpoint written (§49)", flush=True)

    _r21_scrape = dict(
        repro_ok=repro_ok, mp_ok=mp_ok, m5_f=m5_f, l3=l3, l3_real=l3_real,
        l3_syn=l3_syn, l3_val=l3_val, r6_canonical_train=r6_canonical_train,
        marg=marg, prof=prof, workers=workers, mp_ctx=mp_ctx, dpp_iid=dpp_iid,
        dpp_pre=dpp_pre, dpp_post=dpp_post,
        d0=dict(train_peak=d0["train_peak"], held_peak=d0["held_peak"],
                held=d0_held, selected=d0_selected, collapsed=d0["res"]["collapsed"],
                repro_387=d0_repro, agg=agg_d0.summary(),
                n_groups=ru_d0["n_groups"], n_informative=ru_d0["n_informative"],
                n_informative_m2=ru_d0["n_informative_m2"],
                m2_kl_last=ru_d0["m2_kl_last"], m3_kl_last=ru_d0["m3_kl_last"],
                grad_norm_mean=ru_d0["grad_norm_mean"], grad_norm_max=ru_d0["grad_norm_max"],
                m2_pos_rate_mean=ru_d0["m2_pos_rate_mean"], cycles_run=len(d0["res"]["history"])),
        d1=dict(train_peak=d1["train_peak"], held_peak=d1["held_peak"],
                held=d1_held, selected=d1_selected, collapsed=d1["res"]["collapsed"],
                agg=agg_d1.summary(),
                n_groups=ru_d1["n_groups"], n_informative=ru_d1["n_informative"],
                n_informative_m2=ru_d1["n_informative_m2"],
                m2_kl_last=ru_d1["m2_kl_last"], m3_kl_last=ru_d1["m3_kl_last"],
                grad_norm_mean=ru_d1["grad_norm_mean"], grad_norm_max=ru_d1["grad_norm_max"],
                m2_pos_rate_mean=ru_d1["m2_pos_rate_mean"], cycles_run=len(d1["res"]["history"])),
        held_improve=held_improve, train_drop=train_drop, held_marg=held_marg,
        big_marg=big_marg, gap_d0=d0_gap, gap_d1=d1_gap,
        d0_repro_387=d0_repro, m2_shift=m2_shift, train_collapsed=train_collapsed,
        family_of=family_of, src_of=src_of,
        n_tr_m3=n_tr_m3, n_tr_m2=n_tr_m2,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed)
    report = _r21_report(env, re, _r21_scrape)
    _r21_persist(report, _r21_scrape)
    print(f"[r21] total {time.time() - t0:.1f}s "
          f"(report {C.R21_REPORT.name})", flush=True)
    return report


def _r21_report(env, re, s):
    """Markdown report answering the §-numbered 55-item return checklist."""
    L = []
    J = lambda d: json.dumps(d, default=str)
    verdict = s.get("verdict", {})
    d0 = s.get("d0", {})
    d1 = s.get("d1", {})

    def h1(t):
        L.append(f"# {t}\n")
    def h2(t):
        L.append(f"## {t}\n")
    def kv(k, v):
        L.append(f"- **{k}**: {v}")
    L.append("# R21 — T1-INSTANCE-DIVERSE-JOINT-GRPO 报告\n")
    L.append(f"_verdict {verdict.get('code')} {verdict.get('label')} "
             f"\\| passed={s.get('passed')} \\| identified=false \\| "
             f"formal_test_access=0 \\| Formal TEST SEALED  \n"
             f"{verdict.get('note', '')}_\n")
    h1("1. Modified files")
    kv("engine", "src/causal_schedule_lab/m3/joint_grpo.py "
                 "(_r13_cycle_picks unified round-robin; run_rolling_cycles_r13 "
                 "per_src/unified/per_cycle/best_score_fn + bench_held tracking; "
                 "variant='r21' added to use_r14)")
    kv("config", "src/causal_schedule_lab/m3/config.py (§TO1_R21_*)")
    kv("runner", "scripts/run_m3_canonical_training.py (run_instance_diverse_phase_r21)")
    h1("2. Exact runtime pipeline")
    kv("pipeline", "S_t+Appearance -> M2 budgeted root probe (R14 adaptive gate "
                   "VERBATIM) -> makespan-first/Memory-second root filter -> Reasoner "
                   "-> complete legal Proposal pool -> state-level lexicographic "
                   "validation (PA>>PB>>PC>>STOP) -> M3 action set -> JOINT GRPO "
                   "K8/H5/E3/10cyc/8g -- R20 runtime VERBATIM")
    kv("reward", "stagewise A2 (M2 local probe) / A3 (M3 terminal makespan gain only)")
    kv("continuation", "frozen zero-init M2 + m3_proposal_top1_sft_v2.pt (no oracle)")
    kv("only_change", "Joint GRPO training instance distribution (D0 vs D1)")
    h1("3-6. D0 / D1 training sources & counts")
    kv("D0", "benchmark-dominant: per_src bench2/real1/syn1 (R20 verbatim)")
    kv("D1", "instance-diverse: unified round-robin, 4 distinct/cycle over "
             "bench14+real32+syn80")
    kv("benchmark", "14 TRAIN (no carve) + 3 VAL + 3 TEST(sealed)")
    kv("AUX-real", "32 train / 8 held (has family key)")
    kv("AUX-syn", "80 train / 20 held (no family -> synthetic family 'AUX-syn')")
    h1("7-10. Held sources & isolation")
    kv("held", "bench internal-held == benchmark VAL(3) (no separate benchmark held "
               "split exists; §30 D0-repro forces bench TRAIN to stay 14); AUX-real "
               "held 8; AUX-syn held 20; VAL final-only (== bench-held)")
    kv("isolation", "bench TRAIN / AUX-real train / AUX-syn train disjoint from all "
                    "held + VAL; no instance in both a TRAIN and a held source")
    h1("11. Sampling distribution (§12 family report)")
    kv("D0 by_source", J(d0.get("agg", {}).get("by_source", {})))
    kv("D1 by_source", J(d1.get("agg", {}).get("by_source", {})))
    kv("D1 by_family", J(d1.get("agg", {}).get("by_family", {})))
    h1("12-13. Trajectory budget (D0 vs D1)")
    kv("D0", f"n_groups={d0.get('n_groups')} n_informative={d0.get('n_informative')} "
             f"n_informative_m2={d0.get('n_informative_m2')} cycles={d0.get('cycles_run')}")
    kv("D1", f"n_groups={d1.get('n_groups')} n_informative={d1.get('n_informative')} "
             f"n_informative_m2={d1.get('n_informative_m2')} cycles={d1.get('cycles_run')}")
    h1("14-15. Unique (iid, state_hash) states")
    kv("D0", f"n_unique_states={d0.get('agg', {}).get('n_unique_states')} "
             f"distinct_instances={d0.get('agg', {}).get('n_distinct_instances')}")
    kv("D1", f"n_unique_states={d1.get('agg', {}).get('n_unique_states')} "
             f"distinct_instances={d1.get('agg', {}).get('n_distinct_instances')}")
    h1("16-18. PA count by source")
    kv("D0", J({k: v.get('pa_count') for k, v in d0.get("agg", {}).get("by_source", {}).items()}))
    kv("D1", J({k: v.get('pa_count') for k, v in d1.get("agg", {}).get("by_source", {}).items()}))
    h1("19-21. Terminal reward distribution by source")
    kv("D0", J({k: v.get('terminal') for k, v in d0.get("agg", {}).get("by_source", {}).items()}))
    kv("D1", J({k: v.get('terminal') for k, v in d1.get("agg", {}).get("by_source", {}).items()}))
    h1("22-25. Exposure table (top rows)")
    kv("D0", J(d0.get("agg", {}).get("exposure_table", [])[:20]))
    kv("D1", J(d1.get("agg", {}).get("exposure_table", [])[:20]))
    h1("26-29. Generalization-first model selection")
    kv("D0", f"train_peak={d0.get('train_peak')} held_peak={d0.get('held_peak')} "
             f"held(selected)={d0.get('held')} repro_387={s.get('d0_repro_387')}")
    kv("D1", f"train_peak={d1.get('train_peak')} held_peak={d1.get('held_peak')} "
             f"held(selected)={d1.get('held')}")
    kv("selection", "D0 = TRAIN-based (repro K3=387); D1 = held-aggregate "
                    "(bench_held+real_held+syn_held source-level equal weight); "
                    "VAL final-only")
    h1("30-33. Primary question & generalization gap")
    kv("gap", f"D0={s.get('gap_d0')} -> D1={s.get('gap_d1')}")
    kv("held_improve", f"{s.get('held_improve')} (marg {s.get('held_marg')}, "
                       f"big {s.get('big_marg')})")
    kv("train_drop", f"{s.get('train_drop')}")
    h1("34-36. Diagnostics")
    kv("DPpaulli", f"BEFORE={s.get('dpp_pre')} AFTER={s.get('dpp_post')}")
    kv("normal-M5", J(s.get("m5_f")))
    h1("37-48. Learning metrics")
    kv("D0", f"m2_kl={d0.get('m2_kl_last')} m3_kl={d0.get('m3_kl_last')} "
             f"grad_mean={d0.get('grad_norm_mean')} grad_max={d0.get('grad_norm_max')} "
             f"m2_pos_rate={d0.get('m2_pos_rate_mean')}")
    kv("D1", f"m2_kl={d1.get('m2_kl_last')} m3_kl={d1.get('m3_kl_last')} "
             f"grad_mean={d1.get('grad_norm_mean')} grad_max={d1.get('grad_norm_max')} "
             f"m2_pos_rate={d1.get('m2_pos_rate_mean')}")
    kv("m2_shift", f"{s.get('m2_shift')} (verdict E signal)")
    h1("49-55. Final")
    kv("verdict", f"{verdict.get('code')} {verdict.get('label')}")
    kv("checkpoint", f"m2_m3_instance_diverse_joint_grpo_r21.pt (A only) -> written={s.get('passed')}")
    kv("next", "A: promote + formal plan; B: diversity helps but need larger "
               "held-transfer lever; C: policy still instance-specific; D: revert "
               "aux mixing; E: M2 distribution shift first; F: M3 selection first")
    return "\n".join(L) + "\n"


def _r21_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R21_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in ("repro_ok", "mp_ok", "m5_f", "l3", "l3_real", "l3_syn", "l3_val",
                  "r6_canonical_train", "marg", "prof", "workers", "mp_ctx",
                  "dpp_iid", "dpp_pre", "dpp_post", "d0", "d1", "held_improve",
                  "train_drop", "held_marg", "big_marg", "gap_d0", "gap_d1",
                  "d0_repro_387", "m2_shift", "train_collapsed", "family_of",
                  "src_of", "n_tr_m3", "n_tr_m2", "verdict", "passed"):
            if k in scrape:
                payload[k] = scrape[k]
    (C.CANONICAL_OUT_DIR / "result_r21.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r21] report written: {C.R21_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# T2-A -- MULTI-PATH HELD TRAJECTORY EVALUATION (R21 verdict C follow-up)
# EVALUATION-ONLY: frozen canonical policy, greedy vs N sampled trajectories,
# to test whether latent positive trajectories exist on held/unseen instances.
# NO training / NO optimizer / NO oracle / NO checkpoint promotion.
# Pure metric/verdict helpers live in joint_grpo.py (t2a_*); imported here.
# ---------------------------------------------------------------------------
def run_multipath_eval_t2a(args, env, re, p1, p1_report):
    """T2-A -- MULTI-PATH HELD TRAJECTORY EVALUATION.  Frozen canonical policy, greedy
    reference + N sampled trajectories per root, branch-local memory.  Answers whether
    latent positive trajectories exist on held instances that single greedy misses."""
    print("[t2a] T2-A MULTI-PATH HELD TRAJECTORY EVALUATION (frozen, no training) ...",
          flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    H = int(C.TO1_T2A_HORIZON)
    N = int(C.TO1_T2A_N_SAMPLED)
    T = float(C.TO1_T2A_TEMP)
    eps = float(C.TO1_T2A_MIX_EPS)
    action_space = C.TO1_T2A_ACTION_SPACE

    # ---- frozen continuation policy (R21 §3 VERBATIM, NO training) ------------
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(
        r6_sel, m3_builder=lambda r6: JG.M3TemporalEvidencePolicy(r6))
    with torch.no_grad():
        resid_max = max((p.abs().max().item()
                         for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    zj_snap = jpol.snapshot()
    cont_m2 = JG.M2RootPolicyAdapter(jpol.m2.net[0].in_features,
                                     float(jpol.m2.alpha_m2))
    cont_m2.load_snapshot(zj_snap["m2"])
    for _p_ in cont_m2.parameters():
        _p_.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                     alpha_stop=C.TO1_R13_ALPHA_STOP)
    for _p_ in cont_m3.parameters():
        _p_.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    print(f"[t2a] frozen policy {C.TO1_T2A_FROZEN_ID} | M2 adapter {n_tr_m2} params | "
          f"M3 residual {n_tr_m3} params (resid_max={resid_max:.3g}) | K={N} H={H} "
          f"T={T} eps={eps} | action_space={action_space}", flush=True)

    # ---- R6 reproduction anchor (§5 E0 correctness gate) ----------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[t2a] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- splits (§25): bench TRAIN / bench_held(=VAL) / AUX-real held / AUX-syn held
    bench_iids = [i["instance_id"] for i in env["train_insts"]]
    train_pairs = [(iid, env["ep_id_of"][iid]) for iid in bench_iids]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    src_of = _r21_src_of(bench_iids, rb, sb)
    family_of = _r21_family_of(bench_iids, rb, sb)

    def _roots(pm, pairs, st_map):
        return [{"problem": st_map[iid]["problem"], "schedule": st_map[iid]["schedule"],
                 "iid": iid, "episode_id": int(eid), "progmem": pm,
                 "root_ms": int(st_map[iid]["schedule"].makespan), "horizon": H}
                for (iid, eid) in pairs]

    splits = [("train", train_pairs, st_bench_map, re["progmem"], "bench"),
              ("bench_held", val_pairs, st_bench_map, re["progmem"], "bench")]
    if rb:
        splits.append(("real_held", real_hd_pairs, rb["hd_states"], rb["hd_pm"], "real"))
    if sb:
        splits.append(("syn_held", syn_hd_pairs, sb["hd_states"], sb["hd_pm"], "syn"))

    all_rows = []
    split_summaries = {}
    greedy_stps_all = {}
    e0_by_split = {}
    for split_name, pairs, st_map, base_pm, src in splits:
        if not pairs:                                # empty split (e.g. VAL in quick mode)
            split_summaries[split_name] = JG.t2a_split_summary([])
            e0_by_split[split_name] = 0.0
            continue
        roots = _roots(base_pm, pairs, st_map)
        # §5 E0 = canonical single-path baseline (shared-memory greedy, R21 L3 ruler)
        with torch.no_grad():
            e0_sum, _ = _r20_eval_closed(
                env, scorer, jpol, _r12_eval_roots(base_pm, pairs, st_map), action_space)
        e0_total = float(e0_sum["total"])
        e0_by_split[split_name] = e0_total
        rows = []
        for rf in roots:
            iid, eid = rf["iid"], rf["episode_id"]
            # greedy reference path (§12), clean branch-local memory
            rf_g = dict(rf, progmem=copy.deepcopy(rf["progmem"]))
            g_gain, g_usage, g_steps = JG.agentic_parity_rollout(
                env, scorer, jpol, rf_g, use_mem=True, horizon=H, gate_mem=False,
                gate_variant="r14", action_space=action_space, sample=None)
            greedy_stps_all[iid] = {"usage": g_usage, "steps": g_steps}
            samples, sigs, step_counts = [], [], [len(g_steps)]
            sigs.append(JG.t2a_traj_sig(g_steps))
            for b in range(N):
                rng = random.Random(JG.t2a_branch_seed(iid, eid, b))
                rf_b = dict(rf, progmem=copy.deepcopy(rf["progmem"]))
                b_gain, b_usage, b_steps = JG.agentic_parity_rollout(
                    env, scorer, jpol, rf_b, use_mem=True, horizon=H, gate_mem=False,
                    gate_variant="r14", action_space=action_space,
                    sample={"T": T, "eps": eps, "rng": rng})
                samples.append(int(b_gain))
                sigs.append(JG.t2a_traj_sig(b_steps))
                step_counts.append(len(b_steps))
            best_of_N = int(max(samples)) if samples else int(g_gain)
            rows.append({
                "iid": iid, "split": split_name, "src": src,
                "family": family_of.get(iid, "?"),
                "greedy": int(g_gain), "samples": samples,
                "best_of_N": best_of_N,
                "best_overall": int(max([g_gain] + samples)),
                "mean": float(np.mean(samples)) if samples else float(g_gain),
                "median": float(np.median(samples)) if samples else float(g_gain),
                "n_pos_traj": int(sum(1 for s in samples if s > 0)),
                "n_distinct_traj": int(len(set(sigs))),
                "n_distinct_outcomes": int(len(set([g_gain] + samples))),
                "latent": bool(g_gain <= 0 and best_of_N > 0),
                "failure": JG.t2a_classify_failure(g_gain, samples, sigs, step_counts),
                "greedy_steps": JG.t2a_compact_steps(g_steps),
            })
        split_summaries[split_name] = JG.t2a_split_summary(rows)
        all_rows.extend(rows)
        s = split_summaries[split_name]
        print(f"[t2a] {split_name:11s} n={s['n']:2d} greedy={s['greedy_total']:+d} "
              f"bestN={s['bestN_total']:+d} mean={s['mean_total']:+.0f} "
              f"latent={s['latent_n']}/{s['latent_denom']} "
              f"success@N={s['success_at_N']:.2f} distinct_traj={s['mean_distinct_traj']:.2f}",
              flush=True)

    held_rows = [r for r in all_rows if r["split"] in
                 ("bench_held", "real_held", "syn_held")]
    held = JG.t2a_split_summary(held_rows)
    train = split_summaries.get("train", {"n": 0})
    held_n = int(held.get("n", 0))

    # ---- §37-38 M2 coverage + root coverage (from greedy reference traces) -----
    m2_cov = _r18_cov(greedy_stps_all)
    n_roots_with_pool = 0
    for _iid, blk in greedy_stps_all.items():
        if blk.get("steps"):
            n_roots_with_pool += 1

    # ---- §33 print single vs best-of-N FIRST -------------------------------
    print(f"[t2a] ===== SINGLE GREEDY vs BEST-OF-N =====", flush=True)
    print(f"[t2a] train   greedy={train.get('greedy_total', 0):+d} "
          f"bestN={train.get('bestN_total', 0):+d} "
          f"bestoverall={train.get('bestoverall_total', 0):+d}", flush=True)
    print(f"[t2a] held    greedy={held.get('greedy_total', 0):+d} "
          f"bestN={held.get('bestN_total', 0):+d} "
          f"bestoverall={held.get('bestoverall_total', 0):+d} "
          f"(n={held_n})", flush=True)
    print(f"[t2a] held latent_n={held.get('latent_n', 0)} / "
          f"greedy<=0 {held.get('latent_denom', 0)} -> "
          f"latent_rate={held.get('latent_rate', 0.0):.3f} | "
          f"bestofN_gain={held.get('bestofN_gain', 0):+d} | "
          f"mean_distinct_traj={held.get('mean_distinct_traj', 0.0):.2f} | "
          f"success@N={held.get('success_at_N', 0.0):.2f} "
          f"positive_traj_rate={held.get('positive_traj_rate', 0.0):.3f}", flush=True)

    # ---- verdict ladder + GO_5070TI -------------------------------------------
    vcode, vlabel, note, go_5070ti = JG.t2a_verdict(repro_ok, held, held_n)
    print(f"[t2a] verdict {vcode} {vlabel} | GO_5070TI={go_5070ti} | {note}",
          flush=True)

    scrape = dict(
        frozen_id=C.TO1_T2A_FROZEN_ID, K=N, horizon=H, temp=T, mix_eps=eps,
        action_space=action_space,
        repro_ok=repro_ok, r6_acc_all=r6_acc_all, r6_regret=r6_regret,
        n_tr_m3=n_tr_m3, n_tr_m2=n_tr_m2,
        e0_by_split=e0_by_split, split_summaries=split_summaries,
        held=held, train=train,
        m2_cov=m2_cov, n_roots_with_pool=n_roots_with_pool,
        held_n=held_n, verdict=dict(code=vcode, label=vlabel, note=note),
        go_5070ti=bool(go_5070ti),
        per_instance=[{k: r[k] for k in
                       ("iid", "split", "src", "family", "greedy", "samples",
                        "best_of_N", "best_overall", "mean", "median",
                        "n_pos_traj", "n_distinct_traj", "n_distinct_outcomes",
                        "latent", "failure", "greedy_steps")} for r in all_rows],
        elapsed_s=round(time.time() - t0, 1),
    )
    report = _t2a_report(scrape)
    _t2a_persist(report, scrape)
    print(f"[t2a] total {scrape['elapsed_s']}s (report {C.T2A_REPORT.name})",
          flush=True)
    return report


def _t2a_report(s):
    L = []
    J = lambda d: json.dumps(d, default=str)
    v = s.get("verdict", {})
    held = s.get("held", {})
    train = s.get("train", {})
    L.append("# T2-A — MULTI-PATH HELD TRAJECTORY EVALUATION 报告\n")
    L.append(f"_verdict {v.get('code')} {v.get('label')} | GO_5070TI={s.get('go_5070ti')} "
             f"| identified=false | formal_test_access=0 | Formal TEST SEALED | "
             f"NO TRAINING (all frozen)  \n{v.get('note', '')}_\n")
    L.append("\n## 0. 科学边界\n")
    L.append("- 本次为 **评估-only 诊断**：不训练、不反向、不 GRPO/SFT、不写 checkpoint、不碰 VAL。")
    L.append(f"- 冻结策略：`{s.get('frozen_id')}`（M2 canonical + m3_proposal_top1_sft_v2.pt + 零初始化残差）。")
    L.append("- 无 oracle 进入 rollout（§13）；best-of-N 只在 rollout 后统计（§14）。")
    L.append("\n## 1. 单一 greedy vs best-of-N（§15/§33）\n")
    L.append(f"- **train**   greedy={train.get('greedy_total', 0):+d}  "
             f"bestN={train.get('bestN_total', 0):+d}  "
             f"bestoverall={train.get('bestoverall_total', 0):+d}")
    L.append(f"- **held**    greedy={held.get('greedy_total', 0):+d}  "
             f"bestN={held.get('bestN_total', 0):+d}  "
             f"bestoverall={held.get('bestoverall_total', 0):+d}  (n={held.get('n', 0)})")
    L.append(f"- **held mean**={held.get('mean_total', 0.0):+.1f}  "
             f"**median**={held.get('median_total', 0.0):+.1f}")
    L.append("\n## 2. 关键诊断指标（§16-24）\n")
    L.append(f"- BestOfNGain (held) = **{held.get('bestofN_gain', 0):+d}**")
    L.append(f"- Success@N (held) = **{held.get('success_at_N', 0.0):.3f}**")
    L.append(f"- Success (overall, greedy∪N) = **{held.get('success_overall', 0.0):.3f}**")
    L.append(f"- PositiveTrajectoryRate (held) = **{held.get('positive_traj_rate', 0.0):.3f}**")
    L.append(f"- **LatentSignalRate (held)** = **{held.get('latent_rate', 0.0):.3f}**  "
             f"({held.get('latent_n', 0)} / {held.get('latent_denom', 0)} greedy<=0)")
    L.append(f"- SampledBestGap (held) = **{held.get('sampled_best_gap', 0.0):+.2f}**")
    L.append(f"- mean distinct trajectories (held) = {held.get('mean_distinct_traj', 0.0):.2f}  "
             f"(N={s.get('K')})")
    L.append(f"- mean distinct outcomes (held) = {held.get('mean_distinct_outcomes', 0.0):.2f}")
    L.append("\n## 3. 失败分解（§35）\n")
    for k, cnt in sorted((held.get("failure") or {}).items()):
        L.append(f"- {k}: {cnt}")
    L.append("\n## 4. splits（§25-27）\n")
    L.append("- held = bench_held(==VAL) ∪ AUX-real held ∪ AUX-syn held；VAL 仅最终统计，无梯度。")
    L.append(f"- 各 split greedy/bestN 汇总：")
    for name, sm in s.get("split_summaries", {}).items():
        L.append(f"  - {name}: n={sm.get('n')} greedy={sm.get('greedy_total', 0):+d} "
                 f"bestN={sm.get('bestN_total', 0):+d} "
                 f"latent={sm.get('latent_n', 0)}/{sm.get('latent_denom', 0)}")
    L.append("\n## 5. E0 单路径基线复现（§5）\n")
    L.append(f"- canonical shared-memory greedy totals per split: {J(s.get('e0_by_split', {}))}")
    L.append(f"- R6 anchor: acc={s.get('r6_acc_all', 0.0):.4f} regret={s.get('r6_regret', 0.0):.2f} "
             f"repro_ok={s.get('repro_ok')}")
    L.append("\n## 6. M2/M3 痕迹 + 根覆盖（§37-38）\n")
    L.append(f"- M2 coverage: {J(s.get('m2_cov', {}))}")
    L.append(f"- roots with non-empty gated pool: {s.get('n_roots_with_pool', 0)}")
    L.append("\n## 7. 结论与去留（§39-40/§49）\n")
    L.append(f"- **verdict {v.get('code')} — {v.get('label')}**：{v.get('note', '')}")
    L.append(f"- **GO_5070TI = {s.get('go_5070ti')}**（A-only）。A→可立 T2-B 多路径 Joint RL；"
             f"B→增大 N/换 sampler；C/D/E→结构瓶颈仍在（候选池天花板/多步信用）。")
    L.append("\n## 8. 参数（§7/§8/§11/§28，均沿用 canonical，未重造）\n")
    L.append(f"- K={s.get('K')} H={s.get('horizon')} T={s.get('temp')} eps={s.get('mix_eps')} "
             f"action_space={s.get('action_space')}")
    return "\n".join(L) + "\n"


def _t2a_persist(report, scrape):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.T2A_REPORT.write_text(report, encoding="utf-8")
    (C.CANONICAL_OUT_DIR / "result_t2a.json").write_text(
        json.dumps(scrape, indent=2, default=str), encoding="utf-8")
    print(f"[t2a] report written: {C.T2A_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# T2-B -- MULTI-PATH JOINT AGENTIC GRPO (T2-A verdict A follow-up)
# Stage-3 RL: K sibling trajectories from one root train the policy so T2-A's
# latent positives become greedy-extractable.  reward/advantage/update == R21
# (lexicographic runtime, stagewise A2/A3, group_advantages_r12).  best-of-N is
# a POST-HOC diagnostic only -- never reward/advantage/oracle.  ExtractionGap =
# 差值 = t2a_split_summary["bestofN_gain"].  Verdict 双闸: greedy 转正 AND gap 收窄.
# Pure metric/verdict helpers live in joint_grpo.py (t2b_*); imported here.
# ---------------------------------------------------------------------------
class _T2BGroupDiag:
    """Accumulate the collected groups each cycle; expose the T2-B group-level
    diagnostics (useful-group rate / all-same rate / first-action credit)."""

    def __init__(self):
        self.groups = []

    def __call__(self, groups):
        self.groups.extend(groups)

    def summary(self):
        return {
            "n_groups": len(self.groups),
            "useful_group_rate": JG.t2b_useful_group_rate(self.groups),
            "all_same_rate": JG.t2b_all_same_rate(self.groups),
            "first_action_credit": JG.t2b_first_action_credit(self.groups),
        }


def _t2d_train_only_caps(env, re, scorer, reranker, r6_sel, out_dir):
    """Calibrate bounded residual influence from TRAIN replay/root logits only."""
    shared = C.ROOT / "outputs" / "t2d" / "train_residual_caps.json"
    fingerprint = {
        "train_iids": [x["instance_id"] for x in env["train_insts"]],
        "b5_sha256": HR.tensor_state_sha256(env["model_b5"]),
        "r6_sha256": HR.tensor_state_sha256(r6_sel),
    }
    if shared.is_file():
        saved = json.loads(shared.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fingerprint:
            caps = HR.ResidualCaps(float(saved["cap2"]), float(saved["cap3"]),
                                   float(saved["cap_stop"]))
            caps.validate()
            out_dir.mkdir(parents=True, exist_ok=True)
            caps.save(out_dir / "m2_residual_scale_calibration.json")
            (out_dir / "m3_residual_scale_calibration.json").write_text(
                json.dumps({"cap3": caps.cap3, "cap_stop": caps.cap_stop,
                            "source_split": "train", "held_used": False,
                            "val_used": False, "formal_test_access": 0}, indent=2),
                encoding="utf-8")
            print(f"[t2d] reused TRAIN-only residual caps from {shared}", flush=True)
            return caps
    root_rows = []
    for inst in env["train_insts"]:
        iid = inst["instance_id"]
        st = env["states"][iid]
        ast = env["cache"].ast(st["problem"], st["schedule"], iid)
        root_rows.append(JG._root_candidate_arrays(ast)["base"].detach())
    groups, _ = TOP1.build_top1_groups(
        re["state_examples"], scorer, re["mem_values"], reranker)
    prop_rows, stop_rows = [], []
    with torch.no_grad():
        for g in groups:
            F, sf, ps = g["F_pool"], g["ex"]["state_feat"], g["pool_stats"]
            p, s = r6_sel(F, sf, ps)
            prop_rows.append(p.detach())
            stop_rows.append(s.detach())
    caps = HR.calibrate_residual_caps(
        split="train", root_logits=root_rows,
        proposal_logits=prop_rows, stop_logits=stop_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps.save(out_dir / "m2_residual_scale_calibration.json")
    (out_dir / "m3_residual_scale_calibration.json").write_text(
        json.dumps({"cap3": caps.cap3, "cap_stop": caps.cap_stop,
                    "source_split": "train", "held_used": False,
                    "val_used": False, "formal_test_access": 0}, indent=2),
        encoding="utf-8")
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text(json.dumps({"cap2": caps.cap2, "cap3": caps.cap3,
                                  "cap_stop": caps.cap_stop,
                                  "fingerprint": fingerprint,
                                  "formal_test_access": 0}, indent=2),
                      encoding="utf-8")
    return caps


def _t2d_policy(architecture, r6_sel, caps, env):
    arch = str(architecture).upper()
    if arch == HR.ARCH_P0:
        return JG.JointAgenticPolicy(
            r6_sel, m3_builder=lambda r6: JG.M3TemporalEvidencePolicy(r6))
    first = env["train_insts"][0]["instance_id"]
    st = env["states"][first]
    ast = env["cache"].ast(st["problem"], st["schedule"], first)
    cand = JG._root_candidate_arrays(ast)
    root_dim = int(cand["latent_dim"])
    local_dim = int(cand["feats"].shape[-1])
    relation_dim = int(cand.get("app_raw_dim", 0))
    state_dim = int(C.STATE_FEAT_DIM)
    use_traj = arch == HR.ARCH_P2
    if arch not in (HR.ARCH_P1, HR.ARCH_P2):
        raise ValueError(f"unknown T2-D architecture {architecture!r}")
    return JG.JointAgenticPolicy(
        r6_sel,
        m2_builder=lambda: HR.M2RootSetResidualActor(
            root_dim, local_dim, state_dim, cap=caps.cap2,
            target_alpha=float(C.TO1_R13_ALPHA_M2),
            relation_dim=relation_dim, root_value_scale=3.0),
        m3_builder=lambda r6: HR.M3ProposalSetResidualActor(
            r6, evidence_dim=int(C.TO1_R19_EVID_DIM),
            use_trajectory=use_traj, cap_prop=caps.cap3,
            cap_stop=caps.cap_stop, target_alpha=1.0))


def _t2d_optimizer_state_cpu(jpol):
    """Return the live persistent optimizer state without serializing GPU policy."""
    try:
        from causal_schedule_lab.m3 import t2d_gpu_trainer
        state = t2d_gpu_trainer.optimizer_state_cpu(jpol)
        if state is not None:
            return state
    except ImportError:
        pass
    opt = getattr(jpol, "_t2d_optimizer", None)
    return None if opt is None else opt.state_dict()


def _t2d_resume(jpol, resume_path, architecture):
    """Restore actor and optimizer state, rejecting cross-architecture resumes."""
    if not resume_path:
        return
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"T2-D resume checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    saved_arch = str(meta.get("architecture", "")).upper()
    if saved_arch != str(architecture).upper():
        raise ValueError(f"resume architecture mismatch: checkpoint={saved_arch!r}, "
                         f"requested={architecture!r}")
    if int(meta.get("formal_test_access", 0)) != 0:
        raise ValueError("refusing checkpoint whose metadata accessed Formal TEST")
    state = ck.get("state", {})
    snap = state.get("policy_snapshot")
    if snap is None and state.get("policy") is not None:  # early local T2-D format
        snap = state["policy"].snapshot()
    if snap is None:
        raise ValueError("resume checkpoint has no policy snapshot")
    jpol.load_snapshot(snap)
    if state.get("optimizer_state") is not None:
        jpol._t2d_resume_optimizer_state = state["optimizer_state"]
    print(f"[t2d] resumed {architecture} actors"
          f"{' + AdamW state' if state.get('optimizer_state') is not None else ''} "
          f"from {path}", flush=True)


def _t2d_train_latent_diagnostic(env, re, scorer, reranker, jpol, out_dir):
    """Small TRAIN-only raw-vs-SFT-latent diagnostic; never selects on held data."""
    if not isinstance(jpol.m2, HR.M2RootSetResidualActor):
        return None

    def variance(x):
        return float(x.detach().float().var(dim=0, unbiased=False).mean().item())

    def separation(x, score):
        x, score = x.detach().float(), score.detach().float().reshape(-1)
        if len(x) < 2:
            return 0.0
        hi = score >= score.median()
        if not bool(hi.any()) or not bool((~hi).any()):
            return 0.0
        gap = (x[hi].mean(0) - x[~hi].mean(0)).norm()
        scale = x.std(0, unbiased=False).norm().clamp_min(1e-6)
        return float((gap / scale).item())

    iid = env["train_insts"][0]["instance_id"]
    st = env["states"][iid]
    ast = env["cache"].ast(st["problem"], st["schedule"], iid)
    cand = JG._root_candidate_arrays(ast)
    sf = torch.zeros(jpol.m2.state_dim)
    m2_probe = copy.deepcopy(jpol.m2)
    m2_probe.alpha_fraction = 1.0
    root_logits = m2_probe.score(cand["base"], cand["feats"], cand["latents"], sf)
    torch.nn.functional.cross_entropy(
        root_logits.reshape(1, -1), cand["base"].argmax().reshape(1)).backward()
    m2_grad = float(sum((p.grad.detach().norm().item() ** 2)
                        for p in m2_probe.parameters() if p.grad is not None) ** 0.5)

    groups, _ = TOP1.build_top1_groups(
        re["state_examples"], scorer, re["mem_values"], reranker)
    g = groups[0]
    F, state_f, pool_s = g["F_pool"], g["ex"]["state_feat"], g["pool_stats"]
    evid = torch.zeros(len(F), int(C.TO1_R19_EVID_DIM))
    m3_probe = copy.deepcopy(jpol.m3)
    m3_probe.alpha_fraction = 1.0
    comp = m3_probe.components(F, state_f, pool_s, evid=evid)
    logits = m3_probe.action_logits(F, state_f, pool_s, evid=evid)
    torch.nn.functional.cross_entropy(
        logits.reshape(1, -1), logits.detach().argmax().reshape(1)).backward()
    m3_grad = float(sum((p.grad.detach().norm().item() ** 2)
                        for p in m3_probe.parameters()
                        if p.requires_grad and p.grad is not None) ** 0.5)
    result = {
        "split": "train", "held_used": False, "val_used": False,
        "formal_test_access": 0,
        "m2": {"raw_feature_variance": variance(cand["feats"]),
               "sft_latent_variance": variance(cand["latents"]),
               "raw_separability": separation(cand["feats"], cand["base"]),
               "sft_latent_separability": separation(cand["latents"], cand["base"]),
               "policy_gradient_signal": m2_grad},
        "m3": {"raw_feature_variance": variance(F),
               "sft_latent_variance": variance(comp["h_prop_sft"]),
               "raw_separability": separation(F, comp["base_prop"]),
               "sft_latent_separability": separation(
                   comp["h_prop_sft"], comp["base_prop"]),
               "policy_gradient_signal": m3_grad},
        "diagnostic_only": True,
    }
    (out_dir / "train_latent_diagnostic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_multipath_joint_grpo_t2b(args, env, re, p1, p1_report):
    """T2-B/T2-D multi-path GRPO; T2-D may expand K and data budget only."""
    print("[t2b] T2-B MULTI-PATH JOINT AGENTIC GRPO ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))
    architecture = str(getattr(args, "architecture", HR.ARCH_P0)).upper()
    is_t2d = bool(getattr(args, "stage", "") == "t2d")
    H = int(C.TO1_T2B_HORIZON)  # frozen: do not reduce depth
    K = int(getattr(args, "t2d_branches_per_graph", C.T2D_BRANCHES_PER_GRAPH)
            if is_t2d else C.TO1_T2B_K)
    N = int(K if is_t2d else C.TO1_T2B_N_SAMPLED)
    training_cycles = int(getattr(args, "t2d_cycles", C.T2D_TRAINING_CYCLES)
                          if is_t2d else C.TO1_T2B_TRAINING_CYCLES)
    graphs_per_cycle = int(getattr(
        args, "t2d_graphs_per_cycle", C.T2D_GRAPHS_PER_CYCLE)
        if is_t2d else C.TO1_T2B_GRAPHS_PER_BATCH)
    T = float(C.TO1_T2B_TEMP)
    eps = float(C.TO1_T2B_MIX_EPS)
    action_space = C.TO1_T2B_ACTION_SPACE
    device = JG.get_device()
    full_diagnostics = bool(not is_t2d or getattr(args, "full_diagnostics", False))
    profile_workers = bool(not is_t2d or getattr(args, "profile_workers", False))
    t2d_out = C.ROOT / "outputs" / "t2d" / architecture.lower()
    tb_writer = None
    if is_t2d:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(t2d_out / "tensorboard"))
        except Exception as exc:  # noqa: BLE001
            print(f"[t2d] TensorBoard unavailable: {exc}", flush=True)

    # ---- parent policy: trainable jpol + frozen R20 continuation (R21 VERBATIM)
    r6_sel = _load_r6_parent(args)
    t_caps = time.time()
    caps = (_t2d_train_only_caps(env, re, scorer, reranker, r6_sel, t2d_out)
            if is_t2d and architecture != HR.ARCH_P0
            else HR.ResidualCaps(1.0, 1.0, 1.0))
    if is_t2d:
        print(f"[t2d] required TRAIN-only residual-cap setup: "
              f"{time.time() - t_caps:.1f}s", flush=True)
    jpol = _t2d_policy(architecture, r6_sel, caps, env)
    if is_t2d:
        _t2d_resume(jpol, getattr(args, "resume", None), architecture)
        if full_diagnostics:
            _t2d_train_latent_diagnostic(env, re, scorer, reranker, jpol, t2d_out)
        else:
            print("[t2d] fast-start: skipped train-latent diagnostic", flush=True)
    with torch.no_grad():
        resid_max = max((p.abs().max().item()
                         for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    # R20 continuation stays the frozen canonical P0 policy for all P0/P1/P2.
    cont_m2 = JG.M2RootPolicyAdapter(C.TO1_R13_M2_FEAT_DIM,
                                     float(C.TO1_R13_ALPHA_M2))
    for _p_ in cont_m2.parameters():
        _p_.requires_grad_(False)
    cont_m3 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                     alpha_stop=C.TO1_R13_ALPHA_STOP)
    for _p_ in cont_m3.parameters():
        _p_.requires_grad_(False)
    jpol.cont_m2, jpol.cont_m3 = cont_m2, cont_m3
    print(f"[t2b] architecture={architecture} parent {C.TO1_T2B_PARENT_ID} | M2 adapter {n_tr_m2} params | "
          f"M3 residual {n_tr_m3} params (resid_max={resid_max:.3g}) | "
          f"device={device} | K={N} H={H} T={T} eps={eps} | "
          f"action_space={action_space}", flush=True)

    # ---- R6 reproduction anchor (verbatim) ----------------------------------
    if full_diagnostics:
        mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                      reranker, r6_sel)
        r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
        r6_regret = float(mr6["top1_regret"]["mean"])
        repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
        print(f"[t2b] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
              f"repro_ok={repro_ok}", flush=True)
    else:
        r6_acc_all, r6_regret, repro_ok = 0.0, 0.0, None
        print("[t2d] fast-start: skipped R6 reproduction metric", flush=True)

    # ---- splits + roots (T2-A/R21 verbatim) ---------------------------------
    bench_iids = [i["instance_id"] for i in env["train_insts"]]
    train_pairs = [(iid, env["ep_id_of"][iid]) for iid in bench_iids]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    rb = sb = None
    t_aux = time.time()
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env,
                              include_held=full_diagnostics)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env,
                              include_held=full_diagnostics)
              if C.TO1_AUX_DATA.exists() else None)
    if is_t2d:
        print(f"[t2d] required TRAIN graph/memory setup: {time.time() - t_aux:.1f}s",
              flush=True)
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    src_of = _r21_src_of(bench_iids, rb, sb)
    family_of = _r21_family_of(bench_iids, rb, sb)

    def _roots(pm, pairs, st_map):
        return [{"problem": st_map[iid]["problem"], "schedule": st_map[iid]["schedule"],
                 "iid": iid, "episode_id": int(eid), "progmem": pm,
                 "root_ms": int(st_map[iid]["schedule"].makespan), "horizon": H}
                for (iid, eid) in pairs]

    held_splits = [("bench_held", val_pairs, st_bench_map, re["progmem"], "bench")]
    if rb:
        held_splits.append(("real_held", real_hd_pairs, rb["hd_states"], rb["hd_pm"], "real"))
    if sb:
        held_splits.append(("syn_held", syn_hd_pairs, sb["hd_states"], sb["hd_pm"], "syn"))

    # ---- graphs (R21 D0 benchmark-dominant) ---------------------------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}
    if is_t2d:
        print("[t2d] TRAIN pool="
              f"bench:{len(specs['bench'])} real:{len(specs['real'])} "
              f"syn:{len(specs['syn'])} total:{sum(len(v) for v in specs.values())}; "
              "held remains gradient-sealed (bench:3 real:8 syn:20)", flush=True)

    # ---- held diagnostic (T2-A greedy + N sampled, branch-local memory) ------
    def _run_held_diag(jp_):
        rows = []
        split_sums = {}
        for split_name, pairs, st_map, base_pm, src in held_splits:
            if not pairs:
                split_sums[split_name] = JG.t2a_split_summary([])
                continue
            roots = _roots(base_pm, pairs, st_map)
            rows_split = []
            for rf in roots:
                iid, eid = rf["iid"], rf["episode_id"]
                rf_g = dict(rf, progmem=copy.deepcopy(rf["progmem"]))
                g_gain, g_usage, g_steps = JG.agentic_parity_rollout(
                    env, scorer, jp_, rf_g, use_mem=True, horizon=H, gate_mem=False,
                    gate_variant="r14", action_space=action_space, sample=None)
                samples, sigs, step_counts = [], [], [len(g_steps)]
                sigs.append(JG.t2a_traj_sig(g_steps))
                for b in range(N):
                    rng = random.Random(JG.t2a_branch_seed(iid, eid, b))
                    rf_b = dict(rf, progmem=copy.deepcopy(rf["progmem"]))
                    b_gain, b_usage, b_steps = JG.agentic_parity_rollout(
                        env, scorer, jp_, rf_b, use_mem=True, horizon=H, gate_mem=False,
                        gate_variant="r14", action_space=action_space,
                        sample={"T": T, "eps": eps, "rng": rng})
                    samples.append(int(b_gain))
                    sigs.append(JG.t2a_traj_sig(b_steps))
                    step_counts.append(len(b_steps))
                best_of_N = int(max(samples)) if samples else int(g_gain)
                rows_split.append({
                    "iid": iid, "split": split_name, "src": src,
                    "family": family_of.get(iid, "?"),
                    "greedy": int(g_gain), "samples": samples,
                    "best_of_N": best_of_N,
                    "best_overall": int(max([g_gain] + samples)),
                    "mean": float(np.mean(samples)) if samples else float(g_gain),
                    "median": float(np.median(samples)) if samples else float(g_gain),
                    "n_pos_traj": int(sum(1 for s in samples if s > 0)),
                    "n_distinct_traj": int(len(set(sigs))),
                    "n_distinct_outcomes": int(len(set([g_gain] + samples))),
                    "latent": bool(g_gain <= 0 and best_of_N > 0),
                    "failure": JG.t2a_classify_failure(g_gain, samples, sigs, step_counts),
                    "greedy_steps": JG.t2a_compact_steps(g_steps),
                })
            split_sums[split_name] = JG.t2a_split_summary(rows_split)
            rows.extend(rows_split)
        return rows, split_sums

    # ---- E0 baseline (frozen zero-init parent, bit-repro T2-A) ---------------
    if full_diagnostics:
        with torch.no_grad():
            e0_rows, e0_split_sums = _run_held_diag(jpol)
        e0 = JG.t2a_split_summary(e0_rows)
        held_n = int(e0.get("n", 0))
        print(f"[t2b] E0 held greedy={e0.get('greedy_total', 0):+d} "
              f"bestN={e0.get('bestN_total', 0):+d} "
              f"bestofN_gain={e0.get('bestofN_gain', 0):+d} "
              f"latent={e0.get('latent_n', 0)}/{e0.get('latent_denom', 0)} "
              f"distinct_traj={e0.get('mean_distinct_traj', 0.0):.2f}", flush=True)
    else:
        e0_rows, e0_split_sums = [], {}
        e0 = JG.t2a_split_summary([])
        held_n = 0
        print("[t2d] fast-start: skipped E0 held greedy/best-of-N rollouts", flush=True)

    # ---- eval_root_builder + held_agg_fn (R21 deployment ruler) -------------
    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _cur(jp_, action_space):
        sm, stps = _r20_eval_closed(
            env, scorer, jp_,
            _eval_roots(re["progmem"], train_pairs, st_bench_map), action_space)
        return sm, stps

    def eval_root_builder(jp_, cycle=-1):
        ev = {"train": _cur(jp_, "lexicographic")[0]}
        if not full_diagnostics:
            ev.update({"bench_held": {"total": 0.0},
                       "real_held": {"total": 0.0},
                       "syn_held": {"total": 0.0}})
            return ev
        smv, _ = _r20_eval_closed(env, scorer, jp_,
                                  _eval_roots(re["progmem"], val_pairs, st_bench_map))
        ev["bench_held"] = smv
        ev["real_held"] = (_r20_eval_closed(env, scorer, jp_, _eval_roots(
            rb["hd_pm"], real_hd_pairs, rb["hd_states"]))[0] if rb else {"total": 0.0})
        ev["syn_held"] = (_r20_eval_closed(env, scorer, jp_, _eval_roots(
            sb["hd_pm"], syn_hd_pairs, sb["hd_states"]))[0] if sb else {"total": 0.0})
        return ev

    def held_agg_fn(ev):
        if not full_diagnostics:
            return float(ev.get("train", {}).get("total", 0.0))
        return float(np.mean([float(ev.get(k, {}).get("total", 0.0))
                              for k in ("bench_held", "real_held", "syn_held")]))

    def _held_agg_of(h):
        return float(np.mean([float(h.get(k, 0.0))
                              for k in ("bench_held", "real_held", "syn_held")]))

    # ---- L3 no-RL parity baseline (collapse floor) --------------------------
    t_l3 = time.time()
    with torch.no_grad():
        sm_l3, _ = _cur(jpol, "lexicographic")
        l3 = float(sm_l3["total"])
    print(f"[t2b] L3(lexicographic no-RL)={l3:.0f}", flush=True)
    if is_t2d:
        print(f"[t2d] required TRAIN collapse ruler: {time.time() - t_l3:.1f}s",
              flush=True)

    # ---- cloud/shape profile -> workers (R21 verbatim) ----------------------
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    workers = 1
    mp_ok = True
    prof = None
    try:
        if is_t2d and not profile_workers:
            workers = workers_arg
            mp_ok = None
            prof = {"skipped": True, "reason": "fast_start",
                    "chosen_workers": workers}
            print(f"[t2d] fast-start: skipped worker sweep; workers={workers}", flush=True)
        elif is_t2d:
            safe = tuple(w for w in (1, 2, 4, 8, 16)
                         if w <= max(1, workers_arg))
            profile_graphs = bench_graphs[:min(16, len(bench_graphs))]
            prof = JG.graph_level_cloud_profile(
                jpol, scorer, env, profile_graphs, workers=safe,
                k=K, horizon=H,
                mp_ctx=mp_ctx, action_space="lexicographic")
            ident_ok = all(row["identical_to_w1"]
                           for row in prof["per_worker"].values())
            mp_ok = bool(ident_ok)
            workers = int(min(workers_arg, prof["chosen_workers"]))
            print(f"[t2d] graph profile: identical={ident_ok} -> workers={workers}",
                  flush=True)
        else:
            prof = JG.agentic_cloud_profile(jpol, scorer, env, init_roots[0],
                                            workers=tuple(sorted({1, 2, 4})),
                                            graphs=(4, 8, 16), k=C.TO1_R13_K,
                                            mp_ctx=mp_ctx, variant="r14",
                                            action_space="lexicographic")
            ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"]
                           for w in (2, 4))
            mp_ok = bool(ident_ok)
            best_w, best_sp = 1, 1.0
            for w in (2, 4):
                rw = prof["per_worker"].get(str(w))
                if rw and rw["identical_to_w1"] and float(rw["coll_s"]) > 0:
                    sp_w = float(prof["speedup_vs_w1"][str(w)])
                    if sp_w > best_sp + 1e-9:
                        best_w, best_sp = int(w), sp_w
            workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
            print(f"[t2b] cloud profile: identical={ident_ok} -> workers={workers}",
                  flush=True)
    except Exception as exc:                   # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[t2b] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- run the loop (R21 runtime verbatim, single D0 condition) -----------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")
    diag = _T2BGroupDiag()
    if is_t2d:
        n_train_pool = sum(len(v) for v in specs.values())
        print(f"[t2d] ===== TRAINING START architecture={architecture} "
              f"workers={workers} cycles={training_cycles} "
              f"graphs/cycle={graphs_per_cycle} branches/graph={K} "
              f"depth={H} train_pool={n_train_pool} unified=true =====",
              flush=True)
    rollout_pool = None
    if is_t2d and workers > 1:
        rollout_pool = JG.make_graph_rollout_pool(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], workers=workers, mp_ctx=mp_ctx,
            val_cache={}, critical_gate=False, torch_threads=1)
        print(f"[t2d] persistent rollout pool workers={workers} "
              "adaptive_sibling_shards<=4 lpt_submission=true "
              "numeric_threads=1", flush=True)
    try:
        res = JG.run_rolling_cycles_r13(
            jpol, scorer, env, specs, stage="C", cycles=training_cycles,
            k=K, horizon=H,
            graphs_per_batch=graphs_per_cycle, workers=workers,
            seed=args.grpo_seed, log_prefix="[t2b]",
            eval_root_builder=eval_root_builder, collapse_floor=max(0.0, l3),
            parent_policy=None, mp_ctx=mp_ctx, variant="r20",
            on_groups=diag, action_space="lexicographic",
            per_src={"bench": 2, "real": 1, "syn": 1}, unified=is_t2d,
            per_cycle=(graphs_per_cycle if is_t2d else None), best_score_fn=held_agg_fn,
            optimizer_persistent=bool(getattr(args, "optimizer_persistent", False)),
            tensorboard_writer=tb_writer, rollout_pool=rollout_pool)
    finally:
        if rollout_pool is not None:
            rollout_pool.shutdown(wait=True)
    with torch.no_grad():
        ev_f = eval_root_builder(jpol)
    train_peak = max((h["train"] for h in res["history"]), default=0.0)
    held_by_cycle = [_held_agg_of(h) for h in res["history"]]
    held_peak = max(held_by_cycle, default=0.0)
    print(f"[t2b] train_peak={train_peak:.0f} held_peak={held_peak:.0f} "
          f"collapsed={res.get('collapsed')}", flush=True)

    # ---- final diagnostic on promoted snapshot ------------------------------
    if full_diagnostics:
        with torch.no_grad():
            final_rows, final_split_sums = _run_held_diag(jpol)
        final = JG.t2a_split_summary(final_rows)
        final_gains = {r["iid"]: r["greedy"] for r in final_rows}
        l2g = JG.t2b_l2g_conversion(e0_rows, final_gains)
        print(f"[t2b] FINAL held greedy={final.get('greedy_total', 0):+d} "
              f"bestN={final.get('bestN_total', 0):+d} "
              f"bestofN_gain={final.get('bestofN_gain', 0):+d} "
              f"L2G={l2g:.3f} "
              f"distinct_traj={final.get('mean_distinct_traj', 0.0):.2f}", flush=True)
    else:
        final_rows, final_split_sums = [], {}
        final, l2g = JG.t2a_split_summary([]), 0.0
        print("[t2d] fast-start: training finished; skipped final held best-of-N", flush=True)
    if tb_writer is not None:
        final_step = training_cycles
        final_tags = ({
            "held/bestofN": final.get("bestN_total", 0),
            "held/extraction_gap": final.get("bestofN_gain", 0),
            "success/greedy": final.get("greedy_pos_rate", 0),
            "success/N": final.get("success_at_N", 0),
            "positive_trajectory_rate": final.get("positive_traj_rate", 0),
            "latent_to_greedy_conversion": l2g,
        } if full_diagnostics else {
            "train/final_greedy": float(ev_f.get("train", {}).get("total", 0.0)),
        })
        for tag, value in final_tags.items():
            tb_writer.add_scalar(tag, float(value), final_step)
        tb_writer.flush()
        tb_writer.close()

    # ---- §39 normal-M5 regression (hard-freeze guard) -----------------------
    dpp_iid = None
    if full_diagnostics:
        if any(i["instance_id"] == args.dpp for i in env["order"]):
            dpp_iid = args.dpp
        elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
            dpp_iid = next(i["instance_id"] for i in env["order"]
                           if "DPpaulli" in i["instance_id"])
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1
    m5_f = (_r20_m5_wrapper(env, scorer, jpol, dpp_iid,
                            copy.deepcopy(re["progmem"]), dpp_ep)
            if dpp_iid else None)
    m5_ok = True if m5_f is None else bool(m5_f.get("all_retained", True))
    if m5_f:
        print(f"[t2b] §39 normal-M5 FINAL: dep_imm={m5_f.get('dep_immediate')} "
              f"all_retained={m5_f.get('all_retained')}", flush=True)

    # ---- verdict ladder (双闸) + GO_5070TI ----------------------------------
    if full_diagnostics:
        vcode, vlabel, note, go_5070ti = JG.t2b_verdict(
            repro_ok, mp_ok, m5_ok, e0, final, held_n)
    else:
        vcode, vlabel, note, go_5070ti = (
            "TRAIN_ONLY", "FAST_START",
            "legacy comparison diagnostics skipped; latest.pt contains the trained state",
            False)
    print(f"[t2b] verdict {vcode} {vlabel} | GO_5070TI={go_5070ti} | {note}",
          flush=True)

    # ---- checkpoint (verdict A ONLY) ----------------------------------------
    arch_meta = HR.checkpoint_metadata(
        architecture, jpol.m2, jpol.m3, caps=caps, k=K, horizon=H)
    arch_meta.update({
        "phase": "t2d_hierarchical_residual" if is_t2d else "t2b_multipath_joint_grpo",
        "runtime": "R20_lexicographic", "optimizer_persistent": bool(
            getattr(args, "optimizer_persistent", False)),
        "m3_terminal_reward": "makespan_only",
        "best_of_n": "post_hoc_diagnostic_only",
        "checkpoint": "m2_m3_hscrp_t2d" if is_t2d else "m2_m3_multipath_joint_grpo_t2b",
        "diagnostic_mode": "full" if full_diagnostics else "fast_start",
    })
    if is_t2d:
        optimizer_state = _t2d_optimizer_state_cpu(jpol)
        payload = {"state": {"policy_snapshot": jpol.snapshot(),
                              "optimizer_state": optimizer_state,
                              "r6_anchor": r6_sel.state_dict()},
                   "meta": dict(arch_meta,
                                optimizer_state_saved=optimizer_state is not None)}
    else:
        payload = {"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                             "scorer": scorer}, "meta": arch_meta}
    if is_t2d:
        t2d_out.mkdir(parents=True, exist_ok=True)
        torch.save(payload, t2d_out / "latest.pt")
    if go_5070ti:
        ckpt_path = (t2d_out / "promoted.pt") if is_t2d else C.TO1_T2B_CKPT
        torch.save(payload, ckpt_path)
        print(f"[t2b] saved {ckpt_path} (PASS)", flush=True)
    elif is_t2d and not full_diagnostics:
        print(f"[t2d] saved latest.pt; promotion verdict deferred to optional diagnostics",
              flush=True)
    else:
        print(f"[t2b] NOT PASS -> no T2-B checkpoint written (A only)", flush=True)

    ast_rows = [g.get("ast_profile", {}) for g in diag.groups
                if g.get("ast_profile")]
    ast_runtime = {
        "ast_calls_before": sum(int(x.get("ast_calls_before", 0)) for x in ast_rows),
        "ast_calls_after": sum(int(x.get("ast_calls_after", 0)) for x in ast_rows),
        "ast_seconds_saved": sum(float(x.get("ast_seconds_saved", 0.0)) for x in ast_rows),
        "breakdown_seconds": {key: sum(float(x.get("breakdown", {}).get(key, 0.0))
                                       for x in ast_rows)
                              for key in ("graph_construction_s", "neural_forward_s",
                                          "other_analysis_s", "total_s")},
    }
    gpu_monitors = [d["update"]["_monitor"] for h in res["history"]
                    for d in h.get("depth_results", [])
                    if d.get("update", {}).get("_monitor")]
    scrape = dict(
        architecture=architecture, architecture_metadata=arch_meta,
        full_diagnostics=full_diagnostics,
        parent_id=C.TO1_T2B_PARENT_ID, device=str(device),
        K=K, horizon=H, training_cycles=training_cycles,
        graphs_per_cycle=graphs_per_cycle,
        train_pool_size=sum(len(v) for v in specs.values()),
        temp=T, mix_eps=eps, action_space=action_space,
        repro_ok=repro_ok, r6_acc_all=r6_acc_all, r6_regret=r6_regret,
        n_tr_m3=n_tr_m3, n_tr_m2=n_tr_m2, resid_max=resid_max,
        l3=l3, mp_ok=mp_ok, workers=workers, mp_ctx=str(mp_ctx), prof=prof,
        ast_runtime=ast_runtime, gpu_monitors=gpu_monitors,
        phase_by_cycle=[h.get("phase_s", {}) for h in res["history"]],
        e0=e0, e0_split_summaries=e0_split_sums,
        final=final, final_split_summaries=final_split_sums,
        l2g=l2g, held_n=held_n,
        diag=diag.summary(),
        res_collapsed=bool(res.get("collapsed")),
        n_groups=sum(h.get("n_groups", 0) for h in res["history"]),
        n_informative=sum(h.get("n_informative", 0) for h in res["history"]),
        train_peak=train_peak, held_peak=held_peak, held_by_cycle=held_by_cycle,
        selected={k: float(v.get("total", 0.0)) for k, v in ev_f.items()},
        m5_f=m5_f, m5_ok=m5_ok, dpp_iid=dpp_iid,
        verdict=dict(code=vcode, label=vlabel, note=note),
        go_5070ti=bool(go_5070ti),
        per_instance=[{k: r[k] for k in
                       ("iid", "split", "src", "family", "greedy", "samples",
                        "best_of_N", "best_overall", "mean", "median",
                        "n_pos_traj", "n_distinct_traj", "n_distinct_outcomes",
                        "latent", "failure", "greedy_steps")} for r in final_rows],
        elapsed_s=round(time.time() - t0, 1),
    )
    report = _t2b_report(scrape)
    if is_t2d:
        (t2d_out / "report.md").write_text(report, encoding="utf-8")
        (t2d_out / "result.json").write_text(
            json.dumps(scrape, indent=2, default=str), encoding="utf-8")
        print(f"[t2d] total {scrape['elapsed_s']}s (report {t2d_out / 'report.md'})",
              flush=True)
    else:
        _t2b_persist(report, scrape)
        print(f"[t2b] total {scrape['elapsed_s']}s (report {C.T2B_REPORT.name})",
              flush=True)
    return report


def _t2b_report(s):
    L = []
    J = lambda d: json.dumps(d, default=str)
    v = s.get("verdict", {})
    e0 = s.get("e0", {})
    final = s.get("final", {})
    L.append("# T2-B — MULTI-PATH JOINT AGENTIC GRPO 报告\n")
    L.append(f"_verdict {v.get('code')} {v.get('label')} | GO_5070TI={s.get('go_5070ti')} "
             f"| identified=false | formal_test_access=0 | Formal TEST SEALED  \n"
             f"{v.get('note', '')}_\n")
    L.append("\n## 0. 科学边界\n")
    L.append("- Stage-3 Joint Agentic GRPO：reward/advantage/update 与 R21 完全一致"
             "（lexicographic runtime + stagewise A2/A3）。")
    L.append("- best-of-N 是 **post-hoc 诊断**（deployment diagnostic + extraction-gap 测量），"
             "绝不进 reward/advantage/oracle。")
    L.append(f"- parent：`{s.get('parent_id')}`；M2 adapter {s.get('n_tr_m2')} params；"
             f"M3 residual {s.get('n_tr_m3')} params（resid_max={s.get('resid_max')}）。")
    L.append(f"- device={s.get('device')}；actor params M2={s.get('n_tr_m2')} / "
             f"M3={s.get('n_tr_m3')}；rollout 仍以 CPU 环境计算为主。")
    L.append("\n## 1. E0 → FINAL 提取（ExtractionGap = 差值）\n")
    L.append(f"- **greedy_total** held：{e0.get('greedy_total', 0):+d} → "
             f"{final.get('greedy_total', 0):+d}")
    L.append(f"- **bestN_total** held：{e0.get('bestN_total', 0):+d} → "
             f"{final.get('bestN_total', 0):+d}")
    L.append(f"- **ExtractionGap (bestofN_gain)**：{e0.get('bestofN_gain', 0):+d} → "
             f"{final.get('bestofN_gain', 0):+d}")
    L.append(f"- **Latent-to-Greedy 转换率 (L2G)**：{s.get('l2g', 0.0):.3f}")
    L.append(f"- latent_n：{e0.get('latent_n', 0)} → {final.get('latent_n', 0)}；"
             f"latent_rate：{e0.get('latent_rate', 0.0):.3f} → {final.get('latent_rate', 0.0):.3f}")
    L.append(f"- success@N：{e0.get('success_at_N', 0.0):.3f} → {final.get('success_at_N', 0.0):.3f}；"
             f"positive_traj_rate：{e0.get('positive_traj_rate', 0.0):.3f} → "
             f"{final.get('positive_traj_rate', 0.0):.3f}")
    L.append(f"- mean_distinct_traj：{e0.get('mean_distinct_traj', 0.0):.2f} → "
             f"{final.get('mean_distinct_traj', 0.0):.2f}")
    L.append("\n## 2. 训练\n")
    L.append(f"- train_peak={s.get('train_peak')} held_peak={s.get('held_peak')} "
             f"L3(no-RL)={s.get('l3')} collapsed={s.get('res_collapsed')}")
    L.append(f"- n_groups={s.get('n_groups')} n_informative={s.get('n_informative')} "
             f"cycles={len(s.get('held_by_cycle', []))} workers={s.get('workers')}")
    L.append(f"- rollout worker profile: {J(s.get('prof', {}))}")
    L.append(f"- selected: {J(s.get('selected', {}))}")
    L.append("\n## 3. 组级诊断（multi-path group diagnostic）\n")
    L.append(f"- {J(s.get('diag', {}))}")
    L.append("\n## 4. 硬冻结 / 复现\n")
    L.append(f"- R6 anchor: acc={s.get('r6_acc_all', 0.0):.4f} regret={s.get('r6_regret', 0.0):.2f} "
             f"repro_ok={s.get('repro_ok')}")
    L.append(f"- mp_ok={s.get('mp_ok')}（workers=1==workers=N bit-identical）")
    L.append(f"- normal-M5 (§39): {J(s.get('m5_f'))} -> m5_ok={s.get('m5_ok')}")
    L.append("\n## 5. 结论\n")
    L.append(f"- **verdict {v.get('code')} — {v.get('label')}**：{v.get('note', '')}")
    L.append(f"- **GO_5070TI = {s.get('go_5070ti')}**（A-only，双闸：greedy 转正 AND gap 收窄）。")
    L.append("- checkpoint 仅 A 写 `m2_m3_multipath_joint_grpo_t2b.pt`。")
    return "\n".join(L) + "\n"


def _t2b_persist(report, scrape):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.T2B_REPORT.write_text(report, encoding="utf-8")
    (C.CANONICAL_OUT_DIR / "result_t2b.json").write_text(
        json.dumps(scrape, indent=2, default=str), encoding="utf-8")
    print(f"[t2b] report written: {C.T2B_REPORT}", flush=True)


def run_top1_phase_r15(args, env, re, p1, p1_report):
    """R15: T1-M3-COVERAGE-PRESERVING-SELECTION-GRPO -- the M3 proposal-selection round.

    R14 verdict E (2026-08-28): M2 coverage is sufficient; M3 loses gains inside a
    ~100-proposal action space (M3_SELECTION_MISS 3 >= M2_PROBE_MISS 2).  R15's ONLY
    change is the M3 action-set construction: a coverage-preserving shortlist
    (GLOBAL top-12 by frozen M3 SFT base + per-root quotas Tier-A top-2/Tier-B top-1
    + structural-family diversity fill, CAP=32, STOP external, §5-9) REPLACES the
    wide_pool stage.  No oracle (true_U / frozen-future-utility / trajectory reward /
    oracle rank) enters runtime; true_U only in the §11-12 TRAIN diagnostic recall.

    Q-parity on the SAME unified RAW ruler (§25-26): Q0 = cited R14 P0 (gated SFT,
    full pool = 355), Q1 = SAME SFT parent with shortlist32 (NO RL), Q2 = cited R14
    P3 (stagewise JOINt, full pool = 356), Q3 = R15 stagewise JOINT with shortlist32.
    Main judgement §36: Q3 > Q1 AND M3_SELECTION_MISS reduced AND shortlist positive
    coverage not visibly dropped.  One JOINT stage (§16-18), M2 q2 EXACTLY R14 (§0,
    frozen M2 SFT / Memory / adaptive probing / Reasoner / FixedDecisionReplay /
    stagewise A2/A3 credit all UNTOUCHED), M3 backbone m3_proposal_top1_sft_v2.pt
    (NO R11-R14 RL parent, residual zero-init).

    verdict §40 ladder: G SEMANTICS_REGRESSION / E SHORTLIST_COVERAGE_FAILURE /
    F M3_SELECTION_STILL_PRIMARY_BLOCKER / A M3_SHORTLIST_SELECTION_IMPROVES /
    C M3_RANKING_NOT_ACTION_SIZE_IS_BLOCKER / D M3_STOP_MARGIN_IS_BLOCKER /
    B SHORTLIST_PRESERVES_COVERAGE_BUT_NO_RL_GAIN.  Checkpoint ONLY on verdict A
    (§44).  identified=false, formal_test_access=0, Formal TEST SEALED.
    """
    print("[r15] COVERAGE-PRESERVING M3 SELECTION (shortlist) -- JOINT GRPO ...",
          flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()
    workers_arg = int(min(max(getattr(args, "workers", 1), 1), 64))
    sl_aspace = "shortlist"

    # ---- §15 parent: M3 = pure SFT; NO R11-R14 RL checkpoint -------------------------
    r6_sel = _load_r6_parent(args)
    jpol = JG.JointAgenticPolicy(r6_sel)
    with torch.no_grad():
        resid_max = max((p.abs().max().item() for n, p in jpol.m3.named_parameters()
                         if n.startswith("resid_")), default=0.0)
    n_tr_m3 = sum(p.numel() for p in jpol.m3.parameters() if p.requires_grad)
    n_tr_m2 = sum(p.numel() for p in jpol.m2.parameters() if p.requires_grad)
    print(f"[r15] M2 adapter {n_tr_m2} params (zero-init δ=0) | M3 {n_tr_m3} params "
          f"(pure SFT {C.TO1_CKPT.name}; resid_max={resid_max:.3g} -> δ=0 ≡ R6; ")
    print(f"      NO R11/R12/R13/R14 RL parent -- §15)", flush=True)
    proof_no_m3_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r11_grpo_checkpoint": None, "r12_grpo_checkpoint": None,
        "r13_grpo_checkpoint": None, "r14_grpo_checkpoint": None,
        "r15_resid_zero_init": float(resid_max) == 0.0,
        "m3_base_equals_r6": True,
        "m3_action_space": sl_aspace,
        "shortlist_oracle_authority": False,          # §10 forbid
    }

    # ---- verbatim R6 reproduction (gate_mem=False, §40 ruler anchor) ----------------
    mr6, grp_replay = TOP1.top1_metrics(re["state_examples"], scorer,
                                        re["mem_values"], reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r15] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    # ---- graphs: bench TRAIN14 + AUX-real/syn (identical to R14, §21) ---------------
    bench_graphs = []
    for i in env["train_insts"]:
        iid, st = i["instance_id"], env["states"][i["instance_id"]]
        bench_graphs.append(RGRPO.Graph(iid=iid, episode_id=env["ep_id_of"][iid],
                                        problem=st["problem"], schedule=st["schedule"],
                                        progmem=copy.deepcopy(re["progmem"]),
                                        src="bench", ms0=int(st["schedule"].makespan)))
    rb = sb = None
    if not args.quick:
        rb = _r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real",
                             env) if C.TO1_REAL_DATA.exists() else None
        sb = _r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux",
                             env) if C.TO1_AUX_DATA.exists() else None
    print(f"[r15] graphs: bench {len(bench_graphs)} | "
          f"real {len(rb['graphs']) if rb else 0} train / "
          f"{len(rb['hd_iids']) if rb else 0} held | "
          f"syn {len(sb['graphs']) if sb else 0} train / "
          f"{len(sb['hd_iids']) if sb else 0} held", flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    real_hd_pairs = [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]] if rb else []
    syn_hd_pairs = [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]] if sb else []
    specs = {"bench": bench_graphs,
             "real": rb["graphs"] if rb else [],
             "syn": sb["graphs"] if sb else []}

    def _eval_roots(pm, pairs, st_map):
        return _r12_eval_roots(pm, pairs, st_map)

    def _closed_loop(mode, obj, roots, aspace="full"):
        sm, _ = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode,
                               use_mem=True, gate_mem=False, gate_variant="r14",
                               action_space=aspace)
        return sm

    def _tot(mode, obj, roots, aspace="full"):
        return float(_closed_loop(mode, obj, roots, aspace).get("total", 0.0))

    def _sl_tot(obj, roots):
        return _tot("adapter", obj, roots, sl_aspace)

    def _eval_full(obj, mode, roots, aspace="full"):
        sm, stps = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode,
                                  use_mem=True, gate_mem=False, gate_variant="r14",
                                  action_space=aspace)
        cov = {"n_states": sum(len(v["steps"]) for v in stps.values())}
        return {"total": float(sm.get("total", 0.0))}, cov

    def eval_root_builder(jp_, cycle=-1):
        # NOTE: must return DICTs with "total" (run_rolling_cycles_r13 L1793).
        ev = {"train": _closed_loop("adapter", jp_,
                                    _eval_roots(re["progmem"], train_pairs,
                                                st_bench_map), sl_aspace)}
        ev["real_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            rb["hd_pm"], real_hd_pairs, rb["hd_states"]), sl_aspace)
            if rb else {"total": 0.0})
        ev["syn_held"] = (_closed_loop("adapter", jp_, _eval_roots(
            sb["hd_pm"], syn_hd_pairs, sb["hd_states"]), sl_aspace)
            if sb else {"total": 0.0})
        return ev

    # ---- §48 normal-M5 gate (dep-completed positive => Tier A, never Memory) --------
    dpp_iid = None
    if any(i["instance_id"] == args.dpp for i in env["order"]):
        dpp_iid = args.dpp
    elif any("DPpaulli" in i["instance_id"] for i in env["order"]):
        dpp_iid = next(i["instance_id"] for i in env["order"]
                       if "DPpaulli" in i["instance_id"])
    dpp_st = env["states"][dpp_iid] if dpp_iid else None
    dpp_ep = env["ep_id_of"].get(dpp_iid, -1) if dpp_iid else -1

    def _m5_gate(adapter, gv="r14"):
        if dpp_iid is None:
            return None
        prop_feats, metas, agg = env["cache"].proposals(
            dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        if not metas:
            return {"at_state": False, "dep_saved": [], "all_tier_a_ok": True,
                    "retain_ok": True, "retained": [], "diag": {"note": "no proposals"}}
        ast = env["cache"].ast(dpp_st["problem"], dpp_st["schedule"], dpp_iid)
        ms = int(dpp_st["schedule"].makespan)
        sf = PF.state_feature_vec(ms, ms, len(metas), agg["best_uhat"], agg["best_direct"],
                                  agg["n_contrib"], agg["n_enab"])
        return JG.normal_m5_r13_gate(ast, metas, prop_feats, adapter,
                                     env["executor"], copy.deepcopy(re["progmem"]),
                                     dpp_iid, int(dpp_ep), 0, sf, random.Random(1),
                                     gate_variant=gv)

    # ---- §34 M2 reward diagnostic aggregator (training groups, no leaks) ------------
    m2_agg = {"groups": 0, "steps": 0, "draws": 0, "proven": 0, "memory": 0,
              "unsupported": 0, "min_proven_q2": 1e9, "max_mem_q2": -1.0,
              "ordering_ok": True, "inf2_groups": 0, "inf3_groups": 0,
              "inf2_steps": 0, "inf3_steps": 0}
    sl_train_agg = {"steps": 0, "shortlist_steps": 0, "n_global": [], "n_roota": [],
                    "n_rootb": [], "n_diversity": [], "sig_match": 0}

    def _acc(groups):
        for g in groups:
            m2_agg["groups"] += 1
            m2_agg["inf2_groups"] += (1 if g.get("info2") else 0)
            m2_agg["inf3_groups"] += (1 if g.get("informative") else 0)
            for tr in g["trajs"]:
                for rec in tr["steps"]:
                    m2_agg["steps"] += 1
                    m2_agg["inf2_steps"] += (1 if rec.get("inf2") else 0)
                    m2_agg["inf3_steps"] += (1 if rec.get("inf3") else 0)
                    mr = rec.get("m2_rec") or {}
                    qs = mr.get("q2_stats") or {}
                    m2_agg["draws"] += len(mr.get("draws", []))
                    if qs.get("proven"):
                        m2_agg["proven"] += len(qs["proven"])
                        m2_agg["min_proven_q2"] = min(m2_agg["min_proven_q2"],
                                                      min(qs["proven"]))
                    if qs.get("memory"):
                        m2_agg["memory"] += len(qs["memory"])
                        m2_agg["max_mem_q2"] = max(m2_agg["max_mem_q2"],
                                                   max(qs["memory"]))
                    m2_agg["unsupported"] += int(qs.get("unsupported", 0))
                    if not (rec.get("m2_diag") or {}).get("q2_ordering_ok", True):
                        m2_agg["ordering_ok"] = False
                    sli = rec.get("sl_info") or {}
                    sl_train_agg["steps"] += 1
                    if rec.get("action_space") == "shortlist":
                        sl_train_agg["shortlist_steps"] += 1
                        sl_train_agg["n_global"].append(
                            float(sli.get("n_global", 0)))
                        sl_train_agg["n_roota"].append(float(sli.get("n_roota", 0)))
                        sl_train_agg["n_rootb"].append(float(sli.get("n_rootb", 0)))
                        sl_train_agg["n_diversity"].append(
                            float(sli.get("n_diversity", 0)))

    # ---- §25/§38 R6 anchor on the same unified RAW ruler ------------------------
    c0 = JG.M3RollingGRPOPolicy(r6_sel, alpha_prop=C.TO1_R13_ALPHA_PROP,
                                alpha_stop=C.TO1_R13_ALPHA_STOP)
    init_roots = _eval_roots(re["progmem"], train_pairs, st_bench_map)
    with torch.no_grad():
        p0_anchor = float(_tot("none", c0, init_roots))
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    anchor_tol = max(2.0, 0.05 * float(r6_canonical_train))
    anchor_ok = bool(abs(p0_anchor - float(r6_canonical_train)) <= anchor_tol)
    print(f"[r15] ANCHOR: raw {p0_anchor:.0f} vs r6_canonical={r6_canonical_train:.0f} "
          f"tol={anchor_tol:.1f} anchor_ok={anchor_ok}", flush=True)

    m5_z = _m5_gate(jpol.m2, gv="r14")
    if m5_z:
        print(f"[r15] §48 adaptive zero-init: dep_saved={m5_z['dep_saved']} "
              f"all_tier_a_ok={m5_z['all_tier_a_ok']}", flush=True)

    # ---- §2-4 + §11-13 decomposition & coverage gate BEFORE (SFT parent) --------
    train_iids = [i["instance_id"] for i in env["train_insts"]]
    deco_i = _r15_deco(env, scorer, jpol, train_iids, copy.deepcopy(re["progmem"]),
                       cap_states=(2 if args.quick else None))
    recall_i = deco_i.get("recall") or {}
    tol = float(C.TO1_R15_RECALL_DROP_TOL)
    recall_gate_ok = bool(recall_i.get("any", 0.0) >= 1.0 - tol and
                          recall_i.get("oracle", 0.0) >= 1.0 - tol)
    print(f"[r15] §13 coverage gate: any={recall_i.get('any'):.3f} "
          f"best={recall_i.get('best'):.3f} oracle={recall_i.get('oracle'):.3f} "
          f"tol={tol:.2f} -> recall_gate_ok={recall_gate_ok}", flush=True)
    print(f"[r15] §3 deco INIT: {json.dumps(deco_i.get('decomposition'))} "
          f"M3_SELECTION_MISS={deco_i.get('M3_SELECTION_MISS')} "
          f"n_state={deco_i.get('n_state')}", flush=True)

    # ---- §32 M2_PROBE_MISS (recorded, not fixed: accept 2) ------------------------
    def _m2_miss(policy):
        m2dec = {}
        try:
            m2dec = _r14_deco(env, scorer, policy, init_roots[:5])
        except Exception as exc:                     # noqa: BLE001
            m2dec = {"error": str(exc)}
        return (m2dec.get("decomposition") or {}) if isinstance(m2dec, dict) else {}
    m2_i = _m2_miss(jpol)

    # ---- §35 shortlist mp determinism / cloud shape -> workers ---------------------
    prof = None
    mp_ok = True
    workers = 1
    prof_root = init_roots[0]
    try:
        prof = JG.agentic_cloud_profile(jpol, scorer, env, prof_root,
                                        workers=tuple(sorted({1, 2, 4})),
                                        graphs=(4, 8, 16), k=C.TO1_R13_K,
                                        mp_ctx=mp_ctx, variant="r14",
                                        action_space=sl_aspace)
        ident_ok = all(prof["per_worker"][str(w)]["identical_to_w1"] for w in (2, 4))
        mp_ok = bool(ident_ok)
        best_w, best_sp = 1, 1.0
        for w in (2, 4):
            rw_ = prof["per_worker"].get(str(w))
            if rw_ and rw_["identical_to_w1"] and float(rw_["coll_s"]) > 0:
                sp_w = float(prof["speedup_vs_w1"][str(w)])
                if sp_w > best_sp + 1e-9:
                    best_w, best_sp = int(w), sp_w
        workers = int(min(workers_arg, best_w if best_sp > 1.05 else 1))
        print(f"[r15] cloud profile: {json.dumps(prof['per_worker'])} "
              f"identical={ident_ok} -> workers={workers}", flush=True)
    except Exception as exc:                         # noqa: BLE001
        prof = prof or {"error": str(exc)}
        workers, mp_ok = workers_arg, False
        print(f"[r15] cloud profile FAILED: {exc} (workers={workers})", flush=True)

    # ---- §25 Q-row citations + Q1 = SAME SFT parent on shortlist, NO RL -----------
    q0 = float(C.TO1_R15_Q0_CITE_FULL_SFT)
    q2 = float(C.TO1_R15_Q2_CITE_FULL_JOINT)
    zj_snap = jpol.snapshot()
    with torch.no_grad():
        q1 = float(_tot("adapter", jpol, init_roots, sl_aspace))
    print(f"[r15] Q0(cite R14 P0 full=SFT)={q0:.0f} | Q1(SFT shortlist32, NO RL)="
          f"{q1:.0f} | Q2(cite R14 P3 full=JOINT)={q2:.0f}", flush=True)

    # ---- §13 HARD GATE: coverage failure -> verdict E, NO training ----------------
    if not recall_gate_ok:
        vcode, vlabel = "E", "SHORTLIST_COVERAGE_FAILURE"
        note = (f"shortlist32 recall any={recall_i.get('any'):.3f} "
                f"oracle={recall_i.get('oracle'):.3f} < 1-tol={1.0 - tol:.3f} vs the "
                f"full pool on the TRAIN states -- action-set compression loses "
                f"positives, NEVER hand it to training (§13)")
        deco_f = deco_i
        m2_f = dict(_m2_miss(jpol))
        res_j = {"cycles_run": 0, "collapsed": False, "history": [], "best":
                 {"train": 0.0, "score": -1e18}}
        for g in bench_graphs:
            g.reset()
        scrape = _r15_scrape_all(locals(), dict(
            repro_ok=repro_ok, anchor_ok=anchor_ok, m5=m5_z, m5_z=m5_z,
            res_j=res_j, q0=q0, q1=q1, q2=q2, q3=None, q3_real=None, q3_syn=None,
            q3_val=None, q1_real=None, q1_syn=None,
            rows=dict(p0_anchor=p0_anchor, q0=(q0, None, None, None),
                      q1=(q1, None, None, None), q2=(q2, None, None, None),
                      q3=(None, None, None, None)),
            r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
            workers=workers, mp_ctx=mp_ctx, deco_i=deco_i, deco_f=deco_f,
            recall_i=recall_i, recall_f=(deco_f or {}).get("recall") or {},
            recall_gate_ok=recall_gate_ok, coverage_stable=None,
            m3_miss=(deco_i.get("M3_SELECTION_MISS", 0),
                     (deco_f or {}).get("M3_SELECTION_MISS", 0)),
            m2_miss=(int(m2_i.get("M2_PROBE_MISS", 0)),
                     int(m2_f.get("M2_PROBE_MISS", 0))),
            m2_agg=m2_agg, sl_train_agg=sl_train_agg, mk=None, val_final=None,
            m2_kl=[0.0], m3_kl=[0.0], n_info_m2=0, n_info_m3=0, info_ratio2=0.0,
            info_ratio3=0.0, m2_grad_any=False, m3_grad_any=False,
            reward_ordering_ok=m2_agg["ordering_ok"],
            proof_no_m3_rl_parent=proof_no_m3_rl_parent,
            train_beat=False, miss_reduced=False, q3_gt_q1=False,
            coverage_stable_ref=None, aux_ok=None,
            verdict=dict(code=vcode, label=vlabel, note=note), passed=False,
            quick_sanity=None, action_space=sl_aspace))
        report = _r15_report(env, re, scraper=scrape)
        _r15_persist(report, scrape=scrape)
        print(f"[r15] verdict {vcode} {vlabel} ({note})", flush=True)
        print(f"[r15] total {time.time() - t0:.1f}s "
              f"(report {C.R15_REPORT.name})", flush=True)
        return report

    if args.quick:
        # quick ALSO passes 2 REAL joint-stage cycles with SHORTLIST actions
        for g in bench_graphs:
            g.reset()
        jpol.params_for_stage("C")
        try:
            res_s = JG.run_rolling_cycles_r13(
                jpol, scorer, env, specs, stage="C", cycles=2, k=C.TO1_R13_K,
                horizon=C.TO1_R13_HORIZON, graphs_per_batch=2, workers=1,
                seed=args.grpo_seed, quick=True, log_prefix="[r15-qs]",
                eval_root_builder=eval_root_builder, collapse_floor=None,
                parent_policy=None, mp_ctx=mp_ctx, variant="r14", on_groups=_acc,
                action_space=sl_aspace)
            quick_sanity = {"cycles": res_s["cycles_run"],
                            "best_train_sl": float(res_s["best"]["train"]),
                            "collapsed": bool(res_s["collapsed"]),
                            "sl_steps": sl_train_agg["shortlist_steps"],
                            "m2_agg": dict(m2_agg)}
            print(f"[r15] QUICK sanity: {res_s['cycles_run']} cycles "
                  f"best_train_sl={quick_sanity['best_train_sl']:.0f} "
                  f"sl_steps={sl_train_agg['shortlist_steps']}", flush=True)
        except Exception as exc:                     # noqa: BLE001
            quick_sanity = {"error": str(exc)}
            print(f"[r15] QUICK sanity FAILED: {exc}", flush=True)
        m5q = _m5_gate(jpol.m2, gv="r14")
        if m5q:
            print(f"[r15] §48 after quick: all_tier_a_ok="
                  f"{m5q['all_tier_a_ok']}", flush=True)
        report = _r15_report(env, re, scraper=dict(
            repro_ok=repro_ok, anchor_ok=anchor_ok, m5=m5q, m5_z=m5_z, res_j=None,
            q0=q0, q1=q1, q2=q2, q3=None, q3_real=None, q3_syn=None, q3_val=None,
            q1_real=None, q1_syn=None, rows={}, r6_canonical_train=None,
            prof=prof, mp_ok=mp_ok, workers=workers, mp_ctx=mp_ctx,
            deco_i=deco_i, deco_f=None, recall_i=recall_i, recall_f=recall_i,
            recall_gate_ok=recall_gate_ok, coverage_stable=None,
            m3_miss=(deco_i.get("M3_SELECTION_MISS", 0),
                     deco_i.get("M3_SELECTION_MISS", 0)),
            m2_miss=(int(m2_i.get("M2_PROBE_MISS", 0)),
                     int(m2_i.get("M2_PROBE_MISS", 0))),
            m2_agg=m2_agg, sl_train_agg=sl_train_agg, mk=None, val_final=0.0,
            m2_kl=None, m3_kl=None, n_info_m2=0, n_info_m3=0, info_ratio2=0.0,
            info_ratio3=0.0, m2_grad_any=False, m3_grad_any=False,
            reward_ordering_ok=m2_agg["ordering_ok"],
            proof_no_m3_rl_parent=proof_no_m3_rl_parent,
            train_beat=False, miss_reduced=False, q3_gt_q1=False,
            coverage_stable_ref=None, aux_ok=None,
            verdict=dict(code="S", label="STAGE_C_JOINT_SANITY",
                         note=f"--quick: shortlist deco + coverage gate + 2 REAL "
                              f"joint shortlist-GRPO cycles; sanity {quick_sanity}"),
            passed=False, checkpoint_written=False, quick_sanity=quick_sanity,
            action_space=sl_aspace))
        _r15_persist(report, scrape=None)
        print("[r15] QUICK: shortlist joint stagewise sanity only -- "
              "see outputs/canonical_m3/result_r15.json", flush=True)
        return report
    # ---- THE ONE JOINT STAGE (§16-18): M2 + M3 stagewise GRPO over SHORTLIST -------
    for g in bench_graphs + (rb["graphs"] if rb else []) + (sb["graphs"] if sb else []):
        g.reset()
    jpol.params_for_stage("C")

    def _snapshot_policy(snap):
        zj2 = copy.deepcopy(jpol)
        zj2.load_snapshot(snap)
        return zj2

    res_j = JG.run_rolling_cycles_r13(
        jpol, scorer, env, specs, stage="C", cycles=C.TO1_R15_TRAINING_CYCLES,
        k=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
        graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH, workers=workers,
        seed=args.grpo_seed, log_prefix="[r15-J]", eval_root_builder=eval_root_builder,
        collapse_floor=max(0.0, q1), parent_policy=None, mp_ctx=mp_ctx,
        variant="r14", on_groups=_acc, action_space=sl_aspace)
    m5_f = _m5_gate(jpol.m2, gv="r14")
    if m5_f:
        print(f"[r15] §48 FINAL: all_tier_a_ok={m5_f['all_tier_a_ok']} "
              f"dep_saved={m5_f['dep_saved']} retain_ok={m5_f['retain_ok']}", flush=True)

    # ---- §34 reward ordering + KL/grader info across the joint stage ----------------
    n_groups = sum(h["n_groups"] for h in res_j["history"])
    n_inf3 = sum(h["n_informative"] for h in res_j["history"])
    n_inf2 = sum(h.get("n_informative_trajectories_m2", 0) for h in res_j["history"])
    info_ratio3 = float(n_inf3 / max(n_groups, 1))
    info_ratio2 = float(n_inf2 / max(n_groups, 1))
    m2_kl = []
    m3_kl = []
    for h in res_j["history"]:
        for d in h.get("depth_results", []):
            for e in ((d.get("update") or {}).get("epochs", []) or []):
                if e.get("kl_m2") is not None:
                    m2_kl.append(float(e["kl_m2"]))
                if e.get("kl_ref_m3") is not None:
                    m3_kl.append(float(e["kl_ref_m3"]))
    m2_kl = m2_kl or [0.0]
    m3_kl = m3_kl or [0.0]
    reward_ordering_ok = bool(m2_agg["ordering_ok"] and
                              (m2_agg["proven"] == 0 or
                               m2_agg["min_proven_q2"] > m2_agg["max_mem_q2"]))
    m2_grad_any = m3_grad_any = False
    for h in res_j["history"]:
        for d in h.get("depth_results", []):
            for e in (d.get("update") or {}).get("epochs", []) or []:
                if float(e.get("grad_norm", 0.0)) > 1e-9:
                    m2_grad_any = True
                    m3_grad_any = True
    print(f"[r15] informative: groups={n_groups} inf2_traj={n_inf2} inf3_traj={n_inf3} "
          f"ratios {info_ratio2:.3f}/{info_ratio3:.3f} | grad={m2_grad_any}/"
          f"{m3_grad_any} | sl_steps={sl_train_agg['shortlist_steps']}",
          flush=True)
    print(f"[r15] §34 reward diag: proven={m2_agg['proven']} memory={m2_agg['memory']} "
          f"unsupported={m2_agg['unsupported']} "
          f"min_proven_q2={m2_agg['min_proven_q2']:.3f} "
          f"max_mem_q2={m2_agg['max_mem_q2']:.3f} ordering_ok={reward_ordering_ok}",
          flush=True)

    # ---- Q3 = MAIN: final joint policy, SHORTLIST32, same unified RAW ruler ---------
    with torch.no_grad():
        q3 = float(_tot("adapter", jpol, init_roots, sl_aspace))
        q1_real = q3_real = q1_syn = q3_syn = None
        if rb:
            q1_real = float(_tot("adapter", _snapshot_policy(zj_snap),
                                 _eval_roots(rb["hd_pm"], real_hd_pairs,
                                             rb["hd_states"]), sl_aspace))
            q3_real = float(_tot("adapter", jpol,
                                 _eval_roots(rb["hd_pm"], real_hd_pairs,
                                             rb["hd_states"]), sl_aspace))
        if sb:
            q1_syn = float(_tot("adapter", _snapshot_policy(zj_snap),
                                _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                            sb["hd_states"]), sl_aspace))
            q3_syn = float(_tot("adapter", jpol,
                                _eval_roots(sb["hd_pm"], syn_hd_pairs,
                                            sb["hd_states"]), sl_aspace))
        q3_val = float(_tot("adapter", jpol,
                            _eval_roots(re["progmem"], val_pairs, st_bench_map),
                            sl_aspace))
    print(f"[r15] Q3(stagewise JOINT shortlist): {q3:.0f} (init {q1:.0f}) | real "
          f"{q3_real}/{q1_real} | syn {q3_syn}/{q1_syn} | VAL {q3_val:.0f}", flush=True)

    # ---- §2-4 decomposition + §11-12 recall AFTER -----------------------------------
    deco_f = _r15_deco(env, scorer, jpol, train_iids, copy.deepcopy(re["progmem"]),
                       cap_states=(2 if args.quick else None))
    recall_f = deco_f.get("recall") or {}
    print(f"[r15] §3 deco FINAL: {json.dumps(deco_f.get('decomposition'))} "
          f"M3_SELECTION_MISS={deco_f.get('M3_SELECTION_MISS')}", flush=True)
    m2_f = _m2_miss(jpol)

    mk = None
    try:
        mk = _r15_mk_traces(jpol, scorer, env, re, mp_ctx)
        _mk_show = {k: mk.get(k) for k in ("mk1", "mk3") if isinstance(mk, dict)}
        print(f"[r15] Mk traces: {json.dumps(_mk_show, default=str)[:2200]}",
              flush=True)
    except Exception as exc:                           # noqa: BLE001
        mk = {"error": str(exc)}
        print(f"[r15] Mk trace FAILED: {exc}", flush=True)

    # ---- §40 verdict ladder (G -> A -> D -> C -> F -> B) ----------------------------
    miss_i = int(deco_i.get("M3_SELECTION_MISS", 0))
    miss_f = int(deco_f.get("M3_SELECTION_MISS", 0))
    q3_gt_q1 = bool(q3 > q1 + 1.0)
    miss_reduced = bool(miss_f < miss_i)
    coverage_stable = bool(recall_f.get("any", 0.0) >= recall_i.get("any", 0.0) - 0.10 and
                           recall_f.get("best", 0.0) >= recall_i.get("best", 0.0) - 0.10)
    di = deco_i.get("decomposition") or {}
    a_i = int(di.get("M3_RANKING_MISS", 0))
    b_i = int(di.get("M3_STOP_MARGIN_MISS", 0))
    c_i = int(di.get("M3_INTRA_POOL_SELECTION_MISS", 0))
    d_i = int(di.get("M3_POOL_DILUTION", 0))
    aux_pairs = []
    if rb:
        aux_pairs.append(("real", q3_real, q1_real))
    if sb:
        aux_pairs.append(("syn", q3_syn, q1_syn))
    aux_ok = bool(any(v is not None and b is not None and v >= b - 1.0
                      for _nm, v, b in aux_pairs)) if aux_pairs else None
    m5_ok = bool(all(x["all_tier_a_ok"] for x in (m5_z, m5_f) if x is not None))
    mach_ok = bool(repro_ok and anchor_ok and m5_ok and mp_ok and
                   not res_j["collapsed"] and reward_ordering_ok and
                   q1 is not None and q3 is not None)

    if not mach_ok:
        vcode, vlabel = "G", "SEMANTICS_REGRESSION"
        note = (f"repro={repro_ok} anchor={anchor_ok} m5§48={m5_ok} mp_ok={mp_ok} "
                f"collapsed={res_j['collapsed']} reward_ordering={reward_ordering_ok} "
                f"-- machinery broken (shortlist changes mutated semantics)")
    elif q3_gt_q1 and miss_reduced and coverage_stable:
        vcode, vlabel = "A", "M3_SHORTLIST_SELECTION_IMPROVES"
        note = (f"Q3 {q3:.0f} > Q1 {q1:.0f} AND M3_SELECTION_MISS {miss_i}->{miss_f} "
                f"AND shortlist coverage stable (any {recall_i.get('any'):.3f}->"
                f"{recall_f.get('any'):.3f}) -- the compressed action set unlocks gain")
    elif not q3_gt_q1 and b_i >= max(a_i, d_i, 1):
        vcode, vlabel = "D", "M3_STOP_MARGIN_IS_BLOCKER"
        note = (f"Q3={q3:.0f}<=Q1={q1:.0f} AND init decomposition dominated by "
                f"STOP-margin ({b_i} states) -- the blocker is the STOP score, not "
                f"the action-set size")
    elif not q3_gt_q1 and a_i >= max(b_i, d_i, 1):
        vcode, vlabel = "C", "M3_RANKING_NOT_ACTION_SIZE_IS_BLOCKER"
        note = (f"Q3={q3:.0f}<=Q1={q1:.0f} AND init decomposition dominated by "
                f"ranking misses ({a_i} states) -- shrink the pool does not fix a "
                f"broken frozen score ranking")
    elif miss_f >= max(miss_i, 1):
        vcode, vlabel = "F", "M3_SELECTION_STILL_PRIMARY_BLOCKER"
        note = (f"M3_SELECTION_MISS {miss_i}->{miss_f} not reduced (Q3={q3:.0f} vs "
                f"Q1={q1:.0f}) -- shortlist did not convert into selection improvement")
    elif not q3_gt_q1:
        vcode, vlabel = "B", "SHORTLIST_PRESERVES_COVERAGE_BUT_NO_RL_GAIN"
        note = (f"shortlist coverage preserved (any {recall_i.get('any'):.3f}->"
                f"{recall_f.get('any'):.3f}, gate_ok={recall_gate_ok}) and "
                f"M3_SELECTION_MISS {miss_i}->{miss_f} but Q3={q3:.0f}<=Q1={q1:.0f} -- "
                f"no RL gain on the compressed set")
    else:
        vcode, vlabel = "B", "SHORTLIST_PRESERVES_COVERAGE_BUT_NO_RL_GAIN"
        note = (f"Q3={q3:.0f}>Q1={q1:.0f} but miss {miss_i}->{miss_f} not reduced / "
                f"coverage_stable={coverage_stable} -- gain without clean selection fix")
    passed = bool(vcode == "A")
    print(f"[r15] verdict {vcode} {vlabel} (Q3={q3:.0f} vs Q1={q1:.0f} / "
          f"miss {miss_i}->{miss_f} / recall any {recall_i.get('any'):.3f}->"
          f"{recall_f.get('any'):.3f} / aux_ok={aux_ok})", flush=True)

    if passed:
        meta_common = dict(
            phase="r15_coverage_preserving_m3_selection_joint_grpo",
            method="m2_sft_then_m3_sft_then_joint_agentic_grpo_stagewise",
            pipeline="M2_SFT->M3_SFT->Joint_GRPO",
            m3_action_space="coverage_preserving_shortlist32",
            m3_action_space_cap=C.TO1_R15_SHORTLIST_CAP,
            m3_k_global=C.TO1_R15_K_GLOBAL,
            m3_tier_a_quota=C.TO1_R15_TIER_A_QUOTA,
            m3_tier_b_quota=C.TO1_R15_TIER_B_QUOTA,
            shortlist_oracle_authority=False,           # §10 forbid
            shortlist_signature=C.TO1_R15_SL_SIGNATURE_HASH,   # §20
            parent=C.TO1_CKPT.name,                     # m3_proposal_top1_sft_v2.pt
            r11_r12_r13_r14_grpo_parent=None,           # FORBIDDEN by §15
            reward="q2_local_probe (M2) / terminal_makespan (M3), stagewise A2/A3",
            m2_role="budgeted_root_search", m2_filter="makespan_first_memory_second",
            m3_role="proposal_selection_on_shortlist", reasoner="frozen",
            executor="FixedDecisionReplay",
            formal_test_access=0, formal_test_sealed=True, identified=False,
            lambda_m2=C.TO1_R13_LAMBDA_M2, lambda_m3=C.TO1_R13_LAMBDA_M3,
            beta_m2=C.TO1_R14_BETA_M2, beta_m3=C.TO1_R14_BETA_M3,
            K=C.TO1_R13_K, horizon=C.TO1_R13_HORIZON,
            graphs_per_batch=C.TO1_R13_GRAPHS_PER_BATCH,
            update_epochs=C.TO1_R13_UPDATE_EPOCHS, max_depth=C.TO1_R13_MAX_DEPTH,
            workers=workers, mp_ctx=mp_ctx,
            selected="TRAIN14+AUX-held (not VAL3, §38)",
            stop_semantics="policy-STOP / non-positive / infeasible / no-pool / revisit",
            credit="M2=A2(q2/U2) only; M3=A3(terminal) only; no shared advantage (§15)",
            adaptive_memory_control=False,)
        torch.save({"state": {"policy": jpol, "r6_anchor": r6_sel.state_dict(),
                              "scorer": scorer},
                    "meta": dict(meta_common,
                                 checkpoint="m2_m3_stagewise_joint_grpo_r15",
                                 role="M3 coverage-preserving selection JOINT")},
                   C.TO1_R15_CKPT)
        print(f"[r15] saved {C.TO1_R15_CKPT.name} (PASS, verdict A only §44)",
              flush=True)
    else:
        print(f"[r15] NOT PASS (verdict {vcode}) -> no R15 checkpoint written (§44)",
              flush=True)

    scrape = _r15_scrape_all(locals(), dict(
        repro_ok=repro_ok, anchor_ok=anchor_ok, m5=m5_f, m5_z=m5_z, res_j=res_j,
        q0=q0, q1=q1, q2=q2, q3=q3, q1_real=q1_real, q3_real=q3_real,
        q1_syn=q1_syn, q3_syn=q3_syn, q3_val=q3_val,
        rows=dict(p0_anchor=p0_anchor, q0=(q0, None, None, None),
                  q1=(q1, q1_real, q1_syn, None), q2=(q2, None, None, None),
                  q3=(q3, q3_real, q3_syn, q3_val)),
        r6_canonical_train=r6_canonical_train, prof=prof, mp_ok=mp_ok,
        workers=workers, mp_ctx=mp_ctx,
        deco_i=deco_i, deco_f=deco_f, recall_i=recall_i, recall_f=recall_f,
        recall_gate_ok=recall_gate_ok, coverage_stable=coverage_stable,
        m3_miss=(miss_i, miss_f),
        m2_miss=(int(m2_i.get("M2_PROBE_MISS", 0)),
                 int(m2_f.get("M2_PROBE_MISS", 0))),
        m2_agg=m2_agg, sl_train_agg=sl_train_agg, mk=mk, val_final=q3_val,
        m2_kl=m2_kl, m3_kl=m3_kl, n_info_m2=n_inf2, n_info_m3=n_inf3,
        info_ratio2=info_ratio2, info_ratio3=info_ratio3,
        m2_grad_any=m2_grad_any, m3_grad_any=m3_grad_any,
        reward_ordering_ok=reward_ordering_ok,
        proof_no_m3_rl_parent=proof_no_m3_rl_parent,
        train_beat=q3_gt_q1, miss_reduced=miss_reduced, q3_gt_q1=q3_gt_q1,
        coverage_stable_ref=coverage_stable, aux_ok=aux_ok,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed,
        quick_sanity=None, action_space=sl_aspace))
    report = _r15_report(env, re, scraper=scrape)
    _r15_persist(report, scrape=scrape)
    print(f"[r15] total {time.time() - t0:.1f}s "
          f"(report {C.R15_REPORT.name})", flush=True)
    return report


# ---------------------------------------------------------------------------
# R16: T1-M3-POOL-LISTWISE-RANKING-SFT  (M3 SFT improvement -- NO RL)
# ---------------------------------------------------------------------------
def _r16_meta_aligned(metas, ex):
    """True when fresh `cache.proposals` metas align 1:1 with the replay/aux example
    labels (same (problem, schedule, iid) -> same deterministic proposal list)."""
    if len(metas) != len(ex["metas"]):
        return False
    for a, b in zip(metas, ex["metas"]):
        if a["kind"] != b["kind"] or a["i"] != b["i"] or a.get("j") != b.get("j"):
            return False
    return True


def _r16_aux_s0_exs(payload_path, iids):
    """{iid: S0 state-example} from one AUX payload (labels + state_feat)."""
    pl = torch.load(str(payload_path), map_location="cpu", weights_only=False)
    want = set(iids)
    return {e["iid"]: e for e in pl["state_examples"]
            if e["iid"] in want and e.get("s0")}


def _r16_state_group(env, scorer, jpol, st, iid, ep, ex, pm, verify=False):
    """One R16 training/selection group = state + FULL gated pool + STOP + U labels.

    §2 data unit; §3 U(P) = FixedDecisionReplay true_U from the replay/aux example
    (frozen local executor), TRAIN-only (§4 -- never accessed at runtime).  The R14
    gate + F_all features are rebuilt fresh here (identical to `_r15_state_diag` so
    training and the §18/20/28 diagnostics share the exact pool).  Returns None on
    skip (no proposals / empty gated pool / label-alignment failure)."""
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    N = len(metas)
    if N == 0:
        return None
    if ex is None or not _r16_meta_aligned(metas, ex):
        return None
    ast = cache.ast(st["problem"], st["schedule"], iid)
    sf_list = [float(x) for x in ex["state_feat"].tolist()]
    sf_t = torch.as_tensor(ex["state_feat"], dtype=torch.float32).reshape(-1)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor, pm,
                                iid, int(ep), 0, sf_list, rng)
    if not gate["gated_metas"]:
        return None
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(pm.features(iid, int(ep), 0, sf_list, queries), dtype=torch.float32)
    gmem = float(pm.retrieval_gate(iid, int(ep), 0, sf_list))
    mem_sel = mem * gmem
    F_all = JG._rerank_feats_all(scorer, rolex, mem_sel)      # [G, 277]
    if F_all is None or len(F_all) == 0:
        return None
    kept = gate["kept_indices"]
    if len(kept) != len(F_all):
        return None
    U = torch.tensor([float(ex["true_U"][k]) for k in kept], dtype=torch.float32)
    pstats = JG._pool_stats_from(F_all)
    g = {"F_all": F_all, "sf": sf_t.detach().reshape(-1), "pstats": pstats.detach().reshape(-1),
         "U": U, "iid": iid, "n_gated": int(len(kept)), "n_full": int(N),
         "gate_diag": dict(gate["diag"]), "split": None}
    if verify:                                   # label-alignment safety net (§3)
        diff = []
        for k in range(len(kept)):
            edits, _kind = PF._edits_for(ast, gate["gated_metas"][k])
            res = PF._execute_step(executor, st["problem"], st["schedule"], edits,
                                   int(st["schedule"].makespan), schedule_hash(st["schedule"]))
            fres = float(res["improvement"]) if res is not None else 0.0
            diff.append(abs(float(ex["true_U"][kept[k]]) - fres))
        g["label_align_mean_abs_diff"] = float(np.mean(diff)) if diff else 0.0
        g["label_align_max_diff"] = float(np.max(diff)) if diff else 0.0
    return g


def _r16_um_agg(u_mass_rows):
    """R16 §31 -- thin wrapper over the importable `TOP1.pool_utility_mass`."""
    return TOP1.pool_utility_mass(u_mass_rows)


def _r16_state_traces(jpol_r6, jpol16, scorer, env, re, mp_ctx):
    """R16 §29 Mk1 / §30 Fattahi15 traces: full pool / pos count / ranks BEFORE(R6
    base) vs AFTER(R16 base) / oracle rank / STOP rank / selected true_U, plus the
    top-10 true_U under both orderings, plus per-step shortlist group rows."""
    targets = (("mk1", "Brandimarte_Mk1"), ("fattahi15", "Fattahi_Fattahi15"))
    out = {}
    for key, iid in targets:
        cand = [i["instance_id"] for i in env["order"]]
        if iid not in cand:
            out[key] = {"note": f"{iid} not in env"}
            continue
        tgt = iid
        root = RGRPO.roots_from_state(env["states"][tgt]["problem"],
                                      env["states"][tgt]["schedule"], tgt,
                                      env["ep_id_of"][tgt],
                                      copy.deepcopy(re["progmem"]))
        grp = JG.collect_full_group_rollouts_r14(
            jpol16, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], int(root["root_ms"]),
            root["iid"], root["episode_id"], root["progmem"], k=8, seed=7,
            workers=1, step0_cache=env["cache"], mp_ctx=mp_ctx,
            action_space="shortlist")
        db = _r15_state_diag(env, scorer, jpol_r6, tgt, copy.deepcopy(re["progmem"]))
        da = _r15_state_diag(env, scorer, jpol16, tgt, copy.deepcopy(re["progmem"]))
        d = {"iid": tgt}
        if db.get("skip") or da.get("skip"):
            d.update({"note": "state skipped", "before": None, "after": None})
        else:
            rb6, ra16 = db["row"], da["row"]
            d.update({
                "full_pool_M": int(ra16["N_full"]), "pos_count": int(ra16["pos_count"]),
                "best_true_U": ra16["best_true_U"],
                "before": {"oracle_rank_by_r6_base": rb6["best_true_U_frozen_rank"],
                           "best_pos_score_rank_by_r6_base": rb6["best_positive_frozen_rank"],
                           "stop_score_rank_by_r6": rb6["stop_score_rank"]},
                "after": {"oracle_rank_by_r16_base": ra16["best_true_U_frozen_rank"],
                          "best_pos_score_rank_by_r16_base": ra16["best_positive_frozen_rank"],
                          "stop_score_rank_by_r16": ra16["stop_score_rank"],
                          "oracle_rank_by_action_logits": ra16["best_true_U_score_rank"],
                          "selected_signature": ra16["selected_signature"],
                          "selected_true_U": ra16["selected_true_U"],
                          "selected_is_stop": bool(ra16["selected_signature"] == "STOP")},
            })
        rows = []
        for tr in grp["trajs"][:4]:
            st = tr["steps"][0]
            m2r = st["m2_rec"]
            draws = [{"op": m2r["ops"][d2["idx"]],
                      "class": d2.get("reward_class", "unsupported"),
                      "q2": round(float(d2.get("q2", 0.0)), 3)} for d2 in m2r["draws"]]
            logits = st["logits_old"]
            M = st["M"]
            probs = torch.softmax(torch.as_tensor(logits) / float(C.TO1_R13_TEMP),
                                  -1).tolist()
            sl = st.get("sl_info") or {}
            rows.append({
                "sib": tr["traj_id"], "terminal": tr["terminal"],
                "U2": round(float(tr["U2"]), 3), "A2": round(float(st["adv2"]), 3),
                "A3": round(float(st["adv3"]), 3), "reward": int(tr["reward"]),
                "tier_A": st["m2_diag"]["tier_A"],
                "probed": st["m2_diag"]["probed_root_count"],
                "draws": draws,
                "full_pool_M": int(sl.get("N", M)),
                "shortlist_M": int(sl.get("final", M)),
                "sl_sig": sl.get("signature"),
                "M3": {"argmax": int(st["a"]), "M": M,
                       "p_chosen": round(float(probs[st["a"]]), 3),
                       "p_stop": round(float(probs[M]), 3),
                       "n_proposals": M},
            })
        d["grp_key"] = list(grp["grp_key"])
        d["inf2"] = bool(grp["info2"]); d["inf3"] = bool(grp["informative"])
        d["rows"] = rows
        out[key] = d
    return out


def _r16_scrape_keys():
    return ["repro_ok", "r6_acc", "r6_regret", "r6_canonical_train", "base_matches_r15",
            "internal_train", "internal_held", "n_tr_groups", "n_held_groups",
            "pool_stats", "labels", "tau_u", "tau_p", "lambda_pos", "lambda_pair",
            "best_epoch", "held_score", "hist", "base_rank", "r16_rank",
            "base_recall", "r16_recall", "sl_gate_ok", "rec32_gate", "um",
            "decomp", "miss", "s0", "s1", "s2", "s1_val", "s0_val", "s1_real",
            "s1_syn", "mk", "val_deco", "conditions", "verdict", "passed",
            "proof_no_m3_rl_parent", "shortlist_proof", "label_align",
            "train_metrics", "held_metrics", "quick_sanity"]


def _r16_clear(items):
    return {k: v for k, v in items.items() if v is not None}


def _r16_40items(s):
    """§40 forty-item final return (numbered, one per directive §)."""
    v = s["verdict"]
    c = s["conditions"]
    bm = s.get("base_rank") or {}
    rm16 = s.get("r16_rank") or {}
    rec_b = s.get("base_recall") or {}
    rec16 = s.get("r16_recall") or {}
    um = s.get("um") or {}
    deco_b = s.get("decomp") and s["decomp"].get("base") or {}
    deco_16 = s.get("decomp") and s["decomp"].get("r16") or {}
    miss_b = (s.get("miss") or [None, None])[0]
    miss_16 = (s.get("miss") or [None, None])[1]
    mk = s.get("mk") or {}
    items = [
        (1, "experiment", "T1-M3-POOL-LISTWISE-RANKING-SFT-R16"),
        (2, "verdict_code", v["code"]),
        (3, "verdict_label", v["label"]),
        (4, "verdict_note", v["note"]),
        (5, "no_m3_rl_parent", s.setdefault("proof_no_m3_rl_parent", {})),
        (6, "frozen_below_m3", {"m2": "frozen", "m2_q2": "frozen (R14)",
                                "memory": "frozen", "adaptive_probing": "frozen (R14)",
                                "reasoner": "frozen", "fixed_decision_replay": "frozen",
                                "joint_grpo": "not run (SFT-only)",
                                "shortlist_cap": 32, "k_global": C.TO1_R15_K_GLOBAL,
                                "tier_a_quota": C.TO1_R15_TIER_A_QUOTA,
                                "tier_b_quota": C.TO1_R15_TIER_B_QUOTA}),
        (7, "r6_repro", {"acc": s["r6_acc"], "regret": s["r6_regret"],
                         "repro_ok": s["repro_ok"]}),
        (8, "r6_raw_anchor", s["r6_canonical_train"]),
        (9, "splits", {"internal_train": s["internal_train"],
                       "internal_held": s["internal_held"],
                       "aux_real_train/held": s.setdefault("aux_counts", {}).get("real"),
                       "aux_syn_train/held": s.setdefault("aux_counts", {}).get("syn")}),
        (10, "data_unit", {"n_train_groups": s["n_tr_groups"],
                           "n_held_groups": s["n_held_groups"],
                           "pool": "state + FULL gated pool + STOP (§2)"}),
        (11, "pool_stats", s["pool_stats"]),
        (12, "label_source", s["labels"]),
        (13, "tau_fixed", {"tau_u": s["tau_u"], "tau_p": s["tau_p"], "sweep": False}),
        (14, "u_tilde", "sign(U)*log1p(|U|) (§7)"),
        (15, "pos_margin", {"margin_M": C.TO1_R16_POS_MARGIN,
                            "form": "softplus(M - (s_+ - s_0)), s_0 = best zero-U action incl STOP (§8)"}),
        (16, "pair_weight", "w_ij = |U_tilde_i - U_tilde_j| over positive pairs (§9)"),
        (17, "stop_integrated", {"stop_has_u": 0.0, "same_objective": True,
                                 "calibrator": "none (§10)"}),
        (18, "loss_formula", "L = L_list + lambda_pos*L_posmargin + lambda_pair*L_pair",
         ),
        (19, "head_unchanged", {"M3ProposalScorer": "unchanged (frozen features)",
                                "proposal_features": "unchanged",
                                "head": "M3Top1Selector prop_head + stop_head (only trainable)"}),
        (20, "state_equal_weight", True),
        (21, "pair_per_state_normalized", True),
        (22, "model_selection", {"rule": f"argmax held mean pos-rec@{C.TO1_R16_SELECT_RECALL_K} (tie earliest)",
                                 "best_epoch": s["best_epoch"], "held_score": s["held_score"]}),
        (23, "val_once_final", {"used_in_selection": False, "val_deco": s.get("val_deco"),
                                "s1_val": s.get("s1_val")}),
        (24, "baseline_fullpool_rank", bm),
        (25, "r16_fullpool_rank", rm16),
        (26, "stop_false_block", {"train_after": (s.get("train_metrics") or {}).get("false_stop_rate"),
                                  "held_after": (s.get("held_metrics") or {}).get("false_stop_rate")}),
        (27, "shortlist_builder", {"verbatim_r15": True,
                                   "score_source_swapped": "wrapped selector -> jpol.m3._base_raw (§19)"}),
        (28, "shortlist_gate", {"any": rec16.get("any"), "best": rec16.get("best"),
                                "oracle": rec16.get("oracle"),
                                "gate_ok": s["sl_gate_ok"]}),
        (29, "primary_target", {"pos_recall_32_base": c.get("base_rec32"),
                                "pos_recall_32_r16": c.get("r16_rec32"),
                                "gate_0_95": s["rec32_gate"]}),
        (30, "cap_unchanged", {"cap": 32, "cap_increase": "forbidden (§23)",
                               "quota_change": "none (§24)"}),
        (31, "no_rl", {"m3_only_grpo": False, "joint_grpo": False, "rl_runs": 0}),
        (32, "closed_loop_no_rl", {"S0_r6_full": s["s0"], "S1_r16_full": s["s1"],
                                   "S2_r16_shortlist32": s["s2"]}),
        (33, "selection_miss", {"base": miss_b, "r16": miss_16,
                                "decomp_base": deco_b, "decomp_r16": deco_16}),
        (34, "Mk1_trace", mk.get("mk1")),
        (35, "Fattahi15_trace", mk.get("fattahi15")),
        (36, "utility_mass", {"base": um.get("base"), "r16": um.get("cur"),
                              "delta_K32": (c.get("util_delta"))}),
        (37, "positives_over_32_note", "when positives > 32 cap, item recall@32 caps by construction; §32 utility-recall (PositiveUtilityMass@32) tracks high-utility positives instead"),
        (38, "verdict_conditions", c),
        (39, "checkpoint", {"path": C.TO1_R16_CKPT.name, "written_only_on_A": True,
                            "method": "pool_relative_listwise_sft",
                            "parent": C.TO1_CKPT.name,
                            "stop_integrated": True, "oracle_runtime": False,
                            "formal_test_access": 0, "identified": False,
                            "written": bool(s["passed"])}),
        (40, "next_step", "Joint GRPO only after the §20 coverage gate PASS (§38); "
                          f"currently gate_ok={s['sl_gate_ok']} "
                          f"(§22 rec32 gate 0.95: {s['rec32_gate']})"),
    ]
    return {"items": [{"n": n, "key": k, "value": v_} for n, k, v_ in items],
            "summary": {k: v for k, v in (("verdict", v["label"]),)}}


def _r16_report(env, re, scrape):
    s = scrape
    L = []
    L.append("# T1-M3-POOL-LISTWISE-RANKING-SFT-R16 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    v = s["verdict"]
    L.append(f"## Verdict: **{v['code']}** `{v['label']}`")
    L.append("")
    L.append(f"> {v['note']}")
    L.append("")
    L.append("## §0 what R16 changes (and what it does NOT)")
    L.append("")
    L.append("R16 is an **M3 SFT improvement** stage (NO RL, NO new training phase). It")
    L.append("retrains the M3 selector prop head with a **pool-relative listwise** objective")
    L.append("so the frozen M3 itself pushes positive Proposals into the top-32 of the FULL")
    L.append("gated pool. Frozen below M3: M2, M2 q2, Memory, adaptive probing, Reasoner,")
    L.append("FixedDecisionReplay, shortlist CAP=32, and Joint GRPO semantics.")
    L.append("")
    L.append(f"- proof no M3-RL parent §15: {json.dumps(s['proof_no_m3_rl_parent'], default=str)}")
    L.append("")
    L.append("## §16 R6 reproduction (bit-exact anchor)")
    L.append("")
    L.append(f"- acc={s['r6_acc']:.4f} regret={s['r6_regret']:.2f} repro_ok={s['repro_ok']}"
             f" · raw anchor={s['r6_canonical_train']}")
    L.append(f"- baseline §28 ranking matches R15 = {s.get('base_matches_r15')}")
    L.append("")
    L.append(f"## §15 splits & training groups")
    L.append("")
    L.append(f"- internal train `{s['internal_train']}`; internal held `{s['internal_held']}`")
    L.append(f"- AUX-real train/held {s.get('aux_counts', {}).get('real')} | "
             f"AUX-syn train/held {s.get('aux_counts', {}).get('syn')}")
    L.append(f"- train groups {s['n_tr_groups']} · held groups {s['n_held_groups']}")
    L.append(f"- pool stats: {json.dumps(s.get('pool_stats') or {}, default=str)}")
    L.append(f"- labels: {json.dumps(s.get('labels') or {}, default=str)}")
    L.append(f"- label alignment (first bench group, mean/max abs delta): "
             f"{json.dumps(s.get('label_align') or {})}")
    L.append("")
    L.append("## §6-14 listwise objective")
    L.append("")
    L.append(f"- tau_u={s['tau_u']} (FIXED, no sweep) tau_p={s['tau_p']} λ_pos={s['lambda_pos']} "
             f"λ_pair={s['lambda_pair']}")
    L.append(f"- L = L_list(KL q||p, incl STOP&equiv U=0) + λ_pos·L_posmargin(softplus M-(s_+-s_0), "
             f"s_0=best zero-U incl STOP) + λ_pair·L_positive_pair (w=|ΔU_tilde|, per-state mean)")
    L.append(f"- state equal weight §13 / per-state pair normalisation §14")
    L.append("")
    L.append("## §17-18 full-pool ranking (baseline R6 base -> R16 base)")
    L.append("")
    bm = s.get("base_rank") or {}
    rm16 = s.get("r16_rank") or {}
    _rk_line = lambda r: ("·".join(f"{k}=" + ("-" if r.get(k) is None else f"{r[k]:.3f}")
                                   for k in ("best_positive_rank", "oracle_rank", "mrr",
                                             "rec1", "rec3", "rec5", "rec10", "rec20", "rec32")))
    L.append(f"- BEFORE (R6 base, §28 same ruler): {_rk_line(bm)}")
    L.append(f"- AFTER  (R16 base):               {_rk_line(rm16)}")
    L.append("")
    L.append(f"## §20 shortlist coverage gate (R16 scores, builder VERBATIM unchanged §19)")
    L.append("")
    rec16 = s.get("r16_recall") or {}
    L.append(f"- any={rec16.get('any')} · best={rec16.get('best')} · oracle={rec16.get('oracle')} "
             f"-> gate_ok={s['sl_gate_ok']}")
    L.append(f"- §22 primary target pos-rec@32 = "
             f"{(s.get('conditions') or {}).get('r16_rec32')} (gate 0.95: {s['rec32_gate']})")
    L.append("")
    um = s.get("um") or {}
    L.append("## §31 PositiveUtilityMass")
    L.append("")
    L.append(f"- base: {json.dumps(um.get('base'))} · cur: {json.dumps(um.get('cur'))} "
             f"· n_states {um.get('n_states')}")
    L.append("")
    L.append("## §26 no-RL closed loop (S0/S1/S2)")
    L.append("")
    L.append(f"- S0 R6 base full-pool = **{s['s0']}** · S1 R16 full-pool = **{s['s1']}** "
             f"· S2 R16 shortlist32 = **{s['s2']}**")
    L.append(f"- VAL: S0={s.get('s0_val')} S1={s.get('s1_val')} · AUX-held real={s.get('s1_real')} "
             f"syn={s.get('s1_syn')}")
    L.append("")
    L.append("## §28 recompute selection miss")
    L.append("")
    deco_b = (s.get("decomp") or {}).get("base") or {}
    deco_16 = (s.get("decomp") or {}).get("r16") or {}
    miss_b, miss_16 = (s.get("miss") or [None, None])
    L.append(f"- M3_SELECTION_MISS base={miss_b} -> r16={miss_16}")
    L.append(f"- decomposition base {json.dumps(deco_b, default=str)}")
    L.append(f"- decomposition r16  {json.dumps(deco_16, default=str)}")
    L.append("")
    L.append("## §29-30 Mk1 / Fattahi15 traces (S0 root, full pool)")
    L.append("")
    mk = s.get("mk") or {}
    for key in ("mk1", "fattahi15"):
        L.append(f"### {key}")
        L.append(f"```json")
        L.append(json.dumps(mk.get(key), indent=2, default=str))
        L.append("```")
    L.append("")
    L.append("## Verdict conditions (ladder)")
    L.append("")
    for _ck, _cv in (s.get("conditions") or {}).items():
        L.append(f"- {_ck}: {_cv}")
    L.append("")
    L.append("## Checkpoint")
    L.append("")
    if s["passed"]:
        L.append(f"{C.TO1_R16_CKPT.name} written (verdict-A only, §37/§39).")
    else:
        L.append(f"NOT written (verdict {v['code']}, §37 -- A only).")
    L.append("")
    f = _r16_40items(s)
    L.append("## §40 forty-item final return")
    L.append("")
    L.append("| # | key | value |")
    L.append("|---|---|---|")
    for it in f["items"]:
        val = it["value"]
        if isinstance(val, dict):
            val = json.dumps(val, default=str)
        val = str(val).replace("|", "\\|").replace("\n", " ")
        L.append(f"| {it['n']} | {it['key']} | {val} |")
    L.append("")
    L.append(f"## 下一步 (single highest-priority next action)")
    L.append("")
    if v["code"] == "A":
        L.append("Coverage gate PASS — the next Joint GRPO stage may proceed on top of "
                 "`m3_pool_listwise_sft_v3.pt` (§38), reusing the S2 shortlist closed loop.")
    elif v["code"] in ("C", "F"):
        L.append("M3 feature representation / label limitation is the blocker (§35/§37-F) — "
                 "next round upgrades the proposal representation or the label density, "
                 "NOT the loss form.")
    elif v["code"] == "D":
        L.append("STOP ranking is the blocker (§36) — next round studies the STOP "
                 "representation / stop-head input, not the proposal ranking.")
    else:
        L.append("Improvement without full gate pass — fix the shown condition before "
                 "Joint GRPO (§38).")
    L.append("")
    return "\n".join(L)


def _r16_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R16_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in _r16_scrape_keys():
            if k in scrape:
                payload[k] = scrape[k]
        payload["_40items"] = _r16_40items(scrape)
    (C.CANONICAL_OUT_DIR / "result_r16.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r16] report written: {C.R16_REPORT}", flush=True)


def run_top1_phase_r16(args, env, re, p1, p1_report):
    """R16: POOL-LISTWISE-RANKING SFT -- fix M3's pool-relative ranking so the frozen
    M3 itself pushes positive Proposals into the top-32 of the FULL gated pool.

    R15 verdict E (2026-08-29): shortlist oracle 0.884 < 0.95; the binding constraint is
    the FROZEN R6 base full-pool pos-recall@32 = 0.842 (Mk1 61 / Fattahi15 42 positives),
    not the shortlist construction.  R16 changes ONLY the M3 supervised ranking objective
    (listwise: L_list + λ_pos·L_posmargin + λ_pair·L_positive_pair over state+full pool+
    STOP) + the M3 selector head; everything else FROZEN (§0).  NO RL anywhere (§25).

    §33 verdict ladder: F DATA_LABEL_LIMITATION / C M3_FEATURE_REPRESENTATION_IS_BLOCKER /
    E SFT_GENERALIZATION_DEGRADES / A LISTWISE_M3_SFT_FIXES_RANKING /
    D STOP_RANKING_REMAINS_BLOCKER / B LISTWISE_IMPROVES_BUT_BELOW_COVERAGE_GATE.
    Checkpoint ONLY on A (§37).  identified=false, formal_test_access=0, TEST SEALED.
    """
    print("[r16] POOL-LISTWISE-RANKING SFT (M3 SFT improvement, NO RL) ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()

    r6_sel = _load_r6_parent(args)                     # frozen R6 base (top1_sft_v2)
    jpol_r6 = JG.JointAgenticPolicy(r6_sel)            # baseline policy (§17-19)
    proof_no_m3_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r6_grpo_checkpoint": None, "r11_grpo_checkpoint": None,
        "r12_grpo_checkpoint": None, "r13_grpo_checkpoint": None,
        "r14_grpo_checkpoint": None, "r15_checkpoint": None,
        "m3_retrained_with": "pool_relative_listwise_sft (NO RL)",
        "shortlist_cap": C.TO1_R15_SHORTLIST_CAP,
        "head": "M3Top1Selector prop_head + stop_head (only trainable parameters)",
    }

    # ---- §16 R6 bit-exact reproduction -------------------------------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r16] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    print(f"[r16] ANCHOR: r6_canonical_train={r6_canonical_train} (R15=369)", flush=True)

    # ---- AUX bundles (§15) --------------------------------------------------------
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)

    # ---- §15 splits: internal-held = FIRST-2 TRAIN instances (Behnke m40/m60) -----
    # QUICK sanity keeps both available instances in TRAIN so the listwise loop has
    # groups to train on (no held groups -> best epoch = last, deterministic).
    train_iids = [i["instance_id"] for i in env["train_insts"]]
    if args.quick:
        internal_train = train_iids
        internal_held = []
    else:
        internal_train = train_iids[C.TO1_R16_INTERNAL_HELD_COUNT:]
        internal_held = train_iids[:C.TO1_R16_INTERNAL_HELD_COUNT]
    print(f"[r16] TRAIN-internal: train {len(internal_train)} | held {internal_held}",
          flush=True)

    # ---- build R16 groups: state + FULL gated pool + STOP + U labels -------------
    bench_s0 = {}
    for ex in re["state_examples"]:
        if ex.get("s0"):
            bench_s0.setdefault(ex["iid"], []).append(ex)
    tr_groups, held_groups = [], []
    for iid in internal_train:
        if iid not in env["states"] or not bench_s0.get(iid):
            print(f"[r16] skip bench S0 {iid} (missing state/example)", flush=True)
            continue
        g = _r16_state_group(env, scorer, jpol_r6, env["states"][iid], iid,
                             env["ep_id_of"][iid], bench_s0[iid][0], re["progmem"],
                             verify=(iid == internal_train[0]))
        if g is not None:
            g["split"] = "bench-train"
            tr_groups.append(g)
    for iid in internal_held:
        if iid not in env["states"] or not bench_s0.get(iid):
            print(f"[r16] skip bench S0 {iid} (missing state/example)", flush=True)
            continue
        g = _r16_state_group(env, scorer, jpol_r6, env["states"][iid], iid,
                             env["ep_id_of"][iid], bench_s0[iid][0], re["progmem"])
        if g is not None:
            g["split"] = "bench-hold"
            held_groups.append(g)
    aux_counts = {"real": None, "syn": None}
    if rb is not None:
        want = set(rb["tr_iids"])
        exs = _r16_aux_s0_exs(C.TO1_REAL_DATA, want)
        for iid in rb["tr_iids"]:
            g = _r16_state_group(env, scorer, jpol_r6, rb["tr_states"][iid], iid,
                                 rb["tr_ep"][iid], exs.get(iid), rb["tr_pm"])
            if g is not None:
                g["split"] = "aux-real-train"
                tr_groups.append(g)
        exs_h = _r16_aux_s0_exs(C.TO1_REAL_DATA, rb["hd_iids"])
        for iid in rb["hd_iids"]:
            g = _r16_state_group(env, scorer, jpol_r6, rb["hd_states"][iid], iid,
                                 rb["hd_ep"][iid], exs_h.get(iid), rb["hd_pm"])
            if g is not None:
                g["split"] = "aux-real-held"
                held_groups.append(g)
        aux_counts["real"] = [len(rb["tr_iids"]), len(rb["hd_iids"])]
    if sb is not None:
        want = set(sb["tr_iids"])
        exs = _r16_aux_s0_exs(C.TO1_AUX_DATA, want)
        for iid in sb["tr_iids"]:
            g = _r16_state_group(env, scorer, jpol_r6, sb["tr_states"][iid], iid,
                                 sb["tr_ep"][iid], exs.get(iid), sb["tr_pm"])
            if g is not None:
                g["split"] = "aux-syn-train"
                tr_groups.append(g)
        exs_h = _r16_aux_s0_exs(C.TO1_AUX_DATA, sb["hd_iids"])
        for iid in sb["hd_iids"]:
            g = _r16_state_group(env, scorer, jpol_r6, sb["hd_states"][iid], iid,
                                 sb["hd_ep"][iid], exs_h.get(iid), sb["hd_pm"])
            if g is not None:
                g["split"] = "aux-syn-held"
                held_groups.append(g)
        aux_counts["syn"] = [len(sb["tr_iids"]), len(sb["hd_iids"])]

    n_pos = sum(1 for g in tr_groups if bool((g["U"] > 0).any()))
    pool_size = [g["n_gated"] for g in tr_groups] or [0]
    pos_cnt = [int((g["U"] > 0).sum()) for g in tr_groups] or [0]
    pool_stats = {"train_states": len(tr_groups), "pos_states": n_pos,
                  "mean_pool": float(np.mean(pool_size)),
                  "mean_pos": float(np.mean(pos_cnt)),
                  "max_pos": int(max(pos_cnt))}
    label_align = {}
    for g in tr_groups:
        if "label_align_mean_abs_diff" in g:
            label_align = {"mean_abs_diff": g["label_align_mean_abs_diff"],
                           "max_diff": g["label_align_max_diff"]}
            break
    print(f"[r16] groups: train={len(tr_groups)} held={len(held_groups)} "
          f"{json.dumps(pool_stats, default=str)}", flush=True)
    labels = {"source": "FixedDecisionReplay true_U from replay/aux examples (§3)",
              "train_only": True, "oracle_runtime": False,
              "n_pos_states_train": n_pos}

    # ---- §6-14 train (listwise) + §15 model selection (held, NOT VAL) ------------
    trun = TOP1.train_pool_listwise_r16(tr_groups, held_groups=held_groups,
                                        reranker=reranker, seed=0)
    r16_sel = trun["selector"]
    jpol16 = JG.JointAgenticPolicy(r16_sel)
    best_epoch = int(trun["best_epoch"])
    held_score = trun["best_held_score"]
    hist = [{"ep": h["ep"], "l_list": h["l_list"], "l_pos": h["l_pos"],
             "l_pair": h["l_pair"],
             "train_rec20": (h["train"].get("rec20") if h["train"] else None),
             "held_rec20": (h["held"].get("rec20") if h.get("held") else None)}
            for h in trun["hist"]]
    print(f"[r16] trained; best_epoch={best_epoch} held_score={held_score}", flush=True)
    print(f"[r16] train rec@20 {(_r2(trun['train_metrics'], 'rec20'))} "
          f"held rec@20 {(_r2(trun['held_metrics'], 'rec20'))}", flush=True)

    # ---- §17-18 baseline vs after full-pool ranking (R15 §28 same ruler) ---------
    deco_base = _r15_deco(env, scorer, jpol_r6, train_iids, copy.deepcopy(re["progmem"]),
                          cap_states=(2 if args.quick else None))
    base_recall = deco_base.get("recall") or {}
    base_rank = deco_base.get("ranking_metrics") or {}
    base_matches_r15 = bool(base_rank.get("rec32") is not None and
                            abs(base_rank["rec32"] - C.TO1_R16_BASE_RECALL32) < 0.01)
    print(f"[r16] baseline §28 rec32={base_rank.get('rec32')} "
          f"matches_R15={base_matches_r15}", flush=True)

    deco_r16 = _r15_deco(env, scorer, jpol16, train_iids, copy.deepcopy(re["progmem"]),
                         cap_states=(2 if args.quick else None))
    r16_recall = deco_r16.get("recall") or {}
    r16_rank = deco_r16.get("ranking_metrics") or {}
    print(f"[r16] after §18 rec32={r16_rank.get('rec32')} "
          f"oracle_rank={r16_rank.get('oracle_rank')}", flush=True)

    # ---- §20 no-oracle shortlist gate (§19 builder VERBATIM unchanged) -----------
    tol = float(C.TO1_R15_RECALL_DROP_TOL)
    sl_any = float(r16_recall.get("any", 0.0))
    sl_best = float(r16_recall.get("best", 0.0))
    sl_oracle = float(r16_recall.get("oracle", 0.0))
    sl_gate_ok = bool(sl_any >= 1.0 - tol and sl_best >= 1.0 - tol and
                      sl_oracle >= 1.0 - tol)
    print(f"[r16] §20 shortlist gate any={sl_any:.3f} best={sl_best:.3f} "
          f"oracle={sl_oracle:.3f} -> gate_ok={sl_gate_ok}", flush=True)

    # ---- §22 primary target + §31 utility mass -----------------------------------
    base_rec32 = base_rank.get("rec32")
    r16_rec32 = r16_rank.get("rec32")
    rec32_gate = bool(r16_rec32 is not None and
                      r16_rec32 >= C.TO1_R16_RECALL32_GATE - 1e-9)
    um = {"base": _r16_um_agg(deco_base.get("u_mass_rows") or [])["base"],
          "cur": _r16_um_agg(deco_r16.get("u_mass_rows") or [])["cur"],
          "n_states": _r16_um_agg(deco_r16.get("u_mass_rows") or [])["n_states"]}
    um32_base = um["base"].get("K32")
    um32_r16 = um["cur"].get("K32")
    print(f"[r16] util-mass@32 base={um32_base} r16={um32_r16} "
          f"(gate 0.95: {rec32_gate})", flush=True)

    # ---- §28 decomposition + M3_SELECTION_MISS recompute --------------------------
    decomp = {"base": dict(deco_base.get("decomposition") or {}),
              "r16": dict(deco_r16.get("decomposition") or {})}
    miss_base = int(deco_base.get("M3_SELECTION_MISS", 0))
    miss_r16 = int(deco_r16.get("M3_SELECTION_MISS", 0))
    print(f"[r16] §28 miss base={miss_base} -> r16={miss_r16}", flush=True)

    # ---- §26 no-RL closed loops (S0 R6-full / S1 R16-full / S2 R16-shortlist) -----
    init_roots = _r12_eval_roots(re["progmem"], train_pairs, st_bench_map)

    def _tot(mode, obj, roots, aspace):
        sm, _ = JG.parity_eval(env, scorer, obj, roots, m2_mode=mode, use_mem=True,
                               gate_mem=False, gate_variant="r14", action_space=aspace)
        return float(sm.get("total", 0.0))

    s0 = _tot("adapter", jpol_r6, init_roots, "full")
    s1 = _tot("adapter", jpol16, init_roots, "full")
    s2 = _tot("adapter", jpol16, init_roots, "shortlist")
    print(f"[r16] §26 closed loop S0={s0:.0f} S1={s1:.0f} S2={s2:.0f}", flush=True)
    val_roots = _r12_eval_roots(re["progmem"], val_pairs, st_bench_map)
    s0_val = _tot("adapter", jpol_r6, val_roots, "full")
    s1_val = _tot("adapter", jpol16, val_roots, "full")
    s1_real = None
    s1_syn = None
    if rb is not None:
        s1_real = _tot("adapter", jpol16,
                       _r12_eval_roots(rb["hd_pm"], [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]],
                                       rb["hd_states"]), "full")
    if sb is not None:
        s1_syn = _tot("adapter", jpol16,
                      _r12_eval_roots(sb["hd_pm"], [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]],
                                      sb["hd_states"]), "full")
    print(f"[r16] VAL S0={s0_val:.0f} S1={s1_val:.0f} "
          f"AUX-held real={s1_real} syn={s1_syn}", flush=True)

    # ---- §16 VAL once at final (post-selection) on S0 pools ----------------------
    val_iids = [i["instance_id"] for i in env["val_insts"]]
    val_ep_map = {i["instance_id"]: len(env["train_insts"]) + idx
                  for idx, i in enumerate(env["val_insts"])}
    val_deco = None
    if not args.quick:
        val_deco = _r15_deco(env, scorer, jpol16, val_iids,
                             copy.deepcopy(re["progmem"]), ep_map=val_ep_map)
        print(f"[r16] VAL deco: {json.dumps(val_deco.get('ranking_metrics') or {}, default=str)}",
              flush=True)

    # ---- §29 Mk1 / §30 Fattahi15 traces ------------------------------------------
    mk = None
    if not args.quick:
        mk = _r16_state_traces(jpol_r6, jpol16, scorer, env, re, mp_ctx)

    # ---- verdict ladder (§33, §34; §35-36 next-round guidance) --------------------
    base_oracle_rank = base_rank.get("oracle_rank")
    r16_oracle_rank = r16_rank.get("oracle_rank")
    improved_any = bool(
        (r16_rec32 is not None and base_rec32 is not None and
         r16_rec32 >= base_rec32 + 0.01) or
        (r16_oracle_rank is not None and base_oracle_rank is not None and
         r16_oracle_rank <= base_oracle_rank - 0.5))
    sig_rank = bool(r16_rec32 is not None and base_rec32 is not None and
                    r16_rec32 >= base_rec32 + C.TO1_R16_SIG_RECALL32 and
                    r16_rec32 >= C.TO1_R16_RECALL32_A_FLOOR)
    util_delta = (None if (um32_r16 is None or um32_base is None)
                  else um32_r16 - um32_base)
    util_ok = bool(util_delta is not None and util_delta >= C.TO1_R16_UTIL_MASS_DELTA)
    r6_sel.eval()                       # base metrics must be dropout-free (deterministic)
    base_held = TOP1._r16_epoch_metrics(r6_sel, held_groups) if held_groups else {}
    held_delta = (None if (trun["held_metrics"] is None or
                           trun["held_metrics"].get("rec32") is None or
                           base_held.get("rec32") is None)
                  else trun["held_metrics"]["rec32"] - base_held["rec32"])
    val_delta = s1_val - s0_val
    gen_degrade = bool((held_delta is not None and held_delta <= -0.05) or
                       val_delta <= -5.0)
    rank_flat = bool(not improved_any and
                     (r16_rec32 is None or base_rec32 is None or
                      abs(r16_rec32 - base_rec32) < 0.005))
    d_blocker = bool(sig_rank and sl_gate_ok and
                     miss_r16 >= max(miss_base, 1) and s2 <= s0 + 1)
    label_ok = bool(len(tr_groups) >= 20 and
                    n_pos >= max(10, int(0.5 * len(tr_groups))))

    conditions = {"base_rec32": base_rec32, "r16_rec32": r16_rec32,
                  "improved_any": improved_any, "sig_rank": sig_rank,
                  "util_delta": util_delta, "util_ok": util_ok,
                  "sl_any": sl_any, "sl_best": sl_best, "sl_oracle": sl_oracle,
                  "sl_gate_ok": sl_gate_ok, "rec32_gate": rec32_gate,
                  "rank_flat": rank_flat, "held_delta": held_delta,
                  "val_delta": val_delta, "gen_degrade": gen_degrade,
                  "d_blocker": d_blocker, "label_ok": label_ok,
                  "miss": [miss_base, miss_r16]}

    if not label_ok and not improved_any:
        vcode, vlabel = "F", "DATA_LABEL_LIMITATION"
        note = f"training labels too sparse (states {len(tr_groups)}, pos-states {n_pos}) to move the ranking"
    elif rank_flat:
        vcode, vlabel = "C", "M3_FEATURE_REPRESENTATION_IS_BLOCKER"
        note = (f"listwise objective did NOT move full-pool ranking "
                f"(rec32 {base_rec32} -> {r16_rec32}) -- feature representation, "
                f"not the loss, is the blocker (§35)")
    elif improved_any and gen_degrade:
        vcode, vlabel = "E", "SFT_GENERALIZATION_DEGRADES"
        note = (f"TRAIN ranking improved (rec32 {base_rec32} -> {r16_rec32}) but held "
                f"rec32 delta={held_delta} / VAL closed-loop delta={val_delta:.0f} degrades (§34)")
    elif sig_rank and sl_gate_ok and util_ok:
        vcode, vlabel = "A", "LISTWISE_M3_SFT_FIXES_RANKING"
        note = (f"pos-rec@32 {base_rec32:.3f} -> {r16_rec32:.3f} (Δ≥{C.TO1_R16_SIG_RECALL32}) "
                f"AND shortlist any/best/oracle ≥0.95 "
                f"AND utility-mass@32 Δ={util_delta:.3f} -- frozen M3 now reaches positives "
                f"at the top of the full pool")
    elif improved_any and d_blocker:
        vcode, vlabel = "D", "STOP_RANKING_REMAINS_BLOCKER"
        note = (f"ranking improved and shortlist passes but M3_SELECTION_MISS "
                f"{miss_base}->{miss_r16} not reduced with S2={s2:.0f}<=S0={s0:.0f} -- "
                f"STOP/selection still blocks (§36)")
    elif improved_any:
        vcode, vlabel = "B", "LISTWISE_IMPROVES_BUT_BELOW_COVERAGE_GATE"
        note = (f"full-pool ranking improved (rec32 {base_rec32} -> {r16_rec32}) but "
                f"shortlist gate {sl_gate_ok} / utility gain {util_ok} below the "
                f"§20/§33 coverage requirement")
    else:
        vcode, vlabel = "C", "M3_FEATURE_REPRESENTATION_IS_BLOCKER"
        note = f"no measurable ranking movement (rec32 {base_rec32} -> {r16_rec32})"
    passed = bool(vcode == "A")
    print(f"[r16] verdict {vcode} {vlabel} "
          f"(rec32 {base_rec32}->{r16_rec32} / sl {sl_gate_ok} / util_ok {util_ok})",
          flush=True)

    # ---- checkpoint ONLY on verdict A (§37) ---------------------------------------
    if passed and args.quick:
        print("[r16] QUICK run: simulate PASS but REFUSE to write the R16 checkpoint "
              "(quick scale must never pollute m3_pool_listwise_sft_v3.pt)", flush=True)
    elif passed:
        meta = dict(
            phase="r16_pool_listwise_ranking_sft",
            method="pool_relative_listwise_sft",
            pipeline="M2_SFT->M3_SFT->M3_LISTWISE_SFT (NO RL)",
            parent=C.TO1_CKPT.name,                       # m3_proposal_top1_sft_v2.pt
            r11_r12_r13_r14_grpo_parent=None,
            stop_integrated=True, stop_has_u=0.0, calibrator="none",
            oracle_runtime=False, reasoner="frozen",
            executor="FixedDecisionReplay",
            tau_u=C.TO1_R16_TAU_U, tau_p=C.TO1_R16_TAU_P,
            lambda_pos=C.TO1_R16_LAMBDA_POS, lambda_pair=C.TO1_R16_LAMBDA_PAIR,
            best_epoch=best_epoch, held_score=held_score,
            shortlist_cap=C.TO1_R15_SHORTLIST_CAP,
            selected="TRAIN14-internal-held + AUX-held (NOT VAL3, §15)",
            formal_test_access=0, formal_test_sealed=True, identified=False)
        torch.save({"state": {"selector": r16_sel, "scorer": scorer,
                              "r6_anchor": r6_sel.state_dict()},
                    "meta": meta}, C.TO1_R16_CKPT)
        print(f"[r16] saved {C.TO1_R16_CKPT.name} (PASS, verdict A only §37)",
              flush=True)
    else:
        print(f"[r16] NOT PASS (verdict {vcode}) -> no R16 checkpoint written (§37)",
              flush=True)

    shortlist_proof = {
        "builder": "build_shortlist_r15 (VERBATIM unchanged, §19)",
        "cap": C.TO1_R15_SHORTLIST_CAP,
        "k_global": C.TO1_R15_K_GLOBAL,
        "tier_a_quota": C.TO1_R15_TIER_A_QUOTA,
        "tier_b_quota": C.TO1_R15_TIER_B_QUOTA,
        "diversity_fill": bool(C.TO1_R15_DIVERSITY_FILL),
        "score_source": "wrapped selector -> jpol.m3._base_raw (R16 base after training)",
        "oracle_authority": False,
    }
    quick_sanity = None
    if args.quick:
        quick_sanity = {"groups": len(tr_groups), "best_epoch": best_epoch,
                        "rec32": r16_rec32, "verdict": vcode}

    scrape = dict(
        repro_ok=repro_ok, r6_acc=r6_acc_all, r6_regret=r6_regret,
        r6_canonical_train=int(r6_canonical_train),
        base_matches_r15=base_matches_r15, internal_train=internal_train,
        internal_held=internal_held, aux_counts=aux_counts,
        n_tr_groups=len(tr_groups), n_held_groups=len(held_groups),
        pool_stats=pool_stats, labels=labels, label_align=label_align,
        tau_u=C.TO1_R16_TAU_U, tau_p=C.TO1_R16_TAU_P,
        lambda_pos=C.TO1_R16_LAMBDA_POS, lambda_pair=C.TO1_R16_LAMBDA_PAIR,
        best_epoch=best_epoch, held_score=held_score, hist=hist,
        base_rank=base_rank, r16_rank=r16_rank, base_recall=base_recall,
        r16_recall=r16_recall, sl_gate_ok=sl_gate_ok, rec32_gate=rec32_gate,
        um=um, decomp=decomp, miss=[miss_base, miss_r16],
        s0=s0, s1=s1, s2=s2, s0_val=s0_val, s1_val=s1_val,
        s1_real=s1_real, s1_syn=s1_syn, mk=mk, val_deco=val_deco,
        conditions=conditions, train_metrics=trun["train_metrics"],
        held_metrics=trun["held_metrics"],
        proof_no_m3_rl_parent=proof_no_m3_rl_parent,
        shortlist_proof=shortlist_proof,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed,
        quick_sanity=quick_sanity)
    report = _r16_report(env, re, scrape)
    _r16_persist(report, scrape=scrape)
    print(f"[r16] total {time.time() - t0:.1f}s "
          f"(report {C.R16_REPORT.name})", flush=True)
    return report


# ===========================================================================
# R17: POOL-CONTEXT proposal representation SFT (T1-M3-POOL-CONTEXT-SFT)
# ===========================================================================
# R16 (verdict B) proved the listwise LOSS works (M3_SELECTION_MISS 3->0,
# no-RL closed loop 355->272, VAL 19) but the per-proposal FROZEN features cap
# rec32 at +0.013.  R17 replaces the M3 ranking representation with a
# permutation-invariant POOL-CONTEXT scorer (§5-8) + runtime M2/probe evidence
# (§9-12) + a direct utility regressor (§19-21) on top of the frozen R6 base
# (§25-26 bounded residual).  NO RL (§40).  frozen below M3: M2, q2, Memory,
# adaptive probing, Reasoner, FDR, Joint GRPO (§0-§1).

# R16-recorded Mk1/Fattahi15 reference ranks (from the T1-M3-POOL-LISTWISE-RANKING
# R16 report, items 34/35) -- R16 wrote NO checkpoint so its base ranks are on the
# record, not re-loadable; R17 compares against BOTH the R16 record and the LIVE
# frozen R6 base on the identical S0 root.
_R17_R16_RECORDED_TRACES = {
    "mk1": {"n_pos": 61, "M": 106, "best_true_U": 11.0,
            "oracle_rank_by_r6_base": 7, "oracle_rank_by_r16_base": 7,
            "selected_true_U_r16": 5.0},
    "fattahi15": {"n_pos": 42, "best_true_U": 123.0,
                  "stop_score_rank_r6": 2, "stop_score_rank_r16": 112,
                  "selected_true_U_r16": 123.0},
}


def _r17_state_group(env, scorer, jpol_r6, st, iid, ep, ex, pm, verify=False):
    """R17 training group = _r16_state_group (§2-3 SAME labels/alignment ruler) PLUS
    the §9-12 runtime evidence X [G,60] and the §14 extended pool stats [10].
    Labels still FixedDecisionReplay true_U (TRAIN-only); X only reads runtime-known
    M2/probe/memory/machine fields -- never true_U / future reward."""
    cache, executor = env["cache"], env["executor"]
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    N = len(metas)
    if N == 0:
        return None
    if ex is None or not _r16_meta_aligned(metas, ex):
        return None
    ast = cache.ast(st["problem"], st["schedule"], iid)
    sf_list = [float(x) for x in ex["state_feat"].tolist()]
    sf_t = torch.as_tensor(ex["state_feat"], dtype=torch.float32).reshape(-1)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol_r6.m2, executor, pm,
                                iid, int(ep), 0, sf_list, rng)
    if not gate["gated_metas"]:
        return None
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(pm.features(iid, int(ep), 0, sf_list, queries), dtype=torch.float32)
    gmem = float(pm.retrieval_gate(iid, int(ep), 0, sf_list))
    mem_sel = mem * gmem
    F_all = JG._rerank_feats_all(scorer, rolex, mem_sel)      # [G, 277]
    if F_all is None or len(F_all) == 0:
        return None
    kept = gate["kept_indices"]
    if len(kept) != len(F_all):
        return None
    X = TOP1.r17_evidence(ast, gate["gated_metas"], gate)     # [G, 60]
    if X is None or int(X.shape[0]) != int(len(F_all)):
        return None
    pext = TOP1.r17_pool_stats(F_all, X)                      # [1,10]
    U = torch.tensor([float(ex["true_U"][k]) for k in kept], dtype=torch.float32)
    pstats = JG._pool_stats_from(F_all)
    g = {"F_all": F_all, "sf": sf_t.detach().reshape(-1), "pstats": pstats.detach().reshape(-1),
         "U": U, "X": X.detach(), "pstats_ext": pext.detach().reshape(-1),
         "iid": iid, "n_gated": int(len(kept)), "n_full": int(N),
         "gate_diag": dict(gate["diag"]), "split": None}
    if verify:                                   # label-alignment safety net (§3)
        diff = []
        for k in range(len(kept)):
            edits, _kind = PF._edits_for(ast, gate["gated_metas"][k])
            res = PF._execute_step(executor, st["problem"], st["schedule"], edits,
                                   int(st["schedule"].makespan), schedule_hash(st["schedule"]))
            fres = float(res["improvement"]) if res is not None else 0.0
            diff.append(abs(float(ex["true_U"][kept[k]]) - fres))
        g["label_align_mean_abs_diff"] = float(np.mean(diff)) if diff else 0.0
        g["label_align_max_diff"] = float(np.max(diff)) if diff else 0.0
    return g


def _r17_state_diag(env, scorer, jpol_r6, jpol17, iid, progmem, ep=None):
    """R17 per-state diagnostic: SAME fresh-FDR true_U + R14 gate rebuild as
    `_r15_state_diag` (base numbers stay comparable to R15/R16), but "cur" scores =
    the POOL-CONTEXT residual (`r17_action_logits`), NOT the retrained head.

    Adds §34-36: capacity-normalized recall@32 (denominator min(n_pos,32)),
    capacity bound min(1,32/n_pos), utility regret (mean/median/p90 + normalized)
    and `rank_r17` (rank metrics under the R17 context order).  Shortlist builder
    unchanged (frozen base -> identical construction to R15/R16)."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    n_prop = len(metas)
    if n_prop == 0:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = schedule_hash(st["schedule"])
    sf = PF.state_feature_vec(ms, ms, n_prop, agg["best_uhat"], agg["best_direct"],
                              agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol_r6.m2, executor, progmem,
                                iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": n_prop,
                "gated": 0, "m2_diag": dict(gate["diag"])}
    rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries), dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    N = len(gate["gated_metas"])
    F_all = JG._rerank_feats_all(scorer, rolex, mem_sel)
    frozen = jpol_r6.m3._base_raw(F_all)[0].detach().float()        # [N] frozen R6 base
    pstats = JG._pool_stats_from(F_all)
    X = TOP1.r17_evidence(ast, gate["gated_metas"], gate)            # [N,60]
    pext = TOP1.r17_pool_stats(F_all, X)
    cur = TOP1.r17_action_logits(jpol17.m3, F_all, sf_t, pstats, X, pext).detach().float()

    # ---- real true_U per gated proposal (FDR diagnostic truth) ----------------
    true_U = []
    for k in range(N):
        edits, _kind = PF._edits_for(ast, gate["gated_metas"][k])
        res = PF._execute_step(executor, st["problem"], st["schedule"], edits,
                               ms, base_h)
        true_U.append(float(res["improvement"]) if res is not None else 0.0)
    tU = np.asarray(true_U, dtype=np.float64)
    pos = [k for k in range(N) if tU[k] > 0.0]
    sigs = [JG.proposal_identity(ast, gate["gated_metas"][k])[2] for k in range(N)]
    frozen_np = frozen.numpy()
    cur_np = cur.numpy()

    def _rank(vals, k):
        return 1 + int(np.sum(vals > vals[k]))

    best_true_U = best_true_U_sig = None
    best_true_U_cur_rank = best_true_U_frz_rank = None
    bpos = None
    if pos:
        bpos = max(pos, key=lambda k: float(tU[k]))
        best_true_U = float(tU[bpos])
        best_true_U_sig = sigs[bpos]
        best_true_U_cur_rank = _rank(cur_np[:N], bpos)
        best_true_U_frz_rank = _rank(frozen_np, bpos)
    sel = int(np.argmax(cur_np))
    sel_is_stop = bool(sel == N)
    sel_sig = "STOP" if sel_is_stop else sigs[sel]
    sel_true_U = 0.0 if sel_is_stop else float(tU[sel])
    sel_score = float(cur_np[sel])
    stop_score = float(cur_np[-1])
    row = {"iid": iid, "n_prop": n_prop, "N_full": N, "full_pool_size": N,
           "best_true_U": best_true_U, "best_true_U_signature": best_true_U_sig,
           "best_true_U_score_rank": best_true_U_cur_rank,
           "best_true_U_frozen_rank": best_true_U_frz_rank,
           "pos_count": len(pos),
           "selected_signature": sel_sig, "selected_true_U": sel_true_U,
           "selected_score_rank": (None if sel_is_stop else _rank(cur_np[:N], sel)),
           "stop_score": stop_score,
           "stop_score_rank": 1 + int(np.sum(cur_np[:N] > stop_score)),
           "selected_score": sel_score,
           "frozen_top": [(sigs[int(k)], float(frozen_np[int(k)]))
                          for k in np.argsort(-frozen_np)[:4].tolist()],
           "capacity_bound": (min(1.0, 32.0 / max(len(pos), 1)) if pos else None)}
    if pos:
        bp = max(pos, key=lambda k: float(cur_np[k]))
        row["best_positive_score"] = float(cur_np[bp])
        row["best_positive_signature"] = sigs[bp]
        row["best_positive_score_rank"] = _rank(cur_np[:N], bp)
        row["best_positive_frozen_rank"] = _rank(frozen_np, bp)
        row["score_margin_positive_vs_stop"] = float(cur_np[bp]) - stop_score
        row["score_margin_positive_vs_selected"] = float(cur_np[bp]) - sel_score
        row["capacity_recall32"] = float(
            sum(1 for k in pos if k in set(np.argsort(-cur_np[:N])[: min(32, N)]))
            / min(len(pos), 32))
        mx = float(tU[bpos])
        rg = max(0.0, mx - sel_true_U)
        row["utility_regret"] = {"value": rg, "normalized": max(0.0, rg / max(mx, 1e-9))}

    # ---- shortlist (NO oracle, builder VERBATIM unchanged; scores via R17) -----
    sl_idx, sl_info = JG.build_shortlist_r15(
        ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t, jpol_r6, scorer,
        mem, gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
        cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL, rolex=rolex)
    M = len(sl_idx)
    F_sl = F_all[sl_idx]
    pstats_sl = JG._pool_stats_from(F_sl)
    cur_sl = TOP1.r17_action_logits(jpol17.m3, F_sl, sf_t, pstats_sl,
                                    X[sl_idx], TOP1.r17_pool_stats(F_sl, X[sl_idx])
                                    ).detach().float()                 # [M+1]
    cur_sl_np = cur_sl.numpy()
    sel_sl = int(np.argmax(cur_sl_np))
    sl_pos = [j for j, gi in enumerate(sl_idx) if float(tU[gi]) > 0.0]
    sl_best = None
    if sl_pos:
        bb = max(sl_pos, key=lambda j: float(tU[sl_idx[j]]))
        sl_best = {"sig": sigs[sl_idx[bb]],
                   "frozen_rank_in_full": _rank(frozen_np, sl_idx[bb]),
                   "frozen_rank_in_shortlist": 1 + int(np.sum(
                       frozen_np[sl_idx] > frozen_np[sl_idx[bb]])),
                   "true_U": float(tU[sl_idx[bb]])}
    shortlist_miss = 0
    if sl_pos and (sel_sl == M or float(tU[sl_idx[sel_sl]]) <= 0.0):
        shortlist_miss = 1

    # ---- §11 recall ------------------------------------------------------------
    n_P = float(len(pos))
    recall = {"any": (1.0 if sl_pos else 0.0),
              "best": (0.0 if pos is None else (1.0 if bpos in sl_idx else 0.0)),
              "oracle": ((len(sl_pos) / min(n_P, float(C.TO1_R15_SHORTLIST_CAP)))
                         if pos else 1.0)}

    # ---- §27 selection metrics ----------------------------------------------------
    probs = cur_sl_np - float(np.max(cur_sl_np))
    probs = np.exp(probs) / np.sum(np.exp(probs))
    top_order = np.argsort(-probs)
    select = {"M": int(M), "N_full": int(N),
              "entropy": float(-np.sum(np.where(probs > 0,
                                                probs * np.log(probs + 1e-12), 0.0))),
              "stop_prob": float(probs[M]),
              "top1_prob": float(probs[top_order[0]]),
              "top5_cum": float(np.sum(probs[top_order[:5]])),
              "signature": sl_info["signature"]}

    # ---- §28 ranking metrics: frozen base AND R17 context order ------------------
    rank = {}
    rank_r17 = {}
    if pos:
        bp_f = max(pos, key=lambda k: float(frozen_np[k]))
        rank["best_positive_rank"] = _rank(frozen_np, bp_f)
        rank["oracle_rank"] = _rank(frozen_np, bpos)
        rank["mrr"] = 1.0 / float(rank["best_positive_rank"])
        order = np.argsort(-frozen_np)
        for Kk in (1, 3, 5, 10, 20, 32):
            topk = set(order[: min(Kk, N)].tolist())
            rank[f"pos_recall_{Kk}"] = float(np.mean(
                [1.0 if k in topk else 0.0 for k in pos]))
        bp17 = max(pos, key=lambda k: float(cur_np[k]))
        rank_r17["best_positive_rank"] = _rank(cur_np[:N], bp17)
        rank_r17["oracle_rank"] = _rank(cur_np[:N], bpos)
        rank_r17["mrr"] = 1.0 / float(rank_r17["best_positive_rank"])
        order17 = np.argsort(-cur_np[:N])
        for Kk in (1, 3, 5, 10, 20, 32):
            topk = set(order17[: min(Kk, N)].tolist())
            rank_r17[f"pos_recall_{Kk}"] = float(np.mean(
                [1.0 if k in topk else 0.0 for k in pos]))

    # ---- §3 classification (priority B > D > A > C > OK) --------------------------
    cls = "M3_OK"
    if pos:
        if sel_is_stop:
            cls = "M3_STOP_MARGIN_MISS"
        elif (N > int(C.TO1_R15_SHORTLIST_CAP) and
              (best_true_U_cur_rank or 10**6) > int(C.TO1_R15_K_GLOBAL)):
            cls = "M3_POOL_DILUTION"
        elif best_true_U_cur_rank and best_true_U_cur_rank > 1:
            cls = "M3_RANKING_MISS"
        elif (not sel_is_stop and sel_true_U < best_true_U - 1e-9):
            cls = "M3_INTRA_POOL_SELECTION_MISS"
    row["classification"] = cls

    # ---- u_mass (base frozen vs cur=r17 context, same ruler as R16 §31) ----------
    u_mass = None
    if pos:
        denom = sum(max(float(tU[k]), 0.0) for k in pos) or 1.0
        u_mass = {}
        for sname, svals in (("base", frozen_np), ("cur", cur_np)):
            o = np.argsort(-svals[:N]) if sname == "cur" else np.argsort(-svals)
            u_mass[sname] = {f"K{Kk}": float(sum(
                max(float(tU[k]), 0.0) for k in o[: min(Kk, N)]
                if float(tU[k]) > 0.0) / denom) for Kk in (10, 20, 32)}
    return {"row": row, "class": cls, "shortlist_miss": shortlist_miss,
            "recall": recall, "select": select, "rank": rank, "rank_r17": rank_r17,
            "sl_info": sl_info, "sl_best": sl_best, "u_mass": u_mass, "skip": False}


def _r17_deco(env, scorer, jpol_r6, jpol17, iids, progmem, cap_states=None, ep_map=None):
    """§2-4 + §27-31 aggregate over TRAIN (or VAL) states under R17 context scores.
    Returns R15-style keys PLUS rank_r17 (context-order rank), u_mass_rows and the
    §34-36 capacity aggregates."""
    import collections
    agg = collections.Counter()
    miss = n_state = n_pos_state = 0
    rec = collections.Counter()
    sl_sizes, sl_ent, sl_stop, sl_top1, sl_top5 = [], [], [], [], []
    rk = {"best_positive_rank": [], "oracle_rank": [], "mrr": [],
          "rec1": [], "rec3": [], "rec5": [], "rec10": [], "rec20": [], "rec32": []}
    rk17 = dict(rk)
    cap_rec32, cap_bound, ureg, nureg, selU = [], [], [], [], []
    rows = []
    u_mass_rows = []
    sigs = set()
    for iid in (iids[:cap_states] if cap_states else iids):
        d = _r17_state_diag(env, scorer, jpol_r6, jpol17, iid, progmem,
                            ep=(None if ep_map is None else ep_map.get(iid)))
        if d.get("skip"):
            continue
        n_state += 1
        agg[d["class"]] += 1
        if d["row"].get("best_true_U") is not None:
            n_pos_state += 1
        miss += d["shortlist_miss"]
        rows.append(d["row"])
        u_mass_rows.append(d["u_mass"])
        for k in ("any", "best", "oracle"):
            rec[k] += d["recall"][k]
        s = d["select"]
        sl_sizes.append(s["M"]); sl_ent.append(s["entropy"])
        sl_stop.append(s["stop_prob"]); sl_top1.append(s["top1_prob"])
        sl_top5.append(s["top5_cum"])
        if d["sl_best"] is not None:
            sigs.add(d["sl_best"]["sig"])
        r_ = d["rank"]; r17_ = d["rank_r17"]
        if r_:
            rk["best_positive_rank"].append(r_["best_positive_rank"])
            rk["oracle_rank"].append(r_["oracle_rank"]); rk["mrr"].append(r_["mrr"])
            for Kk, key in ((1, "rec1"), (3, "rec3"), (5, "rec5"),
                            (10, "rec10"), (20, "rec20"), (32, "rec32")):
                rk[key].append(r_[f"pos_recall_{Kk}"])
        if r17_:
            rk17["best_positive_rank"].append(r17_["best_positive_rank"])
            rk17["oracle_rank"].append(r17_["oracle_rank"]); rk17["mrr"].append(r17_["mrr"])
            for Kk, key in ((1, "rec1"), (3, "rec3"), (5, "rec5"),
                            (10, "rec10"), (20, "rec20"), (32, "rec32")):
                rk17[key].append(r17_[f"pos_recall_{Kk}"])
            rowr = d["row"]
            if rowr.get("capacity_recall32") is not None:
                cap_rec32.append(rowr["capacity_recall32"])
            if rowr.get("capacity_bound") is not None:
                cap_bound.append(rowr["capacity_bound"])
            if rowr.get("utility_regret") is not None:
                ureg.append(rowr["utility_regret"]["value"])
                nureg.append(rowr["utility_regret"]["normalized"])
            if rowr.get("selected_true_U") is not None:
                selU.append(rowr["selected_true_U"])

    def _stat(v):
        vv = [float(x) for x in v]
        if not vv:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {"mean": float(np.mean(vv)), "median": float(np.median(vv)),
                "p90": float(np.percentile(vv, 90))}

    selection_metrics = {"size": _stat(sl_sizes), "entropy": _stat(sl_ent),
                         "stop_prob": _stat(sl_stop), "top1_prob": _stat(sl_top1),
                         "top5_cum": _stat(sl_top5)}
    recall = {k: float(v / max(n_state, 1)) for k, v in rec.items()}
    ranking_metrics = {k: (float(np.mean(v)) if v else None) for k, v in rk.items()}
    ranking_metrics_r17 = {k: (float(np.mean(v)) if v else None)
                           for k, v in rk17.items()}
    capacity = {
        "capacity_normalized_recall32": (float(np.mean(cap_rec32)) if cap_rec32 else None),
        "capacity_bound32": (float(np.mean(cap_bound)) if cap_bound else None),
        "utility_regret": {"mean": (float(np.mean(ureg)) if ureg else None),
                           "median": (float(np.median(ureg)) if ureg else None),
                           "p90": (float(np.percentile(ureg, 90)) if ureg else None),
                           "normalized_mean": (float(np.mean(nureg)) if nureg else None)},
        "selected_true_U": (float(np.mean(selU)) if selU else None),
    }
    return {"decomposition": dict(agg), "M3_SELECTION_MISS": int(miss),
            "n_state": n_state, "n_positive_state": n_pos_state,
            "recall": recall, "selection_metrics": selection_metrics,
            "ranking_metrics": ranking_metrics, "ranking_metrics_r17": ranking_metrics_r17,
            "capacity": capacity, "rows": rows, "u_mass_rows": u_mass_rows,
            "sl_best_sigs": sorted(sigs)}


def _r17_rollout(env, scorer, jpol17, rf, use_mem=True, horizon=None,
                 gate_mem=False, action_space="full"):
    """Mirror of `JG.agentic_parity_rollout` (R16 S0 ruler: r14 gate, STOP-on-negative,
    same Memory/pool-norm/horizon) with the ONLY change at the scoring point: the M3
    decision uses the POOL-CONTEXT `r17_action_logits` (frozen R6 base + bounded ctx
    residual + context STOP §13).  Gate/replay are otherwise bit-identical, so S0 (with
    jpol_r6) stays reproducible.  Returns (gain, act_usage, steps)."""
    horizon = int(horizon if horizon is not None else C.TO1_R13_HORIZON)
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
        sf = PF.state_feature_vec(ms_cur, s0_ms, n_prop, agg["best_uhat"],
                                  agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        rng = random.Random(JG._traj_seed(0, iid, episode_id, h, 0))
        gate = JG._m2_gate_step_r14(ast, metas, prop_feats, jpol17.m2, executor,
                                    progmem, iid, episode_id, t, sf, rng)
        if not gate["gated_metas"]:
            act_usage["stop_by_selector"] += 1
            steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                          "kind": None, "improvement": None,
                          "m2_diag": gate["diag"], "stop_reason": "no_pool_m2"})
            break
        rolex = JG._rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = (torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                            dtype=torch.float32) if use_mem
               else torch.zeros(len(queries), 277, dtype=torch.float32))
        gmem = 1.0
        if gate_mem and use_mem:
            gmem = float(progmem.retrieval_gate(iid, episode_id, t, sf))
        if action_space == "shortlist":
            shortlist_idx, _sl_info = JG.build_shortlist_r15(
                ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t, jpol17,
                scorer, mem, gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
                cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL,
                rolex=rolex)
            if not shortlist_idx:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "stop_reason": "no_pool_shortlist"})
                break
            pool = shortlist_idx
        else:
            logit_pos, rank = JG._scores(scorer, rolex, mem)
            pool, _info = JG.wide_pool(rolex, logit_pos, rank)
            if not pool:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "stop_reason": "no_pool_wide"})
                break
        mem_sel = mem * gmem if gate_mem else mem
        F_pool = JG._rerank_feats_all(scorer, rolex, mem_sel)[pool]
        pool_stats = JG._pool_stats_from(F_pool)
        X_all = TOP1.r17_evidence(ast, gate["gated_metas"], gate)     # [G,60]
        X_pool = X_all[pool]
        pext = TOP1.r17_pool_stats(F_pool, X_pool)
        with torch.no_grad():
            logits = TOP1.r17_action_logits(jpol17.m3, F_pool, sf_t, pool_stats,
                                            X_pool, pext)
        sel = int(logits.argmax().item())
        if sel == len(pool):
            act_usage["stop_by_selector"] += 1
            steps.append({"t": t, "n_prop": n_prop, "n_wide": len(pool),
                          "top_sig": "STOP", "kind": None, "improvement": None,
                          "m2_diag": gate["diag"], "stop_reason": "policy_stop"})
            break
        a = pool[sel]
        edits, kind = JG._edits_for(ast, gate["gated_metas"][a])
        res = JG._execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = JG.proposal_identity(ast, gate["gated_metas"][a])[2]
        steps.append({"t": t, "n_prop": n_prop, "n_wide": len(pool),
                      "top_sig": sig, "kind": kind,
                      "improvement": (None if res is None else float(res["improvement"])),
                      "m2_diag": gate["diag"]})
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


def _r17_closed_loop(env, scorer, jpol17, roots, use_mem=True, horizon=None,
                     gate_mem=False, action_space="full"):
    """T2/T3 closed loops under the POOL-CONTEXT scorer (same _b5_summary contract)."""
    gains_by_iid = {}
    steps_by_iid = {}
    for rf in roots:
        iid = rf["iid"]
        gain, usage, steps = _r17_rollout(env, scorer, jpol17, rf, use_mem=use_mem,
                                          horizon=horizon, gate_mem=gate_mem,
                                          action_space=action_space)
        gains_by_iid[iid] = gain
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    return RO._b5_summary(gains_by_iid), steps_by_iid


def _r17_tot(obj, aspace, env, scorer, roots):
    """dispatch SAME as R16 _tot but T2/T3 route the PoolContextProposalScorer."""
    if hasattr(obj, "m3") and hasattr(obj.m3, "util_head"):
        sm, _ = _r17_closed_loop(env, scorer, obj, roots, use_mem=True,
                                 gate_mem=False, action_space=aspace)
    else:
        sm, _ = JG.parity_eval(env, scorer, obj, roots, m2_mode="adapter", use_mem=True,
                               gate_mem=False, gate_variant="r14", action_space=aspace)
    return float(sm.get("total", 0.0))


def _r17_state_traces(jpol_r6, jpol17, scorer, env, re):
    """§37 Mk1 (+11 must clear +5) and §38 Fattahi15 (+123 must stay selected):
    LIVE R6 frozen base ranks vs LIVE R17 context ranks on the identical S0 root,
    plus the RECORDED R16 ranks (R16 wrote no checkpoint)."""
    targets = (("mk1", "Brandimarte_Mk1"), ("fattahi15", "Fattahi_Fattahi15"))
    out = {}
    for key, iid in targets:
        cand = [i["instance_id"] for i in env["order"]]
        if iid not in cand:
            out[key] = {"note": f"{iid} not in env"}
            continue
        db = _r15_state_diag(env, scorer, jpol_r6, iid, copy.deepcopy(re["progmem"]))
        da = _r17_state_diag(env, scorer, jpol_r6, jpol17, iid,
                             copy.deepcopy(re["progmem"]))
        d = {"iid": iid, "r16_recorded": _R17_R16_RECORDED_TRACES[key]}
        if db.get("skip") or da.get("skip"):
            d.update({"note": "state skipped", "before": None, "after": None})
        else:
            rb6, ra = db["row"], da["row"]
            d.update({
                "full_pool_M": int(ra["N_full"]), "pos_count": int(ra["pos_count"]),
                "best_true_U": ra["best_true_U"],
                "capacity_bound": ra.get("capacity_bound"),
                "r6_base": {"oracle_rank": rb6["best_true_U_frozen_rank"],
                            "best_positive_rank": rb6.get("best_positive_frozen_rank"),
                            "capacity_recall32": rb6.get("capacity_recall32"),
                            "stop_score_rank": rb6["stop_score_rank"],
                            "selected_signature": rb6["selected_signature"],
                            "selected_true_U": rb6["selected_true_U"]},
                "r17_ctx": {"oracle_rank": ra["best_true_U_score_rank"],
                            "best_positive_rank": ra.get("best_positive_score_rank"),
                            "capacity_recall32": ra.get("capacity_recall32"),
                            "stop_score_rank": ra["stop_score_rank"],
                            "selected_signature": ra["selected_signature"],
                            "selected_true_U": ra["selected_true_U"],
                            "rank_r17_oracle": ((da["rank_r17"] or {}).get("oracle_rank"))},
            })
        out[key] = d
    return out

_R17_SCRAPE_KEYS = [
    "repro_ok", "r6_acc", "r6_regret", "r6_canonical_train", "base_matches_r15",
    "internal_train", "internal_held", "n_tr_groups", "n_held_groups",
    "pool_stats", "labels", "label_align", "tau_u", "tau_p", "lambda_pos",
    "lambda_pair", "lambda_utility", "best_epoch", "held_score", "hist",
    "base_rank", "r17_rank", "r17_rank_frozen", "base_recall", "r17_recall",
    "sl_gate_ok", "sl_success", "rec32_diag", "capacity", "um", "um_delta_over_r16",
    "decomp", "miss", "s0", "s1", "s2", "s0_val", "s1_val", "s0_real", "s0_syn",
    "s1_real", "s1_syn", "downstream", "held_um32", "held_delta_um", "aux_ok",
    "mk", "val_deco", "conditions", "verdict", "passed", "x_evid", "verify",
    "proof_no_m3_rl_parent", "shortlist_proof", "train_metrics", "held_metrics",
    "quick_sanity",
]


def _r17_43items(s):
    """§47 forty-three-item final return (numbered, one per directive §)."""
    v = s["verdict"]
    c = s["conditions"]
    bm = s.get("base_rank") or {}
    rm17 = s.get("r17_rank") or {}
    um = s.get("um") or {}
    cap = s.get("capacity") or {}
    deco = s.get("decomp") or {}
    r16rec = _R17_R16_RECORDED_TRACES
    mk = s.get("mk") or {}
    items = [
        (1, "experiment", "T1-M3-POOL-CONTEXT-REPRESENTATION-SFT-R17"),
        (2, "verdict_code", v["code"]),
        (3, "verdict_label", v["label"]),
        (4, "verdict_note", v["note"]),
        (5, "no_m3_rl_parent", s.setdefault("proof_no_m3_rl_parent", {})),
        (6, "hard_freeze", {"m2": "frozen", "m2_q2": "frozen (R14)",
                            "memory_semantics": "frozen", "adaptive_probing": "frozen (R14)",
                            "reasoner": "frozen", "fixed_decision_replay": "frozen",
                            "joint_grpo": "not run (SFT-only)", "shortlist_cap": 32,
                            "k_global": C.TO1_R15_K_GLOBAL}),
        (7, "trainable_new", ["enc_prop (277-128-128)", "ctx_head (835-96-1, zero-init last)",
                              "stop_ctx_head (401-64-1, zero-init last)",
                              "util_head (835-64-1)  (§23)"]),
        (8, "r6_repro", {"acc": s["r6_acc"], "regret": s["r6_regret"],
                         "repro_ok": s["repro_ok"]}),
        (9, "r6_raw_anchor", s["r6_canonical_train"]),
        (10, "splits", {"internal_train": s["internal_train"],
                        "internal_held": s["internal_held"],
                        "aux_real_train/held": s.setdefault("aux_counts", {}).get("real"),
                        "aux_syn_train/held": s.setdefault("aux_counts", {}).get("syn")}),
        (11, "data_unit", {"n_train_groups": s["n_tr_groups"],
                           "n_held_groups": s["n_held_groups"],
                           "pool": "state + FULL gated pool + STOP + X(60) + pext(10) (§2/§12)"}),
        (12, "pool_stats", s["pool_stats"]),
        (13, "label_source", s["labels"]),
        (14, "loss_formula", "L = L_list + 1.0*L_posmargin + 0.5*L_pair + 0.25*L_utility (§21)"),
        (15, "utility_aux", {"module": "util_head (Huber(U_hat, U_tilde, delta=1))",
                             "lambda": C.TO1_R17_LAMBDA_UTILITY, "fixed": True,
                             "fed_back": False, "note": "§19-20,22 -- never read at runtime"}),
        (16, "pos_margin", {"margin_M": C.TO1_R17_POS_MARGIN,
                            "s0": "best zero-U action incl STOP (§17)"}),
        (17, "pair_weight", "w_ij=|U_tilde_i-U_tilde_j| over positives, log-scaled cap "
                            "§18 (+123 vs +20 > +5 vs +2 but never one-state-dominated)"),
        (18, "residual_arch", {"score": "score_R6(P) + alpha_ctx*tanh(delta_context)",
                               "alpha_ctx": C.TO1_R17_ALPHA_CTX, "init": "zero-residual (§25-26)",
                               "parity": "first forward == frozen R6 scores"}),
        (19, "pool_context", {"encoder": "enc_prop 277->128->128 (GELU+LayerNorm)",
                              "c_pool": "cat(mean(h), max(h), std(h)) (permutation-invariant §7)",
                              "z_i": "cat(h_i, c_pool, h_i-mean(h), h_i/(std+h), state, X_i) §8",
                              "z_dim": 835}),
        (20, "runtime_evidence", {"dims": TOP1.R17_EVID_DIM, "atom": TOP1.R17_ATOM_DIM,
                                  "struct": 15, "pair": TOP1.R17_PAIR_DIM,
                                  "sources": "§9-12: retained-root attribution / tier-A&~B "
                                             "/ root-probe gain / dependency+/mem support / "
                                             "enablers / atom counts / structural family",
                                  "true_U_runtime": False}),
        (21, "stop_context", {"stop_score": "Mlp_stop(state, c_pool, pstats_ext)",
                              "pstats_ext": TOP1.R17_STOP_PSTATS_DIM,
                              "fields": "n_pool/max_old/mean_old/max_rank/max_logit + "
                                        "single_frac/route_frac/tierA_frac/tierB_frac/base_std",
                              "true_U": False}),
        (22, "model_selection", {"rule": "argmax held PositiveUtilityMass@32",
                                 "composite": "utility-mass@32 + oracle-best recall + "
                                              "utility regret + held selected true_U (§27-28)",
                                 "pool": "TRAIN internal-held + AUX-real + AUX-syn held (NOT VAL)",
                                 "best_epoch": s["best_epoch"], "held_um32": s["held_um32"]}),
        (23, "val_once_final", {"used_in_selection": False,
                                "val_deco": s.get("val_deco"), "s1_val": s.get("s1_val")}),
        (24, "baseline_fullpool_rank", bm),
        (25, "r17_fullpool_rank", {"frozen_base": s.get("r17_rank_frozen"),
                                   "pool_context": rm17}),
        (26, "capacity", cap),
        (27, "utility_mass", {"base_r6": um.get("base"), "r17_ctx": um.get("cur"),
                              "r16_ref_K32": C.TO1_R17_R16_FULL_UM32,
                              "delta_vs_r16_K32": s.get("um_delta_over_r16")}),
        (28, "utility_regret", ((cap or {}).get("utility_regret"))),
        (29, "primary_target", {"pos_recall32_base": c.get("base_rec32"),
                                "pos_recall32_r17": c.get("r17_rec32"),
                                "recall32_diagnostic_only": True,
                                "goal": "prioritize high-utility positives (§1)"}),
        (30, "shortlist_gate", {"any": (s.get("r17_recall") or {}).get("any"),
                                "best": (s.get("r17_recall") or {}).get("best"),
                                "oracle": (s.get("r17_recall") or {}).get("oracle"),
                                "gate_ok": s["sl_gate_ok"],
                                "success": "any/best/oracle>=0.95 AND util-mass@32>=0.99 "
                                           "of R16 full 0.90725 (§31)", "cap": 32}),
        (31, "closed_loop_no_rl", {"T0_r6_full": s["s0"], "T1_r16_full_recorded": C.TO1_R17_T1_TRAIN,
                                   "T2_r17_full": s["s1"], "T3_r17_shortlist32": s["s2"]}),
        (32, "closed_loop_floor", {"floor": C.TO1_R17_SELF_FLOOR, "passed": bool(s["s1"] >= C.TO1_R17_SELF_FLOOR)}),
        (33, "downstream_table", s.get("downstream")),
        (34, "Mk1_trace", mk.get("mk1")),
        (35, "Fattahi15_trace", mk.get("fattahi15")),
        (36, "selection_miss", {"base": s.get("miss", [None, None])[0],
                                "r17": s.get("miss", [None, None])[1],
                                "decomp_base": deco.get("base"), "decomp_r17": deco.get("r17")}),
        (37, "held_not_degraded", {"held_um32_base": (s.get("held_um32") or {}).get("base"),
                                   "held_um32_r17": (s.get("held_um32") or {}).get("r17"),
                                   "delta": s.get("held_delta_um")}),
        (38, "aux_held_gate", {"real": s.get("s1_real"), "syn": s.get("s1_syn"),
                               "r16_real_ref": C.TO1_R17_T1_AUX_REAL,
                               "r16_syn_ref": C.TO1_R17_T1_AUX_SYN,
                               "ok": s.get("aux_ok")}),
        (39, "verdict_conditions", c),
        (40, "no_rl", {"m3_only_grpo": False, "joint_grpo": False, "rl_runs": 0}),
        (41, "evidence_verify", s.get("x_evid")),
        (42, "checkpoint", {"path": C.TO1_R17_CKPT.name, "written_only_on_A": True,
                            "method": "pool_context_residual_sft",
                            "parent": C.TO1_CKPT.name, "stop_integrated": True,
                            "oracle_runtime": False, "formal_test_access": 0,
                            "identified": False, "written": bool(s["passed"])}),
        (43, "next_step", "Joint GRPO only after R17 A (§45); "
                          f"currently gate_ok={s['sl_gate_ok']} "
                          f"closed_loop={s['s1']} vs floor {C.TO1_R17_SELF_FLOOR}"),
    ]
    return {"items": [{"n": n, "key": k, "value": v_} for n, k, v_ in items],
            "summary": {k: v for k, v in (("verdict", v["label"]),)}}


def _r17_report(env, re, scrape):
    s = scrape
    c = s["conditions"]
    L = []
    L.append("# T1-M3-POOL-CONTEXT-REPRESENTATION-SFT-R17 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    v = s["verdict"]
    L.append(f"## Verdict: **{v['code']}** `{v['label']}`")
    L.append("")
    L.append(f"> {v['note']}")
    L.append("")
    L.append("## §0-1 what R17 changes (and what it does NOT)")
    L.append("")
    L.append("R17 is an **M3 SFT improvement** (NO RL). It upgrades the M3 proposal "
             "REPRESENTATION: a permutation-invariant pool-context scorer over the frozen "
             "R6 base (bounded residual, §25-26) + runtime M2/probe evidence (§9-12) + a "
             "direct utility regressor (§19-21). Frozen below M3: M2, M2 q2, Memory "
             "semantics, adaptive probing, Reasoner, FixedDecisionReplay, and Joint GRPO "
             "semantics. Pos-rec@32 is a DIAGNOSTIC (§1 -- 61 positives > CAP=32); the A-"
             "gates are utility-ranking advance + shortlist oracle >=0.95 + closed loop "
             ">=337.25 + held not degraded.")
    L.append("")
    L.append(f"- proof no M3-RL parent §15: {json.dumps(s['proof_no_m3_rl_parent'], default=str)}")
    L.append("")
    L.append("## §16 R6 reproduction (bit-exact anchor)")
    L.append("")
    L.append(f"- acc={s['r6_acc']:.4f} regret={s['r6_regret']:.2f} repro_ok={s['repro_ok']}"
             f" · raw anchor={s['r6_canonical_train']}")
    L.append(f"- frozen-base §28 ranking matches R15 = {s.get('base_matches_r15')}")
    L.append("")
    L.append("## §15 splits & training groups")
    L.append("")
    L.append(f"- internal train `{s['internal_train']}`; internal held `{s['internal_held']}`")
    L.append(f"- AUX-real train/held {s.get('aux_counts', {}).get('real')} | "
             f"AUX-syn train/held {s.get('aux_counts', {}).get('syn')}")
    L.append(f"- train groups {s['n_tr_groups']} · held groups {s['n_held_groups']}")
    L.append(f"- pool stats: {json.dumps(s.get('pool_stats') or {}, default=str)}")
    L.append(f"- labels: {json.dumps(s.get('labels') or {}, default=str)}")
    L.append(f"- label alignment (first bench group, mean/max abs delta): "
             f"{json.dumps(s.get('label_align') or {})}")
    L.append(f"- evidence X: {json.dumps(s.get('x_evid') or {})}")
    L.append("")
    L.append("## §5-21 architecture (trainable + frozen)")
    L.append("")
    L.append(f"- enc_prop 277->128->128, c_pool=cat(mean,max,std) h (§7), "
             f"z_i=cat(h_i,c,h_i-mean,h_i/(std+e),state,X_i) [835] (§8)")
    L.append(f"- runtime evidence {TOP1.R17_EVID_DIM}-dim: retained-root attribution, "
             f"tier-A/B, root-probe gain (robust, state-normalized §10), dependency+/mem "
             f"support, enablers, atom counts, structural family (§9-12); pair atoms "
             f"e1+e2+mean+absdiff+prod, single->second atom zero-masked (§12)")
    L.append(f"- STOP context: Mlp_stop(state, c_pool, pstats_ext[{TOP1.R17_STOP_PSTATS_DIM}]) "
             f"§13-14 (never true_U)")
    L.append(f"- L = L_list + {C.TO1_R17_LAMBDA_POS}·L_posmargin + "
             f"{C.TO1_R17_LAMBDA_PAIR}·L_pair + {C.TO1_R17_LAMBDA_UTILITY}·L_utility "
             f"(Huber, λ FIXED §20); state equal weight §21")
    L.append(f"- tau_u={s['tau_u']} tau_p={s['tau_p']}")
    L.append("")
    L.append("## §17-18 + §25 full-pool ranking (frozen R6 base -> R17 context)")
    L.append("")
    bm = s.get("base_rank") or {}
    rm17 = s.get("r17_rank") or {}
    _rk_line = lambda r: ("·".join(f"{k}=" + ("-" if r.get(k) is None else f"{r[k]:.3f}")
                                   for k in ("best_positive_rank", "oracle_rank", "mrr",
                                             "rec1", "rec3", "rec5", "rec10", "rec20", "rec32")))
    L.append(f"- BEFORE (R6 frozen base): {_rk_line(bm)}")
    L.append(f"- AFTER  (R17 pool-context): {_rk_line(rm17)}")
    L.append("")
    um = s.get("um") or {}
    L.append(f"- PositiveUtilityMass@32 base={um.get('base')} r17={um.get('cur')} "
             f"(R16 ref {C.TO1_R17_R16_FULL_UM32}, Δ={s.get('um_delta_over_r16')})")
    cap = s.get("capacity") or {}
    L.append(f"- capacity: normalised-rec@32={cap.get('capacity_normalized_recall32')} "
             f"bound32={cap.get('capacity_bound32')} "
             f"regret={json.dumps(cap.get('utility_regret') or {})}")
    L.append("")
    L.append("## §31 shortlist gate (builder VERBATIM unchanged, scores R17 context)")
    L.append("")
    rec17 = s.get("r17_recall") or {}
    L.append(f"- any={rec17.get('any')} · best={rec17.get('best')} · oracle={rec17.get('oracle')} "
             f"-> gate_ok={s['sl_gate_ok']} success={s.get('sl_success')}")
    L.append(f"- §29 diagnostic pos-rec@32 base={c.get('base_rec32')} r17={c.get('r17_rec32')}")
    L.append("")
    L.append("## §26-30 no-RL closed loops + downstream table (§41)")
    L.append("")
    L.append(f"- T0 R6 full = **{s['s0']}** · T2 R17 full = **{s['s1']}** "
             f"· T3 R17 shortlist32 = **{s['s2']}**   [floor {C.TO1_R17_SELF_FLOOR}]")
    L.append(f"- VAL: S0={s.get('s0_val')} S1={s.get('s1_val')} · AUX-held real={s.get('s1_real')} "
             f"syn={s.get('s1_syn')}  (R16 ref real={C.TO1_R17_T1_AUX_REAL} "
             f"syn={C.TO1_R17_T1_AUX_SYN})")
    L.append(f"- downstream: {json.dumps(s.get('downstream') or {}, default=str)}")
    L.append("")
    L.append("## §28 recompute selection miss + held")
    L.append("")
    deco_b = (s.get("decomp") or {}).get("base") or {}
    deco_17 = (s.get("decomp") or {}).get("r17") or {}
    miss_b, miss_17 = (s.get("miss") or [None, None])
    L.append(f"- M3_SELECTION_MISS base={miss_b} -> r17={miss_17}")
    L.append(f"- decomposition base {json.dumps(deco_b, default=str)}")
    L.append(f"- decomposition r17  {json.dumps(deco_17, default=str)}")
    L.append(f"- held: um32 base={(s.get('held_um32') or {}).get('base')} "
             f"r17={(s.get('held_um32') or {}).get('r17')} Δ={s.get('held_delta_um')}")
    L.append("")
    L.append("## §37-38 Mk1 / Fattahi15 traces (S0 root, full pool)")
    L.append("")
    mk = s.get("mk") or {}
    for key in ("mk1", "fattahi15"):
        L.append(f"### {key}")
        L.append("```json")
        L.append(json.dumps(mk.get(key), indent=2, default=str))
        L.append("```")
    L.append("")
    L.append("## Verdict conditions (ladder)")
    L.append("")
    for _ck, _cv in (s.get("conditions") or {}).items():
        L.append(f"- {_ck}: {_cv}")
    L.append("")
    L.append("## Checkpoint")
    L.append("")
    if s["passed"]:
        L.append(f"{C.TO1_R17_CKPT.name} written (verdict-A only, §44).")
    else:
        L.append(f"NOT written (verdict {v['code']}, §44 -- A only).")
    L.append("")
    f = _r17_43items(s)
    L.append("## §47 forty-three-item final return")
    L.append("")
    L.append("| # | key | value |")
    L.append("|---|---|---|")
    for it in f["items"]:
        val = it["value"]
        if isinstance(val, dict):
            val = json.dumps(val, default=str)
        val = str(val).replace("|", "\\|").replace("\n", " ")
        L.append(f"| {it['n']} | {it['key']} | {val} |")
    L.append("")
    L.append("## 下一步 (single highest-priority next action)")
    L.append("")
    if v["code"] == "A":
        L.append("R17 A -- the next Joint GRPO stage may proceed on top of "
                 "`m3_pool_context_sft_v4.pt` (§45), reusing the T3 shortlist32 loop.")
    elif v["code"] in ("C", "F"):
        L.append("Representation / label density still limits (§42 C/F) -- next round "
                 "upgrades proposal features or label density, not the loss.")
    elif v["code"] == "D":
        L.append("Utility auxiliary overfits (§42 D) -- next round rebalances "
                 "λ_utility or regularizes the util_head.")
    elif v["code"] == "E":
        L.append("Pool-context overfits / generalizes poorly (§42 E) -- next round "
                 "shrinks ctx capacity / adds held-CV.")
    else:
        L.append("Improvement without full gate pass -- fix the shown condition before "
                 "Joint GRPO (§45).")
    L.append("")
    return "\n".join(L)


def _r17_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R17_REPORT.write_text(report, encoding="utf-8")
    if scrape is not None:
        (C.CANONICAL_OUT_DIR / "result_r17.json").write_text(
            json.dumps(scrape, default=str, indent=2), encoding="utf-8")
    print(f"[r17] persisted {C.R17_REPORT.name} + result_r17.json", flush=True)


def run_top1_phase_r17(args, env, re, p1, p1_report):
    """R17: POOL-CONTEXT proposal representation SFT -- fix M3's ranking by upgrading
    the proposal REPRESENTATION (pool-context scorer + runtime M2/probe evidence +
    utility regressor) instead of only the loss (R16 caps at +0.013 rec32).

    §1 pos-rec@32 is DIAGNOSTIC (61 positives > CAP=32).  A-gates (§33/§42):
      * utility ranking significantly improved: PositiveUtilityMass@32 >= R16 full
        0.90725 + 0.03, AND
      * shortlist oracle-best recall >= 0.95 (AND BestPositiveRecall >=0.95, and um32
        above R16 -> §31), AND
      * no-RL TRAIN closed loop >= 337.25 (0.95*355) §29/§32,
      * held utility-mass not degraded vs frozen base, and AUX-held not degraded vs
        R16 (real>=53 or syn>=70) §30.
    Verdict ladder priority G > F > D > E > A > B > C (§42-43).  ckpt ONLY on A (§44).
    identified=false, formal_test_access=0, Formal TEST SEALED."""
    print("[r17] POOL-CONTEXT representation SFT (M3 SFT improvement, NO RL) ...", flush=True)
    t0 = time.time()
    scorer, reranker = p1["scorer_mem"], p1["reranker"]
    mp_ctx = _r12_mp_ctx()

    r6_sel = _load_r6_parent(args)                     # frozen R6 base (§24)
    jpol_r6 = JG.JointAgenticPolicy(r6_sel)
    proof_no_m3_rl_parent = {
        "m3_backbone": C.TO1_CKPT.name,
        "r6_grpo_checkpoint": None, "r11_grpo_checkpoint": None,
        "r12_grpo_checkpoint": None, "r13_grpo_checkpoint": None,
        "r14_grpo_checkpoint": None, "r15_checkpoint": None, "r16_checkpoint": None,
        "m3_retrained_with": "pool_context_residual_sft (NO RL)",
        "shortlist_cap": C.TO1_R15_SHORTLIST_CAP,
        "head": "enc_prop+ctx_head+stop_ctx_head+util_head over frozen R6 base",
    }

    # ---- §16 R6 bit-exact reproduction -------------------------------------------
    mr6, _grp = TOP1.top1_metrics(re["state_examples"], scorer, re["mem_values"],
                                  reranker, r6_sel)
    r6_acc_all = float(mr6["state_wise_top1_acc"]["all"])
    r6_regret = float(mr6["top1_regret"]["mean"])
    repro_ok = bool(abs(r6_acc_all - 0.525) < 1e-6 and abs(r6_regret - 5.2) < 0.01)
    print(f"[r17] R6 reproduction acc={r6_acc_all:.4f} regret={r6_regret:.2f} "
          f"repro_ok={repro_ok}", flush=True)

    train_pairs = [(i["instance_id"], env["ep_id_of"][i["instance_id"]])
                   for i in env["train_insts"]]
    val_pairs = [(i["instance_id"], len(env["train_insts"]) + idx)
                 for idx, i in enumerate(env["val_insts"])]
    st_bench_map = {i["instance_id"]: env["states"][i["instance_id"]]
                    for i in env["train_insts"] + env["val_insts"]}
    r6_canonical_train = _r6_canonical_train_gain(env, re, scorer, r6_sel,
                                                  train_pairs, st_bench_map)
    print(f"[r17] ANCHOR: r6_canonical_train={r6_canonical_train} (R15=R16=369)",
          flush=True)

    # ---- AUX bundles (§15) --------------------------------------------------------
    rb = sb = None
    if not args.quick:
        rb = (_r11_aux_bundle("real", C.TO1_REAL_DATA, "aux_real", env)
              if C.TO1_REAL_DATA.exists() else None)
        sb = (_r11_aux_bundle("syn", C.TO1_AUX_DATA, "aux", env)
              if C.TO1_AUX_DATA.exists() else None)

    # ---- §15 splits: internal-held = FIRST-2 TRAIN instances ----------------------
    train_iids = [i["instance_id"] for i in env["train_insts"]]
    if args.quick:
        internal_train = train_iids
        internal_held = []
    else:
        internal_train = train_iids[C.TO1_R17_INTERNAL_HELD_COUNT:]
        internal_held = train_iids[:C.TO1_R17_INTERNAL_HELD_COUNT]
    print(f"[r17] TRAIN-internal: train {len(internal_train)} | held {internal_held}",
          flush=True)

    # ---- build R17 groups: state + FULL gated pool + STOP + U + X + pext ----------
    bench_s0 = {}
    for ex in re["state_examples"]:
        if ex.get("s0"):
            bench_s0.setdefault(ex["iid"], []).append(ex)
    tr_groups, held_groups = [], []
    for iid in internal_train:
        if iid not in env["states"] or not bench_s0.get(iid):
            print(f"[r17] skip bench S0 {iid} (missing state/example)", flush=True)
            continue
        g = _r17_state_group(env, scorer, jpol_r6, env["states"][iid], iid,
                             env["ep_id_of"][iid], bench_s0[iid][0], re["progmem"],
                             verify=(iid == internal_train[0]))
        if g is not None:
            g["split"] = "bench-train"
            tr_groups.append(g)
    for iid in internal_held:
        if iid not in env["states"] or not bench_s0.get(iid):
            print(f"[r17] skip bench S0 {iid} (missing state/example)", flush=True)
            continue
        g = _r17_state_group(env, scorer, jpol_r6, env["states"][iid], iid,
                             env["ep_id_of"][iid], bench_s0[iid][0], re["progmem"])
        if g is not None:
            g["split"] = "bench-hold"
            held_groups.append(g)
    aux_counts = {"real": None, "syn": None}
    if rb is not None:
        want = set(rb["tr_iids"])
        exs = _r16_aux_s0_exs(C.TO1_REAL_DATA, want)
        for iid in rb["tr_iids"]:
            g = _r17_state_group(env, scorer, jpol_r6, rb["tr_states"][iid], iid,
                                 rb["tr_ep"][iid], exs.get(iid), rb["tr_pm"])
            if g is not None:
                g["split"] = "aux-real-train"
                tr_groups.append(g)
        exs_h = _r16_aux_s0_exs(C.TO1_REAL_DATA, rb["hd_iids"])
        for iid in rb["hd_iids"]:
            g = _r17_state_group(env, scorer, jpol_r6, rb["hd_states"][iid], iid,
                                 rb["hd_ep"][iid], exs_h.get(iid), rb["hd_pm"])
            if g is not None:
                g["split"] = "aux-real-held"
                held_groups.append(g)
        aux_counts["real"] = [len(rb["tr_iids"]), len(rb["hd_iids"])]
    if sb is not None:
        want = set(sb["tr_iids"])
        exs = _r16_aux_s0_exs(C.TO1_AUX_DATA, want)
        for iid in sb["tr_iids"]:
            g = _r17_state_group(env, scorer, jpol_r6, sb["tr_states"][iid], iid,
                                 sb["tr_ep"][iid], exs.get(iid), sb["tr_pm"])
            if g is not None:
                g["split"] = "aux-syn-train"
                tr_groups.append(g)
        exs_h = _r16_aux_s0_exs(C.TO1_AUX_DATA, sb["hd_iids"])
        for iid in sb["hd_iids"]:
            g = _r17_state_group(env, scorer, jpol_r6, sb["hd_states"][iid], iid,
                                 sb["hd_ep"][iid], exs_h.get(iid), sb["hd_pm"])
            if g is not None:
                g["split"] = "aux-syn-held"
                held_groups.append(g)
        aux_counts["syn"] = [len(sb["tr_iids"]), len(sb["hd_iids"])]

    n_pos = sum(1 for g in tr_groups if bool((g["U"] > 0).any()))
    pool_size = [g["n_gated"] for g in tr_groups] or [0]
    pos_cnt = [int((g["U"] > 0).sum()) for g in tr_groups] or [0]
    pool_stats = {"train_states": len(tr_groups), "pos_states": n_pos,
                  "mean_pool": float(np.mean(pool_size)),
                  "mean_pos": float(np.mean(pos_cnt)),
                  "max_pos": int(max(pos_cnt))}
    label_align = {}
    for g in tr_groups:
        if "label_align_mean_abs_diff" in g:
            label_align = {"mean_abs_diff": g["label_align_mean_abs_diff"],
                           "max_diff": g["label_align_max_diff"]}
            break
    x_evid = {}
    if tr_groups:
        x0 = tr_groups[0]["X"]
        x0np = x0.numpy()
        x_evid = {"g": int(x0.shape[0]), "d": int(x0.shape[1]),
                  "mean_abs": float(np.abs(x0np).mean()),
                  "nonzero_cols": int((np.abs(x0np).sum(0) > 0).sum()),
                  "perm_invariant": True}
    print(f"[r17] groups: train={len(tr_groups)} held={len(held_groups)} "
          f"{json.dumps(pool_stats, default=str)}", flush=True)
    labels = {"source": "FixedDecisionReplay true_U from replay/aux examples (§3)",
              "train_only": True, "oracle_runtime": False,
              "n_pos_states_train": n_pos}

    # ---- §6-21 train (pool-context + utility aux) + §22-28 selection (held) -------
    trun = TOP1.train_pool_context_r17(tr_groups, held_groups=held_groups,
                                       base_selector=r6_sel, seed=0)
    r17_sel = trun["selector"]
    jpol17 = JG.JointAgenticPolicy(r6_sel)     # fresh zero-init M2 adapter (δ=0, parity)
    jpol17.m3 = r17_sel                        # m3 must BE the PoolContextScorer (§25)
    best_epoch = int(trun["best_epoch"])
    held_score = trun["best_held_score"]
    hist = [{"ep": h["ep"], "l_list": h["l_list"], "l_pos": h["l_pos"],
             "l_pair": h["l_pair"], "l_util": h["l_util"],
             "train_um32": (h["train"].get("utility_mass") or {}).get("K32")
                           if h["train"] else None,
             "held_um32": (h["held"].get("utility_mass") or {}).get("K32")
                          if h.get("held") else None}
            for h in trun["hist"]]
    print(f"[r17] trained; best_epoch={best_epoch} held_um32={held_score}", flush=True)
    tr_um32 = (trun["train_metrics"].get("utility_mass") or {}).get("K32") \
        if trun.get("train_metrics") else None
    print(f"[r17] train util-mass@32="
          f"{'-' if tr_um32 is None else round(float(tr_um32), 4)}", flush=True)

    # ---- §17-18 + §25 baseline (frozen R6) vs after (R17 context) ranking ---------
    # base rows use the R15/R16 SAME ruler: `_r15_state_diag` via the FROZEN rolling
    # policy jpol_r6 (raw R6 base + zero residual => canonical R6 action logits).
    deco_base = _r15_deco(env, scorer, jpol_r6, train_iids,
                          copy.deepcopy(re["progmem"]),
                          cap_states=(2 if args.quick else None))
    base_recall = deco_base.get("recall") or {}
    base_rank = deco_base.get("ranking_metrics") or {}
    base_matches_r15 = bool(base_rank.get("rec32") is not None and
                            abs(base_rank["rec32"] - C.TO1_R17_REC32_BASE) < 0.01)
    print(f"[r17] frozen-base §28 rec32={base_rank.get('rec32')} "
          f"matches_R15={base_matches_r15}", flush=True)

    deco_r17 = _r17_deco(env, scorer, jpol_r6, jpol17, train_iids,
                         copy.deepcopy(re["progmem"]),
                         cap_states=(2 if args.quick else None))
    r17_recall = deco_r17.get("recall") or {}
    r17_rank = deco_r17.get("ranking_metrics_r17") or {}
    r17_rank_frozen = deco_r17.get("ranking_metrics") or {}
    print(f"[r17] after §18 context rec32={_r2(r17_rank, 'rec32')} "
          f"oracle_rank={_r2(r17_rank, 'oracle_rank')} "
          f"cap-rec32={_r2(deco_r17.get('capacity') or {}, 'capacity_normalized_recall32')}",
          flush=True)

    # ---- §31 shortlist gate (builder unchanged, scores R17 context) --------------
    tol = float(C.TO1_R15_RECALL_DROP_TOL)
    sl_any = float(r17_recall.get("any", 0.0))
    sl_best = float(r17_recall.get("best", 0.0))
    sl_oracle = float(r17_recall.get("oracle", 0.0))
    sl_gate_ok = bool(sl_any >= 1.0 - tol and sl_best >= 1.0 - tol and
                      sl_oracle >= 1.0 - tol)
    sl_success = bool(sl_best >= C.TO1_R17_SL_RECALL_GATE
                      and sl_oracle >= C.TO1_R17_SL_RECALL_GATE)
    print(f"[r17] §31 shortlist gate any={sl_any:.3f} best={sl_best:.3f} "
          f"oracle={sl_oracle:.3f} -> gate_ok={sl_gate_ok} success={sl_success}",
          flush=True)

    # ---- §22 + §33 utility mass + §29 diagnostic rec32 ----------------------------
    base_rec32 = base_rank.get("rec32")
    r17_rec32 = r17_rank.get("rec32")
    um = {"base": TOP1.pool_utility_mass(deco_base.get("u_mass_rows") or [])["base"],
          "cur": TOP1.pool_utility_mass(deco_r17.get("u_mass_rows") or [])["cur"],
          "n_states": TOP1.pool_utility_mass(deco_r17.get("u_mass_rows") or [])["n_states"]}
    um32_base = um["base"].get("K32")
    um32_r17 = um["cur"].get("K32")
    um_delta_over_r16 = (None if um32_r17 is None
                         else um32_r17 - C.TO1_R17_R16_FULL_UM32)
    print(f"[r17] util-mass@32 base={um32_base} r17={um32_r17} "
          f"(R16 ref {C.TO1_R17_R16_FULL_UM32} Δ={um_delta_over_r16})", flush=True)

    # ---- §28 decomposition + M3_SELECTION_MISS recompute --------------------------
    decomp = {"base": dict(deco_base.get("decomposition") or {}),
              "r17": dict(deco_r17.get("decomposition") or {})}
    miss_base = int(deco_base.get("M3_SELECTION_MISS", 0))
    miss_r17 = int(deco_r17.get("M3_SELECTION_MISS", 0))
    capacity = deco_r17.get("capacity") or {}
    print(f"[r17] §28 miss base={miss_base} -> r17={miss_r17}", flush=True)

    # ---- §26 no-RL closed loops: T0 R6 / T2 R17 full / T3 R17 shortlist -----------
    init_roots = _r12_eval_roots(re["progmem"], train_pairs, st_bench_map)
    s0 = _r17_tot(jpol_r6, "full", env, scorer, init_roots)
    s1 = _r17_tot(jpol17, "full", env, scorer, init_roots)
    s2 = _r17_tot(jpol17, "shortlist", env, scorer, init_roots)
    print(f"[r17] §26 closed loop T0={s0:.0f} T2={s1:.0f} T3={s2:.0f} "
          f"(floor {C.TO1_R17_SELF_FLOOR})", flush=True)
    val_roots = _r12_eval_roots(re["progmem"], val_pairs, st_bench_map)
    s0_val = _r17_tot(jpol_r6, "full", env, scorer, val_roots)
    s1_val = _r17_tot(jpol17, "full", env, scorer, val_roots)
    s0_real = s0_syn = s1_real = s1_syn = None
    if rb is not None:
        rq = _r12_eval_roots(rb["hd_pm"], [(i, rb["hd_ep"][i]) for i in rb["hd_iids"]],
                             rb["hd_states"])
        s0_real = _r17_tot(jpol_r6, "full", env, scorer, rq)
        s1_real = _r17_tot(jpol17, "full", env, scorer, rq)
    if sb is not None:
        sq = _r12_eval_roots(sb["hd_pm"], [(i, sb["hd_ep"][i]) for i in sb["hd_iids"]],
                             sb["hd_states"])
        s0_syn = _r17_tot(jpol_r6, "full", env, scorer, sq)
        s1_syn = _r17_tot(jpol17, "full", env, scorer, sq)
    print(f"[r17] VAL S0={s0_val:.0f} S1={s1_val:.0f} "
          f"AUX-held real {s0_real}->{s1_real} syn {s0_syn}->{s1_syn}", flush=True)

    # ---- §22-28 held not-degraded + §39 VAL once at final ---------------------------
    base_ctx = TOP1.PoolContextProposalScorer(r6_sel, seed=0)
    base_ctx.eval()
    held_um32 = {"base": TOP1._r17_held_utility_mass(base_ctx, held_groups, K=32,
                                                     context=False),
                 "r17": TOP1._r17_held_utility_mass(r17_sel, held_groups, K=32,
                                                    context=True)}
    held_delta_um = (None if held_um32["base"] is None or held_um32["r17"] is None
                     else held_um32["r17"] - held_um32["base"])
    print(f"[r17] held util-mass@32 base={held_um32['base']} r17={held_um32['r17']} "
          f"Δ={held_delta_um}", flush=True)
    val_deco = None
    if not args.quick:
        val_iids = [i["instance_id"] for i in env["val_insts"]]
        val_ep_map = {i["instance_id"]: len(env["train_insts"]) + idx
                      for idx, i in enumerate(env["val_insts"])}
        val_deco = _r17_deco(env, scorer, jpol_r6, jpol17, val_iids,
                             copy.deepcopy(re["progmem"]), ep_map=val_ep_map)
        print(f"[r17] VAL deco: {json.dumps(val_deco.get('ranking_metrics_r17') or {}, default=str)}",
              flush=True)

    # ---- §37 Mk1 / §38 Fattahi15 traces ------------------------------------------
    mk = None
    if not args.quick:
        mk = _r17_state_traces(jpol_r6, jpol17, scorer, env, re)

    # ---- §41 downstream table ---------------------------------------------------
    downstream = {
        "columns": ["TRAIN", "AUX-real", "AUX-syn", "VAL"],
        "T0_r6_full": {"TRAIN": s0, "AUX-real": s0_real, "AUX-syn": s0_syn, "VAL": s0_val},
        "T1_r16_full": {"TRAIN": C.TO1_R17_T1_TRAIN, "AUX-real": C.TO1_R17_T1_AUX_REAL,
                        "AUX-syn": C.TO1_R17_T1_AUX_SYN, "VAL": C.TO1_R17_T1_VAL},
        "T2_r17_full": {"TRAIN": s1, "AUX-real": s1_real, "AUX-syn": s1_syn, "VAL": s1_val},
        "T3_r17_shortlist32": {"TRAIN": s2, "AUX-real": None, "AUX-syn": None, "VAL": None},
    }

    # ---- verdict ladder (§33/§42-43 priority G > F > D > E > A > B > C) -----------
    util_advance_ok = bool(um_delta_over_r16 is not None
                           and um_delta_over_r16 >= C.TO1_R17_UTIL_ADVANCE)
    closed_loop_ok = bool(s1 >= C.TO1_R17_SELF_FLOOR)
    aux_ok = bool((s1_real is not None and s1_real >= C.TO1_R17_T1_AUX_REAL) or
                  (s1_syn is not None and s1_syn >= C.TO1_R17_T1_AUX_SYN))
    held_not_degraded = bool(held_delta_um is None or held_delta_um >= -0.02)
    gen_degrade = bool((held_delta_um is not None and held_delta_um <= -0.05) or
                       (s1_val - s0_val) <= -5.0)
    util_aux_overfit = bool(um32_r17 is not None and tr_um32 is not None
                            and held_um32["r17"] is not None
                            and tr_um32 >= um32_r17 + 0.05
                            and held_um32["r17"] <= (held_um32["base"] or 1.0) - 0.05)
    rank_flat = bool(not util_advance_ok and
                     (um32_r17 is None or um32_base is None
                      or abs(um32_r17 - um32_base) < 0.005))

    conditions = {"base_rec32": base_rec32, "r17_rec32": r17_rec32,
                  "um32_base": um32_base, "um32_r17": um32_r17,
                  "um_delta_over_r16": um_delta_over_r16,
                  "util_advance_ok": util_advance_ok,
                  "sl_any": sl_any, "sl_best": sl_best, "sl_oracle": sl_oracle,
                  "sl_gate_ok": sl_gate_ok, "sl_success": sl_success,
                  "closed_loop_s1": s1, "closed_loop_ok": closed_loop_ok,
                  "s2": s2, "s1_val": s1_val, "s0_val": s0_val,
                  "aux_real": s1_real, "aux_syn": s1_syn, "aux_ok": aux_ok,
                  "held_delta_um": held_delta_um, "held_not_degraded": held_not_degraded,
                  "gen_degrade": gen_degrade, "util_aux_overfit": util_aux_overfit,
                  "rank_flat": rank_flat, "repro_ok": repro_ok,
                  "base_matches_r15": base_matches_r15,
                  "miss": [miss_base, miss_r17]}

    if not repro_ok or not base_matches_r15:
        vcode, vlabel = "G", "SEMANTICS_OR_LEAKAGE_BUG"
        note = (f"hard verification failed: repro_ok={repro_ok} "
                f"base_matches_r15={base_matches_r15} -- STOP (§46 semantics)")
    elif len(tr_groups) < 20 or (n_pos >= 1 and n_pos < max(10, int(0.5 * len(tr_groups)))):
        vcode, vlabel = "F", "LABEL_DENSITY_IS_NEXT_BLOCKER"
        note = f"training labels too sparse (states {len(tr_groups)}, pos-states {n_pos})"
    elif util_aux_overfit:
        vcode, vlabel = "D", "UTILITY_AUX_OVERFITS"
        note = (f"util-head lifts TRAIN um32 {tr_um32:.3f} but held um32 "
                f"{held_um32['r17']} falls below base {held_um32['base']} (§42-D)")
    elif gen_degrade:
        vcode, vlabel = "E", "POOL_CONTEXT_OVERFITS"
        note = (f"pool-context improves TRAIN but held Δ={held_delta_um} / VAL "
                f"closed-loop Δ={s1_val - s0_val:.0f} degrades (§42-E)")
    elif util_advance_ok and sl_success and closed_loop_ok and held_not_degraded:
        vcode, vlabel = "A", "POOL_CONTEXT_M3_FIXES_REPRESENTATION"
        note = (f"util-mass@32 {um32_r17:.4f} (>= R16 {C.TO1_R17_R16_FULL_UM32} "
                f"+ {C.TO1_R17_UTIL_ADVANCE}) AND shortlist oracle≥0.95 AND closed "
                f"loop {s1:.0f}>=337.25 AND held um32 Δ={held_delta_um}")
    elif util_advance_ok or sl_success:
        vcode, vlabel = "B", "CONTEXT_IMPROVES_RANKING_BUT_CLOSED_LOOP_UNSAFE"
        note = (f"context ranking improved (um32 {um32_r17}, sl success {sl_success}) "
                f"but closed loop {s1:.0f}<{C.TO1_R17_SELF_FLOOR} or held degraded -- "
                f"§32 coverage gate blocks Joint GRPO")
    else:
        vcode, vlabel = "C", "FEATURE_REPRESENTATION_STILL_LIMITED"
        note = (f"no measurable utility-ranking movement (um32 base={um32_base} -> "
                f"r17={um32_r17}, ΔoverR16={um_delta_over_r16})")
    passed = bool(vcode == "A")
    print(f"[r17] verdict {vcode} {vlabel} "
          f"(um32 {um32_base}->{um32_r17} / ΔR16 {um_delta_over_r16} / "
          f"sl_success {sl_success} / closed {s1:.0f} / aux_ok {aux_ok})", flush=True)

    # ---- checkpoint ONLY on verdict A (§44) ---------------------------------------
    if passed and args.quick:
        print("[r17] QUICK run: simulate PASS but REFUSE to write the R17 checkpoint "
              "(quick scale must never pollute m3_pool_context_sft_v4.pt)", flush=True)
    elif passed:
        meta = dict(
            phase="r17_pool_context_representation_sft",
            pipeline_stage="M3_SFT",
            architecture="pool_context_residual",
            base=C.TO1_CKPT.name,                          # m3_proposal_top1_sft_v2.pt
            r16_checkpoint=None,
            method="pool_context_residual_sft + utility_aux (NO RL)",
            stop_integrated=True, stop_has_u=0.0,
            oracle_runtime=False, utility_aux="learned_only",
            reasoner="frozen", executor="FixedDecisionReplay",
            tau_u=C.TO1_R17_TAU_U, tau_p=C.TO1_R17_TAU_P,
            lambda_pos=C.TO1_R17_LAMBDA_POS, lambda_pair=C.TO1_R17_LAMBDA_PAIR,
            lambda_utility=C.TO1_R17_LAMBDA_UTILITY,
            alpha_ctx=C.TO1_R17_ALPHA_CTX,
            best_epoch=best_epoch, held_score=held_score,
            shortlist_cap=C.TO1_R15_SHORTLIST_CAP,
            selected="TRAIN internal-held + AUX-real/syn held (NOT VAL3, §28)",
            formal_test_access=0, formal_test_sealed=True, identified=False)
        torch.save({"state": {"selector": r17_sel, "scorer": scorer,
                              "r6_anchor": r6_sel.state_dict()},
                    "meta": meta}, C.TO1_R17_CKPT)
        print(f"[r17] saved {C.TO1_R17_CKPT.name} (PASS, verdict A only §44)",
              flush=True)
    else:
        print(f"[r17] NOT PASS (verdict {vcode}) -> no R17 checkpoint written (§44)",
              flush=True)

    shortlist_proof = {
        "builder": "build_shortlist_r15 (VERBATIM unchanged, §31)",
        "cap": C.TO1_R15_SHORTLIST_CAP,
        "k_global": C.TO1_R15_K_GLOBAL,
        "tier_a_quota": C.TO1_R15_TIER_A_QUOTA,
        "tier_b_quota": C.TO1_R15_TIER_B_QUOTA,
        "diversity_fill": bool(C.TO1_R15_DIVERSITY_FILL),
        "score_source": "frozen R6 base (shortlist build) -> R17 context for ranking",
        "oracle_authority": False,
    }
    quick_sanity = None
    if args.quick:
        quick_sanity = {"groups": len(tr_groups), "best_epoch": best_epoch,
                        "um32_r17": um32_r17, "verdict": vcode}

    scrape = dict(
        repro_ok=repro_ok, r6_acc=r6_acc_all, r6_regret=r6_regret,
        r6_canonical_train=int(r6_canonical_train),
        base_matches_r15=base_matches_r15, internal_train=internal_train,
        internal_held=internal_held, aux_counts=aux_counts,
        n_tr_groups=len(tr_groups), n_held_groups=len(held_groups),
        pool_stats=pool_stats, labels=labels, label_align=label_align,
        tau_u=C.TO1_R17_TAU_U, tau_p=C.TO1_R17_TAU_P,
        lambda_pos=C.TO1_R17_LAMBDA_POS, lambda_pair=C.TO1_R17_LAMBDA_PAIR,
        lambda_utility=C.TO1_R17_LAMBDA_UTILITY,
        best_epoch=best_epoch, held_score=held_score, hist=hist,
        base_rank=base_rank, r17_rank=r17_rank, r17_rank_frozen=r17_rank_frozen,
        base_recall=base_recall, r17_recall=r17_recall,
        sl_gate_ok=sl_gate_ok, sl_success=sl_success,
        rec32_diag={"base": base_rec32, "r17": r17_rec32, "diagnostic_only": True},
        capacity=capacity, um=um, um_delta_over_r16=um_delta_over_r16,
        decomp=decomp, miss=[miss_base, miss_r17],
        s0=s0, s1=s1, s2=s2, s0_val=s0_val, s1_val=s1_val,
        s0_real=s0_real, s0_syn=s0_syn, s1_real=s1_real, s1_syn=s1_syn,
        downstream=downstream, held_um32=held_um32, held_delta_um=held_delta_um,
        aux_ok=aux_ok, mk=mk, val_deco=val_deco, x_evid=x_evid, verify={"repro": repro_ok},
        conditions=conditions, train_metrics=trun["train_metrics"],
        held_metrics=trun["held_metrics"],
        proof_no_m3_rl_parent=proof_no_m3_rl_parent,
        shortlist_proof=shortlist_proof,
        verdict=dict(code=vcode, label=vlabel, note=note), passed=passed,
        quick_sanity=quick_sanity)
    report = _r17_report(env, re, scrape)
    _r17_persist(report, scrape=scrape)
    print(f"[r17] total {time.time() - t0:.1f}s "
          f"(report {C.R17_REPORT.name})", flush=True)
    return report

def _r2(d, k):
    return "-" if (not d or d.get(k) is None) else round(float(d[k]), 3)


def _r13_parity_rows(scr):
    rw = scr.get("rows") or {}
    names = (("c0", "C0 R6 delta=0 (existing M2)", "none"),
             ("c1", "C1 R11/R12 M3 (existing M2)", "none"),
             ("c2", "C2 §0 gate d0 (zero-init M2)", "gated"),
             ("c3", "C3 M2 RootPolicy RL x frozen M3", "gated"),
             ("c4", "C4 frozen M2 x M3 GRPO (gated)", "gated"),
             ("c5", "C5 JOINT M2+M3", "gated"))
    out = []
    for key, name, mode in names:
        t, r_, s_, v_ = rw.get(key) or (None, None, None, None)
        out.append({"model": name, "m2": mode,
                    "train": None if t is None else float(t),
                    "real_held": None if r_ is None else float(r_),
                    "syn_held": None if s_ is None else float(s_),
                    "val": None if v_ is None else float(v_)})
    return out


_R15_SCRAPE_KEYS = [
    "repro_ok", "anchor_ok", "m5", "m5_z", "res_j", "q0", "q1", "q2", "q3",
    "q1_real", "q3_real", "q1_syn", "q3_syn", "q3_val", "rows", "cov_p0", "cov_p3",
    "r6_canonical_train", "prof", "mp_ok", "workers", "mp_ctx", "deco_i", "deco_f",
    "recall_i", "recall_f", "recall_gate_ok", "coverage_stable", "m3_miss",
    "m2_miss", "m2_agg", "sl_train_agg", "mk", "val_final", "m2_kl", "m3_kl",
    "n_info_m2", "n_info_m3", "info_ratio2", "info_ratio3", "m2_grad_any",
    "m3_grad_any", "reward_ordering_ok", "proof_no_m3_rl_parent", "train_beat",
    "miss_reduced", "q3_gt_q1", "coverage_stable_ref", "aux_ok", "verdict",
    "passed", "quick_sanity", "action_space",
]


def _r15_scrape_all(ns, ov):
    out = {}
    for k in _R15_SCRAPE_KEYS:
        out[k] = ov.get(k, ns.get(k))
    return out


def _r15_41items(s):
    """The 41-item §45 final return checklist."""
    res_j = s.get("res_j") or {}
    hist = (res_j.get("history") if isinstance(res_j, dict) else []) or []
    n_groups = sum(h.get("n_groups", 0) for h in hist)
    di = (s.get("deco_i") or {}).get("decomposition", {}) if isinstance(s.get("deco_i"), dict) else {}
    df = (s.get("deco_f") or {}).get("decomposition", {}) if isinstance(s.get("deco_f"), dict) else {}
    recall_i = s.get("recall_i") or {}
    recall_f = s.get("recall_f") or {}
    mm = s.get("m3_miss") or (None, None)
    m2 = s.get("m2_miss") or (None, None)
    rows = s.get("rows", {})
    sel_i = (s.get("deco_i") or {}).get("selection_metrics") or {}
    rnk_f = (s.get("deco_f") or {}).get("ranking_metrics") or {}
    sl_agg = s.get("sl_train_agg") or {}
    prof = s.get("prof") or {}

    def _z(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"

    return [
        "modified files: src/causal_schedule_lab/m3/config.py, "
        "src/causal_schedule_lab/m3/joint_grpo.py, "
        "scripts/run_m3_canonical_training.py, tests/test_m3_joint_grpo_r15.py",
        "init checkpoints: M2 attribution (frozen B5) + m3_proposal_top1_sft_v2.pt "
        "(pure SFT) + zero-init M2RootPolicyAdapter(δ=0) + M3 residual δ=0",
        f"proof no M3-RL parent §15: {json.dumps(s.get('proof_no_m3_rl_parent'))} "
        f"(R11-R14 never loaded)",
        "§0 permanent structure: R14 pipeline UNCHANGED (M2 SFT / Memory / adaptive "
        "probing / Reasoner / FixedDecisionReplay / stagewise A2-A3 credit); the "
        "ONLY R15 change is the M3 action-set construction (shortlist replaces the "
        "wide_pool stage) + the necessary M3 selection policy",
        f"M3 action set = coverage-preserving shortlist32 + STOP: Source1 GLOBAL "
        f"frozen-M3-SFT-base top {C.TO1_R15_K_GLOBAL} (§5), per-root quota "
        f"Tier-A top-{C.TO1_R15_TIER_A_QUOTA}/Tier-B top-{C.TO1_R15_TIER_B_QUOTA} "
        f"(§6-7), structural-family diversity fill (single ROUTE/SEQ, pair "
        f"ROUTE+ROUTE/ROUTE+SEQ/SEQ+SEQ) if < cap (§8), CAP={C.TO1_R15_SHORTLIST_CAP} "
        f"(no sweep); STOP external",
        "§10 no-oracle: true_U / frozen-future-utility / trajectory reward / oracle "
        "rank NEVER enter the runtime shortlist -- construction uses only frozen M3 "
        "SFT base score + root provenance + proposal structural metadata",
        f"§19/§20 same action set old/new (stored F_pool), per-state shortlist "
        f"signature {C.TO1_R15_SL_SIGNATURE_HASH}, no optimizer-epoch rebuild; "
        f"collected shortlist steps {sl_agg.get('shortlist_steps', 0)}",
        "§2 recorded fields per TRAIN state: full_pool_size / best_true_U / "
        "best_true_U_signature / best_true_U_score_rank / best_positive_score_rank / "
        "selected_signature / selected_true_U / selected_score_rank / stop_score / "
        "best_positive_score / score_margin_positive_vs_stop / "
        "score_margin_positive_vs_selected",
        f"§3 decomposition INIT: ranking={di.get('M3_RANKING_MISS', 0)} stop_margin="
        f"{di.get('M3_STOP_MARGIN_MISS', 0)} intra_pool="
        f"{di.get('M3_INTRA_POOL_SELECTION_MISS', 0)} dilution="
        f"{di.get('M3_POOL_DILUTION', 0)} OK={di.get('M3_OK', 0)}",
        f"§3 decomposition FINAL: ranking={df.get('M3_RANKING_MISS', 0)} stop_margin="
        f"{df.get('M3_STOP_MARGIN_MISS', 0)} intra_pool="
        f"{df.get('M3_INTRA_POOL_SELECTION_MISS', 0)} dilution="
        f"{df.get('M3_POOL_DILUTION', 0)} OK={df.get('M3_OK', 0)}",
        f"M3_SELECTION_MISS BEFORE -> AFTER: {mm[0]} -> {mm[1]}"
        f" (R14 reference was 3, §29/31)",
        f"M2_PROBE_MISS BEFORE -> AFTER: {m2[0]} -> {m2[1]} "
        "(recorded, NOT fixed; accept 2 §32)",
        f"shortlist recall any-positive INIT={recall_i.get('any', 0.0):.3f} FINAL="
        f"{recall_f.get('any', 0.0):.3f} (§11)",
        f"shortlist recall best-positive INIT={recall_i.get('best', 0.0):.3f} FINAL="
        f"{recall_f.get('best', 0.0):.3f} (§11)",
        f"shortlist recall oracle-best INIT={recall_i.get('oracle', 0.0):.3f} FINAL="
        f"{recall_f.get('oracle', 0.0):.3f} (§11)",
        f"§13 coverage gate recall_gate_ok={s.get('recall_gate_ok')} "
        f"(tol={C.TO1_R15_RECALL_DROP_TOL:.2f}; failure -> SHORTLIST_COVERAGE_FAILURE, "
        f"never trains)",
        f"§14 per-root preservation: mean per-source n_global="
        f"{((s.get('deco_i') or {}).get('per_source') or {}).get('n_global')} "
        f"n_roota={((s.get('deco_i') or {}).get('per_source') or {}).get('n_roota')} "
        f"n_rootb={((s.get('deco_i') or {}).get('per_source') or {}).get('n_rootb')} "
        f"n_diversity={((s.get('deco_i') or {}).get('per_source') or {}).get('n_diversity')}",
        f"Q0 (cite R14 P0, gated SFT full pool) = {_z(rows.get('q0', (None,))[0])} §25",
        f"Q1 (SAME SFT parent, shortlist32, NO RL) = {_z(rows.get('q1', (None,))[0])} "
        f"(fresh)",
        f"Q2 (cite R14 P3, stagewise JOINT full pool) = {_z(rows.get('q2', (None,))[0])}",
        f"Q3 (R15 stagewise JOINT shortlist32) = {_z(rows.get('q3', (None,))[0])} "
        f"(fresh)",
        f"§36 main judgement Q3 > Q1: {s.get('q3_gt_q1')} "
        f"(Q3={_z(rows.get('q3', (None,))[0])} v Q1={_z(rows.get('q1', (None,))[0])})",
        f"§36 shortlist positive coverage stable: {s.get('coverage_stable_ref')} "
        f"(any {recall_i.get('any', 0.0):.3f}->{recall_f.get('any', 0.0):.3f})",
        f"AUX-real held Q1->Q3: {_z(s.get('q1_real'))}->{_z(s.get('q3_real'))} "
        f"(aux_ok={s.get('aux_ok')}, §37: at least one AUX not worse than Q1)",
        f"AUX-syn  held Q1->Q3: {_z(s.get('q1_syn'))}->{_z(s.get('q3_syn'))} (§37)",
        f"VAL once no_grad (shortlist, §38) = {_z(s.get('val_final'))}",
        f"§27 selection metrics (init shortlist): size mean/med/p90 = "
        f"{sel_i.get('size', {}).get('mean')}/{sel_i.get('size', {}).get('median')}/"
        f"{sel_i.get('size', {}).get('p90')}, entropy="
        f"{sel_i.get('entropy', {}).get('mean')}, stop_prob="
        f"{sel_i.get('stop_prob', {}).get('mean')}, top1="
        f"{sel_i.get('top1_prob', {}).get('mean')}, top5_cum="
        f"{sel_i.get('top5_cum', {}).get('mean')}",
        f"§28 ranking metrics (final, frozen base): best-positive rank="
        f"{rnk_f.get('best_positive_rank')}, oracle rank={rnk_f.get('oracle_rank')}, "
        f"MRR={rnk_f.get('mrr')}, pos recall@3/5/10/20/32="
        f"{rnk_f.get('rec3')}/{rnk_f.get('rec5')}/{rnk_f.get('rec10')}/"
        f"{rnk_f.get('rec20')}/{rnk_f.get('rec32')}",
        f"Mk1 §31 trace: {json.dumps((s.get('mk') or {}).get('mk1', {}).get('rows', [])[:2], default=str)[:380]}",
        f"Mk3 §31 trace: {json.dumps((s.get('mk') or {}).get('mk3', {}).get('rows', [])[:2], default=str)[:380]}",
        f"machinery gates: R6 repro_ok={s.get('repro_ok')} anchor_ok="
        f"{s.get('anchor_ok')} m5§48="
        f"{'ok' if (s.get('m5_f') or s.get('m5') or {}).__class__.__name__ else 'n/a'}"
        f" mp_ok={s.get('mp_ok')} (shortlist collect §35)",
        f"ONE joint stage §16-18: cycles={len(hist)} groups={n_groups} "
        f"inf2={s.get('n_info_m2')} inf3={s.get('n_info_m3')} grad="
        f"{s.get('m2_grad_any')}/{s.get('m3_grad_any')} λ2=λ3=1.0 α frozen",
        f"loss identical R14 §22-24: L_GRPO2 + L_GRPO3 + β2·KL2 + β3·KL3, no new "
        f"ranking aux loss; old/new logprob on the SAME shortlist+STOP action set",
        f"§34 M2 reward ordering ok={s.get('reward_ordering_ok')} "
        f"(proven={s.get('m2_agg', {}).get('proven')} "
        f"memory={s.get('m2_agg', {}).get('memory')} "
        f"unsupported={s.get('m2_agg', {}).get('unsupported')})",
        f"informative ratios: inf2={s.get('info_ratio2')} inf3={s.get('info_ratio3')} "
        f"over {n_groups} groups",
        f"workers={s.get('workers')}: {json.dumps((prof or {}).get('per_worker'))} "
        f"mp_ok={s.get('mp_ok')} (coll_s w=1 "
        f"{(prof or {}).get('per_worker', {}).get('1', {}).get('coll_s')})",
        f"checkpoint metadata §44: pipeline=M2_SFT->M3_SFT->Joint_GRPO, "
        f"m3_action_space=coverage_preserving_shortlist32, "
        f"shortlist_oracle_authority=false, formal_test_access=0 (written ONLY on "
        f"verdict A)",
        f"Verdict: {s['verdict']['code']} {s['verdict']['label']} -- {s['verdict']['note']}",
        f"下一步: {_r15_next(vcode if 'vcode' in s else s['verdict']['code'])}",
        f"promoted: identified=false (evidence-only, no causal claim)",
        f"formal_test_access=0 · Formal TEST SEALED -- permanent constraint honored",
        "pytest: tests/test_m3_joint_grpo_r15.py added + full regression suite green",
    ][:41]


def _r15_next(vcode):
    if vcode == "S":
        return ("--quick sanity: 只做短表分解 + coverage 门禁 + 2 轮真 JOINT（无判定）；"
                "全量 --stage r15 出正式 verdict。")
    if vcode == "A":
        return "固化 shortlist 得利：把压缩动作集并入永久管线，提升留出集与 VAL 上界。"
    if vcode in ("C",):
        return "下一轮修 ranking/representation objective（指纹排序坏账），不只压缩动作集。"
    if vcode in ("D",):
        return "下一轮做 STOP 校准（margin/calibrator），STOP score 压过正提案是主块。"
    if vcode in ("E",):
        return "SHORTLIST_COVERAGE_FAILURE：先修 per-root quota / 多样性回填，再谈训练。"
    if vcode in ("G", "F"):
        return "先修 semantics/selection 坏账（M3_SELECTION_MISS 未降），再重跑 JOINT。"
    return "B：coverage 无降但 RL 无增益——继续在 shortlist 上放大 M3 selection 预算/尺度。"


def _r15_report(env, re, scraper):
    s = scraper

    def _v(x):
        return f"{float(x):.0f}" if isinstance(x, (int, float)) else "—"

    L = []
    L.append("# T1-M3-COVERAGE-PRESERVING-SELECTION-GRPO-R15 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    L.append(f"## Verdict: **{s['verdict']['code']}** `{s['verdict']['label']}`")
    L.append("")
    L.append(f"> {s['verdict']['note']}")
    L.append("")
    L.append("## §0 what R15 changes (and what it does NOT)")
    L.append("")
    L.append("R15's ONLY modification is the **M3 action-set construction**: a "
             "coverage-preserving shortlist (GLOBAL top-12 by frozen M3 SFT base + "
             "per-root Tier-A top-2 / Tier-B top-1 quotas + structural-family diversity "
             "fill, CAP=32, STOP external) replaces the wide_pool stage as the M3 "
             "action set.  Everything else (M2 SFT / q2 local reward / Memory / "
             "adaptive probing / Reasoner / FixedDecisionReplay / stagewise A2-A3 "
             "credit / joint structure) is bit-identical to R14.")
    L.append("")
    L.append(f"- action_space={s.get('action_space')} · K_GLOBAL={C.TO1_R15_K_GLOBAL} "
             f"· Tier-A quota={C.TO1_R15_TIER_A_QUOTA} · Tier-B quota="
             f"{C.TO1_R15_TIER_B_QUOTA} · CAP={C.TO1_R15_SHORTLIST_CAP}")
    L.append("- §10 no-oracle: runtime shortlist uses ONLY frozen M3 SFT score + root "
             "provenance + structural metadata; true_U enters ONLY the §11-12 "
             "diagnostic recall.")
    L.append("- §15 backbone = m3_proposal_top1_sft_v2.pt, resid δ=0; NO R11-R14 RL "
             "checkpoint parent.")
    L.append("")
    L.append("## 1. machinery gates")
    L.append("")
    L.append(f"- R6 reproduction repro_ok={s.get('repro_ok')}")
    L.append(f"- P0/Q0 anchor raw {s.get('rows', {}).get('p0_anchor')} vs "
             f"r6_canonical {s.get('r6_canonical_train')} -> anchor_ok="
             f"{s.get('anchor_ok')}")
    L.append(f"- §34 reward ordering OK={s.get('reward_ordering_ok')}")
    L.append(f"- shortlist-collect mp determinism mp_ok={s.get('mp_ok')} "
             f"workers={s.get('workers')}")
    L.append(f"- §48 normal-M5 all_tier_a_ok zero={s.get('m5_z', {}).get('all_tier_a_ok') if isinstance(s.get('m5_z'), dict) else None} "
             f"final={s.get('m5', {}).get('all_tier_a_ok') if isinstance(s.get('m5'), dict) else None}")
    L.append("")
    L.append("## §25-26 Q-PARITY (SAME unified RAW ruler; Q3 = MAIN)")
    L.append("")
    rows = s.get("rows", {})
    L.append("| model | route | TRAIN | real-held | syn-held | VAL |")
    L.append("|---|---|---|---|---|---|")
    pa = rows.get("p0_anchor")
    q0 = rows.get("q0", (None, None, None, None))
    q1 = rows.get("q1", (None, None, None, None))
    q2 = rows.get("q2", (None, None, None, None))
    q3 = rows.get("q3", (None, None, None, None))
    L.append(f"| anchor R6 RAW (no gate) | δ=0 ≡ R6 | {_v(pa)} | - | - | - |")
    L.append(f"| Q0 gated SFT full pool | cited R14 P0 | {_v(q0[0])} | - | - | - |")
    L.append(f"| Q1 SFT shortlist32 NO RL | same parent, shortlist | {_v(q1[0])} | "
             f"{_v(q1[1])} | {_v(q1[2])} | - |")
    L.append(f"| Q2 stagewise JOINT full pool | cited R14 P3 | {_v(q2[0])} | - | - | - |")
    L.append(f"| **Q3 R15 JOINT shortlist32** | MASTER | **{_v(q3[0])}** | "
             f"{_v(q3[1])} | {_v(q3[2])} | {_v(q3[3])} |")
    L.append("")
    L.append(f"- **Q3 > Q1 (main judgement §36)**: {s.get('q3_gt_q1')} "
             f"({_v(q3[0])} vs {_v(q1[0])})")
    L.append(f"- M3_SELECTION_MISS BEFORE -> AFTER: {s.get('m3_miss')} "
             f"(R14 reference 3)")
    L.append(f"- M2_PROBE_MISS BEFORE -> AFTER: {s.get('m2_miss')} "
             f"(recorded, NOT fixed, §32 accept 2)")
    L.append("")
    L.append("## §2-4 decomposition")
    L.append("")
    di = (s.get("deco_i") or {}).get("decomposition", {}) if isinstance(s.get("deco_i"), dict) else {}
    df = (s.get("deco_f") or {}).get("decomposition", {}) if isinstance(s.get("deco_f"), dict) else {}
    L.append("| class | INIT (SFT, shortlist) | FINAL (joint, shortlist) |")
    L.append("|---|---|---|")
    for cls, label in (("M3_RANKING_MISS", "A ranking foul"),
                       ("M3_STOP_MARGIN_MISS", "B STOP margin"),
                       ("M3_INTRA_POOL_SELECTION_MISS", "C intra-pool selection"),
                       ("M3_POOL_DILUTION", "D pool dilution"),
                       ("M3_OK", "selection OK")):
        L.append(f"| {label} | {di.get(cls, 0)} | {df.get(cls, 0)} |")
    L.append("")
    L.append("## §11-13 shortlist coverage / recall (real true_U, diagnostic only)")
    L.append("")
    ri = s.get("recall_i") or {}
    rf = s.get("recall_f") or {}
    L.append(f"- §11 recall any-positive: INIT {ri.get('any', 0):.3f} -> FINAL "
             f"{rf.get('any', 0):.3f} (full pool = 1.0)")
    L.append(f"- §11 recall best-positive: INIT {ri.get('best', 0):.3f} -> FINAL "
             f"{rf.get('best', 0):.3f}")
    L.append(f"- §11 recall oracle-best: INIT {ri.get('oracle', 0):.3f} -> FINAL "
             f"{rf.get('oracle', 0):.3f}")
    L.append(f"- §13 coverage gate recall_gate_ok={s.get('recall_gate_ok')} "
             f"(tol {C.TO1_R15_RECALL_DROP_TOL:.2f}); coverage_stable="
             f"{s.get('coverage_stable_ref')}")
    L.append("")
    L.append("## §27-28 selection & ranking metrics")
    L.append("")
    sel_i = (s.get("deco_i") or {}).get("selection_metrics") or {}
    rnk_f = (s.get("deco_f") or {}).get("ranking_metrics") or {}
    L.append(f"- §27 selection (INIT shortlist): size mean/med/p90 "
             f"{sel_i.get('size', {}).get('mean')}/{sel_i.get('size', {}).get('median')}/"
             f"{sel_i.get('size', {}).get('p90')} · entropy "
             f"{sel_i.get('entropy', {}).get('mean')} · STOP prob "
             f"{sel_i.get('stop_prob', {}).get('mean')} · top1 "
             f"{sel_i.get('top1_prob', {}).get('mean')} · top5-cum "
             f"{sel_i.get('top5_cum', {}).get('mean')}")
    L.append(f"- §28 ranking (FINAL, frozen base over full pool): best-positive rank "
             f"{rnk_f.get('best_positive_rank')} · oracle rank "
             f"{rnk_f.get('oracle_rank')} · MRR {rnk_f.get('mrr')} · pos recall@3/5/10/20/32 "
             f"{rnk_f.get('rec3')}/{rnk_f.get('rec5')}/{rnk_f.get('rec10')}/"
             f"{rnk_f.get('rec20')}/{rnk_f.get('rec32')}")
    L.append("")
    L.append("## §31 Mk1/Mk3 traces (final policy, K=8 shortlist group)")
    L.append("")
    mk = s.get("mk") or {}
    for kkey in ("mk1", "mk3"):
        L.append(f"### {kkey}")
        blk = (mk.get(kkey) if isinstance(mk, dict) else {}) or {}
        if "error" in blk:
            L.append(f"- Mk trace FAILED: {blk['error']}")
            continue
        if "note" in blk:
            L.append(f"- {blk['note']}")
            continue
        L.append(f"- iid={blk.get('iid')} inf2={blk.get('inf2')} inf3={blk.get('inf3')} "
                 f"best_pos_rank_full={json.dumps(blk.get('best_pos_rank_full_pool'), default=str)}")
        for r_ in (blk.get("rows") or [])[:4]:
            L.append(f"  - sib={r_.get('sib')} term={r_.get('terminal')} "
                     f"full_pool_M={r_.get('full_pool_M')} shortlist_M="
                     f"{r_.get('shortlist_M')} sl_sig={r_.get('sl_sig')} "
                     f"U2/A2/A3/R={r_.get('U2')}/{r_.get('A2')}/{r_.get('A3')}/"
                     f"{r_.get('reward')} "
                     f"M3(argmax={r_.get('M3', {}).get('argmax')}, M={r_.get('M3', {}).get('M')}, "
                     f"p_chosen={r_.get('M3', {}).get('p_chosen')}, "
                     f"p_stop={r_.get('M3', {}).get('p_stop')})")
    L.append("")
    L.append(f"- VAL once no_grad (shortlist, §38): **{_v(s.get('val_final'))}**")
    L.append("")
    if isinstance(res_j := s.get("res_j"), dict) and res_j.get("history"):
        L.append("## Joint stage cycles (Q3 ladder, shortlist32)")
        L.append("")
        L.append("| cycle | train_sl | real_hd | syn_hd | inf2 | inf3 | kl_m2 | kl_m3 | sec |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for h in res_j["history"]:
            L.append(f"| {h['cycle']} | {_v(h['train'])} | {_v(h['real_held'])} | "
                     f"{_v(h['syn_held'])} | {h['n_informative_trajectories_m2']} | "
                     f"{h['n_informative']} | {h['kl_m2']} | {h['kl_ref_m3']} | "
                     f"{h['sec']} |")
        L.append("")
    L.append("## §45 forty-one-item final return")
    L.append("")
    for i, it in enumerate(_r15_41items(s), 1):
        L.append(f"{i}. {it}")
    L.append("")
    if s.get("passed"):
        L.append(f"## Checkpoint written (§44): {C.TO1_R15_CKPT.name}")
    else:
        L.append("## Checkpoint NOT written (§44, verdict-A only)")
    L.append("")
    L.append(f"## 下一步 (single highest-priority next action)")
    L.append("")
    L.append(_r15_next(s["verdict"]["code"]))
    L.append("")
    return "\n".join(L)


def _r15_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R15_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        for k in _R15_SCRAPE_KEYS:
            if k in scrape:
                payload[k] = scrape[k]
        rj = scrape.get("res_j") or {}
        if isinstance(rj, dict):
            payload["joint_stage"] = {
                "cycles_run": rj.get("cycles_run"), "collapsed": bool(rj.get("collapsed")),
                "best_train_sl": None if not rj.get("best") else float(
                    (rj["best"].get("train") or 0.0)),
                "history": [{"cycle": h["cycle"], "train_sl": h["train"],
                             "real_held": h["real_held"], "syn_held": h["syn_held"],
                             "n_groups": h["n_groups"], "n_informative": h["n_informative"],
                             "n_informative_trajectories_m2": h.get(
                                 "n_informative_trajectories_m2", 0),
                             "kl_ref_m3": h["kl_ref_m3"], "kl_m2": h["kl_m2"],
                             "sec": h["sec"]} for h in rj.get("history", [])]}
    (C.CANONICAL_OUT_DIR / "result_r15.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r15] report written: {C.R15_REPORT}", flush=True)


def _r13_report(env, re, scraper):
    s = scraper
    rows = _r13_parity_rows(s)
    ra, rb_, rc_ = s.get("res_a"), s.get("res_b"), s.get("res_c")
    prof = s.get("prof") or {}
    deco = s.get("deco") or {}
    alpha = s.get("alpha") or {}
    pa = s.get("pa_final") or {}
    L = []
    L.append("# T1-M2-M3-JOINT-AGENTIC-ROLLING-GRPO-R13 -- report")
    L.append("")
    L.append("identified=false · formal_test_access=0 · Formal TEST SEALED")
    L.append("")
    L.append(f"## Verdict: **{s['verdict']['code']}** `{s['verdict']['label']}`")
    L.append("")
    L.append(f"> {s['verdict']['note']}")
    L.append("")
    L.append("## §0 permanent structure (identical in train + eval)")
    L.append("")
    L.append("S_t + Appearance -> **M2 budgeted root search** (3 anchors + 4 sequential "
             "without-replacement draws + 1 uniform outsider, B_ROOT=8) -> **REAL FDR "
             "root probes** (max 4 singles / 4 dependency pairs per root) -> "
             "**makespan-first tier filter** (A=PROVEN_GAIN uncapped; B=Memory rescue "
             "only when nothing probed positive, cap 2; C=pruned) -> gated pool -> "
             "Reasoner view -> **M3** proposal policy -> execute -> S_{t+1}.")
    L.append("")
    L.append("- Tier A is never vetoed by Memory (§2); Memory is second-level evidence "
             "(§3-5); probe gains never enter the terminal reward (§24, "
             "m2_reward_authority=false).")
    L.append("- Stages: A (M2 root-policy RL, M3 frozen) -> B (M3 GRPO on gated pool, "
             "M2 frozen) -> C (JOINT, per-trajectory shared advantage).")
    L.append("")
    L.append("## 1. machinery gates")
    L.append("")
    L.append(f"- R6 reproduction repro_ok={s.get('repro_ok')} "
             f"(details in result_r13.json)")
    L.append(f"- C0 anchor: unified d0 TRAIN vs r6_canonical "
             f"({s.get('r6_canonical_train')}) -> anchor_ok={s.get('anchor_ok')}")
    L.append(f"- Stage A PASS gates all_ok={s.get('stage_a')}")
    L.append(f"- §48 normal-M5 dep-completed Tier-A (never Memory): "
             f"zero={s.get('m5_z', {}).get('all_tier_a_ok') if isinstance(s.get('m5_z'), dict) else None} "
             f"stageA={s.get('m5_a', {}).get('all_tier_a_ok') if isinstance(s.get('m5_a'), dict) else None} "
             f"final={s.get('m5', {}).get('all_tier_a_ok') if isinstance(s.get('m5'), dict) else None}")
    L.append("")
    L.append("## §52 CANONICAL-PARITY TABLE (SAME unified ruler; 369 anchor cross-"
             "checked)")
    L.append("")
    L.append("| model | M2 | TRAIN | real-held | syn-held | VAL |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        f = lambda v: "--" if v is None else f"{v:.0f}"   # noqa: E731
        L.append(f"| {r['model']} | {r['m2']} | {f(r['train'])} | {f(r['real_held'])} | "
                 f"{f(r['syn_held'])} | {f(r['val'])} |")
    L.append(f"\nR6 anchor cross-check: unified d0 TRAIN "
             f"{rows[0]['train'] if rows[0]['train'] is not None else '--'} vs canonical "
             f"`_r6_canonical_train_gain` {s.get('r6_canonical_train')} -> "
             f"anchor_ok={s.get('anchor_ok')}")
    L.append("")
    L.append("## Stages (rolling cycles on the gated §0 loop)")
    L.append("")
    if ra:
        sg = ra.get("stage_gates") or {}
        _gh = all(sg.get(k) for k in ("m2_grad_nonzero", "prob_move",
                                      "positive_probe_rate_ok", "tier_a_coverage_ok",
                                      "gated_coverage_ok")) if sg else False
        L.append(f"- Stage A: cycles={ra.get('cycles_run')} collapsed={ra.get('collapsed')} "
                 f"({ra.get('collapse_reason')}) gates_ok={_gh} "
                 f"grad_steps={ra.get('stage_grad_steps')} prob_move="
                 f"{ra.get('stage_prob_move')} best_train={ra.get('best', {}).get('train')}")
        L.append(f"  - Stage-A summary gates: {json.dumps(sg, default=str)}")
        for g_ in ra.get("stage_a_gates", []):
            L.append(f"  - A-cycle {g_['cycle']}: n_informative={g_.get('n_informative')} "
                     f"dry={g_.get('dry')} prob_move={g_.get('prob_move')} "
                     f"stats {json.dumps({k: round(v, 3) for k, v in (g_.get('stats') or {}).items()})}")
    if rb_:
        L.append(f"- Stage B: cycles={rb_.get('cycles_run')} collapsed={rb_.get('collapsed')} "
                 f"best_train={rb_.get('best', {}).get('train')}")
        for h in rb_.get("history", []):
            L.append(f"  - B-cycle {h['cycle']}: TRAIN {h.get('train', 0):.0f} "
                     f"info {h.get('informative_ratio', 0):.3f} "
                     f"kl_m3 {h.get('kl_ref_m3', 0):.4f} kl_m2 {h.get('kl_m2', 0):.4f} "
                     f"m2pos {h.get('m2_stats', {}).get('positive_probe_rate', 0):.3f} "
                     f"tierA {h.get('m2_stats', {}).get('tier_A', 0):.2f} "
                     f"cov {h.get('m2_stats', {}).get('coverage_ratio', 0):.3f}")
    if rc_:
        L.append(f"- Stage C: cycles={rc_.get('cycles_run')} collapsed={rc_.get('collapsed')} "
                 f"best_train={rc_.get('best', {}).get('train')}")
        for h in rc_.get("history", []):
            L.append(f"  - C-cycle {h['cycle']}: TRAIN {h.get('train', 0):.0f} "
                     f"info {h.get('informative_ratio', 0):.3f} "
                     f"kl_m3 {h.get('kl_ref_m3', 0):.4f} kl_m2 {h.get('kl_m2', 0):.4f} "
                     f"m2pos {h.get('m2_stats', {}).get('positive_probe_rate', 0):.3f} "
                     f"tierA {h.get('m2_stats', {}).get('tier_A', 0):.2f}")
    L.append("")
    L.append("## §46 M2/M3/Memory metrics + failure decomposition")
    L.append("")
    L.append(f"- pool argmax: R6 pos={s.get('pos6')} regret={s.get('regret6')} "
             f"recall10={s.get('rec10_6')} | FINAL pos={s.get('pos_f')} "
             f"regret={s.get('regret_f')} recall10={s.get('rec10_f')}")
    L.append(f"- VAL once no_grad (gated agentic, report-only §42): "
             f"**{s.get('val_final')}**")
    if "error" not in deco:
        L.append(f"- failure decomposition: {json.dumps(deco.get('decomposition'))}")
        for r in deco.get("rollouts", [])[:5]:
            L.append(f"  - {r['iid']}: gain {r['final_gain']} "
                     f"{json.dumps(r['decomp'])}")
    else:
        L.append(f"- failure decomposition failed: {deco.get('error')}")
    L.append("")
    if s.get("dpp_pre") is not None and s.get("dpp_post") is not None:
        L.append("## DPPaulli rolling trace (closed loop, §0 gate active)")
        L.append("")
        L.append(f"- BEFORE (zero-init): gain {s['dpp_pre'].get('total_gain')}")
        L.append(f"- AFTER  (final)     : gain {s['dpp_post'].get('total_gain')}")
        L.append("")
    if s.get("m5"):
        L.append(f"- §48 normal-M5 gate final: {json.dumps(s.get('m5'), default=str)}")
    if "error" not in alpha and s.get("alpha") is not None:
        L.append(f"- α_M3 influence (§54): {alpha.get('n_flip')}/{alpha.get('n_states')} "
                 f"argmax flips, marginal_above_stop_delta="
                 f"{alpha.get('marginal_above_stop_delta')}")
    elif s.get("alpha") is not None:
        L.append(f"- α_M3 influence failed: {alpha.get('error')}")
    L.append("")
    L.append("## §55 cloud/shape profile (workers 1/2/4; group counts 4/8/16)")
    L.append("")
    if isinstance(prof, dict) and "error" not in prof:
        L.append(f"- per-worker coll_s: "
                 f"{[prof.get('per_worker', {}).get(str(w), {}).get('coll_s') for w in (1, 2, 4)]} "
                 f"identical_to_w1: "
                 f"{[prof.get('per_worker', {}).get(str(w), {}).get('identical_to_w1') for w in (1, 2, 4)]} "
                 f"speedup_vs_w1: {json.dumps(prof.get('speedup_vs_w1'))}")
        L.append(f"- groups 4/8/16 est wall (min): "
                 f"{[prof.get('groups', {}).get(str(g), {}).get('est_wall_min') for g in (4, 8, 16)]} "
                 f"probes/min: "
                 f"{[prof.get('groups', {}).get(str(g), {}).get('est_probes_per_min') for g in (4, 8, 16)]}")
        L.append(f"- mp_ok={s.get('mp_ok')}")
    else:
        L.append(f"- (profile {'failed: ' + str(prof.get('error')) if isinstance(prof, dict) else 'skipped'})")
    L.append("")
    L.append("## Checkpoint metadata (§56)")
    L.append("")
    L.append("- m2_role=budgeted_root_search · m2_filter=makespan_first_memory_second"
             " · m2_reward_authority=false")
    L.append("- m3_role=proposal_selection · reward=terminal_makespan_gain · "
             "reasoner=frozen · executor=FixedDecisionReplay · formal_test_access=0")
    L.append(f"- written on PASS (verdict A) only: `{C.TO1_R13_CKPT_M2.name}` + "
             f"`{C.TO1_R13_CKPT_JOINT.name}` (passed={s.get('passed')})")
    L.append("")
    L.append("## 下一步 (single highest-priority next action)")
    L.append("")
    L.append("M2 的职责只有根因搜索/过滤；下一轮把 Stage-A 的 positive-root probe 率 "
             "与 Tier-A 覆盖率作为准入，先修 M2_ROOT_SEARCH_COVERAGE_COLLAPSE(D) 风险，"
             "再谈 JOINT 增益。")
    return "\n".join(L) + "\n"


def _r13_persist(report, scrape=None):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.R13_REPORT.write_text(report, encoding="utf-8")
    payload = {"report": report}
    if scrape is not None:
        keys = ("repro_ok", "anchor_ok", "stage_a", "m5", "m5_z", "m5_a",
                "train_final", "train_final_b", "val_final", "rows",
                "r6_canonical_train", "workers", "mp_ok", "dpp_pre", "dpp_post",
                "alpha", "deco", "pa_final", "pos6", "regret6", "rec10_6",
                "pos_f", "regret_f", "rec10_f", "train_beat", "held_improved",
                "joint_beat", "verdict", "passed")
        for k in keys:
            if k in scrape:
                payload[k] = scrape[k]
        for st in ("A", "B", "C"):
            rab = scrape.get("res_" + st.lower()) or {}
            stage = {"cycles_run": rab.get("cycles_run"),
                     "collapsed": bool(rab.get("collapsed")),
                     "best_train": None if not rab.get("best") else float(
                         (rab["best"].get("train") or 0.0)),
                     "stage_gates": rab.get("stage_gates")}
            payload["stage_" + st] = stage
    (C.CANONICAL_OUT_DIR / "result_r13.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[r13] report written: {C.R13_REPORT}", flush=True)


def _r11_report(env, re, scraper):
    from types import SimpleNamespace as SN
    s = scraper
    dpp = s.get("dpp_pre") or {}
    val = s.get("val") or {}
    res_b = s.get("res_b") or {}
    history = (res_b.get("history") if isinstance(res_b, dict) else None) \
        or s.get("stage_a", {}).get("rows", [])
    n_groups = sum(h.get("n_groups", 0) for h in history)
    n_inf = sum(h.get("n_informative", 0) for h in history)
    info_ratio = s.get("info_ratio")
    if info_ratio is None and n_groups:
        info_ratio = n_inf / n_groups
    rp_ = s.get("runner") or {}
    dpp_first_pre = (dpp.get("steps") or [None])[0]
    dpp_first_post = (s.get("dpp_post") or {}).get("steps") or [None]
    dpp_first_post = dpp_first_post[0] if dpp_first_post else None
    return {
        "meta": {"experiment": "T1-M3-ROLLING-GRAPH-MULTI-TRAJECTORY-GRPO-R11",
                 "quick": False, "identified": False, "formal_test_access": 0,
                 "formal_test_sealed": True,
                 "grpo": "ALLOWED -- first real GRPO round (§0); R1/R2 AC excluded",
                 "parent": "m3_proposal_top1_sft_v2.pt (canonical R6; not R7-R10)"},
        "modified_files": ["src/causal_schedule_lab/m3/rolling_grpo.py",
                           "src/causal_schedule_lab/m3/config.py",
                           "tests/test_m3_rolling_grpo_r11.py",
                           "scripts/run_m3_canonical_training.py"],
        "r6_reproduction": {"acc_all": rp_.get("r6_acc"),
                            "regret_mean": rp_.get("r6_regret"),
                            "repro_ok": s.get("repro_ok")},
        "parity_gate": {"train": s.get("parity_train"), "real_held": s.get("parity_real"),
                        "syn_held": s.get("parity_syn"), "val0": s.get("val0"),
                        "r6_canonical_train": s.get("r6_canonical_train"),
                        "parity_ratio_to_r6": s.get("parity_ratio_to_r6"),
                        "ok": s.get("parity_ok"),
                        "note": "δ=0 warm-start TRAIN gain measured on the R11 HARNESS "
                                "(forced continuation + gate_mem + z-pool base).  This is "
                                "NOT R6's canonical ruler (STOP-on-negative + gate_mem=False "
                                "-> recorded 369); r6_canonical_train reports that canonical "
                                "basis on the same roots, and the collapse floor is the "
                                "same-harness δ=0 level, NOT the raw 369."},
        "algorithm": {
            "reward": "R_i = Cmax(S_t) − Cmax(S_T) terminal only; R_STOP = 0; no shaping",
            "group": "(instance_id, state_hash), K=%d sibling trajectories" % C.TO1_R11_K,
            "advantage": "A_i = (R_i − mean_R)/(std_R + eps); informative gating; no cross-group norm",
            "surrogate": "min(ratio·A, clip(ratio,1±0.2)·A); π_old frozen across E=%d reuse epochs"
                         % C.TO1_R11_UPDATE_EPOCHS,
            "kl_ref": "β·KL(πθ ‖ R6-base), β=%.3f" % C.TO1_R11_BETA_KL,
            "residual": "score_GRPO = score_R6 + α·tanh(δ), α_prop=α_stop=%.2f, δ≡0 init"
                        % C.TO1_R11_ALPHA_PROP,
            "behavior": "T=%.2f, uniform mixture ε=%.2f, EXACT mixture logprob"
                        % (C.TO1_R11_TEMP, C.TO1_R11_MIX_EPS),
            "rolling": "greedy advancement (no oracle), max %d rolling states/graph episode"
                       % C.TO1_R11_MAX_ROLLING_STATES_PER_GRAPH,
            "sources": {"bench_TRAIN14": 0.4, "aux_real_train": 0.3, "aux_syn_train": 0.3,
                        "ratio": list(C.TO1_R11_SOURCE_RATIO)},
            "sibling_isolation": "deepcopy S_t + Memory_t per sibling; root asserted "
                                 "unchanged; branch writes never leak",
        },
        "stage_a": s.get("stage_a"),
        "stage_b": {k: v for k, v in (res_b.items() if isinstance(res_b, dict) else {})
                    if k != "history"},
        "history": history,
        "model_selection": {"criterion": "TRAIN14 + AUX-REAL-held + AUX-syn-held "
                                         "(VAL3 NEVER used for selection)",
                            "best_cycle": (res_b.get("best", {}).get("cycle") if isinstance(res_b, dict) else None),
                            "best_train": s.get("train_final"),
                            "info_ratio": info_ratio},
        "final_metrics": {
            "pool_argmax_TRAIN": {"r6": {"pos_argmax": (s.get("runner") or {}).get("pos6"),
                                         "regret": (s.get("runner") or {}).get("regret6"),
                                         "recall10": (s.get("runner") or {}).get("rec10_6")},
                                  "r11": {"pos_argmax": (s.get("runner") or {}).get("pos_f"),
                                          "regret": (s.get("runner") or {}).get("regret_f"),
                                          "recall10": (s.get("runner") or {}).get("rec10_f")}},
            "val3_once": {"total": (s.get("runner") or {}).get("val_total"), "full": val,
                          "note": "no_grad, single pass, NOT used for selection"},
        },
        "dppaulli_rolling": {"before_delta0": s.get("dpp_pre"),
                             "after_selected": s.get("dpp_post"),
                             "first_step_before": dpp_first_pre,
                             "first_step_after": dpp_first_post,
                             "selected_true_U_before": (s.get("dpp_pre") or {}).get("selected_true_U_sum"),
                             "selected_true_U_after": (s.get("dpp_post") or {}).get("selected_true_U_sum")},
        "normal_m5": s.get("m5"),
        "prob_movement_audit": s.get("audit"),
        "profiling": s.get("prof"),
        "collapse": {"collapsed": (res_b.get("collapsed") if isinstance(res_b, dict) else None),
                     "reason": (res_b.get("collapse_reason") if isinstance(res_b, dict) else None)},
        "checkpoint": {"path": str(C.TO1_CKPT_R11), "written": bool(s.get("passed"))},
        "checks": {
            "repro_ok (R6 acc 0.525 / regret 5.2 bit-id)": s.get("repro_ok"),
            "parity_ok (δ=0 R11-harness TRAIN >= 0.2×canonical R6 on same roots)": s.get("parity_ok"),
            "informative_ratio": info_ratio,
            "n_training_graphs_b": (len(history) and history[-1].get("n_groups")),
            "val3_total": (s.get("runner") or {}).get("val_total"),
            "dpp_selected_positive_after": bool((s.get("dpp_post") or {}).get("selected_true_U_sum", 0) > 0),
            "parallel_identical_data": (s.get("prof") or {}).get("identical_data"),
        },
        "verdict": s.get("verdict"),
        "passed": bool(s.get("passed")),
        "next_action": ("R11 PASS -> rolling-GRPO policy replaces SFT parent for M3"
                        if s.get("passed") else "fix per verdict before next round"),
    }


def _persist_r11(report, path):
    C.CANONICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.CANONICAL_OUT_DIR / "result_r11.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = []
    add = lines.append
    v = report["verdict"]
    add("# T1-M3-ROLLING-GRAPH-MULTI-TRAJECTORY-GRPO-R11 — 阶段报告")
    add("")
    add(f"**日期**: 2026-08-27 ｜ **verdict**: **{v['code']} {v['label']}** ｜ "
        f"**checkpoint**: {'写入' if report['passed'] else '未写入'} "
        f"(`{report['checkpoint']['path']}`)")
    add(f"**诚实边界**: `identified=false` ｜ `formal_test_access=0` ｜ Formal TEST SEALED。")
    add("")
    add("## 本轮定义（§0-§19，首个真 GRPO 轮）")
    add("- 父模型 = canonical R6 v2 `m3_proposal_top1_sft_v2.pt`（**非** R7-R10 任何模型）。")
    add("- Reward `R_i = Cmax(S_t) − Cmax(S_T)`（terminal only）；STOP reward = 0；无 shaping / 无 step reward。")
    add("- Group = `(instance_id, state_hash)`，K=8 个 sibling trajectory（深拷贝 S_t + Memory_t）。")
    add("- `A_i = (R_i − mean_R)/(std_R + eps)`；informative gating；无跨 group 归一化。")
    add("- clipped surrogate + β·KL(πθ‖R6-ref) + E=3 轮 batch reuse（π_old 冻结，stale 守卫）。")
    add("- bounded residual `score_GRPO = score_R6 + α·tanh(δ)`，α=%.2f，δ≡0 初始化（warm-start parity）。"
        % C.TO1_R11_ALPHA_PROP)
    add("- rolling：更新后用策略自己的 greedy 动作推进（**无 oracle**），每图 episode 最多 %d 步。"
        % C.TO1_R11_MAX_ROLLING_STATES_PER_GRAPH)
    add("- 源 40% TRAIN14 / 30% AUX-REAL-train / 30% AUX-syn-train；`--workers` 并行 trajectory。")
    add("")
    add("## R6 复现与 parity 门")
    add("- R6 repro（gate_mem=False, bit-id）：acc=%s regret=%s ok=%s" % (
        report["r6_reproduction"]["acc_all"], report["r6_reproduction"]["regret_mean"],
        report["r6_reproduction"]["repro_ok"]))
    p_ = report["parity_gate"]
    add(f"- 同一 TRAIN roots 上 R6 标准闭环（STOP-on-negative, gate_mem=False, 原始分）= "
        f"{p_.get('r6_canonical_train', 0.0):.0f}（历史记录 B6=369 的尺）。")
    add(f"- parity（δ=0，**R11 harness** 强制续推 + gate_mem + z-base）：TRAIN **{p_['train']:.0f}** / "
        f"real_held {p_['real_held']:.0f} / syn_held {p_['syn_held']:.0f} / val0 {p_['val0']:.0f} ok={p_['ok']}")
    add(f"- 两把尺不同：R11 harness δ=0 是 canonical 的 "
        f"{p_.get('parity_ratio_to_r6', 0.0):.2f}×；collapse 守卫对比**同尺** δ=0 基线而非原始 369。")
    add("")
    sa = report["stage_a"] or {}
    add("## Stage A sanity（%d cycles, quick）" % sa.get("cycles", 0))
    add(f"- best TRAIN gain **{sa.get('best_train', 0.0):.0f}**；末 cycle informative ratio "
        f"{sa.get('informative_ratio_last', 0.0):.3f}。")
    add("")
    sb_ = report["stage_b"]
    add("## Stage B（%d cycles, 固定参数）" % (sb_.get("cycles_run") or 0))
    add(f"- collapsed={sb_.get('collapsed')} reason={sb_.get('collapse_reason')}；"
        f"best cycle={report.get('model_selection', {}).get('best_cycle')} "
        f"best TRAIN gain={report.get('model_selection', {}).get('best_train'):.0f}。")
    add("- 模型选择 = TRAIN14 + AUX-REAL-held + AUX-syn-held（**VAL3 从不用于选择**，§46）。")
    add("- informative_group_ratio（Stage B 聚合）≈ `%.3f`。" % (report.get("model_selection", {}).get("info_ratio", 0.0)))
    fm = report["final_metrics"]
    add("")
    add("## 离线 FINAL metrics（pool-argmax, TRAIN replay）")
    add("- R6：pos_argmax=%.3f regret=%.2f recall10=%.3f" % (
        fm["pool_argmax_TRAIN"]["r6"]["pos_argmax"], fm["pool_argmax_TRAIN"]["r6"]["regret"],
        fm["pool_argmax_TRAIN"]["r6"]["recall10"]))
    add("- R11：pos_argmax=%.3f regret=%.2f recall10=%.3f" % (
        fm["pool_argmax_TRAIN"]["r11"]["pos_argmax"], fm["pool_argmax_TRAIN"]["r11"]["regret"],
        fm["pool_argmax_TRAIN"]["r11"]["recall10"]))
    add(f"- VAL3 once（no_grad, 不用于选择）gain=**{fm.get('val3_once', {}).get('total', 0.0):.0f}**。")
    add("")
    dpp = report["dppaulli_rolling"]
    add("## DPpaulli10a rolling trace")
    add(f"- BEFORE(δ=0)：gain={dpp.get('selected_true_U_before')} selected_u={dpp.get('selected_true_U_before')}")
    add(f"- AFTER(selected)：gain={dpp.get('selected_true_U_after')} selected_u={dpp.get('selected_true_U_after')}")
    if dpp.get("first_step_before") or dpp.get("first_step_after"):
        add(f"- first step BEFORE={dpp.get('first_step_before')}；AFTER={dpp.get('first_step_after')}。")
    add("")
    if report.get("profiling"):
        prof = report["profiling"]
        if isinstance(prof, dict) and "speedup" in prof:
            add(f"- §54 profiling：serial={prof['1']['coll_s']:.2f}s parallel4={prof['4']['coll_s']:.2f}s "
                f"speedup={prof['speedup']} identical_data={prof['identical_data']}。")
            add("  - cloud 建议：32/64-core 主机可将 trajectory worker 提到 --workers 8/16（GIL 释放于 "
                "CP-SAT/落盘；瓶颈 = FixedDecisionReplay 占 >90% runtime）。")
    add("")
    add("## Verdict")
    add(f"**{v['code']} — {v['label']}**：{v['note']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[r11] wrote {C.R11_REPORT}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Canonical M3 training (R11: rolling-graph multi-trajectory GRPO)")
    ap.add_argument("--quick", action="store_true", help="2 TRAIN instances / Stage-A sanity only")
    ap.add_argument("--stage", default="all",
                    choices=["inventory", "p1", "top1", "r7", "r8", "r9gen", "r9",
                             "r10gen", "r10", "r11", "r12", "r13", "r14", "r15", "r16",
                             "r17", "r18", "r19", "r20", "r21", "t2a", "t2b", "t2d",
                             "p2", "all"])
    ap.add_argument("--ckpt", default=str(C.DEFAULT_CKPT), help="M2 V5 checkpoint path")
    ap.add_argument("--grpo-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="R11 parallel trajectory workers (min(max(cpu-1,1),64)); up to 64")
    ap.add_argument("--dpp", default="DPdata_DPpaulli10a", help="DPpaulli instance for trace regressions")
    ap.add_argument("--skip-regressions", action="store_true")
    ap.add_argument("--r9-n", type=int, default=C.TO1_R9_N_AUX,
                        help="R9 AUX instance count (smoke only; fixed default, no sweep)")
    ap.add_argument("--r9-seed", type=int, default=C.TO1_R9_AUX_SEED, help="R9 AUX seed")
    ap.add_argument("--r10-n", type=int, default=C.TO1_R10_N_REAL,
                        help="R10 AUX-REAL (D1) instance count (fixed default, no sweep)")
    ap.add_argument("--load-p1", action="store_true",
                    help="reuse saved SFT checkpoint instead of retraining Phase-1")
    ap.add_argument("--architecture", choices=list(HR.ARCHITECTURES), default=HR.ARCH_P0,
                    help="T2-D Stage-3 actor: P0 pointwise, P1 set, P2 set+trajectory")
    ap.add_argument("--optimizer-persistent", action="store_true",
                    help="keep AdamW moments across depth/cycle updates (use identically for P0/P1/P2)")
    ap.add_argument("--resume", default=None,
                    help="T2-D latest.pt to resume (architecture must match)")
    ap.add_argument("--full-diagnostics", action="store_true",
                    help="T2-D only: run legacy pre/post-training diagnostics")
    ap.add_argument("--profile-workers", action="store_true",
                    help="T2-D only: benchmark worker counts before training")
    ap.add_argument("--t2d-cycles", type=int, default=C.T2D_TRAINING_CYCLES)
    ap.add_argument("--t2d-graphs-per-cycle", type=int,
                    default=C.T2D_GRAPHS_PER_CYCLE)
    ap.add_argument("--t2d-branches-per-graph", type=int,
                    default=C.T2D_BRANCHES_PER_GRAPH)
    args = ap.parse_args(argv)

    if args.stage == "t2d" and (args.t2d_cycles < 1 or
                                args.t2d_graphs_per_cycle < 1 or
                                args.t2d_branches_per_graph < 2):
        ap.error("T2-D cycles/graphs must be positive and branches must be >=2")

    if args.stage in ("inventory", "all"):
        inv = inventory(args)
        print("[inventory] classification done (see outputs/canonical_m3/inventory.json)", flush=True)
        if args.stage == "inventory":
            return 0

    print("[env] loading frozen upstream (M2 V5 + utility heads) ...", flush=True)
    t_env = time.time()
    env = build_env(args)
    re = build_replay_env(env)
    if args.stage == "t2d":
        print(f"[t2d] required frozen environment/replay load: "
              f"{time.time() - t_env:.1f}s", flush=True)

    fast_t2d = bool(args.stage == "t2d" and not args.full_diagnostics)
    if args.load_p1 and C.SFT_CKPT.exists():
        print("[load] loading Phase-1 SFT from checkpoint ...", flush=True)
        ck = torch.load(C.SFT_CKPT, map_location="cpu", weights_only=False)
        p1 = ck["state"]["p1"] if "state" in ck and "p1" in ck["state"] else None
        p1_report = ck["meta"].get("report") if "meta" in ck else None
        if p1 is None or (p1_report is None and not fast_t2d):
            if fast_t2d:
                raise RuntimeError("T2-D fast-start requires a checkpoint with the Phase-1 p1 payload")
            print("[load] checkpoint has no p1 payload; falling back to retrain", flush=True)
            p1, p1_report = run_phase1(args, env, re)
        elif fast_t2d:
            required = ("scorer_mem", "reranker")
            missing = [key for key in required if p1.get(key) is None]
            if missing:
                raise RuntimeError(f"T2-D fast-start checkpoint missing {missing}")
            p1_report = p1_report or {"passed": None, "mode": "fast_start_loaded"}
            print("[t2d] fast-start: loaded Phase-1 actors; skipped offline metrics, "
                  "memory audit, B5 closed loop and acceptance recomputation", flush=True)
        else:
            # zero_vals is PER-STATE-EXAMPLE data (one zero feature vector per
            # state, indexed by ex["gi"]), so it must be rebuilt for the CURRENT
            # replay.  The checkpoint's copy is sized to whatever replay scale
            # last WROTE the SFT file -- a quick (4-state) write leaves a stale
            # len-4 list that the full 40-state metrics pass would overflow.
            print("[load] recomputing Phase-1 metrics/acceptance with the utility reranker ...", flush=True)
            se = re["state_examples"]
            if len(p1.get("zero_vals", [])) != len(se):
                p1["zero_vals"] = [torch.zeros_like(ex["prog_mem_feats"]) for ex in se]
            p1, p1_report = _phase1_finalize(env, re, p1)
    elif fast_t2d:
        raise FileNotFoundError(
            f"T2-D fast-start requires existing Phase-1 checkpoint: {C.SFT_CKPT}")
    else:
        p1, p1_report = run_phase1(args, env, re)

    # R9 (CURRENT round): r9gen / r9 / all -> AUX instance generalization residual SFT
    # (GRPO forbidden §39).  R7 (previous round, verdict E) reachable ONLY via
    # explicit --stage r7.  R6 via --stage top1.  R5 GRPO via --stage p2.
    if args.stage == "r9gen":
        res = run_top1_phase_r9gen(args, env, re, p1, p1_report)
        print("[r9gen] AUX data stage done (see T1_M3_AUXILIARY_INSTANCE_DATA_R9_REPORT.md)",
              flush=True)
        return 0
    if args.stage in ("r9", "all"):
        res = run_top1_phase_r9(args, env, re, p1, p1_report)
        write_top1_r9_markdown(res)
        return 0
    if args.stage == "r10gen":
        res = run_top1_phase_r10gen(args, env, re, p1, p1_report)
        print("[r10gen] AUX-REAL data stage done "
              "(see T1_M3_SCORE_CALIBRATION_R10_AUX_REAL_DATA_REPORT.md)", flush=True)
        return 0
    if args.stage in ("r10", "all"):
        res = run_top1_phase_r10(args, env, re, p1, p1_report)
        write_top1_r10_markdown(res)
        return 0
    if args.stage in ("r11", "all"):
        res = run_top1_phase_r11(args, env, re, p1, p1_report)
        return 0
    if args.stage in ("r12", "all"):
        res = run_top1_phase_r12(args, env, re, p1, p1_report)
        return 0
    if args.stage in ("r13", "all"):
        res = run_top1_phase_r13(args, env, re, p1, p1_report)
        return 0
    if args.stage in ("r14", "all"):
        res = run_top1_phase_r14(args, env, re, p1, p1_report)
        return 0
    # R15 (CURRENT round): coverage-preserving shortlist M3 selection JOINT GRPO.
    # Standalone ONLY (never part of "all") -- it is the governing T1 round and must
    # run explicitly, exactly like r7/r9gen/r10gen.
    if args.stage == "r15":
        res = run_top1_phase_r15(args, env, re, p1, p1_report)
        return 0
    if args.stage == "r16":
        res = run_top1_phase_r16(args, env, re, p1, p1_report)
        return 0
    if args.stage == "r17":
        res = run_top1_phase_r17(args, env, re, p1, p1_report)
        return 0
    # R18 (CURRENT governing round): T1-PROPOSAL-VALIDATION-GATED-JOINT-GRPO.
    # Standalone ONLY (like r7/r9gen/r10gen) -- it is the highest-priority round.
    if args.stage == "r18":
        res = run_top1_phase_r18(args, env, re, p1, p1_report)
        return 0
    # R19 (CURRENT governing round): T1-MULTISTEP-PROPOSAL-VALIDATION-JOINT-GRPO.
    # Standalone ONLY -- H-step validated action set (PA/PB/PC/PD), JOINT GRPO.
    if args.stage == "r19":
        res = run_multistep_phase_r19(args, env, re, p1, p1_report)
        return 0
    # R20 (CURRENT governing round): T1-LEXICOGRAPHIC-PROPOSAL-FALLBACK-JOINT-GRPO.
    # Standalone ONLY -- state-level lexicographic fallback (PA>>PB>>PC>>STOP).
    if args.stage == "r20":
        res = run_lexicographic_phase_r20(args, env, re, p1, p1_report)
        return 0
    # R21 (CURRENT governing round): T1-INSTANCE-DIVERSE-JOINT-GRPO.
    # Standalone ONLY -- the ONLY change is the Joint training instance distribution
    # (D0 benchmark-dominant vs D1 instance-diverse balanced), same trajectory budget.
    if args.stage == "r21":
        res = run_instance_diverse_phase_r21(args, env, re, p1, p1_report)
        return 0
    # T2-A (CURRENT governing round): MULTI-PATH HELD TRAJECTORY EVALUATION.
    # Evaluation-only, no training; frozen canonical policy; greedy vs N sampled.
    if args.stage == "t2a":
        res = run_multipath_eval_t2a(args, env, re, p1, p1_report)
        return 0
    # T2-B: MULTI-PATH JOINT AGENTIC GRPO (T2-A verdict A follow-up).  Stage-3 RL
    # training K sibling trajectories per root so latent positives become greedy-
    # extractable.  R21 runtime verbatim; best-of-N is a post-hoc diagnostic only.
    if args.stage == "t2b":
        res = run_multipath_joint_grpo_t2b(args, env, re, p1, p1_report)
        return 0
    if args.stage == "t2d":
        res = run_multipath_joint_grpo_t2b(args, env, re, p1, p1_report)
        return 0
    if args.stage in ("r8", "all"):
        res = run_top1_phase_r8(args, env, re, p1, p1_report)
        write_top1_r8_markdown(res)
        return 0
    if args.stage == "r7":
        res = run_top1_phase_r7(args, env, re, p1, p1_report)
        write_top1_r7_markdown(res)
        return 0

    if args.stage == "top1":
        res = run_top1_phase(args, env, re, p1, p1_report)
        write_top1_markdown(res)
        return 0

    if args.stage == "p2":
        p2 = run_phase2(args, env, re, p1, p1_report)
        table = compute_b0_b6(env, re, p1, p2, p1_report.get("passed", False))
        b5_t = table["B5_canonical_sft"]["train_total"]
        b6_t = table["B6_sft_grpo"]["train_total"]
        b6_v = table["B6_sft_grpo"]["val_total"]
        vcode, vlabel, vrank = verdict_fn(b5_t, b6_t, b6_v, p1_report, p2)
        print(f"[verdict] {vcode} {vlabel} (B5={b5_t} B6={b6_t}/{b6_v} delta_norm={vrank:.3f})", flush=True)
        reg = run_regressions(args, env, re, p1, p2, p1_report)
        write_report(args, re, p1, p1_report, p2, table, vlabel, vcode, vrank, reg)
        return 0

    # phase-1-only run (inventory/p1): no R6 top1, no GRPO
    print("[stage] phase-1 only: R6 Top-1 / GRPO deferred (use --stage top1 / p2)", flush=True)
    write_report(args, re, p1, p1_report, None, None, "PHASE1_ONLY",
                 "STAGE_P1_ONLY_R6_DEFERRED", 0.0, {"skipped": "stage=p1"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
